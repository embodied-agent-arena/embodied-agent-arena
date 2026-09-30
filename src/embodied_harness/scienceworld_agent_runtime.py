from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import shutil
from typing import Any, Callable

from .backend import EmbodiedBackend
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[dict[str, Any]], Any]


@dataclass(slots=True)
class ScienceWorldRuntimeConfig:
    task_name: str = "find-non-living-thing"
    variation_idx: int = 0
    simplification: str = "easy"
    env_step_limit: int = 50
    live: bool = True
    server_path: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


class ScienceWorldAgentRuntimeBackend(EmbodiedBackend):
    """ScienceWorld adapter with text-action primitives for coding agents.

    The official ScienceWorld environment exposes rich debugging helpers such as
    gold action sequences. This backend keeps those harness-side: the agent sees
    task text, current text observations, available action strings, and typed
    action primitives that execute through the real `env.step(action)` path.
    """

    def __init__(
        self,
        config: ScienceWorldRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.config = config or ScienceWorldRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._last_observation = ""
        self._last_reward = 0
        self._last_done = False
        self._last_info: JsonDict = {}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        runtime_config = self._merged_config(config or {})
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._env = self._make_env(runtime_config)
        if hasattr(self._env, "load"):
            self._env.load(runtime_config.task_name, runtime_config.variation_idx, runtime_config.simplification)
        observation, info = self._env.reset()
        self._last_observation = str(observation)
        self._last_reward = int(info.get("reward", 0) or 0)
        self._last_done = False
        self._last_info = _to_builtin(dict(info or {}))
        instruction = str(self._last_info.get("taskDesc") or self._safe_call("get_task_description") or "")
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w3:scienceworld:live_text_runtime" if runtime_config.live else "w3:scienceworld:fixture_runtime",
            instruction=instruction,
            goal={
                "task_name": runtime_config.task_name,
                "variation_idx": runtime_config.variation_idx,
                "simplification": runtime_config.simplification,
                "success_source": "harness_only_score_done",
            },
            initial_state={
                "observation": self._last_observation,
                "score": self._score(),
                "moves": self._moves(),
                "valid_action_count": len(self._valid_actions()),
            },
            budgets={"primitive_calls": 24, "verifier_calls": 4, "env_step_limit": runtime_config.env_step_limit},
            tags=["w3", "scienceworld", "text_env", "state_action", "live" if runtime_config.live else "fixture"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "scienceworld",
                "runtime_config": runtime_config.to_dict(),
                "runtime_available": self.runtime_available(live_requested=runtime_config.live),
                "agent_native_contract": {
                    "primitives_accept_prompt_query_agent_context": True,
                    "text_observation_and_action_evidence_returned": True,
                    "raw_step_available_as_named_primitive": True,
                    "gold_action_sequence_exposed": False,
                    "official_success_is_harness_only": True,
                },
            },
        )
        self.record_event("reset", {"task": self._task_spec.to_dict(), "seed": seed})
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        data = self._state_payload(prompt=None, query=None, agent_context={})
        obs = Observation(step=len(self.get_trace().events), data=data, metadata={"benchmark_id": "scienceworld"})
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            _card(
                "get_scienceworld_task_context",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"task_description": "str", "score": "int", "moves": "int", "runtime": "dict"},
                "Return task context and score/move state without exposing gold actions.",
            ),
            _card(
                "observe_scienceworld_state",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "include_actions": "bool"},
                {"observation": "str", "look": "str", "inventory": "str", "valid_actions": "list[str]"},
                "Return current text observation, look, inventory, and optional valid-action evidence.",
            ),
            _card(
                "list_scienceworld_actions",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "action_filter": "str|None", "max_actions": "int"},
                {"valid_actions": "list[str]", "possible_action_templates": "list[str]"},
                "List currently valid ScienceWorld actions, optionally filtered by a substring query.",
            ),
            _card(
                "focus_scienceworld_object",
                "L2",
                {"object_name": "str|None", "query": "str|None", "prompt": "str|None", "agent_context": "dict|None"},
                {"action": "str", "observation": "str", "score": "int", "done": "bool"},
                "Focus on a visible object through the real ScienceWorld step path.",
            ),
            _card(
                "pick_up_scienceworld_object",
                "L2",
                {"object_name": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"action": "str", "observation": "str", "score": "int", "done": "bool"},
                "Pick up an object through ScienceWorld's text action path.",
            ),
            _card(
                "go_scienceworld_location",
                "L2",
                {"location": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"action": "str", "observation": "str", "score": "int", "done": "bool"},
                "Move to a location through a valid ScienceWorld go/teleport action.",
            ),
            _card(
                "move_scienceworld_object_to",
                "L3",
                {"object_name": "str", "destination": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"action": "str", "observation": "str", "score": "int", "done": "bool"},
                "Move an object to a target container/location through the real ScienceWorld step path.",
            ),
            _card(
                "step_scienceworld_action",
                "L3",
                {"action": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"action": "str", "observation": "str", "score": "int", "done": "bool"},
                "Execute an explicit valid ScienceWorld text action.",
            ),
            _card(
                "record_scienceworld_evidence",
                "L1",
                {"key": "str", "value": "any", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"artifact_id": "str"},
                "Record agent-selected task/action/progress evidence in the episode trace.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        allowed = {card.name for card in self.list_primitives()}
        if name not in allowed:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by ScienceWorldAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler is not None else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        score = self._score()
        moves = self._moves()
        if scope == "task":
            ok = bool(self._last_done and score >= 100)
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message="ScienceWorld task reached done with score >= 100" if ok else "ScienceWorld task not complete",
                metrics={"success": float(ok), "score": float(score), "done": float(bool(self._last_done)), "moves": float(moves)},
                metadata={"goal_progress": self._safe_call("get_goal_progress")},
            )
            self.get_trace().final_status = "success" if ok else "failed"
        elif scope == "progress":
            result = VerificationResult(
                ok=score > 0,
                scope=scope,
                message="ScienceWorld positive progress recorded" if score > 0 else "ScienceWorld score is still zero",
                metrics={"score": float(score), "moves": float(moves)},
                metadata={"goal_progress": self._safe_call("get_goal_progress")},
            )
        else:
            result = VerificationResult(ok=False, scope=scope, message=f"Unknown verification scope: {scope}")
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self._trace

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

    def runtime_available(self, live_requested: bool | None = None) -> JsonDict:
        live = self.config.live if live_requested is None else live_requested
        return {
            "live_requested": bool(live),
            "scienceworld_spec_available": importlib.util.find_spec("scienceworld") is not None,
            "java_available": shutil.which("java") is not None,
            "server_path": self.config.server_path,
        }

    def _merged_config(self, overrides: JsonDict) -> ScienceWorldRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return ScienceWorldRuntimeConfig(**data)

    def _make_env(self, config: ScienceWorldRuntimeConfig) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.to_dict())
        if not config.live:
            raise RuntimeError("ScienceWorld live=False requires an env_factory fixture.")
        if importlib.util.find_spec("scienceworld") is None:
            raise RuntimeError("ScienceWorld runtime requires `pip install scienceworld` in the active Python environment.")
        from scienceworld import ScienceWorldEnv

        return ScienceWorldEnv(
            taskName=None,
            serverPath=config.server_path,
            envStepLimit=config.env_step_limit,
        )

    def _primitive_get_scienceworld_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_scienceworld_task_context",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "task_name": self.config.task_name,
                "variation_idx": self.config.variation_idx,
                "simplification": self.config.simplification,
                "task_description": self._task_spec.instruction if self._task_spec else "",
                "score": self._score(),
                "moves": self._moves(),
                "done": bool(self._last_done),
                "runtime": self.runtime_available(),
            },
        )

    def _primitive_observe_scienceworld_state(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        include_actions: bool = True,
    ) -> PrimitiveResult:
        output = self._state_payload(prompt=prompt, query=query, agent_context=agent_context or {})
        if not include_actions:
            output.pop("valid_actions", None)
            output.pop("possible_action_templates", None)
        return PrimitiveResult(name="observe_scienceworld_state", ok=True, output=output)

    def _primitive_list_scienceworld_actions(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        action_filter: str | None = None,
        max_actions: int = 50,
    ) -> PrimitiveResult:
        valid = self._valid_actions()
        if action_filter:
            needle = action_filter.lower()
            valid = [action for action in valid if needle in action.lower()]
        return PrimitiveResult(
            name="list_scienceworld_actions",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "action_filter": action_filter,
                "valid_actions": valid[: max(1, int(max_actions))],
                "valid_action_count": len(valid),
                "possible_action_templates": self._possible_actions(),
            },
        )

    def _primitive_focus_scienceworld_object(
        self,
        object_name: str | None = None,
        query: str | None = None,
        prompt: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        target = object_name or query
        if not target:
            return PrimitiveResult(name="focus_scienceworld_object", ok=False, error="object_name_or_query_required")
        return self._step_named("focus_scienceworld_object", f"focus on {target}", prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_pick_up_scienceworld_object(
        self,
        object_name: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._step_named("pick_up_scienceworld_object", f"pick up {object_name}", prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_go_scienceworld_location(
        self,
        location: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        valid = set(self._valid_actions())
        candidates = [f"go to {location}", f"go {location}", f"teleport to {location}", f"go to door to {location}"]
        action = next((candidate for candidate in candidates if candidate in valid), candidates[0])
        return self._step_named("go_scienceworld_location", action, prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_move_scienceworld_object_to(
        self,
        object_name: str,
        destination: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._step_named(
            "move_scienceworld_object_to",
            f"move {object_name} to {destination}",
            prompt=prompt,
            query=query,
            agent_context=agent_context,
        )

    def _primitive_step_scienceworld_action(
        self,
        action: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._step_named("step_scienceworld_action", action, prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_record_scienceworld_evidence(
        self,
        key: str,
        value: Any,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"scienceworld:evidence:{key}"
        payload = {"key": key, "value": _to_builtin(value), "prompt": prompt, "query": query, "agent_context": agent_context or {}}
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="record_scienceworld_evidence",
            ok=True,
            output={"artifact_id": artifact_id, "prompt": prompt, "query": query, "agent_context": agent_context or {}},
            artifacts=[artifact_id],
        )

    def _step_named(
        self,
        primitive_name: str,
        action: str,
        *,
        prompt: str | None,
        query: str | None,
        agent_context: JsonDict | None,
    ) -> PrimitiveResult:
        valid_before = self._valid_actions()
        if action not in valid_before:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "action": action,
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                    "valid_action_count": len(valid_before),
                    "matching_valid_actions": _matching_actions(valid_before, action),
                },
                error="invalid_scienceworld_action",
            )
        observation, reward, done, info = self._env.step(action)
        self._last_observation = str(observation)
        self._last_reward = int(reward)
        self._last_done = bool(done)
        self._last_info = _to_builtin(dict(info or {}))
        return PrimitiveResult(
            name=primitive_name,
            ok=True,
            output={
                "action": action,
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "observation": self._last_observation,
                "reward": self._last_reward,
                "done": self._last_done,
                "score": self._score(),
                "moves": self._moves(),
                "goal_progress": self._safe_call("get_goal_progress"),
                "valid_action_count": len(self._valid_actions()),
            },
        )

    def _state_payload(self, *, prompt: str | None, query: str | None, agent_context: JsonDict) -> JsonDict:
        return {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context,
            "task_description": self._task_spec.instruction if self._task_spec else "",
            "observation": self._last_observation,
            "look": self._safe_call("look"),
            "inventory": self._safe_call("inventory"),
            "score": self._score(),
            "reward": self._last_reward,
            "done": bool(self._last_done),
            "moves": self._moves(),
            "goal_progress": self._safe_call("get_goal_progress"),
            "valid_actions": self._valid_actions(),
            "possible_action_templates": self._possible_actions(),
        }

    def _safe_call(self, method_name: str) -> Any:
        method = getattr(self._env, method_name, None)
        if not callable(method):
            return None
        try:
            return _to_builtin(method())
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _valid_actions(self) -> list[str]:
        method = getattr(self._env, "get_valid_action_object_combinations", None)
        if callable(method):
            try:
                return [str(action) for action in method()]
            except Exception:
                return []
        return list(self._last_info.get("valid", []) or [])

    def _possible_actions(self) -> list[str]:
        method = getattr(self._env, "get_possible_actions", None)
        if callable(method):
            try:
                return [str(action) for action in method()]
            except Exception:
                return []
        return []

    def _score(self) -> int:
        return int(self._last_info.get("score", 0) or 0)

    def _moves(self) -> int:
        return int(self._last_info.get("moves", 0) or 0)

    def _require_reset(self) -> None:
        if self._task_spec is None or self._trace is None or self._env is None:
            raise RuntimeError("Call reset() before using the backend.")


def _card(name: str, level: str, input_schema: JsonDict, output_schema: JsonDict, description: str) -> PrimitiveCard:
    return PrimitiveCard(
        name=name,
        capability_tags=["scienceworld", "text_env", "state_action", "agent_safe"],
        input_schema=input_schema,
        output_schema=output_schema,
        preconditions=["backend reset"],
        side_effects=["env.step(action)"] if level in {"L2", "L3"} else [],
        failure_modes=["runtime_missing", "invalid_scienceworld_action"],
        abstraction_level=level,
        leakage_risk="low",
        description=description,
    )


def _matching_actions(valid_actions: list[str], attempted_action: str) -> list[str]:
    tokens = [token for token in attempted_action.lower().split() if token not in {"to", "on", "the", "a", "an"}]
    matches = []
    for action in valid_actions:
        lower = action.lower()
        if any(token in lower for token in tokens):
            matches.append(action)
    return matches[:20]


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
