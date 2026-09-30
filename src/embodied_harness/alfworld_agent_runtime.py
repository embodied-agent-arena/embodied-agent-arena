from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import shutil
from typing import Any, Callable

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[dict[str, Any]], Any]
_DEFAULT_ALFWORLD_WORKTREE = str(get_project_paths().external_upstream("alfworld"))


@dataclass(slots=True)
class ALFWorldRuntimeConfig:
    """Configuration for the agent-safe ALFWorld TextWorld runtime."""

    worktree_path: str = _DEFAULT_ALFWORLD_WORKTREE
    split: str = "eval_in_distribution"
    task_types: tuple[int, ...] = (1,)
    batch_size: int = 1
    max_steps: int = 50
    live: bool = True
    game_file: str | None = None
    data_root: str | None = None
    num_eval_games: int = 1

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["task_types"] = list(self.task_types)
        return data


class ALFWorldAgentRuntimeBackend(EmbodiedBackend):
    """Agent-facing primitive wrapper around the official ALFWorld TextWorld env.

    ALFWorld can expose expert trajectories through training wrappers and can
    report `won` in TextWorld infos. This backend keeps those harness-side: the
    agent sees task text, current text observations, admissible commands, and
    typed action primitives that execute through the official `env.step` path.
    """

    def __init__(
        self,
        config: ALFWorldRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.config = config or ALFWorldRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._alf_env: Any | None = None
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._last_observation = ""
        self._last_reward = 0
        self._last_done = False
        self._last_won = False
        self._last_info: JsonDict = {}
        self._steps = 0

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        runtime_config = self._merged_config(config or {})
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._env, self._alf_env = self._make_env(runtime_config)
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        observation, info = self._env.reset()
        self._last_observation = _first_text(observation)
        self._last_reward = 0
        self._last_done = False
        self._last_info = _to_builtin(dict(info or {}))
        self._last_won = self._info_bool("won")
        self._steps = 0
        instruction = _extract_task(self._last_observation)
        game_file = self._first_info("extra.gamefile")
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w3:alfworld:live_textworld_runtime" if runtime_config.live else "w3:alfworld:fixture_runtime",
            instruction=instruction,
            goal={
                "benchmark": "alfworld",
                "task_family": "pick_and_place_simple" if 1 in runtime_config.task_types else "configured",
                "completion_source": "harness_only_textworld_flag",
            },
            initial_state={
                "observation": self._last_observation,
                "admissible_action_count": len(self._valid_actions()),
                "game_file": game_file,
            },
            budgets={"primitive_calls": 24, "verifier_calls": 4, "env_step_limit": runtime_config.max_steps},
            tags=["w3", "alfworld", "textworld", "state_action", "live" if runtime_config.live else "fixture"],
            allowed_primitive_levels=["L1", "L2"],
            metadata={
                "benchmark_id": "alfworld",
                "runtime_config": runtime_config.to_dict(),
                "runtime_available": self.runtime_available(live_requested=runtime_config.live),
                "agent_native_contract": {
                    "primitives_accept_prompt_query_agent_context": True,
                    "admissible_text_actions_returned": True,
                    "privileged_plan_visible": False,
                    "official_completion_flag_is_harness_only": True,
                },
            },
        )
        self.record_event("reset", {"task": self._task_spec.to_dict(), "seed": seed})
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        data = self._state_payload(prompt=None, query=None, agent_context={})
        obs = Observation(step=len(self.get_trace().events), data=data, metadata={"benchmark_id": "alfworld"})
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            _card(
                "get_alfworld_task_context",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"task_description": "str", "steps": "int", "runtime": "dict"},
                "Return task context while keeping privileged completion signals harness-side.",
            ),
            _card(
                "observe_alfworld_state",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "include_actions": "bool"},
                {"observation": "str", "inventory": "str|None", "admissible_actions": "list[str]"},
                "Return current text observation, inventory text, and optional admissible-action evidence.",
            ),
            _card(
                "list_alfworld_actions",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "action_filter": "str|None", "max_actions": "int"},
                {"admissible_actions": "list[str]"},
                "List currently admissible ALFWorld TextWorld actions, optionally filtered.",
            ),
            _card("go_to", "L2", {"location": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Move to a location through `go to <location>`."),
            _card("take", "L2", {"object_name": "str", "source": "str|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Take an object, optionally from a named receptacle."),
            _card("put", "L2", {"object_name": "str", "destination": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Place an object using ALFWorld's admissible move command."),
            _card("open", "L2", {"target": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Open a receptacle through the official text action path."),
            _card("close", "L2", {"target": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Close a receptacle through the official text action path."),
            _card("toggle", "L2", {"target": "str", "on": "bool|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}, {"action": "str", "observation": "str", "done": "bool"}, "Toggle a device using an admissible on/off action."),
            _card(
                "record_alfworld_evidence",
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
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by ALFWorldAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler is not None else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope == "task":
            ok = bool(self._last_won and self._last_done)
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message="ALFWorld official TextWorld won=True and done=True" if ok else "ALFWorld task not complete",
                metrics={"success": float(ok), "won": float(self._last_won), "done": float(self._last_done), "steps": float(self._steps)},
                metadata={"game_file": self._first_info("extra.gamefile")},
            )
            self.get_trace().final_status = "success" if ok else "failed"
        elif scope == "progress":
            ok = self._steps > 0 and self._last_reward >= 0
            result = VerificationResult(ok=ok, scope=scope, message="ALFWorld step path executed" if ok else "No ALFWorld step executed", metrics={"steps": float(self._steps), "last_reward": float(self._last_reward)})
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
        self._alf_env = None

    def runtime_available(self, live_requested: bool | None = None) -> JsonDict:
        live = self.config.live if live_requested is None else live_requested
        root = Path(self.config.worktree_path)
        python_path = root / ".conda-alfworld-py310" / "bin" / "python"
        legacy_python_path = root / ".conda-alfworld" / "bin" / "python"
        return {
            "live_requested": bool(live),
            "alfworld_spec_available": importlib.util.find_spec("alfworld") is not None,
            "textworld_spec_available": importlib.util.find_spec("textworld") is not None,
            "worktree_python": str(python_path),
            "worktree_python_exists": python_path.exists(),
            "legacy_worktree_python": str(legacy_python_path),
            "legacy_worktree_python_exists": legacy_python_path.exists(),
            "java_available": shutil.which("java") is not None,
        }

    def _merged_config(self, overrides: JsonDict) -> ALFWorldRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        if "task_types" in data:
            data["task_types"] = tuple(data["task_types"])
        return ALFWorldRuntimeConfig(**data)

    def _make_env(self, config: ALFWorldRuntimeConfig) -> tuple[Any, Any | None]:
        if self._env_factory is not None:
            return self._env_factory(config.to_dict()), None
        if not config.live:
            raise RuntimeError("ALFWorld live=False requires an env_factory fixture.")
        if importlib.util.find_spec("alfworld") is None:
            raise RuntimeError("ALFWorld runtime requires the live worktree Python or an environment with `alfworld` installed.")
        from alfworld.agents.environment import get_environment

        alf_config = _alfworld_config(config)
        alf_env = get_environment("AlfredTWEnv")(alf_config, train_eval=config.split)
        if config.game_file:
            alf_env.game_files = [str(Path(config.game_file).expanduser())]
            alf_env.num_games = 1
        return alf_env.init_env(batch_size=config.batch_size), alf_env

    def _primitive_get_alfworld_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_alfworld_task_context",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "task_description": self._task_spec.instruction if self._task_spec else "",
                "instruction": self._task_spec.instruction if self._task_spec else "",
                "steps": self._steps,
                "done": bool(self._last_done),
                "game_file": self._first_info("extra.gamefile"),
                "runtime": self.runtime_available(),
            },
        )

    def _primitive_observe_alfworld_state(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        include_actions: bool = True,
    ) -> PrimitiveResult:
        output = self._state_payload(prompt=prompt, query=query, agent_context=agent_context or {})
        if not include_actions:
            output.pop("admissible_actions", None)
        return PrimitiveResult(name="observe_alfworld_state", ok=True, output=output)

    def _primitive_list_alfworld_actions(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        action_filter: str | None = None,
        max_actions: int = 80,
    ) -> PrimitiveResult:
        actions = self._valid_actions()
        if action_filter:
            needle = action_filter.lower()
            actions = [action for action in actions if needle in action.lower()]
        return PrimitiveResult(
            name="list_alfworld_actions",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "action_filter": action_filter,
                "matching_actions": actions[: max(1, int(max_actions))],
                "admissible_actions": actions[: max(1, int(max_actions))],
                "admissible_action_count": len(actions),
            },
        )

    def _primitive_go_to(
        self,
        location: str | None = None,
        target: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        location = location or target
        if not location:
            return PrimitiveResult(name="go_to", ok=False, error="location_or_target_required")
        return self._step_named("go_to", f"go to {location}", prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_take(
        self,
        object_name: str,
        source: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if source:
            action = f"take {object_name} from {source}"
        else:
            action = self._choose_action("take", object_name) or f"take {object_name}"
        return self._step_named("take", action, prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_put(
        self,
        object_name: str,
        destination: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        action = self._choose_action_with_tokens([f"move {object_name} to {destination}"], [object_name, destination])
        return self._step_named("put", action, prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_open(self, target: str, prompt: str | None = None, query: str | None = None, agent_context: JsonDict | None = None) -> PrimitiveResult:
        return self._step_named("open", f"open {target}", prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_close(self, target: str, prompt: str | None = None, query: str | None = None, agent_context: JsonDict | None = None) -> PrimitiveResult:
        return self._step_named("close", f"close {target}", prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_toggle(
        self,
        target: str,
        on: bool | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        prefixes = ["toggle"]
        if on is True:
            prefixes = ["turn on", "activate", "toggle"]
        elif on is False:
            prefixes = ["turn off", "deactivate", "toggle"]
        action = next((candidate for prefix in prefixes if (candidate := f"{prefix} {target}") in self._valid_actions()), f"{prefixes[0]} {target}")
        return self._step_named("toggle", action, prompt=prompt, query=query, agent_context=agent_context)

    def _primitive_record_alfworld_evidence(
        self,
        key: str,
        value: Any,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"alfworld:evidence:{key}"
        payload = {"key": key, "value": _to_builtin(value), "prompt": prompt, "query": query, "agent_context": agent_context or {}}
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="record_alfworld_evidence",
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
                    "admissible_action_count": len(valid_before),
                    "matching_admissible_actions": _matching_actions(valid_before, action),
                },
                error="invalid_alfworld_action",
            )
        observation, reward, done, info = self._env.step([action])
        self._last_observation = _first_text(observation)
        self._last_reward = int(_first_item(reward, 0) or 0)
        self._last_done = bool(_first_item(done, False))
        self._last_info = _to_builtin(dict(info or {}))
        self._last_won = self._info_bool("won")
        self._steps += 1
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
                "steps": self._steps,
                "admissible_action_count": len(self._valid_actions()),
            },
        )

    def _state_payload(self, *, prompt: str | None, query: str | None, agent_context: JsonDict) -> JsonDict:
        return {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context,
            "task_description": self._task_spec.instruction if self._task_spec else "",
            "observation": self._last_observation,
            "inventory": self._inventory_text(),
            "reward": self._last_reward,
            "done": bool(self._last_done),
            "steps": self._steps,
            "admissible_actions": self._valid_actions(),
        }

    def _inventory_text(self) -> str | None:
        actions = self._valid_actions()
        if "inventory" not in actions:
            return None
        return "Use the admissible `inventory` action to inspect held objects."

    def _valid_actions(self) -> list[str]:
        return [str(action) for action in (_first_item(self._last_info.get("admissible_commands"), []) or [])]

    def _choose_action(self, prefix: str, target: str) -> str | None:
        target_tokens = target.lower().split()
        for action in self._valid_actions():
            lower = action.lower()
            if lower.startswith(prefix) and all(token in lower for token in target_tokens):
                return action
        return None

    def _choose_action_with_tokens(self, candidates: list[str], token_groups: list[str]) -> str:
        valid = self._valid_actions()
        for candidate in candidates:
            if candidate in valid:
                return candidate
        tokens = []
        for group in token_groups:
            tokens.extend(str(group).lower().split())
        for action in valid:
            lower = action.lower()
            if all(token in lower for token in tokens):
                return action
        return candidates[0]

    def _first_info(self, key: str, default: Any = None) -> Any:
        return _first_item(self._last_info.get(key), default)

    def _info_bool(self, key: str) -> bool:
        return bool(_first_item(self._last_info.get(key), False))

    def _require_reset(self) -> None:
        if self._task_spec is None or self._trace is None or self._env is None:
            raise RuntimeError("Call reset() before using the backend.")


def _alfworld_config(config: ALFWorldRuntimeConfig) -> JsonDict:
    root = Path(config.worktree_path).expanduser().resolve()
    cache_root = Path(config.data_root).expanduser().resolve() if config.data_root else root / ".cache" / "alfworld"
    data_path = cache_root / "json_2.1.1" / "valid_seen"
    logic_path = cache_root / "logic"
    return {
        "env": {
            "goal_desc_human_anns_prob": 0,
            "task_types": list(config.task_types),
            "domain_randomization": False,
            "expert_type": "handcoded",
        },
        "dataset": {
            "data_path": str(data_path),
            "eval_id_data_path": str(data_path),
            "eval_ood_data_path": str(data_path),
            "num_train_games": 0,
            "num_eval_games": config.num_eval_games,
        },
        "logic": {
            "domain": str(logic_path / "alfred.pddl"),
            "grammar": str(logic_path / "alfred.twl2"),
        },
        "general": {"training_method": "dqn"},
        "rl": {"training": {"max_nb_steps_per_episode": config.max_steps}},
        "dagger": {"training": {"max_nb_steps_per_episode": config.max_steps}},
    }


def _card(name: str, level: str, input_schema: JsonDict, output_schema: JsonDict, description: str) -> PrimitiveCard:
    return PrimitiveCard(
        name=name,
        capability_tags=["alfworld", "textworld", "state_action", "agent_safe"],
        input_schema=input_schema,
        output_schema=output_schema,
        preconditions=["backend reset"],
        side_effects=["env.step(action)"] if level == "L2" else [],
        failure_modes=["runtime_missing", "invalid_alfworld_action"],
        abstraction_level=level,
        leakage_risk="low",
        description=description,
    )


def _extract_task(observation: str) -> str:
    marker = "Your task is to:"
    if marker in observation:
        return f"{marker} {observation.split(marker, 1)[1].strip()}"
    return observation.strip().splitlines()[-1] if observation.strip() else ""


def _first_text(value: Any) -> str:
    return str(_first_item(value, ""))


def _first_item(value: Any, default: Any = None) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else default
    return default if value is None else value


def _matching_actions(valid_actions: list[str], attempted_action: str) -> list[str]:
    tokens = [token for token in attempted_action.lower().split() if token not in {"to", "from", "in", "on", "the", "a", "an"}]
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
