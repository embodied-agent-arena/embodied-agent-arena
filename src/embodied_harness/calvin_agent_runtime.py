from __future__ import annotations

import ast
import contextlib
from copy import deepcopy
import hashlib
import json
import collections
import collections.abc
from dataclasses import asdict, dataclass, field
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import (
    EpisodeTrace,
    Observation,
    PrimitiveCard,
    PrimitiveResult,
    TaskSpec,
    VerificationResult,
)


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]
PolicyCallable = Callable[[str, dict[str, Any]], Any]


@dataclass(slots=True)
class CALVINRuntimeConfig:
    sequence_id: str = "debug_language_sequence"
    dataset_root: str | None = None
    split: str = "debug"
    camera_views: list[str] = field(default_factory=lambda: ["static", "gripper"])
    live: bool = True
    calvin_root: str | None = None
    show_gui: bool = False
    use_egl: bool = False
    env_kwargs: JsonDict = field(default_factory=dict)
    policy_backend: str | None = None
    policy_train_folder: str | None = None
    policy_checkpoint: str | None = None
    policy_dataset_path: str | None = None
    policy_device_id: int = 0
    policy_runtime_dependency_check: bool = True
    official_eval_num_sequences: int = 1000
    official_eval_sequence_index: int = 1
    official_eval_num_workers: int | None = 4
    native_task_key: str = "turn_off_led"
    native_initial_state: JsonDict = field(
        default_factory=lambda: {
            "led": 1,
            "lightbulb": 0,
            "slider": "right",
            "drawer": "closed",
            "red_block": "table",
            "blue_block": "table",
            "pink_block": "slider_right",
            "grasped": 0,
        }
    )

    def to_dict(self) -> JsonDict:
        return asdict(self)


class CALVINAgentRuntimeBackend(EmbodiedBackend):
    """AI-native CALVIN runtime adapter.

    CALVIN success predicates and dataset/oracle helpers remain harness-side.
    The coding agent receives language/state/camera context and can call a
    language-conditioned skill primitive backed by a real policy/controller.
    """

    def __init__(
        self,
        config: CALVINRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
        policy: PolicyCallable | None = None,
    ) -> None:
        self.config = config or CALVINRuntimeConfig()
        self._env_factory = env_factory
        self._policy = policy
        self._env: Any | None = None
        self._last_obs: Any = None
        self._language_subgoal: str | None = None
        self._completed_subgoals: list[str] = []
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._official_policy_model: Any | None = None
        self._official_policy_env: Any | None = None
        self._official_policy_metadata: JsonDict | None = None
        self._official_task_checker: Any | None = None
        self._official_task_key_cache: dict[str, str | None] = {}
        self._official_eval_case: JsonDict | None = None
        self._last_skill_subgoal: str | None = None
        self._last_skill_step_result: Any = None
        self._last_skill_start_info: JsonDict | None = None
        self._last_official_verification: JsonDict | None = None
        self._pool_sequence: JsonDict | None = None
        self._pool_completed_count = 0

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        name = str(coordinate.get("task_id") or "")
        seed = coordinate.get("seed")
        if coordinate.get("variation") != name or not name.startswith("calvin_sequence_"):
            raise ValueError("CALVIN requires an original sequence coordinate")
        if type(seed) is not int or seed != 0:
            raise ValueError("The official CALVIN sequence pool uses seed 0")
        pool_path = os.environ.get("EMBODIED_ARENA_CALVIN_SEQUENCE_POOL")
        if not pool_path:
            raise ValueError("CALVIN requires EMBODIED_ARENA_CALVIN_SEQUENCE_POOL")
        pool = json.loads(Path(pool_path).read_text())["benchmarks"]["calvin"]
        for source, digest in pool["source_files"].items():
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != digest:
                raise ValueError("CALVIN sequence generator source changed")
        matches = [row for row in pool["candidates"] if row["task_id"] == name]
        if len(matches) != 1 or len(matches[0]["native_subgoals"]) != 5:
            raise ValueError("Unknown CALVIN five-subgoal sequence")
        self._pool_sequence = deepcopy(matches[0])
        return {"bound": True, "mode": "native_five_subgoal_sequence", "sequence_id": name, "seed": seed}

    def reset(
        self, task_id: str, seed: int | None = None, config: JsonDict | None = None
    ) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_sequence is not None:
            if seed != self._pool_sequence["seed"]:
                raise ValueError("CALVIN reset seed differs from selected coordinate")
            overrides.update(sequence_id=self._pool_sequence["task_id"],
                             native_task_key=self._pool_sequence["native_subgoals"][0],
                             native_initial_state=deepcopy(self._pool_sequence["native_initial_state"]))
            task_id = self._pool_sequence["task_id"]
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:calvin:live_runtime",
            instruction="Solve a CALVIN language-conditioned manipulation sequence by observing state/cameras and invoking policy-backed skills.",
            goal={
                "sequence_id": runtime_config.sequence_id,
                "success_source": "harness_side_subgoal_verifier",
            },
            budgets={"primitive_calls": 32, "verifier_calls": 5},
            tags=[
                "w4",
                "calvin",
                "ai_native_runtime",
                "live" if runtime_config.live else "dry",
            ],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "calvin",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "language_subgoal_visible": True,
                    "camera_grounding_visible": True,
                    "oracle_primitives_exposed": False,
                    "checker_primitives_exposed": False,
                    "mock_success_for_actions": False,
                },
            },
        )
        if self._pool_sequence is not None:
            self._task_spec.instruction = (
                "Complete all five CALVIN subgoals in order in this episode. "
                "Read the current language_subgoal from observations and submit native 7D actions. "
                "After each code turn the harness checks the current subgoal and exposes the next one when complete."
            )
        self._completed_subgoals = []
        self._pool_completed_count = 0
        self._language_subgoal = None
        self._last_skill_subgoal = None
        self._last_skill_step_result = None
        self._last_skill_start_info = None
        self._last_official_verification = None
        official_readiness = (
            official_mcil_backend_readiness(runtime_config)
            if runtime_config.policy_backend == "calvin_official_mcil"
            else None
        )
        if self._uses_native_agent_backend():
            self._official_eval_case = _native_official_eval_case(runtime_config)
            if self._pool_sequence is not None:
                if self._official_eval_case is None:
                    raise RuntimeError("CALVIN native sequence task configuration unavailable")
                self._official_eval_case["eval_sequence"] = list(self._pool_sequence["native_subgoals"])
                self._official_eval_case["source"] = "calvin_official_five_subgoal_sequence"
        else:
            self._official_eval_case = (
                _first_official_mcil_eval_case(runtime_config)
                if official_readiness is None or official_readiness.get("ready")
                else None
            )
        official_eval_language = (
            str(self._official_eval_case["language"])
            if isinstance(self._official_eval_case, dict)
            and self._official_eval_case.get("language")
            else None
        )
        if runtime_config.live:
            self._env = self._make_env(runtime_config)
            self._last_obs = self._reset_env(seed)
            if self._uses_native_agent_backend():
                self._last_obs = self._reset_native_agent_env_to_eval_initial_state()
            self._language_subgoal = (
                _extract_language(self._last_obs) or official_eval_language
            )
            if self._uses_native_agent_backend():
                self._last_skill_subgoal = self._language_subgoal
                self._last_skill_start_info = _safe_calvin_get_info(
                    _calvin_info_source(self._env, self._official_policy_env, self._env)
                )
        else:
            self._env = None
            self._last_obs = None
            self._language_subgoal = official_eval_language
        self.record_event(
            "reset",
            {
                "task": self._task_spec.to_dict(),
                "seed": seed,
                "runtime": self.runtime_available(),
            },
        )
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "language_subgoal": self._language_subgoal,
                "observation_summary": summarize_calvin_observation(
                    self._last_obs, camera_views=self.config.camera_views
                ),
                "semantic_state": summarize_calvin_semantic_state(self._env),
            },
            metadata={"benchmark_id": "calvin"},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def run_harness_noop_step(self, action: Any | None = None) -> JsonDict:
        """Run one harness-owned CALVIN step for live smoke verification.

        This is intentionally not listed as a coding-agent primitive. It proves
        the official environment can advance without exposing policy/expert or
        success-check shortcuts to the agent-facing surface.
        """
        self._require_reset()
        if self._env is None or not hasattr(self._env, "step"):
            result = {"ok": False, "error": "env_step_unavailable"}
        else:
            chosen_action = (
                action if action is not None else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
            )
            try:
                step_result = self._env.step(chosen_action)
            except Exception as exc:  # pragma: no cover - live dependency path.
                result = {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "action": _to_builtin(chosen_action),
                }
            else:
                self._last_obs = _first_step_observation(step_result)
                result = {
                    "ok": True,
                    "action": _to_builtin(chosen_action),
                    "done": _step_done(step_result),
                    "observation_summary": summarize_calvin_observation(
                        self._last_obs, camera_views=self.config.camera_views
                    ),
                    "raw_step_summary": summarize_calvin_step_result(
                        step_result, camera_views=self.config.camera_views
                    ),
                }
        self.record_event("harness_noop_step", result)
        return result

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_calvin_state",
                "L1",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {"language_subgoal": "str|None", "observation_summary": "dict"},
                (
                    "Observe CALVIN robot/scene state summaries with optional agent prompt/context. "
                    "Call this before selecting or executing a language-conditioned skill."
                ),
            ),
            self._primitive_card(
                "observe_calvin_cameras",
                "L1",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "camera_view": "str|None",
                },
                {
                    "cameras": "dict",
                    "visual_observations": "dict",
                    "evidence_refs": "list[dict]",
                    "visual_runtime": "dict",
                    "artifacts": "list[str]",
                },
                (
                    "Inspect available CALVIN camera modalities for language-conditioned grounding. "
                    "Pass returned evidence_refs to record_calvin_evidence or execute_calvin_language_skill."
                ),
                preconditions=[
                    "Call observe_calvin_state or get_calvin_language_subgoal first so the camera query is tied to a subgoal.",
                    "Pass prompt/query text that describes the object or affordance you need to ground.",
                ],
            ),
            self._primitive_card(
                "get_calvin_language_subgoal",
                "L2",
                {"agent_context": "dict|None"},
                {"subgoal": "str|None"},
                "Return the current language subgoal visible to the policy agent. Use this exact subgoal for downstream skill execution.",
            ),
            self._primitive_card(
                "get_calvin_runtime_context",
                "L2",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "language_subgoal": "str|None",
                    "scene": "dict",
                    "cameras": "dict",
                    "available_skill_schema": "dict",
                },
                "Return agent-facing CALVIN language, scene/camera evidence, and available skill/action schema.",
            ),
            self._primitive_card(
                "execute_calvin_language_skill",
                "L3",
                {
                    "subgoal": "str",
                    "language_goal": "str|None",
                    "horizon": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "grounding": "dict|None",
                    "evidence_refs": "list[str|dict]|None",
                },
                {
                    "skill_executed": "bool",
                    "steps": "int",
                    "termination": "dict",
                    "provenance": "dict",
                    "requires_policy_backend": "bool",
                },
                (
                    "Execute only the agent-supplied language subgoal through a configured policy/controller backend. "
                    "The subgoal should come from get_calvin_language_subgoal or get_calvin_runtime_context, not from a guessed recipe."
                ),
                preconditions=[
                    "Call observe_calvin_state to bind the current scene state.",
                    "Call get_calvin_language_subgoal and use its returned subgoal.",
                    "Call observe_calvin_cameras for visual grounding when camera evidence is available.",
                    "Call record_calvin_evidence with compact state/visual/skill rationale before execution.",
                    "Pass evidence_refs from observe_calvin_cameras or record_calvin_evidence.",
                ],
            ),
            self._primitive_card(
                "submit_calvin_action",
                "L3",
                {
                    "action": "list[float]",
                    "agent_context": "dict|None",
                    "evidence_refs": "list[str|dict]|None",
                },
                {
                    "stepped": "bool",
                    "action": "list[float]",
                    "termination": "dict",
                    "provenance": "dict",
                    "observation_summary": "dict",
                },
                "Submit one agent-supplied 7D action to the live CALVIN env.step boundary.",
            ),
            self._primitive_card(
                "record_calvin_evidence",
                "L1",
                {
                    "key": "str|None",
                    "value": "any|None",
                    "evidence_type": "str|None",
                    "content": "any|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {"artifact_id": "str"},
                "Record agent-selected CALVIN grounding/action evidence. Accepts key/value or evidence_type/content.",
                preconditions=[
                    "Use only public outputs such as language_subgoal, observation_summary, camera evidence_refs, and selected skill inputs.",
                ],
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
            result = PrimitiveResult(
                name=name,
                ok=False,
                error=f"Primitive {name!r} is not exposed by CALVINAgentRuntimeBackend",
            )
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = (
                handler(**kwargs)
                if handler is not None
                else PrimitiveResult(
                    name=name, ok=False, error=f"Missing handler for {name}"
                )
            )
        self.record_event(
            "primitive_call",
            {"name": name, "kwargs": kwargs, "result": result.to_dict()},
        )
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if self._pool_sequence is not None and scope == "task":
            return self._verify_pool_sequence()
        if scope != "task":
            result = VerificationResult(
                ok=False,
                scope=scope,
                message=f"Unsupported CALVIN verification scope: {scope}",
            )
        else:
            required = kwargs.get("required_subgoals") or (
                [self._language_subgoal] if self._language_subgoal else []
            )
            if self._last_skill_subgoal and self._uses_official_task_verifier():
                if self._last_official_verification is None:
                    current_info = _extract_calvin_step_info(
                        self._last_skill_step_result
                    ) or _safe_calvin_get_info(
                        _calvin_info_source(
                            self._official_policy_env,
                            self._official_policy_env,
                            self._env,
                        )
                    )
                    self._last_official_verification = (
                        self._evaluate_official_calvin_task(
                            self._last_skill_subgoal,
                            self._last_skill_start_info,
                            current_info,
                        )
                    )
                verified = bool(self._last_official_verification.get("success"))
                if (
                    verified
                    and self._last_skill_subgoal not in self._completed_subgoals
                ):
                    self._completed_subgoals.append(self._last_skill_subgoal)
            success = bool(required) and all(
                subgoal in self._completed_subgoals for subgoal in required
            )
            result = VerificationResult(
                ok=success,
                scope="task",
                message="CALVIN harness-side subgoal verifier passed"
                if success
                else "CALVIN required subgoals are not complete",
                metrics={
                    "success": float(success),
                    "completed_count": len(self._completed_subgoals),
                },
                metadata={
                    "required_subgoals": required,
                    "completed_subgoals": list(self._completed_subgoals),
                },
            )
        if result.ok:
            self.get_trace().final_status = "success"
        self.record_event("verifier_call", result.to_dict())
        return result

    def _verify_pool_sequence(self) -> VerificationResult:
        # The native checker compares each subtask's own start/end state.
        # Advance without reset, matching the upstream five-subtask rollout.
        goals = self._pool_sequence["native_subgoals"]
        if self._pool_completed_count < len(goals):
            current = _safe_calvin_get_info(self._env)
            verdict = self._evaluate_official_calvin_task(
                self._language_subgoal, self._last_skill_start_info, current)
            if verdict.get("success"):
                self._completed_subgoals.append(goals[self._pool_completed_count])
                self._pool_completed_count += 1
                self._last_skill_start_info = deepcopy(current)
                self._last_official_verification = None
                if self._pool_completed_count < len(goals):
                    next_goal = goals[self._pool_completed_count]
                    self._language_subgoal = _official_validation_language_for_task(self.config, next_goal) or next_goal
                    self._last_skill_subgoal = self._language_subgoal
        success = self._pool_completed_count == len(goals)
        result = VerificationResult(
            ok=success, scope="task",
            message="CALVIN five-subgoal sequence complete" if success else "CALVIN sequence incomplete",
            metrics={"success": float(success), "completed_count": self._pool_completed_count},
            metadata={"completed_subgoals": list(self._completed_subgoals),
                      "language_subgoal": None if success else self._language_subgoal,
                      "required_count": len(goals)})
        if success:
            self.get_trace().final_status = "success"
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self._trace

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
            if hasattr(self._env, "ownsPhysicsClient"):
                self._env.ownsPhysicsClient = False
            if hasattr(self._env, "cid"):
                self._env.cid = -1
        if (
            self._official_policy_env is not None
            and self._official_policy_env is not self._env
        ):
            if hasattr(self._official_policy_env, "close"):
                self._official_policy_env.close()
            wrapped_env = getattr(self._official_policy_env, "env", None)
            if wrapped_env is not None and hasattr(wrapped_env, "ownsPhysicsClient"):
                wrapped_env.ownsPhysicsClient = False
            if wrapped_env is not None and hasattr(wrapped_env, "cid"):
                wrapped_env.cid = -1
        self._env = None
        self._official_policy_env = None

    def runtime_available(self) -> JsonDict:
        _add_calvin_paths(self.config.calvin_root)
        calvin_root = _resolve_calvin_root(self.config.calvin_root)
        status = {
            "live": self.config.live,
            "env_created": self._env is not None,
            "calvin_env_importable": importlib.util.find_spec("calvin_env") is not None,
            "calvin_agent_importable": importlib.util.find_spec("calvin_agent")
            is not None,
            "calvin_source_root": str(calvin_root) if calvin_root is not None else None,
            "dataset_root": self.config.dataset_root,
            "policy_backend": self.config.policy_backend,
            "builtin_policy_backends": [
                "calvin_native_agent",
                "calvin_primitive_action",
                "calvin_official_mcil",
            ],
        }
        if self._uses_official_mcil_backend():
            status["official_mcil_backend"] = _public_official_mcil_readiness(
                official_mcil_backend_readiness(self.config)
            )
            status["official_mcil_eval_case"] = _public_official_eval_case(
                self._official_eval_case
            )
        if self._uses_native_agent_backend():
            status["native_agent_backend"] = {
                "ready": self._env is not None,
                "policy_checkpoint_required": False,
                "action_source": "agent_supplied_7d_actions",
                "official_task_verifier": "harness_only",
                "eval_case": _public_official_eval_case(self._official_eval_case),
            }
        return status

    def _merged_config(self, overrides: JsonDict) -> CALVINRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return CALVINRuntimeConfig(**data)

    def _make_env(self, config: CALVINRuntimeConfig) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.sequence_id, config.to_dict())
        return make_official_calvin_env(config)

    def _reset_env(self, seed: int | None) -> Any:
        if self._env is None:
            return None
        if hasattr(self._env, "reset"):
            try:
                return self._env.reset(seed=seed)
            except TypeError:
                if hasattr(self._env, "seed") and seed is not None:
                    self._env.seed(seed)
                return self._env.reset()
        return None

    def _reset_native_agent_env_to_eval_initial_state(self) -> Any:
        if not isinstance(self._official_eval_case, dict):
            raise RuntimeError("CALVIN native evaluation case is unavailable")
        if self._env is None or not hasattr(self._env, "reset"):
            raise RuntimeError("CALVIN native environment reset is unavailable")
        state = dict(self._official_eval_case["initial_state"])
        if self._pool_sequence is not None:
            robot_obs, scene_obs = _original_calvin_eval_state(self.config, state)
        else:
            robot_obs, scene_obs = _calvin_native_eval_state(state)
        try:
            obs = self._env.reset(robot_obs=robot_obs, scene_obs=scene_obs)
        except Exception as exc:  # pragma: no cover - live dependency path.
            raise RuntimeError(
                f"CALVIN native evaluation-state reset failed: {type(exc).__name__}: {exc}"
            ) from exc
        self._official_eval_case["applied"] = True
        if self._official_eval_case.get("language"):
            self._language_subgoal = str(self._official_eval_case["language"])
        return obs

    def _primitive_observe_calvin_state(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        include_raw: bool = False,
    ) -> PrimitiveResult:
        output = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_ref": self._observation_ref(
                source="calvin_last_state_observation"
            ),
            "language_subgoal": self._language_subgoal,
            "observation_summary": summarize_calvin_observation(
                self._last_obs, camera_views=self.config.camera_views
            ),
            "semantic_state": summarize_calvin_semantic_state(self._env),
            "runtime": self.runtime_available(),
            "raw_observation_omitted": True,
        }
        return PrimitiveResult(name="observe_calvin_state", ok=True, output=output)

    def _primitive_observe_calvin_cameras(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        camera_view: str | None = None,
    ) -> PrimitiveResult:
        cameras = extract_calvin_camera_summaries(
            self._last_obs,
            camera_views=self.config.camera_views,
            camera_view=camera_view,
        )
        visual_observations = extract_calvin_visual_observations(
            self._last_obs,
            camera_views=self.config.camera_views,
            camera_view=camera_view,
        )
        visual_runtime = summarize_calvin_visual_runtime(cameras)
        observation_ref = self._observation_ref(source="calvin_last_camera_observation")
        artifacts: list[str] = []
        evidence_refs: list[JsonDict] = []
        for uid, modalities in visual_observations.items():
            for modality, payload in modalities.items():
                artifact_id = (
                    f"calvin:visual:{uid}:{modality}:{len(self.get_trace().artifacts)}"
                )
                source = _calvin_visual_source(self._last_obs, uid, modality)
                ref = _visual_evidence_ref(
                    "calvin", artifact_id, uid, modality, source, payload
                )
                self.get_trace().add_artifact(
                    artifact_id,
                    {**ref, "observation_ref": observation_ref, "data": payload},
                )
                evidence_refs.append(ref)
                artifacts.append(artifact_id)
        return PrimitiveResult(
            name="observe_calvin_cameras",
            ok=bool(cameras),
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "observation_ref": observation_ref,
                "cameras": cameras,
                "visual_observations": visual_observations,
                "evidence_refs": evidence_refs,
                "visual_runtime": visual_runtime,
            },
            artifacts=artifacts,
            error=None if cameras else "no_camera_observation",
        )

    def _primitive_get_calvin_language_subgoal(
        self, agent_context: JsonDict | None = None
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_calvin_language_subgoal",
            ok=self._language_subgoal is not None,
            output={
                "subgoal": self._language_subgoal,
                "agent_context": agent_context or {},
            },
            error=None
            if self._language_subgoal is not None
            else "language_subgoal_unavailable",
        )

    def _primitive_get_calvin_runtime_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        observation_summary = summarize_calvin_observation(
            self._last_obs, camera_views=self.config.camera_views
        )
        cameras = extract_calvin_camera_summaries(
            self._last_obs, camera_views=self.config.camera_views
        )
        action_schema = infer_calvin_action_schema(self._env)
        output = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_ref": self._observation_ref(
                source="calvin_runtime_context_observation"
            ),
            "language_subgoal": self._language_subgoal,
            "scene": {
                "state": observation_summary.get("state", {}),
                "observation_keys": observation_summary.get("keys", []),
                "semantic_state": summarize_calvin_semantic_state(self._env),
            },
            "cameras": cameras,
            "available_skill_schema": {
                "name": "execute_calvin_language_skill",
                "inputs": {
                    "subgoal": "str|None",
                    "language_goal": "str|None",
                    "horizon": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "grounding": "dict|None",
                },
                "action": action_schema,
                "requires_policy_backend": self._policy is None
                and not self._uses_builtin_policy_backend(),
            },
            "runtime": self.runtime_available(),
        }
        return PrimitiveResult(
            name="get_calvin_runtime_context", ok=True, output=output
        )

    def _primitive_execute_calvin_language_skill(
        self,
        subgoal: str | None = None,
        language_goal: str | None = None,
        horizon: int | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        grounding: JsonDict | None = None,
        evidence_refs: list[Any] | None = None,
    ) -> PrimitiveResult:
        provenance = self._action_provenance(
            "execute_calvin_language_skill", agent_context, evidence_refs
        )
        chosen_subgoal_input = (
            subgoal if isinstance(subgoal, str) and subgoal.strip() else language_goal
        )
        if (
            not isinstance(chosen_subgoal_input, str)
            or not chosen_subgoal_input.strip()
        ):
            return PrimitiveResult(
                name="execute_calvin_language_skill",
                ok=False,
                output={
                    "requires_agent_subgoal": True,
                    "agent_context": agent_context or {},
                    "provenance": provenance,
                },
                error="agent_subgoal_required",
            )
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 0:
            return PrimitiveResult(
                name="execute_calvin_language_skill",
                ok=False,
                output={
                    "requires_agent_horizon": True,
                    "agent_context": agent_context or {},
                    "provenance": provenance,
                },
                error="agent_horizon_required",
            )
        chosen_subgoal = chosen_subgoal_input.strip()
        payload = {
            "subgoal": chosen_subgoal,
            "language_goal": language_goal,
            "horizon": horizon,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "grounding": grounding or {},
            "provenance": provenance,
            "pre_action_observation_ref": self._observation_ref(
                source="calvin_pre_skill_observation"
            ),
        }
        if self._policy is None and not self._uses_builtin_policy_backend():
            return PrimitiveResult(
                name="execute_calvin_language_skill",
                ok=False,
                output={
                    **payload,
                    "skill_executed": False,
                    "requires_policy_backend": True,
                },
                error="policy_backend_missing",
            )
        step_outputs = []
        last_step_result: Any = None
        policy_actions: list[Any] = []
        policy_metadata: JsonDict = {}
        policy_result: JsonDict = {}
        official_task_start_info: JsonDict | None = None
        official_task_verification: JsonDict | None = None
        step_timings: list[JsonDict] = []

        if self._uses_primitive_action_backend():
            if self._uses_native_agent_backend():
                official_task_start_info = (
                    self._last_skill_start_info
                    or _safe_calvin_get_info(
                        _calvin_info_source(
                            self._env, self._official_policy_env, self._env
                        )
                    )
                )
            policy_result = self._builtin_primitive_action_policy(
                chosen_subgoal or "", payload
            )
            if policy_result.get("error"):
                return PrimitiveResult(
                    name="execute_calvin_language_skill",
                    ok=False,
                    output={
                        **payload,
                        "skill_executed": False,
                        "requires_policy_backend": False,
                    },
                    error=str(policy_result["error"]),
                )
            action = policy_result["action"]
            step_action = policy_result.get("step_action", action)
            policy_metadata = _to_builtin(policy_result.get("metadata", {}))
            execution_env = self._env
            for _ in range(max(0, int(horizon))):
                if execution_env is None or not hasattr(execution_env, "step"):
                    break
                step_result = execution_env.step(step_action)
                last_step_result = step_result
                step_outputs.append(_to_builtin(step_result))
                self._last_obs = _first_step_observation(step_result)
                policy_actions.append(_to_builtin(action))
                current_info = _extract_calvin_step_info(
                    step_result
                ) or _safe_calvin_get_info(
                    _calvin_info_source(
                        execution_env, self._official_policy_env, self._env
                    )
                )
                official_task_verification = self._evaluate_official_calvin_task(
                    chosen_subgoal,
                    official_task_start_info,
                    current_info,
                )
                if official_task_verification.get("success") or _step_done(step_result):
                    break
        elif self._uses_official_mcil_backend():
            for step_index in range(max(0, int(horizon))):
                step_timing: JsonDict = {"step_index": step_index}
                policy_started = time.monotonic()
                policy_result = self._official_mcil_policy(
                    chosen_subgoal or "",
                    payload,
                    reset_model=step_index == 0,
                )
                step_timing["policy_seconds"] = round(
                    time.monotonic() - policy_started, 3
                )
                if policy_result.get("error"):
                    step_timing["stop_phase"] = "policy"
                    step_timings.append(step_timing)
                    return PrimitiveResult(
                        name="execute_calvin_language_skill",
                        ok=False,
                        output={
                            **payload,
                            "skill_executed": False,
                            "requires_policy_backend": False,
                            "step_timings": step_timings,
                        },
                        error=str(policy_result["error"]),
                    )
                action = policy_result["action"]
                step_action = policy_result.get("step_action", action)
                policy_metadata = _to_builtin(policy_result.get("metadata", {}))
                execution_env = self._env
                if (
                    isinstance(policy_metadata, dict)
                    and policy_metadata.get("execution_env")
                    == "official_calvin_wrapper"
                    and self._official_policy_env is not None
                ):
                    execution_env = self._official_policy_env
                if execution_env is None or not hasattr(execution_env, "step"):
                    step_timing["stop_phase"] = "execution_env_step_unavailable"
                    step_timings.append(step_timing)
                    break
                if official_task_start_info is None:
                    start_info_started = time.monotonic()
                    official_task_start_info = _safe_calvin_get_info(
                        _calvin_info_source(
                            execution_env, self._official_policy_env, self._env
                        )
                    )
                    step_timing["start_info_seconds"] = round(
                        time.monotonic() - start_info_started, 3
                    )
                env_step_started = time.monotonic()
                step_result = execution_env.step(step_action)
                step_timing["env_step_seconds"] = round(
                    time.monotonic() - env_step_started, 3
                )
                last_step_result = step_result
                step_outputs.append(_to_builtin(step_result))
                self._last_obs = _first_step_observation(step_result)
                policy_actions.append(_to_builtin(action))
                current_info = _extract_calvin_step_info(
                    step_result
                ) or _safe_calvin_get_info(
                    _calvin_info_source(
                        execution_env, self._official_policy_env, self._env
                    )
                )
                hidden_check_started = time.monotonic()
                official_task_verification = self._evaluate_official_calvin_task(
                    chosen_subgoal,
                    official_task_start_info,
                    current_info,
                )
                step_timing["hidden_check_seconds"] = round(
                    time.monotonic() - hidden_check_started, 3
                )
                step_timings.append(step_timing)
                if official_task_verification.get("success") or _step_done(step_result):
                    break
        else:
            policy_result = _normalize_policy_result(
                self._policy(chosen_subgoal or "", payload)
            )  # type: ignore[misc]
            action = policy_result["action"]
            step_action = policy_result.get("step_action", action)
            policy_metadata = _to_builtin(policy_result.get("metadata", {}))
            execution_env = self._env
            for _ in range(max(0, int(horizon))):
                if execution_env is None or not hasattr(execution_env, "step"):
                    break
                step_result = execution_env.step(step_action)
                last_step_result = step_result
                step_outputs.append(_to_builtin(step_result))
                self._last_obs = _first_step_observation(step_result)
                policy_actions.append(_to_builtin(action))
        skill_executed = bool(step_outputs)
        self._last_skill_subgoal = chosen_subgoal
        self._last_skill_step_result = last_step_result
        self._last_skill_start_info = official_task_start_info
        self._last_official_verification = official_task_verification
        last_policy_action = policy_actions[-1] if policy_actions else None
        return PrimitiveResult(
            name="execute_calvin_language_skill",
            ok=skill_executed,
            output={
                **payload,
                "skill_executed": skill_executed,
                "requires_policy_backend": False,
                "steps": len(step_outputs),
                "policy_action": last_policy_action,
                "policy_actions": policy_actions,
                "policy_action_count": len(policy_actions),
                "step_timings": step_timings,
                "policy_metadata": _public_calvin_policy_metadata(policy_metadata),
                "termination": {"done": _step_done(last_step_result)},
                "last_step_summary": _public_calvin_step_summary(
                    last_step_result, camera_views=self.config.camera_views
                ),
                "post_action_observation_ref": self._observation_ref(
                    source="calvin_post_skill_observation"
                ),
            },
        )

    def _primitive_submit_calvin_action(
        self,
        action: Any,
        agent_context: JsonDict | None = None,
        evidence_refs: list[Any] | None = None,
    ) -> PrimitiveResult:
        provenance = self._action_provenance(
            "submit_calvin_action", agent_context, evidence_refs
        )
        normalized_action = _normalize_calvin_action(action)
        if normalized_action is None:
            return PrimitiveResult(
                name="submit_calvin_action",
                ok=False,
                output={
                    "stepped": False,
                    "agent_context": agent_context or {},
                    "provenance": provenance,
                },
                error="action_must_be_7d_numeric",
            )
        if self._env is None or not hasattr(self._env, "step"):
            return PrimitiveResult(
                name="submit_calvin_action",
                ok=False,
                output={
                    "stepped": False,
                    "action": normalized_action,
                    "provenance": provenance,
                    "requires_live_env": True,
                },
                error="live_env_step_unavailable",
            )
        if self._uses_native_agent_backend() and self._last_skill_start_info is None:
            self._last_skill_start_info = _safe_calvin_get_info(
                _calvin_info_source(self._env, self._official_policy_env, self._env)
            )
        step_result = self._env.step(normalized_action)
        self._last_obs = _first_step_observation(step_result)
        if self._uses_native_agent_backend():
            self._last_skill_subgoal = self._language_subgoal
            self._last_skill_step_result = step_result
            current_info = _extract_calvin_step_info(
                step_result
            ) or _safe_calvin_get_info(
                _calvin_info_source(self._env, self._official_policy_env, self._env)
            )
            self._last_official_verification = self._evaluate_official_calvin_task(
                self._last_skill_subgoal,
                self._last_skill_start_info,
                current_info,
            )
        return PrimitiveResult(
            name="submit_calvin_action",
            ok=True,
            output={
                "stepped": True,
                "action": normalized_action,
                "agent_context": agent_context or {},
                "provenance": provenance,
                "termination": {"done": _step_done(step_result)},
                "observation_summary": summarize_calvin_observation(
                    self._last_obs, camera_views=self.config.camera_views
                ),
            },
        )

    def _primitive_record_calvin_evidence(
        self,
        key: str | None = None,
        value: Any | None = None,
        evidence_type: str | None = None,
        content: Any | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence_key = key or evidence_type or "agent_evidence"
        evidence_value = value if value is not None else content
        artifact_id = f"calvin:evidence:{evidence_key}"
        self.get_trace().add_artifact(
            artifact_id,
            {
                "key": evidence_key,
                "value": _to_builtin(evidence_value),
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "accepted_aliases": {"key": key, "evidence_type": evidence_type},
            },
        )
        return PrimitiveResult(
            name="record_calvin_evidence",
            ok=True,
            output={"artifact_id": artifact_id},
            artifacts=[artifact_id],
        )

    def _primitive_card(
        self,
        name: str,
        level: str,
        input_schema: JsonDict,
        output_schema: JsonDict,
        description: str,
        preconditions: list[str] | None = None,
    ) -> PrimitiveCard:
        return PrimitiveCard(
            name=name,
            capability_tags=["w4", "calvin", "agent_native_runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=list(preconditions or []),
            cost={"primitive_calls": 1},
            failure_modes=[
                "backend_not_configured",
                "wrong_arguments",
                "runtime_dependency_missing",
            ],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")

    def _uses_builtin_policy_backend(self) -> bool:
        return self.config.policy_backend in {
            "calvin_native_agent",
            "calvin_primitive_action",
            "calvin_official_mcil",
        }

    def _uses_primitive_action_backend(self) -> bool:
        return self.config.policy_backend in {
            "calvin_native_agent",
            "calvin_primitive_action",
        }

    def _uses_native_agent_backend(self) -> bool:
        return self.config.policy_backend == "calvin_native_agent"

    def _uses_official_mcil_backend(self) -> bool:
        return self.config.policy_backend == "calvin_official_mcil"

    def _uses_official_task_verifier(self) -> bool:
        return self._uses_native_agent_backend() or self._uses_official_mcil_backend()

    def _observation_ref(self, source: str) -> JsonDict:
        return {
            "source": source,
            "trace_event_count": len(self.get_trace().events),
            "runtime_live": self.config.live,
            "env_created": self._env is not None,
            "observation_type": type(self._last_obs).__name__,
        }

    def _normalize_evidence_refs(
        self, evidence_refs: list[Any] | None
    ) -> list[JsonDict]:
        return _normalize_evidence_refs(evidence_refs, self.get_trace().artifacts)

    def _action_provenance(
        self,
        primitive: str,
        agent_context: JsonDict | None,
        evidence_refs: list[Any] | None,
    ) -> JsonDict:
        return {
            "primitive": primitive,
            "agent_context": agent_context or {},
            "evidence_refs": self._normalize_evidence_refs(evidence_refs),
        }

    def _builtin_primitive_action_policy(
        self, subgoal: str, payload: JsonDict
    ) -> JsonDict:
        grounding = (
            payload.get("grounding")
            if isinstance(payload.get("grounding"), dict)
            else {}
        )
        action = grounding.get("action") or grounding.get("policy_action")
        if action is None:
            return {"error": "primitive_action_missing"}
        normalized_action = _normalize_calvin_action(action)
        if normalized_action is None:
            return {"error": "primitive_action_must_be_7d_numeric"}
        return {
            "action": normalized_action,
            "metadata": {
                "source": "calvin_primitive_action",
                "subgoal": subgoal,
            },
        }

    def _official_mcil_policy(
        self, subgoal: str, payload: JsonDict, *, reset_model: bool = True
    ) -> JsonDict:
        readiness = official_mcil_backend_readiness(self.config)
        if not readiness["ready"]:
            return {"error": "official_mcil_backend_not_ready", "metadata": readiness}
        try:
            model = self._load_official_mcil_model(readiness)
            if reset_model and hasattr(model, "reset"):
                model.reset()
            if reset_model and subgoal == self._language_subgoal:
                eval_reset = self._reset_official_mcil_env_to_eval_initial_state()
            elif isinstance(
                self._official_eval_case, dict
            ) and self._official_eval_case.get("applied"):
                eval_reset = {
                    "applied": True,
                    "reason": "continuing_eval_reset_rollout",
                    "source": self._official_eval_case.get("source"),
                    "subtask": self._official_eval_case.get("subtask"),
                    "language": self._official_eval_case.get("language"),
                }
            else:
                eval_reset = {
                    "applied": False,
                    "reason": "subgoal_override_or_reused_rollout",
                }
            policy_obs, observation_source = self._official_mcil_observation()
            raw_action = model.step(policy_obs, subgoal)
            action = _normalize_calvin_action(raw_action)
            if action is None:
                return {
                    "error": "official_mcil_policy_failed:action_must_be_7d_numeric",
                    "metadata": {
                        **readiness,
                        "observation_source": observation_source,
                        "official_eval_reset": eval_reset,
                        "raw_action_summary": _value_summary(raw_action),
                    },
                }
        except Exception as exc:  # pragma: no cover - live dependency path.
            return {
                "error": f"official_mcil_policy_failed:{type(exc).__name__}: {exc}",
                "metadata": readiness,
            }
        use_wrapper_step = self._official_policy_env is not None and hasattr(
            self._official_policy_env, "step"
        )
        return {
            "action": action,
            "step_action": raw_action if use_wrapper_step else action,
            "metadata": {
                "source": "calvin_official_mcil",
                "subgoal": subgoal,
                "train_folder": readiness["train_folder"],
                "checkpoint": readiness["checkpoint"],
                "dataset_path": readiness["dataset_path"],
                "observation_source": observation_source,
                "execution_env": "official_calvin_wrapper"
                if use_wrapper_step
                else "raw_calvin_env",
                "official_eval_case": _public_official_eval_case(
                    self._official_eval_case
                ),
                "official_eval_reset": eval_reset,
            },
        }

    def _load_official_mcil_model(self, readiness: JsonDict) -> Any:
        if self._official_policy_model is not None:
            return self._official_policy_model
        _add_calvin_paths(self.config.calvin_root)
        _clear_hydra_global_state()
        _seed_official_calvin_eval()
        official_utils = importlib.import_module("calvin_agent.evaluation.utils")
        from omegaconf import OmegaConf

        original_load = OmegaConf.load
        torch_module = getattr(official_utils, "torch", None)
        force_cpu_policy = _calvin_force_cpu_policy(torch_module)
        original_torch_device = (
            getattr(torch_module, "device", None) if force_cpu_policy else None
        )
        torch_nn = getattr(torch_module, "nn", None) if force_cpu_policy else None
        torch_module_cls = (
            getattr(torch_nn, "Module", None) if torch_nn is not None else None
        )
        original_module_cuda = (
            getattr(torch_module_cls, "cuda", None)
            if torch_module_cls is not None
            else None
        )
        cpu_device = (
            original_torch_device("cpu") if callable(original_torch_device) else None
        )

        def load_with_runtime_overrides(path: Any) -> Any:
            loaded = original_load(path)
            _force_calvin_runtime_flags(loaded, self.config)
            return loaded

        def module_cuda_as_cpu(module_self: Any, device: Any = None) -> Any:
            if hasattr(module_self, "to") and callable(module_self.to):
                return module_self.to(cpu_device)
            return module_self

        OmegaConf.load = load_with_runtime_overrides
        if (
            force_cpu_policy
            and cpu_device is not None
            and torch_module_cls is not None
            and callable(original_module_cuda)
        ):
            torch_module_cls.cuda = module_cuda_as_cpu
        restore_egl_lookup = _install_calvin_egl_device_lookup_fallback()
        try:
            model, policy_env, _ = official_utils.get_default_model_and_env(
                Path(str(readiness["train_folder"])).expanduser(),
                str(Path(str(readiness["dataset_path"])).expanduser()),
                Path(str(readiness["checkpoint"])).expanduser(),
                env=None,
                device_id=self.config.policy_device_id,
            )
        finally:
            restore_egl_lookup()
            OmegaConf.load = original_load
            if (
                force_cpu_policy
                and cpu_device is not None
                and torch_module_cls is not None
                and callable(original_module_cuda)
            ):
                torch_module_cls.cuda = original_module_cuda
        if (
            force_cpu_policy
            and cpu_device is not None
            and hasattr(policy_env, "device")
        ):
            policy_env.device = cpu_device
        self._official_policy_model = model
        self._official_policy_env = policy_env
        self._official_policy_metadata = readiness
        return model

    def _reset_official_mcil_env_to_eval_initial_state(self) -> JsonDict:
        if not isinstance(self._official_eval_case, dict):
            return {"applied": False, "reason": "official_eval_case_unavailable"}
        if self._official_eval_case.get("applied"):
            return {
                "applied": True,
                "reason": "already_applied",
                "source": self._official_eval_case.get("source"),
                "subtask": self._official_eval_case.get("subtask"),
                "language": self._official_eval_case.get("language"),
            }
        if self._official_policy_env is None or not hasattr(
            self._official_policy_env, "reset"
        ):
            return {"applied": False, "reason": "official_policy_env_reset_unavailable"}
        try:
            _add_calvin_paths(self.config.calvin_root)
            from calvin_agent.evaluation.utils import (
                get_env_state_for_initial_condition,
            )

            robot_obs, scene_obs = get_env_state_for_initial_condition(
                self._official_eval_case["initial_state"]
            )
            obs = self._official_policy_env.reset(
                robot_obs=robot_obs, scene_obs=scene_obs
            )
        except Exception as exc:  # pragma: no cover - live dependency path.
            return {"applied": False, "reason": f"{type(exc).__name__}: {exc}"}
        self._last_obs = obs
        if self._official_eval_case.get("language"):
            self._language_subgoal = str(self._official_eval_case["language"])
        self._official_eval_case["applied"] = True
        return {
            "applied": True,
            "source": self._official_eval_case.get("source"),
            "subtask": self._official_eval_case.get("subtask"),
            "language": self._official_eval_case.get("language"),
        }

    def _official_mcil_observation(self) -> tuple[Any, str]:
        if _looks_like_official_mcil_observation(self._last_obs):
            return self._last_obs, "last_observation_already_official_mcil_tensor_obs"
        if self._official_policy_env is not None and hasattr(
            self._official_policy_env, "transform_observation"
        ):
            return self._official_policy_env.transform_observation(
                self._last_obs
            ), "official_wrapper_transform_observation"
        if self._official_policy_env is not None and hasattr(
            self._official_policy_env, "get_obs"
        ):
            return self._official_policy_env.get_obs(), "official_wrapper_get_obs"
        return self._last_obs, "last_raw_observation"

    def _evaluate_official_calvin_task(
        self,
        language_subgoal: str | None,
        start_info: JsonDict | None,
        current_info: JsonDict | None,
    ) -> JsonDict:
        if not self._uses_official_task_verifier():
            return {
                "available": False,
                "success": False,
                "source": "calvin_official_task_config",
                "task_key": None,
                "achieved_tasks": [],
                "error": "official_task_verifier_backend_not_used",
            }
        task_key = self._official_calvin_task_key(language_subgoal)
        if not task_key:
            return {
                "available": False,
                "success": False,
                "source": "calvin_official_task_config",
                "task_key": None,
                "achieved_tasks": [],
                "error": "task_key_unresolved",
            }
        if start_info is None or current_info is None:
            return {
                "available": False,
                "success": False,
                "source": "calvin_official_task_config",
                "task_key": task_key,
                "achieved_tasks": [],
                "error": "task_info_unavailable",
            }
        try:
            checker = self._load_official_calvin_task_checker()
            achieved = checker.get_task_info_for_set(
                start_info, current_info, {task_key}
            )
        except Exception as exc:  # pragma: no cover - live dependency path.
            return {
                "available": False,
                "success": False,
                "source": "calvin_official_task_config",
                "task_key": task_key,
                "achieved_tasks": [],
                "error": f"{type(exc).__name__}: {exc}",
            }
        achieved_tasks = sorted(str(item) for item in achieved)
        return {
            "available": True,
            "success": task_key in achieved_tasks,
            "source": "calvin_official_task_config",
            "task_key": task_key,
            "achieved_tasks": achieved_tasks,
            "error": None,
        }

    def _official_calvin_task_key(self, language_subgoal: str | None) -> str | None:
        if not language_subgoal:
            return None
        if language_subgoal in self._official_task_key_cache:
            return self._official_task_key_cache[language_subgoal]
        key = resolve_official_calvin_task_key(self.config, language_subgoal)
        self._official_task_key_cache[language_subgoal] = key
        return key

    def _load_official_calvin_task_checker(self) -> Any:
        if self._official_task_checker is not None:
            return self._official_task_checker
        _add_calvin_paths(self.config.calvin_root)
        task_config_path = official_calvin_task_config_path(self.config)
        if task_config_path is None or not task_config_path.exists():
            raise RuntimeError("CALVIN official task config not found")
        from omegaconf import OmegaConf

        try:
            import hydra

            cfg = OmegaConf.load(task_config_path)
            self._official_task_checker = hydra.utils.instantiate(cfg)
            return self._official_task_checker
        except Exception:
            pass

        module = importlib.import_module("calvin_env.envs.tasks")
        tasks_cls = getattr(module, "Tasks")
        cfg = OmegaConf.load(task_config_path)
        tasks = OmegaConf.to_container(cfg.get("tasks", {}), resolve=True)
        self._official_task_checker = tasks_cls(tasks)
        return self._official_task_checker


def summarize_calvin_observation(obs: Any, camera_views: list[str]) -> JsonDict:
    if obs is None:
        return {}
    if isinstance(obs, dict):
        nested_rgb = obs.get("rgb_obs") if isinstance(obs.get("rgb_obs"), dict) else {}
        nested_depth = (
            obs.get("depth_obs") if isinstance(obs.get("depth_obs"), dict) else {}
        )
        return {
            "keys": sorted(str(key) for key in obs),
            "state": {
                str(key): _to_builtin(value)
                for key, value in obs.items()
                if _looks_like_state_key(str(key))
            },
            "cameras": extract_calvin_camera_summaries(obs, camera_views=camera_views),
            "nested_camera_keys": {
                "rgb_obs": sorted(str(key) for key in nested_rgb),
                "depth_obs": sorted(str(key) for key in nested_depth),
            },
        }
    return {"type": type(obs).__name__, "repr": repr(obs)}


def summarize_calvin_semantic_state(env: Any) -> JsonDict:
    """Expose current simulator state and geometry without task predicates."""

    info = _safe_calvin_get_info(env)
    if not isinstance(info, dict):
        return {}
    robot_info = (
        info.get("robot_info") if isinstance(info.get("robot_info"), dict) else {}
    )
    scene_info = (
        info.get("scene_info") if isinstance(info.get("scene_info"), dict) else {}
    )
    result: JsonDict = {
        "robot": {
            key: _to_builtin(robot_info.get(key))
            for key in (
                "tcp_pos",
                "tcp_orn",
                "gripper_opening_width",
                "gripper_action",
                "arm_joint_states",
            )
            if robot_info.get(key) is not None
        },
        "robot_contact_count": len(robot_info.get("contacts") or ()),
        "scene": {},
    }
    for group in ("movable_objects", "doors", "buttons", "switches", "lights"):
        values = scene_info.get(group)
        if not isinstance(values, dict):
            continue
        result["scene"][group] = {
            str(name): {
                str(key): _to_builtin(value)
                for key, value in payload.items()
                if key
                in {
                    "current_pos",
                    "current_orn",
                    "current_state",
                    "joint_state",
                    "logical_state",
                    "current_lin_vel",
                    "current_ang_vel",
                }
            }
            for name, payload in values.items()
            if isinstance(payload, dict)
        }
    fixed_objects = scene_info.get("fixed_objects")
    physics = getattr(env, "p", None)
    physics_client_id = getattr(env, "cid", None)
    fixed_geometry: JsonDict = {}
    if isinstance(fixed_objects, dict) and physics is not None:
        for object_name, payload in fixed_objects.items():
            if not isinstance(payload, dict) or payload.get("uid") is None:
                continue
            object_geometry: JsonDict = {}
            for link_name, link_index in (payload.get("links") or {}).items():
                try:
                    if int(link_index) >= 0:
                        link_state = physics.getLinkState(
                            int(payload["uid"]),
                            int(link_index),
                            physicsClientId=physics_client_id,
                        )
                        position, orientation = link_state[:2]
                    else:
                        position, orientation = physics.getBasePositionAndOrientation(
                            int(payload["uid"]), physicsClientId=physics_client_id
                        )
                    aabb = physics.getAABB(
                        int(payload["uid"]),
                        int(link_index),
                        physicsClientId=physics_client_id,
                    )
                except Exception:
                    continue
                object_geometry[str(link_name)] = {
                    "position": _to_builtin(position),
                    "orientation_xyzw": _to_builtin(orientation),
                    "aabb_min": _to_builtin(aabb[0]),
                    "aabb_max": _to_builtin(aabb[1]),
                }
            if object_geometry:
                fixed_geometry[str(object_name)] = object_geometry
    result["fixed_link_geometry"] = fixed_geometry
    return result


def summarize_calvin_step_result(step_result: Any, camera_views: list[str]) -> JsonDict:
    if not isinstance(step_result, tuple):
        return {"type": type(step_result).__name__}
    info = step_result[-1] if step_result and isinstance(step_result[-1], dict) else {}
    summary: JsonDict = {
        "type": "tuple",
        "length": len(step_result),
        "done": _step_done(step_result),
        "observation": summarize_calvin_observation(
            _first_step_observation(step_result), camera_views=camera_views
        ),
        "info_keys": sorted(str(key) for key in info),
    }
    return summary


def _public_calvin_step_summary(step_result: Any, camera_views: list[str]) -> JsonDict:
    summary = summarize_calvin_step_result(step_result, camera_views=camera_views)
    summary["info_keys"] = [
        key
        for key in summary.get("info_keys", [])
        if not any(
            token in key.lower()
            for token in ("success", "reward", "checker", "predicate", "goal")
        )
    ]
    return summary


def _public_calvin_policy_metadata(metadata: JsonDict) -> JsonDict:
    forbidden = (
        "success",
        "reward",
        "checker",
        "predicate",
        "goal",
        "oracle",
        "expert",
        "evaluate_policy",
        "task_config",
        "validation_annotations",
        "subtask",
        "initial_state",
        "eval_sequence",
        "waypoint",
        "offset",
    )

    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                str(key): sanitize(item)
                for key, item in value.items()
                if not any(token in str(key).lower() for token in forbidden)
            }
        if isinstance(value, list):
            return [sanitize(item) for item in value]
        return value

    return sanitize(metadata)


def _public_official_mcil_readiness(readiness: JsonDict) -> JsonDict:
    policy_assets = {"train_folder", "train_config", "checkpoint", "dataset_path"}
    missing = [
        str(item) for item in readiness.get("missing", []) if str(item) in policy_assets
    ]
    return {
        "ready": bool(readiness.get("ready")),
        "missing": missing,
        "runtime_missing": [str(item) for item in readiness.get("runtime_missing", [])],
        "runtime_dependency": _public_official_mcil_runtime_dependency(
            readiness.get("runtime_dependency")
        ),
        "train_folder": readiness.get("train_folder"),
        "train_config": readiness.get("train_config"),
        "checkpoint": readiness.get("checkpoint"),
        "dataset_path": readiness.get("dataset_path"),
        "policy_contract": readiness.get("policy_contract"),
    }


def _public_official_mcil_runtime_dependency(runtime_dependency: Any) -> JsonDict:
    if not isinstance(runtime_dependency, dict):
        return {}
    incompatible_versions = runtime_dependency.get("incompatible_versions")
    version_warnings = runtime_dependency.get("version_warnings")
    return {
        "ready": bool(runtime_dependency.get("ready")),
        "skipped": bool(runtime_dependency.get("skipped")),
        "missing_modules": [
            str(item) for item in runtime_dependency.get("missing_modules", [])
        ],
        "versions": {
            str(key): str(value)
            for key, value in runtime_dependency.get("versions", {}).items()
            if value is not None
        },
        "incompatible_versions": incompatible_versions
        if isinstance(incompatible_versions, dict)
        else {},
        "required_min_versions": runtime_dependency.get("required_min_versions", {}),
        "recommended_min_versions": runtime_dependency.get(
            "recommended_min_versions", {}
        ),
        "version_warnings": version_warnings
        if isinstance(version_warnings, dict)
        else {},
        "policy_runtime_contract": runtime_dependency.get("policy_runtime_contract"),
    }


def extract_calvin_camera_summaries(
    obs: Any, camera_views: list[str], camera_view: str | None = None
) -> JsonDict:
    visual = extract_calvin_visual_observations(
        obs, camera_views=camera_views, camera_view=camera_view
    )
    cameras: JsonDict = {}
    for view, modalities in visual.items():
        source_modality = next(iter(modalities))
        source_key = _calvin_visual_source(obs, view, source_modality)
        cameras[view] = {
            "source_key": source_key,
            "modalities": sorted(modalities),
            "shape": _shape_of(modalities[source_modality]),
            **{
                f"{modality}_shape": _shape_of(value)
                for modality, value in modalities.items()
            },
        }
    return cameras


def extract_calvin_visual_observations(
    obs: Any, camera_views: list[str], camera_view: str | None = None
) -> JsonDict:
    if not isinstance(obs, dict):
        return {}
    wanted = [camera_view] if camera_view else camera_views
    observations: JsonDict = {}
    for view in wanted:
        if view is None:
            continue
        for modality in ("rgb", "depth", "mask"):
            value = _calvin_visual_value(obs, view, modality)
            if value is not None:
                observations.setdefault(view, {})[modality] = _to_builtin(value)
    return observations


def _calvin_visual_value(obs: JsonDict, view: str, modality: str) -> Any:
    aliases = (
        ("seg", "mask", "segmentation")
        if modality in {"mask", "segmentation"}
        else (modality,)
    )
    for alias in aliases:
        nested = obs.get(f"{alias}_obs")
        key = f"{alias}_{view}"
        if isinstance(nested, dict) and key in nested:
            return nested[key]
        if key in obs:
            return obs[key]
        slash_key = f"{alias}_obs/{view}"
        if slash_key in obs:
            return obs[slash_key]
    return None


def _calvin_visual_source(obs: Any, view: str, modality: str) -> str:
    if not isinstance(obs, dict):
        return f"{modality}_{view}"
    aliases = (
        ("seg", "mask", "segmentation")
        if modality in {"mask", "segmentation"}
        else (modality,)
    )
    for alias in aliases:
        key = f"{alias}_{view}"
        if isinstance(obs.get(f"{alias}_obs"), dict) and key in obs[f"{alias}_obs"]:
            return f"{alias}_obs.{key}"
        if key in obs:
            return key
        slash_key = f"{alias}_obs/{view}"
        if slash_key in obs:
            return slash_key
    return f"{modality}_{view}"


def _visual_evidence_ref(
    benchmark_id: str,
    artifact_id: str,
    camera: str,
    modality: str,
    source: str,
    data: Any,
) -> JsonDict:
    return {
        "artifact_id": artifact_id,
        "evidence_type": "visual_observation",
        "benchmark_id": benchmark_id,
        "camera": camera,
        "modality": modality,
        "source": source,
        "shape": _shape_of(data),
    }


def _normalize_evidence_refs(
    evidence_refs: list[Any] | None, artifacts: dict[str, JsonDict]
) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for value in evidence_refs or []:
        artifact_id = (
            value
            if isinstance(value, str)
            else value.get("artifact_id")
            if isinstance(value, dict)
            else None
        )
        if not isinstance(artifact_id, str) or artifact_id not in artifacts:
            continue
        artifact = artifacts[artifact_id]
        normalized.append(
            {
                key: artifact.get(key)
                for key in (
                    "artifact_id",
                    "evidence_type",
                    "benchmark_id",
                    "camera",
                    "modality",
                    "source",
                    "shape",
                )
                if artifact.get(key) is not None
            }
        )
    return normalized


def summarize_calvin_visual_runtime(cameras: JsonDict) -> JsonDict:
    modalities_by_camera = {
        str(uid): sorted(str(item) for item in payload.get("modalities", []))
        for uid, payload in cameras.items()
        if isinstance(payload, dict)
    }
    return {
        "visual_ready": bool(cameras),
        "camera_views": sorted(str(uid) for uid in cameras),
        "modalities_by_camera": modalities_by_camera,
        "has_rgb": any(
            "rgb" in modalities for modalities in modalities_by_camera.values()
        ),
        "has_depth": any(
            "depth" in modalities for modalities in modalities_by_camera.values()
        ),
        "has_segmentation": any(
            "mask" in modalities or "segmentation" in modalities
            for modalities in modalities_by_camera.values()
        ),
    }


def _extract_language(obs: Any) -> str | None:
    if isinstance(obs, dict):
        for key in ("language", "lang", "instruction", "subgoal"):
            if key in obs and obs[key] is not None:
                return str(obs[key])
    return None


def _native_official_eval_case(config: CALVINRuntimeConfig) -> JsonDict | None:
    """Build one checkpoint-free official CALVIN task instance.

    The task key, language annotation, environment reset schema, and hidden
    success predicate all come from the pinned CALVIN checkout.  Only policy
    inference is omitted: the evaluated agent supplies the 7D actions.
    """

    task_key = str(config.native_task_key or "").strip()
    if not task_key:
        return None
    task_config = official_calvin_task_config_path(config)
    if task_config is None or not task_config.is_file():
        return None
    try:
        from omegaconf import OmegaConf

        task_cfg = OmegaConf.load(task_config)
        if task_key not in {str(key) for key in task_cfg.get("tasks", {}).keys()}:
            return None
    except Exception:
        return None
    language = _official_validation_language_for_task(config, task_key) or task_key
    return {
        "source": "calvin_native_agent_representative_task",
        "initial_state_source": (
            "calvin_agent.evaluation.utils.get_env_state_for_initial_condition_contract"
        ),
        "initial_state": _to_builtin(dict(config.native_initial_state)),
        "eval_sequence": [task_key],
        "subtask": task_key,
        "language": language,
        "applied": False,
    }


def _original_calvin_eval_state(config: CALVINRuntimeConfig, state: JsonDict) -> tuple[Any, Any]:
    # Load unchanged upstream reset utilities without importing its policy model.
    import numpy as np
    import pyhash

    root = Path(config.calvin_root) if config.calvin_root else get_project_paths().external_upstream("calvin")
    source = root / "calvin_models/calvin_agent/evaluation/utils.py"
    nodes = [node for node in ast.parse(source.read_text()).body
             if isinstance(node, ast.FunctionDef)
             and node.name in {"temp_seed", "get_env_state_for_initial_condition"}]
    if len(nodes) != 2:
        raise RuntimeError("Pinned CALVIN reset helpers are missing")
    namespace = {"np": np, "pi": np.pi, "contextlib": contextlib, "hasher": pyhash.fnv1_32()}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["get_env_state_for_initial_condition"](state)


def _calvin_native_eval_state(initial_condition: JsonDict) -> tuple[Any, Any]:
    """Translate CALVIN's public evaluation-state schema to env.reset arrays.

    This is the small, checkpoint-independent part of the upstream evaluation
    utility.  Object yaw values are fixed inside the same official sampling
    range because they are irrelevant to the light-switch representative case.
    """

    import numpy as np

    required = {
        "slider",
        "drawer",
        "lightbulb",
        "led",
        "red_block",
        "blue_block",
        "pink_block",
        "grasped",
    }
    missing = sorted(required - set(initial_condition))
    if missing:
        raise ValueError(
            "CALVIN native initial state is missing: " + ", ".join(missing)
        )
    robot_obs = np.asarray(
        [
            0.02586889,
            -0.2313129,
            0.5712808,
            3.09045411,
            -0.02908596,
            1.50013585,
            0.07999963,
            -1.21779124,
            1.03987629,
            2.11978254,
            -2.34205014,
            -0.87015899,
            1.64119093,
            0.55344928,
            1.0,
        ],
        dtype=np.float64,
    )
    block_slider_left = np.asarray([-0.240851662, 0.0924044687, 0.460990009])
    block_slider_right = np.asarray([0.070341633, 0.0924044687, 0.460990009])
    block_table = (
        np.asarray([0.0500000896, -0.120000177, 0.459990009]),
        np.asarray([0.229995412, -0.11999514, 0.45999001]),
    )

    def block_position(value: Any, table_index: int) -> Any:
        if value == "slider_right":
            return block_slider_right
        if value == "slider_left":
            return block_slider_left
        return block_table[table_index]

    scene_obs = np.zeros(24, dtype=np.float64)
    if initial_condition["slider"] == "left":
        scene_obs[0] = 0.28
    if initial_condition["drawer"] == "open":
        scene_obs[1] = 0.22
    if int(initial_condition["lightbulb"]) == 1:
        scene_obs[3] = 0.088
    scene_obs[4] = int(initial_condition["lightbulb"])
    scene_obs[5] = int(initial_condition["led"])
    scene_obs[6:9] = block_position(initial_condition["red_block"], 0)
    scene_obs[11] = 1.45
    scene_obs[12:15] = block_position(initial_condition["blue_block"], 1)
    scene_obs[17] = 1.57
    scene_obs[18:21] = block_position(initial_condition["pink_block"], 1)
    scene_obs[23] = 1.69
    return robot_obs, scene_obs


def _first_official_mcil_eval_case(config: CALVINRuntimeConfig) -> JsonDict | None:
    if config.policy_backend != "calvin_official_mcil":
        return None
    try:
        _add_calvin_paths(config.calvin_root)
        from calvin_agent.evaluation.multistep_sequences import get_sequences

        num_sequences = max(
            1,
            _env_int(
                "CALVIN_OFFICIAL_EVAL_NUM_SEQUENCES", config.official_eval_num_sequences
            ),
        )
        sequence_index = max(
            0,
            _env_int(
                "CALVIN_OFFICIAL_EVAL_SEQUENCE_INDEX",
                config.official_eval_sequence_index,
            ),
        )
        num_workers = _env_optional_int(
            "CALVIN_OFFICIAL_EVAL_NUM_WORKERS", config.official_eval_num_workers
        )
        sequences = get_sequences(num_sequences, num_workers=num_workers)
        if not sequences:
            return None
        selected_index = min(sequence_index, len(sequences) - 1)
        initial_state, eval_sequence = sequences[selected_index]
        subtask = str(eval_sequence[0]) if eval_sequence else None
        if not subtask:
            return None
        language = _official_validation_language_for_task(config, subtask) or subtask
        return {
            "source": "calvin_agent.evaluation.multistep_sequences.get_sequences",
            "num_sequences": num_sequences,
            "sequence_index": selected_index,
            "num_workers": num_workers,
            "initial_state": _to_builtin(initial_state),
            "eval_sequence": [str(item) for item in eval_sequence],
            "subtask": subtask,
            "language": language,
            "applied": False,
        }
    except Exception:
        return None


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def _env_optional_int(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        return default


def _official_validation_language_for_task(
    config: CALVINRuntimeConfig, task_key: str
) -> str | None:
    annotations_path = official_calvin_validation_annotations_path(config)
    if annotations_path is None or not annotations_path.exists():
        return None
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(annotations_path)
        annotations = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        return None
    if not isinstance(annotations, dict) or task_key not in annotations:
        return None
    utterances = annotations[task_key]
    utterance_list = utterances if isinstance(utterances, list) else [utterances]
    return str(utterance_list[0]) if utterance_list else None


def _public_official_eval_case(eval_case: JsonDict | None) -> JsonDict | None:
    if not isinstance(eval_case, dict):
        return None
    return {
        "source": eval_case.get("source"),
        "language": eval_case.get("language"),
        "applied": bool(eval_case.get("applied")),
    }


def _looks_like_state_key(key: str) -> bool:
    return any(
        token in key for token in ("robot", "scene", "state", "proprio", "actions")
    )


def _first_step_observation(step_result: Any) -> Any:
    if isinstance(step_result, tuple) and step_result:
        return step_result[0]
    return step_result


def _step_done(step_result: Any) -> bool:
    if isinstance(step_result, tuple):
        if len(step_result) >= 5:
            return bool(step_result[2] or step_result[3])
        if len(step_result) >= 4:
            return bool(step_result[2])
    return False


def _normalize_calvin_action(action: Any) -> list[float] | None:
    if hasattr(action, "detach") and callable(action.detach):
        action = action.detach()
    if hasattr(action, "cpu") and callable(action.cpu):
        action = action.cpu()
    if hasattr(action, "tolist"):
        action = action.tolist()
    while (
        isinstance(action, (list, tuple))
        and len(action) == 1
        and isinstance(action[0], (list, tuple))
    ):
        action = action[0]
    if not isinstance(action, (list, tuple)) or len(action) != 7:
        return None
    try:
        return [float(value) for value in action]
    except (TypeError, ValueError):
        return None


def _looks_like_official_mcil_observation(obs: Any) -> bool:
    if not isinstance(obs, dict):
        return False
    robot_shape = _shape_of(obs.get("robot_obs"))
    if robot_shape is not None and len(robot_shape) >= 3:
        return True
    rgb_obs = obs.get("rgb_obs")
    if isinstance(rgb_obs, dict):
        for value in rgb_obs.values():
            shape = _shape_of(value)
            if shape is not None and len(shape) >= 5:
                return True
    return False


def make_official_calvin_env(config: CALVINRuntimeConfig) -> Any:
    """Create the official CALVIN PyBullet env from an installed CALVIN checkout.

    The official helper loads ``<dataset_root>/.hydra/merged_config.yaml`` via
    ``calvin_env.envs.play_table_env.get_env``. This keeps the harness on the
    real PyBullet/config path and fails loudly when the package or debug dataset
    is missing.
    """
    _add_calvin_paths(config.calvin_root)
    dataset_root = _resolve_dataset_root(config)
    if dataset_root is None:
        raise RuntimeError(
            "CALVIN dataset_root is required for live env construction. Set --dataset-root, "
            "CALVIN_DATASET_ROOT, or CALVIN_ROOT pointing to a checkout with dataset/task_D_D/training."
        )
    merged_config = Path(dataset_root) / ".hydra" / "merged_config.yaml"
    if not merged_config.exists():
        raise RuntimeError(f"CALVIN dataset config missing: {merged_config}")
    _install_collections_abc_compat()
    try:
        module = importlib.import_module("calvin_env.envs.play_table_env")
    except Exception as exc:  # pragma: no cover - dependency specific.
        raise RuntimeError(
            f"CALVIN import failed for calvin_env.envs.play_table_env: {type(exc).__name__}: {exc}"
        ) from exc
    restore_git_hash = _install_calvin_git_hash_fallback(module)
    try:
        return _call_calvin_get_env(module, dataset_root, merged_config, config)
    except Exception as exc:  # pragma: no cover - dependency specific.
        raise RuntimeError(
            f"CALVIN official get_env failed for {dataset_root}: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        restore_git_hash()


def _install_collections_abc_compat() -> None:
    """Provide removed collections aliases needed by older CALVIN dependencies."""
    for name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))


def _install_calvin_git_hash_fallback(play_table_module: Any) -> Callable[[], None]:
    """Keep CALVIN env reset from failing on non-task Git metadata checks."""

    try:
        utils_module = importlib.import_module("calvin_env.utils.utils")
    except Exception:
        utils_module = None
    original_play_table = getattr(play_table_module, "get_git_commit_hash", None)
    original_utils = (
        getattr(utils_module, "get_git_commit_hash", None)
        if utils_module is not None
        else None
    )

    def get_git_commit_hash_with_fallback(repo_path: Path) -> str:
        if callable(original_utils):
            try:
                value = original_utils(repo_path)
            except Exception:
                value = None
            if isinstance(value, str) and value.strip():
                return value
        return "0" * 40

    play_table_module.get_git_commit_hash = get_git_commit_hash_with_fallback
    if utils_module is not None:
        utils_module.get_git_commit_hash = get_git_commit_hash_with_fallback

    def restore() -> None:
        if original_play_table is not None:
            play_table_module.get_git_commit_hash = original_play_table
        if utils_module is not None and original_utils is not None:
            utils_module.get_git_commit_hash = original_utils

    return restore


def _install_calvin_egl_device_lookup_fallback() -> Callable[[], None]:
    """Tolerate CALVIN's brittle EGL helper output parser on Slurm nodes.

    The upstream wrapper already falls back to EGL device 0 when no CUDA/EGL
    mapping is found, but its bundled ``EGL_options.o`` parser can raise
    ``IndexError`` or ``ValueError`` before that fallback is reached. Keep the
    official wrapper and environment construction intact while extending only
    that existing compatibility fallback to malformed helper output.
    """

    try:
        wrapper_module = importlib.import_module(
            "calvin_agent.wrappers.calvin_env_wrapper"
        )
    except Exception:
        return lambda: None
    original_lookup = getattr(wrapper_module, "get_egl_device_id", None)
    if not callable(original_lookup):
        return lambda: None

    def lookup_with_parser_fallback(cuda_id: int) -> int:
        try:
            return int(original_lookup(cuda_id))
        except (IndexError, ValueError, UnicodeDecodeError):
            configured = (
                os.environ.get("EGL_VISIBLE_DEVICES", "").split(",", 1)[0].strip()
            )
            try:
                return int(configured) if configured else 0
            except ValueError:
                return 0

    wrapper_module.get_egl_device_id = lookup_with_parser_fallback

    def restore() -> None:
        wrapper_module.get_egl_device_id = original_lookup

    return restore


def _call_calvin_get_env(
    module: Any, dataset_root: Path, merged_config: Path, config: CALVINRuntimeConfig
) -> Any:
    """Call upstream get_env while making the harness EGL flag effective.

    CALVIN's get_env accepts **kwargs but the upstream implementation does not
    pass use_egl through to Hydra. Patch only the in-memory OmegaConf load for
    this dataset config so the checked-out official helper still owns env
    construction while the harness can select DIRECT vs EGL deterministically.
    """
    from omegaconf import OmegaConf

    original_load = OmegaConf.load

    def load_with_runtime_overrides(path: Any) -> Any:
        loaded = original_load(path)
        try:
            same_config = Path(path).expanduser().resolve() == merged_config.resolve()
        except TypeError:
            same_config = False
        if same_config:
            _force_calvin_runtime_flags(loaded, config)
        return loaded

    kwargs = dict(config.env_kwargs)
    kwargs.setdefault("use_egl", config.use_egl)
    OmegaConf.load = load_with_runtime_overrides
    try:
        return module.get_env(
            str(dataset_root),
            obs_space=_calvin_obs_space(config.camera_views),
            show_gui=config.show_gui,
            **kwargs,
        )
    finally:
        OmegaConf.load = original_load


def _force_calvin_runtime_flags(
    loaded_config: Any, config: CALVINRuntimeConfig
) -> None:
    try:
        from omegaconf import OmegaConf

        if OmegaConf.select(loaded_config, "env") is None:
            return
        OmegaConf.update(
            loaded_config, "env.use_egl", bool(config.use_egl), force_add=True
        )
        OmegaConf.update(
            loaded_config, "env.show_gui", bool(config.show_gui), force_add=True
        )
    except Exception:
        if hasattr(loaded_config, "env"):
            loaded_config.env.use_egl = config.use_egl
            loaded_config.env.show_gui = config.show_gui


def _calvin_force_cpu_policy(torch_module: Any) -> bool:
    cuda = getattr(torch_module, "cuda", None)
    is_available = getattr(cuda, "is_available", None)
    if not callable(is_available):
        return False
    try:
        return not bool(is_available())
    except Exception:
        return True


def _calvin_obs_space(camera_views: list[str]) -> JsonDict | None:
    if not camera_views:
        return None
    return {
        "rgb_obs": [f"rgb_{view}" for view in camera_views],
        "depth_obs": [f"depth_{view}" for view in camera_views],
    }


def infer_calvin_action_schema(env: Any) -> JsonDict:
    robot = getattr(env, "robot", None)
    relative_control = {
        "order": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "normalized_range": [-1.0, 1.0],
        "max_position_delta_m_per_action": _to_builtin(
            getattr(robot, "max_rel_pos", 0.02)
        ),
        "max_orientation_delta_rad_per_action": _to_builtin(
            getattr(robot, "max_rel_orn", 0.05)
        ),
        "gripper": {"open": 1.0, "close": -1.0},
    }
    action_space = getattr(env, "action_space", None)
    if action_space is not None:
        return {
            "source": "env.action_space",
            "shape": _shape_of(action_space),
            "repr": repr(action_space),
            "relative_cartesian_control": relative_control,
        }
    return {
        "source": "calvin_default_cartesian_rel",
        "shape": [7],
        "dtype": "float",
        "example_noop": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "relative_cartesian_control": relative_control,
    }


def official_mcil_backend_readiness(config: CALVINRuntimeConfig) -> JsonDict:
    _add_calvin_paths(config.calvin_root)
    root = _resolve_calvin_root(config.calvin_root)
    agent_root = root / "calvin_models" / "calvin_agent" if root is not None else None
    train_folder = (
        Path(config.policy_train_folder).expanduser()
        if config.policy_train_folder
        else (
            agent_root / "D_D_static_rgb_baseline" if agent_root is not None else None
        )
    )
    checkpoint = (
        Path(config.policy_checkpoint).expanduser()
        if config.policy_checkpoint
        else (train_folder / "mcil_baseline.ckpt" if train_folder is not None else None)
    )
    dataset_path = (
        Path(config.policy_dataset_path).expanduser()
        if config.policy_dataset_path
        else _policy_dataset_path(config, root)
    )
    train_config = (
        train_folder / ".hydra" / "config.yaml" if train_folder is not None else None
    )
    evaluator = (
        agent_root / "evaluation" / "evaluate_policy.py"
        if agent_root is not None
        else None
    )
    task_config = official_calvin_task_config_path(config, root=root)
    validation_annotations = official_calvin_validation_annotations_path(
        config, root=root
    )
    missing = []
    for key, path in (
        ("train_folder", train_folder),
        ("train_config", train_config),
        ("checkpoint", checkpoint),
        ("dataset_path", dataset_path),
        ("evaluator", evaluator),
    ):
        if path is None or not path.exists():
            missing.append(key)
    runtime_dependency = (
        official_mcil_runtime_dependency_readiness()
        if config.policy_runtime_dependency_check
        else {
            "ready": True,
            "skipped": True,
            "missing_modules": [],
            "versions": {},
            "incompatible_versions": {},
            "required_min_versions": {"torch": "1.13"},
            "recommended_min_versions": {"torch": "2.1"},
            "version_warnings": {},
            "policy_runtime_contract": "official_mcil_dependency_check_disabled_for_test_or_external_probe",
        }
    )
    runtime_missing = _official_mcil_runtime_missing(runtime_dependency)
    return {
        "ready": not missing and bool(runtime_dependency.get("ready")),
        "missing": missing,
        "runtime_missing": runtime_missing,
        "runtime_dependency": runtime_dependency,
        "calvin_root": str(root) if root is not None else None,
        "train_folder": str(train_folder) if train_folder is not None else None,
        "train_config": str(train_config) if train_config is not None else None,
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "dataset_path": str(dataset_path) if dataset_path is not None else None,
        "evaluator": str(evaluator) if evaluator is not None else None,
        "task_config": str(task_config) if task_config is not None else None,
        "validation_annotations": str(validation_annotations)
        if validation_annotations is not None
        else None,
        "policy_contract": "calvin_agent.evaluation.utils.get_default_model_and_env(...).model.step(obs, language_goal)",
        "agent_safe": True,
        "leakage_boundary": {
            "evaluate_policy_exposed_as_agent_primitive": False,
            "task_oracle_exposed_as_agent_primitive": False,
            "dataset_action_replay_exposed_as_agent_primitive": False,
        },
    }


def official_mcil_runtime_dependency_readiness() -> JsonDict:
    """Report whether the selected Python can load the official CALVIN MCIL stack."""

    required_modules = {
        "torch": "torch",
        "torchvision": "torchvision",
        "pytorch_lightning": "pytorch-lightning",
        "pyhash": "pyhash",
        "wandb": "wandb",
        "calvin_env": "calvin_env",
        "calvin_agent": "calvin_agent",
    }
    required_min_versions = {"torch": "1.13"}
    recommended_min_versions = {"torch": "2.1"}
    missing_modules: list[str] = []
    versions: JsonDict = {}
    version_errors: JsonDict = {}
    for module_name, distribution_name in required_modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing_modules.append(module_name)
            continue
        try:
            versions[module_name] = importlib_metadata.version(distribution_name)
        except importlib_metadata.PackageNotFoundError:
            versions[module_name] = None
        except Exception as exc:  # pragma: no cover - dependency metadata edge case.
            version_errors[module_name] = f"{type(exc).__name__}: {exc}"
    incompatible_versions: JsonDict = {}
    version_warnings: JsonDict = {}
    torch_version = versions.get("torch")
    if isinstance(torch_version, str) and _version_tuple(
        torch_version
    ) < _version_tuple(required_min_versions["torch"]):
        incompatible_versions["torch"] = {
            "found": torch_version,
            "required_min": required_min_versions["torch"],
            "reason": "official CALVIN MCIL policy loading has not been validated below this version",
        }
    elif isinstance(torch_version, str) and _version_tuple(
        torch_version
    ) < _version_tuple(recommended_min_versions["torch"]):
        version_warnings["torch"] = {
            "found": torch_version,
            "recommended_min": recommended_min_versions["torch"],
            "reason": "some optional libraries warn below this version, but local official MCIL strict smoke has loaded and executed with torch 1.13.x",
            "evidence": "artifacts/environment-preflight/calvin/compute-render-probe-v3.json",
        }
    return {
        "ready": not missing_modules and not incompatible_versions,
        "skipped": False,
        "missing_modules": missing_modules,
        "versions": versions,
        "version_errors": version_errors,
        "incompatible_versions": incompatible_versions,
        "required_min_versions": required_min_versions,
        "recommended_min_versions": recommended_min_versions,
        "version_warnings": version_warnings,
        "python_executable": sys.executable,
        "policy_runtime_contract": "torch>=1.13 and the complete official MCIL module surface is discoverable; strict readiness additionally requires the sealed official model-step compute probe",
    }


def _official_mcil_runtime_missing(runtime_dependency: JsonDict) -> list[str]:
    missing = [
        f"module:{item}" for item in runtime_dependency.get("missing_modules", [])
    ]
    incompatible_versions = runtime_dependency.get("incompatible_versions")
    if isinstance(incompatible_versions, dict):
        missing.extend(f"version:{key}" for key in sorted(incompatible_versions))
    return missing


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.split("+", 1)[0].split("-", 1)[0].split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def official_calvin_task_config_path(
    config: CALVINRuntimeConfig, root: Path | None = None
) -> Path | None:
    if root is None:
        root = _resolve_calvin_root(config.calvin_root)
    candidates: list[Path] = []
    if root is not None:
        candidates.extend(
            [
                root
                / "calvin_models"
                / "conf"
                / "callbacks"
                / "rollout"
                / "tasks"
                / "new_playtable_tasks.yaml",
                root / "calvin_env" / "conf" / "tasks" / "new_playtable_tasks.yaml",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def official_calvin_validation_annotations_path(
    config: CALVINRuntimeConfig, root: Path | None = None
) -> Path | None:
    if root is None:
        root = _resolve_calvin_root(config.calvin_root)
    candidates: list[Path] = []
    if root is not None:
        candidates.append(
            root
            / "calvin_models"
            / "conf"
            / "annotations"
            / "new_playtable_validation.yaml"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def resolve_official_calvin_task_key(
    config: CALVINRuntimeConfig, language_subgoal: str
) -> str | None:
    normalized_language = _normalize_text_key(language_subgoal)
    task_config_path = official_calvin_task_config_path(config)
    validation_annotations_path = official_calvin_validation_annotations_path(config)
    try:
        from omegaconf import OmegaConf

        if task_config_path is not None and task_config_path.exists():
            task_cfg = OmegaConf.load(task_config_path)
            task_names = {str(key) for key in task_cfg.get("tasks", {}).keys()}
            if language_subgoal in task_names:
                return language_subgoal
            for task_name in task_names:
                if _normalize_text_key(task_name) == normalized_language:
                    return task_name
        if (
            validation_annotations_path is not None
            and validation_annotations_path.exists()
        ):
            annotations_cfg = OmegaConf.load(validation_annotations_path)
            annotations = OmegaConf.to_container(annotations_cfg, resolve=True)
            if isinstance(annotations, dict):
                for task_name, utterances in annotations.items():
                    if str(task_name) == language_subgoal:
                        return str(task_name)
                    utterance_list = (
                        utterances if isinstance(utterances, list) else [utterances]
                    )
                    if any(
                        _normalize_text_key(str(utterance)) == normalized_language
                        for utterance in utterance_list
                    ):
                        return str(task_name)
    except Exception:
        return None
    return None


def _normalize_text_key(value: str) -> str:
    return " ".join(value.strip().lower().replace("_", " ").split())


def _calvin_info_source(
    execution_env: Any, official_policy_env: Any | None, raw_env: Any | None
) -> Any | None:
    candidates = [
        execution_env,
        official_policy_env,
        getattr(official_policy_env, "env", None),
        raw_env,
    ]
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, "get_info"):
            return candidate
    return None


def _safe_calvin_get_info(source: Any | None) -> JsonDict | None:
    if source is None or not hasattr(source, "get_info"):
        return None
    try:
        info = source.get_info()
    except Exception:
        return None
    return info if isinstance(info, dict) else None


def _extract_calvin_step_info(step_result: Any) -> JsonDict | None:
    if not isinstance(step_result, tuple) or not step_result:
        return None
    info = step_result[-1]
    return info if isinstance(info, dict) else None


def _clear_hydra_global_state() -> None:
    """Allow CALVIN's env and model loaders to install different Hydra roots."""
    try:
        from hydra.core.global_hydra import GlobalHydra

        instance = GlobalHydra.instance()
        if instance.is_initialized():
            instance.clear()
    except Exception:
        return


def _seed_official_calvin_eval(seed: int = 0) -> None:
    """Mirror upstream CALVIN evaluation seeding before model/env construction."""
    try:
        from pytorch_lightning import seed_everything

        seed_everything(seed, workers=True)
    except Exception:
        try:
            import random
            import numpy as np

            random.seed(seed)
            np.random.seed(seed)
        except Exception:
            return


def _policy_dataset_path(config: CALVINRuntimeConfig, root: Path | None) -> Path | None:
    if config.dataset_root:
        dataset = Path(config.dataset_root).expanduser()
        return dataset.parent if dataset.name in {"training", "validation"} else dataset
    if root is None:
        return None
    return root / "dataset" / "task_D_D"


def _normalize_policy_result(value: Any) -> JsonDict:
    if isinstance(value, dict):
        action = value.get("action", value.get("policy_action"))
        if action is None:
            action = value.get("actions")
        return {
            "action": action,
            "metadata": value.get("metadata", {}),
        }
    return {"action": value, "metadata": {}}


def _add_calvin_paths(calvin_root: str | None) -> None:
    root = _resolve_calvin_root(calvin_root)
    if root is None:
        return
    os.environ.setdefault(
        "CALVIN_ASSET_DATA_ROOT", str((root / "calvin_env" / "data").resolve())
    )
    candidates = [
        root,
        root / "calvin_env",
        root / "calvin_env" / "tacto",
        root / "calvin_models",
    ]
    for path in candidates:
        text = str(path)
        if path.exists() and text not in sys.path:
            sys.path.insert(0, text)
    _ensure_git_safe_directories(candidates)


def _ensure_git_safe_directories(paths: list[Path]) -> None:
    try:
        count = int(os.environ.get("GIT_CONFIG_COUNT", "0") or "0")
    except ValueError:
        count = 0
    existing = {
        os.environ.get(f"GIT_CONFIG_VALUE_{index}", "")
        for index in range(count)
        if os.environ.get(f"GIT_CONFIG_KEY_{index}") == "safe.directory"
    }
    if "*" in existing:
        return
    next_index = count
    for path in paths:
        if not path.exists():
            continue
        text = str(path.resolve())
        if text in existing:
            continue
        os.environ[f"GIT_CONFIG_KEY_{next_index}"] = "safe.directory"
        os.environ[f"GIT_CONFIG_VALUE_{next_index}"] = text
        existing.add(text)
        next_index += 1
    if next_index != count:
        os.environ["GIT_CONFIG_COUNT"] = str(next_index)


def _resolve_calvin_root(calvin_root: str | None) -> Path | None:
    configured = calvin_root or os.environ.get("CALVIN_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (get_project_paths().external_upstream("calvin"),):
        if (candidate / "calvin_env").is_dir() and (
            candidate / "calvin_models"
        ).is_dir():
            return candidate.resolve()
    return None


def _resolve_dataset_root(config: CALVINRuntimeConfig) -> Path | None:
    explicit = config.dataset_root or os.environ.get("CALVIN_DATASET_ROOT")
    if explicit:
        return Path(explicit).expanduser().resolve()
    root = _resolve_calvin_root(config.calvin_root)
    if root is None:
        return None
    dataset_root = root / "dataset"
    for relative in (
        "calvin_debug_dataset/training",
        "task_D_D/training",
        "task_D_D/validation",
        "debug/training",
        "training",
    ):
        candidate = dataset_root / relative
        if (candidate / ".hydra" / "merged_config.yaml").exists():
            return candidate.resolve()
    return None


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return [int(dim) for dim in shape]
    inferred: list[int] = []
    nested = value
    while isinstance(nested, list):
        inferred.append(len(nested))
        if not nested:
            break
        nested = nested[0]
    return inferred or None


def _value_summary(value: Any) -> JsonDict:
    summary: JsonDict = {"type": type(value).__name__}
    shape = _shape_of(value)
    if shape is not None:
        summary["shape"] = shape
    if isinstance(value, (list, tuple)):
        summary["length"] = len(value)
    return summary


def _to_builtin(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
