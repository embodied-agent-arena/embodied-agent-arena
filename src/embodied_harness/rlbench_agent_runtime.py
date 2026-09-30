from __future__ import annotations

from dataclasses import asdict, dataclass, field
import ast
import importlib
import importlib.util
import os
from pathlib import Path
from typing import Any, Callable

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]
SkillBackend = Callable[[str, dict[str, Any]], PrimitiveResult | JsonDict]

_PROJECT_PATHS = get_project_paths()
RLBENCH_REQUIRED_LIVE_MODULES = ("rlbench", "pyrep")


def _repo_local_rlbench_python() -> Path | None:
    candidates = (
        _PROJECT_PATHS.external_environment("rlbench") / "bin/python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return None


def _repo_local_coppeliasim_root() -> Path | None:
    candidates = (
        _PROJECT_PATHS.external_assets("rlbench") / "CoppeliaSim",
    )
    for candidate in candidates:
        if _looks_like_coppeliasim_root(candidate):
            return candidate.absolute()
    return None


def _rlbench_live_module_status() -> dict[str, bool]:
    status: dict[str, bool] = {}
    for module_name in RLBENCH_REQUIRED_LIVE_MODULES:
        try:
            status[module_name] = importlib.util.find_spec(module_name) is not None
        except Exception:
            status[module_name] = False
    return status


def _coppeliasim_launcher_env(coppelia_root: str | Path | None) -> JsonDict | None:
    if coppelia_root is None:
        return None
    root = str(Path(coppelia_root).absolute())
    return {
        "COPPELIASIM_ROOT": root,
        "QT_QPA_PLATFORM_PLUGIN_PATH": root,
        "LD_LIBRARY_PATH": root,
    }


@dataclass(slots=True)
class RLBenchRuntimeConfig:
    task_name: str = "ReachTarget"
    observation_mode: str = "vision"
    action_mode: str = "MoveArmThenGripper(JointVelocity, Discrete)"
    live: bool = True
    dataset_root: str | None = None
    coppeliasim_root: str | None = None
    headless: bool = True
    env_kwargs: JsonDict = field(default_factory=dict)
    action_skill_backend: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RLBenchAgentRuntimeBackend(EmbodiedBackend):
    """AI-native RLBench runtime adapter.

    The adapter owns reset, task loading, and harness-side verification. Agent
    primitives accept prompt/query/context, expose observations and action-skill
    hooks, and intentionally do not expose demonstrations, waypoints, or success
    checker calls as primitives.
    """

    def __init__(
        self,
        config: RLBenchRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
        skill_backend: SkillBackend | None = None,
    ) -> None:
        self.config = config or RLBenchRuntimeConfig()
        self._env_factory = env_factory
        self._skill_backend = skill_backend
        self._env: Any | None = None
        self._task: Any | None = None
        self._last_obs: Any = None
        self._last_descriptions: list[str] = []
        self._last_reward: float | None = None
        self._last_terminate: bool | None = None
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._pool_task: tuple[str, int, int] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        task_name = str(coordinate.get("task_id") or "")
        seed = coordinate.get("seed")
        variation = str(coordinate.get("variation") or "")
        prefix = f"{task_name}::variation_"
        if not variation.startswith(prefix) or not variation[len(prefix):].isdigit():
            raise ValueError("RLBench coordinate requires TaskName::variation_N")
        index = int(variation[len(prefix):])
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("RLBench reset seed must be an integer in [0, 2**32)")
        source = get_project_paths().external_upstream("rlbench") / "rlbench/tasks/__init__.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        names = {alias.asname or alias.name for statement in module.body
                 if isinstance(statement, ast.ImportFrom)
                 and (statement.module or "").startswith("rlbench.tasks.")
                 for alias in statement.names}
        if task_name not in names:
            raise ValueError(f"Unknown pinned RLBench task: {task_name!r}")
        self._pool_task = (task_name, index, seed)
        return {"bound": True, "mode": "native_task_variation",
                "task_name": task_name, "variation_index": index, "seed": seed}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_task is not None:
            task_name, variation_index, selected_seed = self._pool_task
            if seed != selected_seed:
                raise ValueError("RLBench reset seed differs from selected pool coordinate")
            overrides["task_name"] = task_name
            task_id = f"rlbench:{task_name}:variation_{variation_index}:seed_{seed}"
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:rlbench:live_runtime",
            instruction="Solve an RLBench task from language, camera/state observation, grounding, and arm/gripper skill primitives.",
            goal={"task_name": runtime_config.task_name, "success_source": "task_success_or_episode_termination"},
            budgets={"primitive_calls": 24, "verifier_calls": 3},
            tags=["w4", "rlbench", "ai_native_runtime", "live" if runtime_config.live else "dry"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "rlbench",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "language_descriptions_visible": True,
                    "camera_grounding_visible": True,
                    "mask_object_pose_evidence_visible": True,
                    "action_schema_visible": True,
                    "demo_waypoints_exposed": False,
                    "checker_primitives_exposed": False,
                    "mock_success_for_actions": False,
                },
            },
        )
        self._last_descriptions = []
        self._last_reward = None
        self._last_terminate = None
        if runtime_config.live:
            if self._pool_task is not None:
                import random
                import numpy as np
                random.seed(seed)
                np.random.seed(seed)
            self._env = self._make_env(runtime_config)
            if hasattr(self._env, "launch"):
                self._env.launch()
            self._task = self._get_task(self._env, runtime_config.task_name)
            if self._pool_task is not None:
                self._task.set_variation(self._pool_task[1])
                self._task_spec.metadata["native_variation_index"] = self._pool_task[1]
            elif hasattr(self._task, "sample_variation"):
                self._task.sample_variation()
            reset_result = self._task.reset()
            self._last_descriptions, self._last_obs = _split_rlbench_reset(reset_result)
        else:
            self._env = None
            self._task = None
            self._last_obs = None
        self.record_event("reset", {"task": self._task_spec.to_dict(), "seed": seed, "runtime": self.runtime_available()})
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "descriptions": list(self._last_descriptions),
                "observation_summary": summarize_rlbench_observation(self._last_obs),
                "object_pose_evidence": self._object_pose_evidence(),
                "action_schema": self.action_schema(),
            },
            metadata={"benchmark_id": "rlbench"},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_rlbench_scene",
                "L1",
                {"prompt": "str|None", "agent_context": "dict|None", "include_raw": "bool"},
                {
                    "descriptions": "list[str]",
                    "observation_summary": "dict",
                    "object_pose_evidence": "dict",
                    "action_schema": "dict",
                },
                (
                    "Observe RLBench language descriptions plus multi-camera, mask, state, pose, and action schema summaries. "
                    "Use this before grounding a target or constructing an arm action."
                ),
            ),
            self._primitive_card(
                "inspect_rlbench_visual_evidence",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_name": "str|None", "target_name": "str|None"},
                {
                    "descriptions": "list[str]",
                    "visual_evidence": "dict",
                    "visual_observations": "dict",
                    "evidence_refs": "list[dict]",
                    "mask_evidence": "dict",
                    "pose_evidence": "dict",
                    "action_schema": "dict",
                },
                (
                    "Inspect RLBench camera modalities, masks, and target pose evidence for agent-side visual reasoning. "
                    "Use target_name from the task language when available, then pass returned evidence_refs to grounding/action primitives."
                ),
                preconditions=[
                    "Call observe_rlbench_scene first so language descriptions and action schema are available.",
                    "Pass prompt/query text that describes the target you are trying to reach or manipulate.",
                ],
            ),
            self._primitive_card(
                "ground_rlbench_target",
                "L2",
                {"query": "str|None", "target_name": "str|None", "agent_context": "dict|None", "return_types": "list[str]|None", "evidence_refs": "list[str|dict]|None"},
                {"selected": "dict", "candidates": "list[dict]", "evidence_refs": "list[dict]", "mask_evidence": "dict", "pose_evidence": "dict"},
                (
                    "Prompt/query-conditioned target grounding from descriptions, multi-camera/mask fields, state, and optional named targets. "
                    "Use the selected candidate pose/evidence_refs as the public grounding source for downstream arm actions."
                ),
                preconditions=[
                    "Call inspect_rlbench_visual_evidence first for the relevant camera/target.",
                    "Use target_name only when it is derived from the public task description or prior visual inspection.",
                ],
            ),
            self._primitive_card(
                "step_rlbench_action",
                "L3",
                {"action": "list[float]", "agent_context": "dict|None", "evidence_refs": "list[str|dict]|None"},
                {"stepped": "bool", "action": "list[float]", "provenance": "dict", "observation_summary": "dict", "action_schema": "dict"},
                "Submit one caller-authored RLBench action and return post-action observation evidence without verifier signals.",
                preconditions=[
                    "Call inspect_rlbench_visual_evidence and ground_rlbench_target before acting.",
                    "Pass evidence_refs from the visual inspection or target grounding.",
                    "Call record_rlbench_evidence with the selected target/action rationale before acting.",
                ],
            ),
            self._primitive_card(
                "move_rlbench_arm_to",
                "L3",
                {"target_name": "str|None", "target_pose": "list[float]|None", "agent_context": "dict|None", "strategy": "str", "evidence_refs": "list[str|dict]|None"},
                {"success": "bool", "requires_action_skill_backend": "bool"},
                (
                    "Real arm action-skill hook. Requires a configured controller/policy backend or built-in pose action strategy. "
                    "Use a target_pose derived from ground_rlbench_target or inspect_rlbench_visual_evidence."
                ),
                preconditions=[
                    "Call inspect_rlbench_visual_evidence and ground_rlbench_target before moving.",
                    "Call record_rlbench_evidence with the selected target/action rationale before moving.",
                    "Pass evidence_refs from the visual inspection or target grounding.",
                ],
            ),
            self._primitive_card(
                "open_rlbench_gripper",
                "L3",
                {"agent_context": "dict|None", "strategy": "str", "evidence_refs": "list[str|dict]|None"},
                {"success": "bool", "requires_action_skill_backend": "bool"},
                "Real gripper-open skill hook. Requires a configured controller/policy backend.",
            ),
            self._primitive_card(
                "close_rlbench_gripper",
                "L3",
                {"agent_context": "dict|None", "strategy": "str", "evidence_refs": "list[str|dict]|None"},
                {"success": "bool", "requires_action_skill_backend": "bool"},
                "Real gripper-close skill hook. Requires a configured controller/policy backend.",
            ),
            self._primitive_card(
                "record_rlbench_evidence",
                "L1",
                {"key": "str", "value": "any"},
                {"artifact_id": "str"},
                "Record agent-selected RLBench grounding/action evidence before executing an action.",
                preconditions=[
                    "Use only public primitive outputs such as descriptions, target_name, target_pose, and evidence_refs.",
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
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by RLBenchAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler is not None else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported RLBench verification scope: {scope}")
        else:
            success = _task_reports_success(self._task) or bool(self._last_terminate and self._last_reward and self._last_reward > 0)
            result = VerificationResult(
                ok=success,
                scope="task",
                message="RLBench task success reported by task/env" if success else "RLBench task has not reported success",
                metrics={"success": float(success), "last_reward": self._last_reward},
                metadata={"terminate": self._last_terminate},
            )
        if result.ok:
            self.get_trace().final_status = "success"
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self._trace

    def close(self) -> None:
        if self._env is not None and hasattr(self._env, "shutdown"):
            self._env.shutdown()
        elif self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

    def runtime_available(self) -> JsonDict:
        module_status = _rlbench_live_module_status()
        coppelia_root = self._coppeliasim_root()
        repo_python = _repo_local_rlbench_python()
        repo_coppelia_root = _repo_local_coppeliasim_root()
        return {
            "live": self.config.live,
            "env_created": self._env is not None,
            "rlbench_importable": module_status["rlbench"],
            "pyrep_importable": module_status["pyrep"],
            "required_live_modules": module_status,
            "missing_live_modules": [name for name, ok in module_status.items() if not ok],
            "action_skill_backend": self.config.action_skill_backend,
            "coppeliasim_root": coppelia_root,
            "repo_local_python": str(repo_python) if repo_python is not None else None,
            "repo_local_coppeliasim_root": str(repo_coppelia_root) if repo_coppelia_root is not None else None,
            "coppeliasim_launcher_env": _coppeliasim_launcher_env(coppelia_root),
            "display": os.environ.get("DISPLAY"),
            "headless": self.config.headless,
        }

    def _merged_config(self, overrides: JsonDict) -> RLBenchRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return RLBenchRuntimeConfig(**data)

    def _make_env(self, config: RLBenchRuntimeConfig) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.task_name, config.to_dict())
        module_status = _rlbench_live_module_status()
        missing_modules = [name for name, ok in module_status.items() if not ok]
        coppelia_root = self._coppeliasim_root(config)
        repo_python = _repo_local_rlbench_python()
        if missing_modules:
            hint_parts = [
                "RLBench live runtime requires importable Python modules `rlbench` and `pyrep`.",
                f"Missing Python modules: {', '.join(missing_modules)}.",
                f"Detected CoppeliaSim root: {coppelia_root or 'not found'}.",
            ]
            if repo_python is not None:
                hint_parts.append(
                    "Repo-local RLBench Python exists at "
                    f"{repo_python}; launch it with "
                    f"{_coppeliasim_launcher_env(coppelia_root) or 'a valid COPPELIASIM_ROOT/LD_LIBRARY_PATH'}."
                )
            raise RuntimeError(
                " ".join(hint_parts)
            )
        if coppelia_root is None:
            raise RuntimeError("RLBench live runtime requires CoppeliaSim v4.1.0; set COPPELIASIM_ROOT or install it.")
        _configure_coppeliasim_env(coppelia_root)
        try:
            from rlbench.action_modes.action_mode import MoveArmThenGripper
            from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK, EndEffectorPoseViaPlanning, JointVelocity
            from rlbench.action_modes.gripper_action_modes import Discrete
            from rlbench.environment import Environment
            from rlbench.observation_config import ObservationConfig
        except Exception as exc:
            launcher_env = _coppeliasim_launcher_env(coppelia_root)
            raise RuntimeError(
                "RLBench/PyRep import failed after CoppeliaSim discovery. "
                f"Launch the Python process with CoppeliaSim env {launcher_env}. "
                f"Original error: {exc}"
            ) from exc

        if "EndEffectorPoseViaPlanning" in config.action_mode:
            arm_action_mode = EndEffectorPoseViaPlanning()
        elif "EndEffectorPoseViaIK" in config.action_mode:
            arm_action_mode = EndEffectorPoseViaIK()
        else:
            arm_action_mode = JointVelocity()
        action_mode = MoveArmThenGripper(arm_action_mode=arm_action_mode, gripper_action_mode=Discrete())
        kwargs = dict(config.env_kwargs)
        if config.dataset_root is not None:
            kwargs["dataset_root"] = config.dataset_root
        kwargs.setdefault("obs_config", ObservationConfig(task_low_dim_state=True))
        return Environment(action_mode, headless=config.headless, **kwargs)

    def _get_task(self, env: Any, task_name: str) -> Any:
        if hasattr(env, "get_task"):
            task_module = importlib.import_module("rlbench.tasks")
            task_cls = getattr(task_module, task_name)
            return env.get_task(task_cls)
        if hasattr(env, "task"):
            return env.task
        raise RuntimeError("RLBench environment does not provide get_task().")

    def _primitive_observe_rlbench_scene(
        self,
        prompt: str | None = None,
        agent_context: JsonDict | None = None,
        include_raw: bool = False,
    ) -> PrimitiveResult:
        output = {
            "prompt": prompt,
            "agent_context": agent_context or {},
            "descriptions": list(self._last_descriptions),
            "observation_summary": summarize_rlbench_observation(self._last_obs),
            "object_pose_evidence": self._object_pose_evidence(),
            "action_schema": self.action_schema(),
            "runtime": self.runtime_available(),
        }
        if include_raw:
            output["raw_observation"] = _to_builtin(self._last_obs)
        return PrimitiveResult(name="observe_rlbench_scene", ok=True, output=output)

    def _primitive_inspect_rlbench_visual_evidence(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        camera_name: str | None = None,
        target_name: str | None = None,
    ) -> PrimitiveResult:
        summary = summarize_rlbench_observation(self._last_obs)
        visual_evidence = filter_rlbench_visual_summary(summary, camera_name=camera_name)
        visual_observations, evidence_refs, artifacts = self._capture_visual_evidence(camera_name=camera_name)
        selected_target = target_name
        output = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "camera_name": camera_name,
            "target_name": selected_target,
            "descriptions": list(self._last_descriptions),
            "visual_evidence": visual_evidence,
            "visual_observations": visual_observations,
            "evidence_refs": evidence_refs,
            "mask_evidence": _mask_evidence_for_target(self._last_obs, selected_target),
            "pose_evidence": self._object_pose_evidence(selected_target),
            "action_schema": self.action_schema(),
        }
        return PrimitiveResult(
            name="inspect_rlbench_visual_evidence",
            ok=bool(visual_evidence.get("camera_fields")),
            output=output,
            artifacts=artifacts,
            error=None if visual_evidence.get("camera_fields") else "no_rlbench_camera_fields",
        )

    def _primitive_ground_rlbench_target(
        self,
        query: str | None = None,
        target_name: str | None = None,
        agent_context: JsonDict | None = None,
        return_types: list[str] | None = None,
        evidence_refs: list[Any] | None = None,
    ) -> PrimitiveResult:
        requested = return_types or ["language", "camera", "state"]
        normalized_refs = self._normalize_evidence_refs(evidence_refs)
        selected_name = target_name
        candidates = []
        if selected_name:
            pose_evidence = self._object_pose_evidence(selected_name)
            mask_evidence = _mask_evidence_for_target(self._last_obs, selected_name)
            pose_world = pose_evidence.get("pose_world")
            candidates.append(
                {
                    "label": selected_name,
                    "score": 1.0,
                    "pose_world": pose_world,
                    "pose": pose_world,
                    "target_pose": pose_world,
                    "mask_evidence": mask_evidence,
                    "evidence": {
                        "source": "caller_selected_name_and_observation",
                        "descriptions": list(self._last_descriptions),
                        "requested": requested,
                        "agent_context": agent_context or {},
                        "camera_fields": summarize_rlbench_observation(self._last_obs).get("camera_fields", {}),
                        "pose_source": pose_evidence.get("source"),
                        "evidence_refs": normalized_refs,
                    },
                }
            )
        return PrimitiveResult(
            name="ground_rlbench_target",
            ok=bool(candidates),
            output={
                "query": query,
                "agent_context": agent_context or {},
                "return_types": requested,
                "selected": candidates[0] if candidates else None,
                "candidates": candidates,
                "evidence_refs": normalized_refs,
                "mask_evidence": _mask_evidence_for_target(self._last_obs, selected_name),
                "pose_evidence": self._object_pose_evidence(selected_name),
                "action_schema": self.action_schema(),
            },
            error=None if candidates else "target_not_found",
        )

    def _primitive_step_rlbench_action(
        self,
        action: list[float] | None = None,
        agent_context: JsonDict | None = None,
        evidence_refs: list[Any] | None = None,
    ) -> PrimitiveResult:
        provenance = self._action_provenance("step_rlbench_action", agent_context, evidence_refs)
        if self._task is None or not hasattr(self._task, "step"):
            return PrimitiveResult(
                name="step_rlbench_action",
                ok=False,
                output={"success": False, "requires_live_task": True, "provenance": provenance, "action_schema": self.action_schema()},
                error="live_task_step_unavailable",
            )
        if action is None:
            return PrimitiveResult(
                name="step_rlbench_action",
                ok=False,
                output={"stepped": False, "provenance": provenance, "action_schema": self.action_schema()},
                error="action_required",
            )
        action_values = action
        try:
            step_result = self._task.step(action_values)
            self._last_obs, self._last_reward, self._last_terminate = _split_rlbench_step(step_result)
        except Exception as exc:
            return PrimitiveResult(
                name="step_rlbench_action",
                ok=False,
                output={
                    "stepped": False,
                    "action": _to_builtin(action_values),
                    "agent_context": agent_context or {},
                    "provenance": provenance,
                    "action_schema": self.action_schema(),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        return PrimitiveResult(
            name="step_rlbench_action",
            ok=True,
            output={
                "stepped": True,
                "action": _to_builtin(action_values),
                "agent_context": agent_context or {},
                "provenance": provenance,
                "observation_summary": summarize_rlbench_observation(self._last_obs),
                "object_pose_evidence": self._object_pose_evidence(),
                "action_schema": self.action_schema(),
            },
        )

    def _primitive_move_rlbench_arm_to(
        self,
        target_name: str | None = None,
        target_pose: list[float] | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "action_skill_backend",
        evidence_refs: list[Any] | None = None,
    ) -> PrimitiveResult:
        provenance = self._action_provenance("move_rlbench_arm_to", agent_context, evidence_refs)
        if self._uses_builtin_pose_action_backend(strategy):
            return self._step_pose_action(
                primitive_name="move_rlbench_arm_to",
                target_name=target_name,
                target_pose=target_pose,
                gripper_command=1.0,
                agent_context=agent_context or {},
                strategy=strategy,
                provenance=provenance,
            )
        return self._call_skill_or_fail("move_rlbench_arm_to", {"target_name": target_name, "target_pose": target_pose, "agent_context": agent_context or {}, "strategy": strategy, "provenance": provenance})

    def _primitive_open_rlbench_gripper(self, agent_context: JsonDict | None = None, strategy: str = "action_skill_backend", evidence_refs: list[Any] | None = None) -> PrimitiveResult:
        if self._skill_backend is None:
            return self._native_gripper_action("open_rlbench_gripper", 1.0, agent_context, evidence_refs)
        return self._call_skill_or_fail("open_rlbench_gripper", {"agent_context": agent_context or {}, "strategy": strategy, "provenance": self._action_provenance("open_rlbench_gripper", agent_context, evidence_refs)})

    def _primitive_close_rlbench_gripper(self, agent_context: JsonDict | None = None, strategy: str = "action_skill_backend", evidence_refs: list[Any] | None = None) -> PrimitiveResult:
        if self._skill_backend is None:
            return self._native_gripper_action("close_rlbench_gripper", 0.0, agent_context, evidence_refs)
        return self._call_skill_or_fail("close_rlbench_gripper", {"agent_context": agent_context or {}, "strategy": strategy, "provenance": self._action_provenance("close_rlbench_gripper", agent_context, evidence_refs)})

    def _native_gripper_action(self, name, command, agent_context, evidence_refs):
        if self._is_pose_action_mode():
            pose = _normalize_pose_action(_to_builtin(getattr(self._last_obs, "gripper_pose", None)))
            if pose is None:
                return PrimitiveResult(name=name, ok=False, error="current_gripper_pose_unavailable")
            action = pose + [command]
        elif "JointVelocity" in self.config.action_mode:
            action = [0.0] * 7 + [command]
        else:
            return PrimitiveResult(name=name, ok=False, error="gripper_action_mode_not_supported")
        result = self._primitive_step_rlbench_action(action, agent_context=agent_context, evidence_refs=evidence_refs)
        return PrimitiveResult(name=name, ok=result.ok, output=result.output,
                               artifacts=result.artifacts, error=result.error, metadata=result.metadata)

    def _primitive_record_rlbench_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"rlbench:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value)})
        return PrimitiveResult(name="record_rlbench_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _call_skill_or_fail(self, name: str, payload: JsonDict) -> PrimitiveResult:
        if self._skill_backend is None:
            return PrimitiveResult(
                name=name,
                ok=False,
                output={**payload, "success": False, "requires_action_skill_backend": True},
                error="action_skill_backend_missing",
            )
        raw = self._skill_backend(name, payload)
        result = raw if isinstance(raw, PrimitiveResult) else PrimitiveResult(name=name, ok=bool(raw.get("success")), output=raw)
        return PrimitiveResult(
            name=result.name,
            ok=result.ok,
            output={**_public_rlbench_skill_output(result.output), "provenance": payload.get("provenance", {})},
            artifacts=list(result.artifacts),
            error=result.error,
            metadata=_public_rlbench_skill_output(result.metadata),
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
            capability_tags=["w4", "rlbench", "agent_native_runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=list(preconditions or []),
            cost={"primitive_calls": 1},
            failure_modes=["backend_not_configured", "wrong_arguments", "runtime_dependency_missing"],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def action_schema(self) -> JsonDict:
        if self._is_pose_action_mode():
            minimal_action = {
                "shape": [8],
                "dtype": "float32-compatible",
                "semantics": "7 end-effector pose values [x, y, z, qx, qy, qz, qw] followed by discrete gripper command",
            }
        else:
            minimal_action = {
                "shape": [8],
                "dtype": "float32-compatible",
                "semantics": "7 joint velocities followed by discrete gripper command for MoveArmThenGripper(JointVelocity, Discrete)",
            }
        return {
            "action_mode": self.config.action_mode,
            "minimal_step_action": minimal_action,
            "skill_primitives": {
                "move_rlbench_arm_to": {"target_name": "str|None", "target_pose": "list[float]|None"},
                "open_rlbench_gripper": {"agent_context": "dict|None"},
                "close_rlbench_gripper": {"agent_context": "dict|None"},
            },
        }

    def _object_pose_evidence(self, target_name: str | None = None) -> JsonDict:
        names = _target_name_aliases(target_name)
        evidence = _pose_from_pyrep_object(names)
        if evidence is not None:
            return evidence
        obs_summary = summarize_rlbench_observation(self._last_obs)
        gripper_pose = obs_summary.get("state_fields", {}).get("gripper_pose")
        task_low_dim_state = obs_summary.get("state_fields", {}).get("task_low_dim_state")
        pose_world = _pose_from_task_low_dim_state(task_low_dim_state, gripper_pose)
        if pose_world is not None:
            return {
                "source": "observation_task_low_dim_state",
                "target_name": target_name,
                "pose_world": pose_world,
                "pose": pose_world,
                "target_pose": pose_world,
                "task_low_dim_state": list(task_low_dim_state),
                "available_state_pose": {"gripper_pose": gripper_pose, "task_low_dim_state": task_low_dim_state},
            }
        return {
            "source": "observation_state_only",
            "target_name": target_name,
            "pose_world": None,
            "available_state_pose": {"gripper_pose": gripper_pose},
        }

    def _uses_builtin_pose_action_backend(self, strategy: str | None) -> bool:
        return self._is_pose_action_mode() and (
            self.config.action_skill_backend == "rlbench_pose_action" or strategy in {"rlbench_pose_action", "end_effector_pose"}
        )

    def _is_pose_action_mode(self) -> bool:
        return "EndEffectorPoseViaPlanning" in self.config.action_mode or "EndEffectorPoseViaIK" in self.config.action_mode

    def _step_pose_action(
        self,
        *,
        primitive_name: str,
        target_name: str | None,
        target_pose: list[float] | None,
        gripper_command: float,
        agent_context: JsonDict,
        strategy: str,
        provenance: JsonDict,
    ) -> PrimitiveResult:
        if self._task is None or not hasattr(self._task, "step"):
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={"success": False, "requires_live_task": True, "provenance": provenance, "action_schema": self.action_schema()},
                error="live_task_step_unavailable",
            )
        pose_source = target_pose if target_pose is not None else self._object_pose_evidence(target_name).get("pose_world")
        pose = _normalize_pose_action(pose_source)
        if pose is None:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "success": False,
                    "target_name": target_name,
                    "target_pose": target_pose,
                    "agent_context": agent_context,
                    "strategy": strategy,
                    "provenance": provenance,
                    "requires_action_skill_backend": False,
                    "action_schema": self.action_schema(),
                },
                error="target_pose_unavailable",
            )
        action_values = pose + [float(gripper_command)]
        try:
            step_result = self._task.step(action_values)
            self._last_obs, self._last_reward, self._last_terminate = _split_rlbench_step(step_result)
        except Exception as exc:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "success": False,
                    "target_name": target_name,
                    "target_pose": pose,
                    "action": _to_builtin(action_values),
                    "agent_context": agent_context,
                    "strategy": strategy,
                    "provenance": provenance,
                    "requires_action_skill_backend": False,
                    "action_schema": self.action_schema(),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        return PrimitiveResult(
            name=primitive_name,
            ok=True,
            output={
                "executed": True,
                "target_name": target_name,
                "target_pose": pose,
                "action": _to_builtin(action_values),
                "agent_context": agent_context,
                "strategy": strategy,
                "provenance": provenance,
                "observation_summary": summarize_rlbench_observation(self._last_obs),
                "object_pose_evidence": self._object_pose_evidence(target_name),
                "action_schema": self.action_schema(),
            },
        )

    def _capture_visual_evidence(self, camera_name: str | None = None) -> tuple[JsonDict, list[JsonDict], list[str]]:
        observations = extract_rlbench_visual_observations(self._last_obs, camera_name=camera_name)
        refs: list[JsonDict] = []
        artifacts: list[str] = []
        for camera, modalities in observations.items():
            for modality, data in modalities.items():
                artifact_id = f"rlbench:visual:{camera}:{modality}:{len(self.get_trace().artifacts)}"
                ref = _visual_evidence_ref("rlbench", artifact_id, camera, modality, f"{camera}_{modality}", data)
                self.get_trace().add_artifact(artifact_id, {**ref, "data": data})
                refs.append(ref)
                artifacts.append(artifact_id)
        return observations, refs, artifacts

    def _normalize_evidence_refs(self, evidence_refs: list[Any] | None) -> list[JsonDict]:
        return _normalize_evidence_refs(evidence_refs, self.get_trace().artifacts)

    def _action_provenance(self, primitive: str, agent_context: JsonDict | None, evidence_refs: list[Any] | None) -> JsonDict:
        return {
            "primitive": primitive,
            "agent_context": agent_context or {},
            "evidence_refs": self._normalize_evidence_refs(evidence_refs),
        }

    def _coppeliasim_root(self, config: RLBenchRuntimeConfig | None = None) -> str | None:
        configured = (config or self.config).coppeliasim_root
        return configured or find_coppeliasim_root()

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def summarize_rlbench_observation(obs: Any) -> JsonDict:
    if obs is None:
        return {}
    names = [name for name in dir(obs) if not name.startswith("_")]
    camera_fields = [name for name in names if name.endswith(("_rgb", "_depth", "_mask", "_point_cloud"))]
    state_fields = [name for name in names if name in {"joint_positions", "joint_velocities", "gripper_pose", "gripper_open", "task_low_dim_state"}]
    mask_fields = [name for name in camera_fields if name.endswith("_mask")]
    return {
        "type": type(obs).__name__,
        "camera_fields": {name: _shape_of(getattr(obs, name, None)) for name in camera_fields},
        "multi_camera": _camera_modalities(camera_fields),
        "mask_fields": {name: _mask_summary(getattr(obs, name, None)) for name in mask_fields},
        "state_fields": {name: _to_builtin(getattr(obs, name, None)) for name in state_fields},
    }


def extract_rlbench_visual_observations(obs: Any, camera_name: str | None = None) -> JsonDict:
    if obs is None:
        return {}
    observations: JsonDict = {}
    for field_name in dir(obs):
        if field_name.startswith("_") or not field_name.endswith(("_rgb", "_depth", "_mask", "_point_cloud")):
            continue
        camera, _, modality = field_name.rpartition("_")
        if camera_name is not None and camera != camera_name:
            continue
        value = getattr(obs, field_name, None)
        if value is not None:
            observations.setdefault(camera, {})[modality] = _to_builtin(value)
    return {camera: observations[camera] for camera in sorted(observations)}


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


def _normalize_evidence_refs(evidence_refs: list[Any] | None, artifacts: dict[str, JsonDict]) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for value in evidence_refs or []:
        artifact_id = value if isinstance(value, str) else value.get("artifact_id") if isinstance(value, dict) else None
        if not isinstance(artifact_id, str) or artifact_id not in artifacts:
            continue
        artifact = artifacts[artifact_id]
        normalized.append(
            {
                key: artifact.get(key)
                for key in ("artifact_id", "evidence_type", "benchmark_id", "camera", "modality", "source", "shape")
                if artifact.get(key) is not None
            }
        )
    return normalized


def filter_rlbench_visual_summary(summary: JsonDict, camera_name: str | None = None) -> JsonDict:
    if camera_name is None:
        return {
            "camera_fields": summary.get("camera_fields", {}),
            "multi_camera": summary.get("multi_camera", {}),
            "mask_fields": summary.get("mask_fields", {}),
        }
    prefix = f"{camera_name}_"
    return {
        "camera_fields": {key: value for key, value in summary.get("camera_fields", {}).items() if key.startswith(prefix)},
        "multi_camera": {key: value for key, value in summary.get("multi_camera", {}).items() if key == camera_name},
        "mask_fields": {key: value for key, value in summary.get("mask_fields", {}).items() if key.startswith(prefix)},
    }


def _split_rlbench_reset(reset_result: Any) -> tuple[list[str], Any]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        descriptions, obs = reset_result
        return [str(item) for item in descriptions], obs
    return [], reset_result


def _split_rlbench_step(step_result: Any) -> tuple[Any, float | None, bool | None]:
    if isinstance(step_result, tuple):
        if len(step_result) >= 3:
            return step_result[0], _maybe_float(step_result[1]), bool(step_result[2])
        if len(step_result) == 2:
            return step_result[0], _maybe_float(step_result[1]), None
    return step_result, None, None


def _target_name_aliases(target_name: str | None) -> list[str | None]:
    return [target_name] if target_name is not None else []


def _pose_from_task_low_dim_state(task_low_dim_state: Any, gripper_pose: Any) -> list[float] | None:
    state = _to_builtin(task_low_dim_state)
    if not isinstance(state, list) or len(state) < 3:
        return None
    try:
        xyz = [float(state[0]), float(state[1]), float(state[2])]
    except (TypeError, ValueError):
        return None
    gripper = _to_builtin(gripper_pose)
    quat: list[float] | None = None
    if isinstance(gripper, list) and len(gripper) >= 7:
        try:
            quat = [float(item) for item in gripper[3:7]]
        except (TypeError, ValueError):
            quat = None
    return xyz + (quat or [0.0, 0.0, 0.0, 1.0])


def _normalize_pose_action(value: Any) -> list[float] | None:
    if value is None:
        return None
    pose = _to_builtin(value)
    if not isinstance(pose, list):
        return None
    if len(pose) >= 7:
        return [float(item) for item in pose[:7]]
    return None


def _public_rlbench_skill_output(value: JsonDict) -> JsonDict:
    forbidden = ("reward", "terminate", "checker", "predicate", "task_success")

    def sanitize(item: Any) -> Any:
        if isinstance(item, dict):
            return {
                str(key): sanitize(nested)
                for key, nested in item.items()
                if not any(token in str(key).lower() for token in forbidden)
            }
        if isinstance(item, list):
            return [sanitize(nested) for nested in item]
        return item

    return sanitize(value)


def _task_reports_success(task: Any) -> bool:
    if task is None:
        return False
    for name in ("success", "_success"):
        attr = getattr(task, name, None)
        if callable(attr):
            try:
                value = attr()
                if isinstance(value, tuple):
                    return bool(value[0])
                return bool(value)
            except TypeError:
                continue
    return False


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


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _camera_modalities(camera_fields: list[str]) -> JsonDict:
    cameras: dict[str, list[str]] = {}
    for field_name in camera_fields:
        camera_name, _, modality = field_name.rpartition("_")
        cameras.setdefault(camera_name, []).append(modality)
    return {name: sorted(modalities) for name, modalities in sorted(cameras.items())}


def _mask_summary(value: Any) -> JsonDict:
    shape = _shape_of(value)
    if value is None:
        return {"shape": None, "unique_ids_sample": []}
    unique_ids: list[Any] = []
    try:
        import numpy as np

        unique_ids = [_to_builtin(item) for item in np.unique(value)[:16]]
    except Exception:
        unique_ids = []
    return {"shape": shape, "unique_ids_sample": unique_ids}


def _mask_evidence_for_target(obs: Any, target_name: str | None) -> JsonDict:
    summary = summarize_rlbench_observation(obs)
    return {
        "target_name": target_name,
        "source": "rlbench_observation_mask_fields",
        "mask_fields": summary.get("mask_fields", {}),
        "note": "RLBench masks expose object ids; semantic id-to-name mapping stays harness-side unless configured.",
    }


def _pose_from_pyrep_object(names: list[str | None]) -> JsonDict | None:
    if importlib.util.find_spec("pyrep") is None:
        return None
    try:
        from pyrep.objects.object import Object
    except Exception:
        return None
    for name in names:
        if not name:
            continue
        try:
            obj = Object.get_object(str(name))
            return {"source": "pyrep_object_pose", "target_name": str(name), "pose_world": _to_builtin(obj.get_pose())}
        except Exception:
            continue
    return None


def find_coppeliasim_root() -> str | None:
    candidates = [
        os.environ.get("COPPELIASIM_ROOT"),
        os.environ.get("COPPELIASIM_ROOT_DIR"),
        str(_repo_local_coppeliasim_root()) if _repo_local_coppeliasim_root() is not None else None,
        str(Path.home() / "CoppeliaSim"),
    ]
    for candidate in candidates:
        if candidate and _looks_like_coppeliasim_root(Path(candidate)):
            return str(Path(candidate))
    for base in (Path.home(), Path("/opt"), Path("/usr/local")):
        try:
            for child in base.glob("CoppeliaSim*"):
                if _looks_like_coppeliasim_root(child):
                    return str(child)
        except OSError:
            continue
    return None


def _looks_like_coppeliasim_root(path: Path) -> bool:
    return path.is_dir() and ((path / "coppeliaSim.sh").exists() or (path / "libcoppeliaSim.so").exists())


def _configure_coppeliasim_env(coppelia_root: str) -> None:
    root = str(Path(coppelia_root))
    os.environ.setdefault("COPPELIASIM_ROOT", root)
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", root)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if root not in current.split(":"):
        os.environ["LD_LIBRARY_PATH"] = f"{current}:{root}" if current else root


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
