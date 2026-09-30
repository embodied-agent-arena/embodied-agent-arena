from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import importlib
import importlib.util
import os
from pathlib import Path
import random
import sys
import types
from typing import Any, Callable

import numpy as np

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], tuple[Any, Any]]

CLIPORT_REQUIRED_LIVE_MODULES = ("cliport", "pybullet")


def _repo_local_cliport_python() -> Path | None:
    candidates = (
        get_project_paths().external_environment("cliport") / "bin/python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return None


def _cliport_live_module_status() -> dict[str, bool]:
    _activate_repo_local_cliport_source()
    return {module_name: _module_importable(module_name) for module_name in CLIPORT_REQUIRED_LIVE_MODULES}


@dataclass(slots=True)
class CLIPortRuntimeConfig:
    task_name: str = "stack-block-pyramid-seq-seen-colors"
    mode: str = "test"
    assets_root: str | None = None
    disp: bool = False
    shared_memory: bool = False
    hz: int = 480
    live: bool = True
    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class CLIPortGroundingCandidate:
    object_id: int
    pose0: list[Any] | None = None
    bbox_xyxy: list[int] | None = None
    camera_uid: str | None = None
    dimensions: list[float] | None = None
    geometry_tags: list[str] = field(default_factory=list)
    affordance_hints: list[str] = field(default_factory=list)
    visual_parts: list[JsonDict] = field(default_factory=list)
    surface_regions: list[JsonDict] = field(default_factory=list)
    visual_rgbs: list[list[float]] = field(default_factory=list)
    dominant_rgb: list[float] | None = None
    dominant_color_name: str | None = None
    evidence: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class CLIPortAgentRuntimeBackend(EmbodiedBackend):
    """Agent-native CLIPort/Ravens runtime adapter.

    Live mode creates the upstream PyBullet CLIPort environment and exposes
    native RGB-D/segmentation instance measurements plus the benchmark's
    pick-place action hook.
    The harness may call `verify`, but oracle policies, reward checks, and
    `check_*` helpers are not listed as coding-agent primitives.
    """

    def __init__(self, config: CLIPortRuntimeConfig | None = None, env_factory: EnvFactory | None = None) -> None:
        self.config = config or CLIPortRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._task: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._episode_return = 0.0
        self._native_steps = 0
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._pool_task: tuple[str, int] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        task_name = str(coordinate.get("task_id") or "")
        seed = coordinate.get("seed")
        if type(seed) is not int or seed < 10001 or seed % 2 != 1:
            raise ValueError("CLIPort test reset seeds must be odd integers >= 10001")
        source = get_project_paths().external_upstream("cliport") / "cliport/tasks/__init__.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        names = set()
        for statement in module.body:
            if (isinstance(statement, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "names" for target in statement.targets)
                and isinstance(statement.value, ast.Dict)):
                names.update(key.value for key in statement.value.keys
                             if isinstance(key, ast.Constant) and isinstance(key.value, str))
        if task_name not in names:
            raise ValueError(f"Unknown pinned CLIPort task: {task_name!r}")
        self._pool_task = (task_name, seed)
        return {"mode": "native_task_config", "bound": True,
                "task_name": task_name, "mode_name": "test", "seed": seed}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_task is not None:
            task_name, selected_seed = self._pool_task
            if seed != selected_seed:
                raise ValueError("CLIPort reset seed differs from the selected pool coordinate")
            overrides.update(task_name=task_name, mode="test")
            task_id = f"cliport:{task_name}:test:{seed}"
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:cliport:live_runtime",
            instruction=(
                "Solve a CLIPort/Ravens tabletop task by grounding the language goal in RGB-D "
                "evidence and invoking the task's original motion primitive."
            ),
            goal={"task_name": runtime_config.task_name, "mode": runtime_config.mode, "success_source": "cliport_reward_done"},
            initial_state={},
            budgets={"primitive_calls": 48, "verifier_calls": 6},
            tags=["w4", "cliport", "ravens", "ai_native_runtime", "live" if runtime_config.live else "dry"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "cliport",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "instance_measurements_return_bbox_and_pose": True,
                    "semantic_selection_owned_by_agent": True,
                    "mock_success_for_actions": False,
                    "oracle_exposed_to_agent": False,
                },
            },
        )
        self._last_reward = 0.0
        self._last_done = False
        self._episode_return = 0.0
        self._native_steps = 0
        if runtime_config.live:
            self._env, self._task = self._make_env(runtime_config)
            self._seed(seed)
            self._env.set_task(self._task)
            self._last_obs = self._env.reset()
            self._last_info = _to_builtin(getattr(self._env, "info", {}) or {})
        else:
            self._env = None
            self._task = None
            self._last_obs = {"color": (), "depth": ()}
            self._last_info = {"lang_goal": f"dry CLIPort task: {runtime_config.task_name}"}
        self._task_spec.instruction = self._language_goal()
        self._task_spec.metadata["native_action_primitive"] = self._native_action_primitive()
        self.record_event(
            "reset",
            {"task": self._task_spec.to_dict(), "seed": seed, "runtime_available": self.runtime_available()},
        )
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        data = {
            "runtime": self.runtime_available(),
            "language_goal": self._language_goal(),
            "rgbd_summary": summarize_cliport_rgbd(self._last_obs),
            "object_ids": self._object_ids(),
        }
        obs = Observation(step=len(self.get_trace().events), data=data, metadata={"benchmark_id": "cliport"})
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_cliport_rgbd",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "include_raw": "bool"},
                {
                    "language_goal": "str",
                    "rgbd_summary": "dict",
                    "heightmap_evidence": "dict",
                    "segmentation_evidence": "dict",
                    "object_ids": "list[int]",
                    "camera_observations": "dict",
                },
                (
                    "Observe CLIPort RGB-D/depth/segmentation evidence and language goal with optional agent "
                    "prompt/context. Use this before selecting source or target instances for a visual pick-place action."
                ),
            ),
            self._primitive_card(
                "get_cliport_task_language_goal",
                "L1",
                {"query": "str|None", "agent_context": "dict|None"},
                {"language_goal": "str"},
                "Return the current upstream CLIPort language goal so subsequent visual grounding is tied to the task text.",
            ),
            self._primitive_card(
                "inspect_cliport_instances",
                "L2",
                {
                    "camera_uid": "str|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None",
                },
                {
                    "instances": (
                        "list[dict with object_id:int, pose0:[[x,y,z],[qx,qy,qz,qw]], "
                        "dimensions:list[float]|None, visual_rgbs:list[list[float]], "
                        "dominant_rgb:list[float]|None, dominant_color_name:str|None, "
                        "bbox_xyxy:list[int]|None, camera_uid:str|None]"
                    ),
                    "query_grounding": (
                        "dict mapping requested color/type words to ranked candidates with object_id, pose0, "
                        "dimensions, dominant_color_name, dominant_rgb, score"
                    ),
                    "evidence_id": "str",
                },
                (
                    "Enumerate native CLIPort segmentation instances with measured pose, size, RGB values, and bboxes. "
                    "Use query_grounding when prompt/query names visual attributes. pose0 is the object-body pose, "
                    "not the suction end-effector pose: choose contact/placement from public geometry; "
                    "the native downward suction pick uses quaternion [0,0,0,1]. After choosing source/target candidates, "
                    "call inspect_cliport_instance for each selected object id to bind exact visual evidence."
                ),
                preconditions=[
                    "Call get_cliport_task_language_goal or observe_cliport_rgbd first when the action depends on task language.",
                    "Pass prompt/query text that names the visual attributes you are grounding.",
                ],
            ),
            self._primitive_card(
                "inspect_cliport_instance",
                "L2",
                {"object_id": "int", "camera_uid": "str|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {
                    "instance": (
                        "dict with object_id:int, pose0:[[x,y,z],[qx,qy,qz,qw]], dimensions:list[float]|None, "
                        "visual_rgbs:list[list[float]], dominant_rgb:list[float]|None, "
                        "dominant_color_name:str|None, bbox_xyxy:list[int]|None, camera_uid:str|None"
                    ),
                    "evidence_id": "str",
                },
                (
                    "Measure one caller-selected native segmentation instance by exact object id. "
                    "pose0 is the object-body pose. Construct the suction end-effector contact/placement pose from public "
                    "geometry; do not copy a tilted object quaternion into the downward suction pick. "
                    "Pass the returned evidence_id to the action primitive."
                ),
                preconditions=[
                    "Choose object_id from inspect_cliport_instances or observe_cliport_rgbd object_ids.",
                    "Call this separately for each source or target instance that will be used by an action.",
                ],
            ),
            self._primitive_card(
                "submit_cliport_pick_place_action",
                "L3",
                {
                    "pick_pose": "list|None",
                    "place_pose": "list|None",
                    "source_object_id": "int|None",
                    "target_object_id": "int|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "evidence_ids": "list[str]|None",
                },
                {"stepped": "bool", "action": "dict", "action_schema": "dict", "observation_summary": "dict"},
                (
                    "Submit one original CLIPort task motion using caller-selected poses or exact instance ids. "
                    "pick_pose/place_pose map to native pose0/pose1; the task chooses Push or PickPlace. Use the pose0 format returned by inspect_cliport_instance(s). "
                    "Use evidence_ids from selected instance inspections, not hidden task metadata."
                ),
                preconditions=[
                    "Call get_cliport_task_language_goal to bind the task text.",
                    "Call observe_cliport_rgbd and inspect_cliport_instances to ground visible candidates.",
                    "Call inspect_cliport_instance for the selected source and target before acting.",
                    "Call record_cliport_evidence with a compact source/target/action rationale before acting.",
                    "Pass evidence_ids returned by inspect_cliport_instance for the selected source and target.",
                ],
            ),
            self._primitive_card(
                "record_cliport_evidence",
                "L1",
                {"key": "str", "value": "any"},
                {"artifact_id": "str"},
                "Record agent-selected CLIPort grounding/action evidence in the trace before executing an action.",
                preconditions=[
                    "Use only public primitive outputs such as language_goal, object_id, pose0, bbox, color, and evidence_id.",
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
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by CLIPortAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler is not None else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
        self.record_event("primitive_call", {"name": name, "kwargs": _to_builtin(kwargs), "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported CLIPort verification scope: {scope}")
        else:
            # Environment.done also means a primitive timed out. Only the
            # original task predicate establishes success; eval.py scores the
            # sum of step rewards and bounds actions by task.max_steps.
            predicate = getattr(self._task, "done", None)
            available = callable(predicate)
            ok = bool(predicate()) if available else False
            result = VerificationResult(
                ok=ok,
                scope="task",
                message="CLIPort native task completion verified" if ok else "CLIPort native task is not complete",
                metrics={"reward": float(self._last_reward), "done": float(bool(self._last_done)),
                         "native_episode_return": self._episode_return, "native_steps": self._native_steps},
                metadata={"info_summary": _to_builtin(self._last_info),
                          "native_episode_done": self._native_episode_done(),
                          "native_success_defined": available,
                          "native_success_source": "task.done()" if available else None,
                          "native_max_steps": self._native_max_steps()},
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
        if self._env is not None and hasattr(self._env, "close"):
            self._env.close()
        self._env = None

    def runtime_available(self) -> JsonDict:
        source_path = _activate_repo_local_cliport_source()
        module_status = _cliport_live_module_status()
        repo_python = _repo_local_cliport_python()
        return {
            "live": self.config.live,
            "env_created": self._env is not None,
            "cliport_importable": module_status["cliport"],
            "pybullet_importable": module_status["pybullet"],
            "required_live_modules": module_status,
            "missing_live_modules": [name for name, ok in module_status.items() if not ok],
            "assets_root": self.config.assets_root or _default_assets_root(),
            "repo_local_source_path": source_path,
            "repo_local_python": str(repo_python) if repo_python is not None else None,
        }

    def _primitive_observe_cliport_rgbd(
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
            "language_goal": self._language_goal(),
            "rgbd_summary": summarize_cliport_rgbd(self._last_obs),
            "heightmap_evidence": summarize_cliport_heightmap(self._last_obs),
            "segmentation_evidence": self._segmentation_evidence(),
            "object_ids": self._object_ids(),
            "camera_observations": self._camera_observations(include_raw=include_raw),
        }
        if include_raw:
            output["raw_observation"] = _to_builtin(self._last_obs)
        return PrimitiveResult(name="observe_cliport_rgbd", ok=True, output=output)

    def _native_action_primitive(self) -> str | None:
        primitive = getattr(self._task, "primitive", None)
        return getattr(primitive, "__name__", type(primitive).__name__) if primitive is not None else None

    def _native_max_steps(self) -> int | None:
        value = getattr(self._task, "max_steps", None)
        return int(value) if isinstance(value, (int, np.integer)) and value > 0 else None

    def _native_episode_done(self) -> bool:
        limit = self._native_max_steps()
        return self._last_done or (limit is not None and self._native_steps >= limit)

    def _primitive_get_cliport_task_language_goal(
        self,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_cliport_task_language_goal",
            ok=True,
            output={"query": query, "agent_context": agent_context or {}, "language_goal": self._language_goal(),
                    "native_action_primitive": self._native_action_primitive(),
                    "native_max_steps": self._native_max_steps(),
                    "action_guidance": "pose0 is the start pose and pose1 the end pose; env.step dispatches the original task primitive (e.g. Push or PickPlace)."},
        )

    def _primitive_inspect_cliport_instances(
        self,
        camera_uid: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        frames = self._render_segmentation_frames(camera_uid)
        instances = [self._instance_evidence(object_id, camera_uid=camera_uid, frames=frames).to_dict() for object_id in self._object_ids()]
        evidence_id = f"cliport:instances:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "instances": instances,
            "query_grounding": _ground_cliport_instances_from_text(instances, " ".join(part for part in [prompt, query] if part)),
        }
        self.get_trace().add_artifact(evidence_id, payload)
        return PrimitiveResult(
            name="inspect_cliport_instances",
            ok=True,
            output={**payload, "evidence_id": evidence_id},
            artifacts=[evidence_id],
        )

    def _primitive_inspect_cliport_instance(
        self,
        object_id: int,
        camera_uid: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if object_id not in self._object_ids():
            return PrimitiveResult(
                name="inspect_cliport_instance",
                ok=False,
                output={"object_id": object_id},
                error="object_id_not_found",
            )
        instance = self._instance_evidence(object_id, camera_uid=camera_uid).to_dict()
        artifact_id = f"cliport:instance:{object_id}:{len(self.get_trace().artifacts)}"
        payload = {"kind": "visual_grounding", "prompt": prompt, "query": query, "agent_context": agent_context or {}, "instance": instance}
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="inspect_cliport_instance",
            ok=True,
            output={**payload, "evidence_id": artifact_id},
            artifacts=[artifact_id],
        )

    def _primitive_submit_cliport_pick_place_action(
        self,
        pick_pose: list[Any] | tuple[Any, Any] | None = None,
        place_pose: list[Any] | tuple[Any, Any] | None = None,
        source_object_id: int | None = None,
        target_object_id: int | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        if self._native_episode_done():
            return PrimitiveResult(name="submit_cliport_pick_place_action", ok=False,
                                   error="native_episode_terminated", output={"stepped": False})
        evidence_refs, evidence_error = self._resolve_visual_evidence(evidence_ids)
        if evidence_error is not None:
            return PrimitiveResult(name="submit_cliport_pick_place_action", ok=False, error=evidence_error)
        if pick_pose is None and source_object_id is not None:
            pick_pose = self._pose_for_object(source_object_id)
        if place_pose is None and target_object_id is not None:
            place_pose = self._pose_for_object(target_object_id)
        pose0 = _normalize_pose(pick_pose)
        pose1 = _normalize_pose(place_pose)
        if pose0 is None or pose1 is None:
            return PrimitiveResult(
                name="submit_cliport_pick_place_action",
                ok=False,
                output={
                    "pick_pose_available": pose0 is not None,
                    "place_pose_available": pose1 is not None,
                    "source_object_id": source_object_id,
                    "target_object_id": target_object_id,
                    "evidence_ids": evidence_refs,
                },
                error="pick_and_place_pose_required",
            )
        action = {"pose0": pose0, "pose1": pose1}
        action_schema = _cliport_action_schema()
        if self._env is None:
            return PrimitiveResult(
                name="submit_cliport_pick_place_action",
                ok=False,
                output={"action": action, "action_schema": action_schema, "requires_live_runtime": True},
                error="live_runtime_required",
            )
        self._last_obs, reward, done, info = self._env.step(action)
        self._last_reward = float(reward)
        self._episode_return += float(reward)
        self._native_steps += 1
        self._last_done = bool(done)
        self._last_info = _to_builtin(info or {})
        output = {
            "stepped": True,
            "action": _to_builtin(action),
            "action_schema": action_schema,
            "source_object_id": source_object_id,
            "target_object_id": target_object_id,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "evidence_ids": evidence_refs,
            "observation_summary": summarize_cliport_rgbd(self._last_obs),
        }
        return PrimitiveResult(name="submit_cliport_pick_place_action", ok=True, output=output)

    def _primitive_record_cliport_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"cliport:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value)})
        return PrimitiveResult(name="record_cliport_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _make_env(self, config: CLIPortRuntimeConfig) -> tuple[Any, Any]:
        if self._env_factory is not None:
            return self._env_factory(config.task_name, config.to_dict())
        _activate_repo_local_cliport_source()
        module_status = _cliport_live_module_status()
        missing_modules = [name for name, ok in module_status.items() if not ok]
        if missing_modules:
            repo_python = _repo_local_cliport_python()
            hint = (
                ""
                if repo_python is None
                else (
                    f" Repo-local runtime Python detected at {repo_python}; configure the native harness to launch "
                    "that environment or install the same CLIPort dependency stack into the current interpreter."
                )
            )
            raise RuntimeError(
                "CLIPort live runtime requires importable Python modules `cliport` and `pybullet`. "
                f"Missing Python modules: {', '.join(missing_modules)}.{hint}"
            )
        try:
            from cliport import tasks
            from cliport.environments.environment import Environment
        except ModuleNotFoundError:
            tasks, environment_module = _import_cliport_env_only()
            Environment = environment_module.Environment

        if config.task_name not in tasks.names:
            raise RuntimeError(f"Unknown CLIPort task {config.task_name!r}; expected one of cliport.tasks.names.")
        assets_root = config.assets_root or _default_assets_root()
        if assets_root is None:
            raise RuntimeError("Set CLIPORT_ROOT or pass assets_root to create the CLIPort environment.")
        env = Environment(assets_root, disp=config.disp, shared_memory=config.shared_memory, hz=config.hz)
        task = tasks.names[config.task_name]()
        task.mode = config.mode
        return env, task

    def _instance_evidence(self, object_id: int, camera_uid: str | None, frames: list | None = None) -> CLIPortGroundingCandidate:
        boxes = self._render_segmentation_bboxes(object_id, camera_uid=camera_uid, frames=frames)
        bbox = boxes[0] if boxes else None
        visual_parts = self._visual_parts_for_object(object_id)
        visual_rgbs = [list(part["rgb"]) for part in visual_parts if isinstance(part.get("rgb"), list)]
        if not visual_rgbs:
            visual_rgbs = [list(rgb) for rgb in self._visual_rgbs_for_object(object_id)]
        dominant_rgb = _dominant_rgb(visual_rgbs)
        dimensions = self._dimensions_for_object(object_id)
        pose0 = self._pose_for_object(object_id)
        return CLIPortGroundingCandidate(
            object_id=object_id,
            pose0=pose0,
            bbox_xyxy=bbox["bbox_xyxy"] if bbox else None,
            camera_uid=bbox["camera_uid"] if bbox else None,
            dimensions=dimensions,
            geometry_tags=_cliport_geometry_tags(dimensions),
            affordance_hints=_cliport_affordance_hints(dimensions),
            visual_parts=visual_parts,
            surface_regions=_cliport_surface_regions(object_id=object_id, dimensions=dimensions, object_pose0=pose0, visual_parts=visual_parts),
            visual_rgbs=visual_rgbs,
            dominant_rgb=dominant_rgb,
            dominant_color_name=_coarse_color_name(dominant_rgb),
            evidence={"source": "env.info+rgbd_segmentation+pybullet_visual_shape"},
        )

    def _object_ids(self) -> list[int]:
        ids: list[int] = []
        for key in self._last_info:
            if isinstance(key, int):
                ids.append(key)
            elif isinstance(key, str) and key.isdigit():
                ids.append(int(key))
        return sorted(ids)

    def _pose_for_object(self, object_id: int) -> list[Any] | None:
        record = self._last_info.get(object_id, self._last_info.get(str(object_id)))
        if not isinstance(record, (list, tuple)) or len(record) < 2:
            return None
        return [_to_builtin(record[0]), _to_builtin(record[1])]

    def _dimensions_for_object(self, object_id: int) -> list[float] | None:
        record = self._last_info.get(object_id, self._last_info.get(str(object_id)))
        if not isinstance(record, (list, tuple)) or len(record) < 3:
            return None
        return _to_builtin(record[2])

    def _bbox_for_object(self, object_id: int, camera_uid: str | None) -> JsonDict | None:
        camera_results = self._render_segmentation_bboxes(object_id=object_id, camera_uid=camera_uid)
        return camera_results[0] if camera_results else None

    def _render_segmentation_frames(self, camera_uid: str | None = None) -> list:
        """Render once per query; never retain frames across environment actions."""
        if self._env is None or not hasattr(self._env, "render_camera"):
            return []
        cameras = list(getattr(self._env, "agent_cams", []) or [])
        results: list = []
        for index, camera in enumerate(cameras):
            uid = str(camera.get("name", f"camera_{index}")) if isinstance(camera, dict) else f"camera_{index}"
            if camera_uid is not None and uid != camera_uid:
                continue
            try:
                _, _, segm = self._env.render_camera(camera)
            except Exception:  # noqa: BLE001 - camera rendering should not abort grounding.
                continue
            results.append((uid, segm))
        return results

    def _render_segmentation_bboxes(self, object_id: int, camera_uid: str | None = None, frames: list | None = None) -> list[JsonDict]:
        frames = self._render_segmentation_frames(camera_uid) if frames is None else frames
        results = []
        for uid, segm in frames:
            bbox = _segmentation_bbox(segm, object_id)
            if bbox is not None:
                results.append({"camera_uid": uid, "bbox_xyxy": bbox})
        return results

    def _segmentation_evidence(self) -> JsonDict:
        frames = self._render_segmentation_frames()
        objects: list[JsonDict] = []
        for object_id in self._object_ids():
            objects.append(
                {
                    "object_id": object_id,
                    "pose0": self._pose_for_object(object_id),
                    "dimensions": self._dimensions_for_object(object_id),
                    "visual_rgbs": [list(rgb) for rgb in self._visual_rgbs_for_object(object_id)],
                    "visible_bboxes": self._render_segmentation_bboxes(object_id, frames=frames),
                }
            )
        return {"source": "env.render_camera_segmentation", "objects": objects}

    def _camera_observations(self, include_raw: bool) -> JsonDict:
        if self._env is None or not hasattr(self._env, "render_camera"):
            return {}
        output: JsonDict = {}
        for index, camera in enumerate(list(getattr(self._env, "agent_cams", []) or [])):
            uid = str(camera.get("name", f"camera_{index}")) if isinstance(camera, dict) else f"camera_{index}"
            try:
                rgb, depth, segmentation = self._env.render_camera(camera)
            except Exception:  # noqa: BLE001 - preserve other usable upstream cameras.
                continue
            payload = {
                "rgb": _array_summary(rgb),
                "depth": _array_summary(depth),
                "segmentation": _array_summary(segmentation),
            }
            if include_raw:
                payload["raw"] = {
                    "rgb": _to_builtin(rgb),
                    "depth": _to_builtin(depth),
                    "segmentation": _to_builtin(segmentation),
                }
            output[uid] = payload
        return output

    def _resolve_visual_evidence(self, evidence_ids: list[str] | None) -> tuple[list[str], str | None]:
        refs = [str(value) for value in (evidence_ids or [])]
        for evidence_id in refs:
            payload = self.get_trace().artifacts.get(evidence_id)
            if not isinstance(payload, dict) or payload.get("kind") != "visual_grounding":
                return refs, f"visual_evidence_not_found: {evidence_id}"
        return refs, None

    def _visual_rgbs_for_object(self, object_id: int) -> list[tuple[float, float, float]]:
        try:
            pybullet = importlib.import_module("pybullet")
            visual_shapes = pybullet.getVisualShapeData(object_id)
        except Exception:  # noqa: BLE001 - pybullet evidence is optional in unit/dry contexts.
            return []
        rgbs: list[tuple[float, float, float]] = []
        for shape in visual_shapes or []:
            if len(shape) < 8 or shape[7] is None:
                continue
            rgba = shape[7]
            if len(rgba) < 3:
                continue
            rgb = (float(rgba[0]), float(rgba[1]), float(rgba[2]))
            if rgb not in rgbs:
                rgbs.append(rgb)
        return rgbs

    def _visual_parts_for_object(self, object_id: int) -> list[JsonDict]:
        try:
            pybullet = importlib.import_module("pybullet")
            visual_shapes = pybullet.getVisualShapeData(object_id)
        except Exception:  # noqa: BLE001 - pybullet evidence is optional in unit/dry contexts.
            return []
        parts: list[JsonDict] = []
        for index, shape in enumerate(visual_shapes or []):
            if len(shape) < 8 or shape[7] is None:
                continue
            rgba = shape[7]
            if len(rgba) < 3:
                continue
            rgb = [round(float(rgba[0]), 4), round(float(rgba[1]), 4), round(float(rgba[2]), 4)]
            link_index = int(shape[1]) if len(shape) > 1 else -1
            dimensions = _to_builtin(shape[3]) if len(shape) > 3 else None
            local_pos = _to_builtin(shape[5]) if len(shape) > 5 else [0.0, 0.0, 0.0]
            local_orn = _to_builtin(shape[6]) if len(shape) > 6 else [0.0, 0.0, 0.0, 1.0]
            pose0 = _visual_part_world_pose(pybullet, object_id, link_index, local_pos, local_orn)
            part = {
                "part_index": index,
                "link_index": link_index,
                "rgb": rgb,
                "color_name": _coarse_color_name(rgb),
                "brightness": round(sum(rgb) / 3.0, 4),
                "dimensions": dimensions,
                "local_pose0": [local_pos, local_orn],
                "pose0": pose0,
            }
            parts.append(part)
        return parts

    def _language_goal(self) -> str:
        if "lang_goal" in self._last_info:
            return str(self._last_info["lang_goal"])
        if self._env is not None and hasattr(self._env, "get_lang_goal"):
            try:
                return str(self._env.get_lang_goal())
            except Exception:  # noqa: BLE001
                return ""
        return ""

    def _seed(self, seed: int | None) -> None:
        if seed is None:
            return
        np.random.seed(seed)
        random.seed(seed)
        if self._env is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)

    def _merged_config(self, overrides: JsonDict) -> CLIPortRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return CLIPortRuntimeConfig(**data)

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
            capability_tags=["w4", "cliport", "ravens", "agent_native_runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=list(preconditions or []),
            cost={"primitive_calls": 1},
            failure_modes=["backend_not_configured", "wrong_arguments", "runtime_dependency_missing", "grounding_not_found"],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def summarize_cliport_rgbd(obs: Any) -> JsonDict:
    if not isinstance(obs, dict):
        return {"type": type(obs).__name__}
    summary: JsonDict = {}
    for key in ("color", "depth"):
        frames = obs.get(key, ())
        if not isinstance(frames, (list, tuple)):
            frames = (frames,)
        summary[key] = [_array_summary(frame) for frame in frames]
    return summary


def summarize_cliport_heightmap(obs: Any) -> JsonDict:
    if not isinstance(obs, dict):
        return {"type": type(obs).__name__, "source": "unavailable"}
    depth_frames = obs.get("depth", ())
    if not isinstance(depth_frames, (list, tuple)):
        depth_frames = (depth_frames,)
    heightmaps: list[JsonDict] = []
    for index, frame in enumerate(depth_frames):
        if hasattr(frame, "shape"):
            array = np.asarray(frame)
            finite = array[np.isfinite(array)] if np.issubdtype(array.dtype, np.number) else np.asarray([])
            heightmaps.append(
                {
                    "camera_index": index,
                    "source": "cliport_depth_frame",
                    "shape": list(array.shape),
                    "dtype": str(array.dtype),
                    "z_min": float(np.min(finite)) if finite.size else None,
                    "z_max": float(np.max(finite)) if finite.size else None,
                    "valid_pixel_count": int(finite.size),
                }
            )
        else:
            heightmaps.append({"camera_index": index, "type": type(frame).__name__})
    return {"source": "obs.depth_summary", "heightmaps": heightmaps}


def _array_summary(value: Any) -> JsonDict:
    if hasattr(value, "shape"):
        array = np.asarray(value)
        return {"shape": list(array.shape), "dtype": str(array.dtype), "min": _safe_min(array), "max": _safe_max(array)}
    return {"type": type(value).__name__}


def _safe_min(array: np.ndarray) -> float | None:
    return float(np.min(array)) if array.size else None


def _safe_max(array: np.ndarray) -> float | None:
    return float(np.max(array)) if array.size else None


def _segmentation_bbox(segm: Any, object_id: int) -> list[int] | None:
    array = np.asarray(segm)
    if array.ndim > 2:
        array = array[..., 0]
    mask = array == object_id
    if not np.any(mask):
        mask = (array & ((1 << 24) - 1)) == object_id
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _dominant_rgb(visual_rgbs: list[list[float]]) -> list[float] | None:
    if not visual_rgbs:
        return None
    array = np.asarray(visual_rgbs, dtype=float)
    if array.size == 0:
        return None
    rgb = np.mean(array[:, :3], axis=0)
    return [round(float(value), 4) for value in rgb.tolist()]


def _coarse_color_name(rgb: list[float] | None) -> str | None:
    if rgb is None or len(rgb) < 3:
        return None
    red, green, blue = [float(value) for value in rgb[:3]]
    scale = 255.0 if max(red, green, blue) > 1.5 else 1.0
    red, green, blue = red / scale, green / scale, blue / scale
    if red >= green >= blue and red - blue >= 0.02:
        if red >= 0.68 and green >= 0.55 and blue >= 0.45:
            return "light_brown"
        if red <= 0.4 and green <= 0.35 and blue <= 0.3:
            return "dark_brown"
        if red <= 0.8 and green <= 0.65 and blue <= 0.55:
            return "brown"
    channels = {"red": red, "green": green, "blue": blue}
    strongest = max(channels, key=channels.get)
    strongest_value = channels[strongest]
    second_value = sorted(channels.values())[-2]
    if strongest_value < 0.2 or strongest_value - second_value < 0.1:
        return "neutral"
    return strongest


def _ground_cliport_instances_from_text(instances: list[JsonDict], text: str) -> JsonDict:
    query = text.lower()
    grounding: JsonDict = {}
    color_words = ["red", "green", "blue", "yellow", "purple", "orange", "white", "black", "brown"]
    for color in color_words:
        if color not in query:
            continue
        candidates = _rank_instances_by_color(instances, color)
        if candidates:
            grounding[color] = candidates[:5]
    if "brown" in query and any(word in query for word in ["lightest", "light", "palest", "brightest"]):
        candidates = _rank_instances_by_light_brown_block(instances)
        if candidates:
            grounding["lightest_brown_block"] = candidates[:5]
        region_candidates = _rank_brown_surface_regions(instances, rank="lightest")
        if region_candidates:
            grounding["lightest_brown_support_region"] = region_candidates[:5]
    if any(word in query for word in ["bowl", "container", "cup", "plate"]):
        candidates = _rank_instances_by_container_shape(instances)
        if candidates:
            grounding["container_like"] = candidates[:5]
    if any(word in query for word in ["block", "cube"]):
        candidates = _rank_instances_by_block_shape(instances)
        if candidates:
            grounding["block_like"] = candidates[:5]
    if any(word in query for word in ["surface", "base", "zone", "mat", "plate"]):
        candidates = _rank_instances_by_surface_shape(instances)
        if candidates:
            grounding["thin_surface"] = candidates[:5]
    return grounding


def _rank_instances_by_color(instances: list[JsonDict], color: str) -> list[JsonDict]:
    prototypes: dict[str, tuple[float, float, float]] = {
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "purple": (0.5, 0.0, 0.8),
        "orange": (1.0, 0.45, 0.0),
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "brown": (0.62, 0.46, 0.34),
    }
    target = np.asarray(prototypes[color], dtype=float)
    scored: list[tuple[float, JsonDict]] = []
    for instance in instances:
        rgb = instance.get("dominant_rgb")
        if not isinstance(rgb, list) or len(rgb) < 3:
            if instance.get("dominant_color_name") == color:
                scored.append((0.0, instance))
            continue
        vector = np.asarray([float(value) for value in rgb[:3]], dtype=float)
        if vector.size != 3:
            continue
        if np.max(vector) > 1.5:
            vector = vector / 255.0
        distance = float(np.linalg.norm(vector - target))
        scored.append((distance, instance))
    return [_candidate_summary(instance, score=round(1.0 / (1.0 + distance), 4)) for distance, instance in sorted(scored, key=lambda item: item[0])]


def _rank_instances_by_container_shape(instances: list[JsonDict]) -> list[JsonDict]:
    scored: list[tuple[float, JsonDict]] = []
    for instance in instances:
        dims = instance.get("dimensions")
        if not isinstance(dims, list) or len(dims) < 3:
            continue
        x, y, z = [abs(float(value)) for value in dims[:3]]
        footprint = x * y
        flatness = footprint / max(z, 1e-6)
        score = footprint + 0.01 * flatness
        scored.append((-score, instance))
    return [_candidate_summary(instance, score=round(-score, 4)) for score, instance in sorted(scored, key=lambda item: item[0])]


def _rank_instances_by_block_shape(instances: list[JsonDict]) -> list[JsonDict]:
    scored: list[tuple[float, JsonDict]] = []
    for instance in instances:
        dims = instance.get("dimensions")
        if not isinstance(dims, list) or len(dims) < 3:
            continue
        tags = set(instance.get("geometry_tags") or [])
        values = [abs(float(value)) for value in dims[:3]]
        mean = sum(values) / 3.0
        equal_axes = sum(abs(value - mean) for value in values) / max(mean, 1e-6)
        volume = values[0] * values[1] * values[2]
        if "thin_surface" in tags and "block_like" not in tags:
            equal_axes += 10.0
        scored.append((equal_axes + volume, instance))
    return [_candidate_summary(instance, score=round(1.0 / (1.0 + score), 4)) for score, instance in sorted(scored, key=lambda item: item[0])]


def _rank_instances_by_surface_shape(instances: list[JsonDict]) -> list[JsonDict]:
    scored: list[tuple[float, JsonDict]] = []
    for instance in instances:
        dims = instance.get("dimensions")
        if not isinstance(dims, list) or len(dims) < 3:
            continue
        x, y, z = [abs(float(value)) for value in dims[:3]]
        thinness = min(x, y) / max(z, 1e-6)
        scored.append((-thinness, instance))
    return [_candidate_summary(instance, score=round(-score, 4)) for score, instance in sorted(scored, key=lambda item: item[0])]


def _rank_brown_surface_regions(instances: list[JsonDict], *, rank: str) -> list[JsonDict]:
    regions: list[JsonDict] = []
    for instance in instances:
        for region in instance.get("surface_regions") or []:
            if not isinstance(region, dict):
                continue
            label = str(region.get("region_label") or "")
            if rank and not label.startswith(rank):
                continue
            regions.append(region)
    return [
        {
            "object_id": region.get("object_id"),
            "parent_object_id": region.get("parent_object_id"),
            "region_label": region.get("region_label"),
            "pose0": region.get("pose0"),
            "dimensions": region.get("dimensions"),
            "geometry_tags": region.get("geometry_tags"),
            "affordance_hints": region.get("affordance_hints"),
            "dominant_color_name": region.get("dominant_color_name"),
            "dominant_rgb": region.get("dominant_rgb"),
            "score": region.get("dominant_rgb", [0, 0, 0])[0] if isinstance(region.get("dominant_rgb"), list) else 1.0,
        }
        for region in regions
    ]


def _rank_instances_by_light_brown_block(instances: list[JsonDict]) -> list[JsonDict]:
    scored: list[tuple[float, JsonDict]] = []
    for instance in instances:
        tags = set(instance.get("geometry_tags") or [])
        if "block_like" not in tags:
            continue
        rgb = instance.get("dominant_rgb")
        if not isinstance(rgb, list) or len(rgb) < 3:
            continue
        red, green, blue = [float(value) for value in rgb[:3]]
        if max(red, green, blue) > 1.5:
            red, green, blue = red / 255.0, green / 255.0, blue / 255.0
        warm_neutral = red >= green >= blue and red - blue >= 0.02
        if not warm_neutral and instance.get("dominant_color_name") not in {"brown", "light_brown"}:
            continue
        brightness = (red + green + blue) / 3.0
        scored.append((-brightness, instance))
    return [_candidate_summary(instance, score=round(-score, 4)) for score, instance in sorted(scored, key=lambda item: item[0])]


def _candidate_summary(instance: JsonDict, *, score: float) -> JsonDict:
    return {
        "object_id": instance.get("object_id"),
        "pose0": instance.get("pose0"),
        "dimensions": instance.get("dimensions"),
        "geometry_tags": instance.get("geometry_tags"),
        "affordance_hints": instance.get("affordance_hints"),
        "dominant_color_name": instance.get("dominant_color_name"),
        "dominant_rgb": instance.get("dominant_rgb"),
        "score": score,
    }


def _cliport_geometry_tags(dimensions: list[float] | None) -> list[str]:
    if not isinstance(dimensions, list) or len(dimensions) < 3:
        return []
    x, y, z = [abs(float(value)) for value in dimensions[:3]]
    horizontal_min = min(x, y)
    horizontal_max = max(x, y)
    max_dim = max(x, y, z)
    min_dim = min(x, y, z)
    tags: list[str] = []
    if min_dim > 0 and max_dim / min_dim <= 1.4:
        tags.append("block_like")
    if horizontal_min > 0 and z / horizontal_min <= 0.35:
        tags.append("thin_surface")
    if horizontal_min > 0 and z / horizontal_min <= 0.5 and horizontal_max >= 0.06:
        tags.append("container_like")
    return tags


def _cliport_affordance_hints(dimensions: list[float] | None) -> list[str]:
    tags = set(_cliport_geometry_tags(dimensions))
    hints: list[str] = []
    if "block_like" in tags:
        hints.append("grasp_candidate")
        hints.append("stackable_candidate")
    if "thin_surface" in tags or "container_like" in tags:
        hints.append("place_support_candidate")
    if "thin_surface" in tags and "block_like" not in tags:
        hints.append("low_profile_surface")
    return hints


def _cliport_surface_regions(
    *,
    object_id: int,
    dimensions: list[float] | None,
    object_pose0: list[Any] | None,
    visual_parts: list[JsonDict],
) -> list[JsonDict]:
    if "thin_surface" not in _cliport_geometry_tags(dimensions) or len(visual_parts) < 2:
        return []
    brown_parts = [
        part for part in visual_parts
        if str(part.get("color_name")) in {"light_brown", "brown", "dark_brown"}
    ]
    if len(brown_parts) < 2:
        return []
    sorted_parts = sorted(brown_parts, key=lambda part: float(part.get("brightness") or 0.0), reverse=True)
    rank_labels = ["lightest_brown_block", "middle_brown_block", "darkest_brown_block"]
    regions: list[JsonDict] = []
    for index, part in enumerate(sorted_parts):
        rank_label = rank_labels[index] if index < len(rank_labels) else f"brown_surface_region_{index}"
        pose0 = part.get("pose0") or object_pose0
        regions.append(
            {
                "region_label": rank_label,
                "label": rank_label.replace("_", " "),
                "aliases": [rank_label.replace("_", " "), rank_label],
                "parent_object_id": object_id,
                "object_id": object_id,
                "part_index": part.get("part_index"),
                "link_index": part.get("link_index"),
                "pose0": pose0,
                "dimensions": part.get("dimensions") or dimensions,
                "dominant_rgb": part.get("rgb"),
                "dominant_color_name": part.get("color_name"),
                "geometry_tags": ["thin_surface", "place_region"],
                "affordance_hints": ["place_support_candidate", "low_profile_surface"],
                "source": "pybullet_visual_shape_region",
            }
        )
    return regions


def _visual_part_world_pose(
    pybullet: Any,
    object_id: int,
    link_index: int,
    local_pos: Any,
    local_orn: Any,
) -> list[Any] | None:
    try:
        if link_index >= 0:
            link_state = pybullet.getLinkState(object_id, link_index, computeForwardKinematics=True)
            if isinstance(link_state, (list, tuple)) and len(link_state) >= 6:
                base_pos, base_orn = link_state[4], link_state[5]
            elif isinstance(link_state, (list, tuple)) and len(link_state) >= 2:
                base_pos, base_orn = link_state[0], link_state[1]
            else:
                return None
        else:
            base_pos, base_orn = pybullet.getBasePositionAndOrientation(object_id)
        position, orientation = pybullet.multiplyTransforms(base_pos, base_orn, local_pos, local_orn)
    except Exception:  # noqa: BLE001 - visual part pose is helpful but not required.
        return None
    return [_to_builtin(position), _to_builtin(orientation)]


def _normalize_pose(pose: Any) -> list[Any] | None:
    if pose is None:
        return None
    if isinstance(pose, dict):
        position = pose.get("position") or pose.get("xyz")
        rotation = pose.get("rotation") or pose.get("quaternion") or pose.get("quat")
        if position is not None and rotation is not None:
            return [_to_builtin(position), _to_builtin(rotation)]
    if isinstance(pose, (list, tuple)) and len(pose) == 2:
        return [_to_builtin(pose[0]), _to_builtin(pose[1])]
    return None


def _cliport_action_schema() -> JsonDict:
    pose_schema = {
        "type": "pose",
        "format": "[position_xyz, quaternion_xyzw]",
        "position_xyz": "list[float] length 3, world frame",
        "quaternion_xyzw": "list[float] length 4",
    }
    return {
        "native_env_step": "env.step({'pose0': pose0, 'pose1': pose1})",
        "pose0": pose_schema,
        "pose1": pose_schema,
    }


def _import_cliport_env_only() -> tuple[Any, Any]:
    spec = importlib.util.find_spec("cliport")
    if spec is None or spec.submodule_search_locations is None:
        raise RuntimeError("CLIPort live runtime requires the upstream `cliport` package in this Python environment.")
    package_paths = list(spec.submodule_search_locations)
    for name in list(sys.modules):
        if name == "cliport" or name.startswith("cliport."):
            del sys.modules[name]
    package = types.ModuleType("cliport")
    package.__file__ = spec.origin
    package.__path__ = package_paths  # type: ignore[attr-defined]
    package.__package__ = "cliport"
    package.__spec__ = spec
    sys.modules["cliport"] = package
    tasks = importlib.import_module("cliport.tasks")
    environment_module = importlib.import_module("cliport.environments.environment")
    return tasks, environment_module


def _module_importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        return name in sys.modules


def _activate_repo_local_cliport_source() -> str | None:
    configured = os.environ.get("CLIPORT_SOURCE_DIR")
    project_paths = get_project_paths()
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_paths.external_upstream("cliport"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        source_dir = candidate.resolve()
        if not (source_dir / "cliport").is_dir():
            continue
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        return source_text
    return None


def _default_assets_root() -> str | None:
    root = os.environ.get("CLIPORT_ROOT")
    if root:
        return str(Path(root) / "cliport" / "environments" / "assets")
    source_path = _activate_repo_local_cliport_source()
    if source_path:
        candidate = Path(source_path) / "cliport" / "environments" / "assets"
        if candidate.exists():
            return str(candidate)
    return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(_to_builtin(key)): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
