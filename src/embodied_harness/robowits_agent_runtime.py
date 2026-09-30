from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib.util
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import sys
import sysconfig
from typing import Any, Callable

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]


DEFAULT_ROBOWITS_REPO_CANDIDATES = (
    get_project_paths().external_upstream("robowits"),
)
DEFAULT_LEROBOT_REPO = get_project_paths().external_upstream("lerobot")
DEFAULT_ROBOWITS_REPO = next((path for path in DEFAULT_ROBOWITS_REPO_CANDIDATES if path.exists()), DEFAULT_ROBOWITS_REPO_CANDIDATES[0])
GENESIS_CONSTRAINT_SOLVER_ENV = "EMBODIED_ARENA_GENESIS_CONSTRAINT_SOLVER"
_GENESIS_PREFER_DECOMPOSED_SOLVER = -1


def _repo_local_robowits_python(repo_path: str | Path) -> Path | None:
    repo = Path(repo_path).expanduser().absolute()
    candidate = repo / ".venv/bin/python"
    if candidate.is_file():
        return candidate
    return None


ROBOWITS_PRIMITIVE_ALIASES = {
    "inspect_scene": "inspect_robowits_scene_or_tool",
    "observe_state": "observe_robowits_state",
    "inspect_camera_pixels": "inspect_robowits_camera_pixels",
    "inspect_object_poses": "inspect_robowits_object_poses",
    "locate_entity": "locate_robowits_entity",
    "inspect_grasp_state": "inspect_robowits_grasp_state",
    "inspect_contact_stability": "inspect_robowits_contact_stability",
    "query_wrist_orientations": "query_robowits_wrist_orientations",
    "query_motion": "query_robowits_motion",
    "control_ee": "execute_robowits_ee_control",
    "inspect_motion_outcome": "inspect_robowits_motion_outcome",
    "record_evidence": "record_robowits_evidence",
}


def _configure_genesis_constraint_solver(mode: str) -> None:
    """Select Genesis' public rigid constraint-solver implementation.

    Quadrants 0.8.0 ships its GPU graph-control helper as an architecture-
    specific fatbin.  The wheel used by the pinned RoboWits runtime does not
    contain SM90 code, while Genesis' monolithic solver uses ordinary kernels
    and implements the same constraint model.  Keep the switch local to the
    RoboWits worker instead of modifying the installed runtime.
    """

    normalized = str(mode or "auto").strip().lower()
    values = {"auto": -1, "monolithic": 0, "decomposed": 1}
    if normalized not in values:
        raise ValueError(
            "genesis_constraint_solver must be one of: auto, monolithic, decomposed"
        )
    if normalized == "auto":
        return

    global _GENESIS_PREFER_DECOMPOSED_SOLVER
    _GENESIS_PREFER_DECOMPOSED_SOLVER = values[normalized]

    import genesis as gs

    # RigidSolver cannot be imported before gs.init(): Genesis has not created
    # qd_float and the annotated solver modules fail during import. Install a
    # tiny one-shot init hook when the runtime is still uninitialized.
    if not hasattr(gs, "qd_float"):
        original_attribute = "_embodied_arena_original_init"
        if not hasattr(gs, original_attribute):
            original_init = gs.init
            setattr(gs, original_attribute, original_init)

            def _init_with_solver_selection(*args: Any, **kwargs: Any) -> Any:
                result = original_init(*args, **kwargs)
                _install_genesis_rigid_solver_selection()
                return result

            _init_with_solver_selection.__name__ = original_init.__name__
            _init_with_solver_selection.__doc__ = original_init.__doc__
            gs.init = _init_with_solver_selection
        return
    _install_genesis_rigid_solver_selection()


def _install_genesis_rigid_solver_selection() -> None:
    from genesis.engine.solvers.rigid.rigid_solver import RigidSolver

    target_attribute = "_embodied_arena_prefer_decomposed_solver"
    original_attribute = "_embodied_arena_original_build_static_config"
    if not hasattr(RigidSolver, original_attribute):
        original = RigidSolver._build_static_config
        setattr(RigidSolver, original_attribute, original)

        def _build_static_config_with_solver_selection(self: Any) -> None:
            original(self)
            selected = getattr(type(self), target_attribute)
            self._static_rigid_sim_config.prefer_decomposed_solver = selected

        _build_static_config_with_solver_selection.__name__ = original.__name__
        _build_static_config_with_solver_selection.__doc__ = original.__doc__
        RigidSolver._build_static_config = _build_static_config_with_solver_selection
    setattr(
        RigidSolver,
        target_attribute,
        _GENESIS_PREFER_DECOMPOSED_SOLVER,
    )


@dataclass(slots=True)
class RoboWitsRuntimeConfig:
    repo_path: str = str(DEFAULT_ROBOWITS_REPO)
    task_id: str = "01"
    dataset_split: str = "eval_dataset_50"
    dataset_revision_manifest: str | None = field(default_factory=lambda: os.environ.get("EMBODIED_ARENA_ROBOWITS_DATASET_REVISION_MANIFEST") or None)
    episode_index: int = 0
    control_mode: str = "EE_ABS"
    observation_mode: str = "EE"
    device: str = "cpu"
    genesis_backend: str = "cpu"
    genesis_constraint_solver: str = "auto"
    live: bool = False
    allow_dataset_scene_mismatch_for_diagnostics: bool = False
    env_kwargs: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RoboWitsAgentRuntimeBackend(EmbodiedBackend):
    """Agent-native RoboWits adapter.

    Agent primitives expose benchmark observations, public geometry and robot
    state, generic motion queries, bounded low-level control, and evidence
    recording. Evaluation remains outside the primitive surface.
    """

    def __init__(
        self,
        config: RoboWitsRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.config = config or RoboWitsRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._task_name: str | None = None
        self._task_class_name: str | None = None
        self._episode_scene: JsonDict = {}
        self._dataset_summary: JsonDict = {}
        self._visual_handles: dict[str, JsonDict] = {}
        self._episode_status: JsonDict = _initial_robowits_episode_status()
        self._pool_episode: tuple[str, int, int, str] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        task_id = str(coordinate.get("task_id") or "")
        variation = str(coordinate.get("variation") or "")
        prefix = f"{task_id}::episode_"
        if (len(task_id) != 2 or not task_id.isdigit() or not 1 <= int(task_id) <= 30
                or not variation.startswith(prefix) or not variation[len(prefix):].isdigit()):
            raise ValueError("RoboWits requires a native task 01..30 and TaskID::episode_N")
        index = int(variation[len(prefix):])
        seed = coordinate.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("RoboWits reset seed must be an integer in [0, 2**32)")
        repo = get_project_paths().external_upstream("robowits")
        binding_config = self._merged_config({"repo_path": str(repo), "task_id": task_id,
                                              "episode_index": index, "dataset_split": "eval_dataset_50"})
        path = self._dataset_json_path(binding_config)
        episodes = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(episodes, list) or not 0 <= index < len(episodes):
            raise ValueError("Selected RoboWits episode is absent; index will not be clamped")
        self._pool_episode = (task_id, index, seed, str(repo))
        return {"bound": True, "mode": "native_dataset_episode", "task_id": task_id,
                "episode_index": index, "seed": seed, "dataset_file": str(path)}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_episode is not None:
            task_id, index, selected_seed, repo = self._pool_episode
            if seed != selected_seed:
                raise ValueError("RoboWits reset seed differs from selected pool coordinate")
            overrides.update(repo_path=repo, task_id=task_id, episode_index=index,
                             dataset_split="eval_dataset_50")
        runtime_config = self._merged_config(overrides)
        runtime_config.task_id = task_id or runtime_config.task_id
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=runtime_config.task_id)
        self._env = None
        self._last_obs = None
        self._last_info = {}
        self._visual_handles = {}
        self._robowits_held_objects = {}
        self._robowits_hold_action = None
        self._episode_status = _initial_robowits_episode_status()

        self._task_name, self._task_class_name = self._resolve_task_metadata(runtime_config)
        self._episode_scene, self._dataset_summary = self._load_episode_scene(runtime_config)
        self._task_spec = TaskSpec(
            task_id=runtime_config.task_id,
            source="w7:robowits:live_runtime_boundary",
            instruction=self._task_instruction(),
            goal={
                "benchmark": "RoboWits",
                "execution_contract": "benchmark_native_primitives",
            },
            initial_state={
                "episode_index": runtime_config.episode_index,
                "scene_summary": summarize_scene(self._episode_scene),
            },
            budgets={"primitive_calls": 48, "verifier_calls": 6},
            tags=["w7", "robowits", "creative_tool_use", "bimanual", "live" if runtime_config.live else "dataset_boundary"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "robowits",
                "runtime_config": runtime_config.to_dict(),
                "upstream": inspect_robowits_repo(runtime_config.repo_path),
                "dataset": self._dataset_summary,
                "agent_native_contract": {
                    "primitives_accept_prompt_query_context": True,
                    "scene_uses_real_eval_json": True,
                    "live_env_requires_genesis_assets": True,
                    "public_surface": "observation_geometry_motion_control_state_evidence",
                    "visual_provenance_required_by_motion": True,
                    "native_sensor_modalities": ["rgb"],
                    "native_depth_segmentation_pointcloud": False,
                },
            },
        )

        if runtime_config.live:
            self._env = self._make_env(runtime_config, seed=seed)
            selectors = []
            for candidate in _robowits_env_candidates(self._env):
                setter = getattr(candidate, "set_eval_dataset_idx", None)
                if callable(setter):
                    setter(runtime_config.episode_index)
                    selectors.append(type(candidate).__name__)
            if runtime_config.episode_index != 0 and not selectors:
                raise RuntimeError("selected_episode_cannot_be_bound_to_live_environment")
            self._task_spec.metadata["live_episode_binding"] = {
                "episode_index": runtime_config.episode_index,
                "selection_applied_before_reset": bool(selectors),
                "selector_types": selectors,
            }
            reset_result = self._env.reset(seed=seed) if seed is not None else self._env.reset()
            self._last_obs, self._last_info = _split_reset_result(reset_result)
            live_description = _robowits_env_task_description(self._env)
            if live_description:
                self._task_spec.instruction = live_description
                self._task_spec.metadata["instruction_source"] = "live_env.task_description"
            live_scene = self._public_scene()
            expected = set(self._episode_scene)
            actual = set(live_scene)
            # Native collect_objs_info also includes the fixed table, which is
            # not a sampled object in eval JSON.
            fixed_entities = {"table"}
            compatibility = {
                "compatible": bool(expected) and bool(actual) and (expected - fixed_entities) == (actual - fixed_entities),
                "dataset_only": sorted(expected - actual - fixed_entities),
                "live_only": sorted(actual - expected - fixed_entities),
                "episode_index": runtime_config.episode_index,
            }
            self.record_event("dataset_scene_compatibility", compatibility)
            self._task_spec.initial_state["scene_summary"] = summarize_scene(live_scene)
            self._task_spec.metadata["dataset"] = self._public_dataset_summary()
            self._task_spec.metadata["agent_native_contract"].update(
                scene_uses_real_eval_json=False,
                scene_source="live_public_object_state",
            )
            self._task_spec.metadata["dataset_scene_compatible"] = compatibility["compatible"]
            if not compatibility["compatible"] and not runtime_config.allow_dataset_scene_mismatch_for_diagnostics:
                self.close()
                raise RuntimeError("robowits_dataset_scene_mismatch: " + json.dumps(compatibility, sort_keys=True))

        self.record_event(
            "reset",
            {
                "task": self._task_spec.to_dict(),
                "seed": seed,
                "live_env": self._env is not None,
                "runtime_available": self.runtime_available(),
            },
        )
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        live_visual_evidence = robowits_live_visual_evidence(self._last_obs)
        visual_handles = (
            self._register_visual_handles(
                prompt=None,
                query=None,
                context={"source_interface": "scene.observe"},
            )
            if live_visual_evidence.get("visual_ready")
            else []
        )
        ee_state = robowits_public_ee_tracking_evidence(
            self._last_obs,
            target_action=None,
            schema=self._action_schema(),
            robot_base_position=_robowits_env_robot_base_position(self._env),
        )
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "task_context": self._task_context(),
                "scene_summary": summarize_scene(self._public_scene()),
                "observation_summary": summarize_observation(self._last_obs),
                "live_visual_evidence": live_visual_evidence,
                "evidence_handles": visual_handles,
                "native_visual_evidence_handles": visual_handles,
                "visual_provenance_source": "live_genesis_observation" if visual_handles else None,
                "ee_state": ee_state,
                "action_schema": self._action_schema(),
            },
            metadata={"benchmark_id": "robowits", "task_id": self.config.task_id},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "get_robowits_task_context",
                "L1",
                {"prompt": "str|None", "query": "str|None", "context": "dict|None"},
                {"task_context": "dict", "scene_summary": "dict", "runtime": "dict"},
                "Expose task language, dataset schema, and runtime availability without evaluator internals.",
            ),
            self._primitive_card(
                "observe_robowits_state",
                "L1",
                {"names": "list[str]|None", "query": "str|None", "context": "dict|None"},
                {"observation_summary": "dict", "live_visual_evidence": "dict", "evidence_handles": "list[str]", "object_state": "dict", "ee_state": "dict"},
                "Observe live visual, robot, and requested public object state.",
            ),
            self._primitive_card(
                "inspect_robowits_scene_or_tool",
                "L2",
                {"query": "str|None", "context": "dict|None"},
                {"selected": "dict|None", "candidates": "list[dict]", "scene_summary": "dict"},
                "Inspect current live public object/tool metadata by prompt/query/context matching; dataset-only mode uses eval JSON.",
            ),
            self._primitive_card(
                "inspect_robowits_scene_evidence",
                "L2",
                {"prompt": "str|None", "query": "str|None", "context": "dict|None"},
                {"scene_evidence": "dict", "geometry_candidates": "list[dict]", "workspace_bounds": "dict"},
                "Inspect prompt/query-conditioned current public object poses and bounds. Live queries never fall back to stale dataset geometry.",
            ),
            self._primitive_card(
                "locate_robowits_entity",
                "L2",
                {"query": "str|None", "entity_name": "str|None", "context": "dict|None", "return_types": "list[str]|None"},
                {"selected": "dict|None", "candidates": "list[dict]", "pose_world": "list[float]|None", "bbox_3d": "list|None"},
                "Locate a scene entity from public eval JSON names, pose, bounds, and agent query/context.",
            ),
            self._primitive_card(
                "inspect_robowits_live_observation",
                "L2",
                {"prompt": "str|None", "query": "str|None", "context": "dict|None"},
                {"live_visual_evidence": "dict", "evidence_handles": "list[str]", "runtime": "dict"},
                "Inspect native live camera arrays and issue provenance handles for later generic motion.",
            ),
            self._primitive_card(
                "inspect_robowits_camera_pixels",
                "L2",
                {"camera_name": "str|None", "bbox": "list[int]|None", "query": "str|None", "context": "dict|None"},
                {"camera": "dict", "regions": "list[dict]", "evidence": "dict"},
                "Inspect real RGB pixels from a live Genesis camera and return query-conditioned auditable region statistics.",
            ),
            self._primitive_card(
                "inspect_robowits_object_poses",
                "L2",
                {"names": "list[str]|None", "query": "str|None", "context": "dict|None"},
                {"object_poses": "dict", "diagnostics": "dict"},
                "Inspect live object poses, bounds, velocities, and public geometric diagnostics.",
            ),
            self._primitive_card(
                "inspect_robowits_grasp_state",
                "L2",
                {"object_name": "str", "arm": "right|left|auto", "context": "dict|None"},
                {"object": "dict", "arm": "str", "ee_distance": "float|None", "gripper_width": "float|None", "grasp_likely": "bool"},
                "Estimate grasp state from public object bounds, end-effector pose, gripper width, and object motion.",
            ),
            self._primitive_card(
                "inspect_robowits_contact_stability",
                "L2",
                {"names": "list[str]", "velocity_threshold": "float|None", "contact_tolerance": "float|None", "context": "dict|None"},
                {"objects": "dict", "pairwise_contacts": "list[dict]", "all_stable": "bool"},
                "Inspect generic pairwise AABB contact and velocity-based stability from public simulator state.",
            ),
            self._primitive_card(
                "query_robowits_wrist_orientations",
                "L2",
                {
                    "object_name": "str|None",
                    "position": "list[float]|None",
                    "arm": "right|left|auto",
                    "reference_direction": "list[float]|None",
                    "yaw_offsets": "list[float]|None",
                    "ignored_names": "list[str]|None",
                    "context": "dict|None",
                },
                {"candidates": "list[dict]", "collision_free_candidates": "list[str]", "selected": "None"},
                "Query task-agnostic wrist-yaw candidates and public AABB clearances without choosing a task objective or action.",
            ),
            self._primitive_card(
                "query_robowits_motion",
                "L2",
                {
                    "target_position": "list[float]",
                    "arm": "right|left|auto",
                    "axis_angle": "list[float]|None",
                    "ignored_names": "list[str]|None",
                    "clearance_radius": "float|None",
                    "samples": "int|None",
                    "evidence_handles": "list[str]",
                    "context": "dict|None",
                },
                {"ik": "dict", "path": "dict", "start_position": "list[float]", "target_position": "list[float]"},
                "Query generic IK reachability and straight-line public-geometry clearance without moving the robot.",
            ),
            self._primitive_card(
                "execute_robowits_ee_control",
                "L3",
                {
                    "target_position": "list[float]",
                    "arm": "right|left|auto",
                    "axis_angle": "list[float]|None",
                    "gripper_width": "float|None",
                    "repeat_steps": "int|None",
                    "hold_steps": "int|None",
                    "max_translation": "float|None",
                    "observe_names": "list[str]|None",
                    "evidence_handles": "list[str]",
                    "context": "dict|None",
                },
                {
                    "executed": "bool",
                    "public_state_before": "dict",
                    "public_state_after": "dict",
                    "tracking": "dict",
                    "termination": "dict",
                },
                "Execute one caller-specified bounded EE_ABS/gripper command and re-observe public robot/object state.",
            ),
            self._primitive_card(
                "inspect_robowits_motion_outcome",
                "L2",
                {
                    "object_name": "str",
                    "arm": "right|left|auto",
                    "previous_object_position": "list[float]|None",
                    "previous_ee_position": "list[float]|None",
                    "reference_object_ee_offset": "list[float]|None",
                    "context": "dict|None",
                },
                {"object_displacement": "dict", "attachment": "dict", "failure_signals": "list[str]"},
                "Diagnose displacement or suspected attachment loss from public state only; return no recovery action or task decision.",
            ),
            self._primitive_card(
                "record_robowits_evidence",
                "L1",
                {"key": "str", "value": "any", "context": "dict|None"},
                {"artifact_id": "str"},
                "Record agent-selected task, scene, visual, or tool-use evidence in the trace.",
            ),
        ]
        for card in cards:
            if card.name == "execute_robowits_ee_control":
                card.description += " Long targets are split into legal waypoints within max_translation (never above 0.3m). Total repeat_steps/hold_steps are not increased. Each waypoint is checked; inspect target_reached and segments, not executed alone."
        from .robowits_manipulation import cards as manipulation_cards
        cards.extend(manipulation_cards(self))
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        requested_name = name
        name = ROBOWITS_PRIMITIVE_ALIASES.get(name, name)
        allowed = {card.name for card in self.list_primitives()}
        if name not in allowed:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by RoboWitsAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            if handler is None:
                from .robowits_manipulation import HANDLERS
                if name in HANDLERS:
                    handler = lambda **arguments: HANDLERS[name](self, **arguments)
            try:
                result = handler(**kwargs) if handler is not None else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
            except (TypeError, ValueError) as exc:
                result = PrimitiveResult(name=name, ok=False, error=f"invalid_arguments:{exc}")
        self.record_event("primitive_call", {"name": name, "requested_name": requested_name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "runtime_boundary", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope == "task":
            private_success = inspect_robowits_harness_private_success(self)
            if not private_success["available"]:
                message = "RoboWits task success is harness-private and no live success signal is available yet."
                blocker = "private_success_signal_unavailable"
            elif private_success["is_success"] is True:
                message = "RoboWits harness-private task success signal is true."
                blocker = None
            else:
                message = "RoboWits harness-private task success signal is available but false."
                blocker = "private_success_false_after_action_sequence"
            metadata = {
                "private_success": private_success,
                "native_episode_done": bool(self.episode_status()["reached"]),
                "native_episode_status": self.episode_status(),
                "success_function_exposed_to_agent": False,
                "agent_primitive_exposed": False,
                "harness_only": True,
            }
            if blocker is not None:
                metadata["blocker"] = blocker
            result = VerificationResult(
                ok=private_success["is_success"] is True,
                scope=scope,
                message=message,
                metrics={
                    "success": 1.0 if private_success["is_success"] is True else 0.0,
                    "success_signal_available": float(bool(private_success["available"])),
                    "env_created": float(self._env is not None),
                },
                metadata=metadata,
            )
        elif scope == "runtime_boundary":
            available = self.runtime_available()
            ok = bool(available.get("gs_gym_importable") and self._dataset_summary.get("loaded"))
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message="RoboWits package and dataset boundary are import/load ready" if ok else "RoboWits live runtime boundary is blocked",
                metrics={"dataset_loaded": float(bool(self._dataset_summary.get("loaded"))), "env_created": float(self._env is not None)},
                metadata=available,
            )
        elif scope in {"official_success_readiness", "controller_readiness"}:
            private_success = inspect_robowits_harness_private_success(self)
            success_ready = private_success["is_success"] is True
            blocker = None
            if not private_success.get("available"):
                blocker = "private_success_signal_unavailable"
            elif not success_ready:
                blocker = "private_success_false_after_agent_execution"
            result = VerificationResult(
                ok=success_ready,
                scope=scope,
                message="Harness-private official verifier is true." if success_ready else "Harness-private official verifier has not returned true.",
                metrics={
                    "private_success": 1.0 if success_ready else 0.0,
                    "success_signal_available": float(bool(private_success.get("available"))),
                },
                metadata={
                    "private_success": private_success,
                    "blocker": blocker,
                    "readiness_route": "harness_private_official_verifier_only",
                    "agent_primitive_exposed": False,
                    "success_function_exposed_to_agent": False,
                    "harness_only": True,
                },
            )
        else:
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported RoboWits verification scope: {scope}")
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

    def episode_status(self) -> JsonDict:
        """Return the public terminal boundary state without evaluator signals."""

        return deepcopy(self._episode_status)

    def runtime_available(self) -> JsonDict:
        result = robowits_preflight_summary(
            repo_path=self.config.repo_path,
            dataset_split=self.config.dataset_split,
            dataset_revision_manifest=self.config.dataset_revision_manifest,
            task_id=self.config.task_id,
            episode_index=self.config.episode_index,
            live=self.config.live,
            env_created=self._env is not None,
            task_name=self._task_name,
        )
        if self._env is not None and "dataset" in result:
            result["dataset"] = self._public_dataset_summary()
        return result

    def _public_dataset_summary(self) -> JsonDict:
        summary = deepcopy(self._dataset_summary)
        if self.config.live:
            # Historical names and geometry remain in the evaluator trace/data,
            # not in the agent's description of the current scene.
            summary.pop("object_names", None)
            summary.pop("object_count", None)
            summary["current_entities_source"] = "live_public_object_state"
        return summary

    def _public_scene(self) -> JsonDict:
        if not self.config.live:
            return self._episode_scene
        if self._env is None:
            return {}
        records, error = _robowits_live_records(self._env)
        if error:
            return {}  # Do not silently substitute initial dataset geometry.
        return {name: {**record, "source": "robowits_live_public_object_state"}
                for name, record in records.items()}

    def _merged_config(self, overrides: JsonDict) -> RoboWitsRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return RoboWitsRuntimeConfig(**data)

    def _make_env(self, config: RoboWitsRuntimeConfig, seed: int | None) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.task_id, {**config.to_dict(), "seed": seed})
        repo = Path(config.repo_path)
        gs_gym_ok, gs_gym_error = _actual_importable("gs_gym", repo)
        if not gs_gym_ok:
            repo_python = _repo_local_robowits_python(repo)
            hint = (
                f" Repo-local runtime Python detected at {repo_python}; configure the native harness "
                "to launch that interpreter or install the same Genesis dependency stack into the current interpreter."
                if repo_python is not None
                else ""
            )
            detail = f" Import error: {gs_gym_error}." if gs_gym_error else ""
            raise RuntimeError(
                "RoboWits live runtime requires importable repo-local `gs_gym` with Genesis dependencies installed."
                f"{detail}{hint}"
            )
        import sys

        sys.path.insert(0, str(repo))
        import gs_gym
        from gs_gym.envs.robowits.utils import resolve_task

        _configure_genesis_constraint_solver(
            os.environ.get(
                GENESIS_CONSTRAINT_SOLVER_ENV,
                config.genesis_constraint_solver,
            )
        )
        task_name = resolve_task(config.task_id)
        kwargs = dict(config.env_kwargs)
        kwargs.update(
            {
                "control_mode": config.control_mode,
                "observation_mode": config.observation_mode,
                "device": config.device,
                "genesis_backend": config.genesis_backend,
                "eval_dataset_json": str(self._dataset_json_path(config)),
            }
        )
        return gs_gym.make(task_name, n_envs=1, **kwargs)

    def _primitive_get_robowits_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_robowits_task_context",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "context": context or {},
                "task_context": self._task_context(prompt=prompt, query=query, context=context or {}),
                "scene_summary": summarize_scene(self._public_scene()),
                "runtime": self.runtime_available(),
            },
        )

    def _primitive_inspect_robowits_scene_or_tool(
        self,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        scene = self._public_scene()
        candidates = scene_candidates(scene, query=query, context=context or {})
        selected = candidates[0] if candidates else None
        return PrimitiveResult(
            name="inspect_robowits_scene_or_tool",
            ok=selected is not None,
            output={
                "query": query,
                "context": context or {},
                "selected": selected,
                "candidates": candidates,
                "scene_summary": summarize_scene(scene),
                "evidence": {"source": "robowits_live_public_object_state" if self.config.live else "robowits_eval_json", "episode_index": self.config.episode_index},
            },
            error=None if selected is not None else "scene_entity_not_found",
        )

    def _primitive_inspect_robowits_scene_evidence(
        self,
        prompt: str | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence = robowits_scene_evidence(self._public_scene(), prompt=prompt, query=query, context=context or {})
        if self.config.live:
            evidence["source"] = "robowits_live_public_object_state"
            evidence["scene_evidence"]["live_rgb_available"] = bool(robowits_live_visual_evidence(self._last_obs).get("visual_ready"))
        return PrimitiveResult(
            name="inspect_robowits_scene_evidence",
            ok=bool(evidence["geometry_candidates"]),
            output=evidence,
            error=None if evidence["geometry_candidates"] else "scene_entity_not_found",
        )

    def _primitive_locate_robowits_entity(
        self,
        query: str | None = None,
        entity_name: str | None = None,
        context: JsonDict | None = None,
        return_types: list[str] | None = None,
    ) -> PrimitiveResult:
        ctx = context or {}
        candidates = geometry_candidates(self._public_scene(), query=entity_name or query, context=ctx)
        selected = _select_robowits_entity(candidates, entity_name=entity_name, query=query)
        requested = return_types or ["pose_world", "bbox_3d", "dimensions", "spatial_affordances"]
        return PrimitiveResult(
            name="locate_robowits_entity",
            ok=selected is not None,
            output={
                "query": query,
                "entity_name": entity_name,
                "context": ctx,
                "return_types": requested,
                "selected": selected,
                "candidates": candidates,
                "pose_world": _robowits_pose_world(selected),
                "bbox_3d": selected.get("bbox_3d") if selected else None,
                "evidence": {
                    "source": "robowits_live_public_object_state" if self.config.live else "robowits_eval_json_scene_geometry",
                    "uses_public_pose_and_bounds": True,
                    "requires_live_env_for_pixels": True,
                },
            },
            error=None if selected is not None else "scene_entity_not_found",
        )

    def _primitive_inspect_robowits_live_observation(
        self,
        prompt: str | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence = robowits_live_visual_evidence(self._last_obs)
        handles = self._register_visual_handles(prompt=prompt, query=query, context=context or {})
        runtime = self.runtime_available()
        if self._env is None:
            error = "live_env_not_created"
        elif not evidence["visual_ready"]:
            error = "live_visual_observation_unavailable"
        else:
            error = None
        return PrimitiveResult(
            name="inspect_robowits_live_observation",
            ok=error is None,
            output={
                "prompt": prompt,
                "query": query,
                "context": context or {},
                "live_visual_evidence": evidence,
                "evidence_handles": handles,
                "provenance_source": "live_genesis_observation",
                "observation_summary": summarize_observation(self._last_obs),
                "runtime": runtime,
                "dataset_geometry_boundary": {
                    "eval_json_is_not_rgb": True,
                    "live_genesis_env_required_for_camera_frames": True,
                },
            },
            error=error,
        )

    def _primitive_inspect_robowits_camera_pixels(
        self,
        camera_name: str | None = None,
        bbox: Any = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        ctx = context or {}
        evidence = robowits_camera_pixel_evidence(
            self._last_obs,
            camera_name=camera_name,
            bbox=bbox,
            query=query,
            context=ctx,
        )
        error = evidence.get("error")
        if self._env is None:
            error = "live_env_not_created"
        handles = self._register_visual_handles(
            prompt=None,
            query=query,
            context=ctx,
            camera_name=evidence.get("selected_camera"),
            modalities={"rgb"},
        )
        return PrimitiveResult(
            name="inspect_robowits_camera_pixels",
            ok=error is None,
            output={
                "camera_name": camera_name,
                "query": query,
                "context": ctx,
                **evidence,
                "evidence_handles": handles,
                "runtime": self.runtime_available(),
            },
            error=error,
        )

    def _primitive_inspect_robowits_object_poses(
        self,
        names: list[str] | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence = robowits_live_object_pose_evidence(
            self._env,
            names=names,
            query=query,
            context=context or {},
            reference_scene=self._episode_scene,
        )
        return PrimitiveResult(
            name="inspect_robowits_object_poses",
            ok=bool(evidence.get("available")),
            output=evidence,
            error=evidence.get("error"),
        )

    def _primitive_observe_robowits_state(
        self,
        names: list[str] | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        object_state = robowits_live_object_pose_evidence(
            self._env,
            names=names,
            query=query,
            context=context or {},
            reference_scene=self._episode_scene,
        )
        live_visual_evidence = robowits_live_visual_evidence(self._last_obs)
        visual_handles = (
            self._register_visual_handles(
                prompt=None,
                query=query,
                context={**(context or {}), "source_primitive": "observe_robowits_state"},
            )
            if live_visual_evidence.get("visual_ready")
            else []
        )
        ee_state = robowits_public_ee_tracking_evidence(
            self._last_obs,
            target_action=None,
            schema=self._action_schema(),
            robot_base_position=_robowits_env_robot_base_position(self._env),
        )
        output = {
            "query": query,
            "context": context or {},
            "observation_summary": summarize_observation(self._last_obs),
            "live_visual_evidence": live_visual_evidence,
            "evidence_handles": visual_handles,
            "native_visual_evidence_handles": visual_handles,
            "provenance_source": "live_genesis_observation" if visual_handles else None,
            "object_state": object_state,
            "ee_state": ee_state,
        }
        available = self._env is not None and any(
            (
                bool(object_state.get("available")),
                bool(live_visual_evidence.get("visual_ready")),
                bool(ee_state.get("available")),
            )
        )
        return PrimitiveResult(
            name="observe_robowits_state",
            ok=available,
            output=output,
            error=None if available else str(object_state.get("error") or "live_env_not_created"),
        )

    def _primitive_inspect_robowits_grasp_state(
        self,
        object_name: str,
        arm: str = "auto",
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence = robowits_grasp_state_evidence(
            self._env,
            self._last_obs,
            object_name=object_name,
            arm=arm,
            context=context or {},
        )
        return PrimitiveResult(
            name="inspect_robowits_grasp_state",
            ok=bool(evidence.get("available")),
            output=evidence,
            error=evidence.get("error"),
        )

    def _primitive_inspect_robowits_contact_stability(
        self,
        names: list[str],
        velocity_threshold: float | None = None,
        contact_tolerance: float | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence = robowits_contact_stability_evidence(
            self._env,
            names=names,
            velocity_threshold=velocity_threshold,
            contact_tolerance=contact_tolerance,
            context=context or {},
        )
        return PrimitiveResult(
            name="inspect_robowits_contact_stability",
            ok=bool(evidence.get("available")),
            output=evidence,
            error=evidence.get("error"),
        )

    def _primitive_query_robowits_wrist_orientations(
        self,
        object_name: str | None = None,
        position: list[float] | None = None,
        arm: str = "auto",
        reference_direction: list[float] | None = None,
        yaw_offsets: list[float] | None = None,
        ignored_names: list[str] | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        output = robowits_wrist_orientation_query(
            self._env,
            self._last_obs,
            object_name=object_name,
            position=position,
            arm=arm,
            reference_direction=reference_direction,
            yaw_offsets=yaw_offsets,
            ignored_names=ignored_names,
            context=context or {},
        )
        return PrimitiveResult(
            name="query_robowits_wrist_orientations",
            ok=bool(output.get("available")),
            output=output,
            error=output.get("error"),
        )

    def _primitive_query_robowits_motion(
        self,
        target_position: list[float],
        arm: str = "auto",
        axis_angle: list[float] | None = None,
        ignored_names: list[str] | None = None,
        clearance_radius: float | None = None,
        samples: int | None = None,
        evidence_handles: list[str] | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="query_robowits_motion",
                ok=False,
                output={"evidence_handles": evidence_handles or []},
                error=provenance_error,
            )
        output = robowits_motion_query(
            self._env,
            self._last_obs,
            target_position=target_position,
            arm=arm,
            axis_angle=axis_angle,
            ignored_names=ignored_names,
            clearance_radius=clearance_radius,
            samples=samples,
            context=context or {},
        )
        output["grounding_provenance"] = provenance
        return PrimitiveResult(
            name="query_robowits_motion",
            ok=bool(output.get("available")),
            output=output,
            error=output.get("error"),
        )

    def _primitive_execute_robowits_ee_control(
        self,
        target_position: list[float],
        arm: str = "auto",
        axis_angle: list[float] | None = None,
        gripper_width: float | None = None,
        repeat_steps: int | None = None,
        hold_steps: int | None = None,
        max_translation: float | None = None,
        observe_names: list[str] | None = None,
        evidence_handles: list[str] | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="execute_robowits_ee_control",
                ok=False,
                output={"evidence_handles": evidence_handles or [], "executed": False},
                error=provenance_error,
            )
        output = execute_robowits_ee_control(
            self,
            target_position=target_position,
            arm=arm,
            axis_angle=axis_angle,
            gripper_width=gripper_width,
            repeat_steps=repeat_steps,
            hold_steps=hold_steps,
            max_translation=max_translation,
            observe_names=observe_names,
            context=context or {},
        )
        output["grounding_provenance"] = provenance
        return PrimitiveResult(
            name="execute_robowits_ee_control",
            ok=bool(output.get("target_reached")),
            output=output,
            error=output.get("error"),
        )

    def _primitive_inspect_robowits_motion_outcome(
        self,
        object_name: str,
        arm: str = "auto",
        previous_object_position: list[float] | None = None,
        previous_ee_position: list[float] | None = None,
        reference_object_ee_offset: list[float] | None = None,
        context: JsonDict | None = None,
    ) -> PrimitiveResult:
        output = robowits_motion_outcome_evidence(
            self._env,
            self._last_obs,
            object_name=object_name,
            arm=arm,
            previous_object_position=previous_object_position,
            previous_ee_position=previous_ee_position,
            reference_object_ee_offset=reference_object_ee_offset,
            context=context or {},
        )
        return PrimitiveResult(
            name="inspect_robowits_motion_outcome",
            ok=bool(output.get("available")),
            output=output,
            error=output.get("error"),
        )

    def _primitive_record_robowits_evidence(self, key: str, value: Any, context: JsonDict | None = None) -> PrimitiveResult:
        artifact_id = f"robowits:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value), "context": context or {}})
        return PrimitiveResult(name="record_robowits_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _register_visual_handles(
        self,
        *,
        prompt: str | None,
        query: str | None,
        context: JsonDict,
        camera_name: str | None = None,
        modalities: set[str] | None = None,
    ) -> list[str]:
        handles: list[str] = []
        for source_name, modality, value in _robowits_visual_arrays(self._last_obs):
            if camera_name and source_name != camera_name:
                continue
            if modalities and modality not in modalities:
                continue
            handle = f"robowits:visual:{len(self._visual_handles)}"
            self._visual_handles[handle] = {
                "handle": handle,
                "source": "live_genesis_observation",
                "camera_name": source_name,
                "modality": modality,
                "summary": summarize_observation(value),
                "prompt": prompt,
                "query": query,
                "context": _to_builtin(context),
                "raw_value": value,
            }
            handles.append(handle)
        return handles

    def _consume_visual_handles(self, handles: list[str] | None) -> tuple[list[JsonDict], str | None]:
        requested = [str(handle) for handle in handles or []]
        if not requested:
            return [], "visual_evidence_handles_required"
        missing = [handle for handle in requested if handle not in self._visual_handles]
        if missing:
            return [], f"unknown_visual_evidence_handles:{','.join(missing)}"
        return [
            {key: value for key, value in self._visual_handles[handle].items() if key != "raw_value"}
            for handle in requested
        ], None

    def _resolve_task_metadata(self, config: RoboWitsRuntimeConfig) -> tuple[str | None, str | None]:
        repo = Path(config.repo_path)
        if not _actual_importable("gs_gym", repo)[0]:
            return None, None
        import sys

        try:
            sys.path.insert(0, str(repo))
            import gs_gym
            from gs_gym.envs.robowits.utils import resolve_task

            task_name = resolve_task(config.task_id)
            task_class = gs_gym.get_task_class(task_name)
            return task_name, task_class.__name__
        except Exception:
            return None, None

    def _load_episode_scene(self, config: RoboWitsRuntimeConfig) -> tuple[JsonDict, JsonDict]:
        path = self._dataset_json_path(config)
        summary: JsonDict = {"path": str(path), "loaded": False}
        if config.dataset_revision_manifest:
            manifest = json.loads(Path(config.dataset_revision_manifest).read_text())
            entry = manifest["tasks"].get(_dataset_task_stem(config.task_id))
            if entry and config.episode_index in entry["episodes"]:
                summary["dataset_revision"] = {"revision": manifest["revision"], **entry}
        if not path.exists():
            summary["error"] = "dataset_json_missing"
            return {}, summary
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not data:
            summary["error"] = "dataset_json_not_nonempty_list"
            return {}, summary
        index = config.episode_index
        if type(index) is not int or not 0 <= index < len(data):
            raise ValueError("Selected RoboWits episode is absent; index will not be clamped")
        scene = data[index]
        summary.update(
            {
                "loaded": True,
                "episodes": len(data),
                "selected_episode_index": index,
                "object_count": len(scene) if isinstance(scene, dict) else None,
                "object_names": sorted(str(key) for key in scene) if isinstance(scene, dict) else [],
                "split": config.dataset_split,
            }
        )
        return scene if isinstance(scene, dict) else {}, summary

    def _dataset_json_path(self, config: RoboWitsRuntimeConfig) -> Path:
        task_stem = _dataset_task_stem(config.task_id)
        original = Path(config.repo_path) / "dataset" / "robowits" / config.dataset_split / f"{task_stem}.json"
        if not config.dataset_revision_manifest:
            return original
        manifest_path = Path(config.dataset_revision_manifest)
        entry = json.loads(manifest_path.read_text())["tasks"].get(task_stem)
        if not entry or config.episode_index not in entry["episodes"]:
            return original
        revised = manifest_path.parent / entry["file"]
        checks = [(original, entry["source_sha256"]), (revised, entry["sha256"])]
        if entry.get("environment_file"):
            checks.append((Path(config.repo_path) / entry["environment_file"], entry["environment_sha256"]))
        for path, expected in checks:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"robowits_dataset_revision_hash_mismatch: {path}")
        return revised

    def _task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        context: JsonDict | None = None,
    ) -> JsonDict:
        return {
            "prompt": prompt,
            "query": query,
            "context": context or {},
            "task_id": self.config.task_id,
            "task_name": self._task_name,
            "task_class": self._task_class_name,
            "instruction": self._task_instruction(),
            "dataset": self._public_dataset_summary(),
            "control_mode": self.config.control_mode,
            "observation_mode": self.config.observation_mode,
        }

    def _task_instruction(self) -> str:
        live_description = _robowits_env_task_description(self._env)
        if live_description:
            return live_description
        if self._task_name:
            readable = self._task_name.split("/")[-1].replace("-v0", "").replace("_", " ").replace("-", " ")
            return f"Solve the RoboWits bimanual creative/tool-use manipulation task: {readable}."
        return f"Solve RoboWits task {self.config.task_id} with bimanual creative tool use."

    def _action_schema(self) -> JsonDict:
        dim = 14 if self.config.control_mode.startswith("EE") else 16
        return {
            "control_mode": self.config.control_mode,
            "observation_mode": self.config.observation_mode,
            "action_dimension": dim,
            "layout": _robowits_action_layout(self.config.control_mode),
            "format": "RoboWits/Genesis action array or structured answer for harness-side policy integration",
            "private_eval_signal_exposed": False,
        }

    def _primitive_card(
        self,
        name: str,
        level: str,
        input_schema: JsonDict,
        output_schema: JsonDict,
        description: str,
    ) -> PrimitiveCard:
        return PrimitiveCard(
            name=name,
            capability_tags=["w7", "robowits", "creative_tool_use", "bimanual"],
            input_schema=input_schema,
            output_schema=output_schema,
            cost={"primitive_calls": 1},
            failure_modes=["runtime_dependency_missing", "asset_missing", "live_env_not_created", "wrong_arguments"],
            abstraction_level=level,
            leakage_risk="L2_privileged_scene_state" if name == "inspect_robowits_scene_or_tool" else "none",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def inspect_robowits_repo(repo_path: str) -> JsonDict:
    repo = Path(repo_path)
    pyproject = repo / "pyproject.toml"
    readme = repo / "README.md"
    official_policy = inspect_robowits_official_policy_controller(repo)
    return {
        "repo_path": str(repo),
        "repo_exists": repo.exists(),
        "readme_exists": readme.exists(),
        "pyproject_exists": pyproject.exists(),
        "dataset_eval_50_json": len(list((repo / "dataset/robowits/eval_dataset_50").glob("*.json"))),
        "dataset_mutation_json": len(list((repo / "dataset/robowits/eval_dataset_mutation_10").glob("*.json"))),
        "assets_setup_script": str(repo / "assets/setup_assets.sh"),
        "simulator": "Genesis World",
        "policy_eval": "LeRobot lerobot-eval scripts",
        "official_policy_controller": official_policy,
    }


def inspect_robowits_official_policy_controller(repo: Path) -> JsonDict:
    """Discover upstream policy/controller entrypoints without importing them."""

    candidates = [
        repo / "scripts/robowits/eval/eval.sh",
        repo / "scripts/robowits/eval/eval_mutation.sh",
        repo / "scripts/robowits/examples/run_env.py",
        repo / "scripts/robowits/train/train_act.sh",
        repo / "scripts/robowits/train/train_pi0.sh",
        repo / "scripts/robowits/train/train_pi05.sh",
    ]
    entrypoints: list[JsonDict] = []
    policy_families: set[str] = set()
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        rel = str(path.relative_to(repo))
        uses_lerobot_eval = "lerobot-eval" in text
        requires_checkpoint = "CHECKPOINT_PATH" in text or "--policy.path" in text
        if "--policy.type=act" in text:
            policy_families.add("act")
        if "--policy.type=pi0" in text:
            policy_families.add("pi0")
        if "--policy.type=pi05" in text:
            policy_families.add("pi05")
        kind = "policy_eval" if uses_lerobot_eval else "env_controller_smoke" if "env.step" in text else "policy_training"
        entrypoints.append(
            {
                "path": rel,
                "kind": kind,
                "uses_lerobot_eval": uses_lerobot_eval,
                "requires_checkpoint_path": requires_checkpoint,
                "uses_env_step": "env.step" in text,
            }
        )
    public_controller = _scan_public_controller_surfaces(repo)
    checkpoint_env_var = None
    checkpoint_path = None
    if os.environ.get("ROBOWITS_CHECKPOINT_PATH"):
        checkpoint_env_var = "ROBOWITS_CHECKPOINT_PATH"
        checkpoint_path = os.environ.get("ROBOWITS_CHECKPOINT_PATH")
    elif os.environ.get("CHECKPOINT_PATH"):
        checkpoint_env_var = "CHECKPOINT_PATH"
        checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    checkpoint_exists = bool(checkpoint_path and Path(checkpoint_path).expanduser().is_file())
    lerobot_eval_cli, lerobot_eval_source, lerobot_eval_configured_path = _resolve_robowits_command(
        "lerobot-eval",
        env_names=("ROBOWITS_LEROBOT_EVAL", "LEROBOT_EVAL"),
    )
    single_case_command = None
    eval_script = repo / "scripts/robowits/eval/eval.sh"
    mutation_eval_script = repo / "scripts/robowits/eval/eval_mutation.sh"
    if eval_script.is_file():
        single_case_command = (
            "TASK_IDS=${TASK_IDS:?Set TASK_IDS} N_EPISODES=${N_EPISODES:-1} ENV_DEVICE=${ENV_DEVICE:-cpu} "
            "EVAL_JSON_DIR=${EVAL_JSON_DIR:-dataset/robowits/eval_dataset_50} "
            "CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an official trained policy checkpoint} "
            "bash scripts/robowits/eval/eval.sh"
        )
    mutation_command = None
    if mutation_eval_script.is_file():
        mutation_command = (
            "TASK_IDS=${TASK_IDS:?Set TASK_IDS} MUTATION_IDS=${MUTATION_IDS:?Set MUTATION_IDS} "
            "N_EPISODES=${N_EPISODES:-1} ENV_DEVICE=${ENV_DEVICE:-cpu} "
            "EVAL_JSON_DIR=${EVAL_JSON_DIR:-dataset/robowits/eval_dataset_mutation_10} "
            "CHECKPOINT_PATH=${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to an official trained policy checkpoint} "
            "bash scripts/robowits/eval/eval_mutation.sh"
        )
    official_policy_eval_available = bool(eval_script.is_file() or mutation_eval_script.is_file())
    return {
        "repo_exists": repo.exists(),
        "entrypoints": entrypoints,
        "entrypoint_count": len(entrypoints),
        "policy_families": sorted(policy_families),
        "official_single_case_eval_available": bool(eval_script.is_file()),
        "official_mutation_eval_available": bool(mutation_eval_script.is_file()),
        "official_policy_eval_available": official_policy_eval_available,
        "requires_trained_checkpoint": official_policy_eval_available,
        "checkpoint_path_configured": bool(checkpoint_path),
        "checkpoint_env_var": checkpoint_env_var,
        "checkpoint_path": "<configured_redacted>" if checkpoint_path else None,
        "checkpoint_path_exists": checkpoint_exists,
        "lerobot_eval_cli_available": bool(lerobot_eval_cli),
        "lerobot_eval_cli": lerobot_eval_cli,
        "lerobot_eval_cli_source": lerobot_eval_source,
        "lerobot_eval_configured_path": lerobot_eval_configured_path,
        "lerobot_source_tree_available": (DEFAULT_LEROBOT_REPO / "pyproject.toml").is_file(),
        "lerobot_install_hint": "pip install -e external/upstreams/lerobot",
        "official_policy_eval_ready": bool(official_policy_eval_available and checkpoint_path and checkpoint_exists and lerobot_eval_cli),
        "official_policy_eval_blockers": _official_policy_eval_blockers(
            policy_eval_available=official_policy_eval_available,
            checkpoint_path_configured=bool(checkpoint_path),
            checkpoint_path_exists=checkpoint_exists,
            lerobot_eval_cli_available=bool(lerobot_eval_cli),
        ),
        "single_case_eval_command": single_case_command,
        "mutation_eval_command": mutation_command,
        "controller_smoke_command": (
            "python scripts/robowits/examples/run_env.py --task-id ${TASK_ID:?Set TASK_ID} "
            "--steps ${STEPS:-1} --device ${ENV_DEVICE:-cpu} --episode-id ${EPISODE_ID:-0}"
            if (repo / "scripts/robowits/examples/run_env.py").is_file()
            else None
        ),
        "public_scripted_controller_available": public_controller["available"],
        "public_controller_surfaces": public_controller["surfaces"],
        "official_single_case_blockers": _official_single_case_blockers(
            eval_script_exists=eval_script.is_file(),
            checkpoint_path_configured=bool(checkpoint_path),
            public_controller_available=bool(public_controller["available"]),
        ),
        "agent_primitive_exposes_policy_or_controller": False,
    }


def _scan_public_controller_surfaces(repo: Path) -> JsonDict:
    """Find public non-checkpoint controller hints while excluding reward/eval internals."""

    surfaces: list[JsonDict] = []
    candidate_roots = [repo / "scripts/robowits", repo / "gs_gym/solvers"]
    for root in candidate_roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh"}:
                continue
            rel = str(path.relative_to(repo))
            text = path.read_text(encoding="utf-8", errors="replace")
            lower = text.lower()
            if "lerobot-eval" in lower or "checkpoint_path" in lower or "--policy.path" in lower:
                continue
            if "_check_success" in lower or "get_reward" in lower:
                continue
            has_step = "env.step" in lower
            has_noop = "noop" in lower or 'obs["agent_pos"]' in text or "obs['agent_pos']" in text
            scripted_terms = any(term in lower for term in ("scripted", "heuristic", "controller", "waypoint", "motion plan"))
            if has_step or scripted_terms:
                surfaces.append(
                    {
                        "path": rel,
                        "env_step_smoke": has_step,
                        "noop_only": bool(has_step and has_noop and not scripted_terms),
                        "scripted_controller_terms": scripted_terms,
                    }
                )
    available = any(item["scripted_controller_terms"] and not item["noop_only"] for item in surfaces)
    return {"available": available, "surfaces": surfaces}


def _official_single_case_blockers(
    *,
    eval_script_exists: bool,
    checkpoint_path_configured: bool,
    public_controller_available: bool,
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if not eval_script_exists:
        blockers.append({"id": "official_eval_script_missing", "detail": "scripts/robowits/eval/eval.sh not found"})
    if eval_script_exists and not checkpoint_path_configured:
        blockers.append({"id": "checkpoint_path_missing", "detail": "Set CHECKPOINT_PATH or ROBOWITS_CHECKPOINT_PATH to a trained RoboWits policy checkpoint."})
    if not public_controller_available:
        blockers.append(
            {
                "id": "public_scripted_controller_missing",
                "detail": "Inspected public scripts expose env smoke/noop or trained-policy eval, not a non-checkpoint solved controller.",
            }
        )
    return blockers


def _official_policy_eval_blockers(
    *,
    policy_eval_available: bool,
    checkpoint_path_configured: bool,
    checkpoint_path_exists: bool,
    lerobot_eval_cli_available: bool,
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if not policy_eval_available:
        blockers.append(
            {
                "id": "official_policy_eval_script_missing",
                "detail": "scripts/robowits/eval/eval.sh or eval_mutation.sh not found",
            }
        )
    if policy_eval_available and not checkpoint_path_configured:
        blockers.append(
            {
                "id": "checkpoint_path_missing",
                "detail": "Set CHECKPOINT_PATH or ROBOWITS_CHECKPOINT_PATH to a trained RoboWits policy checkpoint.",
            }
        )
    if policy_eval_available and checkpoint_path_configured and not checkpoint_path_exists:
        blockers.append(
            {
                "id": "checkpoint_path_not_found",
                "detail": "Configured RoboWits checkpoint path does not point to a local file.",
            }
        )
    if policy_eval_available and not lerobot_eval_cli_available:
        blockers.append(
            {
                "id": "lerobot_eval_cli_missing",
                "detail": "Install external/upstreams/lerobot so the lerobot-eval console script is available.",
            }
        )
    return blockers


def _resolve_robowits_command(command: str, *, env_names: tuple[str, ...] = ()) -> tuple[str | None, str | None, str | None]:
    for env_name in env_names:
        configured = os.environ.get(env_name)
        if not configured:
            continue
        configured_path = Path(configured).expanduser()
        if configured_path.is_file() and os.access(configured_path, os.X_OK):
            return str(configured_path), env_name, configured
        return None, None, configured
    for sibling in (Path(sys.executable).parent / command, Path(sys.executable).resolve().parent / command):
        if sibling.is_file() and os.access(sibling, os.X_OK):
            return str(sibling), "current_python_bin", None
    path_value = shutil.which(command)
    if path_value:
        return path_value, "PATH", None
    return None, None, None


def _dataset_task_stem(task_id: str) -> str:
    """Return the base eval JSON stem for short, mutation, or full task ids."""

    task_stem = task_id.split("/")[-1].replace("-v0", "")
    if "_" in task_stem:
        task_stem = task_stem.split("_", 1)[0]
    return task_stem.zfill(2)


def robowits_preflight_summary(
    *,
    repo_path: str | None = None,
    dataset_split: str = "eval_dataset_50",
    dataset_revision_manifest: str | None = None,
    task_id: str = "01",
    episode_index: int = 0,
    live: bool = False,
    env_created: bool = False,
    task_name: str | None = None,
) -> JsonDict:
    """Return machine-readable RoboWits runtime readiness without secret values."""

    config = RoboWitsRuntimeConfig(
        repo_path=repo_path or str(DEFAULT_ROBOWITS_REPO),
        dataset_split=dataset_split,
        dataset_revision_manifest=dataset_revision_manifest,
        task_id=task_id,
        episode_index=episode_index,
        live=live,
    )
    repo = Path(config.repo_path)
    gs_gym_ok, gs_gym_error = _actual_importable("gs_gym", repo)
    dataset_path = RoboWitsAgentRuntimeBackend(config)._dataset_json_path(config)
    dataset_summary = _dataset_file_summary(dataset_path, episode_index=episode_index)
    gs_gym_discoverable = _module_discoverable("gs_gym", repo)
    genesis_probe = _import_probe("genesis")
    gymnasium_probe = _import_probe("gymnasium")
    lerobot_probe = _import_probe("lerobot")
    torch_probe = _torch_probe()
    pymeshlab_probe = _import_probe("pymeshlab")
    payload = {
        "runtime": {
            "live": live,
            "env_created": env_created,
            "repo_exists": repo.exists(),
            "repo_path": str(repo),
            "task_id": task_id,
            "task_name": task_name,
            "dataset_split": dataset_split,
            "dataset_loaded": bool(dataset_summary.get("loaded")),
        },
        "python": _python_platform_summary(),
        "commands": {
            "uv": shutil.which("uv"),
            "blender": shutil.which("blender"),
            "nvidia_smi": shutil.which("nvidia-smi"),
        },
        "repo_local_python": str(_repo_local_robowits_python(repo)) if _repo_local_robowits_python(repo) is not None else None,
        "imports": {
            "gs_gym": {
                "discoverable": gs_gym_discoverable,
                "importable": gs_gym_ok,
                "error": gs_gym_error,
            },
            "genesis": genesis_probe,
            "gymnasium": gymnasium_probe,
            "lerobot": lerobot_probe,
            "torch": torch_probe,
            "pymeshlab": pymeshlab_probe,
        },
        "dataset": dataset_summary,
        "assets": _asset_preflight(repo),
        "upstream": inspect_robowits_repo(str(repo)),
        "blockers": _robowits_blockers(
            repo=repo,
            dataset=dataset_summary,
            gs_gym_ok=gs_gym_ok,
            gs_gym_error=gs_gym_error,
        ),
    }
    payload.update(
        {
            "live": live,
            "env_created": env_created,
            "repo_exists": repo.exists(),
            "gs_gym_package_discoverable": gs_gym_discoverable,
            "gs_gym_importable": gs_gym_ok,
            "gs_gym_import_error": gs_gym_error,
            "genesis_importable": bool(genesis_probe.get("importable")),
            "gymnasium_importable": bool(gymnasium_probe.get("importable")),
            "lerobot_importable": bool(lerobot_probe.get("importable")),
            "dataset_loaded": bool(dataset_summary.get("loaded")),
            "task_name": task_name,
        }
    )
    return payload


def run_robowits_preflight_probe(
    *,
    repo_path: str | None = None,
    dataset_split: str = "eval_dataset_50",
    task_id: str = "01",
    episode_index: int = 0,
    create_env: bool = False,
) -> JsonDict:
    config = RoboWitsRuntimeConfig(
        repo_path=repo_path or str(DEFAULT_ROBOWITS_REPO),
        dataset_split=dataset_split,
        task_id=task_id,
        episode_index=episode_index,
        live=create_env,
    )
    payload = robowits_preflight_summary(
        repo_path=config.repo_path,
        dataset_split=dataset_split,
        dataset_revision_manifest=config.dataset_revision_manifest,
        task_id=task_id,
        episode_index=episode_index,
        live=create_env,
    )
    payload["env_probe"] = None
    if create_env:
        backend = RoboWitsAgentRuntimeBackend(config)
        try:
            task = backend.reset(task_id)
            obs = backend.observe()
            runtime = backend.runtime_available()
            payload["env_probe"] = {
                "ok": True,
                "task": task.to_dict(),
                "observation": obs.to_dict(),
                "runtime": runtime,
            }
            payload["env_created"] = True
            payload["task_name"] = runtime.get("task_name")
            if isinstance(payload.get("runtime"), dict):
                payload["runtime"]["env_created"] = True
                payload["runtime"]["task_name"] = runtime.get("task_name")
        except Exception as exc:
            payload["env_probe"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "runtime": backend.runtime_available(),
            }
        finally:
            backend.close()
    return payload


def inspect_robowits_harness_private_success(backend: RoboWitsAgentRuntimeBackend) -> JsonDict:
    """Read RoboWits success signals for reports, not as an agent primitive."""

    signals: list[JsonDict] = []

    def add_signal(source: str, value: Any, *, error: str | None = None) -> None:
        record: JsonDict = {
            "source": source,
            "value": _to_builtin(value),
            "available": error is None,
        }
        if error is not None:
            record["error"] = error
        else:
            record["bool_value"] = _boolish_success_value(value)
        signals.append(record)

    last_info = getattr(backend, "_last_info", {}) or {}
    if isinstance(last_info, dict):
        for key in ("is_success", "success"):
            if key in last_info:
                add_signal(f"last_step_info.{key}", last_info[key])

    env = getattr(backend, "_env", None)
    candidates = _robowits_env_candidates(env)
    for index, candidate in enumerate(candidates):
        if hasattr(candidate, "get_extra_infos"):
            try:
                info = candidate.get_extra_infos()
                if isinstance(info, dict):
                    for key in ("is_success", "success"):
                        if key in info:
                            add_signal(f"env[{index}].get_extra_infos.{key}", info[key])
                else:
                    add_signal(f"env[{index}].get_extra_infos", info)
            except Exception as exc:
                add_signal(f"env[{index}].get_extra_infos", None, error=f"{type(exc).__name__}: {exc}")
        if hasattr(candidate, "_check_success"):
            try:
                add_signal(f"env[{index}]._check_success", candidate._check_success())
            except Exception as exc:
                add_signal(f"env[{index}]._check_success", None, error=f"{type(exc).__name__}: {exc}")

    available_signals = [signal for signal in signals if signal.get("available")]
    success_values = [signal.get("bool_value") for signal in available_signals if signal.get("bool_value") is not None]
    return {
        "available": bool(success_values),
        "is_success": bool(any(success_values)) if success_values else None,
        "signals": signals,
        "success_signal_count": len(success_values),
        "agent_primitive_exposed": False,
        "harness_only": True,
        "boundary": "RoboWits success/reward/checker remains private and is read only after agent action submission.",
    }


def summarize_scene(scene: JsonDict) -> JsonDict:
    entities = scene_candidates(scene)
    materials = sorted({str(item.get("material")) for item in entities if item.get("material") is not None})
    return {
        "entity_count": len(entities),
        "entity_names": [item["name"] for item in entities],
        "materials": materials,
        "workspace_bounds": workspace_bounds(entities),
    }


def scene_candidates(scene: JsonDict, query: str | None = None, context: JsonDict | None = None) -> list[JsonDict]:
    terms = _query_terms(query, context or {})
    candidates: list[JsonDict] = []
    for name, payload in sorted(scene.items()):
        if not isinstance(payload, dict):
            continue
        text = f"{name} {payload.get('material', '')}".lower()
        score = sum(1 for term in terms if term in text)
        if terms and score == 0:
            continue
        candidates.append(
            {
                "name": str(name),
                "material": payload.get("material"),
                "pos": _to_builtin(payload.get("pos")),
                "euler": _to_builtin(payload.get("euler")),
                "bounds": _to_builtin(payload.get("bounds")),
                "score": float(score if terms else 1),
                "source": payload.get("source", "robowits_eval_json"),
            }
        )
    return sorted(candidates, key=lambda item: (-float(item["score"]), item["name"]))


def robowits_scene_evidence(scene: JsonDict, *, prompt: str | None = None, query: str | None = None, context: JsonDict | None = None) -> JsonDict:
    ctx = context or {}
    all_entities = geometry_candidates(scene)
    selected_entities = geometry_candidates(scene, query=query, context=ctx)
    if query or ctx:
        candidates = selected_entities
    else:
        candidates = all_entities
    return {
        "source": "robowits_eval_json_scene_geometry",
        "prompt": prompt,
        "query": query,
        "context": ctx,
        "scene_evidence": {
            "entity_count": len(all_entities),
            "entity_names": [item["name"] for item in all_entities],
            "materials": sorted({str(item.get("material")) for item in all_entities if item.get("material") is not None}),
            "agent_visible_fields": ["name", "material", "pos", "euler", "bounds", "center", "dimensions"],
            "live_rgb_available": False,
            "genesis_live_required_for_camera_frames": True,
        },
        "geometry_candidates": candidates,
        "workspace_bounds": workspace_bounds(all_entities),
        "pairwise_distances": pairwise_entity_distances(candidates[:6]),
    }


def geometry_candidates(scene: JsonDict, query: str | None = None, context: JsonDict | None = None) -> list[JsonDict]:
    terms = _query_terms(query, context or {})
    candidates: list[JsonDict] = []
    for item in scene_candidates(scene, query=None):
        text = f"{item.get('name', '')} {item.get('material', '')}".lower()
        score = sum(1 for term in terms if term in text)
        if terms and score == 0:
            continue
        bounds = item.get("bounds")
        center = _entity_center(item)
        dimensions = _entity_dimensions(bounds)
        candidates.append(
            {
                **item,
                "score": float(score if terms else 1),
                "center": center,
                "dimensions": dimensions,
                "bbox_3d": bounds,
                "spatial_affordances": _spatial_affordances(item, dimensions),
            }
        )
    return sorted(candidates, key=lambda value: (-float(value["score"]), value["name"]))


def _select_robowits_entity(candidates: list[JsonDict], *, entity_name: str | None = None, query: str | None = None) -> JsonDict | None:
    if not candidates:
        return None
    if entity_name:
        entity_text = entity_name.lower()
        for item in candidates:
            if str(item.get("name", "")).lower() == entity_text:
                return item
        for item in candidates:
            if entity_text in str(item.get("name", "")).lower():
                return item
    if query:
        query_text = query.lower()
        for item in candidates:
            if str(item.get("name", "")).lower() in query_text:
                return item
    return candidates[0]


def _robowits_pose_world(entity: JsonDict | None) -> list[float] | None:
    if entity is None:
        return None
    center = entity.get("center") or entity.get("pos")
    if not isinstance(center, list) or len(center) < 3:
        return None
    euler = entity.get("euler")
    pose = [float(center[0]), float(center[1]), float(center[2])]
    if isinstance(euler, list) and len(euler) >= 3:
        pose.extend([float(euler[0]), float(euler[1]), float(euler[2])])
    return pose


def workspace_bounds(entities: list[JsonDict]) -> JsonDict:
    mins: list[list[float]] = []
    maxs: list[list[float]] = []
    for item in entities:
        bounds = item.get("bounds")
        if _is_bounds(bounds):
            mins.append([float(v) for v in bounds[0]])
            maxs.append([float(v) for v in bounds[1]])
    if not mins or not maxs:
        return {"available": False}
    return {
        "available": True,
        "min": [min(values[index] for values in mins) for index in range(3)],
        "max": [max(values[index] for values in maxs) for index in range(3)],
    }


def pairwise_entity_distances(entities: list[JsonDict]) -> list[JsonDict]:
    distances: list[JsonDict] = []
    for left_index, left in enumerate(entities):
        left_center = left.get("center")
        if left_center is None:
            continue
        for right in entities[left_index + 1 :]:
            right_center = right.get("center")
            if right_center is None:
                continue
            distances.append(
                {
                    "a": left["name"],
                    "b": right["name"],
                    "center_distance": sum((float(left_center[i]) - float(right_center[i])) ** 2 for i in range(3)) ** 0.5,
                }
            )
    return distances


def _entity_center(item: JsonDict) -> list[float] | None:
    bounds = item.get("bounds")
    if _is_bounds(bounds):
        return [(float(bounds[0][axis]) + float(bounds[1][axis])) / 2.0 for axis in range(3)]
    pos = item.get("pos")
    if isinstance(pos, list) and len(pos) >= 3:
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    return None


def _entity_dimensions(bounds: Any) -> list[float] | None:
    if not _is_bounds(bounds):
        return None
    return [abs(float(bounds[1][axis]) - float(bounds[0][axis])) for axis in range(3)]


def _is_bounds(bounds: Any) -> bool:
    return (
        isinstance(bounds, list)
        and len(bounds) == 2
        and all(isinstance(row, list) and len(row) >= 3 for row in bounds)
    )


def _spatial_affordances(item: JsonDict, dimensions: list[float] | None) -> list[str]:
    material = str(item.get("material") or "").lower()
    name = str(item.get("name") or "").lower()
    affordances = []
    if dimensions:
        longest = max(dimensions)
        shortest = min(dimensions)
        if longest >= 4 * max(shortest, 1e-6):
            affordances.append("long_straight_edge")
        if dimensions[2] < 0.08:
            affordances.append("low_profile_contact_tool")
    if any(token in name for token in ("ruler", "stick", "rod")):
        affordances.append("alignment_or_pushing_tool")
    if any(token in material for token in ("rubber", "wood", "plastic", "metal")):
        affordances.append(f"material_{material}")
    return sorted(set(affordances))


def summarize_observation(obs: Any) -> JsonDict:
    if obs is None:
        return {"type": "NoneType"}
    if isinstance(obs, dict):
        return {str(key): summarize_observation(value) for key, value in obs.items()}
    shape = getattr(obs, "shape", None)
    dtype = getattr(obs, "dtype", None)
    if shape is not None:
        return {"type": type(obs).__name__, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(obs, (list, tuple)):
        return {"type": type(obs).__name__, "length": len(obs)}
    return {"type": type(obs).__name__, "value": _to_builtin(obs)}


def summarize_action(action: Any, *, schema: JsonDict) -> JsonDict:
    import numpy as np

    array = np.asarray(action, dtype=np.float32)
    flat = array.reshape(-1)
    summary: JsonDict = {
        "validated": True,
        "schema": schema,
        "shape": [int(dim) for dim in array.shape],
        "dtype": str(array.dtype),
        "nonzero_count": int(np.count_nonzero(flat)),
        "l2_norm": round(float(np.linalg.norm(flat)), 6),
        "min": round(float(np.min(flat)), 6) if flat.size else None,
        "max": round(float(np.max(flat)), 6) if flat.size else None,
        "values": _to_builtin(np.round(array.astype(float), 6).tolist()) if flat.size <= 64 else None,
    }
    if array.ndim == 2 and array.shape[0] >= 1:
        summary["components"] = _action_components(array[0], schema=schema)
    return summary


def _coerce_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _clamp_float(value: float, low: float, high: float) -> float:
    return min(max(float(value), float(low)), float(high))


def _extract_agent_pos(obs: Any, *, action_dim: int) -> tuple[list[float] | None, str | None]:
    if not isinstance(obs, dict):
        return None, None
    keys = ("agent_pos_ee", "agent_pos") if action_dim == 14 else ("agent_pos", "agent_pos_ee")
    for key in keys:
        value = obs.get(key)
        if value is None:
            continue
        try:
            import numpy as np

            array = np.asarray(value, dtype=float)
        except Exception:
            continue
        if array.ndim == 2 and array.shape[0] >= 1:
            array = array[0]
        if array.ndim == 1 and array.shape[0] >= action_dim:
            return [round(float(value), 6) for value in array[:action_dim]], key
    return None, None


def robowits_public_ee_tracking_evidence(
    obs: Any,
    *,
    target_action: Any | None,
    schema: JsonDict,
    robot_base_position: list[float] | None = None,
) -> JsonDict:
    """Compare public EE observation against a submitted EE_ABS target."""

    action_dim = int(schema.get("action_dimension") or 14)
    ee_state, source_key = _extract_agent_pos(obs, action_dim=action_dim)
    evidence: JsonDict = {
        "available": ee_state is not None,
        "source": source_key,
    }
    if ee_state is None:
        evidence["error"] = "public_ee_observation_unavailable"
        return evidence
    base_position = list(robot_base_position or [0.0, 0.0, 0.0])
    observation_state = ee_state
    tracking_state = ee_state
    if action_dim == 14 and len(base_position) >= 3:
        evidence["position_frame"] = "world"
        evidence["robot_base_position"] = [round(float(value), 6) for value in base_position[:3]]
        world_state = list(ee_state)
        for offset in (0, 3):
            world_state[offset : offset + 3] = [
                round(float(world_state[offset + idx]) + float(base_position[idx]), 6) for idx in range(3)
            ]
        evidence["ee_state_robot_base_relative"] = ee_state
        evidence["ee_state_world"] = world_state
        observation_state = world_state
    evidence["ee_state"] = observation_state
    evidence["components"] = _action_components(observation_state, schema=schema)
    if target_action is None:
        return evidence
    try:
        import numpy as np

        target = np.asarray(target_action, dtype=float)
        if target.ndim == 2 and target.shape[0] >= 1:
            target = target[0]
        if target.ndim != 1 or target.shape[0] < action_dim:
            evidence["target_error"] = "target_action_shape_mismatch"
            return evidence
        target_values = [round(float(value), 6) for value in target[:action_dim]]
    except Exception as exc:
        evidence["target_error"] = f"{type(exc).__name__}: {exc}"
        return evidence
    evidence["target_action"] = target_values
    evidence["target_position_frame"] = "robot_base_relative"
    evidence["tracking_position_frame"] = "robot_base_relative"
    evidence["target_components"] = _action_components(target_values, schema=schema)
    tracking_components = _action_components(tracking_state, schema=schema)
    evidence["tracking_components"] = tracking_components
    component_errors: JsonDict = {}
    for name in ("right_pos", "left_pos", "right_gripper", "left_gripper"):
        actual_values = (tracking_components.get(name) or {}).get("values")
        target_values_for_name = (evidence["target_components"].get(name) or {}).get("values")
        if not (isinstance(actual_values, list) and isinstance(target_values_for_name, list)):
            continue
        diffs = [round(float(actual_values[idx]) - float(target_values_for_name[idx]), 6) for idx in range(min(len(actual_values), len(target_values_for_name)))]
        component_errors[name] = {
            "actual_minus_target": diffs,
            "l2": round(sum(float(value) ** 2 for value in diffs) ** 0.5, 6),
        }
    evidence["component_errors"] = component_errors
    position_errors = [
        float(component_errors.get("right_pos", {}).get("l2", 0.0)),
        float(component_errors.get("left_pos", {}).get("l2", 0.0)),
    ]
    evidence["max_position_l2_error"] = round(max(position_errors) if position_errors else 0.0, 6)
    evidence["tracking_ok_within_2cm"] = bool(evidence["max_position_l2_error"] <= 0.02)
    return evidence


def _extract_ee_abs_action_base(obs: Any, *, action_dim: int) -> tuple[list[float] | None, JsonDict]:
    """Extract a RoboWits EE_ABS action base from public EE observations.

    Current upstream RoboWits `agent_pos_ee` is already laid out as
    [R_pos, L_pos, R_axis_angle, L_axis_angle, R_grip, L_grip], matching
    the 14D EE_ABS action layout. Prefer `agent_pos_ee` over `agent_pos`
    so JOINT observation mode cannot be mistaken for an EE action base.
    """

    raw, source_key = _extract_agent_pos(obs, action_dim=action_dim)
    metadata: JsonDict = {
        "source": f"live_observation_{source_key}" if source_key else "live_observation_agent_pos",
        "available": raw is not None,
        "observation_frame": "robot_base_relative_agent_pos",
        "observation_key": source_key,
        "observation_layout": "unknown",
        "action_layout": "EE_ABS",
        "converted": False,
    }
    if raw is None:
        metadata["source"] = "zero_action_base_fallback"
        metadata["observation_frame"] = "zero_fallback_no_live_agent_pos"
        return None, metadata
    if action_dim == 14:
        metadata.update(
            {
                "observation_layout": "EE_state:[R_pos,L_pos,R_axis_angle,L_axis_angle,R_grip,L_grip]",
                "action_layout": "EE_ABS_action:[R_pos,L_pos,R_axis_angle,L_axis_angle,R_grip,L_grip]",
                "converted": False,
                "layout_source": "RoboWits/gs_gym/envs/robowits/robowits.py:state_ee",
            }
        )
        return [round(float(value), 6) for value in raw[:14]], metadata
    return raw, metadata


def _robowits_step_action(action: Any, *, action_dim: int) -> Any:
    if isinstance(action, dict):
        if "action" in action:
            action = action["action"]
        elif "values" in action:
            action = action["values"]
        else:
            raise ValueError("Structured RoboWits action dictionaries must contain an 'action' or 'values' numeric vector.")
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - live runtime requires numpy.
        raise RuntimeError("numpy is required to submit RoboWits actions.") from exc
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("RoboWits action must be a 1D action vector or batched 2D action array.")
    if array.ndim == 1:
        if array.shape[0] != action_dim:
            raise ValueError(f"RoboWits action length {array.shape[0]} does not match expected dimension {action_dim}.")
        array = array.reshape(1, action_dim)
    if array.ndim != 2 or array.shape[-1] != action_dim:
        raise ValueError(f"RoboWits action shape {list(array.shape)} does not match expected (*, {action_dim}).")
    return array


def _slerp_axis_angle(start: list[float], target: list[float], alpha: float) -> list[float]:
    """Interpolate two axis-angle rotations without crossing the long quaternion arc."""

    import math

    start_quat = _axis_angle_to_quat(start)
    target_quat = _axis_angle_to_quat(target)
    dot = sum(float(left) * float(right) for left, right in zip(start_quat, target_quat))
    if dot < 0.0:
        target_quat = [-float(value) for value in target_quat]
        dot = -dot
    dot = _clamp_float(dot, -1.0, 1.0)
    if dot > 0.9995:
        quat = [float(left) + alpha * (float(right) - float(left)) for left, right in zip(start_quat, target_quat)]
        norm = sum(value**2 for value in quat) ** 0.5
        quat = [value / norm for value in quat] if norm > 1e-12 else list(start_quat)
    else:
        theta = math.acos(dot)
        denominator = math.sin(theta)
        start_weight = math.sin((1.0 - alpha) * theta) / denominator
        target_weight = math.sin(alpha * theta) / denominator
        quat = [
            start_weight * float(left) + target_weight * float(right)
            for left, right in zip(start_quat, target_quat)
        ]
    return _quat_to_axis_angle(quat)


def _interpolate_robowits_ee_abs_action(
    start_action: list[float],
    target_action: list[float],
    *,
    arm: str,
    alpha: float,
) -> list[float]:
    """Build one generic smooth EE_ABS setpoint for the caller-selected arm."""

    fraction = _clamp_float(float(alpha), 0.0, 1.0)
    position_offset = 0 if arm == "right" else 3
    axis_offset = 6 if arm == "right" else 9
    grip_offset = 12 if arm == "right" else 13
    action = list(start_action)
    action[position_offset : position_offset + 3] = [
        float(start_action[position_offset + index])
        + fraction * (float(target_action[position_offset + index]) - float(start_action[position_offset + index]))
        for index in range(3)
    ]
    action[axis_offset : axis_offset + 3] = _slerp_axis_angle(
        [float(value) for value in start_action[axis_offset : axis_offset + 3]],
        [float(value) for value in target_action[axis_offset : axis_offset + 3]],
        fraction,
    )
    action[grip_offset] = float(start_action[grip_offset]) + fraction * (
        float(target_action[grip_offset]) - float(start_action[grip_offset])
    )
    return [round(float(value), 6) for value in action[:14]]


def _robowits_action_layout(control_mode: str) -> list[JsonDict]:
    if control_mode in ("EE_ABS", "EE_DELTA") or control_mode.startswith("EE"):
        return [
            {"name": "right_pos", "indices": [0, 1, 2], "units": "meters" if control_mode == "EE_ABS" else "meters_delta"},
            {"name": "left_pos", "indices": [3, 4, 5], "units": "meters" if control_mode == "EE_ABS" else "meters_delta"},
            {"name": "right_axis_angle", "indices": [6, 7, 8], "units": "radians"},
            {"name": "left_axis_angle", "indices": [9, 10, 11], "units": "radians"},
            {"name": "right_gripper", "indices": [12], "units": "width_or_delta"},
            {"name": "left_gripper", "indices": [13], "units": "width_or_delta"},
        ]
    return [
        {"name": "right_arm_joints", "indices": list(range(0, 7)), "units": "radians"},
        {"name": "right_gripper", "indices": [7], "units": "width_or_delta"},
        {"name": "left_arm_joints", "indices": list(range(8, 15)), "units": "radians"},
        {"name": "left_gripper", "indices": [15], "units": "width_or_delta"},
    ]


def _action_components(row: Any, *, schema: JsonDict) -> JsonDict:
    components: JsonDict = {}
    for item in schema.get("layout") or []:
        name = str(item.get("name"))
        indices = [int(index) for index in item.get("indices", [])]
        values = [round(float(row[index]), 6) for index in indices if index < len(row)]
        components[name] = {"indices": indices, "values": values, "units": item.get("units")}
    return components


def robowits_live_visual_evidence(obs: Any) -> JsonDict:
    cameras = _robowits_camera_summaries(obs)
    modalities_by_camera = {name: sorted(payload) for name, payload in cameras.items()}
    return {
        "visual_ready": bool(cameras),
        "camera_names": sorted(cameras),
        "modalities_by_camera": modalities_by_camera,
        "has_rgb": any("rgb" in modalities for modalities in modalities_by_camera.values()),
        "has_depth": any("depth" in modalities for modalities in modalities_by_camera.values()),
        "has_segmentation": any("segmentation" in modalities for modalities in modalities_by_camera.values()),
        "cameras": cameras,
        "source": "live_genesis_observation",
    }


def robowits_camera_pixel_evidence(
    obs: Any,
    *,
    camera_name: str | None = None,
    bbox: Any = None,
    query: str | None = None,
    context: JsonDict | None = None,
) -> JsonDict:
    rgb_arrays = _robowits_rgb_arrays(obs)
    if not rgb_arrays:
        return {
            "ok": False,
            "error": "live_rgb_pixels_unavailable",
            "available_cameras": [],
            "regions": [],
            "evidence": {"source": "live_genesis_observation", "pixel_inspection": False},
        }
    selected_name = _select_camera_name(sorted(rgb_arrays), camera_name=camera_name, query=query, context=context or {})
    if selected_name not in rgb_arrays:
        return {
            "ok": False,
            "error": "camera_not_found",
            "available_cameras": sorted(rgb_arrays),
            "selected_camera": selected_name,
            "regions": [],
            "evidence": {"source": "live_genesis_observation", "pixel_inspection": False},
        }
    try:
        image = _rgb_array_to_hwc_uint8(rgb_arrays[selected_name])
        height, width = int(image.shape[0]), int(image.shape[1])
        requested_bbox = _normalize_pixel_bbox(bbox, width=width, height=height)
    except Exception as exc:
        return {
            "ok": False,
            "error": f"invalid_live_rgb_pixels:{type(exc).__name__}:{exc}",
            "available_cameras": sorted(rgb_arrays),
            "selected_camera": selected_name,
            "regions": [],
            "evidence": {"source": "live_genesis_observation", "pixel_inspection": False},
        }

    regions = [_summarize_rgb_region(image, name="full_image", bbox=(0, 0, width, height))]
    center_bbox = _center_bbox(width=width, height=height, fraction=0.35)
    if center_bbox not in {(0, 0, width, height), requested_bbox}:
        regions.append(_summarize_rgb_region(image, name="center_region", bbox=center_bbox))
    if requested_bbox != (0, 0, width, height):
        regions.append(_summarize_rgb_region(image, name="requested_bbox", bbox=requested_bbox))
    return {
        "ok": True,
        "available_cameras": sorted(rgb_arrays),
        "selected_camera": selected_name,
        "camera": {
            "name": selected_name,
            "width": width,
            "height": height,
            "channels": 3,
            "dtype": str(image.dtype),
        },
        "bbox": list(requested_bbox),
        "regions": regions,
        "evidence": {
            "source": "live_genesis_observation",
            "pixel_inspection": True,
            "query": query,
            "context": context or {},
            "camera_selection": "explicit" if camera_name else "query_or_first_camera",
            "detector_model_used": None,
            "segmentation_model_used": None,
        },
    }


def robowits_live_object_pose_evidence(
    env: Any,
    *,
    names: list[str] | None = None,
    query: str | None = None,
    context: JsonDict | None = None,
    reference_scene: JsonDict | None = None,
) -> JsonDict:
    ctx = context or {}
    requested_names = [str(name) for name in names or [] if str(name)]
    payload: JsonDict = {
        "query": query,
        "context": ctx,
        "requested_names": requested_names or None,
        "available": False,
        "object_poses": {},
        "diagnostics": {},
        "source": None,
    }
    if env is None:
        payload["error"] = "live_env_not_created"
        return payload
    terms = _query_terms(query, ctx)
    for index, candidate in enumerate(_robowits_env_candidates(env)):
        collector = getattr(candidate, "collect_objs_info", None)
        if not callable(collector):
            continue
        try:
            raw = collector(names=requested_names or None)
        except TypeError:
            try:
                raw = collector(requested_names or None)
            except Exception as exc:
                payload["error"] = f"collect_objs_info_failed:{type(exc).__name__}: {exc}"
                continue
        except Exception as exc:
            payload["error"] = f"collect_objs_info_failed:{type(exc).__name__}: {exc}"
            continue
        if not isinstance(raw, dict):
            continue
        records: JsonDict = {}
        for name, info in raw.items():
            if requested_names and str(name) not in requested_names:
                continue
            if terms and not requested_names:
                haystack = f"{name} {json.dumps(_to_builtin(info), ensure_ascii=False)}".lower()
                if not any(term in haystack for term in terms):
                    continue
            records[str(name)] = _robowits_object_pose_record(str(name), info)
        if records:
            payload.update(
                {
                    "available": True,
                    "object_poses": records,
                    "diagnostics": _robowits_pose_diagnostics(records, reference_scene=reference_scene or {}),
                    "source": f"env[{index}].collect_objs_info",
                    "error": None,
                }
            )
            return payload
    payload["error"] = payload.get("error") or "live_object_pose_unavailable"
    return payload


def _robowits_object_pose_record(name: str, info: Any) -> JsonDict:
    info_dict = info if isinstance(info, dict) else {}
    pos = _numeric_vector(info_dict.get("pos"), limit=3)
    vel = _numeric_vector(info_dict.get("vel"), limit=3)
    bounds = _numeric_matrix(info_dict.get("bounds"), rows=2, cols=3)
    center = None
    if bounds is not None:
        center = [round((float(bounds[0][idx]) + float(bounds[1][idx])) / 2.0, 6) for idx in range(3)]
    elif pos is not None:
        center = pos
    return {
        "name": name,
        "material": _to_builtin(info_dict.get("material")),
        "pos": pos,
        "vel": vel,
        "vel_norm": _vector_norm(vel),
        "euler": _numeric_vector(info_dict.get("euler"), limit=3),
        "bounds": bounds,
        "center": center,
        "convex_hull_2d_available": info_dict.get("convex_hull_2d") is not None,
    }


def _robowits_env_robot_base_position(env: Any) -> list[float]:
    for candidate in _robowits_env_candidates(env):
        robot = getattr(candidate, "robot", None) or getattr(candidate, "_robot", None)
        args = getattr(robot, "args", None)
        position = getattr(args, "position", None)
        values = _numeric_vector(position, limit=3)
        if values is not None:
            return values
    return [0.0, 0.0, 0.0]


def _robowits_live_records(env: Any, names: list[str] | None = None) -> tuple[JsonDict, str | None]:
    for candidate in _robowits_env_candidates(env):
        collector = getattr(candidate, "collect_objs_info", None)
        if not callable(collector):
            continue
        try:
            raw = collector(names=names or None)
        except TypeError:
            raw = collector(names or None)
        except Exception:
            continue
        if isinstance(raw, dict):
            return {str(name): _robowits_object_pose_record(str(name), info) for name, info in raw.items()}, None
    return {}, "live_object_pose_unavailable"


def _robowits_world_ee_state(env: Any, obs: Any) -> tuple[list[float] | None, JsonDict]:
    state, metadata = _extract_ee_abs_action_base(obs, action_dim=14)
    if state is None:
        return None, metadata
    base = _robowits_env_robot_base_position(env)
    world = list(state)
    for offset in (0, 3):
        world[offset : offset + 3] = [round(float(world[offset + idx]) + float(base[idx]), 6) for idx in range(3)]
    metadata.update(
        {
            "action_frame": "world",
            "robot_base_position": base,
            "converted": bool(any(abs(float(value)) > 1e-9 for value in base)),
            "conversion": "world_position=robot_base_relative_position+robot_base_position",
        }
    )
    return world, metadata


def robowits_grasp_state_evidence(
    env: Any,
    obs: Any,
    *,
    object_name: str,
    arm: str = "auto",
    context: JsonDict | None = None,
) -> JsonDict:
    records, error = _robowits_live_records(env)
    record = records.get(object_name)
    ee_state, metadata = _robowits_world_ee_state(env, obs)
    if record is None or ee_state is None:
        return {
            "available": False,
            "object_name": object_name,
            "context": context or {},
            "error": error or "public_object_or_ee_state_unavailable",
        }
    bounds = record.get("bounds")
    center = record.get("center")
    if not (_is_bounds(bounds) and isinstance(center, list) and len(center) >= 3):
        return {"available": False, "object_name": object_name, "error": "object_bounds_unavailable"}
    candidates = {
        "right": (ee_state[0:3], float(ee_state[12])),
        "left": (ee_state[3:6], float(ee_state[13])),
    }
    requested_arm = str(arm or "auto").lower()
    if requested_arm not in candidates:
        requested_arm = min(candidates, key=lambda key: sum((float(candidates[key][0][idx]) - float(center[idx])) ** 2 for idx in range(3)))
    ee_position, gripper_width = candidates[requested_arm]
    delta = [float(ee_position[idx]) - float(center[idx]) for idx in range(3)]
    distance = sum(value**2 for value in delta) ** 0.5
    dimensions = [float(bounds[1][idx]) - float(bounds[0][idx]) for idx in range(3)]
    inside_expanded = all(float(bounds[0][idx]) - 0.035 <= float(ee_position[idx]) <= float(bounds[1][idx]) + 0.035 for idx in range(3))
    closed_enough = gripper_width <= max(dimensions[0], dimensions[1]) + 0.015
    return {
        "available": True,
        "object_name": object_name,
        "object": record,
        "arm": requested_arm,
        "ee_position_world": [round(float(value), 6) for value in ee_position],
        "ee_distance": round(distance, 6),
        "gripper_width": round(gripper_width, 6),
        "ee_inside_expanded_bounds": inside_expanded,
        "gripper_closed_around_object_scale": closed_enough,
        "grasp_likely": bool(inside_expanded and closed_enough),
        "grasp_confirmed": None,
        "method": "proximity_and_width_proxy_only",
        "confirmation_required": "lift_and_object_ee_motion_coupling",
        "object_velocity_norm": record.get("vel_norm"),
        "action_base": metadata,
        "context": context or {},
    }


def _aabb_contact(left: Any, right: Any, *, tolerance: float) -> JsonDict:
    if not (_is_bounds(left) and _is_bounds(right)):
        return {"available": False, "contact": False}
    axis_gaps: list[float] = []
    axis_overlaps: list[float] = []
    for axis in range(3):
        gap = max(float(right[0][axis]) - float(left[1][axis]), float(left[0][axis]) - float(right[1][axis]), 0.0)
        overlap = min(float(left[1][axis]), float(right[1][axis])) - max(float(left[0][axis]), float(right[0][axis]))
        axis_gaps.append(round(gap, 6))
        axis_overlaps.append(round(overlap, 6))
    return {
        "available": True,
        "contact": bool(all(gap <= tolerance for gap in axis_gaps)),
        "axis_gaps": axis_gaps,
        "axis_overlaps": axis_overlaps,
        "tolerance": tolerance,
    }


def robowits_contact_stability_evidence(
    env: Any,
    *,
    names: list[str],
    velocity_threshold: float | None = None,
    contact_tolerance: float | None = None,
    context: JsonDict | None = None,
) -> JsonDict:
    requested = [str(name) for name in names if str(name)]
    records, error = _robowits_live_records(env, requested)
    if not records:
        return {"available": False, "objects": {}, "pairwise_contacts": [], "error": error or "live_object_pose_unavailable"}
    velocity_limit = float(velocity_threshold if velocity_threshold is not None else 0.01)
    tolerance = float(contact_tolerance if contact_tolerance is not None else 0.01)
    object_state = {
        name: {**record, "stable": bool((record.get("vel_norm") or 0.0) <= velocity_limit)} for name, record in records.items()
    }
    pairs: list[JsonDict] = []
    ordered = [name for name in requested if name in object_state]
    for index, left_name in enumerate(ordered):
        for right_name in ordered[index + 1 :]:
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    **_aabb_contact(object_state[left_name].get("bounds"), object_state[right_name].get("bounds"), tolerance=tolerance),
                }
            )
    return {
        "available": True,
        "objects": object_state,
        "pairwise_contacts": pairs,
        "all_stable": bool(object_state) and all(item["stable"] for item in object_state.values()),
        "velocity_threshold": velocity_limit,
        "contact_tolerance": tolerance,
        "context": context or {},
    }


def _axis_angle_to_quat(values: list[float]) -> list[float]:
    import math

    angle = sum(float(value) ** 2 for value in values[:3]) ** 0.5
    if angle <= 1e-9:
        return [1.0, 0.0, 0.0, 0.0]
    scale = math.sin(angle / 2.0) / angle
    return [math.cos(angle / 2.0), *(float(value) * scale for value in values[:3])]


def _quat_multiply(left: list[float], right: list[float]) -> list[float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return [
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ]


def _quat_to_axis_angle(values: list[float]) -> list[float]:
    import math

    norm = sum(float(value) ** 2 for value in values[:4]) ** 0.5
    quat = [float(value) / max(norm, 1e-12) for value in values[:4]]
    if quat[0] < 0.0:
        quat = [-value for value in quat]
    vector_norm = sum(value**2 for value in quat[1:]) ** 0.5
    if vector_norm <= 1e-9:
        return [0.0, 0.0, 0.0]
    angle = 2.0 * math.atan2(vector_norm, max(quat[0], 1e-12))
    return [round(value / vector_norm * angle, 6) for value in quat[1:]]


def _rotate_axis_angle_about_world_z(values: list[float], yaw_delta: float) -> list[float]:
    import math

    yaw = [math.cos(yaw_delta / 2.0), 0.0, 0.0, math.sin(yaw_delta / 2.0)]
    return _quat_to_axis_angle(_quat_multiply(yaw, _axis_angle_to_quat(values)))


def _point_aabb_distance_xy(point: list[float], bounds: Any) -> float:
    if not _is_bounds(bounds):
        return float("inf")
    dx = max(float(bounds[0][0]) - float(point[0]), float(point[0]) - float(bounds[1][0]), 0.0)
    dy = max(float(bounds[0][1]) - float(point[1]), float(point[1]) - float(bounds[1][1]), 0.0)
    return (dx**2 + dy**2) ** 0.5


def robowits_wrist_orientation_candidates(
    records: JsonDict,
    *,
    object_name: str | None,
    action_base: list[float],
    arm: str,
    position: list[float],
    reference_direction: list[float] | None = None,
    yaw_offsets: list[float] | None = None,
    ignored_names: list[str] | None = None,
    collision_margin: float = 0.004,
) -> JsonDict:
    """Generate task-agnostic wrist-yaw candidates from public geometry.

    The query intentionally returns no selected candidate.  Choosing the wrist
    objective remains the coding agent's responsibility.
    """

    import math

    axis_offset = 6 if arm == "right" else 9
    initial = [float(value) for value in action_base[axis_offset : axis_offset + 3]]
    record = records.get(object_name) or {} if object_name else {}
    bounds = record.get("bounds")
    dimensions = (
        [float(bounds[1][index]) - float(bounds[0][index]) for index in range(3)]
        if _is_bounds(bounds)
        else [0.05, 0.05, 0.05]
    )
    direction = _numeric_vector(reference_direction, limit=2)
    reference_angle = math.atan2(float(direction[1]), float(direction[0])) if direction and any(abs(value) > 1e-9 for value in direction) else 0.0
    offsets = yaw_offsets or [0.0, math.pi / 4.0, -math.pi / 4.0, math.pi / 2.0, -math.pi / 2.0, math.pi]
    half_span = max(min(dimensions[0], dimensions[1]) / 2.0 + 0.012, 0.035)
    finger_vertical_half_span = min(max(dimensions[2] * 0.2, 0.008), 0.018)
    ignored = set(str(name) for name in ignored_names or [])
    if object_name:
        ignored.add(object_name)
    obstacles = {
        name: item
        for name, item in records.items()
        if name not in ignored
        and _is_bounds(item.get("bounds"))
        and float(item["bounds"][1][2]) >= float(position[2]) - finger_vertical_half_span
        and float(item["bounds"][0][2]) <= float(position[2]) + finger_vertical_half_span
    }
    candidates: list[JsonDict] = []
    for index, raw_offset in enumerate(offsets):
        yaw_delta = float(raw_offset)
        sample_angle = reference_angle + yaw_delta
        finger_points = [
            [float(position[0]) + sign * math.cos(sample_angle) * half_span, float(position[1]) + sign * math.sin(sample_angle) * half_span]
            for sign in (-1.0, 1.0)
        ]
        clearances = [
            _point_aabb_distance_xy(point, obstacle.get("bounds"))
            for obstacle in obstacles.values()
            for point in finger_points
        ]
        clearance = min(clearances) if clearances else 0.25
        candidates.append(
            {
                "name": f"yaw_candidate_{index}",
                "yaw_delta_radians": round(yaw_delta, 6),
                "axis_angle": _rotate_axis_angle_about_world_z(initial, yaw_delta),
                "minimum_public_aabb_clearance_xy": round(clearance, 6),
                "collision_rejected": bool(clearance < float(collision_margin)),
                "finger_sample_points_xy": [[round(value, 6) for value in point] for point in finger_points],
            }
        )
    return {
        "available": True,
        "arm": arm,
        "position": [round(float(value), 6) for value in position[:3]],
        "reference_direction": direction,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "collision_free_candidates": [candidate["name"] for candidate in candidates if not candidate["collision_rejected"]],
        "finger_vertical_half_span": round(finger_vertical_half_span, 6),
        "selected": None,
        "selection_deferred_to_coding_agent": True,
        "uses_public_geometry_only": True,
    }


def robowits_wrist_orientation_query(
    env: Any,
    obs: Any,
    *,
    object_name: str | None,
    position: list[float] | None,
    arm: str,
    reference_direction: list[float] | None,
    yaw_offsets: list[float] | None,
    ignored_names: list[str] | None,
    context: JsonDict,
) -> JsonDict:
    records, error = _robowits_live_records(env)
    action_base, _ = _extract_ee_abs_action_base(obs, action_dim=14)
    if action_base is None:
        return {"available": False, "error": "public_ee_state_unavailable"}
    query_position = _numeric_vector(position, limit=3)
    if query_position is None and object_name:
        query_position = _numeric_vector((records.get(object_name) or {}).get("center"), limit=3)
    if query_position is None:
        return {"available": False, "error": error or "query_position_unavailable"}
    base = _robowits_env_robot_base_position(env)
    world_positions = {
        "right": [float(action_base[index]) + float(base[index]) for index in range(3)],
        "left": [float(action_base[index + 3]) + float(base[index]) for index in range(3)],
    }
    selected_arm = str(arm or "auto").lower()
    if selected_arm not in world_positions:
        selected_arm = min(
            world_positions,
            key=lambda name: sum((float(world_positions[name][index]) - float(query_position[index])) ** 2 for index in range(3)),
        )
    result = robowits_wrist_orientation_candidates(
        records,
        object_name=object_name,
        action_base=action_base,
        arm=selected_arm,
        position=query_position,
        reference_direction=reference_direction,
        yaw_offsets=yaw_offsets,
        ignored_names=ignored_names,
        collision_margin=_clamp_float(float(context.get("collision_margin", 0.004)), 0.0, 0.05),
    )
    return {**result, "object_name": object_name, "context": context}


def _point_aabb_clearance_3d(point: list[float], bounds: Any) -> float:
    if not _is_bounds(bounds):
        return float("inf")
    deltas = [
        max(float(bounds[0][index]) - float(point[index]), float(point[index]) - float(bounds[1][index]), 0.0)
        for index in range(3)
    ]
    return sum(value**2 for value in deltas) ** 0.5


def _robowits_path_clearance_query(
    records: JsonDict,
    *,
    start: list[float],
    target: list[float],
    ignored_names: list[str] | None,
    clearance_radius: float,
    samples: int,
) -> JsonDict:
    ignored = set(str(name) for name in ignored_names or [])
    count = max(2, min(int(samples), 101))
    obstacles = {name: record for name, record in records.items() if name not in ignored and _is_bounds(record.get("bounds"))}
    sampled: list[JsonDict] = []
    minimum = float("inf")
    colliding_names: set[str] = set()
    for index in range(count):
        alpha = index / (count - 1)
        point = [float(start[axis]) + (float(target[axis]) - float(start[axis])) * alpha for axis in range(3)]
        per_object = {name: _point_aabb_clearance_3d(point, record.get("bounds")) for name, record in obstacles.items()}
        local_minimum = min(per_object.values()) if per_object else float("inf")
        minimum = min(minimum, local_minimum)
        collisions = sorted(name for name, value in per_object.items() if value <= clearance_radius)
        colliding_names.update(collisions)
        sampled.append(
            {
                "index": index,
                "position": [round(value, 6) for value in point],
                "minimum_aabb_clearance": None if local_minimum == float("inf") else round(local_minimum, 6),
                "collisions": collisions,
            }
        )
    return {
        "clear": not colliding_names,
        "clearance_radius": round(clearance_radius, 6),
        "minimum_public_aabb_clearance": None if minimum == float("inf") else round(minimum, 6),
        "colliding_names": sorted(colliding_names),
        "samples": sampled,
        "ignored_names": sorted(ignored),
        "geometry_model": "sampled_ee_sphere_against_public_world_aabbs",
    }


def _robowits_ik_query(
    env: Any,
    *,
    action_base: list[float],
    arm: str,
    target_world: list[float],
    axis_angle: list[float],
) -> JsonDict:
    """Run upstream Pink IK as a read-only model query; simulator state is not stepped."""

    robot = None
    for candidate in _robowits_env_candidates(env):
        candidate_robot = getattr(candidate, "robot", None) or getattr(candidate, "_robot", None)
        if candidate_robot is not None and callable(getattr(candidate_robot, "run_bimanual_ik", None)):
            robot = candidate_robot
            break
    if robot is None:
        return {"available": False, "reachable": None, "error": "upstream_ik_query_unavailable"}
    try:
        import numpy as np
        import torch

        base = _robowits_env_robot_base_position(env)
        target_relative = [float(target_world[index]) - float(base[index]) for index in range(3)]
        right_position = list(action_base[0:3])
        left_position = list(action_base[3:6])
        right_axis_angle = list(action_base[6:9])
        left_axis_angle = list(action_base[9:12])
        if arm == "right":
            right_position = target_relative
            right_axis_angle = axis_angle
        else:
            left_position = target_relative
            left_axis_angle = axis_angle

        device = getattr(robot, "device", "cpu")
        right_pos_tensor = torch.tensor([right_position], dtype=torch.float32, device=device)
        left_pos_tensor = torch.tensor([left_position], dtype=torch.float32, device=device)
        right_quat_tensor = torch.tensor([_axis_angle_to_quat(right_axis_angle)], dtype=torch.float32, device=device)
        left_quat_tensor = torch.tensor([_axis_angle_to_quat(left_axis_angle)], dtype=torch.float32, device=device)
        right_joints, left_joints = robot.run_bimanual_ik(
            right_pos_tensor,
            right_quat_tensor,
            left_pos_tensor,
            left_quat_tensor,
        )
        solver = robot._get_ik_solver()
        right_fk, _ = solver.forward_kinematics("Gripper_Tip_R")
        left_fk, _ = solver.forward_kinematics("Gripper_Tip_L")
        selected_fk = right_fk if arm == "right" else left_fk
        residual = float(np.linalg.norm(np.asarray(selected_fk, dtype=float) - np.asarray(target_relative, dtype=float)))
        joint_values = right_joints if arm == "right" else left_joints
        finite = bool(torch.isfinite(joint_values).all().item())
        return {
            "available": True,
            "reachable": bool(finite and residual <= 0.015),
            "arm": arm,
            "target_position_robot_base_relative": [round(value, 6) for value in target_relative],
            "position_residual": round(residual, 6),
            "joint_solution": [round(float(value), 6) for value in joint_values[0].detach().cpu().tolist()],
            "solver": "upstream_pink_bimanual_ik",
            "simulator_stepped": False,
        }
    except Exception as exc:
        return {
            "available": True,
            "reachable": False,
            "error": f"upstream_ik_failed:{type(exc).__name__}:{exc}",
            "simulator_stepped": False,
        }


def robowits_motion_query(
    env: Any,
    obs: Any,
    *,
    target_position: list[float],
    arm: str,
    axis_angle: list[float] | None,
    ignored_names: list[str] | None,
    clearance_radius: float | None,
    samples: int | None,
    context: JsonDict,
) -> JsonDict:
    target = _numeric_vector(target_position, limit=3)
    action_base, _ = _extract_ee_abs_action_base(obs, action_dim=14)
    if env is None:
        return {"available": False, "error": "live_env_not_created"}
    if target is None or action_base is None:
        return {"available": False, "error": "target_or_public_ee_state_unavailable"}
    base = _robowits_env_robot_base_position(env)
    world_positions = {
        "right": [float(action_base[index]) + float(base[index]) for index in range(3)],
        "left": [float(action_base[index + 3]) + float(base[index]) for index in range(3)],
    }
    selected_arm = str(arm or "auto").lower()
    if selected_arm not in world_positions:
        selected_arm = min(
            world_positions,
            key=lambda name: sum((float(world_positions[name][index]) - float(target[index])) ** 2 for index in range(3)),
        )
    selected_axis_angle = _numeric_vector(axis_angle, limit=3)
    if selected_axis_angle is None:
        offset = 6 if selected_arm == "right" else 9
        selected_axis_angle = [float(value) for value in action_base[offset : offset + 3]]
    records, _ = _robowits_live_records(env)
    radius = _clamp_float(float(clearance_radius if clearance_radius is not None else 0.025), 0.0, 0.15)
    path = _robowits_path_clearance_query(
        records,
        start=world_positions[selected_arm],
        target=target,
        ignored_names=ignored_names,
        clearance_radius=radius,
        samples=11 if samples is None else samples,
    )
    ik = _robowits_ik_query(
        env,
        action_base=action_base,
        arm=selected_arm,
        target_world=target,
        axis_angle=selected_axis_angle,
    )
    return {
        "available": True,
        "arm": selected_arm,
        "start_position": [round(value, 6) for value in world_positions[selected_arm]],
        "target_position": [round(value, 6) for value in target],
        "axis_angle": [round(value, 6) for value in selected_axis_angle],
        "translation_distance": round(sum((float(target[index]) - float(world_positions[selected_arm][index])) ** 2 for index in range(3)) ** 0.5, 6),
        "ik": ik,
        "path": path,
        "selection_deferred_to_coding_agent": True,
        "context": context,
    }


def _execute_robowits_ee_control_single(
    backend: RoboWitsAgentRuntimeBackend,
    *,
    target_position: list[float],
    arm: str,
    axis_angle: list[float] | None,
    gripper_width: float | None,
    repeat_steps: int | None,
    hold_steps: int | None,
    max_translation: float | None,
    observe_names: list[str] | None,
    context: JsonDict,
) -> JsonDict:
    episode_status = backend.episode_status()
    if episode_status["reached"]:
        return {
            "executed": False,
            "error": "episode_boundary_reached",
            "episode_status": episode_status,
            "context": context,
        }
    target = _numeric_vector(target_position, limit=3)
    action_base, _ = _extract_ee_abs_action_base(backend._last_obs, action_dim=14)
    if backend._env is None:
        return {"executed": False, "error": "live_env_not_created"}
    if int(backend._action_schema().get("action_dimension") or 0) != 14:
        return {"executed": False, "error": "controlled_ee_requires_14d_ee_mode"}
    if target is None or action_base is None:
        return {"executed": False, "error": "target_or_public_ee_state_unavailable"}
    base = _robowits_env_robot_base_position(backend._env)
    world_positions = {
        "right": [float(action_base[index]) + float(base[index]) for index in range(3)],
        "left": [float(action_base[index + 3]) + float(base[index]) for index in range(3)],
    }
    selected_arm = str(arm or "auto").lower()
    if selected_arm not in world_positions:
        selected_arm = min(
            world_positions,
            key=lambda name: sum((float(world_positions[name][index]) - float(target[index])) ** 2 for index in range(3)),
        )
    distance = sum((float(target[index]) - float(world_positions[selected_arm][index])) ** 2 for index in range(3)) ** 0.5
    translation_limit = _clamp_float(float(max_translation if max_translation is not None else 0.08), 0.005, 0.3)
    if distance > translation_limit + 1e-9:
        return {
            "executed": False,
            "error": "translation_exceeds_agent_supplied_control_bound",
            "translation_distance": round(distance, 6),
            "max_translation": round(translation_limit, 6),
        }
    selected_axis_angle = _numeric_vector(axis_angle, limit=3)
    position_offset = 0 if selected_arm == "right" else 3
    axis_offset = 6 if selected_arm == "right" else 9
    grip_offset = 12 if selected_arm == "right" else 13
    # Keep the other arm at its commanded pose. Reusing its measured (sagged)
    # position every chunk integrates gravity/servo error into a new target.
    held_action = getattr(backend, "_robowits_hold_action", None)
    if held_action is not None:
        other_indices = (3, 4, 5, 9, 10, 11, 13) if selected_arm == "right" else (0, 1, 2, 6, 7, 8, 12)
        for index in other_indices:
            action_base[index] = held_action[index]
        if gripper_width is None:
            action_base[grip_offset] = held_action[grip_offset]
    action = list(action_base)
    action[position_offset : position_offset + 3] = [float(target[index]) - float(base[index]) for index in range(3)]
    if selected_axis_angle is not None:
        action[axis_offset : axis_offset + 3] = selected_axis_angle
    if gripper_width is not None:
        action[grip_offset] = _clamp_float(float(gripper_width), 0.0, 0.1)
    action = [round(float(value), 6) for value in action[:14]]
    target_step_action = _robowits_step_action(action, action_dim=14)
    before_records, _ = _robowits_live_records(backend._env, observe_names)
    before_ee = robowits_public_ee_tracking_evidence(
        backend._last_obs,
        target_action=None,
        schema=backend._action_schema(),
        robot_base_position=base,
    )
    executed_steps = 0
    terminated = False
    truncated = False
    command_steps = max(1, min(int(repeat_steps or 1), 64))
    for step_index in range(command_steps):
        interpolated_action = _interpolate_robowits_ee_abs_action(
            action_base,
            action,
            arm=selected_arm,
            alpha=(step_index + 1) / command_steps,
        )
        step_action = _robowits_step_action(interpolated_action, action_dim=14)
        result = backend._env.step(step_action)
        backend._last_obs, _reward, terminated, truncated, step_info = _split_step_result(result)
        backend._last_info = step_info
        executed_steps += 1
        if terminated or truncated:
            backend._episode_status = _terminal_robowits_episode_status(
                terminated=terminated,
                truncated=truncated,
                steps_executed=executed_steps,
            )
            break
    requested_hold_steps = max(0, min(int(hold_steps or 0), 64))
    held_steps = 0
    while held_steps < requested_hold_steps and not (terminated or truncated):
        result = backend._env.step(target_step_action)
        backend._last_obs, _reward, terminated, truncated, step_info = _split_step_result(result)
        backend._last_info = step_info
        held_steps += 1
        executed_steps += 1
        if terminated or truncated:
            backend._episode_status = _terminal_robowits_episode_status(
                terminated=terminated,
                truncated=truncated,
                steps_executed=executed_steps,
            )
    after_records, _ = _robowits_live_records(backend._env, observe_names)
    if executed_steps:
        backend._robowits_hold_action = list(action)
    tracking = robowits_public_ee_tracking_evidence(
        backend._last_obs,
        target_action=target_step_action,
        schema=backend._action_schema(),
        robot_base_position=base,
    )
    return {
        "executed": True,
        "arm": selected_arm,
        "target_position_world": [round(value, 6) for value in target],
        "action_position_robot_base_relative": [round(float(value), 6) for value in action[position_offset : position_offset + 3]],
        "axis_angle": [round(float(value), 6) for value in action[axis_offset : axis_offset + 3]],
        "gripper_width": round(float(action[grip_offset]), 6),
        "translation_distance": round(distance, 6),
        "max_translation": round(translation_limit, 6),
        "steps_executed": executed_steps,
        "control_trajectory": {
            "mode": "smooth_ee_abs_interpolation",
            "command_steps": command_steps,
            "hold_steps": held_steps,
            "position": "linear",
            "orientation": "quaternion_shortest_arc",
            "gripper": "linear",
            "caller_target_unchanged": True,
        },
        "public_state_before": {"objects": before_records, "ee": before_ee},
        "public_state_after": {"objects": after_records, "ee": tracking},
        "tracking": tracking,
        "termination": {
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "replan_decision_deferred_to_caller": True,
        },
        "episode_status": backend.episode_status(),
        "context": context,
    }


def _initial_robowits_episode_status() -> JsonDict:
    return {
        "reached": False,
        "terminated": False,
        "truncated": False,
        "source": None,
    }


def _terminal_robowits_episode_status(*, terminated: bool, truncated: bool, steps_executed: int) -> JsonDict:
    return {
        "reached": True,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "source": "environment_step",
        "control_steps_executed": int(steps_executed),
        "diagnosis": "episode_terminated" if terminated else "episode_truncated",
    }


def robowits_motion_outcome_evidence(
    env: Any,
    obs: Any,
    *,
    object_name: str,
    arm: str,
    previous_object_position: list[float] | None,
    previous_ee_position: list[float] | None,
    reference_object_ee_offset: list[float] | None,
    context: JsonDict,
) -> JsonDict:
    records, error = _robowits_live_records(env, [object_name])
    record = records.get(object_name)
    ee_state, _ = _robowits_world_ee_state(env, obs)
    center = _numeric_vector((record or {}).get("center"), limit=3)
    if center is None or ee_state is None:
        return {"available": False, "error": error or "public_object_or_ee_state_unavailable"}
    candidates = {"right": ee_state[0:3], "left": ee_state[3:6]}
    selected_arm = str(arm or "auto").lower()
    if selected_arm not in candidates:
        selected_arm = min(candidates, key=lambda name: sum((float(candidates[name][index]) - float(center[index])) ** 2 for index in range(3)))
    ee_position = [float(value) for value in candidates[selected_arm]]
    old_object = _numeric_vector(previous_object_position, limit=3)
    old_ee = _numeric_vector(previous_ee_position, limit=3)
    reference_offset = _numeric_vector(reference_object_ee_offset, limit=3)
    current_offset = [float(center[index]) - float(ee_position[index]) for index in range(3)]
    object_delta = None if old_object is None else [float(center[index]) - float(old_object[index]) for index in range(3)]
    ee_delta = None if old_ee is None else [float(ee_position[index]) - float(old_ee[index]) for index in range(3)]
    offset_drift = None if reference_offset is None else sum((current_offset[index] - float(reference_offset[index])) ** 2 for index in range(3)) ** 0.5
    object_motion = None if object_delta is None else sum(value**2 for value in object_delta) ** 0.5
    ee_motion = None if ee_delta is None else sum(value**2 for value in ee_delta) ** 0.5
    displacement_threshold = _clamp_float(float(context.get("displacement_threshold", 0.01)), 0.001, 0.2)
    attachment_threshold = _clamp_float(float(context.get("attachment_offset_tolerance", 0.035)), 0.005, 0.2)
    failure_signals: list[str] = []
    if object_motion is not None and object_motion >= displacement_threshold:
        failure_signals.append("object_displaced")
    if offset_drift is not None and offset_drift > attachment_threshold:
        failure_signals.append("attachment_lost_suspected")
    return {
        "available": True,
        "object_name": object_name,
        "arm": selected_arm,
        "object": record,
        "ee_position_world": [round(value, 6) for value in ee_position],
        "object_ee_offset": [round(value, 6) for value in current_offset],
        "object_displacement": {
            "delta": None if object_delta is None else [round(value, 6) for value in object_delta],
            "distance": None if object_motion is None else round(object_motion, 6),
            "threshold": round(displacement_threshold, 6),
        },
        "ee_displacement": {
            "delta": None if ee_delta is None else [round(value, 6) for value in ee_delta],
            "distance": None if ee_motion is None else round(ee_motion, 6),
        },
        "attachment": {
            "reference_offset": reference_offset,
            "offset_drift": None if offset_drift is None else round(offset_drift, 6),
            "consistent": None if offset_drift is None else bool(offset_drift <= attachment_threshold),
            "tolerance": round(attachment_threshold, 6),
        },
        "failure_signals": failure_signals,
        "recovery_action_selected": False,
        "selection_deferred_to_coding_agent": True,
        "context": context,
    }


def _robowits_pose_diagnostics(records: JsonDict, *, reference_scene: JsonDict) -> JsonDict:
    diagnostics: JsonDict = {"object_count": len(records)}
    pairwise: list[JsonDict] = []
    names = sorted(records)
    for left_index, left_name in enumerate(names):
        left = records[left_name]
        left_center = left.get("center") if isinstance(left, dict) else None
        for right_name in names[left_index + 1 :]:
            right = records[right_name]
            right_center = right.get("center") if isinstance(right, dict) else None
            if not (
                isinstance(left_center, list)
                and isinstance(right_center, list)
                and len(left_center) >= 3
                and len(right_center) >= 3
            ):
                continue
            delta = [round(float(right_center[idx]) - float(left_center[idx]), 6) for idx in range(3)]
            pairwise.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "delta": delta,
                    "distance_xy": round((delta[0] ** 2 + delta[1] ** 2) ** 0.5, 6),
                    "distance_3d": round(sum(value**2 for value in delta) ** 0.5, 6),
                }
            )
    diagnostics["pairwise_geometry"] = pairwise
    reference = {item["name"]: item for item in geometry_candidates(reference_scene)}
    displacement = {}
    for name, record in records.items():
        ref_center = reference.get(name, {}).get("center")
        center = record.get("center")
        if isinstance(ref_center, list) and isinstance(center, list) and len(ref_center) >= 3 and len(center) >= 3:
            displacement[name] = [round(float(center[idx]) - float(ref_center[idx]), 6) for idx in range(3)]
    if displacement:
        diagnostics["displacement_from_eval_json_center"] = displacement
    return diagnostics


def _numeric_vector(value: Any, *, limit: int) -> list[float] | None:
    try:
        import numpy as np

        array = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return None
    if array.size < limit:
        return None
    return [round(float(item), 6) for item in array[:limit]]


def _numeric_matrix(value: Any, *, rows: int, cols: int) -> list[list[float]] | None:
    try:
        import numpy as np

        array = np.asarray(value, dtype=float)
    except Exception:
        return None
    if array.ndim < 2 or array.shape[0] < rows or array.shape[1] < cols:
        return None
    return [[round(float(array[row, col]), 6) for col in range(cols)] for row in range(rows)]


def _vector_norm(values: list[float] | None) -> float | None:
    if values is None:
        return None
    return round(sum(float(value) ** 2 for value in values) ** 0.5, 6)


def _robowits_camera_summaries(obs: Any) -> dict[str, JsonDict]:
    if not isinstance(obs, dict):
        return {}
    cameras: dict[str, JsonDict] = {}
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for camera_name, value in pixels.items():
            cameras.setdefault(str(camera_name), {})["rgb"] = summarize_observation(value)
    camera_obs = obs.get("camera_observations")
    if isinstance(camera_obs, dict):
        for key, value in camera_obs.items():
            _add_robowits_camera_modality(cameras, str(key), value)
    for key, value in obs.items():
        if isinstance(key, str):
            _add_robowits_camera_modality(cameras, key, value)
    return cameras


def _robowits_visual_arrays(obs: Any) -> list[tuple[str, str, Any]]:
    if not isinstance(obs, dict):
        return []
    arrays: dict[tuple[str, str], Any] = {}
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for camera_name, value in pixels.items():
            arrays[(str(camera_name), "rgb")] = value

    def add(key: str, value: Any) -> None:
        for suffix, modality in (
            ("_rgb", "rgb"),
            ("_image", "rgb"),
            ("_depth", "depth"),
            ("_segmentation", "segmentation"),
            ("_seg", "segmentation"),
            ("_pointcloud", "pointcloud"),
            ("_point_cloud", "pointcloud"),
        ):
            if key.endswith(suffix):
                arrays.setdefault((key[: -len(suffix)] or "camera", modality), value)
                return

    camera_obs = obs.get("camera_observations")
    if isinstance(camera_obs, dict):
        for key, value in camera_obs.items():
            if isinstance(value, dict):
                for modality, array in value.items():
                    lowered = str(modality).lower()
                    if lowered in {"rgb", "depth", "segmentation", "seg", "pointcloud", "point_cloud"}:
                        normalized = "segmentation" if lowered == "seg" else "pointcloud" if lowered == "point_cloud" else lowered
                        arrays.setdefault((str(key), normalized), array)
            else:
                add(str(key), value)
    for key, value in obs.items():
        if isinstance(key, str):
            add(key, value)
    return [(camera, modality, value) for (camera, modality), value in sorted(arrays.items())]


def _add_robowits_camera_modality(cameras: dict[str, JsonDict], key: str, value: Any) -> None:
    for suffix, modality in (
        ("_rgb", "rgb"),
        ("_image", "rgb"),
        ("_depth", "depth"),
        ("_segmentation", "segmentation"),
        ("_seg", "segmentation"),
    ):
        if key.endswith(suffix):
            camera_name = key[: -len(suffix)] or "camera"
            cameras.setdefault(camera_name, {})[modality] = summarize_observation(value)
            return


def _robowits_rgb_arrays(obs: Any) -> dict[str, Any]:
    if not isinstance(obs, dict):
        return {}
    arrays: dict[str, Any] = {}
    pixels = obs.get("pixels")
    if isinstance(pixels, dict):
        for camera_name, value in pixels.items():
            arrays[str(camera_name)] = value
    camera_obs = obs.get("camera_observations")
    if isinstance(camera_obs, dict):
        for key, value in camera_obs.items():
            _add_robowits_rgb_array(arrays, str(key), value)
    for key, value in obs.items():
        if isinstance(key, str):
            _add_robowits_rgb_array(arrays, key, value)
    return arrays


def _add_robowits_rgb_array(arrays: dict[str, Any], key: str, value: Any) -> None:
    for suffix in ("_rgb", "_image"):
        if key.endswith(suffix):
            camera_name = key[: -len(suffix)] or "camera"
            arrays.setdefault(camera_name, value)
            return


def _select_camera_name(cameras: list[str], *, camera_name: str | None, query: str | None, context: JsonDict) -> str:
    if camera_name:
        return camera_name
    terms = _query_terms(query, context)
    for camera in cameras:
        camera_text = camera.lower().replace("_", " ")
        if any(term in camera_text for term in terms):
            return camera
    for preferred in ("ego", "front", "corner2", "wrist_left", "wrist_right"):
        if preferred in cameras:
            return preferred
    return cameras[0]


def _rgb_array_to_hwc_uint8(value: Any) -> Any:
    import numpy as np

    array = np.asarray(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected RGB array with shape HxWx3 or 1xHxWx3, got {list(array.shape)}")
    array = array[..., :3]
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating):
            max_value = float(np.nanmax(array)) if array.size else 0.0
            if max_value <= 1.0:
                array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _normalize_pixel_bbox(bbox: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
    if bbox is None:
        return (0, 0, width, height)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be a four-item [x0, y0, x1, y1] list")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox has empty area after clipping: {[x0, y0, x1, y1]}")
    return (x0, y0, x1, y1)


def _center_bbox(*, width: int, height: int, fraction: float) -> tuple[int, int, int, int]:
    crop_w = max(1, int(width * fraction))
    crop_h = max(1, int(height * fraction))
    x0 = max(0, (width - crop_w) // 2)
    y0 = max(0, (height - crop_h) // 2)
    return (x0, y0, min(width, x0 + crop_w), min(height, y0 + crop_h))


def _summarize_rgb_region(image: Any, *, name: str, bbox: tuple[int, int, int, int]) -> JsonDict:
    import numpy as np

    x0, y0, x1, y1 = bbox
    region = image[y0:y1, x0:x1, :3]
    flat = region.reshape(-1, 3).astype(float)
    mean_rgb = flat.mean(axis=0) if flat.size else np.zeros(3)
    std_rgb = flat.std(axis=0) if flat.size else np.zeros(3)
    min_rgb = flat.min(axis=0) if flat.size else np.zeros(3)
    max_rgb = flat.max(axis=0) if flat.size else np.zeros(3)
    return {
        "name": name,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "width": int(x1 - x0),
        "height": int(y1 - y0),
        "area_pixels": int(max(0, x1 - x0) * max(0, y1 - y0)),
        "channels": ["R", "G", "B"],
        "mean_rgb": [round(float(value), 3) for value in mean_rgb],
        "std_rgb": [round(float(value), 3) for value in std_rgb],
        "min_rgb": [int(value) for value in min_rgb],
        "max_rgb": [int(value) for value in max_rgb],
    }


def _python_platform_summary() -> JsonDict:
    try:
        from packaging import tags

        tag_head = [str(tag) for tag in list(tags.sys_tags())[:12]]
    except Exception as exc:
        tag_head = [f"packaging.tags unavailable: {type(exc).__name__}: {exc}"]
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "platform": platform.platform(),
        "sysconfig_platform": sysconfig.get_platform(),
        "libc": list(platform.libc_ver()),
        "compatible_tag_head": tag_head,
    }


def _import_probe(module_name: str) -> JsonDict:
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return {"discoverable": False, "importable": False, "error": "module_not_found"}
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        return {"discoverable": True, "importable": True, "version": str(version) if version is not None else None}
    except Exception as exc:
        return {"discoverable": True, "importable": False, "error": f"{type(exc).__name__}: {exc}"}


def _torch_probe() -> JsonDict:
    result = _import_probe("torch")
    if not result.get("importable"):
        return result
    try:
        import torch

        result["cuda_available"] = bool(torch.cuda.is_available())
        result["cuda_device_count"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception as exc:
        result["cuda_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _dataset_file_summary(path: Path, *, episode_index: int) -> JsonDict:
    summary: JsonDict = {"path": str(path), "loaded": False}
    if not path.exists():
        summary["error"] = "dataset_json_missing"
        return summary
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
        return summary
    if not isinstance(data, list) or not data:
        summary["error"] = "dataset_json_not_nonempty_list"
        return summary
    selected_index = max(0, min(episode_index, len(data) - 1))
    scene = data[selected_index]
    summary.update(
        {
            "loaded": True,
            "episodes": len(data),
            "selected_episode_index": selected_index,
            "object_count": len(scene) if isinstance(scene, dict) else None,
            "object_names": sorted(str(key) for key in scene) if isinstance(scene, dict) else [],
        }
    )
    return summary


def _asset_preflight(repo: Path) -> JsonDict:
    assets_dir = repo / "assets"
    hf_assets = assets_dir / "hf_assets"
    metadata = assets_dir / "metadata.json"
    return {
        "assets_dir_exists": assets_dir.exists(),
        "setup_script_exists": (assets_dir / "setup_assets.sh").exists(),
        "metadata_json_exists": metadata.exists(),
        "hf_assets_dir_exists": hf_assets.exists(),
        "hf_asset_file_count": len(list(hf_assets.rglob("*"))) if hf_assets.exists() else 0,
        "blenderkit_secret_configured": bool(
            os.environ.get("BLENDERKIT_API_KEY")
            or os.environ.get("BLENDERKIT_KEY")
        ),
        "secret_values_redacted": True,
    }


def _robowits_blockers(*, repo: Path, dataset: JsonDict, gs_gym_ok: bool, gs_gym_error: str | None) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    if not repo.exists():
        blockers.append({"id": "repo_missing", "detail": str(repo)})
    if not dataset.get("loaded"):
        blockers.append({"id": "dataset_missing_or_invalid", "detail": dataset.get("error")})
    if not gs_gym_ok:
        blockers.append({"id": "gs_gym_import_failed", "detail": gs_gym_error})
    if importlib.util.find_spec("genesis") is None:
        blockers.append({"id": "genesis_missing", "detail": "Install Genesis/gs_gym dependency stack before live env reset."})
    if shutil.which("blender") is None:
        blockers.append({"id": "blender_missing", "detail": "RoboWits asset conversion expects blender on PATH."})
    if shutil.which("nvidia-smi") is None:
        blockers.append({"id": "gpu_probe_unavailable", "detail": "nvidia-smi not found; GPU readiness is unverified."})
    assets = _asset_preflight(repo)
    if not assets["hf_assets_dir_exists"] or assets["hf_asset_file_count"] == 0:
        blockers.append({"id": "assets_missing", "detail": "assets/hf_assets is absent or empty."})
    return blockers


def _module_discoverable(module_name: str, repo: Path) -> bool:
    if importlib.util.find_spec(module_name) is not None:
        return True
    if module_name == "gs_gym" and (repo / "gs_gym").exists():
        try:
            import sys

            sys.path.insert(0, str(repo))
            return importlib.util.find_spec(module_name) is not None
        except Exception:
            return False
    return False


def _actual_importable(module_name: str, repo: Path) -> tuple[bool, str | None]:
    if module_name == "gs_gym" and (repo / "gs_gym").exists():
        import sys

        sys.path.insert(0, str(repo))
    try:
        __import__(module_name)
        return True, None
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _query_terms(query: str | None, context: JsonDict) -> list[str]:
    raw = " ".join([query or "", " ".join(str(value) for value in context.values())])
    return [term.lower() for term in raw.replace("_", " ").replace("-", " ").split() if len(term) > 2]


def _split_reset_result(result: Any) -> tuple[Any, JsonDict]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result, {}


def _split_step_result(result: Any) -> tuple[Any, Any, bool, bool, JsonDict]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, bool(terminated), bool(truncated), dict(info or {})
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, reward, bool(done), False, dict(info or {})
    return result, None, False, False, {}


def _robowits_env_candidates(env: Any) -> list[Any]:
    if env is None:
        return []
    candidates: list[Any] = [env]
    unwrapped = getattr(env, "unwrapped", None)
    if unwrapped is not None and unwrapped is not env:
        candidates.append(unwrapped)
    envs = getattr(env, "envs", None)
    if isinstance(envs, list | tuple):
        candidates.extend(item for item in envs if item is not None)
    unique: list[Any] = []
    seen: set[int] = set()
    for candidate in candidates:
        ident = id(candidate)
        if ident in seen:
            continue
        seen.add(ident)
        unique.append(candidate)
    return unique


def _robowits_env_task_description(env: Any) -> str | None:
    """Read the benchmark's public natural-language instruction, if available."""

    for candidate in _robowits_env_candidates(env):
        try:
            description = getattr(candidate, "task_description", None)
            if callable(description):
                description = description()
        except Exception:
            continue
        if isinstance(description, str) and description.strip():
            return description.strip()
    return None


def _boolish_success_value(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, dict):
        for key in ("is_success", "success"):
            if key in value:
                return _boolish_success_value(value[key])
        return None
    try:
        import numpy as np

        array = np.asarray(_to_builtin(value))
        if array.size == 0:
            return None
        return bool(array.astype(float).max() > 0.0)
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return bool(value.item())
        except Exception:
            return None
    if isinstance(value, (list, tuple)):
        bools = [_boolish_success_value(item) for item in value]
        bools = [item for item in bools if item is not None]
        return bool(any(bools)) if bools else None
    return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
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


def _primitive_output(value: Any) -> Any:
    if hasattr(value, "output"):
        return getattr(value, "output")
    if isinstance(value, dict) and "output" in value and len(value) <= 4:
        return value["output"]
    return value


def execute_robowits_ee_control(backend, **kwargs):
    from .robowits_segmented_control import execute
    return execute(backend, _execute_robowits_ee_control_single, **kwargs)
