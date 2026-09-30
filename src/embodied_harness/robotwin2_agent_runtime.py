from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Iterator

import numpy as np

from .backend import EmbodiedBackend
from .paths import get_project_paths, resolve_project_root
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]

_REPO_ROOT = resolve_project_root()
DEFAULT_ROBOTWIN2_REPO_CANDIDATES = (
    get_project_paths().external_upstream("robotwin2"),
)
DEFAULT_ROBOTWIN2_REPO = next(
    (path for path in DEFAULT_ROBOTWIN2_REPO_CANDIDATES if path.exists()),
    DEFAULT_ROBOTWIN2_REPO_CANDIDATES[0],
)
DEFAULT_ROBOTWIN2_PUBLIC_CASE = (
    _REPO_ROOT / "benchmarks/operation/robotwin2/cases/place_empty_cup_visual_low_level.json"
)
ROBOTWIN2_HF_DATASET_REPO = "TianxingChen/RoboTwin2.0"
ROBOTWIN2_BACKGROUND_TEXTURE_ZIP_SIZE = 10_970_687_027
ROBOTWIN2_BACKGROUND_TEXTURE_SHA256 = "54ede0fb5b783e0faa2bc98720d3affd6ca3bb9280b225b48c1aafaf31473070"
ROBOTWIN2_REQUIRED_LIVE_MODULES = ("sapien", "gymnasium")


class RoboTwin2CuroboPlannerUnavailable(RuntimeError):
    """Raised when upstream RoboTwin2 cannot materialize its cuRobo planner."""


def _repo_local_robotwin2_python(repo_path: str | Path) -> Path | None:
    repo = Path(repo_path).expanduser().resolve()
    candidates = (
        repo.parent.parent / ".venv-robotwin2/bin/python",
        get_project_paths().external_environment("robotwin2") / "bin/python",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.absolute()
    return None


@dataclass(slots=True)
class RoboTwin2RuntimeConfig:
    repo_path: str = str(DEFAULT_ROBOTWIN2_REPO)
    assets_path: str | None = None
    task_name: str = "place_empty_cup"
    task_config: str = "demo_clean"
    embodiment: list[Any] | None = None
    action_type: str = "ee"
    live: bool = False
    render_freq: int = 0
    data_type: JsonDict = field(
        default_factory=lambda: {
            "rgb": True,
            "depth": True,
            "pointcloud": False,
            "mesh_segmentation": True,
            "actor_segmentation": True,
            "endpose": True,
            "qpos": True,
            "observer": False,
            "third_view": False,
        }
    )
    env_kwargs: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RoboTwin2AgentRuntimeBackend(EmbodiedBackend):
    """Agent-native RoboTwin 2.0 adapter.

    The public surface keeps native camera data and generic arm control while
    excluding task recipes, actor-keypoint oracles, and success checkers.
    """

    def __init__(
        self,
        config: RoboTwin2RuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.config = config or RoboTwin2RuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._actors: dict[str, Any] = {}
        self._visual_handles: dict[str, JsonDict] = {}
        self._observation_round = 0
        self._selected_native_episode: JsonDict | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        task_name = str(coordinate.get("task_id") or "")
        parts = str(coordinate.get("variation") or "").split("::")
        repo = Path(self.config.repo_path).resolve()
        if (len(parts) != 3 or parts[:2] != ["demo_clean", task_name]
                or not parts[2].startswith("accepted_episode_")
                or not parts[2][17:].isdigit() or not task_name.isidentifier()
                or task_name.startswith("_") or not (repo / "envs" / f"{task_name}.py").is_file()
                or type(coordinate.get("seed")) is not int or coordinate["seed"] != 0):
            raise ValueError("Invalid native RoboTwin2 task/evaluation ordinal/user-seed coordinate")
        index = int(parts[2][17:])
        if not 0 <= index < 100:
            raise ValueError("RoboTwin2 native evaluation has 100 accepted episodes per task")
        path = Path(os.environ.get("ROBOTWIN2_NATIVE_SEED_INDEX") or
                    get_project_paths().external_assets("robotwin2") / "evaluation_seeds" / f"{task_name}.json")
        content = path.read_bytes()
        digest = hashlib.sha256(content).hexdigest()
        expected = os.environ.get("ROBOTWIN2_NATIVE_SEED_INDEX_SHA256")
        if expected is not None and digest != expected:
            raise ValueError("RoboTwin2 native accepted-seed index digest changed")
        data = json.loads(content)
        source = repo / "script/eval_policy.py"
        if (data.get("task") != task_name or data.get("task_config") != "demo_clean"
                or data.get("user_seed") != 0 or data.get("native_test_num") != 100
                or data.get("generator_sha256") != hashlib.sha256(source.read_bytes()).hexdigest()
                or data.get("complete_object_assets") is not True):
            raise ValueError("RoboTwin2 index is not bound to the original native evaluation and complete assets")
        episodes = data.get("episodes", [])
        if index >= len(episodes):
            raise ValueError("Selected native accepted episode has not been materialized")
        episode = episodes[index]
        if (episode.get("episode_index") != index or type(episode.get("seed")) is not int
                or episode["seed"] < 100000 or not isinstance(episode.get("episode_info", {}).get("info"), dict)):
            raise ValueError("Invalid original expert-accepted RoboTwin2 episode metadata")
        self._selected_native_episode = {"task_name": task_name, "episode_index": index,
            "native_seed": episode["seed"], "episode_info": deepcopy(episode["episode_info"])}
        return {"bound": True, "mode": "native_expert_accepted_episode", "task_name": task_name,
                "episode_index": index, "user_seed": 0, "native_seed": episode["seed"],
                "seed_index_sha256": digest, "seed_index_file": str(path)}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        runtime_config = self._merged_config(config or {})
        if self._selected_native_episode is not None:
            selected = self._selected_native_episode
            runtime_config.task_name = selected["task_name"]
            runtime_config.task_config = "demo_clean"
            runtime_config.env_kwargs.update(eval_mode=True, is_test=True,
                now_ep_num=selected["episode_index"], eval_video_log=False)
            seed = selected["native_seed"]
        if runtime_config.embodiment is None:
            runtime_config.embodiment = _load_task_config(runtime_config.repo_path, runtime_config.task_config).get(
                "embodiment", ["aloha-agilex"]
            )
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._last_obs = None
        self._last_info = {}
        self._actors = {}
        self._visual_handles = {}
        self._observation_round = 0
        repo = inspect_robotwin2_repo(runtime_config.repo_path)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w7:robotwin2:agent_runtime",
            instruction=(
                "Solve a RoboTwin 2.0 bimanual manipulation task from native camera evidence "
                "and caller-authored generic dual-arm controls."
            ),
            goal={
                "benchmark": "RoboTwin2",
                "task_name": runtime_config.task_name,
                "task_config": runtime_config.task_config,
                "success_source": "private_official_environment_verifier_only",
                "success_exposed_to_agent": False,
            },
            initial_state={
                "repo": repo,
                "actor_registry_source": "live_environment_only",
            },
            budgets={"primitive_calls": 48, "verifier_calls": 6},
            tags=["w7", "robotwin2", "bimanual", "sapien", "live" if runtime_config.live else "source_boundary"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "robotwin2",
                "runtime_config": runtime_config.to_dict(),
                "upstream": repo,
                "agent_native_contract": {
                    "primitives_accept_prompt_query_context": True,
                    "dual_arm_generic_controls_exposed": True,
                    "actor_keypoints_exposed": False,
                    "caller_authored_arm_motion_exposed": True,
                    "visual_modalities_exposed_when_live": True,
                    "actor_visual_world_geometry_exposed_when_live": True,
                    "read_only_motion_plan_probe_exposed": True,
                    "current_arm_pose_introspection_exposed": True,
                    "visual_provenance_required_by_actions": True,
                    "native_sensor_modalities": ["rgb", "depth", "mesh_segmentation", "actor_segmentation", "pointcloud"],
                    "task_recipe_primitives_exposed": False,
                    "success_function_exposed_as_primitive": False,
                    "expert_play_once_exposed_as_primitive": False,
                    "demo_replay_exposed_as_primitive": False,
                },
            },
        )

        if runtime_config.live:
            self._env = self._make_env(runtime_config, seed=seed)
            if self._selected_native_episode is not None:
                self._task_spec.instruction = str(self._env.get_instruction())
                self._task_spec.source = f"robotwin2:{runtime_config.task_name}:native_eval"
            self._refresh_actor_registry()
            self._capture_live_observation()
        else:
            self._env = None

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
        if self._env is not None:
            self._refresh_actor_registry()
            self._capture_live_observation()
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "task": {
                    "task_name": self.config.task_name,
                    "task_config": self.config.task_config,
                    "embodiment": self.config.embodiment,
                    "action_type": self.config.action_type,
                },
                "actor_names": sorted(self._actors),
                "actor_evidence": self._actor_evidence(),
                "visual_evidence": robotwin2_visual_evidence(self._last_obs),
                "observation_summary": summarize_observation(self._last_obs),
                "camera_schema": self._camera_schema(),
                "object_schema": self._object_schema(),
                "arm_schema": self._arm_schema(),
                "action_schema": self._action_schema(),
            },
            metadata={"benchmark_id": "robotwin2", "task_name": self.config.task_name},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_robotwin2_scene",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"task": "dict", "actor_names": "list[str]", "actor_evidence": "dict", "action_schema": "dict"},
                "Observe RoboTwin2 task, actor, embodiment, and action-schema evidence.",
            ),
            self._primitive_card(
                "observe_robotwin2_visual",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_name": "str|None"},
                {"visual_evidence": "dict", "evidence_handles": "list[str]", "observation_summary": "dict"},
                "Inspect native live RGB/depth/segmentation/point-cloud arrays and issue provenance handles.",
            ),
            self._primitive_card(
                "check_robotwin2_asset_readiness",
                "L1",
                {"task_config": "str|None", "agent_context": "dict|None"},
                {"readiness": "dict", "selected_config_ready": "bool", "blockers": "list[dict]"},
                "Expose demo_clean, demo_randomized, and full-suite asset readiness before starting a live env.",
            ),
            self._primitive_card(
                "locate_robotwin2_actor",
                "L2",
                {"query": "str|None", "actor_name": "str|None", "agent_context": "dict|None"},
                {"selected": "dict|None", "candidates": "list[dict]", "pose_world": "list[float]|None"},
                "Enumerate live SAPIEN actors and inspect only an exact caller-selected registry or native actor name.",
            ),
            self._primitive_card(
                "measure_robotwin2_actor_visual",
                "L2",
                {
                    "segmentation_id": "int|list[int]|None",
                    "actor_label": "str|None",
                    "camera_name": "str",
                    "depth_scale": "float",
                    "quaternion_wxyz": "list[float]|None",
                    "max_points": "int",
                    "prompt": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "bbox_pixels": "list[int]",
                    "centroid_pixels": "list[float]",
                    "world_points": "list[list[float]]",
                    "centroid_world": "list[float]",
                    "pose_world": "list[float]",
                    "evidence_handles": "list[str]",
                },
                "Measure an explicitly selected actor mask from native RGB, depth, actor segmentation, and camera calibration.",
            ),
            self._primitive_card(
                "inspect_robotwin2_actor_points",
                "L2",
                {"actor_name": "str", "point_type": "contact|functional|target|orientation", "agent_context": "dict|None"},
                {"pose_world": "list[float]|None", "points": "list[dict]", "visual_evidence": "false"},
                "Inspect upstream simulator geometry, explicitly labeled non-visual and ineligible as action provenance.",
            ),
            self._primitive_card(
                "inspect_robotwin2_arm_pose",
                "L1",
                {"arm_tag": "left|right", "agent_context": "dict|None"},
                {"arm_tag": "str", "pose_world": "list[float]|None", "source": "str"},
                "Read the current caller-selected end-effector pose from the official environment.",
            ),
            self._primitive_card(
                "probe_robotwin2_motion_plan",
                "L2",
                {
                    "arm_tag": "left|right",
                    "target_pose": "list[float]|visual_pose_reference|None",
                    "evidence_handles": "list[str]",
                    "agent_context": "dict|None",
                },
                {
                    "arm_tag": "str",
                    "current_pose": "list[float]|None",
                    "target_pose": "list[float]",
                    "status": "str|None",
                    "path_summary": "dict",
                    "plan_success_unchanged": "bool",
                },
                "Probe an official arm motion path to a caller target, or the current pose when omitted, without executing it.",
            ),
            self._primitive_card(
                "execute_robotwin2_actions",
                "L3",
                {
                    "left": "action_descriptor|None",
                    "right": "action_descriptor|None",
                    "evidence_handles": "list[str]",
                    "agent_context": "dict|None",
                },
                {"executed": "bool", "arms": "list[str]", "actions": "dict"},
                "Execute one caller-selected action per arm, synchronously when both left and right are supplied.",
            ),
            self._primitive_card(
                "execute_robotwin2_pick_place",
                "L3",
                {
                    "source_actor_label": "str",
                    "target_actor_label": "str",
                    "arm_tag": "left|right",
                    "evidence_handles": "list[str]",
                    "pre_grasp_distance": "float",
                    "lift_distance": "float",
                    "pre_place_distance": "float",
                    "agent_context": "dict|None",
                },
                {"executed": "bool", "arm_tag": "str", "completed_stages": "list[dict]"},
                "Execute a caller-selected semantic pick/place with upstream generic grasp_actor and place_actor helpers.",
            ),
            self._primitive_card(
                "move_robotwin2_arm",
                "L3",
                {
                    "arm_tag": "left|right",
                    "target_pose": "list[float]|None",
                    "delta": "list[float]|None",
                    "move_axis": "world|arm",
                    "evidence_handles": "list[str]",
                    "agent_context": "dict|None",
                },
                {"executed": "bool", "arm_tag": "str", "requires_live_env": "bool"},
                "Move one arm to an absolute pose or by displacement using upstream move_to_pose/move_by_displacement.",
            ),
            self._primitive_card(
                "set_robotwin2_gripper",
                "L3",
                {
                    "arm_tag": "left|right",
                    "command": "open|close",
                    "pos": "float|None",
                    "evidence_handles": "list[str]",
                    "agent_context": "dict|None",
                },
                {"executed": "bool", "arm_tag": "str", "command": "str", "requires_live_env": "bool"},
                "Open or close one RoboTwin2 gripper through upstream open_gripper/close_gripper and move().",
            ),
            self._primitive_card(
                "submit_robotwin2_ee_action",
                "L3",
                {"action": "list[float]", "action_type": "ee|delta_ee|qpos|None", "evidence_handles": "list[str]", "agent_context": "dict|None"},
                {"submitted": "bool", "action_type": "str", "requires_live_env": "bool"},
                "Submit a raw RoboTwin2 deployment action through take_action; useful for policy-as-skill testing.",
            ),
            self._primitive_card(
                "record_robotwin2_evidence",
                "L1",
                {"key": "str", "value": "any", "agent_context": "dict|None"},
                {"artifact_id": "str"},
                "Record agent-selected RoboTwin2 evidence in the trace.",
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
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by RoboTwin2AgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported RoboTwin2 verification scope: {scope}")
        elif self._env is None:
            result = VerificationResult(
                ok=False,
                scope="task",
                message="RoboTwin2 live env is not created; source-boundary contract only.",
                metrics={"success": 0.0},
                metadata={"live_env": False},
            )
        elif not hasattr(self._env, "check_success"):
            result = VerificationResult(
                ok=False,
                scope="task",
                message="RoboTwin2 env does not expose check_success.",
                metrics={"success": 0.0},
                metadata={"live_env": True},
            )
        else:
            success = bool(self._env.check_success())
            result = VerificationResult(
                ok=success,
                scope="task",
                message=(
                    "RoboTwin2 official task verifier passed"
                    if success
                    else "RoboTwin2 official task verifier has not passed"
                ),
                metrics={"success": float(success)},
                metadata={"plan_success": bool(getattr(self._env, "plan_success", False))},
            )
        if result.ok:
            self.get_trace().final_status = "success"
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("RoboTwin2 backend must be reset before reading trace")
        return self._trace

    def runtime_available(self) -> JsonDict:
        repo = inspect_robotwin2_repo(self.config.repo_path)
        module_status = {
            module: importlib.util.find_spec(module) is not None
            for module in ROBOTWIN2_REQUIRED_LIVE_MODULES
        }
        repo_python = _repo_local_robotwin2_python(self.config.repo_path)
        curobo_src = _robotwin2_vendored_curobo_src(Path(self.config.repo_path).expanduser())
        return {
            "live": self._env is not None,
            "repo_ready": repo["repo_ready"],
            "assets_ready": repo["assets"]["extracted_ready"],
            "asset_readiness": repo["asset_readiness"]["readiness"],
            "task_name": self.config.task_name,
            "task_config": self.config.task_config,
            "required_live_modules": module_status,
            "missing_live_modules": [name for name, ok in module_status.items() if not ok],
            "sapien_importable": module_status["sapien"],
            "gymnasium_importable": module_status["gymnasium"],
            "repo_local_python": str(repo_python) if repo_python is not None else None,
            "vendored_curobo_src": str(curobo_src) if curobo_src.exists() else None,
        }

    def _primitive_observe_robotwin2_scene(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        obs = self.observe()
        payload = deepcopy(obs.data)
        payload.update({"prompt": prompt, "query": query, "agent_context": agent_context or {}})
        return PrimitiveResult(name="observe_robotwin2_scene", ok=True, output=payload, artifacts=obs.artifacts)

    def _primitive_observe_robotwin2_visual(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        camera_name: str | None = None,
    ) -> PrimitiveResult:
        if self._env is not None:
            self._capture_live_observation()
        visual = robotwin2_visual_evidence(self._last_obs, camera_name=camera_name)
        handles = self._register_visual_handles(
            prompt=prompt,
            query=query,
            camera_name=camera_name,
            agent_context=agent_context or {},
        )
        artifact_id = f"robotwin2_visual_{len(self.get_trace().artifacts)}"
        artifact = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "camera_name": camera_name,
            "visual_evidence": visual,
            "evidence_handles": handles,
            "provenance_source": "live_env.get_obs",
        }
        self.get_trace().add_artifact(artifact_id, artifact)
        return PrimitiveResult(
            name="observe_robotwin2_visual",
            ok=bool(visual["visual_ready"]),
            output={
                **artifact,
                "observation_summary": summarize_observation(self._last_obs),
                "artifacts": [artifact_id],
                "requires_live_env_for_pixels": self._env is None,
            },
            artifacts=[artifact_id],
            error=None if visual["visual_ready"] else "live_visual_observation_unavailable",
        )

    def _primitive_check_robotwin2_asset_readiness(
        self,
        task_config: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        selected_config = task_config or self.config.task_config
        readiness = robotwin2_asset_readiness(
            self.config.repo_path, assets_path=self.config.assets_path
        )
        selected_ready = _robotwin2_config_assets_ready(readiness, selected_config)
        blockers = _robotwin2_config_asset_blockers(readiness, selected_config)
        return PrimitiveResult(
            name="check_robotwin2_asset_readiness",
            ok=selected_ready,
            output={
                "task_config": selected_config,
                "agent_context": agent_context or {},
                "selected_config_ready": selected_ready,
                "blockers": blockers,
                "readiness": readiness,
            },
            error=None if selected_ready else "robotwin2_assets_not_ready",
        )

    def _primitive_locate_robotwin2_actor(
        self,
        query: str | None = None,
        actor_name: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        self._refresh_actor_registry()
        candidates = self._actor_candidates(query=query, context=agent_context)
        selected = _select_candidate(candidates, explicit=actor_name)
        ok = selected is not None if actor_name is not None else bool(candidates)
        return PrimitiveResult(
            name="locate_robotwin2_actor",
            ok=ok,
            output={
                "query": query,
                "actor_name": actor_name,
                "agent_context": agent_context or {},
                "selected": selected,
                "candidates": candidates,
                "pose_world": selected.get("pose_world") if selected else None,
                "selection_performed": selected is not None,
                "source": "live_env_actor_registry" if self._env is not None else "unavailable_without_live_env",
            },
            error=None if ok else ("actor_not_found" if actor_name is not None else "live_actor_registry_unavailable"),
        )

    def _primitive_inspect_robotwin2_actor_points(
        self,
        actor_name: str,
        point_type: str = "functional",
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        actor = self._resolve_actor(actor_name)
        points = _extract_actor_points(actor, point_type)
        return PrimitiveResult(
            name="inspect_robotwin2_actor_points",
            ok=bool(points),
            output={
                "actor_name": actor_name,
                "point_type": point_type,
                "agent_context": agent_context or {},
                "pose_world": _actor_pose(actor),
                "points": points,
                "source": "live_env_actor_keypoint_api",
                "evidence_type": "simulator_oracle_geometry",
                "visual_evidence": False,
                "eligible_as_visual_provenance": False,
            },
            error=None if points else ("actor_not_found" if actor is None else "actor_points_unavailable"),
        )

    def _primitive_measure_robotwin2_actor_visual(
        self,
        segmentation_id: int | list[int] | None = None,
        actor_label: str | None = None,
        camera_name: str = "head_camera",
        depth_scale: float = 0.001,
        quaternion_wxyz: list[float] | None = None,
        max_points: int = 256,
        prompt: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result(
                "measure_robotwin2_actor_visual",
                camera_name=camera_name,
                actor_label=actor_label,
                segmentation_id=segmentation_id,
            )
        self._capture_live_observation()
        actor = self._resolve_actor_exact(actor_label) if actor_label is not None else None
        if actor_label is not None and actor is None:
            return PrimitiveResult(
                name="measure_robotwin2_actor_visual",
                ok=False,
                output={"actor_label": actor_label, "camera_name": camera_name},
                error="exact_actor_label_not_found",
            )
        actor_id = _actor_segmentation_id(actor) if actor is not None else None
        if segmentation_id is None:
            segmentation_id = actor_id
        elif actor_id is not None and _normalize_segmentation_selector(segmentation_id) != _normalize_segmentation_selector(actor_id):
            return PrimitiveResult(
                name="measure_robotwin2_actor_visual",
                ok=False,
                output={"actor_label": actor_label, "segmentation_id": segmentation_id, "actor_segmentation_id": actor_id},
                error="actor_label_segmentation_id_mismatch",
            )
        if segmentation_id is None:
            return PrimitiveResult(
                name="measure_robotwin2_actor_visual",
                ok=False,
                output={"actor_label": actor_label, "camera_name": camera_name},
                error="segmentation_id_or_resolvable_actor_label_required",
            )
        try:
            measurement = _measure_robotwin2_actor_geometry(
                self._last_obs,
                camera_name=camera_name,
                segmentation_id=segmentation_id,
                depth_scale=depth_scale,
                quaternion_wxyz=quaternion_wxyz,
                max_points=max_points,
            )
        except ValueError as exc:
            return PrimitiveResult(
                name="measure_robotwin2_actor_visual",
                ok=False,
                output={
                    "actor_label": actor_label,
                    "segmentation_id": _to_builtin(segmentation_id),
                    "camera_name": camera_name,
                },
                error=str(exc),
            )
        source_handles = self._register_visual_handles(
            prompt=prompt,
            query=actor_label,
            camera_name=camera_name,
            agent_context=agent_context or {},
        )
        handle = f"robotwin2:visual:{len(self._visual_handles)}"
        self._visual_handles[handle] = {
            "handle": handle,
            "source": "live_env.get_obs:actor_visual_measurement",
            "observation_round": self._observation_round,
            "camera_name": camera_name,
            "modality": "actor_visual_measurement",
            "actor_label": actor_label,
            "segmentation_id": _to_builtin(segmentation_id),
            "source_evidence_handles": source_handles,
            "summary": {key: value for key, value in measurement.items() if key != "world_points"},
            "prompt": prompt,
            "agent_context": _to_builtin(agent_context or {}),
            "raw_value": measurement,
        }
        artifact_id = f"robotwin2_actor_visual_measurement_{len(self.get_trace().artifacts)}"
        artifact = {
            **measurement,
            "actor_label": actor_label,
            "segmentation_id": _to_builtin(segmentation_id),
            "camera_name": camera_name,
            "evidence_handles": [handle],
            "source_evidence_handles": source_handles,
            "provenance_source": "live_env.get_obs",
            "observation_round": self._observation_round,
        }
        self.get_trace().add_artifact(artifact_id, artifact)
        return PrimitiveResult(
            name="measure_robotwin2_actor_visual",
            ok=True,
            output={**artifact, "artifacts": [artifact_id]},
            artifacts=[artifact_id],
        )

    def _primitive_inspect_robotwin2_arm_pose(
        self,
        arm_tag: str,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result("inspect_robotwin2_arm_pose", arm_tag=arm_tag, agent_context=agent_context)
        try:
            pose = self._current_arm_pose(arm_tag)
        except ValueError as exc:
            return PrimitiveResult(
                name="inspect_robotwin2_arm_pose",
                ok=False,
                output={"arm_tag": arm_tag, "agent_context": agent_context or {}},
                error=str(exc),
            )
        return PrimitiveResult(
            name="inspect_robotwin2_arm_pose",
            ok=pose is not None,
            output={
                "arm_tag": arm_tag,
                "pose_world": pose,
                "source": "official_env.get_arm_pose_or_observation_endpose",
                "agent_context": agent_context or {},
            },
            error=None if pose is not None else "robotwin2_arm_pose_unavailable",
        )

    def _primitive_probe_robotwin2_motion_plan(
        self,
        arm_tag: str,
        target_pose: list[float] | JsonDict | None = None,
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result("probe_robotwin2_motion_plan", arm_tag=arm_tag, agent_context=agent_context)
        if arm_tag not in {"left", "right"}:
            return PrimitiveResult(
                name="probe_robotwin2_motion_plan",
                ok=False,
                output={"arm_tag": arm_tag},
                error="arm_tag_must_be_left_or_right",
            )
        provenance: list[JsonDict] = []
        if target_pose is None:
            resolved_target = self._current_arm_pose(arm_tag)
            target_source = "current_arm_pose"
        else:
            provenance, provenance_error = self._consume_visual_handles(evidence_handles)
            if provenance_error:
                return PrimitiveResult(
                    name="probe_robotwin2_motion_plan",
                    ok=False,
                    output={"arm_tag": arm_tag, "evidence_handles": evidence_handles or []},
                    error=provenance_error,
                )
            try:
                resolved_target = self._resolve_visual_pose_reference(target_pose, evidence_handles)
            except (TypeError, ValueError) as exc:
                return PrimitiveResult(
                    name="probe_robotwin2_motion_plan",
                    ok=False,
                    output={"arm_tag": arm_tag},
                    error=str(exc),
                )
            target_source = "caller_target"
        if resolved_target is None:
            return PrimitiveResult(
                name="probe_robotwin2_motion_plan",
                ok=False,
                output={"arm_tag": arm_tag},
                error="robotwin2_arm_pose_unavailable",
            )
        robot = getattr(self._env, "robot", None)
        plan_path = getattr(robot, f"{arm_tag}_plan_path", None)
        if not callable(plan_path):
            return PrimitiveResult(
                name="probe_robotwin2_motion_plan",
                ok=False,
                output={"arm_tag": arm_tag, "target_pose": resolved_target},
                error="official_motion_plan_path_unavailable",
            )
        plan_success_before = getattr(self._env, "plan_success", None)
        try:
            raw_plan = plan_path(resolved_target)
        except BaseException as exc:
            raw_plan = {"status": "Exception", "exception": f"{type(exc).__name__}: {exc}"}
        finally:
            if hasattr(self._env, "plan_success"):
                self._env.plan_success = plan_success_before
        summary = _summarize_robotwin2_motion_plan(raw_plan)
        status = summary.get("status")
        ok = status == "Success"
        return PrimitiveResult(
            name="probe_robotwin2_motion_plan",
            ok=ok,
            output={
                "arm_tag": arm_tag,
                "current_pose": self._current_arm_pose(arm_tag),
                "target_pose": resolved_target,
                "target_source": target_source,
                "status": status,
                "path_summary": summary,
                "plan_success_before": _to_builtin(plan_success_before),
                "plan_success_after": _to_builtin(getattr(self._env, "plan_success", None)),
                "plan_success_unchanged": getattr(self._env, "plan_success", None) == plan_success_before,
                "grounding_provenance": provenance,
                "agent_context": agent_context or {},
            },
            error=None if ok else "robotwin2_motion_plan_probe_failed",
        )

    def _primitive_move_robotwin2_arm(
        self,
        arm_tag: str,
        target_pose: list[float] | JsonDict | None = None,
        delta: list[float] | None = None,
        move_axis: str = "world",
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result("move_robotwin2_arm", arm_tag=arm_tag, agent_context=agent_context)
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="move_robotwin2_arm",
                ok=False,
                output={"arm_tag": arm_tag, "evidence_handles": evidence_handles or []},
                error=provenance_error,
            )
        if target_pose is not None:
            try:
                resolved_target_pose = self._resolve_visual_pose_reference(target_pose, evidence_handles)
            except ValueError as exc:
                return PrimitiveResult(name="move_robotwin2_arm", ok=False, output={"arm_tag": arm_tag}, error=str(exc))
            action_seq = self._env.move_to_pose(arm_tag=arm_tag, target_pose=resolved_target_pose)
        elif delta is not None:
            resolved_target_pose = None
            padded = list(delta) + [0.0, 0.0, 0.0]
            action_seq = self._env.move_by_displacement(arm_tag=arm_tag, x=padded[0], y=padded[1], z=padded[2], move_axis=move_axis)
        else:
            return PrimitiveResult(name="move_robotwin2_arm", ok=False, output={"arm_tag": arm_tag}, error="target_pose_or_delta_required")
        executed = bool(self._env.move(action_seq))
        self._last_obs = _safe_call(self._env, "get_obs") or self._last_obs
        diagnostics = self._move_failure_diagnostics([arm_tag], executed)
        return PrimitiveResult(
            name="move_robotwin2_arm",
            ok=executed,
            output={
                "executed": executed,
                "arm_tag": arm_tag,
                "target_pose": resolved_target_pose,
                "target_pose_request": _to_builtin(target_pose),
                "delta": delta,
                "move_axis": move_axis,
                "requires_live_env": False,
                "agent_context": agent_context or {},
                "grounding_provenance": provenance,
                "diagnostic_evidence": diagnostics,
            },
            error=None if executed else "robotwin2_move_failed",
        )

    def _primitive_execute_robotwin2_actions(
        self,
        left: JsonDict | None = None,
        right: JsonDict | None = None,
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result(
                "execute_robotwin2_actions",
                left=left,
                right=right,
                agent_context=agent_context,
            )
        requested = {arm: spec for arm, spec in (("left", left), ("right", right)) if spec is not None}
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="execute_robotwin2_actions",
                ok=False,
                output={"actions": _to_builtin(requested), "evidence_handles": evidence_handles or []},
                error=provenance_error,
            )
        if not requested:
            return PrimitiveResult(
                name="execute_robotwin2_actions",
                ok=False,
                output={"actions": {}, "agent_context": agent_context or {}},
                error="left_or_right_action_required",
            )
        sequences: dict[str, Any] = {}
        normalized: JsonDict = {}
        try:
            for arm_tag, spec in requested.items():
                if not isinstance(spec, dict):
                    raise ValueError(f"{arm_tag}_action_must_be_an_object")
                sequence, action = self._build_action_sequence(arm_tag, spec, evidence_handles)
                sequences[arm_tag] = sequence
                normalized[arm_tag] = action
        except (TypeError, ValueError) as exc:
            return PrimitiveResult(
                name="execute_robotwin2_actions",
                ok=False,
                output={"actions": _to_builtin(requested), "agent_context": agent_context or {}},
                error=str(exc),
            )
        routed_arms = [
            arm_tag
            for arm_tag, sequence in sequences.items()
            if isinstance(sequence, dict) and sequence.get("kind") == "vertical_clearance_route"
        ]
        route_execution: JsonDict | None = None
        if routed_arms:
            if len(sequences) != 1:
                return PrimitiveResult(
                    name="execute_robotwin2_actions",
                    ok=False,
                    output={"actions": normalized, "agent_context": agent_context or {}},
                    error="vertical_clearance_route_requires_single_arm_action",
                )
            route_arm = routed_arms[0]
            executed, route_execution = self._execute_vertical_clearance_route(route_arm, sequences[route_arm])
        else:
            executed = bool(self._env.move(*sequences.values()))
        self._last_obs = _safe_call(self._env, "get_obs") or self._last_obs
        diagnostics = self._move_failure_diagnostics(sorted(normalized), executed)
        if diagnostics is not None and route_execution is not None:
            diagnostics["route_execution"] = route_execution
        return PrimitiveResult(
            name="execute_robotwin2_actions",
            ok=executed,
            output={
                "executed": executed,
                "arms": sorted(normalized),
                "simultaneous": len(normalized) == 2,
                "actions": normalized,
                "agent_context": agent_context or {},
                "grounding_provenance": provenance,
                "post_arm_schema": self._arm_schema(),
                "route_execution": route_execution,
                "diagnostic_evidence": diagnostics,
            },
            error=None if executed else "robotwin2_move_failed",
        )

    def _primitive_execute_robotwin2_pick_place(
        self,
        source_actor_label: str,
        target_actor_label: str,
        arm_tag: str,
        evidence_handles: list[str] | None = None,
        pre_grasp_distance: float = 0.1,
        lift_distance: float = 0.08,
        pre_place_distance: float = 0.05,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        """Compile semantic pick/place into upstream generic manipulation helpers.

        The caller selects both actors and the arm.  Fresh visual evidence is
        mandatory, while task success remains available only through verify().
        No expert episode or task play_once method is invoked.
        """

        if self._env is None:
            return _live_required_result(
                "execute_robotwin2_pick_place",
                source_actor_label=source_actor_label,
                target_actor_label=target_actor_label,
                arm_tag=arm_tag,
                agent_context=agent_context,
            )
        if arm_tag not in {"left", "right"}:
            return PrimitiveResult(
                name="execute_robotwin2_pick_place",
                ok=False,
                output={"executed": False, "arm_tag": arm_tag},
                error="robotwin2_arm_must_be_left_or_right",
            )
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error is not None:
            return PrimitiveResult(
                name="execute_robotwin2_pick_place",
                ok=False,
                output={"executed": False, "arm_tag": arm_tag},
                error=provenance_error,
            )
        grounded_labels = {
            str(item.get("actor_label"))
            for item in provenance
            if item.get("actor_label") is not None
        }
        missing_grounding = [
            label
            for label in (source_actor_label, target_actor_label)
            if label not in grounded_labels
        ]
        if missing_grounding:
            return PrimitiveResult(
                name="execute_robotwin2_pick_place",
                ok=False,
                output={"executed": False, "arm_tag": arm_tag},
                error=f"robotwin2_pick_place_visual_grounding_missing:{','.join(missing_grounding)}",
            )
        source_actor = self._resolve_actor_exact(source_actor_label)
        target_actor = self._resolve_actor_exact(target_actor_label)
        if source_actor is None or target_actor is None:
            missing = source_actor_label if source_actor is None else target_actor_label
            return PrimitiveResult(
                name="execute_robotwin2_pick_place",
                ok=False,
                output={"executed": False, "arm_tag": arm_tag},
                error=f"exact_actor_label_not_found:{missing}",
            )

        pre_grasp_distance = max(0.02, min(0.25, float(pre_grasp_distance)))
        lift_distance = max(0.02, min(0.25, float(lift_distance)))
        pre_place_distance = max(0.01, min(0.25, float(pre_place_distance)))
        completed_stages: list[JsonDict] = []

        def execute_stage(phase: str, sequence: Any) -> bool:
            executed = bool(self._env.move(sequence))
            completed_stages.append({"phase": phase, "ok": executed})
            self._capture_live_observation()
            return executed

        # A partially closed preshape keeps thin vessels between the fingers
        # while the official generic grasp planner selects a collision-free pose.
        if not execute_stage("preshape", self._env.close_gripper(arm_tag, pos=0.6)):
            failed_phase = "preshape"
        elif not execute_stage(
            "grasp",
            self._env.grasp_actor(
                source_actor,
                arm_tag,
                pre_grasp_dis=pre_grasp_distance,
                gripper_pos=0.0,
            ),
        ):
            failed_phase = "grasp"
        elif not execute_stage(
            "lift",
            self._env.move_by_displacement(
                arm_tag,
                z=lift_distance,
                move_axis="arm",
            ),
        ):
            failed_phase = "lift"
        else:
            try:
                target_pose = target_actor.get_functional_point(0, "pose")
            except TypeError:
                target_pose = target_actor.get_functional_point(0)
            if target_pose is None:
                return PrimitiveResult(
                    name="execute_robotwin2_pick_place",
                    ok=False,
                    output={
                        "executed": False,
                        "arm_tag": arm_tag,
                        "completed_stages": completed_stages,
                    },
                    error="robotwin2_target_functional_point_unavailable",
                )
            # Upstream uses the actor pose when no source functional point exists.
            source_point = source_actor.get_functional_point(0)
            if not execute_stage(
                "place",
                self._env.place_actor(
                    source_actor,
                    arm_tag,
                    target_pose=target_pose,
                    functional_point_id=0 if source_point is not None else None,
                    pre_dis=pre_place_distance,
                ),
            ):
                failed_phase = "place"
            elif not execute_stage(
                "retreat",
                self._env.move_by_displacement(
                    arm_tag,
                    z=0.05,
                    move_axis="arm",
                ),
            ):
                failed_phase = "retreat"
            else:
                failed_phase = None

        executed = failed_phase is None
        return PrimitiveResult(
            name="execute_robotwin2_pick_place",
            ok=executed,
            output={
                "executed": executed,
                "arm_tag": arm_tag,
                "source_actor_label": source_actor_label,
                "target_actor_label": target_actor_label,
                "completed_stages": completed_stages,
                "grounding_provenance": provenance,
                "uses_private_success": False,
                "agent_context": agent_context or {},
            },
            error=None if executed else f"robotwin2_pick_place_failed:{failed_phase}",
        )

    def _build_action_sequence(
        self,
        arm_tag: str,
        spec: JsonDict,
        evidence_handles: list[str] | None = None,
    ) -> tuple[Any, JsonDict]:
        operation = str(spec.get("operation", "")).strip().lower()
        supported = {
            "move_to_pose",
            "move_by_displacement",
            "open_gripper",
            "close_gripper",
            "back_to_origin",
        }
        if operation not in supported:
            raise ValueError(f"unsupported_robotwin2_action:{operation or 'missing'}")
        action = {**_to_builtin(spec), "operation": operation}
        if operation == "move_to_pose":
            target_pose = spec.get("target_pose")
            if target_pose is None:
                raise ValueError("target_pose_required")
            resolved = self._resolve_visual_pose_reference(target_pose, evidence_handles)
            action["target_pose_request"] = _to_builtin(target_pose)
            action["target_pose"] = resolved
            route = spec.get("route")
            if route is not None:
                sequence, route_evidence = self._build_vertical_clearance_route(arm_tag, resolved, route)
                action["route"] = route_evidence
                return sequence, action
            return self._env.move_to_pose(arm_tag=arm_tag, target_pose=resolved), action
        if operation == "move_by_displacement":
            delta = list(spec.get("delta") or [])
            if not delta:
                raise ValueError("delta_required")
            delta = delta + [0.0, 0.0, 0.0]
            action["delta"] = delta[:3]
            return self._env.move_by_displacement(
                arm_tag=arm_tag,
                x=delta[0],
                y=delta[1],
                z=delta[2],
                quat=spec.get("quat"),
                move_axis=str(spec.get("move_axis", "world")),
            ), action
        if operation in {"open_gripper", "close_gripper"}:
            default_pos = 1.0 if operation == "open_gripper" else 0.0
            pos = float(spec.get("pos", default_pos))
            action["pos"] = pos
            method = self._env.open_gripper if operation == "open_gripper" else self._env.close_gripper
            return method(arm_tag=arm_tag, pos=pos), action
        if operation == "back_to_origin":
            return self._env.back_to_origin(arm_tag=arm_tag), action

        raise ValueError(f"unsupported_robotwin2_action:{operation}")

    def _build_vertical_clearance_route(
        self,
        arm_tag: str,
        target_pose: list[float],
        route: Any,
    ) -> tuple[Any, JsonDict]:
        if not isinstance(route, dict):
            raise ValueError("move_route_must_be_an_object")
        strategy = str(route.get("strategy", "")).strip().lower()
        if strategy != "vertical_clearance":
            raise ValueError(f"unsupported_robotwin2_move_route:{strategy or 'missing'}")
        clearance = float(route.get("clearance", 0.1))
        if not np.isfinite(clearance) or clearance <= 0.0 or clearance > 0.5:
            raise ValueError("vertical_clearance_must_be_in_range_0_to_0.5")
        max_upward_margin = float(route.get("max_upward_margin", 0.04))
        if not np.isfinite(max_upward_margin) or max_upward_margin <= 0.0 or max_upward_margin > 0.25:
            raise ValueError("vertical_clearance_max_upward_margin_must_be_in_range_0_to_0.25")
        recovery_xy_step = float(route.get("recovery_xy_step", 0.08))
        if not np.isfinite(recovery_xy_step) or recovery_xy_step <= 0.0 or recovery_xy_step > 0.25:
            raise ValueError("vertical_clearance_recovery_xy_step_must_be_in_range_0_to_0.25")
        current_pose = self._current_arm_pose(arm_tag)
        if current_pose is None or len(current_pose) != 7:
            raise ValueError("vertical_clearance_requires_current_arm_pose")

        route_floor_z = max(float(current_pose[2]), float(target_pose[2]))
        requested_clearance_z = route_floor_z + clearance
        local_reachable_ceiling_z = float(current_pose[2]) + max_upward_margin
        workspace = self._public_arm_workspace_z_bounds(arm_tag)
        ceiling_candidates = [local_reachable_ceiling_z]
        if workspace is not None:
            ceiling_candidates.append(float(workspace["z_max"]))
        effective_ceiling_z = max(route_floor_z, min(ceiling_candidates))
        clearance_z = min(requested_clearance_z, effective_ceiling_z)
        waypoints = [
            [*map(float, current_pose[:2]), clearance_z, *map(float, current_pose[3:7])],
            [*map(float, target_pose[:2]), clearance_z, *map(float, target_pose[3:7])],
            [float(value) for value in target_pose],
        ]
        if not callable(getattr(self._env, "move_to_pose", None)):
            raise ValueError("official_move_to_pose_unavailable")
        return {
            "kind": "vertical_clearance_route",
            "waypoints": waypoints,
            "recovery_xy_step": recovery_xy_step,
        }, {
            "strategy": strategy,
            "requested_clearance": clearance,
            "max_upward_margin": max_upward_margin,
            "recovery_xy_step": recovery_xy_step,
            "requested_clearance_z": requested_clearance_z,
            "applied_clearance_z": clearance_z,
            "clearance_clamped": clearance_z < requested_clearance_z,
            "local_reachable_ceiling_z": local_reachable_ceiling_z,
            "public_workspace": workspace,
            "pose_source": "official_current_arm_pose_plus_visual_target",
            "waypoints": waypoints,
        }

    def _public_arm_workspace_z_bounds(self, arm_tag: str) -> JsonDict | None:
        owners = (("official_env", self._env), ("official_env.robot", getattr(self._env, "robot", None)))
        for owner_name, owner in owners:
            if owner is None:
                continue
            get_workspace = getattr(owner, "get_arm_workspace", None)
            if callable(get_workspace):
                try:
                    raw = get_workspace(arm_tag=arm_tag)
                except TypeError:
                    raw = get_workspace(arm_tag)
                bounds = _robotwin2_workspace_z_bounds(raw, arm_tag)
                if bounds is not None:
                    return {**bounds, "source": f"{owner_name}.get_arm_workspace"}
            for attribute in (f"{arm_tag}_arm_workspace", f"{arm_tag}_workspace"):
                raw = getattr(owner, attribute, None)
                bounds = _robotwin2_workspace_z_bounds(raw, arm_tag)
                if bounds is not None:
                    return {**bounds, "source": f"{owner_name}.{attribute}"}
        return None

    def _execute_vertical_clearance_route(self, arm_tag: str, route_sequence: JsonDict) -> tuple[bool, JsonDict]:
        waypoints = route_sequence.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError("invalid_vertical_clearance_route_sequence")

        def execute_waypoint(index: int, waypoint: list[float], phase: str) -> JsonDict:
            self._capture_live_observation()
            sequence = self._env.move_to_pose(arm_tag=arm_tag, target_pose=waypoint)
            if not isinstance(sequence, tuple) or len(sequence) != 2 or sequence[0] != arm_tag:
                raise ValueError("official_move_to_pose_returned_invalid_sequence")
            pose_before = self._current_arm_pose(arm_tag)
            plan_success_before = getattr(self._env, "plan_success", None)
            executed = bool(self._env.move(sequence))
            plan_success_at_result = getattr(self._env, "plan_success", None)
            pose_after = self._current_arm_pose(arm_tag)
            history = getattr(self._env, f"{arm_tag}_joint_path", None)
            latest_path = history[-1] if isinstance(history, list) and history else None
            attempt: JsonDict = {
                "index": index,
                "phase": phase,
                "observation_round": self._observation_round,
                "target_pose": _to_builtin(waypoint),
                "pose_before": pose_before,
                "pose_after": pose_after,
                "executed": executed,
                "plan_success_before": _to_builtin(plan_success_before),
                "plan_success_at_result": _to_builtin(plan_success_at_result),
                "latest_official_path": _summarize_robotwin2_motion_plan(latest_path),
                "planner_latch_restored": False,
            }
            if not executed:
                if hasattr(self._env, "plan_success") and isinstance(plan_success_before, bool):
                    self._env.plan_success = plan_success_before
                    attempt["planner_latch_restored"] = getattr(self._env, "plan_success", None) == plan_success_before
            return attempt

        attempts: list[JsonDict] = []
        for index, waypoint in enumerate(waypoints):
            attempt = execute_waypoint(index, waypoint, "primary")
            attempts.append(attempt)
            if not attempt["executed"]:
                recovery: JsonDict = {
                    "triggered": index == 0 and attempt["planner_latch_restored"],
                    "strategy": "current_height_transit",
                    "attempts": [],
                }
                if recovery["triggered"]:
                    pose_before = attempt["pose_before"]
                    target_pose = [float(value) for value in waypoints[-1]]
                    recovery_xy_step = float(route_sequence.get("recovery_xy_step", 0.08))
                    if not np.isfinite(recovery_xy_step) or recovery_xy_step <= 0.0 or recovery_xy_step > 0.25:
                        raise ValueError("vertical_clearance_recovery_xy_step_must_be_in_range_0_to_0.25")
                    if not isinstance(pose_before, list) or len(pose_before) != 7:
                        recovery["failure"] = "official_current_arm_pose_unavailable"
                        return False, {
                            "strategy": "vertical_clearance",
                            "completed_waypoints": index,
                            "failed_waypoint_index": index,
                            "recovered": False,
                            "attempts": attempts,
                            "recovery": recovery,
                        }
                    xy_distance = float(np.linalg.norm(np.asarray(target_pose[:2]) - np.asarray(pose_before[:2])))
                    xy_tolerance = min(0.01, recovery_xy_step * 0.25)
                    min_xy_step = min(0.005, recovery_xy_step * 0.5)
                    active_xy_step = recovery_xy_step
                    max_replans = max(2, int(np.ceil(xy_distance / min_xy_step)) + 4)
                    recovery_waypoints: list[list[float]] = []
                    recovery.update(
                        {
                            "strategy": "adaptive_current_pose_transit",
                            "xy_distance": xy_distance,
                            "max_xy_step": recovery_xy_step,
                            "min_xy_step": min_xy_step,
                            "active_xy_step": active_xy_step,
                            "xy_tolerance": xy_tolerance,
                            "max_replans": max_replans,
                            "horizontal_steps": 0,
                            "successful_horizontal_steps": 0,
                            "step_backoffs": [],
                            "replan_source": "official_current_arm_pose_plus_visual_target",
                            "orientation_policy": "hold_latest_official_arm_orientation_during_xy_transit",
                        }
                    )
                    recovery["waypoints"] = recovery_waypoints
                    for recovery_index in range(max_replans):
                        current_pose = self._current_arm_pose(arm_tag)
                        if current_pose is None or len(current_pose) != 7:
                            recovery["failure"] = "official_current_arm_pose_unavailable"
                            break
                        remaining_xy = np.asarray(target_pose[:2]) - np.asarray(current_pose[:2])
                        remaining_distance = float(np.linalg.norm(remaining_xy))
                        recovery["remaining_xy_distance"] = remaining_distance
                        if remaining_distance <= xy_tolerance:
                            break
                        required_steps = max(1, int(np.ceil(remaining_distance / active_xy_step)))
                        next_xy = np.asarray(current_pose[:2]) + remaining_xy / required_steps
                        recovery_waypoint = [
                            float(next_xy[0]),
                            float(next_xy[1]),
                            float(current_pose[2]),
                            *[float(value) for value in current_pose[3:7]],
                        ]
                        recovery_waypoints.append(recovery_waypoint)
                        recovery_attempt = execute_waypoint(recovery_index, recovery_waypoint, "recovery")
                        recovery["attempts"].append(recovery_attempt)
                        recovery["horizontal_steps"] += 1
                        if not recovery_attempt["executed"]:
                            can_backoff = bool(recovery_attempt["planner_latch_restored"]) and active_xy_step > (
                                min_xy_step + 1e-9
                            )
                            if not can_backoff:
                                recovery["failure"] = (
                                    "adaptive_min_xy_step_failed"
                                    if recovery_attempt["planner_latch_restored"]
                                    else "official_planner_latch_not_restored"
                                )
                                return False, {
                                    "strategy": "vertical_clearance",
                                    "completed_waypoints": index,
                                    "failed_waypoint_index": index,
                                    "recovered": False,
                                    "attempts": attempts,
                                    "recovery": recovery,
                                }
                            previous_xy_step = active_xy_step
                            if active_xy_step > 0.01 + 1e-9:
                                active_xy_step = max(0.01, active_xy_step * 0.5)
                            else:
                                active_xy_step = max(min_xy_step, active_xy_step * 0.5)
                            recovery["active_xy_step"] = active_xy_step
                            recovery["step_backoffs"].append(
                                {
                                    "attempt_index": recovery_index,
                                    "from_xy_step": previous_xy_step,
                                    "to_xy_step": active_xy_step,
                                    "pose_source": "official_current_arm_pose_after_failed_move",
                                }
                            )
                            continue
                        recovery["successful_horizontal_steps"] += 1
                    current_pose = self._current_arm_pose(arm_tag)
                    remaining_distance = None
                    if current_pose is not None and len(current_pose) == 7:
                        remaining_distance = float(
                            np.linalg.norm(np.asarray(target_pose[:2]) - np.asarray(current_pose[:2]))
                        )
                    recovery["remaining_xy_distance"] = remaining_distance
                    if remaining_distance is None or remaining_distance > xy_tolerance:
                        recovery.setdefault(
                            "failure",
                            "official_current_arm_pose_unavailable"
                            if remaining_distance is None
                            else "adaptive_recovery_exhausted",
                        )
                        return False, {
                            "strategy": "vertical_clearance",
                            "completed_waypoints": index,
                            "failed_waypoint_index": index,
                            "recovered": False,
                            "attempts": attempts,
                            "recovery": recovery,
                        }
                    recovery_waypoints.append(target_pose)
                    final_attempt = execute_waypoint(len(recovery["attempts"]), target_pose, "recovery")
                    recovery["attempts"].append(final_attempt)
                    if not final_attempt["executed"]:
                        return False, {
                            "strategy": "vertical_clearance",
                            "completed_waypoints": index,
                            "failed_waypoint_index": index,
                            "recovered": False,
                            "attempts": attempts,
                            "recovery": recovery,
                        }
                    return True, {
                        "strategy": "vertical_clearance",
                        "completed_waypoints": recovery["successful_horizontal_steps"] + 1,
                        "failed_waypoint_index": index,
                        "recovered": True,
                        "attempts": attempts,
                        "recovery": recovery,
                    }
                return False, {
                    "strategy": "vertical_clearance",
                    "completed_waypoints": index,
                    "failed_waypoint_index": index,
                    "recovered": False,
                    "attempts": attempts,
                    "recovery": recovery,
                }
        return True, {
            "strategy": "vertical_clearance",
            "completed_waypoints": len(attempts),
            "failed_waypoint_index": None,
            "recovered": False,
            "attempts": attempts,
            "recovery": {"triggered": False, "strategy": "current_height_transit", "attempts": []},
        }

    def _resolve_visual_pose_reference(
        self,
        value: list[float] | JsonDict,
        evidence_handles: list[str] | None,
    ) -> list[float]:
        if not isinstance(value, dict):
            pose = [float(item) for item in value]
            if len(pose) != 7:
                raise ValueError("target_pose_must_have_7_values")
            return pose
        handle = str(value.get("evidence_handle") or "")
        if not handle:
            raise ValueError("target_pose_evidence_handle_required")
        if handle not in {str(item) for item in evidence_handles or []}:
            raise ValueError("target_pose_evidence_handle_not_supplied")
        record = self._visual_handles.get(handle)
        if record is None or record.get("modality") != "actor_visual_measurement":
            raise ValueError("target_pose_requires_actor_visual_measurement_handle")
        measurement = record.get("raw_value")
        if not isinstance(measurement, dict):
            raise ValueError("actor_visual_measurement_unavailable")
        field = str(value.get("field", "pose_world"))
        if field == "pose_world":
            base = list(measurement.get("pose_world") or [])
        elif field == "centroid_world":
            base = list(measurement.get("centroid_world") or []) + list(measurement.get("pose_world", [])[3:7])
        else:
            raise ValueError(f"unsupported_actor_visual_pose_field:{field}")
        if len(base) != 7:
            raise ValueError("actor_visual_pose_unavailable")
        offset = list(value.get("offset") or [0.0, 0.0, 0.0])
        offset += [0.0, 0.0, 0.0]
        base[:3] = [float(base[index]) + float(offset[index]) for index in range(3)]
        quaternion = value.get("quaternion_wxyz")
        if quaternion is not None:
            quaternion = [float(item) for item in quaternion]
            if len(quaternion) != 4:
                raise ValueError("quaternion_wxyz_must_have_4_values")
            base[3:7] = quaternion
        return [float(item) for item in base]

    def _primitive_set_robotwin2_gripper(
        self,
        arm_tag: str,
        command: str = "close",
        pos: float | None = None,
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result("set_robotwin2_gripper", arm_tag=arm_tag, command=command, agent_context=agent_context)
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="set_robotwin2_gripper",
                ok=False,
                output={"arm_tag": arm_tag, "command": command, "evidence_handles": evidence_handles or []},
                error=provenance_error,
            )
        normalized = command.lower().strip()
        if normalized not in {"open", "close"}:
            return PrimitiveResult(
                name="set_robotwin2_gripper",
                ok=False,
                output={"arm_tag": arm_tag, "command": command, "agent_context": agent_context or {}},
                error="unsupported_gripper_command",
            )
        if normalized == "open":
            action_seq = self._env.open_gripper(arm_tag=arm_tag, pos=1.0 if pos is None else pos)
        else:
            action_seq = self._env.close_gripper(arm_tag=arm_tag, pos=0.0 if pos is None else pos)
        executed = bool(self._env.move(action_seq))
        self._last_obs = _safe_call(self._env, "get_obs") or self._last_obs
        diagnostics = self._move_failure_diagnostics([arm_tag], executed)
        return PrimitiveResult(
            name="set_robotwin2_gripper",
            ok=executed,
            output={
                "executed": executed,
                "arm_tag": arm_tag,
                "command": normalized,
                "pos": 1.0 if normalized == "open" and pos is None else 0.0 if pos is None else pos,
                "requires_live_env": False,
                "agent_context": agent_context or {},
                "grounding_provenance": provenance,
                "diagnostic_evidence": diagnostics,
            },
            error=None if executed else "robotwin2_move_failed",
        )

    def _primitive_submit_robotwin2_ee_action(
        self,
        action: list[float],
        action_type: str | None = None,
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None:
            return _live_required_result("submit_robotwin2_ee_action", action_type=action_type or self.config.action_type, agent_context=agent_context)
        provenance, provenance_error = self._consume_visual_handles(evidence_handles)
        if provenance_error:
            return PrimitiveResult(
                name="submit_robotwin2_ee_action",
                ok=False,
                output={"action_type": action_type or self.config.action_type, "evidence_handles": evidence_handles or []},
                error=provenance_error,
            )
        selected_type = action_type or self.config.action_type
        action_count_before = getattr(self._env, "take_action_cnt", None)
        result = self._env.take_action(action, action_type=selected_type)
        action_count_after = getattr(self._env, "take_action_cnt", None)
        step_advanced = (
            isinstance(action_count_before, (int, float))
            and isinstance(action_count_after, (int, float))
            and action_count_after > action_count_before
        )
        submitted = _robotwin2_take_action_ok(result) if result is not None else step_advanced
        self._last_obs = _safe_call(self._env, "get_obs") or self._last_obs
        return PrimitiveResult(
            name="submit_robotwin2_ee_action",
            ok=submitted,
            output={
                "submitted": submitted,
                "action_type": selected_type,
                "action_length": len(action),
                "raw_result": _to_builtin(result),
                "take_action_count_before": _to_builtin(action_count_before),
                "take_action_count_after": _to_builtin(action_count_after),
                "step_advanced": step_advanced,
                "requires_live_env": False,
                "agent_context": agent_context or {},
                "grounding_provenance": provenance,
            },
            error=None if submitted else "robotwin2_take_action_failed",
        )

    def _primitive_record_robotwin2_evidence(
        self,
        key: str,
        value: Any,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"robotwin2_agent_evidence_{len(self.get_trace().artifacts)}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value), "agent_context": agent_context or {}})
        return PrimitiveResult(name="record_robotwin2_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _register_visual_handles(
        self,
        *,
        prompt: str | None,
        query: str | None,
        camera_name: str | None,
        agent_context: JsonDict,
    ) -> list[str]:
        handles: list[str] = []
        for source_name, modality, value in _robotwin2_visual_arrays(self._last_obs, camera_name=camera_name):
            handle = f"robotwin2:visual:{len(self._visual_handles)}"
            self._visual_handles[handle] = {
                "handle": handle,
                "source": "live_env.get_obs",
                "observation_round": self._observation_round,
                "camera_name": source_name,
                "modality": modality,
                "summary": summarize_observation(value),
                "prompt": prompt,
                "query": query,
                "agent_context": _to_builtin(agent_context),
                "raw_value": value,
            }
            handles.append(handle)
        return handles

    def _capture_live_observation(self) -> Any:
        if self._env is None:
            return self._last_obs
        latest = _safe_call(self._env, "get_obs")
        if latest is not None:
            self._last_obs = latest
            self._observation_round += 1
        return self._last_obs

    def _consume_visual_handles(self, handles: list[str] | None) -> tuple[list[JsonDict], str | None]:
        requested = [str(handle) for handle in handles or []]
        if not requested:
            return [], "visual_evidence_handles_required"
        missing = [handle for handle in requested if handle not in self._visual_handles]
        if missing:
            return [], f"unknown_visual_evidence_handles:{','.join(missing)}"
        provenance = [
            {key: value for key, value in self._visual_handles[handle].items() if key != "raw_value"}
            for handle in requested
        ]
        return provenance, None

    def _make_env(self, config: RoboTwin2RuntimeConfig, seed: int | None = None) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.task_name, config.to_dict())
        repo = Path(config.repo_path).expanduser().resolve()
        if not repo.exists():
            raise FileNotFoundError(f"RoboTwin2 repo not found: {repo}")
        missing_modules = [
            module for module in ROBOTWIN2_REQUIRED_LIVE_MODULES
            if importlib.util.find_spec(module) is None
        ]
        if missing_modules:
            repo_python = _repo_local_robotwin2_python(repo)
            hint = (
                f" Repo-local runtime Python detected at {repo_python}; align the OpenHands/Docker "
                "interpreter with that environment or install the same dependencies into the current interpreter."
                if repo_python is not None
                else ""
            )
            required = "`, `".join(ROBOTWIN2_REQUIRED_LIVE_MODULES)
            missing = "`, `".join(missing_modules)
            raise RuntimeError(
                f"RoboTwin2 live runtime requires `{required}` in the current Python interpreter. "
                f"Missing: `{missing}`.{hint}"
            )
        args = build_robotwin2_env_args(config, seed=seed)
        assets = _resolve_robotwin2_assets(repo, config.assets_path)
        with _repo_import_context(repo, assets_path=assets):
            _reload_stale_robotwin2_curobo_planner(repo)
            try:
                env_module = importlib.import_module(f"envs.{config.task_name}")
            except ImportError as exc:
                if _is_robotwin2_curobo_planner_import_error(exc):
                    raise RoboTwin2CuroboPlannerUnavailable(
                        _robotwin2_curobo_planner_unavailable_message(repo, exc)
                    ) from exc
                raise
            env_class = getattr(env_module, config.task_name)
            env = env_class()
            env.setup_demo(**args)
            if self._selected_native_episode is not None:
                _restore_native_verifier_context(env, env_module, config.task_name, self._selected_native_episode["episode_info"])
                source = repo / "description/utils/generate_episode_instructions.py"
                spec = importlib.util.spec_from_file_location("_robotwin2_native_instructions", source)
                if spec is None or spec.loader is None:
                    raise RuntimeError("Original RoboTwin2 instruction generator unavailable")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                descriptions = module.generate_episode_descriptions(config.task_name,
                    [deepcopy(self._selected_native_episode["episode_info"]["info"])], 100)
                instruction = str(np.random.choice(descriptions[0]["unseen"]))
                env.set_instruction(instruction=instruction)
            return env

    def _refresh_actor_registry(self) -> None:
        if self._env is None:
            return
        actors: dict[str, Any] = {}
        for name, value in vars(self._env).items():
            if name.startswith("_"):
                continue
            if _looks_like_actor(value):
                actors[name] = value
                actor_name = _actor_name(value)
                if actor_name and actor_name not in actors:
                    actors[actor_name] = value
        self._actors = actors

    def _actor_candidates(self, query: str | None = None, context: JsonDict | None = None) -> list[JsonDict]:
        del query, context
        self._refresh_actor_registry()
        names = sorted(self._actors)
        candidates = []
        for name in names:
            actor = self._actors.get(name)
            candidates.append(
                {
                    "name": name,
                    "actor_name": _actor_name(actor) if actor is not None else None,
                    "pose_world": _actor_pose(actor) if actor is not None else None,
                    "source": "live_env_actor_registry",
                }
            )
        return sorted(candidates, key=lambda item: item["name"])

    def _resolve_actor(self, name: str | None) -> Any | None:
        if not name:
            return None
        self._refresh_actor_registry()
        if name in self._actors:
            return self._actors[name]
        lowered = name.lower()
        for candidate_name, actor in self._actors.items():
            if lowered == candidate_name.lower():
                return actor
            actor_name = _actor_name(actor)
            if actor_name and lowered == actor_name.lower():
                return actor
        return None

    def _resolve_actor_exact(self, name: str | None) -> Any | None:
        if not name:
            return None
        self._refresh_actor_registry()
        if name in self._actors:
            return self._actors[name]
        matches = [actor for actor in self._actors.values() if _actor_name(actor) == name]
        unique = {id(actor): actor for actor in matches}
        return next(iter(unique.values())) if len(unique) == 1 else None

    def _actor_evidence(self) -> JsonDict:
        self._refresh_actor_registry()
        return {
            name: {
                "pose_world": _actor_pose(actor),
                "actor_name": _actor_name(actor),
                "source": "live_env_actor_registry",
            }
            for name, actor in sorted(self._actors.items())
        }

    def _camera_schema(self) -> JsonDict:
        visual = robotwin2_visual_evidence(self._last_obs)
        return {
            "frame": "camera-specific pixels with upstream calibration to world",
            "available_cameras": sorted(visual["cameras"]),
            "modalities_by_camera": visual["modalities_by_camera"],
            "calibration_fields": ["intrinsic_cv", "extrinsic_cv", "cam2world_gl"],
            "cameras": visual["cameras"],
        }

    def _object_schema(self) -> JsonDict:
        return {
            "selection": "caller_selected_registry_or_native_actor_name",
            "pose_format": "[x,y,z,qw,qx,qy,qz] in world frame",
            "visual_status": "actor registry poses are simulator geometry, not visual evidence",
            "objects": self._actor_candidates(),
        }

    def _arm_schema(self) -> JsonDict:
        endpose = self._last_obs.get("endpose", {}) if isinstance(self._last_obs, dict) else {}
        return {
            "selection": "caller_selected",
            "arm_tags": ["left", "right"],
            "pose_format": "[x,y,z,qw,qx,qy,qz] in world frame",
            "arms": {
                arm: {
                    "endpose": _to_builtin(endpose.get(f"{arm}_endpose")),
                    "gripper": _to_builtin(endpose.get(f"{arm}_gripper")),
                }
                for arm in ("left", "right")
            },
        }

    def _current_arm_pose(self, arm_tag: str) -> list[float] | None:
        if arm_tag not in {"left", "right"}:
            raise ValueError("arm_tag_must_be_left_or_right")
        get_arm_pose = getattr(self._env, "get_arm_pose", None)
        if callable(get_arm_pose):
            pose = get_arm_pose(arm_tag=arm_tag)
            converted = _to_builtin(pose)
            if isinstance(converted, list) and len(converted) == 7:
                return [float(item) for item in converted]
        endpose = self._last_obs.get("endpose", {}) if isinstance(self._last_obs, dict) else {}
        pose = _to_builtin(endpose.get(f"{arm_tag}_endpose")) if isinstance(endpose, dict) else None
        return [float(item) for item in pose] if isinstance(pose, list) and len(pose) == 7 else None

    def _move_failure_diagnostics(self, arms: list[str], executed: bool) -> JsonDict | None:
        if executed:
            return None
        path_history: JsonDict = {}
        for arm_tag in arms:
            history = getattr(self._env, f"{arm_tag}_joint_path", None)
            latest = history[-1] if isinstance(history, list) and history else None
            path_history[arm_tag] = _summarize_robotwin2_motion_plan(latest)
        return {
            "failure": "official_env.move_returned_false",
            "plan_success": _to_builtin(getattr(self._env, "plan_success", None)),
            "arms": {
                arm_tag: {
                    "current_pose": self._current_arm_pose(arm_tag),
                    "latest_official_path": path_history[arm_tag],
                }
                for arm_tag in arms
            },
        }

    def _action_schema(self) -> JsonDict:
        return {
            "official_action_types": {
                "qpos": "[left_arm_joints + left_gripper + right_arm_joints + right_gripper]",
                "ee": "[left_pose7 + left_gripper + right_pose7 + right_gripper]",
                "delta_ee": "[left_delta_pose7 + left_gripper + right_delta_pose7 + right_gripper]",
            },
            "default_action_type": self.config.action_type,
            "caller_selected_action_descriptor": {
                "operation": [
                    "move_to_pose",
                    "move_by_displacement",
                    "open_gripper",
                    "close_gripper",
                    "back_to_origin",
                ],
                "arm_binding": "left/right dictionary key supplied by caller",
                "execution": "one descriptor per selected arm; two descriptors execute synchronously through upstream move",
                "grounding": "caller target or actor-visual measurement reference plus issued evidence_handles",
                "move_route": {
                    "strategy": "vertical_clearance",
                    "clearance": "requested positive meters, at most 0.5",
                    "max_upward_margin": "local reachability clamp above current EE pose; default 0.04m",
                    "workspace_clamp": "uses a public arm workspace z bound when the official env publishes one",
                    "evidence": "official current arm pose plus visually grounded target pose",
                },
            },
            "dual_arm_primitives": [
                "probe_robotwin2_motion_plan",
                "execute_robotwin2_actions",
                "move_robotwin2_arm",
                "set_robotwin2_gripper",
                "submit_robotwin2_ee_action",
            ],
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
            capability_tags=["robotwin2", "bimanual", "manipulation", level],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=["reset_called"],
            side_effects=["trace_event"] if level in {"L1", "L2"} else ["trace_event", "may_step_live_simulator"],
            cost={"sim_steps": "depends_on_upstream_action_sequence" if level == "L3" else 0},
            failure_modes=["live_env_not_created", "actor_not_found", "motion_plan_failed"] if level == "L3" else ["actor_not_found"],
            abstraction_level=level,
            leakage_risk="low",
            description=description,
        )

    def _merged_config(self, config: JsonDict) -> RoboTwin2RuntimeConfig:
        base = self.config.to_dict()
        base.update(config)
        return RoboTwin2RuntimeConfig(**base)

    def _require_reset(self) -> None:
        if self._task_spec is None or self._trace is None:
            raise RuntimeError("RoboTwin2AgentRuntimeBackend.reset must be called before use")


def inspect_robotwin2_repo(repo_path: str | Path) -> JsonDict:
    repo = Path(repo_path).expanduser()
    env_dir = repo / "envs"
    task_config_dir = repo / "task_config"
    assets = repo / "assets"
    task_files = sorted(path.stem for path in env_dir.glob("*.py") if not path.name.startswith("_")) if env_dir.exists() else []
    config_files = sorted(path.stem for path in task_config_dir.glob("*.yml")) if task_config_dir.exists() else []
    asset_readiness = robotwin2_asset_readiness(repo)
    asset_state = {
        "assets_dir_exists": assets.exists(),
        "download_script_exists": (repo / "assets/_download.py").exists(),
        "zip_files": sorted(path.name for path in assets.glob("*.zip")) if assets.exists() else [],
        "extracted_ready": all((assets / name).exists() for name in ("objects", "embodiments", "background_texture")),
        "objects_dir_exists": (assets / "objects").exists(),
        "embodiments_dir_exists": (assets / "embodiments").exists(),
        "background_texture_dir_exists": (assets / "background_texture").exists(),
        "hf_dataset_repo": ROBOTWIN2_HF_DATASET_REPO,
    }
    return {
        "repo_path": str(repo),
        "repo_ready": (repo / "README.md").exists() and env_dir.exists() and task_config_dir.exists(),
        "head_commit": _git_head(repo),
        "task_count": len(task_files),
        "task_names_sample": task_files[:12],
        "has_50_task_surface": len(task_files) >= 45,
        "config_count": len(config_files),
        "config_names": config_files,
        "assets": asset_state,
        "install": {
            "python": "3.10",
            "simulator": "SAPIEN 3.0.0b1",
            "core_requirements": ["torch==2.4.1", "sapien==3.0.0b1", "mplib==0.2.1", "gymnasium==0.29.1", "open3d==0.18.0"],
            "needs_vulkan": True,
            "needs_curobo_for_full_collection": True,
        },
        "case_asset_readiness": {
            "demo_clean_minimal_ready": asset_state["objects_dir_exists"] and asset_state["embodiments_dir_exists"],
            "demo_randomized_full_ready": asset_state["extracted_ready"],
            "background_texture_required_for_demo_clean": False,
        },
        "asset_readiness": asset_readiness,
    }


def robotwin2_asset_readiness(
    repo_path: str | Path, *, assets_path: str | Path | None = None
) -> JsonDict:
    repo = Path(repo_path).expanduser()
    assets = _resolve_robotwin2_assets(repo, assets_path)
    required = {
        "demo_clean": ["objects", "embodiments"],
        "demo_randomized": ["objects", "embodiments", "background_texture"],
        "full_suite": ["objects", "embodiments", "background_texture"],
    }
    installed = {
        "objects": _asset_component_state(assets, "objects"),
        "embodiments": _asset_component_state(assets, "embodiments"),
        "background_texture": _asset_component_state(
            assets,
            "background_texture",
            expected_zip_size=ROBOTWIN2_BACKGROUND_TEXTURE_ZIP_SIZE,
            expected_sha256=ROBOTWIN2_BACKGROUND_TEXTURE_SHA256,
        ),
    }
    readiness = {
        key: all(installed[name]["extracted_ready"] for name in names)
        for key, names in required.items()
    }
    missing = {
        key: [name for name in names if not installed[name]["extracted_ready"]]
        for key, names in required.items()
    }
    blockers: list[JsonDict] = []
    if missing["demo_randomized"]:
        blockers.append(
            {
                "id": "robotwin2_demo_randomized_assets_missing",
                "missing_components": missing["demo_randomized"],
                "detail": "demo_randomized requires assets/background_texture in addition to objects and embodiments.",
            }
        )
    if missing["full_suite"]:
        blockers.append(
            {
                "id": "robotwin2_full_suite_assets_missing",
                "missing_components": missing["full_suite"],
                "detail": "Do not claim randomized/full-suite readiness until every required asset component is extracted.",
            }
        )
    return {
        "assets_dir": str(assets),
        "hf_dataset_repo": ROBOTWIN2_HF_DATASET_REPO,
        "download_command": "cd external/upstreams/robotwin2 && bash script/_download_assets.sh",
        "background_texture_hf_lfs": {
            "filename": "background_texture.zip",
            "size_bytes": ROBOTWIN2_BACKGROUND_TEXTURE_ZIP_SIZE,
            "sha256": ROBOTWIN2_BACKGROUND_TEXTURE_SHA256,
        },
        "hf_token_configured": any(os.environ.get(name) for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN")),
        "required_components": required,
        "installed_components": installed,
        "missing_components": missing,
        "readiness": readiness,
        "blockers": blockers,
    }


def build_robotwin2_env_args(config: RoboTwin2RuntimeConfig, seed: int | None = None) -> JsonDict:
    args = _load_task_config(config.repo_path, config.task_config)
    args["task_name"] = config.task_name
    args["task_config"] = config.task_config
    args["render_freq"] = config.render_freq
    args["save_data"] = False
    args["collect_data"] = False
    args["eval_mode"] = False
    args["need_plan"] = True
    args["seed"] = seed if seed is not None else args.get("seed", 0)
    if config.embodiment is not None:
        args["embodiment"] = config.embodiment
    args["data_type"] = {**dict(args.get("data_type", {})), **config.data_type}
    args.update(config.env_kwargs)
    repo = Path(config.repo_path).expanduser()
    assets = _resolve_robotwin2_assets(repo.resolve(), config.assets_path)
    embodiment_config = _load_yaml(repo / "task_config/_embodiment_config.yml")
    embodiment = args.get("embodiment", ["aloha-agilex"])

    def get_embodiment_file(name: str) -> str:
        entry = embodiment_config.get(name, {})
        robot_file = entry.get("file_path")
        if robot_file is None:
            raise FileNotFoundError(f"RoboTwin2 embodiment {name!r} is missing file_path")
        robot_path = Path(str(robot_file)).expanduser()
        if not robot_path.is_absolute():
            repo_robot_path = repo / robot_path
            asset_robot_path = assets.joinpath(*robot_path.parts[1:]) if robot_path.parts and robot_path.parts[0] == "assets" else None
            if asset_robot_path is not None and asset_robot_path.exists():
                robot_path = asset_robot_path
            else:
                robot_path = repo_robot_path
        return str(robot_path.resolve())

    if len(embodiment) == 1:
        args["left_robot_file"] = get_embodiment_file(str(embodiment[0]))
        args["right_robot_file"] = get_embodiment_file(str(embodiment[0]))
        args["dual_arm_embodied"] = True
        args["embodiment_name"] = str(embodiment[0])
    elif len(embodiment) == 3:
        args["left_robot_file"] = get_embodiment_file(str(embodiment[0]))
        args["right_robot_file"] = get_embodiment_file(str(embodiment[1]))
        args["embodiment_dis"] = embodiment[2]
        args["dual_arm_embodied"] = False
        args["embodiment_name"] = f"{embodiment[0]}+{embodiment[1]}"
    else:
        raise ValueError("RoboTwin2 embodiment must be [dual-arm] or [left, right, interval]")
    args["left_embodiment_config"] = _load_yaml(Path(args["left_robot_file"]) / "config.yml")
    args["right_embodiment_config"] = _load_yaml(Path(args["right_robot_file"]) / "config.yml")
    return args


def run_robotwin2_preflight_probe(
    repo_path: str | Path = DEFAULT_ROBOTWIN2_REPO,
    task_name: str = "place_empty_cup",
    task_config: str = "demo_clean",
    create_env: bool = False,
    run_case: bool = False,
    case_path: str | Path = DEFAULT_ROBOTWIN2_PUBLIC_CASE,
    output: str | Path | None = None,
    env_factory: EnvFactory | None = None,
) -> JsonDict:
    config = RoboTwin2RuntimeConfig(repo_path=str(repo_path), task_name=task_name, task_config=task_config, live=create_env)
    repo_report = inspect_robotwin2_repo(repo_path)
    asset_readiness = repo_report["asset_readiness"]
    report: JsonDict = {
        "benchmark": "robotwin2",
        "repo": repo_report,
        "asset_readiness": asset_readiness,
        "task_name": task_name,
        "task_config": task_config,
        "create_env_requested": create_env,
        "runtime_contract": {
            "agent_visible_primitives": [
                "observe_robotwin2_scene",
                "observe_robotwin2_visual",
                "check_robotwin2_asset_readiness",
                "locate_robotwin2_actor",
                "measure_robotwin2_actor_visual",
                "inspect_robotwin2_actor_points",
                "inspect_robotwin2_arm_pose",
                "probe_robotwin2_motion_plan",
                "execute_robotwin2_actions",
                "move_robotwin2_arm",
                "set_robotwin2_gripper",
                "submit_robotwin2_ee_action",
            ],
            "private_success_primitives": [],
            "task_recipe_primitives": [],
        },
    }
    if create_env and not _robotwin2_config_assets_ready(asset_readiness, task_config):
        report.update(
            {
                "ok": False,
                "preflight_guard": {
                    "id": "robotwin2_live_env_assets_not_ready",
                    "task_config": task_config,
                    "blockers": _robotwin2_config_asset_blockers(asset_readiness, task_config),
                },
                "error": "RoboTwin2 live env creation skipped because selected asset set is incomplete.",
            }
        )
        if output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            Path(output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return report
    backend = RoboTwin2AgentRuntimeBackend(config, env_factory=env_factory)
    try:
        task = backend.reset(f"robotwin2_preflight_{task_name}", seed=0)
        obs = backend.observe()
        report.update(
            {
                "ok": bool(repo_report["repo_ready"]) and ((not create_env) or bool(backend._env is not None)),
                "task": task.to_dict(),
                "observation": obs.to_dict(),
                "primitive_count": len(backend.list_primitives()),
            }
        )
        if run_case:
            case = _run_robotwin2_public_action_case(backend, case_path)
            verification = backend.verify("task").to_dict()
            report["case"] = case
            report["verification"] = verification
            report["ok"] = bool(report["ok"] and case["action_chain_ok"] and verification["ok"])
            if not report["ok"]:
                report["error"] = case.get("error") or "robotwin2_official_verifier_failed_after_actions"
        else:
            report["verification"] = backend.verify("task").to_dict()
    except Exception as exc:  # pragma: no cover - live dependency failures are environment-specific.
        report.update({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _run_robotwin2_public_action_case(
    backend: RoboTwin2AgentRuntimeBackend,
    case_path: str | Path,
) -> JsonDict:
    """Execute an evaluator-supplied low-level program through public primitives only."""
    path = Path(case_path).expanduser().resolve()
    program = json.loads(path.read_text(encoding="utf-8"))
    measurements = program.get("measurements")
    actions = program.get("actions")
    if not isinstance(measurements, list) or not isinstance(actions, list):
        raise ValueError("robotwin2_public_case_requires_measurements_and_actions_lists")

    case: JsonDict = {
        "case_path": str(path),
        "mode": "evaluator_supplied_public_primitives_only",
        "program_exposed_to_agent": False,
        "expert_trajectory_used": False,
        "hidden_oracle_used_as_agent_input": False,
        "forbidden_primitives_called": [],
        "measurements": [],
        "stage_observations": [],
        "actions": [],
        "evidence_bindings": {},
        "action_attempted": False,
        "action_chain_ok": False,
    }
    measurement_handles: dict[str, str] = {}
    measurement_requests: dict[str, JsonDict] = {}
    for index, request in enumerate(measurements):
        if not isinstance(request, dict):
            case["error"] = f"measurement_{index}_must_be_an_object"
            return case
        name = str(request.get("name") or "").strip()
        actor_label = str(request.get("actor_label") or "").strip()
        camera_name = str(request.get("camera_name") or "head_camera")
        if not name or not actor_label:
            case["error"] = f"measurement_{index}_requires_name_and_actor_label"
            return case
        measurement_requests[name] = request
        result = backend.call_primitive(
            "measure_robotwin2_actor_visual",
            actor_label=actor_label,
            camera_name=camera_name,
            prompt=request.get("prompt"),
            agent_context={"case_measurement": name, "source": "evaluator_public_case"},
        )
        evidence = _robotwin2_measurement_case_evidence(name, result)
        case["measurements"].append(evidence)
        if not result.ok:
            case["error"] = result.error or f"measurement_{name}_failed"
            return case
        handles = result["evidence_handles"]
        if not isinstance(handles, list) or not handles:
            case["error"] = f"measurement_{name}_did_not_issue_visual_evidence"
            return case
        measurement_handles[name] = str(handles[0])
        case["evidence_bindings"][name] = str(handles[0])

    for index, stage in enumerate(actions):
        if not isinstance(stage, dict) or not stage:
            case["error"] = f"action_{index}_must_be_a_nonempty_object"
            return case
        phase = str(stage.get("phase") or f"action_{index}")
        action_stage = stage.get("actions", stage)
        reobserve = stage.get("reobserve", []) if action_stage is not stage else sorted(measurement_requests)
        if action_stage is not stage:
            extra = sorted(set(stage) - {"phase", "reobserve", "actions"})
            if extra:
                case["error"] = f"action_{index}_has_unsupported_stage_fields:{','.join(extra)}"
                return case
        elif "phase" in stage or "reobserve" in stage:
            case["error"] = f"action_{index}_stage_metadata_requires_actions_object"
            return case
        if not isinstance(action_stage, dict) or not action_stage:
            case["error"] = f"action_{index}_actions_must_be_a_nonempty_object"
            return case
        if not isinstance(reobserve, list) or any(not isinstance(name, str) for name in reobserve):
            case["error"] = f"action_{index}_reobserve_must_be_a_string_list"
            return case
        stage_observations: list[JsonDict] = []
        for name in reobserve:
            request = measurement_requests.get(name)
            if request is None:
                case["error"] = f"action_{index}_reobserve_unknown_measurement:{name}"
                return case
            result = backend.call_primitive(
                "measure_robotwin2_actor_visual",
                actor_label=str(request["actor_label"]),
                camera_name=str(request.get("camera_name") or "head_camera"),
                prompt=request.get("prompt"),
                agent_context={"case_measurement": name, "case_phase": phase, "source": "evaluator_public_case_reobserve"},
            )
            evidence = _robotwin2_measurement_case_evidence(name, result)
            stage_observations.append(evidence)
            if not result.ok:
                case["stage_observations"].append({"index": index, "phase": phase, "measurements": stage_observations})
                case["error"] = result.error or f"action_{index}_reobserve_{name}_failed"
                return case
            handles = result["evidence_handles"]
            if not isinstance(handles, list) or not handles:
                case["error"] = f"action_{index}_reobserve_{name}_did_not_issue_visual_evidence"
                return case
            measurement_handles[name] = str(handles[0])
            case["evidence_bindings"][name] = str(handles[0])
        case["stage_observations"].append({"index": index, "phase": phase, "measurements": stage_observations})
        requested_arms = [arm for arm in ("left", "right") if action_stage.get(arm) is not None]
        if len(requested_arms) != len(action_stage):
            case["error"] = f"action_{index}_may_only_contain_left_or_right"
            return case
        bound_stage: JsonDict = {}
        try:
            for arm in requested_arms:
                bound_stage[arm] = _bind_robotwin2_public_case_action(action_stage[arm], measurement_handles)
        except (TypeError, ValueError) as exc:
            case["error"] = f"action_{index}_binding_failed:{exc}"
            return case
        case["action_attempted"] = True
        bound_handles = list(measurement_handles.values())
        result = backend.call_primitive(
            "execute_robotwin2_actions",
            **bound_stage,
            evidence_handles=bound_handles,
            agent_context={"case_action_index": index, "case_phase": phase, "source": "evaluator_public_case"},
        )
        action_evidence = _robotwin2_action_case_evidence(index, action_stage, result)
        action_evidence["phase"] = phase
        action_evidence["reobservations"] = stage_observations
        case["actions"].append(action_evidence)
        if not result.ok:
            case["error"] = result.error or f"action_{index}_failed"
            return case

    case["action_chain_ok"] = bool(actions) and len(case["actions"]) == len(actions)
    return case


def _bind_robotwin2_public_case_action(spec: Any, measurement_handles: dict[str, str]) -> JsonDict:
    if not isinstance(spec, dict):
        raise TypeError("action_descriptor_must_be_an_object")
    forbidden = {
        "actor_name",
        "contact_point_id",
        "functional_point_id",
        "target_actor_name",
        "target_point_id",
        "target_point_type",
        "joint_path",
        "expert_action",
    }
    leaked = sorted(forbidden.intersection(spec))
    if leaked:
        raise ValueError(f"forbidden_oracle_or_expert_fields:{','.join(leaked)}")
    bound = deepcopy(spec)
    measurement_name = bound.pop("measurement", None)
    if measurement_name is not None:
        handle = measurement_handles.get(str(measurement_name))
        if handle is None:
            raise ValueError(f"unknown_measurement:{measurement_name}")
        bound["target_pose"] = {
            "evidence_handle": handle,
            "field": "centroid_world",
            "offset": bound.pop("offset", [0.0, 0.0, 0.0]),
        }
        quaternion = bound.pop("quaternion_wxyz", None)
        if quaternion is not None:
            bound["target_pose"]["quaternion_wxyz"] = quaternion
    elif str(bound.get("operation", "")).strip().lower() == "move_to_pose":
        raise ValueError("move_to_pose_requires_named_visual_measurement")
    return bound


def _robotwin2_measurement_case_evidence(name: str, result: PrimitiveResult) -> JsonDict:
    return {
        "name": name,
        "primitive": result.name,
        "ok": result.ok,
        "error": result.error,
        "actor_label": result.output.get("actor_label"),
        "camera_name": result.output.get("camera_name"),
        "bbox_pixels": result.output.get("bbox_pixels"),
        "centroid_pixels": result.output.get("centroid_pixels"),
        "centroid_world": result.output.get("centroid_world"),
        "pose_world": result.output.get("pose_world"),
        "geometry_source": result.output.get("geometry_source"),
        "observation_round": result.output.get("observation_round"),
        "evidence_handles": result.output.get("evidence_handles", []),
        "source_evidence_handles": result.output.get("source_evidence_handles", []),
    }


def _robotwin2_action_case_evidence(index: int, request: JsonDict, result: PrimitiveResult) -> JsonDict:
    return {
        "index": index,
        "primitive": result.name,
        "request": _to_builtin(request),
        "ok": result.ok,
        "error": result.error,
        "executed": result.output.get("executed", False),
        "arms": result.output.get("arms", []),
        "resolved_actions": result.output.get("actions", {}),
        "grounding_provenance": result.output.get("grounding_provenance", []),
        "post_arm_schema": result.output.get("post_arm_schema", {}),
        "route_execution": result.output.get("route_execution"),
        "diagnostic_evidence": result.output.get("diagnostic_evidence", {}),
    }


def robotwin2_visual_evidence(obs: Any, camera_name: str | None = None) -> JsonDict:
    evidence = {"visual_ready": False, "cameras": {}, "modalities_by_camera": {}, "has_rgb": False, "has_depth": False, "has_segmentation": False, "has_pointcloud": False}
    if not isinstance(obs, dict):
        return evidence
    observation = obs.get("observation", obs)
    if isinstance(observation, dict):
        for name, value in observation.items():
            if camera_name and name != camera_name:
                continue
            if not isinstance(value, dict):
                continue
            camera: JsonDict = {}
            modalities: list[str] = []
            for key, modality in value.items():
                lowered = str(key).lower()
                summary = summarize_observation(modality)
                if lowered in {"intrinsic_cv", "extrinsic_cv", "cam2world_gl"}:
                    camera[lowered] = _to_builtin(modality)
                elif "depth" in lowered:
                    camera["depth"] = summary
                    evidence["has_depth"] = True
                    modalities.append("depth")
                elif "segmentation" in lowered:
                    camera["segmentation"] = summary
                    evidence["has_segmentation"] = True
                    modalities.append("segmentation")
                elif "rgb" in lowered or _looks_like_rgb(modality):
                    camera["rgb"] = summary
                    evidence["has_rgb"] = True
                    modalities.append("rgb")
            if camera:
                evidence["cameras"][name] = camera
                evidence["modalities_by_camera"][name] = sorted(set(modalities))
    pointcloud = obs.get("pointcloud")
    if pointcloud is not None and getattr(pointcloud, "size", len(pointcloud) if hasattr(pointcloud, "__len__") else 0):
        evidence["has_pointcloud"] = True
        evidence["pointcloud"] = summarize_observation(pointcloud)
    evidence["visual_ready"] = any(evidence[key] for key in ("has_rgb", "has_depth", "has_segmentation", "has_pointcloud"))
    return evidence


def _measure_robotwin2_actor_geometry(
    obs: Any,
    *,
    camera_name: str,
    segmentation_id: int | list[int],
    depth_scale: float,
    quaternion_wxyz: list[float] | None,
    max_points: int,
) -> JsonDict:
    if not isinstance(obs, dict):
        raise ValueError("visual_world_geometry_unavailable")
    observation = obs.get("observation", obs)
    camera = observation.get(camera_name) if isinstance(observation, dict) else None
    if not isinstance(camera, dict):
        raise ValueError("visual_world_geometry_unavailable:camera")
    required = ("rgb", "depth", "actor_segmentation")
    if any(camera.get(key) is None for key in required):
        raise ValueError("visual_world_geometry_unavailable:rgb_depth_actor_segmentation")
    rgb = np.asarray(camera["rgb"])
    depth = np.asarray(camera["depth"], dtype=float)
    segmentation = np.asarray(camera["actor_segmentation"])
    if rgb.ndim < 2 or depth.ndim != 2 or rgb.shape[:2] != depth.shape or segmentation.shape[:2] != depth.shape:
        raise ValueError("visual_world_geometry_unavailable:unaligned_camera_modalities")
    mask = _robotwin2_segmentation_mask(segmentation, segmentation_id)
    if not np.any(mask):
        raise ValueError("selected_actor_not_visible_in_segmentation")
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        raise ValueError("visual_world_geometry_unavailable:selected_actor_has_no_depth")
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError("depth_scale_must_be_positive")
    world = _robotwin2_mask_world_points(camera, obs, valid, depth, float(depth_scale))
    if world.size == 0 or not np.all(np.isfinite(world)):
        raise ValueError("visual_world_geometry_unavailable:world_points")
    ys, xs = np.nonzero(mask)
    valid_ys, valid_xs = np.nonzero(valid)
    centroid_world = np.median(world, axis=0)
    quaternion = [1.0, 0.0, 0.0, 0.0] if quaternion_wxyz is None else [float(item) for item in quaternion_wxyz]
    if len(quaternion) != 4 or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion_wxyz_must_have_4_finite_values")
    if max_points < 1:
        raise ValueError("max_points_must_be_positive")
    sample_indices = np.linspace(0, len(world) - 1, min(int(max_points), len(world)), dtype=int)
    return {
        "bbox_pixels": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "centroid_pixels": [float(np.median(valid_xs)), float(np.median(valid_ys))],
        "mask_pixel_count": int(mask.sum()),
        "valid_depth_pixel_count": int(valid.sum()),
        "world_points": world[sample_indices].astype(float).tolist(),
        "world_point_count": int(len(world)),
        "centroid_world": centroid_world.astype(float).tolist(),
        "pose_world": [*centroid_world.astype(float).tolist(), *quaternion],
        "depth_scale": float(depth_scale),
        "geometry_source": "aligned_pointcloud" if _aligned_pointcloud(obs, depth.shape) is not None else "depth_intrinsic_extrinsic",
        "rgb_shape": [int(item) for item in rgb.shape],
    }


def _robotwin2_segmentation_mask(segmentation: np.ndarray, selector: int | list[int]) -> np.ndarray:
    normalized = _normalize_segmentation_selector(selector)
    if segmentation.ndim == 2:
        if not isinstance(normalized, int):
            raise ValueError("scalar_actor_segmentation_requires_integer_id")
        return segmentation.astype(np.int64) == normalized
    if segmentation.ndim == 3 and segmentation.shape[2] >= 3:
        if isinstance(normalized, int):
            color = np.asarray(_robotwin2_actor_id_color(normalized), dtype=segmentation.dtype)
        else:
            color = np.asarray(normalized, dtype=segmentation.dtype)
        return np.all(segmentation[..., :3] == color[:3], axis=2)
    raise ValueError("unsupported_actor_segmentation_shape")


def _normalize_segmentation_selector(selector: int | list[int]) -> int | list[int]:
    if isinstance(selector, (int, np.integer)):
        return int(selector)
    values = [int(item) for item in selector]
    if len(values) != 3 or any(item < 0 or item > 255 for item in values):
        raise ValueError("segmentation_color_must_have_3_byte_values")
    return values


def _robotwin2_actor_id_color(actor_id: int) -> tuple[int, int, int]:
    if actor_id < 0:
        raise ValueError("segmentation_id_must_be_nonnegative")
    try:
        from PIL import ImageColor
    except ModuleNotFoundError as exc:  # pragma: no cover - Pillow is an upstream RoboTwin dependency.
        raise ValueError("actor_id_palette_unavailable") from exc
    palette = sorted(set(ImageColor.colormap.values()))
    if actor_id >= len(palette):
        raise ValueError("segmentation_id_outside_robotwin2_palette")
    return ImageColor.getrgb(palette[actor_id])


def _aligned_pointcloud(obs: Any, image_shape: tuple[int, int]) -> np.ndarray | None:
    pointcloud = obs.get("pointcloud") if isinstance(obs, dict) else None
    array = np.asarray(pointcloud) if pointcloud is not None else np.asarray([])
    return array if array.ndim == 3 and array.shape[:2] == image_shape and array.shape[2] >= 3 else None


def _robotwin2_mask_world_points(
    camera: JsonDict,
    obs: JsonDict,
    valid: np.ndarray,
    depth: np.ndarray,
    depth_scale: float,
) -> np.ndarray:
    aligned = _aligned_pointcloud(obs, depth.shape)
    if aligned is not None:
        return np.asarray(aligned[valid, :3], dtype=float)
    intrinsic = camera.get("intrinsic_cv")
    extrinsic = camera.get("extrinsic_cv")
    if intrinsic is None or extrinsic is None:
        raise ValueError("visual_world_geometry_unavailable:camera_calibration")
    intrinsic_array = _robotwin2_calibration_matrix(intrinsic, (3, 3))
    extrinsic_array = _robotwin2_calibration_matrix(extrinsic, (4, 4))
    try:
        inverse_intrinsic = np.linalg.inv(intrinsic_array)
        camera_to_world = np.linalg.inv(extrinsic_array)
    except np.linalg.LinAlgError as exc:
        raise ValueError("visual_world_geometry_unavailable:singular_camera_calibration") from exc
    ys, xs = np.nonzero(valid)
    z = depth[valid] * depth_scale
    pixels = np.stack((xs.astype(float), ys.astype(float), np.ones_like(z)), axis=0)
    camera_points = (inverse_intrinsic @ pixels) * z
    homogeneous = np.vstack((camera_points, np.ones((1, camera_points.shape[1]), dtype=float)))
    return (camera_to_world @ homogeneous)[:3].T


def _robotwin2_calibration_matrix(value: Any, expected_shape: tuple[int, int]) -> np.ndarray:
    """Normalize direct, affine, singleton-batched, or flat calibration matrices."""
    array = np.asarray(value, dtype=float)
    squeezed = np.squeeze(array)
    if squeezed.shape == expected_shape:
        return squeezed
    if expected_shape[0] == expected_shape[1] and squeezed.shape == (expected_shape[0] - 1, expected_shape[1]):
        homogeneous_row = np.zeros((1, expected_shape[1]), dtype=float)
        homogeneous_row[0, -1] = 1.0
        return np.vstack((squeezed, homogeneous_row))
    if squeezed.ndim == 1 and squeezed.size == expected_shape[0] * expected_shape[1]:
        return squeezed.reshape(expected_shape)
    raise ValueError("visual_world_geometry_unavailable:camera_calibration_shape")


def _robotwin2_visual_arrays(obs: Any, camera_name: str | None = None) -> list[tuple[str, str, Any]]:
    arrays: list[tuple[str, str, Any]] = []
    if not isinstance(obs, dict):
        return arrays
    observation = obs.get("observation", obs)
    if isinstance(observation, dict):
        for name, camera in observation.items():
            if camera_name and name != camera_name:
                continue
            if not isinstance(camera, dict):
                continue
            for key, value in camera.items():
                lowered = str(key).lower()
                if (
                    "rgb" in lowered
                    or "depth" in lowered
                    or "segmentation" in lowered
                    or lowered in {"intrinsic_cv", "extrinsic_cv", "cam2world_gl"}
                ):
                    arrays.append((str(name), lowered, value))
    pointcloud = obs.get("pointcloud")
    if pointcloud is not None and not camera_name:
        size = getattr(pointcloud, "size", len(pointcloud) if hasattr(pointcloud, "__len__") else 0)
        if size:
            arrays.append(("combined", "pointcloud", pointcloud))
    return arrays


def summarize_observation(value: Any, *, max_items: int = 8) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return {str(key): summarize_observation(val, max_items=max_items) for key, val in list(value.items())[:max_items]}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "length": len(value), "items": [summarize_observation(v, max_items=max_items) for v in list(value)[:max_items]]}
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"type": type(value).__name__, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(value, (str, int, float, bool)):
        return value
    return {"type": type(value).__name__, "repr": repr(value)[:160]}


def _summarize_robotwin2_motion_plan(value: Any) -> JsonDict:
    if not isinstance(value, dict):
        return {"status": None, "available": False, "raw_type": type(value).__name__}
    summary: JsonDict = {
        "status": _to_builtin(value.get("status")),
        "available": True,
        "fields": sorted(str(key) for key in value),
    }
    for key in ("position", "velocity", "acceleration", "time"):
        item = value.get(key)
        shape = getattr(item, "shape", None)
        if shape is not None:
            summary[f"{key}_shape"] = [int(dim) for dim in shape]
        elif isinstance(item, (list, tuple)):
            summary[f"{key}_length"] = len(item)
    position = value.get("position")
    if position is not None:
        try:
            summary["path_length"] = int(len(position))
        except TypeError:
            pass
    if value.get("exception") is not None:
        summary["exception"] = str(value["exception"])
    return summary


def _robotwin2_take_action_ok(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("ok", "accepted", "success", "submitted"):
            if key in value:
                return bool(value[key])
        status = str(value.get("status", "")).strip().lower()
        if status:
            return status in {"ok", "accepted", "success", "submitted"}
    return bool(value) if value is not None else False


def _asset_component_state(
    assets: Path,
    name: str,
    *,
    expected_zip_size: int | None = None,
    expected_sha256: str | None = None,
) -> JsonDict:
    directory = assets / name
    archive = assets / f"{name}.zip"
    incomplete_paths = []
    download_cache = assets / ".cache/huggingface/download"
    if download_cache.exists():
        incomplete_paths = sorted(
            {
                path.name: path.stat().st_size
                for path in download_cache.glob("*.incomplete")
                if name in path.name or (expected_sha256 and expected_sha256 in path.name)
            }.items()
        )
    archive_size = archive.stat().st_size if archive.exists() else None
    archive_size_matches = archive_size == expected_zip_size if expected_zip_size is not None and archive_size is not None else None
    return {
        "directory": str(directory),
        "zip_path": str(archive),
        "directory_exists": directory.exists(),
        "directory_has_entries": _dir_has_entries(directory),
        "extracted_ready": directory.exists() and _dir_has_entries(directory),
        "zip_exists": archive.exists(),
        "zip_size_bytes": archive_size,
        "expected_zip_size_bytes": expected_zip_size,
        "zip_size_matches_expected": archive_size_matches,
        "expected_sha256": expected_sha256,
        "incomplete_downloads": [{"name": name, "size_bytes": size} for name, size in incomplete_paths],
    }


def _dir_has_entries(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        next(path.iterdir())
    except StopIteration:
        return False
    except OSError:
        return False
    return True


def _robotwin2_config_assets_ready(readiness: JsonDict, task_config: str) -> bool:
    key = "demo_randomized" if "random" in task_config.lower() else "demo_clean"
    return bool(readiness.get("readiness", {}).get(key, False))


def _robotwin2_config_asset_blockers(readiness: JsonDict, task_config: str) -> list[JsonDict]:
    key = "demo_randomized" if "random" in task_config.lower() else "demo_clean"
    missing = list(readiness.get("missing_components", {}).get(key, []))
    if not missing:
        return []
    return [
        {
            "id": f"robotwin2_{key}_assets_missing",
            "task_config": task_config,
            "missing_components": missing,
            "hf_dataset_repo": readiness.get("hf_dataset_repo", ROBOTWIN2_HF_DATASET_REPO),
            "download_command": readiness.get("download_command"),
        }
    ]


def _load_task_config(repo_path: str | Path, task_config: str) -> JsonDict:
    path = Path(repo_path).expanduser() / "task_config" / f"{task_config}.yml"
    if not path.exists():
        return {}
    return _load_yaml(path)


def _load_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ModuleNotFoundError("PyYAML is required to load RoboTwin2 task configs") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _git_head(repo: Path) -> str | None:
    head = repo / ".git/HEAD"
    if not head.exists():
        return None
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if text.startswith("ref:"):
        ref = repo / ".git" / text.split(" ", 1)[1]
        return ref.read_text(encoding="utf-8", errors="replace").strip() if ref.exists() else text
    return text


def _select_candidate(candidates: list[JsonDict], explicit: str | None = None) -> JsonDict | None:
    if explicit is None:
        return None
    lowered = explicit.lower()
    for candidate in candidates:
        names = [str(candidate.get("name", "")), str(candidate.get("actor_name", ""))]
        if any(lowered == name.lower() for name in names if name):
            return candidate
    return None


def _looks_like_actor(value: Any) -> bool:
    return value is not None and hasattr(value, "get_pose") and hasattr(value, "get_name")


def _actor_name(actor: Any) -> str | None:
    if actor is None:
        return None
    if hasattr(actor, "get_name"):
        try:
            return str(actor.get_name())
        except Exception:
            return None
    wrapped = getattr(actor, "actor", None)
    if wrapped is not None and hasattr(wrapped, "get_name"):
        try:
            return str(wrapped.get_name())
        except Exception:
            return None
    return None


def _actor_pose(actor: Any) -> list[float] | None:
    if actor is None or not hasattr(actor, "get_pose"):
        return None
    try:
        pose = actor.get_pose()
    except Exception:
        return None
    p = getattr(pose, "p", None)
    q = getattr(pose, "q", None)
    if p is None:
        return None
    return _as_float_list(p) + (_as_float_list(q) if q is not None else [])


def _actor_segmentation_id(actor: Any) -> int | None:
    if actor is None:
        return None
    candidates = [actor, getattr(actor, "actor", None), getattr(actor, "entity", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        for method_name in ("get_per_scene_id", "get_id"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    value = method()
                except Exception:
                    continue
                if isinstance(value, (int, np.integer)):
                    return int(value)
        for attribute in ("segmentation_id", "per_scene_id", "id"):
            value = getattr(candidate, attribute, None)
            if isinstance(value, (int, np.integer)):
                return int(value)
    return None


def _extract_actor_points(actor: Any, point_type: str) -> list[JsonDict]:
    if actor is None:
        return []
    method_name = {
        "contact": "get_contact_point",
        "functional": "get_functional_point",
        "target": "get_target_point",
        "orientation": "get_orientation_point",
    }.get(point_type, point_type)
    method = getattr(actor, method_name, None)
    if method is None:
        return []
    if point_type == "orientation":
        for args in (("list",), ()):
            try:
                value = method(*args)
            except Exception:
                continue
            return [{"index": 0, "point_type": point_type, "pose": _to_builtin(value)}]
        return []
    points: list[JsonDict] = []
    for index in range(16):
        value = None
        for args in ((index, "list"), (index,)):
            try:
                value = method(*args)
            except Exception:
                continue
            if value is not None:
                break
        if value is None:
            if index == 0:
                continue
            break
        points.append({"index": index, "point_type": point_type, "pose": _to_builtin(value)})
    return points


def _robotwin2_workspace_z_bounds(value: Any, arm_tag: str) -> JsonDict | None:
    value = _to_builtin(value)
    if isinstance(value, dict):
        if arm_tag in value:
            nested = _robotwin2_workspace_z_bounds(value[arm_tag], arm_tag)
            if nested is not None:
                return nested
        if "bounds" in value:
            nested = _robotwin2_workspace_z_bounds(value["bounds"], arm_tag)
            if nested is not None:
                return nested
        z_value = value.get("z") or value.get("z_bounds")
        if isinstance(z_value, list) and len(z_value) == 2:
            z_min, z_max = map(float, z_value)
        elif value.get("z_min") is not None and value.get("z_max") is not None:
            z_min, z_max = float(value["z_min"]), float(value["z_max"])
        else:
            return None
    elif (
        isinstance(value, list)
        and len(value) == 3
        and isinstance(value[2], list)
        and len(value[2]) == 2
    ):
        z_min, z_max = map(float, value[2])
    else:
        return None
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min >= z_max:
        return None
    return {"z_min": z_min, "z_max": z_max}


def _as_float_list(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        return [float(value)]
    return [float(item) for item in value]


def _to_builtin(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "tolist"):
        return _to_builtin(value.tolist())
    p = getattr(value, "p", None)
    q = getattr(value, "q", None)
    if p is not None:
        return _as_float_list(p) + (_as_float_list(q) if q is not None else [])
    return repr(value)


def _safe_call(obj: Any, name: str) -> Any:
    if obj is None or not hasattr(obj, name):
        return None
    try:
        return getattr(obj, name)()
    except Exception:
        return None


def _live_required_result(name: str, **output: Any) -> PrimitiveResult:
    return PrimitiveResult(
        name=name,
        ok=False,
        output={**output, "executed": False, "submitted": False, "requires_live_env": True},
        error="live_env_not_created",
    )


def _looks_like_rgb(value: Any) -> bool:
    shape = getattr(value, "shape", None)
    if shape is None or len(shape) < 3:
        return False
    return int(shape[-1]) in {3, 4}


def _robotwin2_vendored_curobo_src(repo: Path) -> Path:
    return repo / "envs" / "curobo" / "src"


def _reload_stale_robotwin2_curobo_planner(repo: Path) -> bool:
    """Reload only a half-initialized planner module from this RoboTwin checkout.

    Upstream intentionally catches optional cuRobo import failures. That leaves a
    successfully cached ``envs.robot.planner`` module without ``CuroboPlanner``;
    a later reset would otherwise reuse it after CUDA/assets become available.
    """

    module = sys.modules.get("envs.robot.planner")
    if module is None or hasattr(module, "CuroboPlanner"):
        return False
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    expected = (repo / "envs" / "robot" / "planner.py").resolve()
    try:
        loaded_from = Path(module_file).expanduser().resolve()
    except (OSError, RuntimeError):
        return False
    if loaded_from != expected:
        return False
    reloaded = importlib.reload(module)
    return hasattr(reloaded, "CuroboPlanner")


def _is_robotwin2_curobo_planner_import_error(exc: ImportError) -> bool:
    text = str(exc)
    return "CuroboPlanner" in text and "envs.robot.planner" in text


def _robotwin2_curobo_planner_unavailable_message(repo: Path, exc: ImportError) -> str:
    curobo_src = _robotwin2_vendored_curobo_src(repo)
    return (
        "RoboTwin2 cuRobo planner is unavailable: upstream `envs.robot.planner` did not define "
        "`CuroboPlanner` after its optional cuRobo/CUDA import. This usually means the OpenHands "
        "DockerWorkspace or Slurm node is missing an NVIDIA driver/CUDA runtime, or the repo-local "
        "RoboTwin2 Python environment has not installed/imported cuRobo correctly. "
        f"vendored_curobo_src={curobo_src}; original_import_error={exc}"
    )


def _resolve_robotwin2_assets(
    repo: Path, assets_path: str | Path | None = None
) -> Path:
    configured = assets_path or os.environ.get("ROBOTWIN2_ASSETS_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else (repo / "assets").resolve()
    )


@contextmanager
def _repo_import_context(
    repo: Path, *, assets_path: Path | None = None
) -> Iterator[None]:
    cwd = Path.cwd()
    inserted_paths: list[str] = []
    assets = (assets_path or (repo / "assets")).resolve()
    runtime_view: tempfile.TemporaryDirectory[str] | None = None
    global_config: Any | None = None
    original_asset_paths: dict[str, Any] = {}
    for path in (repo, _robotwin2_vendored_curobo_src(repo)):
        if not path.exists():
            continue
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)
            inserted_paths.append(path_text)
    try:
        if assets != (repo / "assets").resolve():
            runtime_view = tempfile.TemporaryDirectory(
                prefix="agentic_embodied_arena_robotwin2_"
            )
            Path(runtime_view.name, "assets").symlink_to(
                assets, target_is_directory=True
            )
            os.chdir(runtime_view.name)
            global_config = importlib.import_module("envs._GLOBAL_CONFIGS")
            for name in ("ASSETS_PATH", "EMBODIMENTS_PATH"):
                original_asset_paths[name] = getattr(global_config, name, None)
            global_config.ASSETS_PATH = f"{assets}{os.sep}"
            global_config.EMBODIMENTS_PATH = f"{assets / 'embodiments'}{os.sep}"
        else:
            os.chdir(repo)
        yield
    finally:
        os.chdir(cwd)
        if global_config is not None:
            for name, value in original_asset_paths.items():
                setattr(global_config, name, value)
        if runtime_view is not None:
            runtime_view.cleanup()
        for path_text in inserted_paths:
            try:
                sys.path.remove(path_text)
            except ValueError:
                pass


def _restore_native_verifier_context(env: Any, native_module: Any, task_name: str, episode_info: JsonDict) -> None:
    """Restore metadata retained by upstream's expert-check then setup_demo cycle.

    These tasks set verifier inputs in play_once before the first move. The
    accepted episode already records its native arm selection; no expert action
    is replayed and check_success remains the original benchmark implementation.
    """
    if task_name not in {"open_laptop", "place_object_scale", "put_object_cabinet"}:
        return
    arm = episode_info.get("info", {}).get("{a}")
    if arm not in {"left", "right"}:
        raise ValueError("Native accepted episode lacks verifier arm metadata")
    env.arm_tag = native_module.ArmTag(arm)
    if task_name == "put_object_cabinet":
        env.origin_z = env.object.get_pose().p[2]
