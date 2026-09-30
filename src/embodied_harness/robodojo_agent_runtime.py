from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import numpy as np

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .robodojo_policy_skill import (
    RoboDojoPolicySkillConfig,
    XPolicyLabClient,
    create_robodojo_policy_client,
    inspect_robodojo_policy_skill,
    inspect_robodojo_rgb_observation,
    list_robodojo_policy_adapters,
    run_robodojo_policy_skill,
)
from .schemas import (
    EpisodeTrace,
    Observation,
    PrimitiveCard,
    PrimitiveResult,
    TaskSpec,
    VerificationResult,
)


JsonDict = dict[str, Any]

DEFAULT_ROBODOJO_REPO = get_project_paths().external_upstream("robodojo")
OBJECT_SECTIONS = ("Rigid", "Geometry", "Articulation", "Deformable")
ROBODOJO_LIVE_OFFICIAL_RECEIPT_SCHEMA = (
    "agentic-embodied-arena/robodojo-live-official-receipt/v1"
)
ROBODOJO_LIVE_OFFICIAL_OPERATION = "robodojo_general_pickup_current_interface"
ROBODOJO_OFFICIAL_REWARD_SYMBOL = (
    "task/RoboDojo/tasks/general_pickup.py:GeneralPickupCommon.run_reward"
)
ROBODOJO_OFFICIAL_REWARD_SHA256 = (
    "1a9eef7fe518f9d2c6805ad1a8f97c1045fa83726f84b9c63f67ffdfabbc40a7"
)

_ROBODOJO_LIVE_EXIT_REPORT_FIELDS = (
    "ok",
    "stage",
    "error_type",
    "error",
    "system_exit_code",
    "elapsed_seconds",
    "blocker",
    "runtime_diagnostics",
)


def _robodojo_json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _robodojo_text_tail(path: Path, max_chars: int) -> str:
    if not path.is_file():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:]
    except OSError as exc:
        return f"<unreadable:{type(exc).__name__}:{exc}>"


def _robodojo_live_exit_diagnostic(
    *,
    report_path: Path,
    stdout_path: Path,
    stderr_path: Path,
) -> str:
    """Return bounded, credential-safe evidence from an exited live Isaac probe."""

    report_summary: JsonDict = {}
    if report_path.is_file():
        try:
            loaded = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            report_summary["report_load_error"] = f"{type(exc).__name__}:{exc}"
        else:
            if isinstance(loaded, dict):
                report_summary.update(
                    (field, loaded[field])
                    for field in _ROBODOJO_LIVE_EXIT_REPORT_FIELDS
                    if field in loaded
                )
                trace_events = loaded.get("trace_events")
                if isinstance(trace_events, list):
                    report_summary["trace_events_tail"] = trace_events[-8:]
            else:
                report_summary["report_load_error"] = "report_not_mapping"
    else:
        report_summary["report_load_error"] = "report_missing"
    report_text = json.dumps(
        report_summary,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        default=repr,
    )[-16000:]
    return "; ".join(
        (
            f"report={report_text}",
            f"stdout_tail={_robodojo_text_tail(stdout_path, 8000)!r}",
            f"stderr_tail={_robodojo_text_tail(stderr_path, 16000)!r}",
        )
    )


@dataclass(slots=True)
class RoboDojoIsaacLiveOfficialProxy:
    """Evaluator-only handle to the still-live Isaac process owning the episode."""

    socket_path: Path
    auth_token: str
    episode_instance_nonce: str
    probe_runtime: JsonDict
    submitted_actions_digest: str
    timeout_seconds: float = 900.0
    on_complete: Callable[[], None] | None = field(default=None, repr=False)
    persistent_episode: bool = False
    _consumed: bool = field(default=False, init=False, repr=False)

    def capture_live_official(self, request: JsonDict) -> JsonDict:
        if self._consumed:
            raise RuntimeError("RoboDojo live official capture already consumed")
        if not self.socket_path.is_absolute() or not self.socket_path.exists():
            raise RuntimeError("RoboDojo live Isaac control socket is unavailable")
        bound_request = {
            **dict(request),
            "operation_id": ROBODOJO_LIVE_OFFICIAL_OPERATION,
            "episode_instance_nonce": self.episode_instance_nonce,
            "expected_probe_runtime": dict(self.probe_runtime),
            "expected_submitted_actions_digest": self.submitted_actions_digest,
        }
        self._consumed = not self.persistent_episode
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout_seconds)
            client.connect(str(self.socket_path))
            client.sendall(
                (
                    json.dumps(
                        {"authorization": self.auth_token, "request": bound_request},
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            raw = b""
            while b"\n" not in raw and len(raw) <= 1024 * 1024:
                chunk = client.recv(65536)
                if not chunk:
                    break
                raw += chunk
        if b"\n" not in raw or len(raw) > 1024 * 1024:
            raise RuntimeError("RoboDojo live official response is incomplete")
        response = json.loads(raw.split(b"\n", 1)[0])
        if not isinstance(response, dict) or response.get("ok") is not True:
            raise RuntimeError(
                f"RoboDojo live official capture failed: {response.get('error') if isinstance(response, dict) else 'invalid_response'}"
            )
        receipt = response.get("receipt")
        if not isinstance(receipt, dict):
            raise RuntimeError("RoboDojo live official receipt is missing")
        expected = {
            "schema_version": ROBODOJO_LIVE_OFFICIAL_RECEIPT_SCHEMA,
            "operation_id": ROBODOJO_LIVE_OFFICIAL_OPERATION,
            "episode_instance_nonce": self.episode_instance_nonce,
            "probe_runtime": self.probe_runtime,
            "submitted_actions_digest": self.submitted_actions_digest,
            "request_binding": bound_request,
            "request_binding_digest": _robodojo_json_digest(bound_request),
            "official_symbol": ROBODOJO_OFFICIAL_REWARD_SYMBOL,
            "official_source_sha256": ROBODOJO_OFFICIAL_REWARD_SHA256,
        }
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise RuntimeError(
                    f"RoboDojo live official receipt binding mismatch: {key}"
                )
        official_result = receipt.get("official_result")
        if (
            not isinstance(official_result, dict)
            or official_result.get("source") != "upstream_official_result"
            or type(official_result.get("success")) is not bool
        ):
            raise RuntimeError("RoboDojo live official result is invalid")
        if self.on_complete is not None:
            self.on_complete()
        return receipt


@dataclass(slots=True)
class RoboDojoRuntimeConfig:
    repo_path: str = str(DEFAULT_ROBODOJO_REPO)
    task_name: str = ""
    env_cfg: str = "arx_x5"
    sim_env: str = "RoboDojo"
    policy_name: str = "ACT"
    policy_checkpoint: str = ""
    policy_server_url: str = "ws://127.0.0.1:6000"
    policy_port: int = 6060
    action_type: str = "joint"
    live: bool = False
    source_boundary_ok: bool = True
    env_kwargs: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RoboDojoAgentRuntimeBackend(EmbodiedBackend):
    """Agent-native RoboDojo adapter.

    RoboDojo is an eval-only Isaac Sim benchmark in the public release: the
    simulator client, tasks, configs, assets, and result artifacts live in this
    repo, while policy servers are owned by XPolicyLab. This adapter exposes the
    source/task/config evidence and action-plan boundary to a coding agent, and
    keeps official Isaac/XPolicyLab success as a harness-only gate.
    """

    def __init__(
        self,
        config: RoboDojoRuntimeConfig | None = None,
        *,
        policy_client: XPolicyLabClient | None = None,
        action_executor: Callable[[JsonDict], Any] | None = None,
        observation_provider: Callable[[], JsonDict] | None = None,
        official_verifier: Callable[[], Any] | None = None,
    ) -> None:
        self.config = config or RoboDojoRuntimeConfig()
        self._policy_client = policy_client
        self._policy_client_owned = False
        self._policy_client_key: tuple[str, ...] | None = None
        self._action_executor = action_executor
        self._observation_provider = observation_provider
        self._official_verifier = official_verifier
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._repo: JsonDict = {}
        self._task_config: JsonDict = {}
        self._env_config: JsonDict = {}
        self._entities: list[JsonDict] = []
        self._submitted_plan: JsonDict | None = None
        self._current_runtime_observation: JsonDict | None = None
        self._current_visual_evidence_refs: list[JsonDict] = []
        self.isaac_subprocess_live_general_pickup: (
            RoboDojoIsaacLiveOfficialProxy | None
        ) = None
        self._isaac_process: subprocess.Popen[str] | None = None
        self._isaac_terminal_error: str | None = None
        self._isaac_runtime_dir: Path | None = None
        self._isaac_stdout: Any = None
        self._isaac_stderr: Any = None
        self._isaac_report_path: Path | None = None
        self._isaac_action_socket_path: Path | None = None
        self._isaac_action_auth_token: str | None = None
        self._isaac_episode_nonce: str | None = None
        self._pool_layout: tuple[str, str, int] | None = None
        self._isaac_official_socket_path: Path | None = None
        self._isaac_official_auth_token: str | None = None

    def close(self) -> None:
        """Release the live Isaac episode and any backend-owned policy client."""

        self._close_live_isaac_subprocess()
        policy_client = self._policy_client
        if (
            policy_client is not None
            and self._policy_client_owned
            and hasattr(policy_client, "close")
        ):
            try:
                policy_client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._policy_client = None
        self._policy_client_owned = False
        self._policy_client_key = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        name = str(coordinate.get("task_id") or "")
        parts = str(coordinate.get("variation") or "").split("::")
        seed = coordinate.get("seed")
        if (name != "general_pickup" or len(parts) != 3 or parts[1] != name
                or parts[0] != "arx_x5" or not parts[2].startswith("episode_")
                or not parts[2][8:].isdigit()):
            raise ValueError("RoboDojo live binding supports native arx_x5 general_pickup layouts")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("RoboDojo evaluation seed must be an integer in [0, 2**32)")
        filename = f"{name}_{int(parts[2][8:])}.json"
        env_cfg = load_robodojo_env_config(self.config.repo_path, parts[0])
        layout = (Path(self.config.repo_path) / "Assets/Eval_Layout/RoboDojo"
                  / str(env_cfg["config_name"]) / str(seed) / filename)
        if not layout.is_file():
            raise ValueError(f"Selected original RoboDojo layout is missing: {filename}")
        self._pool_layout = (parts[0], filename, seed)
        return {"bound": True, "mode": "native_saved_layout", "task_name": name,
                "env_cfg": parts[0], "eval_seed": seed, "layout_file": str(layout),
                "layout_sha256": hashlib.sha256(layout.read_bytes()).hexdigest()}

    def reset(
        self, task_id: str, seed: int | None = None, config: JsonDict | None = None
    ) -> TaskSpec:
        self._isaac_terminal_error = None
        overrides = dict(config or {})
        if self._pool_layout is not None:
            env_cfg, filename, selected_seed = self._pool_layout
            if seed != selected_seed:
                raise ValueError("RoboDojo reset seed differs from selected pool coordinate")
            env_kwargs = {**self.config.env_kwargs, **overrides.get("env_kwargs", {}),
                          "seed": selected_seed, "selected_layout_file": filename}
            overrides.update(task_name="general_pickup", env_cfg=env_cfg, env_kwargs=env_kwargs)
            task_id = f"robodojo:{env_cfg}:{filename}:seed_{seed}"
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._repo = inspect_robodojo_repo(runtime_config.repo_path)
        self._task_config = load_robodojo_task_config(
            runtime_config.repo_path, runtime_config.task_name
        )
        self._env_config = load_robodojo_env_config(
            runtime_config.repo_path, runtime_config.env_cfg
        )
        self._entities = extract_robodojo_entities(self._task_config)
        self._submitted_plan = None
        self._current_runtime_observation = None
        self._current_visual_evidence_refs = []
        self._close_live_isaac_subprocess()
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w7:robodojo:agent_runtime",
            instruction=(
                "Use only benchmark-native runtime observations and the public robot action schema. "
                "Task interpretation and low-level action ordering belong to the coding agent; public "
                "tools do not expose task planners, subgoal synthesis, reward, checker, demos, or priors."
            ),
            goal={
                "benchmark": "RoboDojo",
                "task_name": runtime_config.task_name,
                "env_cfg": runtime_config.env_cfg,
                "success_source": "harness_only_robodojo_eval_result_or_isaac_verifier",
                "success_exposed_to_agent": False,
            },
            initial_state={
                "repo": self._repo,
                "env_cfg": runtime_config.env_cfg,
                "runtime_observation_required": True,
            },
            budgets={"primitive_calls": 48, "verifier_calls": 6},
            tags=[
                "w7",
                "robodojo",
                "isaacsim",
                "isaaclab",
                "source_boundary" if not runtime_config.live else "live",
            ],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "robodojo",
                "runtime_config": runtime_config.to_dict(),
                "upstream": self._repo,
                "source_preflight": robodojo_source_preflight(
                    runtime_config.repo_path,
                    runtime_config.task_name,
                    runtime_config.env_cfg,
                ),
                "asset_gate": robodojo_asset_gate(runtime_config.repo_path),
                "agent_native_contract": {
                    "benchmark_native_runtime_observation_only": True,
                    "task_config_entities_exposed": False,
                    "task_or_subgoal_synthesis_exposed": False,
                    "low_level_action_submission_exposed": True,
                    "official_success_exposed_as_primitive": False,
                    "expert_trajectory_or_demo_replay_exposed": False,
                    "default_low_level_skill": "XPolicyLab policy adapter",
                    "default_policy_name": runtime_config.policy_name,
                    "policy_prompt_is_agent_parameterized": True,
                    "hand_written_ik_is_default": False,
                    "live_isaac_eval_required_for_official_success": True,
                },
            },
        )
        self.record_event(
            "reset",
            {
                "task": self._task_spec.to_dict(),
                "seed": seed,
                "runtime_available": self.runtime_available(),
            },
        )
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        if (
            self._observation_provider is None
            and self._current_runtime_observation is None
            and self.config.live
            and self.config.task_name == "general_pickup"
        ):
            self._ensure_live_isaac_observation_episode()
        live_observation = (
            self._observation_provider()
            if self._observation_provider is not None
            else self._current_runtime_observation
        )
        if isinstance(live_observation, dict):
            self._current_runtime_observation = live_observation
        visual_contract = _robodojo_visual_observation_contract(live_observation)
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "env_cfg": summarize_robodojo_env_cfg(self._env_config),
                "action_schema": self._action_schema(),
                "public_runtime_observation": live_observation,
                "visual_observation_contract": visual_contract,
                "runtime_observation_contract": {
                    "source": "live_observation_provider_or_observe_robodojo_runtime_report",
                    "required_rgb_views": [
                        "cam_head",
                        "cam_left_wrist",
                        "cam_right_wrist",
                    ],
                    "raw_rgb_frames_required_for_pi05": True,
                    "vision_summary_is_not_visual_grounding": True,
                    "fields": [
                        "observation.instruction",
                        "observation.state",
                        "observation.vision_summary",
                        "observation.vision.cam_head.color|rgb",
                        "observation.vision.cam_left_wrist.color|rgb",
                        "observation.vision.cam_right_wrist.color|rgb",
                        "scene.objects[].pose",
                        "scene.objects[].bbox",
                        "robot.robots[].joint_state",
                        "robot.robots[].gripper_state",
                        "robot.robots[].ee_pose",
                    ],
                },
            },
            metadata={"benchmark_id": "robodojo", "task_name": self.config.task_name},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_robodojo_runtime_report",
                "L1",
                {"report_path": "str|None", "agent_context": "dict|None"},
                {
                    "public_runtime_state": "dict|None",
                    "public_runtime_state_after_action": "dict|None",
                    "public_runtime_state_trace": "list[dict]",
                    "action_step": "dict",
                    "controller_feedback": "dict",
                },
                "Read benchmark-native instruction, scene, robot, and sanitized IK/tracking evidence; private verifier fields are removed.",
            ),
            self._primitive_card(
                "observe_robodojo_visual",
                "L1",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "camera_names": "list[str]|None",
                    "agent_context": "dict|None",
                },
                {
                    "observation": "dict",
                    "visual_observation_contract": "dict",
                    "rgb_summaries": "dict",
                    "raw_frame_refs": "list[dict]",
                    "evidence_refs": "list[dict]",
                    "camera_names": "list[str]",
                },
                "Capture the current native RoboDojo three-view RGB observation and reusable raw-frame evidence references; summary-only inputs are rejected.",
            ),
            self._primitive_card(
                "get_robodojo_robot_action_schema",
                "L1",
                {"agent_context": "dict|None"},
                {
                    "action_type": "str",
                    "robot_action_schema": "dict",
                    "runtime_observation_fields": "list[str]",
                },
                "Expose only the public robot action dimensions and benchmark-native runtime observation fields.",
            ),
            self._primitive_card(
                "list_robodojo_policy_skills",
                "L1",
                {"agent_context": "dict|None"},
                {"default_policy_name": "str", "adapters": "list[dict]"},
                "List official XPolicyLab adapter capabilities and local checkpoint availability without selecting a task recipe.",
            ),
            self._primitive_card(
                "inspect_robodojo_policy_skill",
                "L1",
                {
                    "policy_name": "str|None",
                    "checkpoint_name": "str|None",
                    "server_url": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "adapter_ready": "bool",
                    "checkpoint_ready": "bool",
                    "operation_ready": "bool",
                },
                "Inspect one caller-selected XPolicyLab adapter, runtime, checkpoint, and server contract; demo zero-action wiring is explicitly non-operational.",
            ),
            self._primitive_card(
                "measure_robodojo_public_geometry",
                "L2",
                {
                    "runtime_state_before": "dict",
                    "runtime_state_after": "dict|None",
                    "object_labels": "list[str]|None",
                    "agent_context": "dict|None",
                },
                {
                    "objects": "dict",
                    "robots": "list[dict]",
                    "pairwise_distances": "dict",
                },
                "Measure object and robot geometry from public poses without selecting a task target or applying task thresholds.",
            ),
            self._primitive_card(
                "inspect_robodojo_collision_geometry",
                "L2",
                {
                    "runtime_state": "dict",
                    "arm": "left|right",
                    "agent_context": "dict|None",
                },
                {
                    "available": "bool",
                    "contacts": "list[dict]",
                    "local_contact_center": "list[float]",
                    "local_jaw_axis": "list[float]",
                    "local_approach_axis": "list[float]",
                    "contact_opening": "float",
                    "world_xyz": "list[float]",
                },
                "Inspect public URDF finger collision centers, jaw orientation, and contact opening for a caller-selected arm; no object or grasp pose is chosen.",
            ),
            self._primitive_card(
                "compile_robodojo_ee_path",
                "L3",
                {
                    "runtime_state": "dict",
                    "path_points": "list[dict]",
                    "arm": "left|right",
                    "contact_frame": "public_urdf|list[float]|None",
                    "max_cartesian_step": "float|None",
                    "agent_context": "dict|None",
                },
                {
                    "low_level_actions": "list[dict[str,list[float]]]",
                    "path_points": "list[dict]",
                    "path_contract": "dict",
                },
                "Compile caller-authored EE path points, contact axes, and physical jaw openings to schema-valid low-level actions using current robot state and public collision geometry, with optional bounded Cartesian interpolation.",
            ),
            self._primitive_card(
                "compile_robodojo_contact_lift_actions",
                "L3",
                {
                    "runtime_state": "dict",
                    "object_label": "str",
                    "arm": "left|right",
                    "approach_axis": "list[float]|str|dict",
                    "jaw_axis": "list[float]|str|dict",
                    "lift_distance": "float|None",
                    "lift_vector": "list[float]|None",
                    "precontact_distance": "float|None",
                    "contact_depth": "float|None",
                    "contact_position_offset": "list[float]|None",
                    "open_gripper_opening": "float|None",
                    "grasp_gripper_opening": "float|None",
                    "grasp_compression": "float",
                    "precontact_repeat_steps": "int",
                    "contact_repeat_steps": "int",
                    "grasp_repeat_steps": "int",
                    "lift_repeat_steps": "int",
                    "max_cartesian_step": "float|None",
                    "agent_context": "dict|None",
                },
                {
                    "low_level_actions": "list[dict[str,list[float]]]",
                    "path_points": "list[dict]",
                    "contact_lift_contract": "dict",
                },
                "Compile a caller-parameterized contact/lift action sequence from public object geometry and public robot collision geometry; the agent selects object, arm, axes, lift, openings, and repeats.",
            ),
            self._primitive_card(
                "build_robodojo_joint_action",
                "L3",
                {
                    "joint_targets": "dict|list",
                    "gripper_targets": "dict|float|list",
                    "agent_context": "dict|None",
                },
                {
                    "action": "dict[str,list[float]]",
                    "action_schema": "dict",
                    "action_dims": "dict",
                },
                "Encode one caller-specified low-level joint/gripper action; missing targets are rejected rather than filled by a hidden pose.",
            ),
            self._primitive_card(
                "measure_robodojo_contact_state",
                "L2",
                {
                    "runtime_state_before": "dict",
                    "runtime_state_after": "dict|None",
                    "object_label": "str",
                    "arm": "left|right",
                    "criteria": "dict|None",
                    "agent_context": "dict|None",
                },
                {
                    "contact_evidence": "dict",
                    "holding_evidence": "dict",
                    "stability_evidence": "dict",
                },
                "Record raw contact/holding/stability evidence and evaluate only caller-supplied criteria; no task completion rule is embedded.",
            ),
            self._primitive_card(
                "run_robodojo_policy_skill",
                "L3",
                {
                    "observation": "dict",
                    "agent_prompt": "str|None",
                    "policy_name": "str|None",
                    "checkpoint_name": "str|None",
                    "max_actions": "int",
                    "reset_policy": "bool",
                    "evidence_refs": "list[str|dict]|None",
                    "agent_context": "dict|None",
                },
                {
                    "actions": "list[dict]",
                    "env_step_executed": "bool",
                    "observations_after_action": "list[dict]",
                    "observation_evidence_refs": "list[dict]",
                    "action_chunk_evidence_refs": "list[dict]",
                },
                "Run one bounded action chunk through a caller-selected official XPolicyLab policy adapter. The caller controls the prompt and budget; no object, pose, or task plan is embedded.",
            ),
            self._primitive_card(
                "submit_robodojo_low_level_actions",
                "L3",
                {
                    "actions": "list[dict]",
                    "reobserve_after_each_action": "bool",
                    "control_horizon_frames": "int|None",
                    "execution_mode": "point_ik|curobo_trajectory",
                    "planner_arm": "left|right|None",
                    "evidence_refs": "list[str|dict]|None",
                    "agent_context": "dict|None",
                },
                {
                    "accepted": "bool",
                    "action_count": "int",
                    "action_dims": "list[dict]",
                    "live_action_payload": "dict",
                    "env_step_executed_by_this_primitive": "bool",
                    "execution_results": "list",
                    "observations_after_action": "list[dict]",
                    "controller_feedback": "dict",
                    "provenance": "dict",
                },
                "Validate caller-authored low-level actions and, when a live executor is bound, execute them in order with public-state reobservation and controller feedback.",
            ),
            self._primitive_card(
                "record_robodojo_evidence",
                "L1",
                {"key": "str", "value": "dict"},
                {"artifact_id": "str"},
                "Record compact evidence used by the coding agent.",
            ),
        ]
        if level:
            cards = [card for card in cards if card.abstraction_level == level]
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        handlers = {
            "observe_robodojo_runtime_report": self._observe_runtime_report,
            "observe_robodojo_visual": self._observe_visual,
            "get_robodojo_robot_action_schema": self._get_robot_action_schema,
            "list_robodojo_policy_skills": self._list_policy_skills,
            "inspect_robodojo_policy_skill": self._inspect_policy_skill,
            "measure_robodojo_public_geometry": self._measure_public_geometry,
            "inspect_robodojo_collision_geometry": self._inspect_collision_geometry,
            "compile_robodojo_ee_path": self._compile_ee_path,
            "compile_robodojo_contact_lift_actions": self._compile_contact_lift_actions,
            "build_robodojo_joint_action": self._build_joint_action,
            "measure_robodojo_contact_state": self._measure_contact_state,
            "run_robodojo_policy_skill": self._run_policy_skill,
            "submit_robodojo_low_level_actions": self._submit_low_level_actions,
            "record_robodojo_evidence": self._record_evidence,
        }
        handler = handlers.get(name)
        if handler is None:
            result = PrimitiveResult(
                name=name, ok=False, error=f"unknown_primitive:{name}"
            )
        else:
            result = handler(**kwargs)
        self.record_event(
            "primitive_call",
            {
                "name": name,
                "kwargs": _jsonable(kwargs),
                "result": _primitive_result_event_payload(result),
            },
        )
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        runtime = self.runtime_available()
        official_receipt = None
        official_verifier = self._official_verifier
        proxy = self.isaac_subprocess_live_general_pickup
        if scope == "task" and self.config.live and official_verifier is None and proxy is not None:
            # The batch runner calls backend.verify(). Capture the original reward
            # from the process that executed the submitted actions, just as the
            # evaluator's live-official path does. Keep the receipt evaluator-only.
            official_receipt = proxy.capture_live_official({
                "identity": {"case_id": ROBODOJO_LIVE_OFFICIAL_OPERATION,
                             "task_id": self._task_spec.task_id},
                "environment_digest": _robodojo_json_digest(self.config.to_dict()),
                "transcript_digest": _robodojo_json_digest(_jsonable(self.get_trace().to_dict())),
            })
            official_verifier = lambda: official_receipt["official_result"]
        if scope == "source_boundary":
            ok = bool(runtime["source_ready"] and self._submitted_plan)
            message = (
                "RoboDojo source-boundary plan accepted"
                if ok
                else "RoboDojo source-boundary evidence incomplete"
            )
        elif scope == "task":
            if self.config.live:
                if official_verifier is None:
                    ok = False
                    message = "RoboDojo live Isaac official verifier is not bound"
                else:
                    official = official_verifier()
                    if isinstance(official, VerificationResult):
                        ok = bool(official.ok)
                    elif isinstance(official, dict):
                        ok = bool(
                            official.get(
                                "official_success", official.get("success", False)
                            )
                        )
                    else:
                        ok = bool(official)
                    message = (
                        "RoboDojo live Isaac official verifier passed"
                        if ok
                        else "RoboDojo live Isaac official verifier did not pass"
                    )
            else:
                ok = False
                source_boundary_complete = bool(
                    self.config.source_boundary_ok
                    and runtime["source_ready"]
                    and self._submitted_plan
                )
                message = (
                    "RoboDojo source-boundary plan accepted, but the official live Isaac task verifier was not run"
                    if source_boundary_complete
                    else "RoboDojo official live Isaac task verifier was not run"
                )
        else:
            ok = bool(runtime["source_ready"])
            message = (
                "RoboDojo runtime source inspected"
                if ok
                else "RoboDojo source unavailable"
            )
        self.get_trace().final_status = "verified" if ok else "not_verified"
        return VerificationResult(
            ok=ok,
            scope=scope,
            message=message,
            metrics={
                "source_ready": runtime["source_ready"],
                "asset_ready": runtime["asset_ready"],
                "live_ready": runtime["live_ready"],
                "official_task_success_claimed": bool(
                    scope == "task"
                    and self.config.live
                    and official_verifier is not None
                    and ok
                ),
                "submitted_plan": self._submitted_plan is not None,
            },
            metadata={
                "runtime": runtime,
                "not_paper_official_ready": not runtime["live_ready"],
                **({"same_episode_native_verifier_receipt": official_receipt}
                   if official_receipt is not None else {}),
            },
        )

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            self._trace = EpisodeTrace(task_id=self.config.task_name)
        return self._trace

    def runtime_available(self) -> JsonDict:
        source = robodojo_source_preflight(
            self.config.repo_path, self.config.task_name, self.config.env_cfg
        )
        asset = robodojo_asset_gate(self.config.repo_path)
        env = robodojo_python_env_gate(self.config.sim_env)
        return {
            "source_ready": bool(source["ready"]),
            "asset_ready": bool(asset["ready"]),
            "python_env_ready": bool(env["ready"]),
            "live_ready": bool(
                source["ready"] and asset["ready"] and env["ready"] and self.config.live
            ),
            "source_preflight": source,
            "asset_gate": asset,
            "python_env_gate": env,
        }

    def _get_robot_action_schema(
        self, agent_context: JsonDict | None = None, **_: Any
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_robodojo_robot_action_schema",
            ok=True,
            output={**self._action_schema(), "agent_context": agent_context or {}},
        )

    def _policy_config(
        self,
        *,
        policy_name: str | None = None,
        checkpoint_name: str | None = None,
        server_url: str | None = None,
    ) -> RoboDojoPolicySkillConfig:
        return RoboDojoPolicySkillConfig(
            repo_path=self.config.repo_path,
            policy_name=str(policy_name or self.config.policy_name),
            checkpoint_name=str(
                checkpoint_name
                if checkpoint_name is not None
                else self.config.policy_checkpoint
            ),
            env_cfg=self.config.env_cfg,
            action_type=self.config.action_type,
            server_url=str(server_url or self.config.policy_server_url),
        )

    def _list_policy_skills(
        self,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        adapters = list_robodojo_policy_adapters(self.config.repo_path)
        return PrimitiveResult(
            name="list_robodojo_policy_skills",
            ok=bool(adapters),
            output={
                "default_policy_name": self.config.policy_name,
                "adapters": adapters,
                "agent_context": agent_context or {},
                "selection_performed_by_primitive": False,
                "demo_policy_operation_ready": False,
            },
            error=None if adapters else "xpolicylab_policy_adapters_missing",
        )

    def _inspect_policy_skill(
        self,
        policy_name: str | None = None,
        checkpoint_name: str | None = None,
        server_url: str | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        status = inspect_robodojo_policy_skill(
            self._policy_config(
                policy_name=policy_name,
                checkpoint_name=checkpoint_name,
                server_url=server_url,
            )
        )
        return PrimitiveResult(
            name="inspect_robodojo_policy_skill",
            ok=bool(status["adapter_ready"]),
            output={**status, "agent_context": agent_context or {}},
            error=None
            if status["adapter_ready"]
            else "xpolicylab_policy_adapter_incomplete",
        )

    def _run_policy_skill(
        self,
        observation: JsonDict | None = None,
        agent_prompt: str | None = None,
        policy_name: str | None = None,
        checkpoint_name: str | None = None,
        server_url: str | None = None,
        max_actions: int = 8,
        reset_policy: bool = False,
        evidence_refs: list[Any] | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        current_observation = observation
        retained_observation = (
            current_observation is None
            and self._current_runtime_observation is not None
        )
        if retained_observation:
            current_observation = self._current_runtime_observation
        elif current_observation is None and self._observation_provider is not None:
            current_observation = self._observation_provider()
        if current_observation is None:
            return PrimitiveResult(
                name="run_robodojo_policy_skill",
                ok=False,
                output={"agent_context": agent_context or {}},
                error="public_observation_required",
            )
        config = self._policy_config(
            policy_name=policy_name,
            checkpoint_name=checkpoint_name,
            server_url=server_url,
        )
        status = inspect_robodojo_policy_skill(config)
        if status["dummy_policy"]:
            return PrimitiveResult(
                name="run_robodojo_policy_skill",
                ok=False,
                output={"policy_status": status, "agent_context": agent_context or {}},
                error="demo_policy_is_zero_action_wiring_not_an_operation_skill",
            )
        if self._policy_client is None and not status["operation_ready"]:
            return PrimitiveResult(
                name="run_robodojo_policy_skill",
                ok=False,
                output={"policy_status": status, "agent_context": agent_context or {}},
                error="policy_checkpoint_or_runtime_not_ready",
            )
        if self._policy_client is None and not status["policy_server_ready"]:
            endpoint = status.get("policy_server_endpoint") or {}
            return PrimitiveResult(
                name="run_robodojo_policy_skill",
                ok=False,
                output={"policy_status": status, "agent_context": agent_context or {}},
                error=str(endpoint.get("blocker") or "policy_server_unreachable"),
            )
        try:
            policy_client = (
                self._policy_client
                if self._policy_client is not None and not self._policy_client_owned
                else self._policy_client_for_config(config)
            )
            output = run_robodojo_policy_skill(
                config,
                observation=current_observation,
                agent_prompt=agent_prompt,
                agent_context=agent_context,
                max_actions=max_actions,
                reset_policy=reset_policy,
                client=policy_client,
                action_validator=lambda actions: validate_robodojo_low_level_actions(
                    self.config.repo_path,
                    self.config.env_cfg,
                    actions=actions,
                    runtime_state=self._current_runtime_observation,
                ),
                action_executor=self._action_executor,
                observation_provider=self._observation_provider,
                evidence_refs=(
                    evidence_refs
                    if evidence_refs is not None
                    else (
                        self._current_visual_evidence_refs
                        if retained_observation
                        else None
                    )
                ),
            )
        except Exception as exc:
            return PrimitiveResult(
                name="run_robodojo_policy_skill",
                ok=False,
                output={"policy_status": status, "agent_context": agent_context or {}},
                error=f"policy_skill_failed:{type(exc).__name__}:{exc}",
            )
        artifact_id = "robodojo:policy_skill:last_chunk"
        observations_after_action = output.get("observations_after_action") or []
        if observations_after_action and isinstance(
            observations_after_action[-1], dict
        ):
            self._current_runtime_observation = observations_after_action[-1]
        self.get_trace().add_artifact(artifact_id, output)
        execution_ok = bool(output.get("execution_ok", True))
        return PrimitiveResult(
            name="run_robodojo_policy_skill",
            ok=bool(output["actions"]) and execution_ok,
            output={**output, "policy_status": status, "artifact_id": artifact_id},
            error=None if execution_ok else "policy_action_execution_rejected",
        )

    def _policy_client_for_config(
        self, config: RoboDojoPolicySkillConfig
    ) -> XPolicyLabClient:
        key = (
            config.repo_path,
            config.policy_name,
            config.checkpoint_name,
            config.env_cfg,
            config.action_type,
            config.server_url,
        )
        if (
            self._policy_client is not None
            and self._policy_client_owned
            and self._policy_client_key == key
        ):
            return self._policy_client
        if (
            self._policy_client is not None
            and self._policy_client_owned
            and hasattr(self._policy_client, "close")
        ):
            try:
                self._policy_client.close()  # type: ignore[attr-defined]
            except Exception:
                pass
        self._policy_client = create_robodojo_policy_client(config)
        self._policy_client_owned = True
        self._policy_client_key = key
        return self._policy_client

    def _build_joint_action(
        self,
        joint_targets: Any | None = None,
        gripper_targets: Any | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        try:
            payload = build_robodojo_joint_action(
                self.config.repo_path,
                self.config.env_cfg,
                joint_targets=joint_targets,
                gripper_targets=gripper_targets,
                agent_context=agent_context,
            )
        except Exception as exc:
            return PrimitiveResult(
                name="build_robodojo_joint_action",
                ok=False,
                output={"agent_context": agent_context or {}},
                error=f"joint_action_build_failed:{type(exc).__name__}:{exc}",
            )
        return PrimitiveResult(
            name="build_robodojo_joint_action", ok=True, output=payload
        )

    def _observe_runtime_report(
        self,
        report_path: str | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        path = report_path or str(
            (self.config.env_kwargs or {}).get("latest_live_report_path") or ""
        )
        live_general_pickup = (
            self.config.live and self.config.task_name == "general_pickup"
        )
        configured_payload = load_robodojo_runtime_report(path) if path else {}
        configured_public_state = extract_robodojo_public_runtime_state(
            configured_payload
        )
        configured_current_state = configured_public_state.get(
            "public_runtime_state"
        ) or configured_public_state.get("public_runtime_state_after_action")
        if (
            not isinstance(configured_current_state, dict)
            and self._current_runtime_observation is None
            and live_general_pickup
        ):
            # The built-in case retains the last host report path for offline
            # replay.  That path is intentionally absent from the strict
            # Agent Server capsule, so a live run must start a fresh Isaac
            # episode instead of returning a stale-report failure.
            self._ensure_live_isaac_observation_episode()
            path = (
                str(self._isaac_report_path)
                if self._isaac_report_path is not None
                else ""
            )
        if not path and self._isaac_report_path is not None:
            path = str(self._isaac_report_path)
        payload = (
            load_robodojo_runtime_report(path)
            if path
            else {
                "ok": bool(self._current_runtime_observation),
                "public_runtime_state": self._current_runtime_observation,
            }
        )
        public_state = extract_robodojo_public_runtime_state(payload)
        current_state = public_state.get("public_runtime_state") or public_state.get(
            "public_runtime_state_after_action"
        )
        if isinstance(current_state, dict):
            self._current_runtime_observation = current_state
        visual_contract = _robodojo_visual_observation_contract(current_state)
        ok = bool(
            public_state.get("public_runtime_state")
            or public_state.get("public_runtime_state_after_action")
        )
        return PrimitiveResult(
            name="observe_robodojo_runtime_report",
            ok=ok,
            output={
                **public_state,
                "visual_observation_contract": visual_contract,
                "report_path": path or None,
                "agent_context": agent_context or {},
                "official_task_success_claimed": False,
                "agent_visible_success_checker": False,
                "oracle_or_demo_replay_used": False,
            },
            error=None if ok else "public_runtime_state_missing",
        )

    def _observe_visual(
        self,
        prompt: str | None = None,
        query: str | None = None,
        camera_names: list[str] | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        current: Any = None
        if (
            self._observation_provider is None
            and self._current_runtime_observation is None
            and self.config.live
            and self.config.task_name == "general_pickup"
        ):
            self._ensure_live_isaac_observation_episode()
        if self._observation_provider is not None:
            current = self._observation_provider()
            if isinstance(current, dict):
                self._current_runtime_observation = current
        elif self._current_runtime_observation is not None:
            current = self._current_runtime_observation
        if not isinstance(current, dict):
            return PrimitiveResult(
                name="observe_robodojo_visual",
                ok=False,
                output={
                    "evidence_refs": [],
                    "agent_context": agent_context or {},
                    "query": query,
                    "frames_fabricated": False,
                },
                error="native_runtime_observation_unavailable",
            )
        observation = deepcopy(_robodojo_policy_observation_from_runtime(current))
        if prompt is not None:
            instruction = prompt.strip()
            if not instruction:
                return PrimitiveResult(
                    name="observe_robodojo_visual",
                    ok=False,
                    output={
                        "evidence_refs": [],
                        "agent_context": agent_context or {},
                        "query": query,
                    },
                    error="prompt_cannot_be_empty",
                )
            observation["instruction"] = instruction
        visual_contract, raw_refs = inspect_robodojo_rgb_observation(observation)
        if not visual_contract["ready"]:
            return PrimitiveResult(
                name="observe_robodojo_visual",
                ok=False,
                output={
                    "observation": observation,
                    "visual_observation_contract": visual_contract,
                    "evidence_refs": [],
                    "agent_context": agent_context or {},
                    "query": query,
                    "frames_fabricated": False,
                },
                error="native_three_view_rgb_observation_required",
            )
        requested = list(camera_names or visual_contract["required_views"])
        unknown = sorted(set(requested) - set(visual_contract["required_views"]))
        if not requested or unknown:
            return PrimitiveResult(
                name="observe_robodojo_visual",
                ok=False,
                output={
                    "available_camera_names": visual_contract["required_views"],
                    "unknown_camera_names": unknown,
                },
                error="camera_names_must_select_native_rgb_views",
            )
        evidence_refs: list[JsonDict] = []
        for raw_ref in raw_refs:
            if raw_ref["camera"] not in requested:
                continue
            artifact_id = (
                f"robodojo:visual:{raw_ref['camera']}:{len(self.get_trace().artifacts)}"
            )
            raw_frame = _robodojo_value_at_path(
                observation, str(raw_ref["source_path"])
            )
            self.get_trace().add_artifact(
                artifact_id,
                {
                    "camera": raw_ref["camera"],
                    "source_path": raw_ref["source_path"],
                    "raw_rgb_frame_metadata": _raw_rgb_frame_metadata(raw_frame),
                    "raw_rgb_frame": _jsonable(raw_frame),
                    "frames_fabricated": False,
                },
            )
            evidence_refs.append({**raw_ref, "artifact_id": artifact_id})
        output = {
            "observation": observation,
            "visual_observation_contract": visual_contract,
            "rgb_summaries": visual_contract["views"],
            "camera_names": requested,
            "raw_frame_refs": evidence_refs,
            "evidence_refs": evidence_refs,
            "prompt_applied": prompt is not None,
            "query": query,
            "agent_context": agent_context or {},
            "frames_fabricated": False,
            "official_task_success_claimed": False,
        }
        self._current_runtime_observation = observation
        self._current_visual_evidence_refs = evidence_refs
        return PrimitiveResult(
            name="observe_robodojo_visual",
            ok=True,
            output=output,
            artifacts=[ref["artifact_id"] for ref in evidence_refs],
        )

    def _measure_public_geometry(
        self,
        runtime_state_before: JsonDict | None = None,
        runtime_state_after: JsonDict | None = None,
        object_labels: list[str] | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        payload = measure_robodojo_public_geometry(
            runtime_state_before,
            runtime_state_after,
            object_labels=object_labels,
            agent_context=agent_context,
        )
        return PrimitiveResult(
            name="measure_robodojo_public_geometry",
            ok=bool(payload["objects"] or payload["robots"]),
            output=payload,
            error=None
            if payload["objects"] or payload["robots"]
            else "public_geometry_missing",
        )

    def _inspect_collision_geometry(
        self,
        runtime_state: JsonDict | None = None,
        arm: str | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        runtime = _robodojo_public_runtime_state(runtime_state)
        side = str(arm or "").strip().lower()
        if side not in {"left", "right"}:
            return PrimitiveResult(
                name="inspect_robodojo_collision_geometry",
                ok=False,
                output={"agent_context": agent_context or {}},
                error="caller_selected_arm_required",
            )
        payload = robodojo_public_tool_contact_offset(
            self.config.repo_path,
            robots=_robodojo_runtime_robots(runtime),
            active_side=side,
        )
        return PrimitiveResult(
            name="inspect_robodojo_collision_geometry",
            ok=bool(payload.get("available")),
            output={**payload, "agent_context": agent_context or {}},
            error=None
            if payload.get("available")
            else str(payload.get("error") or "collision_geometry_missing"),
        )

    def _compile_ee_path(
        self,
        runtime_state: JsonDict | None = None,
        path_points: list[JsonDict] | None = None,
        arm: str | None = None,
        contact_frame: Any | None = None,
        max_cartesian_step: float | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        try:
            payload = compile_robodojo_ee_path(
                self.config.repo_path,
                self.config.env_cfg,
                runtime_state=runtime_state,
                path_points=path_points,
                arm=arm,
                contact_frame=contact_frame,
                max_cartesian_step=max_cartesian_step,
                agent_context=agent_context,
            )
        except Exception as exc:
            return PrimitiveResult(
                name="compile_robodojo_ee_path",
                ok=False,
                output={
                    "path_point_count": len(path_points or []),
                    "agent_context": agent_context or {},
                },
                error=f"ee_path_compile_failed:{type(exc).__name__}:{exc}",
            )
        return PrimitiveResult(
            name="compile_robodojo_ee_path",
            ok=bool(payload["low_level_actions"]),
            output=payload,
            error=None
            if payload["low_level_actions"]
            else str(payload.get("error") or "empty_ee_path"),
        )

    def _compile_contact_lift_actions(
        self,
        runtime_state: JsonDict | None = None,
        object_label: str | None = None,
        arm: str | None = None,
        approach_axis: Any | None = None,
        jaw_axis: Any | None = None,
        lift_distance: float | None = None,
        lift_vector: list[float] | None = None,
        lift_axis: Any | None = None,
        precontact_distance: float | None = None,
        contact_depth: float | None = None,
        contact_position_offset: list[float] | None = None,
        open_gripper_opening: float | None = None,
        grasp_gripper_opening: float | None = None,
        grasp_compression: float = 0.0,
        precontact_repeat_steps: int = 1,
        contact_repeat_steps: int = 1,
        grasp_repeat_steps: int = 2,
        lift_repeat_steps: int = 2,
        max_cartesian_step: float | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        try:
            payload = compile_robodojo_contact_lift_actions(
                self.config.repo_path,
                self.config.env_cfg,
                runtime_state=runtime_state,
                object_label=object_label,
                arm=arm,
                approach_axis=approach_axis,
                jaw_axis=jaw_axis,
                lift_distance=lift_distance,
                lift_vector=lift_vector,
                lift_axis=lift_axis,
                precontact_distance=precontact_distance,
                contact_depth=contact_depth,
                contact_position_offset=contact_position_offset,
                open_gripper_opening=open_gripper_opening,
                grasp_gripper_opening=grasp_gripper_opening,
                grasp_compression=grasp_compression,
                precontact_repeat_steps=precontact_repeat_steps,
                contact_repeat_steps=contact_repeat_steps,
                grasp_repeat_steps=grasp_repeat_steps,
                lift_repeat_steps=lift_repeat_steps,
                max_cartesian_step=max_cartesian_step,
                agent_context=agent_context,
            )
        except Exception as exc:
            return PrimitiveResult(
                name="compile_robodojo_contact_lift_actions",
                ok=False,
                output={"agent_context": agent_context or {}},
                error=f"contact_lift_compile_failed:{type(exc).__name__}:{exc}",
            )
        return PrimitiveResult(
            name="compile_robodojo_contact_lift_actions",
            ok=bool(payload["low_level_actions"]),
            output=payload,
            error=None
            if payload["low_level_actions"]
            else str(payload.get("error") or "empty_contact_lift_actions"),
        )

    def _measure_contact_state(
        self,
        runtime_state_before: JsonDict | None = None,
        runtime_state_after: JsonDict | None = None,
        object_label: str | None = None,
        arm: str | None = None,
        criteria: JsonDict | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        payload = measure_robodojo_contact_state(
            runtime_state_before,
            runtime_state_after,
            object_label=object_label,
            arm=arm,
            criteria=criteria,
            repo_path=self.config.repo_path,
            agent_context=agent_context,
        )
        return PrimitiveResult(
            name="measure_robodojo_contact_state",
            ok=bool(payload["contact_evidence"].get("object_position_after")),
            output=payload,
            error=None
            if payload["contact_evidence"].get("object_position_after")
            else "public_contact_evidence_missing",
        )

    def _submit_low_level_actions(
        self,
        actions: list[JsonDict] | None = None,
        reobserve_after_each_action: bool = False,
        control_horizon_frames: int | None = None,
        execution_mode: str = "point_ik",
        planner_arm: str | None = None,
        evidence_refs: list[Any] | None = None,
        agent_context: JsonDict | None = None,
        **_: Any,
    ) -> PrimitiveResult:
        try:
            clean_actions = validate_robodojo_low_level_actions(
                self.config.repo_path,
                self.config.env_cfg,
                actions=actions,
                runtime_state=self._current_runtime_observation,
            )
            normalized_evidence_refs = _normalize_robodojo_evidence_refs(evidence_refs)
            normalized_control_horizon = None
            if control_horizon_frames is not None:
                if isinstance(control_horizon_frames, bool):
                    raise ValueError("control_horizon_frames_must_be_positive_integer")
                normalized_control_horizon = int(control_horizon_frames)
                if (
                    normalized_control_horizon <= 0
                    or normalized_control_horizon != control_horizon_frames
                ):
                    raise ValueError("control_horizon_frames_must_be_positive_integer")
            normalized_execution_mode = (
                str(execution_mode or "point_ik").strip().lower()
            )
            if normalized_execution_mode not in {"point_ik", "curobo_trajectory"}:
                raise ValueError("execution_mode_must_be_point_ik_or_curobo_trajectory")
            normalized_planner_arm = None
            if planner_arm is not None:
                normalized_planner_arm = (
                    str(planner_arm).strip().lower().removesuffix("_arm")
                )
                if normalized_planner_arm not in {"left", "right"}:
                    raise ValueError("planner_arm_must_be_left_or_right")
            if (
                normalized_execution_mode == "curobo_trajectory"
                and normalized_planner_arm is None
            ):
                raise ValueError("planner_arm_required_for_curobo_trajectory")
        except Exception as exc:
            return PrimitiveResult(
                name="submit_robodojo_low_level_actions",
                ok=False,
                output={"accepted": False, "agent_context": agent_context or {}},
                error=f"low_level_action_validation_failed:{type(exc).__name__}:{exc}",
            )
        live_action_payload = {
            "low_level_actions": clean_actions,
            "reobserve_after_each_action": bool(reobserve_after_each_action),
        }
        if normalized_control_horizon is not None:
            live_action_payload["control_horizon_frames"] = normalized_control_horizon
        if normalized_planner_arm is not None:
            live_action_payload["execution_mode"] = normalized_execution_mode
            live_action_payload["planner_arm"] = normalized_planner_arm
        action_provenance = {
            "evidence_refs": normalized_evidence_refs,
            "agent_context": agent_context or {},
            "task_recipe_embedded": False,
        }
        self._submitted_plan = {
            "env_cfg": self.config.env_cfg,
            **live_action_payload,
            "provenance": action_provenance,
        }
        artifact_id = "robodojo:submitted_low_level_actions"
        execution_results: list[Any] = []
        observations_after_action: list[JsonDict] = []
        execution_error: str | None = None
        if self._action_executor is not None:
            try:
                for action in clean_actions:
                    execution_result = _jsonable(self._action_executor(action))
                    execution_results.append(execution_result)
                    if (
                        reobserve_after_each_action
                        and self._observation_provider is not None
                    ):
                        observation = _jsonable(self._observation_provider())
                        if isinstance(observation, dict):
                            observations_after_action.append(observation)
                            self._current_runtime_observation = observation
                    incremental_feedback = _summarize_robodojo_controller_feedback(
                        [execution_result]
                    )
                    if not _robodojo_execution_result_ok(execution_result) or (
                        incremental_feedback["diagnostic_count"]
                        and not incremental_feedback["all_ik_succeeded"]
                    ):
                        break
            except Exception as exc:
                execution_error = (
                    f"low_level_action_execution_failed:{type(exc).__name__}:{exc}"
                )
        elif self.config.live and self.config.task_name == "general_pickup":
            try:
                live_result = self._start_live_isaac_action_episode(
                    self._submitted_plan
                )
                execution_results.extend(
                    {**live_result, "action_index": index}
                    for index in range(len(clean_actions))
                )
            except Exception as exc:
                self._close_live_isaac_subprocess()
                execution_error = (
                    f"low_level_action_execution_failed:{type(exc).__name__}:{exc}"
                )
                self._isaac_terminal_error = execution_error
        controller_feedback = _summarize_robodojo_controller_feedback(execution_results)
        executed = any(not isinstance(item, dict) or item.get("env_step_executed", True)
                       for item in execution_results)
        execution_complete = execution_error is None and (
            self._action_executor is None or len(execution_results) == len(clean_actions)
        ) and all(not isinstance(item, dict) or item.get("execution_complete", True)
                  for item in execution_results)
        controller_rejected = bool(
            controller_feedback["diagnostic_count"]
        ) and not bool(controller_feedback["all_ik_succeeded"])
        accepted = (
            execution_complete
            and not controller_rejected
            and all(_robodojo_execution_result_ok(item) for item in execution_results)
        )
        stored_artifact = {
            **self._submitted_plan,
            "execution_results": execution_results,
            "observations_after_action": observations_after_action,
            "controller_feedback": controller_feedback,
        }
        self.get_trace().add_artifact(artifact_id, stored_artifact)
        return PrimitiveResult(
            name="submit_robodojo_low_level_actions",
            ok=accepted,
            output={
                "accepted": accepted,
                "action_count": len(clean_actions),
                "action_dims": robodojo_action_dims(clean_actions),
                "submission_channel": (
                    "live-action-executor"
                    if self._action_executor is not None
                    else (
                        "live-isaac-subprocess"
                        if self.config.live
                        and self.config.task_name == "general_pickup"
                        else "trace-only-boundary"
                    )
                ),
                "env_step_executed_by_this_primitive": executed,
                "execution_complete": execution_complete,
                "execution_results": execution_results,
                "observations_after_action": observations_after_action,
                "controller_feedback": controller_feedback,
                "artifact_id": artifact_id,
                "live_action_payload": live_action_payload,
                "provenance": action_provenance,
                "agent_context": agent_context or {},
            },
            error=execution_error
            or (
                "low_level_action_controller_ik_failed" if controller_rejected else None
            )
            or (None if accepted else "low_level_action_executor_rejected_action"),
        )

    def _ensure_live_isaac_observation_episode(self) -> JsonDict:
        if self._current_runtime_observation is not None:
            return self._current_runtime_observation
        if self._isaac_process is None:
            self._start_live_isaac_observation_episode()
        if self._current_runtime_observation is None:
            raise RuntimeError("RoboDojo live Isaac episode has no public reset state")
        return self._current_runtime_observation

    def _start_live_isaac_observation_episode(self) -> JsonDict:
        """Launch Isaac once and pause after reset for an agent-authored action."""

        if self._isaac_terminal_error is not None:
            raise RuntimeError("RoboDojo episode terminated; explicit reset required: " + self._isaac_terminal_error)
        if self._isaac_process is not None:
            raise RuntimeError(
                "RoboDojo live Isaac observation episode already started"
            )
        project_root = get_project_paths().project_root
        repo = Path(self.config.repo_path).resolve()
        python = Path(
            os.environ.get(
                "AGENTIC_EMBODIED_ARENA_NATIVE_PYTHON_ROBODOJO",
                str(project_root / "external/environments/robodojo/bin/python3.11"),
            )
        ).expanduser().absolute()
        probe = project_root / "scripts/robodojo_live_reset_probe.py"
        experience_source = repo / "third_party/IsaacLab/apps/isaaclab.python.kit"
        for required in (python, probe, experience_source):
            if not required.is_file():
                raise RuntimeError(
                    f"RoboDojo live Isaac dependency missing: {required}"
                )
        runtime_base = Path(os.environ.get("ROBODOJO_LIVE_RUNTIME_BASE", "/tmp"))
        if not runtime_base.is_absolute() or runtime_base.is_symlink():
            raise RuntimeError(
                "RoboDojo live runtime base must be an absolute non-symlink path"
            )
        runtime_base.mkdir(parents=True, exist_ok=True)
        runtime_dir = Path(tempfile.mkdtemp(prefix="rd-live-", dir=runtime_base))
        os.chmod(runtime_dir, 0o700)
        report_path = runtime_dir / "report.json"
        action_socket_path = runtime_dir / "action.sock"
        socket_path = runtime_dir / "official.sock"
        stdout_path = runtime_dir / "stdout.log"
        stderr_path = runtime_dir / "stderr.log"
        experience_path = runtime_dir / "isaaclab.python.kit"
        portable_root = runtime_dir / "kit-portable"
        portable_root.mkdir()
        experience = experience_source.read_text(encoding="utf-8")
        isaaclab_root = experience_source.parent.parent
        experience_path.write_text(
            experience.replace(
                '    "${app}", # needed to find other app files',
                f'    "{isaaclab_root / "apps"}", # locked IsaacLab app files',
            ).replace(
                '    "${app}/../source", # needed to find extensions in Isaac Lab',
                f'    "{isaaclab_root / "source"}", # locked IsaacLab extensions',
            ),
            encoding="utf-8",
        )
        action_auth_token = secrets.token_hex(32)
        official_auth_token = secrets.token_hex(32)
        episode_nonce = secrets.token_hex(32)
        action_auth_digest = hashlib.sha256(action_auth_token.encode()).hexdigest()
        official_auth_digest = hashlib.sha256(official_auth_token.encode()).hexdigest()
        device_id = int(self.config.env_kwargs.get("robodojo_device_id", 0))
        command = [
            str(python),
            str(probe),
            "--repo-path",
            str(repo),
            "--task-name",
            "general_pickup",
            "--env-cfg",
            self.config.env_cfg,
            "--num-envs",
            "1",
            "--seed",
            str(int(self.config.env_kwargs.get("seed", 0))),
            "--robodojo-device-id",
            str(device_id),
            "--persistent-live-episode",
            "--require-action-step",
            "--evaluate-task-success-after-action",
            "--action-repeat-steps",
            "1",
            "--live-action-socket",
            str(action_socket_path),
            "--live-action-auth-sha256",
            action_auth_digest,
            "--live-action-episode-nonce",
            episode_nonce,
            "--live-official-socket",
            str(socket_path),
            "--live-official-auth-sha256",
            official_auth_digest,
            "--live-official-episode-nonce",
            episode_nonce,
            "--output",
            str(report_path),
            "--include-camera-observations",
            "--experience",
            str(experience_path),
            "--kit_args",
            f"--portable-root {portable_root} --/app/extensions/fsWatcherEnabled=false",
            "--headless",
            "--enable_cameras",
            "--device",
            f"cuda:{device_id}",
        ]
        if self.config.env_kwargs.get("selected_layout_file"):
            command.extend(["--layout-file-name", self.config.env_kwargs["selected_layout_file"]])
        environment = os.environ.copy()
        environment.update(
            PYTHONNOUSERSITE="1",
            OMNI_KIT_ACCEPT_EULA="YES",
            ROBODOJO_ENABLE_CAMERAS="1",
            ROBODOJO_RAW_CAMERA_SIDECARS="1",
            ROBODOJO_STACK_DUMP_SECONDS="0",
            ROBODOJO_OWNER_PID=str(os.getpid()),
            PYTHONPATH=":".join(
                (str(repo), str(repo / "XPolicyLab"), str(project_root / "src"))
            ),
        )
        self._isaac_stdout = stdout_path.open("w", encoding="utf-8")
        self._isaac_stderr = stderr_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=project_root,
            env=environment,
            stdout=self._isaac_stdout,
            stderr=self._isaac_stderr,
            text=True,
        )
        self._isaac_process = process
        self._isaac_runtime_dir = runtime_dir
        self._isaac_report_path = report_path
        self._isaac_action_socket_path = action_socket_path
        self._isaac_action_auth_token = action_auth_token
        self._isaac_episode_nonce = episode_nonce
        self._isaac_official_socket_path = socket_path
        self._isaac_official_auth_token = official_auth_token
        timeout = float(
            self.config.env_kwargs.get("live_start_timeout_seconds", 3600.0)
        )
        deadline = time.monotonic() + timeout
        binding: JsonDict | None = None
        report: JsonDict = {}
        while time.monotonic() < deadline:
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    report = {}
                candidate = report.get("live_action_binding")
                if (
                    report.get("stage") == "awaiting_live_agent_action"
                    and isinstance(candidate, dict)
                    and action_socket_path.is_socket()
                ):
                    binding = candidate
                    break
                if report.get("stage") in {"action_step_failed", "cleanup_failed"}:
                    raise RuntimeError(
                        f"RoboDojo reset episode failed at {report.get('stage')}: {report.get('error')}"
                    )
            return_code = process.poll()
            if return_code is not None:
                diagnostic = _robodojo_live_exit_diagnostic(
                    report_path=report_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                )
                raise RuntimeError(
                    f"RoboDojo live Isaac exited before capture: {return_code}: {diagnostic}"
                )
            time.sleep(0.1)
        if binding is None or not action_socket_path.is_socket():
            diagnostic = _robodojo_live_exit_diagnostic(
                report_path=report_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
            raise TimeoutError(
                "RoboDojo live Isaac did not reach the agent-action boundary: "
                f"{diagnostic}"
            )
        if binding.get("episode_instance_nonce") != episode_nonce or not isinstance(
            binding.get("probe_runtime"), dict
        ):
            raise RuntimeError("RoboDojo live Isaac reset binding mismatch")
        public = extract_robodojo_public_runtime_state(
            load_robodojo_runtime_report(report_path)
        ).get("public_runtime_state")
        if not isinstance(public, dict):
            raise RuntimeError(
                "RoboDojo live Isaac reset did not expose public runtime state"
            )
        self._current_runtime_observation = public
        return public

    def _start_live_isaac_action_episode(self, submitted_plan: JsonDict) -> JsonDict:
        """Submit actions to the already-observed live Isaac episode."""

        if self._isaac_process is None:
            self._start_live_isaac_observation_episode()
        process = self._isaac_process
        report_path = self._isaac_report_path
        action_socket_path = self._isaac_action_socket_path
        action_auth_token = self._isaac_action_auth_token
        episode_nonce = self._isaac_episode_nonce
        socket_path = self._isaac_official_socket_path
        official_auth_token = self._isaac_official_auth_token
        if (
            process is None
            or report_path is None
            or action_socket_path is None
            or action_auth_token is None
            or episode_nonce is None
            or socket_path is None
            or official_auth_token is None
        ):
            raise RuntimeError("RoboDojo live Isaac action channel is incomplete")
        if process.poll() is not None or not action_socket_path.is_socket():
            raise RuntimeError("RoboDojo live Isaac action channel is unavailable")

        envelope = {
            "authorization": action_auth_token,
            "episode_instance_nonce": episode_nonce,
            "action_payload": submitted_plan,
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(
                float(self.config.env_kwargs.get("live_action_timeout_seconds", 900.0))
            )
            client.connect(str(action_socket_path))
            client.sendall(
                (
                    json.dumps(
                        envelope,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
            )
            raw = b""
            while b"\n" not in raw and len(raw) <= 1024 * 1024:
                chunk = client.recv(65536)
                if not chunk:
                    break
                raw += chunk
        if b"\n" not in raw:
            raise RuntimeError(
                "RoboDojo live Isaac action acknowledgement is incomplete"
            )
        acknowledgement = json.loads(raw.split(b"\n", 1)[0])
        actions_digest = _robodojo_json_digest(submitted_plan)
        if (
            not isinstance(acknowledgement, dict)
            or not isinstance(acknowledgement.get("ok"), bool)
            or acknowledgement.get("episode_instance_nonce") != episode_nonce
            or acknowledgement.get("submitted_actions_digest") != actions_digest
        ):
            raise RuntimeError("RoboDojo live Isaac action acknowledgement mismatch")

        timeout = float(
            self.config.env_kwargs.get("live_action_timeout_seconds", 900.0)
        )
        deadline = time.monotonic() + timeout
        binding: JsonDict | None = None
        report: JsonDict = {}
        while time.monotonic() < deadline:
            if report_path.is_file():
                try:
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    report = {}
                candidate = report.get("live_official_binding")
                if (
                    report.get("stage") in {"awaiting_live_official_capture", "awaiting_live_agent_action"}
                    and isinstance(candidate, dict)
                    and candidate.get("submitted_actions_digest") == actions_digest
                    and socket_path.is_socket()
                ):
                    binding = candidate
                    break
                if report.get("stage") == "action_step_failed":
                    action_step = report.get("action_step")
                    detail = (
                        action_step.get("error")
                        if isinstance(action_step, dict)
                        else report.get("error")
                    )
                    raise RuntimeError(f"RoboDojo action step failed: {detail}")
            return_code = process.poll()
            if return_code is not None:
                runtime_dir = self._isaac_runtime_dir or report_path.parent
                diagnostic = _robodojo_live_exit_diagnostic(
                    report_path=report_path,
                    stdout_path=runtime_dir / "stdout.log",
                    stderr_path=runtime_dir / "stderr.log",
                )
                raise RuntimeError(
                    f"RoboDojo live Isaac exited before official capture: {return_code}: {diagnostic}"
                )
            time.sleep(0.1)
        if binding is None or not socket_path.is_socket():
            raise TimeoutError(
                "RoboDojo live Isaac did not reach the official capture boundary"
            )
        if (
            binding.get("episode_instance_nonce") != episode_nonce
            or binding.get("submitted_actions_digest") != actions_digest
            or not isinstance(binding.get("probe_runtime"), dict)
        ):
            raise RuntimeError("RoboDojo live Isaac episode binding mismatch")
        public_after = extract_robodojo_public_runtime_state(
            load_robodojo_runtime_report(report_path)
        ).get("public_runtime_state_after_action")
        if isinstance(public_after, dict):
            self._current_runtime_observation = public_after
        self.isaac_subprocess_live_general_pickup = RoboDojoIsaacLiveOfficialProxy(
            socket_path=socket_path,
            auth_token=official_auth_token,
            episode_instance_nonce=episode_nonce,
            probe_runtime=dict(binding["probe_runtime"]),
            submitted_actions_digest=actions_digest,
            timeout_seconds=float(
                self.config.env_kwargs.get("live_capture_timeout_seconds", 900.0)
            ),
            on_complete=None if report.get("persistent_live_episode") else self._close_live_isaac_subprocess,
            persistent_episode=bool(report.get("persistent_live_episode")),
        )
        return {
            "accepted": acknowledgement["ok"],
            "env_step_executed": bool((report.get("action_step") or {}).get("executed_step_count", 0)),
            "execution_complete": acknowledgement["ok"],
            "error": (report.get("action_step") or {}).get("error") if not acknowledgement["ok"] else None,
            "same_episode_live_official_ready": True,
            "episode_instance_nonce": episode_nonce,
            "submitted_actions_digest": actions_digest,
            "probe_runtime": binding["probe_runtime"],
        }

    def _close_live_isaac_subprocess(self) -> None:
        process = self._isaac_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                # The parent bridge allows 10 seconds for worker shutdown.
                # Reap Isaac before that deadline so the worker is not killed
                # first, leaving a live simulator orphan on the GPU.
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        for stream_name in ("_isaac_stdout", "_isaac_stderr"):
            stream = getattr(self, stream_name)
            if stream is not None:
                stream.close()
                setattr(self, stream_name, None)
        self._isaac_process = None
        runtime_dir = self._isaac_runtime_dir
        self._isaac_runtime_dir = None
        self._isaac_report_path = None
        self._isaac_action_socket_path = None
        self._isaac_action_auth_token = None
        self._isaac_episode_nonce = None
        self._isaac_official_socket_path = None
        self._isaac_official_auth_token = None
        self.isaac_subprocess_live_general_pickup = None
        if runtime_dir is not None and os.environ.get(
            "ROBODOJO_PRESERVE_LIVE_RUNTIME_DIAGNOSTICS"
        ) not in {"1", "true", "TRUE", "yes", "on"}:
            shutil.rmtree(runtime_dir, ignore_errors=True)

    def _record_evidence(self, key: str, value: Any, **_: Any) -> PrimitiveResult:
        artifact_id = f"robodojo:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, _jsonable(value))
        return PrimitiveResult(
            name="record_robodojo_evidence",
            ok=True,
            output={"artifact_id": artifact_id},
        )

    def _action_schema(self) -> JsonDict:
        robot_action_schema = robodojo_robot_action_schema(
            self.config.repo_path, self.config.env_cfg
        )
        runtime = _robodojo_public_runtime_state(self._current_runtime_observation)
        robot_action_schema = _robodojo_action_schema_with_public_runtime_fallback(
            robot_action_schema,
            robots=_robodojo_runtime_robots(runtime),
        )
        return {
            "action_type": self.config.action_type,
            "robot_action_schema": robot_action_schema,
            "runtime_observation_fields": [
                "observation.instruction",
                "scene.objects[].pose",
                "scene.objects[].bbox",
                "robot.robots[].joint_state",
                "robot.robots[].gripper_state",
                "robot.robots[].ee_pose",
                "observation.state",
                "observation.vision_summary",
                "observation.vision.cam_head.color|rgb",
                "observation.vision.cam_left_wrist.color|rgb",
                "observation.vision.cam_right_wrist.color|rgb",
            ],
            "pi05_visual_contract": {
                "required_rgb_views": ["cam_head", "cam_left_wrist", "cam_right_wrist"],
                "raw_rgb_frames_required": True,
                "summary_only_visual_input_accepted": False,
            },
            "task_synthesis_exposed": False,
            "verifier_exposed": False,
        }

    def _primitive_card(
        self,
        name: str,
        level: str,
        input_schema: JsonDict,
        output_schema: JsonDict,
        description: str,
    ) -> PrimitiveCard:
        capability_tags = ["robodojo", "isaacsim", "agent_native", level.lower()]
        if name == "observe_robodojo_visual":
            capability_tags.extend(["vision", "rgb"])
        elif name in {
            "measure_robodojo_public_geometry",
            "inspect_robodojo_collision_geometry",
        }:
            capability_tags.extend(["geometry", "grounding"])
        return PrimitiveCard(
            name=name,
            capability_tags=capability_tags,
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=["reset_called"],
            side_effects=(
                ["runs_policy_action_chunk", "may_step_live_environment"]
                if name == "run_robodojo_policy_skill"
                else (
                    ["records_low_level_actions"]
                    if name == "submit_robodojo_low_level_actions"
                    else []
                )
            ),
            cost={
                "sim_steps": "bounded_by_max_actions"
                if name == "run_robodojo_policy_skill"
                else 0,
                "requires_live_env": name == "run_robodojo_policy_skill",
            },
            failure_modes=[
                "repo_missing",
                "task_config_missing",
                "asset_or_isaac_not_ready",
                "policy_adapter_checkpoint_or_server_not_ready",
            ],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._task_spec is None:
            raise RuntimeError("reset must be called before using RoboDojo runtime")

    def _merged_config(self, overrides: JsonDict) -> RoboDojoRuntimeConfig:
        data = self.config.to_dict()
        data.update(
            {key: value for key, value in overrides.items() if value is not None}
        )
        return RoboDojoRuntimeConfig(**data)


def inspect_robodojo_repo(repo_path: str | Path) -> JsonDict:
    root = Path(repo_path)
    commit = None
    if (root / ".git").exists() and shutil.which("git"):
        try:
            commit = subprocess.check_output(
                ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            commit = None
    task_dir = root / "task/RoboDojo/tasks"
    config_dir = root / "task/RoboDojo/config"
    task_files = (
        sorted(
            path.stem
            for path in task_dir.glob("*.py")
            if not path.name.startswith("__")
        )
        if task_dir.exists()
        else []
    )
    config_files = (
        sorted(
            path.stem
            for path in config_dir.glob("*.yml")
            if not path.name.startswith("_")
        )
        if config_dir.exists()
        else []
    )
    return {
        "repo_path": str(root),
        "exists": root.exists(),
        "commit": commit,
        "readme_exists": (root / "README.md").exists(),
        "pyproject_exists": (root / "pyproject.toml").exists(),
        "dockerfile_exists": (root / "Dockerfile").exists(),
        "task_count": len(task_files),
        "config_count": len(config_files),
        "tasks": task_files,
        "configs": config_files,
    }


def robodojo_source_preflight(
    repo_path: str | Path, task_name: str = "", env_cfg: str = "arx_x5"
) -> JsonDict:
    root = Path(repo_path)
    blockers: list[str] = []
    required_files = {
        "launcher": root / "scripts/robodojo.sh",
        "inventory": root / "scripts/internal/task_inventory.py",
        "task_registry": root / "task/RoboDojo/task_registry.py",
        "task_py": root / f"task/RoboDojo/tasks/{task_name}.py",
        "task_config": root / f"task/RoboDojo/config/{task_name}.yml",
        "env_cfg": root / f"env_cfg/{env_cfg}.yml",
    }
    for name, path in required_files.items():
        if not path.exists():
            blockers.append(f"{name}_missing:{path}")
    inventory = inspect_robodojo_repo(root)
    if inventory["task_count"] == 0 or inventory["config_count"] == 0:
        blockers.append("task_or_config_inventory_empty")
    env_refs = robodojo_env_cfg_reference_gate(root, env_cfg)
    if not env_refs["ready"]:
        blockers.extend(env_refs["blockers"])
    return {
        "ready": not blockers,
        "blockers": blockers,
        "required_files": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required_files.items()
        },
        "inventory": {
            key: inventory[key]
            for key in ("task_count", "config_count", "commit", "repo_path")
        },
        "env_cfg_references": env_refs,
    }


def robodojo_asset_gate(repo_path: str | Path) -> JsonDict:
    root = Path(repo_path)
    required_dirs = {
        "robot_assets": root / "Assets/Robots",
        "object_assets": root / "Assets/Object/RoboDojo",
        "eval_layouts": root / "Assets/Eval_Layout/RoboDojo",
        "materials": root / "Assets/Material",
    }
    missing = [name for name, path in required_dirs.items() if not path.exists()]
    return {
        "ready": not missing,
        "blockers": [f"{name}_missing:{required_dirs[name]}" for name in missing],
        "required_dirs": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required_dirs.items()
        },
        "download_command": "cd external/upstreams/robodojo && bash scripts/init_assets.sh",
        "hf_dataset": "RoboDojo-Benchmark/RoboDojo",
    }


def robodojo_python_env_gate(sim_env: str = "RoboDojo") -> JsonDict:
    pinned_candidates = [
        os.environ.get("AGENTIC_EMBODIED_ARENA_NATIVE_PYTHON_ROBODOJO"),
        os.environ.get("AGENTIC_EMBODIED_ARENA_NATIVE_PYTHON"),
        str(
            get_project_paths().project_root
            / "external/environments/robodojo/bin/python3.11"
        ),
        sys.executable,
    ]
    for raw_candidate in pinned_candidates:
        if not raw_candidate:
            continue
        candidate = Path(raw_candidate).expanduser()
        if not candidate.is_file():
            continue
        # Resolving a venv's Python symlink selects the base interpreter and
        # drops the installed simulator packages from its import path.
        resolved = candidate.expanduser().absolute()
        if "external/environments/robodojo" not in resolved.as_posix():
            continue
        return {
            "ready": True,
            "blockers": [],
            "sim_env": sim_env,
            "python_executable": str(resolved),
            "source": "pinned_operation_environment",
            "isaac_import_probe_command": (
                f"{resolved} -c 'import isaacsim, isaaclab'"
            ),
        }
    conda = shutil.which("conda")
    if not conda:
        return {"ready": False, "blockers": ["conda_not_found"], "sim_env": sim_env}
    try:
        output = subprocess.check_output(
            [conda, "env", "list"], text=True, stderr=subprocess.STDOUT, timeout=15
        )
    except Exception as exc:
        return {
            "ready": False,
            "blockers": [f"conda_env_list_failed:{type(exc).__name__}"],
            "sim_env": sim_env,
        }
    env_names = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        env_names.append(stripped.split()[0])
    env_exists = sim_env in env_names
    return {
        "ready": env_exists,
        "blockers": [] if env_exists else [f"conda_env_missing:{sim_env}"],
        "sim_env": sim_env,
        "isaac_import_probe_command": f"conda run -n {sim_env} python -c 'import isaacsim, isaaclab'",
    }


def robodojo_env_cfg_reference_gate(repo_path: str | Path, env_cfg: str) -> JsonDict:
    root = Path(repo_path)
    cfg = load_robodojo_env_config(root, env_cfg)
    refs = dict(cfg.get("config") or {})
    required = {
        "sim": root / "env_cfg/sim" / f"{refs.get('sim')}.yml",
        "scene": root / "env_cfg/scene" / f"{refs.get('scene')}.yml",
        "robot": root / "env_cfg/robot" / f"{refs.get('robot')}.yml",
        "camera": root / "env_cfg/camera" / f"{refs.get('camera')}.yml",
        "robot_info": root / "env_cfg/robot/_robot_info.json",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    return {
        "ready": not missing,
        "blockers": [
            f"env_cfg_reference_missing:{name}:{required[name]}" for name in missing
        ],
        "refs": refs,
        "required": {
            name: {"path": str(path), "exists": path.exists()}
            for name, path in required.items()
        },
    }


def robodojo_robot_action_schema(repo_path: str | Path, env_cfg: str) -> JsonDict:
    root = Path(repo_path)
    cfg = load_robodojo_env_config(root, env_cfg)
    robot_name = str((cfg.get("config") or {}).get("robot") or "")
    info_path = root / "env_cfg/robot/_robot_info.json"
    robot_info: JsonDict = {}
    if info_path.exists():
        try:
            all_info = json.loads(info_path.read_text(encoding="utf-8"))
            if isinstance(all_info, dict):
                selected = all_info.get(robot_name) or {}
                robot_info = selected if isinstance(selected, dict) else {}
        except Exception:
            robot_info = {}
    arm_dims = list(robot_info.get("arm_dim") or [])
    ee_dims = list(robot_info.get("ee_dim") or [])
    arm_count = max(len(arm_dims), len(ee_dims))
    if arm_count == 1:
        keys = {
            "joint": {
                "arm_joint_state": int(arm_dims[0]) if arm_dims else None,
                "ee_joint_state": int(ee_dims[0]) if ee_dims else None,
            },
            "ee": {"ee_pose": 7, "tcp_pose": 7, "delta_ee_pose": 7},
        }
    elif arm_count == 2:
        keys = {
            "joint": {
                "left_arm_joint_state": int(arm_dims[0]) if len(arm_dims) > 0 else None,
                "left_ee_joint_state": int(ee_dims[0]) if len(ee_dims) > 0 else None,
                "right_arm_joint_state": int(arm_dims[1])
                if len(arm_dims) > 1
                else None,
                "right_ee_joint_state": int(ee_dims[1]) if len(ee_dims) > 1 else None,
            },
            "ee": {
                "left_ee_pose": 7,
                "left_tcp_pose": 7,
                "left_delta_ee_pose": 7,
                "right_ee_pose": 7,
                "right_tcp_pose": 7,
                "right_delta_ee_pose": 7,
            },
        }
    else:
        keys = {"joint": {}, "ee": {}}
    return {
        "env_cfg": env_cfg,
        "robot_config_name": robot_name or None,
        "robot_info_path": str(info_path),
        "robot_info_available": bool(robot_info),
        "arm_dim": arm_dims,
        "ee_dim": ee_dims,
        "arm_count": arm_count,
        "action_keys": keys,
        "notes": "These are public policy action schemas from RoboDojo env_cfg/robot/_robot_info.json; success remains harness-only.",
    }


def build_robodojo_joint_action(
    repo_path: str | Path,
    env_cfg: str,
    *,
    joint_targets: Any | None = None,
    gripper_targets: Any | None = None,
    agent_context: JsonDict | None = None,
) -> JsonDict:
    """Encode exactly one caller-authored joint/gripper action."""

    schema = robodojo_robot_action_schema(repo_path, env_cfg)
    joint_keys = dict((schema.get("action_keys") or {}).get("joint") or {})
    action: dict[str, list[float]] = {}
    for key, dim in joint_keys.items():
        expected_dim = int(dim) if dim is not None else 0
        target = _target_for_action_key(
            key, joint_targets=joint_targets, gripper_targets=gripper_targets
        )
        if target is None:
            raise ValueError(f"caller_target_required:{key}")
        action[key] = _coerce_float_vector(target, expected_dim, fill=0.0)
    return {
        "action": action,
        "action_dims": {key: len(value) for key, value in action.items()},
        "action_schema": schema,
        "action_source": "caller_authored_public_robot_action_schema",
        "hidden_pose_or_target_fill": False,
        "official_task_success_claimed": False,
        "agent_visible_success_checker": False,
        "oracle_or_demo_replay_used": False,
        "agent_context": agent_context or {},
    }


def load_robodojo_runtime_report(report_path: str | Path) -> JsonDict:
    path = Path(report_path).expanduser().resolve()
    if not path.exists():
        return {"ok": False, "error": f"report_missing:{path}"}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "ok": False,
            "error": f"report_json_load_failed:{type(exc).__name__}:{exc}",
            "report_path": str(path),
        }
    if not isinstance(data, dict):
        return {"ok": False, "error": "report_not_mapping", "report_path": str(path)}
    data["_runtime_report_path"] = str(path)
    return data


def _hydrate_robodojo_rgb_sidecars(
    report: JsonDict, runtime_state: Any
) -> tuple[Any, JsonDict]:
    state = (
        deepcopy(runtime_state) if isinstance(runtime_state, dict) else runtime_state
    )
    artifacts = report.get("agent_visual_artifacts")
    views = artifacts.get("views") if isinstance(artifacts, dict) else None
    report_path = report.get("_runtime_report_path")
    if (
        not isinstance(state, dict)
        or not isinstance(views, dict)
        or not isinstance(report_path, str)
    ):
        return state, {"attempted": False, "hydrated_views": [], "errors": {}}
    observation = state.get("observation")
    if not isinstance(observation, dict):
        return state, {
            "attempted": True,
            "hydrated_views": [],
            "errors": {"observation": "missing"},
        }
    vision = observation.setdefault("vision", {})
    if not isinstance(vision, dict):
        return state, {
            "attempted": True,
            "hydrated_views": [],
            "errors": {"vision": "invalid"},
        }

    report_dir = Path(report_path).parent
    hydrated: list[str] = []
    errors: dict[str, str] = {}
    for camera in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
        metadata = views.get(camera)
        if not isinstance(metadata, dict):
            errors[camera] = "metadata_missing"
            continue
        sidecar_value = metadata.get("path")
        if not isinstance(sidecar_value, str) or not sidecar_value:
            errors[camera] = "path_missing"
            continue
        sidecar_path = Path(sidecar_value)
        if not sidecar_path.is_absolute():
            sidecar_path = report_dir / sidecar_path
        try:
            frame = np.load(sidecar_path, allow_pickle=False)
            frame = np.ascontiguousarray(frame)
            expected_shape = [int(item) for item in metadata.get("shape", [])]
            expected_dtype = str(metadata.get("dtype") or "")
            expected_sha256 = str(metadata.get("sha256") or "")
            actual_shape = [int(item) for item in frame.shape]
            actual_dtype = str(frame.dtype)
            actual_sha256 = hashlib.sha256(frame.tobytes(order="C")).hexdigest()
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("rgb_shape_invalid")
            if expected_shape and actual_shape != expected_shape:
                raise ValueError("shape_mismatch")
            if expected_dtype and actual_dtype != expected_dtype:
                raise ValueError("dtype_mismatch")
            if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
                raise ValueError("sha256_mismatch")
            source_path = str(metadata.get("source_path") or "")
            source_key = source_path.rsplit(".", 1)[-1] if source_path else "color"
            if source_key not in {"color", "rgb"}:
                raise ValueError("source_path_invalid")
            payload = vision.setdefault(camera, {})
            if not isinstance(payload, dict):
                payload = {}
                vision[camera] = payload
            payload[source_key] = frame
            hydrated.append(camera)
        except Exception as exc:
            errors[camera] = f"{type(exc).__name__}:{exc}"
    return state, {"attempted": True, "hydrated_views": hydrated, "errors": errors}


def extract_robodojo_public_runtime_state(report: JsonDict | None) -> JsonDict:
    data = _primitive_output(report) if report is not None else {}
    if not isinstance(data, dict):
        data = {}
    runtime_state, hydration = _hydrate_robodojo_rgb_sidecars(
        data, data.get("public_runtime_state")
    )
    after_state = data.get("public_runtime_state_after_action")
    raw_action_step = (
        data.get("action_step") if isinstance(data.get("action_step"), dict) else {}
    )
    public_action_step = _sanitize_robodojo_action_step_trace(raw_action_step)
    public_runtime_state_trace = public_action_step.get("reobservations") or []
    clean_report = {
        "ok": bool(data.get("ok")),
        "stage": data.get("stage"),
        "task_name": data.get("task_name"),
        "env_cfg": data.get("env_cfg"),
        "action_step_ok": data.get("action_step_ok"),
        "low_level_action_count": data.get("low_level_action_count"),
        "agent_action_payload_loaded": data.get("agent_action_payload_loaded"),
        "action_step": public_action_step,
        "official_task_success_claimed": False,
        "agent_visible_success_checker": False,
        "oracle_or_demo_replay_used": False,
        "visual_artifact_hydration": hydration,
    }
    return {
        "public_runtime_state": runtime_state
        if isinstance(runtime_state, dict)
        else None,
        "public_runtime_state_after_action": after_state
        if isinstance(after_state, dict)
        else None,
        "public_runtime_state_trace": public_runtime_state_trace,
        "action_step": public_action_step,
        "controller_feedback": public_action_step.get("controller_feedback") or {},
        "runtime_report": clean_report,
        "leakage_boundary": {
            "reward_manager_read": False,
            "success_predicate_read": False,
            "demo_replay_used": False,
        },
    }


def compile_robodojo_ee_path(
    repo_path: str | Path,
    env_cfg: str,
    *,
    runtime_state: JsonDict | None = None,
    path_points: list[JsonDict] | None = None,
    arm: str | None = None,
    contact_frame: Any | None = None,
    max_cartesian_step: float | None = None,
    agent_context: JsonDict | None = None,
) -> JsonDict:
    """Compile explicit caller-authored EE path points against public robot state.

    This geometry/path primitive does not inspect a task identifier, select an
    object, choose path points, add offsets, order manipulation phases, or read
    a completion condition. Every position, contact-axis alignment, physical
    jaw opening (or normalized gripper command), and repeat is supplied by the
    caller.
    """

    runtime = _robodojo_public_runtime_state(runtime_state)
    robots = _robodojo_runtime_robots(runtime)
    schema = _robodojo_action_schema_with_public_runtime_fallback(
        robodojo_robot_action_schema(repo_path, env_cfg),
        robots=robots,
    )
    requested_points = [
        dict(item) for item in (path_points or []) if isinstance(item, dict)
    ]
    if not requested_points:
        return {
            "low_level_actions": [],
            "action_dims": [],
            "path_points": [],
            "action_schema": schema,
            "error": "caller_authored_path_points_required",
            "official_task_success_claimed": False,
            "agent_visible_success_checker": False,
            "oracle_or_demo_replay_used": False,
            "agent_context": agent_context or {},
        }

    first = requested_points[0]
    first_pose = (
        _coerce_float_vector(first.get("pose"), 7)
        if first.get("pose") is not None
        else []
    )
    first_position = (
        first_pose[:3]
        if len(first_pose) == 7
        else _coerce_float_vector(first.get("position"), 3)
    )
    if len(first_position) != 3:
        return {
            "low_level_actions": [],
            "action_dims": [],
            "path_points": [],
            "action_schema": schema,
            "error": "path_point_0_pose_or_position_required",
            "official_task_success_claimed": False,
            "agent_visible_success_checker": False,
            "oracle_or_demo_replay_used": False,
            "agent_context": agent_context or {},
        }
    active_side = _select_robodojo_arm_side(arm=arm)
    hold_poses = _robodojo_hold_ee_poses(schema, robots=robots)
    default_orientation = _robodojo_active_orientation(hold_poses, active_side)
    active_pose_key = (
        f"{active_side}_ee_pose" if active_side in {"left", "right"} else "ee_pose"
    )
    current_active_pose = _coerce_float_vector(hold_poses.get(active_pose_key), 7)
    if len(current_active_pose) != 7:
        raise ValueError(f"public_current_ee_pose_required:{active_pose_key}")
    hold_grippers = _robodojo_hold_gripper_states(
        schema, robots=robots, repo_path=repo_path
    )
    tool_offset_evidence: JsonDict = {
        "source": "caller_value",
        "available": contact_frame is not None,
    }
    local_tool_offset: list[float] | None = None
    if (
        isinstance(contact_frame, str)
        and contact_frame.strip().lower() == "public_urdf"
    ):
        tool_offset_evidence = robodojo_public_tool_contact_offset(
            repo_path, robots=robots, active_side=active_side
        )
        if not tool_offset_evidence.get("available"):
            raise ValueError(
                f"public_contact_geometry_unavailable:{tool_offset_evidence.get('error')}"
            )
        local_tool_offset = _coerce_float_vector(
            tool_offset_evidence.get("local_contact_center"), 3
        )
        if len(local_tool_offset) != 3:
            raise ValueError("public_contact_geometry_missing_local_center")
        tool_offset = [0.0, 0.0, 0.0]
    else:
        tool_offset = _coerce_float_vector(contact_frame, 3, fill=0.0)

    sequence: list[JsonDict] = []
    summaries: list[JsonDict] = []
    for index, point in enumerate(requested_points):
        pose_vector = (
            _coerce_float_vector(point.get("pose"), 7)
            if point.get("pose") is not None
            else []
        )
        position = (
            pose_vector[:3]
            if len(pose_vector) == 7
            else _coerce_float_vector(point.get("position"), 3)
        )
        if len(position) != 3:
            raise ValueError(f"path_point_{index}_pose_or_position_required")
        explicit_orientation = (
            pose_vector[3:7]
            if len(pose_vector) == 7
            else _coerce_float_vector(point.get("orientation"), 4, fill=0.0)
            if "orientation" in point
            else []
        )
        requested_approach_axis = (
            _coerce_float_vector(point.get("approach_axis"), 3)
            if "approach_axis" in point
            else []
        )
        requested_jaw_axis = (
            _coerce_float_vector(point.get("jaw_axis"), 3)
            if "jaw_axis" in point
            else []
        )
        if explicit_orientation and (requested_approach_axis or requested_jaw_axis):
            raise ValueError(f"path_point_{index}_orientation_axis_alignment_ambiguous")
        orientation_source = "caller_orientation"
        if requested_approach_axis or requested_jaw_axis:
            if len(requested_approach_axis) != 3 or len(requested_jaw_axis) != 3:
                raise ValueError(
                    f"path_point_{index}_approach_and_jaw_axes_required_together"
                )
            if not tool_offset_evidence.get("available"):
                raise ValueError(
                    f"path_point_{index}_public_contact_geometry_required_for_axis_alignment"
                )
            orientation = _quaternion_aligning_contact_axes(
                local_approach_axis=_coerce_float_vector(
                    tool_offset_evidence.get("local_approach_axis"), 3
                ),
                local_jaw_axis=_coerce_float_vector(
                    tool_offset_evidence.get("local_jaw_axis"), 3
                ),
                world_approach_axis=requested_approach_axis,
                world_jaw_axis=requested_jaw_axis,
            )
            orientation_source = "caller_contact_axes"
        elif explicit_orientation:
            orientation = explicit_orientation
        else:
            orientation = list(default_orientation)
            orientation_source = "public_current_ee_pose"
        if len(orientation) != 4:
            raise ValueError(
                f"path_point_{index}_orientation_or_public_current_pose_required"
            )
        if "gripper" in point and "gripper_opening" in point:
            raise ValueError(f"path_point_{index}_gripper_command_opening_ambiguous")
        if "gripper_opening" in point:
            if not tool_offset_evidence.get("gripper_command_mapping"):
                raise ValueError(
                    f"path_point_{index}_public_contact_geometry_required_for_gripper_opening"
                )
            requested_gripper_opening = float(point["gripper_opening"])
            gripper = _robodojo_gripper_command_for_opening(
                tool_offset_evidence, requested_gripper_opening
            )
            predicted_gripper_opening = _robodojo_contact_opening_for_command(
                tool_offset_evidence, gripper
            )
            gripper_source = "caller_physical_opening"
        elif "gripper" in point:
            requested_gripper_opening = None
            gripper = float(point["gripper"])
            predicted_gripper_opening = (
                _robodojo_contact_opening_for_command(tool_offset_evidence, gripper)
                if tool_offset_evidence.get("gripper_command_mapping")
                else None
            )
            gripper_source = "caller_normalized_command"
        else:
            raise ValueError(f"path_point_{index}_gripper_or_gripper_opening_required")
        position_frame = str(point.get("position_frame") or "wrist").strip().lower()
        commanded_position = list(position)
        applied_contact_offset = [0.0, 0.0, 0.0]
        if position_frame == "contact":
            applied_contact_offset = (
                _rotate_vector_by_quaternion_wxyz(local_tool_offset, orientation)
                if local_tool_offset is not None
                else list(tool_offset)
            )
            commanded_position = [
                commanded_position[axis] - applied_contact_offset[axis]
                for axis in range(3)
            ]
        elif position_frame != "wrist":
            raise ValueError(
                f"path_point_{index}_invalid_position_frame:{position_frame}"
            )
        pose = [round(float(value), 5) for value in commanded_position] + [
            float(value) for value in orientation
        ]
        configured_step = point.get("max_cartesian_step", max_cartesian_step)
        interpolation_limit = (
            None if configured_step is None else float(configured_step)
        )
        if interpolation_limit is not None and (
            not math.isfinite(interpolation_limit) or interpolation_limit <= 0.0
        ):
            raise ValueError(f"path_point_{index}_max_cartesian_step_must_be_positive")
        transition_distance = _vector_norm(
            _vector_delta(current_active_pose[:3], pose[:3])
        )
        interpolation_steps = (
            max(1, int(math.ceil(transition_distance / interpolation_limit)))
            if interpolation_limit is not None
            else 1
        )
        point_actions: list[JsonDict] = []
        for interpolation_index in range(1, interpolation_steps + 1):
            fraction = interpolation_index / interpolation_steps
            interpolated_pose = [
                round(
                    float(current_active_pose[axis])
                    + (float(pose[axis]) - float(current_active_pose[axis])) * fraction,
                    5,
                )
                for axis in range(3)
            ] + _nlerp_quaternion_wxyz(current_active_pose[3:7], pose[3:7], fraction)
            point_actions.append(
                _robodojo_ee_action_from_pose(
                    schema,
                    active_side=active_side,
                    active_pose=interpolated_pose,
                    gripper_value=gripper,
                    hold_poses=hold_poses,
                    hold_grippers=hold_grippers,
                )
            )
        action = point_actions[-1]
        repeat = max(1, int(point.get("repeat_steps") or 1))
        for point_action in point_actions:
            for _ in range(repeat):
                sequence.append(dict(point_action))
        summaries.append(
            {
                "index": index,
                "active_side": active_side,
                "repeat_steps": repeat,
                "position_frame": position_frame,
                "requested_pose": [float(value) for value in position]
                + [float(value) for value in orientation],
                "active_ee_pose": pose,
                "applied_contact_offset": [
                    round(float(value), 6) for value in applied_contact_offset
                ],
                "gripper": gripper,
                "gripper_source": gripper_source,
                "requested_gripper_opening": requested_gripper_opening,
                "predicted_gripper_opening": predicted_gripper_opening,
                "orientation_source": orientation_source,
                "requested_approach_axis": requested_approach_axis or None,
                "requested_jaw_axis": requested_jaw_axis or None,
                "transition_distance": transition_distance,
                "max_cartesian_step": interpolation_limit,
                "interpolation_step_count": interpolation_steps,
                "compiled_action_count": interpolation_steps * repeat,
                "action_dims": {key: len(value) for key, value in action.items()},
            }
        )
        current_active_pose = list(pose)
    return {
        "low_level_actions": sequence,
        "action_dims": robodojo_action_dims(sequence),
        "path_points": summaries,
        "action_schema": schema,
        "path_contract": {
            "caller_authored_path_points_required": True,
            "caller_path_point_count": len(requested_points),
            "object_selection_performed_by_primitive": False,
            "offsets_added_by_primitive": False,
            "task_id_read": False,
            "task_completion_condition_read": False,
            "runtime_state_used": bool(runtime),
            "inactive_arm_pose_source": "public_current_robot_state",
            "inactive_gripper_source": "public_current_robot_state",
            "contact_frame_translation": tool_offset
            if local_tool_offset is None
            else None,
            "contact_frame_local_translation": local_tool_offset,
            "contact_frame_rotated_by_each_requested_orientation": local_tool_offset
            is not None,
            "contact_frame_evidence": tool_offset_evidence,
            "cartesian_interpolation_supported": True,
            "repeat_steps_apply_to_each_interpolated_waypoint": True,
            "default_max_cartesian_step": max_cartesian_step,
        },
        "controller_bridge": {
            "accepted_by_upstream_validate_action_dict": True,
            "action_type": "ee",
            "upstream_mapping": "EvalEnv.take_action -> robot_manager.solve_ik -> control_manager",
            "requires_live_isaac": True,
        },
        "action_source": "caller_authored_public_runtime_ee_path",
        "official_task_success_claimed": False,
        "agent_visible_success_checker": False,
        "oracle_or_demo_replay_used": False,
        "agent_context": agent_context or {},
    }


def compile_robodojo_contact_lift_actions(
    repo_path: str | Path,
    env_cfg: str,
    *,
    runtime_state: JsonDict | None = None,
    object_label: str | None = None,
    arm: str | None = None,
    approach_axis: Any | None = None,
    jaw_axis: Any | None = None,
    lift_distance: float | None = None,
    lift_vector: list[float] | None = None,
    lift_axis: Any | None = None,
    precontact_distance: float | None = None,
    contact_depth: float | None = None,
    contact_position_offset: list[float] | None = None,
    open_gripper_opening: float | None = None,
    grasp_gripper_opening: float | None = None,
    grasp_compression: float = 0.0,
    precontact_repeat_steps: int = 1,
    contact_repeat_steps: int = 1,
    grasp_repeat_steps: int = 2,
    lift_repeat_steps: int = 2,
    max_cartesian_step: float | None = None,
    agent_context: JsonDict | None = None,
) -> JsonDict:
    """Compile a caller-parameterized contact/lift sequence.

    This is a manipulation-skill compiler, not a task planner: the caller must
    name the public object label, arm, contact axes, lift amount/vector, and
    repeat/opening parameters. The function derives only geometry quantities
    that are directly measurable from public object poses/bboxes and public
    robot collision geometry.
    """

    runtime = _robodojo_public_runtime_state(runtime_state)
    label = str(object_label or "").strip()
    active_side = _select_robodojo_arm_side(arm=arm)
    if not label:
        return _empty_robodojo_contact_lift_payload(
            "caller_selected_object_label_required",
            agent_context=agent_context,
        )
    geometry = measure_robodojo_public_geometry(
        runtime,
        object_labels=[label],
        agent_context={"source": "compile_robodojo_contact_lift_actions"},
    )
    object_geometry = (geometry.get("objects") or {}).get(label) or {}
    obb = (
        object_geometry.get("before_oriented_bbox")
        if isinstance(object_geometry.get("before_oriented_bbox"), dict)
        else {}
    )
    center = _coerce_float_vector(
        obb.get("world_center") or object_geometry.get("before_position"), 3
    )
    if len(center) != 3:
        return _empty_robodojo_contact_lift_payload(
            f"public_object_geometry_missing:{label}",
            agent_context=agent_context,
            geometry=geometry,
        )
    axis_map = _robodojo_obb_axis_map(obb)
    if approach_axis is None or jaw_axis is None:
        return _empty_robodojo_contact_lift_payload(
            "caller_selected_approach_and_jaw_axes_required",
            agent_context=agent_context,
            geometry=geometry,
            available_object_axes=axis_map,
        )
    approach = _resolve_robodojo_world_axis(
        approach_axis, axis_map, label="approach_axis"
    )
    jaw = _resolve_robodojo_world_axis(jaw_axis, axis_map, label="jaw_axis")
    if _vector_norm(_vector_cross(approach, jaw)) <= 1e-6:
        raise ValueError("approach_axis_and_jaw_axis_must_not_be_parallel")

    robots = _robodojo_runtime_robots(runtime)
    contact = robodojo_public_tool_contact_offset(
        repo_path, robots=robots, active_side=active_side
    )
    if not contact.get("available"):
        raise ValueError(f"public_contact_geometry_unavailable:{contact.get('error')}")
    opening_range = sorted(
        float(value) for value in (contact.get("contact_opening_range") or [])
    )
    if len(opening_range) != 2:
        raise ValueError("public_gripper_opening_range_missing")
    min_opening, max_opening = opening_range
    open_opening = (
        float(open_gripper_opening) if open_gripper_opening is not None else max_opening
    )
    if grasp_gripper_opening is None:
        projected_width = _robodojo_projected_obb_extent(obb, jaw)
        grasp_opening = max(
            min_opening,
            min(max_opening, projected_width - max(0.0, float(grasp_compression))),
        )
        grasp_opening_source = "public_object_projected_extent_minus_caller_compression"
    else:
        grasp_opening = float(grasp_gripper_opening)
        grasp_opening_source = "caller_grasp_gripper_opening"
    if not min_opening - 1e-9 <= open_opening <= max_opening + 1e-9:
        raise ValueError(f"open_gripper_opening_out_of_public_range:{open_opening}")
    if not min_opening - 1e-9 <= grasp_opening <= max_opening + 1e-9:
        raise ValueError(f"grasp_gripper_opening_out_of_public_range:{grasp_opening}")

    approach_extent = _robodojo_projected_obb_extent(obb, approach)
    if precontact_distance is None:
        precontact = max(
            float(contact.get("contact_opening") or 0.0), approach_extent * 0.25
        )
        precontact_source = "public_contact_opening_or_object_extent"
    else:
        precontact = float(precontact_distance)
        precontact_source = "caller_precontact_distance"
    if precontact < 0.0:
        raise ValueError("precontact_distance_must_be_nonnegative")

    depth = float(contact_depth or 0.0)
    if not math.isfinite(depth) or depth < 0.0:
        raise ValueError("contact_depth_must_be_finite_and_nonnegative")
    offset = [0.0, 0.0, 0.0]
    if contact_position_offset is not None:
        offset = _coerce_float_vector(contact_position_offset, 3)
        if len(offset) != 3 or not all(math.isfinite(value) for value in offset):
            raise ValueError("contact_position_offset_must_be_three_finite_values")

    lift_delta = _robodojo_lift_delta(
        lift_vector=lift_vector, lift_axis=lift_axis, lift_distance=lift_distance
    )
    contact_base_position = [center[i] + offset[i] for i in range(3)]
    contact_position = [
        contact_base_position[i] + approach[i] * depth for i in range(3)
    ]
    precontact_position = [
        contact_base_position[i] - approach[i] * precontact for i in range(3)
    ]
    lifted_position = [contact_position[i] + lift_delta[i] for i in range(3)]
    common = {
        "position_frame": "contact",
        "approach_axis": approach,
        "jaw_axis": jaw,
    }
    path_points = [
        {
            **common,
            "position": precontact_position,
            "gripper_opening": open_opening,
            "repeat_steps": max(1, int(precontact_repeat_steps)),
        },
        {
            **common,
            "position": contact_position,
            "gripper_opening": open_opening,
            "repeat_steps": max(1, int(contact_repeat_steps)),
        },
        {
            **common,
            "position": contact_position,
            "gripper_opening": grasp_opening,
            "repeat_steps": max(1, int(grasp_repeat_steps)),
        },
        {
            **common,
            "position": lifted_position,
            "gripper_opening": grasp_opening,
            "repeat_steps": max(1, int(lift_repeat_steps)),
        },
    ]
    compiled = compile_robodojo_ee_path(
        repo_path,
        env_cfg,
        runtime_state=runtime,
        path_points=path_points,
        arm=active_side,
        contact_frame="public_urdf",
        max_cartesian_step=max_cartesian_step,
        agent_context={
            **(agent_context or {}),
            "source": "compile_robodojo_contact_lift_actions",
        },
    )
    return {
        "low_level_actions": compiled["low_level_actions"],
        "action_dims": compiled["action_dims"],
        "path_points": compiled["path_points"],
        "action_schema": compiled["action_schema"],
        "contact_lift_contract": {
            "object_label_caller_selected": True,
            "arm_caller_selected": True,
            "approach_axis_caller_selected": True,
            "jaw_axis_caller_selected": True,
            "lift_vector_or_distance_caller_selected": True,
            "task_id_read": False,
            "task_target_selected": False,
            "reward_or_verifier_read": False,
            "demo_or_expert_replay_used": False,
            "fixed_waypoints_used": False,
            "contact_depth_caller_parameterized": True,
            "contact_position_offset_caller_parameterized": True,
            "waypoint_positions_derived_from_public_object_geometry": True,
            "public_collision_geometry_used": True,
            "precontact_distance_source": precontact_source,
            "grasp_opening_source": grasp_opening_source,
        },
        "public_geometry": {
            "object_label": label,
            "object_center": center,
            "contact_position": contact_position,
            "contact_depth": depth,
            "contact_position_offset": offset,
            "available_object_axes": axis_map,
            "selected_approach_axis": approach,
            "selected_jaw_axis": jaw,
            "projected_approach_extent": approach_extent,
            "projected_jaw_extent": _robodojo_projected_obb_extent(obb, jaw),
            "lift_delta": lift_delta,
            "open_gripper_opening": open_opening,
            "grasp_gripper_opening": grasp_opening,
            "public_gripper_opening_range": opening_range,
            "contact_geometry": contact,
        },
        "controller_bridge": compiled.get("controller_bridge", {}),
        "action_source": "caller_parameterized_public_geometry_contact_lift",
        "official_task_success_claimed": False,
        "agent_visible_success_checker": False,
        "oracle_or_demo_replay_used": False,
        "agent_context": agent_context or {},
    }


def _empty_robodojo_contact_lift_payload(
    error: str,
    *,
    agent_context: JsonDict | None,
    geometry: JsonDict | None = None,
    available_object_axes: JsonDict | None = None,
) -> JsonDict:
    return {
        "low_level_actions": [],
        "action_dims": [],
        "path_points": [],
        "action_schema": {},
        "error": error,
        "contact_lift_contract": {
            "task_id_read": False,
            "task_target_selected": False,
            "reward_or_verifier_read": False,
            "demo_or_expert_replay_used": False,
        },
        "public_geometry": {
            "geometry": geometry or {},
            "available_object_axes": available_object_axes or {},
        },
        "official_task_success_claimed": False,
        "agent_visible_success_checker": False,
        "oracle_or_demo_replay_used": False,
        "agent_context": agent_context or {},
    }


def _robodojo_obb_axis_map(obb: JsonDict) -> JsonDict:
    axis_map: dict[str, JsonDict] = {}
    for axis in obb.get("world_axes") or []:
        if not isinstance(axis, dict):
            continue
        name = str(axis.get("local_axis") or "").strip().lower()
        direction = _normalized_vector(
            _coerce_float_vector(axis.get("world_direction"), 3)
        )
        if name and _vector_norm(direction) > 0.0:
            axis_map[name] = {
                "local_axis": name,
                "world_direction": direction,
                "extent": float(axis.get("extent") or 0.0),
            }
    return axis_map


def _resolve_robodojo_world_axis(
    value: Any, axis_map: JsonDict, *, label: str
) -> list[float]:
    if isinstance(value, str):
        axis_name = value.split(":", 1)[1] if value.startswith("obb:") else value
        axis_name = axis_name.strip().lower()
        sign = -1.0 if axis_name.startswith("-") else 1.0
        axis_name = axis_name.lstrip("+-")
        axis = axis_map.get(axis_name)
        if not isinstance(axis, dict):
            raise ValueError(f"{label}_unknown_object_obb_axis:{axis_name}")
        direction = _normalized_vector(
            _coerce_float_vector(axis.get("world_direction"), 3)
        )
        return [sign * item for item in direction]
    if isinstance(value, dict):
        source = str(value.get("source") or "vector").strip().lower()
        sign = float(value.get("sign", 1.0))
        if source in {"object_obb", "obb"}:
            axis_name = (
                str(value.get("axis") or value.get("local_axis") or "").strip().lower()
            )
            axis = axis_map.get(axis_name)
            if not isinstance(axis, dict):
                raise ValueError(f"{label}_unknown_object_obb_axis:{axis_name}")
            direction = _normalized_vector(
                _coerce_float_vector(axis.get("world_direction"), 3)
            )
            return [sign * item for item in direction]
        vector = _normalized_vector(_coerce_float_vector(value.get("vector"), 3))
    else:
        vector = _normalized_vector(_coerce_float_vector(value, 3))
    if _vector_norm(vector) <= 0.0:
        raise ValueError(f"{label}_zero_vector")
    return vector


def _robodojo_projected_obb_extent(obb: JsonDict, world_axis: list[float]) -> float:
    axis = _normalized_vector(world_axis)
    extent = 0.0
    for item in obb.get("world_axes") or []:
        if not isinstance(item, dict):
            continue
        direction = _normalized_vector(
            _coerce_float_vector(item.get("world_direction"), 3)
        )
        extent += abs(_vector_dot(axis, direction)) * float(item.get("extent") or 0.0)
    return float(extent)


def _robodojo_lift_delta(
    *,
    lift_vector: list[float] | None,
    lift_axis: Any | None,
    lift_distance: float | None,
) -> list[float]:
    if lift_vector is not None:
        delta = _coerce_float_vector(lift_vector, 3)
        if len(delta) != 3:
            raise ValueError("lift_vector_must_be_xyz")
        return delta
    if lift_distance is None:
        raise ValueError("lift_vector_or_lift_distance_required")
    axis = _resolve_robodojo_world_axis(
        lift_axis if lift_axis is not None else [0.0, 0.0, 1.0], {}, label="lift_axis"
    )
    return [axis[i] * float(lift_distance) for i in range(3)]


def robodojo_public_tool_contact_offset(
    repo_path: str | Path,
    *,
    robots: list[JsonDict],
    active_side: str,
) -> JsonDict:
    """Derive finger contact geometry from public robot config/URDF."""

    robot = next(
        (
            item
            for item in robots
            if str(item.get("arm_name") or "").startswith(active_side)
        ),
        {},
    )
    robot_name = str(robot.get("robot_name") or "")
    config_path = Path(repo_path) / "Assets/Robots" / robot_name / "robot_config.yml"
    if not robot_name or not config_path.exists():
        return {
            "available": False,
            "source": "public_robot_urdf",
            "error": "robot_config_missing",
            "world_xy": [0.0, 0.0],
        }
    config = _load_yaml(config_path)
    urdf_value = str(config.get("urdf_path") or "")
    urdf_path = (config_path.parent / urdf_value).resolve() if urdf_value else None
    gripper_joints = [str(item) for item in config.get("gripper_joints_name") or []]
    if urdf_path is None or not urdf_path.exists() or not gripper_joints:
        return {
            "available": False,
            "source": "public_robot_urdf",
            "robot_name": robot_name,
            "robot_config_path": str(config_path),
            "error": "urdf_or_gripper_joints_missing",
            "world_xy": [0.0, 0.0],
        }
    try:
        import xml.etree.ElementTree as ET

        root = ET.parse(urdf_path).getroot()
        contacts: list[JsonDict] = []
        links = {str(link.attrib.get("name")): link for link in root.findall("link")}
        for joint in root.findall("joint"):
            joint_name = str(joint.attrib.get("name"))
            if joint_name not in gripper_joints:
                continue
            origin = joint.find("origin")
            if origin is None:
                continue
            xyz = [
                float(item)
                for item in str(origin.attrib.get("xyz") or "0 0 0").split()[:3]
            ]
            if len(xyz) == 3:
                axis_node = joint.find("axis")
                joint_axis = _normalized_vector(
                    _coerce_float_vector(
                        str(axis_node.attrib.get("xyz") or "0 0 0").split()
                        if axis_node is not None
                        else None,
                        3,
                    )
                )
                limit_node = joint.find("limit")
                joint_limits = [
                    float(limit_node.attrib.get("lower", 0.0))
                    if limit_node is not None
                    else 0.0,
                    float(limit_node.attrib.get("upper", 0.0))
                    if limit_node is not None
                    else 0.0,
                ]
                child = joint.find("child")
                child_link_name = (
                    str(child.attrib.get("link")) if child is not None else ""
                )
                child_link = links.get(child_link_name)
                mesh = (
                    child_link.find("collision/geometry/mesh")
                    if child_link is not None
                    else None
                )
                filename = (
                    str(mesh.attrib.get("filename") or "") if mesh is not None else ""
                )
                mesh_path = (
                    (urdf_path.parent / filename).resolve()
                    if filename and not filename.startswith("package://")
                    else None
                )
                mesh_center = (
                    _stl_bbox_center(mesh_path) if mesh_path is not None else None
                )
                local_center = list(xyz)
                center_source = "gripper_joint_origin"
                if mesh_center is not None:
                    scale = _coerce_float_vector(
                        str(mesh.attrib.get("scale") or "1 1 1").split(), 3, fill=1.0
                    )
                    collision = child_link.find("collision")
                    collision_origin = (
                        collision.find("origin") if collision is not None else None
                    )
                    collision_xyz = _coerce_float_vector(
                        str(collision_origin.attrib.get("xyz") or "0 0 0").split()
                        if collision_origin is not None
                        else None,
                        3,
                    )
                    local_center = [
                        xyz[index]
                        + collision_xyz[index]
                        + mesh_center[index] * scale[index]
                        for index in range(3)
                    ]
                    center_source = "gripper_collision_mesh_bbox_center"
                contacts.append(
                    {
                        "joint_name": joint_name,
                        "child_link": child_link_name,
                        "local_center": local_center,
                        "joint_axis": joint_axis,
                        "joint_limits": joint_limits,
                        "center_source": center_source,
                    }
                )
        if not contacts:
            raise ValueError("gripper_joint_origins_missing")
        contact_sources = [item["local_center"] for item in contacts]
        local_center = [
            sum(item[index] for item in contact_sources) / len(contact_sources)
            for index in range(3)
        ]
        if len(contact_sources) >= 2:
            jaw_delta = [
                contact_sources[1][index] - contact_sources[0][index]
                for index in range(3)
            ]
            local_jaw_axis = _normalized_vector(jaw_delta)
            contact_opening = math.sqrt(sum(value * value for value in jaw_delta))
        else:
            local_jaw_axis = [0.0, 0.0, 0.0]
            contact_opening = 0.0
        local_approach_axis = _normalized_vector(local_center)
        gripper_move = (
            config.get("gripper_move")
            if isinstance(config.get("gripper_move"), dict)
            else {}
        )
        gripper_scale = _coerce_float_vector(config.get("gripper_scale"), 2)
        command_mapping = {
            "base_joint": str(gripper_move.get("base") or gripper_joints[0]),
            "sign": float(gripper_move.get("sign", 1.0)),
            "mimic": list(gripper_move.get("mimic") or []),
            "gripper_scale": gripper_scale,
        }
        contact_model = {
            "contacts": contacts,
            "gripper_command_mapping": command_mapping,
        }
        command_openings = (
            {
                "closed_command_0": _robodojo_contact_opening_for_command(
                    contact_model, 0.0
                ),
                "open_command_1": _robodojo_contact_opening_for_command(
                    contact_model, 1.0
                ),
            }
            if len(gripper_scale) == 2
            else {}
        )
        current_joint_positions = _coerce_float_vector(
            robot.get("gripper_state"), len(contacts)
        )
        current_contact_opening = _robodojo_contact_opening_at_joint_positions(
            contacts,
            current_joint_positions,
        )
        command_range = sorted(float(value) for value in command_openings.values()) or [
            contact_opening,
            contact_opening,
        ]
        ee_pose = _coerce_float_vector(robot.get("ee_pose"), 7)
        quaternion = ee_pose[3:7] if len(ee_pose) >= 7 else [1.0, 0.0, 0.0, 0.0]
        world_offset = _rotate_vector_by_quaternion_wxyz(local_center, quaternion)
        world_jaw_axis = _rotate_vector_by_quaternion_wxyz(local_jaw_axis, quaternion)
        world_approach_axis = _rotate_vector_by_quaternion_wxyz(
            local_approach_axis, quaternion
        )
        for contact in contacts:
            contact["local_center"] = [
                round(float(value), 6) for value in contact["local_center"]
            ]
            contact["world_offset"] = [
                round(value, 6)
                for value in _rotate_vector_by_quaternion_wxyz(
                    contact["local_center"], quaternion
                )
            ]
    except Exception as exc:
        return {
            "available": False,
            "source": "public_robot_urdf",
            "robot_name": robot_name,
            "robot_config_path": str(config_path),
            "urdf_path": str(urdf_path),
            "error": f"{type(exc).__name__}:{exc}",
            "world_xy": [0.0, 0.0],
        }
    return {
        "available": True,
        "source": (
            "public_robot_urdf_gripper_collision_center"
            if any(
                item["center_source"] == "gripper_collision_mesh_bbox_center"
                for item in contacts
            )
            else "public_robot_urdf_gripper_joint_origin"
        ),
        "robot_name": robot_name,
        "active_side": active_side,
        "robot_config_path": str(config_path),
        "urdf_path": str(urdf_path),
        "gripper_joints": gripper_joints,
        "collision_mesh_centers_used": sum(
            item["center_source"] == "gripper_collision_mesh_bbox_center"
            for item in contacts
        ),
        "contacts": contacts,
        "local_contact_center": [round(value, 6) for value in local_center],
        "local_jaw_axis": [round(value, 6) for value in local_jaw_axis],
        "local_approach_axis": [round(value, 6) for value in local_approach_axis],
        "contact_opening": round(contact_opening, 6),
        "contact_opening_range": [round(value, 6) for value in command_range],
        "current_contact_opening": (
            round(float(current_contact_opening), 6)
            if current_contact_opening is not None
            else None
        ),
        "current_gripper_joint_positions": current_joint_positions,
        "gripper_command_mapping": command_mapping,
        "gripper_command_openings": {
            key: round(float(value), 6) for key, value in command_openings.items()
        },
        "ee_quaternion_wxyz": quaternion,
        "world_xyz": [round(value, 6) for value in world_offset],
        "world_xy": [round(world_offset[0], 6), round(world_offset[1], 6)],
        "world_jaw_axis": [round(value, 6) for value in world_jaw_axis],
        "world_approach_axis": [round(value, 6) for value in world_approach_axis],
    }


def _robodojo_contact_opening_at_joint_positions(
    contacts: list[JsonDict],
    joint_positions: list[float],
) -> float | None:
    if len(contacts) < 2 or len(joint_positions) < len(contacts):
        return None
    centers: list[list[float]] = []
    for index, contact in enumerate(contacts):
        center = _coerce_float_vector(contact.get("local_center"), 3)
        axis = _coerce_float_vector(contact.get("joint_axis"), 3)
        limits = _coerce_float_vector(contact.get("joint_limits"), 2)
        if len(center) != 3 or len(axis) != 3:
            return None
        position = float(joint_positions[index])
        if len(limits) == 2:
            position = min(max(position, min(limits)), max(limits))
        centers.append([center[i] + axis[i] * position for i in range(3)])
    return _vector_norm(_vector_delta(centers[0], centers[1]))


def _robodojo_contact_opening_for_command(evidence: JsonDict, command: float) -> float:
    contacts = [
        dict(item) for item in evidence.get("contacts") or [] if isinstance(item, dict)
    ]
    if len(contacts) < 2:
        raise ValueError("paired_public_gripper_contacts_required")
    mapping = evidence.get("gripper_command_mapping")
    mapping = mapping if isinstance(mapping, dict) else {}
    scale = _coerce_float_vector(mapping.get("gripper_scale"), 2)
    if len(scale) != 2:
        raise ValueError("public_gripper_command_scale_required")
    normalized = min(max(float(command), 0.0), 1.0)
    if float(mapping.get("sign", 1.0)) == 1.0:
        base_position = normalized * (scale[1] - scale[0]) + scale[0]
    else:
        base_position = (1.0 - normalized) * (scale[1] - scale[0]) + scale[0]
    base_joint = str(mapping.get("base_joint") or "")
    mimic = list(mapping.get("mimic") or [])
    mimic_joint = str(mimic[0]) if mimic else ""
    mimic_scale = float(mimic[1]) if len(mimic) > 1 else 1.0
    mimic_offset = float(mimic[2]) if len(mimic) > 2 else 0.0
    positions: list[float] = []
    for contact in contacts:
        joint_name = str(contact.get("joint_name") or "")
        if joint_name == base_joint:
            positions.append(base_position)
        elif joint_name == mimic_joint:
            positions.append(base_position * mimic_scale + mimic_offset)
        else:
            positions.append(base_position)
    opening = _robodojo_contact_opening_at_joint_positions(contacts, positions)
    if opening is None:
        raise ValueError("public_contact_opening_unavailable")
    return float(opening)


def _robodojo_gripper_command_for_opening(
    evidence: JsonDict, requested_opening: float
) -> float:
    requested = float(requested_opening)
    low_command = 0.0
    high_command = 1.0
    low_opening = _robodojo_contact_opening_for_command(evidence, low_command)
    high_opening = _robodojo_contact_opening_for_command(evidence, high_command)
    increasing = high_opening >= low_opening
    minimum = min(low_opening, high_opening)
    maximum = max(low_opening, high_opening)
    if requested <= minimum:
        return low_command if increasing else high_command
    if requested >= maximum:
        return high_command if increasing else low_command
    for _ in range(60):
        midpoint = (low_command + high_command) * 0.5
        opening = _robodojo_contact_opening_for_command(evidence, midpoint)
        if (opening < requested) == increasing:
            low_command = midpoint
        else:
            high_command = midpoint
    return round((low_command + high_command) * 0.5, 6)


def _stl_bbox_center(path: Path | None) -> list[float] | None:
    if path is None or not path.is_file():
        return None
    data = path.read_bytes()
    vertices: list[tuple[float, float, float]] = []
    if len(data) >= 84:
        triangle_count = struct.unpack_from("<I", data, 80)[0]
        expected_size = 84 + triangle_count * 50
        if triangle_count > 0 and expected_size <= len(data):
            for triangle_index in range(triangle_count):
                vertex_offset = 84 + triangle_index * 50 + 12
                for vertex_index in range(3):
                    vertices.append(
                        struct.unpack_from(
                            "<3f", data, vertex_offset + vertex_index * 12
                        )
                    )
    if not vertices:
        for raw_line in data.decode("utf-8", errors="ignore").splitlines():
            parts = raw_line.strip().split()
            if len(parts) == 4 and parts[0].lower() == "vertex":
                try:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
    if not vertices:
        return None
    return [
        (
            min(vertex[axis] for vertex in vertices)
            + max(vertex[axis] for vertex in vertices)
        )
        * 0.5
        for axis in range(3)
    ]


def _rotate_vector_by_quaternion_wxyz(
    vector: list[float], quaternion: list[float]
) -> list[float]:
    w, x, y, z = _coerce_float_vector(quaternion, 4)
    vx, vy, vz = _coerce_float_vector(vector, 3)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _vector_dot(left: list[float], right: list[float]) -> float:
    return sum(float(left[index]) * float(right[index]) for index in range(3))


def _vector_cross(left: list[float], right: list[float]) -> list[float]:
    return [
        float(left[1]) * float(right[2]) - float(left[2]) * float(right[1]),
        float(left[2]) * float(right[0]) - float(left[0]) * float(right[2]),
        float(left[0]) * float(right[1]) - float(left[1]) * float(right[0]),
    ]


def _orthonormal_axis_pair(
    primary: list[float], secondary: list[float]
) -> tuple[list[float], list[float], list[float]]:
    first = _normalized_vector(primary)
    if _vector_norm(first) == 0.0:
        raise ValueError("contact_primary_axis_zero")
    projected = [
        float(secondary[i]) - _vector_dot(secondary, first) * first[i] for i in range(3)
    ]
    second = _normalized_vector(projected)
    if _vector_norm(second) == 0.0:
        raise ValueError("contact_axes_parallel")
    third = _normalized_vector(_vector_cross(first, second))
    return first, second, third


def _quaternion_from_rotation_matrix(matrix: list[list[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2][1] - matrix[1][2]) / scale
        y = (matrix[0][2] - matrix[2][0]) / scale
        z = (matrix[1][0] - matrix[0][1]) / scale
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        w = (matrix[2][1] - matrix[1][2]) / scale
        x = 0.25 * scale
        y = (matrix[0][1] + matrix[1][0]) / scale
        z = (matrix[0][2] + matrix[2][0]) / scale
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        w = (matrix[0][2] - matrix[2][0]) / scale
        x = (matrix[0][1] + matrix[1][0]) / scale
        y = 0.25 * scale
        z = (matrix[1][2] + matrix[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        w = (matrix[1][0] - matrix[0][1]) / scale
        x = (matrix[0][2] + matrix[2][0]) / scale
        y = (matrix[1][2] + matrix[2][1]) / scale
        z = 0.25 * scale
    quaternion = [w, x, y, z]
    norm = math.sqrt(sum(value * value for value in quaternion))
    return [value / norm for value in quaternion]


def _nlerp_quaternion_wxyz(
    start: list[float], end: list[float], fraction: float
) -> list[float]:
    """Interpolate unit quaternions along the shortest sign-equivalent arc."""

    start_values = _coerce_float_vector(start, 4)
    end_values = _coerce_float_vector(end, 4)
    if len(start_values) != 4 or len(end_values) != 4:
        raise ValueError("quaternion_interpolation_requires_wxyz")
    start_norm = math.sqrt(sum(value * value for value in start_values))
    end_norm = math.sqrt(sum(value * value for value in end_values))
    if start_norm <= 1e-12 or end_norm <= 1e-12:
        raise ValueError("quaternion_interpolation_requires_nonzero_quaternions")
    left = [value / start_norm for value in start_values]
    right = [value / end_norm for value in end_values]
    if sum(left[index] * right[index] for index in range(4)) < 0.0:
        right = [-value for value in right]
    alpha = min(max(float(fraction), 0.0), 1.0)
    blended = [(1.0 - alpha) * left[index] + alpha * right[index] for index in range(4)]
    norm = math.sqrt(sum(value * value for value in blended))
    if norm <= 1e-12:
        raise ValueError("quaternion_interpolation_degenerate")
    return [value / norm for value in blended]


def _quaternion_aligning_contact_axes(
    *,
    local_approach_axis: list[float],
    local_jaw_axis: list[float],
    world_approach_axis: list[float],
    world_jaw_axis: list[float],
) -> list[float]:
    local = _orthonormal_axis_pair(local_approach_axis, local_jaw_axis)
    world = _orthonormal_axis_pair(world_approach_axis, world_jaw_axis)
    matrix = [
        [sum(world[k][row] * local[k][column] for k in range(3)) for column in range(3)]
        for row in range(3)
    ]
    return _quaternion_from_rotation_matrix(matrix)


def _normalized_vector(vector: list[float]) -> list[float]:
    values = _coerce_float_vector(vector, 3)
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [value / norm for value in values]


def measure_robodojo_public_geometry(
    runtime_state_before: JsonDict | None,
    runtime_state_after: JsonDict | None = None,
    *,
    object_labels: list[str] | None = None,
    agent_context: JsonDict | None = None,
) -> JsonDict:
    """Measure public object/robot geometry without task semantics or thresholds."""

    before = _robodojo_public_runtime_state(runtime_state_before)
    after = (
        _robodojo_public_runtime_state(runtime_state_after)
        if runtime_state_after is not None
        else before
    )
    before_objects = _robodojo_runtime_objects(before)
    after_objects = _robodojo_runtime_objects(after)
    available_labels = sorted(set(before_objects) | set(after_objects))
    labels = (
        [str(label) for label in object_labels]
        if object_labels is not None
        else available_labels
    )
    objects: dict[str, JsonDict] = {}
    for label in labels:
        before_item = before_objects.get(label)
        after_item = after_objects.get(label)
        before_position = _robodojo_object_pose(before_objects, label)
        after_position = _robodojo_object_pose(after_objects, label)
        before_bbox = _robodojo_oriented_bbox_geometry(before_item)
        after_bbox = _robodojo_oriented_bbox_geometry(after_item)
        translation = _vector_delta(before_position, after_position)
        objects[label] = {
            "before_position": before_position,
            "after_position": after_position,
            "before_oriented_bbox": before_bbox,
            "after_oriented_bbox": after_bbox,
            "before_contact_center": before_bbox.get("world_center") or before_position,
            "after_contact_center": after_bbox.get("world_center") or after_position,
            "translation": translation,
            "translation_norm": _vector_norm(translation),
            "bbox_extent_before": _robodojo_object_extent(before_item),
            "bbox_extent_after": _robodojo_object_extent(after_item),
            "present_before": before_item is not None,
            "present_after": after_item is not None,
        }

    pairwise_distances: dict[str, float] = {}
    for index, left_label in enumerate(labels):
        left_position = objects.get(left_label, {}).get("after_contact_center")
        if not isinstance(left_position, list):
            continue
        for right_label in labels[index + 1 :]:
            right_position = objects.get(right_label, {}).get("after_contact_center")
            if not isinstance(right_position, list):
                continue
            pairwise_distances[f"{left_label}|{right_label}"] = _vector_norm(
                _vector_delta(left_position, right_position)
            )

    return {
        "objects": objects,
        "robots": _robodojo_robot_motion_deltas(
            _robodojo_runtime_robots(before), _robodojo_runtime_robots(after)
        ),
        "pairwise_distances": pairwise_distances,
        "measurement_contract": {
            "available_object_labels": available_labels,
            "caller_filter_applied": object_labels is not None,
            "task_id_read": False,
            "task_target_selected": False,
            "task_thresholds_applied": False,
            "verifier_read": False,
        },
        "agent_context": agent_context or {},
    }


def measure_robodojo_contact_state(
    runtime_state_before: JsonDict | None,
    runtime_state_after: JsonDict | None = None,
    *,
    object_label: str | None,
    arm: str | None,
    criteria: JsonDict | None = None,
    repo_path: str | Path | None = None,
    agent_context: JsonDict | None = None,
) -> JsonDict:
    """Measure caller-selected contact/holding/stability evidence.

    The function reports raw public geometry and evaluates only criteria supplied
    by the caller. It does not infer a task target or task completion condition.
    """

    label = str(object_label or "").strip()
    side = str(arm or "").strip().lower()
    if not label:
        raise ValueError("caller_selected_object_label_required")
    if side not in {"left", "right"}:
        raise ValueError("caller_selected_arm_required")

    before = _robodojo_public_runtime_state(runtime_state_before)
    after = (
        _robodojo_public_runtime_state(runtime_state_after)
        if runtime_state_after is not None
        else before
    )
    before_objects = _robodojo_runtime_objects(before)
    after_objects = _robodojo_runtime_objects(after)
    object_before_item = before_objects.get(label, {})
    object_after_item = after_objects.get(label, {})
    object_before_origin = _robodojo_object_pose(before_objects, label)
    object_after_origin = _robodojo_object_pose(after_objects, label)
    object_before_geometry = _robodojo_oriented_bbox_geometry(object_before_item)
    object_after_geometry = _robodojo_oriented_bbox_geometry(object_after_item)
    object_before = object_before_geometry.get("world_center") or object_before_origin
    object_after = object_after_geometry.get("world_center") or object_after_origin
    object_orientation_before = _robodojo_object_orientation(before_objects, label)
    object_orientation_after = _robodojo_object_orientation(after_objects, label)

    def selected_robot(state: JsonDict) -> JsonDict:
        return next(
            (
                item
                for item in _robodojo_runtime_robots(state)
                if str(item.get("arm_name") or "").startswith(side)
            ),
            {},
        )

    robot_before = selected_robot(before)
    robot_after = selected_robot(after)
    ee_before_pose = _coerce_float_vector(robot_before.get("ee_pose"), 7)
    ee_after_pose = _coerce_float_vector(robot_after.get("ee_pose"), 7)
    ee_before = ee_before_pose[:3] if len(ee_before_pose) >= 3 else None
    ee_after = ee_after_pose[:3] if len(ee_after_pose) >= 3 else None
    ee_orientation_before = ee_before_pose[3:7] if len(ee_before_pose) >= 7 else None
    ee_orientation_after = ee_after_pose[3:7] if len(ee_after_pose) >= 7 else None
    object_delta = _vector_delta(object_before, object_after)
    ee_delta = _vector_delta(ee_before, ee_after)
    comotion_delta = (
        [round(float(object_delta[i]) - float(ee_delta[i]), 6) for i in range(3)]
        if object_delta is not None and ee_delta is not None
        else None
    )
    separation_before = _vector_norm(_vector_delta(object_before, ee_before))
    separation_after = _vector_norm(_vector_delta(object_after, ee_after))
    public_contact_geometry = robodojo_public_tool_contact_offset(
        repo_path or DEFAULT_ROBODOJO_REPO,
        robots=_robodojo_runtime_robots(after),
        active_side=side,
    )

    def tool_contact_center(ee_pose: list[float] | None) -> list[float] | None:
        if (
            not public_contact_geometry.get("available")
            or ee_pose is None
            or len(ee_pose) < 7
        ):
            return None
        local_center = _coerce_float_vector(
            public_contact_geometry.get("local_contact_center"), 3
        )
        if len(local_center) != 3:
            return None
        world_offset = _rotate_vector_by_quaternion_wxyz(local_center, ee_pose[3:7])
        return [float(ee_pose[i]) + float(world_offset[i]) for i in range(3)]

    tool_contact_before = tool_contact_center(ee_before_pose)
    tool_contact_after = tool_contact_center(ee_after_pose)
    tool_object_vector_before = _vector_delta(object_before, tool_contact_before)
    tool_object_vector_after = _vector_delta(object_after, tool_contact_after)
    gripper_before = _coerce_float_vector(
        robot_before.get("gripper_state"),
        len(robot_before.get("gripper_state") or []) or 1,
    )
    gripper_after = _coerce_float_vector(
        robot_after.get("gripper_state"),
        len(robot_after.get("gripper_state") or []) or 1,
    )

    public_native_contact = {
        key: _jsonable(value)
        for item in (object_after_item, robot_after)
        for key, value in item.items()
        if key
        in {"contact", "contacts", "contact_state", "grasp_state", "holding_state"}
    }
    contact_evidence = {
        "object_label": label,
        "arm": side,
        "object_position_before": object_before,
        "object_position_after": object_after,
        "object_pose_origin_before": object_before_origin,
        "object_pose_origin_after": object_after_origin,
        "object_oriented_bbox_before": object_before_geometry,
        "object_oriented_bbox_after": object_after_geometry,
        "object_orientation_before_wxyz": object_orientation_before,
        "object_orientation_after_wxyz": object_orientation_after,
        "object_orientation_change_degrees": _quaternion_angular_distance_degrees(
            object_orientation_before,
            object_orientation_after,
        ),
        "ee_position_before": ee_before,
        "ee_position_after": ee_after,
        "ee_orientation_before_wxyz": ee_orientation_before,
        "ee_orientation_after_wxyz": ee_orientation_after,
        "ee_orientation_change_degrees": _quaternion_angular_distance_degrees(
            ee_orientation_before,
            ee_orientation_after,
        ),
        "ee_object_distance_before": separation_before,
        "ee_object_distance_after": separation_after,
        "tool_contact_center_before": tool_contact_before,
        "tool_contact_center_after": tool_contact_after,
        "tool_contact_to_object_vector_before": tool_object_vector_before,
        "tool_contact_to_object_vector_after": tool_object_vector_after,
        "tool_contact_object_distance_before": _vector_norm(tool_object_vector_before),
        "tool_contact_object_distance_after": _vector_norm(tool_object_vector_after),
        "public_tool_contact_geometry": public_contact_geometry,
        "bbox_extent": _robodojo_object_extent(object_after_item),
        "native_contact_state": public_native_contact or None,
    }
    holding_evidence = {
        "object_translation": object_delta,
        "object_translation_norm": _vector_norm(object_delta),
        "ee_translation": ee_delta,
        "ee_translation_norm": _vector_norm(ee_delta),
        "comotion_residual": comotion_delta,
        "comotion_residual_norm": _vector_norm(comotion_delta),
        "gripper_state_before": gripper_before,
        "gripper_state_after": gripper_after,
        "gripper_delta": _component_delta(gripper_before, gripper_after),
    }
    before_env = (
        before.get("env_summary") if isinstance(before.get("env_summary"), dict) else {}
    )
    after_env = (
        after.get("env_summary") if isinstance(after.get("env_summary"), dict) else {}
    )
    stability_evidence = {
        "unstable_nums_before": _jsonable(before_env.get("unstable_nums")),
        "unstable_nums_after": _jsonable(after_env.get("unstable_nums")),
        "object_translation_norm": _vector_norm(object_delta),
        "ee_translation_norm": _vector_norm(ee_delta),
    }

    supplied = dict(criteria or {})
    evaluated: dict[str, bool] = {}
    values = {
        "max_ee_object_distance": separation_after,
        "max_comotion_residual": holding_evidence["comotion_residual_norm"],
        "max_object_motion": holding_evidence["object_translation_norm"],
        "min_object_motion": holding_evidence["object_translation_norm"],
        "max_gripper_value": max(gripper_after) if gripper_after else None,
        "min_gripper_value": min(gripper_after) if gripper_after else None,
        "max_object_orientation_change_degrees": contact_evidence[
            "object_orientation_change_degrees"
        ],
        "min_object_orientation_change_degrees": contact_evidence[
            "object_orientation_change_degrees"
        ],
        "max_ee_orientation_change_degrees": contact_evidence[
            "ee_orientation_change_degrees"
        ],
        "min_ee_orientation_change_degrees": contact_evidence[
            "ee_orientation_change_degrees"
        ],
    }
    for key, threshold in supplied.items():
        if key not in values or values[key] is None:
            continue
        value = float(values[key])
        limit = float(threshold)
        evaluated[key] = value >= limit if key.startswith("min_") else value <= limit

    return {
        "contact_evidence": contact_evidence,
        "holding_evidence": holding_evidence,
        "stability_evidence": stability_evidence,
        "criteria": {
            "caller_supplied": supplied,
            "evaluated": evaluated,
            "all_met": all(evaluated.values()) if evaluated else None,
        },
        "measurement_contract": {
            "caller_selected_object": True,
            "caller_selected_arm": True,
            "default_thresholds_used": False,
            "task_completion_condition_read": False,
            "verifier_read": False,
        },
        "agent_context": agent_context or {},
    }


def validate_robodojo_low_level_actions(
    repo_path: str | Path,
    env_cfg: str,
    *,
    actions: list[JsonDict] | None,
    runtime_state: JsonDict | None = None,
) -> list[JsonDict]:
    """Validate caller-authored action dictionaries without changing their order."""

    if not isinstance(actions, list) or not actions:
        raise ValueError("caller_authored_low_level_actions_required")
    runtime = _robodojo_public_runtime_state(runtime_state)
    schema = _robodojo_action_schema_with_public_runtime_fallback(
        robodojo_robot_action_schema(repo_path, env_cfg),
        robots=_robodojo_runtime_robots(runtime),
    )
    action_keys = (
        schema.get("action_keys") if isinstance(schema.get("action_keys"), dict) else {}
    )
    expected: dict[str, int] = {}
    expected.update(
        {
            str(key): int(dim)
            for key, dim in dict(action_keys.get("joint") or {}).items()
        }
    )
    expected.update(
        {str(key): int(dim) for key, dim in dict(action_keys.get("ee") or {}).items()}
    )
    clean: list[JsonDict] = []
    for index, raw_action in enumerate(actions):
        if not isinstance(raw_action, dict) or not raw_action:
            raise ValueError(f"action_{index}_dictionary_required")
        unknown = sorted(set(raw_action) - set(expected))
        if unknown:
            raise ValueError(f"action_{index}_unknown_keys:{','.join(unknown)}")
        action: JsonDict = {}
        for key, value in raw_action.items():
            vector = _coerce_float_vector(value, expected[key])
            if len(vector) != expected[key]:
                raise ValueError(f"action_{index}_invalid_dim:{key}")
            action[key] = vector
        clean.append(action)
    return clean


def _robodojo_action_schema_with_public_runtime_fallback(
    schema: JsonDict,
    *,
    robots: list[JsonDict],
) -> JsonDict:
    """Fill a missing repository schema from public robot-state dimensions.

    RoboDojo runtime reports expose each arm's action-key names and public
    joint/EE state.  That is enough to validate the pose and joint actions used
    by this adapter when the source checkout is not mounted (for example while
    replaying a captured public observation).  The fallback never reads task
    configuration, reward state, or a success predicate.
    """

    action_keys = (
        schema.get("action_keys") if isinstance(schema.get("action_keys"), dict) else {}
    )
    existing_joint = dict(action_keys.get("joint") or {})
    existing_ee = dict(action_keys.get("ee") or {})
    if existing_joint or existing_ee or not robots:
        return schema

    joint: dict[str, int] = {}
    ee: dict[str, int] = {}
    arm_dims: list[int] = []
    ee_dims: list[int] = []
    for robot in robots:
        public_keys = (
            robot.get("action_keys")
            if isinstance(robot.get("action_keys"), dict)
            else {}
        )
        arm_key = public_keys.get("arm_joint")
        joint_state = _coerce_float_vector(
            robot.get("joint_state"), len(robot.get("joint_state") or [])
        )
        if isinstance(arm_key, str) and joint_state:
            joint[arm_key] = len(joint_state)
            arm_dims.append(len(joint_state))

        gripper_key = public_keys.get("gripper_joint")
        if isinstance(gripper_key, str):
            # RoboDojo's public policy interface commands each end effector
            # with one normalized scalar even when the observed mechanism has
            # multiple coupled finger joints.
            joint[gripper_key] = 1
            ee_dims.append(1)

        ee_pose_key = public_keys.get("ee_pose")
        ee_pose = _coerce_float_vector(robot.get("ee_pose"), 7)
        if isinstance(ee_pose_key, str) and len(ee_pose) == 7:
            ee[ee_pose_key] = 7

    if not joint and not ee:
        return schema
    return {
        **schema,
        "robot_info_available": False,
        "arm_dim": arm_dims,
        "ee_dim": ee_dims,
        "arm_count": len(robots),
        "action_keys": {"joint": joint, "ee": ee},
        "notes": (
            "Repository robot_info was unavailable; action names and dimensions "
            "were derived only from the public RoboDojo robot-state contract."
        ),
    }


def robodojo_action_dims(actions: list[JsonDict]) -> list[JsonDict]:
    dims: list[JsonDict] = []
    for action in actions:
        dims.append(
            {
                key: len(value) if isinstance(value, list) else None
                for key, value in action.items()
            }
        )
    return dims


def _robodojo_public_runtime_state(value: Any | None) -> JsonDict:
    raw = _primitive_output(value)
    if isinstance(raw, dict) and isinstance(raw.get("public_runtime_state"), dict):
        return raw["public_runtime_state"]
    if isinstance(raw, dict) and isinstance(
        raw.get("public_runtime_state_after_action"), dict
    ):
        return raw["public_runtime_state_after_action"]
    return raw if isinstance(raw, dict) else {}


def _robodojo_visual_observation_contract(value: Any | None) -> JsonDict:
    runtime = _robodojo_public_runtime_state(value)
    observation = (
        runtime.get("observation")
        if isinstance(runtime.get("observation"), dict)
        else runtime
    )
    if not isinstance(observation, dict):
        observation = {}
    contract, refs = inspect_robodojo_rgb_observation(observation)
    return {
        **contract,
        "evidence_refs": refs,
        "source_available": bool(value is not None),
        "vision_summary_is_visual_evidence": False,
    }


def _robodojo_policy_observation_from_runtime(value: JsonDict) -> JsonDict:
    runtime = _robodojo_public_runtime_state(value)
    nested = runtime.get("observation")
    if not isinstance(nested, dict):
        return runtime
    observation = _shallow_container_copy(nested)
    for key in ("robot", "scene", "env_summary"):
        if key in runtime and key not in observation:
            observation[key] = _shallow_container_copy(runtime[key])
    return observation


def _shallow_container_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _shallow_container_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_shallow_container_copy(item) for item in value)
    if isinstance(value, list):
        return list(value)
    return value


def _robodojo_value_at_path(observation: JsonDict, source_path: str) -> Any:
    value: Any = observation
    for part in source_path.split("."):
        if not isinstance(value, dict) or part not in value:
            raise KeyError(f"visual evidence source path is unavailable: {source_path}")
        value = value[part]
    return value


def _raw_rgb_frame_metadata(frame: Any) -> JsonDict:
    shape = getattr(frame, "shape", None)
    dtype = getattr(frame, "dtype", None)
    try:
        payload = (
            frame.tobytes()
            if hasattr(frame, "tobytes")
            else json.dumps(frame, sort_keys=True, default=str).encode()
        )
        sha256 = hashlib.sha256(payload).hexdigest()
    except Exception:
        sha256 = None
    return {
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else type(frame).__name__,
        "sha256": sha256,
        "frames_fabricated": False,
    }


def _normalize_robodojo_evidence_refs(
    evidence_refs: list[Any] | None,
) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for ref in evidence_refs or []:
        if isinstance(ref, str) and ref.strip():
            normalized.append({"artifact_id": ref.strip()})
        elif isinstance(ref, dict) and ref:
            normalized.append(
                {str(key): _jsonable(value) for key, value in ref.items()}
            )
        else:
            raise ValueError("evidence_refs must contain non-empty strings or mappings")
    return normalized


def _robodojo_public_runtime_state_after_action(value: Any | None) -> JsonDict:
    raw = _primitive_output(value)
    if isinstance(raw, dict) and isinstance(
        raw.get("public_runtime_state_after_action"), dict
    ):
        return raw["public_runtime_state_after_action"]
    return {}


def _robodojo_runtime_objects(runtime_state: JsonDict) -> dict[str, JsonDict]:
    scene = runtime_state.get("scene") if isinstance(runtime_state, dict) else {}
    objects = scene.get("objects") if isinstance(scene, dict) else []
    by_label: dict[str, JsonDict] = {}
    if isinstance(objects, list):
        for item in objects:
            if not isinstance(item, dict):
                continue
            label = item.get("label")
            if label is not None:
                by_label[str(label)] = item
    return by_label


def _robodojo_runtime_robots(runtime_state: JsonDict) -> list[JsonDict]:
    robot = runtime_state.get("robot") if isinstance(runtime_state, dict) else {}
    robots = robot.get("robots") if isinstance(robot, dict) else []
    return (
        [dict(item) for item in robots if isinstance(item, dict)]
        if isinstance(robots, list)
        else []
    )


def _robodojo_object_pose(
    objects: dict[str, JsonDict], label: str | None
) -> list[float] | None:
    if label is None:
        return None
    item = objects.get(str(label))
    pose = item.get("pose") if isinstance(item, dict) else None
    position = pose.get("position") if isinstance(pose, dict) else None
    if position is None:
        return None
    vector = _coerce_float_vector(position, 3)
    return vector if len(vector) >= 3 else None


def _robodojo_object_orientation(
    objects: dict[str, JsonDict], label: str | None
) -> list[float] | None:
    if label is None:
        return None
    item = objects.get(str(label))
    pose = item.get("pose") if isinstance(item, dict) else None
    quaternion = pose.get("quaternion_wxyz") if isinstance(pose, dict) else None
    if quaternion is None:
        return None
    vector = _coerce_float_vector(quaternion, 4)
    return vector if len(vector) == 4 else None


def _select_robodojo_arm_side(*, arm: str | None) -> str:
    requested = str(arm or "").strip().lower()
    if requested not in {"left", "right"}:
        raise ValueError("caller_selected_arm_required:left_or_right")
    return requested


def _robodojo_hold_ee_poses(
    schema: JsonDict, *, robots: list[JsonDict]
) -> dict[str, list[float]]:
    hold: dict[str, list[float]] = {}
    for robot in robots:
        action_keys = (
            robot.get("action_keys")
            if isinstance(robot.get("action_keys"), dict)
            else {}
        )
        ee_key = action_keys.get("ee_pose")
        if isinstance(ee_key, str):
            pose = _coerce_float_vector(robot.get("ee_pose"), 7)
            if len(pose) == 7:
                hold[ee_key] = pose
    return hold


def _robodojo_hold_gripper_states(
    schema: JsonDict,
    *,
    robots: list[JsonDict],
    repo_path: str | Path,
) -> dict[str, list[float]]:
    joint_keys = dict((schema.get("action_keys") or {}).get("joint") or {})
    hold: dict[str, list[float]] = {}
    for robot in robots:
        action_keys = (
            robot.get("action_keys")
            if isinstance(robot.get("action_keys"), dict)
            else {}
        )
        key = action_keys.get("gripper_joint")
        if not isinstance(key, str) or key not in joint_keys:
            continue
        expected_dim = int(joint_keys.get(key) or 1)
        value = _coerce_float_vector(robot.get("gripper_state"), expected_dim)
        if len(value) == expected_dim:
            robot_name = str(robot.get("robot_name") or "")
            config_path = (
                Path(repo_path) / "Assets/Robots" / robot_name / "robot_config.yml"
            )
            config = (
                _load_yaml(config_path) if robot_name and config_path.exists() else {}
            )
            scale = _coerce_float_vector(config.get("gripper_scale"), 2)
            gripper_move = (
                config.get("gripper_move")
                if isinstance(config.get("gripper_move"), dict)
                else {}
            )
            if len(scale) == 2 and abs(scale[1] - scale[0]) > 1e-12:
                normalized = (float(value[0]) - scale[0]) / (scale[1] - scale[0])
                if float(gripper_move.get("sign", 1.0)) != 1.0:
                    normalized = 1.0 - normalized
                normalized = min(max(normalized, 0.0), 1.0)
                hold[key] = [normalized] * expected_dim
            else:
                hold[key] = value
    return hold


def _robodojo_active_orientation(
    hold_poses: dict[str, list[float]], active_side: str
) -> list[float]:
    key = f"{active_side}_ee_pose" if active_side in {"left", "right"} else "ee_pose"
    pose = hold_poses.get(key)
    if pose and len(pose) >= 7:
        return [float(value) for value in pose[3:7]]
    return []


def _robodojo_ee_action_from_pose(
    schema: JsonDict,
    *,
    active_side: str,
    active_pose: list[float],
    gripper_value: float,
    hold_poses: dict[str, list[float]],
    hold_grippers: dict[str, list[float]],
) -> JsonDict:
    action: JsonDict = {}
    ee_keys = dict((schema.get("action_keys") or {}).get("ee") or {})
    joint_keys = dict((schema.get("action_keys") or {}).get("joint") or {})
    active_pose_key = (
        f"{active_side}_ee_pose" if active_side in {"left", "right"} else "ee_pose"
    )
    for key in ee_keys:
        if key not in {"ee_pose", "left_ee_pose", "right_ee_pose"}:
            continue
        pose = active_pose if key == active_pose_key else hold_poses.get(key)
        if pose is None:
            raise ValueError(f"public_current_ee_pose_required:{key}")
        action[key] = _coerce_float_vector(pose, 7)
    for key, dim in joint_keys.items():
        if "_ee_" not in key and not key.startswith("ee_") and "gripper" not in key:
            continue
        expected_dim = int(dim) if dim is not None else 1
        side = (
            "left"
            if key.startswith("left_")
            else "right"
            if key.startswith("right_")
            else active_side
        )
        if side == active_side:
            action[key] = _coerce_float_vector(
                [float(gripper_value)], expected_dim, fill=float(gripper_value)
            )
            continue
        value = hold_grippers.get(key)
        if value is None:
            raise ValueError(f"public_current_gripper_state_required:{key}")
        action[key] = _coerce_float_vector(value, expected_dim)
    return action


def _robodojo_object_extent(item: JsonDict | None) -> list[float] | None:
    bbox = item.get("bbox") if isinstance(item, dict) else None
    extent = bbox.get("local_extent") if isinstance(bbox, dict) else None
    if extent is None:
        return None
    vector = _coerce_float_vector(extent, 3)
    return vector if len(vector) >= 3 else None


def _robodojo_oriented_bbox_geometry(item: JsonDict | None) -> JsonDict:
    if not isinstance(item, dict):
        return {}
    pose = item.get("pose") if isinstance(item.get("pose"), dict) else {}
    bbox = item.get("bbox") if isinstance(item.get("bbox"), dict) else {}
    position = _coerce_float_vector(pose.get("position"), 3)
    quaternion = _coerce_float_vector(pose.get("quaternion_wxyz"), 4)
    local_min = _coerce_float_vector(bbox.get("local_min"), 3)
    local_max = _coerce_float_vector(bbox.get("local_max"), 3)
    if not (
        len(position) == 3
        and len(quaternion) == 4
        and len(local_min) == 3
        and len(local_max) == 3
    ):
        return {}
    local_center = [(local_min[i] + local_max[i]) * 0.5 for i in range(3)]
    local_extent = [local_max[i] - local_min[i] for i in range(3)]
    rotated_center = _rotate_vector_by_quaternion_wxyz(local_center, quaternion)
    world_center = [position[i] + rotated_center[i] for i in range(3)]
    local_basis = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])
    axis_names = ("x", "y", "z")
    axes = [
        {
            "local_axis": axis_names[index],
            "world_direction": _normalized_vector(
                _rotate_vector_by_quaternion_wxyz(list(local_basis[index]), quaternion)
            ),
            "extent": local_extent[index],
        }
        for index in range(3)
    ]
    corners: list[list[float]] = []
    for x in (local_min[0], local_max[0]):
        for y in (local_min[1], local_max[1]):
            for z in (local_min[2], local_max[2]):
                rotated = _rotate_vector_by_quaternion_wxyz([x, y, z], quaternion)
                corners.append([position[i] + rotated[i] for i in range(3)])
    world_min = [min(corner[i] for corner in corners) for i in range(3)]
    world_max = [max(corner[i] for corner in corners) for i in range(3)]
    return {
        "world_center": [round(float(value), 6) for value in world_center],
        "local_center": [round(float(value), 6) for value in local_center],
        "local_extent": [round(float(value), 6) for value in local_extent],
        "world_axes": [
            {
                **axis,
                "world_direction": [
                    round(float(value), 6) for value in axis["world_direction"]
                ],
                "extent": round(float(axis["extent"]), 6),
            }
            for axis in axes
        ],
        "world_aabb_min": [round(float(value), 6) for value in world_min],
        "world_aabb_max": [round(float(value), 6) for value in world_max],
        "world_aabb_extent": [
            round(float(world_max[i] - world_min[i]), 6) for i in range(3)
        ],
        "source": "public_pose_plus_local_bbox",
    }


def _robodojo_object_motion_deltas(
    before: dict[str, JsonDict], after: dict[str, JsonDict]
) -> dict[str, JsonDict]:
    labels = sorted(set(before) | set(after))
    deltas: dict[str, JsonDict] = {}
    for label in labels:
        before_pose = _robodojo_object_pose(before, label)
        after_pose = _robodojo_object_pose(after, label)
        delta = _vector_delta(before_pose, after_pose)
        deltas[label] = {
            "before_position": before_pose,
            "after_position": after_pose,
            "translation": delta,
            "translation_norm": _vector_norm(delta),
        }
    return deltas


def _robodojo_robot_motion_deltas(
    before: list[JsonDict], after: list[JsonDict]
) -> list[JsonDict]:
    before_by_name = {
        str(item.get("arm_name") or index): item for index, item in enumerate(before)
    }
    after_by_name = {
        str(item.get("arm_name") or index): item for index, item in enumerate(after)
    }
    names = sorted(set(before_by_name) | set(after_by_name))
    deltas: list[JsonDict] = []
    for name in names:
        before_robot = before_by_name.get(name, {})
        after_robot = after_by_name.get(name, {})
        before_ee = _coerce_float_vector(before_robot.get("ee_pose"), 7)
        after_ee = _coerce_float_vector(after_robot.get("ee_pose"), 7)
        ee_delta = _vector_delta(before_ee[:3], after_ee[:3])
        gripper_delta = _vector_delta(
            _coerce_float_vector(
                before_robot.get("gripper_state"),
                len(before_robot.get("gripper_state") or []) or 1,
            ),
            _coerce_float_vector(
                after_robot.get("gripper_state"),
                len(after_robot.get("gripper_state") or []) or 1,
            ),
        )
        deltas.append(
            {
                "arm_name": name,
                "ee_position_before": before_ee[:3] if len(before_ee) >= 3 else None,
                "ee_position_after": after_ee[:3] if len(after_ee) >= 3 else None,
                "ee_translation": ee_delta,
                "ee_translation_norm": _vector_norm(ee_delta),
                "gripper_delta_norm": _vector_norm(gripper_delta),
            }
        )
    return deltas


def _sanitize_robodojo_action_step_trace(action_step: JsonDict) -> JsonDict:
    steps = (
        action_step.get("steps") if isinstance(action_step.get("steps"), list) else []
    )
    existing_samples = (
        action_step.get("step_samples")
        if isinstance(action_step.get("step_samples"), list)
        else []
    )
    samples = []
    for index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        sample = (
            step.get("submitted_action_sample")
            if isinstance(step.get("submitted_action_sample"), dict)
            else {}
        )
        controller_diagnostics = _sanitize_robodojo_controller_diagnostics(
            step.get("controller_diagnostics")
        )
        samples.append(
            {
                "index": index,
                "ok": bool(step.get("ok")),
                "submitted_action_type": step.get("submitted_action_type"),
                "submitted_action_keys": list(step.get("submitted_action_keys") or []),
                "submitted_action_sample": _jsonable(sample),
                "take_action_cnt_before": step.get("take_action_cnt_before"),
                "take_action_cnt_after": step.get("take_action_cnt_after"),
                "control_empty_before": step.get("control_empty_before"),
                "control_empty_after": step.get("control_empty_after"),
                "controller_diagnostics": controller_diagnostics,
            }
        )
    if not samples and existing_samples:
        samples = [
            _jsonable(item) for item in existing_samples if isinstance(item, dict)
        ]
    reobservations = [
        {
            "repeat_index": item.get("repeat_index"),
            "action_index": item.get("action_index"),
            "runtime_state": _jsonable(item.get("runtime_state")),
        }
        for item in action_step.get("reobservations") or []
        if isinstance(item, dict) and isinstance(item.get("runtime_state"), dict)
    ]
    controller_feedback = _summarize_robodojo_controller_feedback(samples)
    return {
        "ok": bool(action_step.get("ok")),
        "action_boundary_executed": bool(action_step.get("ok"))
        and int(action_step.get("executed_step_count") or 0) > 0,
        "action_source": action_step.get("action_source"),
        "repeat_steps": action_step.get("repeat_steps"),
        "submitted_action_count": action_step.get("submitted_action_count"),
        "executed_step_count": action_step.get("executed_step_count"),
        "submitted_action_types": list(action_step.get("submitted_action_types") or []),
        "submitted_action_keys": list(action_step.get("submitted_action_keys") or []),
        "submitted_action_dims": _jsonable(
            action_step.get("submitted_action_dims") or {}
        ),
        "take_action_cnt_before": action_step.get("take_action_cnt_before"),
        "take_action_cnt_after": action_step.get("take_action_cnt_after"),
        "control_empty_before": action_step.get("control_empty_before"),
        "control_empty_after": action_step.get("control_empty_after"),
        "step_samples": samples,
        "controller_feedback": controller_feedback,
        "reobservations": reobservations,
        "public_state_reobserved_after_each_action": bool(
            (action_step.get("execution_contract") or {}).get(
                "public_state_reobserved_after_each_action"
            )
        ),
        "private_verifier_fields_removed": True,
    }


def _sanitize_robodojo_controller_diagnostics(value: Any) -> JsonDict:
    if not isinstance(value, dict):
        return {}
    ik_results = []
    for item in value.get("ik_results") or []:
        if not isinstance(item, dict):
            continue
        ik_results.append(
            {
                key: _jsonable(item.get(key))
                for key in (
                    "arm_name",
                    "target_pose",
                    "status",
                    "joint_value",
                    "joint_value_dim",
                    "error",
                )
                if key in item
            }
        )
    tracking: dict[str, JsonDict] = {}
    raw_tracking = (
        value.get("tracking") if isinstance(value.get("tracking"), dict) else {}
    )
    allowed_tracking = {
        "requested_ee_pose",
        "requested_arm_joint_state",
        "actual_ee_pose_before",
        "actual_ee_pose_after",
        "actual_arm_joint_state_before",
        "actual_arm_joint_state_after",
        "actual_gripper_joint_state_before",
        "actual_gripper_joint_state_after",
        "ee_position_error",
        "ee_orientation_error_radians",
        "arm_joint_l2_error",
    }
    for side, raw in raw_tracking.items():
        if not isinstance(raw, dict):
            continue
        tracking[str(side)] = {
            key: _jsonable(raw.get(key)) for key in allowed_tracking if key in raw
        }
    return {
        "contract": str(value.get("contract") or "controller-execution-diagnostics-v1"),
        "task_signal_read": bool(value.get("task_signal_read")),
        "reward_or_verifier_read": bool(value.get("reward_or_verifier_read")),
        "ik_attempt_count": int(value.get("ik_attempt_count") or len(ik_results)),
        "ik_success_count": int(
            value.get("ik_success_count")
            if value.get("ik_success_count") is not None
            else sum(str(item.get("status")) == "Success" for item in ik_results)
        ),
        "ik_results": ik_results,
        "tracking": tracking,
    }


def _robodojo_controller_diagnostics_from_result(value: Any) -> list[JsonDict]:
    diagnostics: list[JsonDict] = []
    if isinstance(value, dict):
        direct = _sanitize_robodojo_controller_diagnostics(
            value.get("controller_diagnostics")
        )
        if direct:
            diagnostics.append(direct)
        for key in ("steps", "execution_results"):
            for item in value.get(key) or []:
                diagnostics.extend(_robodojo_controller_diagnostics_from_result(item))
    elif isinstance(value, list):
        for item in value:
            diagnostics.extend(_robodojo_controller_diagnostics_from_result(item))
    return diagnostics


def _summarize_robodojo_controller_feedback(results: Any) -> JsonDict:
    diagnostics = _robodojo_controller_diagnostics_from_result(results)
    attempts = sum(int(item.get("ik_attempt_count") or 0) for item in diagnostics)
    successes = sum(int(item.get("ik_success_count") or 0) for item in diagnostics)
    failed_ik = []
    position_errors: list[float] = []
    orientation_errors: list[float] = []
    joint_errors: list[float] = []
    leakage_detected = False
    for diagnostic_index, item in enumerate(diagnostics):
        leakage_detected = (
            leakage_detected
            or bool(item.get("task_signal_read"))
            or bool(item.get("reward_or_verifier_read"))
        )
        for result in item.get("ik_results") or []:
            if str(result.get("status")) != "Success":
                failed_ik.append(
                    {
                        "diagnostic_index": diagnostic_index,
                        "arm_name": result.get("arm_name"),
                        "status": result.get("status"),
                        "target_pose": result.get("target_pose"),
                        "error": result.get("error"),
                    }
                )
        for tracking in (item.get("tracking") or {}).values():
            if not isinstance(tracking, dict):
                continue
            for key, sink in (
                ("ee_position_error", position_errors),
                ("ee_orientation_error_radians", orientation_errors),
                ("arm_joint_l2_error", joint_errors),
            ):
                value = tracking.get(key)
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    sink.append(float(value))
    return {
        "contract": "controller-feedback-summary-v1",
        "diagnostic_count": len(diagnostics),
        "ik_attempt_count": attempts,
        "ik_success_count": successes,
        "all_ik_succeeded": attempts > 0 and successes == attempts,
        "failed_ik": failed_ik,
        "max_ee_position_error": max(position_errors) if position_errors else None,
        "max_ee_orientation_error_radians": max(orientation_errors)
        if orientation_errors
        else None,
        "max_arm_joint_l2_error": max(joint_errors) if joint_errors else None,
        "task_or_verifier_signal_read": leakage_detected,
        "task_specific_threshold_applied": False,
    }


def _robodojo_execution_result_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return True
    if "ok" in value:
        return bool(value.get("ok"))
    if "accepted" in value:
        return bool(value.get("accepted"))
    return True


def _vector_delta(
    before: list[float] | None, after: list[float] | None
) -> list[float] | None:
    if before is None or after is None or len(before) < 3 or len(after) < 3:
        return None
    return [round(float(after[index]) - float(before[index]), 6) for index in range(3)]


def _component_delta(
    before: list[float] | None, after: list[float] | None
) -> list[float] | None:
    if before is None or after is None or len(before) != len(after):
        return None
    return [
        round(float(right) - float(left), 6)
        for left, right in zip(before, after, strict=True)
    ]


def _quaternion_angular_distance_degrees(
    before: list[float] | None,
    after: list[float] | None,
) -> float | None:
    if before is None or after is None or len(before) != 4 or len(after) != 4:
        return None
    before_norm = math.sqrt(sum(float(value) ** 2 for value in before))
    after_norm = math.sqrt(sum(float(value) ** 2 for value in after))
    if before_norm <= 1e-12 or after_norm <= 1e-12:
        return None
    dot = abs(
        sum(
            float(left) * float(right)
            for left, right in zip(before, after, strict=True)
        )
        / (before_norm * after_norm)
    )
    return round(math.degrees(2.0 * math.acos(max(-1.0, min(1.0, dot)))), 6)


def _vector_norm(delta: list[float] | None) -> float:
    if not delta:
        return 0.0
    return round(math.sqrt(sum(float(value) * float(value) for value in delta)), 6)


def load_robodojo_task_config(repo_path: str | Path, task_name: str) -> JsonDict:
    return _load_yaml(Path(repo_path) / f"task/RoboDojo/config/{task_name}.yml")


def load_robodojo_env_config(repo_path: str | Path, env_cfg: str) -> JsonDict:
    return _load_yaml(Path(repo_path) / f"env_cfg/{env_cfg}.yml")


def extract_robodojo_entities(task_config: JsonDict) -> list[JsonDict]:
    entities: list[JsonDict] = []
    for section in OBJECT_SECTIONS:
        for group_index, group in enumerate(task_config.get(section) or []):
            if not isinstance(group, dict):
                continue
            common = dict(group.get("common") or {})
            select = dict(group.get("select_mode") or {})
            labels = _labels_from_select(select)
            categories = _categories_from_group(group)
            if not labels:
                labels = [
                    f"{section.lower()}_{group_index}_{idx}"
                    for idx in range(max(1, len(categories)))
                ]
            for label_index, label in enumerate(labels):
                category = (
                    categories[min(label_index, len(categories) - 1)]
                    if categories
                    else {}
                )
                entities.append(
                    {
                        "label": str(label),
                        "section": section,
                        "group_index": group_index,
                        "label_index": label_index,
                        "category_name": category.get("name"),
                        "category": category,
                        "placement": {
                            "xlim": common.get("xlim"),
                            "ylim": common.get("ylim"),
                            "zlim": common.get("zlim"),
                            "relative_plane": common.get("relative_plane"),
                            "qpos": common.get("qpos"),
                            "rotate_rand": common.get("rotate_rand"),
                            "rotate_deg": common.get("rotate_deg"),
                            "margin": common.get("margin"),
                        },
                        "select_mode": select,
                    }
                )
    return entities


def summarize_robodojo_scene(
    task_config: JsonDict, entities: list[JsonDict]
) -> JsonDict:
    section_counts = {
        section: len(task_config.get(section) or [])
        for section in OBJECT_SECTIONS + ("Clutter", "ProhibitedArea")
    }
    return {
        "section_counts": section_counts,
        "entity_count": len(entities),
        "entities": entities,
        "has_clutter": bool(task_config.get("Clutter")),
        "prohibited_area_count": len(task_config.get("ProhibitedArea") or []),
    }


def summarize_robodojo_env_cfg(env_cfg: JsonDict) -> JsonDict:
    return {
        "config_name": env_cfg.get("config_name"),
        "config_refs": dict(env_cfg.get("config") or {}),
        "observation": dict(env_cfg.get("observation") or {}),
        "annotator": dict(env_cfg.get("annotator") or {}),
        "camera_names": [key for key in env_cfg if str(key).startswith("cam_")],
    }


def _target_for_action_key(
    key: str, *, joint_targets: Any | None, gripper_targets: Any | None
) -> Any:
    is_gripper = "_ee_" in key or key.startswith("ee_") or "gripper" in key
    source = gripper_targets if is_gripper else joint_targets
    if isinstance(source, dict):
        side = (
            "left"
            if key.startswith("left_")
            else "right"
            if key.startswith("right_")
            else None
        )
        aliases = [key]
        if is_gripper:
            aliases.extend(
                filter(
                    None,
                    [
                        f"{side}_gripper" if side else None,
                        f"{side}_ee" if side else None,
                        "gripper",
                        "ee",
                    ],
                )
            )
        else:
            aliases.extend(
                filter(
                    None,
                    [
                        f"{side}_arm" if side else None,
                        f"{side}_joint" if side else None,
                        "arm",
                        "joint",
                    ],
                )
            )
        for alias in aliases:
            if alias in source:
                return source[alias]
        return None
    return source


def _coerce_float_vector(
    value: Any | None, expected_dim: int, *, fill: float = 0.0
) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    flat: list[float] = []

    def append(item: Any) -> None:
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, dict):
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                append(child)
            return
        if item is None:
            return
        try:
            flat.append(float(item))
        except Exception:
            return

    append(value)
    if expected_dim <= 0:
        return flat
    if len(flat) >= expected_dim:
        return flat[:expected_dim]
    return flat + [fill] * (expected_dim - len(flat))


def _primitive_output(value: Any) -> Any:
    if isinstance(value, PrimitiveResult):
        return value.output
    return value


def _labels_from_select(select: JsonDict) -> list[str]:
    labels = select.get("label")
    if isinstance(labels, list):
        return [str(label) for label in labels]
    if isinstance(labels, str):
        return [labels]
    prefix = select.get("label_prefix")
    if isinstance(prefix, list):
        count = _select_count(select)
        return [f"{prefix[min(i, len(prefix) - 1)]}_{i}" for i in range(count)]
    if isinstance(prefix, str):
        return [f"{prefix}_{i}" for i in range(_select_count(select))]
    return []


def _select_count(select: JsonDict) -> int:
    nums = select.get("nums") or select.get("select_instance_nums") or 1
    if isinstance(nums, list):
        numeric = [int(value) for value in nums if isinstance(value, int | float)]
        return max(numeric) if numeric else 1
    try:
        return int(nums)
    except Exception:
        return 1


def _categories_from_group(group: JsonDict) -> list[JsonDict]:
    categories = group.get("category") or []
    if isinstance(categories, dict):
        categories = [categories]
    clean: list[JsonDict] = []
    for category in categories:
        if isinstance(category, dict):
            clean.append(dict(category))
    return clean


def _load_yaml(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - yaml is available in harness envs
        raise RuntimeError("PyYAML is required for RoboDojo config parsing") from exc
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        return repr(value)


def _primitive_result_event_payload(result: PrimitiveResult) -> JsonDict:
    output: JsonDict = {}
    for key, value in result.output.items():
        if key == "observation":
            output[key] = _observation_event_summary(value)
        elif key in {"raw_rgb_frame", "raw_rgb_frames"}:
            output[key] = _raw_rgb_frame_metadata(value)
        else:
            output[key] = _jsonable(value)
    return {
        "name": result.name,
        "ok": result.ok,
        "output": output,
        "artifacts": list(result.artifacts),
        "error": result.error,
        "metadata": _jsonable(result.metadata),
    }


def _observation_event_summary(value: Any) -> JsonDict:
    if not isinstance(value, dict):
        return {
            "type": type(value).__name__,
            "raw_payload_included": False,
            "visual_observation_contract": _robodojo_visual_observation_contract(None),
        }
    contract = _robodojo_visual_observation_contract(value)
    observation = (
        value.get("observation")
        if isinstance(value.get("observation"), dict)
        else value
    )
    vision = (
        observation.get("vision") if isinstance(observation.get("vision"), dict) else {}
    )
    return {
        "keys": sorted(str(key) for key in value.keys()),
        "observation_keys": sorted(str(key) for key in observation.keys())
        if isinstance(observation, dict)
        else [],
        "vision_camera_names": sorted(str(key) for key in vision.keys()),
        "visual_observation_contract": contract,
        "raw_payload_included": False,
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="RoboDojo agent-runtime source/asset preflight."
    )
    parser.add_argument("--repo-path", default=str(DEFAULT_ROBODOJO_REPO))
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--env-cfg", default="arx_x5")
    parser.add_argument("--sim-env", default="RoboDojo")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    config = RoboDojoRuntimeConfig(
        repo_path=args.repo_path,
        task_name=args.task_name,
        env_cfg=args.env_cfg,
        sim_env=args.sim_env,
    )
    if args.smoke:
        backend = RoboDojoAgentRuntimeBackend(config)
        backend.reset(f"robodojo_{args.task_name}_source_boundary", seed=0)
        schema = backend.call_primitive("get_robodojo_robot_action_schema")
        observation = backend.observe()
        payload = {
            "ok": schema.ok,
            "task_name": args.task_name,
            "runtime": backend.runtime_available(),
            "public_tools": [card.name for card in backend.list_primitives()],
            "observation": observation.to_dict(),
            "trace": backend.get_trace().to_dict(),
        }
    else:
        payload = {
            "repo": inspect_robodojo_repo(args.repo_path),
            "source_preflight": robodojo_source_preflight(
                args.repo_path, args.task_name, args.env_cfg
            ),
            "asset_gate": robodojo_asset_gate(args.repo_path),
            "python_env_gate": robodojo_python_env_gate(args.sim_env),
        }
        payload["ready"] = bool(
            payload["source_preflight"]["ready"]
            and payload["asset_gate"]["ready"]
            and payload["python_env_gate"]["ready"]
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if payload.get("ok") or args.preflight else 0


if __name__ == "__main__":
    raise SystemExit(main())
