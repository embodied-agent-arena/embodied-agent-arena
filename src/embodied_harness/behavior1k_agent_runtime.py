from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import ast
import csv
import hashlib
import importlib.util
import inspect
import json
import math
import os
import platform
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import types
from typing import Any, Callable

import numpy as np

from .backend import EmbodiedBackend
from .paths import get_project_paths, resolve_project_root
from .schemas import (
    EpisodeTrace,
    Observation,
    PrimitiveCard,
    PrimitiveResult,
    TaskSpec,
    VerificationResult,
)


JsonDict = dict[str, Any]
EnvFactory = Callable[[dict[str, Any]], Any]
ENV_PROBE_MARKER = "BEHAVIOR1K_ENV_PROBE_JSON="
BEHAVIOR1K_BOOTSTRAP_VISUAL_MODE_ENV = "BEHAVIOR1K_BOOTSTRAP_VISUAL_MODE"
DEFERRED_VISION_MODALITIES = frozenset(
    {
        "rgb",
        "depth",
        "depth_linear",
        "normal",
        "seg_semantic",
        "seg_instance",
        "seg_instance_id",
        "flow",
        "bbox_2d_tight",
        "bbox_2d_loose",
        "bbox_3d",
        "camera_params",
        "pointcloud",
    }
)
DEFAULT_BEHAVIOR1K_ACTIVITY = "turning_on_radio"
DEFAULT_BEHAVIOR1K_SCENE_MODEL = "house_double_floor_lower"
DEFAULT_BEHAVIOR1K_SCENE_INSTANCE = (
    "house_double_floor_lower_task_turning_on_radio_0_0_template"
)
DEFAULT_BEHAVIOR1K_ASSET_QUERY = "radio receiver"


@contextmanager
def _behavior1k_bootstrap_timeout(seconds: int | float | None) -> Any:
    """Bound a headless OmniGibson bootstrap that can otherwise wait forever."""

    try:
        timeout_seconds = float(seconds or 0)
    except (TypeError, ValueError):
        timeout_seconds = 0.0
    if timeout_seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0.0)

    def _raise_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(
            f"omnigibson_env_bootstrap_timeout_after_{timeout_seconds:g}s"
        )

    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])


@dataclass(slots=True)
class Behavior1KRuntimeConfig:
    task_name: str = "turning_on_radio"
    env_config: JsonDict = field(default_factory=lambda: {"scene": {"type": "Scene"}})
    live: bool = True
    headless: bool = True
    dataset_path: str | None = None
    omnigibson_root: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


def build_behavior1k_full_bddl_env_config(
    *,
    activity_name: str = DEFAULT_BEHAVIOR1K_ACTIVITY,
    scene_model: str = DEFAULT_BEHAVIOR1K_SCENE_MODEL,
    scene_instance: str | None = None,
    image_size: int = 64,
    max_steps: int = 300,
    online_object_sampling: bool | None = None,
) -> JsonDict:
    """Build the validated full-scene BEHAVIOR task config used by live agent cases."""

    online_sampling_env = os.getenv("BEHAVIOR1K_ONLINE_OBJECT_SAMPLING")
    use_online_sampling = (
        True
        if online_object_sampling is None and online_sampling_env is None
        else _env_truthy("BEHAVIOR1K_ONLINE_OBJECT_SAMPLING")
        if online_object_sampling is None
        else bool(online_object_sampling)
    )
    # OmniGibson resolves an explicit ``scene_instance`` from the official
    # challenge-task-instance tree, while an omitted instance resolves the
    # base ``<scene_model>_best.json`` from behavior-1k-assets.  Online object
    # sampling needs that base scene, so do not spell the implicit best name
    # as an explicit task instance.
    resolved_scene_instance = (
        scene_instance
        if scene_instance is not None
        else None
        if use_online_sampling
        else f"{scene_model}_task_{activity_name}_0_0_template"
    )
    sensor_size = max(16, int(image_size))
    return {
        "env": {
            "action_frequency": 30,
            "rendering_frequency": 30,
            "physics_frequency": 120,
        },
        "scene": {
            "type": "InteractiveTraversableScene",
            "scene_model": scene_model,
            **(
                {"scene_instance": resolved_scene_instance}
                if resolved_scene_instance is not None
                else {}
            ),
            "load_task_relevant_only": True,
            "not_load_object_categories": ["ceilings"],
        },
        "robots": [
            {
                "type": "Fetch",
                # Match OmniGibson's official learning wrapper ordering. In
                # particular, semantic labels must exist before instance-ID
                # labels are initialized by Replicator.
                "obs_modalities": [
                    "rgb",
                    "depth_linear",
                    "seg_semantic",
                    "seg_instance_id",
                    "proprio",
                ],
                "sensor_config": {
                    "VisionSensor": {
                        "sensor_kwargs": {
                            "image_width": sensor_size,
                            "image_height": sensor_size,
                        }
                    }
                },
            }
        ],
        "task": {
            "type": "BehaviorTask",
            "activity_name": activity_name,
            "activity_definition_id": 0,
            "activity_instance_id": 0,
            "predefined_problem": None,
            "online_object_sampling": use_online_sampling,
            "use_presampled_robot_pose": not use_online_sampling,
            "debug_object_sampling": False,
            "highlight_task_relevant_objects": False,
            "termination_config": {"max_steps": max(1, int(max_steps))},
            "reward_config": {"r_potential": 1.0},
            "include_obs": False,
        },
    }


def _behavior1k_native_function(path: Path, name: str, namespace: dict[str, Any], *, owner: str | None = None) -> Any:
    """Load an unchanged native helper without starting its CLI or policy imports."""
    nodes = ast.parse(path.read_text(encoding="utf-8")).body
    if owner is not None:
        nodes = next(node.body for node in nodes if isinstance(node, ast.ClassDef) and node.name == owner)
    function = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def _behavior1k_public_instance_env_config(selected: JsonDict) -> JsonDict:
    import runpy

    upstream = get_project_paths().external_upstream("behavior1k")
    utils_root = upstream / "joylo/gello/robots/sim_robot"
    namespace = runpy.run_path(str(utils_root / "og_teleop_cfg.py"))
    for rule in namespace["DISABLED_TRANSITION_RULES"]:
        rule.ENABLED = False
    _behavior1k_native_function(utils_root / "og_teleop_utils.py", "infer_torso_qpos_from_trunk_translate", namespace)
    make_robot = _behavior1k_native_function(utils_root / "og_teleop_utils.py", "generate_robot_config", namespace)
    make_env = _behavior1k_native_function(upstream / "OmniGibson/omnigibson/learning/utils/eval_utils.py", "generate_basic_environment_config", {})
    cfg = make_env(selected["task"], selected["task_cfg"])
    robot = make_robot(task_name=selected["task"], task_cfg=selected["task_cfg"])
    robot["obs_modalities"] = ["proprio", "rgb", "depth_linear", "seg_semantic", "seg_instance_id"]
    robot["sensor_config"]["VisionSensor"]["sensor_kwargs"].update(image_width=64, image_height=64)
    cfg["robots"] = [_to_builtin(robot)]
    cfg["scene"]["scene_instance"] = selected["template"]
    data = get_project_paths().external_assets("behavior1k") / "datasets/2025-challenge-task-instances/metadata/episodes.jsonl"
    lengths = [row["length"] for line in data.read_text().splitlines()
               if (row := json.loads(line))["episode_index"] // 10000 == selected["task_index"]]
    if not lengths:
        raise ValueError("Selected BEHAVIOR task lacks native human episode lengths")
    cfg["task"]["termination_config"]["max_steps"] = int(2 * sum(lengths) / len(lengths))
    return cfg


def _restore_behavior1k_public_instance(env: Any, instance_id: int) -> None:
    import omnigibson as og
    from omnigibson.macros import gm
    from omnigibson.utils.asset_utils import get_task_instance_path
    from omnigibson.utils.python_utils import recursively_convert_to_torch

    root = get_project_paths().external_upstream("behavior1k")
    namespace = dict(os=os, json=json, og=og, gm=gm, get_task_instance_path=get_task_instance_path,
                     recursively_convert_to_torch=recursively_convert_to_torch)
    restore = _behavior1k_native_function(root / "OmniGibson/omnigibson/learning/eval.py", "load_task_instance", namespace, owner="Evaluator")
    native_env = _unwrap_env(env)
    robot = native_env.scene.object_registry("name", "robot_r1")
    restore(types.SimpleNamespace(env=native_env, robot=robot), instance_id, test_hidden=False)


class Behavior1KAgentRuntimeBackend(EmbodiedBackend):
    """BEHAVIOR-1K / OmniGibson adapter with agent-safe primitives.

    The live path creates the lightest configured OmniGibson environment. The
    agent-facing surface is observation/evidence only; official task checking,
    evaluator websocket plumbing, demos, and replay hooks remain harness-side.
    """

    def __init__(
        self,
        config: Behavior1KRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
    ) -> None:
        self.config = config or Behavior1KRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._objects: dict[str, JsonDict] = {}
        self._probe_progress: Callable[..., None] | None = None
        self._deferred_visual_modalities: list[JsonDict] = []
        self._deferred_visual_attached = False
        self._pool_instance: dict[str, Any] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        import yaml

        task = str(coordinate.get("task_id") or "")
        variation = str(coordinate.get("variation") or "")
        prefix = task + "::instance_"
        if not variation.startswith(prefix) or not variation[len(prefix):].isdigit():
            raise ValueError("Invalid BEHAVIOR public task instance coordinate")
        instance = int(variation[len(prefix):])
        seed = coordinate.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("BEHAVIOR seed must be an integer in [0, 2**32)")
        paths = get_project_paths()
        upstream = paths.external_upstream("behavior1k")
        dataset = paths.external_assets("behavior1k") / "datasets"
        with (dataset / "2025-challenge-task-instances/metadata/test_instances.csv").open() as f:
            rows = list(csv.reader(f))[1:]
        eligible = next((row for row in rows if row[1] == task), None)
        if eligible is None or instance not in [int(x) for x in eligible[2].strip().split(",")[:10]]:
            raise ValueError("BEHAVIOR coordinate is absent from the original public evaluation split")
        configs = yaml.safe_load((upstream / "joylo/sampled_task/available_tasks.yaml").read_text())
        task_cfg = configs[task][0]
        scene = task_cfg["scene_model"]
        template = f"{scene}_task_{task}_0_0_template"
        state = dataset / "2025-challenge-task-instances/scenes" / scene / "json" / f"{scene}_task_{task}_instances" / f"{scene}_task_{task}_0_{instance}_template-tro_state.json"
        if not state.is_file():
            raise FileNotFoundError(f"Selected native BEHAVIOR state is missing: {state}")
        self._pool_instance = dict(task=task, instance=instance, seed=seed, scene=scene,
                                   template=template, task_cfg=task_cfg, task_index=int(eligible[0]))
        return {"bound": True, "mode": "native_public_tro_instance", "task_name": task,
                "instance_id": instance, "scene_model": scene, "robot_type": "R1Pro",
                "state_file": str(state), "state_sha256": hashlib.sha256(state.read_bytes()).hexdigest(),
                "actual_task_id": f"behavior1k:{task}:instance_{instance}:seed_{seed}"}

    def reset(
        self, task_id: str, seed: int | None = None, config: JsonDict | None = None
    ) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_instance is not None:
            selected = self._pool_instance
            if seed is not None and seed != selected["seed"]:
                raise ValueError("BEHAVIOR reset seed differs from the bound native coordinate")
            seed = selected["seed"]
            task_id = f"behavior1k:{selected['task']}:instance_{selected['instance']}:seed_{seed}"
            overrides.update(task_name=selected["task"], env_config=build_behavior1k_full_bddl_env_config(
                activity_name=selected["task"], scene_model=selected["scene"],
                scene_instance=selected["template"], online_object_sampling=False))
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w7:behavior1k:live_runtime_preflight",
            instruction=(
                "Ground a BEHAVIOR-1K / OmniGibson household task from allowed "
                "agent observations and record auditable observation/action/asset evidence."
            ),
            goal={
                "task_name": runtime_config.task_name,
                "success_source": "harness_only_official_evaluator",
            },
            budgets={"primitive_calls": 32, "verifier_calls": 6},
            tags=[
                "w7",
                "behavior1k",
                "omnigibson",
                "bddl",
                "live" if runtime_config.live else "dry",
            ],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "behavior1k",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_prompt_query_agent_context": True,
                    "observation_action_asset_evidence_returned": True,
                    "action_primitive_steps_live_env": True,
                    "oracle_checker_success_demo_replay_primitives_exposed": False,
                    "official_evaluator_is_harness_only": True,
                },
            },
        )
        self._last_obs = {}
        self._last_info = {}
        self._objects = {}
        if runtime_config.live:
            self._emit_probe_progress("reset_make_env_start")
            self._env = self._make_env(runtime_config, seed=seed)
            self._emit_probe_progress("reset_make_env_done")
            self._emit_probe_progress("env_reset_start")
            reset_result = self._env.reset() if hasattr(self._env, "reset") else None
            self._emit_probe_progress("env_reset_done")
            self._last_obs, self._last_info = _split_reset_result(reset_result)
            if self._pool_instance is not None:
                _restore_behavior1k_public_instance(self._env, self._pool_instance["instance"])
                self._last_obs, self._last_info = _split_reset_result(self._env.reset())
                self._task_spec.instruction = "Complete the native household task: " + runtime_config.task_name.replace("_", " ") + "."
                self._task_spec.metadata["runtime_config"] = _to_builtin(runtime_config.to_dict())
            self._emit_probe_progress("asset_registry_start")
            self._refresh_asset_registry()
            self._emit_probe_progress(
                "asset_registry_done", asset_count=len(self._objects)
            )
        else:
            self._env = None
        self.record_event(
            "reset",
            {
                "task": self._task_spec.to_dict(),
                "seed": seed,
                "live_env": self._env is not None,
                "runtime": self.runtime_available(),
            },
        )
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        data = {
            "language_task": self._language_task(),
            "runtime": self.runtime_available(),
            "observation_summary": summarize_data(self._last_obs),
            "visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                self._last_obs, self._objects
            ),
            "asset_evidence": deepcopy(self._objects),
            "action_evidence": {
                "last_info_summary": summarize_data(
                    _public_behavior1k_info(self._last_info)
                )
            },
        }
        obs = Observation(
            step=len(self.get_trace().events),
            data=data,
            metadata={"benchmark_id": "behavior1k"},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            _card(
                "observe_behavior1k_state",
                "L1",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "include_raw": "bool",
                },
                {
                    "observation_evidence": "dict",
                    "action_evidence": "dict",
                    "asset_evidence": "dict",
                },
                "Summarize current BEHAVIOR-1K / OmniGibson observation and asset registry evidence.",
            ),
            _card(
                "get_behavior1k_task_context",
                "L1",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {"language_task": "dict", "bddl_task": "dict", "runtime": "dict"},
                "Return prompt/query/task grounding context without exposing task success predicates.",
            ),
            _card(
                "inspect_behavior1k_asset",
                "L2",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "asset_name": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "selected": "dict",
                    "candidates": "list[dict]",
                    "asset_evidence": "dict",
                },
                "Enumerate OmniGibson scene objects and inspect only an exact caller-selected asset name.",
            ),
            _card(
                "inspect_behavior1k_visual_evidence",
                "L2",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "camera_name": "str|None",
                    "entity_name": "str|None",
                    "instance_id": "int|None",
                    "label_kind": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "visual_runtime": "dict",
                    "sensor_summaries": "list[dict]",
                    "calibration": "list[dict]",
                    "segmentation_instances": "list[dict]",
                    "groundings": "list[dict]",
                    "evidence_handles": "list[str]",
                    "asset_candidates": "list[dict]",
                },
                "Expose promptable native RGB/depth/instance/semantic regions, with RGBD+asset-pose fallback groundings when native masks are absent.",
            ),
            _card(
                "convert_behavior1k_grounding_to_controller_input",
                "L2",
                {
                    "evidence_handle": "str",
                    "robot_name": "str",
                    "controller_name": "str",
                    "source_field": "str",
                    "component_indices": "list[int]|None",
                    "scale": "list[float]|float|None",
                    "offset": "list[float]|float|None",
                    "agent_context": "dict|None",
                },
                {
                    "controller_input": "dict",
                    "action": "dict|list",
                    "action_schema": "dict",
                    "evidence_handle": "str",
                },
                "Convert a caller-selected visual evidence field into native controller indices without stepping or choosing a policy.",
            ),
            _card(
                "inspect_behavior1k_object_state",
                "L2",
                {
                    "asset_name": "str",
                    "state_name": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "selected_asset": "dict",
                    "object_states": "dict",
                    "read_only": "bool",
                },
                "Read caller-selected OmniGibson object states without mutating simulator state.",
            ),
            _card(
                "inspect_behavior1k_contacts",
                "L2",
                {"asset_name": "str", "agent_context": "dict|None"},
                {
                    "selected_asset": "dict",
                    "contacts": "list[dict]",
                    "read_only": "bool",
                },
                "Read native contact_list evidence for a caller-selected scene object.",
            ),
            _card(
                "inspect_behavior1k_control",
                "L2",
                {"robot_name": "str|None", "agent_context": "dict|None"},
                {
                    "selected_robot": "dict",
                    "robots": "dict",
                    "action_schema": "dict",
                    "read_only": "bool",
                },
                "Inspect live robot poses, end-effector poses, and native controller action indices without stepping.",
            ),
            _card(
                "settle_behavior1k",
                "L3",
                {"steps": "int", "agent_context": "dict|None"},
                {
                    "settled": "bool",
                    "step_count": "int",
                    "termination": "dict",
                    "observation_evidence": "dict",
                },
                "Advance the live OmniGibson environment for an agent-specified number of neutral settle steps.",
            ),
            _card(
                "record_behavior1k_evidence",
                "L1",
                {"key": "str", "value": "any", "agent_context": "dict|None"},
                {"artifact_id": "str", "agent_context": "dict"},
                "Record agent-selected observation/action/asset evidence in the episode trace.",
            ),
            _card(
                "submit_behavior1k_action",
                "L3",
                {
                    "action": "dict|list",
                    "prompt": "str|None",
                    "query": "str|None",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                },
                {
                    "stepped": "bool",
                    "terminated": "bool|None",
                    "truncated": "bool|None",
                    "action_schema": "dict",
                    "observation_evidence": "dict",
                    "visual_evidence": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Submit an agent-chosen action to the live OmniGibson env.step boundary and return post-step evidence.",
            ),
            _card(
                "step_behavior1k_action",
                "L3",
                {
                    "action": "dict|list",
                    "prompt": "str|None",
                    "query": "str|None",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                },
                {
                    "stepped": "bool",
                    "terminated": "bool|None",
                    "truncated": "bool|None",
                    "action_schema": "dict",
                    "observation_evidence": "dict",
                    "visual_evidence": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Agent-native alias for one real live OmniGibson env.step(action); dry/no-live returns an explicit blocker.",
            ),
            _card(
                "execute_behavior1k_action_sequence",
                "L3",
                {
                    "actions": "list[dict|list]",
                    "prompt": "str|None",
                    "query": "str|None",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                    "stop_on_termination": "bool",
                },
                {
                    "stepped": "bool",
                    "step_count": "int",
                    "steps": "list[dict]",
                    "final_observation_evidence": "dict",
                    "action_schema": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Execute a bounded agent-provided action sequence through repeated live OmniGibson env.step(action) calls.",
            ),
            _card(
                "execute_behavior1k_controller_command",
                "L3",
                {
                    "robot_name": "str",
                    "commands": "dict[str,list|float]",
                    "repeat": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                },
                {
                    "stepped": "bool",
                    "step_count": "int",
                    "controller_command": "dict",
                    "control_state": "dict",
                    "control_response": "dict",
                    "observation_evidence": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Map caller-supplied commands through native controller_action_idx and step the live environment.",
            ),
            _card(
                "navigate_behavior1k_base_to_pose",
                "L3",
                {
                    "robot_name": "str",
                    "base_controller_name": "str",
                    "target_pose_xyyaw": "list[float]",
                    "max_steps": "int",
                    "distance_tolerance": "float",
                    "yaw_tolerance": "float",
                    "linear_gain": "float",
                    "angular_gain": "float",
                    "linear_limit": "float",
                    "angular_limit": "float",
                    "turn_in_place_yaw_threshold": "float|None",
                    "command_mode": "str",
                    "repeat_per_command": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                },
                {
                    "stepped": "bool",
                    "reached": "bool",
                    "step_count": "int",
                    "trace": "list[dict]",
                    "final_progress": "dict",
                    "control_state": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Drive a caller-selected base controller toward an agent-provided world pose using native controller feedback.",
            ),
            _card(
                "run_behavior1k_semantic_action",
                "L3",
                {
                    "semantic_action": "str",
                    "asset_name": "str",
                    "secondary_asset_name": "str|None",
                    "robot_name": "str|None",
                    "attempts": "int",
                    "max_steps": "int",
                    "evidence_handle": "str|None",
                    "evidence_handles": "list[str]|None",
                    "suspend_deferred_visual_modalities": "bool",
                    "include_post_step_visual_evidence": "bool",
                    "agent_context": "dict|None",
                },
                {
                    "stepped": "bool",
                    "step_count": "int",
                    "semantic_action": "str",
                    "selected_assets": "dict",
                    "object_states_before": "dict",
                    "object_states_after": "dict",
                    "trace": "list[dict]",
                    "observation_evidence": "dict",
                    "visual_evidence": "dict",
                    "evidence_handles": "list[str]",
                    "evidence_provenance": "list[dict]",
                },
                "Run a caller-selected official OmniGibson semantic action primitive over observed scene assets and real env.step actions.",
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
                error=f"Primitive {name!r} is not exposed by Behavior1KAgentRuntimeBackend",
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
        if scope == "preflight":
            runtime = self.runtime_available()
            runtime["bddl_importable"] = _module_imports("bddl")[0]
            runtime["omnigibson_importable"] = _module_imports("omnigibson")[0]
            result = VerificationResult(
                ok=bool(
                    runtime["omnigibson_importable"] and runtime["bddl_importable"]
                ),
                scope=scope,
                message="BEHAVIOR-1K Python imports are available"
                if runtime["omnigibson_importable"] and runtime["bddl_importable"]
                else "BEHAVIOR-1K Python imports are incomplete",
                metadata=runtime,
            )
        elif scope == "object_scene_boundary":
            readiness = behavior1k_task_readiness(
                config=self.config,
                env=self._env,
                obs=self._last_obs,
                objects=self._objects,
                info=self._last_info,
            )
            ok = bool(readiness["object_scene_boundary"]["ready"])
            result = VerificationResult(
                ok=ok,
                scope=scope,
                message=(
                    "BEHAVIOR-1K bounded object-scene observation/action boundary is ready"
                    if ok
                    else "BEHAVIOR-1K bounded object-scene boundary is not ready"
                ),
                metrics={
                    "live_env_created": float(readiness["live_env_created"]),
                    "visual_boundary_ready": float(
                        readiness["visual_boundary"]["ready"]
                    ),
                    "object_scene_boundary_ready": float(
                        readiness["object_scene_boundary"]["ready"]
                    ),
                    "full_task_ready": 0.0,
                },
                metadata=readiness,
            )
        else:
            readiness = behavior1k_task_readiness(
                config=self.config,
                env=self._env,
                obs=self._last_obs,
                objects=self._objects,
                info=self._last_info,
            )
            official_success = _behavior1k_harness_task_success(self._env)
            if official_success["available"] and official_success["success"]:
                readiness["full_bddl_task"]["ready"] = True
                readiness["full_bddl_task"]["official_success"] = official_success
                readiness["full_bddl_task"]["blockers"] = []
                result = VerificationResult(
                    ok=True,
                    scope=scope,
                    message="BEHAVIOR-1K full BDDL task success verified by harness-only official task state",
                    metrics={
                        "live_env_created": float(readiness["live_env_created"]),
                        "visual_boundary_ready": float(
                            readiness["visual_boundary"]["ready"]
                        ),
                        "object_scene_boundary_ready": float(
                            readiness["object_scene_boundary"]["ready"]
                        ),
                        "full_task_ready": 1.0,
                        "official_success": 1.0,
                    },
                    metadata=readiness,
                )
                self.record_event("verifier_call", result.to_dict())
                return result
            readiness["full_bddl_task"]["official_success"] = official_success
            result = VerificationResult(
                ok=False,
                scope=scope,
                message=(
                    "BEHAVIOR-1K full BDDL task success is not ready; "
                    "runtime evidence is classified separately from the harness-only official evaluator."
                ),
                metrics={
                    "live_env_created": float(readiness["live_env_created"]),
                    "visual_boundary_ready": float(
                        readiness["visual_boundary"]["ready"]
                    ),
                    "object_scene_boundary_ready": float(
                        readiness["object_scene_boundary"]["ready"]
                    ),
                    "full_task_ready": 0.0,
                },
                metadata=readiness,
            )
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self._trace

    def close(self) -> None:
        environment, self._env = self._env, None
        if environment is None:
            return
        try:
            close = getattr(environment, "close", None)
            if callable(close):
                close()
        finally:
            # OmniGibson Environment.close() is a no-op. Shut down Kit while
            # Python's rendering objects still exist, before interpreter exit.
            if self._env_factory is None:
                _shutdown_loaded_omnigibson()

    def runtime_available(self) -> JsonDict:
        return behavior1k_preflight_summary(live_requested=self.config.live)

    def _merged_config(self, overrides: JsonDict) -> Behavior1KRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return Behavior1KRuntimeConfig(**data)

    def _make_env(self, config: Behavior1KRuntimeConfig, seed: int | None) -> Any:
        if self._env_factory is not None:
            self._deferred_visual_modalities = []
            self._deferred_visual_attached = True
            return self._env_factory({**config.to_dict(), "seed": seed})
        source_paths = _configure_behavior1k_source_paths(config)
        asset_preflight = prepare_behavior1k_runtime_assets(config)
        if not asset_preflight["ok"]:
            raise FileNotFoundError(
                "BEHAVIOR-1K asset preflight failed before OmniGibson import: "
                + json.dumps(asset_preflight, sort_keys=True)
            )
        if _find_module_spec("omnigibson", bootstrap_behavior1k=False) is None:
            raise RuntimeError(
                "BEHAVIOR-1K live runtime requires `omnigibson`; run the official BEHAVIOR-1K setup first "
                f"or set BEHAVIOR1K_OMNIGIBSON_ROOT. source_path_bootstrap={source_paths}"
            )
        if config.headless:
            # OmniGibson freezes gm.HEADLESS while importing its macros. Set
            # the process contract before the first import so Isaac Sim never
            # starts the GUI extension path on batch nodes.
            os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
        _configure_isaacsim_noninteractive_eula(config)
        omnigibson_ok, _omnigibson_module, omnigibson_error = _module_imports(
            "omnigibson", bootstrap_behavior1k=False
        )
        if not omnigibson_ok:
            raise RuntimeError(
                f"BEHAVIOR-1K live runtime import failed for omnigibson: {omnigibson_error}"
            )
        import omnigibson as og

        if self._pool_instance is not None:
            config.env_config = _behavior1k_public_instance_env_config(self._pool_instance)
            prepare_behavior1k_runtime_assets(config)
        _configure_omnigibson_headless_viewer(config, og)
        viewport_patch = _maybe_patch_isaacsim_headless_viewport_wait(config)
        self._emit_probe_progress(
            "isaacsim_headless_viewport_wait_patch", patch=viewport_patch
        )
        fast_mesh_patch = _maybe_patch_omnigibson_fast_mesh_triangulation()
        self._emit_probe_progress(
            "omnigibson_fast_mesh_triangulation_patch", patch=fast_mesh_patch
        )
        _maybe_patch_omnigibson_meta_root_links()
        _maybe_patch_fetch_eef_link_fallback()
        _maybe_patch_empty_finger_property_inference()
        _maybe_patch_behavior_task_reset_none_object_scope()
        _maybe_patch_behavior1k_empty_semantic_remapper()
        _maybe_apply_minimal_kit_no_flowusd_override()
        _maybe_install_offline_omni_particle_stubs()
        bootstrap_visual_mode = _behavior1k_bootstrap_visual_mode()
        bootstrap_config, deferred_modalities = _defer_behavior1k_visual_modalities(
            config.env_config,
            bootstrap_visual_mode=bootstrap_visual_mode,
        )
        self._deferred_visual_modalities = deferred_modalities
        self._deferred_visual_attached = not bool(deferred_modalities)
        bootstrap_timeout_seconds = int(
            os.environ.get("BEHAVIOR1K_OMNIGIBSON_BOOTSTRAP_TIMEOUT_SECONDS", "0") or 0
        )
        bootstrap_robot_modalities = []
        robots = (
            bootstrap_config.get("robots")
            if isinstance(bootstrap_config, dict)
            else None
        )
        if isinstance(robots, list):
            for robot_config in robots:
                if isinstance(robot_config, dict):
                    modalities = robot_config.get("obs_modalities")
                    if isinstance(modalities, (list, tuple, set)):
                        bootstrap_robot_modalities.append(
                            [str(modality) for modality in modalities]
                        )
        self._emit_probe_progress(
            "omnigibson_env_bootstrap_config_prepared",
            bootstrap_visual_mode=bootstrap_visual_mode,
            diagnostic_no_visual_bootstrap=bootstrap_visual_mode == "none",
            bootstrap_robot_modalities=bootstrap_robot_modalities,
            deferred_visual_modalities=deferred_modalities,
        )
        self._emit_probe_progress(
            "omnigibson_env_bootstrap_start",
            bootstrap_visual_mode=bootstrap_visual_mode,
            diagnostic_no_visual_bootstrap=bootstrap_visual_mode == "none",
            deferred_visual_modalities=deferred_modalities,
            timeout_seconds=bootstrap_timeout_seconds,
        )
        with _behavior1k_bootstrap_timeout(bootstrap_timeout_seconds):
            env = og.Environment(configs=bootstrap_config)
        self._emit_probe_progress("omnigibson_env_bootstrap_done")
        viewport_patch_after_bootstrap = _maybe_patch_isaacsim_headless_viewport_wait(
            config
        )
        self._emit_probe_progress(
            "isaacsim_headless_viewport_wait_patch_after_bootstrap",
            patch=viewport_patch_after_bootstrap,
        )
        self._emit_probe_progress(
            "omnigibson_deferred_visual_modalities_pending",
            deferred_visual_modalities=deferred_modalities,
        )
        return env

    def _ensure_behavior1k_deferred_visual_modalities_attached(self) -> None:
        if self._deferred_visual_attached:
            return
        self._emit_probe_progress(
            "omnigibson_deferred_visual_modalities_start",
            deferred_visual_modalities=self._deferred_visual_modalities,
        )
        _attach_behavior1k_deferred_visual_modalities(
            self._env, self._deferred_visual_modalities
        )
        self._deferred_visual_attached = True
        self._emit_probe_progress(
            "omnigibson_deferred_visual_modalities_done",
            deferred_visual_modalities=self._deferred_visual_modalities,
        )
        refreshed = _behavior1k_read_observation(self._env)
        if refreshed is not None:
            self._last_obs, self._last_info = _split_reset_result(refreshed)
            return
        settle_records = self._settle_behavior1k_env(1)
        if not settle_records or not settle_records[-1].get("ok"):
            self._emit_probe_progress(
                "omnigibson_deferred_visual_observation_refresh_unavailable",
                settle_records=settle_records,
            )

    def _suspend_behavior1k_deferred_visual_modalities(self) -> None:
        if not self._deferred_visual_attached or not self._deferred_visual_modalities:
            return
        self._emit_probe_progress(
            "omnigibson_deferred_visual_modalities_suspend_start",
            deferred_visual_modalities=self._deferred_visual_modalities,
        )
        _remove_behavior1k_deferred_visual_modalities(
            self._env, self._deferred_visual_modalities
        )
        self._deferred_visual_attached = False
        self._emit_probe_progress(
            "omnigibson_deferred_visual_modalities_suspend_done",
            deferred_visual_modalities=self._deferred_visual_modalities,
        )

    def _refresh_asset_registry(self) -> None:
        scene = getattr(_unwrap_env(self._env), "scene", None)
        objects = (
            getattr(scene, "objects", None) or getattr(scene, "_objects", None) or []
        )
        registry: dict[str, JsonDict] = {}
        if isinstance(objects, dict):
            iterable = objects.items()
        else:
            iterable = [
                (getattr(obj, "name", f"asset_{idx}"), obj)
                for idx, obj in enumerate(objects)
            ]
        for name, obj in iterable:
            registry[str(name)] = _asset_payload(obj)
        self._objects = registry

    def _primitive_observe_behavior1k_state(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        include_raw: bool = False,
    ) -> PrimitiveResult:
        output = {
            "language_task": self._language_task(
                prompt=prompt, query=query, agent_context=agent_context or {}
            ),
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_evidence": summarize_data(self._last_obs),
            "visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                self._last_obs, self._objects
            ),
            "visual_evidence": behavior1k_visual_evidence(
                self._last_obs,
                assets=self._objects,
                observation_info=self._last_info,
                prompt=prompt,
                query=query,
                agent_context=agent_context or {},
            ),
            "action_evidence": {
                "last_info_summary": summarize_data(
                    _public_behavior1k_info(self._last_info)
                )
            },
            "asset_evidence": deepcopy(self._objects),
            "runtime": self.runtime_available(),
        }
        if include_raw:
            output["raw_observation"] = _to_builtin(self._last_obs)
        return PrimitiveResult(name="observe_behavior1k_state", ok=True, output=output)

    def _primitive_get_behavior1k_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="get_behavior1k_task_context",
            ok=True,
            output={
                "language_task": self._language_task(
                    prompt=prompt, query=query, agent_context=agent_context or {}
                ),
                "bddl_task": {
                    "task_name": self.config.task_name,
                    "definitions_location": "bddl3/bddl/activity_definitions/<task_name>/problem_0.bddl",
                    "predicate_success_exposed": False,
                },
                "runtime": self.runtime_available(),
            },
        )

    def _primitive_inspect_behavior1k_asset(
        self,
        prompt: str | None = None,
        query: str | None = None,
        asset_name: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        selected_name = _select_name(self._objects, asset_name)
        candidates = [
            {**deepcopy(payload), "name": name}
            for name, payload in sorted(self._objects.items())
        ]
        if selected_name is None:
            return PrimitiveResult(
                name="inspect_behavior1k_asset",
                ok=False,
                output={
                    "language_task": self._language_task(
                        prompt=prompt, query=query, agent_context=agent_context or {}
                    ),
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                    "candidates": candidates,
                    "asset_evidence": deepcopy(self._objects),
                },
                error="asset_not_found",
            )
        selected = {**deepcopy(self._objects[selected_name]), "name": selected_name}
        selected["evidence"] = {
            "source": "omnigibson_scene_assets",
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
        }
        return PrimitiveResult(
            name="inspect_behavior1k_asset",
            ok=True,
            output={
                "language_task": self._language_task(
                    prompt=prompt, query=query, agent_context=agent_context or {}
                ),
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "selected": selected,
                "candidates": candidates,
                "asset_evidence": deepcopy(self._objects),
            },
        )

    def _primitive_inspect_behavior1k_visual_evidence(
        self,
        prompt: str | None = None,
        query: str | None = None,
        camera_name: str | None = None,
        entity_name: str | None = None,
        instance_id: int | None = None,
        label_kind: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        self._ensure_behavior1k_deferred_visual_modalities_attached()
        output = behavior1k_visual_evidence(
            self._last_obs,
            assets=self._objects,
            observation_info=self._last_info,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
            camera_name=camera_name,
            entity_name=entity_name,
            instance_id=instance_id,
            label_kind=label_kind,
        )
        artifacts: list[str] = []
        for grounding in output["groundings"]:
            artifact_id = (
                f"behavior1k:visual_grounding:{len(self.get_trace().artifacts)}"
            )
            grounding["evidence_handle"] = artifact_id
            self.get_trace().add_artifact(
                artifact_id,
                {
                    "source": "omnigibson_observation",
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                    "grounding": deepcopy(grounding),
                },
            )
            artifacts.append(artifact_id)
        output["evidence_handles"] = artifacts
        return PrimitiveResult(
            name="inspect_behavior1k_visual_evidence",
            ok=bool(output["visual_runtime"]["available"]),
            output=output,
            artifacts=artifacts,
        )

    def _primitive_convert_behavior1k_grounding_to_controller_input(
        self,
        evidence_handle: str,
        robot_name: str,
        controller_name: str,
        source_field: str,
        component_indices: list[int] | None = None,
        scale: list[float] | float | None = None,
        offset: list[float] | float | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact = self.get_trace().artifacts.get(evidence_handle)
        grounding = artifact.get("grounding") if isinstance(artifact, dict) else None
        if not isinstance(grounding, dict):
            return PrimitiveResult(
                name="convert_behavior1k_grounding_to_controller_input",
                ok=False,
                output={
                    "evidence_handle": evidence_handle,
                    "available_handles": sorted(self.get_trace().artifacts),
                },
                error="visual_grounding_handle_not_found",
            )
        try:
            command_values, conversion = _behavior1k_grounding_controller_values(
                grounding,
                source_field=source_field,
                component_indices=component_indices,
                scale=scale,
                offset=offset,
            )
            action, command_evidence = _behavior1k_controller_action(
                self._env,
                robot_name,
                {controller_name: command_values},
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return PrimitiveResult(
                name="convert_behavior1k_grounding_to_controller_input",
                ok=False,
                output={
                    "evidence_handle": evidence_handle,
                    "action_schema": behavior1k_action_schema(self._env),
                    "control_state": _behavior1k_control_schema(self._env),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        return PrimitiveResult(
            name="convert_behavior1k_grounding_to_controller_input",
            ok=True,
            output={
                "evidence_handle": evidence_handle,
                "controller_input": {
                    "robot_name": robot_name,
                    "controller_name": controller_name,
                    "values": _to_builtin(command_values),
                    "conversion": conversion,
                    "native_controller": command_evidence,
                },
                "action": _to_builtin(action),
                "action_schema": behavior1k_action_schema(self._env),
                "stepped": False,
                "agent_context": agent_context or {},
            },
            artifacts=[evidence_handle],
        )

    def _primitive_inspect_behavior1k_object_state(
        self,
        asset_name: str,
        state_name: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        payload = self._objects.get(asset_name)
        if not isinstance(payload, dict):
            return PrimitiveResult(
                name="inspect_behavior1k_object_state",
                ok=False,
                output={"read_only": True, "available_assets": sorted(self._objects)},
                error="asset_not_found",
            )
        states = deepcopy(payload.get("object_states", {}))
        if state_name is not None:
            states = (
                {state_name: states.get(state_name)} if state_name in states else {}
            )
        return PrimitiveResult(
            name="inspect_behavior1k_object_state",
            ok=state_name is None or bool(states),
            output={
                "selected_asset": {"name": asset_name, **deepcopy(payload)},
                "object_states": states,
                "interaction_sites": _behavior1k_object_interaction_sites(
                    _behavior1k_lookup_scene_object(self._env, asset_name),
                    state_name=state_name,
                ),
                "state_name": state_name,
                "read_only": True,
                "agent_context": agent_context or {},
            },
            error=None if state_name is None or states else "object_state_not_found",
        )

    def _primitive_inspect_behavior1k_control(
        self,
        robot_name: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        robots = _behavior1k_control_schema(self._env)
        selected_name = (
            robot_name
            if robot_name in robots
            else (
                next(iter(robots)) if robot_name is None and len(robots) == 1 else None
            )
        )
        return PrimitiveResult(
            name="inspect_behavior1k_control",
            ok=selected_name is not None,
            output={
                "selected_robot": deepcopy(robots.get(selected_name, {}))
                if selected_name is not None
                else None,
                "selected_robot_name": selected_name,
                "robots": robots,
                "action_schema": behavior1k_action_schema(self._env),
                "read_only": True,
                "agent_context": agent_context or {},
            },
            error=None if selected_name is not None else "robot_not_found",
        )

    def _primitive_inspect_behavior1k_contacts(
        self,
        asset_name: str,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        obj = _behavior1k_lookup_scene_object(self._env, asset_name)
        if obj is None:
            return PrimitiveResult(
                name="inspect_behavior1k_contacts",
                ok=False,
                output={"read_only": True, "available_assets": sorted(self._objects)},
                error="asset_not_found",
            )
        contact_list = getattr(obj, "contact_list", None)
        if not callable(contact_list):
            return PrimitiveResult(
                name="inspect_behavior1k_contacts",
                ok=False,
                output={"selected_asset": {"name": asset_name}, "read_only": True},
                error="contact_api_unavailable",
            )
        try:
            contacts = [
                _behavior1k_contact_payload(contact) for contact in contact_list()
            ]
        except Exception as exc:
            return PrimitiveResult(
                name="inspect_behavior1k_contacts",
                ok=False,
                output={"selected_asset": {"name": asset_name}, "read_only": True},
                error=f"{type(exc).__name__}: {exc}",
            )
        return PrimitiveResult(
            name="inspect_behavior1k_contacts",
            ok=True,
            output={
                "selected_asset": {
                    "name": asset_name,
                    **deepcopy(self._objects.get(asset_name, {})),
                },
                "contacts": contacts,
                "read_only": True,
                "agent_context": agent_context or {},
            },
        )

    def _primitive_record_behavior1k_evidence(
        self,
        key: str,
        value: Any,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"behavior1k:evidence:{key}"
        payload = {
            "key": key,
            "value": _to_builtin(value),
            "agent_context": agent_context or {},
        }
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="record_behavior1k_evidence",
            ok=True,
            output={"artifact_id": artifact_id, "agent_context": agent_context or {}},
            artifacts=[artifact_id],
        )

    def _primitive_settle_behavior1k(
        self,
        steps: int,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            return PrimitiveResult(
                name="settle_behavior1k",
                ok=False,
                output={
                    "settled": False,
                    "step_count": 0,
                    "agent_context": agent_context or {},
                },
                error="steps_must_be_nonnegative_int",
            )
        if self._env is None or not hasattr(self._env, "step"):
            return PrimitiveResult(
                name="settle_behavior1k",
                ok=False,
                output={"settled": False, "step_count": 0, "requires_live_env": True},
                error="live_env_step_unavailable",
            )
        records = self._settle_behavior1k_env(steps)
        self._refresh_asset_registry()
        terminated = any(
            bool(record.get("terminated") or record.get("truncated"))
            for record in records
        )
        return PrimitiveResult(
            name="settle_behavior1k",
            ok=len(records) == steps and all(record.get("ok") for record in records),
            output={
                "settled": len(records) == steps
                and all(record.get("ok") for record in records),
                "step_count": len(records),
                "steps": records,
                "termination": {"terminated": terminated},
                "agent_context": agent_context or {},
                "observation_evidence": summarize_data(self._last_obs),
                "visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                    self._last_obs, self._objects
                ),
            },
            error=None
            if len(records) == steps and all(record.get("ok") for record in records)
            else "settle_incomplete",
        )

    def _primitive_submit_behavior1k_action(
        self,
        action: Any | None = None,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._step_behavior1k_action(
            primitive_name="submit_behavior1k_action",
            action=action,
            prompt=prompt,
            query=query,
            evidence_handle=evidence_handle,
            evidence_handles=evidence_handles,
            suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
            include_post_step_visual_evidence=include_post_step_visual_evidence,
            agent_context=agent_context,
        )

    def _primitive_step_behavior1k_action(
        self,
        action: Any | None = None,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._step_behavior1k_action(
            primitive_name="step_behavior1k_action",
            action=action,
            prompt=prompt,
            query=query,
            evidence_handle=evidence_handle,
            evidence_handles=evidence_handles,
            suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
            include_post_step_visual_evidence=include_post_step_visual_evidence,
            agent_context=agent_context,
        )

    def _primitive_execute_behavior1k_action_sequence(
        self,
        actions: list[Any] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
        stop_on_termination: bool = True,
    ) -> PrimitiveResult:
        if self._env is None or not hasattr(self._env, "step"):
            return PrimitiveResult(
                name="execute_behavior1k_action_sequence",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "requires_live_env": True,
                    "blocker": "live_env_step_unavailable",
                    "action_schema": behavior1k_action_schema(self._env),
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                },
                error="live_env_step_unavailable",
            )
        if actions is None:
            return PrimitiveResult(
                name="execute_behavior1k_action_sequence",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error="agent_actions_required",
            )
        if not isinstance(actions, list):
            return PrimitiveResult(
                name="execute_behavior1k_action_sequence",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "action_schema": behavior1k_action_schema(self._env),
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                },
                error="actions_must_be_list",
            )
        validated_handles, provenance, evidence_error = (
            self._behavior1k_action_evidence_provenance(
                evidence_handle=evidence_handle,
                evidence_handles=evidence_handles,
            )
        )
        if evidence_error is not None:
            return PrimitiveResult(
                name="execute_behavior1k_action_sequence",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "evidence_handles": validated_handles,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error=evidence_error,
            )

        steps: list[JsonDict] = []
        stopped_on_termination = False
        for index, action in enumerate(actions):
            result = self._step_behavior1k_action(
                primitive_name="step_behavior1k_action",
                action=action,
                prompt=prompt,
                query=query,
                evidence_handles=validated_handles,
                suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
                include_post_step_visual_evidence=include_post_step_visual_evidence,
                agent_context={**(agent_context or {}), "sequence_index": index},
            )
            if not result.ok:
                return PrimitiveResult(
                    name="execute_behavior1k_action_sequence",
                    ok=False,
                    output={
                        "stepped": bool(steps),
                        "step_count": len(steps),
                        "steps": steps,
                        "failed_step_index": index,
                        "failed_step": result.to_dict(),
                        "action_schema": behavior1k_action_schema(self._env),
                        "prompt": prompt,
                        "query": query,
                        "evidence_handles": validated_handles,
                        "evidence_provenance": provenance,
                        "agent_context": agent_context or {},
                    },
                    error=result.error,
                )
            step_payload = deepcopy(result.output)
            step_payload["sequence_index"] = index
            steps.append(step_payload)
            if stop_on_termination and bool(
                step_payload.get("terminated") or step_payload.get("truncated")
            ):
                stopped_on_termination = True
                break

        return PrimitiveResult(
            name="execute_behavior1k_action_sequence",
            ok=True,
            output={
                "stepped": bool(steps),
                "step_count": len(steps),
                "steps": steps,
                "stopped_on_termination": stopped_on_termination,
                "action_schema": behavior1k_action_schema(self._env),
                "prompt": prompt,
                "query": query,
                "evidence_handles": validated_handles,
                "evidence_provenance": provenance,
                "agent_context": agent_context or {},
                "language_task": self._language_task(
                    prompt=prompt, query=query, agent_context=agent_context or {}
                ),
                "final_observation_evidence": summarize_data(self._last_obs),
                "final_visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                    self._last_obs, self._objects
                ),
                "final_visual_evidence": behavior1k_visual_evidence(
                    self._last_obs,
                    assets=self._objects,
                    observation_info=self._last_info,
                    prompt=prompt,
                    query=query,
                    agent_context=agent_context or {},
                ),
                "asset_evidence": deepcopy(self._objects),
            },
            artifacts=validated_handles,
        )

    def _primitive_execute_behavior1k_controller_command(
        self,
        robot_name: str,
        commands: JsonDict,
        repeat: int = 1,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if (
            not isinstance(repeat, int)
            or isinstance(repeat, bool)
            or repeat < 1
            or repeat > 256
        ):
            return PrimitiveResult(
                name="execute_behavior1k_controller_command",
                ok=False,
                output={"stepped": False, "step_count": 0},
                error="repeat_must_be_int_between_1_and_256",
            )
        try:
            action, command_evidence = _behavior1k_controller_action(
                self._env, robot_name, commands
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as exc:
            return PrimitiveResult(
                name="execute_behavior1k_controller_command",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "control_state": _behavior1k_control_schema(self._env),
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error=f"{type(exc).__name__}: {exc}",
            )
        before_control_state = _behavior1k_control_schema(self._env)
        result = self._primitive_execute_behavior1k_action_sequence(
            actions=[action for _ in range(repeat)],
            prompt=prompt,
            query=query,
            evidence_handle=evidence_handle,
            evidence_handles=evidence_handles,
            suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
            include_post_step_visual_evidence=include_post_step_visual_evidence,
            agent_context=agent_context,
            stop_on_termination=True,
        )
        after_control_state = _behavior1k_control_schema(self._env)
        output = deepcopy(result.output)
        output["controller_command"] = command_evidence
        output["control_state"] = after_control_state
        output["control_response"] = _behavior1k_control_response(
            before_control_state,
            after_control_state,
            robot_name=robot_name,
        )
        return PrimitiveResult(
            name="execute_behavior1k_controller_command",
            ok=result.ok,
            output=output,
            error=result.error,
            artifacts=result.artifacts,
        )

    def _primitive_navigate_behavior1k_base_to_pose(
        self,
        robot_name: str,
        base_controller_name: str,
        target_pose_xyyaw: list[float],
        max_steps: int = 160,
        distance_tolerance: float = 0.08,
        yaw_tolerance: float = 0.12,
        linear_gain: float = 1.5,
        angular_gain: float = 1.8,
        linear_limit: float = 1.0,
        angular_limit: float = 1.0,
        turn_in_place_yaw_threshold: float | None = 0.35,
        command_mode: str = "auto",
        repeat_per_command: int = 1,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        validated_handles, provenance, evidence_error = (
            self._behavior1k_action_evidence_provenance(
                evidence_handle=evidence_handle,
                evidence_handles=evidence_handles,
            )
        )
        if evidence_error is not None:
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={
                    "stepped": False,
                    "reached": False,
                    "step_count": 0,
                    "evidence_handles": validated_handles,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error=evidence_error,
            )
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
            or max_steps > 2048
        ):
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={"stepped": False, "reached": False, "step_count": 0},
                error="max_steps_must_be_int_between_1_and_2048",
            )
        if (
            not isinstance(repeat_per_command, int)
            or isinstance(repeat_per_command, bool)
            or repeat_per_command < 1
            or repeat_per_command > 64
        ):
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={"stepped": False, "reached": False, "step_count": 0},
                error="repeat_per_command_must_be_int_between_1_and_64",
            )
        if (
            not isinstance(target_pose_xyyaw, (list, tuple))
            or len(target_pose_xyyaw) < 3
        ):
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={"stepped": False, "reached": False, "step_count": 0},
                error="target_pose_xyyaw_requires_x_y_yaw",
            )
        try:
            target = [
                float(target_pose_xyyaw[0]),
                float(target_pose_xyyaw[1]),
                float(target_pose_xyyaw[2]),
            ]
            distance_tol = max(0.0, float(distance_tolerance))
            yaw_tol = max(0.0, float(yaw_tolerance))
            linear_gain_value = max(0.0, float(linear_gain))
            angular_gain_value = max(0.0, float(angular_gain))
            linear_limit_value = max(0.0, float(linear_limit))
            angular_limit_value = max(0.0, float(angular_limit))
            turn_threshold_value = (
                None
                if turn_in_place_yaw_threshold is None
                else max(0.0, float(turn_in_place_yaw_threshold))
            )
        except Exception as exc:
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={"stepped": False, "reached": False, "step_count": 0},
                error=f"{type(exc).__name__}: invalid_navigation_numeric_parameter",
            )
        control_state = _behavior1k_control_schema(self._env)
        robot = (
            control_state.get(robot_name) if isinstance(control_state, dict) else None
        )
        if not isinstance(robot, dict):
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={
                    "stepped": False,
                    "reached": False,
                    "step_count": 0,
                    "control_state": control_state,
                },
                error=f"unknown_robot:{robot_name}",
            )
        controllers = (
            robot.get("controllers")
            if isinstance(robot.get("controllers"), dict)
            else {}
        )
        controller = (
            controllers.get(base_controller_name)
            if isinstance(controllers, dict)
            else None
        )
        if not isinstance(controller, dict):
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={
                    "stepped": False,
                    "reached": False,
                    "step_count": 0,
                    "control_state": control_state,
                },
                error=f"unknown_base_controller:{base_controller_name}",
            )

        try:
            mode = _behavior1k_resolve_base_command_mode(command_mode, controller)
        except ValueError as exc:
            return PrimitiveResult(
                name="navigate_behavior1k_base_to_pose",
                ok=False,
                output={
                    "stepped": False,
                    "reached": False,
                    "step_count": 0,
                    "control_state": control_state,
                },
                error=str(exc),
            )
        trace: list[JsonDict] = []
        total_steps = 0
        reached = False
        final_progress: JsonDict = {}
        terminated = False
        truncated = False
        last_result: PrimitiveResult | None = None
        for iteration in range(max_steps):
            control_state = _behavior1k_control_schema(self._env)
            robot = (
                control_state.get(robot_name)
                if isinstance(control_state, dict)
                else None
            )
            pose = robot.get("pose_world") if isinstance(robot, dict) else None
            if not isinstance(pose, list):
                return PrimitiveResult(
                    name="navigate_behavior1k_base_to_pose",
                    ok=False,
                    output={
                        "stepped": bool(total_steps),
                        "reached": False,
                        "step_count": total_steps,
                        "trace": trace,
                        "control_state": control_state,
                        "evidence_handles": validated_handles,
                        "evidence_provenance": provenance,
                    },
                    error="robot_pose_unavailable",
                    artifacts=validated_handles,
                )
            try:
                command, progress = _behavior1k_base_pose_controller_command(
                    pose,
                    target,
                    mode=mode,
                    distance_tolerance=distance_tol,
                    yaw_tolerance=yaw_tol,
                    linear_gain=linear_gain_value,
                    angular_gain=angular_gain_value,
                    linear_limit=linear_limit_value,
                    angular_limit=angular_limit_value,
                    turn_in_place_yaw_threshold=turn_threshold_value,
                )
            except ValueError as exc:
                return PrimitiveResult(
                    name="navigate_behavior1k_base_to_pose",
                    ok=False,
                    output={
                        "stepped": bool(total_steps),
                        "reached": False,
                        "step_count": total_steps,
                        "trace": trace,
                        "control_state": control_state,
                        "evidence_handles": validated_handles,
                        "evidence_provenance": provenance,
                    },
                    error=str(exc),
                    artifacts=validated_handles,
                )
            final_progress = progress
            if bool(progress.get("reached")):
                reached = True
                break
            step = self._primitive_execute_behavior1k_controller_command(
                robot_name=robot_name,
                commands={base_controller_name: command},
                repeat=repeat_per_command,
                prompt=prompt,
                query=query,
                evidence_handles=validated_handles,
                suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
                include_post_step_visual_evidence=include_post_step_visual_evidence,
                agent_context={
                    **(agent_context or {}),
                    "target_pose_xyyaw": target,
                    "iteration": iteration,
                    "command_mode": mode,
                    "base_controller_name": base_controller_name,
                },
            )
            last_result = step
            if not step.ok:
                return PrimitiveResult(
                    name="navigate_behavior1k_base_to_pose",
                    ok=False,
                    output={
                        "stepped": bool(total_steps),
                        "reached": False,
                        "step_count": total_steps,
                        "trace": trace,
                        "final_progress": final_progress,
                        "control_state": _behavior1k_control_schema(self._env),
                        "evidence_handles": validated_handles,
                        "evidence_provenance": provenance,
                    },
                    error=step.error,
                    artifacts=validated_handles,
                )
            total_steps += int(step.output.get("step_count", 0))
            terminated = bool(step.output.get("terminated"))
            truncated = bool(step.output.get("truncated"))
            if iteration == 0 or (iteration + 1) % 10 == 0 or terminated or truncated:
                self._emit_probe_progress(
                    "behavior1k_navigation_step",
                    iteration=iteration,
                    step_count=total_steps,
                    target_pose_xyyaw=target,
                    progress=progress,
                    command_mode=mode,
                    terminated=terminated,
                    truncated=truncated,
                )
            if len(trace) < 16 or iteration % 10 == 0:
                trace.append(
                    {
                        "iteration": iteration,
                        "step_count": total_steps,
                        "target_pose_xyyaw": target,
                        "command": command,
                        "progress": progress,
                        "control_response": step.output.get("control_response"),
                    }
                )
            if terminated or truncated:
                break

        final_control_state = _behavior1k_control_schema(self._env)
        if not reached:
            robot = (
                final_control_state.get(robot_name)
                if isinstance(final_control_state, dict)
                else None
            )
            pose = robot.get("pose_world") if isinstance(robot, dict) else None
            if isinstance(pose, list):
                _, final_progress = _behavior1k_base_pose_controller_command(
                    pose,
                    target,
                    mode=mode,
                    distance_tolerance=distance_tol,
                    yaw_tolerance=yaw_tol,
                    linear_gain=linear_gain_value,
                    angular_gain=angular_gain_value,
                    linear_limit=linear_limit_value,
                    angular_limit=angular_limit_value,
                    turn_in_place_yaw_threshold=turn_threshold_value,
                )
                reached = bool(final_progress.get("reached"))
        output: JsonDict = {
            "stepped": bool(total_steps),
            "reached": bool(reached),
            "step_count": total_steps,
            "trace": trace,
            "final_progress": final_progress,
            "target_pose_xyyaw": target,
            "command_mode": mode,
            "controller_type": controller.get("type"),
            "control_state": final_control_state,
            "evidence_handles": validated_handles,
            "evidence_provenance": provenance,
            "agent_context": agent_context or {},
            "language_task": self._language_task(
                prompt=prompt, query=query, agent_context=agent_context or {}
            ),
        }
        if last_result is not None:
            output["last_control_response"] = last_result.output.get("control_response")
            output["terminated"] = terminated
            output["truncated"] = truncated
        return PrimitiveResult(
            name="navigate_behavior1k_base_to_pose",
            ok=True,
            output=output,
            artifacts=validated_handles,
        )

    def _primitive_run_behavior1k_semantic_action(
        self,
        semantic_action: str,
        asset_name: str,
        secondary_asset_name: str | None = None,
        robot_name: str | None = None,
        attempts: int = 3,
        max_steps: int = 512,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None or not hasattr(self._env, "step"):
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "requires_live_env": True,
                    "blocker": "live_env_step_unavailable",
                    "action_schema": behavior1k_action_schema(self._env),
                    "agent_context": agent_context or {},
                },
                error="live_env_step_unavailable",
            )
        if (
            not isinstance(attempts, int)
            or isinstance(attempts, bool)
            or attempts < 1
            or attempts > 16
        ):
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={"stepped": False, "step_count": 0},
                error="attempts_must_be_int_between_1_and_16",
            )
        if (
            not isinstance(max_steps, int)
            or isinstance(max_steps, bool)
            or max_steps < 1
            or max_steps > 4096
        ):
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={"stepped": False, "step_count": 0},
                error="max_steps_must_be_int_between_1_and_4096",
            )

        validated_handles, provenance, evidence_error = (
            self._behavior1k_action_evidence_provenance(
                evidence_handle=evidence_handle,
                evidence_handles=evidence_handles,
            )
        )
        if evidence_error is not None:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "evidence_handles": validated_handles,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error=evidence_error,
            )

        primary_obj = _behavior1k_lookup_scene_object(self._env, asset_name)
        secondary_obj = _behavior1k_lookup_scene_object(self._env, secondary_asset_name)
        if primary_obj is None:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "available_assets": sorted(self._objects),
                },
                error="asset_not_found",
            )
        if secondary_asset_name is not None and secondary_obj is None:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "available_assets": sorted(self._objects),
                },
                error="secondary_asset_not_found",
            )

        robots = _behavior1k_robots(self._env)
        selected_robot_name = (
            robot_name
            if robot_name in robots
            else (
                next(iter(robots)) if robot_name is None and len(robots) == 1 else None
            )
        )
        if selected_robot_name is None:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "available_robots": sorted(robots),
                },
                error="robot_not_found",
            )
        robot = robots[selected_robot_name]

        normalized_semantic_action = str(semantic_action).strip().upper()
        physical_toggle_plan = _behavior1k_physical_toggle_plan(
            self._env,
            primary_obj,
            selected_robot_name,
        )
        if (
            normalized_semantic_action in {"TOGGLE_ON", "TOGGLE_OFF"}
            and physical_toggle_plan is not None
        ):
            return self._run_behavior1k_physical_toggle(
                semantic_action=normalized_semantic_action,
                asset_name=asset_name,
                primary_obj=primary_obj,
                selected_robot_name=selected_robot_name,
                plan=physical_toggle_plan,
                max_steps=max_steps,
                validated_handles=validated_handles,
                provenance=provenance,
                include_post_step_visual_evidence=include_post_step_visual_evidence,
                agent_context=agent_context or {},
            )

        try:
            controller_cls, primitive_set, primitive = (
                _behavior1k_semantic_action_runtime(semantic_action)
            )
        except RuntimeError as exc:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "semantic_action": semantic_action,
                    "available_semantic_actions": _behavior1k_public_semantic_action_names(),
                    "agent_context": agent_context or {},
                },
                error=str(exc),
            )

        args = [primary_obj]
        if str(primitive.name).upper() in {"PLACE_ON_TOP", "PLACE_INSIDE"}:
            if secondary_obj is None:
                return PrimitiveResult(
                    name="run_behavior1k_semantic_action",
                    ok=False,
                    output={
                        "stepped": False,
                        "step_count": 0,
                        "semantic_action": primitive.name,
                    },
                    error="secondary_asset_required_for_place_action",
                )
            args = [secondary_obj]

        selected_assets = {
            "asset_name": asset_name,
            "asset_payload": deepcopy(
                self._objects.get(asset_name, _asset_payload(primary_obj))
            ),
            "secondary_asset_name": secondary_asset_name,
            "secondary_asset_payload": deepcopy(
                self._objects.get(secondary_asset_name, _asset_payload(secondary_obj))
            )
            if secondary_asset_name is not None and secondary_obj is not None
            else None,
        }
        object_states_before = {
            "primary": _behavior1k_public_object_states(primary_obj),
            "secondary": _behavior1k_public_object_states(secondary_obj)
            if secondary_obj is not None
            else {},
        }
        trace: list[JsonDict] = []
        step_count = 0
        completed = False
        terminated = False
        truncated = False
        error: str | None = None

        try:
            controller = controller_cls(self._env, robot, enable_head_tracking=False)
            if suspend_deferred_visual_modalities:
                self._suspend_behavior1k_deferred_visual_modalities()
            for step_index, action in enumerate(
                controller.apply_ref(primitive, *args, attempts=attempts)
            ):
                if step_index >= max_steps:
                    error = "semantic_action_step_budget_exhausted"
                    break
                if action is None:
                    continue
                step_result = self._env.step(action)
                obs, _reward, terminated, truncated, info = _split_step_result(
                    step_result
                )
                step_count += 1
                self._last_obs = obs
                self._last_info = info
                self._refresh_asset_registry()
                if (
                    len(trace) < 24
                    or terminated
                    or truncated
                    or (step_index + 1) % 25 == 0
                ):
                    trace.append(
                        {
                            "step_index": step_index,
                            "action_summary": summarize_data(action),
                            "terminated": terminated,
                            "truncated": truncated,
                            "observation_evidence": summarize_data(obs),
                        }
                    )
                if terminated or truncated:
                    break
            else:
                completed = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        object_states_after = {
            "primary": _behavior1k_public_object_states(primary_obj),
            "secondary": _behavior1k_public_object_states(secondary_obj)
            if secondary_obj is not None
            else {},
        }
        output: JsonDict = {
            "stepped": step_count > 0,
            "step_count": step_count,
            "completed": completed,
            "terminated": terminated,
            "truncated": truncated,
            "semantic_action": str(primitive.name),
            "available_semantic_actions": _behavior1k_public_semantic_action_names(),
            "selected_robot_name": selected_robot_name,
            "selected_assets": selected_assets,
            "object_states_before": object_states_before,
            "object_states_after": object_states_after,
            "trace": trace,
            "action_schema": behavior1k_action_schema(self._env),
            "observation_evidence": summarize_data(self._last_obs),
            "visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                self._last_obs, self._objects
            ),
            "visual_evidence": (
                behavior1k_visual_evidence(
                    self._last_obs,
                    assets=self._objects,
                    observation_info=self._last_info,
                    prompt=None,
                    query=asset_name,
                    agent_context=agent_context or {},
                )
                if include_post_step_visual_evidence
                else {}
            ),
            "evidence_handles": validated_handles,
            "evidence_provenance": provenance,
            "agent_context": agent_context or {},
        }
        return PrimitiveResult(
            name="run_behavior1k_semantic_action",
            ok=completed and error is None,
            output=output,
            error=error,
            artifacts=validated_handles,
        )

    def _run_behavior1k_physical_toggle(
        self,
        *,
        semantic_action: str,
        asset_name: str,
        primary_obj: Any,
        selected_robot_name: str,
        plan: JsonDict,
        max_steps: int,
        validated_handles: list[str],
        provenance: JsonDict,
        include_post_step_visual_evidence: bool,
        agent_context: JsonDict,
    ) -> PrimitiveResult:
        """Execute a toggle through native navigation, arm, and finger controllers.

        The public semantic request stays backend-independent. This implementation
        replaces OmniGibson's currently unimplemented starter toggle primitive with
        a physical finger/contact sequence grounded only in the public interaction
        link pose and native controller feedback. Official task success remains a
        harness-only verifier concern after execution.
        """

        object_states_before = _behavior1k_public_object_states(primary_obj)
        desired_value = semantic_action == "TOGGLE_ON"
        current_value = object_states_before.get("ToggledOn")
        selected_assets = {
            "asset_name": asset_name,
            "asset_payload": deepcopy(
                self._objects.get(asset_name, _asset_payload(primary_obj))
            ),
            "secondary_asset_name": None,
            "secondary_asset_payload": None,
        }
        if isinstance(current_value, bool) and current_value == desired_value:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=True,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "completed": True,
                    "terminated": False,
                    "truncated": False,
                    "semantic_action": semantic_action,
                    "execution_mode": "physical_native_controller_toggle",
                    "available_semantic_actions": _behavior1k_public_semantic_action_names(),
                    "selected_robot_name": selected_robot_name,
                    "selected_assets": selected_assets,
                    "object_states_before": {
                        "primary": object_states_before,
                        "secondary": {},
                    },
                    "object_states_after": {
                        "primary": deepcopy(object_states_before),
                        "secondary": {},
                    },
                    "trace": [],
                    "physical_toggle_plan": deepcopy(plan),
                    "action_schema": behavior1k_action_schema(self._env),
                    "evidence_handles": validated_handles,
                    "evidence_provenance": provenance,
                    "agent_context": agent_context,
                },
                artifacts=validated_handles,
            )

        total_steps = 0
        trace: list[JsonDict] = []
        terminated = False
        truncated = False
        error: str | None = None
        completed = False
        initial_target_pose = [float(value) for value in plan["target_pose"][:7]]
        target_pose = list(initial_target_pose)
        surface_normal = list(plan["surface_normal"])
        base_name = plan["base_controller_name"]
        arm_name = plan["arm_controller_name"]
        gripper_name = plan["gripper_controller_name"]
        logical_arm_name = plan["logical_arm_name"]
        arm_position_scale = float(plan.get("arm_position_command_scale", 0.2) or 0.2)
        standoff_distance = float(plan.get("standoff_distance", 0.84) or 0.84)
        navigation_replan: JsonDict | None = None

        def refresh_live_target() -> JsonDict | None:
            nonlocal target_pose, surface_normal
            live = _behavior1k_live_toggle_target(
                primary_obj, standoff=standoff_distance
            )
            if live is not None:
                target_pose = list(live["target_pose"])
                surface_normal = list(live["surface_normal"])
                plan["live_target_pose"] = list(target_pose)
                plan["surface_normal"] = list(surface_normal)
                plan["live_standoff_pose"] = list(live["standoff_pose"])
            return live

        def navigate_support_aware(
            live_target: JsonDict,
            *,
            budget: int,
            phase: str,
        ) -> tuple[PrimitiveResult, int, JsonDict, list[JsonDict]]:
            control_state = _behavior1k_control_schema(self._env)
            robot = (
                control_state.get(selected_robot_name)
                if isinstance(control_state, dict)
                else None
            )
            robot_pose = robot.get("pose_world") if isinstance(robot, dict) else None
            route = _behavior1k_support_aware_navigation_path(
                self._env,
                primary_obj,
                robot_pose,
                live_target["standoff_pose"],
                live_target["target_pose"],
            )
            targets = [list(item) for item in route.get("waypoints", [])]
            if route.get("direct_path_clear") is False and not targets:
                blocked = PrimitiveResult(
                    name="navigate_behavior1k_base_to_pose",
                    ok=False,
                    output={
                        "reached": False,
                        "step_count": 0,
                        "control_state": control_state,
                        "support_aware_path": deepcopy(route),
                    },
                    error="support_aware_navigation_detour_unavailable",
                )
                return blocked, 0, route, []
            targets.append(list(live_target["standoff_pose"]))
            route_steps = 0
            legs: list[JsonDict] = []
            last: PrimitiveResult | None = None
            for index, route_target in enumerate(targets):
                remaining = max(0, int(budget) - route_steps)
                if remaining <= 0:
                    break
                is_waypoint = index < len(targets) - 1
                leg_budget = (
                    min(200, max(1, remaining - 1)) if is_waypoint else remaining
                )
                last = self._primitive_navigate_behavior1k_base_to_pose(
                    robot_name=selected_robot_name,
                    base_controller_name=base_name,
                    target_pose_xyyaw=route_target,
                    max_steps=leg_budget,
                    distance_tolerance=0.08,
                    yaw_tolerance=math.pi if is_waypoint else 0.18,
                    command_mode="auto",
                    repeat_per_command=1,
                    query=(
                        "public support-AABB safe navigation waypoint"
                        if is_waypoint
                        else "public interaction-link surface-normal navigation"
                    ),
                    evidence_handles=validated_handles,
                    suspend_deferred_visual_modalities=True,
                    include_post_step_visual_evidence=False,
                    agent_context={
                        **agent_context,
                        "semantic_action": semantic_action,
                        "phase": phase,
                        "navigation_leg": "support_detour_waypoint"
                        if is_waypoint
                        else "live_standoff",
                        "target_source": "public_scene_geometry",
                    },
                )
                steps = int(last.output.get("step_count", 0))
                route_steps += steps
                legs.append(
                    {
                        "kind": "support_detour_waypoint"
                        if is_waypoint
                        else "live_standoff",
                        "target_pose_xyyaw": route_target,
                        "budget": leg_budget,
                        "step_count": steps,
                        "ok": last.ok,
                        "reached": bool(last.output.get("reached")),
                        "final_progress": deepcopy(
                            last.output.get("final_progress", {})
                        ),
                    }
                )
                if not last.ok or not bool(last.output.get("reached")):
                    break
            if last is None:
                last = PrimitiveResult(
                    name="navigate_behavior1k_base_to_pose",
                    ok=False,
                    output={
                        "reached": False,
                        "step_count": 0,
                        "control_state": control_state,
                    },
                    error="support_aware_navigation_budget_exhausted",
                )
            return last, route_steps, route, legs

        navigation_budget = max(0, max_steps - 120)
        if navigation_budget <= 0:
            return PrimitiveResult(
                name="run_behavior1k_semantic_action",
                ok=False,
                output={
                    "stepped": False,
                    "step_count": 0,
                    "physical_toggle_plan": deepcopy(plan),
                },
                error="physical_toggle_step_budget_too_small",
                artifacts=validated_handles,
            )
        initial_live_target: JsonDict = {
            "target_pose": list(initial_target_pose),
            "standoff_pose": list(plan["standoff_pose"]),
        }
        navigation, navigation_steps, initial_navigation_path, navigation_legs = (
            navigate_support_aware(
                initial_live_target,
                budget=navigation_budget,
                phase="navigate",
            )
        )
        total_steps += navigation_steps
        plan["initial_navigation_path"] = deepcopy(initial_navigation_path)
        trace.append(
            {
                "phase": "navigate",
                "ok": navigation.ok,
                "reached": bool(navigation.output.get("reached")),
                "step_count": navigation_steps,
                "support_aware_path": deepcopy(initial_navigation_path),
                "navigation_legs": deepcopy(navigation_legs),
                "final_progress": deepcopy(navigation.output.get("final_progress", {})),
            }
        )
        if not navigation.ok:
            error = navigation.error or "physical_toggle_navigation_failed"
        elif not bool(navigation.output.get("reached")):
            error = "physical_toggle_navigation_target_not_reached"

        control_state = (
            navigation.output.get("control_state", {})
            if isinstance(navigation.output, dict)
            else {}
        )
        robot_state = (
            control_state.get(selected_robot_name)
            if isinstance(control_state, dict)
            else None
        )
        live_after_navigation = refresh_live_target()
        stale_target_delta = (
            float(
                np.linalg.norm(
                    np.asarray(target_pose[:3]) - np.asarray(initial_target_pose[:3])
                )
            )
            if live_after_navigation is not None
            else 0.0
        )
        plan["initial_target_pose"] = list(initial_target_pose)
        plan["live_target_pose"] = list(target_pose)
        plan["stale_target_delta_m"] = stale_target_delta
        plan["navigation_replanned"] = False
        trace.append(
            {
                "phase": "refresh_live_target_after_navigation",
                "available": live_after_navigation is not None,
                "initial_target_pose": list(initial_target_pose),
                "live_target_pose": list(target_pose),
                "stale_target_delta_m": stale_target_delta,
                "replan_threshold_m": 0.02,
            }
        )
        if error is None and live_after_navigation is None:
            error = "physical_toggle_live_target_unavailable_after_navigation"
        elif error is None and stale_target_delta > 0.02:
            replan_budget = max(0, max_steps - total_steps - 80)
            if replan_budget <= 0:
                error = "physical_toggle_replan_step_budget_exhausted"
            else:
                replan_navigation, replan_steps, replan_path, replan_legs = (
                    navigate_support_aware(
                        live_after_navigation,
                        budget=replan_budget,
                        phase="navigate_live_replan",
                    )
                )
                total_steps += replan_steps
                navigation_replan = replan_navigation.to_dict()
                plan["navigation_replanned"] = True
                plan["replan_standoff_pose"] = list(
                    live_after_navigation["standoff_pose"]
                )
                plan["replan_navigation_path"] = deepcopy(replan_path)
                trace.append(
                    {
                        "phase": "navigate_live_replan",
                        "ok": replan_navigation.ok,
                        "reached": bool(replan_navigation.output.get("reached")),
                        "step_count": replan_steps,
                        "target_pose": list(target_pose),
                        "standoff_pose": list(live_after_navigation["standoff_pose"]),
                        "support_aware_path": deepcopy(replan_path),
                        "navigation_legs": deepcopy(replan_legs),
                        "final_progress": deepcopy(
                            replan_navigation.output.get("final_progress", {})
                        ),
                    }
                )
                if not replan_navigation.ok:
                    error = (
                        replan_navigation.error
                        or "physical_toggle_replan_navigation_failed"
                    )
                elif not bool(replan_navigation.output.get("reached")):
                    error = "physical_toggle_replan_navigation_target_not_reached"
                control_state = (
                    replan_navigation.output.get("control_state", {})
                    if isinstance(replan_navigation.output, dict)
                    else {}
                )
                robot_state = (
                    control_state.get(selected_robot_name)
                    if isinstance(control_state, dict)
                    else None
                )
                final_live = refresh_live_target()
                if error is None and final_live is None:
                    error = "physical_toggle_live_target_unavailable_after_replan"

        selected_finger_link_name: str | None = "right"
        selected_finger_link: JsonDict | None = None
        contact_anchor_pose: list[float] | None = None
        contact_surface_normal: list[float] | None = None
        contact_eef_from_finger_offset: list[float] | None = None
        contact_finger_pose: list[float] | None = None
        contact_press_normal: list[float] | None = None
        contact_press_normal_source: str | None = None
        contact_observed = False
        contact_consecutive_steps = 0
        max_contact_consecutive_steps = 0
        contact_hold_completed = False

        # Close the gripper while the arm is still clear of the interaction
        # site. Approaching with the Fetch fingers open lets the unused finger
        # or palm catch the small movable appliance before the selected finger
        # reaches the toggle marker.
        if error is None and not terminated and not truncated:
            for close_index in range(min(8, max(0, max_steps - total_steps))):
                close = self._primitive_execute_behavior1k_controller_command(
                    robot_name=selected_robot_name,
                    commands={gripper_name: [-1.0]},
                    repeat=1,
                    query="close native fingers before approaching the public interaction link",
                    evidence_handles=validated_handles,
                    suspend_deferred_visual_modalities=True,
                    include_post_step_visual_evidence=False,
                    agent_context={
                        **agent_context,
                        "semantic_action": semantic_action,
                        "phase": "close_gripper_clear",
                        "close_index": close_index,
                    },
                )
                total_steps += int(close.output.get("step_count", 0))
                trace.append(
                    {
                        "phase": "close_gripper_clear",
                        "iteration": close_index,
                        "ok": close.ok,
                        "step_count": total_steps,
                    }
                )
                if not close.ok:
                    error = close.error or "physical_toggle_clear_gripper_close_failed"
                    break
                terminated = bool(close.output.get("terminated"))
                truncated = bool(close.output.get("truncated"))
                next_control_state = close.output.get("control_state")
                robot_state = (
                    next_control_state.get(selected_robot_name)
                    if isinstance(next_control_state, dict)
                    else None
                )
                if terminated or truncated:
                    break

        phase_specs = (
            # Slow the final part of this collision-free pre-contact motion
            # below.  A full one-step IK correction overshoots near the target
            # even though the pose remains reachable.
            ("approach", 0.015, 65, [-1.0]),
            # The official state checks finger rigid bodies, not the EEF. The
            # contact phase therefore tracks a public native finger-link pose
            # directly and uses a tighter target tolerance than the old EEF
            # collision-envelope approximation.
            ("contact", 0.012, 20, [-1.0]),
        )
        for phase, tolerance, phase_budget, gripper_command in phase_specs:
            if error is not None:
                break
            reached_phase = False
            for iteration in range(min(phase_budget, max(0, max_steps - total_steps))):
                live = refresh_live_target()
                if live is None:
                    error = f"physical_toggle_{phase}_live_target_unavailable"
                    break
                contact_offset_m: float | None = None
                if phase == "approach":
                    pose = [
                        float(target_pose[index]) - 0.08 * float(surface_normal[index])
                        for index in range(3)
                    ]
                else:
                    # Freeze the public marker pose before first contact. A
                    # toggleable object is free to move, so chasing its live
                    # marker after contact adds force and can tip the object.
                    if contact_anchor_pose is None:
                        contact_anchor_pose = [
                            float(value) for value in target_pose[:3]
                        ]
                        normal = np.asarray(surface_normal[:3], dtype=np.float64)
                        normal_norm = float(np.linalg.norm(normal))
                        if normal_norm <= 1e-6:
                            error = "physical_toggle_contact_surface_normal_unavailable"
                            break
                        contact_surface_normal = (normal / normal_norm).tolist()
                        plan["contact_anchor_pose"] = list(contact_anchor_pose)
                        plan["contact_surface_normal_unit"] = list(
                            contact_surface_normal
                        )
                    # Aim at the frozen marker center, but stop on the first
                    # public native finger contact. Earlier point chasing
                    # pushed the object only because it kept applying force
                    # after contact while following the moving marker.
                    contact_offset_m = 0.0
                    pose = list(contact_anchor_pose)
                if not isinstance(robot_state, dict):
                    error = "physical_toggle_control_state_unavailable"
                    break
                robot_pose = robot_state.get("pose_world")
                end_effectors = robot_state.get("end_effectors")
                eef = (
                    end_effectors.get(logical_arm_name)
                    if isinstance(end_effectors, dict)
                    else None
                )
                if not isinstance(eef, dict) and isinstance(end_effectors, dict):
                    eef = next(
                        (
                            value
                            for value in end_effectors.values()
                            if isinstance(value, dict)
                        ),
                        None,
                    )
                eef_pose = eef.get("pose_world") if isinstance(eef, dict) else None
                if not isinstance(robot_pose, list) or not isinstance(eef_pose, list):
                    error = "physical_toggle_robot_or_eef_pose_unavailable"
                    break
                tracking_pose = eef_pose
                tracking_link = "eef"
                controller_target_pose = list(pose)
                finger_distance: float | None = None
                finger_contacts: list[JsonDict] = []
                finger_link = _behavior1k_closest_finger_link(
                    robot_state,
                    logical_arm_name,
                    pose,
                    preferred_name=selected_finger_link_name,
                )
                if finger_link is not None and isinstance(
                    finger_link.get("pose_world"), list
                ):
                    selected_finger_link = finger_link
                    finger_pose = finger_link["pose_world"]
                    current_eef_from_finger_offset = [
                        float(eef_pose[index]) - float(finger_pose[index])
                        for index in range(3)
                    ]
                    if phase == "contact":
                        if contact_eef_from_finger_offset is None:
                            contact_eef_from_finger_offset = [
                                float(value) for value in current_eef_from_finger_offset
                            ]
                            plan["contact_eef_from_finger_offset"] = list(
                                contact_eef_from_finger_offset
                            )
                    else:
                        plan["approach_eef_from_finger_offset"] = list(
                            current_eef_from_finger_offset
                        )
                    # Recompute the public EEF-to-finger translation each
                    # step. The wrist orientation changes during IK, so a
                    # reset-time offset is not a stable proxy for the rigid
                    # body whose overlap the official Toggle state checks.
                    controller_target_pose = [
                        float(pose[index])
                        + float(current_eef_from_finger_offset[index])
                        for index in range(3)
                    ]
                    finger_distance = float(
                        np.linalg.norm(
                            np.asarray(pose[:3], dtype=np.float64)
                            - np.asarray(finger_pose[:3], dtype=np.float64)
                        )
                    )
                    selected_finger_link_name = str(
                        finger_link.get("name")
                        or finger_link.get("body_name")
                        or finger_link.get("prim_path")
                        or ""
                    )
                    tracking_link = f"eef_calibrated_for:{selected_finger_link_name or 'finger_link'}"
                    if phase == "contact":
                        finger_contacts = _behavior1k_public_finger_contacts(
                            primary_obj, selected_finger_link
                        )
                        if finger_contacts:
                            contact_observed = True
                            contact_finger_pose = [
                                float(value) for value in finger_pose[:3]
                            ]
                            contact_press_normal, contact_press_normal_source = (
                                _behavior1k_contact_press_normal(
                                    finger_contacts,
                                    contact_surface_normal or surface_normal,
                                )
                            )
                            plan["contact_finger_pose"] = list(contact_finger_pose)
                            plan["contact_press_normal"] = list(contact_press_normal)
                            plan["contact_press_normal_source"] = (
                                contact_press_normal_source
                            )
                            plan["contact_press_depth_m"] = 0.006
                            contact_consecutive_steps = 1
                            max_contact_consecutive_steps = max(
                                max_contact_consecutive_steps, 1
                            )
                            reached_phase = True
                            trace.append(
                                {
                                    "phase": "contact_detected",
                                    "iteration": iteration,
                                    "tracking_link": tracking_link,
                                    "contact_offset_m": contact_offset_m,
                                    "finger_contacts": finger_contacts,
                                    "step_count": total_steps,
                                }
                            )
                            break
                command, distance = _behavior1k_arm_delta_command(
                    robot_pose,
                    tracking_pose,
                    controller_target_pose,
                    position_scale=arm_position_scale,
                )
                fine_gain = 1.0
                if phase == "contact":
                    fine_gain = 0.20
                elif distance <= 0.25:
                    fine_gain = 0.25
                if fine_gain < 1.0:
                    command[:3] = [fine_gain * float(value) for value in command[:3]]
                if distance <= tolerance and phase == "approach":
                    reached_phase = True
                    break
                step = self._primitive_execute_behavior1k_controller_command(
                    robot_name=selected_robot_name,
                    commands={arm_name: command, gripper_name: gripper_command},
                    repeat=1,
                    query="public interaction-link end-effector pose control",
                    evidence_handles=validated_handles,
                    suspend_deferred_visual_modalities=True,
                    include_post_step_visual_evidence=False,
                    agent_context={
                        **agent_context,
                        "semantic_action": semantic_action,
                        "phase": phase,
                        "target_pose": list(pose[:3]),
                        "target_source": "native_object_state_link",
                    },
                )
                total_steps += int(step.output.get("step_count", 0))
                if not step.ok:
                    error = step.error or f"physical_toggle_{phase}_controller_failed"
                    break
                terminated = bool(step.output.get("terminated"))
                truncated = bool(step.output.get("truncated"))
                next_control_state = step.output.get("control_state")
                robot_state = (
                    next_control_state.get(selected_robot_name)
                    if isinstance(next_control_state, dict)
                    else None
                )
                next_eef_pose: list[float] | None = None
                next_finger_pose: list[float] | None = None
                if isinstance(robot_state, dict):
                    next_end_effectors = robot_state.get("end_effectors")
                    next_eef = (
                        next_end_effectors.get(logical_arm_name)
                        if isinstance(next_end_effectors, dict)
                        else None
                    )
                    if isinstance(next_eef, dict) and isinstance(
                        next_eef.get("pose_world"), list
                    ):
                        next_eef_pose = list(next_eef["pose_world"])
                    next_finger = _behavior1k_closest_finger_link(
                        robot_state,
                        logical_arm_name,
                        pose,
                        preferred_name=selected_finger_link_name,
                    )
                    if isinstance(next_finger, dict) and isinstance(
                        next_finger.get("pose_world"), list
                    ):
                        selected_finger_link = next_finger
                        next_finger_pose = list(next_finger["pose_world"])
                finger_contacts = []
                if phase == "contact" and selected_finger_link is not None:
                    finger_contacts = _behavior1k_public_finger_contacts(
                        primary_obj, selected_finger_link
                    )
                    if finger_contacts:
                        contact_observed = True
                        if next_finger_pose is not None:
                            contact_finger_pose = [
                                float(value) for value in next_finger_pose[:3]
                            ]
                        contact_press_normal, contact_press_normal_source = (
                            _behavior1k_contact_press_normal(
                                finger_contacts,
                                contact_surface_normal or surface_normal,
                            )
                        )
                        if contact_finger_pose is not None:
                            plan["contact_finger_pose"] = list(contact_finger_pose)
                        plan["contact_press_normal"] = list(contact_press_normal)
                        plan["contact_press_normal_source"] = (
                            contact_press_normal_source
                        )
                        plan["contact_press_depth_m"] = 0.006
                        contact_consecutive_steps = 1
                        max_contact_consecutive_steps = max(
                            max_contact_consecutive_steps, 1
                        )
                        reached_phase = True
                if (
                    phase == "contact"
                    or len(trace) < 24
                    or iteration % 10 == 0
                    or terminated
                    or truncated
                ):
                    trace.append(
                        {
                            "phase": phase,
                            "iteration": iteration,
                            "tracking_link": tracking_link,
                            "live_target_pose": list(target_pose),
                            "distance": distance,
                            "finger_distance": finger_distance,
                            "controller_target_pose": controller_target_pose,
                            "tracking_pose_before": list(tracking_pose),
                            "eef_pose_after": next_eef_pose,
                            "finger_pose_after": next_finger_pose,
                            "command": command,
                            "fine_gain": fine_gain,
                            "contact_offset_m": contact_offset_m,
                            "finger_contacts": finger_contacts,
                            "step_count": total_steps,
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                    )
                if contact_observed:
                    trace.append(
                        {
                            "phase": "contact_detected",
                            "iteration": iteration,
                            "tracking_link": tracking_link,
                            "contact_offset_m": contact_offset_m,
                            "finger_contacts": finger_contacts,
                            "step_count": total_steps,
                        }
                    )
                    break
                if terminated or truncated:
                    reached_phase = True
                    break
            if error is None and not reached_phase:
                if phase == "contact":
                    # Marker overlap is an official simulator condition, not
                    # a point-to-point kinematic invariant.  Continue into the
                    # fixed closed-loop press window even when the finger-link
                    # origin cannot numerically settle inside the marker.
                    trace.append(
                        {
                            "phase": "contact_transition",
                            "target_reached": False,
                            "reason": "continue_to_official_contact_hold",
                            "step_count": total_steps,
                        }
                    )
                else:
                    error = f"physical_toggle_{phase}_target_not_reached"
            if error is not None or terminated or truncated:
                break
            if phase == "approach":
                for close_index in range(min(8, max(0, max_steps - total_steps))):
                    live = refresh_live_target()
                    if live is None or not isinstance(robot_state, dict):
                        error = "physical_toggle_precontact_close_state_unavailable"
                        break
                    robot_pose = robot_state.get("pose_world")
                    end_effectors = robot_state.get("end_effectors")
                    eef = (
                        end_effectors.get(logical_arm_name)
                        if isinstance(end_effectors, dict)
                        else None
                    )
                    if not isinstance(eef, dict) and isinstance(end_effectors, dict):
                        eef = next(
                            (
                                value
                                for value in end_effectors.values()
                                if isinstance(value, dict)
                            ),
                            None,
                        )
                    eef_pose = eef.get("pose_world") if isinstance(eef, dict) else None
                    if not isinstance(robot_pose, list) or not isinstance(
                        eef_pose, list
                    ):
                        error = "physical_toggle_precontact_close_pose_unavailable"
                        break
                    close_target = [
                        float(target_pose[index]) - 0.08 * float(surface_normal[index])
                        for index in range(3)
                    ]
                    close_controller_target = list(close_target)
                    close_tracking_link = "eef"
                    close_finger_distance: float | None = None
                    close_finger = _behavior1k_closest_finger_link(
                        robot_state,
                        logical_arm_name,
                        close_target,
                        preferred_name=selected_finger_link_name,
                    )
                    if isinstance(close_finger, dict) and isinstance(
                        close_finger.get("pose_world"), list
                    ):
                        selected_finger_link = close_finger
                        close_finger_pose = close_finger["pose_world"]
                        close_offset = [
                            float(eef_pose[index]) - float(close_finger_pose[index])
                            for index in range(3)
                        ]
                        close_controller_target = [
                            float(close_target[index]) + float(close_offset[index])
                            for index in range(3)
                        ]
                        close_finger_distance = float(
                            np.linalg.norm(
                                np.asarray(close_target[:3], dtype=np.float64)
                                - np.asarray(close_finger_pose[:3], dtype=np.float64)
                            )
                        )
                        close_tracking_link = "eef_calibrated_for_closed_finger"
                    close_command, close_distance = _behavior1k_arm_delta_command(
                        robot_pose,
                        eef_pose,
                        close_controller_target,
                        position_scale=arm_position_scale,
                    )
                    close_command[:3] = [
                        0.30 * float(value) for value in close_command[:3]
                    ]
                    close = self._primitive_execute_behavior1k_controller_command(
                        robot_name=selected_robot_name,
                        commands={arm_name: close_command, gripper_name: [-1.0]},
                        repeat=1,
                        query="close native fingers while maintaining public precontact pose",
                        evidence_handles=validated_handles,
                        suspend_deferred_visual_modalities=True,
                        include_post_step_visual_evidence=False,
                        agent_context={
                            **agent_context,
                            "semantic_action": semantic_action,
                            "phase": "close_precontact",
                            "close_index": close_index,
                        },
                    )
                    total_steps += int(close.output.get("step_count", 0))
                    trace.append(
                        {
                            "phase": "close_precontact",
                            "iteration": close_index,
                            "ok": close.ok,
                            "distance": close_distance,
                            "finger_distance": close_finger_distance,
                            "tracking_link": close_tracking_link,
                            "controller_target_pose": close_controller_target,
                            "command": close_command,
                            "step_count": total_steps,
                        }
                    )
                    if not close.ok:
                        error = close.error or "physical_toggle_precontact_close_failed"
                        break
                    terminated = bool(close.output.get("terminated"))
                    truncated = bool(close.output.get("truncated"))
                    next_control_state = close.output.get("control_state")
                    robot_state = (
                        next_control_state.get(selected_robot_name)
                        if isinstance(next_control_state, dict)
                        else None
                    )
                    if terminated or truncated:
                        break
                if error is not None or terminated or truncated:
                    break

        if (
            error is None
            and not terminated
            and not truncated
            and total_steps < max_steps
        ):
            # ToggledOn requires five consecutive native steps of finger
            # contact/marker overlap. Hold the controller at contact instead
            # of chasing the now-movable marker; if contact is briefly lost,
            # use only a damped re-contact command toward the frozen anchor.
            for hold_index in range(min(16, max_steps - total_steps)):
                live = refresh_live_target()
                if live is None:
                    error = "physical_toggle_hold_live_target_unavailable"
                    break
                if not isinstance(robot_state, dict):
                    error = "physical_toggle_hold_control_state_unavailable"
                    break
                robot_pose = robot_state.get("pose_world")
                end_effectors = robot_state.get("end_effectors")
                eef = (
                    end_effectors.get(logical_arm_name)
                    if isinstance(end_effectors, dict)
                    else None
                )
                if not isinstance(eef, dict) and isinstance(end_effectors, dict):
                    eef = next(
                        (
                            value
                            for value in end_effectors.values()
                            if isinstance(value, dict)
                        ),
                        None,
                    )
                eef_pose = eef.get("pose_world") if isinstance(eef, dict) else None
                if not isinstance(robot_pose, list) or not isinstance(eef_pose, list):
                    error = "physical_toggle_hold_robot_or_eef_pose_unavailable"
                    break
                tracking_pose = eef_pose
                tracking_link = "eef"
                hold_target_pose = list(contact_anchor_pose or target_pose[:3])
                hold_finger_pose: list[float] | None = None
                hold_eef_from_finger_offset: list[float] | None = None
                finger_link = _behavior1k_closest_finger_link(
                    robot_state,
                    logical_arm_name,
                    hold_target_pose,
                    preferred_name=selected_finger_link_name,
                )
                if finger_link is not None and isinstance(
                    finger_link.get("pose_world"), list
                ):
                    selected_finger_link = finger_link
                    hold_finger_pose = [
                        float(value) for value in finger_link["pose_world"]
                    ]
                    hold_eef_from_finger_offset = [
                        float(eef_pose[index]) - float(hold_finger_pose[index])
                        for index in range(3)
                    ]
                    selected_finger_link_name = str(
                        finger_link.get("name")
                        or finger_link.get("body_name")
                        or finger_link.get("prim_path")
                        or ""
                    )
                    tracking_link = f"eef_calibrated_for:{selected_finger_link_name or 'finger_link'}"
                contacts_before = _behavior1k_public_finger_contacts(
                    primary_obj, selected_finger_link
                )
                hold_controller_target_pose = list(hold_target_pose)
                if (
                    contact_finger_pose is not None
                    and contact_press_normal is not None
                    and hold_finger_pose is not None
                ):
                    # The official toggle overlap is evaluated on a native
                    # finger rigid body. Preserve that first-contact frame and
                    # recalibrate the live EEF-to-finger offset every step so
                    # wrist drift cannot pull the collider off the button.
                    hold_controller_target_pose = _behavior1k_contact_hold_target(
                        contact_finger_pose=contact_finger_pose,
                        current_eef_pose=eef_pose,
                        current_finger_pose=hold_finger_pose,
                        press_normal=contact_press_normal,
                        press_depth=0.006,
                    )
                elif contact_eef_from_finger_offset is not None:
                    # No official finger contact was observed during the
                    # bounded press phase. Keep using the current native
                    # finger frame rather than the stale offset captured when
                    # the wrist had a different orientation.
                    current_offset = (
                        hold_eef_from_finger_offset or contact_eef_from_finger_offset
                    )
                    hold_controller_target_pose = [
                        float(hold_target_pose[index]) + float(current_offset[index])
                        for index in range(3)
                    ]
                command, distance = _behavior1k_arm_delta_command(
                    robot_pose,
                    tracking_pose,
                    hold_controller_target_pose,
                    position_scale=arm_position_scale,
                )
                hold_gain = 0.50 if contact_finger_pose is not None else 0.15
                command[:3] = [hold_gain * float(value) for value in command[:3]]
                hold = self._primitive_execute_behavior1k_controller_command(
                    robot_name=selected_robot_name,
                    commands={arm_name: command, gripper_name: [-1.0]},
                    repeat=1,
                    query="actively maintain native finger contact with public interaction link",
                    evidence_handles=validated_handles,
                    suspend_deferred_visual_modalities=True,
                    include_post_step_visual_evidence=False,
                    agent_context={
                        **agent_context,
                        "semantic_action": semantic_action,
                        "phase": "contact_hold",
                        "hold_index": hold_index,
                    },
                )
                total_steps += int(hold.output.get("step_count", 0))
                trace.append(
                    {
                        "phase": "contact_hold",
                        "iteration": hold_index,
                        "tracking_link": tracking_link,
                        "live_target_pose": list(target_pose),
                        "distance": distance,
                        "controller_target_pose": hold_controller_target_pose,
                        "finger_pose_before": hold_finger_pose,
                        "finger_distance_before": (
                            float(
                                np.linalg.norm(
                                    np.asarray(hold_target_pose[:3], dtype=np.float64)
                                    - np.asarray(hold_finger_pose[:3], dtype=np.float64)
                                )
                            )
                            if hold_finger_pose is not None
                            else None
                        ),
                        "command": command,
                        "ok": hold.ok,
                        "step_count": total_steps,
                    }
                )
                if not hold.ok:
                    error = hold.error or "physical_toggle_contact_hold_failed"
                    break
                terminated = bool(hold.output.get("terminated"))
                truncated = bool(hold.output.get("truncated"))
                next_control_state = hold.output.get("control_state")
                robot_state = (
                    next_control_state.get(selected_robot_name)
                    if isinstance(next_control_state, dict)
                    else None
                )
                contacts_after = _behavior1k_public_finger_contacts(
                    primary_obj, selected_finger_link
                )
                if contacts_after:
                    contact_observed = True
                    contact_consecutive_steps += 1
                    max_contact_consecutive_steps = max(
                        max_contact_consecutive_steps, contact_consecutive_steps
                    )
                else:
                    contact_consecutive_steps = 0
                trace[-1]["contacts_before"] = contacts_before
                trace[-1]["contacts_after"] = contacts_after
                trace[-1]["contact_consecutive_steps"] = contact_consecutive_steps
                if contact_consecutive_steps >= 5:
                    contact_hold_completed = True
                    trace.append(
                        {
                            "phase": "contact_hold_complete",
                            "contact_consecutive_steps": contact_consecutive_steps,
                            "step_count": total_steps,
                        }
                    )
                    break
                if terminated or truncated:
                    break

        if (
            error is None
            and not terminated
            and not truncated
            and contact_hold_completed
            and contact_press_normal is not None
            and total_steps < max_steps
        ):
            retract_target_pose: list[float] | None = None
            for retract_index in range(min(4, max_steps - total_steps)):
                if not isinstance(robot_state, dict):
                    error = "physical_toggle_retract_control_state_unavailable"
                    break
                robot_pose = robot_state.get("pose_world")
                end_effectors = robot_state.get("end_effectors")
                eef = (
                    end_effectors.get(logical_arm_name)
                    if isinstance(end_effectors, dict)
                    else None
                )
                if not isinstance(eef, dict) and isinstance(end_effectors, dict):
                    eef = next(
                        (
                            value
                            for value in end_effectors.values()
                            if isinstance(value, dict)
                        ),
                        None,
                    )
                eef_pose = eef.get("pose_world") if isinstance(eef, dict) else None
                if not isinstance(robot_pose, list) or not isinstance(eef_pose, list):
                    error = "physical_toggle_retract_robot_or_eef_pose_unavailable"
                    break
                if retract_target_pose is None:
                    retract_target_pose = [
                        float(eef_pose[index])
                        - 0.03 * float(contact_press_normal[index])
                        for index in range(3)
                    ]
                    plan["contact_retract_distance_m"] = 0.03
                    plan["contact_retract_steps"] = 4
                command, distance = _behavior1k_arm_delta_command(
                    robot_pose,
                    eef_pose,
                    retract_target_pose,
                    position_scale=arm_position_scale,
                )
                command[:3] = [0.50 * float(value) for value in command[:3]]
                retract = self._primitive_execute_behavior1k_controller_command(
                    robot_name=selected_robot_name,
                    commands={arm_name: command, gripper_name: [-1.0]},
                    repeat=1,
                    query="retract from public interaction link along contact normal",
                    evidence_handles=validated_handles,
                    suspend_deferred_visual_modalities=True,
                    include_post_step_visual_evidence=False,
                    agent_context={
                        **agent_context,
                        "semantic_action": semantic_action,
                        "phase": "contact_retract",
                        "retract_index": retract_index,
                    },
                )
                total_steps += int(retract.output.get("step_count", 0))
                trace.append(
                    {
                        "phase": "contact_retract",
                        "iteration": retract_index,
                        "distance": distance,
                        "controller_target_pose": list(retract_target_pose),
                        "command": command,
                        "ok": retract.ok,
                        "step_count": total_steps,
                    }
                )
                if not retract.ok:
                    error = retract.error or "physical_toggle_contact_retract_failed"
                    break
                terminated = bool(retract.output.get("terminated"))
                truncated = bool(retract.output.get("truncated"))
                next_control_state = retract.output.get("control_state")
                robot_state = (
                    next_control_state.get(selected_robot_name)
                    if isinstance(next_control_state, dict)
                    else None
                )
                if terminated or truncated:
                    break

        completed = error is None
        self._refresh_asset_registry()
        object_states_after = _behavior1k_public_object_states(primary_obj)
        output: JsonDict = {
            "stepped": total_steps > 0,
            "step_count": total_steps,
            "completed": completed,
            "terminated": terminated,
            "truncated": truncated,
            "semantic_action": semantic_action,
            "execution_mode": "physical_native_controller_toggle",
            "available_semantic_actions": _behavior1k_public_semantic_action_names(),
            "selected_robot_name": selected_robot_name,
            "selected_assets": selected_assets,
            "object_states_before": {"primary": object_states_before, "secondary": {}},
            "object_states_after": {"primary": object_states_after, "secondary": {}},
            "trace": trace,
            "physical_toggle_plan": {
                **deepcopy(plan),
                "selected_finger_link_name": selected_finger_link_name,
                "contact_observed": contact_observed,
                "max_contact_consecutive_steps": max_contact_consecutive_steps,
            },
            "navigation": navigation.to_dict(),
            "navigation_replan": navigation_replan,
            "action_schema": behavior1k_action_schema(self._env),
            "observation_evidence": summarize_data(self._last_obs),
            "visual_runtime": _behavior1k_visual_runtime_with_asset_poses(
                self._last_obs, self._objects
            ),
            "visual_evidence": (
                behavior1k_visual_evidence(
                    self._last_obs,
                    assets=self._objects,
                    observation_info=self._last_info,
                    prompt=None,
                    query=asset_name,
                    agent_context=agent_context,
                )
                if include_post_step_visual_evidence
                else {}
            ),
            "evidence_handles": validated_handles,
            "evidence_provenance": provenance,
            "agent_context": agent_context,
        }
        return PrimitiveResult(
            name="run_behavior1k_semantic_action",
            ok=completed,
            output=output,
            error=error,
            artifacts=validated_handles,
        )

    def _step_behavior1k_action(
        self,
        *,
        primitive_name: str,
        action: Any | None = None,
        prompt: str | None = None,
        query: str | None = None,
        evidence_handle: str | None = None,
        evidence_handles: list[str] | None = None,
        suspend_deferred_visual_modalities: bool = False,
        include_post_step_visual_evidence: bool = True,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if self._env is None or not hasattr(self._env, "step"):
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "stepped": False,
                    "requires_live_env": True,
                    "blocker": "live_env_step_unavailable",
                    "action_schema": behavior1k_action_schema(self._env),
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                },
                error="live_env_step_unavailable",
            )
        if action is None:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "stepped": False,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error="agent_action_required",
            )
        validated_handles, provenance, evidence_error = (
            self._behavior1k_action_evidence_provenance(
                evidence_handle=evidence_handle,
                evidence_handles=evidence_handles,
            )
        )
        if evidence_error is not None:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "stepped": False,
                    "evidence_handles": validated_handles,
                    "action_schema": behavior1k_action_schema(self._env),
                },
                error=evidence_error,
            )
        schema = behavior1k_action_schema(self._env)
        try:
            step_action, action_source = _behavior1k_step_action(
                action, schema=schema, env=self._env
            )
            self._emit_probe_progress(
                "behavior1k_action_step_start",
                primitive_name=primitive_name,
                action_source=action_source,
                suspend_deferred_visual_modalities=suspend_deferred_visual_modalities,
                include_post_step_visual_evidence=include_post_step_visual_evidence,
            )
            if suspend_deferred_visual_modalities:
                self._suspend_behavior1k_deferred_visual_modalities()
            self._emit_probe_progress(
                "behavior1k_env_step_start", primitive_name=primitive_name
            )
            step_result = self._env.step(step_action)
            self._emit_probe_progress(
                "behavior1k_env_step_done", primitive_name=primitive_name
            )
            obs, _reward, terminated, truncated, info = _split_step_result(step_result)
        except Exception as exc:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "stepped": False,
                    "action_schema": schema,
                    "prompt": prompt,
                    "query": query,
                    "evidence_handles": validated_handles,
                    "evidence_provenance": provenance,
                    "agent_context": agent_context or {},
                },
                error=f"{type(exc).__name__}: {exc}",
                artifacts=validated_handles,
            )
        self._last_obs = obs
        self._last_info = info
        self._emit_probe_progress(
            "behavior1k_asset_refresh_start", primitive_name=primitive_name
        )
        self._refresh_asset_registry()
        self._emit_probe_progress(
            "behavior1k_asset_refresh_done",
            primitive_name=primitive_name,
            asset_count=len(self._objects),
        )
        visual_runtime = _behavior1k_visual_runtime_with_asset_poses(
            self._last_obs, self._objects
        )
        self._emit_probe_progress(
            "behavior1k_post_step_visual_evidence_start",
            primitive_name=primitive_name,
            include_post_step_visual_evidence=include_post_step_visual_evidence,
        )
        visual_evidence = (
            behavior1k_visual_evidence(
                self._last_obs,
                assets=self._objects,
                observation_info=self._last_info,
                prompt=prompt,
                query=query,
                agent_context=agent_context or {},
            )
            if include_post_step_visual_evidence
            else {
                "available": bool(visual_runtime.get("available")),
                "post_step_visual_evidence_omitted": True,
                "reason": "agent_requested_lightweight_action_step",
            }
        )
        self._emit_probe_progress(
            "behavior1k_post_step_visual_evidence_done",
            primitive_name=primitive_name,
            include_post_step_visual_evidence=include_post_step_visual_evidence,
        )
        return PrimitiveResult(
            name=primitive_name,
            ok=True,
            output={
                "stepped": True,
                "action_source": action_source,
                "submitted_action_summary": summarize_data(step_action),
                "terminated": terminated,
                "truncated": truncated,
                "info_summary": summarize_data(_public_behavior1k_info(info)),
                "action_schema": behavior1k_action_schema(self._env),
                "prompt": prompt,
                "query": query,
                "evidence_handles": validated_handles,
                "evidence_provenance": provenance,
                "agent_context": agent_context or {},
                "language_task": self._language_task(
                    prompt=prompt, query=query, agent_context=agent_context or {}
                ),
                "observation_evidence": summarize_data(self._last_obs),
                "visual_runtime": visual_runtime,
                "visual_evidence": visual_evidence,
                "asset_evidence": deepcopy(self._objects),
            },
            artifacts=validated_handles,
        )

    def _settle_behavior1k_env(self, steps: int) -> list[JsonDict]:
        records: list[JsonDict] = []
        if steps <= 0 or self._env is None or not hasattr(self._env, "step"):
            return records
        schema = behavior1k_action_schema(self._env)
        for index in range(steps):
            try:
                step_action, action_source = _behavior1k_step_action(
                    None, schema=schema, env=self._env
                )
                step_result = self._env.step(step_action)
                obs, _reward, terminated, truncated, info = _split_step_result(
                    step_result
                )
            except Exception as exc:
                records.append(
                    {
                        "step_index": index,
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                break
            self._last_obs = obs
            self._last_info = info
            records.append(
                {
                    "step_index": index,
                    "ok": True,
                    "action_source": action_source,
                    "terminated": _to_builtin(terminated),
                    "truncated": _to_builtin(truncated),
                }
            )
            if terminated or truncated:
                break
        return records

    def _language_task(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> JsonDict:
        return {
            "instruction": self._task_spec.instruction
            if self._task_spec is not None
            else "",
            "task_name": self.config.task_name,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
        }

    def _behavior1k_action_evidence_provenance(
        self,
        *,
        evidence_handle: str | None,
        evidence_handles: list[str] | None,
    ) -> tuple[list[str], list[JsonDict], str | None]:
        requested: list[str] = []
        if evidence_handle is not None:
            if not isinstance(evidence_handle, str) or not evidence_handle:
                return [], [], "evidence_handle_must_be_nonempty_str"
            requested.append(evidence_handle)
        if evidence_handles is not None:
            if not isinstance(evidence_handles, list) or not all(
                isinstance(handle, str) and bool(handle) for handle in evidence_handles
            ):
                return requested, [], "evidence_handles_must_be_list_of_nonempty_str"
            requested.extend(evidence_handles)
        requested = list(dict.fromkeys(requested))
        missing = [
            handle for handle in requested if handle not in self.get_trace().artifacts
        ]
        if missing:
            return requested, [], "evidence_handle_not_found"
        provenance = []
        for handle in requested:
            artifact = self.get_trace().artifacts[handle]
            provenance.append(
                {
                    "evidence_handle": handle,
                    "source": artifact.get("source")
                    if isinstance(artifact, dict)
                    else None,
                    "artifact_fields": sorted(artifact)
                    if isinstance(artifact, dict)
                    else [],
                    "same_episode_trace": True,
                }
            )
        return requested, provenance, None

    def _require_reset(self) -> None:
        if self._task_spec is None or self._trace is None:
            raise RuntimeError("Call reset() before using the backend.")

    def _emit_probe_progress(self, event: str, **extra: Any) -> None:
        if self._probe_progress is None:
            return
        self._probe_progress(event, **extra)


def _shutdown_loaded_omnigibson() -> None:
    """Close an already running native app without importing/starting one."""
    og = sys.modules.get("omnigibson")
    app = getattr(og, "app", None)
    if app is not None and not getattr(app, "_exiting", False):
        og.shutdown()


def _normalize_behavior1k_bootstrap_visual_mode(raw: str | None) -> str:
    normalized = str(raw or "").strip().lower().replace("-", "_")
    if normalized in {
        "none",
        "off",
        "0",
        "false",
        "no",
        "no_visual",
        "novision",
        "proprio",
        "proprio_only",
    }:
        return "none"
    if normalized in {"native", "full", "all", "no_defer", "nodefer"}:
        return "native"
    return "rgb"


def _behavior1k_bootstrap_visual_mode(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    return _normalize_behavior1k_bootstrap_visual_mode(
        env.get(BEHAVIOR1K_BOOTSTRAP_VISUAL_MODE_ENV)
    )


def _defer_behavior1k_visual_modalities(
    env_config: JsonDict,
    *,
    bootstrap_visual_mode: str | None = None,
) -> tuple[JsonDict, list[JsonDict]]:
    """Bootstrap BEHAVIOR visual sensors with a selectable diagnostic strategy.

    OmniGibson's own learning wrappers use this order because Replicator depth and
    segmentation annotators are substantially more reliable once the simulator,
    scene, render product, and base RGB camera have initialized.

    The default ``rgb`` mode preserves the strict native-vision path: RGB is
    created during ``og.Environment`` bootstrap and depth/segmentation are added
    later. ``none`` is a diagnostic-only crash isolation mode that removes all
    visual modalities from bootstrap; it never grants strict readiness credit.
    """

    mode = (
        _behavior1k_bootstrap_visual_mode()
        if bootstrap_visual_mode is None
        else _normalize_behavior1k_bootstrap_visual_mode(bootstrap_visual_mode)
    )
    bootstrap_config = deepcopy(env_config)
    deferred: list[JsonDict] = []
    if mode == "native":
        return bootstrap_config, deferred
    robots = bootstrap_config.get("robots")
    if not isinstance(robots, list):
        return bootstrap_config, deferred
    for robot_index, robot_config in enumerate(robots):
        if not isinstance(robot_config, dict):
            continue
        modalities = robot_config.get("obs_modalities")
        if not isinstance(modalities, (list, tuple, set)):
            continue
        requested = [str(modality) for modality in modalities]
        if mode == "none":
            delayed = [
                modality
                for modality in requested
                if modality in DEFERRED_VISION_MODALITIES
            ]
            bootstrap = [
                modality
                for modality in requested
                if modality not in DEFERRED_VISION_MODALITIES
            ]
        else:
            delayed = [
                modality
                for modality in requested
                if modality in DEFERRED_VISION_MODALITIES and modality != "rgb"
            ]
            bootstrap = [
                modality
                for modality in requested
                if modality not in DEFERRED_VISION_MODALITIES or modality == "rgb"
            ]
        if not delayed:
            continue
        if mode == "rgb" and "rgb" not in bootstrap:
            bootstrap.append("rgb")
        if "proprio" in requested and "proprio" not in bootstrap:
            bootstrap.append("proprio")
        robot_config["obs_modalities"] = bootstrap
        deferred.append({"robot_index": robot_index, "modalities": delayed})
    return bootstrap_config, deferred


def _attach_behavior1k_deferred_visual_modalities(
    env: Any, deferred: list[JsonDict]
) -> None:
    if not deferred:
        return
    unwrapped = _unwrap_env(env)
    robots = getattr(unwrapped, "robots", None)
    if not isinstance(robots, (list, tuple)):
        raise RuntimeError(
            "OmniGibson environment did not expose robots for deferred visual setup"
        )
    for request in deferred:
        robot_index = int(request["robot_index"])
        if robot_index < 0 or robot_index >= len(robots):
            raise RuntimeError(
                f"Deferred visual setup referenced unavailable robot index {robot_index}"
            )
        robot = robots[robot_index]
        sensors = getattr(robot, "sensors", None)
        sensor_values = sensors.values() if isinstance(sensors, dict) else sensors
        if not sensor_values:
            raise RuntimeError(
                f"Robot {robot_index} exposed no sensors for deferred visual setup"
            )
        requested = [str(modality) for modality in request.get("modalities", [])]
        requested.sort(key=_behavior1k_deferred_modality_order)
        attached = {modality: False for modality in requested}
        for sensor in sensor_values:
            supported = getattr(sensor, "all_modalities", ())
            for modality in requested:
                if modality in supported:
                    sensor.add_modality(modality)
                    attached[modality] = True
        missing = sorted(modality for modality, ok in attached.items() if not ok)
        if missing:
            raise RuntimeError(
                f"Robot {robot_index} has no native sensor supporting deferred modalities {missing}"
            )
    if hasattr(unwrapped, "load_observation_space"):
        unwrapped.load_observation_space()


def _remove_behavior1k_deferred_visual_modalities(
    env: Any, deferred: list[JsonDict]
) -> None:
    if not deferred:
        return
    unwrapped = _unwrap_env(env)
    robots = getattr(unwrapped, "robots", None)
    if not isinstance(robots, (list, tuple)):
        raise RuntimeError(
            "OmniGibson environment did not expose robots for deferred visual teardown"
        )
    for request in deferred:
        robot_index = int(request["robot_index"])
        if robot_index < 0 or robot_index >= len(robots):
            raise RuntimeError(
                f"Deferred visual teardown referenced unavailable robot index {robot_index}"
            )
        robot = robots[robot_index]
        sensors = getattr(robot, "sensors", None)
        sensor_values = sensors.values() if isinstance(sensors, dict) else sensors
        if not sensor_values:
            continue
        requested = [str(modality) for modality in request.get("modalities", [])]
        requested.sort(key=_behavior1k_deferred_modality_order, reverse=True)
        for sensor in sensor_values:
            active = getattr(sensor, "modalities", ())
            for modality in requested:
                if modality in active and hasattr(sensor, "remove_modality"):
                    sensor.remove_modality(modality)
    if hasattr(unwrapped, "load_observation_space"):
        unwrapped.load_observation_space()


def _behavior1k_deferred_modality_order(modality: str) -> tuple[int, str]:
    """Respect the dependency order used by OmniGibson's official wrapper."""

    if modality == "rgb":
        return 0, modality
    if modality in {"depth", "depth_linear"}:
        return 1, modality
    if modality == "seg_semantic":
        return 2, modality
    if modality in {"seg_instance", "seg_instance_id"}:
        return 3, modality
    return 4, modality


def behavior1k_preflight_summary(live_requested: bool = True) -> JsonDict:
    source_paths = _configure_behavior1k_source_paths()
    minimal_kit = _minimal_kit_no_flowusd_path()
    isaac_path = os.environ.get("ISAAC_PATH")
    exp_path = os.environ.get("EXP_PATH")
    carb_app_path = os.environ.get("CARB_APP_PATH")
    omnigibson_dataset_path = os.environ.get("OMNIGIBSON_DATASET_PATH")
    omnigibson_data_path = os.environ.get("OMNIGIBSON_DATA_PATH")
    omnigibson_asset_path = os.environ.get("OMNIGIBSON_ASSET_PATH")
    omnigibson_key_path = os.environ.get("OMNIGIBSON_KEY_PATH")
    effective_dataset_path = omnigibson_dataset_path or omnigibson_data_path
    effective_asset_path = omnigibson_asset_path
    if effective_dataset_path and not effective_asset_path:
        default_asset_path = _default_omnigibson_asset_root(
            Path(effective_dataset_path).expanduser()
        )
        effective_asset_path = (
            str(default_asset_path) if default_asset_path.exists() else None
        )
    effective_key_path = omnigibson_key_path
    if not effective_key_path:
        default_key_path = _default_omnigibson_key_path()
        effective_key_path = (
            str(default_key_path) if default_key_path.exists() else None
        )
    isaac_version_file = Path(isaac_path) / "VERSION" if isaac_path else None
    isaacsim_spec_available = (
        _find_module_spec("isaacsim", bootstrap_behavior1k=False) is not None
    )
    omnigibson_importable, _omnigibson_module, omnigibson_import_error = (
        _module_imports("omnigibson", bootstrap_behavior1k=False)
    )
    bddl_importable, _bddl_module, bddl_import_error = _module_imports(
        "bddl", bootstrap_behavior1k=False
    )
    return {
        "live_requested": live_requested,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "behavior1k_source_paths": source_paths,
        "omnigibson_spec_available": _find_module_spec(
            "omnigibson", bootstrap_behavior1k=False
        )
        is not None,
        "isaacsim_spec_available": isaacsim_spec_available,
        "bddl_spec_available": _find_module_spec("bddl", bootstrap_behavior1k=False)
        is not None,
        "omnigibson_importable": omnigibson_importable,
        "bddl_importable": bddl_importable,
        "omnigibson_import_error": omnigibson_import_error,
        "bddl_import_error": bddl_import_error,
        "isaac_runtime": {
            "isaacsim_spec_available": isaacsim_spec_available,
            "isaac_path_set": bool(isaac_path),
            "exp_path_set": bool(exp_path),
            "carb_app_path_set": bool(carb_app_path),
            "isaac_version_file": str(isaac_version_file)
            if isaac_version_file is not None
            else None,
            "isaac_version_file_exists": isaac_version_file.exists()
            if isaac_version_file is not None
            else False,
            "expected_for_launcher_install": [
                "ISAAC_PATH",
                "EXP_PATH",
                "CARB_APP_PATH",
            ],
            "pip_install_can_set_paths_on_import": isaacsim_spec_available,
        },
        "conda_available": shutil.which("conda") is not None,
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
        "display": bool(os.environ.get("DISPLAY")),
        "omnigibson_data_path_set": bool(os.environ.get("OMNIGIBSON_DATA_PATH")),
        "omnigibson_dataset_path_set": bool(omnigibson_dataset_path),
        "omnigibson_asset_path_set": bool(omnigibson_asset_path),
        "omnigibson_dataset_path": omnigibson_dataset_path,
        "omnigibson_asset_path": omnigibson_asset_path,
        "omnigibson_effective_dataset_path": effective_dataset_path,
        "omnigibson_effective_asset_path": effective_asset_path,
        "omnigibson_key_path_set": bool(omnigibson_key_path),
        "omnigibson_effective_key_path_exists": Path(effective_key_path).exists()
        if effective_key_path
        else False,
        "behavior1k_meta_root_link_patch_enabled": not _env_falsey(
            "BEHAVIOR1K_PATCH_META_ROOT_LINKS"
        ),
        "behavior_dataset_path_set": bool(os.environ.get("BEHAVIOR_DATASET_PATH")),
        "minimal_kit_no_flowusd_requested": _minimal_kit_no_flowusd_enabled(),
        "minimal_kit_no_flowusd_path": str(minimal_kit),
        "minimal_kit_no_flowusd_exists": minimal_kit.exists(),
        "bootstrap_visual_mode_env": BEHAVIOR1K_BOOTSTRAP_VISUAL_MODE_ENV,
        "bootstrap_visual_mode": _behavior1k_bootstrap_visual_mode(),
        "bootstrap_visual_mode_options": ["rgb", "none", "native"],
        "diagnostic_no_visual_bootstrap": _behavior1k_bootstrap_visual_mode() == "none",
    }


def behavior1k_task_readiness(
    *,
    config: Behavior1KRuntimeConfig,
    env: Any,
    obs: Any,
    objects: dict[str, JsonDict],
    info: JsonDict,
) -> JsonDict:
    """Classify live BEHAVIOR evidence without reading task success predicates."""

    visual_runtime = _behavior1k_visual_runtime_with_asset_poses(obs, objects)
    action_schema = behavior1k_action_schema(env)
    scene_config = (
        config.env_config.get("scene") if isinstance(config.env_config, dict) else {}
    )
    scene_type = scene_config.get("type") if isinstance(scene_config, dict) else None
    scene_model = (
        scene_config.get("scene_model") if isinstance(scene_config, dict) else None
    )
    partial_categories = (
        scene_config.get("load_object_categories")
        if isinstance(scene_config, dict)
        else None
    )
    live_env_created = env is not None
    action_boundary_ready = bool(live_env_created and action_schema.get("available"))
    visual_boundary_ready = bool(
        live_env_created
        and visual_runtime.get("rgb_ready")
        and visual_runtime.get("depth_ready")
        and visual_runtime.get("segmentation_ready")
    )
    object_scene_ready = bool(
        visual_boundary_ready
        and scene_type == "InteractiveTraversableScene"
        and scene_model
        and objects
    )
    full_scene_requested = bool(object_scene_ready and not partial_categories)
    bounded_scene_requested = bool(object_scene_ready and partial_categories)
    blockers: list[JsonDict] = []
    if not live_env_created:
        blockers.append(
            {
                "id": "live_env_not_created",
                "detail": "No OmniGibson env is attached to the runtime.",
            }
        )
    if live_env_created and not visual_boundary_ready:
        blockers.append(
            {
                "id": "visual_boundary_incomplete",
                "detail": "RGB, depth, and segmentation evidence are all required for object-scene readiness.",
                "visual_runtime": visual_runtime,
            }
        )
    if visual_boundary_ready and scene_type != "InteractiveTraversableScene":
        blockers.append(
            {
                "id": "object_scene_not_requested",
                "detail": "The current env is a visual/action boundary, not an official InteractiveTraversableScene object-scene.",
            }
        )
    if scene_type == "InteractiveTraversableScene" and not objects:
        blockers.append(
            {
                "id": "object_registry_empty",
                "detail": "No scene object poses were registered after reset.",
            }
        )
    full_task_blockers = list(blockers)
    full_task_blockers.append(
        {
            "id": "official_bddl_evaluator_not_run",
            "detail": "Run the harness-side OmniGibson/BDDL evaluator for task success; no success predicate is exposed as a primitive.",
        }
    )
    if bounded_scene_requested:
        full_task_blockers.append(
            {
                "id": "bounded_partial_scene_not_full_task",
                "detail": "scene.load_object_categories proves a bounded official object-scene subtask, not full BDDL task readiness.",
            }
        )
    return {
        "task_name": config.task_name,
        "live_env_created": live_env_created,
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
            "load_object_categories": partial_categories,
            "full_scene_requested": full_scene_requested,
            "bounded_partial_scene_requested": bounded_scene_requested,
        },
        "visual_boundary": {"ready": visual_boundary_ready, "runtime": visual_runtime},
        "action_boundary": {"ready": action_boundary_ready, "schema": action_schema},
        "object_scene_boundary": {
            "ready": object_scene_ready,
            "official_scene_model": scene_model,
            "asset_count": len(objects),
            "bounded_partial_scene": bounded_scene_requested,
            "full_object_scene": full_scene_requested,
        },
        "full_bddl_task": {
            "ready": False,
            "success_source": "harness_only_official_evaluator",
            "official_evaluator": "python -m omnigibson.eval.eval",
            "oracle_checker_success_demo_replay_primitives_exposed": False,
            "blockers": full_task_blockers,
        },
        "last_info_summary": summarize_data(info),
    }


def run_preflight_probe(
    create_env: bool = False, env_timeout_seconds: int = 900
) -> JsonDict:
    result: JsonDict = {
        "runtime": behavior1k_preflight_summary(live_requested=create_env),
        "imports": {},
        "env_probe": None,
    }
    for name in ("bddl", "omnigibson"):
        ok, module, error = _module_imports(name)
        result["runtime"][f"{name}_importable"] = ok
        if ok:
            result["imports"][name] = {
                "ok": True,
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }
        else:
            result["imports"][name] = {"ok": False, "error": error}
    if create_env:
        result["env_probe"] = run_env_probe_subprocess(
            timeout_seconds=env_timeout_seconds
        )
    return result


def run_env_probe_inline() -> JsonDict:
    backend: Behavior1KAgentRuntimeBackend | None = None
    stage = _behavior1k_probe_stage()
    env_config = _behavior1k_env_config_from_env()
    config = Behavior1KRuntimeConfig(live=True, env_config=env_config)
    _configure_omnigibson_asset_env(config)
    code_path = Path(__file__).resolve()
    code_identity = {
        "code_file": str(code_path),
        "code_mtime": code_path.stat().st_mtime,
    }
    started_at = time.time()

    def progress(event: str, **extra: Any) -> None:
        _write_env_probe_progress(
            {
                "ok": False,
                "probe_stage": stage,
                "progress_event": event,
                "elapsed_seconds": round(time.time() - started_at, 3),
                "env_config": env_config,
                **code_identity,
                **extra,
            }
        )

    progress("child_start")
    try:
        backend = Behavior1KAgentRuntimeBackend(config=config)
        backend._probe_progress = progress
        progress("backend_initialized")
        if stage == "construct":
            progress("construct_make_env_start")
            backend._env = backend._make_env(config, seed=None)
            progress("construct_make_env_done")
            backend._refresh_asset_registry()
            return {
                "ok": True,
                "probe_stage": stage,
                "env_config": env_config,
                "runtime": backend.runtime_available(),
                "asset_evidence": deepcopy(backend._objects),
                **code_identity,
            }
        progress("reset_start")
        task = backend.reset("behavior1k_preflight_empty_scene")
        progress("reset_done", task=task.to_dict())
        if stage == "reset":
            return {
                "ok": True,
                "probe_stage": stage,
                "env_config": env_config,
                "task": task.to_dict(),
                "runtime": backend.runtime_available(),
                **code_identity,
            }
        progress("observe_start", task=task.to_dict())
        obs = backend.observe()
        progress(
            "observe_done",
            task=task.to_dict(),
            observation_summary=summarize_data(obs.to_dict()),
        )
        if stage == "observe":
            return {
                "ok": True,
                "probe_stage": stage,
                "env_config": env_config,
                "task": task.to_dict(),
                "observation": obs.to_dict(),
                "runtime": backend.runtime_available(),
                **code_identity,
            }
        if stage == "visual":
            progress("visual_primitive_start", task=task.to_dict())
            visual = backend.call_primitive(
                "inspect_behavior1k_visual_evidence",
                prompt="preflight inspect native RGB-D and segmentation evidence",
                query="return agent-visible camera, mask, and asset evidence",
                agent_context={"case": "behavior1k_visual_probe"},
            )
            progress("visual_primitive_done", visual_ok=bool(visual.ok))
            return {
                "ok": bool(visual.ok),
                "probe_stage": stage,
                "env_config": env_config,
                "task": task.to_dict(),
                "observation": obs.to_dict(),
                "visual_primitive": visual.to_dict(),
                "runtime": backend.runtime_available(),
                **code_identity,
            }
        progress("action_step_start", task=task.to_dict())
        action = backend.call_primitive(
            "submit_behavior1k_action",
            action=None,
            prompt="preflight one legal default action",
            query="prove the live OmniGibson env.step boundary",
            agent_context={"case": "behavior1k_env_probe"},
        )
        progress("action_step_done", action_ok=bool(action.ok))
        return {
            "ok": bool(action.ok),
            "probe_stage": stage,
            "env_config": env_config,
            "task": task.to_dict(),
            "observation": obs.to_dict(),
            "default_action_step": action.to_dict(),
            "action_step_ok": bool(action.ok),
            **code_identity,
        }
    except BaseException as exc:
        progress("exception", error=f"{type(exc).__name__}: {exc}")
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )[-6000:],
            **code_identity,
        }
    finally:
        if backend is not None:
            backend.close()


def _write_env_probe_progress(payload: JsonDict) -> None:
    sidecar_output = os.getenv("BEHAVIOR1K_ENV_PROBE_OUTPUT")
    if not sidecar_output:
        return
    try:
        output_path = Path(sidecar_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return


def _behavior1k_probe_stage() -> str:
    stage = os.getenv("BEHAVIOR1K_PROBE_STAGE", "action").strip().lower()
    aliases = {
        "env": "construct",
        "create": "construct",
        "vision": "visual",
        "step": "action",
    }
    stage = aliases.get(stage, stage)
    if stage not in {"construct", "reset", "observe", "visual", "action"}:
        raise ValueError(
            f"Unsupported BEHAVIOR1K_PROBE_STAGE={stage!r}; expected construct, reset, observe, visual, or action."
        )
    return stage


def _behavior1k_env_config_from_env() -> JsonDict:
    raw = os.getenv("BEHAVIOR1K_ENV_CONFIG_JSON")
    if not raw:
        return {"scene": {"type": "Scene"}}
    loaded = json.loads(raw)
    if not isinstance(loaded, dict):
        raise ValueError("BEHAVIOR1K_ENV_CONFIG_JSON must decode to a JSON object.")
    return loaded


def _configure_omnigibson_asset_env(config: Behavior1KRuntimeConfig) -> None:
    dataset_root = (
        config.dataset_path
        or os.getenv("BEHAVIOR1K_DATASET_PATH")
        or os.getenv("OMNIGIBSON_DATASET_PATH")
        or os.getenv("OMNIGIBSON_DATA_PATH")
    )
    if not dataset_root:
        default_dataset_root = _default_behavior1k_dataset_root()
        if default_dataset_root.exists():
            dataset_root = str(default_dataset_root)
    if dataset_root:
        source_dataset_root = Path(dataset_root).expanduser()
        og_dataset_root = _ensure_behavior1k_dataset_compat_root(source_dataset_root)
        os.environ["OMNIGIBSON_DATASET_PATH"] = str(og_dataset_root)
        os.environ["OMNIGIBSON_DATA_PATH"] = str(og_dataset_root)
        os.environ.setdefault("BEHAVIOR_DATASET_PATH", str(source_dataset_root))
        asset_root = _default_omnigibson_asset_root(source_dataset_root)
        if asset_root.exists():
            os.environ.setdefault("OMNIGIBSON_ASSET_PATH", str(asset_root))
    key_path = _default_omnigibson_key_path()
    if key_path.exists():
        os.environ.setdefault("OMNIGIBSON_KEY_PATH", str(key_path))


def _configure_omnigibson_headless_viewer(
    config: Behavior1KRuntimeConfig, og_module: Any
) -> JsonDict:
    """Disable only the GUI viewer camera for headless runs.

    Robot VisionSensor observations still render normally; this avoids a
    separate UI viewport dependency that is brittle on batch GPU nodes.
    """

    if not config.headless:
        return {"applied": False, "reason": "not_headless"}
    try:
        gm = getattr(__import__("omnigibson.macros", fromlist=["gm"]), "gm")
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"macro_import_failed:{type(exc).__name__}: {exc}",
        }
    previous = {
        "HEADLESS": getattr(gm, "HEADLESS", None),
        "RENDER_VIEWER_CAMERA": getattr(gm, "RENDER_VIEWER_CAMERA", None),
        "GUI_VIEWPORT_ONLY": getattr(gm, "GUI_VIEWPORT_ONLY", None),
    }
    unlock = getattr(gm, "unlocked", None)
    if callable(unlock):
        with unlock():
            gm.HEADLESS = True
            gm.RENDER_VIEWER_CAMERA = False
            gm.GUI_VIEWPORT_ONLY = False
    else:
        gm.HEADLESS = True
        gm.RENDER_VIEWER_CAMERA = False
        gm.GUI_VIEWPORT_ONLY = False
    setattr(
        og_module,
        "_embodiedai_headless_viewer_config",
        {"applied": True, "previous": previous},
    )
    return {"applied": True, "previous": previous}


def _maybe_patch_isaacsim_headless_viewport_wait(
    config: Behavior1KRuntimeConfig,
) -> JsonDict:
    """Bound IsaacSim's GUI viewport wait on headless batch nodes.

    IsaacSim 4.x waits for an active UI viewport handle during SimulationApp
    construction even when OmniGibson's GUI viewer camera is disabled. On some
    headless GPU nodes that handle never appears, so the app hangs before
    robot VisionSensor observations can initialize. This patch only caps that
    UI wait; it does not synthesize observations or bypass simulator stepping.
    """

    if not config.headless:
        return {"applied": False, "reason": "not_headless"}
    if _env_falsey("BEHAVIOR1K_PATCH_ISAACSIM_VIEWPORT_WAIT"):
        return {"applied": False, "reason": "disabled_by_env"}
    _configure_isaacsim_noninteractive_eula(config)
    import_errors: list[str] = []
    simulation_app_cls = None
    for module_name in (
        "isaacsim",
        "isaacsim.simulation_app",
        "isaacsim.simulation_app.simulation_app",
    ):
        try:
            sim_app_module = __import__(module_name, fromlist=["SimulationApp"])
            simulation_app_cls = getattr(sim_app_module, "SimulationApp", None)
            if simulation_app_cls is not None:
                break
            import_errors.append(f"{module_name}:missing SimulationApp")
        except Exception as exc:
            import_errors.append(f"{module_name}:{type(exc).__name__}: {exc}")
    if simulation_app_cls is None:
        return {
            "applied": False,
            "reason": "import_failed:" + " | ".join(import_errors),
        }
    if getattr(simulation_app_cls, "_behavior1k_headless_viewport_wait_patch", False):
        return {"applied": True, "reason": "already_applied"}

    original_wait = getattr(simulation_app_cls, "_wait_for_viewport", None)
    if not callable(original_wait):
        return {"applied": False, "reason": "missing_wait_for_viewport"}

    max_frames = max(0, _env_int("BEHAVIOR1K_VIEWPORT_WAIT_MAX_FRAMES", 120))
    dock_frames = max(0, _env_int("BEHAVIOR1K_VIEWPORT_DOCK_FRAMES", 2))

    def bounded_wait_for_viewport(self: Any) -> None:  # type: ignore[no-untyped-def]
        try:
            from omni.kit.viewport.utility import get_active_viewport

            if getattr(self, "config", {}).get("create_new_stage") is False:
                raise Exception("create_new_stage is False")
            viewport_api = get_active_viewport()
            for _ in range(max_frames):
                frame_info = getattr(viewport_api, "frame_info", {}) or {}
                if frame_info.get("viewport_handle", None) is not None:
                    break
                self._app.update()
        except Exception:
            pass

        for _ in range(dock_frames):
            self._app.update()

    simulation_app_cls._wait_for_viewport = bounded_wait_for_viewport
    simulation_app_cls._behavior1k_headless_viewport_wait_patch = True
    simulation_app_cls._behavior1k_headless_viewport_wait_original = original_wait
    simulation_app_cls._behavior1k_headless_viewport_wait_limits = {
        "max_frames": max_frames,
        "dock_frames": dock_frames,
    }
    return {"applied": True, "max_frames": max_frames, "dock_frames": dock_frames}


def _configure_isaacsim_noninteractive_eula(
    config: Behavior1KRuntimeConfig,
) -> JsonDict:
    """Make IsaacSim imports non-interactive on batch nodes after EULA approval."""

    if not config.headless:
        return {"applied": False, "reason": "not_headless"}
    if _env_falsey("BEHAVIOR1K_ACCEPT_NVIDIA_EULA"):
        return {"applied": False, "reason": "disabled_by_env"}
    previous = {
        "ACCEPT_EULA": os.environ.get("ACCEPT_EULA"),
        "OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
    }
    os.environ.setdefault("ACCEPT_EULA", "Y")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    return {
        "applied": True,
        "previous": previous,
        "value": {
            "ACCEPT_EULA": os.environ.get("ACCEPT_EULA"),
            "OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA"),
        },
    }


def _maybe_patch_omnigibson_meta_root_links() -> JsonDict:
    """Patch OG 1.1.x root-link inference for BEHAVIOR 3.9 meta-state links.

    Some BEHAVIOR scene assets add state metadata links such as
    `meta__base_link_openfillable_0_0_link` next to the physical `base_link`.
    OG 1.1.x treats both as root-link candidates and raises before the scene can
    load. The compatibility patch filters those metadata candidates only when a
    single physical root remains.
    """

    if _env_falsey("BEHAVIOR1K_PATCH_META_ROOT_LINKS"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import omnigibson.prims.entity_prim as entity_module
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed:{type(exc).__name__}: {exc}",
        }
    entity_cls = getattr(entity_module, "EntityPrim", None)
    if entity_cls is None:
        return {"applied": False, "reason": "missing_entity_prim"}
    if getattr(entity_cls, "_behavior1k_meta_root_link_patch", False):
        return {"applied": True, "reason": "already_applied"}

    required = ("ClothPrim", "PrimType", "absolute_prim_path_to_scene_relative")
    missing = [name for name in required if not hasattr(entity_module, name)]
    has_rigid_prim = hasattr(entity_module, "RigidPrim")
    has_rigid_dynamic_prim = hasattr(entity_module, "RigidDynamicPrim")
    has_rigid_kinematic_prim = hasattr(entity_module, "RigidKinematicPrim")
    if not (has_rigid_prim or (has_rigid_dynamic_prim and has_rigid_kinematic_prim)):
        missing.append("RigidPrim_or_RigidDynamicPrim/RigidKinematicPrim")
    if missing:
        return {"applied": False, "reason": f"missing_symbols:{','.join(missing)}"}

    original_update_links = entity_cls.update_links

    def update_links_with_meta_filter(self: Any) -> None:  # type: ignore[no-untyped-def]
        joint_children = set()
        links_to_create = {}
        for prim in self._prim.GetChildren():
            link_cls = None
            link_name = prim.GetName()
            prim_type_name = prim.GetPrimTypeInfo().GetTypeName()
            if (
                self._prim_type == entity_module.PrimType.RIGID
                and prim_type_name == "Xform"
            ):
                link_cls = entity_module.PrimType.RIGID
                for child_prim in prim.GetChildren():
                    if (
                        "joint"
                        not in child_prim.GetPrimTypeInfo().GetTypeName().lower()
                    ):
                        continue
                    relationships = {
                        r.GetName(): r for r in child_prim.GetRelationships()
                    }
                    body0 = relationships.get("physics:body0")
                    body1 = relationships.get("physics:body1")
                    if body0 is None or body1 is None:
                        continue
                    if len(body0.GetTargets()) > 0 and len(body1.GetTargets()) > 0:
                        joint_children.add(
                            body1.GetTargets()[0].pathString.split("/")[-1]
                        )
            elif (
                self._prim_type == entity_module.PrimType.CLOTH
                and prim_type_name == "Mesh"
            ):
                link_cls = entity_module.PrimType.CLOTH

            if link_cls is not None:
                links_to_create[link_name] = (link_cls, prim)

        valid_root_links = list(set(links_to_create.keys()) - joint_children)
        filtered_root_links = _behavior1k_filter_meta_root_links(valid_root_links)
        assert len(filtered_root_links) == 1, (
            f"Only a single root link should have been found for {self.name}, "
            f"but found multiple instead: {valid_root_links}"
        )
        self._root_link_name = filtered_root_links[0]

        self._links = dict()
        for link_name, (link_type, prim) in links_to_create.items():
            is_root_link = link_name == self._root_link_name
            is_kinematic = (
                self._load_config.get("kinematic_only", False)
                if is_root_link
                else False
            )
            link_load_config = {
                "kinematic_only": is_kinematic,
                "belongs_to_articulation": self._articulation_view is not None
                and not is_root_link,
                "remesh": self._load_config.get("remesh", True),
                "xform_props_pre_loaded": self._load_config.get(
                    "xform_props_pre_loaded", False
                ),
                "scale": self._load_config.get("scale", None),
            }
            if link_type == entity_module.PrimType.RIGID:
                if has_rigid_prim:
                    link_cls = entity_module.RigidPrim
                else:
                    link_cls = (
                        entity_module.RigidKinematicPrim
                        if is_kinematic
                        else entity_module.RigidDynamicPrim
                    )
            else:
                link_cls = entity_module.ClothPrim
            if getattr(
                getattr(entity_module, "gm", None), "USE_PBR_MATERIALS", False
            ) and hasattr(entity_module, "force_pbr_material_for_link"):
                entity_module.force_pbr_material_for_link(self._prim, link_name)
            self._links[link_name] = link_cls(
                relative_prim_path=entity_module.absolute_prim_path_to_scene_relative(
                    self.scene, prim.GetPrimPath().__str__()
                ),
                name=f"{self._name}:{link_name}",
                load_config=link_load_config,
            )
            self._links[link_name].load(self.scene)

    entity_cls._behavior1k_original_update_links = original_update_links
    entity_cls.update_links = update_links_with_meta_filter
    entity_cls._behavior1k_meta_root_link_patch = True
    return {"applied": True, "reason": "patched"}


def _behavior1k_triangulate_face_indices(
    face_vertex_counts: Any,
    face_vertex_indices: Any,
) -> np.ndarray:
    """Vectorize USD polygon fan triangulation without changing face geometry."""

    counts = np.asarray(face_vertex_counts, dtype=np.int64).reshape(-1)
    indices = np.asarray(face_vertex_indices, dtype=np.int64).reshape(-1)
    if np.any(counts < 0):
        raise ValueError("USD face vertex counts must be non-negative")
    consumed = int(counts.sum())
    if consumed > indices.size:
        raise ValueError(
            f"USD face vertex indices are truncated: expected {consumed}, got {indices.size}"
        )
    triangle_counts = np.maximum(counts - 2, 0)
    triangle_total = int(triangle_counts.sum())
    if triangle_total == 0:
        return np.empty((0, 3), dtype=np.int64)

    face_starts = np.cumsum(np.concatenate((np.zeros(1, dtype=np.int64), counts[:-1])))
    valid = triangle_counts > 0
    valid_triangle_counts = triangle_counts[valid]
    repeated_starts = np.repeat(face_starts[valid], valid_triangle_counts)
    triangle_group_starts = np.repeat(
        np.cumsum(valid_triangle_counts) - valid_triangle_counts,
        valid_triangle_counts,
    )
    fan_offsets = np.arange(triangle_total, dtype=np.int64) - triangle_group_starts
    return np.column_stack(
        (
            indices[repeated_starts],
            indices[repeated_starts + fan_offsets + 1],
            indices[repeated_starts + fan_offsets + 2],
        )
    )


def _maybe_patch_omnigibson_fast_mesh_triangulation() -> JsonDict:
    """Avoid OG 1.1.x's per-triangle Torch-scalar conversion bottleneck.

    The upstream implementation builds a Python list containing Torch scalar
    tensors for every triangle. Large official BEHAVIOR scenes then spend many
    minutes converting that list back to NumPy inside Trimesh. This patch keeps
    the exact USD fan-triangulation rule while constructing arrays directly.
    """

    if _env_falsey("BEHAVIOR1K_PATCH_FAST_MESH_TRIANGULATION"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import omnigibson.utils.usd_utils as usd_utils
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed:{type(exc).__name__}: {exc}",
        }
    if getattr(usd_utils, "_behavior1k_fast_mesh_triangulation_patch", False):
        return {"applied": True, "reason": "already_applied"}
    trimesh_module = getattr(usd_utils, "trimesh", None)
    if trimesh_module is None:
        return {"applied": False, "reason": "missing_trimesh_module"}

    original = usd_utils.mesh_prim_mesh_to_trimesh_mesh

    def fast_mesh_prim_mesh_to_trimesh_mesh(
        mesh_prim: Any,
        include_normals: bool = True,
        include_texcoord: bool = True,
    ) -> Any:
        mesh_type = mesh_prim.GetPrimTypeInfo().GetTypeName()
        assert mesh_type == "Mesh", (
            f"Expected mesh prim to have type Mesh, got {mesh_type}"
        )
        counts = mesh_prim.GetAttribute("faceVertexCounts").Get()
        vertices = np.asarray(mesh_prim.GetAttribute("points").Get(), dtype=np.float64)
        face_indices = mesh_prim.GetAttribute("faceVertexIndices").Get()
        kwargs: JsonDict = {
            "vertices": vertices,
            "faces": _behavior1k_triangulate_face_indices(counts, face_indices),
        }
        if include_normals:
            normals = mesh_prim.GetAttribute("normals").Get()
            if normals is not None:
                kwargs["vertex_normals"] = np.asarray(normals, dtype=np.float64)
        if include_texcoord:
            raw_texture = mesh_prim.GetAttribute("primvars:st").Get()
            if raw_texture is not None:
                kwargs["visual"] = trimesh_module.visual.TextureVisuals(
                    uv=np.asarray(raw_texture, dtype=np.float64)
                )
        return trimesh_module.Trimesh(**kwargs)

    usd_utils._behavior1k_original_mesh_prim_mesh_to_trimesh_mesh = original
    usd_utils.mesh_prim_mesh_to_trimesh_mesh = fast_mesh_prim_mesh_to_trimesh_mesh
    usd_utils._behavior1k_fast_mesh_triangulation_patch = True
    return {"applied": True, "reason": "patched_exact_fan_triangulation"}


def _behavior1k_filter_meta_root_links(root_links: list[str]) -> list[str]:
    if len(root_links) <= 1:
        return list(root_links)
    physical = [link for link in root_links if not str(link).startswith("meta__")]
    if len(physical) == 1:
        return physical
    if "base_link" in physical and all(
        link == "base_link" or _behavior1k_is_part_link(link) for link in physical
    ):
        return ["base_link"]
    return list(root_links)


def _behavior1k_is_part_link(link_name: str) -> bool:
    text = str(link_name)
    return text.startswith("link_") or text in {
        "door",
        "drawer",
        "glass",
        "glass_base",
        "handle",
        "leaf",
        "lid",
        "panel",
        "shelf",
    }


def _maybe_patch_fetch_eef_link_fallback() -> JsonDict:
    """Handle OG Fetch assets whose USD exposes gripper links but no synthetic `eef_link`."""

    if _env_falsey("BEHAVIOR1K_PATCH_FETCH_EEF_LINK"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import omnigibson.robots.fetch as fetch_module
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed: {type(exc).__name__}: {exc}",
        }
    fetch_cls = getattr(fetch_module, "Fetch", None)
    if fetch_cls is None:
        return {"applied": False, "reason": "Fetch_class_missing"}
    if getattr(fetch_cls, "_behavior1k_fetch_eef_link_fallback_patch", False):
        return {"applied": True, "reason": "already_patched"}
    original_eef_link_names = getattr(fetch_cls, "eef_link_names", None)

    def eef_link_names_with_asset_fallback(self: Any) -> dict[str, str]:
        default_arm = getattr(self, "default_arm", "0")
        links = getattr(self, "_links", None)
        if isinstance(links, dict):
            for candidate in (
                "eef_link",
                "gripper_link",
                "wrist_roll_link",
                "r_gripper_finger_link",
                "l_gripper_finger_link",
            ):
                if candidate in links:
                    return {default_arm: candidate}
            for candidate in (
                "eef_link",
                "gripper_link",
                "wrist_roll_link",
                "r_gripper_finger_link",
                "l_gripper_finger_link",
            ):
                for link_name in links:
                    if str(link_name).endswith(candidate):
                        return {default_arm: str(link_name)}
        if original_eef_link_names is not None:
            try:
                value = original_eef_link_names.__get__(self, type(self))
                if isinstance(value, dict):
                    return value
            except Exception:
                pass
        return {default_arm: "eef_link"}

    fetch_cls._behavior1k_original_eef_link_names = original_eef_link_names
    fetch_cls.eef_link_names = property(eef_link_names_with_asset_fallback)
    fetch_cls._behavior1k_fetch_eef_link_fallback_patch = True
    return {"applied": True, "reason": "patched"}


def _maybe_patch_empty_finger_property_inference() -> JsonDict:
    """Let OmniGibson continue when a robot finger mesh has no usable collision points.

    BEHAVIOR assets can expose Fetch finger links whose collision boundary point
    filtering leaves an empty tensor. That breaks assisted-grasp point inference
    during environment construction, before any agent primitive can observe the
    scene. The patch preserves the upstream implementation for normal robots and
    only falls back for this specific empty-point reduction error.
    """

    if _env_falsey("BEHAVIOR1K_PATCH_EMPTY_FINGER_PROPERTIES"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import torch as th
        import omnigibson.robots.manipulation_robot as manipulation_module
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed: {type(exc).__name__}: {exc}",
        }
    robot_cls = getattr(manipulation_module, "ManipulationRobot", None)
    grasping_point_cls = getattr(manipulation_module, "GraspingPoint", None)
    if robot_cls is None or grasping_point_cls is None:
        return {"applied": False, "reason": "missing_manipulation_robot_symbols"}
    if getattr(robot_cls, "_behavior1k_empty_finger_inference_patch", False):
        return {"applied": True, "reason": "already_patched"}
    original_infer = getattr(robot_cls, "_infer_finger_properties", None)
    if not callable(original_infer):
        return {"applied": False, "reason": "missing_infer_finger_properties"}

    def infer_finger_properties_with_empty_point_fallback(self: Any) -> Any:  # type: ignore[no-untyped-def]
        try:
            return original_infer(self)
        except RuntimeError as exc:
            message = str(exc)
            if (
                "input.numel() == 0" not in message
                and "Expected reduction dim" not in message
            ):
                raise
            self._eef_to_fingertip_lengths = {}
            self._default_ag_start_points = {}
            self._default_ag_end_points = {}
            finger_links_by_arm = getattr(self, "finger_links", {}) or {}
            for arm, finger_links in dict(finger_links_by_arm).items():
                self._eef_to_fingertip_lengths[arm] = {}
                start_points = []
                end_points = []
                for index, link in enumerate(list(finger_links)[:2]):
                    link_name = str(
                        getattr(link, "body_name", None)
                        or getattr(link, "name", None)
                        or getattr(link, "prim_path", f"finger_{index}")
                    )
                    self._eef_to_fingertip_lengths[arm][link_name] = 0.05
                    point = grasping_point_cls(
                        link_name=link_name, position=th.tensor([0.0, 0.0, 0.03])
                    )
                    if index == 0:
                        start_points.append(point)
                    else:
                        end_points.append(point)
                self._default_ag_start_points[arm] = start_points
                self._default_ag_end_points[arm] = end_points or list(start_points)
            return None

    robot_cls._behavior1k_original_infer_finger_properties = original_infer
    robot_cls._infer_finger_properties = (
        infer_finger_properties_with_empty_point_fallback
    )
    robot_cls._behavior1k_empty_finger_inference_patch = True
    return {"applied": True, "reason": "patched"}


def _maybe_patch_behavior_task_reset_none_object_scope() -> JsonDict:
    """Let BEHAVIOR task reset tolerate unresolved optional object-scope entries.

    Some full-BDDL task / scene combinations in the local BEHAVIOR-1K 3.9
    assets leave a sampled object-scope entry as ``None`` after reset. Upstream
    then crashes while force-waking objects via ``obj.exists`` before any
    agent-facing primitive can observe the scene. This patch is deliberately
    narrow: it only catches that exact wake-loop failure, records which scope
    entries were skipped on the task instance, and wakes all valid DatasetObject
    entries. It does not mark task success, alter predicates, or expose checker
    state to the agent runtime.
    """

    if _env_falsey("BEHAVIOR1K_PATCH_RESET_NONE_OBJECT_SCOPE"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import omnigibson.tasks.behavior_task as behavior_task_module
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed: {type(exc).__name__}: {exc}",
        }
    task_cls = getattr(behavior_task_module, "BehaviorTask", None)
    if task_cls is None:
        return {"applied": False, "reason": "missing_BehaviorTask"}
    if getattr(task_cls, "_behavior1k_reset_none_scope_patch", False):
        return {"applied": True, "reason": "already_patched"}
    original_reset = getattr(task_cls, "reset", None)
    if not callable(original_reset):
        return {"applied": False, "reason": "missing_reset"}
    dataset_object_cls = getattr(behavior_task_module, "DatasetObject", None)

    def reset_with_none_object_scope_fallback(self: Any, env: Any) -> Any:  # type: ignore[no-untyped-def]
        try:
            return original_reset(self, env)
        except AttributeError as exc:
            if "'NoneType' object has no attribute 'exists'" not in str(exc):
                raise
            object_scope = getattr(self, "object_scope", None)
            if not isinstance(object_scope, dict):
                raise
            skipped = [str(name) for name, obj in object_scope.items() if obj is None]
            if not skipped:
                raise
            woke: list[str] = []
            resume_after_missing = False
            for name, obj in object_scope.items():
                if obj is None:
                    resume_after_missing = True
                    continue
                if not resume_after_missing:
                    continue
                if not bool(getattr(obj, "exists", False)):
                    continue
                if dataset_object_cls is not None and not isinstance(
                    obj, dataset_object_cls
                ):
                    continue
                wake = getattr(obj, "wake", None)
                if callable(wake):
                    wake()
                    woke.append(str(name))
            self._behavior1k_reset_none_scope_skipped = skipped
            self._behavior1k_reset_none_scope_woke = woke
            return None

    task_cls._behavior1k_original_reset = original_reset
    task_cls.reset = reset_with_none_object_scope_fallback
    task_cls._behavior1k_reset_none_scope_patch = True
    return {"applied": True, "reason": "patched"}


def _maybe_patch_behavior1k_empty_semantic_remapper() -> JsonDict:
    """Keep native visual reads alive when Replicator returns an empty semantic frame."""

    if _env_falsey("BEHAVIOR1K_PATCH_EMPTY_SEMANTIC_REMAPPER"):
        return {"applied": False, "reason": "disabled_by_env"}
    try:
        import omnigibson.utils.vision_utils as vision_utils
    except Exception as exc:
        return {
            "applied": False,
            "reason": f"import_failed: {type(exc).__name__}: {exc}",
        }
    remapper_cls = getattr(vision_utils, "Remapper", None)
    if remapper_cls is None:
        return {"applied": False, "reason": "missing_Remapper"}
    if getattr(remapper_cls, "_behavior1k_empty_semantic_patch", False):
        return {"applied": True, "reason": "already_patched"}
    original_remap = getattr(remapper_cls, "remap", None)
    if not callable(original_remap):
        return {"applied": False, "reason": "missing_remap"}

    def remap_with_empty_image_fallback(
        self: Any,
        old_mapping: Any,
        new_mapping: Any,
        image: Any,
        image_keys: Any = None,
    ) -> Any:
        try:
            if hasattr(image, "numel") and int(image.numel()) == 0:
                return image, dict(new_mapping or {})
        except Exception:
            pass
        if not old_mapping:
            return image, dict(new_mapping or {})
        return original_remap(self, old_mapping, new_mapping, image, image_keys)

    remapper_cls._behavior1k_original_remap = original_remap
    remapper_cls.remap = remap_with_empty_image_fallback
    remapper_cls._behavior1k_empty_semantic_patch = True
    return {"applied": True, "reason": "patched"}


def _default_omnigibson_key_path() -> Path:
    return (
        get_project_paths().external_assets("behavior1k")
        / "runtime_keys/omnigibson.key"
    )


def _default_behavior1k_dataset_root() -> Path:
    return get_project_paths().external_assets("behavior1k") / "datasets"


def prepare_behavior1k_runtime_assets(config: Behavior1KRuntimeConfig) -> JsonDict:
    """Resolve and validate BEHAVIOR assets before OmniGibson caches its path macros."""

    scene = (
        config.env_config.get("scene") if isinstance(config.env_config, dict) else None
    )
    requested_scene_file = scene.get("scene_file") if isinstance(scene, dict) else None
    _configure_omnigibson_asset_env(config)
    _configure_behavior1k_scene_file_adapter(config)
    effective_root = Path(os.environ.get("OMNIGIBSON_DATASET_PATH", "")).expanduser()
    scenes_path = effective_root / "behavior-1k-assets" / "scenes"
    scene = (
        config.env_config.get("scene") if isinstance(config.env_config, dict) else None
    )
    scene_model = scene.get("scene_model") if isinstance(scene, dict) else None
    scene_instance = scene.get("scene_instance") if isinstance(scene, dict) else None
    scene_file = scene.get("scene_file") if isinstance(scene, dict) else None
    scene_model_path = scenes_path / str(scene_model) if scene_model else None
    activity_scene_contract = _behavior1k_activity_scene_contract(config)
    base_scene_instance = _behavior1k_is_base_scene_instance(
        scene_model, scene_instance
    )
    task_scene_dir = (
        effective_root / "2025-challenge-task-instances" / "scenes" / str(scene_model)
        if scene_model
        and scene_instance
        and not base_scene_instance
        and requested_scene_file is None
        else None
    )
    task_scene_file = (
        task_scene_dir / "json" / f"{scene_instance}.json"
        if task_scene_dir is not None
        else None
    )
    blockers: list[str] = []
    if not effective_root.is_dir():
        blockers.append("omnigibson_dataset_root_missing")
    if not scenes_path.is_dir():
        blockers.append("behavior1k_scenes_path_missing")
    if scene_model_path is not None and not scene_model_path.is_dir():
        blockers.append(f"behavior1k_scene_model_missing:{scene_model}")
    if task_scene_dir is not None and not task_scene_dir.is_dir():
        blockers.append(f"behavior1k_task_scene_dir_missing:{scene_model}")
    if task_scene_file is not None and not task_scene_file.is_file():
        blockers.append(f"behavior1k_scene_instance_missing:{scene_instance}")
    if scene_file is not None and not Path(str(scene_file)).expanduser().is_file():
        blockers.append("behavior1k_scene_file_missing")
    blockers.extend(activity_scene_contract["blockers"])
    return {
        "ok": not blockers,
        "configured_dataset_path": config.dataset_path
        or os.getenv("BEHAVIOR1K_DATASET_PATH"),
        "effective_dataset_path": str(effective_root),
        "scenes_path": str(scenes_path),
        "scene_model": scene_model,
        "scene_instance": scene_instance,
        "scene_model_path": str(scene_model_path)
        if scene_model_path is not None
        else None,
        "task_scene_dir": str(task_scene_dir) if task_scene_dir is not None else None,
        "task_scene_file": str(task_scene_file)
        if task_scene_file is not None
        else None,
        "scene_file": str(scene_file) if scene_file is not None else None,
        "activity_scene_contract": activity_scene_contract,
        "blockers": blockers,
    }


def _behavior1k_activity_scene_contract(config: Behavior1KRuntimeConfig) -> JsonDict:
    """Validate an official activity / scene pairing without importing OmniGibson."""

    task = (
        config.env_config.get("task") if isinstance(config.env_config, dict) else None
    )
    scene = (
        config.env_config.get("scene") if isinstance(config.env_config, dict) else None
    )
    activity_name = task.get("activity_name") if isinstance(task, dict) else None
    scene_model = scene.get("scene_model") if isinstance(scene, dict) else None
    source_root = get_project_paths().external_upstream("behavior1k")
    definition_file = (
        source_root
        / "bddl3"
        / "bddl"
        / "activity_definitions"
        / str(activity_name)
        / "problem0.bddl"
        if activity_name
        else None
    )
    sampling_manifest = (
        source_root
        / "OmniGibson"
        / "omnigibson"
        / "sampling"
        / "task_custom_lists.json"
    )
    declared_scenes: list[str] = []
    manifest_error: str | None = None
    if sampling_manifest.is_file() and activity_name:
        try:
            manifest = json.loads(sampling_manifest.read_text(encoding="utf-8"))
            activity_entry = (
                manifest.get(str(activity_name)) if isinstance(manifest, dict) else None
            )
            if isinstance(activity_entry, dict):
                declared_scenes = sorted(
                    str(key)
                    for key, value in activity_entry.items()
                    if key != "room_types" and isinstance(value, dict)
                )
        except (OSError, ValueError) as exc:
            manifest_error = f"{type(exc).__name__}: {exc}"
    blockers: list[str] = []
    if activity_name and (definition_file is None or not definition_file.is_file()):
        blockers.append(f"behavior1k_activity_definition_missing:{activity_name}")
    if manifest_error:
        blockers.append("behavior1k_sampling_manifest_unreadable")
    if declared_scenes and scene_model not in declared_scenes:
        blockers.append(
            f"behavior1k_activity_scene_mapping_mismatch:{activity_name}:{scene_model}:expected={','.join(declared_scenes)}"
        )
    return {
        "ok": not blockers,
        "activity_name": activity_name,
        "scene_model": scene_model,
        "declared_scenes": declared_scenes,
        "definition_file": str(definition_file)
        if definition_file is not None
        else None,
        "sampling_manifest": str(sampling_manifest),
        "sampling_manifest_error": manifest_error,
        "blockers": blockers,
    }


def _behavior1k_is_base_scene_instance(scene_model: Any, scene_instance: Any) -> bool:
    return bool(
        scene_model and scene_instance and str(scene_instance) == f"{scene_model}_best"
    )


def _configure_behavior1k_scene_file_adapter(config: Behavior1KRuntimeConfig) -> None:
    """Point OG v1.1.x at a cache-adapted scene JSON when datasets use the older registry nesting."""

    scene = config.env_config.get("scene")
    if not isinstance(scene, dict):
        return
    empty_visual_scene = bool(scene.pop("empty_visual_scene", False)) or _env_truthy(
        "BEHAVIOR1K_EMPTY_VISUAL_SCENE"
    )
    scene_type = str(scene.get("type") or "")
    if scene_type not in {"InteractiveTraversableScene", "TraversableScene"}:
        return
    scene_file = scene.get("scene_file")
    dataset_root = (
        config.dataset_path
        or os.getenv("OMNIGIBSON_DATASET_PATH")
        or os.getenv("OMNIGIBSON_DATA_PATH")
    )
    if dataset_root and _behavior1k_uses_flat_asset_layout(
        Path(dataset_root).expanduser()
    ):
        scene.setdefault("dataset_name", ".")
        scene.setdefault("_harness_dataset_layout", "flat_behavior_assets")
    if scene_file is None:
        scene_model = scene.get("scene_model")
        if not scene_model:
            return
        scene_instance = scene.get("scene_instance")
        if not dataset_root:
            return
        dataset_path = Path(dataset_root).expanduser()
        if scene_instance and not _behavior1k_is_base_scene_instance(
            scene_model, scene_instance
        ):
            scene_file = (
                dataset_path
                / "2025-challenge-task-instances"
                / "scenes"
                / str(scene_model)
                / "json"
                / f"{scene_instance}.json"
            )
            scene["scene_file"] = str(scene_file)
        else:
            base_instance = str(scene_instance or f"{scene_model}_best")
            scene_file = (
                dataset_path
                / "scenes"
                / str(scene_model)
                / "json"
                / f"{base_instance}.json"
            )
        if not scene_file.is_file():
            return
        scene["scene_file"] = str(scene_file)
    if empty_visual_scene:
        adapted = _behavior1k_empty_visual_scene_file(Path(scene_file).expanduser())
        if adapted is not None:
            scene["scene_file"] = str(adapted)
            scene["_harness_scene_file_adapter"] = "empty_visual_scene"
            return
    adapted = _behavior1k_scene_file_with_runtime_registry(
        Path(scene_file).expanduser()
    )
    if adapted is not None:
        scene["scene_file"] = str(adapted)
        scene.setdefault(
            "_harness_scene_file_adapter", "state.registry_to_runtime_registries"
        )


def _behavior1k_empty_visual_scene_file(scene_file: Path) -> Path | None:
    adapted = _behavior1k_scene_info_with_runtime_registry(scene_file)
    if adapted is None:
        return None
    adapted.setdefault("metadata", {})
    if isinstance(adapted["metadata"], dict):
        adapted["metadata"].setdefault("harness_adapter", {})
        if isinstance(adapted["metadata"]["harness_adapter"], dict):
            adapted["metadata"]["harness_adapter"]["empty_visual_scene"] = True
            adapted["metadata"]["harness_adapter"]["source_scene_file"] = str(
                scene_file
            )
    adapted["objects_info"] = {"init_info": {}}
    adapted["state"]["object_registry"] = {}
    adapted["state"]["system_registry"] = {}
    adapted["state"]["registry"] = {"object_registry": {}, "system_registry": {}}
    return _write_behavior1k_scene_cache(
        scene_file=scene_file, scene_info=adapted, suffix="empty_visual_scene"
    )


def _behavior1k_uses_flat_asset_layout(dataset_root: Path) -> bool:
    return (dataset_root / "scenes").is_dir() and not (
        dataset_root / "behavior-1k-assets" / "scenes"
    ).is_dir()


def _behavior1k_runtime_cache_root() -> Path:
    """Return a writable cache root, honoring strict-runtime cache mounts."""

    candidates = (
        os.getenv("BEHAVIOR1K_RUNTIME_CACHE_ROOT"),
        os.getenv("XDG_CACHE_HOME"),
        os.getenv("OMNIGIBSON_APPDATA_PATH"),
    )
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser()
        if root.is_absolute():
            return root / "agentic-embodied-arena" / "behavior1k"
    return resolve_project_root() / ".cache" / "behavior1k"


def _ensure_behavior1k_dataset_compat_root(dataset_root: Path) -> Path:
    dataset_root = dataset_root.expanduser()
    if not _behavior1k_uses_flat_asset_layout(dataset_root):
        return dataset_root
    digest = hashlib.sha1(str(dataset_root.resolve()).encode("utf-8")).hexdigest()[:12]
    compat_root = _behavior1k_runtime_cache_root() / "dataset_compat" / digest
    compat_root.mkdir(parents=True, exist_ok=True)
    robot_assets_root = _default_omnigibson_asset_root(dataset_root)
    links = {
        "behavior-1k-assets": dataset_root,
        "scenes": dataset_root / "scenes",
        "objects": dataset_root / "objects",
        "metadata": dataset_root / "metadata",
        "systems": dataset_root / "systems",
        "2025-challenge-task-instances": dataset_root / "2025-challenge-task-instances",
    }
    key_path = _default_omnigibson_key_path()
    if key_path.exists():
        links["omnigibson.key"] = key_path
    for name, target in links.items():
        if not target.exists():
            continue
        link = compat_root / name
        if link.exists() or link.is_symlink():
            try:
                if link.resolve() == target.resolve():
                    continue
            except OSError:
                continue
            continue
        try:
            link.symlink_to(target, target_is_directory=target.is_dir())
        except OSError:
            continue
    if robot_assets_root.exists():
        _ensure_behavior1k_robot_assets_compat_root(
            compat_root / "omnigibson-robot-assets", robot_assets_root
        )
    return compat_root


def _ensure_behavior1k_robot_assets_compat_root(
    compat_robot_root: Path, robot_assets_root: Path
) -> None:
    """Expose robot assets in the layout expected by OG 1.1 robot classes."""

    if compat_robot_root.is_symlink():
        try:
            if (
                compat_robot_root.resolve() == robot_assets_root.resolve()
                and (
                    robot_assets_root / "models" / "fetch" / "usd" / "fetch.usda"
                ).exists()
            ):
                return
            compat_robot_root.unlink()
        except OSError:
            return
    compat_robot_root.mkdir(parents=True, exist_ok=True)
    for child in robot_assets_root.iterdir():
        if child.name == "models":
            continue
        link = compat_robot_root / child.name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(child, target_is_directory=child.is_dir())
        except OSError:
            continue

    src_models = robot_assets_root / "models"
    compat_models = compat_robot_root / "models"
    compat_models.mkdir(exist_ok=True)
    if src_models.exists():
        for model_dir in src_models.iterdir():
            if model_dir.name == "fetch":
                continue
            link = compat_models / model_dir.name
            if link.exists() or link.is_symlink():
                continue
            try:
                link.symlink_to(model_dir, target_is_directory=model_dir.is_dir())
            except OSError:
                continue

    src_fetch = src_models / "fetch"
    if not src_fetch.exists():
        return
    compat_fetch = compat_models / "fetch"
    compat_fetch.mkdir(exist_ok=True)
    for child in src_fetch.iterdir():
        if child.name in {"usd", "urdf"}:
            continue
        link = compat_fetch / child.name
        if link.exists() or link.is_symlink():
            continue
        try:
            link.symlink_to(child, target_is_directory=child.is_dir())
        except OSError:
            continue
    fetch_usd = src_fetch / "fetch" / "fetch.usd"
    fetch_eef_usd = src_fetch / "fetch" / "fetch_eef.usd"
    fetch_urdf = src_fetch / "fetch.urdf"
    fetch_gripper_urdf = src_fetch / "fetch_gripper.urdf"
    usda_reference_specs = {
        compat_fetch / "usd" / "fetch.usda": (fetch_usd, "fetch"),
        compat_fetch / "usd" / "fetch_eef.usda": (fetch_eef_usd, "fetch_eef"),
    }
    for alias, (target, default_prim) in usda_reference_specs.items():
        if target.exists():
            _write_behavior1k_usda_reference_alias(
                alias, target, default_prim=default_prim
            )

    alias_specs = {
        compat_fetch / "usd" / "fetch.usd": fetch_usd,
        compat_fetch / "usd" / "fetch_eef.usd": fetch_eef_usd,
        compat_fetch / "urdf" / "fetch.urdf": fetch_urdf,
        compat_fetch / "urdf" / "fetch_gripper.urdf": fetch_gripper_urdf,
    }
    for alias, target in alias_specs.items():
        if not target.exists():
            continue
        alias.parent.mkdir(parents=True, exist_ok=True)
        if alias.exists() or alias.is_symlink():
            continue
        try:
            alias.symlink_to(target, target_is_directory=False)
        except OSError:
            continue


def _write_behavior1k_usda_reference_alias(
    alias: Path, target: Path, default_prim: str
) -> None:
    alias.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=alias.parent)
    text = (
        "#usda 1.0\n"
        "(\n"
        f'    defaultPrim = "{default_prim}"\n'
        "    subLayers = [\n"
        f"        @{relative_target}@\n"
        "    ]\n"
        ")\n\n"
    )
    try:
        if alias.is_symlink():
            alias.unlink()
        if (
            not alias.exists()
            or alias.read_text(encoding="utf-8", errors="ignore") != text
        ):
            alias.write_text(text, encoding="utf-8")
    except OSError:
        return


def _behavior1k_scene_file_with_runtime_registry(scene_file: Path) -> Path | None:
    adapted_info = _behavior1k_scene_info_with_runtime_registry(scene_file)
    if adapted_info is None:
        return None
    return _write_behavior1k_scene_cache(
        scene_file=scene_file, scene_info=adapted_info, suffix="registry_v1_1"
    )


def _behavior1k_scene_info_with_runtime_registry(scene_file: Path) -> JsonDict | None:
    if not scene_file.exists():
        return None
    try:
        scene_info = json.loads(scene_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(scene_info, dict):
        return None
    state = scene_info.get("state")
    if not isinstance(state, dict):
        return None
    if "object_registry" in state and "system_registry" in state:
        return deepcopy(scene_info)
    registry = state.get("registry")
    if not isinstance(registry, dict):
        return None
    object_registry = registry.get("object_registry")
    system_registry = registry.get("system_registry")
    if not isinstance(object_registry, dict) or not isinstance(system_registry, dict):
        return None
    adapted_info = deepcopy(scene_info)
    adapted_info["state"]["object_registry"] = object_registry
    adapted_info["state"]["system_registry"] = system_registry
    adapted_info.setdefault("metadata", {})
    if isinstance(adapted_info["metadata"], dict):
        adapted_info["metadata"].setdefault("harness_adapter", {})
        if isinstance(adapted_info["metadata"]["harness_adapter"], dict):
            adapted_info["metadata"]["harness_adapter"]["state_registry_layout"] = (
                "runtime_v1_1_compatible"
            )
            adapted_info["metadata"]["harness_adapter"]["source_scene_file"] = str(
                scene_file
            )
    return adapted_info


def _write_behavior1k_scene_cache(
    scene_file: Path, scene_info: JsonDict, suffix: str
) -> Path:
    digest_source = f"{scene_file.resolve()}:{scene_file.stat().st_mtime_ns}:{scene_file.stat().st_size}"
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:12]
    cache_dir = _behavior1k_runtime_cache_root() / "scene_adapters"
    cache_dir.mkdir(parents=True, exist_ok=True)
    adapted_path = cache_dir / f"{scene_file.stem}.{suffix}.{digest}.json"
    adapted_path.write_text(json.dumps(scene_info), encoding="utf-8")
    return adapted_path


def _default_omnigibson_asset_root(dataset_root: Path) -> Path:
    repo_assets = get_project_paths().external_assets("behavior1k") / "og_assets_full"
    if (repo_assets / "models" / "fetch" / "fetch" / "fetch.usd").exists():
        return repo_assets
    return dataset_root / "omnigibson-robot-assets"


def run_env_probe_subprocess(timeout_seconds: int = 900) -> JsonDict:
    command = [
        sys.executable,
        "-m",
        "embodied_harness.behavior1k_preflight",
        "--env-probe-child",
    ]
    env = os.environ.copy()
    source_root = resolve_project_root() / "src"
    env["PYTHONPATH"] = f"{source_root}:{env.get('PYTHONPATH', '')}"
    env.setdefault("BEHAVIOR1K_ENV_PROBE_FAST_EXIT", "1")
    started_at = time.time()
    sidecar_path = _env_probe_sidecar_path(started_at)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    env["BEHAVIOR1K_ENV_PROBE_OUTPUT"] = str(sidecar_path)
    try:
        completed = subprocess.run(
            command,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=max(1, int(timeout_seconds)),
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        parsed = parse_env_probe_sidecar(sidecar_path)
        if parsed is not None:
            parsed["blocker"] = "timeout_after_sidecar_marker"
            parsed["timeout_seconds"] = int(timeout_seconds)
            parsed["command"] = command
            parsed["marker_source"] = "sidecar_json"
            parsed["sidecar_path"] = str(sidecar_path)
            parsed["stdout_tail"] = output[-8000:]
            parsed["stdout_blockers"] = _classify_env_probe_stdout_blockers(output)
            parsed["process_ok"] = False
            parsed["ok"] = False
            return parsed
        return {
            "ok": False,
            "blocker": "timeout",
            "timeout_seconds": int(timeout_seconds),
            "command": command,
            "sidecar_path": str(sidecar_path),
            "sidecar_exists": sidecar_path.exists(),
            "stdout_tail": output[-8000:],
            "stdout_blockers": _classify_env_probe_stdout_blockers(output),
        }
    parsed = parse_env_probe_sidecar(sidecar_path)
    marker_source = "sidecar_json"
    if parsed is None:
        parsed = parse_env_probe_output(completed.stdout)
        marker_source = "subprocess_stdout"
    if parsed is None:
        parsed = parse_latest_omnigibson_kit_marker(min_mtime=started_at - 5)
        marker_source = "omnigibson_kit_log"
    if parsed is None:
        return {
            "ok": False,
            "blocker": "missing_env_probe_marker",
            "returncode": completed.returncode,
            "command": command,
            "sidecar_path": str(sidecar_path),
            "sidecar_exists": sidecar_path.exists(),
            "stdout_tail": completed.stdout[-8000:],
            "stdout_blockers": _classify_env_probe_stdout_blockers(completed.stdout),
        }
    parsed["returncode"] = completed.returncode
    parsed["command"] = command
    parsed["marker_source"] = marker_source
    parsed["sidecar_path"] = str(sidecar_path)
    parsed["sidecar_exists"] = sidecar_path.exists()
    parsed["stdout_tail"] = completed.stdout[-8000:]
    parsed["stdout_blockers"] = _classify_env_probe_stdout_blockers(completed.stdout)
    parsed["process_ok"] = completed.returncode == 0
    if completed.returncode != 0:
        parsed["native_process_error_after_marker"] = True
    parsed["ok"] = bool(parsed.get("ok"))
    return parsed


def _classify_env_probe_stdout_blockers(output: str) -> list[str]:
    blockers: list[str] = []
    if "Only a single root link should have been found" in output:
        blockers.append("omnigibson_meta_root_link_assertion")
    if (
        "omnigibson_vray_mtl.mdl" in output
        and "Failed to resolve USD Asset Identifier" in output
    ):
        blockers.append("omnigibson_vray_material_identifier_unresolved")
    if "Segmentation fault" in output or "returncode=-11" in output:
        blockers.append("native_process_segfault")
    if "GLFW" in output:
        blockers.append("glfw_windowing_warning_or_error")
    if "Failed to create MDL shade node" in output:
        blockers.append("mdl_shade_node_create_failed")
    if "Unable to create convex mesh" in output:
        blockers.append("physx_convex_mesh_create_failed")
    return blockers


def _env_probe_sidecar_path(started_at: float) -> Path:
    configured = os.getenv("BEHAVIOR1K_ENV_PROBE_OUTPUT")
    if configured:
        return Path(configured).expanduser()
    root = resolve_project_root()
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(started_at))
    return root / "reports" / f"behavior1k_env_probe_child_{os.getpid()}_{stamp}.json"


def parse_env_probe_sidecar(path: str | Path) -> JsonDict | None:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return loaded if isinstance(loaded, dict) else None


def parse_env_probe_output(output: str) -> JsonDict | None:
    for line in reversed(output.splitlines()):
        marker_index = line.find(ENV_PROBE_MARKER)
        if marker_index >= 0:
            loaded = json.loads(line[marker_index + len(ENV_PROBE_MARKER) :])
            return loaded if isinstance(loaded, dict) else None
    return None


def parse_latest_omnigibson_kit_marker(
    min_mtime: float | None = None,
) -> JsonDict | None:
    configured_appdata = os.getenv("OMNIGIBSON_APPDATA_PATH")
    if configured_appdata:
        logs_root = (
            Path(configured_appdata).expanduser()
            / "local"
            / "logs"
            / "Kit"
            / "OmniGibson"
        )
        if not logs_root.exists():
            logs_root = (
                Path(configured_appdata).expanduser() / "logs" / "Kit" / "OmniGibson"
            )
    else:
        logs_root = (
            get_project_paths().external_environment("behavior1k")
            / "appdata/local/logs/Kit/OmniGibson"
        )
    candidates = sorted(
        logs_root.glob("*/*kit_*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates[:5]:
        if min_mtime is not None and path.stat().st_mtime < min_mtime:
            continue
        try:
            parsed = parse_env_probe_output(path.read_text(errors="replace"))
        except OSError:
            continue
        if parsed is not None:
            parsed["kit_log_path"] = str(path)
            return parsed
    return None


def official_version_probe() -> JsonDict:
    cmd = [
        "git",
        "ls-remote",
        "--tags",
        "https://github.com/StanfordVL/BEHAVIOR-1K.git",
        "refs/tags/v*",
    ]
    try:
        proc = subprocess.run(
            cmd, check=False, capture_output=True, text=True, timeout=30
        )
    except Exception as exc:
        return {"ok": False, "command": cmd, "error": f"{type(exc).__name__}: {exc}"}
    tags = []
    for line in proc.stdout.splitlines():
        ref = line.rsplit("/", 1)[-1]
        if ref and "^{}" not in ref:
            tags.append(ref)
    return {
        "ok": proc.returncode == 0,
        "command": cmd,
        "returncode": proc.returncode,
        "latest_tag_seen": tags[-1] if tags else None,
        "tags_tail": tags[-8:],
    }


def _maybe_apply_minimal_kit_no_flowusd_override() -> None:
    """Launch OmniGibson with a local kit that omits omni.flowusd when available.

    This is a bounded preflight fallback for offline clusters where the
    Omniverse extension registry cannot resolve `omni.flowusd`. It is enabled
    by default when the local kit exists, and can be disabled explicitly with
    BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD=0.
    """

    if not _minimal_kit_no_flowusd_enabled():
        return
    import omnigibson.simulator as simulator

    explicit_kit_path = os.environ.get("BEHAVIOR1K_MINIMAL_KIT_PATH")
    kit_path = (
        Path(explicit_kit_path).expanduser().resolve()
        if explicit_kit_path
        else _minimal_kit_no_flowusd_path()
    )
    if not kit_path.exists():
        raise RuntimeError(
            f"Requested BEHAVIOR1K minimal kit does not exist: {kit_path}"
        )
    simulator._embodiedai_minimal_kit_path = str(kit_path)
    simulator.m.KIT_FILES[(4, 1, 0)] = kit_path.name
    simulator.m.KIT_FILES[(4, 0, 0)] = kit_path.name
    simulator.m.KIT_FILES[(4, 5, 0)] = kit_path.name
    _patch_omnigibson_offline_extension_enables(simulator)


def _patch_omnigibson_offline_extension_enables(simulator: Any) -> None:
    if getattr(simulator, "_embodiedai_offline_extension_patch", False):
        return
    source = inspect.getsource(simulator._launch_app)
    replacements = {
        "kit_file = Path(__file__).parent / kit_file_name": (
            "kit_file = Path(_embodiedai_minimal_kit_path)"
        ),
        "kit_file_target = Path(exp_path) / kit_file_name": (
            "kit_file_target = Path(exp_path) / kit_file.name"
        ),
        'lazy.omni.isaac.core.utils.extensions.enable_extension("omni.flowusd")': (
            'log.warning("Skipping omni.flowusd extension enable for BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD preflight.")'
        ),
        'lazy.omni.isaac.core.utils.extensions.enable_extension("omni.particle.system.bundle")': (
            'log.warning("Skipping omni.particle.system.bundle extension enable for BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD preflight.")'
        ),
    }
    patched = source
    replacement_count = 0
    for old, new in replacements.items():
        if old in patched:
            replacement_count += patched.count(old)
            patched = patched.replace(old, new)
    if patched == source:
        simulator._embodiedai_offline_extension_patch = True
        simulator._embodiedai_offline_extension_patch_replacements = 0
        simulator._embodiedai_offline_extension_patch_note = (
            "No omni.flowusd or omni.particle.system.bundle enable calls were found in upstream _launch_app; "
            "continuing with the selected local kit file."
        )
        return
    exec(
        compile(patched, str(Path(simulator.__file__).resolve()), "exec"),
        simulator.__dict__,
    )
    simulator._embodiedai_offline_extension_patch = True
    simulator._embodiedai_offline_extension_patch_replacements = replacement_count


def _maybe_install_offline_omni_particle_stubs() -> None:
    """Provide import-only particle modules when the particle extension is disabled.

    OmniGibson v1.1.1 imports particle Core/Utils classes from
    `deprecated_utils` even for basic camera pose queries. On clusters where the
    Omniverse extension registry cannot resolve `omni.particle.system.bundle`,
    the live preflight disables that extension; this shim keeps the import path
    available without pretending particle simulation itself is supported.
    """

    if not _minimal_kit_no_flowusd_enabled():
        return
    if "omni.particle.system.core.scripts.core" in sys.modules:
        return

    class _OfflineParticleCore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._offline_stub_args = args
            self._offline_stub_kwargs = kwargs

    class _OfflineParticleUtils:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._offline_stub_args = args
            self._offline_stub_kwargs = kwargs

    package_names = [
        "omni.particle",
        "omni.particle.system",
        "omni.particle.system.core",
        "omni.particle.system.core.scripts",
    ]
    for name in package_names:
        module = sys.modules.setdefault(name, types.ModuleType(name))
        module.__dict__.setdefault("__path__", [])

    core_module = types.ModuleType("omni.particle.system.core.scripts.core")
    core_module.Core = _OfflineParticleCore
    utils_module = types.ModuleType("omni.particle.system.core.scripts.utils")
    utils_module.Utils = _OfflineParticleUtils
    sys.modules[core_module.__name__] = core_module
    sys.modules[utils_module.__name__] = utils_module

    parent_links = [
        ("omni", "particle", "omni.particle"),
        ("omni.particle", "system", "omni.particle.system"),
        ("omni.particle.system", "core", "omni.particle.system.core"),
        ("omni.particle.system.core", "scripts", "omni.particle.system.core.scripts"),
        ("omni.particle.system.core.scripts", "core", core_module.__name__),
        ("omni.particle.system.core.scripts", "utils", utils_module.__name__),
    ]
    for parent_name, attr, child_name in parent_links:
        parent = sys.modules.get(parent_name)
        child = sys.modules.get(child_name)
        if parent is not None and child is not None:
            setattr(parent, attr, child)


def _minimal_kit_no_flowusd_path() -> Path:
    asset_root = get_project_paths().external_assets("behavior1k") / "runtime"
    default_root = (
        get_project_paths().external_upstream("behavior1k") / "OmniGibson/omnigibson"
    )
    for candidate in (
        asset_root / "omnigibson_4_5_0_no_flowusd_no_xr.kit",
        default_root / "omnigibson_4_5_0_no_flowusd_no_xr.kit",
        default_root / "omnigibson_4_5_0_no_flowusd.kit",
    ):
        if candidate.exists():
            return candidate
    spec = _find_module_spec("omnigibson")
    if spec is not None and spec.submodule_search_locations:
        candidate = (
            Path(next(iter(spec.submodule_search_locations))) / "omnigibson_4_1_0.kit"
        )
        if candidate.exists():
            return candidate
    return asset_root / "omnigibson_4_5_0_no_flowusd_no_xr.kit"


def _find_module_spec(name: str, *, bootstrap_behavior1k: bool = False) -> Any:
    if bootstrap_behavior1k and name in {"omnigibson", "bddl"}:
        _configure_behavior1k_source_paths()
    try:
        return importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None


def _configure_behavior1k_source_paths(
    config: Behavior1KRuntimeConfig | None = None,
) -> JsonDict:
    """Expose repo-local BEHAVIOR-1K source checkouts when they are present.

    This does not make the simulator ready by itself; it only lets preflight
    reach the next real blocker, such as missing h5py, Isaac, assets, or GPU
    runtime, instead of failing at an avoidable PYTHONPATH boundary.
    """

    configured_root = (
        config.omnigibson_root
        if config is not None and config.omnigibson_root
        else os.getenv("BEHAVIOR1K_OMNIGIBSON_ROOT")
    )
    omnigibson_root = (
        Path(configured_root).expanduser()
        if configured_root
        else get_project_paths().external_upstream("behavior1k") / "OmniGibson"
    )
    bddl_root = Path(
        os.getenv("BEHAVIOR1K_BDDL_ROOT")
        or get_project_paths().external_upstream("behavior1k") / "bddl3"
    ).expanduser()
    inserted: list[str] = []
    missing: list[str] = []
    for path in (omnigibson_root, bddl_root):
        if path.exists():
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
                inserted.append(resolved)
        else:
            missing.append(str(path))
    return {
        "omnigibson_root": str(omnigibson_root),
        "bddl_root": str(bddl_root),
        "inserted": inserted,
        "missing": missing,
    }


def _minimal_kit_no_flowusd_enabled() -> bool:
    if _env_falsey("BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD"):
        return False
    if _env_truthy("BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD"):
        return True
    return _minimal_kit_no_flowusd_path().exists()


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "False").lower() in ("1", "true", "t", "yes", "y")


def _env_falsey(name: str) -> bool:
    return os.getenv(name, "").lower() in ("0", "false", "f", "no", "n")


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def summarize_data(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return {str(key): summarize_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return {
            "type": type(value).__name__,
            "length": len(value),
            "items": [summarize_data(item) for item in list(value)[:5]],
        }
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        payload = {
            "type": type(value).__name__,
            "shape": list(shape),
            "dtype": str(dtype),
        }
        payload.update(_compact_array_stats(value))
        return payload
    return {"type": type(value).__name__, "value": _to_builtin(value)}


def summarize_behavior1k_visual_runtime(obs: Any) -> JsonDict:
    sensor_summaries = summarize_behavior1k_sensors(obs)
    modalities = sorted({summary["modality"] for summary in sensor_summaries})
    return {
        "available": bool(sensor_summaries),
        "modalities": modalities,
        "camera_names": sorted(
            {summary["camera_name"] for summary in sensor_summaries}
        ),
        "sensor_count": len(sensor_summaries),
        "rgb_ready": "rgb" in modalities,
        "depth_ready": "depth" in modalities,
        "segmentation_ready": "segmentation" in modalities,
        "point_cloud_ready": "point_cloud" in modalities,
        "pose_ready": "pose" in modalities,
        "native_masks_preferred_over_sam": "segmentation" in modalities,
    }


def behavior1k_visual_evidence(
    obs: Any,
    *,
    assets: dict[str, JsonDict],
    observation_info: JsonDict | None = None,
    prompt: str | None = None,
    query: str | None = None,
    agent_context: JsonDict | None = None,
    camera_name: str | None = None,
    entity_name: str | None = None,
    instance_id: int | None = None,
    label_kind: str | None = None,
) -> JsonDict:
    context = agent_context or {}
    sensor_summaries = [
        summary
        for summary in summarize_behavior1k_sensors(obs)
        if camera_name is None or summary["camera_name"] == camera_name
    ]
    calibration = summarize_behavior1k_calibration(obs, camera_name=camera_name)
    id_to_entity = _behavior1k_segmentation_id_entity_map(observation_info)
    instances = extract_behavior1k_segmentation_instances(
        obs, camera_name=camera_name, id_to_entity=id_to_entity
    )
    _canonicalize_behavior1k_instance_entities(instances, assets)
    selected_instances = [
        instance
        for instance in instances
        if instance.get("entity_name") != "background"
        and (entity_name is None or instance.get("entity_name") == entity_name)
        and (instance_id is None or instance.get("instance_id") == instance_id)
        and (label_kind is None or instance.get("label_kind") == label_kind)
    ]
    groundings = [
        _behavior1k_depth_grounding(obs, instance) for instance in selected_instances
    ]
    asset_candidates = _behavior1k_asset_candidates(
        assets, query=query, agent_context=context
    )
    object_masks = _behavior1k_object_mask_summaries(instances, assets)
    visual_runtime = _behavior1k_visual_runtime_with_asset_poses(obs, assets)
    if (
        not groundings
        and entity_name is not None
        and visual_runtime.get("rgb_ready")
        and visual_runtime.get("depth_ready")
    ):
        fallback = _behavior1k_asset_pose_rgbd_grounding(
            obs,
            asset_name=entity_name,
            asset=assets.get(entity_name),
            sensor_summaries=sensor_summaries,
            camera_name=camera_name,
        )
        if fallback is not None:
            groundings.append(fallback)
    return {
        "source": "omnigibson_observation",
        "prompt": prompt,
        "query": query,
        "agent_context": context,
        "camera_name": camera_name,
        "visual_runtime": visual_runtime,
        "sensor_summaries": sensor_summaries,
        "calibration": calibration,
        "segmentation_instances": instances,
        "grounding_selector": {
            "camera_name": camera_name,
            "entity_name": entity_name,
            "instance_id": instance_id,
            "label_kind": label_kind,
            "selection_source": "caller_exact_selector"
            if any(
                value is not None for value in (entity_name, instance_id, label_kind)
            )
            else "all_native_regions",
            "query_used_for_selection": False,
        },
        "groundings": groundings,
        "object_masks": object_masks,
        "asset_candidates": asset_candidates,
        "agent_notes": {
            "native_segmentation_instances_are_agent_visible": bool(instances),
            "native_semantic_regions_are_agent_visible": any(
                instance.get("label_kind") == "seg_semantic" for instance in instances
            ),
            "depth_backed_groundings_are_agent_visible": any(
                grounding.get("depth", {}).get("valid_pixel_count", 0) > 0
                for grounding in groundings
            ),
            "object_masks_linked_to_scene_assets": bool(object_masks),
            "asset_pose_rgbd_fallback_groundings_are_agent_visible": any(
                grounding.get("label_kind") == "asset_pose_rgbd"
                for grounding in groundings
            ),
            "sam_needed": not bool(groundings)
            and any(summary["modality"] == "rgb" for summary in sensor_summaries),
        },
    }


def _behavior1k_asset_pose_rgbd_grounding(
    obs: Any,
    *,
    asset_name: str,
    asset: JsonDict | None,
    sensor_summaries: list[JsonDict] | None = None,
    camera_name: str | None = None,
) -> JsonDict | None:
    arrays = dict(_iter_observation_arrays(obs))
    available_cameras = sorted(
        {
            _camera_name_from_path(path, modality)
            for path in arrays
            for modality in (_behavior1k_modality(path),)
            if modality in {"rgb", "depth"}
        }
    )
    candidates = [camera_name] if camera_name is not None else available_cameras
    for candidate_camera in candidates:
        if candidate_camera is None:
            continue
        depth_key, depth = _behavior1k_camera_array(arrays, candidate_camera, "depth")
        rgb_key, rgb = _behavior1k_camera_array(arrays, candidate_camera, "rgb")
        depth_plane = _behavior1k_image_plane(depth)
        rgb_array = np.asarray(rgb) if rgb is not None else None
        if depth_plane is None or rgb_array is None or rgb_array.ndim < 2:
            continue
        valid_depth = np.asarray(depth_plane, dtype=np.float64)
        valid_depth = valid_depth[np.isfinite(valid_depth) & (valid_depth > 0)]
        if valid_depth.size <= 0:
            continue
        height, width = depth_plane.shape[:2]
        center_xy = [float(max(width - 1, 1) / 2.0), float(max(height - 1, 1) / 2.0)]
        rgb_summary: JsonDict | None = None
        if rgb_array.ndim >= 3 and rgb_array.shape[-1] >= 3:
            flat_rgb = rgb_array.reshape(-1, rgb_array.shape[-1])
            if flat_rgb.size:
                rgb_summary = {
                    "channel_mean": _to_builtin(
                        np.asarray(flat_rgb, dtype=np.float64).mean(axis=0)
                    ),
                    "channel_min": _to_builtin(np.asarray(flat_rgb).min(axis=0)),
                    "channel_max": _to_builtin(np.asarray(flat_rgb).max(axis=0)),
                }
        has_asset_pose = isinstance(asset, dict) and bool(asset.get("pose_world"))
        return {
            "source": (
                "omnigibson_native_rgbd_plus_scene_asset_pose"
                if has_asset_pose
                else "omnigibson_native_rgbd_frame"
            ),
            "camera_name": candidate_camera,
            "entity_name": asset_name,
            "label_kind": "asset_pose_rgbd" if has_asset_pose else "rgbd_frame",
            "label_id": None,
            "segmentation_source_key": None,
            "depth_source_key": depth_key,
            "rgb_source_key": rgb_key,
            "intrinsics_source_key": None,
            "pixel_count": int(height * width),
            "bbox_xyxy": [0, 0, int(width - 1), int(height - 1)],
            "center_xy": center_xy,
            "center_normalized_xy": [0.5, 0.5],
            "pose_world": deepcopy(asset.get("pose_world"))
            if isinstance(asset, dict)
            else None,
            "asset_category": asset.get("category")
            if isinstance(asset, dict)
            else None,
            "asset_model": asset.get("model") if isinstance(asset, dict) else None,
            "asset_prim_path": asset.get("prim_path")
            if isinstance(asset, dict)
            else None,
            "depth": {
                "valid_pixel_count": int(valid_depth.size),
                "min": float(valid_depth.min()),
                "median": float(np.median(valid_depth)),
                "max": float(valid_depth.max()),
            },
            "depth_median": float(np.median(valid_depth)),
            "camera_point_xyz": None,
            "camera_point_source": None,
            "rgb_region_summary": rgb_summary,
        }
    return _behavior1k_sensor_summary_rgbd_grounding(
        sensor_summaries or [],
        asset_name=asset_name,
        asset=asset,
        camera_name=camera_name,
    )


def _behavior1k_sensor_summary_rgbd_grounding(
    sensor_summaries: list[JsonDict],
    *,
    asset_name: str,
    asset: JsonDict | None,
    camera_name: str | None = None,
) -> JsonDict | None:
    by_camera: dict[str, dict[str, JsonDict]] = {}
    for summary in sensor_summaries:
        if not isinstance(summary, dict):
            continue
        summary_camera = str(summary.get("camera_name") or "default")
        if camera_name is not None and summary_camera != camera_name:
            continue
        modality = summary.get("modality")
        if modality in {"rgb", "depth"}:
            by_camera.setdefault(summary_camera, {})[str(modality)] = summary

    for candidate_camera in sorted(by_camera):
        pair = by_camera[candidate_camera]
        rgb_summary = pair.get("rgb")
        depth_summary = pair.get("depth")
        if not isinstance(rgb_summary, dict) or not isinstance(depth_summary, dict):
            continue

        depth_stats = (
            depth_summary.get("data_summary")
            if isinstance(depth_summary.get("data_summary"), dict)
            else {}
        )
        valid_pixel_count = _positive_int(depth_stats.get("size"))
        depth_shape = (
            depth_summary.get("shape")
            if isinstance(depth_summary.get("shape"), list)
            else []
        )
        rgb_shape = (
            rgb_summary.get("shape")
            if isinstance(rgb_summary.get("shape"), list)
            else []
        )
        height, width = (
            _behavior1k_image_hw_from_shape(depth_shape)
            or _behavior1k_image_hw_from_shape(rgb_shape)
            or (1, 1)
        )
        if valid_pixel_count is None:
            valid_pixel_count = int(max(height, 1) * max(width, 1))
        if valid_pixel_count <= 0:
            continue

        rgb_stats = (
            rgb_summary.get("data_summary")
            if isinstance(rgb_summary.get("data_summary"), dict)
            else {}
        )
        has_asset_pose = isinstance(asset, dict) and bool(asset.get("pose_world"))
        depth_mean = _optional_float(depth_stats.get("mean"))
        return {
            "source": (
                "omnigibson_native_rgbd_sensor_summary_plus_scene_asset_pose"
                if has_asset_pose
                else "omnigibson_native_rgbd_sensor_summary"
            ),
            "camera_name": candidate_camera,
            "entity_name": asset_name,
            "label_kind": "asset_pose_rgbd" if has_asset_pose else "rgbd_frame",
            "label_id": None,
            "segmentation_source_key": None,
            "depth_source_key": depth_summary.get("source_key"),
            "rgb_source_key": rgb_summary.get("source_key"),
            "intrinsics_source_key": None,
            "pixel_count": int(max(height, 1) * max(width, 1)),
            "bbox_xyxy": [0, 0, int(max(width - 1, 0)), int(max(height - 1, 0))],
            "center_xy": [
                float(max(width - 1, 1) / 2.0),
                float(max(height - 1, 1) / 2.0),
            ],
            "center_normalized_xy": [0.5, 0.5],
            "pose_world": deepcopy(asset.get("pose_world"))
            if isinstance(asset, dict)
            else None,
            "asset_category": asset.get("category")
            if isinstance(asset, dict)
            else None,
            "asset_model": asset.get("model") if isinstance(asset, dict) else None,
            "asset_prim_path": asset.get("prim_path")
            if isinstance(asset, dict)
            else None,
            "depth": {
                "valid_pixel_count": int(valid_pixel_count),
                "min": _optional_float(depth_stats.get("min")),
                "median": depth_mean,
                "max": _optional_float(depth_stats.get("max")),
            },
            "depth_median": depth_mean,
            "camera_point_xyz": None,
            "camera_point_source": None,
            "rgb_region_summary": {
                "summary_source": "native_sensor_summary",
                "channel_mean": _to_builtin(rgb_stats.get("mean")),
                "channel_min": _to_builtin(rgb_stats.get("min")),
                "channel_max": _to_builtin(rgb_stats.get("max")),
            },
        }
    return None


def _behavior1k_image_hw_from_shape(shape: list[Any]) -> tuple[int, int] | None:
    if len(shape) < 2:
        return None
    try:
        height = int(shape[0])
        width = int(shape[1])
    except (TypeError, ValueError):
        return None
    if height <= 0 or width <= 0:
        return None
    return height, width


def _positive_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _behavior1k_depth_grounding(obs: Any, instance: JsonDict) -> JsonDict:
    arrays = dict(_iter_observation_arrays(obs))
    source_key = str(instance.get("source_key", ""))
    source_camera = str(instance.get("camera_name", "default"))
    segmentation = _behavior1k_image_plane(arrays.get(source_key))
    label_id = instance.get("instance_id")
    mask = segmentation == label_id if segmentation is not None else None
    depth_key, depth = _behavior1k_camera_array(arrays, source_camera, "depth")
    rgb_key, rgb = _behavior1k_camera_array(arrays, source_camera, "rgb")
    depth_plane = _behavior1k_image_plane(depth)
    valid_depth: np.ndarray[Any, Any] = np.asarray([], dtype=np.float64)
    if mask is not None and depth_plane is not None and mask.shape == depth_plane.shape:
        values = np.asarray(depth_plane, dtype=np.float64)[mask]
        valid_depth = values[np.isfinite(values) & (values > 0)]

    center_xy = [float(value) for value in instance.get("center_xy", [0.0, 0.0])]
    image_height, image_width = mask.shape if mask is not None else (0, 0)
    center_normalized = [
        center_xy[0] / max(image_width - 1, 1),
        center_xy[1] / max(image_height - 1, 1),
    ]
    depth_median = float(np.median(valid_depth)) if valid_depth.size else None
    intrinsics_key, intrinsics = _behavior1k_camera_calibration_array(
        arrays, source_camera, "intrinsic"
    )
    camera_point = _behavior1k_unproject_pixel(center_xy, depth_median, intrinsics)

    rgb_summary: JsonDict | None = None
    rgb_array = np.asarray(rgb) if rgb is not None else None
    if (
        mask is not None
        and rgb_array is not None
        and rgb_array.ndim >= 3
        and rgb_array.shape[:2] == mask.shape
    ):
        region = rgb_array[mask]
        if region.size:
            rgb_summary = {
                "channel_mean": _to_builtin(
                    np.asarray(region, dtype=np.float64).mean(axis=0)
                ),
                "channel_min": _to_builtin(np.asarray(region).min(axis=0)),
                "channel_max": _to_builtin(np.asarray(region).max(axis=0)),
            }

    return {
        "source": "omnigibson_native_visual_observation",
        "camera_name": source_camera,
        "entity_name": instance.get("entity_name"),
        "label_kind": instance.get("label_kind"),
        "label_id": label_id,
        "segmentation_source_key": source_key,
        "depth_source_key": depth_key,
        "rgb_source_key": rgb_key,
        "intrinsics_source_key": intrinsics_key,
        "pixel_count": instance.get("pixel_count"),
        "bbox_xyxy": deepcopy(instance.get("bbox_xyxy")),
        "center_xy": center_xy,
        "center_normalized_xy": center_normalized,
        "depth": {
            "valid_pixel_count": int(valid_depth.size),
            "min": float(valid_depth.min()) if valid_depth.size else None,
            "median": depth_median,
            "max": float(valid_depth.max()) if valid_depth.size else None,
        },
        "depth_median": depth_median,
        "camera_point_xyz": camera_point,
        "camera_point_source": "depth_and_camera_intrinsics"
        if camera_point is not None
        else None,
        "rgb_region_summary": rgb_summary,
    }


def _behavior1k_image_plane(value: Any) -> np.ndarray[Any, Any] | None:
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return array if array.ndim == 2 else None


def _behavior1k_camera_array(
    arrays: dict[str, Any], camera_name: str, modality: str
) -> tuple[str | None, Any | None]:
    for path, value in arrays.items():
        if (
            _behavior1k_modality(path) == modality
            and _camera_name_from_path(path, modality) == camera_name
        ):
            return path, value
    return None, None


def _behavior1k_camera_calibration_array(
    arrays: dict[str, Any], camera_name: str, marker: str
) -> tuple[str | None, Any | None]:
    for path, value in arrays.items():
        if (
            marker in path.lower()
            and _camera_name_from_calibration_path(path) == camera_name
        ):
            return path, value
    return None, None


def _behavior1k_unproject_pixel(
    center_xy: list[float], depth: float | None, intrinsics: Any
) -> list[float] | None:
    if depth is None or intrinsics is None:
        return None
    try:
        matrix = np.asarray(intrinsics, dtype=np.float64)
        if matrix.shape != (3, 3):
            return None
        ray = np.linalg.solve(
            matrix, np.asarray([center_xy[0], center_xy[1], 1.0], dtype=np.float64)
        )
        return _to_builtin(ray * depth)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def summarize_behavior1k_sensors(obs: Any) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for path, value in _iter_observation_arrays(obs):
        modality = _behavior1k_modality(path)
        if modality is None:
            continue
        shape = [int(dim) for dim in getattr(value, "shape", [])]
        summaries.append(
            {
                "source_key": path,
                "camera_name": _camera_name_from_path(path, modality),
                "modality": modality,
                "shape": shape,
                "dtype": str(getattr(value, "dtype", "")),
                "data_summary": _compact_array_stats(value),
            }
        )
    return summaries


def summarize_behavior1k_calibration(
    obs: Any, camera_name: str | None = None
) -> list[JsonDict]:
    calibration: list[JsonDict] = []
    markers = (
        "intrinsic",
        "extrinsic",
        "camera_matrix",
        "projection_matrix",
        "view_matrix",
    )
    for path, value in _iter_observation_arrays(obs):
        if not any(marker in path.lower() for marker in markers):
            continue
        source_camera = _camera_name_from_calibration_path(path)
        if camera_name is not None and source_camera != camera_name:
            continue
        calibration.append(
            {
                "source_key": path,
                "camera_name": source_camera,
                "shape": [int(dim) for dim in getattr(value, "shape", [])],
                "dtype": str(getattr(value, "dtype", "")),
                "data": _to_builtin(value),
            }
        )
    return sorted(calibration, key=lambda item: item["source_key"])


def _camera_name_from_calibration_path(path: str) -> str:
    tokens = path.replace("[", ".").replace("]", "").split(".")
    markers = (
        "intrinsic",
        "extrinsic",
        "camera_matrix",
        "projection_matrix",
        "view_matrix",
    )
    prefix = [
        token
        for token in tokens
        if token and not any(marker in token.lower() for marker in markers)
    ]
    return ".".join(prefix) or "default"


def _behavior1k_visual_runtime_with_asset_poses(
    obs: Any, assets: dict[str, JsonDict]
) -> JsonDict:
    visual_runtime = summarize_behavior1k_visual_runtime(obs)
    if any(
        isinstance(payload, dict) and payload.get("pose_world")
        for payload in assets.values()
    ):
        visual_runtime["pose_ready"] = True
        visual_runtime["pose_source"] = "omnigibson_scene_asset_registry"
    return visual_runtime


def extract_behavior1k_segmentation_instances(
    obs: Any,
    *,
    camera_name: str | None = None,
    min_pixel_count: int = 1,
    id_to_entity: dict[tuple[str | None, str, int], str] | None = None,
) -> list[JsonDict]:
    instances: list[JsonDict] = []
    id_to_entity = id_to_entity or {}
    for path, value in _iter_observation_arrays(obs):
        modality = _behavior1k_modality(path)
        if modality != "segmentation":
            continue
        source_camera = _camera_name_from_path(path, "segmentation")
        if camera_name is not None and source_camera != camera_name:
            continue
        label_kind = _behavior1k_segmentation_label_kind(path)
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.ndim >= 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            continue
        for raw_id in np.unique(arr):
            mask = arr == raw_id
            pixel_count = int(mask.sum())
            if pixel_count < min_pixel_count:
                continue
            ys, xs = np.where(mask)
            if len(xs) == 0 or len(ys) == 0:
                continue
            label_id = _to_builtin(raw_id)
            entity_name = "background" if label_id == 0 else None
            try:
                lookup_id = int(raw_id)
            except (TypeError, ValueError):
                lookup_id = None
            if lookup_id is not None:
                entity_name = (
                    id_to_entity.get((source_camera, label_kind, lookup_id))
                    or id_to_entity.get((None, label_kind, lookup_id))
                    or entity_name
                )
            instances.append(
                {
                    "instance_id": label_id,
                    "entity_name": entity_name,
                    "label_kind": label_kind,
                    "source_key": path,
                    "camera_name": source_camera,
                    "pixel_count": pixel_count,
                    "bbox_xyxy": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max() + 1),
                        int(ys.max() + 1),
                    ],
                    "center_xy": [float(xs.mean()), float(ys.mean())],
                }
            )
    return sorted(
        instances,
        key=lambda item: (-int(item["pixel_count"]), str(item["instance_id"])),
    )


def _behavior1k_segmentation_label_kind(path: str) -> str:
    lower = path.lower()
    if "semantic" in lower:
        return "seg_semantic"
    if "instance" in lower or "seg" in lower or "mask" in lower:
        return "seg_instance"
    return "segmentation"


def _behavior1k_segmentation_id_entity_map(
    observation_info: JsonDict | None,
) -> dict[tuple[str | None, str, int], str]:
    if not isinstance(observation_info, dict):
        return {}
    root = observation_info.get("obs_info", observation_info)
    if not isinstance(root, dict):
        return {}
    mapping: dict[tuple[str | None, str, int], str] = {}

    def visit(value: Any, path: list[str]) -> None:
        if not isinstance(value, dict):
            return
        for key, item in value.items():
            key_str = str(key)
            label_kind = _behavior1k_segmentation_label_kind(key_str)
            if label_kind in {"seg_instance", "seg_semantic"} and isinstance(
                item, dict
            ):
                camera = ".".join(path) if path else None
                for raw_id, raw_name in item.items():
                    try:
                        label_id = int(raw_id)
                    except (TypeError, ValueError):
                        continue
                    entity_name = _behavior1k_obs_info_entity_name(raw_name)
                    if entity_name:
                        mapping[(camera, label_kind, label_id)] = entity_name
                        mapping[(None, label_kind, label_id)] = entity_name
                continue
            visit(item, [*path, key_str])

    visit(root, [])
    return mapping


def _behavior1k_obs_info_entity_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        raw = value.get("value")
        if isinstance(raw, str):
            return raw
        raw_name = value.get("name") or value.get("entity_name")
        if isinstance(raw_name, str):
            return raw_name
    return None


def _behavior1k_object_mask_summaries(
    instances: list[JsonDict], assets: dict[str, JsonDict]
) -> list[JsonDict]:
    summaries: list[JsonDict] = []
    for instance in instances:
        entity_name = instance.get("entity_name")
        if not isinstance(entity_name, str) or entity_name == "background":
            continue
        asset = assets.get(entity_name)
        if asset is None:
            asset = _match_behavior1k_asset_by_entity_name(entity_name, assets)
        summary = {
            "entity_name": entity_name,
            "instance_id": instance.get("instance_id"),
            "label_kind": instance.get("label_kind"),
            "camera_name": instance.get("camera_name"),
            "source_key": instance.get("source_key"),
            "pixel_count": instance.get("pixel_count"),
            "bbox_xyxy": instance.get("bbox_xyxy"),
            "center_xy": instance.get("center_xy"),
            "pose_world": deepcopy(asset.get("pose_world"))
            if isinstance(asset, dict)
            else None,
            "asset_category": asset.get("category")
            if isinstance(asset, dict)
            else None,
            "asset_model": asset.get("model") if isinstance(asset, dict) else None,
            "asset_prim_path": asset.get("prim_path")
            if isinstance(asset, dict)
            else None,
        }
        summaries.append(summary)
    return summaries


def _canonicalize_behavior1k_instance_entities(
    instances: list[JsonDict], assets: dict[str, JsonDict]
) -> None:
    """Join native instance-ID prim paths back to the scene asset registry."""

    for instance in instances:
        entity_name = instance.get("entity_name")
        if not isinstance(entity_name, str) or entity_name == "background":
            continue
        matched_name = _match_behavior1k_asset_name(entity_name, assets)
        if matched_name is not None:
            instance["native_entity_name"] = entity_name
            instance["entity_name"] = matched_name


def _match_behavior1k_asset_name(
    entity_name: str, assets: dict[str, JsonDict]
) -> str | None:
    if entity_name in assets:
        return entity_name
    normalized = entity_name.rstrip("/")
    for name, payload in assets.items():
        prim_path = payload.get("prim_path") if isinstance(payload, dict) else None
        if isinstance(prim_path, str):
            normalized_prim = prim_path.rstrip("/")
            if normalized == normalized_prim or normalized.startswith(
                f"{normalized_prim}/"
            ):
                return name
        if name.startswith(f"{entity_name}_") or entity_name.startswith(f"{name}_"):
            return name
    return None


def _match_behavior1k_asset_by_entity_name(
    entity_name: str, assets: dict[str, JsonDict]
) -> JsonDict | None:
    matched_name = _match_behavior1k_asset_name(entity_name, assets)
    if matched_name is not None:
        return assets[matched_name]
    category = entity_name.rsplit("_", 2)[0]
    for payload in assets.values():
        if payload.get("category") == category:
            return payload
    return None


def _iter_observation_arrays(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_iter_observation_arrays(item, path))
        return items
    if isinstance(value, (list, tuple)):
        items = []
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]" if prefix else f"[{index}]"
            items.extend(_iter_observation_arrays(item, path))
        return items
    if getattr(value, "shape", None) is not None:
        return [(prefix or type(value).__name__, value)]
    return []


def _behavior1k_modality(path: str) -> str | None:
    lower = path.lower()
    if any(token in lower for token in ("seg", "mask", "semantic", "instance")):
        return "segmentation"
    if "depth" in lower:
        return "depth"
    if any(token in lower for token in ("pointcloud", "point_cloud", "pcd", "xyz")):
        return "point_cloud"
    if any(token in lower for token in ("pose", "proprioception", "joint", "eef")):
        return "pose"
    if any(token in lower for token in ("rgb", "image", "color")):
        return "rgb"
    return None


def _camera_name_from_path(path: str, modality: str) -> str:
    normalized = path.replace("[", ".").split(".")
    modality_tokens = {
        "rgb": {"rgb", "image", "color"},
        "depth": {"depth"},
        "segmentation": {"seg", "segmentation", "mask", "semantic", "instance"},
        "point_cloud": {"pointcloud", "point_cloud", "pcd", "xyz"},
        "pose": {"pose", "proprioception", "joint", "eef"},
    }.get(modality, set())
    prefix = []
    for token in normalized:
        clean = token.strip("]").lower()
        if clean in modality_tokens or any(
            marker in clean for marker in modality_tokens
        ):
            break
        prefix.append(token.strip("]"))
    return ".".join(part for part in prefix if part) or "default"


def _behavior1k_asset_candidates(
    assets: dict[str, JsonDict],
    query: str | None,
    agent_context: JsonDict,
) -> list[JsonDict]:
    del query, agent_context
    return [
        {**deepcopy(payload), "name": name} for name, payload in sorted(assets.items())
    ]


def _module_imports(
    name: str, *, bootstrap_behavior1k: bool = True
) -> tuple[bool, Any | None, str | None]:
    if bootstrap_behavior1k and name in {"omnigibson", "bddl"}:
        _configure_behavior1k_source_paths()
    try:
        module = __import__(name)
        return True, module, None
    except Exception as exc:
        return False, None, f"{type(exc).__name__}: {exc}"


def _split_reset_result(result: Any) -> tuple[Any, JsonDict]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info or {})
    return result if result is not None else {}, {}


def _behavior1k_read_observation(env: Any) -> Any | None:
    """Read the current observation through public/common env accessors."""

    if env is None:
        return None
    candidates = (
        "get_obs",
        "get_observation",
        "get_observations",
        "_get_obs",
        "_get_observation",
        "_get_observations",
    )
    for name in candidates:
        getter = getattr(env, name, None)
        if not callable(getter):
            getter = getattr(_unwrap_env(env), name, None)
        if not callable(getter):
            continue
        try:
            return getter()
        except TypeError:
            continue
    return None


def _split_step_result(
    result: Any,
) -> tuple[Any, float | None, bool | None, bool | None, JsonDict]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return (
            obs,
            _optional_float(reward),
            bool(terminated),
            bool(truncated),
            dict(info or {}),
        )
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, _optional_float(reward), bool(done), False, dict(info or {})
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, None, None, None, dict(info or {})
    return result if result is not None else {}, None, None, None, {}


def behavior1k_action_schema(env: Any) -> JsonDict:
    action_space = (
        getattr(_unwrap_env(env), "action_space", None) if env is not None else None
    )
    if action_space is None and env is not None:
        action_space = getattr(env, "action_space", None)
    if action_space is None:
        return {"available": False}
    return _space_schema(action_space)


def _behavior1k_robots(env: Any) -> dict[str, Any]:
    if env is None:
        return {}
    unwrapped = _unwrap_env(env)
    robots = getattr(unwrapped, "robots", None)
    if robots is None:
        scene = getattr(unwrapped, "scene", None)
        robots = getattr(scene, "robots", None) if scene is not None else None
    if isinstance(robots, dict):
        return {str(name): robot for name, robot in robots.items()}
    if robots is None:
        return {}
    return {
        str(getattr(robot, "name", f"robot_{index}")): robot
        for index, robot in enumerate(robots)
    }


def _behavior1k_control_schema(env: Any) -> JsonDict:
    result: JsonDict = {}
    for robot_name, robot in _behavior1k_robots(env).items():
        payload = _asset_payload(robot)
        payload["action_dim"] = _optional_int(getattr(robot, "action_dim", None))
        payload["action_normalize"] = _to_builtin(
            getattr(robot, "_action_normalize", None)
        )
        controllers = (
            getattr(robot, "controllers", None)
            or getattr(robot, "_controllers", None)
            or {}
        )
        action_indices = getattr(robot, "controller_action_idx", {})
        controller_payload: JsonDict = {}
        for name, controller in controllers.items():
            indices = (
                action_indices.get(name, []) if isinstance(action_indices, dict) else []
            )
            item: JsonDict = {
                "type": type(controller).__name__,
                "action_indices": _index_list(indices),
                "command_dim": _optional_int(getattr(controller, "command_dim", None)),
            }
            for attr in ("mode", "motor_type", "use_delta_commands", "control_type"):
                if hasattr(controller, attr):
                    item[attr] = _to_builtin(getattr(controller, attr))
            for attr in ("command_input_limits", "command_output_limits"):
                if hasattr(controller, attr):
                    item[attr] = summarize_data(getattr(controller, attr))
            item["position_command_scale"] = _behavior1k_controller_position_scale(
                controller
            )
            controller_payload[str(name)] = item
        payload["controllers"] = controller_payload
        arm_names = getattr(robot, "arm_names", []) or []
        eef_poses: JsonDict = {}
        for arm_name in arm_names:
            getter = getattr(robot, "get_eef_pose", None)
            relative_getter = getattr(robot, "get_relative_eef_pose", None)
            try:
                if callable(getter):
                    pos, quat = getter(arm=arm_name)
                    eef_poses[str(arm_name)] = {
                        "pose_world": _to_builtin(list(pos) + list(quat))
                    }
                if callable(relative_getter):
                    pos, quat = relative_getter(arm=arm_name)
                    eef_poses.setdefault(str(arm_name), {})["pose_robot"] = _to_builtin(
                        list(pos) + list(quat)
                    )
            except Exception as exc:
                eef_poses[str(arm_name)] = {"error": f"{type(exc).__name__}: {exc}"}
        payload["end_effectors"] = eef_poses
        finger_link_poses: JsonDict = {}
        try:
            finger_links_by_arm = getattr(robot, "finger_links", {}) or {}
        except Exception as exc:
            finger_links_by_arm = {}
            finger_link_poses["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(finger_links_by_arm, dict):
            for arm_name, finger_links in finger_links_by_arm.items():
                arm_payload: list[JsonDict] = []
                for link in list(finger_links or []):
                    item: JsonDict = {
                        "name": str(getattr(link, "name", "")),
                        "body_name": str(getattr(link, "body_name", "")),
                        "prim_path": str(getattr(link, "prim_path", "")),
                    }
                    getter = getattr(link, "get_position_orientation", None)
                    try:
                        if callable(getter):
                            pos, quat = getter()
                            item["pose_world"] = _to_builtin(list(pos) + list(quat))
                    except Exception as exc:
                        item["error"] = f"{type(exc).__name__}: {exc}"
                    arm_payload.append(item)
                finger_link_poses[str(arm_name)] = arm_payload
        payload["finger_links"] = finger_link_poses
        result[robot_name] = payload
    return result


def _behavior1k_controller_action(
    env: Any, robot_name: str, commands: JsonDict
) -> tuple[Any, JsonDict]:
    if not isinstance(commands, dict) or not commands:
        raise TypeError(
            "commands must be a non-empty mapping of native controller names to numeric values"
        )
    robots = _behavior1k_robots(env)
    if robot_name not in robots:
        raise KeyError(f"unknown robot {robot_name!r}; available={sorted(robots)}")
    robot = robots[robot_name]
    controllers = (
        getattr(robot, "controllers", None)
        or getattr(robot, "_controllers", None)
        or {}
    )
    action_indices = getattr(robot, "controller_action_idx", {})
    unknown = sorted(set(commands) - set(controllers))
    if unknown:
        raise KeyError(
            f"unknown controllers {unknown}; available={sorted(controllers)}"
        )
    action_dim = int(getattr(robot, "action_dim", 0) or 0)
    if action_dim <= 0:
        all_indices = (
            [_index_list(indices) for indices in action_indices.values()]
            if isinstance(action_indices, dict)
            else []
        )
        action_dim = 1 + max(
            (index for indices in all_indices for index in indices), default=-1
        )
    if action_dim <= 0:
        raise RuntimeError("robot does not expose action_dim or controller_action_idx")
    action = np.zeros(action_dim, dtype=np.float32)
    control_dict = None
    get_control_dict = getattr(robot, "get_control_dict", None)
    if callable(get_control_dict):
        try:
            control_dict = get_control_dict()
        except Exception:
            control_dict = None
    action_space = getattr(_unwrap_env(env), "action_space", None) or getattr(
        env, "action_space", None
    )
    spaces = getattr(action_space, "spaces", None)
    action_space_keys = (
        sorted(str(key) for key in spaces) if isinstance(spaces, dict) else []
    )
    evidence: JsonDict = {
        "robot_name": robot_name,
        "commands": {},
        "native_controller_index_mapping": True,
        "action_space_type": type(action_space).__name__
        if action_space is not None
        else None,
        "action_space_keys": action_space_keys,
    }
    for name, controller in controllers.items():
        indices = (
            _index_list(action_indices.get(name, []))
            if isinstance(action_indices, dict)
            else []
        )
        if not indices:
            continue
        if name in commands:
            values = np.asarray(commands[name], dtype=np.float32).reshape(-1)
            source = "caller"
        else:
            compute_no_op = getattr(controller, "compute_no_op_action", None)
            if callable(compute_no_op) and control_dict is not None:
                try:
                    values = np.asarray(
                        _to_builtin(compute_no_op(control_dict)), dtype=np.float32
                    ).reshape(-1)
                    source = "native_controller.compute_no_op_action"
                except Exception:
                    values = np.zeros(len(indices), dtype=np.float32)
                    source = "zero_fallback"
            else:
                values = np.zeros(len(indices), dtype=np.float32)
                source = "zero_fallback"
        if len(values) != len(indices):
            raise ValueError(
                f"controller {name!r} requires {len(indices)} values, received {len(values)}"
            )
        action[indices] = values
        evidence["commands"][str(name)] = {
            "action_indices": indices,
            "values": _to_builtin(values),
            "source": source,
            "controller_type": type(controller).__name__,
        }
    wrapped_action: Any = {robot_name: action} if isinstance(spaces, dict) else action
    evidence["wrapped_action_kind"] = (
        "dict_by_robot_name" if isinstance(spaces, dict) else "array"
    )
    evidence["full_action_summary"] = summarize_data(wrapped_action)
    return wrapped_action, evidence


def _behavior1k_control_response(
    before: JsonDict, after: JsonDict, *, robot_name: str
) -> JsonDict:
    before_robot = before.get(robot_name) if isinstance(before, dict) else None
    after_robot = after.get(robot_name) if isinstance(after, dict) else None
    if not isinstance(before_robot, dict) or not isinstance(after_robot, dict):
        return {
            "robot_name": robot_name,
            "available": False,
            "reason": "robot_state_missing",
        }

    robot_delta = _behavior1k_pose_delta(
        before_robot.get("pose_world"), after_robot.get("pose_world")
    )
    end_effectors: JsonDict = {}
    before_eefs = (
        before_robot.get("end_effectors")
        if isinstance(before_robot.get("end_effectors"), dict)
        else {}
    )
    after_eefs = (
        after_robot.get("end_effectors")
        if isinstance(after_robot.get("end_effectors"), dict)
        else {}
    )
    for name in sorted(set(before_eefs) | set(after_eefs)):
        before_pose = (
            before_eefs.get(name, {}).get("pose_world")
            if isinstance(before_eefs.get(name), dict)
            else None
        )
        after_pose = (
            after_eefs.get(name, {}).get("pose_world")
            if isinstance(after_eefs.get(name), dict)
            else None
        )
        end_effectors[str(name)] = _behavior1k_pose_delta(before_pose, after_pose)

    translation_values = [robot_delta.get("translation_distance")]
    translation_values.extend(
        delta.get("translation_distance")
        for delta in end_effectors.values()
        if isinstance(delta, dict)
    )
    moved = any(
        isinstance(value, (int, float)) and float(value) > 1e-5
        for value in translation_values
    )
    return {
        "robot_name": robot_name,
        "available": True,
        "robot_pose_delta": robot_delta,
        "end_effector_pose_deltas": end_effectors,
        "any_pose_changed": moved,
        "change_threshold_m": 1e-5,
    }


def _behavior1k_resolve_base_command_mode(
    command_mode: str, controller: JsonDict
) -> str:
    requested = str(command_mode or "auto").strip().lower()
    if requested in {"linear_angular", "lin_ang", "linear_velocity_angular_velocity"}:
        return "linear_angular"
    if requested in {"differential_wheel", "left_right_wheel", "wheel_velocity"}:
        return "left_right_wheel"
    if requested != "auto":
        raise ValueError(f"unsupported_base_command_mode:{command_mode}")
    # OmniGibson's DifferentialDriveController accepts normalized
    # [linear_velocity, angular_velocity] commands and internally converts them
    # to wheel velocities. Left/right wheel commands remain available through an
    # explicit command_mode for runtimes whose native interface actually expects
    # wheel velocities.
    return "linear_angular"


def _behavior1k_rotate_vector_xyzw(quaternion: Any, vector: Any) -> list[float]:
    quat = np.asarray(_to_builtin(quaternion), dtype=np.float64).reshape(-1)
    direction = np.asarray(_to_builtin(vector), dtype=np.float64).reshape(-1)
    if quat.size < 4 or direction.size < 3:
        raise ValueError("xyzw_quaternion_and_xyz_vector_required")
    x, y, z, w = (float(value) for value in quat[:4])
    vx, vy, vz = (float(value) for value in direction[:3])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _behavior1k_interaction_surface_standoff(
    target_pose: Any,
    *,
    standoff: float = 0.84,
) -> tuple[list[float], list[float]]:
    pose = np.asarray(_to_builtin(target_pose), dtype=np.float64).reshape(-1)
    if pose.size < 7:
        raise ValueError("interaction_site_pose_requires_xyz_and_xyzw")
    surface_normal = _behavior1k_rotate_vector_xyzw(pose[3:7], [0.0, 1.0, 0.0])
    xy_norm = max(math.hypot(surface_normal[0], surface_normal[1]), 1e-6)
    normal_xy = [surface_normal[0] / xy_norm, surface_normal[1] / xy_norm]
    return (
        [
            float(pose[0]) - float(standoff) * normal_xy[0],
            float(pose[1]) - float(standoff) * normal_xy[1],
            math.atan2(normal_xy[1], normal_xy[0]),
        ],
        [normal_xy[0], normal_xy[1], float(surface_normal[2])],
    )


def _behavior1k_live_toggle_target(
    obj: Any, *, standoff: float = 0.84
) -> JsonDict | None:
    """Read the current public toggle-link pose without consulting task state.

    The interaction link is ordinary scene geometry exposed by OmniGibson.  It
    may move while the robot navigates, so physical execution must not retain a
    reset-time pose as if it were immutable.
    """

    sites = _behavior1k_object_interaction_sites(obj, state_name="ToggledOn")
    if not sites:
        return None
    target_pose = sites[0].get("pose_world")
    if not isinstance(target_pose, list) or len(target_pose) < 7:
        return None
    try:
        standoff_pose, surface_normal = _behavior1k_interaction_surface_standoff(
            target_pose,
            standoff=standoff,
        )
    except (TypeError, ValueError):
        return None
    return {
        "interaction_site": deepcopy(sites[0]),
        "target_pose": [float(value) for value in target_pose[:7]],
        "surface_normal": surface_normal,
        "standoff_pose": standoff_pose,
        "standoff_distance": float(standoff),
        "source": "public_native_object_state_link",
        "verifier_feedback_used": False,
    }


def _behavior1k_public_object_aabb(obj: Any) -> JsonDict | None:
    """Return a scene object's public axis-aligned bounds when available."""

    values: dict[str, np.ndarray] = {}
    for attr in ("aabb_center", "aabb_extent"):
        raw = getattr(obj, attr, None)
        try:
            raw = raw() if callable(raw) else raw
            array = np.asarray(_to_builtin(raw), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            return None
        if array.size < 3 or not np.all(np.isfinite(array[:3])):
            return None
        values[attr] = array[:3]
    extent = values["aabb_extent"]
    if np.any(extent <= 0.0):
        return None
    return {
        "center": values["aabb_center"].tolist(),
        "extent": extent.tolist(),
        "minimum": (values["aabb_center"] - extent / 2.0).tolist(),
        "maximum": (values["aabb_center"] + extent / 2.0).tolist(),
    }


def _behavior1k_segment_intersects_aabb_xy(
    start_xy: Any,
    end_xy: Any,
    minimum_xy: Any,
    maximum_xy: Any,
) -> bool:
    """Liang-Barsky segment test against a closed 2-D AABB."""

    start = np.asarray(_to_builtin(start_xy), dtype=np.float64).reshape(-1)[:2]
    end = np.asarray(_to_builtin(end_xy), dtype=np.float64).reshape(-1)[:2]
    minimum = np.asarray(_to_builtin(minimum_xy), dtype=np.float64).reshape(-1)[:2]
    maximum = np.asarray(_to_builtin(maximum_xy), dtype=np.float64).reshape(-1)[:2]
    if min(start.size, end.size, minimum.size, maximum.size) < 2:
        return False
    direction = end - start
    lower_t = 0.0
    upper_t = 1.0
    for axis in range(2):
        if abs(float(direction[axis])) <= 1e-12:
            if float(start[axis]) < float(minimum[axis]) or float(start[axis]) > float(
                maximum[axis]
            ):
                return False
            continue
        first = (float(minimum[axis]) - float(start[axis])) / float(direction[axis])
        second = (float(maximum[axis]) - float(start[axis])) / float(direction[axis])
        enter, leave = sorted((first, second))
        lower_t = max(lower_t, enter)
        upper_t = min(upper_t, leave)
        if lower_t > upper_t:
            return False
    return upper_t >= 0.0 and lower_t <= 1.0


def _behavior1k_support_aware_navigation_path(
    env: Any,
    primary_obj: Any,
    robot_pose: Any,
    target_pose_xyyaw: Any,
    interaction_pose: Any,
    *,
    base_clearance: float = 0.40,
    corner_margin: float = 0.12,
    support_top_tolerance: float = 0.25,
) -> JsonDict:
    """Plan a shortest public-AABB detour around the target's support object."""

    robot = np.asarray(_to_builtin(robot_pose), dtype=np.float64).reshape(-1)
    target = np.asarray(_to_builtin(target_pose_xyyaw), dtype=np.float64).reshape(-1)
    interaction = np.asarray(_to_builtin(interaction_pose), dtype=np.float64).reshape(
        -1
    )
    direct = {
        "support_aware": True,
        "support_found": False,
        "direct_path_clear": True,
        "waypoints": [],
        "base_clearance": float(base_clearance),
    }
    if robot.size < 2 or target.size < 3 or interaction.size < 3:
        return direct
    scene = getattr(_unwrap_env(env), "scene", None)
    objects = getattr(scene, "objects", None) or getattr(scene, "_objects", None) or []
    iterable = list(objects.values()) if isinstance(objects, dict) else list(objects)
    supports: list[tuple[float, str, Any, JsonDict]] = []
    for obj in iterable:
        if obj is primary_obj:
            continue
        bounds = _behavior1k_public_object_aabb(obj)
        if bounds is None:
            continue
        minimum = bounds["minimum"]
        maximum = bounds["maximum"]
        contains_xy = (
            float(minimum[0]) - 0.03
            <= float(interaction[0])
            <= float(maximum[0]) + 0.03
            and float(minimum[1]) - 0.03
            <= float(interaction[1])
            <= float(maximum[1]) + 0.03
        )
        top_delta = abs(float(interaction[2]) - float(maximum[2]))
        if contains_xy and top_delta <= float(support_top_tolerance):
            supports.append((top_delta, str(getattr(obj, "name", "")), obj, bounds))
    if not supports:
        return direct
    _delta, support_name, _support_obj, bounds = min(
        supports, key=lambda item: (item[0], item[1])
    )
    minimum = np.asarray(bounds["minimum"][:2], dtype=np.float64) - float(
        base_clearance
    )
    maximum = np.asarray(bounds["maximum"][:2], dtype=np.float64) + float(
        base_clearance
    )
    result: JsonDict = {
        **direct,
        "support_found": True,
        "support_name": support_name,
        "support_aabb": deepcopy(bounds),
        "expanded_minimum_xy": minimum.tolist(),
        "expanded_maximum_xy": maximum.tolist(),
    }
    if not _behavior1k_segment_intersects_aabb_xy(
        robot[:2], target[:2], minimum, maximum
    ):
        return result
    result["direct_path_clear"] = False
    candidates = [
        [float(minimum[0] - corner_margin), float(minimum[1] - corner_margin)],
        [float(minimum[0] - corner_margin), float(maximum[1] + corner_margin)],
        [float(maximum[0] + corner_margin), float(minimum[1] - corner_margin)],
        [float(maximum[0] + corner_margin), float(maximum[1] + corner_margin)],
    ]
    safe_candidates = [
        point
        for point in candidates
        if not _behavior1k_segment_intersects_aabb_xy(
            robot[:2], point, minimum, maximum
        )
        and not _behavior1k_segment_intersects_aabb_xy(
            point, target[:2], minimum, maximum
        )
    ]
    if not safe_candidates:
        result["detour_unavailable"] = True
        return result
    waypoint_xy = min(
        safe_candidates,
        key=lambda point: float(
            np.linalg.norm(robot[:2] - point) + np.linalg.norm(target[:2] - point)
        ),
    )
    waypoint_yaw = math.atan2(
        float(target[1]) - waypoint_xy[1], float(target[0]) - waypoint_xy[0]
    )
    result["waypoints"] = [[waypoint_xy[0], waypoint_xy[1], waypoint_yaw]]
    result["path_length"] = float(
        np.linalg.norm(robot[:2] - waypoint_xy)
        + np.linalg.norm(target[:2] - waypoint_xy)
    )
    return result


def _behavior1k_physical_toggle_plan(
    env: Any, obj: Any, robot_name: str
) -> JsonDict | None:
    control_state = _behavior1k_control_schema(env)
    robot = control_state.get(robot_name) if isinstance(control_state, dict) else None
    controllers = robot.get("controllers") if isinstance(robot, dict) else None
    if not isinstance(controllers, dict):
        return None
    base_name = next((name for name in controllers if name == "base"), None)
    arm_name = next(
        (name for name in controllers if str(name).startswith("arm_")), None
    )
    gripper_name = next(
        (name for name in controllers if str(name).startswith("gripper_")), None
    )
    if base_name is None or arm_name is None or gripper_name is None:
        return None
    live_target = _behavior1k_live_toggle_target(obj)
    if live_target is None:
        return None
    logical_arm_name = str(arm_name).split("arm_", 1)[-1]
    return {
        "source": "public_native_object_state_link_and_controller_schema",
        "interaction_site": deepcopy(live_target["interaction_site"]),
        "target_pose": deepcopy(live_target["target_pose"]),
        "initial_target_pose": deepcopy(live_target["target_pose"]),
        "live_target_pose": deepcopy(live_target["target_pose"]),
        "surface_normal": deepcopy(live_target["surface_normal"]),
        "standoff_pose": deepcopy(live_target["standoff_pose"]),
        "standoff_distance": live_target["standoff_distance"],
        "base_controller_name": str(base_name),
        "arm_controller_name": str(arm_name),
        "gripper_controller_name": str(gripper_name),
        "logical_arm_name": logical_arm_name,
        "arm_position_command_scale": float(
            controllers.get(str(arm_name), {}).get("position_command_scale", 0.2)
        ),
        "stale_target_delta_m": 0.0,
        "navigation_replanned": False,
        "verifier_feedback_used": False,
    }


def _behavior1k_controller_position_scale(
    controller: Any, *, default: float = 0.2
) -> float:
    """Return the native OSC xyz output delta represented by normalized 1.0."""

    limits = getattr(controller, "command_output_limits", None)
    try:
        array = np.asarray(_to_builtin(limits), dtype=np.float64)
    except (TypeError, ValueError):
        return float(default)
    if array.ndim == 0 or array.size == 0:
        return float(default)
    xyz = array[..., :3] if array.shape[-1] >= 3 else array
    finite = np.abs(xyz[np.isfinite(xyz)])
    if finite.size == 0:
        return float(default)
    scale = float(np.max(finite))
    return scale if scale > 1e-6 else float(default)


def _behavior1k_arm_delta_command(
    robot_pose: Any,
    eef_pose: Any,
    target_position: Any,
    *,
    position_scale: float = 0.2,
) -> tuple[list[float], float]:
    robot = np.asarray(_to_builtin(robot_pose), dtype=np.float64).reshape(-1)
    eef = np.asarray(_to_builtin(eef_pose), dtype=np.float64).reshape(-1)
    target = np.asarray(_to_builtin(target_position), dtype=np.float64).reshape(-1)
    if robot.size < 7 or eef.size < 3 or target.size < 3:
        raise ValueError("robot_xyzw_pose_eef_xyz_and_target_xyz_required")
    pose_summary = _behavior1k_pose_xyz_yaw(robot[:7].tolist())
    if not pose_summary or pose_summary.get("yaw_radians") is None:
        raise ValueError("robot_pose_yaw_unavailable")
    delta_world = target[:3] - eef[:3]
    yaw = float(pose_summary["yaw_radians"])
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    local_x = cosine * float(delta_world[0]) + sine * float(delta_world[1])
    local_y = -sine * float(delta_world[0]) + cosine * float(delta_world[1])
    scale = max(float(position_scale), 1e-6)
    command = [
        max(-1.0, min(1.0, local_x / scale)),
        max(-1.0, min(1.0, local_y / scale)),
        max(-1.0, min(1.0, float(delta_world[2]) / scale)),
        0.0,
        0.0,
        0.0,
    ]
    return command, float(np.linalg.norm(delta_world))


def _behavior1k_closest_finger_link(
    robot_state: Any,
    logical_arm_name: str,
    target_position: Any,
    *,
    preferred_name: str | None = None,
) -> JsonDict | None:
    """Return the public finger-link pose nearest an interaction marker.

    OmniGibson toggle markers are checked against finger rigid bodies rather
    than the synthetic EEF frame. Tracking a native finger-link origin keeps
    the physical controller aligned with the geometry that can satisfy that
    overlap test without consulting the task verifier.
    """

    if not isinstance(robot_state, dict):
        return None
    finger_links = robot_state.get("finger_links")
    if not isinstance(finger_links, dict):
        return None
    arm_links = finger_links.get(str(logical_arm_name))
    candidates = arm_links if isinstance(arm_links, list) else []
    if not candidates:
        candidates = [
            item
            for value in finger_links.values()
            if isinstance(value, list)
            for item in value
            if isinstance(item, dict)
        ]
    if preferred_name:
        preferred_token = preferred_name.lower()
        preferred = [
            item
            for item in candidates
            if any(
                preferred_token in candidate_name.lower()
                for candidate_name in (
                    str(item.get("name") or ""),
                    str(item.get("body_name") or ""),
                    str(item.get("prim_path") or ""),
                )
            )
        ]
        if preferred:
            candidates = preferred
    target = np.asarray(_to_builtin(target_position), dtype=np.float64).reshape(-1)
    if target.size < 3:
        return None
    ranked: list[tuple[float, str, JsonDict]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        pose = item.get("pose_world")
        try:
            position = np.asarray(_to_builtin(pose), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if position.size < 3 or not np.all(np.isfinite(position[:3])):
            continue
        distance = float(np.linalg.norm(target[:3] - position[:3]))
        name = str(
            item.get("name") or item.get("body_name") or item.get("prim_path") or ""
        )
        ranked.append((distance, name, item))
    if not ranked:
        return None
    distance, _name, selected = min(ranked, key=lambda row: (row[0], row[1]))
    return {**deepcopy(selected), "distance_to_target": distance}


def _behavior1k_base_pose_controller_command(
    robot_pose: Any,
    target_pose_xyyaw: list[float],
    *,
    mode: str,
    distance_tolerance: float,
    yaw_tolerance: float,
    linear_gain: float,
    angular_gain: float,
    linear_limit: float,
    angular_limit: float,
    turn_in_place_yaw_threshold: float | None = 0.35,
) -> tuple[list[float], JsonDict]:
    pose = _behavior1k_pose_xyz_yaw(robot_pose)
    if not pose or pose.get("yaw_radians") is None:
        raise ValueError("robot_pose_requires_xyz_and_yaw")
    xyz = pose["xyz"]
    current_yaw = float(pose["yaw_radians"])
    dx = float(target_pose_xyyaw[0]) - float(xyz[0])
    dy = float(target_pose_xyyaw[1]) - float(xyz[1])
    distance = float(math.hypot(dx, dy))
    travel_yaw = (
        math.atan2(dy, dx)
        if distance > max(distance_tolerance, 1e-6)
        else float(target_pose_xyyaw[2])
    )
    yaw_error = _behavior1k_wrap_angle(travel_yaw - current_yaw)
    final_yaw_error = _behavior1k_wrap_angle(float(target_pose_xyyaw[2]) - current_yaw)
    reached = distance <= distance_tolerance and abs(final_yaw_error) <= yaw_tolerance
    if distance <= distance_tolerance:
        linear = 0.0
        angular = max(
            -angular_limit, min(angular_limit, final_yaw_error * angular_gain)
        )
    else:
        forward_alignment = max(0.0, math.cos(yaw_error))
        linear = max(0.0, min(linear_limit, distance * linear_gain * forward_alignment))
        angular = max(-angular_limit, min(angular_limit, yaw_error * angular_gain))
        if (
            turn_in_place_yaw_threshold is not None
            and abs(yaw_error) >= turn_in_place_yaw_threshold
        ):
            linear = 0.0
        if abs(yaw_error) > (math.pi * 0.72):
            linear = 0.0
    if mode == "linear_angular":
        command = [linear, angular]
    elif mode == "left_right_wheel":
        left = max(-linear_limit, min(linear_limit, linear - angular))
        right = max(-linear_limit, min(linear_limit, linear + angular))
        command = [left, right]
    else:
        raise ValueError(f"unsupported_base_command_mode:{mode}")
    return command, {
        "distance": distance,
        "yaw_error": yaw_error,
        "final_yaw_error": final_yaw_error,
        "linear_component": linear,
        "angular_component": angular,
        "turn_in_place_yaw_threshold": turn_in_place_yaw_threshold,
        "mode": mode,
        "reached": reached,
        "distance_tolerance": distance_tolerance,
        "yaw_tolerance": yaw_tolerance,
    }


def _behavior1k_pose_delta(before_pose: Any, after_pose: Any) -> JsonDict:
    before_summary = _behavior1k_pose_xyz_yaw(before_pose)
    after_summary = _behavior1k_pose_xyz_yaw(after_pose)
    if not before_summary or not after_summary:
        return {"available": False, "before": before_summary, "after": after_summary}
    before_xyz = before_summary["xyz"]
    after_xyz = after_summary["xyz"]
    translation_delta = [
        float(after_xyz[index]) - float(before_xyz[index]) for index in range(3)
    ]
    payload: JsonDict = {
        "available": True,
        "before": before_summary,
        "after": after_summary,
        "translation_delta_xyz": translation_delta,
        "translation_distance": float(
            np.linalg.norm(np.asarray(translation_delta, dtype=np.float64))
        ),
    }
    if (
        before_summary.get("yaw_radians") is not None
        and after_summary.get("yaw_radians") is not None
    ):
        payload["yaw_delta_radians"] = _behavior1k_wrap_angle(
            float(after_summary["yaw_radians"]) - float(before_summary["yaw_radians"])
        )
    return payload


def _behavior1k_pose_xyz_yaw(pose: Any) -> JsonDict:
    if not isinstance(pose, (list, tuple)) or len(pose) < 3:
        return {}
    try:
        xyz = [float(pose[index]) for index in range(3)]
    except Exception:
        return {}
    payload: JsonDict = {"xyz": xyz}
    if len(pose) >= 7:
        try:
            qx, qy, qz, qw = (float(value) for value in pose[3:7])
            payload["yaw_radians"] = math.atan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz),
            )
        except Exception:
            payload["yaw_radians"] = None
    return payload


def _behavior1k_wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2.0 * math.pi) - math.pi)


def _behavior1k_grounding_controller_values(
    grounding: JsonDict,
    *,
    source_field: str,
    component_indices: list[int] | None,
    scale: list[float] | float | None,
    offset: list[float] | float | None,
) -> tuple[list[float], JsonDict]:
    allowed_fields = {
        "center_xy",
        "center_normalized_xy",
        "camera_point_xyz",
        "depth_median",
    }
    if source_field not in allowed_fields:
        raise KeyError(
            f"unsupported source_field {source_field!r}; available={sorted(allowed_fields)}"
        )
    raw_value = grounding.get(source_field)
    if raw_value is None:
        raise ValueError(f"grounding field {source_field!r} is unavailable")
    values = np.asarray(raw_value, dtype=np.float64).reshape(-1)
    if component_indices is not None:
        if not isinstance(component_indices, list) or not all(
            isinstance(index, int) and not isinstance(index, bool)
            for index in component_indices
        ):
            raise TypeError("component_indices must be a list of integer indices")
        if any(index < 0 or index >= len(values) for index in component_indices):
            raise ValueError(
                f"component_indices out of range for {source_field!r} with {len(values)} values"
            )
        values = values[component_indices]
    scale_values = _behavior1k_broadcast_conversion_parameter(
        scale, len(values), default=1.0, name="scale"
    )
    offset_values = _behavior1k_broadcast_conversion_parameter(
        offset, len(values), default=0.0, name="offset"
    )
    converted = values * scale_values + offset_values
    return [float(value) for value in converted], {
        "source_field": source_field,
        "source_values": _to_builtin(values),
        "component_indices": deepcopy(component_indices),
        "scale": _to_builtin(scale_values),
        "offset": _to_builtin(offset_values),
        "formula": "controller_input = selected_grounding_values * scale + offset",
        "policy_selected_by_runtime": False,
    }


def _behavior1k_broadcast_conversion_parameter(
    value: list[float] | float | None,
    size: int,
    *,
    default: float,
    name: str,
) -> np.ndarray:
    if value is None:
        return np.full(size, default, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        return np.full(size, float(array[0]), dtype=np.float64)
    if array.size != size:
        raise ValueError(f"{name} must contain one value or {size} values")
    return array


def _behavior1k_object_interaction_sites(
    obj: Any, state_name: str | None = None
) -> list[JsonDict]:
    if obj is None:
        return []
    raw_states = getattr(obj, "states", None)
    if not isinstance(raw_states, dict):
        return []
    sites: list[JsonDict] = []
    for key, state in raw_states.items():
        key_name = getattr(key, "__name__", None) or str(key)
        if state_name is not None and key_name.lower() != state_name.lower():
            continue
        link = getattr(state, "link", None)
        if link is None or not hasattr(link, "get_position_orientation"):
            continue
        try:
            pos, quat = link.get_position_orientation()
        except Exception:
            continue
        item: JsonDict = {
            "state_name": key_name,
            "link_name": str(getattr(link, "name", "")),
            "prim_path": str(getattr(link, "prim_path", "")),
            "pose_world": _to_builtin(list(pos) + list(quat)),
            "source": "native_object_state_link",
        }
        for attr in ("scale", "extent"):
            if hasattr(link, attr):
                item[attr] = _to_builtin(getattr(link, attr))
        sites.append(item)
    return sites


def _index_list(value: Any) -> list[int]:
    try:
        return [
            int(item) for item in np.asarray(_to_builtin(value)).reshape(-1).tolist()
        ]
    except Exception:
        return []


def _optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def _space_schema(space: Any) -> JsonDict:
    schema: JsonDict = {
        "available": True,
        "type": type(space).__name__,
        "repr": str(space),
        "sample_available": hasattr(space, "sample"),
    }
    shape = getattr(space, "shape", None)
    if shape is not None:
        schema["shape"] = [int(dim) for dim in shape]
    dtype = getattr(space, "dtype", None)
    if dtype is not None:
        schema["dtype"] = str(dtype)
    for attr in ("low", "high"):
        if hasattr(space, attr):
            value = getattr(space, attr)
            summary = summarize_data(value)
            if isinstance(summary, dict) and "value" in summary:
                schema[attr] = summary["value"]
            else:
                schema[f"{attr}_summary"] = summary
                compact = _compact_array_stats(value)
                if compact:
                    schema[f"{attr}_compact"] = compact
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        schema["spaces"] = {
            str(key): _space_schema(item) for key, item in spaces.items()
        }
    return schema


def _behavior1k_step_action(
    action: Any | None, *, schema: JsonDict, env: Any
) -> tuple[Any, str]:
    action_space = getattr(_unwrap_env(env), "action_space", None) or getattr(
        env, "action_space", None
    )
    if action is None:
        zero = _zero_action_for_space(action_space)
        if zero is not None:
            return zero, "zero_action_from_action_space"
        if action_space is not None and hasattr(action_space, "sample"):
            return action_space.sample(), "action_space.sample"
        raise RuntimeError(
            "BEHAVIOR-1K env does not expose an action_space or accept a default action."
        )
    if isinstance(action, dict) and "action" in action and not schema.get("spaces"):
        action = action["action"]
    if action_space is not None and schema.get("shape") is not None:
        return np.asarray(
            action, dtype=getattr(action_space, "dtype", np.float32)
        ), "caller"
    return action, "caller"


def _zero_action_for_space(space: Any) -> Any | None:
    if space is None:
        return None
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, dict):
        return {key: _zero_action_for_space(item) for key, item in spaces.items()}
    shape = getattr(space, "shape", None)
    if shape is not None:
        dtype = getattr(space, "dtype", np.float32)
        return np.zeros(shape, dtype=dtype)
    if hasattr(space, "n"):
        return 0
    return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _compact_array_stats(value: Any, *, sample_size: int = 8) -> JsonDict:
    try:
        arr = np.asarray(value)
    except Exception:
        return {}
    if arr.size == 0:
        return {"size": 0}
    flat = arr.reshape(-1)
    payload: JsonDict = {
        "size": int(arr.size),
        "sample": _to_builtin(flat[:sample_size]),
    }
    if np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.bool_):
        finite = arr[np.isfinite(arr)] if np.issubdtype(arr.dtype, np.floating) else arr
        if finite.size:
            payload["min"] = _to_builtin(np.min(finite))
            payload["max"] = _to_builtin(np.max(finite))
            payload["mean"] = _to_builtin(np.mean(finite))
    return payload


def _unwrap_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def _asset_payload(obj: Any) -> JsonDict:
    payload: JsonDict = {"type": type(obj).__name__}
    for attr in ("category", "model", "prim_path"):
        if hasattr(obj, attr):
            payload[attr] = _to_builtin(getattr(obj, attr))
    if hasattr(obj, "get_position_orientation"):
        try:
            pos, orn = obj.get_position_orientation()
            payload["pose_world"] = _to_builtin(list(pos) + list(orn))
        except Exception:
            pass
    states = _behavior1k_public_object_states(obj)
    if states:
        payload["object_states"] = states
    return payload


def _behavior1k_lookup_scene_object(env: Any, asset_name: str | None) -> Any | None:
    if env is None or not asset_name:
        return None
    scene = getattr(_unwrap_env(env), "scene", None)
    objects = getattr(scene, "objects", None) or getattr(scene, "_objects", None) or []
    if isinstance(objects, dict):
        iterable = objects.items()
    else:
        iterable = [
            (getattr(obj, "name", f"asset_{idx}"), obj)
            for idx, obj in enumerate(objects)
        ]
    target = asset_name.lower()
    for name, obj in iterable:
        candidates = {
            str(name).lower(),
            str(getattr(obj, "name", "")).lower(),
            str(getattr(obj, "category", "")).lower(),
            str(getattr(obj, "model", "")).lower(),
        }
        if target in candidates:
            return obj
    for name, obj in iterable:
        haystack = " ".join(
            str(part).lower()
            for part in [
                name,
                getattr(obj, "name", ""),
                getattr(obj, "category", ""),
                getattr(obj, "model", ""),
            ]
        )
        if target and target in haystack:
            return obj
    return None


def _behavior1k_public_semantic_action_names() -> list[str]:
    return [
        "GRASP",
        "PLACE_ON_TOP",
        "PLACE_INSIDE",
        "OPEN",
        "CLOSE",
        "NAVIGATE_TO",
        "TOGGLE_ON",
        "TOGGLE_OFF",
    ]


def _configure_native_curobo_collision_cache() -> None:
    """Allocate the native scene's meshes before CuRobo captures CUDA graphs."""
    from functools import wraps
    from omnigibson.action_primitives import curobo
    original = curobo.create_world_mesh_collision
    if getattr(original, "_arena_scene_sized_cache", False):
        return

    @wraps(original)
    def create_world(tensor_args, obb_cache_size=10, mesh_cache_size=2048, max_distance=0.05):
        import omnigibson as og
        # Including robot meshes is a harmless upper bound: the native world
        # builder excludes the controlled robot, but retains every other mesh.
        scene_meshes = int(og.sim.floor_plane is not None) + sum(
            len(link.collision_meshes)
            for scene in og.sim.scenes
            for obj in scene.objects if not obj.visual_only
            for link in obj.links.values()
        )
        return original(tensor_args, obb_cache_size=obb_cache_size,
                        mesh_cache_size=max(mesh_cache_size, scene_meshes),
                        max_distance=max_distance)

    create_world._arena_scene_sized_cache = True
    curobo.create_world_mesh_collision = create_world


def _configure_native_curobo_shared_buffers() -> bool:
    """Select CuRobo's native alternate kernel for affected Isaac deployments."""
    if os.environ.get("EMBODIED_ARENA_CUROBO_SHARED_BUFFERS") != "0":
        return False
    from functools import wraps
    from curobo.opt.newton.lbfgs import LBFGSOpt
    _configure_native_curobo_collision_cache()
    original = LBFGSOpt.__init__
    if getattr(original, "_arena_native_no_shared_buffers", False):
        return True

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.use_shared_buffers_kernel = False

    initialize._arena_native_no_shared_buffers = True
    LBFGSOpt.__init__ = initialize
    return True


def _behavior1k_semantic_action_runtime(semantic_action: str) -> tuple[Any, Any, Any]:
    _configure_native_curobo_shared_buffers()
    try:
        from omnigibson.action_primitives.starter_semantic_action_primitives import (  # type: ignore
            StarterSemanticActionPrimitives,
            StarterSemanticActionPrimitiveSet,
        )
    except Exception as exc:
        raise RuntimeError(
            f"semantic_action_runtime_unavailable:{type(exc).__name__}: {exc}"
        ) from exc
    try:
        primitive = StarterSemanticActionPrimitiveSet[
            str(semantic_action).strip().upper()
        ]
    except KeyError as exc:
        raise RuntimeError(
            "unknown_semantic_action:"
            + str(semantic_action)
            + "; available="
            + ",".join(_behavior1k_public_semantic_action_names())
        ) from exc
    return StarterSemanticActionPrimitives, StarterSemanticActionPrimitiveSet, primitive


def _behavior1k_public_object_states(obj: Any) -> JsonDict:
    states: JsonDict = {}
    for state_name in ("ToggledOn", "Open"):
        value = _behavior1k_object_state_value(obj, state_name)
        if value is not None:
            states[state_name] = value
    return states


def _behavior1k_object_state_entry(obj: Any, state_name: str) -> Any | None:
    raw_states = getattr(obj, "states", None)
    if not isinstance(raw_states, dict):
        return None
    state_class = _behavior1k_object_state_class(state_name)
    candidates: list[Any] = []
    if state_class is not None:
        candidates.append(state_class)
    candidates.extend([state_name, state_name.lower()])
    for candidate in candidates:
        if candidate in raw_states:
            return raw_states[candidate]
    for key, state in raw_states.items():
        key_name = getattr(key, "__name__", None) or str(key)
        if key_name == state_name or key_name.lower() == state_name.lower():
            return state
    return None


def _behavior1k_object_state_class(state_name: str) -> Any | None:
    try:
        from omnigibson import object_states  # type: ignore

        return getattr(object_states, state_name, None)
    except Exception:
        return None


def _behavior1k_object_state_value(obj: Any, state_name: str) -> bool | None:
    state = _behavior1k_object_state_entry(obj, state_name)
    if state is None or not hasattr(state, "get_value"):
        return None
    try:
        return bool(state.get_value())
    except Exception:
        return None


def _behavior1k_set_object_state(obj: Any, state_name: str, value: bool) -> JsonDict:
    state = _behavior1k_object_state_entry(obj, state_name)
    if state is None or not hasattr(state, "set_value"):
        return {
            "available": False,
            "state_name": state_name,
            "error": "object_state_not_found",
        }
    try:
        ok = state.set_value(bool(value))
    except Exception as exc:
        return {
            "available": True,
            "state_name": state_name,
            "set_value_returned": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "available": True,
        "state_name": state_name,
        "set_value_returned": _to_builtin(ok),
        "private_success_signal": False,
    }


def _behavior1k_harness_task_success(env: Any) -> JsonDict:
    if env is None:
        return {"available": False, "success": False, "source": "env_unavailable"}
    task = getattr(_unwrap_env(env), "task", None) or getattr(env, "task", None)
    if task is None:
        return {"available": False, "success": False, "source": "env.task_unavailable"}
    if hasattr(task, "success"):
        try:
            return {
                "available": True,
                "success": bool(getattr(task, "success")),
                "source": "env.task.success",
            }
        except Exception as exc:
            return {
                "available": False,
                "success": False,
                "source": "env.task.success",
                "error": f"{type(exc).__name__}: {exc}",
            }
    options = getattr(task, "ground_goal_state_options", None)
    if options:
        try:
            option_successes = [
                all(bool(pred.evaluate()) for pred in option) for option in options
            ]
            return {
                "available": True,
                "success": any(option_successes),
                "source": "env.task.ground_goal_state_options.evaluate",
                "option_successes": option_successes,
            }
        except Exception as exc:
            return {
                "available": False,
                "success": False,
                "source": "env.task.ground_goal_state_options.evaluate",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {
        "available": False,
        "success": False,
        "source": "official_task_success_unavailable",
    }


def _select_name(
    registry: dict[str, JsonDict], explicit_name: str | None
) -> str | None:
    return explicit_name if explicit_name in registry else None


def _public_behavior1k_info(info: JsonDict) -> JsonDict:
    forbidden = ("success", "reward", "checker", "predicate", "goal")

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

    return sanitize(info)


def _behavior1k_contact_payload(contact: Any) -> JsonDict:
    fields = ("body0", "body1", "position", "normal", "impulse", "distance")
    payload = {
        field: _to_builtin(getattr(contact, field))
        for field in fields
        if hasattr(contact, field)
    }
    if payload:
        return payload
    return {"value": _to_builtin(contact)}


def _behavior1k_public_finger_contacts(obj: Any, finger_link: Any) -> list[JsonDict]:
    """Return public object contacts involving one native finger link."""

    if not isinstance(finger_link, dict):
        return []
    contact_list = getattr(obj, "contact_list", None)
    if not callable(contact_list):
        return []
    identity_tokens = {
        str(finger_link.get(field) or "")
        for field in ("name", "body_name", "prim_path")
        if str(finger_link.get(field) or "")
    }
    if not identity_tokens:
        return []

    def matches(body: Any) -> bool:
        body_text = str(body or "")
        return any(
            body_text == token
            or body_text.endswith(f"/{token}")
            or body_text.endswith(f":{token}")
            for token in identity_tokens
        )

    try:
        payloads = [_behavior1k_contact_payload(contact) for contact in contact_list()]
    except Exception:
        return []
    return [
        payload
        for payload in payloads
        if matches(payload.get("body0")) or matches(payload.get("body1"))
    ]


def _behavior1k_contact_press_normal(
    contacts: Any,
    surface_normal: Any,
) -> tuple[list[float], str]:
    """Return a stable public press direction aligned with the interaction surface."""

    surface = np.asarray(_to_builtin(surface_normal), dtype=np.float64).reshape(-1)
    if surface.size < 3 or not np.all(np.isfinite(surface[:3])):
        raise ValueError("behavior1k_contact_surface_normal_unavailable")
    surface = surface[:3]
    surface_norm = float(np.linalg.norm(surface))
    if surface_norm <= 1e-6:
        raise ValueError("behavior1k_contact_surface_normal_unavailable")
    surface /= surface_norm
    for contact in contacts if isinstance(contacts, list) else []:
        if not isinstance(contact, dict):
            continue
        raw_normal = contact.get("normal")
        try:
            normal = np.asarray(_to_builtin(raw_normal), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if normal.size < 3 or not np.all(np.isfinite(normal[:3])):
            continue
        normal = normal[:3]
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-6:
            continue
        normal /= normal_norm
        # PhysX contact ordering may reverse the reported normal. Align it to
        # the public interaction-link surface direction before pressing.
        if float(np.dot(normal, surface)) < 0.0:
            normal *= -1.0
        return normal.tolist(), "public_contact_normal_aligned_to_surface"
    return surface.tolist(), "public_interaction_surface_normal"


def _behavior1k_contact_hold_target(
    *,
    contact_finger_pose: Any,
    current_eef_pose: Any,
    current_finger_pose: Any,
    press_normal: Any,
    press_depth: float,
) -> list[float]:
    """Bind hold control to the finger contact frame, not a stale EEF pose."""

    anchor = np.asarray(_to_builtin(contact_finger_pose), dtype=np.float64).reshape(-1)
    eef = np.asarray(_to_builtin(current_eef_pose), dtype=np.float64).reshape(-1)
    finger = np.asarray(_to_builtin(current_finger_pose), dtype=np.float64).reshape(-1)
    normal = np.asarray(_to_builtin(press_normal), dtype=np.float64).reshape(-1)
    if any(value.size < 3 for value in (anchor, eef, finger, normal)):
        raise ValueError("behavior1k_contact_frame_xyz_required")
    normal_norm = float(np.linalg.norm(normal[:3]))
    if normal_norm <= 1e-6:
        raise ValueError("behavior1k_contact_press_normal_unavailable")
    normal = normal[:3] / normal_norm
    live_eef_from_finger = eef[:3] - finger[:3]
    target_finger = anchor[:3] + max(0.0, float(press_depth)) * normal
    return (target_finger + live_eef_from_finger).tolist()


def _card(
    name: str,
    level: str,
    input_schema: JsonDict,
    output_schema: JsonDict,
    description: str,
) -> PrimitiveCard:
    return PrimitiveCard(
        name=name,
        capability_tags=["behavior1k", "omnigibson", "bddl", "agent_safe"],
        input_schema=input_schema,
        output_schema=output_schema,
        preconditions=["backend reset"],
        side_effects=[],
        failure_modes=["runtime_missing", "asset_not_found"],
        abstraction_level=level,
        leakage_risk="low",
        description=description,
    )


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
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
