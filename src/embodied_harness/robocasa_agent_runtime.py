from __future__ import annotations

from copy import deepcopy
import ast
import inspect
import json
from dataclasses import asdict, dataclass, field
import importlib.util
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable

import numpy as np

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]
SkillExecutor = Callable[[Any, str, dict[str, Any]], PrimitiveResult]
ROBOCASA_REQUIRED_LIVE_MODULES = ("robocasa", "robosuite", "mujoco", "gymnasium")
_UNCHANGED_POSE_REUSE_SKILLS = {
    "grasp_robocasa_object",
    "place_robocasa_object_at",
    "move_robocasa_ee_to",
    "open_robocasa_fixture",
    "close_robocasa_fixture",
    "press_robocasa_fixture_button",
    "navigate_robocasa_to_object",
}
_UNCHANGED_POSE_REUSE_TOLERANCE_M = 0.03


def _invoke_primitive_handler(handler: Callable[..., PrimitiveResult], kwargs: JsonDict) -> PrimitiveResult:
    from .robocasa_interface_repair import invoke
    return invoke(handler, kwargs)


def _robocasa_live_module_status() -> dict[str, bool]:
    return {
        module: importlib.util.find_spec(module) is not None
        for module in ROBOCASA_REQUIRED_LIVE_MODULES
    }


@dataclass(slots=True)
class RoboCasaRuntimeConfig:
    env_id: str = "robocasa/PickPlaceCounterToCabinet"
    split: str = "target"
    robot: str = "PandaOmron"
    camera_names: list[str] = field(
        default_factory=lambda: ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    )
    camera_widths: int = 128
    camera_heights: int = 128
    camera_depths: bool = True
    render_onscreen: bool = False
    live: bool = True
    use_gymnasium_wrapper: bool = True
    asset_cache_dir: str | None = None
    env_kwargs: JsonDict = field(default_factory=dict)
    motion_backend: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)


class RoboCasaAgentRuntimeBackend(EmbodiedBackend):
    """AI-native RoboCasa runtime adapter.

    The live path creates a real RoboCasa/Gymnasium kitchen environment. Agent
    primitives expose observation, prompt-conditioned object/fixture grounding,
    and action-skill hooks. The task success predicate is intentionally only
    available through ``verify()`` and is never listed as an agent primitive.
    """

    def __init__(
        self,
        config: RoboCasaRuntimeConfig | None = None,
        env_factory: EnvFactory | None = None,
        skill_executor: SkillExecutor | None = None,
    ) -> None:
        self.config = config or RoboCasaRuntimeConfig()
        self._env_factory = env_factory
        self._skill_executor = skill_executor
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._objects: dict[str, JsonDict] = {}
        self._fixtures: dict[str, JsonDict] = {}
        self._visual_grounding_handles: dict[str, JsonDict] = {}
        self._held_object_handle: str | None = None
        self._held_object_name: str | None = None
        self._observation_serial = 0
        self._pool_task: tuple[str, str, int] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        name = str(coordinate.get("task_id") or "")
        variation = str(coordinate.get("variation") or "")
        seed = coordinate.get("seed")
        split, separator, selected_name = variation.partition("::")
        if not separator or selected_name != name or split not in {"target", "pretrain"}:
            raise ValueError("RoboCasa requires split::NativeTaskName")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("RoboCasa reset seed must be an integer in [0, 2**32)")
        source = get_project_paths().external_upstream("robocasa") / "robocasa/utils/dataset_registry.py"
        names = set()
        for node in ast.parse(source.read_text()).body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == "OrderedDict"):
                for entry in node.value.keywords:
                    if isinstance(entry.value, ast.Call) and any(k.arg == split for k in entry.value.keywords):
                        names.add(entry.arg)
        if getattr(self.config, "suite_variant", None) == "robocasa365" and split == "pretrain":
            # The released task catalogue includes 48 tasks without demonstration datasets.
            catalogue = source.parents[2] / "docs/composite_tasks/task_attributes.json"
            names.update(row["name"] for row in json.loads(catalogue.read_text())["tasks"])
        if name not in names:
            raise ValueError(f"Unknown pinned RoboCasa {split} dataset task: {name!r}")
        self._pool_task = (name, split, seed)
        return {"bound": True, "mode": "native_task_split", "env_name": name, "split": split, "seed": seed}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_task is not None:
            name, split, selected_seed = self._pool_task
            if seed != selected_seed:
                raise ValueError("RoboCasa reset seed differs from selected pool coordinate")
            overrides.update(env_id=f"robocasa/{name}", split=split)
            suite = getattr(self.config, "suite_variant", "robocasa")
            if suite == "robocasa365":
                overrides.update(dataset_root=None, episode_id=None, task_family=name)
            task_id = f"{suite}:{split}:{name}:seed_{seed}"
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:robocasa:live_runtime",
            instruction=(
                "Solve a RoboCasa kitchen manipulation task by grounding real kitchen object/fixture state "
                "and invoking open/close/manipulate action-skill primitives."
            ),
            goal={"env_id": runtime_config.env_id, "success_source": "robocasa_env_info_or_harness_verifier"},
            initial_state={},
            budgets={"primitive_calls": 40, "verifier_calls": 5},
            tags=["w4", "robocasa", "ai_native_runtime", "live" if runtime_config.live else "dry"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "robocasa",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "grounding_uses_real_state_and_rgbd": True,
                    "entity_selection_requires_exact_caller_name_or_id": True,
                    "button_selection_requires_exact_caller_name_or_pose": True,
                    "query_or_task_lexical_auto_selection": False,
                    "runtime_recommendation_or_recovery_planner": False,
                    "action_skills_require_real_backend": True,
                    "private_success_checker_exposed_as_primitive": False,
                },
            },
        )
        self._last_info = {}
        self._visual_grounding_handles = {}
        self._held_object_handle = None
        self._held_object_name = None
        self._observation_serial = 0
        if runtime_config.live:
            self._env = self._make_env(runtime_config, seed=seed)
            reset_result = self._env.reset()
            self._last_obs, self._last_info = _split_reset_result(reset_result)
            self._observation_serial = 1
            benchmark_instruction = self._benchmark_task_description()
            if benchmark_instruction:
                self._task_spec.instruction = benchmark_instruction
                self._task_spec.metadata["instruction_source"] = "robocasa_episode_metadata"
            self._refresh_scene_registry()
        else:
            self._env = None
            self._last_obs = {}
            self._objects = {}
            self._fixtures = {}
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
        data = {
            "language_task": self._language_task(),
            "runtime": self.runtime_available(),
            "observation_summary": summarize_observation(self._last_obs),
            "objects": deepcopy(self._objects),
            "fixtures": deepcopy(self._fixtures),
            "pose_evidence": self._pose_evidence(),
            "visual_evidence": self._visual_evidence(),
            "action_skill_schema": self._action_skill_schema(),
        }
        obs = Observation(step=len(self.get_trace().events), data=data, metadata={"benchmark_id": "robocasa"})
        self.record_event("observe", obs.to_dict())
        observation_artifact = f"robocasa:observation:{self._observation_serial}"
        if observation_artifact not in self.get_trace().artifacts:
            self.get_trace().add_artifact(observation_artifact, _to_builtin(data))
        return Observation(step=obs.step, data=self._compact_kitchen_state(), metadata=obs.metadata)

    def _compact_kitchen_state(self) -> JsonDict:
        """Expose a task-oriented scene index; keep the full state in the trace."""
        task = self._language_task()
        task_text = " ".join(str(task.get(k) or "") for k in ("instruction", "benchmark_task_description"))
        task_key = re.sub(r"[^a-z0-9]", "", task_text.casefold())

        def compact_registry(registry: dict[str, JsonDict], *, include_all: bool = False) -> tuple[list[str], JsonDict]:
            names = sorted(registry)
            relevant: JsonDict = {}
            for name in names:
                payload = registry[name]
                aliases = [name, *payload.get("agent_aliases", [])]
                if not include_all and not any(
                    (key := re.sub(r"[^a-z0-9]", "", str(alias).casefold()).rstrip("0123456789"))
                    and len(key) >= 3 and key in task_key for alias in aliases
                ):
                    continue
                sites = payload.get("affordance_sites")
                relevant[name] = {
                    "position": self._registry_position(name),
                    "size": next((payload[k] for k in ("size", "bbox_size", "dimensions") if k in payload), None),
                    "affordance_sites": sorted(sites) if isinstance(sites, dict) else [],
                }
            return names, relevant

        object_names, objects = compact_registry(self._objects, include_all=len(self._objects) <= 12)
        fixture_names, fixtures = compact_registry(self._fixtures)
        obs = self._last_obs if isinstance(self._last_obs, dict) else {}
        return {
            "language_task": task,
            "objects": objects,
            "fixtures": fixtures,
            "available_object_names": object_names,
            "available_fixture_names": fixture_names,
            "eef_position": _to_builtin(_vector_to_list(_vector(obs, "robot0_eef_pos"))),
            "gripper_qpos": _observation_numeric_list(obs, "robot0_gripper_qpos", "state.gripper_qpos"),
            "held_object_name": self._held_object_name,
            "held_object_handle": self._held_object_handle,
            "camera_names": list(self.config.camera_names),
            "observation_serial": self._observation_serial,
            "evidence_handles": [
                {"handle": handle, "entity_name": item.get("entity_name")}
                for handle, item in list(self._visual_grounding_handles.items())[-8:]
                if item.get("observation_serial") == self._observation_serial
            ],
            "grounding_hint": "ground_robocasa_visual_target(entity_name=canonical_name); pass its evidence_handle to an action",
        }

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_robocasa_kitchen_state",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {
                    "language_task": "dict",
                    "fixtures": "dict",
                    "objects": "dict",
                    "pose_evidence": "dict",
                    "visual_evidence": "dict",
                    "action_skill_schema": "dict",
                },
                "Observe real RoboCasa object and fixture state summaries.",
            ),
            self._primitive_card(
                "observe_robocasa_rgbd",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_name": "str|None"},
                {
                    "language_task": "dict",
                    "rgbd_evidence": "dict",
                    "segmentation_evidence": "dict",
                    "visual_runtime": "dict",
                    "segmentation_instances": "dict[str,list[dict]]",
                    "artifacts": "list[str]",
                },
                "Inspect RGB-D cameras, modalities, and segmentation instances from the live RoboCasa environment.",
            ),
            self._primitive_card(
                "ground_robocasa_visual_target",
                "L2",
                {
                    "prompt": "str|None",
                    "entity_name": "str|None",
                    "camera_name": "str|None",
                    "segmentation_id": "int|None",
                    "world_position": "list[float]|None",
                    "max_world_distance": "float|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "evidence_handle": "str",
                    "grounding": "dict",
                    "native_modalities": "dict",
                },
                "Bind a caller-selected segmentation instance to a prompt using native RGB, depth, and segmentation.",
            ),
            self._primitive_card(
                "inspect_robocasa_object",
                "L2",
                {"prompt": "str|None", "object_name": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"language_task": "dict", "selected": "dict", "candidates": "list[dict]", "pose_evidence": "dict"},
                "Inspect an exact caller-selected object name or id, or list candidates when omitted.",
            ),
            self._primitive_card(
                "locate_robocasa_object",
                "L2",
                {"prompt": "str|None", "object_name": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"language_task": "dict", "selected": "dict", "candidates": "list[dict]", "pose_evidence": "dict"},
                "Locate an exact caller-selected RoboCasa object name or id for downstream actions.",
            ),
            self._primitive_card(
                "inspect_robocasa_fixture",
                "L2",
                {"prompt": "str|None", "fixture_name": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"language_task": "dict", "selected": "dict", "candidates": "list[dict]", "pose_evidence": "dict"},
                "Inspect an exact caller-selected fixture name or id, or list candidates when omitted.",
            ),
            self._primitive_card(
                "locate_robocasa_fixture",
                "L2",
                {"prompt": "str|None", "fixture_name": "str|None", "query": "str|None", "agent_context": "dict|None"},
                {"language_task": "dict", "selected": "dict", "candidates": "list[dict]", "pose_evidence": "dict"},
                "Locate an exact caller-selected RoboCasa fixture name or id for downstream actions.",
            ),
            self._primitive_card(
                "inspect_robocasa_affordance",
                "L2",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "object_name": "str|None",
                    "target_name": "str|None",
                    "relation": "str",
                    "agent_context": "dict|None",
                },
                {
                    "language_task": "dict",
                    "selected_object": "dict|None",
                    "selected_target": "dict|None",
                    "grasp_affordance": "dict",
                    "grasp_candidates": "list[dict]",
                    "placement_affordance": "dict",
                },
                "Report object/fixture geometry and grasp/place affordance estimates from live RoboCasa state.",
            ),
            self._primitive_card(
                "inspect_robocasa_button_contact_frame",
                "L2",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "fixture_name": "str",
                    "button_name": "str|None",
                    "button_position": "list[float]|None",
                    "visual_binding_tolerance": "float|None",
                    "evidence_handles": "list[str]|None",
                    "candidate_limit": "int",
                    "agent_context": "dict|None",
                },
                {
                    "language_task": "dict",
                    "selected_fixture": "dict",
                    "selected_button": "dict",
                    "visual_binding": "dict",
                    "contact_frame_candidates": "list[dict]",
                    "press_primitive_parameters": "dict",
                },
                "Inspect caller-selected button geometry, same-observation visual grounding, and gripper contact-frame candidates for a later press action.",
            ),
            self._primitive_card(
                "open_robocasa_fixture",
                "L3",
                {"prompt": "str|None", "query": "str|None", "fixture_name": "str", "agent_context": "dict|None", "strategy": "str"},
                {"action_skill_schema": "dict", "execution_status": "str", "requires_motion_backend": "bool"},
                "Real action-skill hook for opening a fixture; fails explicitly without a configured backend.",
            ),
            self._primitive_card(
                "close_robocasa_fixture",
                "L3",
                {"prompt": "str|None", "query": "str|None", "fixture_name": "str", "agent_context": "dict|None", "strategy": "str"},
                {"action_skill_schema": "dict", "execution_status": "str", "requires_motion_backend": "bool"},
                "Real action-skill hook for closing a fixture; fails explicitly without a configured backend.",
            ),
            self._primitive_card(
                "press_robocasa_fixture_button",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "fixture_name": "str",
                    "button_name": "str|None",
                    "button_position": "list[float]|None",
                    "visual_binding_tolerance": "float|None",
                    "use_visual_button_anchor_position": "bool",
                    "button_offset": "list[float]|None",
                    "bind_gripper_contact_geometry": "bool",
                    "gripper_contact_surface_axis": "list[float]|None",
                    "agent_context": "dict|None",
                    "strategy": "str",
                    "horizon": "int",
                    "approach_steps": "int|None",
                    "approach_distance": "float",
                    "approach_stop_distance": "float|None",
                    "approach_patience_steps": "int",
                    "tolerance": "float",
                    "press_depth": "float",
                    "max_press_depth": "float|None",
                    "press_contact_seek_steps": "int",
                    "press_contact_seek_depth": "float|None",
                    "press_direction_sign": "float",
                    "press_direction_vector": "list[float]|None",
                    "press_steps": "int",
                    "hold_steps": "int",
                    "retreat_steps": "int|None",
                    "retreat_distance": "float",
                    "gain": "float",
                    "max_delta": "float",
                    "mobile_base_enabled": "bool",
                    "mobile_base_active_phases": "list[str]|None",
                    "base_gain": "float",
                    "base_max_delta": "float",
                    "base_command_sign": "float",
                    "base_mode_value": "float",
                    "base_xy_deadband": "float|None",
                    "arm_delta_frame": "str|None",
                    "gripper_command": "float",
                    "motion_trace_limit": "int",
                },
                {
                    "action_skill_schema": "dict",
                    "execution_status": "str",
                    "motion_status": "str",
                    "grounded_action": "dict",
                    "button_affordance": "dict",
                    "fixture_state_before": "dict|None",
                    "fixture_state_after": "dict|None",
                },
                "Press an exact caller-selected button name or world pose through env.step actions, then retreat.",
            ),
            self._primitive_card(
                "refine_robocasa_button_contact_search",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "fixture_name": "str|None",
                    "button_name": "str|None",
                    "button_position": "list[float]|None",
                    "evidence_handles": "list[str]|None",
                    "visual_binding_tolerance": "float|None",
                    "previous_action_output": "dict|None",
                    "contact_frame_candidate": "dict|None",
                    "move_to_previous_best": "bool",
                    "prealign_steps": "int",
                    "prealign_gain": "float",
                    "prealign_max_delta": "float",
                    "use_visual_button_anchor_position": "bool",
                    "button_offset": "list[float]|None",
                    "bind_gripper_contact_geometry": "bool",
                    "gripper_contact_surface_axis": "list[float]|None",
                    "horizon": "int",
                    "approach_steps": "int|None",
                    "approach_distance": "float",
                    "approach_stop_distance": "float|None",
                    "approach_patience_steps": "int",
                    "tolerance": "float",
                    "press_depth": "float",
                    "max_press_depth": "float|None",
                    "press_contact_seek_steps": "int",
                    "press_contact_seek_depth": "float|None",
                    "press_direction_sign": "float",
                    "press_direction_vector": "list[float]|None",
                    "press_steps": "int",
                    "hold_steps": "int",
                    "retreat_steps": "int|None",
                    "retreat_distance": "float",
                    "gain": "float",
                    "max_delta": "float",
                    "mobile_base_enabled": "bool",
                    "mobile_base_active_phases": "list[str]|None",
                    "base_gain": "float",
                    "base_max_delta": "float",
                    "base_command_sign": "float",
                    "base_mode_value": "float",
                    "base_delta_frame": "str",
                    "base_xy_deadband": "float|None",
                    "arm_delta_frame": "str|None",
                    "gripper_command": "float",
                    "motion_trace_limit": "int",
                    "agent_context": "dict|None",
                },
                {
                    "prealign": "dict|None",
                    "action": "dict",
                    "recovery_source": "dict",
                    "verifier_boundary": "dict",
                },
                "Continue a visually grounded button contact search from previous action trace evidence and caller-selected contact-frame parameters.",
            ),
            self._primitive_card(
                "sweep_robocasa_button_contact_candidates",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "fixture_name": "str|None",
                    "button_name": "str|None",
                    "button_position": "list[float]|None",
                    "evidence_handles": "list[str]|None",
                    "visual_binding_tolerance": "float|None",
                    "previous_action_output": "dict|None",
                    "contact_frame_candidates": "list[dict]",
                    "candidate_indices": "list[int]|None",
                    "max_attempts": "int|None",
                    "stop_on_interaction": "bool",
                    "move_to_previous_best": "bool",
                    "prealign_steps": "int",
                    "prealign_gain": "float",
                    "prealign_max_delta": "float",
                    "use_visual_button_anchor_position": "bool",
                    "bind_gripper_contact_geometry": "bool",
                    "horizon": "int",
                    "approach_steps": "int|None",
                    "approach_distance": "float",
                    "approach_stop_distance": "float|None",
                    "approach_patience_steps": "int",
                    "tolerance": "float",
                    "press_depth": "float",
                    "max_press_depth": "float|None",
                    "press_contact_seek_steps": "int",
                    "press_contact_seek_depth": "float|None",
                    "press_direction_sign": "float",
                    "press_steps": "int",
                    "hold_steps": "int",
                    "retreat_steps": "int|None",
                    "retreat_distance": "float",
                    "gain": "float",
                    "max_delta": "float",
                    "mobile_base_enabled": "bool",
                    "mobile_base_active_phases": "list[str]|None",
                    "base_gain": "float",
                    "base_max_delta": "float",
                    "base_command_sign": "float",
                    "base_mode_value": "float",
                    "base_delta_frame": "str",
                    "base_xy_deadband": "float|None",
                    "arm_delta_frame": "str|None",
                    "gripper_command": "float",
                    "motion_trace_limit": "int",
                    "agent_context": "dict|None",
                },
                {
                    "attempts": "list[dict]",
                    "best_attempt": "dict|None",
                    "final_action": "dict|None",
                    "sweep_policy": "dict",
                    "verifier_boundary": "dict",
                },
                "Try caller-ordered visual contact-frame candidates with the generic refine primitive until contact, state change, or the requested attempt budget is exhausted.",
            ),
            self._primitive_card(
                "settle_robocasa_environment",
                "L3",
                {
                    "steps": "int",
                    "gripper_command": "float",
                    "base_mode_value": "float|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "execution_status": "str",
                    "steps_executed": "int",
                    "step_summaries": "list[dict]",
                    "live_observation": "dict",
                    "visual_evidence": "dict",
                    "pose_evidence": "dict",
                },
                "Advance the live simulator with caller-selected no-op steps, then expose refreshed public observation evidence.",
            ),
            self._primitive_card(
                "grasp_robocasa_object",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "object_name": "str",
                    "offset": "list[float]|None",
                    "grasp_approach": "center|near_side",
                    "grasp_wrist_yaw_offset_rad": "float",
                    "horizon": "int",
                    "grasp_contact_steps": "int",
                    "grasp_contact_offset": "list[float]|None",
                    "grasp_contact_tolerance": "float|None",
                    "force_grasp_contact_steps": "bool",
                    "grasp_hold_steps": "int",
                    "gripper_command": "float",
                    "grasp_lift_delta": "float",
                    "contact_tolerance": "float",
                    "gain": "float",
                    "max_delta": "float",
                    "arm_delta_frame": "str|None",
                    "mobile_base_enabled": "bool",
                    "base_gain": "float",
                    "base_max_delta": "float",
                    "base_command_sign": "float",
                    "base_mode_value": "float",
                    "motion_trace_limit": "int",
                    "agent_context": "dict|None",
                    "strategy": "str",
                },
                {
                    "action_skill_schema": "dict",
                    "execution_status": "str",
                    "motion_status": "str",
                    "grounded_action": "dict",
                    "distance_before": "float",
                    "distance_after": "float",
                },
                "Grasp a grounded object with center or near-side approach and optional bounded wrist yaw; verify contact and lift.",
            ),
            self._primitive_card(
                "place_robocasa_object_at",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "object_name": "str",
                    "held_object_handle": "str|None",
                    "target_name": "str|None",
                    "target_position": "list[float]|None",
                    "relation": "str",
                    "offset": "list[float]|None",
                    "use_affordance_site": "bool",
                    "affordance_site_name": "str|None",
                    "horizon": "int",
                    "gripper_command": "float",
                    "transport_gripper_command": "float|None",
                    "release_gripper_command": "float",
                    "release_only_when_ready": "bool",
                    "release_xy_tolerance": "float|None",
                    "release_requires_contact": "bool",
                    "release_settle_steps": "int",
                    "post_release_retreat_offset": "list[float]|None",
                    "post_release_retreat_steps": "int",
                    "tolerance": "float",
                    "object_relative_control": "bool",
                    "object_error_gain": "float",
                    "object_error_clip": "float",
                    "object_xy_push_steps": "int",
                    "object_xy_push_align_steps": "int",
                    "object_xy_push_backoff": "float",
                    "object_xy_push_through": "float",
                    "object_xy_push_z_offset": "float",
                    "object_xy_push_reacquire_from_side": "bool",
                    "object_xy_contact_seek_steps": "int",
                    "object_xy_contact_seek_backoff": "float|None",
                    "object_xy_contact_seek_z_offset": "float|None",
                    "contact_guard_enabled": "bool",
                    "contact_tolerance": "float",
                    "contact_guard_tolerance": "float|None",
                    "contact_guard_recover_steps": "int",
                    "contact_guard_offset": "list[float]|None",
                    "gain": "float",
                    "max_delta": "float",
                    "arm_delta_frame": "str|None",
                    "mobile_base_enabled": "bool",
                    "base_gain": "float",
                    "base_max_delta": "float",
                    "base_command_sign": "float",
                    "base_mode_value": "float",
                    "motion_trace_limit": "int",
                    "agent_context": "dict|None",
                    "strategy": "str",
                },
                {
                    "action_skill_schema": "dict",
                    "execution_status": "str",
                    "motion_status": "str",
                    "grounded_action": "dict",
                    "distance_before": "float",
                    "distance_after": "float",
                },
                "Move a grounded object toward a caller-supplied object, fixture, or pose.",
            ),
            self._primitive_card(
                "inspect_robocasa_transport_state",
                "L2",
                {
                    "motion_result": "dict|PrimitiveResult|None",
                    "runtime_report": "dict|None",
                    "placement_refinement_hint": "dict|PrimitiveResult|None",
                    "object_name": "str|None",
                    "previous_state": "dict|PrimitiveResult|None",
                    "contact_tolerance": "float",
                    "gripper_closed_threshold": "float",
                    "coupling_tolerance": "float",
                    "minimum_coupling_motion": "float",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "heldness": "dict",
                    "contact_stability": "dict",
                    "release_gate": "dict",
                    "settle_state": "dict",
                    "live_observation": "dict",
                    "verifier_boundary": "dict",
                },
                "Report live physical heldness, contact, coupling, release, and settle facts without choosing the next call.",
            ),
            self._primitive_card(
                "move_robocasa_ee_to",
                "L3",
                {
                    "prompt": "str|None",
                    "query": "str|None",
                    "target_name": "str|None",
                    "target_position": "list[float]|None",
                    "offset": "list[float]|None",
                    "horizon": "int",
                    "gain": "float",
                    "max_delta": "float",
                    "arm_delta_frame": "str|None",
                    "tolerance": "float",
                    "gripper_command": "float",
                    "avoidance_point": "list[float]|None",
                    "min_distance_from_avoidance": "float|None",
                    "avoidance_tolerance": "float",
                    "stop_when_avoidance_reached": "bool",
                    "agent_context": "dict|None",
                    "strategy": "str",
                },
                {
                    "action_skill_schema": "dict",
                    "execution_status": "str",
                    "motion_status": "str",
                    "moved": "bool",
                    "distance_before": "float",
                    "distance_after": "float",
                    "avoidance_clearance_reached": "bool|None",
                },
                "Move the robot end effector toward a grounded RoboCasa object/pose, optionally enforcing a generic clearance distance from a caller-supplied avoidance point.",
            ),
            self._primitive_card(
                "apply_robocasa_control",
                "L3",
                {
                    "eef_position_delta": "list[float]|None",
                    "eef_rotation_delta": "list[float]|None",
                    "base_delta": "list[float]|None",
                    "gripper_command": "float",
                    "repeat": "int",
                    "arm_delta_frame": "str|None",
                    "base_mode_value": "float",
                    "object_name": "str|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {
                    "execution_status": "str",
                    "runtime_action_schema": "dict",
                    "steps": "list[dict]",
                    "live_observation": "dict",
                },
                "Submit caller-authored low-level end-effector, base, and gripper deltas to the live RoboCasa action space.",
            ),
            self._primitive_card(
                "record_robocasa_evidence",
                "L1",
                {"key": "str", "value": "any"},
                {"artifact_id": "str"},
                "Record agent-selected RoboCasa evidence in the episode trace.",
            ),
        ]
        for card in cards:
            if card.abstraction_level == "L3" and card.name != "apply_robocasa_control":
                card.input_schema = {**card.input_schema, "evidence_handles": "list[str]"}
            if card.name == "move_robocasa_ee_to":
                card.description += (
                    " With evidence_handles, the observed anchor supplies target_position; "
                    "use offset for a caller-chosen approach or manipulation waypoint relative "
                    "to that anchor. Fixture handles are in inspect_robocasa_fixture.affordance_geometries."
                )
        for card in cards:
            card.description += " Names accept unique case/spacing variants and listed aliases only. Common parameter aliases: object/object_name, fixture/fixture_name, evidence_handle/evidence_handles. Actions report action_feedback: arrived, grasped, released; null means unconfirmed, not success."
        if self.config.motion_backend == "robocasa_state_delta_motion":
            cards = [card for card in cards if card.name not in
                     {"open_robocasa_fixture", "close_robocasa_fixture"}]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        allowed = {card.name for card in self.list_primitives()}
        if name not in allowed:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by RoboCasaAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = (
                _invoke_primitive_handler(handler, kwargs)
                if handler is not None
                else PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
            )
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        artifact_id = f"robocasa:diagnostic:{len(self.get_trace().artifacts)}"
        self.get_trace().add_artifact(artifact_id, {"name": name, "kwargs": _to_builtin(kwargs), "result": _to_builtin(result.to_dict())})
        result.artifacts.append(artifact_id)
        from .robocasa_interface_repair import compact_public_result
        return compact_public_result(result)

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported RoboCasa verification scope: {scope}")
        else:
            success, source = self._extract_task_success()
            result = VerificationResult(
                ok=success,
                scope="task",
                message="RoboCasa task success reported by harness verifier" if success else "RoboCasa task is not yet successful",
                metrics={"success": float(success)},
                metadata={"success_source": source, "info_summary": _to_builtin(self._last_info)},
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
        module_status = _robocasa_live_module_status()
        return {
            "live": self.config.live,
            "env_created": self._env is not None,
            "required_live_modules": module_status,
            "missing_live_modules": [name for name, ok in module_status.items() if not ok],
            "robocasa_importable": module_status["robocasa"],
            "robosuite_importable": module_status["robosuite"],
            "mujoco_importable": module_status["mujoco"],
            "gymnasium_importable": module_status["gymnasium"],
            "motion_backend": self.config.motion_backend,
            "skill_executor": self._skill_executor is not None,
            "use_gymnasium_wrapper": self.config.use_gymnasium_wrapper,
        }

    def run_minimal_rollout(self, steps: int = 1) -> JsonDict:
        """Harness-only real env.step smoke; intentionally not an agent primitive."""
        self._require_reset()
        if self._env is None:
            result = {"ok": False, "error": "live_env_not_created", "steps": []}
            self.record_event("harness_minimal_rollout", result)
            return result
        step_action, _, _, _, _ = _zero_action(self._env)
        if step_action is None:
            result = {"ok": False, "error": "action_space_unavailable", "steps": []}
            self.record_event("harness_minimal_rollout", result)
            return result
        records: list[JsonDict] = []
        for step_idx in range(max(1, steps)):
            template, action, _, _, action_key = _zero_action(self._env)
            formatted_action = _format_step_action(template, action, action_key, gripper_command=0.0)
            step_result = self._env.step(formatted_action)
            obs, reward, terminated, truncated, info = _split_step_result(step_result)
            self._last_obs = obs
            self._last_info = dict(info or {})
            self._refresh_scene_registry()
            records.append(
                {
                    "step_index": step_idx,
                    "action_summary": summarize_observation(formatted_action),
                    "reward": _to_builtin(reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_keys": sorted(str(key) for key in self._last_info),
                    "observation_summary": summarize_observation(obs),
                }
            )
            if terminated or truncated:
                break
        result = {"ok": bool(records), "steps": records}
        self.record_event("harness_minimal_rollout", result)
        return result

    def _primitive_settle_robocasa_environment(
        self,
        steps: int = 8,
        gripper_command: float = 0.0,
        base_mode_value: float | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        self._require_reset()
        if self._env is None:
            return PrimitiveResult(
                name="settle_robocasa_environment",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
                    "execution_status": "not_executed",
                    "steps_executed": 0,
                    "private_success_signal_exposed": False,
                },
                error="live_env_not_created",
            )
        step_action, action_vector, _, _, action_key = _zero_action(self._env)
        if step_action is None or action_vector is None:
            return PrimitiveResult(
                name="settle_robocasa_environment",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
                    "execution_status": "not_executed",
                    "steps_executed": 0,
                    "runtime_action_schema": self._runtime_action_schema(),
                    "private_success_signal_exposed": False,
                },
                error="action_space_unavailable",
            )
        records: list[JsonDict] = []
        requested_steps = max(0, int(steps))
        for step_idx in range(requested_steps):
            template, vector, _, _, current_action_key = _zero_action(self._env)
            if template is None or vector is None:
                break
            formatted_action = _format_step_action(
                template,
                vector,
                current_action_key,
                gripper_command=float(gripper_command),
                base_mode_value=base_mode_value,
            )
            obs, _, terminated, truncated, info = _split_step_result(self._env.step(formatted_action))
            self._last_obs = obs
            self._last_info = dict(info or {})
            self._observation_serial += 1
            self._refresh_scene_registry()
            records.append(
                {
                    "step_index": step_idx,
                    "action_summary": summarize_observation(formatted_action),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_keys": sorted(str(key) for key in self._last_info),
                    "observation_summary": summarize_observation(obs),
                }
            )
            if terminated or truncated:
                break
        return PrimitiveResult(
            name="settle_robocasa_environment",
            ok=requested_steps == 0 or bool(records),
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "execution_status": "executed" if records or requested_steps == 0 else "not_executed",
                "requested_steps": requested_steps,
                "steps_executed": len(records),
                "step_summaries": records,
                "runtime_action_schema": self._runtime_action_schema(),
                "live_observation": summarize_observation(self._last_obs),
                "objects": deepcopy(self._objects),
                "fixtures": deepcopy(self._fixtures),
                "pose_evidence": self._pose_evidence(),
                "visual_evidence": self._visual_evidence(),
                "private_success_signal_exposed": False,
                "verifier_boundary": {
                    "task_completion_claimed_by_primitive": False,
                    "private_task_signal_exposed": False,
                    "harness_verify_required": True,
                },
            },
            error=None if records or requested_steps == 0 else "no_settle_steps_executed",
        )

    def _merged_config(self, overrides: JsonDict) -> RoboCasaRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return RoboCasaRuntimeConfig(**data)

    def _make_env(self, config: RoboCasaRuntimeConfig, seed: int | None) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.env_id, {**config.to_dict(), "seed": seed})
        module_status = _robocasa_live_module_status()
        missing_modules = [name for name, ok in module_status.items() if not ok]
        if missing_modules:
            required = "`, `".join(ROBOCASA_REQUIRED_LIVE_MODULES)
            missing = "`, `".join(missing_modules)
            raise RuntimeError(
                f"RoboCasa live runtime requires importable Python modules `{required}` plus MuJoCo assets. "
                f"Missing Python modules: `{missing}`."
            )
        _configure_external_numba_cache()
        import robocasa  # noqa: F401 - registers RoboCasa Gymnasium env IDs.
        # A strict OpenHands run supplies an attempt-private, writable mirror
        # through the environment.  Prefer that sealed coordinator binding
        # over the case's repo-local default, which is mounted read-only in the
        # Agent Server container.
        _activate_robocasa_asset_cache(
            os.environ.get("ROBOCASA_ASSET_CACHE_DIR") or config.asset_cache_dir
        )

        if not config.use_gymnasium_wrapper:
            return _make_direct_robosuite_env(config, seed)

        import gymnasium as gym

        kwargs = dict(config.env_kwargs)
        kwargs.update(
            {
                "split": config.split,
                "robots": config.robot,
                "camera_names": config.camera_names,
                "camera_widths": config.camera_widths,
                "camera_heights": config.camera_heights,
                "camera_depths": config.camera_depths,
                "seed": seed,
            }
        )
        return gym.make(config.env_id, **kwargs)

    def _refresh_scene_registry(self) -> None:
        unwrapped = _unwrap_env(self._env)
        self._objects = _extract_objects(unwrapped)
        self._fixtures = _extract_fixtures(unwrapped)

    def _benchmark_task_description(self) -> str | None:
        return _task_description(self._last_obs) or _env_task_description(self._env)

    def _primitive_observe_robocasa_kitchen_state(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        include_raw: bool = False,
        verbose: bool = False,
    ) -> PrimitiveResult:
        output = {
            "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "runtime": self.runtime_available(),
            "objects": deepcopy(self._objects),
            "fixtures": deepcopy(self._fixtures),
            "pose_evidence": self._pose_evidence(),
            "visual_evidence": self._visual_evidence(),
            "action_skill_schema": self._action_skill_schema(),
            "state_summary": summarize_observation(self._last_obs),
            "raw_observation_omitted": True,
        }
        if verbose:
            return PrimitiveResult(name="observe_robocasa_kitchen_state", ok=True, output=output)
        self.get_trace().add_artifact(f"robocasa:kitchen-diagnostic:{len(self.get_trace().artifacts)}", _to_builtin(output))
        return PrimitiveResult(name="observe_robocasa_kitchen_state", ok=True, output=self._compact_kitchen_state())

    def _primitive_observe_robocasa_rgbd(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        camera_name: str | None = None,
    ) -> PrimitiveResult:
        self._last_obs = _inject_direct_robocasa_segmentation(self._env, self._last_obs, self.config)
        cameras = extract_camera_summaries(self._last_obs, camera_name=camera_name)
        segmentation_instances = self._robocasa_segmentation_instances(camera_name=camera_name)
        visual_runtime = summarize_robocasa_visual_runtime(cameras)
        observation_ref = self._observation_ref(source="robocasa_last_rgbd_observation")
        artifacts: list[str] = []
        for uid, summary in cameras.items():
            artifact_id = f"robocasa:rgbd:{uid}:{len(self.get_trace().artifacts)}"
            self.get_trace().add_artifact(
                artifact_id,
                {
                    "camera_name": uid,
                    "observation_ref": observation_ref,
                    "summary": summary,
                    "modalities": visual_runtime.get("modalities_by_camera", {}).get(uid, []),
                    "segmentation_instances": segmentation_instances.get(uid, []),
                },
            )
            artifacts.append(artifact_id)
        return PrimitiveResult(
            name="observe_robocasa_rgbd",
            ok=bool(cameras),
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "observation_ref": observation_ref,
                "cameras": cameras,
                "rgbd_evidence": _filter_camera_modalities(cameras, {"rgb", "depth"}),
                "segmentation_evidence": _filter_camera_modalities(cameras, {"segmentation"}),
                "visual_runtime": visual_runtime,
                "segmentation_instances": segmentation_instances,
            },
            artifacts=artifacts,
            error=None if cameras else "no_rgbd_camera_data",
        )

    def _primitive_ground_robocasa_visual_target(
        self,
        prompt: str | None = None,
        camera_name: str | None = None,
        segmentation_id: int | None = None,
        entity_name: str | None = None,
        world_position: list[float] | None = None,
        max_world_distance: float | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        """Bind a caller-selected native visual target to an action handle.

        The caller can select an exact segmentation id, or provide an observed
        world-space affordance point and let the primitive bind the nearest
        native visual instance for the same entity. The latter keeps the
        operation visual-grounded when the simulator exposes a button affordance
        point in state but only exposes the parent fixture geometry in the
        segmentation stream.
        """

        cameras = extract_camera_summaries(self._last_obs, camera_name=camera_name)
        instances_by_camera = self._robocasa_segmentation_instances(camera_name=camera_name)
        requested_world_position = _coerce_vector_list(world_position)
        prompt = prompt or (f"Ground the caller-selected entity {entity_name}" if entity_name else "Ground the selected visual instance")
        if camera_name is None and segmentation_id is None and requested_world_position is None and entity_name:
            matches = [
                (uid, item)
                for uid, instances in sorted(instances_by_camera.items())
                for item in instances
                if item.get("entity_name") == entity_name
            ]
            if matches:
                camera_name, selected = matches[0]
                segmentation_id = int(selected["segmentation_id"])
            else:
                # Public scene pose plus a current RGB/depth observation is the
                # existing fallback; this does not consult task success.
                requested_world_position = self._registry_position(entity_name)
        nearest_selection: JsonDict | None = None
        if (camera_name is None or segmentation_id is None) and requested_world_position is not None:
            nearest_selection = _nearest_robocasa_visual_instance(
                instances_by_camera,
                entity_name=entity_name,
                world_position=requested_world_position,
                max_world_distance=max_world_distance,
            )
            if nearest_selection is not None:
                camera_name = str(nearest_selection["camera_name"])
                segmentation_id = int(nearest_selection["segmentation_id"])
        if (camera_name is None or segmentation_id is None) and requested_world_position is not None and entity_name:
            selected_camera_name: str | None = None
            selected_camera: JsonDict | None = None
            if camera_name is not None:
                camera = cameras.get(camera_name)
                if isinstance(camera, dict) and {"rgb", "depth"}.issubset(camera):
                    selected_camera_name = camera_name
                    selected_camera = camera
            if selected_camera_name is None:
                for candidate_name, camera in sorted(cameras.items()):
                    if isinstance(camera, dict) and {"rgb", "depth"}.issubset(camera):
                        selected_camera_name = str(candidate_name)
                        selected_camera = camera
                        break
            if selected_camera_name is not None and selected_camera is not None:
                entity_payload = self._objects.get(entity_name) or self._fixtures.get(entity_name) or {}
                entity_kind = entity_payload.get("kind")
                if entity_kind is None:
                    if entity_name in self._objects:
                        entity_kind = "object"
                    elif entity_name in self._fixtures:
                        entity_kind = "fixture"
                handle = (
                    f"robocasa:visual:{self._observation_serial}:{selected_camera_name}:"
                    f"state:{len(self._visual_grounding_handles)}"
                )
                grounding = {
                    "prompt": prompt,
                    "query": query,
                    "agent_context": dict(agent_context or {}),
                    "entity_name": entity_name,
                    "camera_name": selected_camera_name,
                    "segmentation_id": None,
                    "instance": {
                        "entity_name": entity_name,
                        "entity_kind": entity_kind,
                        "binding_source": "public_state_pose_with_rgbd_observation",
                    },
                    "depth_statistics": None,
                    "point_world": requested_world_position,
                    "entity_kind": entity_kind,
                    "button_name": None,
                    "native_geometry": deepcopy(entity_payload) if isinstance(entity_payload, dict) else {},
                    "requested_world_position": requested_world_position,
                    "distance_to_requested_world_position": 0.0,
                    "observation_serial": self._observation_serial,
                    "registry_pose_world": self._registry_position(entity_name),
                    "source": "upstream_robocasa_rgb_depth_and_public_state_pose",
                    "fallback_mode": "rgbd_state_pose_no_segmentation",
                }
                self._visual_grounding_handles[handle] = grounding
                self.get_trace().add_artifact(handle, deepcopy(grounding))
                return PrimitiveResult(
                    name="ground_robocasa_visual_target",
                    ok=True,
                    output={"evidence_handle": handle, "entity_name": entity_name, "grounding": grounding, "native_modalities": selected_camera},
                    artifacts=[handle],
                )
        if camera_name is None or segmentation_id is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "prompt": prompt,
                    "query": query,
                    "camera_name": camera_name,
                    "segmentation_id": segmentation_id,
                    "world_position": requested_world_position,
                    "fabricated_evidence": False,
                },
                error="visual_grounding_requires_segmentation_id_or_world_position",
            )
        camera = cameras.get(camera_name, {})
        missing = sorted({"rgb", "depth", "segmentation"}.difference(camera))
        if missing:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "prompt": prompt,
                    "query": query,
                    "camera_name": camera_name,
                    "segmentation_id": int(segmentation_id),
                    "native_modalities": camera,
                    "unsupported_modalities": missing,
                    "fabricated_evidence": False,
                },
                error="visual_grounding_modalities_unsupported",
            )
        instances = instances_by_camera.get(camera_name, [])
        selected = next(
            (item for item in instances if int(item.get("segmentation_id", -1)) == int(segmentation_id)),
            None,
        )
        if selected is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "prompt": prompt,
                    "query": query,
                    "camera_name": camera_name,
                    "segmentation_id": int(segmentation_id),
                    "available_instances": instances,
                    "fabricated_evidence": False,
                },
                error="segmentation_instance_not_observed",
            )
        observed_name = selected.get("entity_name")
        if observed_name is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={"instance": selected, "fabricated_evidence": False},
                error="segmentation_instance_unbound",
            )
        if entity_name is not None and observed_name is not None and entity_name != observed_name:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={"requested_entity": entity_name, "observed_entity": observed_name, "instance": selected},
                error="visual_entity_binding_mismatch",
            )
        depth_statistics = _robocasa_segmented_depth_statistics(
            self._last_obs, camera_name=camera_name, segmentation_id=int(segmentation_id)
        )
        if depth_statistics is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={"instance": selected, "fabricated_evidence": False},
                error="segmented_depth_evidence_unavailable",
            )
        handle = (
            f"robocasa:visual:{self._observation_serial}:{camera_name}:"
            f"{int(segmentation_id)}:{len(self._visual_grounding_handles)}"
        )
        point_world = _robocasa_segmented_world_point(
            self._last_obs,
            camera_name=camera_name,
            segmentation_id=int(segmentation_id),
        )
        if point_world is None:
            point_world = _vector_to_list(selected.get("native_world_position"))
        if point_world is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={"instance": selected, "fabricated_evidence": False},
                error="visual_grounding_world_point_unavailable",
            )
        grounding = {
            "prompt": prompt,
            "query": query,
            "agent_context": dict(agent_context or {}),
            "entity_name": entity_name or observed_name,
            "camera_name": camera_name,
            "segmentation_id": int(segmentation_id),
            "instance": deepcopy(selected),
            "depth_statistics": depth_statistics,
            "point_world": point_world,
            "entity_kind": selected.get("entity_kind"),
            "button_name": selected.get("button_name"),
            "native_geometry": deepcopy(selected.get("native_geometry")),
            "requested_world_position": requested_world_position,
            "distance_to_requested_world_position": (
                nearest_selection.get("distance_to_requested_world_position")
                if isinstance(nearest_selection, dict)
                else None
            ),
            "observation_serial": self._observation_serial,
            "registry_pose_world": self._registry_position(entity_name or observed_name),
            "source": "upstream_robocasa_rgb_depth_segmentation",
        }
        self._visual_grounding_handles[handle] = grounding
        self.get_trace().add_artifact(handle, deepcopy(grounding))
        return PrimitiveResult(
            name="ground_robocasa_visual_target",
            ok=True,
            output={"evidence_handle": handle, "entity_name": entity_name or observed_name, "grounding": grounding, "native_modalities": camera},
            artifacts=[handle],
        )

    def _primitive_inspect_robocasa_object(
        self,
        prompt: str | None = None,
        object_name: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._inspect_entity("inspect_robocasa_object", self._objects, object_name, prompt, query, agent_context or {})

    def _primitive_locate_robocasa_object(
        self,
        prompt: str | None = None,
        object_name: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._inspect_entity("locate_robocasa_object", self._objects, object_name, prompt, query, agent_context or {})

    def _primitive_inspect_robocasa_fixture(
        self,
        prompt: str | None = None,
        fixture_name: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._inspect_entity("inspect_robocasa_fixture", self._fixtures, fixture_name, prompt, query, agent_context or {})

    def _primitive_locate_robocasa_fixture(
        self,
        prompt: str | None = None,
        fixture_name: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._inspect_entity("locate_robocasa_fixture", self._fixtures, fixture_name, prompt, query, agent_context or {})

    def _primitive_inspect_robocasa_affordance(
        self,
        prompt: str | None = None,
        query: str | None = None,
        object_name: str | None = None,
        target_name: str | None = None,
        relation: str = "at",
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        context = agent_context or {}
        selected_object_name = _select_name(self._objects, object_name)
        selected_target_name, selected_target_registry = self._select_target_entity(target_name)
        object_payload = deepcopy(self._objects[selected_object_name]) if selected_object_name in self._objects else None
        target_payload = (
            deepcopy(selected_target_registry[selected_target_name])
            if selected_target_name is not None and selected_target_name in selected_target_registry
            else None
        )
        object_position = self._entity_position(selected_object_name, object_payload, fallback_key="obj")
        target_position = self._entity_position(selected_target_name, target_payload, fallback_key=selected_target_name)
        eef_position = _vector(_read_current_observation(self._env) or self._last_obs, "robot0_eef_pos") if self._env is not None else None
        affordance_target = _placement_affordance_target(
            relation,
            target_payload=target_payload,
            query=query,
            agent_context=context,
            site_name=None,
        )
        placement_offset = (
            [0.0, 0.0, 0.0]
            if affordance_target is not None
            else _placement_offset(relation, target_payload=target_payload, query=query, agent_context=context)
        )
        if affordance_target is not None:
            target_position = affordance_target["position"]
        grasp_offset = _grasp_offset_for_object(object_payload)
        grasp_distance = _distance_between(eef_position, _add_vectors(object_position, grasp_offset))
        grasp_candidates = [{"id": "center", "grasp_approach": "center", "grasp_wrist_yaw_offset_rad": 0.0}]
        side_offsets = _near_side_grasp_offsets(object_position, eef_position, object_payload)
        if side_offsets is not None:
            grasp_candidates.extend([
                {"id": "near_side", "grasp_approach": "near_side", "grasp_wrist_yaw_offset_rad": 0.0,
                 "pregrasp_offset": side_offsets[0], "contact_offset": side_offsets[1]},
                {"id": "near_side_rotated", "grasp_approach": "near_side", "grasp_wrist_yaw_offset_rad": .35,
                 "pregrasp_offset": side_offsets[0], "contact_offset": side_offsets[1]},
            ])
        placement_distance = _distance_between(object_position, _add_vectors(target_position, placement_offset))
        output = {
            "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
            "prompt": prompt,
            "query": query,
            "agent_context": context,
            "relation": relation,
            "selected_object": {"name": selected_object_name, **object_payload} if object_payload is not None else None,
            "selected_target": {"name": selected_target_name, **target_payload} if target_payload is not None else None,
            "grasp_affordance": {
                "target_position": _to_builtin(_vector_to_list(_add_vectors(object_position, grasp_offset))),
                "offset": grasp_offset,
                "distance_from_eef": grasp_distance,
                "requires_contact_or_gripper_observation": True,
                "estimated_reachable": grasp_distance is None or grasp_distance < 1.25,
            },
            "grasp_candidates": grasp_candidates,
            "placement_affordance": {
                "target_position": _to_builtin(_vector_to_list(_add_vectors(target_position, placement_offset))),
                "offset": placement_offset,
                "affordance_target": affordance_target,
                "relation": relation,
                "object_to_target_distance": placement_distance,
                "target_entity_kind": target_payload.get("kind") if isinstance(target_payload, dict) else None,
                "estimated_feasible": target_position is not None,
            },
            "visual_evidence": self._visual_evidence(),
            "pose_evidence": self._pose_evidence(),
            "verifier_boundary": {
                "task_completion_claimed_by_primitive": False,
                "private_task_signal_exposed": False,
                "harness_verify_required": True,
            },
        }
        return PrimitiveResult(
            name="inspect_robocasa_affordance",
            ok=object_payload is not None and (target_name is None or target_payload is not None),
            output=output,
            error=None if object_payload is not None and (target_name is None or target_payload is not None) else "affordance_entities_not_found",
        )

    def _primitive_inspect_robocasa_button_contact_frame(
        self,
        fixture_name: str,
        button_name: str | None = None,
        button_position: list[float] | None = None,
        visual_binding_tolerance: float | None = None,
        evidence_handles: list[str] | None = None,
        candidate_limit: int = 8,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        context = agent_context or {}
        selected_fixture_name = _select_name(self._fixtures, fixture_name)
        fixture_payload = (
            deepcopy(self._fixtures[selected_fixture_name])
            if selected_fixture_name is not None and selected_fixture_name in self._fixtures
            else None
        )
        if selected_fixture_name is None or fixture_payload is None:
            return PrimitiveResult(
                name="inspect_robocasa_button_contact_frame",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                    "fixture_name": fixture_name,
                    "available_fixtures": sorted(self._fixtures),
                    "fabricated_evidence": False,
                },
                error="fixture_not_found",
            )

        resolved_button_position = _coerce_vector_value(button_position)
        resolved_button_name = str(button_name) if button_name is not None else None
        if resolved_button_position is None:
            button_selection = _select_fixture_button_affordance(
                fixture_payload,
                button_name=button_name,
            )
            if button_selection is not None:
                resolved_button_name, resolved_button_position = button_selection
        if resolved_button_position is None:
            return PrimitiveResult(
                name="inspect_robocasa_button_contact_frame",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                    "selected_fixture": {"name": selected_fixture_name, **fixture_payload},
                    "available_buttons": _fixture_button_names(fixture_payload),
                    "fabricated_evidence": False,
                },
                error="fixture_button_affordance_unavailable",
            )

        obs = _read_current_observation(self._env) or self._last_obs
        eef_position = _vector(obs, "robot0_eef_pos")
        fixture_position = _coerce_vector_value(fixture_payload.get("pos"))
        if fixture_position is None:
            fixture_position = _coerce_vector_value(fixture_payload.get("pose_world"))
        raw_button_position = np.asarray(resolved_button_position, dtype=np.float32)
        binding_kwargs: JsonDict = {
            "fixture_name": selected_fixture_name,
            "button_name": resolved_button_name,
            "button_position": _vector_to_list(raw_button_position),
            "visual_binding_tolerance": visual_binding_tolerance,
        }
        visual_binding_error = None
        visual_grounding_evidence: list[JsonDict] = []
        supplied_handles = [str(item) for item in evidence_handles or []]
        if supplied_handles:
            unknown = [handle for handle in supplied_handles if handle not in self._visual_grounding_handles]
            stale = [
                handle
                for handle in supplied_handles
                if handle in self._visual_grounding_handles
                and int(self._visual_grounding_handles[handle].get("observation_serial", -1)) != self._observation_serial
            ]
            if unknown:
                visual_binding_error = "visual_grounding_evidence_invalid"
            elif stale:
                visual_binding_error = "visual_grounding_evidence_stale"
            else:
                visual_grounding_evidence = [
                    deepcopy(self._visual_grounding_handles[handle]) for handle in supplied_handles
                ]
                bound_names = {
                    str(item["entity_name"])
                    for item in visual_grounding_evidence
                    if item.get("entity_name") is not None
                }
                if selected_fixture_name not in bound_names:
                    visual_binding_error = "visual_grounding_entity_mismatch"
                else:
                    visual_binding_error = _bind_robocasa_visual_action_target(
                        "press_robocasa_fixture_button",
                        binding_kwargs,
                        visual_grounding_evidence,
                    )

        bound_button_position = _coerce_vector_value(binding_kwargs.get("button_position"))
        if bound_button_position is None:
            bound_button_position = raw_button_position
        visual_anchor_position = _coerce_vector_value(binding_kwargs.get("visual_button_anchor_position"))
        if visual_anchor_position is None:
            visual_anchor_position = bound_button_position if supplied_handles and visual_binding_error is None else None

        candidates = _robocasa_button_contact_frame_candidates(
            self._env,
            fixture_position=fixture_position,
            button_position=bound_button_position,
            eef_position=eef_position,
            candidate_limit=candidate_limit,
        )
        output = {
            "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
            "prompt": prompt,
            "query": query,
            "agent_context": context,
            "selected_fixture": {"name": selected_fixture_name, **fixture_payload},
            "selected_button": {
                "button_name": resolved_button_name,
                "raw_position": _to_builtin(_vector_to_list(raw_button_position)),
                "bound_position": _to_builtin(_vector_to_list(bound_button_position)),
                "position_source": (
                    "caller.button_position" if button_position is not None else "fixture_affordance_sites.start_buttons"
                ),
            },
            "visual_binding": {
                "evidence_handles": supplied_handles,
                "ok": bool(supplied_handles and visual_binding_error is None),
                "error": visual_binding_error,
                "mode": binding_kwargs.get("visual_button_binding_mode"),
                "visual_target_source": binding_kwargs.get("visual_target_source"),
                "visual_anchor_position": _to_builtin(_vector_to_list(visual_anchor_position)),
                "grounding_evidence": visual_grounding_evidence,
                "action_requires_same_observation_evidence": True,
            },
            "eef_position": _to_builtin(_vector_to_list(eef_position)),
            "fixture_position": _to_builtin(_vector_to_list(fixture_position)),
            "contact_frame_candidates": candidates,
            "press_primitive_parameters": {
                "primitive": "press_robocasa_fixture_button",
                "fixture_name": selected_fixture_name,
                "button_name": resolved_button_name,
                "button_position": _to_builtin(_vector_to_list(bound_button_position)),
                "evidence_handles": supplied_handles,
                "candidate_fields_for_action": [
                    "button_offset",
                    "press_direction_vector",
                    "gripper_contact_surface_axis",
                ],
            },
            "verifier_boundary": {
                "task_completion_claimed_by_primitive": False,
                "private_task_signal_exposed": False,
                "harness_verify_required": True,
            },
            "fabricated_evidence": False,
        }
        ok = bool(candidates) and (not supplied_handles or visual_binding_error is None)
        return PrimitiveResult(
            name="inspect_robocasa_button_contact_frame",
            ok=ok,
            output=output,
            error=None if ok else (visual_binding_error or "contact_frame_candidates_unavailable"),
        )

    def _primitive_open_robocasa_fixture(
        self,
        fixture_name: str,
        evidence_handles: list[str] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "open_robocasa_fixture", fixture_name=fixture_name, evidence_handles=evidence_handles, prompt=prompt, query=query, agent_context=agent_context or {}, strategy=strategy
        )

    def _primitive_close_robocasa_fixture(
        self,
        fixture_name: str,
        evidence_handles: list[str] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "close_robocasa_fixture", fixture_name=fixture_name, evidence_handles=evidence_handles, prompt=prompt, query=query, agent_context=agent_context or {}, strategy=strategy
        )

    def _primitive_press_robocasa_fixture_button(
        self,
        fixture_name: str,
        evidence_handles: list[str] | None = None,
        button_name: str | None = None,
        button_position: list[float] | None = None,
        visual_binding_tolerance: float | None = None,
        use_visual_button_anchor_position: bool = False,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 80,
        approach_steps: int | None = None,
        approach_distance: float = 0.08,
        approach_stop_distance: float | None = None,
        approach_patience_steps: int = 12,
        button_offset: list[float] | None = None,
        button_anchor_adjustment: list[float] | None = None,
        bind_gripper_contact_geometry: bool = False,
        gripper_contact_surface_axis: list[float] | None = None,
        tolerance: float = 0.025,
        press_depth: float = 0.035,
        max_press_depth: float | None = None,
        press_contact_seek_steps: int = 0,
        press_contact_seek_depth: float | None = None,
        press_direction_sign: float = -1.0,
        press_direction_vector: list[float] | None = None,
        press_steps: int = 16,
        hold_steps: int = 4,
        retreat_steps: int | None = None,
        retreat_distance: float = 0.22,
        gain: float = 2.0,
        max_delta: float = 0.3,
        mobile_base_enabled: bool = False,
        mobile_base_active_phases: list[str] | None = None,
        base_gain: float = 1.0,
        base_max_delta: float = 0.35,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_delta_frame: str = "world",
        base_xy_deadband: float | None = None,
        arm_delta_frame: str | None = None,
        gripper_command: float = 1.0,
        motion_trace_limit: int = 5,
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "press_robocasa_fixture_button",
            fixture_name=fixture_name,
            evidence_handles=evidence_handles,
            button_name=button_name,
            button_position=button_position,
            visual_binding_tolerance=visual_binding_tolerance,
            use_visual_button_anchor_position=use_visual_button_anchor_position,
            button_offset=button_offset,
            button_anchor_adjustment=button_anchor_adjustment,
            bind_gripper_contact_geometry=bind_gripper_contact_geometry,
            gripper_contact_surface_axis=gripper_contact_surface_axis,
            horizon=horizon,
            approach_steps=approach_steps,
            approach_distance=approach_distance,
            approach_stop_distance=approach_stop_distance,
            approach_patience_steps=approach_patience_steps,
            tolerance=tolerance,
            press_depth=press_depth,
            max_press_depth=max_press_depth,
            press_contact_seek_steps=press_contact_seek_steps,
            press_contact_seek_depth=press_contact_seek_depth,
            press_direction_sign=press_direction_sign,
            press_direction_vector=press_direction_vector,
            press_steps=press_steps,
            hold_steps=hold_steps,
            retreat_steps=retreat_steps,
            retreat_distance=retreat_distance,
            gain=gain,
            max_delta=max_delta,
            mobile_base_enabled=mobile_base_enabled,
            mobile_base_active_phases=mobile_base_active_phases,
            base_gain=base_gain,
            base_max_delta=base_max_delta,
            base_command_sign=base_command_sign,
            base_mode_value=base_mode_value,
            base_delta_frame=base_delta_frame,
            base_xy_deadband=base_xy_deadband,
            arm_delta_frame=arm_delta_frame,
            gripper_command=gripper_command,
            motion_trace_limit=motion_trace_limit,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
            strategy=strategy,
        )

    def _primitive_refine_robocasa_button_contact_search(
        self,
        fixture_name: str | None = None,
        evidence_handles: list[str] | None = None,
        button_name: str | None = None,
        button_position: list[float] | None = None,
        visual_binding_tolerance: float | None = None,
        previous_action_output: JsonDict | None = None,
        contact_frame_candidate: JsonDict | None = None,
        move_to_previous_best: bool = True,
        prealign_steps: int = 24,
        prealign_gain: float = 2.0,
        prealign_max_delta: float = 0.35,
        use_visual_button_anchor_position: bool = False,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 160,
        approach_steps: int | None = 24,
        approach_distance: float = 0.02,
        approach_stop_distance: float | None = 0.02,
        approach_patience_steps: int = 8,
        button_offset: list[float] | None = None,
        button_anchor_adjustment: list[float] | None = None,
        bind_gripper_contact_geometry: bool = True,
        gripper_contact_surface_axis: list[float] | None = None,
        tolerance: float = 0.02,
        press_depth: float = 0.06,
        max_press_depth: float | None = 0.12,
        press_contact_seek_steps: int = 80,
        press_contact_seek_depth: float | None = 0.18,
        press_direction_sign: float = -1.0,
        press_direction_vector: list[float] | None = None,
        press_steps: int = 60,
        hold_steps: int = 20,
        retreat_steps: int | None = 0,
        retreat_distance: float = 0.0,
        gain: float = 3.0,
        max_delta: float = 0.7,
        mobile_base_enabled: bool = False,
        mobile_base_active_phases: list[str] | None = None,
        base_gain: float = 1.0,
        base_max_delta: float = 0.15,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_delta_frame: str = "world",
        base_xy_deadband: float | None = None,
        arm_delta_frame: str | None = None,
        gripper_command: float = 1.0,
        motion_trace_limit: int = 8,
    ) -> PrimitiveResult:
        context = agent_context or {}
        previous = previous_action_output if isinstance(previous_action_output, dict) else {}
        candidate = contact_frame_candidate if isinstance(contact_frame_candidate, dict) else {}
        previous_affordance = previous.get("button_affordance") if isinstance(previous.get("button_affordance"), dict) else {}
        previous_best = (
            (previous.get("step_summary") or {}).get("best_step")
            if isinstance(previous.get("step_summary"), dict)
            else None
        )
        previous_best = previous_best if isinstance(previous_best, dict) else {}

        resolved_fixture_name = fixture_name or previous.get("fixture_name")
        if not isinstance(resolved_fixture_name, str) or not resolved_fixture_name:
            return PrimitiveResult(
                name="refine_robocasa_button_contact_search",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                    "previous_action_output_available": bool(previous),
                    "verifier_boundary": _verifier_boundary(),
                },
                error="fixture_name_required",
            )

        resolved_button_position = _coerce_vector_value(button_position)
        if resolved_button_position is None:
            for source in (
                previous.get("button_position"),
                previous_affordance.get("raw_position"),
                previous_affordance.get("position"),
                candidate.get("target_position"),
            ):
                resolved_button_position = _coerce_vector_value(source)
                if resolved_button_position is not None:
                    break
        if resolved_button_position is None:
            return PrimitiveResult(
                name="refine_robocasa_button_contact_search",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                    "fixture_name": resolved_fixture_name,
                    "previous_action_output_available": bool(previous),
                    "verifier_boundary": _verifier_boundary(),
                },
                error="button_position_required",
            )

        resolved_button_offset = button_offset
        if resolved_button_offset is None:
            resolved_button_offset = candidate.get("button_offset") or previous.get("button_offset")
        resolved_button_anchor_adjustment = button_anchor_adjustment
        if resolved_button_anchor_adjustment is None and candidate:
            resolved_button_anchor_adjustment = candidate.get("button_anchor_adjustment")
        if resolved_button_anchor_adjustment is None and not candidate:
            resolved_button_anchor_adjustment = previous.get("button_anchor_adjustment")
        resolved_button_anchor_adjustment_source = (
            candidate.get("button_anchor_adjustment_source") if candidate else previous.get("button_anchor_adjustment_source")
        )
        resolved_press_direction_vector = press_direction_vector
        if resolved_press_direction_vector is None:
            resolved_press_direction_vector = candidate.get("press_direction_vector") or previous.get("press_direction_vector")
        resolved_contact_surface_axis = gripper_contact_surface_axis
        if resolved_contact_surface_axis is None:
            resolved_contact_surface_axis = (
                candidate.get("gripper_contact_surface_axis")
                or previous.get("gripper_contact_surface_axis")
                or resolved_press_direction_vector
            )

        prealign = None
        prealign_target = _coerce_vector_value(previous_best.get("eef_position"))
        if move_to_previous_best and prealign_target is not None and prealign_steps > 0:
            prealign = self._primitive_move_robocasa_ee_to(
                target_position=_to_builtin(_vector_to_list(prealign_target)),
                evidence_handles=evidence_handles,
                horizon=prealign_steps,
                gain=prealign_gain,
                max_delta=prealign_max_delta,
                arm_delta_frame=arm_delta_frame,
                query=query,
                agent_context={
                    **context,
                    "refinement_stage": "return_to_previous_best_eef_position",
                    "previous_best_distance": previous_best.get("distance_to_target"),
                },
            )

        action_context = {
            **context,
            "refinement_stage": "local_button_contact_search",
            "previous_best_step_index": previous_best.get("step_index"),
            "previous_best_phase": previous_best.get("phase"),
            "previous_best_distance_to_button": previous_best.get("distance_to_target"),
            "contact_frame_id": candidate.get("contact_frame_id"),
            "control_policy": "generic_trace_conditioned_visual_button_contact_refinement",
        }
        action = self._run_action_skill(
            "press_robocasa_fixture_button",
            fixture_name=resolved_fixture_name,
            evidence_handles=evidence_handles,
            button_name=button_name or previous.get("button_name"),
            button_position=_to_builtin(_vector_to_list(resolved_button_position)),
            visual_binding_tolerance=visual_binding_tolerance,
            use_visual_button_anchor_position=use_visual_button_anchor_position,
            button_offset=resolved_button_offset,
            button_anchor_adjustment=resolved_button_anchor_adjustment,
            button_anchor_adjustment_source=resolved_button_anchor_adjustment_source,
            bind_gripper_contact_geometry=bind_gripper_contact_geometry,
            gripper_contact_surface_axis=resolved_contact_surface_axis,
            horizon=horizon,
            approach_steps=approach_steps,
            approach_distance=approach_distance,
            approach_stop_distance=approach_stop_distance,
            approach_patience_steps=approach_patience_steps,
            tolerance=tolerance,
            press_depth=press_depth,
            max_press_depth=max_press_depth,
            press_contact_seek_steps=press_contact_seek_steps,
            press_contact_seek_depth=press_contact_seek_depth,
            press_direction_sign=press_direction_sign,
            press_direction_vector=resolved_press_direction_vector,
            press_steps=press_steps,
            hold_steps=hold_steps,
            retreat_steps=retreat_steps,
            retreat_distance=retreat_distance,
            gain=gain,
            max_delta=max_delta,
            mobile_base_enabled=mobile_base_enabled,
            mobile_base_active_phases=mobile_base_active_phases,
            base_gain=base_gain,
            base_max_delta=base_max_delta,
            base_command_sign=base_command_sign,
            base_mode_value=base_mode_value,
            base_delta_frame=base_delta_frame,
            base_xy_deadband=base_xy_deadband,
            arm_delta_frame=arm_delta_frame,
            gripper_command=gripper_command,
            motion_trace_limit=motion_trace_limit,
            prompt=prompt,
            query=query,
            agent_context=action_context,
            strategy=strategy,
        )
        return PrimitiveResult(
            name="refine_robocasa_button_contact_search",
            ok=bool((prealign is None or prealign.ok) and action.ok),
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                "prealign": None if prealign is None else prealign.to_dict(),
                "action": action.to_dict(),
                "recovery_source": {
                    "previous_action_output_available": bool(previous),
                    "previous_best_step_available": bool(previous_best),
                    "prealign_target_position": _to_builtin(_vector_to_list(prealign_target)),
                    "contact_frame_candidate_available": bool(candidate),
                    "contact_frame_id": candidate.get("contact_frame_id"),
                    "resolved_fixture_name": resolved_fixture_name,
                    "resolved_button_position": _to_builtin(_vector_to_list(resolved_button_position)),
                    "resolved_button_offset": _to_builtin(_vector_to_list(_coerce_vector_value(resolved_button_offset))),
                    "resolved_button_anchor_adjustment": _to_builtin(
                        _vector_to_list(_coerce_vector_value(resolved_button_anchor_adjustment))
                    ),
                    "resolved_button_anchor_adjustment_source": resolved_button_anchor_adjustment_source,
                    "resolved_press_direction_vector": _to_builtin(
                        _vector_to_list(_coerce_vector_value(resolved_press_direction_vector))
                    ),
                    "resolved_gripper_contact_surface_axis": _to_builtin(
                        _vector_to_list(_coerce_vector_value(resolved_contact_surface_axis))
                    ),
                },
                "official_task_completion_claimed": False,
                "verifier_boundary": _verifier_boundary(),
            },
            error=None if bool((prealign is None or prealign.ok) and action.ok) else (action.error or (prealign.error if prealign is not None else None)),
        )

    def _primitive_sweep_robocasa_button_contact_candidates(
        self,
        contact_frame_candidates: list[JsonDict] | None = None,
        candidate_indices: list[int] | None = None,
        max_attempts: int | None = None,
        stop_on_interaction: bool = True,
        fixture_name: str | None = None,
        evidence_handles: list[str] | None = None,
        button_name: str | None = None,
        button_position: list[float] | None = None,
        visual_binding_tolerance: float | None = None,
        previous_action_output: JsonDict | None = None,
        move_to_previous_best: bool = True,
        prealign_steps: int = 24,
        prealign_gain: float = 2.0,
        prealign_max_delta: float = 0.35,
        use_visual_button_anchor_position: bool = False,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 160,
        approach_steps: int | None = 24,
        approach_distance: float = 0.02,
        approach_stop_distance: float | None = 0.02,
        approach_patience_steps: int = 8,
        bind_gripper_contact_geometry: bool = True,
        tolerance: float = 0.02,
        press_depth: float = 0.06,
        max_press_depth: float | None = 0.12,
        press_contact_seek_steps: int = 80,
        press_contact_seek_depth: float | None = 0.18,
        press_direction_sign: float = -1.0,
        press_steps: int = 60,
        hold_steps: int = 20,
        retreat_steps: int | None = 0,
        retreat_distance: float = 0.0,
        gain: float = 3.0,
        max_delta: float = 0.7,
        mobile_base_enabled: bool = False,
        mobile_base_active_phases: list[str] | None = None,
        base_gain: float = 1.0,
        base_max_delta: float = 0.15,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_delta_frame: str = "world",
        base_xy_deadband: float | None = None,
        arm_delta_frame: str | None = None,
        gripper_command: float = 1.0,
        motion_trace_limit: int = 8,
    ) -> PrimitiveResult:
        context = agent_context or {}
        candidates = [candidate for candidate in (contact_frame_candidates or []) if isinstance(candidate, dict)]
        if not candidates:
            return PrimitiveResult(
                name="sweep_robocasa_button_contact_candidates",
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                    "sweep_policy": {
                        "candidate_count": 0,
                        "candidate_indices": candidate_indices,
                        "stop_on_interaction": bool(stop_on_interaction),
                    },
                    "verifier_boundary": _verifier_boundary(),
                },
                error="contact_frame_candidates_required",
            )

        if candidate_indices is None:
            ordered_indices = list(range(len(candidates)))
        else:
            ordered_indices = []
            for raw_index in candidate_indices:
                index = int(raw_index)
                if index < 0 or index >= len(candidates):
                    return PrimitiveResult(
                        name="sweep_robocasa_button_contact_candidates",
                        ok=False,
                        output={
                            "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                            "candidate_count": len(candidates),
                            "candidate_indices": candidate_indices,
                            "verifier_boundary": _verifier_boundary(),
                        },
                        error=f"candidate_index_out_of_range:{index}",
                    )
                if index not in ordered_indices:
                    ordered_indices.append(index)
        if max_attempts is not None:
            ordered_indices = ordered_indices[: max(0, int(max_attempts))]

        attempts: list[JsonDict] = []
        best_attempt: JsonDict | None = None
        confirmed_attempt: JsonDict | None = None
        best_distance: float | None = None
        final_action: JsonDict | None = None
        current_previous = previous_action_output if isinstance(previous_action_output, dict) else None
        stopped_reason = "attempt_budget_exhausted"

        for attempt_index, candidate_index in enumerate(ordered_indices):
            candidate = candidates[candidate_index]
            resolved_button_offset = candidate.get("button_offset")
            resolved_button_anchor_adjustment = candidate.get("button_anchor_adjustment")
            resolved_press_direction_vector = candidate.get("press_direction_vector")
            resolved_contact_surface_axis = candidate.get("gripper_contact_surface_axis") or resolved_press_direction_vector
            result = self._primitive_refine_robocasa_button_contact_search(
                fixture_name=fixture_name,
                evidence_handles=evidence_handles,
                button_name=button_name,
                button_position=button_position,
                visual_binding_tolerance=visual_binding_tolerance,
                previous_action_output=current_previous,
                contact_frame_candidate=candidate,
                move_to_previous_best=move_to_previous_best,
                prealign_steps=prealign_steps,
                prealign_gain=prealign_gain,
                prealign_max_delta=prealign_max_delta,
                use_visual_button_anchor_position=use_visual_button_anchor_position,
                button_offset=resolved_button_offset,
                button_anchor_adjustment=resolved_button_anchor_adjustment,
                bind_gripper_contact_geometry=bind_gripper_contact_geometry,
                gripper_contact_surface_axis=resolved_contact_surface_axis,
                horizon=horizon,
                approach_steps=approach_steps,
                approach_distance=approach_distance,
                approach_stop_distance=approach_stop_distance,
                approach_patience_steps=approach_patience_steps,
                tolerance=tolerance,
                press_depth=press_depth,
                max_press_depth=max_press_depth,
                press_contact_seek_steps=press_contact_seek_steps,
                press_contact_seek_depth=press_contact_seek_depth,
                press_direction_sign=press_direction_sign,
                press_direction_vector=resolved_press_direction_vector,
                press_steps=press_steps,
                hold_steps=hold_steps,
                retreat_steps=retreat_steps,
                retreat_distance=retreat_distance,
                gain=gain,
                max_delta=max_delta,
                mobile_base_enabled=mobile_base_enabled,
                mobile_base_active_phases=mobile_base_active_phases,
                base_gain=base_gain,
                base_max_delta=base_max_delta,
                base_command_sign=base_command_sign,
                base_mode_value=base_mode_value,
                base_delta_frame=base_delta_frame,
                base_xy_deadband=base_xy_deadband,
                arm_delta_frame=arm_delta_frame,
                gripper_command=gripper_command,
                motion_trace_limit=motion_trace_limit,
                prompt=prompt,
                query=query,
                agent_context={
                    **context,
                    "sweep_attempt_index": attempt_index,
                    "candidate_list_index": candidate_index,
                    "candidate_index": candidate.get("candidate_index", candidate_index),
                    "contact_frame_id": candidate.get("contact_frame_id"),
                    "control_policy": "generic_visual_contact_frame_candidate_sweep",
                },
                strategy=strategy,
            )
            action = result.output.get("action") if isinstance(result.output, dict) else None
            action_output = action.get("output") if isinstance(action, dict) and isinstance(action.get("output"), dict) else {}
            step_summary = action_output.get("step_summary") if isinstance(action_output, dict) else {}
            final_summary = step_summary.get("final") if isinstance(step_summary, dict) else {}
            candidate_best_distance = action_output.get("best_distance")
            if candidate_best_distance is None and isinstance(final_summary, dict):
                candidate_best_distance = final_summary.get("best_distance_to_button")
            interaction_confirmed = bool(
                result.ok
                or action_output.get("interaction_confirmed")
                or action_output.get("button_contact_seen")
                or action_output.get("state_changed")
                or (isinstance(final_summary, dict) and final_summary.get("interaction_confirmed"))
            )
            attempt = {
                "attempt_index": attempt_index,
                "candidate_list_index": candidate_index,
                "candidate_index": candidate.get("candidate_index", candidate_index),
                "contact_frame_id": candidate.get("contact_frame_id"),
                "ok": bool(result.ok),
                "interaction_confirmed": interaction_confirmed,
                "best_distance": _to_builtin(candidate_best_distance),
                "result": result.to_dict(),
            }
            attempts.append(attempt)
            final_action = result.to_dict()
            if action_output:
                current_previous = action_output
            try:
                numeric_distance = None if candidate_best_distance is None else float(candidate_best_distance)
            except (TypeError, ValueError):
                numeric_distance = None
            if best_attempt is None or (
                numeric_distance is not None and (best_distance is None or numeric_distance < best_distance)
            ):
                best_attempt = attempt
                best_distance = numeric_distance
            if interaction_confirmed and confirmed_attempt is None:
                confirmed_attempt = attempt
            if stop_on_interaction and interaction_confirmed:
                stopped_reason = "interaction_confirmed"
                break

        selected_attempt = confirmed_attempt or best_attempt
        ok = confirmed_attempt is not None
        return PrimitiveResult(
            name="sweep_robocasa_button_contact_candidates",
            ok=ok,
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                "attempts": attempts,
                "best_attempt": selected_attempt,
                "best_distance_attempt": best_attempt,
                "final_action": final_action,
                "sweep_policy": {
                    "candidate_count": len(candidates),
                    "candidate_indices": ordered_indices,
                    "max_attempts": max_attempts,
                    "stop_on_interaction": bool(stop_on_interaction),
                    "stopped_reason": stopped_reason,
                    "control_policy": "agent_ordered_generic_visual_contact_frame_candidate_sweep",
                },
                "failure_diagnostics": {
                    "status": "interaction_confirmed" if ok else "no_candidate_confirmed_interaction",
                    "attempt_count": len(attempts),
                    "confirmed_attempt_available": confirmed_attempt is not None,
                    "best_distance": _to_builtin(best_distance),
                    "best_contact_frame_id": None if best_attempt is None else best_attempt.get("contact_frame_id"),
                    "next_native_debug_focus": (
                        []
                        if ok
                        else [
                            "button_contact_geom_pair",
                            "gripper_pad_surface_offset",
                            "arm_base_controller_reachability_near_best_candidate",
                        ]
                    ),
                    "verifier_or_reward_used": False,
                },
                "official_task_completion_claimed": False,
                "verifier_boundary": _verifier_boundary(),
            },
            error=None if ok else "no_candidate_confirmed_interaction",
        )

    def _primitive_manipulate_robocasa_object(
        self,
        object_name: str,
        evidence_handles: list[str] | None = None,
        target_name: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "manipulate_robocasa_object",
            object_name=object_name,
            evidence_handles=evidence_handles,
            target_name=target_name,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
            strategy=strategy,
        )

    def _primitive_grasp_robocasa_object(
        self,
        object_name: str,
        evidence_handles: list[str] | None = None,
        offset: list[float] | None = None,
        grasp_approach: str = "center",
        grasp_wrist_yaw_offset_rad: float = 0.0,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 80,
        grasp_contact_steps: int = 50,
        grasp_contact_offset: list[float] | None = None,
        grasp_contact_tolerance: float | None = None,
        force_grasp_contact_steps: bool = True,
        grasp_hold_steps: int = 12,
        gripper_command: float = 1.0,
        grasp_lift_delta: float = 0.06,
        contact_tolerance: float = 0.08,
        gain: float = 8.0,
        max_delta: float = 0.4,
        arm_delta_frame: str | None = None,
        mobile_base_enabled: bool = False,
        base_gain: float = 1.0,
        base_max_delta: float = 0.35,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_xy_deadband: float = 0.12,
        motion_trace_limit: int = 5,
    ) -> PrimitiveResult:
        try:
            grasp_wrist_yaw_offset_rad = float(grasp_wrist_yaw_offset_rad)
        except (TypeError, ValueError):
            grasp_wrist_yaw_offset_rad = float("nan")
        if grasp_approach not in {"center", "near_side"} or not np.isfinite(grasp_wrist_yaw_offset_rad) or abs(grasp_wrist_yaw_offset_rad) > .6:
            return PrimitiveResult(name="grasp_robocasa_object", ok=False, error="invalid_grasp_pose_candidate",
                                   output={"executed": False, "allowed_approaches": ["center", "near_side"], "max_abs_wrist_yaw_rad": .6})
        if mobile_base_enabled:
            horizon = max(int(horizon), 160)
            grasp_contact_steps = max(int(grasp_contact_steps), 16)
            base_gain = max(float(base_gain), 0.6)
            base_max_delta = max(float(base_max_delta), 0.25)
        return self._run_action_skill(
            "grasp_robocasa_object",
            object_name=object_name,
            evidence_handles=evidence_handles,
            target_name=object_name,
            offset=offset,
            grasp_approach=grasp_approach,
            grasp_wrist_yaw_offset_rad=grasp_wrist_yaw_offset_rad,
            horizon=horizon,
            grasp_contact_steps=grasp_contact_steps,
            grasp_contact_offset=grasp_contact_offset,
            grasp_contact_tolerance=grasp_contact_tolerance,
            force_grasp_contact_steps=force_grasp_contact_steps,
            grasp_hold_steps=grasp_hold_steps,
            gripper_command=gripper_command,
            grasp_lift_delta=grasp_lift_delta,
            contact_tolerance=contact_tolerance,
            gain=gain,
            max_delta=max_delta,
            arm_delta_frame=arm_delta_frame,
            mobile_base_enabled=mobile_base_enabled,
            base_gain=base_gain,
            base_max_delta=base_max_delta,
            base_command_sign=base_command_sign,
            base_mode_value=base_mode_value,
            base_xy_deadband=base_xy_deadband,
            motion_trace_limit=motion_trace_limit,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
            strategy=strategy,
        )

    def _primitive_place_robocasa_object_at(
        self,
        object_name: str,
        evidence_handles: list[str] | None = None,
        held_object_handle: str | None = None,
        target_name: str | None = None,
        target_position: list[float] | None = None,
        relation: str = "at",
        offset: list[float] | None = None,
        use_affordance_site: bool = True,
        affordance_site_name: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 30,
        gripper_command: float = 1.0,
        transport_gripper_command: float | None = None,
        release_gripper_command: float = 0.0,
        release_only_when_ready: bool = True,
        release_xy_tolerance: float | None = None,
        release_requires_contact: bool = False,
        release_settle_steps: int = 3,
        post_release_retreat_offset: list[float] | None = None,
        post_release_retreat_steps: int = 12,
        post_release_retreat_min_distance: float = 0.0,
        post_release_retreat_max_steps: int = 0,
        tolerance: float = 0.035,
        object_relative_control: bool = True,
        object_error_gain: float = 1.0,
        object_error_clip: float = 0.18,
        object_xy_push_steps: int = 0,
        object_xy_push_align_steps: int = 8,
        object_xy_push_backoff: float = 0.07,
        object_xy_push_through: float = 0.03,
        object_xy_push_z_offset: float = 0.02,
        object_xy_push_reacquire_from_side: bool = True,
        object_xy_contact_seek_steps: int = 0,
        object_xy_contact_seek_backoff: float | None = None,
        object_xy_contact_seek_z_offset: float | None = None,
        object_xy_early_stop_enabled: bool | None = None,
        object_xy_stop_when_within: float | None = None,
        object_xy_stop_requires_contact: bool = False,
        contact_guard_enabled: bool = True,
        contact_tolerance: float = 0.08,
        contact_guard_tolerance: float | None = None,
        contact_guard_recover_steps: int = 12,
        contact_guard_offset: list[float] | None = None,
        gain: float = 2.0,
        max_delta: float = 0.3,
        arm_delta_frame: str | None = None,
        mobile_base_enabled: bool = False,
        base_gain: float = 1.0,
        base_max_delta: float = 0.35,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_xy_deadband: float = 0.12,
        motion_trace_limit: int = 5,
    ) -> PrimitiveResult:
        context = dict(agent_context or {})
        resolved_target_name = target_name
        target_payload = None
        if target_name in self._fixtures:
            target_payload = self._fixtures[target_name]
        elif target_name in self._objects:
            target_payload = self._objects[target_name]
        resolved_relation = relation
        affordance_target = (
            _placement_affordance_target(
                resolved_relation,
                target_payload=target_payload,
                query=query,
                agent_context=context,
                site_name=affordance_site_name,
            )
            if use_affordance_site
            else None
        )
        if affordance_target is not None:
            target_position = affordance_target["position"]
        if resolved_target_name is not None and target_position is None:
            target_position = self._registry_position(resolved_target_name)
        if target_position is None and target_payload is None:
            return PrimitiveResult(
                name="place_robocasa_object_at",
                ok=False,
                output={
                    "object_name": object_name,
                    "target_name": target_name,
                    "resolved_target_name": resolved_target_name if target_payload is not None else None,
                    "relation": resolved_relation,
                    "requested_relation": relation,
                    "prompt": prompt,
                    "query": query,
                    "agent_context": context,
                    **self._action_evidence(
                        "place_robocasa_object_at",
                        {
                            "object_name": object_name,
                            "target_name": target_name,
                            "target_position": target_position,
                            "relation": resolved_relation,
                            "prompt": prompt,
                            "query": query,
                            "agent_context": context,
                            "strategy": strategy,
                        },
                        target_name=target_name,
                        target_position=target_position,
                    ),
                    "available_targets": sorted([*self._fixtures.keys(), *self._objects.keys()]),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                    "motion_backend": self.config.motion_backend,
                },
                error="placement_target_not_found",
            )
        resolved_offset = (
            offset
            if offset is not None
            else [0.0, 0.0, 0.0]
            if affordance_target is not None
            else _placement_offset(resolved_relation, target_payload=target_payload, query=query, agent_context=context)
        )
        resolved_release_xy_tolerance = (
            float(release_xy_tolerance)
            if release_xy_tolerance is not None
            else _placement_release_xy_tolerance(target_payload, affordance_target=affordance_target)
        )
        resolved_object_xy_stop_when_within = (
            float(object_xy_stop_when_within)
            if object_xy_stop_when_within is not None
            else resolved_release_xy_tolerance
        )
        return self._run_action_skill(
            "place_robocasa_object_at",
            object_name=object_name,
            evidence_handles=evidence_handles,
            held_object_handle=held_object_handle,
            target_name=resolved_target_name,
            target_position=target_position,
            relation=resolved_relation,
            requested_relation=relation,
            offset=resolved_offset,
            use_affordance_site=use_affordance_site,
            affordance_site_name=affordance_site_name,
            affordance_target=affordance_target,
            horizon=horizon,
            release_settle_steps=release_settle_steps,
            post_release_retreat_offset=post_release_retreat_offset,
            post_release_retreat_steps=post_release_retreat_steps,
            post_release_retreat_min_distance=post_release_retreat_min_distance,
            post_release_retreat_max_steps=post_release_retreat_max_steps,
            tolerance=tolerance,
            object_relative_control=object_relative_control,
            object_error_gain=object_error_gain,
            object_error_clip=object_error_clip,
            object_xy_push_steps=object_xy_push_steps,
            object_xy_push_align_steps=object_xy_push_align_steps,
            object_xy_push_backoff=object_xy_push_backoff,
            object_xy_push_through=object_xy_push_through,
            object_xy_push_z_offset=object_xy_push_z_offset,
            object_xy_push_reacquire_from_side=object_xy_push_reacquire_from_side,
            object_xy_contact_seek_steps=object_xy_contact_seek_steps,
            object_xy_contact_seek_backoff=object_xy_contact_seek_backoff,
            object_xy_contact_seek_z_offset=object_xy_contact_seek_z_offset,
            object_xy_early_stop_enabled=object_xy_early_stop_enabled,
            object_xy_stop_when_within=resolved_object_xy_stop_when_within,
            object_xy_stop_requires_contact=object_xy_stop_requires_contact,
            contact_guard_enabled=contact_guard_enabled,
            contact_tolerance=contact_tolerance,
            contact_guard_tolerance=contact_guard_tolerance,
            contact_guard_recover_steps=contact_guard_recover_steps,
            contact_guard_offset=contact_guard_offset,
            gain=gain,
            max_delta=max_delta,
            arm_delta_frame=arm_delta_frame,
            mobile_base_enabled=mobile_base_enabled,
            base_gain=base_gain,
            base_max_delta=base_max_delta,
            base_command_sign=base_command_sign,
            base_mode_value=base_mode_value,
            base_xy_deadband=base_xy_deadband,
            motion_trace_limit=motion_trace_limit,
            gripper_command=gripper_command,
            transport_gripper_command=transport_gripper_command,
            release_gripper_command=release_gripper_command,
            release_only_when_ready=release_only_when_ready,
            release_xy_tolerance=resolved_release_xy_tolerance,
            release_requires_contact=release_requires_contact,
            prompt=prompt,
            query=query,
            agent_context=context,
            strategy=strategy,
        )

    def _primitive_inspect_robocasa_transport_state(
        self,
        motion_result: Any | None = None,
        runtime_report: JsonDict | None = None,
        placement_refinement_hint: Any | None = None,
        object_name: str | None = None,
        previous_state: Any | None = None,
        contact_tolerance: float = 0.08,
        gripper_closed_threshold: float = 0.035,
        coupling_tolerance: float = 0.02,
        minimum_coupling_motion: float = 0.005,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        context = dict(agent_context or {})
        state = _robocasa_transport_state_from_public_inputs(
            motion_result=motion_result,
            runtime_report=runtime_report,
            placement_refinement_hint=placement_refinement_hint,
        )
        live_observation = _robocasa_live_transport_observation(
            self._env,
            _read_current_observation(self._env) or self._last_obs,
            object_name=object_name,
            previous_state=previous_state,
            contact_tolerance=contact_tolerance,
            gripper_closed_threshold=gripper_closed_threshold,
            coupling_tolerance=coupling_tolerance,
            minimum_coupling_motion=minimum_coupling_motion,
        )
        state = _merge_robocasa_live_transport_state(state, live_observation)
        return PrimitiveResult(
            name="inspect_robocasa_transport_state",
            ok=True,
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                "prompt": prompt,
                "query": query,
                "agent_context": context,
                **state,
                "live_observation": live_observation,
                "private_task_signal_exposed": False,
                "official_task_completion_claimed": False,
                "verifier_boundary": {
                    "call_after_agent_visible_sequence": "verify(scope='task')",
                    "private_task_signal_exposed": False,
                    "primitive_claims_task_completion": False,
                },
            },
        )

    def _primitive_plan_robocasa_place_recovery(
        self,
        placement_refinement_hint: Any,
        strategy: str | None = None,
        attempts: int = 3,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        hint = _extract_robocasa_refinement_hint(placement_refinement_hint)
        if not hint.get("available"):
            return PrimitiveResult(
                name="plan_robocasa_place_recovery",
                ok=False,
                output={
                    "available": False,
                    "reason": hint.get("reason") or "placement_refinement_hint_unavailable",
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                    "private_success_signal_exposed": False,
                    "official_task_completion_claimed": False,
                },
                error="placement_refinement_hint_unavailable",
            )
        plan = _robocasa_place_recovery_plan_from_hint(
            hint,
            strategy=strategy,
            attempts=attempts,
            prompt=prompt,
            query=query,
            agent_context=agent_context,
        )
        return PrimitiveResult(name="plan_robocasa_place_recovery", ok=True, output=plan)

    def _primitive_refine_robocasa_place_until_ready(
        self,
        object_name: str,
        target_name: str | None = None,
        target_position: list[float] | None = None,
        relation: str = "at",
        offset: list[float] | None = None,
        use_affordance_site: bool = True,
        affordance_site_name: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        attempts: int = 3,
        horizon: int = 30,
        gripper_command: float = 1.0,
        transport_gripper_command: float | None = None,
        release_gripper_command: float = 0.0,
        release_xy_tolerance: float | None = None,
        release_requires_contact: bool = False,
        release_settle_steps: int = 3,
        post_release_retreat_offset: list[float] | None = None,
        post_release_retreat_steps: int = 12,
        post_release_retreat_min_distance: float = 0.0,
        post_release_retreat_max_steps: int = 0,
        tolerance: float = 0.035,
        object_relative_control: bool = True,
        object_error_gain: float = 1.0,
        object_error_clip: float = 0.18,
        object_xy_push_steps: int = 12,
        object_xy_push_align_steps: int = 8,
        object_xy_push_backoff: float = 0.07,
        object_xy_push_through: float = 0.03,
        object_xy_push_z_offset: float = 0.02,
        object_xy_push_reacquire_from_side: bool = True,
        object_xy_contact_seek_steps: int = 0,
        object_xy_contact_seek_backoff: float | None = None,
        object_xy_contact_seek_z_offset: float | None = None,
        object_xy_stop_when_within: float | None = None,
        object_xy_stop_requires_contact: bool = False,
        contact_guard_enabled: bool = True,
        contact_tolerance: float = 0.08,
        contact_guard_tolerance: float | None = None,
        contact_guard_recover_steps: int = 12,
        contact_guard_offset: list[float] | None = None,
        gain: float = 2.0,
        max_delta: float = 0.3,
        arm_delta_frame: str | None = None,
        mobile_base_enabled: bool = False,
        base_gain: float = 1.0,
        base_max_delta: float = 0.35,
        base_command_sign: float = 1.0,
        base_mode_value: float = 1.0,
        base_xy_deadband: float = 0.12,
        motion_trace_limit: int = 5,
    ) -> PrimitiveResult:
        context = dict(agent_context or {})
        max_attempts = max(1, int(attempts))
        attempt_results: list[JsonDict] = []
        params: JsonDict = {
            "object_name": object_name,
            "target_name": target_name,
            "target_position": target_position,
            "relation": relation,
            "offset": offset,
            "use_affordance_site": use_affordance_site,
            "affordance_site_name": affordance_site_name,
            "prompt": prompt,
            "query": query,
            "strategy": strategy,
            "horizon": horizon,
            "gripper_command": gripper_command,
            "transport_gripper_command": transport_gripper_command,
            "release_gripper_command": release_gripper_command,
            "release_only_when_ready": True,
            "release_xy_tolerance": release_xy_tolerance,
            "release_requires_contact": release_requires_contact,
            "release_settle_steps": release_settle_steps,
            "post_release_retreat_offset": post_release_retreat_offset,
            "post_release_retreat_steps": post_release_retreat_steps,
            "post_release_retreat_min_distance": post_release_retreat_min_distance,
            "post_release_retreat_max_steps": post_release_retreat_max_steps,
            "tolerance": tolerance,
            "object_relative_control": object_relative_control,
            "object_error_gain": object_error_gain,
            "object_error_clip": object_error_clip,
            "object_xy_push_steps": object_xy_push_steps,
            "object_xy_push_align_steps": object_xy_push_align_steps,
            "object_xy_push_backoff": object_xy_push_backoff,
            "object_xy_push_through": object_xy_push_through,
            "object_xy_push_z_offset": object_xy_push_z_offset,
            "object_xy_push_reacquire_from_side": object_xy_push_reacquire_from_side,
            "object_xy_contact_seek_steps": object_xy_contact_seek_steps,
            "object_xy_contact_seek_backoff": object_xy_contact_seek_backoff,
            "object_xy_contact_seek_z_offset": object_xy_contact_seek_z_offset,
            "object_xy_early_stop_enabled": True,
            "object_xy_stop_when_within": object_xy_stop_when_within,
            "object_xy_stop_requires_contact": object_xy_stop_requires_contact,
            "contact_guard_enabled": contact_guard_enabled,
            "contact_tolerance": contact_tolerance,
            "contact_guard_tolerance": contact_guard_tolerance,
            "contact_guard_recover_steps": contact_guard_recover_steps,
            "contact_guard_offset": contact_guard_offset,
            "gain": gain,
            "max_delta": max_delta,
            "arm_delta_frame": arm_delta_frame,
            "mobile_base_enabled": mobile_base_enabled,
            "base_gain": base_gain,
            "base_max_delta": base_max_delta,
            "base_command_sign": base_command_sign,
            "base_mode_value": base_mode_value,
            "base_xy_deadband": base_xy_deadband,
            "motion_trace_limit": motion_trace_limit,
        }
        last_result: PrimitiveResult | None = None
        stop_reason = "attempt_budget_exhausted"
        allowed_recommendation_keys = {
            "object_error_gain",
            "object_error_clip",
            "object_xy_push_align_steps",
            "object_xy_push_steps",
            "object_xy_push_backoff",
            "object_xy_push_through",
            "object_xy_push_z_offset",
            "object_xy_contact_seek_steps",
            "object_xy_contact_seek_backoff",
            "object_xy_contact_seek_z_offset",
            "release_xy_tolerance",
            "release_requires_contact",
            "contact_guard_enabled",
            "contact_guard_tolerance",
            "contact_guard_recover_steps",
            "object_xy_early_stop_enabled",
            "object_xy_stop_when_within",
        }
        for attempt_index in range(max_attempts):
            attempt_context = {
                **context,
                "meta_primitive": "refine_robocasa_place_until_ready",
                "refinement_attempt": attempt_index + 1,
                "refinement_attempts": max_attempts,
            }
            result = self._primitive_place_robocasa_object_at(**{**params, "agent_context": attempt_context})
            last_result = result
            result_dict = result.to_dict()
            attempt_results.append(result_dict)
            output = result.output
            hint = output.get("placement_refinement_hint") if isinstance(output, dict) else None
            if output.get("release_executed") is True and output.get("release_stable") is True:
                stop_reason = "release_stable"
                break
            if not isinstance(hint, dict) or not hint.get("available"):
                stop_reason = "refinement_hint_unavailable"
                break
            if hint.get("release_ready") is True and hint.get("release_xy_ready") is True:
                stop_reason = "release_ready_but_not_executed"
                break
            recommended = hint.get("recommended_next_call")
            if isinstance(recommended, dict):
                for key in allowed_recommendation_keys:
                    if key in recommended:
                        params[key] = recommended[key]
            params["object_error_gain"] = min(float(params.get("object_error_gain", 1.0)) * 1.05, 4.5)

        final_output = dict(last_result.output) if last_result is not None else {}
        final_hint = final_output.get("placement_refinement_hint")
        final_release_executed = bool(final_output.get("release_executed"))
        final_release_ready = bool(final_output.get("release_ready"))
        final_ok = bool(last_result.ok) if last_result is not None else False
        error = None if final_ok else (last_result.error if last_result is not None and last_result.error else "release_not_ready_after_refinement")
        return PrimitiveResult(
            name="refine_robocasa_place_until_ready",
            ok=final_ok,
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=context),
                "prompt": prompt,
                "query": query,
                "agent_context": context,
                "attempts_requested": max_attempts,
                "attempts_executed": len(attempt_results),
                "attempt_results": attempt_results,
                "final_result": final_output,
                "final_refinement_hint": final_hint,
                "final_release_ready": final_release_ready,
                "final_release_executed": final_release_executed,
                "stop_reason": stop_reason,
                "next_call_parameters": params,
                "private_task_signal_exposed": False,
                "official_task_completion_claimed": False,
            },
            error=error,
        )

    def _primitive_move_robocasa_ee_to(
        self,
        target_name: str | None = None,
        target_position: list[float] | None = None,
        evidence_handles: list[str] | None = None,
        offset: list[float] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        strategy: str = "motion_backend",
        horizon: int = 20,
        gain: float = 2.0,
        max_delta: float = 0.3,
        arm_delta_frame: str | None = None,
        tolerance: float = 0.06,
        gripper_command: float = 1.0,
        avoidance_point: list[float] | None = None,
        min_distance_from_avoidance: float | None = None,
        avoidance_tolerance: float = 0.0,
        stop_when_avoidance_reached: bool = False,
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "move_robocasa_ee_to",
            target_name=target_name,
            target_position=target_position,
            evidence_handles=evidence_handles,
            offset=offset,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
            strategy=strategy,
            horizon=horizon,
            gain=gain,
            max_delta=max_delta,
            arm_delta_frame=arm_delta_frame,
            tolerance=tolerance,
            gripper_command=gripper_command,
            avoidance_point=avoidance_point,
            min_distance_from_avoidance=min_distance_from_avoidance,
            avoidance_tolerance=avoidance_tolerance,
            stop_when_avoidance_reached=stop_when_avoidance_reached,
        )

    def _primitive_apply_robocasa_control(
        self,
        eef_position_delta: list[float] | None = None,
        eef_rotation_delta: list[float] | None = None,
        base_delta: list[float] | None = None,
        gripper_command: float = 0.0,
        repeat: int = 1,
        arm_delta_frame: str | None = None,
        base_mode_value: float = 1.0,
        object_name: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return self._run_action_skill(
            "apply_robocasa_control",
            eef_position_delta=eef_position_delta,
            eef_rotation_delta=eef_rotation_delta,
            base_delta=base_delta,
            gripper_command=gripper_command,
            repeat=repeat,
            arm_delta_frame=arm_delta_frame,
            base_mode_value=base_mode_value,
            object_name=object_name,
            prompt=prompt,
            query=query,
            agent_context=agent_context or {},
        )

    def _primitive_record_robocasa_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"robocasa:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value)})
        return PrimitiveResult(name="record_robocasa_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _inspect_entity(
        self,
        primitive_name: str,
        registry: dict[str, JsonDict],
        explicit_name: str | None,
        prompt: str | None,
        query: str | None,
        agent_context: JsonDict,
    ) -> PrimitiveResult:
        grounding_context = dict(agent_context)
        selected_name = _select_name(registry, explicit_name)
        candidates = [
            {**deepcopy(payload), "name": name, "score": 1.0 if name == selected_name else 0.5}
            for name, payload in sorted(registry.items())
        ]
        if selected_name is None:
            return PrimitiveResult(
                name=primitive_name,
                ok=False,
                output={
                    "language_task": self._language_task(prompt=prompt, query=query, agent_context=grounding_context),
                    "prompt": prompt,
                    "query": query,
                    "agent_context": grounding_context,
                    "candidates": candidates,
                    "pose_evidence": self._pose_evidence(),
                    "observation_ref": self._observation_ref(source="robocasa_scene_registry"),
                },
                error="entity_not_found",
            )
        selected = {**deepcopy(registry[selected_name]), "name": selected_name, "score": 1.0}
        selected["evidence"] = {
            "source": "robocasa_scene_state",
            "observation_ref": self._observation_ref(source="robocasa_scene_registry"),
            "prompt": prompt,
            "query": query,
            "agent_context": grounding_context,
        }
        return PrimitiveResult(
            name=primitive_name,
            ok=True,
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=grounding_context),
                "prompt": prompt,
                "query": query,
                "agent_context": grounding_context,
                "selected": selected,
                "candidates": candidates,
                "pose_evidence": self._pose_evidence(selected_name),
                "visual_evidence": self._visual_evidence(),
                "observation_ref": self._observation_ref(source="robocasa_scene_registry"),
            },
        )

    def _run_action_skill(self, skill_name: str, **kwargs: Any) -> PrimitiveResult:
        evidence_error = self._validate_visual_grounding_for_action(skill_name, kwargs)
        if evidence_error is not None:
            return evidence_error
        if self._skill_executor is not None:
            result = self._skill_executor(self._env, skill_name, kwargs)
            latest_obs = _read_current_observation(self._env)
            if latest_obs is not None:
                self._last_obs = latest_obs
                self._observation_serial += 1
            self._refresh_scene_registry()
            return result
        if self.config.motion_backend == "robocasa_state_delta_motion":
            result = self._run_state_delta_motion_skill(skill_name, **kwargs)
            if result.output.get("execution_status") in {"stepped", "executed"}:
                self._observation_serial += 1
            self._refresh_scene_registry()
            return result
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                **self._action_evidence(skill_name, kwargs),
                "action_skill_schema": self._action_skill_schema(skill_name),
                "execution_status": "not_executed",
                "requires_motion_backend": True,
            },
            error="motion_backend_missing",
        )

    def _validate_visual_grounding_for_action(
        self,
        skill_name: str,
        kwargs: JsonDict,
    ) -> PrimitiveResult | None:
        if skill_name == "apply_robocasa_control":
            return None
        supplied = kwargs.get("evidence_handles")
        handles = ([str(supplied)] if isinstance(supplied, str) else
                   [str(item) for item in supplied] if isinstance(supplied, (list, tuple)) else [])
        entity_to_ground = kwargs.get("target_name") if skill_name == "place_robocasa_object_at" else (
            kwargs.get("object_name") or kwargs.get("fixture_name") or kwargs.get("target_name")
        )
        repair_call = {"name": "ground_robocasa_visual_target", "parameters": {"entity_name": entity_to_ground}}
        if not handles:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    "execution_status": "not_executed",
                    "next_call": repair_call,
                },
                error="visual_grounding_evidence_required",
            )
        unknown = [handle for handle in handles if handle not in self._visual_grounding_handles]
        if unknown:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={"execution_status": "not_executed", "unknown_evidence_handles": unknown, "next_call": repair_call},
                error="visual_grounding_evidence_invalid",
            )
        stale = [
            handle
            for handle in handles
            if int(self._visual_grounding_handles[handle].get("observation_serial", -1)) != self._observation_serial
        ]
        if stale:
            static_reuse = _robocasa_static_fixture_evidence_reuse_for_action(
                skill_name,
                kwargs,
                handles=handles,
                stale_handles=stale,
                grounding_handles=self._visual_grounding_handles,
                observation_serial=self._observation_serial,
            )
            if not static_reuse.get("allowed"):
                pose_reuse = self._reuse_unchanged_pose_visual_evidence(
                    skill_name,
                    kwargs,
                    handles=handles,
                    stale_handles=stale,
                )
                if not pose_reuse.get("allowed"):
                    return PrimitiveResult(
                        name=skill_name,
                        ok=False,
                        output={"execution_status": "not_executed", "stale_evidence_handles": stale, "next_call": repair_call},
                        error="visual_grounding_evidence_stale",
                    )
                kwargs["visual_evidence_freshness"] = pose_reuse
            else:
                kwargs["visual_evidence_freshness"] = static_reuse
            for handle in stale:
                cached = self._visual_grounding_handles[handle]
                cached["previous_observation_serial"] = cached.get("observation_serial")
                cached["observation_serial"] = self._observation_serial
                cached["auto_refreshed"] = True
                cached["refresh_source"] = kwargs["visual_evidence_freshness"].get("reason")
            self.record_event("visual_evidence_auto_refresh", {
                "skill_name": skill_name, "handles": stale,
                "current_observation_serial": self._observation_serial,
                "refresh_source": kwargs["visual_evidence_freshness"].get("reason"),
            })
        expected_names = {
            str(value)
            for key in ("object_name", "fixture_name", "target_name")
            if (value := kwargs.get(key)) is not None
        }
        if (skill_name == "place_robocasa_object_at"
                and kwargs.get("held_object_handle") == self._held_object_handle
                and kwargs.get("object_name") == self._held_object_name):
            expected_names.discard(str(kwargs["object_name"]))
        bound_names = {
            str(payload["entity_name"])
            for handle in handles
            if (payload := self._visual_grounding_handles[handle]).get("entity_name") is not None
        }
        missing_bindings = sorted(expected_names.difference(bound_names))
        if missing_bindings:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    "execution_status": "not_executed",
                    "evidence_bound_entities": sorted(bound_names),
                    "missing_entity_bindings": missing_bindings,
                },
                error="visual_grounding_entity_mismatch",
            )
        evidence = [deepcopy(self._visual_grounding_handles[handle]) for handle in handles]
        target_error = _bind_robocasa_visual_action_target(skill_name, kwargs, evidence)
        if target_error is not None:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={**kwargs, "execution_status": "not_executed", "fabricated_evidence": False},
                error=target_error,
            )
        kwargs["visual_grounding_evidence"] = evidence
        return None

    def _reuse_unchanged_pose_visual_evidence(
        self,
        skill_name: str,
        kwargs: JsonDict,
        *,
        handles: list[str],
        stale_handles: list[str],
    ) -> JsonDict:
        """Reuse a handle after robot-only motion when the public target pose did not move."""

        if skill_name not in _UNCHANGED_POSE_REUSE_SKILLS:
            return {"allowed": False, "reason": "skill_requires_fresh_visual_evidence"}
        target_name = next(
            (
                str(kwargs[key])
                for key in ("object_name", "fixture_name", "target_name")
                if kwargs.get(key) is not None
            ),
            None,
        )
        if not target_name:
            return {"allowed": False, "reason": "target_name_required"}
        current_pose = self._registry_position(target_name)
        for handle in stale_handles:
            payload = self._visual_grounding_handles.get(handle) or {}
            if payload.get("entity_name") != target_name:
                return {"allowed": False, "reason": "stale_entity_mismatch"}
            recorded_registry = _coerce_vector_value(payload.get("registry_pose_world"))
            recorded_points = []
            for candidate in (
                payload.get("point_world"),
                (payload.get("native_geometry") or {}).get("pose_world") if isinstance(payload.get("native_geometry"), dict) else None,
            ):
                vector = _coerce_vector_value(candidate)
                if vector is not None:
                    recorded_points.append(vector)
            if recorded_registry is None and not recorded_points:
                return {"allowed": False, "reason": "stale_pose_unavailable"}
            if current_pose is None:
                refresh_point = recorded_registry or recorded_points[0]
            elif recorded_registry is not None:
                delta = float(np.linalg.norm(np.asarray(current_pose[:3], dtype=float) - np.asarray(recorded_registry[:3], dtype=float)))
                if delta > _UNCHANGED_POSE_REUSE_TOLERANCE_M:
                    return {"allowed": False, "reason": "public_pose_moved", "pose_delta_m": delta}
                refresh_point = current_pose
            else:
                refresh_point = current_pose
            payload["point_world"] = [float(value) for value in refresh_point[:3]]
            payload["registry_pose_world"] = [float(value) for value in refresh_point[:3]]
        return {
            "allowed": True,
            "reason": "unchanged_public_pose_reused_after_robot_only_motion",
            "current_observation_serial": int(self._observation_serial),
            "stale_evidence_handles": list(stale_handles),
            "current_evidence_handles": sorted(set(handles).difference(stale_handles)),
            "target_name": target_name,
        }

    def _run_state_delta_motion_skill(self, skill_name: str, **kwargs: Any) -> PrimitiveResult:
        if skill_name not in {
            "move_robocasa_ee_to",
            "press_robocasa_fixture_button",
            "manipulate_robocasa_object",
            "grasp_robocasa_object",
            "place_robocasa_object_at",
            "apply_robocasa_control",
        }:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                },
                error="state_delta_skill_unsupported",
            )
        if self._env is None:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                },
                error="live_env_not_created",
            )
        step_action, action, low, high, action_key = _zero_action(self._env)
        obs = _read_current_observation(self._env)
        if step_action is None or action is None:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                },
                error="action_space_unavailable",
            )
        if not isinstance(obs, dict):
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                },
                error="observation_unavailable",
            )

        runtime_action_schema = self._runtime_action_schema()
        if skill_name == "apply_robocasa_control":
            try:
                import numpy as np
            except Exception as exc:  # pragma: no cover - numpy is a project test dependency.
                return PrimitiveResult(
                    name=skill_name,
                    ok=False,
                    output={**kwargs, "execution_status": "not_executed"},
                    error=f"numpy_unavailable: {exc}",
                )
            position_delta = _coerce_vector_value(kwargs.get("eef_position_delta"))
            rotation_delta = _coerce_vector_value(kwargs.get("eef_rotation_delta"))
            base_delta = _coerce_vector_value(kwargs.get("base_delta"))
            position_delta = position_delta if position_delta is not None else np.zeros(3, dtype=np.float32)
            rotation_delta = rotation_delta if rotation_delta is not None else np.zeros(3, dtype=np.float32)
            base_delta = base_delta if base_delta is not None else np.zeros(3, dtype=np.float32)
            repeat = max(1, int(kwargs.get("repeat", 1)))
            gripper_command = float(kwargs.get("gripper_command", 0.0))
            arm_delta_frame = str(
                kwargs.get("arm_delta_frame")
                or runtime_action_schema.get("controller_input_ref_frame")
                or "world"
            )
            base_mode_value = float(kwargs.get("base_mode_value", 1.0))
            object_name = kwargs.get("object_name")
            records: list[JsonDict] = []
            latest_reward: Any = None
            latest_info: JsonDict = {}
            terminated = False
            truncated = False
            for step_index in range(repeat):
                template, action, low, high, action_key = _zero_action(self._env)
                if template is None or action is None:
                    break
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                transformed_position_delta = _delta_for_robocasa_action_frame(position_delta, obs, arm_delta_frame)
                action[: min(3, action.shape[0])] = transformed_position_delta[: min(3, action.shape[0])]
                if action.shape[0] > 3:
                    rotation_width = min(3, action.shape[0] - 3)
                    action[3 : 3 + rotation_width] = rotation_delta[:rotation_width]
                _set_flat_gripper_command_if_present(action, gripper_command, action_key=action_key)
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                formatted_action = _format_step_action(
                    template,
                    bounded_action,
                    action_key,
                    gripper_command,
                    base_action=base_delta,
                    base_mode_value=base_mode_value,
                )
                step_result = self._env.step(formatted_action)
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                records.append(
                    {
                        "step_index": step_index,
                        "eef_position_delta": _to_builtin(_vector_to_list(position_delta)),
                        "eef_rotation_delta": _to_builtin(_vector_to_list(rotation_delta)),
                        "base_delta": _to_builtin(_vector_to_list(base_delta)),
                        "gripper_command": gripper_command,
                        "arm_delta_frame": arm_delta_frame,
                        "action_summary": summarize_observation(formatted_action),
                        "observation_summary": summarize_observation(step_obs),
                        "reward": _to_builtin(latest_reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )
                if terminated or truncated:
                    break
            self._refresh_scene_registry()
            live_observation = _robocasa_live_transport_observation(
                self._env,
                obs,
                object_name=object_name if isinstance(object_name, str) else None,
                previous_state=None,
                contact_tolerance=0.08,
                gripper_closed_threshold=0.035,
                coupling_tolerance=0.02,
                minimum_coupling_motion=0.005,
            )
            return PrimitiveResult(
                name=skill_name,
                ok=bool(records),
                output={
                    **kwargs,
                    "execution_status": "stepped" if records else "not_executed",
                    "runtime_action_schema": self._runtime_action_schema(),
                    "steps": records,
                    "live_observation": live_observation,
                    "latest_info_keys": [key for key in sorted(str(key) for key in latest_info) if "success" not in key.lower()],
                    "official_task_completion_claimed": False,
                },
                error=None if records else "action_space_unavailable",
            )

        target_name = kwargs.get("target_name") or kwargs.get("object_name") or "obj"
        if skill_name == "press_robocasa_fixture_button":
            target_name = kwargs.get("fixture_name") or target_name
        explicit_target_position = kwargs.get("target_position")
        if explicit_target_position is None and skill_name == "press_robocasa_fixture_button":
            explicit_target_position = kwargs.get("button_position")
        visual_target_position = _visual_target_position(kwargs.get("visual_grounding_evidence"), target_name)
        if explicit_target_position is None and visual_target_position is not None:
            explicit_target_position = visual_target_position
        if explicit_target_position is None and kwargs.get("evidence_handles"):
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs, target_name=target_name),
                    "execution_status": "not_executed",
                    "fabricated_evidence": False,
                },
                error="visual_grounding_world_point_unavailable",
            )
        if explicit_target_position is None and isinstance(target_name, str):
            explicit_target_position = self._registry_position(target_name)
        target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
        eef_position = _vector(obs, "robot0_eef_pos")
        selected_button_name: str | None = None
        button_position = _coerce_vector_value(kwargs.get("button_position"))
        if skill_name == "press_robocasa_fixture_button":
            fixture_name = str(kwargs.get("fixture_name") or target_name)
            if button_position is None and visual_target_position is not None:
                button_position = _coerce_vector_value(visual_target_position)
            if button_position is None:
                button_selection = _select_fixture_button_affordance(
                    self._fixtures.get(fixture_name),
                    button_name=kwargs.get("button_name"),
                )
                if button_selection is not None:
                    selected_button_name, button_position = button_selection
            else:
                selected_button_name = str(kwargs.get("button_name")) if kwargs.get("button_name") is not None else None
            if button_position is None:
                return PrimitiveResult(
                    name=skill_name,
                    ok=False,
                    output={
                        **kwargs,
                        **self._action_evidence(skill_name, kwargs, target_name=target_name),
                        "execution_status": "not_executed",
                        "requires_motion_backend": True,
                        "available_buttons": _fixture_button_names(self._fixtures.get(fixture_name)),
                    },
                    error="fixture_button_affordance_unavailable",
                )
            if target_position is None:
                target_position = button_position
        if target_position is None or eef_position is None:
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs, target_name=target_name),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                    "motion_backend": "robocasa_state_delta_motion",
                    "available_state_keys": sorted(str(key) for key in obs),
                },
                error="state_grounding_unavailable",
            )

        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - numpy is a project test dependency.
            return PrimitiveResult(
                name=skill_name,
                ok=False,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs, target_name=target_name),
                    "execution_status": "not_executed",
                    "requires_motion_backend": True,
                },
                error=f"numpy_unavailable: {exc}",
            )

        if skill_name == "press_robocasa_fixture_button":
            if button_position is None:
                return PrimitiveResult(
                    name=skill_name,
                    ok=False,
                    output={
                        **kwargs,
                        **self._action_evidence(skill_name, kwargs, target_name=target_name),
                        "execution_status": "not_executed",
                        "requires_motion_backend": True,
                    },
                    error="fixture_button_affordance_unavailable",
                )
            fixture_name = str(kwargs.get("fixture_name") or target_name)
            fixture_payload = self._fixtures.get(fixture_name, {})
            fixture_position = _coerce_vector_value(fixture_payload.get("pos") if isinstance(fixture_payload, dict) else None)
            if fixture_position is None:
                fixture_position = target_position
            button_position = np.asarray(button_position, dtype=np.float32)
            fixture_position = np.asarray(fixture_position, dtype=np.float32)
            eef_position = np.asarray(eef_position, dtype=np.float32)
            outward = button_position - fixture_position
            outward[2] = 0.0
            if float(np.linalg.norm(outward)) < 1e-5:
                outward = button_position - eef_position
                outward[2] = 0.0
            if float(np.linalg.norm(outward)) < 1e-5:
                outward = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            outward = outward / float(np.linalg.norm(outward))
            raw_button_position = button_position.copy()
            button_anchor_adjustment = _coerce_vector_value(kwargs.get("button_anchor_adjustment"))
            button_anchor_adjustment_source = None
            if button_anchor_adjustment is not None:
                button_position = button_position + np.asarray(button_anchor_adjustment, dtype=np.float32)
                button_anchor_adjustment_source = str(
                    kwargs.get("button_anchor_adjustment_source") or "public_button_anchor_micro_sweep"
                )
            anchor_adjusted_button_position = button_position.copy()
            button_offset = _coerce_vector_value(kwargs.get("button_offset"))
            button_offset_source = "caller.button_offset" if button_offset is not None else None
            horizon = max(1, int(kwargs.get("horizon", 80)))
            max_delta = float(kwargs.get("max_delta", 0.3))
            gain = float(kwargs.get("gain", 2.0))
            arm_delta_frame = str(
                kwargs.get("arm_delta_frame")
                or runtime_action_schema.get("controller_input_ref_frame")
                or "world"
            )
            tolerance = float(kwargs.get("tolerance", 0.025))
            approach_distance = max(0.0, float(kwargs.get("approach_distance", 0.08)))
            raw_approach_stop_distance = kwargs.get("approach_stop_distance")
            approach_stop_distance = (
                None
                if raw_approach_stop_distance is None
                else max(0.0, float(raw_approach_stop_distance))
            )
            if approach_stop_distance is None:
                approach_stop_distance = max(tolerance, 0.035)
            approach_patience_steps = max(0, int(kwargs.get("approach_patience_steps", 12)))
            press_depth = max(0.0, float(kwargs.get("press_depth", 0.035)))
            raw_max_press_depth = kwargs.get("max_press_depth")
            max_press_depth = (
                press_depth
                if raw_max_press_depth is None
                else max(press_depth, float(raw_max_press_depth))
            )
            press_contact_seek_steps = max(0, int(kwargs.get("press_contact_seek_steps", 0)))
            raw_press_contact_seek_depth = kwargs.get("press_contact_seek_depth")
            press_contact_seek_depth = (
                max_press_depth
                if raw_press_contact_seek_depth is None
                else max(max_press_depth, float(raw_press_contact_seek_depth))
            )
            raw_contact_seek_stop_distance = kwargs.get("contact_seek_stop_distance")
            contact_seek_stop_distance = (
                max(tolerance, approach_stop_distance or tolerance)
                if raw_contact_seek_stop_distance is None
                else max(0.0, float(raw_contact_seek_stop_distance))
            )
            contact_seek_patience_steps = max(
                0,
                int(kwargs.get("contact_seek_patience_steps", max(6, approach_patience_steps))),
            )
            press_direction_sign = float(kwargs.get("press_direction_sign", -1.0))
            retreat_distance = max(0.0, float(kwargs.get("retreat_distance", 0.22)))
            press_steps = max(1, int(kwargs.get("press_steps", 16)))
            hold_steps = max(0, int(kwargs.get("hold_steps", 4)))
            mobile_base_enabled = bool(kwargs.get("mobile_base_enabled", False))
            raw_mobile_base_active_phases = kwargs.get("mobile_base_active_phases")
            mobile_base_active_phases = (
                None
                if raw_mobile_base_active_phases is None
                else {str(phase) for phase in raw_mobile_base_active_phases}
            )
            base_gain = float(kwargs.get("base_gain", 1.0))
            base_max_delta = float(kwargs.get("base_max_delta", 0.35))
            base_command_sign = float(kwargs.get("base_command_sign", 1.0))
            base_mode_value = float(kwargs.get("base_mode_value", 1.0))
            base_delta_frame = str(kwargs.get("base_delta_frame") or "world")
            raw_base_xy_deadband = kwargs.get("base_xy_deadband")
            base_xy_deadband = (
                0.0
                if raw_base_xy_deadband is None
                else max(0.0, float(raw_base_xy_deadband))
            )
            configured_retreat_steps = kwargs.get("retreat_steps")
            configured_approach_steps = kwargs.get("approach_steps")
            retreat_steps = (
                max(0, int(configured_retreat_steps))
                if configured_retreat_steps is not None
                else int(max(8, horizon // 4))
            )
            approach_steps = max(
                1,
                int(configured_approach_steps)
                if configured_approach_steps is not None
                else int(max(1, horizon - press_steps - hold_steps - retreat_steps)),
            )
            raw_press_direction_vector = _coerce_vector_value(kwargs.get("press_direction_vector"))
            if raw_press_direction_vector is not None:
                press_direction = np.asarray(raw_press_direction_vector, dtype=np.float32)
                if float(np.linalg.norm(press_direction)) < 1e-5:
                    press_direction = outward * press_direction_sign
                else:
                    press_direction = press_direction / float(np.linalg.norm(press_direction))
            else:
                press_direction = outward * press_direction_sign
            if button_offset is None and bool(kwargs.get("bind_gripper_contact_geometry", False)):
                raw_contact_axis = _coerce_vector_value(kwargs.get("gripper_contact_surface_axis"))
                contact_axis = raw_contact_axis if raw_contact_axis is not None else press_direction
                inferred_button_offset = _infer_eef_target_offset_for_gripper_contact(
                    self._env,
                    surface_axis=contact_axis,
                )
                if inferred_button_offset is not None:
                    button_offset = np.asarray(inferred_button_offset, dtype=np.float32)
                    button_offset_source = (
                        "gripper_contact_surface_axis"
                        if raw_contact_axis is not None
                        else "resolved_press_direction"
                    )
            if button_offset is not None:
                button_position = button_position + np.asarray(button_offset, dtype=np.float32)
            gripper_command = float(kwargs.get("gripper_command", 1.0))
            fixture_state_before = _env_fixture_state(self._env, fixture_name)
            distance_before = float(np.linalg.norm(eef_position - button_position))
            best_distance = distance_before
            final_distance = distance_before
            latest_reward = None
            latest_info: JsonDict = {}
            terminated = False
            truncated = False
            step_records: list[JsonDict] = []
            stage_targets = [
                ("button_approach", button_position + outward * approach_distance, approach_steps, True),
                ("button_press", button_position + press_direction * max_press_depth, press_steps, False),
                ("button_hold", button_position + press_direction * max_press_depth, hold_steps, False),
                (
                    "button_contact_seek",
                    button_position + press_direction * press_contact_seek_depth,
                    press_contact_seek_steps,
                    False,
                ),
                ("button_retreat", button_position + outward * retreat_distance, retreat_steps, True),
            ]
            phase_exit_reasons: JsonDict = {}
            contact_hold_target = None
            base_handoff_reached = False
            base_handoff_phase = None
            base_handoff_step = None
            for phase, phase_target_position, stage_steps, break_on_tolerance in stage_targets:
                phase_best_distance: float | None = None
                phase_steps_since_best = 0
                for stage_step_index in range(stage_steps):
                    press_depth_target = None
                    if phase == "button_press":
                        press_fraction = float(stage_step_index + 1) / float(stage_steps)
                        press_depth_target = press_depth + press_fraction * (max_press_depth - press_depth)
                        eef_target_position = button_position + press_direction * press_depth_target
                    elif phase == "button_contact_seek":
                        press_fraction = float(stage_step_index + 1) / float(stage_steps)
                        press_depth_target = max_press_depth + press_fraction * (
                            press_contact_seek_depth - max_press_depth
                        )
                        eef_target_position = button_position + press_direction * press_depth_target
                    elif phase == "button_hold" and contact_hold_target is not None:
                        eef_target_position = contact_hold_target
                    else:
                        eef_target_position = phase_target_position
                    eef_position = _vector(obs, "robot0_eef_pos")
                    if eef_position is None:
                        break
                    delta = eef_target_position - np.asarray(eef_position, dtype=np.float32)
                    base_action = None
                    step_base_mode_value = None
                    phase_mobile_base_enabled = mobile_base_enabled and (
                        mobile_base_active_phases is None or phase in mobile_base_active_phases
                    )
                    if phase_mobile_base_enabled:
                        base_action = np.zeros(3, dtype=np.float32)
                        planar_button_distance = float(
                            np.linalg.norm(
                                np.asarray(eef_position, dtype=np.float32)[:2]
                                - button_position[:2]
                            )
                        )
                        if (
                            not base_handoff_reached
                            and base_xy_deadband > 0.0
                            and planar_button_distance <= base_xy_deadband
                        ):
                            base_handoff_reached = True
                            base_handoff_phase = phase
                            base_handoff_step = len(step_records)
                        if not base_handoff_reached:
                            base_delta = _delta_for_robocasa_action_frame(
                                delta,
                                obs,
                                base_delta_frame,
                            )
                            base_action[:2] = np.clip(
                                np.asarray(base_delta, dtype=np.float32)[:2]
                                * base_gain
                                * base_command_sign,
                                -base_max_delta,
                                base_max_delta,
                            )
                        step_base_mode_value = -1.0 if base_handoff_reached else base_mode_value
                    action = np.asarray(action, dtype=np.float32)
                    action[...] = 0.0
                    action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                    action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                    _set_flat_gripper_command_if_present(action, gripper_command, action_key=action_key)
                    bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                    step_result = self._env.step(
                        _format_step_action(
                            step_action,
                            bounded_action,
                            action_key,
                            gripper_command,
                            base_action=base_action,
                            base_mode_value=step_base_mode_value,
                        )
                    )
                    step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                    self._last_obs = step_obs
                    self._last_info = dict(latest_info or {})
                    control_obs = _read_current_observation(self._env)
                    obs = control_obs if isinstance(control_obs, dict) else step_obs
                    eef_after = _vector(obs, "robot0_eef_pos")
                    if eef_after is None:
                        distance = None
                        target_distance = None
                    else:
                        eef_after = np.asarray(eef_after, dtype=np.float32)
                        distance = float(np.linalg.norm(eef_after - button_position))
                        target_distance = float(np.linalg.norm(eef_after - eef_target_position))
                        best_distance = min(best_distance, distance)
                        final_distance = distance
                        if phase_best_distance is None or distance < phase_best_distance - 1e-4:
                            phase_best_distance = distance
                            phase_steps_since_best = 0
                        else:
                            phase_steps_since_best += 1
                    button_contact = _env_fixture_button_contact(self._env, fixture_name, selected_button_name)
                    fixture_state = _env_fixture_state(self._env, fixture_name)
                    step_records.append(
                        {
                            "step_index": len(step_records),
                            "phase": phase,
                            "distance_to_target": distance,
                            "target_distance": target_distance,
                            "button_contact": button_contact,
                            "contact_confirmed": bool(button_contact),
                            "fixture_state": _to_builtin(fixture_state),
                            "button_name": selected_button_name,
                            "button_position": _to_builtin(_vector_to_list(button_position)),
                            "eef_position": _to_builtin(_vector_to_list(eef_after)) if eef_after is not None else None,
                            "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                            "press_depth_target": press_depth_target,
                            "gripper_command": gripper_command,
                            "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                            "base_mode_value": step_base_mode_value,
                            "base_handoff_reached": base_handoff_reached,
                            "arm_delta_frame": arm_delta_frame,
                            "action_summary": summarize_observation(bounded_action),
                            "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                        }
                    )
                    if terminated or truncated:
                        break
                    exit_reason = None
                    if break_on_tolerance and target_distance is not None and target_distance <= tolerance:
                        exit_reason = "target_tolerance"
                    elif (
                        phase == "button_approach"
                        and distance is not None
                        and approach_stop_distance is not None
                        and distance <= approach_stop_distance
                    ):
                        exit_reason = "approach_stop_distance"
                    elif (
                        phase == "button_approach"
                        and approach_patience_steps > 0
                        and phase_best_distance is not None
                        and approach_stop_distance is not None
                        and phase_best_distance <= approach_stop_distance
                        and phase_steps_since_best >= approach_patience_steps
                    ):
                        exit_reason = "approach_stalled_near_button"
                    elif (
                        phase == "button_contact_seek"
                        and contact_seek_patience_steps > 0
                        and phase_best_distance is not None
                        and phase_best_distance <= contact_seek_stop_distance
                        and phase_steps_since_best >= contact_seek_patience_steps
                    ):
                        exit_reason = "contact_seek_stalled_near_button"
                    elif (
                        phase == "button_press"
                        and contact_seek_patience_steps > 0
                        and phase_best_distance is not None
                        and phase_best_distance <= contact_seek_stop_distance
                        and phase_steps_since_best >= contact_seek_patience_steps
                    ):
                        exit_reason = "button_press_stalled_near_button"
                    elif phase in {"button_press", "button_hold", "button_contact_seek"} and button_contact:
                        exit_reason = "button_contact"
                    elif (
                        phase in {"button_press", "button_hold", "button_contact_seek"}
                        and fixture_state is not None
                        and fixture_state != fixture_state_before
                    ):
                        exit_reason = "fixture_state_changed"
                    if exit_reason is not None:
                        if phase in {"button_press", "button_contact_seek"} and exit_reason in {
                            "button_contact",
                            "fixture_state_changed",
                        }:
                            contact_hold_target = eef_after.copy() if eef_after is not None else eef_target_position.copy()
                        phase_exit_reasons[phase] = exit_reason
                        if step_records:
                            step_records[-1]["phase_exit_reason"] = exit_reason
                        break
                if terminated or truncated:
                    break
            fixture_state_after = _env_fixture_state(self._env, fixture_name)
            button_contact_seen = any(bool(record.get("button_contact")) for record in step_records)
            state_changed = fixture_state_after is not None and fixture_state_after != fixture_state_before
            if state_changed:
                motion_status = "fixture_state_changed"
            elif button_contact_seen:
                motion_status = "button_contacted"
            elif best_distance < distance_before:
                motion_status = "distance_reduced"
            else:
                motion_status = "no_distance_reduction"
            interaction_confirmed = bool(state_changed or button_contact_seen)
            filtered_info_keys = [key for key in sorted(str(key) for key in dict(latest_info or {})) if "success" not in key.lower()]
            motion_trace_limit = max(1, int(kwargs.get("motion_trace_limit", 5)))
            return PrimitiveResult(
                name=skill_name,
                ok=interaction_confirmed,
                output={
                    **kwargs,
                    **self._action_evidence(skill_name, kwargs, target_name=fixture_name, target_position=_to_builtin(_vector_to_list(button_position))),
                    "execution_status": "executed",
                    "requires_motion_backend": True,
                    "motion_backend": "robocasa_state_delta_motion",
                    "motion_status": motion_status,
                    "fixture_name": fixture_name,
                    "button_name": selected_button_name,
                    "button_affordance": {
                        "source": (
                            "caller.button_position"
                            if kwargs.get("button_position") is not None
                            else "fixture_affordance_sites.start_buttons"
                        ),
                        "position": _to_builtin(_vector_to_list(button_position)),
                        "raw_position": _to_builtin(_vector_to_list(raw_button_position)),
                        "anchor_adjusted_position": _to_builtin(_vector_to_list(anchor_adjusted_button_position)),
                        "button_anchor_adjustment": (
                            _to_builtin(_vector_to_list(button_anchor_adjustment))
                            if button_anchor_adjustment is not None
                            else None
                        ),
                        "button_anchor_adjustment_source": button_anchor_adjustment_source,
                        "button_offset": _to_builtin(_vector_to_list(button_offset)) if button_offset is not None else None,
                        "button_offset_source": button_offset_source,
                        "gripper_contact_surface_axis": _to_builtin(
                            _vector_to_list(_coerce_vector_value(kwargs.get("gripper_contact_surface_axis")))
                        )
                        if _coerce_vector_value(kwargs.get("gripper_contact_surface_axis")) is not None
                        else None,
                        "press_direction": _to_builtin(_vector_to_list(press_direction)),
                        "press_direction_sign": press_direction_sign,
                        "press_direction_vector": _to_builtin(_vector_to_list(raw_press_direction_vector))
                        if raw_press_direction_vector is not None
                        else None,
                        "approach_direction": _to_builtin(_vector_to_list(outward)),
                    },
                    "fixture_state_before": _to_builtin(fixture_state_before),
                    "fixture_state_after": _to_builtin(fixture_state_after),
                    "button_contact_seen": button_contact_seen,
                    "state_changed": state_changed,
                    "interaction_confirmed": interaction_confirmed,
                    "distance_before": distance_before,
                    "distance_after": final_distance,
                    "best_distance": best_distance,
                    "approach_steps": approach_steps,
                    "approach_distance": approach_distance,
                    "approach_stop_distance": approach_stop_distance,
                    "approach_patience_steps": approach_patience_steps,
                    "press_depth": press_depth,
                    "max_press_depth": max_press_depth,
                    "press_contact_seek_steps": press_contact_seek_steps,
                    "press_contact_seek_depth": press_contact_seek_depth,
                    "contact_seek_stop_distance": contact_seek_stop_distance,
                    "contact_seek_patience_steps": contact_seek_patience_steps,
                    "button_anchor_adjustment": (
                        _to_builtin(_vector_to_list(button_anchor_adjustment))
                        if button_anchor_adjustment is not None
                        else None
                    ),
                    "button_anchor_adjustment_source": button_anchor_adjustment_source,
                    "button_offset": _to_builtin(_vector_to_list(button_offset)) if button_offset is not None else None,
                    "button_offset_source": button_offset_source,
                    "press_direction_sign": press_direction_sign,
                    "press_direction_vector": _to_builtin(_vector_to_list(raw_press_direction_vector))
                    if raw_press_direction_vector is not None
                    else None,
                    "press_steps": press_steps,
                    "hold_steps": hold_steps,
                    "retreat_steps": retreat_steps,
                    "retreat_distance": retreat_distance,
                    "mobile_base_enabled": mobile_base_enabled,
                    "mobile_base_active_phases": (
                        None if mobile_base_active_phases is None else sorted(mobile_base_active_phases)
                    ),
                    "base_gain": base_gain,
                    "base_max_delta": base_max_delta,
                    "base_command_sign": base_command_sign,
                    "base_mode_value": base_mode_value,
                    "base_delta_frame": base_delta_frame,
                    "base_xy_deadband": base_xy_deadband,
                    "base_handoff_reached": base_handoff_reached,
                    "base_handoff_phase": base_handoff_phase,
                    "base_handoff_step": base_handoff_step,
                    "arm_delta_frame": arm_delta_frame,
                    "motion_trace_limit": motion_trace_limit,
                    "steps": len(step_records),
                    "official_task_completion_claimed": False,
                    "step_summary": {
                        "reward": _to_builtin(latest_reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "info_keys": filtered_info_keys,
                        "observation_summary": summarize_observation(self._last_obs),
                        "phase_summary": _motion_phase_summary(step_records),
                        "phase_exit_reasons": phase_exit_reasons,
                        "best_step": _best_motion_record(step_records),
                        "final": {
                            "distance_to_button": final_distance,
                            "best_distance_to_button": best_distance,
                            "button_contact_seen": button_contact_seen,
                            "state_changed": state_changed,
                            "interaction_confirmed": interaction_confirmed,
                        },
                        "motion_trace": step_records[-motion_trace_limit:],
                    },
                },
            )

        if skill_name == "manipulate_robocasa_object":
            default_offset = [0.0, 0.0, 0.12]
        elif skill_name == "grasp_robocasa_object":
            default_offset = [0.0, 0.0, 0.03]
            if kwargs.get("grasp_approach") == "near_side":
                side_offsets = _near_side_grasp_offsets(
                    target_position, eef_position, self._objects.get(kwargs.get("object_name"))
                )
                if side_offsets is None:
                    return PrimitiveResult(name=skill_name, ok=False, error="near_side_grasp_geometry_unavailable",
                                           output={"execution_status": "not_executed"})
                if kwargs.get("offset") is None:
                    default_offset = side_offsets[0]
                if kwargs.get("grasp_contact_offset") is None:
                    kwargs["grasp_contact_offset"] = side_offsets[1]
        elif skill_name == "place_robocasa_object_at":
            target_payload = None
            if target_name in self._fixtures:
                target_payload = self._fixtures[target_name]
            elif target_name in self._objects:
                target_payload = self._objects[target_name]
            default_offset = _placement_offset(
                str(kwargs.get("relation", "at")),
                target_payload=target_payload,
                query=kwargs.get("query"),
                agent_context=kwargs.get("agent_context"),
            )
        else:
            default_offset = [0.0, 0.0, 0.0]
        offset = _vector({"offset": kwargs.get("offset") or default_offset}, "offset")
        offset = offset if offset is not None else np.zeros(3, dtype=np.float32)
        target_with_offset = target_position + offset
        last_eef_target_position = target_with_offset
        place_object_relative_control = (
            bool(kwargs.get("object_relative_control", True)) if skill_name == "place_robocasa_object_at" else False
        )
        place_transport_offset = None
        if place_object_relative_control:
            object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
            if object_position is not None:
                place_transport_offset = eef_position - object_position
        distance_metric = "object_to_target" if skill_name == "place_robocasa_object_at" else "eef_to_target"
        object_distance_before = None
        if skill_name == "place_robocasa_object_at":
            object_distance_before = _object_to_target_distance(
                obs,
                object_name=kwargs.get("object_name"),
                target_name=target_name,
                explicit_target_position=explicit_target_position,
                offset=offset,
            )
            if object_distance_before is None:
                return PrimitiveResult(
                    name=skill_name,
                    ok=False,
                    output={
                        **kwargs,
                        **self._action_evidence(skill_name, kwargs, target_name=target_name),
                        "execution_status": "not_executed",
                        "requires_motion_backend": True,
                        "motion_backend": "robocasa_state_delta_motion",
                        "distance_metric": distance_metric,
                        "available_state_keys": sorted(str(key) for key in obs),
                    },
                    error="object_place_grounding_unavailable",
                )
        distance_before = object_distance_before if object_distance_before is not None else float(np.linalg.norm(target_with_offset - eef_position))
        horizon = max(1, int(kwargs.get("horizon", 20)))
        max_delta = float(kwargs.get("max_delta", 0.3))
        gain = float(kwargs.get("gain", 2.0))
        # World-space target errors must be expressed in the live controller's
        # input frame when the caller leaves the conversion unspecified.
        arm_delta_frame = str(
            kwargs.get("arm_delta_frame")
            or runtime_action_schema.get("controller_input_ref_frame")
            or "world"
        )
        mobile_base_enabled = bool(kwargs.get("mobile_base_enabled", False))
        base_gain = float(kwargs.get("base_gain", 1.0))
        base_max_delta = float(kwargs.get("base_max_delta", 0.35))
        base_command_sign = float(kwargs.get("base_command_sign", 1.0))
        base_mode_value = float(kwargs.get("base_mode_value", 1.0))
        base_xy_deadband = max(0.0, float(kwargs.get("base_xy_deadband", 0.12)))
        tolerance = float(kwargs.get("tolerance", 0.035 if skill_name == "place_robocasa_object_at" else 0.06))
        contact_tolerance = float(kwargs.get("contact_tolerance", 0.08))
        gripper_command = float(kwargs.get("gripper_command", 1.0))
        transport_gripper_command = (
            float(kwargs.get("transport_gripper_command"))
            if skill_name == "place_robocasa_object_at" and kwargs.get("transport_gripper_command") is not None
            else (-1.0 if skill_name == "grasp_robocasa_object" else
                  1.0 if skill_name == "place_robocasa_object_at" else gripper_command)
        )
        release_gripper_command = float(
            kwargs.get("release_gripper_command", 0.0 if skill_name == "place_robocasa_object_at" else gripper_command)
        )
        release_only_when_ready = (
            bool(kwargs.get("release_only_when_ready", True)) if skill_name == "place_robocasa_object_at" else False
        )
        release_xy_tolerance = None
        if skill_name == "place_robocasa_object_at":
            configured_release_xy_tolerance = kwargs.get("release_xy_tolerance")
            release_xy_tolerance = (
                float(configured_release_xy_tolerance)
                if configured_release_xy_tolerance is not None
                else float(tolerance)
            )
        release_requires_contact = (
            bool(kwargs.get("release_requires_contact", False)) if skill_name == "place_robocasa_object_at" else False
        )
        raw_xy_early_stop_enabled = kwargs.get("object_xy_early_stop_enabled")
        object_xy_early_stop_enabled = False
        if skill_name == "place_robocasa_object_at":
            object_xy_early_stop_enabled = (
                bool(raw_xy_early_stop_enabled)
                if raw_xy_early_stop_enabled is not None
                else kwargs.get("object_xy_stop_when_within") is not None
            )
        object_xy_stop_when_within = None
        if skill_name == "place_robocasa_object_at":
            raw_xy_stop = kwargs.get("object_xy_stop_when_within")
            object_xy_stop_when_within = (
                float(raw_xy_stop)
                if raw_xy_stop is not None
                else (float(release_xy_tolerance) if release_xy_tolerance is not None else None)
            )
            if object_xy_stop_when_within is not None and object_xy_stop_when_within <= 0.0:
                object_xy_stop_when_within = None
        object_xy_stop_requires_contact = (
            bool(kwargs.get("object_xy_stop_requires_contact", False)) if skill_name == "place_robocasa_object_at" else False
        )
        grasp_contact_steps = max(0, int(kwargs.get("grasp_contact_steps", 0)))
        grasp_contact_offset = _vector(
            {"grasp_contact_offset": kwargs.get("grasp_contact_offset") or [0.0, 0.0, 0.0]},
            "grasp_contact_offset",
        )
        grasp_contact_offset = grasp_contact_offset if grasp_contact_offset is not None else np.zeros(3, dtype=np.float32)
        grasp_contact_tolerance_value = kwargs.get("grasp_contact_tolerance")
        grasp_contact_tolerance = (
            min(contact_tolerance, tolerance)
            if grasp_contact_tolerance_value is None
            else float(grasp_contact_tolerance_value)
        )
        force_grasp_contact_steps = bool(kwargs.get("force_grasp_contact_steps", True))
        grasp_hold_steps = max(0, int(kwargs.get("grasp_hold_steps", 0)))
        release_settle_steps = max(0, int(kwargs.get("release_settle_steps", 0)))
        avoidance_point = None
        min_distance_from_avoidance = None
        avoidance_tolerance = 0.0
        stop_when_avoidance_reached = False
        avoidance_distance_before = None
        if skill_name == "move_robocasa_ee_to":
            avoidance_point = _coerce_vector_value(kwargs.get("avoidance_point"))
            raw_min_distance_from_avoidance = kwargs.get("min_distance_from_avoidance")
            min_distance_from_avoidance = (
                None
                if raw_min_distance_from_avoidance is None
                else max(0.0, float(raw_min_distance_from_avoidance))
            )
            avoidance_tolerance = max(0.0, float(kwargs.get("avoidance_tolerance", 0.0)))
            stop_when_avoidance_reached = bool(kwargs.get("stop_when_avoidance_reached", False))
            if avoidance_point is not None:
                avoidance_distance_before = float(
                    np.linalg.norm(np.asarray(eef_position, dtype=np.float32) - avoidance_point)
                )
        post_release_retreat_steps = max(0, int(kwargs.get("post_release_retreat_steps", 0)))
        post_release_retreat_offset = _vector(
            {"post_release_retreat_offset": kwargs.get("post_release_retreat_offset") or [0.0, -0.30, 0.15]},
            "post_release_retreat_offset",
        )
        post_release_retreat_offset = (
            post_release_retreat_offset if post_release_retreat_offset is not None else np.zeros(3, dtype=np.float32)
        )
        post_release_retreat_min_distance = max(0.0, float(kwargs.get("post_release_retreat_min_distance", 0.0)))
        post_release_retreat_max_steps = max(0, int(kwargs.get("post_release_retreat_max_steps", 0)))
        grasp_lift_delta = float(kwargs.get("grasp_lift_delta", 0.04 if skill_name == "grasp_robocasa_object" else 0.0))
        object_error_gain = float(kwargs.get("object_error_gain", 1.0 if skill_name == "place_robocasa_object_at" else 0.0))
        object_error_clip = max(0.0, float(kwargs.get("object_error_clip", 0.18)))
        object_xy_push_steps = max(0, int(kwargs.get("object_xy_push_steps", 0)))
        object_xy_push_align_steps = max(0, int(kwargs.get("object_xy_push_align_steps", 8)))
        object_xy_push_backoff = max(0.0, float(kwargs.get("object_xy_push_backoff", 0.07)))
        object_xy_push_through = max(0.0, float(kwargs.get("object_xy_push_through", 0.03)))
        object_xy_push_z_offset = float(kwargs.get("object_xy_push_z_offset", 0.02))
        object_xy_push_reacquire_from_side = bool(kwargs.get("object_xy_push_reacquire_from_side", True))
        object_xy_contact_seek_steps = max(0, int(kwargs.get("object_xy_contact_seek_steps", 0)))
        raw_contact_seek_backoff = kwargs.get("object_xy_contact_seek_backoff")
        object_xy_contact_seek_backoff = (
            object_xy_push_backoff if raw_contact_seek_backoff is None else max(0.0, float(raw_contact_seek_backoff))
        )
        raw_contact_seek_z_offset = kwargs.get("object_xy_contact_seek_z_offset")
        object_xy_contact_seek_z_offset = (
            object_xy_push_z_offset if raw_contact_seek_z_offset is None else float(raw_contact_seek_z_offset)
        )
        contact_guard_enabled = bool(kwargs.get("contact_guard_enabled", True)) if skill_name == "place_robocasa_object_at" else False
        raw_contact_guard_tolerance = kwargs.get("contact_guard_tolerance")
        contact_guard_tolerance = (
            float(raw_contact_guard_tolerance)
            if raw_contact_guard_tolerance is not None
            else max(contact_tolerance * 1.25, contact_tolerance + 0.02)
        )
        contact_guard_recover_steps = max(0, int(kwargs.get("contact_guard_recover_steps", 12)))
        raw_contact_guard_offset = kwargs.get("contact_guard_offset")
        contact_guard_offset = _vector({"contact_guard_offset": raw_contact_guard_offset}, "contact_guard_offset")
        step_records: list[JsonDict] = []
        latest_reward: Any = None
        latest_info: JsonDict = {}
        terminated = False
        truncated = False
        best_distance = distance_before
        final_xy_distance = (
            _object_to_target_xy_distance(
                obs,
                object_name=kwargs.get("object_name"),
                target_name=target_name,
                explicit_target_position=explicit_target_position,
                offset=offset,
            )
            if skill_name == "place_robocasa_object_at"
            else None
        )
        best_xy_distance = final_xy_distance
        final_contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
        best_contact_distance = final_contact_distance
        ever_contact_confirmed = final_contact_distance is not None and final_contact_distance <= contact_tolerance
        transport_contact_confirmed = False
        release_ready: bool | None = None
        release_xy_ready: bool | None = None
        release_contact_ready: bool | None = None
        release_executed = False
        release_skipped_reason: str | None = None
        post_release_drift_detected = False
        xy_early_stop_triggered = False
        xy_early_stop_reason: str | None = None
        xy_early_stop_step: int | None = None
        xy_early_stop_phase: str | None = None
        if contact_guard_offset is None:
            if place_transport_offset is not None and ever_contact_confirmed:
                contact_guard_offset = place_transport_offset.copy()
            else:
                contact_guard_offset = np.zeros(3, dtype=np.float32)

        def base_action_for_delta(delta_vector: Any) -> tuple[Any | None, float | None]:
            if not mobile_base_enabled:
                return None, None
            delta_array = np.asarray(delta_vector, dtype=np.float32).reshape(-1)
            if skill_name == "grasp_robocasa_object":
                # The mobile-base controller consumes motion in the current
                # base frame; grasp targets are expressed in world coordinates.
                delta_array = _delta_for_robocasa_action_frame(delta_array, obs, "base")
            base_action = np.zeros(3, dtype=np.float32)
            if float(np.linalg.norm(delta_array[:2])) > base_xy_deadband:
                base_action[:2] = np.clip(delta_array[:2] * base_gain * base_command_sign, -base_max_delta, base_max_delta)
            return base_action, base_mode_value

        def place_xy_stop_ready(xy_distance: Any, contact_distance: Any) -> bool:
            if (
                skill_name != "place_robocasa_object_at"
                or not object_xy_early_stop_enabled
                or object_xy_stop_when_within is None
                or xy_distance is None
            ):
                return False
            if float(xy_distance) > float(object_xy_stop_when_within):
                return False
            if object_xy_stop_requires_contact:
                return contact_distance is not None and float(contact_distance) <= float(contact_tolerance)
            return True

        def maybe_mark_place_xy_stop(record: JsonDict) -> bool:
            nonlocal xy_early_stop_triggered, xy_early_stop_reason, xy_early_stop_step, xy_early_stop_phase
            if not place_xy_stop_ready(record.get("xy_distance_to_target"), record.get("object_to_eef_distance")):
                return False
            xy_early_stop_triggered = True
            xy_early_stop_reason = "object_xy_stop_when_within"
            xy_early_stop_step = int(record.get("step_index", len(step_records) - 1))
            xy_early_stop_phase = str(record.get("phase") or "")
            record["phase_exit_reason"] = xy_early_stop_reason
            record["object_xy_stop_when_within"] = object_xy_stop_when_within
            record["object_xy_stop_requires_contact"] = object_xy_stop_requires_contact
            return True

        def avoidance_distance(current_obs: JsonDict) -> float | None:
            if skill_name != "move_robocasa_ee_to" or avoidance_point is None:
                return None
            current_eef = _vector(current_obs, "robot0_eef_pos")
            if current_eef is None:
                return None
            return float(np.linalg.norm(np.asarray(current_eef, dtype=np.float32) - avoidance_point))

        def avoidance_reached(distance_value: Any) -> bool | None:
            if skill_name != "move_robocasa_ee_to" or min_distance_from_avoidance is None:
                return None
            if distance_value is None:
                return False
            return float(distance_value) >= float(min_distance_from_avoidance) - float(avoidance_tolerance)

        for step_index in range(horizon):
            eef_position = _vector(obs, "robot0_eef_pos")
            target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
            if eef_position is None or target_position is None:
                break
            target_with_offset = target_position + offset
            eef_target_position = target_with_offset
            object_error_vector = None
            phase = "transport"
            pre_step_contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
            if place_object_relative_control and place_transport_offset is not None:
                eef_target_position = target_with_offset + place_transport_offset
                if object_error_gain:
                    object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                    if object_position is not None:
                        object_error_vector = target_with_offset - object_position
                        if object_error_clip:
                            object_error_vector = np.clip(object_error_vector, -object_error_clip, object_error_clip)
                        eef_target_position = eef_target_position + object_error_vector * object_error_gain
                        if (
                            contact_guard_enabled
                            and pre_step_contact_distance is not None
                            and pre_step_contact_distance > contact_guard_tolerance
                        ):
                            eef_target_position = object_position + contact_guard_offset
                            phase = "contact_guard"
            last_eef_target_position = eef_target_position
            delta = eef_target_position - eef_position
            action = np.asarray(action, dtype=np.float32)
            action[...] = 0.0
            action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
            action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
            _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
            base_action, step_base_mode_value = base_action_for_delta(delta)
            bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
            step_result = self._env.step(
                _format_step_action(
                    step_action,
                    bounded_action,
                    action_key,
                    transport_gripper_command,
                    base_action=base_action,
                    base_mode_value=step_base_mode_value,
                )
            )
            step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
            self._last_obs = step_obs
            self._last_info = dict(latest_info or {})
            control_obs = _read_current_observation(self._env)
            obs = control_obs if isinstance(control_obs, dict) else step_obs
            if skill_name == "place_robocasa_object_at":
                distance = _object_to_target_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
            else:
                distance = _distance(obs, target_name=target_name, explicit_position=explicit_target_position, offset=offset)
            if distance is not None:
                best_distance = min(best_distance, distance)
            if skill_name == "place_robocasa_object_at":
                xy_distance = _object_to_target_xy_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if xy_distance is not None:
                    final_xy_distance = xy_distance
                    best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
            else:
                xy_distance = None
            contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
            if contact_distance is not None:
                final_contact_distance = contact_distance
                best_contact_distance = (
                    contact_distance
                    if best_contact_distance is None
                    else min(best_contact_distance, contact_distance)
                )
            step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
            ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
            transport_contact_confirmed = transport_contact_confirmed or step_contact_confirmed
            step_avoidance_distance = avoidance_distance(obs)
            step_avoidance_reached = avoidance_reached(step_avoidance_distance)
            step_records.append(
                {
                    "step_index": step_index,
                    "phase": phase,
                    "distance_to_target": distance,
                    "distance_metric": distance_metric,
                    "xy_distance_to_target": xy_distance,
                    "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                    "object_error_vector": _to_builtin(_vector_to_list(object_error_vector)),
                    "object_error_gain": object_error_gain if skill_name == "place_robocasa_object_at" else None,
                    "object_to_eef_distance": contact_distance,
                    "pre_step_object_to_eef_distance": pre_step_contact_distance,
                    "avoidance_distance": step_avoidance_distance,
                    "avoidance_reached": step_avoidance_reached,
                    "contact_guard_tolerance": contact_guard_tolerance if skill_name == "place_robocasa_object_at" else None,
                    "contact_confirmed": step_contact_confirmed,
                    "gripper_command": transport_gripper_command,
                    "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                    "base_mode_value": step_base_mode_value,
                    "action_summary": summarize_observation(bounded_action),
                    "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                }
            )
            if maybe_mark_place_xy_stop(step_records[-1]):
                break
            if stop_when_avoidance_reached and step_avoidance_reached is True:
                step_records[-1]["phase_exit_reason"] = "avoidance_clearance_reached"
                break
            if distance is not None and distance <= tolerance:
                if stop_when_avoidance_reached and step_avoidance_reached is not True:
                    step_records[-1]["target_tolerance_without_avoidance_clearance"] = True
                else:
                    step_records[-1]["phase_exit_reason"] = "target_tolerance"
                    break
            if terminated or truncated:
                break

        grasp_preclose_failure: str | None = None
        grasp_wrist_yaw_error_rad: float | None = None
        wrist_yaw_offset = float(kwargs.get("grasp_wrist_yaw_offset_rad", 0.0)) if skill_name == "grasp_robocasa_object" else 0.0
        if skill_name == "grasp_robocasa_object" and step_records and abs(wrist_yaw_offset) > .01 and not terminated and not truncated:
            initial_rotation = _quat_xyzw_to_rotation_matrix(_lookup_observation_value(obs, "robot0_eef_quat"))
            if initial_rotation is None or action.shape[0] < 6:
                grasp_preclose_failure = "wrist_rotation_unavailable"
            else:
                horizontal_axis = 0 if np.linalg.norm(initial_rotation[:2, 0]) >= np.linalg.norm(initial_rotation[:2, 1]) else 1
                target_yaw = float(np.arctan2(initial_rotation[1, horizontal_axis], initial_rotation[0, horizontal_axis]) + wrist_yaw_offset)
                object_before_wrist = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                for _ in range(12):
                    current_rotation = _quat_xyzw_to_rotation_matrix(_lookup_observation_value(obs, "robot0_eef_quat"))
                    if current_rotation is None:
                        grasp_preclose_failure = "wrist_rotation_unavailable"
                        break
                    current_yaw = float(np.arctan2(current_rotation[1, horizontal_axis], current_rotation[0, horizontal_axis]))
                    grasp_wrist_yaw_error_rad = float(np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw)))
                    if abs(grasp_wrist_yaw_error_rad) <= .09:
                        break
                    base_rotation = _quat_xyzw_to_rotation_matrix(_lookup_observation_value(obs, "robot0_base_quat"))
                    yaw_axis = base_rotation.T @ np.asarray([0., 0., 1.], dtype=np.float32) if base_rotation is not None else np.asarray([0., 0., 1.], dtype=np.float32)
                    action = np.asarray(action, dtype=np.float32)
                    action[...] = 0.0
                    action[3:6] = np.clip(yaw_axis * grasp_wrist_yaw_error_rad * 2.0, -.45, .45)
                    _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
                    bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                    step_result = self._env.step(_format_step_action(
                        step_action, bounded_action, action_key, transport_gripper_command,
                        base_action=np.zeros(3, dtype=np.float32) if mobile_base_enabled else None,
                        base_mode_value=base_mode_value if mobile_base_enabled else None,
                    ))
                    step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                    self._last_obs = step_obs
                    self._last_info = dict(latest_info or {})
                    control_obs = _read_current_observation(self._env)
                    obs = control_obs if isinstance(control_obs, dict) else step_obs
                    step_records.append({"step_index": len(step_records), "phase": "grasp_wrist_align",
                                         "yaw_error_rad": grasp_wrist_yaw_error_rad, "gripper_command": transport_gripper_command,
                                         "base_action": [0.0, 0.0, 0.0] if mobile_base_enabled else None})
                    object_after_wrist = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                    if object_before_wrist is not None and object_after_wrist is not None and float(
                        np.linalg.norm(object_after_wrist - object_before_wrist)
                    ) > .12:
                        grasp_preclose_failure = "target_displaced_during_wrist_alignment"
                    if grasp_preclose_failure is not None or terminated or truncated:
                        break
                current_rotation = _quat_xyzw_to_rotation_matrix(_lookup_observation_value(obs, "robot0_eef_quat"))
                if current_rotation is not None:
                    current_yaw = float(np.arctan2(current_rotation[1, horizontal_axis], current_rotation[0, horizontal_axis]))
                    grasp_wrist_yaw_error_rad = float(np.arctan2(np.sin(target_yaw - current_yaw), np.cos(target_yaw - current_yaw)))
                if grasp_preclose_failure is None and grasp_wrist_yaw_error_rad is not None and abs(grasp_wrist_yaw_error_rad) > .15:
                    grasp_preclose_failure = "wrist_alignment_unreached"
        if (
            skill_name == "grasp_robocasa_object"
            and step_records
            and grasp_preclose_failure is None
            and (force_grasp_contact_steps or not ever_contact_confirmed)
            and grasp_contact_steps > 0
            and not terminated
            and not truncated
        ):
            contact_start_object_position = _target_position(
                obs, target_name=kwargs.get("object_name"), explicit_position=None
            )
            for _ in range(grasp_contact_steps):
                eef_position = _vector(obs, "robot0_eef_pos")
                object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                if eef_position is None or object_position is None:
                    break
                eef_target_position = object_position + grasp_contact_offset
                last_eef_target_position = eef_target_position
                delta = eef_target_position - eef_position
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
                # Once the arm takes over fine contact, moving the base can
                # push a small object away before the fingers close.
                base_action = np.zeros(3, dtype=np.float32) if mobile_base_enabled else None
                step_base_mode_value = base_mode_value if mobile_base_enabled else None
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        transport_gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _distance(obs, target_name=target_name, explicit_position=explicit_target_position, offset=offset)
                if distance is not None:
                    best_distance = min(best_distance, distance)
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                gripper_object_contact = _env_gripper_object_contact(self._env, kwargs.get("object_name"))
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": "grasp_contact_approach",
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                        "object_to_eef_distance": contact_distance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_object_contact": gripper_object_contact,
                        "grasp_contact_tolerance": grasp_contact_tolerance,
                        "gripper_command": transport_gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                # Center distance is only a rough proximity cue. On real RoboCasa
                # objects it can be small while the open fingers have not made
                # contact, so closing here loses the object. Keep approaching
                # until physical contact or the bounded approach budget ends.
                grasp_contact_reached = contact_distance is not None and contact_distance <= grasp_contact_tolerance
                current_object_position = _target_position(
                    obs, target_name=kwargs.get("object_name"), explicit_position=None
                )
                if contact_start_object_position is not None and current_object_position is not None and float(
                    np.linalg.norm(current_object_position - contact_start_object_position)
                ) > .12:
                    grasp_preclose_failure = "target_displaced_during_approach"
                    step_records[-1]["phase_exit_reason"] = grasp_preclose_failure
                    break
                if (grasp_contact_reached and gripper_object_contact is True) or terminated or truncated:
                    break
            if (
                grasp_preclose_failure is None
                and final_contact_distance is not None
                and final_contact_distance > .12
                and _env_gripper_object_contact(self._env, kwargs.get("object_name")) is not True
            ):
                grasp_preclose_failure = "grasp_target_unreached"

        grasp_lift_evidence: JsonDict | None = None
        if skill_name == "grasp_robocasa_object" and step_records and not terminated and not truncated and grasp_preclose_failure is None:
            for _ in range(grasp_hold_steps):
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                _set_flat_gripper_command_if_present(action, gripper_command, action_key=action_key)
                base_action, step_base_mode_value = base_action_for_delta(np.zeros(3, dtype=np.float32))
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _distance(obs, target_name=target_name, explicit_position=explicit_target_position, offset=offset)
                if distance is not None:
                    best_distance = min(best_distance, distance)
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": "grasp_hold",
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "object_to_eef_distance": contact_distance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_command": gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                if terminated or truncated:
                    break

            # Measure object/EEF coupling during a separate 4–6 cm lift, not
            # across the whole approach where a stationary object can look held.
            if not terminated and not truncated:
                before_lift = _robocasa_live_transport_observation(
                    self._env, obs, object_name=kwargs.get("object_name"), previous_state=None,
                    contact_tolerance=contact_tolerance, gripper_closed_threshold=.035,
                    coupling_tolerance=.02, minimum_coupling_motion=.005,
                )
                lift_start = _vector(obs, "robot0_eef_pos")
                if lift_start is not None:
                    lift_target = lift_start + np.asarray([0., 0., min(.06, max(.04, grasp_lift_delta))], dtype=np.float32)
                    for _ in range(36):
                        current_eef = _vector(obs, "robot0_eef_pos")
                        if current_eef is None:
                            break
                        delta = lift_target - current_eef
                        if float(np.linalg.norm(delta)) <= .008:
                            break
                        action = np.asarray(action, dtype=np.float32)
                        action[...] = 0.0
                        action[:3] = np.clip(
                            _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame) * gain,
                            -max_delta, max_delta,
                        )
                        _set_flat_gripper_command_if_present(action, gripper_command, action_key=action_key)
                        bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                        step_result = self._env.step(_format_step_action(
                            step_action, bounded_action, action_key, gripper_command,
                            base_action=np.zeros(3, dtype=np.float32) if mobile_base_enabled else None,
                            base_mode_value=base_mode_value if mobile_base_enabled else None,
                        ))
                        step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                        self._last_obs = step_obs
                        self._last_info = dict(latest_info or {})
                        control_obs = _read_current_observation(self._env)
                        obs = control_obs if isinstance(control_obs, dict) else step_obs
                        step_records.append({
                            "step_index": len(step_records), "phase": "grasp_lift",
                            "eef_target_position": _to_builtin(_vector_to_list(lift_target)),
                            "eef_position": _to_builtin(_vector_to_list(_vector(obs, "robot0_eef_pos"))),
                            "object_position": _to_builtin(_vector_to_list(_target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None))),
                            "object_to_eef_distance": _object_to_eef_distance(obs, object_name=kwargs.get("object_name")),
                            "gripper_command": gripper_command,
                        })
                        if terminated or truncated:
                            break
                    grasp_lift_evidence = _robocasa_live_transport_observation(
                        self._env, obs, object_name=kwargs.get("object_name"), previous_state=before_lift,
                        contact_tolerance=contact_tolerance, gripper_closed_threshold=.035,
                        coupling_tolerance=.02, minimum_coupling_motion=.005,
                    )

        if (
            skill_name == "place_robocasa_object_at"
            and step_records
            and contact_guard_enabled
            and contact_guard_recover_steps > 0
            and not xy_early_stop_triggered
            and not terminated
            and not truncated
        ):
            for _ in range(contact_guard_recover_steps):
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None and contact_distance <= contact_tolerance:
                    break
                eef_position = _vector(obs, "robot0_eef_pos")
                object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                if eef_position is None or object_position is None:
                    break
                eef_target_position = object_position + contact_guard_offset
                last_eef_target_position = eef_target_position
                delta = eef_target_position - eef_position
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
                base_action, step_base_mode_value = base_action_for_delta(delta)
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        transport_gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _object_to_target_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if distance is not None:
                    best_distance = min(best_distance, distance)
                xy_distance = _object_to_target_xy_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if xy_distance is not None:
                    final_xy_distance = xy_distance
                    best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                transport_contact_confirmed = transport_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": "contact_guard_recover",
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "xy_distance_to_target": xy_distance,
                        "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                        "object_to_eef_distance": contact_distance,
                        "contact_guard_tolerance": contact_guard_tolerance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_command": transport_gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                if maybe_mark_place_xy_stop(step_records[-1]):
                    break
                if terminated or truncated:
                    break

        if (
            skill_name == "place_robocasa_object_at"
            and step_records
            and object_xy_contact_seek_steps > 0
            and not xy_early_stop_triggered
            and not terminated
            and not truncated
        ):
            for _ in range(object_xy_contact_seek_steps):
                contact_distance_before_seek = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance_before_seek is not None and contact_distance_before_seek <= contact_tolerance:
                    break
                eef_position = _vector(obs, "robot0_eef_pos")
                object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
                if eef_position is None or object_position is None or target_position is None:
                    break
                target_with_offset = target_position + offset
                xy_error = target_with_offset[:2] - object_position[:2]
                xy_distance = float(np.linalg.norm(xy_error))
                push_unit = xy_error / max(xy_distance, 1e-6)
                eef_target_position = object_position.copy()
                eef_target_position[:2] = object_position[:2] - push_unit * object_xy_contact_seek_backoff
                eef_target_position[2] = object_position[2] + object_xy_contact_seek_z_offset
                last_eef_target_position = eef_target_position
                delta = eef_target_position - eef_position
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
                base_action, step_base_mode_value = base_action_for_delta(delta)
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        transport_gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _object_to_target_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if distance is not None:
                    best_distance = min(best_distance, distance)
                xy_distance_after = _object_to_target_xy_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if xy_distance_after is not None:
                    final_xy_distance = xy_distance_after
                    best_xy_distance = (
                        xy_distance_after if best_xy_distance is None else min(best_xy_distance, xy_distance_after)
                    )
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                transport_contact_confirmed = transport_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": "xy_push_contact_seek",
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "xy_distance_to_target": xy_distance_after,
                        "xy_distance_before_step": xy_distance,
                        "push_unit_xy": _to_builtin([float(push_unit[0]), float(push_unit[1])]),
                        "object_xy_contact_seek_backoff": object_xy_contact_seek_backoff,
                        "object_xy_contact_seek_z_offset": object_xy_contact_seek_z_offset,
                        "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                        "object_to_eef_distance": contact_distance,
                        "pre_step_object_to_eef_distance": contact_distance_before_seek,
                        "contact_tolerance": contact_tolerance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_command": transport_gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                if maybe_mark_place_xy_stop(step_records[-1]):
                    break
                if step_contact_confirmed or terminated or truncated:
                    break

        if (
            skill_name == "place_robocasa_object_at"
            and step_records
            and object_xy_push_steps > 0
            and not xy_early_stop_triggered
            and not terminated
            and not truncated
        ):
            refine_budget = object_xy_push_align_steps + object_xy_push_steps
            for refine_index in range(refine_budget):
                eef_position = _vector(obs, "robot0_eef_pos")
                object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
                if eef_position is None or object_position is None or target_position is None:
                    break
                target_with_offset = target_position + offset
                xy_error = target_with_offset[:2] - object_position[:2]
                xy_distance = float(np.linalg.norm(xy_error))
                contact_distance_before_push = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if place_xy_stop_ready(xy_distance, contact_distance_before_push):
                    xy_early_stop_triggered = True
                    xy_early_stop_reason = "object_xy_stop_when_within"
                    xy_early_stop_step = len(step_records) - 1 if step_records else None
                    xy_early_stop_phase = str(step_records[-1].get("phase") or "") if step_records else None
                    if step_records:
                        step_records[-1]["phase_exit_reason"] = xy_early_stop_reason
                        step_records[-1]["object_xy_stop_when_within"] = object_xy_stop_when_within
                        step_records[-1]["object_xy_stop_requires_contact"] = object_xy_stop_requires_contact
                    break
                if xy_distance <= tolerance:
                    break
                push_unit = xy_error / max(xy_distance, 1e-6)
                contact_preserve_tolerance = min(float(contact_guard_tolerance), max(float(contact_tolerance), 1e-6))
                if (
                    contact_guard_enabled
                    and contact_distance_before_push is not None
                    and contact_distance_before_push > contact_preserve_tolerance
                ):
                    phase = "xy_push_reacquire" if object_xy_push_reacquire_from_side else "xy_push_contact_guard"
                elif refine_index < object_xy_push_align_steps:
                    phase = "xy_push_align"
                else:
                    phase = "xy_push"
                eef_target_position = object_position.copy()
                adaptive_backoff = object_xy_push_backoff
                adaptive_through = object_xy_push_through
                if phase == "xy_push_contact_guard":
                    eef_target_position = object_position + contact_guard_offset
                elif phase in {"xy_push_align", "xy_push_reacquire"}:
                    if xy_distance <= max(float(release_xy_tolerance or tolerance) * 2.5, 0.10):
                        adaptive_backoff = min(object_xy_push_backoff, max(0.012, xy_distance * 0.35))
                    else:
                        adaptive_backoff = object_xy_push_backoff
                    eef_target_position[:2] = object_position[:2] - push_unit * adaptive_backoff
                else:
                    if xy_distance <= max(float(release_xy_tolerance or tolerance) * 2.5, 0.10):
                        adaptive_through = max(object_xy_push_through, min(0.14, xy_distance + 0.025))
                    else:
                        adaptive_through = object_xy_push_through
                    eef_target_position[:2] = target_with_offset[:2] + push_unit * adaptive_through
                eef_target_position[2] = object_position[2] + object_xy_push_z_offset
                last_eef_target_position = eef_target_position
                delta = eef_target_position - eef_position
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                _set_flat_gripper_command_if_present(action, transport_gripper_command, action_key=action_key)
                base_action, step_base_mode_value = base_action_for_delta(delta)
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        transport_gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _object_to_target_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if distance is not None:
                    best_distance = min(best_distance, distance)
                xy_distance_after = _object_to_target_xy_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if xy_distance_after is not None:
                    final_xy_distance = xy_distance_after
                    best_xy_distance = (
                        xy_distance_after if best_xy_distance is None else min(best_xy_distance, xy_distance_after)
                    )
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                transport_contact_confirmed = transport_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": phase,
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "xy_distance_to_target": xy_distance_after,
                        "xy_distance_before_step": xy_distance,
                        "push_unit_xy": _to_builtin([float(push_unit[0]), float(push_unit[1])]),
                        "object_xy_push_backoff": object_xy_push_backoff,
                        "object_xy_push_through": object_xy_push_through,
                        "adaptive_object_xy_push_backoff": adaptive_backoff,
                        "adaptive_object_xy_push_through": adaptive_through,
                        "contact_preserve_tolerance": contact_preserve_tolerance,
                        "object_xy_push_reacquire_from_side": object_xy_push_reacquire_from_side,
                        "eef_target_position": _to_builtin(_vector_to_list(eef_target_position)),
                        "object_to_eef_distance": contact_distance,
                        "pre_step_object_to_eef_distance": contact_distance_before_push,
                        "contact_guard_tolerance": contact_guard_tolerance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_command": transport_gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                if maybe_mark_place_xy_stop(step_records[-1]):
                    break
                if terminated or truncated:
                    break

        if skill_name == "place_robocasa_object_at" and step_records and not terminated and not truncated:
            release_xy_ready = (
                final_xy_distance is not None
                and release_xy_tolerance is not None
                and final_xy_distance <= release_xy_tolerance
            )
            release_contact_ready = (
                not release_requires_contact
                or (final_contact_distance is not None and final_contact_distance <= contact_tolerance)
            )
            release_ready = bool((release_xy_ready and release_contact_ready) or not release_only_when_ready)
            if not release_ready:
                if not release_xy_ready:
                    release_skipped_reason = "release_skipped_not_within_xy_tolerance"
                elif not release_contact_ready:
                    release_skipped_reason = "release_skipped_contact_not_stable"
                else:
                    release_skipped_reason = "release_skipped_not_ready"

        if (
            skill_name == "place_robocasa_object_at"
            and step_records
            and not terminated
            and not truncated
            and release_ready is not False
        ):
            release_executed = True
            action = np.asarray(action, dtype=np.float32)
            action[...] = 0.0
            _set_flat_gripper_command_if_present(action, release_gripper_command, action_key=action_key)
            base_action, step_base_mode_value = base_action_for_delta(np.zeros(3, dtype=np.float32))
            bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
            step_result = self._env.step(
                _format_step_action(
                    step_action,
                    bounded_action,
                    action_key,
                    release_gripper_command,
                    base_action=base_action,
                    base_mode_value=step_base_mode_value,
                )
            )
            step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
            self._last_obs = step_obs
            self._last_info = dict(latest_info or {})
            control_obs = _read_current_observation(self._env)
            obs = control_obs if isinstance(control_obs, dict) else step_obs
            distance = _object_to_target_distance(
                obs,
                object_name=kwargs.get("object_name"),
                target_name=target_name,
                explicit_target_position=explicit_target_position,
                offset=offset,
            )
            if distance is not None:
                best_distance = min(best_distance, distance)
            xy_distance = _object_to_target_xy_distance(
                obs,
                object_name=kwargs.get("object_name"),
                target_name=target_name,
                explicit_target_position=explicit_target_position,
                offset=offset,
            )
            if xy_distance is not None:
                final_xy_distance = xy_distance
                best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
            contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
            if contact_distance is not None:
                final_contact_distance = contact_distance
                best_contact_distance = (
                    contact_distance
                    if best_contact_distance is None
                    else min(best_contact_distance, contact_distance)
                )
            step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
            ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
            step_records.append(
                {
                    "step_index": len(step_records),
                    "phase": "release",
                    "distance_to_target": distance,
                    "distance_metric": distance_metric,
                    "xy_distance_to_target": xy_distance,
                    "object_to_eef_distance": contact_distance,
                    "contact_confirmed": step_contact_confirmed,
                    "gripper_command": release_gripper_command,
                    "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                    "base_mode_value": step_base_mode_value,
                    "action_summary": summarize_observation(bounded_action),
                    "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                }
            )
            if (
                release_only_when_ready
                and release_xy_tolerance is not None
                and xy_distance is not None
                and xy_distance > release_xy_tolerance
            ):
                post_release_drift_detected = True
                step_records[-1]["phase_exit_reason"] = "released_object_drifted_outside_xy_tolerance"
            for _ in range(release_settle_steps):
                if terminated or truncated or post_release_drift_detected:
                    break
                action = np.asarray(action, dtype=np.float32)
                action[...] = 0.0
                _set_flat_gripper_command_if_present(action, release_gripper_command, action_key=action_key)
                base_action, step_base_mode_value = base_action_for_delta(np.zeros(3, dtype=np.float32))
                bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                step_result = self._env.step(
                    _format_step_action(
                        step_action,
                        bounded_action,
                        action_key,
                        release_gripper_command,
                        base_action=base_action,
                        base_mode_value=step_base_mode_value,
                    )
                )
                step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                self._last_obs = step_obs
                self._last_info = dict(latest_info or {})
                control_obs = _read_current_observation(self._env)
                obs = control_obs if isinstance(control_obs, dict) else step_obs
                distance = _object_to_target_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if distance is not None:
                    best_distance = min(best_distance, distance)
                xy_distance = _object_to_target_xy_distance(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                )
                if xy_distance is not None:
                    final_xy_distance = xy_distance
                    best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
                contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                if contact_distance is not None:
                    final_contact_distance = contact_distance
                    best_contact_distance = (
                        contact_distance
                        if best_contact_distance is None
                        else min(best_contact_distance, contact_distance)
                    )
                step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                step_records.append(
                    {
                        "step_index": len(step_records),
                        "phase": "settle",
                        "distance_to_target": distance,
                        "distance_metric": distance_metric,
                        "xy_distance_to_target": xy_distance,
                        "object_to_eef_distance": contact_distance,
                        "contact_confirmed": step_contact_confirmed,
                        "gripper_command": release_gripper_command,
                        "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                        "base_mode_value": step_base_mode_value,
                        "action_summary": summarize_observation(bounded_action),
                        "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                    }
                )
                if (
                    release_only_when_ready
                    and release_xy_tolerance is not None
                    and xy_distance is not None
                    and xy_distance > release_xy_tolerance
                ):
                    post_release_drift_detected = True
                    step_records[-1]["phase_exit_reason"] = "released_object_drifted_outside_xy_tolerance"
                    break
            if post_release_retreat_steps and not post_release_drift_detected and not terminated and not truncated:
                eef_position = _vector(obs, "robot0_eef_pos")
                retreat_target = eef_position + post_release_retreat_offset if eef_position is not None else None
                for _ in range(post_release_retreat_steps):
                    if terminated or truncated or retreat_target is None:
                        break
                    eef_position = _vector(obs, "robot0_eef_pos")
                    if eef_position is None:
                        break
                    delta = retreat_target - eef_position
                    action = np.asarray(action, dtype=np.float32)
                    action[...] = 0.0
                    action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                    action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                    _set_flat_gripper_command_if_present(action, release_gripper_command, action_key=action_key)
                    base_action, step_base_mode_value = base_action_for_delta(delta)
                    bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                    step_result = self._env.step(
                        _format_step_action(
                            step_action,
                            bounded_action,
                            action_key,
                            release_gripper_command,
                            base_action=base_action,
                            base_mode_value=step_base_mode_value,
                        )
                    )
                    step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                    self._last_obs = step_obs
                    self._last_info = dict(latest_info or {})
                    control_obs = _read_current_observation(self._env)
                    obs = control_obs if isinstance(control_obs, dict) else step_obs
                    distance = _object_to_target_distance(
                        obs,
                        object_name=kwargs.get("object_name"),
                        target_name=target_name,
                        explicit_target_position=explicit_target_position,
                        offset=offset,
                    )
                    if distance is not None:
                        best_distance = min(best_distance, distance)
                    xy_distance = _object_to_target_xy_distance(
                        obs,
                        object_name=kwargs.get("object_name"),
                        target_name=target_name,
                        explicit_target_position=explicit_target_position,
                        offset=offset,
                    )
                    if xy_distance is not None:
                        final_xy_distance = xy_distance
                        best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
                    contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                    if contact_distance is not None:
                        final_contact_distance = contact_distance
                        best_contact_distance = (
                            contact_distance
                            if best_contact_distance is None
                            else min(best_contact_distance, contact_distance)
                        )
                    step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                    ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                    step_records.append(
                        {
                            "step_index": len(step_records),
                            "phase": "post_release_retreat",
                            "distance_to_target": distance,
                            "distance_metric": distance_metric,
                            "xy_distance_to_target": xy_distance,
                            "object_to_eef_distance": contact_distance,
                            "contact_confirmed": step_contact_confirmed,
                            "gripper_command": release_gripper_command,
                            "eef_target_position": _to_builtin(_vector_to_list(retreat_target)),
                            "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                            "base_mode_value": step_base_mode_value,
                            "action_summary": summarize_observation(bounded_action),
                            "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                        }
                    )
            if (
                post_release_retreat_min_distance > 0.0
                and post_release_retreat_max_steps > 0
                and not post_release_drift_detected
                and not terminated
                and not truncated
            ):
                for _ in range(post_release_retreat_max_steps):
                    contact_distance_before_clearance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                    if (
                        contact_distance_before_clearance is not None
                        and contact_distance_before_clearance >= post_release_retreat_min_distance
                    ):
                        break
                    eef_position = _vector(obs, "robot0_eef_pos")
                    object_position = _target_position(obs, target_name=kwargs.get("object_name"), explicit_position=None)
                    if eef_position is None or object_position is None:
                        break
                    away = np.asarray(eef_position, dtype=np.float32) - np.asarray(object_position, dtype=np.float32)
                    if float(np.linalg.norm(away)) < 1e-5:
                        away = np.asarray(post_release_retreat_offset, dtype=np.float32)
                    if float(np.linalg.norm(away)) < 1e-5:
                        away = np.array([0.0, -1.0, 0.25], dtype=np.float32)
                    away = away / float(np.linalg.norm(away))
                    clearance_target = np.asarray(object_position, dtype=np.float32) + away * post_release_retreat_min_distance
                    clearance_target[2] += min(0.08, max(0.02, post_release_retreat_min_distance * 0.25))
                    delta = clearance_target - np.asarray(eef_position, dtype=np.float32)
                    action = np.asarray(action, dtype=np.float32)
                    action[...] = 0.0
                    action_delta = _delta_for_robocasa_action_frame(delta, obs, arm_delta_frame)
                    action[:3] = np.clip(action_delta * gain, -max_delta, max_delta)
                    _set_flat_gripper_command_if_present(action, release_gripper_command, action_key=action_key)
                    base_action, step_base_mode_value = base_action_for_delta(delta)
                    bounded_action = np.clip(action, low, high) if low is not None and high is not None else action
                    step_result = self._env.step(
                        _format_step_action(
                            step_action,
                            bounded_action,
                            action_key,
                            release_gripper_command,
                            base_action=base_action,
                            base_mode_value=step_base_mode_value,
                        )
                    )
                    step_obs, latest_reward, terminated, truncated, latest_info = _split_step_result(step_result)
                    self._last_obs = step_obs
                    self._last_info = dict(latest_info or {})
                    control_obs = _read_current_observation(self._env)
                    obs = control_obs if isinstance(control_obs, dict) else step_obs
                    distance = _object_to_target_distance(
                        obs,
                        object_name=kwargs.get("object_name"),
                        target_name=target_name,
                        explicit_target_position=explicit_target_position,
                        offset=offset,
                    )
                    if distance is not None:
                        best_distance = min(best_distance, distance)
                    xy_distance = _object_to_target_xy_distance(
                        obs,
                        object_name=kwargs.get("object_name"),
                        target_name=target_name,
                        explicit_target_position=explicit_target_position,
                        offset=offset,
                    )
                    if xy_distance is not None:
                        final_xy_distance = xy_distance
                        best_xy_distance = xy_distance if best_xy_distance is None else min(best_xy_distance, xy_distance)
                    contact_distance = _object_to_eef_distance(obs, object_name=kwargs.get("object_name"))
                    if contact_distance is not None:
                        final_contact_distance = contact_distance
                        best_contact_distance = (
                            contact_distance
                            if best_contact_distance is None
                            else min(best_contact_distance, contact_distance)
                        )
                    step_contact_confirmed = contact_distance is not None and contact_distance <= contact_tolerance
                    ever_contact_confirmed = ever_contact_confirmed or step_contact_confirmed
                    step_records.append(
                        {
                            "step_index": len(step_records),
                            "phase": "post_release_clearance",
                            "distance_to_target": distance,
                            "distance_metric": distance_metric,
                            "xy_distance_to_target": xy_distance,
                            "object_to_eef_distance": contact_distance,
                            "object_to_eef_distance_before_clearance": contact_distance_before_clearance,
                            "post_release_retreat_min_distance": post_release_retreat_min_distance,
                            "clearance_reached": bool(
                                contact_distance is not None
                                and contact_distance >= post_release_retreat_min_distance
                            ),
                            "contact_confirmed": step_contact_confirmed,
                            "gripper_command": release_gripper_command,
                            "eef_target_position": _to_builtin(_vector_to_list(clearance_target)),
                            "base_action": _to_builtin(_vector_to_list(base_action)) if base_action is not None else None,
                            "base_mode_value": step_base_mode_value,
                            "action_summary": summarize_observation(bounded_action),
                            "used_control_observation_fallback": isinstance(control_obs, dict) and control_obs is not step_obs,
                        }
                    )
                    if contact_distance is not None and contact_distance >= post_release_retreat_min_distance:
                        break

        if skill_name == "place_robocasa_object_at":
            distance_after = _object_to_target_distance(
                obs,
                object_name=kwargs.get("object_name"),
                target_name=target_name,
                explicit_target_position=explicit_target_position,
                offset=offset,
            )
        else:
            distance_after = _distance(obs, target_name=target_name, explicit_position=explicit_target_position, offset=offset)
        if distance_after is not None:
            best_distance = min(best_distance, distance_after)
        avoidance_distance_after = avoidance_distance(obs)
        final_avoidance_reached = avoidance_reached(avoidance_distance_after)
        contact_confirmed = final_contact_distance is not None and final_contact_distance <= contact_tolerance
        release_stable = bool(
            release_executed
            and final_xy_distance is not None
            and release_xy_tolerance is not None
            and final_xy_distance <= release_xy_tolerance
            and not post_release_drift_detected
        )
        moved = distance_after is not None and distance_after < distance_before
        ok = bool(moved and step_records)
        if skill_name == "place_robocasa_object_at" and release_only_when_ready and not release_executed:
            ok = False
        if skill_name == "place_robocasa_object_at" and release_only_when_ready and release_executed and not release_stable:
            ok = False
        if skill_name == "grasp_robocasa_object":
            ok = bool(ok and contact_confirmed and runtime_action_schema.get("has_gripper_control"))
        if skill_name == "move_robocasa_ee_to" and min_distance_from_avoidance is not None:
            ok = bool(ok and final_avoidance_reached)
        filtered_info_keys = [key for key in sorted(str(key) for key in dict(latest_info or {})) if "success" not in key.lower()]
        motion_status = "distance_reduced" if moved else "no_distance_reduction"
        if skill_name == "move_robocasa_ee_to" and min_distance_from_avoidance is not None:
            motion_status = (
                "avoidance_clearance_reached"
                if final_avoidance_reached
                else "avoidance_clearance_incomplete"
            )
        if skill_name == "place_robocasa_object_at" and post_release_drift_detected:
            motion_status = "released_object_drifted_outside_xy_tolerance"
        if skill_name == "place_robocasa_object_at" and release_skipped_reason is not None and moved:
            motion_status = release_skipped_reason
        skill_error = None
        if skill_name == "grasp_robocasa_object" and grasp_preclose_failure is not None:
            skill_error = grasp_preclose_failure
        if skill_name == "grasp_robocasa_object" and moved and not runtime_action_schema.get("has_gripper_control"):
            motion_status = "gripper_control_unavailable"
            skill_error = "gripper_control_unavailable_for_grasp"
        motion_trace_limit = max(1, int(kwargs.get("motion_trace_limit", 5)))
        return PrimitiveResult(
            name=skill_name,
            ok=ok,
            output={
                **kwargs,
                **self._action_evidence(
                    skill_name,
                    kwargs,
                    target_name=target_name,
                    target_position=_to_builtin(target_with_offset.tolist()),
                ),
                "execution_status": "stepped" if step_records else "not_executed",
                "requires_motion_backend": False,
                "motion_backend": "robocasa_state_delta_motion",
                "motion_status": motion_status,
                "moved": bool(moved),
                "runtime_action_schema": runtime_action_schema,
                "arm_delta_frame": arm_delta_frame,
                "distance_metric": distance_metric,
                "avoidance_point": _to_builtin(_vector_to_list(avoidance_point))
                if skill_name == "move_robocasa_ee_to"
                else None,
                "min_distance_from_avoidance": min_distance_from_avoidance
                if skill_name == "move_robocasa_ee_to"
                else None,
                "avoidance_tolerance": avoidance_tolerance if skill_name == "move_robocasa_ee_to" else None,
                "stop_when_avoidance_reached": stop_when_avoidance_reached
                if skill_name == "move_robocasa_ee_to"
                else None,
                "avoidance_distance_before": avoidance_distance_before
                if skill_name == "move_robocasa_ee_to"
                else None,
                "avoidance_distance_after": avoidance_distance_after
                if skill_name == "move_robocasa_ee_to"
                else None,
                "avoidance_clearance_reached": final_avoidance_reached
                if skill_name == "move_robocasa_ee_to"
                else None,
                "transport_gripper_command": transport_gripper_command if skill_name == "place_robocasa_object_at" else None,
                "release_gripper_command": release_gripper_command if skill_name == "place_robocasa_object_at" else None,
                "release_only_when_ready": release_only_when_ready if skill_name == "place_robocasa_object_at" else None,
                "release_xy_tolerance": release_xy_tolerance if skill_name == "place_robocasa_object_at" else None,
                "release_requires_contact": release_requires_contact if skill_name == "place_robocasa_object_at" else None,
                "object_xy_early_stop_enabled": object_xy_early_stop_enabled
                if skill_name == "place_robocasa_object_at"
                else None,
                "object_xy_stop_when_within": object_xy_stop_when_within
                if skill_name == "place_robocasa_object_at"
                else None,
                "object_xy_stop_requires_contact": object_xy_stop_requires_contact
                if skill_name == "place_robocasa_object_at"
                else None,
                "xy_early_stop_triggered": xy_early_stop_triggered if skill_name == "place_robocasa_object_at" else None,
                "xy_early_stop_reason": xy_early_stop_reason if skill_name == "place_robocasa_object_at" else None,
                "xy_early_stop_step": xy_early_stop_step if skill_name == "place_robocasa_object_at" else None,
                "xy_early_stop_phase": xy_early_stop_phase if skill_name == "place_robocasa_object_at" else None,
                "release_ready": release_ready if skill_name == "place_robocasa_object_at" else None,
                "release_xy_ready": release_xy_ready if skill_name == "place_robocasa_object_at" else None,
                "release_contact_ready": release_contact_ready if skill_name == "place_robocasa_object_at" else None,
                "release_executed": release_executed if skill_name == "place_robocasa_object_at" else None,
                "release_stable": release_stable if skill_name == "place_robocasa_object_at" else None,
                "post_release_drift_detected": post_release_drift_detected
                if skill_name == "place_robocasa_object_at"
                else None,
                "release_skipped_reason": release_skipped_reason if skill_name == "place_robocasa_object_at" else None,
                "release_settle_steps": release_settle_steps if skill_name == "place_robocasa_object_at" else None,
                "post_release_retreat_offset": _to_builtin(_vector_to_list(post_release_retreat_offset))
                if skill_name == "place_robocasa_object_at"
                else None,
                "post_release_retreat_steps": post_release_retreat_steps if skill_name == "place_robocasa_object_at" else None,
                "post_release_retreat_min_distance": post_release_retreat_min_distance
                if skill_name == "place_robocasa_object_at"
                else None,
                "post_release_retreat_max_steps": post_release_retreat_max_steps
                if skill_name == "place_robocasa_object_at"
                else None,
                "post_release_clearance_reached": (
                    final_contact_distance is not None
                    and post_release_retreat_min_distance > 0.0
                    and final_contact_distance >= post_release_retreat_min_distance
                )
                if skill_name == "place_robocasa_object_at"
                else None,
                "tolerance": tolerance,
                "object_error_gain": object_error_gain if skill_name == "place_robocasa_object_at" else None,
                "object_error_clip": object_error_clip if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_steps": object_xy_push_steps if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_align_steps": object_xy_push_align_steps if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_backoff": object_xy_push_backoff if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_through": object_xy_push_through if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_z_offset": object_xy_push_z_offset if skill_name == "place_robocasa_object_at" else None,
                "object_xy_push_reacquire_from_side": object_xy_push_reacquire_from_side
                if skill_name == "place_robocasa_object_at"
                else None,
                "object_xy_contact_seek_steps": object_xy_contact_seek_steps
                if skill_name == "place_robocasa_object_at"
                else None,
                "object_xy_contact_seek_backoff": object_xy_contact_seek_backoff
                if skill_name == "place_robocasa_object_at"
                else None,
                "object_xy_contact_seek_z_offset": object_xy_contact_seek_z_offset
                if skill_name == "place_robocasa_object_at"
                else None,
                "contact_guard_enabled": contact_guard_enabled if skill_name == "place_robocasa_object_at" else None,
                "contact_guard_tolerance": contact_guard_tolerance if skill_name == "place_robocasa_object_at" else None,
                "contact_guard_recover_steps": contact_guard_recover_steps if skill_name == "place_robocasa_object_at" else None,
                "contact_guard_offset": _to_builtin(_vector_to_list(contact_guard_offset))
                if skill_name == "place_robocasa_object_at"
                else None,
                "placement_refinement_hint": _robocasa_place_refinement_hint(
                    obs,
                    object_name=kwargs.get("object_name"),
                    target_name=target_name,
                    explicit_target_position=explicit_target_position,
                    offset=offset,
                    best_xy_distance=best_xy_distance,
                    final_xy_distance=final_xy_distance,
                    release_xy_tolerance=release_xy_tolerance,
                    release_ready=release_ready,
                    release_xy_ready=release_xy_ready,
                    release_contact_ready=release_contact_ready,
                    release_executed=release_executed,
                    release_skipped_reason=release_skipped_reason,
                    contact_confirmed=contact_confirmed,
                    transport_contact_confirmed=transport_contact_confirmed,
                    ever_contact_confirmed=ever_contact_confirmed,
                    best_contact_distance=best_contact_distance,
                    final_contact_distance=final_contact_distance,
                    contact_tolerance=contact_tolerance,
                    best_step=_best_motion_record(step_records),
                    best_xy_step=_best_xy_motion_record(step_records),
                    object_error_gain=object_error_gain,
                    object_error_clip=object_error_clip,
                    object_xy_push_steps=object_xy_push_steps,
                    object_xy_push_align_steps=object_xy_push_align_steps,
                    object_xy_contact_seek_steps=object_xy_contact_seek_steps,
                    release_only_when_ready=release_only_when_ready,
                    object_xy_early_stop_enabled=object_xy_early_stop_enabled,
                    object_xy_stop_when_within=object_xy_stop_when_within,
                    xy_early_stop_triggered=xy_early_stop_triggered,
                    xy_early_stop_reason=xy_early_stop_reason,
                )
                if skill_name == "place_robocasa_object_at"
                else None,
                "target_name": target_name,
                "target_position": _to_builtin(target_with_offset.tolist()),
                "eef_target_position": _to_builtin(_vector_to_list(last_eef_target_position)),
                "place_object_relative_control": place_object_relative_control if skill_name == "place_robocasa_object_at" else None,
                "distance_before": distance_before,
                "distance_after": distance_after,
                "best_distance": best_distance,
                "xy_distance_after": final_xy_distance if skill_name == "place_robocasa_object_at" else None,
                "best_xy_distance": best_xy_distance if skill_name == "place_robocasa_object_at" else None,
                "contact_tolerance": contact_tolerance,
                "object_to_eef_distance_after": final_contact_distance,
                "best_object_to_eef_distance": best_contact_distance,
                "contact_confirmed": contact_confirmed,
                "ever_contact_confirmed": ever_contact_confirmed,
                "transport_contact_confirmed": transport_contact_confirmed if skill_name == "place_robocasa_object_at" else None,
                "place_transport_offset": _to_builtin(_vector_to_list(place_transport_offset)),
                "object_distance_before": object_distance_before,
                "object_distance_after": distance_after if skill_name == "place_robocasa_object_at" else None,
                "grasp_contact_steps": grasp_contact_steps if skill_name == "grasp_robocasa_object" else None,
                "grasp_contact_offset": _to_builtin(_vector_to_list(grasp_contact_offset))
                if skill_name == "grasp_robocasa_object"
                else None,
                "grasp_contact_tolerance": grasp_contact_tolerance if skill_name == "grasp_robocasa_object" else None,
                "force_grasp_contact_steps": force_grasp_contact_steps if skill_name == "grasp_robocasa_object" else None,
                "grasp_hold_steps": grasp_hold_steps if skill_name == "grasp_robocasa_object" else None,
                "grasp_lift_delta": grasp_lift_delta if skill_name == "grasp_robocasa_object" else None,
                "grasp_lift_evidence": grasp_lift_evidence if skill_name == "grasp_robocasa_object" else None,
                "grasp_preclose_failure": grasp_preclose_failure if skill_name == "grasp_robocasa_object" else None,
                "grasp_wrist_yaw_error_rad": grasp_wrist_yaw_error_rad if skill_name == "grasp_robocasa_object" else None,
                "mobile_base_enabled": mobile_base_enabled,
                "base_gain": base_gain if mobile_base_enabled else None,
                "base_max_delta": base_max_delta if mobile_base_enabled else None,
                "base_command_sign": base_command_sign if mobile_base_enabled else None,
                "base_mode_value": base_mode_value if mobile_base_enabled else None,
                "base_xy_deadband": base_xy_deadband if mobile_base_enabled else None,
                "motion_trace_limit": motion_trace_limit,
                "steps": len(step_records),
                "official_task_completion_claimed": False,
                "step_summary": {
                    "reward": _to_builtin(latest_reward),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info_keys": filtered_info_keys,
                    "observation_summary": summarize_observation(self._last_obs),
                    "control_observation_summary": summarize_observation(obs) if obs is not self._last_obs else None,
                    "final": {
                        "distance_to_target": distance_after,
                        "best_distance": best_distance,
                        "xy_distance_to_target": final_xy_distance if skill_name == "place_robocasa_object_at" else None,
                        "best_xy_distance": best_xy_distance if skill_name == "place_robocasa_object_at" else None,
                        "object_to_eef_distance": final_contact_distance,
                        "best_object_to_eef_distance": best_contact_distance,
                        "contact_confirmed": contact_confirmed,
                        "ever_contact_confirmed": ever_contact_confirmed,
                        "transport_contact_confirmed": transport_contact_confirmed if skill_name == "place_robocasa_object_at" else None,
                        "contact_tolerance": contact_tolerance,
                        "release_gate": {
                            "release_only_when_ready": release_only_when_ready,
                            "release_xy_tolerance": release_xy_tolerance,
                            "release_requires_contact": release_requires_contact,
                            "object_xy_early_stop_enabled": object_xy_early_stop_enabled,
                            "object_xy_stop_when_within": object_xy_stop_when_within,
                            "object_xy_stop_requires_contact": object_xy_stop_requires_contact,
                            "xy_early_stop_triggered": xy_early_stop_triggered,
                            "xy_early_stop_reason": xy_early_stop_reason,
                            "xy_early_stop_step": xy_early_stop_step,
                            "xy_early_stop_phase": xy_early_stop_phase,
                            "release_ready": release_ready,
                            "release_xy_ready": release_xy_ready,
                            "release_contact_ready": release_contact_ready,
                            "release_executed": release_executed,
                            "release_stable": release_stable,
                            "post_release_drift_detected": post_release_drift_detected,
                            "release_skipped_reason": release_skipped_reason,
                        }
                        if skill_name == "place_robocasa_object_at"
                        else None,
                    },
                    "phase_summary": _motion_phase_summary(step_records),
                    "best_step": _best_motion_record(step_records),
                    "best_xy_step": _best_xy_motion_record(step_records),
                    "motion_trace": step_records[-motion_trace_limit:],
                },
            },
            error=skill_error,
        )

    def _action_evidence(
        self,
        skill_name: str,
        kwargs: JsonDict,
        *,
        target_name: str | None = None,
        target_position: Any | None = None,
    ) -> JsonDict:
        object_name = kwargs.get("object_name")
        fixture_name = kwargs.get("fixture_name")
        target_name = target_name or kwargs.get("target_name") or object_name or fixture_name
        grounded: JsonDict = {}
        if object_name in self._objects:
            grounded["object"] = {"name": object_name, **deepcopy(self._objects[object_name])}
        if fixture_name in self._fixtures:
            grounded["fixture"] = {"name": fixture_name, **deepcopy(self._fixtures[fixture_name])}
        if target_name in self._objects:
            grounded["target"] = {"name": target_name, **deepcopy(self._objects[target_name])}
        elif target_name in self._fixtures:
            grounded["target"] = {"name": target_name, **deepcopy(self._fixtures[target_name])}
        elif target_position is not None:
            grounded["target"] = {"name": target_name, "pose_world": target_position, "source": "action_target_position"}
        return {
            "language_task": self._language_task(
                prompt=kwargs.get("prompt"),
                query=kwargs.get("query"),
                agent_context=kwargs.get("agent_context") if isinstance(kwargs.get("agent_context"), dict) else {},
            ),
            "grounded_action": {
                "skill": skill_name,
                "strategy": kwargs.get("strategy"),
                "target_name": target_name,
                "entities": grounded,
                "source": "robocasa_scene_registry_and_observation_state",
            },
            "pose_evidence": self._pose_evidence(target_name if isinstance(target_name, str) else None),
            "verifier_boundary": {
                "task_completion_claimed_by_primitive": False,
                "harness_verify_required": True,
                "private_task_signal_exposed": False,
            },
        }

    def _language_task(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> JsonDict:
        return {
            "instruction": self._task_spec.instruction if self._task_spec is not None else "",
            "env_id": self.config.env_id,
            "split": self.config.split,
            "benchmark_task_description": self._benchmark_task_description(),
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
        }

    def _registry_position(self, name: str) -> list[float] | None:
        for registry in (self._objects, self._fixtures):
            payload = registry.get(name)
            if not isinstance(payload, dict):
                continue
            for key in ("pose_world", "pos", "position"):
                vector = _coerce_vector_value(payload.get(key))
                if vector is not None:
                    return _to_builtin(vector.tolist())
        return None

    def _entity_position(self, name: str | None, payload: JsonDict | None, *, fallback_key: str | None = None) -> Any | None:
        if isinstance(payload, dict):
            for key in ("pose_world", "pos", "position"):
                vector = _coerce_vector_value(payload.get(key))
                if vector is not None:
                    return vector
        obs = _read_current_observation(self._env) if self._env is not None else self._last_obs
        if isinstance(obs, dict) and fallback_key:
            direct = _target_position(obs, target_name=fallback_key, explicit_position=None)
            if direct is not None:
                return direct
        if name is not None:
            registered = self._registry_position(name)
            if registered is not None:
                return _coerce_vector_value(registered)
        return None

    def _select_target_entity(
        self,
        target_name: str | None,
    ) -> tuple[str | None, dict[str, JsonDict]]:
        selected_fixture = _select_name(self._fixtures, target_name)
        if selected_fixture is not None:
            return selected_fixture, self._fixtures
        selected_object = _select_name(self._objects, target_name)
        if selected_object is not None:
            return selected_object, self._objects
        return None, {}

    def _pose_evidence(self, selected_name: str | None = None) -> JsonDict:
        objects = {
            name: payload.get("pose_world")
            for name, payload in sorted(self._objects.items())
            if payload.get("pose_world") is not None and (selected_name is None or name == selected_name)
        }
        fixtures = {
            name: payload.get("pose_world")
            for name, payload in sorted(self._fixtures.items())
            if payload.get("pose_world") is not None and (selected_name is None or name == selected_name)
        }
        return {"source": "robocasa_scene_registry", "objects": objects, "fixtures": fixtures}

    def _visual_evidence(self) -> JsonDict:
        cameras = extract_camera_summaries(self._last_obs)
        return {
            "observation_ref": self._observation_ref(source="robocasa_last_visual_observation"),
            "rgbd": _filter_camera_modalities(cameras, {"rgb", "depth"}),
            "segmentation": _filter_camera_modalities(cameras, {"segmentation"}),
            "visual_runtime": summarize_robocasa_visual_runtime(cameras),
            "segmentation_instances": self._robocasa_segmentation_instances(),
        }

    def _observation_ref(self, source: str) -> JsonDict:
        return {
            "source": source,
            "trace_event_count": len(self.get_trace().events),
            "runtime_live": self.config.live,
            "env_created": self._env is not None,
            "observation_type": type(self._last_obs).__name__,
        }

    def _robocasa_segmentation_instances(self, camera_name: str | None = None) -> dict[str, list[JsonDict]]:
        instances = extract_robocasa_segmentation_instances(self._last_obs, camera_name=camera_name)
        modes = _robocasa_segmentation_modes(self._last_obs)
        for uid, camera_instances in instances.items():
            bindings = _robocasa_native_segmentation_bindings(
                self._env,
                self._objects,
                self._fixtures,
                mode=modes.get(uid, "unknown"),
            )
            for instance in camera_instances:
                binding = bindings.get(int(instance["segmentation_id"]))
                if binding is not None:
                    instance.update(deepcopy(binding))
        return instances

    def _action_skill_schema(self, skill_name: str | None = None) -> JsonDict:
        schemas: dict[str, JsonDict] = {
            "open_robocasa_fixture": {"fixture_name": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
            "close_robocasa_fixture": {"fixture_name": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
            "press_robocasa_fixture_button": {
                "fixture_name": "str",
                "button_name": "str|None",
                "button_offset": "list[float]|None",
                "use_visual_button_anchor_position": "bool",
                "horizon": "int",
                "approach_steps": "int|None",
                "approach_distance": "float",
                "approach_stop_distance": "float|None",
                "approach_patience_steps": "int",
                "tolerance": "float",
                "press_depth": "float",
                "max_press_depth": "float|None",
                "press_contact_seek_steps": "int",
                "press_contact_seek_depth": "float|None",
                "press_direction_sign": "float",
                "press_direction_vector": "list[float]|None",
                "press_steps": "int",
                "hold_steps": "int",
                "retreat_steps": "int|None",
                "retreat_distance": "float",
                "gain": "float",
                "max_delta": "float",
                "mobile_base_enabled": "bool",
                "mobile_base_active_phases": "list[str]|None",
                "base_gain": "float",
                "base_max_delta": "float",
                "base_command_sign": "float",
                "base_mode_value": "float",
                "arm_delta_frame": "str|None",
                "gripper_command": "float",
                "motion_trace_limit": "int",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
            "refine_robocasa_button_contact_search": {
                "fixture_name": "str|None",
                "button_name": "str|None",
                "button_position": "list[float]|None",
                "evidence_handles": "list[str]|None",
                "visual_binding_tolerance": "float|None",
                "previous_action_output": "dict|None",
                "contact_frame_candidate": "dict|None",
                "move_to_previous_best": "bool",
                "prealign_steps": "int",
                "prealign_gain": "float",
                "prealign_max_delta": "float",
                "use_visual_button_anchor_position": "bool",
                "button_offset": "list[float]|None",
                "bind_gripper_contact_geometry": "bool",
                "gripper_contact_surface_axis": "list[float]|None",
                "horizon": "int",
                "approach_steps": "int|None",
                "approach_distance": "float",
                "approach_stop_distance": "float|None",
                "approach_patience_steps": "int",
                "tolerance": "float",
                "press_depth": "float",
                "max_press_depth": "float|None",
                "press_contact_seek_steps": "int",
                "press_contact_seek_depth": "float|None",
                "press_direction_sign": "float",
                "press_direction_vector": "list[float]|None",
                "press_steps": "int",
                "hold_steps": "int",
                "retreat_steps": "int|None",
                "retreat_distance": "float",
                "gain": "float",
                "max_delta": "float",
                "mobile_base_enabled": "bool",
                "mobile_base_active_phases": "list[str]|None",
                "base_gain": "float",
                "base_max_delta": "float",
                "base_command_sign": "float",
                "base_mode_value": "float",
                "base_delta_frame": "str",
                "base_xy_deadband": "float|None",
                "arm_delta_frame": "str|None",
                "gripper_command": "float",
                "motion_trace_limit": "int",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
            "move_robocasa_ee_to": {
                "target_name": "str|None",
                "target_position": "list[float]|None",
                "offset": "list[float]|None",
                "horizon": "int",
                "gain": "float",
                "max_delta": "float",
                "arm_delta_frame": "str|None",
                "tolerance": "float",
                "gripper_command": "float",
                "avoidance_point": "list[float]|None",
                "min_distance_from_avoidance": "float|None",
                "avoidance_tolerance": "float",
                "stop_when_avoidance_reached": "bool",
                "release_settle_steps": "int",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
            "apply_robocasa_control": {
                "eef_position_delta": "list[float]|None",
                "eef_rotation_delta": "list[float]|None",
                "base_delta": "list[float]|None",
                "gripper_command": "float",
                "repeat": "int",
                "arm_delta_frame": "str|None",
                "base_mode_value": "float",
                "object_name": "str|None",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
            "grasp_robocasa_object": {
                "object_name": "str",
                "offset": "list[float]|None",
                "grasp_approach": "center|near_side",
                "grasp_wrist_yaw_offset_rad": "float",
                "horizon": "int",
                "grasp_contact_steps": "int",
                "grasp_contact_offset": "list[float]|None",
                "grasp_contact_tolerance": "float|None",
                "force_grasp_contact_steps": "bool",
                "grasp_hold_steps": "int",
                "gripper_command": "float",
                "grasp_lift_delta": "float",
                "contact_tolerance": "float",
                "gain": "float",
                "max_delta": "float",
                "arm_delta_frame": "str|None",
                "mobile_base_enabled": "bool",
                "base_gain": "float",
                "base_max_delta": "float",
                "base_command_sign": "float",
                "base_mode_value": "float",
                "base_xy_deadband": "float",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
            "place_robocasa_object_at": {
                "object_name": "str",
                "target_name": "str|None",
                "target_position": "list[float]|None",
                "relation": "str",
                "offset": "list[float]|None",
                "use_affordance_site": "bool",
                "affordance_site_name": "str|None",
                "horizon": "int",
                "gripper_command": "float",
                "transport_gripper_command": "float|None",
                "release_gripper_command": "float",
                "release_only_when_ready": "bool",
                "release_xy_tolerance": "float|None",
                "release_requires_contact": "bool",
                "object_xy_early_stop_enabled": "bool",
                "object_xy_stop_when_within": "float|None",
                "object_xy_stop_requires_contact": "bool",
                "release_settle_steps": "int",
                "post_release_retreat_offset": "list[float]|None",
                "post_release_retreat_steps": "int",
                "post_release_retreat_min_distance": "float",
                "post_release_retreat_max_steps": "int",
                "tolerance": "float",
                "object_relative_control": "bool",
                "object_error_gain": "float",
                "object_error_clip": "float",
                "object_xy_push_steps": "int",
                "object_xy_push_align_steps": "int",
                "object_xy_push_backoff": "float",
                "object_xy_push_through": "float",
                "object_xy_push_z_offset": "float",
                "object_xy_push_reacquire_from_side": "bool",
                "object_xy_contact_seek_steps": "int",
                "object_xy_contact_seek_backoff": "float|None",
                "object_xy_contact_seek_z_offset": "float|None",
                "contact_guard_enabled": "bool",
                "contact_tolerance": "float",
                "contact_guard_tolerance": "float|None",
                "contact_guard_recover_steps": "int",
                "contact_guard_offset": "list[float]|None",
                "gain": "float",
                "max_delta": "float",
                "arm_delta_frame": "str|None",
                "mobile_base_enabled": "bool",
                "base_gain": "float",
                "base_max_delta": "float",
                "base_command_sign": "float",
                "base_mode_value": "float",
                "base_xy_deadband": "float",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            },
        }
        if self.config.motion_backend == "robocasa_state_delta_motion":
            schemas.pop("open_robocasa_fixture", None)
            schemas.pop("close_robocasa_fixture", None)
        selected = {skill_name: schemas[skill_name]} if skill_name in schemas else schemas
        return {
            "type": "runtime_action_skill_hooks",
            "motion_backend": self.config.motion_backend,
            "skills": selected,
            "runtime_action_schema": self._runtime_action_schema(),
            "private_task_signal_exposed": False,
        }

    def _runtime_action_schema(self) -> JsonDict:
        if self._env is None:
            return {
                "env_created": False,
                "available": False,
                "kind": "unavailable",
                "has_gripper_control": False,
                "has_base_control": False,
                "grasp_primitives_supported": False,
            }
        try:
            import numpy as np
        except Exception:
            return {
                "env_created": True,
                "available": False,
                "kind": "numpy_unavailable",
                "has_gripper_control": False,
                "has_base_control": False,
                "grasp_primitives_supported": False,
            }
        try:
            step_action, action_vector, _, _, action_key = _zero_action(self._env)
        except Exception as exc:
            return {
                "env_created": True,
                "available": False,
                "kind": "probe_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "has_gripper_control": False,
                "has_base_control": False,
                "grasp_primitives_supported": False,
            }
        if step_action is None or action_vector is None:
            return {
                "env_created": True,
                "available": False,
                "kind": "action_space_unavailable",
                "has_gripper_control": False,
                "has_base_control": False,
                "grasp_primitives_supported": False,
            }
        if isinstance(step_action, dict) and step_action.get(_DIRECT_ROBOCASA_ACTION_MARKER):
            template_action = np.asarray(step_action.get("template_action", []), dtype=np.float32).reshape(-1)
            has_gripper = step_action.get("gripper_slice") is not None
            has_base = step_action.get("base_slice") is not None
            return {
                "env_created": True,
                "available": True,
                "kind": "direct_composite",
                "action_shape": list(template_action.shape),
                "eef_part": step_action.get("eef_part"),
                "eef_slice": step_action.get("eef_slice"),
                "gripper_part": step_action.get("gripper_part"),
                "gripper_slice": step_action.get("gripper_slice"),
                "base_slice": step_action.get("base_slice"),
                "torso_slice": step_action.get("torso_slice"),
                "base_mode_index": step_action.get("base_mode_index"),
                "controller_input_ref_frame": step_action.get("controller_input_ref_frame"),
                "has_gripper_control": bool(has_gripper),
                "has_base_control": bool(has_base),
                "grasp_primitives_supported": bool(has_gripper),
            }
        if isinstance(step_action, dict):
            keys = sorted(str(key) for key in step_action)
            gripper_keys = [key for key in keys if "gripper" in key.lower()]
            base_keys = [key for key in keys if "base" in key.lower()]
            action_shape = list(np.asarray(action_vector, dtype=np.float32).reshape(-1).shape)
            return {
                "env_created": True,
                "available": True,
                "kind": "dict_action_space",
                "action_key": action_key,
                "action_shape": action_shape,
                "keys": keys,
                "gripper_keys": gripper_keys,
                "base_keys": base_keys,
                "has_gripper_control": bool(gripper_keys),
                "has_base_control": bool(base_keys),
                "grasp_primitives_supported": bool(gripper_keys),
            }
        flat = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        has_gripper = bool(flat.shape[0] in {4, 7} or flat.shape[0] > 7)
        return {
            "env_created": True,
            "available": True,
            "kind": "flat_action_space",
            "action_key": action_key,
            "action_shape": list(flat.shape),
            "has_gripper_control": has_gripper,
            "has_base_control": False,
            "grasp_primitives_supported": has_gripper,
            "gripper_interpretation": "last_channel" if has_gripper else "none_detected",
        }

    def _extract_task_success(self) -> tuple[bool, str]:
        for key in ("success", "is_success", "task_success"):
            if key in self._last_info:
                return bool(_to_builtin(self._last_info[key])), f"info.{key}"
        unwrapped = _unwrap_env(self._env)
        checker = getattr(unwrapped, "_check_success", None)
        if callable(checker):
            return bool(checker()), "env._check_success"
        return False, "unavailable"

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
            capability_tags=["w4", "robocasa", "agent_native_runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            cost={"primitive_calls": 1},
            failure_modes=["backend_not_configured", "wrong_arguments", "runtime_dependency_missing"],
            abstraction_level=level,
            leakage_risk=(
                "L2_privileged_state"
                if name
                in {
                    "observe_robocasa_kitchen_state",
                    "inspect_robocasa_object",
                    "locate_robocasa_object",
                    "inspect_robocasa_fixture",
                    "locate_robocasa_fixture",
                    "inspect_robocasa_affordance",
                }
                else "none"
            ),
            description=description,
        )

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def summarize_observation(obs: Any) -> JsonDict:
    if isinstance(obs, dict):
        return {str(key): summarize_observation(value) for key, value in obs.items()}
    shape = getattr(obs, "shape", None)
    dtype = getattr(obs, "dtype", None)
    if shape is not None:
        return {"type": type(obs).__name__, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(obs, (list, tuple)):
        return {"type": type(obs).__name__, "length": len(obs)}
    return {"type": type(obs).__name__, "value": _to_builtin(obs)}


def extract_camera_summaries(obs: Any, camera_name: str | None = None) -> dict[str, JsonDict]:
    if not isinstance(obs, dict):
        return {}
    cameras: dict[str, JsonDict] = {}
    sensor_data = obs.get("sensor_data", {})
    if isinstance(sensor_data, dict):
        for uid, payload in sensor_data.items():
            if camera_name is not None and uid != camera_name:
                continue
            if isinstance(payload, dict):
                cameras[str(uid)] = {str(key): summarize_observation(value) for key, value in payload.items()}
    for key, value in obs.items():
        if not isinstance(key, str):
            continue
        suffix = _camera_suffix(key)
        if suffix is None:
            continue
        uid = key[: -len(suffix)]
        if camera_name is not None and uid != camera_name:
            continue
        modality = (
            "rgb"
            if suffix == "_image"
            else "depth"
            if suffix == "_depth"
            else "segmentation"
        )
        cameras.setdefault(uid, {})[modality] = summarize_observation(value)
    return cameras


def _camera_suffix(key: str) -> str | None:
    for suffix in (
        "_segmentation_instance",
        "_segmentation_class",
        "_segmentation_element",
        "_segmentation",
        "_image",
        "_depth",
        "_seg",
    ):
        if key.endswith(suffix):
            return suffix
    return None


def _filter_camera_modalities(cameras: dict[str, JsonDict], modalities: set[str]) -> dict[str, JsonDict]:
    filtered: dict[str, JsonDict] = {}
    for uid, payload in cameras.items():
        selected = {name: value for name, value in payload.items() if name in modalities}
        if selected:
            filtered[uid] = selected
    return filtered


def summarize_robocasa_visual_runtime(cameras: dict[str, JsonDict]) -> JsonDict:
    modalities_by_camera = {uid: sorted(payload) for uid, payload in cameras.items()}
    return {
        "visual_ready": bool(cameras),
        "camera_names": sorted(cameras),
        "modalities_by_camera": modalities_by_camera,
        "has_rgb": any("rgb" in modalities for modalities in modalities_by_camera.values()),
        "has_depth": any("depth" in modalities for modalities in modalities_by_camera.values()),
        "has_segmentation": any("segmentation" in modalities for modalities in modalities_by_camera.values()),
    }


def extract_robocasa_segmentation_instances(
    obs: Any,
    camera_name: str | None = None,
    id_to_name: dict[int, str] | None = None,
    min_pixel_count: int = 1,
) -> dict[str, list[JsonDict]]:
    try:
        import numpy as np
    except Exception:
        return {}
    if not isinstance(obs, dict):
        return {}
    instances_by_camera: dict[str, list[JsonDict]] = {}
    modes = _robocasa_segmentation_modes(obs)
    segmentations: dict[str, Any] = {}
    sensor_data = obs.get("sensor_data", {})
    if isinstance(sensor_data, dict):
        for uid, payload in sensor_data.items():
            if camera_name is not None and uid != camera_name:
                continue
            if isinstance(payload, dict) and "segmentation" in payload:
                segmentations[str(uid)] = payload["segmentation"]
    for key, value in obs.items():
        if not isinstance(key, str):
            continue
        suffix = _camera_suffix(key)
        if suffix not in {
            "_segmentation_instance",
            "_segmentation_class",
            "_segmentation_element",
            "_segmentation",
            "_seg",
        }:
            continue
        uid = key[: -len(suffix)]
        if camera_name is not None and uid != camera_name:
            continue
        segmentations[uid] = value
    for uid, raw_segmentation in segmentations.items():
        try:
            segmentation = np.asarray(raw_segmentation)
        except Exception:
            continue
        if segmentation.ndim >= 3:
            segmentation = segmentation[..., 0]
        instances: list[JsonDict] = []
        for raw_id in np.unique(segmentation):
            seg_id = int(raw_id)
            if seg_id < 0 or (seg_id == 0 and modes.get(uid) != "element"):
                continue
            ys, xs = np.where(segmentation == seg_id)
            if xs.size < min_pixel_count:
                continue
            instances.append(
                {
                    "camera_name": uid,
                    "segmentation_id": seg_id,
                    "entity_name": (id_to_name or {}).get(seg_id),
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "center_xy": [float(xs.mean()), float(ys.mean())],
                    "pixel_count": int(xs.size),
                    "image_shape_hw": [int(segmentation.shape[0]), int(segmentation.shape[1])],
                }
            )
        instances.sort(key=lambda item: int(item["pixel_count"]), reverse=True)
        instances_by_camera[uid] = instances
    return instances_by_camera


def _robocasa_segmented_depth_statistics(
    obs: Any,
    camera_name: str,
    segmentation_id: int,
) -> JsonDict | None:
    try:
        import numpy as np
    except Exception:
        return None
    if not isinstance(obs, dict):
        return None
    depth = None
    segmentation = None
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict) and isinstance(sensor_data.get(camera_name), dict):
        payload = sensor_data[camera_name]
        depth = payload.get("depth")
        segmentation = payload.get("segmentation")
    if depth is None:
        depth = obs.get(f"{camera_name}_depth")
    if segmentation is None:
        for suffix in ("_segmentation_instance", "_segmentation_element", "_segmentation", "_seg"):
            key = f"{camera_name}{suffix}"
            if key in obs:
                segmentation = obs[key]
                break
    if depth is None or segmentation is None:
        return None
    depth_array = np.asarray(depth)
    segmentation_array = np.asarray(segmentation)
    if depth_array.ndim >= 3:
        depth_array = depth_array[..., 0]
    if segmentation_array.ndim >= 3:
        segmentation_array = segmentation_array[..., 0]
    if depth_array.shape != segmentation_array.shape:
        return None
    values = depth_array[segmentation_array == int(segmentation_id)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return {
        "sample_count": int(values.size),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
        "source": "upstream_depth_pixels_selected_by_upstream_segmentation",
    }


def _robocasa_segmented_world_point(
    obs: Any,
    camera_name: str,
    segmentation_id: int,
) -> list[float] | None:
    """Read an upstream organized world point cloud; never infer an oracle pose."""

    try:
        import numpy as np
    except Exception:
        return None
    if not isinstance(obs, dict):
        return None
    point_cloud = None
    segmentation = None
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict) and isinstance(sensor_data.get(camera_name), dict):
        payload = sensor_data[camera_name]
        point_cloud = payload.get("point_cloud")
        if point_cloud is None:
            point_cloud = payload.get("pointcloud")
        segmentation = payload.get("segmentation")
    if point_cloud is None:
        for suffix in ("_point_cloud", "_pointcloud", "_xyz"):
            key = f"{camera_name}{suffix}"
            if key in obs:
                point_cloud = obs[key]
                break
    if segmentation is None:
        for suffix in ("_segmentation_instance", "_segmentation_element", "_segmentation", "_seg"):
            key = f"{camera_name}{suffix}"
            if key in obs:
                segmentation = obs[key]
                break
    if point_cloud is None or segmentation is None:
        return None
    points = np.asarray(point_cloud)
    labels = np.asarray(segmentation)
    if labels.ndim >= 3:
        labels = labels[..., 0]
    if points.ndim != 3 or points.shape[-1] < 3 or points.shape[:2] != labels.shape:
        return None
    selected = points[..., :3][labels == int(segmentation_id)]
    selected = selected[np.all(np.isfinite(selected), axis=1)]
    if selected.size == 0:
        return None
    return [float(value) for value in np.median(selected, axis=0)]


def _visual_target_position(evidence: Any, target_name: Any) -> list[float] | None:
    if not isinstance(evidence, list):
        return None
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if target_name is not None and item.get("entity_name") != target_name:
            continue
        point = _coerce_vector_value(item.get("point_world"))
        if point is not None:
            return [float(value) for value in point[:3]]
    return None


def _robocasa_static_fixture_evidence_reuse_for_action(
    skill_name: str,
    kwargs: JsonDict,
    *,
    handles: list[str],
    stale_handles: list[str],
    grounding_handles: dict[str, JsonDict],
    observation_serial: int,
) -> JsonDict:
    """Allow static fixture visual evidence to survive robot-only prealignment."""

    if skill_name != "press_robocasa_fixture_button":
        return {"allowed": False, "reason": "skill_requires_fresh_visual_evidence"}
    fixture_name = kwargs.get("fixture_name")
    if not isinstance(fixture_name, str) or not fixture_name:
        return {"allowed": False, "reason": "fixture_name_required"}
    if _coerce_vector_list(kwargs.get("button_position")) is None:
        return {"allowed": False, "reason": "explicit_button_position_required"}
    requested_button = kwargs.get("button_name")
    stale_payloads = [grounding_handles.get(handle, {}) for handle in stale_handles]
    if not stale_payloads:
        return {"allowed": False, "reason": "no_stale_payloads"}
    for payload in stale_payloads:
        if payload.get("entity_name") != fixture_name:
            return {"allowed": False, "reason": "stale_entity_mismatch"}
        if payload.get("entity_kind") not in {"fixture", "fixture_button"}:
            return {"allowed": False, "reason": "stale_entity_not_static_fixture"}
        payload_button = payload.get("button_name")
        if requested_button is not None and payload_button is not None and payload_button != requested_button:
            return {"allowed": False, "reason": "stale_button_mismatch"}
    current_handles = sorted(set(handles).difference(stale_handles))
    return {
        "allowed": True,
        "reason": "static_fixture_visual_evidence_reused_after_robot_only_motion",
        "current_observation_serial": int(observation_serial),
        "stale_evidence_handles": list(stale_handles),
        "current_evidence_handles": current_handles,
        "requires_explicit_button_position": True,
        "allowed_entity_kinds": ["fixture", "fixture_button"],
    }


def _bind_robocasa_visual_action_target(
    skill_name: str,
    kwargs: JsonDict,
    evidence: list[JsonDict],
) -> str | None:
    """Bind action coordinates only from same-observation visual evidence."""

    if skill_name not in {
        "press_robocasa_fixture_button",
        "move_robocasa_ee_to",
        "grasp_robocasa_object",
        "place_robocasa_object_at",
    }:
        return None
    target_name = (
        kwargs.get("fixture_name")
        if skill_name == "press_robocasa_fixture_button"
        else kwargs.get("target_name") or kwargs.get("object_name")
    )
    if (
        skill_name == "move_robocasa_ee_to"
        and target_name is None
        and _coerce_vector_list(kwargs.get("target_position")) is not None
    ):
        kwargs["visual_target_source"] = "caller_explicit_target_position_with_visual_evidence_provenance"
        return None
    matching = [item for item in evidence if item.get("entity_name") == target_name]
    if skill_name == "press_robocasa_fixture_button":
        requested_button = kwargs.get("button_name")
        explicit_button_position = _coerce_vector_list(kwargs.get("button_position"))
        if requested_button is not None:
            exact_matches = [item for item in matching if item.get("button_name") == requested_button]
            if exact_matches:
                matching = exact_matches
                binding_mode = "exact_native_button_name"
            else:
                pose_matches = _pose_bound_button_visual_matches(
                    matching,
                    explicit_button_position,
                    tolerance=kwargs.get("visual_binding_tolerance"),
                )
                if pose_matches:
                    matching = pose_matches
                    binding_mode = "caller_pose_to_same_fixture_native_visual_anchor"
                else:
                    fixture_context_matches = _affordance_pose_backed_fixture_visual_matches(
                        matching,
                        explicit_button_position,
                    )
                    if not fixture_context_matches:
                        return "visual_button_binding_required"
                    matching = fixture_context_matches
                    binding_mode = "fixture_visual_evidence_plus_affordance_pose"
        else:
            named_matches = [item for item in matching if item.get("button_name") is not None]
            pose_matches = _pose_bound_button_visual_matches(
                matching,
                explicit_button_position,
                tolerance=kwargs.get("visual_binding_tolerance"),
            )
            if named_matches:
                matching = named_matches
                binding_mode = "visible_native_button_name"
            elif pose_matches:
                matching = pose_matches
                binding_mode = "caller_pose_to_same_fixture_native_visual_anchor"
            else:
                fixture_context_matches = _affordance_pose_backed_fixture_visual_matches(
                    matching,
                    explicit_button_position,
                )
                if fixture_context_matches:
                    matching = fixture_context_matches
                    binding_mode = "fixture_visual_evidence_plus_affordance_pose"
                else:
                    matching = []
                    binding_mode = None
        if not matching:
            return "visual_button_binding_required"
    points = [_vector_to_list(item.get("point_world")) for item in matching]
    points = [point for point in points if point is not None]
    if not points:
        return "visual_grounding_world_point_unavailable"
    point = points[0]
    if not (
        skill_name == "press_robocasa_fixture_button"
        and binding_mode == "fixture_visual_evidence_plus_affordance_pose"
    ):
        if any(not np.allclose(point, candidate, rtol=0.0, atol=1e-6) for candidate in points[1:]):
            return "visual_grounding_world_target_ambiguous"
    if skill_name == "press_robocasa_fixture_button":
        if binding_mode == "fixture_visual_evidence_plus_affordance_pose":
            if explicit_button_position is None:
                return "visual_button_binding_required"
            kwargs["button_position"] = explicit_button_position
            kwargs["requested_button_position"] = explicit_button_position
            kwargs["visual_button_anchor_position"] = point
        elif (
            binding_mode == "caller_pose_to_same_fixture_native_visual_anchor"
            and explicit_button_position is not None
            and not bool(kwargs.get("use_visual_button_anchor_position", False))
        ):
            kwargs["button_position"] = explicit_button_position
            kwargs["visual_button_anchor_position"] = point
        else:
            if explicit_button_position is not None:
                kwargs["requested_button_position"] = explicit_button_position
            kwargs["button_position"] = point
            kwargs["visual_button_anchor_position"] = point
        if kwargs.get("button_name") is None:
            kwargs["button_name"] = matching[0].get("button_name")
        kwargs["visual_button_binding_mode"] = binding_mode
    else:
        kwargs["target_position"] = point
    kwargs["visual_target_source"] = (
        "cached_visual_binding_refreshed_from_unchanged_public_pose"
        if any(item.get("auto_refreshed") for item in matching)
        else "same_observation_rgbd_state_pose_fallback"
        if any(item.get("fallback_mode") == "rgbd_state_pose_no_segmentation" for item in matching)
        else "same_observation_native_segmentation_geometry"
    )
    return None


def _affordance_pose_backed_fixture_visual_matches(
    evidence: list[JsonDict],
    button_position: list[float] | None,
) -> list[JsonDict]:
    """Return fixture visual evidence explicitly grounded against a caller affordance pose."""

    if button_position is None:
        return []
    try:
        target = np.asarray(button_position, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return []
    records: list[JsonDict] = []
    for item in evidence:
        if item.get("entity_kind") not in {None, "fixture"}:
            continue
        requested = _vector_to_list(item.get("requested_world_position"))
        if requested is None:
            continue
        if np.allclose(np.asarray(requested, dtype=np.float64)[:3], target[:3], rtol=0.0, atol=1e-6):
            records.append(item)
    return records


def _pose_bound_button_visual_matches(
    evidence: list[JsonDict],
    button_position: list[float] | None,
    *,
    tolerance: Any = None,
) -> list[JsonDict]:
    """Return same-fixture visual anchors close enough to a caller-supplied button pose."""

    if button_position is None:
        return []
    try:
        target = np.asarray(button_position, dtype=np.float64).reshape(-1)
    except Exception:
        return []
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return []
    threshold = 0.08 if tolerance is None else max(0.0, float(tolerance))
    records: list[tuple[float, JsonDict]] = []
    for item in evidence:
        point = _vector_to_list(item.get("point_world"))
        if point is None:
            continue
        distance = float(np.linalg.norm(np.asarray(point, dtype=np.float64)[:3] - target[:3]))
        if distance <= threshold:
            records.append((distance, item))
            continue
        requested = _vector_to_list(item.get("requested_world_position"))
        raw_distance = item.get("distance_to_requested_world_position")
        try:
            requested_distance = float(raw_distance) if raw_distance is not None else None
        except (TypeError, ValueError):
            requested_distance = None
        if (
            requested is not None
            and np.allclose(np.asarray(requested, dtype=np.float64)[:3], target[:3], rtol=0.0, atol=1e-6)
            and requested_distance is not None
            and requested_distance <= threshold
        ):
            records.append((requested_distance, item))
    records.sort(key=lambda record: record[0])
    return [item for _, item in records]


def _robocasa_segmentation_modes(obs: Any) -> dict[str, str]:
    if not isinstance(obs, dict):
        return {}
    modes: dict[str, str] = {}
    sensor_data = obs.get("sensor_data")
    if isinstance(sensor_data, dict):
        for uid, payload in sensor_data.items():
            if not isinstance(payload, dict) or "segmentation" not in payload:
                continue
            mode = payload.get("segmentation_type") or payload.get("segmentation_mode") or "unknown"
            modes[str(uid)] = str(mode).lower()
    for key in obs:
        if not isinstance(key, str):
            continue
        suffix = _camera_suffix(key)
        if suffix not in {
            "_segmentation_instance",
            "_segmentation_class",
            "_segmentation_element",
            "_segmentation",
            "_seg",
        }:
            continue
        uid = key[: -len(suffix)]
        modes[uid] = suffix.removeprefix("_segmentation_") if suffix.startswith("_segmentation_") else "unknown"
    return modes


def _robocasa_native_segmentation_bindings(
    env: Any,
    objects: dict[str, JsonDict],
    fixtures: dict[str, JsonDict],
    *,
    mode: str,
) -> dict[int, JsonDict]:
    """Map native RoboSuite mask IDs through public task/model geometry APIs."""

    unwrapped = _unwrap_env(env)
    task_model = getattr(unwrapped, "model", None)
    sim = getattr(unwrapped, "sim", None)
    sim_model = getattr(sim, "model", None)
    sim_data = getattr(sim, "data", None)
    bindings: dict[int, JsonDict] = {}

    instance_ids = getattr(task_model, "instances_to_ids", None)
    if mode in {"instance", "unknown"} and isinstance(instance_ids, dict):
        for segmentation_id, (native_name, native_ids) in enumerate(instance_ids.items(), start=1):
            entity = _native_instance_entity(str(native_name), objects, fixtures)
            if entity is None:
                continue
            kind, entity_name = entity
            geom_ids = native_ids.get("geom", []) if isinstance(native_ids, dict) else []
            site_ids = native_ids.get("site", []) if isinstance(native_ids, dict) else []
            position = _median_native_geom_position(sim_data, geom_ids)
            if position is None:
                position = _median_native_site_position(sim_data, site_ids)
            _merge_native_binding(
                bindings,
                segmentation_id,
                _native_visual_binding(
                    entity_name,
                    kind,
                    position,
                    native_instance_name=str(native_name),
                    geom_ids=geom_ids,
                    site_ids=site_ids,
                ),
            )

    geom_to_instance = getattr(task_model, "geom_ids_to_instances", None)
    if mode in {"element", "unknown"}:
        button_geoms = _native_fixture_button_geom_bindings(unwrapped, fixtures)
        for geom_id, button in button_geoms.items():
            segmentation_id = _robosuite_element_segmentation_id(geom_id)
            binding = _native_visual_binding(
                str(button["entity_name"]),
                str(button["entity_kind"]),
                _native_geom_position(sim_data, geom_id),
                native_instance_name=_native_geom_name(sim_model, geom_id),
                geom_ids=[geom_id],
                geom_name=_native_geom_name(sim_model, geom_id),
            )
            binding.update(button)
            _merge_native_binding(bindings, segmentation_id, binding)
        for raw_geom_id, native_name in (geom_to_instance.items() if isinstance(geom_to_instance, dict) else ()):
            try:
                geom_id = int(raw_geom_id)
            except (TypeError, ValueError):
                continue
            segmentation_id = _robosuite_element_segmentation_id(geom_id)
            position = _native_geom_position(sim_data, geom_id)
            button = button_geoms.get(geom_id)
            if button is not None:
                binding = _native_visual_binding(
                    str(button["entity_name"]),
                    str(button["entity_kind"]),
                    position,
                    native_instance_name=str(native_name),
                    geom_ids=[geom_id],
                    geom_name=_native_geom_name(sim_model, geom_id),
                )
                binding.update(button)
                _merge_native_binding(bindings, segmentation_id, binding)
                continue
            entity = _native_instance_entity(str(native_name), objects, fixtures)
            if entity is None:
                continue
            kind, entity_name = entity
            binding = _native_visual_binding(
                entity_name,
                kind,
                position,
                native_instance_name=str(native_name),
                geom_ids=[geom_id],
                geom_name=_native_geom_name(sim_model, geom_id),
            )
            _merge_native_binding(bindings, segmentation_id, binding)

    if mode == "unknown":
        for registry, kind in ((objects, "object"), (fixtures, "fixture")):
            for entity_name, payload in registry.items():
                try:
                    body_id = int(payload.get("body_id"))
                except (TypeError, ValueError):
                    continue
                position = _vector_to_list(payload.get("pose_world"))
                _merge_native_binding(
                    bindings,
                    body_id,
                    _native_visual_binding(entity_name, kind, position, body_id=body_id),
                )
    return {seg_id: binding for seg_id, binding in bindings.items() if not binding.get("ambiguous")}


def _native_instance_entity(
    native_name: str,
    objects: dict[str, JsonDict],
    fixtures: dict[str, JsonDict],
) -> tuple[str, str] | None:
    for registry, kind in ((objects, "object"), (fixtures, "fixture")):
        if native_name in registry:
            return kind, native_name
        matches = [
            name
            for name, payload in registry.items()
            if native_name in {str(payload.get("name")), str(payload.get("id"))}
        ]
        if len(matches) == 1:
            return kind, matches[0]
    return None


def _robosuite_element_segmentation_id(geom_id: int) -> int:
    """Convert a MuJoCo geom id to RoboSuite's raw element-mask label.

    RoboSuite reads the renderer's object-id channel directly for element masks;
    MuJoCo reserves zero for background, so visible geom labels are one-based.
    """

    return int(geom_id) + 1


def _native_visual_binding(
    entity_name: str,
    entity_kind: str,
    position: Any,
    **geometry: Any,
) -> JsonDict:
    return {
        "entity_name": entity_name,
        "entity_kind": entity_kind,
        "native_world_position": _vector_to_list(position),
        "native_geometry": {key: _to_builtin(value) for key, value in geometry.items() if value is not None},
        "binding_source": "upstream_robosuite_model_id_mappings",
    }


def _merge_native_binding(bindings: dict[int, JsonDict], segmentation_id: int, candidate: JsonDict) -> None:
    existing = bindings.get(segmentation_id)
    if existing is None:
        bindings[segmentation_id] = candidate
        return
    existing_identity = (existing.get("entity_name"), existing.get("button_name"))
    candidate_identity = (candidate.get("entity_name"), candidate.get("button_name"))
    if existing_identity != candidate_identity:
        bindings[segmentation_id] = {"ambiguous": True}
    elif existing.get("native_world_position") is None and candidate.get("native_world_position") is not None:
        bindings[segmentation_id] = candidate


def _median_native_geom_position(data: Any, geom_ids: Any) -> list[float] | None:
    positions = [_native_geom_position(data, geom_id) for geom_id in geom_ids or []]
    positions = [position for position in positions if position is not None]
    if not positions:
        return None
    try:
        import numpy as np

        return [float(value) for value in np.median(np.asarray(positions, dtype=np.float64), axis=0)]
    except Exception:
        return None


def _median_native_site_position(data: Any, site_ids: Any) -> list[float] | None:
    positions: list[list[float]] = []
    for site_id in site_ids or []:
        try:
            position = _vector_to_list(data.site_xpos[int(site_id)])
        except Exception:
            position = None
        if position is not None:
            positions.append(position)
    if not positions:
        return None
    try:
        import numpy as np

        return [float(value) for value in np.median(np.asarray(positions, dtype=np.float64), axis=0)]
    except Exception:
        return None


def _native_geom_position(data: Any, geom_id: Any) -> list[float] | None:
    try:
        return _vector_to_list(data.geom_xpos[int(geom_id)])
    except Exception:
        return None


def _native_geom_name(model: Any, geom_id: int) -> str | None:
    try:
        return str(model.geom_id2name(geom_id))
    except Exception:
        return None


def _native_fixture_button_geom_bindings(env: Any, fixtures: dict[str, JsonDict]) -> dict[int, JsonDict]:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    if model is None:
        return {}
    result: dict[int, JsonDict] = {}
    for fixture_name, payload in fixtures.items():
        affordance_geometries = payload.get("affordance_geometries") if isinstance(payload, dict) else None
        start_buttons = (
            affordance_geometries.get("start_buttons")
            if isinstance(affordance_geometries, dict)
            else None
        )
        if isinstance(start_buttons, dict):
            for button_name, geometries in start_buttons.items():
                if not isinstance(geometries, list):
                    continue
                for geometry in geometries:
                    if not isinstance(geometry, dict):
                        continue
                    try:
                        geom_id = int(geometry["geom_id"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    result[geom_id] = {
                        "entity_name": fixture_name,
                        "entity_kind": "fixture_button",
                        "button_name": str(button_name),
                        "native_world_position": _vector_to_list(geometry.get("position_world")),
                        "binding_source": "upstream_robocasa_fixture_button_geometry",
                    }
            if start_buttons:
                continue
        fixture = _lookup_env_fixture(env, fixture_name)
        button_names = getattr(fixture, "_start_button_names", None)
        if not isinstance(button_names, (list, tuple)):
            continue
        for button_name in button_names:
            resolved_ids: list[int] = []
            site_positions: list[list[float]] = []
            for geom_name in _fixture_button_geom_candidates(fixture, str(button_name)):
                try:
                    geom_id = int(model.geom_name2id(geom_name))
                    if geom_id >= 0:
                        resolved_ids.append(geom_id)
                except Exception:
                    pass
                try:
                    site_id = int(model.site_name2id(geom_name))
                    if site_id < 0:
                        continue
                    site_position = _vector_to_list(sim.data.site_xpos[site_id])
                    if site_position is not None:
                        site_positions.append(site_position)
                except Exception:
                    pass
            for resolved_id in sorted(set(resolved_ids)):
                result[resolved_id] = {
                    "entity_name": fixture_name,
                    "entity_kind": "fixture_button",
                    "button_name": str(button_name),
                    **({"native_world_position": site_positions[0]} if len(site_positions) == 1 else {}),
                    "binding_source": "upstream_robocasa_fixture_button_geometry",
                }
    return result


def _extract_objects(env: Any) -> dict[str, JsonDict]:
    objects: dict[str, JsonDict] = {}
    body_ids = getattr(env, "obj_body_id", {}) or {}
    for name, body_id in dict(body_ids).items():
        objects[str(name)] = {"kind": "object", "body_id": _to_builtin(body_id), "pose_world": _body_pose(env, body_id)}
    for attr in ("objects", "objs"):
        raw = getattr(env, attr, None)
        if isinstance(raw, dict):
            for name, obj in raw.items():
                objects.setdefault(str(name), {"kind": "object"})
                objects[str(name)].update(_entity_payload(obj))
    return objects


def _unwrap_fixture_reference(value: Any) -> Any:
    # Newer RoboCasa stores fixture_refs[name] = (fixture, full_depth_region).
    # The public native register_fixture_ref returns the first element as well.
    # Preserve old releases whose mapping contains the fixture directly.
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], bool):
        return value[0]
    return value


def _extract_fixtures(env: Any) -> dict[str, JsonDict]:
    fixtures: dict[str, JsonDict] = {}
    fixture_names_by_identity: dict[int, list[str]] = {}
    for attr in ("fixture_refs", "fixtures"):
        raw = getattr(env, attr, None)
        if isinstance(raw, dict):
            for name, fixture in raw.items():
                fixture = _unwrap_fixture_reference(fixture)
                payload = {"kind": "fixture", **_entity_payload(fixture)}
                payload.setdefault("fixture_type", type(fixture).__name__)
                root_body = getattr(fixture, "root_body", None)
                sim_model = getattr(getattr(env, "sim", None), "model", None)
                if isinstance(root_body, str) and sim_model is not None:
                    try:
                        body_id = int(sim_model.body_name2id(root_body))
                    except Exception:
                        body_id = None
                    if body_id is not None:
                        payload["body_id"] = body_id
                        body_pose = _body_pose(env, body_id)
                        if body_pose is not None:
                            payload["pose_world"] = body_pose
                affordance_sites = _fixture_affordance_sites(env, fixture)
                if affordance_sites:
                    payload["affordance_sites"] = affordance_sites
                affordance_geometries = _fixture_affordance_geometries(env, fixture)
                if affordance_geometries:
                    payload["affordance_geometries"] = affordance_geometries
                fixtures[str(name)] = payload
                fixture_names_by_identity.setdefault(id(fixture), []).append(str(name))
    for alias, fixture in vars(env).items():
        fixture_names = fixture_names_by_identity.get(id(fixture))
        if not fixture_names or alias.startswith("_"):
            continue
        normalized_alias = _normalize_text(alias)
        if not normalized_alias or normalized_alias in {"fixture", "fixtures", "fixture refs"}:
            continue
        for fixture_name in fixture_names:
            aliases = fixtures[fixture_name].setdefault("agent_aliases", [])
            if normalized_alias not in aliases:
                aliases.append(normalized_alias)
    return fixtures


def _fixture_affordance_sites(env: Any, fixture: Any) -> JsonDict:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    prefix = str(getattr(fixture, "naming_prefix", ""))
    sites: JsonDict = {}
    receptacle_site = getattr(fixture, "_receptacle_pouring_site", None)
    if receptacle_site is not None:
        site_name = f"{prefix}receptacle_place_site"
        site_position = _sim_site_position(model, data, site_name)
        if site_position is not None:
            sites["receptacle_place_site"] = site_position
    button_names = _native_fixture_button_names(model, fixture)
    if button_names:
        buttons: JsonDict = {}
        for button_name in button_names:
            geom_position = None
            for geom_name in _fixture_button_geom_candidates(fixture, str(button_name)):
                geom_position = _sim_geom_position(model, data, geom_name)
                if geom_position is not None:
                    break
            if geom_position is not None:
                buttons[str(button_name)] = geom_position
        if buttons:
            sites["start_buttons"] = buttons
    return sites


def _fixture_affordance_geometries(env: Any, fixture: Any) -> JsonDict:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    button_names = _native_fixture_button_names(model, fixture)
    buttons: JsonDict = {}
    for button_name in button_names:
        geometries: list[JsonDict] = []
        seen: set[int] = set()
        for geom_name in _fixture_button_geom_candidates(fixture, str(button_name)):
            try:
                geom_id = int(model.geom_name2id(geom_name))
            except Exception:
                continue
            if geom_id < 0 or geom_id in seen:
                continue
            seen.add(geom_id)
            geometries.append(
                {
                    "geom_id": geom_id,
                    "geom_name": _native_geom_name(model, geom_id) or geom_name,
                    "position_world": _native_geom_position(data, geom_id),
                }
            )
        if geometries:
            buttons[str(button_name)] = geometries
    output = {"start_buttons": buttons} if buttons else {}
    # Same L2 geometry boundary as buttons: live fixture-owned handles only.
    # No task label, success threshold, desired joint value or motion is exposed.
    prefix = str(getattr(fixture, "naming_prefix", ""))
    if not prefix:
        name = str(getattr(fixture, "name", ""))
        prefix = name + "_" if name else ""
    handles = []
    if prefix:
        for geom_id in range(int(getattr(model, "ngeom", 0))):
            geom_name = _native_geom_name(model, geom_id)
            if not geom_name or not geom_name.startswith(prefix):
                continue
            if "handle" not in geom_name[len(prefix):].lower():
                continue
            handles.append({"geom_id": geom_id, "geom_name": geom_name,
                            "position_world": _native_geom_position(data, geom_id)})
    if handles:
        output["handles"] = handles
    return output


def _native_fixture_button_names(model: Any, fixture: Any) -> list[str]:
    """Discover fixture-owned button geoms without requiring fixture profiles."""

    discovered = {
        str(name)
        for name in (getattr(fixture, "_start_button_names", None) or ())
        if str(name)
    }
    prefix = str(getattr(fixture, "naming_prefix", ""))
    fixture_name = str(getattr(fixture, "name", ""))
    prefixes = [candidate for candidate in (prefix, f"{fixture_name}_" if fixture_name else "") if candidate]
    try:
        geom_count = int(model.ngeom)
    except Exception:
        geom_count = 0
    for geom_id in range(geom_count):
        geom_name = _native_geom_name(model, geom_id)
        if not geom_name:
            continue
        suffix = next((geom_name[len(owner) :] for owner in prefixes if geom_name.startswith(owner)), None)
        if not suffix:
            continue
        if suffix == "button" or suffix.endswith("_button") or "_button_" in suffix:
            discovered.add(suffix)
    return sorted(discovered)


def _fixture_button_geom_candidates(fixture: Any, button_name: str) -> list[str]:
    prefix = str(getattr(fixture, "naming_prefix", ""))
    fixture_name = str(getattr(fixture, "name", ""))
    candidates = [
        f"{prefix}{button_name}",
        f"{fixture_name}_{button_name}" if fixture_name else "",
        button_name,
        f"{prefix}{button_name}_main",
        f"{fixture_name}_{button_name}_main" if fixture_name else "",
    ]
    seen: set[str] = set()
    unique: list[str] = []
    for candidate in candidates:
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _fixture_button_names(fixture_payload: JsonDict | None) -> list[str]:
    if not isinstance(fixture_payload, dict):
        return []
    affordance_sites = fixture_payload.get("affordance_sites")
    start_buttons = affordance_sites.get("start_buttons") if isinstance(affordance_sites, dict) else None
    if not isinstance(start_buttons, dict):
        return []
    return sorted(str(name) for name in start_buttons)


def _verifier_boundary() -> JsonDict:
    return {
        "task_completion_claimed_by_primitive": False,
        "private_task_signal_exposed": False,
        "harness_verify_required": True,
    }


def _select_fixture_button_affordance(
    fixture_payload: JsonDict | None,
    *,
    button_name: Any | None,
) -> tuple[str, Any] | None:
    if not isinstance(fixture_payload, dict):
        return None
    affordance_sites = fixture_payload.get("affordance_sites")
    start_buttons = affordance_sites.get("start_buttons") if isinstance(affordance_sites, dict) else None
    if not isinstance(start_buttons, dict) or not start_buttons:
        return None
    explicit = str(button_name) if button_name is not None else None
    if explicit is None or explicit not in start_buttons:
        return None
    return explicit, start_buttons[explicit]


def _lookup_env_fixture(env: Any, fixture_name: str | None) -> Any | None:
    if env is None or not fixture_name:
        return None
    unwrapped = getattr(env, "unwrapped", env)
    for attr in ("fixture_refs", "fixtures"):
        raw = getattr(unwrapped, attr, None)
        if isinstance(raw, dict):
            if fixture_name in raw:
                return _unwrap_fixture_reference(raw[fixture_name])
            for fixture in raw.values():
                fixture = _unwrap_fixture_reference(fixture)
                if any(str(getattr(fixture, key, "")) == fixture_name for key in ("id", "name")):
                    return fixture
    return None


def _lookup_env_object(env: Any, object_name: str | None) -> Any | None:
    if env is None or not object_name:
        return None
    for candidate in reversed(_env_candidates(env)):
        for attr in ("objects", "objs"):
            raw = getattr(candidate, attr, None)
            if not isinstance(raw, dict):
                continue
            if object_name in raw:
                return raw[object_name]
            for name, obj in raw.items():
                if str(name) == object_name or any(str(getattr(obj, key, "")) == object_name for key in ("id", "name")):
                    return obj
    return None


def _env_gripper_object_contact(env: Any, object_name: str | None) -> bool | None:
    if env is None or not object_name:
        return None
    obj = _lookup_env_object(env, object_name)
    if obj is None:
        return None
    for candidate in reversed(_env_candidates(env)):
        checker = getattr(candidate, "check_contact", None)
        robots = getattr(candidate, "robots", None)
        if not callable(checker) or not robots:
            continue
        try:
            grippers = getattr(robots[0], "gripper", None)
            if isinstance(grippers, dict):
                gripper = grippers.get("right") or next(iter(grippers.values()), None)
            else:
                gripper = grippers
            if gripper is None:
                continue
            return bool(checker(gripper, obj))
        except Exception:
            continue
    return None


def _infer_eef_target_offset_for_gripper_contact(
    env: Any,
    *,
    surface_axis: Any = None,
) -> list[float] | None:
    try:
        unwrapped = getattr(env, "unwrapped", env)
        robot = list(unwrapped.robots)[0]
        gripper = robot.gripper["right"]
        important_geoms = gripper.important_geoms
        pad_names = [
            str(name)
            for key in ("left_fingerpad", "right_fingerpad")
            for name in important_geoms.get(key, [])
        ]
        axis = np.asarray(surface_axis, dtype=np.float32).reshape(-1)
        if axis.shape[0] != 3 or float(np.linalg.norm(axis)) <= 1e-8:
            return None
        axis = axis / float(np.linalg.norm(axis))
        support_points = []
        for name in pad_names:
            geom_id = unwrapped.sim.model.geom_name2id(name)
            center = np.asarray(unwrapped.sim.data.geom_xpos[geom_id], dtype=np.float32)
            size = np.asarray(unwrapped.sim.model.geom_size[geom_id], dtype=np.float32).reshape(-1)
            rotation = np.asarray(unwrapped.sim.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
            local_axis = rotation.T @ axis
            support_radius = float(np.sum(np.abs(local_axis[: min(3, size.shape[0])]) * size[:3]))
            support_points.append(center + axis * support_radius)
        eef_position = np.asarray(unwrapped.sim.data.site_xpos[robot.eef_site_id["right"]], dtype=np.float32)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    if not support_points:
        return None
    contact_surface_center = np.mean(np.stack(support_points), axis=0)
    return [float(value) for value in eef_position - contact_surface_center]


def _robocasa_button_contact_frame_candidates(
    env: Any,
    *,
    fixture_position: Any,
    button_position: Any,
    eef_position: Any,
    candidate_limit: int = 8,
) -> list[JsonDict]:
    button = _coerce_vector_value(button_position)
    if button is None:
        return []
    fixture = _coerce_vector_value(fixture_position)
    eef = _coerce_vector_value(eef_position)
    candidates: list[JsonDict] = []

    outward = None
    if fixture is not None:
        outward = np.asarray(button, dtype=np.float32) - np.asarray(fixture, dtype=np.float32)
        outward[2] = 0.0
        if float(np.linalg.norm(outward)) <= 1e-5:
            outward = None
    if outward is None and eef is not None:
        outward = np.asarray(button, dtype=np.float32) - np.asarray(eef, dtype=np.float32)
        outward[2] = 0.0
        if float(np.linalg.norm(outward)) <= 1e-5:
            outward = None
    if outward is not None:
        outward = outward / float(np.linalg.norm(outward))
        press_axis = -outward
        candidates.append(
            _robocasa_button_contact_candidate(
                env,
                contact_frame_id="button_fixture_normal_press",
                axis_source="fixture_button_vector",
                press_direction_vector=press_axis,
                gripper_contact_surface_axis=press_axis,
                button_position=button,
                eef_position=eef,
            )
        )
        def append_z_lift_candidate(lift_index: int, lift_meters: float) -> None:
            candidates.append(
                _robocasa_button_contact_candidate(
                    env,
                    contact_frame_id=f"button_fixture_normal_press_z_lift_{lift_index}",
                    axis_source="fixture_button_vector_plus_public_anchor_micro_sweep",
                    press_direction_vector=press_axis,
                    gripper_contact_surface_axis=press_axis,
                    button_position=button,
                    eef_position=eef,
                    button_offset_adjustment=[0.0, 0.0, lift_meters],
                    extra={"contact_frame_perturbation": {"type": "z_lift", "meters": lift_meters}},
                )
            )

        append_z_lift_candidate(1, 0.02)
        append_z_lift_candidate(2, 0.04)
        tangent = np.asarray([-outward[1], outward[0], 0.0], dtype=np.float32)
        if float(np.linalg.norm(tangent)) > 1e-5:
            tangent = tangent / float(np.linalg.norm(tangent))
            for lift_meters in (0.04, 0.02):
                for tangent_sign, tangent_label in ((-1.0, "tangent_neg"), (1.0, "tangent_pos")):
                    tangent_meters = 0.015 * tangent_sign
                    adjustment = tangent * tangent_meters
                    candidates.append(
                        _robocasa_button_contact_candidate(
                            env,
                            contact_frame_id=f"button_fixture_normal_press_z_lift_{int(lift_meters * 100):02d}_{tangent_label}",
                            axis_source="fixture_button_vector_plus_public_anchor_tangent_micro_sweep",
                            press_direction_vector=press_axis,
                            gripper_contact_surface_axis=press_axis,
                            button_position=button,
                            eef_position=eef,
                            button_offset_adjustment=[0.0, 0.0, lift_meters],
                            button_anchor_adjustment=[
                                float(adjustment[0]),
                                float(adjustment[1]),
                                0.0,
                            ],
                            extra={
                                "contact_frame_perturbation": {
                                    "type": "z_lift_tangent",
                                    "z_meters": lift_meters,
                                    "tangent_meters": tangent_meters,
                                    "tangent_axis": _to_builtin(_vector_to_list(tangent)),
                                }
                            },
                        )
                    )
                    if lift_meters == 0.04 and tangent_label == "tangent_neg":
                        for normal_sign, normal_label in ((1.0, "normal_in"), (-1.0, "normal_out")):
                            normal_meters = 0.008 * normal_sign
                            normal_adjustment = adjustment + press_axis * normal_meters
                            candidates.append(
                                _robocasa_button_contact_candidate(
                                    env,
                                    contact_frame_id=(
                                        "button_fixture_normal_press_z_lift_04_"
                                        f"{tangent_label}_{normal_label}"
                                    ),
                                    axis_source="fixture_button_vector_plus_public_anchor_tangent_normal_micro_sweep",
                                    press_direction_vector=press_axis,
                                    gripper_contact_surface_axis=press_axis,
                                    button_position=button,
                                    eef_position=eef,
                                    button_offset_adjustment=[0.0, 0.0, lift_meters],
                                    button_anchor_adjustment=[
                                        float(normal_adjustment[0]),
                                        float(normal_adjustment[1]),
                                        0.0,
                                    ],
                                    extra={
                                        "contact_frame_perturbation": {
                                            "type": "z_lift_tangent_normal",
                                            "z_meters": lift_meters,
                                            "tangent_meters": tangent_meters,
                                            "normal_meters": normal_meters,
                                            "tangent_axis": _to_builtin(_vector_to_list(tangent)),
                                            "press_axis": _to_builtin(_vector_to_list(press_axis)),
                                        }
                                    },
                                )
                            )
        append_z_lift_candidate(3, -0.02)
        candidates.append(
            _robocasa_button_contact_candidate(
                env,
                contact_frame_id="button_fixture_normal_reverse",
                axis_source="fixture_button_vector",
                press_direction_vector=outward,
                gripper_contact_surface_axis=outward,
                button_position=button,
                eef_position=eef,
            )
        )

    if eef is not None:
        approach_axis = np.asarray(button, dtype=np.float32) - np.asarray(eef, dtype=np.float32)
        if float(np.linalg.norm(approach_axis)) > 1e-5:
            approach_axis = approach_axis / float(np.linalg.norm(approach_axis))
            candidates.append(
                _robocasa_button_contact_candidate(
                    env,
                    contact_frame_id="eef_to_button_line",
                    axis_source="current_eef_and_button_vector",
                    press_direction_vector=approach_axis,
                    gripper_contact_surface_axis=approach_axis,
                    button_position=button,
                    eef_position=eef,
                )
            )

    default_press_axis = _coerce_vector_value(candidates[0].get("press_direction_vector")) if candidates else None
    for index, axis_record in enumerate(_gripper_contact_surface_axis_candidates(env)):
        axis = _coerce_vector_value(axis_record.get("axis"))
        if axis is None:
            continue
        candidates.append(
            _robocasa_button_contact_candidate(
                env,
                contact_frame_id=f"gripper_surface_axis_{index}",
                axis_source=str(axis_record.get("source") or "robot_gripper_geometry"),
                press_direction_vector=default_press_axis if default_press_axis is not None else axis,
                gripper_contact_surface_axis=axis,
                button_position=button,
                eef_position=eef,
                extra={"gripper_axis_record": axis_record},
            )
        )

    deduped: list[JsonDict] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in candidates:
        key = (
            candidate.get("contact_frame_id"),
            _rounded_vector_key(candidate.get("button_offset")),
            _rounded_vector_key(candidate.get("press_direction_vector")),
            _rounded_vector_key(candidate.get("gripper_contact_surface_axis")),
        )
        if key in seen:
            continue
        seen.add(key)
        candidate["candidate_index"] = len(deduped)
        deduped.append(candidate)
        if len(deduped) >= max(1, int(candidate_limit)):
            break
    return deduped


def _robocasa_button_contact_candidate(
    env: Any,
    *,
    contact_frame_id: str,
    axis_source: str,
    press_direction_vector: Any,
    gripper_contact_surface_axis: Any,
    button_position: Any,
    eef_position: Any,
    button_offset_adjustment: Any = None,
    button_anchor_adjustment: Any = None,
    extra: JsonDict | None = None,
) -> JsonDict:
    press_axis = _normalize_vector_or_none(press_direction_vector)
    contact_axis = _normalize_vector_or_none(gripper_contact_surface_axis)
    offset = (
        _infer_eef_target_offset_for_gripper_contact(env, surface_axis=contact_axis)
        if contact_axis is not None
        else None
    )
    offset_adjustment = _coerce_vector_value(button_offset_adjustment)
    if offset_adjustment is not None:
        offset = offset_adjustment if offset is None else np.asarray(offset, dtype=np.float32) + offset_adjustment
    anchor_adjustment = _coerce_vector_value(button_anchor_adjustment)
    button = _coerce_vector_value(button_position)
    eef = _coerce_vector_value(eef_position)
    adjusted_button = _add_vectors(_add_vectors(button, anchor_adjustment), offset)
    return {
        "contact_frame_id": contact_frame_id,
        "axis_source": axis_source,
        "press_direction_vector": _to_builtin(_vector_to_list(press_axis)),
        "gripper_contact_surface_axis": _to_builtin(_vector_to_list(contact_axis)),
        "button_offset": _to_builtin(_vector_to_list(offset)),
        "button_offset_adjustment": _to_builtin(_vector_to_list(offset_adjustment)),
        "button_offset_source": (
            "robot_gripper_contact_geometry_plus_public_anchor_micro_sweep"
            if offset is not None and offset_adjustment is not None
            else ("robot_gripper_contact_geometry" if offset is not None else None)
        ),
        "button_anchor_adjustment": _to_builtin(_vector_to_list(anchor_adjustment)),
        "button_anchor_adjustment_source": (
            "public_button_anchor_tangent_micro_sweep" if anchor_adjustment is not None else None
        ),
        "target_position": _to_builtin(_vector_to_list(adjusted_button)),
        "target_distance_from_eef": _distance_between(eef, adjusted_button),
        "requires_action_execution_for_contact_confirmation": True,
        **(extra or {}),
    }


def _gripper_contact_surface_axis_candidates(env: Any) -> list[JsonDict]:
    try:
        unwrapped = getattr(env, "unwrapped", env)
        robot = list(unwrapped.robots)[0]
        gripper = robot.gripper["right"]
        important_geoms = gripper.important_geoms
        pad_names = [
            str(name)
            for key in ("left_fingerpad", "right_fingerpad")
            for name in important_geoms.get(key, [])
        ]
    except (AttributeError, IndexError, KeyError, TypeError):
        return []
    axes: list[JsonDict] = []
    seen: set[tuple[float, float, float]] = set()
    for name in pad_names:
        try:
            geom_id = unwrapped.sim.model.geom_name2id(name)
            rotation = np.asarray(unwrapped.sim.data.geom_xmat[geom_id], dtype=np.float32).reshape(3, 3)
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
        for local_axis_index in range(3):
            axis = rotation[:, local_axis_index]
            for sign in (-1.0, 1.0):
                signed_axis = _normalize_vector_or_none(axis * sign)
                if signed_axis is None:
                    continue
                key = _rounded_vector_key(signed_axis)
                if key in seen:
                    continue
                seen.add(key)
                axes.append(
                    {
                        "axis": _to_builtin(_vector_to_list(signed_axis)),
                        "source": "robot_gripper_pad_geometry",
                        "geom_name": name,
                        "local_axis_index": local_axis_index,
                    }
                )
    return axes


def _normalize_vector_or_none(value: Any) -> np.ndarray | None:
    vector = _coerce_vector_value(value)
    if vector is None:
        return None
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if vector.shape[0] != 3 or float(np.linalg.norm(vector)) <= 1e-8:
        return None
    return vector / float(np.linalg.norm(vector))


def _rounded_vector_key(value: Any, digits: int = 5) -> tuple[Any, ...]:
    vector = _coerce_vector_value(value)
    if vector is None:
        return (None,)
    return tuple(round(float(item), digits) for item in vector[:3])


def _env_fixture_state(env: Any, fixture_name: str | None) -> JsonDict | None:
    fixture = _lookup_env_fixture(env, fixture_name)
    if fixture is None or not hasattr(fixture, "get_state"):
        return None
    try:
        state = fixture.get_state(env)
    except TypeError:
        try:
            state = fixture.get_state()
        except Exception:
            return None
    except Exception:
        return None
    return _to_builtin(state) if isinstance(state, dict) else {"state": _to_builtin(state)}


def _env_fixture_button_contact(env: Any, fixture_name: str | None, button_name: str | None) -> bool | None:
    if env is None or button_name is None:
        return None
    fixture = _lookup_env_fixture(env, fixture_name)
    try:
        robot = env.robots[0]
        gripper = robot.gripper["right"]
        fixture_for_names = fixture if fixture is not None else type("FixtureName", (), {"name": fixture_name or "", "naming_prefix": ""})()
        for geom_name in _fixture_button_geom_candidates(fixture_for_names, str(button_name)):
            if bool(env.check_contact(gripper, geom_name)):
                return True
        return False
    except Exception:
        return None


def _sim_site_position(model: Any, data: Any, site_name: str) -> list[float] | None:
    try:
        site_id = model.site_name2id(site_name)
        if int(site_id) < 0:
            return None
        return _to_builtin(_vector_to_list(data.site_xpos[site_id]))
    except Exception:
        return None


def _sim_geom_position(model: Any, data: Any, geom_name: str) -> list[float] | None:
    try:
        geom_id = model.geom_name2id(geom_name)
        if int(geom_id) < 0:
            return None
        return _to_builtin(_vector_to_list(data.geom_xpos[geom_id]))
    except Exception:
        return None


def _entity_payload(obj: Any) -> JsonDict:
    payload: JsonDict = {"type": type(obj).__name__}
    for attr in ("name", "id", "rot", "pos", "size", "width", "depth", "height"):
        if hasattr(obj, attr):
            payload[attr] = _to_builtin(getattr(obj, attr))
    pose = _pose_to_list(obj)
    if pose is not None:
        payload["pose_world"] = pose
    return payload


def _body_pose(env: Any, body_id: Any) -> list[float] | None:
    sim = getattr(env, "sim", None)
    data = getattr(sim, "data", None)
    xpos = getattr(data, "body_xpos", None)
    xquat = getattr(data, "body_xquat", None)
    if xpos is None or body_id is None:
        return None
    try:
        pos = _flatten_numeric_list(_to_builtin(xpos[body_id]))
        quat = _flatten_numeric_list(_to_builtin(xquat[body_id])) if xquat is not None else []
        return pos + quat
    except Exception:
        return None


def _select_name(registry: dict[str, JsonDict], explicit_name_or_id: Any | None) -> str | None:
    from .robocasa_interface_repair import resolve_name
    return resolve_name(registry, explicit_name_or_id)


def _annotate_task_object_hints(objects: dict[str, JsonDict], task_description: str | None) -> None:
    if not objects or not task_description:
        return
    hints = _task_object_hints(task_description)
    if not hints:
        return
    if len(objects) == 1:
        payload = next(iter(objects.values()))
        payload["task_referent"] = hints[0]
        payload["semantic_hints"] = hints
        payload["agent_aliases"] = sorted(set([str(payload.get("name", "")), *hints]) - {""})


def _task_object_hints(task_description: str) -> list[str]:
    text = task_description.strip()
    patterns = (
        r"\bpick\s+(?:up\s+)?(?:the\s+|a\s+|an\s+)?([A-Za-z0-9_ -]+?)\s+from\b",
        r"\bplace\s+(?:the\s+|a\s+|an\s+)?([A-Za-z0-9_ -]+?)\s+(?:under|below|beneath|on|onto|inside|in|into|at)\b",
        r"\bmove\s+(?:the\s+|a\s+|an\s+)?([A-Za-z0-9_ -]+?)\s+(?:to|under|on|into|inside)\b",
    )
    hints: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            hint = _clean_task_object_hint(match.group(1))
            if hint and hint not in hints:
                hints.append(hint)
    return hints


def _clean_task_object_hint(value: str) -> str:
    cleaned = re.sub(r"\b(from|to|under|below|beneath|on|onto|inside|into|at|the|a|an)\b.*$", "", value, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^A-Za-z0-9_ -]+", " ", cleaned).strip().lower()
    return " ".join(cleaned.split())


def _entity_search_text(payload: JsonDict) -> str:
    parts: list[str] = []
    for key in ("name", "type", "fixture_type", "kind", "category", "model", "task_referent"):
        value = payload.get(key)
        if value is not None:
            parts.append(str(value))
    for key in ("semantic_hints", "agent_aliases"):
        value = payload.get(key)
        if isinstance(value, (list, tuple, set)):
            parts.extend(str(item) for item in value)
    return " ".join(parts)


def _normalize_text(text: str) -> str:
    camel_split = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", camel_split).lower()
    return " ".join(normalized.split())


def _placement_affordance_target(
    relation: str,
    target_payload: JsonDict | None = None,
    query: str | None = None,
    agent_context: JsonDict | None = None,
    site_name: str | None = None,
) -> JsonDict | None:
    if not isinstance(target_payload, dict):
        return None
    affordance_sites = target_payload.get("affordance_sites")
    if not isinstance(affordance_sites, dict):
        return None
    if not site_name:
        return None
    selected_site_name = site_name
    site_vec = _coerce_vector_value(affordance_sites.get(selected_site_name))
    if site_vec is None:
        return None
    return {
        "source": f"fixture_affordance_sites.{selected_site_name}",
        "site_name": selected_site_name,
        "position": _to_builtin(_vector_to_list(site_vec)),
        "reason": "caller_selected_affordance_site",
        "private_success_signal": False,
    }


def _placement_offset(
    relation: str,
    target_payload: JsonDict | None = None,
    query: str | None = None,
    agent_context: JsonDict | None = None,
) -> list[float]:
    normalized = relation.strip().lower().replace("_", " ")
    if normalized in {"under", "below", "beneath"} or "under" in normalized:
        return [0.0, 0.0, -0.12]
    if normalized in {"on", "onto", "on top", "above"} or "top" in normalized or "above" in normalized:
        size = _coerce_vector_value(target_payload.get("size") if isinstance(target_payload, dict) else None)
        height = target_payload.get("height") if isinstance(target_payload, dict) else None
        try:
            height_value = float(height if height is not None else (size[2] if size is not None and len(size) > 2 else 0.24))
        except Exception:
            height_value = 0.24
        return [0.0, 0.0, max(0.12, abs(height_value) * 0.5 + 0.05)]
    if normalized in {"inside", "in"}:
        return [0.0, 0.0, 0.02]
    return [0.0, 0.0, 0.0]


def _placement_release_xy_tolerance(
    target_payload: JsonDict | None,
    *,
    affordance_target: JsonDict | None,
) -> float:
    if affordance_target is not None:
        return 0.04
    size = _coerce_vector_value(target_payload.get("size") if isinstance(target_payload, dict) else None)
    width = target_payload.get("width") if isinstance(target_payload, dict) else None
    depth = target_payload.get("depth") if isinstance(target_payload, dict) else None
    try:
        width_value = abs(float(width if width is not None else (size[0] if size is not None else 0.16)))
        depth_value = abs(float(depth if depth is not None else (size[1] if size is not None and len(size) > 1 else 0.16)))
    except Exception:
        return 0.04
    # A support fixture can be much wider than the object. Keep the geometric
    # gate local enough that entering the fixture's broad footprint is not
    # mistaken for completing a transport onto a free surface point.
    return max(0.04, min(0.06, min(width_value, depth_value) * 0.25))


def _grasp_offset_for_object(payload: JsonDict | None) -> list[float]:
    size = _coerce_vector_value(payload.get("size") if isinstance(payload, dict) else None)
    if size is None:
        return [0.0, 0.0, 0.03]
    z_extent = float(max(0.02, min(0.08, abs(float(size[2])) * 0.35)))
    return [0.0, 0.0, z_extent]


def _near_side_grasp_offsets(object_position: Any, eef_position: Any, payload: JsonDict | None) -> tuple[list[float], list[float]] | None:
    """Two small world-space offsets toward the robot, using public object geometry."""
    object_xyz = _coerce_vector_value(object_position)
    eef_xyz = _coerce_vector_value(eef_position)
    if object_xyz is None or eef_xyz is None:
        return None
    direction = np.asarray(eef_xyz[:2] - object_xyz[:2], dtype=np.float32)
    length = float(np.linalg.norm(direction))
    if length < .03:
        return None
    direction /= length
    size = _coerce_vector_value(payload.get("size") if isinstance(payload, dict) else None)
    radius = .04 if size is None else .5 * float(max(abs(size[0]), abs(size[1])))
    pregrasp = min(.14, max(.08, radius + .06))
    contact = min(.03, max(.01, radius * .35))
    return ([float(direction[0] * pregrasp), float(direction[1] * pregrasp), .04],
            [float(direction[0] * contact), float(direction[1] * contact), .01])


def _add_vectors(position: Any | None, offset: list[float] | None) -> Any | None:
    if position is None:
        return None
    base = _coerce_vector_value(position)
    delta = _coerce_vector_value([0.0, 0.0, 0.0] if offset is None else offset)
    if base is None or delta is None:
        return None
    return base + delta


def _distance_between(source: Any | None, target: Any | None) -> float | None:
    try:
        import numpy as np
    except Exception:
        return None
    source_vec = _coerce_vector_value(source)
    target_vec = _coerce_vector_value(target)
    if source_vec is None or target_vec is None:
        return None
    return float(np.linalg.norm(target_vec - source_vec))


def _vector_to_list(vector: Any | None) -> list[float] | None:
    coerced = _coerce_vector_value(vector)
    if coerced is None:
        return None
    return [float(value) for value in coerced.tolist()]


def _robocasa_affordance_recommendations(
    object_name: str | None,
    target_name: str | None,
    relation: str,
    grasp_offset: list[float],
    placement_offset: list[float],
) -> list[JsonDict]:
    return [
        {
            "primitive": "grasp_robocasa_object",
            "arguments": {"object_name": object_name, "offset": grasp_offset, "strategy": "robocasa_state_delta_motion"},
            "purpose": "close gripper after moving to the estimated object contact pose",
        },
        {
            "primitive": "move_robocasa_ee_to",
            "arguments": {"target_name": object_name, "offset": [0.0, 0.0, 0.12], "strategy": "robocasa_state_delta_motion"},
            "purpose": "lift or clear the object before placement",
        },
        {
            "primitive": "place_robocasa_object_at",
            "arguments": {
                "object_name": object_name,
                "target_name": target_name,
                "relation": relation,
                "offset": placement_offset,
                "strategy": "robocasa_state_delta_motion",
            },
            "purpose": "move the grasped object toward the grounded target relation",
        },
    ]


def _robocasa_subgoal_sequence(
    *,
    object_name: str | None,
    target_name: str | None,
    relation: str,
    grasp_offset: list[float],
    placement_offset: list[float],
    prompt: str | None,
    query: str | None,
    agent_context: JsonDict,
) -> list[JsonDict]:
    context = deepcopy(agent_context)
    return [
        {
            "step": 1,
            "primitive": "observe_robocasa_kitchen_state",
            "arguments": {"prompt": prompt, "query": query, "agent_context": {**context, "plan_step": "observe symbolic kitchen state"}},
            "expected_evidence": ["objects", "fixtures", "pose_evidence"],
        },
        {
            "step": 2,
            "primitive": "observe_robocasa_rgbd",
            "arguments": {"prompt": prompt, "query": query, "agent_context": {**context, "plan_step": "inspect RGB-D and segmentation"}},
            "expected_evidence": ["rgbd_evidence", "segmentation_instances"],
        },
        {
            "step": 3,
            "primitive": "locate_robocasa_object",
            "arguments": {"object_name": object_name, "query": query, "agent_context": {**context, "plan_step": "ground source object"}},
            "expected_evidence": ["selected", "pose_evidence"],
        },
        {
            "step": 4,
            "primitive": "locate_robocasa_fixture",
            "arguments": {"fixture_name": target_name, "query": query, "agent_context": {**context, "plan_step": "ground placement target"}},
            "expected_evidence": ["selected", "pose_evidence"],
        },
        {
            "step": 5,
            "primitive": "inspect_robocasa_affordance",
            "arguments": {
                "object_name": object_name,
                "target_name": target_name,
                "relation": relation,
                "prompt": prompt,
                "query": query,
                "agent_context": {**context, "plan_step": "derive grasp and placement affordances"},
            },
            "expected_evidence": ["grasp_affordance", "placement_affordance"],
        },
        {
            "step": 6,
            "primitive": "grasp_robocasa_object",
            "arguments": {
                "object_name": object_name,
                "offset": grasp_offset,
                "horizon": 40,
                "strategy": "robocasa_state_delta_motion",
                "agent_context": {**context, "plan_step": "execute grasp primitive"},
            },
            "expected_evidence": ["execution_status", "motion_status"],
        },
        {
            "step": 7,
            "primitive": "place_robocasa_object_at",
            "arguments": {
                "object_name": object_name,
                "target_name": target_name,
                "relation": relation,
                "offset": placement_offset,
                "horizon": 80,
                "strategy": "robocasa_state_delta_motion",
                "agent_context": {**context, "plan_step": "execute placement primitive"},
            },
            "expected_evidence": ["execution_status", "motion_status"],
        },
        {
            "step": 8,
            "primitive": "record_robocasa_evidence",
            "arguments": {"key": "robocasa_subgoal_plan_trace", "value": {"object": object_name, "target": target_name, "relation": relation}},
            "expected_evidence": ["artifact_id"],
        },
    ]


def _task_description(obs: Any) -> str | None:
    if isinstance(obs, dict):
        for key, value in obs.items():
            lowered = str(key).lower()
            if "task_description" in lowered or lowered.endswith("task description"):
                if isinstance(value, str):
                    return value
                builtin = _to_builtin(value)
                if isinstance(builtin, str):
                    return builtin
            nested = _task_description(value)
            if nested:
                return nested
    if isinstance(obs, (list, tuple)):
        for item in obs:
            nested = _task_description(item)
            if nested:
                return nested
    return None


def _env_task_description(env: Any) -> str | None:
    for candidate in reversed(_env_candidates(env)):
        get_ep_meta = getattr(candidate, "get_ep_meta", None)
        if not callable(get_ep_meta):
            continue
        try:
            metadata = get_ep_meta()
        except Exception:
            continue
        if not isinstance(metadata, dict):
            continue
        for key in ("lang", "task_description", "instruction"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _split_reset_result(reset_result: Any) -> tuple[Any, JsonDict]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        return obs, dict(info or {})
    return reset_result, {}


def _split_step_result(step_result: Any) -> tuple[Any, Any, bool, bool, JsonDict]:
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, reward, bool(terminated), bool(truncated), dict(info or {})
    if isinstance(step_result, tuple) and len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, reward, bool(done), False, dict(info or {})
    raise RuntimeError(f"Unexpected RoboCasa step() result shape: {type(step_result).__name__}")


def _read_current_observation(env: Any) -> Any | None:
    if env is None:
        return None
    for candidate in (env, _unwrap_env(env)):
        for method_name in ("_get_observations", "_get_obs"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    return method()
                except Exception:
                    continue
    return None


def _sample_action_spec(action_spec: Any) -> Any:
    if isinstance(action_spec, tuple) and len(action_spec) == 2:
        low, high = action_spec
        try:
            import numpy as np

            low_arr = np.asarray(low)
            high_arr = np.asarray(high)
            return np.clip(np.zeros_like(low_arr, dtype=np.float32), low_arr, high_arr)
        except Exception:
            pass
    raise RuntimeError("action_space_unavailable")


_DIRECT_ROBOCASA_ACTION_MARKER = "__robocasa_direct_composite_action__"


def _zero_action(env: Any) -> tuple[Any | None, Any | None, Any | None, Any | None, str | None]:
    try:
        import numpy as np
    except Exception:
        return None, None, None, None, None

    action_space = getattr(env, "action_space", None)
    action_space_sample = None
    if action_space is not None and hasattr(action_space, "sample"):
        action_space_sample = action_space.sample()
        if isinstance(action_space_sample, dict):
            template: dict[str, Any] = {}
            spaces = getattr(action_space, "spaces", {}) or {}
            for key, value in action_space_sample.items():
                template[str(key)] = _zero_for_action_value(value, spaces.get(key) if hasattr(spaces, "get") else None)
            action_key = _select_robocasa_delta_action_key(template)
            if action_key is None:
                return None, None, None, None, None
            vector = np.asarray(template[action_key], dtype=np.float32).reshape(-1)
            low, high = _dict_action_bounds(spaces.get(action_key) if hasattr(spaces, "get") else None, vector)
            return template, vector, low, high, action_key
    direct_template = _direct_composite_action_template(env)
    if direct_template is not None:
        return direct_template
    if action_space_sample is not None:
        sample = np.asarray(action_space_sample, dtype=np.float32)
        low = np.asarray(getattr(action_space, "low", np.full_like(sample, -1.0)), dtype=np.float32)
        high = np.asarray(getattr(action_space, "high", np.full_like(sample, 1.0)), dtype=np.float32)
        action = np.clip(np.zeros_like(sample, dtype=np.float32), low, high)
        return action.copy(), action, low, high, None
    for candidate in _env_candidates(env):
        action_spec = getattr(candidate, "action_spec", None)
        if isinstance(action_spec, tuple) and len(action_spec) == 2:
            low = np.asarray(action_spec[0], dtype=np.float32)
            high = np.asarray(action_spec[1], dtype=np.float32)
            action = np.clip(np.zeros_like(low, dtype=np.float32), low, high)
            return action.copy(), action, low, high, None
    return None, None, None, None, None


def _direct_composite_action_template(env: Any) -> tuple[Any, Any, Any, Any, str] | None:
    try:
        import numpy as np
    except Exception:
        return None

    for candidate in _env_candidates(env):
        robots = getattr(candidate, "robots", None)
        if not robots:
            continue
        for robot in list(robots):
            controller = getattr(robot, "composite_controller", None)
            splits = getattr(controller, "_action_split_indexes", None)
            if not splits:
                continue
            try:
                low, high = controller.action_limits
                low = np.asarray(low, dtype=np.float32).reshape(-1)
                high = np.asarray(high, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            split_indexes: dict[str, tuple[int, int]] = {}
            for part_name, raw_slice in dict(splits).items():
                try:
                    start, end = int(raw_slice[0]), int(raw_slice[1])
                except Exception:
                    continue
                if end > start:
                    split_indexes[str(part_name)] = (start, end)
            if not split_indexes:
                continue
            eef_part = _select_direct_eef_part(split_indexes, robot, controller)
            if eef_part is None:
                continue
            eef_start, eef_end = split_indexes[eef_part]
            highest_split = max(end for _, end in split_indexes.values())
            low, high = _direct_composite_action_bounds(
                candidate,
                robot,
                low,
                high,
                minimum_dim=highest_split + (1 if _direct_controller_uses_base_mode(controller) else 0),
            )
            action = np.clip(np.zeros_like(low, dtype=np.float32), low, high)
            base_mode_index = int(action.shape[0] - 1) if action.shape[0] > highest_split else None
            if base_mode_index is not None:
                action[base_mode_index] = -1.0
            gripper_part = _select_direct_gripper_part(split_indexes, eef_part)
            template = {
                _DIRECT_ROBOCASA_ACTION_MARKER: True,
                "template_action": action.copy(),
                "action_low": low.copy(),
                "action_high": high.copy(),
                "eef_part": eef_part,
                "eef_slice": [eef_start, eef_end],
                "gripper_part": gripper_part,
                "gripper_slice": list(split_indexes[gripper_part]) if gripper_part is not None else None,
                "base_slice": list(split_indexes["base"]) if "base" in split_indexes else None,
                "torso_slice": list(split_indexes["torso"]) if "torso" in split_indexes else None,
                "base_mode_index": base_mode_index,
                "base_mode_value": -1.0,
                "controller_input_ref_frame": _direct_eef_input_ref_frame(controller, eef_part),
                "split_indexes": {name: [start, end] for name, (start, end) in split_indexes.items()},
            }
            return template, action[eef_start:eef_end].copy(), low[eef_start:eef_end], high[eef_start:eef_end], eef_part
    return None


def _direct_composite_action_bounds(
    env_candidate: Any,
    robot: Any,
    low: Any,
    high: Any,
    *,
    minimum_dim: int,
) -> tuple[Any, Any]:
    try:
        import numpy as np
    except Exception:
        return low, high

    for owner in (env_candidate, robot):
        action_spec = getattr(owner, "action_spec", None)
        if isinstance(action_spec, tuple) and len(action_spec) == 2:
            try:
                candidate_low = np.asarray(action_spec[0], dtype=np.float32).reshape(-1)
                candidate_high = np.asarray(action_spec[1], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if candidate_low.shape == candidate_high.shape and candidate_low.shape[0] >= minimum_dim:
                return candidate_low, candidate_high
        action_limits = getattr(owner, "action_limits", None)
        if isinstance(action_limits, tuple) and len(action_limits) == 2:
            try:
                candidate_low = np.asarray(action_limits[0], dtype=np.float32).reshape(-1)
                candidate_high = np.asarray(action_limits[1], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if candidate_low.shape == candidate_high.shape and candidate_low.shape[0] >= minimum_dim:
                return candidate_low, candidate_high
    low = np.asarray(low, dtype=np.float32).reshape(-1)
    high = np.asarray(high, dtype=np.float32).reshape(-1)
    if low.shape[0] >= minimum_dim:
        return low, high
    pad = int(minimum_dim - low.shape[0])
    return (
        np.concatenate([low, np.full(pad, -1.0, dtype=np.float32)]),
        np.concatenate([high, np.full(pad, 1.0, dtype=np.float32)]),
    )


def _direct_controller_uses_base_mode(controller: Any) -> bool:
    name = str(getattr(controller, "name", "")).upper()
    type_name = type(controller).__name__.upper()
    return "HYBRID_MOBILE_BASE" in name or "HYBRID" in type_name


def _direct_eef_input_ref_frame(controller: Any, eef_part: str) -> str | None:
    part_controllers = getattr(controller, "part_controllers", None)
    if hasattr(part_controllers, "get"):
        part_controller = part_controllers.get(eef_part)
        frame = getattr(part_controller, "input_ref_frame", None)
        if frame:
            return str(frame)
    config = getattr(controller, "composite_controller_specific_config", None)
    if isinstance(config, dict):
        frame = config.get("ik_input_ref_frame")
        if frame:
            return str(frame)
    return None


def _select_direct_eef_part(split_indexes: dict[str, tuple[int, int]], robot: Any, controller: Any) -> str | None:
    arms = [str(arm) for arm in (getattr(robot, "arms", None) or getattr(controller, "arms", None) or [])]
    for candidate in ["right", *arms, "left"]:
        if candidate in split_indexes and split_indexes[candidate][1] - split_indexes[candidate][0] >= 3:
            return candidate
    for name, (start, end) in split_indexes.items():
        lowered = name.lower()
        if ("arm" in lowered or "right" in lowered or "eef" in lowered) and end - start >= 3:
            return name
    for name, (start, end) in split_indexes.items():
        if "gripper" not in name.lower() and end - start >= 3:
            return name
    return None


def _select_direct_gripper_part(split_indexes: dict[str, tuple[int, int]], eef_part: str) -> str | None:
    preferred = [f"{eef_part}_gripper", f"{eef_part}gripper", "right_gripper", "left_gripper"]
    for candidate in preferred:
        if candidate in split_indexes:
            return candidate
    for name, (start, end) in split_indexes.items():
        if "gripper" in name.lower() and end > start:
            return name
    return None


def _env_candidates(env: Any) -> list[Any]:
    candidates: list[Any] = []
    current = env
    for _ in range(4):
        if current is None or any(current is item for item in candidates):
            break
        candidates.append(current)
        unwrapped = getattr(current, "unwrapped", None)
        nested = getattr(current, "env", None)
        next_candidate = unwrapped if unwrapped is not None and unwrapped is not current else nested
        if next_candidate is None or next_candidate is current:
            break
        current = next_candidate
    return candidates


def _zero_for_action_value(value: Any, space: Any | None = None) -> Any:
    try:
        import numpy as np
    except Exception:
        return value

    if hasattr(space, "n"):
        return 0
    if hasattr(space, "low") and hasattr(space, "high"):
        low = np.asarray(space.low, dtype=np.float32)
        high = np.asarray(space.high, dtype=np.float32)
        return np.clip(np.zeros_like(low, dtype=np.float32), low, high)
    array = np.asarray(value)
    if array.shape == ():
        return type(value)(0) if isinstance(value, (int, float, bool)) else 0
    return np.zeros_like(array, dtype=np.float32)


def _select_robocasa_delta_action_key(template: dict[str, Any]) -> str | None:
    preferred = [
        "action.end_effector_position",
        "robot0_right",
        "robot0_right_pos",
        "right",
    ]
    for key in preferred:
        if key in template and _action_value_size(template[key]) >= 3:
            return key
    for key in sorted(template):
        lowered = key.lower()
        if "end_effector_position" in lowered and _action_value_size(template[key]) >= 3:
            return key
    for key in sorted(template):
        lowered = key.lower()
        if ("right" in lowered or "arm" in lowered or "eef" in lowered) and _action_value_size(template[key]) >= 3:
            return key
    return None


def _action_value_size(value: Any) -> int:
    try:
        import numpy as np

        return int(np.asarray(value).reshape(-1).shape[0])
    except Exception:
        return 0


def _dict_action_bounds(space: Any | None, vector: Any) -> tuple[Any, Any]:
    try:
        import numpy as np
    except Exception:
        return None, None
    if space is not None and hasattr(space, "low") and hasattr(space, "high"):
        return np.asarray(space.low, dtype=np.float32).reshape(-1), np.asarray(space.high, dtype=np.float32).reshape(-1)
    return np.full_like(vector, -1.0, dtype=np.float32), np.full_like(vector, 1.0, dtype=np.float32)


def _set_flat_gripper_command_if_present(action_vector: Any, gripper_command: float, *, action_key: str | None) -> None:
    """Write a gripper command only for flat action vectors that actually expose one.

    Some RoboCasa / robosuite controllers expose only a 3D position or 6D
    pose delta. In those cases ``action[-1]`` is a motion channel, not a
    gripper channel, so writing a gripper command can destabilize the arm.
    """
    if action_key is not None:
        return
    try:
        flat = action_vector.reshape(-1)
    except Exception:
        return
    if flat.shape[0] in {4, 7} or flat.shape[0] > 7:
        flat[-1] = gripper_command


def _format_step_action(
    template: Any,
    action_vector: Any,
    action_key: str | None,
    gripper_command: float,
    *,
    base_action: Any | None = None,
    torso_action: Any | None = None,
    base_mode_value: float | None = None,
) -> Any:
    if action_key is None:
        return action_vector
    try:
        import numpy as np
    except Exception:
        return template
    if isinstance(template, dict) and template.get(_DIRECT_ROBOCASA_ACTION_MARKER):
        full_action = np.asarray(template["template_action"], dtype=np.float32).copy()
        start, end = [int(value) for value in template["eef_slice"]]
        shaped = np.asarray(action_vector, dtype=np.float32).reshape(-1)
        full_action[start:end] = 0.0
        full_action[start : start + min(end - start, shaped.shape[0])] = shaped[: end - start]
        gripper_slice = template.get("gripper_slice")
        if gripper_slice is not None:
            gripper_start, gripper_end = [int(value) for value in gripper_slice]
            full_action[gripper_start:gripper_end] = _direct_gripper_command(gripper_command)
        base_slice = template.get("base_slice")
        if base_action is not None and base_slice is not None:
            base_start, base_end = [int(value) for value in base_slice]
            base_vector = np.asarray(base_action, dtype=np.float32).reshape(-1)
            width = max(0, base_end - base_start)
            full_action[base_start:base_end] = 0.0
            full_action[base_start : base_start + min(width, base_vector.shape[0])] = base_vector[:width]
        torso_slice = template.get("torso_slice")
        if torso_action is not None and torso_slice is not None:
            torso_start, torso_end = [int(value) for value in torso_slice]
            torso_vector = np.asarray(torso_action, dtype=np.float32).reshape(-1)
            width = max(0, torso_end - torso_start)
            full_action[torso_start:torso_end] = 0.0
            full_action[torso_start : torso_start + min(width, torso_vector.shape[0])] = torso_vector[:width]
        base_mode_index = template.get("base_mode_index")
        if base_mode_index is not None:
            full_action[int(base_mode_index)] = float(
                template.get("base_mode_value", -1.0) if base_mode_value is None else base_mode_value
            )
        low = np.asarray(template.get("action_low", np.full_like(full_action, -1.0)), dtype=np.float32)
        high = np.asarray(template.get("action_high", np.full_like(full_action, 1.0)), dtype=np.float32)
        return np.clip(full_action, low, high)
    action = deepcopy(template)
    original = np.asarray(action[action_key], dtype=np.float32)
    shaped = np.asarray(action_vector, dtype=np.float32).reshape(original.shape)
    action[action_key] = shaped
    for key in list(action):
        lowered = key.lower()
        if "gripper_close" in lowered:
            action[key] = np.asarray([gripper_command], dtype=np.float32)
        elif "gripper" in lowered and np.asarray(action[key]).reshape(-1).shape[0] == 1:
            action[key] = np.asarray([gripper_command], dtype=np.float32)
        elif base_action is not None and "base" in lowered and "mode" not in lowered:
            original_base = np.asarray(action[key], dtype=np.float32)
            base_vector = np.asarray(base_action, dtype=np.float32).reshape(-1)
            shaped_base = np.zeros(original_base.size, dtype=np.float32)
            shaped_base[: min(shaped_base.shape[0], base_vector.shape[0])] = base_vector[: shaped_base.shape[0]]
            action[key] = shaped_base.reshape(original_base.shape)
        elif base_mode_value is not None and "base_mode" in lowered:
            action[key] = int(1 if base_mode_value > 0 else 0)
    return action


def _direct_gripper_command(gripper_command: float) -> float:
    return 1.0 if float(gripper_command) > 0.5 else -1.0


def _motion_phase_summary(step_records: list[JsonDict]) -> JsonDict:
    summary: JsonDict = {}
    for record in step_records:
        phase = str(record.get("phase", "unknown"))
        payload = summary.setdefault(
            phase,
            {
                "steps": 0,
                "contact_confirmed": False,
                "base_action_seen": False,
            },
        )
        payload["steps"] = int(payload["steps"]) + 1
        for field_name, label in (
            ("distance_to_target", "distance"),
            ("xy_distance_to_target", "xy_distance"),
            ("object_to_eef_distance", "object_to_eef_distance"),
            ("pre_step_object_to_eef_distance", "pre_step_object_to_eef_distance"),
        ):
            value = record.get(field_name)
            if value is None:
                continue
            payload.setdefault(f"first_{label}", value)
            payload[f"last_{label}"] = value
            best_key = f"best_{label}"
            if payload.get(best_key) is None or float(value) < float(payload[best_key]):
                payload[best_key] = value
        payload["contact_confirmed"] = bool(payload.get("contact_confirmed")) or bool(record.get("contact_confirmed"))
        if record.get("base_action") is not None:
            payload["base_action_seen"] = True
            payload["last_base_action"] = record.get("base_action")
        if record.get("eef_target_position") is not None:
            payload["last_eef_target_position"] = record.get("eef_target_position")
        if record.get("phase_exit_reason"):
            payload["exit_reason"] = record.get("phase_exit_reason")
    return summary


def _best_motion_record(step_records: list[JsonDict]) -> JsonDict | None:
    best: JsonDict | None = None
    best_distance: float | None = None
    for record in step_records:
        distance = record.get("distance_to_target")
        if distance is None:
            continue
        if best_distance is None or float(distance) < best_distance:
            best_distance = float(distance)
            best = record
    return deepcopy(best) if best is not None else None


def _best_xy_motion_record(step_records: list[JsonDict]) -> JsonDict | None:
    best: JsonDict | None = None
    best_xy_distance: float | None = None
    for record in step_records:
        distance = record.get("xy_distance_to_target")
        if distance is None:
            continue
        if best_xy_distance is None or float(distance) < best_xy_distance:
            best_xy_distance = float(distance)
            best = record
    return deepcopy(best) if best is not None else None


def _robocasa_place_refinement_hint(
    obs: Any,
    *,
    object_name: Any,
    target_name: str | None,
    explicit_target_position: Any,
    offset: Any | None,
    best_xy_distance: float | None,
    final_xy_distance: float | None,
    release_xy_tolerance: float | None,
    release_ready: bool,
    release_xy_ready: bool,
    release_contact_ready: bool,
    release_executed: bool,
    release_skipped_reason: str | None,
    contact_confirmed: bool,
    transport_contact_confirmed: bool,
    ever_contact_confirmed: bool,
    best_contact_distance: float | None,
    final_contact_distance: float | None,
    contact_tolerance: float,
    best_step: JsonDict | None,
    best_xy_step: JsonDict | None,
    object_error_gain: float,
    object_error_clip: float,
    object_xy_push_steps: int,
    object_xy_push_align_steps: int,
    object_xy_contact_seek_steps: int,
    release_only_when_ready: bool,
    object_xy_early_stop_enabled: bool,
    object_xy_stop_when_within: float | None,
    xy_early_stop_triggered: bool,
    xy_early_stop_reason: str | None,
) -> JsonDict:
    hint: JsonDict = {
        "available": False,
        "reason": release_skipped_reason,
        "release_ready": bool(release_ready),
        "release_xy_ready": bool(release_xy_ready),
        "release_contact_ready": bool(release_contact_ready),
        "release_only_when_ready": bool(release_only_when_ready),
        "release_xy_tolerance": release_xy_tolerance,
        "best_xy_distance": best_xy_distance,
        "final_xy_distance": final_xy_distance,
        "best_object_to_eef_distance": best_contact_distance,
        "final_object_to_eef_distance": final_contact_distance,
        "contact_tolerance": contact_tolerance,
        "ever_contact_confirmed": bool(ever_contact_confirmed),
        "contact_confirmed": bool(contact_confirmed),
        "transport_contact_confirmed": bool(transport_contact_confirmed),
        "object_xy_early_stop_enabled": bool(object_xy_early_stop_enabled),
        "object_xy_stop_when_within": object_xy_stop_when_within,
        "xy_early_stop_triggered": bool(xy_early_stop_triggered),
        "xy_early_stop_reason": xy_early_stop_reason,
        "private_success_signal": False,
    }
    if not isinstance(obs, dict) or not isinstance(object_name, str):
        hint["reason"] = hint["reason"] or "object_pose_unavailable"
        return hint

    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project dependency.
        hint["reason"] = hint["reason"] or "numpy_unavailable"
        return hint

    object_position = _target_position(obs, target_name=object_name, explicit_position=None)
    target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
    if object_position is None or target_position is None:
        hint["reason"] = hint["reason"] or "object_or_target_pose_unavailable"
        return hint
    offset_vector = _coerce_vector_value(offset)
    if offset_vector is None:
        offset_vector = np.zeros(3, dtype=np.float32)
    target_with_offset = target_position + offset_vector
    error_vector = target_with_offset - object_position
    xy_error = error_vector[:2]
    xy_norm = float(np.linalg.norm(xy_error))
    xy_error_list = [float(value) for value in np.asarray(xy_error, dtype=np.float32).reshape(-1)[:2].tolist()]
    xy_unit = xy_error / max(xy_norm, 1e-6)
    xy_unit_list = [float(value) for value in np.asarray(xy_unit, dtype=np.float32).reshape(-1)[:2].tolist()]
    threshold = float(release_xy_tolerance) if release_xy_tolerance is not None else None
    xy_margin = xy_norm - threshold if threshold is not None else None

    hint.update(
        {
            "available": True,
            "object_position": _to_builtin(_vector_to_list(object_position)),
            "target_position": _to_builtin(_vector_to_list(target_with_offset)),
            "object_to_target_error": _to_builtin(_vector_to_list(error_vector)),
            "object_to_target_xy_error": _to_builtin(xy_error_list),
            "object_to_target_xy_unit": _to_builtin(xy_unit_list),
            "object_to_target_xy_frame": "world_xy_target_minus_object",
            "xy_error_norm": xy_norm,
            "xy_margin_to_release_threshold": xy_margin,
            "best_step_phase": best_step.get("phase") if isinstance(best_step, dict) else None,
            "best_step_xy_distance": best_step.get("xy_distance_to_target") if isinstance(best_step, dict) else None,
            "best_xy_step_phase": best_xy_step.get("phase") if isinstance(best_xy_step, dict) else None,
            "best_xy_step_index": best_xy_step.get("step_index") if isinstance(best_xy_step, dict) else None,
            "best_xy_step_exit_reason": best_xy_step.get("phase_exit_reason") if isinstance(best_xy_step, dict) else None,
        }
    )
    return hint


def _robocasa_place_motion_strategy(
    *,
    release_ready: bool,
    release_executed: bool,
    needs_xy_refinement: bool,
    needs_contact_reacquire: bool,
    ever_contact_confirmed: bool,
) -> str:
    if release_ready or release_executed:
        return "release_and_settle"
    if not ever_contact_confirmed:
        return "regrasp_before_place"
    if needs_contact_reacquire:
        return "contact_reacquire_then_xy_push"
    if needs_xy_refinement:
        return "xy_push_inside_release_window"
    return "hold_and_reobserve"


def _robocasa_place_motion_hypotheses(
    *,
    recommended_motion_strategy: str,
    xy_error_norm: float,
    release_xy_tolerance: float,
    needs_xy_refinement: bool,
    needs_contact_reacquire: bool,
    ever_contact_confirmed: bool,
    contact_confirmed: bool,
) -> list[JsonDict]:
    public_signal = {
        "xy_error_norm": round(float(xy_error_norm), 6),
        "release_xy_tolerance": round(float(release_xy_tolerance), 6),
        "needs_xy_refinement": bool(needs_xy_refinement),
        "needs_contact_reacquire": bool(needs_contact_reacquire),
        "ever_contact_confirmed": bool(ever_contact_confirmed),
        "contact_confirmed": bool(contact_confirmed),
    }
    return [
        {
            "id": "regrasp_before_place",
            "selected": recommended_motion_strategy == "regrasp_before_place",
            "when_to_use": "no public contact was observed during grasp/place, so another grasp/contact approach should precede placement",
            "parameter_bias": {
                "primitive_sequence": ["grasp_robocasa_object", "place_robocasa_object_at"],
                "grasp_contact_steps": "increase",
                "force_grasp_contact_steps": True,
            },
            "public_signal": public_signal,
        },
        {
            "id": "contact_reacquire_then_xy_push",
            "selected": recommended_motion_strategy == "contact_reacquire_then_xy_push",
            "when_to_use": "contact existed earlier but is lost before release",
            "parameter_bias": {
                "contact_guard_enabled": True,
                "contact_guard_recover_steps": "increase",
                "object_xy_contact_seek_steps": "increase",
                "release_only_when_ready": True,
            },
            "public_signal": public_signal,
        },
        {
            "id": "xy_push_inside_release_window",
            "selected": recommended_motion_strategy == "xy_push_inside_release_window",
            "when_to_use": "object remains grasped/contacted but public XY error is outside release tolerance",
            "parameter_bias": {
                "object_error_gain": "increase_bounded",
                "object_xy_push_steps": "increase",
                "object_xy_push_align_steps": "keep_or_increase",
                "object_xy_stop_when_within": round(float(release_xy_tolerance), 6),
            },
            "public_signal": public_signal,
        },
        {
            "id": "release_and_settle",
            "selected": recommended_motion_strategy == "release_and_settle",
            "when_to_use": "public release gate is satisfied",
            "parameter_bias": {
                "release_only_when_ready": True,
                "release_settle_steps": "keep",
                "post_release_retreat_steps": "task_dependent",
            },
            "public_signal": public_signal,
        },
        {
            "id": "hold_and_reobserve",
            "selected": recommended_motion_strategy == "hold_and_reobserve",
            "when_to_use": "public signals are ambiguous; keep holding and collect another observation/pose diagnostic",
            "parameter_bias": {
                "release_only_when_ready": True,
                "object_xy_push_steps": "do_not_increase_until_reobserved",
            },
            "public_signal": public_signal,
        },
    ]


def _extract_robocasa_refinement_hint(value: Any) -> JsonDict:
    raw = value.output if isinstance(value, PrimitiveResult) else value
    if isinstance(raw, dict) and isinstance(raw.get("placement_refinement_hint"), dict):
        hint = dict(raw["placement_refinement_hint"])
        step_summary = raw.get("step_summary")
        if isinstance(step_summary, dict):
            final = step_summary.get("final")
            if isinstance(final, dict):
                hint.setdefault("public_step_final", deepcopy(final))
            phase_summary = step_summary.get("phase_summary")
            if isinstance(phase_summary, dict):
                hint.setdefault("public_step_phase_summary", deepcopy(phase_summary))
            best_xy_step = step_summary.get("best_xy_step")
            if isinstance(best_xy_step, dict):
                hint.setdefault("public_best_xy_step_phase", best_xy_step.get("phase"))
                hint.setdefault("public_best_xy_step_index", best_xy_step.get("step_index"))
        raw = hint
    if isinstance(raw, dict) and isinstance(raw.get("final_refinement_hint"), dict):
        raw = raw["final_refinement_hint"]
    return dict(raw) if isinstance(raw, dict) else {"available": False, "reason": "hint_not_mapping"}


def _robocasa_transport_state_from_public_inputs(
    *,
    motion_result: Any | None,
    runtime_report: JsonDict | None,
    placement_refinement_hint: Any | None,
) -> JsonDict:
    output = _robocasa_public_result_output(motion_result)
    report_output = _robocasa_latest_transport_output(runtime_report or {})
    if not output and report_output:
        output = report_output

    hint = _extract_robocasa_refinement_hint(placement_refinement_hint) if placement_refinement_hint is not None else {}
    if not hint.get("available"):
        hint = _extract_robocasa_refinement_hint(output)
    if not hint.get("available") and output.get("final_refinement_hint"):
        hint = _extract_robocasa_refinement_hint(output.get("final_refinement_hint"))

    step_final = _robocasa_transport_step_final(output, hint)
    release_gate = step_final.get("release_gate") if isinstance(step_final.get("release_gate"), dict) else {}
    phase_summary = output.get("phase_summary")
    if not isinstance(phase_summary, dict):
        step_summary = output.get("step_summary")
        phase_summary = step_summary.get("phase_summary") if isinstance(step_summary, dict) else {}
    if not isinstance(phase_summary, dict):
        phase_summary = hint.get("public_step_phase_summary") if isinstance(hint.get("public_step_phase_summary"), dict) else {}

    contact_tolerance = _first_float(
        output.get("contact_tolerance"),
        hint.get("contact_tolerance"),
        release_gate.get("contact_tolerance"),
        0.08,
    )
    release_xy_tolerance = _first_float(
        output.get("release_xy_tolerance"),
        hint.get("release_xy_tolerance"),
        release_gate.get("release_xy_tolerance"),
        0.04,
    )
    object_to_eef = _first_float(
        output.get("object_distance_after"),
        output.get("best_object_to_eef_distance"),
        hint.get("object_to_eef_distance_after"),
        release_gate.get("object_to_eef_distance"),
    )
    best_xy = _min_float(
        output.get("best_xy_distance"),
        hint.get("best_xy_distance"),
        release_gate.get("best_xy_distance"),
        step_final.get("best_xy_distance"),
    )
    final_xy = _first_float(
        output.get("final_xy_distance"),
        output.get("distance_after") if output.get("distance_metric") == "object_to_target" else None,
        hint.get("xy_error_norm"),
        hint.get("final_xy_distance"),
        release_gate.get("xy_distance_to_target"),
        step_final.get("xy_distance_to_target"),
    )
    contact_confirmed = _first_bool(output.get("contact_confirmed"), hint.get("contact_confirmed"), release_gate.get("contact_confirmed"))
    ever_contact_confirmed = _first_bool(
        output.get("ever_contact_confirmed"),
        hint.get("ever_contact_confirmed"),
        release_gate.get("ever_contact_confirmed"),
        contact_confirmed,
    )
    release_xy_ready = _first_bool(
        output.get("release_xy_ready"),
        hint.get("release_xy_ready"),
        release_gate.get("release_xy_ready"),
    )
    if release_xy_ready is None and release_xy_tolerance is not None:
        xy_for_gate = final_xy if final_xy is not None else best_xy
        release_xy_ready = xy_for_gate is not None and xy_for_gate <= release_xy_tolerance
    release_requires_contact = _first_bool(output.get("release_requires_contact"), hint.get("release_requires_contact"), release_gate.get("release_requires_contact"))
    release_contact_ready = _first_bool(
        output.get("release_contact_ready"),
        hint.get("release_contact_ready"),
        release_gate.get("release_contact_ready"),
    )
    if release_contact_ready is None:
        release_contact_ready = (not bool(release_requires_contact)) or bool(contact_confirmed)
    release_ready = _first_bool(output.get("release_ready"), hint.get("release_ready"), release_gate.get("release_ready"))
    if release_ready is None:
        release_ready = bool(release_xy_ready and release_contact_ready)
    release_executed = _first_bool(
        output.get("release_executed"),
        hint.get("release_executed"),
        release_gate.get("release_executed"),
        False,
    )
    held = bool(contact_confirmed) or (object_to_eef is not None and contact_tolerance is not None and object_to_eef <= contact_tolerance)
    heldness_status = "likely_held" if held else "contact_lost" if ever_contact_confirmed else "not_established"
    contact_status = (
        "stable_contact"
        if bool(contact_confirmed)
        else "contact_was_seen_but_lost"
        if bool(ever_contact_confirmed)
        else "not_confirmed"
    )
    stability_signal = _robocasa_post_release_stability_signal(
        {
            **hint,
            "public_step_final": step_final,
            "public_step_phase_summary": phase_summary,
            "release_ready": release_ready,
            "release_xy_ready": release_xy_ready,
            "release_contact_ready": release_contact_ready,
            "release_executed": release_executed,
            "release_xy_tolerance": release_xy_tolerance,
            "best_xy_distance": best_xy,
            "xy_error_norm": final_xy,
        }
    )
    drifted_after_release = bool(stability_signal.get("drifted_after_release"))
    return {
        "heldness": {
            "status": heldness_status,
            "likely_held": held,
            "object_to_eef_distance_after": object_to_eef,
            "contact_tolerance": contact_tolerance,
            "source": "public_motion_trace_or_refinement_hint",
        },
        "contact_stability": {
            "status": contact_status,
            "contact_confirmed": bool(contact_confirmed),
            "ever_contact_confirmed": bool(ever_contact_confirmed),
            "phase_summary": _compact_robocasa_phase_summary(phase_summary),
        },
        "release_gate": {
            "release_ready": bool(release_ready),
            "release_xy_ready": bool(release_xy_ready),
            "release_contact_ready": bool(release_contact_ready),
            "release_requires_contact": bool(release_requires_contact),
            "release_xy_tolerance": release_xy_tolerance,
            "best_xy_distance": best_xy,
            "final_xy_distance": final_xy,
            "release_executed": bool(release_executed),
        },
        "settle_state": {
            "release_executed": bool(release_executed),
            "post_release_stability": stability_signal,
            "drifted_after_release": drifted_after_release,
        },
        "public_metrics": {
            "distance_metric": output.get("distance_metric"),
            "distance_before": output.get("distance_before"),
            "distance_after": output.get("distance_after"),
            "best_distance": output.get("best_distance"),
            "best_xy_distance": best_xy,
            "final_xy_distance": final_xy,
            "motion_status": output.get("motion_status"),
            "execution_status": output.get("execution_status"),
        },
    }


def _robocasa_live_transport_observation(
    env: Any,
    obs: Any,
    *,
    object_name: str | None,
    previous_state: Any | None,
    contact_tolerance: float,
    gripper_closed_threshold: float,
    coupling_tolerance: float,
    minimum_coupling_motion: float,
) -> JsonDict:
    if env is None or not isinstance(obs, dict) or not object_name:
        return {
            "available": False,
            "object_name": object_name,
            "reason": "live_env_observation_or_object_name_unavailable",
        }
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project test dependency.
        return {"available": False, "object_name": object_name, "reason": "numpy_unavailable"}

    eef_position = _vector(obs, "robot0_eef_pos")
    object_position = _target_position(obs, target_name=object_name, explicit_position=None)
    object_to_eef_distance = _object_to_eef_distance(obs, object_name=object_name)
    gripper_qpos = _observation_numeric_list(obs, "robot0_gripper_qpos", "state.gripper_qpos")
    gripper_qvel = _observation_numeric_list(obs, "robot0_gripper_qvel", "state.gripper_qvel")
    gripper_closed = (
        all(abs(float(value)) < float(gripper_closed_threshold) for value in gripper_qpos)
        if gripper_qpos
        else None
    )
    physical_contact = _env_gripper_object_contact(env, object_name)

    previous = _robocasa_previous_live_observation(previous_state)
    coupling: JsonDict = {
        "available": False,
        "coupled": None,
        "coupling_error": None,
        "eef_displacement": None,
        "object_displacement": None,
        "minimum_motion": float(minimum_coupling_motion),
        "tolerance": float(coupling_tolerance),
    }
    previous_eef = _coerce_vector_value(previous.get("eef_position"))
    previous_object = _coerce_vector_value(previous.get("object_position"))
    if eef_position is not None and object_position is not None and previous_eef is not None and previous_object is not None:
        eef_displacement = np.asarray(eef_position, dtype=np.float32) - previous_eef
        object_displacement = np.asarray(object_position, dtype=np.float32) - previous_object
        eef_motion = float(np.linalg.norm(eef_displacement))
        object_motion = float(np.linalg.norm(object_displacement))
        coupling_error = float(np.linalg.norm(eef_displacement - object_displacement))
        coupling = {
            "available": True,
            "coupled": bool(eef_motion >= float(minimum_coupling_motion) and coupling_error <= float(coupling_tolerance)),
            "coupling_error": coupling_error,
            "eef_motion": eef_motion,
            "object_motion": object_motion,
            "eef_displacement": _to_builtin(_vector_to_list(eef_displacement)),
            "object_displacement": _to_builtin(_vector_to_list(object_displacement)),
            "minimum_motion": float(minimum_coupling_motion),
            "tolerance": float(coupling_tolerance),
        }

    distance_contact = (
        object_to_eef_distance is not None and object_to_eef_distance <= float(contact_tolerance)
    )
    coupled = coupling.get("coupled") is True
    if physical_contact is True and (gripper_closed is True or coupled):
        likely_held: bool | None = True
        held_status = "likely_held"
    elif physical_contact is False:
        likely_held = False
        held_status = "physical_contact_absent"
    elif coupled:
        likely_held = True
        held_status = "motion_coupled"
    elif physical_contact is None and gripper_closed is None:
        likely_held = None
        held_status = "insufficient_physical_state"
    else:
        likely_held = False
        held_status = "not_established"
    return {
        "available": True,
        "object_name": object_name,
        "eef_position": _to_builtin(_vector_to_list(eef_position)),
        "object_position": _to_builtin(_vector_to_list(object_position)),
        "object_to_eef_distance": object_to_eef_distance,
        "contact_tolerance": float(contact_tolerance),
        "distance_contact": bool(distance_contact),
        "gripper_object_contact": physical_contact,
        "gripper_qpos": gripper_qpos,
        "gripper_qvel": gripper_qvel,
        "gripper_closed_threshold": float(gripper_closed_threshold),
        "gripper_closed": gripper_closed,
        "coupling": coupling,
        "likely_held": likely_held,
        "held_status": held_status,
        "source": "live_observation_and_public_sim_contact",
        "private_task_signal_exposed": False,
    }


def _merge_robocasa_live_transport_state(state: JsonDict, live: JsonDict) -> JsonDict:
    merged = deepcopy(state)
    if not live.get("available"):
        return merged
    heldness = dict(merged.get("heldness") or {})
    likely_held = live.get("likely_held")
    if isinstance(likely_held, bool):
        heldness["likely_held"] = likely_held
        heldness["status"] = live.get("held_status")
    heldness.update(
        {
            "object_to_eef_distance_after": live.get("object_to_eef_distance"),
            "gripper_object_contact": live.get("gripper_object_contact"),
            "gripper_closed": live.get("gripper_closed"),
            "motion_coupled": (live.get("coupling") or {}).get("coupled"),
            "source": "live_observation_and_public_sim_contact",
        }
    )
    contact_stability = dict(merged.get("contact_stability") or {})
    if isinstance(live.get("gripper_object_contact"), bool):
        contact_stability["contact_confirmed"] = live["gripper_object_contact"]
        contact_stability["status"] = "physical_contact" if live["gripper_object_contact"] else "physical_contact_absent"
    contact_stability["live_coupling"] = live.get("coupling")
    merged["heldness"] = heldness
    merged["contact_stability"] = contact_stability
    return merged


def _robocasa_previous_live_observation(value: Any) -> JsonDict:
    raw = value.to_dict() if isinstance(value, PrimitiveResult) else value
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("output"), dict):
        raw = raw["output"]
    if isinstance(raw.get("live_observation"), dict):
        raw = raw["live_observation"]
    return dict(raw) if isinstance(raw, dict) else {}


def _observation_numeric_list(obs: Any, *keys: str) -> list[float] | None:
    for key in keys:
        value = _lookup_observation_value(obs, key)
        if value is None:
            continue
        try:
            import numpy as np

            array = np.asarray(value, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            continue
        if array.size:
            return [float(item) for item in array.tolist()]
    return None


def _robocasa_public_result_output(value: Any) -> JsonDict:
    raw = value.to_dict() if isinstance(value, PrimitiveResult) else value
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("output"), dict):
        output = dict(raw["output"])
        if isinstance(raw.get("error"), str):
            output.setdefault("primitive_error", raw["error"])
        return output
    return dict(raw)


def _robocasa_latest_transport_output(runtime_report: JsonDict) -> JsonDict:
    for item in reversed(runtime_report.get("motion_sequence") or []):
        if not isinstance(item, dict):
            continue
        if item.get("name") in {"place_robocasa_object_at", "refine_robocasa_place_until_ready"}:
            return _robocasa_public_result_output(item)
    return {}


def _robocasa_transport_step_final(output: JsonDict, hint: JsonDict) -> JsonDict:
    step_summary = output.get("step_summary")
    if isinstance(step_summary, dict) and isinstance(step_summary.get("final"), dict):
        return dict(step_summary["final"])
    if isinstance(hint.get("public_step_final"), dict):
        return dict(hint["public_step_final"])
    return {}


def _first_float(*values: Any) -> float | None:
    for value in values:
        parsed = _robocasa_hint_float(value)
        if parsed is not None:
            return parsed
    return None


def _min_float(*values: Any) -> float | None:
    parsed = [_robocasa_hint_float(value) for value in values]
    parsed = [value for value in parsed if value is not None]
    return min(parsed) if parsed else None


def _first_bool(*values: Any) -> bool | None:
    for value in values:
        if value is None:
            continue
        return bool(value)
    return None


def _compact_robocasa_phase_summary(phase_summary: Any) -> JsonDict:
    if not isinstance(phase_summary, dict):
        return {}
    compact: JsonDict = {}
    for phase, payload in phase_summary.items():
        if not isinstance(payload, dict):
            continue
        compact[str(phase)] = {
            key: payload.get(key)
            for key in (
                "steps",
                "first_xy_distance",
                "best_xy_distance",
                "last_xy_distance",
                "contact_confirmed",
                "release_ready",
            )
            if key in payload
        }
    return compact


def _robocasa_place_recovery_plan_from_hint(
    hint: JsonDict,
    *,
    strategy: str | None,
    attempts: int,
    prompt: str | None,
    query: str | None,
    agent_context: JsonDict | None,
) -> JsonDict:
    selected_strategy, selection_source = _robocasa_select_recovery_strategy(hint, strategy)
    max_attempts = max(1, int(attempts))
    base_next_call = {
        key: deepcopy(value)
        for key, value in dict(hint.get("recommended_next_call") or {}).items()
        if key != "primitive"
    }
    entity_parameter_keys = {
        "primitive",
        "object_name",
        "target_name",
        "target_position",
        "relation",
        "offset",
        "use_affordance_site",
        "affordance_site_name",
    }
    override_allowed = set((hint.get("agent_override_schema") or {}).get("allowed_public_parameters") or [])
    if override_allowed:
        allowed = override_allowed | entity_parameter_keys
    else:
        allowed = entity_parameter_keys | {
            "object_error_gain",
            "object_error_clip",
            "object_xy_push_steps",
            "object_xy_push_align_steps",
            "object_xy_push_backoff",
            "object_xy_push_through",
            "object_xy_push_z_offset",
            "object_xy_contact_seek_steps",
            "object_xy_contact_seek_backoff",
            "object_xy_contact_seek_z_offset",
            "object_xy_push_reacquire_from_side",
            "contact_guard_enabled",
            "contact_guard_tolerance",
            "contact_guard_recover_steps",
            "release_only_when_ready",
            "release_xy_tolerance",
            "release_gripper_command",
            "release_requires_contact",
            "release_settle_steps",
            "post_release_retreat_offset",
            "post_release_retreat_steps",
            "post_release_retreat_min_distance",
            "post_release_retreat_max_steps",
            "object_xy_stop_when_within",
            "object_xy_stop_requires_contact",
            "transport_gripper_command",
            "mobile_base_enabled",
            "base_gain",
            "base_max_delta",
            "base_command_sign",
            "base_mode_value",
            "base_xy_deadband",
        }
    if _robocasa_public_xy_push_ineffective(hint):
        allowed |= {
            "object_xy_push_reacquire_from_side",
            "release_requires_contact",
            "transport_gripper_command",
            "object_xy_contact_seek_steps",
            "object_xy_contact_seek_backoff",
            "object_xy_contact_seek_z_offset",
            "contact_guard_enabled",
            "contact_guard_tolerance",
            "contact_guard_recover_steps",
            "object_xy_stop_requires_contact",
        }
    if _robocasa_public_post_release_drift(hint):
        allowed |= {
            "release_gripper_command",
            "release_requires_contact",
            "release_settle_steps",
            "post_release_retreat_offset",
            "post_release_retreat_steps",
            "post_release_retreat_min_distance",
            "post_release_retreat_max_steps",
            "object_xy_stop_requires_contact",
            "transport_gripper_command",
        }

    parameter_schedule: list[JsonDict] = []
    primitive_sequence: list[JsonDict] = []
    for attempt_index in range(max_attempts):
        if _robocasa_recovery_needs_pre_place_grasp(selected_strategy, hint):
            primitive_sequence.append(
                {
                    "attempt": attempt_index + 1,
                    "primitive": "grasp_robocasa_object",
                    "strategy": selected_strategy,
                    "parameters": _robocasa_recovery_grasp_attempt_params(
                        base_next_call,
                        attempt_index=attempt_index,
                        hint=hint,
                    ),
                }
            )
        params = _robocasa_recovery_attempt_params(
            base_next_call,
            selected_strategy=selected_strategy,
            attempt_index=attempt_index,
            hint=hint,
        )
        params = {key: value for key, value in params.items() if key in allowed}
        schedule_item = {
            "attempt": attempt_index + 1,
            "primitive": "place_robocasa_object_at",
            "strategy": selected_strategy,
            "parameters": params,
        }
        parameter_schedule.append(
            schedule_item,
        )
        primitive_sequence.append(
            {
                "primitive": "place_robocasa_object_at",
                "parameters": schedule_item["parameters"],
                "attempt": schedule_item["attempt"],
                "strategy": selected_strategy,
            }
        )

    return {
        "available": True,
        "selected_strategy": selected_strategy,
        "selection_source": selection_source,
        "parameter_schedule": parameter_schedule,
        "primitive_sequence": primitive_sequence,
        "motion_hypotheses": hint.get("motion_hypotheses") or [],
        "public_failure_evidence": {
            "xy_error_norm": hint.get("xy_error_norm"),
            "release_xy_tolerance": hint.get("release_xy_tolerance"),
            "needs_xy_refinement": hint.get("needs_xy_refinement"),
            "needs_contact_reacquire": hint.get("needs_contact_reacquire"),
            "contact_confirmed": hint.get("contact_confirmed"),
            "ever_contact_confirmed": hint.get("ever_contact_confirmed"),
            "best_xy_step_phase": hint.get("best_xy_step_phase"),
            "best_step_phase": hint.get("best_step_phase"),
            "object_xy_push_ineffective": _robocasa_public_xy_push_ineffective(hint),
            "public_xy_push_effectiveness": hint.get("public_xy_push_effectiveness"),
            "post_release_drift": _robocasa_public_post_release_drift(hint),
            "public_post_release_stability": hint.get("public_post_release_stability"),
        },
        "allowed_public_parameters_used": sorted(
            {
                key
                for item in parameter_schedule
                for key in item.get("parameters", {})
            }
        ),
        "prompt": prompt,
        "query": query,
        "agent_context": agent_context or {},
        "private_success_signal_exposed": False,
        "official_task_completion_claimed": False,
        "verifier_boundary": {
            "call_after_sequence": "verify(scope='task')",
            "private_task_signal_exposed": False,
            "primitive_claims_task_completion": False,
        },
    }


def _robocasa_hint_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _robocasa_public_xy_push_ineffective(hint: JsonDict) -> bool:
    explicit = hint.get("object_xy_push_ineffective")
    if explicit is not None:
        return bool(explicit)
    effectiveness = hint.get("public_xy_push_effectiveness")
    if isinstance(effectiveness, dict) and effectiveness.get("ineffective") is not None:
        return bool(effectiveness.get("ineffective"))
    phase_summary = hint.get("public_step_phase_summary")
    if not isinstance(phase_summary, dict):
        phase_summary = hint.get("phase_summary")
    if not isinstance(phase_summary, dict):
        return False
    signal = _robocasa_xy_push_effectiveness_from_phase_summary(phase_summary, hint=hint)
    hint.setdefault("public_xy_push_effectiveness", signal)
    return bool(signal.get("ineffective"))


def _robocasa_xy_push_effectiveness_from_phase_summary(phase_summary: JsonDict, *, hint: JsonDict) -> JsonDict:
    push_phases = ("xy_push_reacquire", "xy_push_contact_guard", "xy_push_align", "xy_push")
    first_xy: float | None = None
    best_xy: float | None = None
    last_xy: float | None = None
    push_steps = 0
    contact_seen = False
    for phase in push_phases:
        payload = phase_summary.get(phase)
        if not isinstance(payload, dict):
            continue
        push_steps += int(payload.get("steps") or 0)
        contact_seen = contact_seen or bool(payload.get("contact_confirmed"))
        phase_first = _robocasa_hint_float(payload.get("first_xy_distance"))
        phase_best = _robocasa_hint_float(payload.get("best_xy_distance"))
        phase_last = _robocasa_hint_float(payload.get("last_xy_distance"))
        if phase_first is not None and first_xy is None:
            first_xy = phase_first
        if phase_best is not None:
            best_xy = phase_best if best_xy is None else min(best_xy, phase_best)
        if phase_last is not None:
            last_xy = phase_last
    release_xy_tolerance = _robocasa_hint_float(hint.get("release_xy_tolerance")) or 0.04
    xy_error = _robocasa_hint_float(hint.get("xy_error_norm"))
    if xy_error is None:
        xy_error = _robocasa_hint_float(hint.get("final_xy_distance"))
    if xy_error is None:
        xy_error = _robocasa_hint_float(hint.get("best_xy_distance"))
    improvement = (first_xy - best_xy) if first_xy is not None and best_xy is not None else None
    min_useful_improvement = min(0.01, max(0.003, release_xy_tolerance * 0.20))
    ineffective = bool(
        push_steps >= 8
        and xy_error is not None
        and xy_error > release_xy_tolerance
        and improvement is not None
        and improvement < min_useful_improvement
    )
    return {
        "push_steps": push_steps,
        "first_xy_distance": first_xy,
        "best_xy_distance": best_xy,
        "last_xy_distance": last_xy,
        "best_xy_improvement": improvement,
        "min_useful_improvement": min_useful_improvement,
        "contact_seen_during_push": contact_seen,
        "ineffective": ineffective,
    }


def _robocasa_public_post_release_drift(hint: JsonDict) -> bool:
    explicit = hint.get("post_release_drift")
    if explicit is not None:
        return bool(explicit)
    stability = hint.get("public_post_release_stability")
    if isinstance(stability, dict) and stability.get("drifted_after_release") is not None:
        return bool(stability.get("drifted_after_release"))
    signal = _robocasa_post_release_stability_signal(hint)
    hint.setdefault("public_post_release_stability", signal)
    return bool(signal.get("drifted_after_release"))


def _robocasa_post_release_stability_signal(hint: JsonDict) -> JsonDict:
    final = hint.get("public_step_final")
    if not isinstance(final, dict):
        final = hint.get("step_final") if isinstance(hint.get("step_final"), dict) else {}
    release_gate = final.get("release_gate") if isinstance(final.get("release_gate"), dict) else {}
    phase_summary = hint.get("public_step_phase_summary")
    if not isinstance(phase_summary, dict):
        phase_summary = hint.get("phase_summary") if isinstance(hint.get("phase_summary"), dict) else {}
    release_xy_tolerance = _robocasa_hint_float(hint.get("release_xy_tolerance")) or _robocasa_hint_float(release_gate.get("release_xy_tolerance")) or 0.04
    best_xy = _robocasa_hint_float(final.get("best_xy_distance"))
    if best_xy is None:
        best_xy = _robocasa_hint_float(hint.get("best_xy_distance"))
    final_xy = _robocasa_hint_float(final.get("xy_distance_to_target"))
    if final_xy is None:
        final_xy = _robocasa_hint_float(final.get("final_xy_distance"))
    if final_xy is None:
        final_xy = _robocasa_hint_float(hint.get("xy_error_norm"))
    post_release_best: float | None = None
    for phase in ("post_release_clearance", "post_release_retreat"):
        payload = phase_summary.get(phase) if isinstance(phase_summary, dict) else None
        if not isinstance(payload, dict):
            continue
        value = _robocasa_hint_float(payload.get("best_xy_distance"))
        if value is not None:
            post_release_best = value if post_release_best is None else max(post_release_best, value)
    release_ready = bool(
        release_gate.get("release_ready") is True
        or (
            hint.get("release_ready") is True
            and hint.get("release_xy_ready") is True
            and hint.get("release_contact_ready") is True
        )
    )
    release_executed = release_gate.get("release_executed")
    if release_executed is None:
        release_executed = release_ready
    entered_release_window = best_xy is not None and best_xy <= release_xy_tolerance
    drift_margin = max(0.01, release_xy_tolerance * 0.25)
    drifted_after_release = bool(
        release_ready
        and bool(release_executed)
        and entered_release_window
        and final_xy is not None
        and final_xy > release_xy_tolerance + drift_margin
    )
    return {
        "release_ready": release_ready,
        "release_executed": bool(release_executed),
        "release_xy_tolerance": release_xy_tolerance,
        "best_xy_distance": best_xy,
        "final_xy_distance": final_xy,
        "post_release_best_xy_distance": post_release_best,
        "drift_margin": drift_margin,
        "entered_release_window": entered_release_window,
        "post_release_left_release_window": bool(
            release_ready
            and bool(release_executed)
            and entered_release_window
            and post_release_best is not None
            and post_release_best > release_xy_tolerance
        ),
        "drifted_after_release": drifted_after_release,
    }


def _robocasa_select_recovery_strategy(hint: JsonDict, strategy: str | None) -> tuple[str, str]:
    if strategy:
        return str(strategy), "agent_override"
    if _robocasa_public_post_release_drift(hint):
        stability = hint.get("public_post_release_stability")
        if isinstance(stability, dict) and stability.get("post_release_left_release_window"):
            return "settled_release_no_retreat", "inferred_from_public_post_release_retreat_drift_trace"
        return "stabilized_release_retreat", "inferred_from_public_post_release_drift_trace"
    if _robocasa_public_xy_push_ineffective(hint):
        return "regrasp_then_verified_contact_push", "inferred_from_public_ineffective_xy_push_trace"
    recommended = hint.get("recommended_motion_strategy")
    if recommended:
        return str(recommended), "placement_refinement_hint.recommended_motion_strategy"
    if hint.get("needs_contact_reacquire") or hint.get("contact_confirmed") is False:
        return "contact_reacquire_then_xy_push", "inferred_from_public_contact_flags"
    xy_error = _robocasa_hint_float(hint.get("xy_error_norm"))
    release_xy_tolerance = _robocasa_hint_float(hint.get("release_xy_tolerance")) or 0.04
    if hint.get("needs_xy_refinement") or (xy_error is not None and xy_error > release_xy_tolerance):
        return "xy_push_inside_release_window", "inferred_from_public_xy_error"
    if hint.get("release_ready"):
        return "release_and_settle", "inferred_from_public_release_gate"
    return "hold_and_reobserve", "fallback_no_public_recovery_signal"


def _robocasa_recovery_needs_pre_place_grasp(selected_strategy: str, hint: JsonDict) -> bool:
    return selected_strategy == "regrasp_then_verified_contact_push" or (
        selected_strategy in {"regrasp_before_place", "contact_reacquire_then_xy_push"}
        and (bool(hint.get("needs_contact_reacquire")) or hint.get("contact_confirmed") is False)
    )


def _robocasa_recovery_grasp_attempt_params(
    base_next_call: JsonDict,
    *,
    attempt_index: int,
    hint: JsonDict,
) -> JsonDict:
    object_name = base_next_call.get("object_name") or hint.get("object_name") or "object"
    contact_tolerance = _robocasa_hint_float(base_next_call.get("contact_tolerance")) or 0.08
    return {
        "object_name": object_name,
        "offset": [0.0, 0.0, 0.02 if _robocasa_public_xy_push_ineffective(hint) else 0.03],
        "horizon": max(int(base_next_call.get("horizon") or 80), 80),
        "grasp_contact_steps": 50 + 8 * attempt_index,
        "grasp_contact_offset": [0.0, 0.0, 0.0],
        "grasp_contact_tolerance": round(min(contact_tolerance, 0.012), 3),
        "force_grasp_contact_steps": True,
        "grasp_hold_steps": (18 if _robocasa_public_xy_push_ineffective(hint) else 12) + 4 * attempt_index,
        "gripper_command": 1.0,
        "grasp_lift_delta": 0.06,
        "contact_tolerance": contact_tolerance,
        "gain": _robocasa_hint_float(base_next_call.get("gain")) or 8.0,
        "max_delta": _robocasa_hint_float(base_next_call.get("max_delta")) or 0.4,
        "arm_delta_frame": base_next_call.get("arm_delta_frame"),
        "mobile_base_enabled": bool(base_next_call.get("mobile_base_enabled", False)),
        "base_gain": _robocasa_hint_float(base_next_call.get("base_gain")) or 1.0,
        "base_max_delta": _robocasa_hint_float(base_next_call.get("base_max_delta")) or 0.35,
        "base_command_sign": _robocasa_hint_float(base_next_call.get("base_command_sign")) or 1.0,
        "base_mode_value": _robocasa_hint_float(base_next_call.get("base_mode_value")) or 1.0,
        "base_xy_deadband": _robocasa_hint_float(base_next_call.get("base_xy_deadband")) or 0.12,
    }


def _robocasa_recovery_attempt_params(
    base_next_call: JsonDict,
    *,
    selected_strategy: str,
    attempt_index: int,
    hint: JsonDict,
) -> JsonDict:
    params = dict(base_next_call)
    release_threshold = float(hint.get("release_xy_tolerance") or 0.04)
    params.setdefault("release_only_when_ready", True)
    params.setdefault("release_xy_tolerance", release_threshold)
    params.setdefault("object_xy_early_stop_enabled", True)
    params.setdefault("object_xy_stop_when_within", release_threshold)
    xy_error = _robocasa_hint_float(hint.get("xy_error_norm"))
    near_goal_xy_refine = xy_error is not None and xy_error <= max(release_threshold * 2.5, 0.10)
    if selected_strategy == "regrasp_then_verified_contact_push":
        params["contact_guard_enabled"] = True
        params["contact_guard_tolerance"] = round(max(float(params.get("contact_guard_tolerance") or 0.08), 0.08 + 0.005 * attempt_index), 3)
        params["contact_guard_recover_steps"] = max(int(params.get("contact_guard_recover_steps") or 12), 32 + 8 * attempt_index)
        params["object_xy_contact_seek_steps"] = max(int(params.get("object_xy_contact_seek_steps") or 0), 24 + 8 * attempt_index)
        params["object_xy_contact_seek_backoff"] = round(
            max(0.0, min(float(params.get("object_xy_contact_seek_backoff") or 0.018), 0.025)),
            3,
        )
        params["object_xy_contact_seek_z_offset"] = round(
            max(0.0, min(float(params.get("object_xy_contact_seek_z_offset") or 0.006), 0.01)),
            3,
        )
        params["object_xy_push_align_steps"] = max(int(params.get("object_xy_push_align_steps") or 0), 4 + 2 * attempt_index)
        params["object_xy_push_steps"] = max(12, min(int(params.get("object_xy_push_steps") or 18), 24 + 4 * attempt_index))
        params["object_xy_push_backoff"] = round(max(0.01, min(float(params.get("object_xy_push_backoff") or 0.018), 0.024)), 3)
        params["object_xy_push_through"] = round(
            max(float(params.get("object_xy_push_through") or 0.06), min(0.11, max(0.06, float(xy_error or 0.0) * 0.75))),
            3,
        )
        params["object_xy_push_z_offset"] = round(max(0.004, min(float(params.get("object_xy_push_z_offset") or 0.006), 0.008)), 3)
        params["object_xy_push_reacquire_from_side"] = False
        params["object_xy_stop_requires_contact"] = True
        params["release_requires_contact"] = True
        params["transport_gripper_command"] = 1.0
        params["base_xy_deadband"] = min(float(params.get("base_xy_deadband") or 0.025), 0.025)
        params["base_max_delta"] = min(float(params.get("base_max_delta") or 0.18), 0.18)
    elif selected_strategy == "contact_reacquire_then_xy_push":
        params["contact_guard_enabled"] = True
        params["contact_guard_tolerance"] = round(max(float(params.get("contact_guard_tolerance") or 0.09), 0.09 + 0.01 * attempt_index), 3)
        params["contact_guard_recover_steps"] = max(int(params.get("contact_guard_recover_steps") or 20), 40 + 12 * attempt_index)
        params["object_xy_contact_seek_steps"] = max(int(params.get("object_xy_contact_seek_steps") or 0), 16 + 8 * attempt_index)
        params["object_xy_push_steps"] = max(int(params.get("object_xy_push_steps") or 0), (44 if near_goal_xy_refine else 32) + 8 * attempt_index)
        params["object_xy_push_backoff"] = round(
            max(0.012, min(float(params.get("object_xy_push_backoff") or 0.035), (0.02 if near_goal_xy_refine else 0.045) - 0.003 * attempt_index)),
            3,
        )
        params["object_xy_push_through"] = round(
            max(
                float(params.get("object_xy_push_through") or 0.055),
                (min(0.14, max(0.075, float(xy_error or 0.0) + 0.025)) if near_goal_xy_refine else 0.055)
                + 0.012 * attempt_index,
            ),
            3,
        )
        params["object_xy_push_z_offset"] = round(
            max(0.004, min(float(params.get("object_xy_push_z_offset") or 0.012), 0.006 if near_goal_xy_refine else 0.012)),
            3,
        )
        params["object_xy_contact_seek_backoff"] = round(
            max(0.0, min(float(params.get("object_xy_contact_seek_backoff") or 0.03), 0.03)),
            3,
        )
        params["object_xy_contact_seek_z_offset"] = round(
            max(0.0, min(float(params.get("object_xy_contact_seek_z_offset") or 0.01), 0.012)),
            3,
        )
        params["object_xy_stop_requires_contact"] = True
    elif selected_strategy == "xy_push_inside_release_window":
        params["object_error_gain"] = round(min(max(float(params.get("object_error_gain") or 2.0), 2.4) + 0.25 * attempt_index, 4.0), 3)
        params["object_xy_push_steps"] = max(int(params.get("object_xy_push_steps") or 0), (36 if near_goal_xy_refine else 24) + 8 * attempt_index)
        params["object_xy_push_align_steps"] = max(int(params.get("object_xy_push_align_steps") or 0), (4 if near_goal_xy_refine else 8) + 4 * attempt_index)
        if near_goal_xy_refine:
            params["object_xy_push_backoff"] = round(max(0.012, min(float(params.get("object_xy_push_backoff") or 0.02), 0.02)), 3)
            params["object_xy_push_through"] = round(
                max(float(params.get("object_xy_push_through") or 0.075), min(0.14, max(0.075, float(xy_error or 0.0) + 0.025)) + 0.01 * attempt_index),
                3,
            )
            params["object_xy_push_z_offset"] = round(max(0.004, min(float(params.get("object_xy_push_z_offset") or 0.006), 0.006)), 3)
        params["object_xy_stop_requires_contact"] = bool(hint.get("contact_confirmed"))
    elif selected_strategy == "release_and_settle":
        params["release_only_when_ready"] = True
        params["object_xy_push_steps"] = 0
        params["object_xy_contact_seek_steps"] = 0
    elif selected_strategy == "stabilized_release_retreat":
        params["release_only_when_ready"] = True
        params["release_requires_contact"] = True
        params["object_xy_stop_requires_contact"] = True
        params["transport_gripper_command"] = 1.0
        params["object_xy_push_steps"] = min(int(params.get("object_xy_push_steps") or 0), 8)
        params["object_xy_contact_seek_steps"] = min(int(params.get("object_xy_contact_seek_steps") or 0), 8)
        params["release_settle_steps"] = max(int(params.get("release_settle_steps") or 0), 14 + 4 * attempt_index)
        params["release_gripper_command"] = float(params.get("release_gripper_command") if params.get("release_gripper_command") is not None else 0.0)
        params["post_release_retreat_offset"] = [0.0, 0.0, round(0.14 + 0.02 * attempt_index, 3)]
        params["post_release_retreat_steps"] = max(4, min(int(params.get("post_release_retreat_steps") or 8), 8 + 2 * attempt_index))
        params["post_release_retreat_min_distance"] = round(max(0.02, min(float(params.get("post_release_retreat_min_distance") or 0.04), 0.06)), 3)
        params["post_release_retreat_max_steps"] = max(4, min(int(params.get("post_release_retreat_max_steps") or 8), 8 + 2 * attempt_index))
    elif selected_strategy == "settled_release_no_retreat":
        params["release_only_when_ready"] = True
        params["release_requires_contact"] = True
        params["object_xy_stop_requires_contact"] = True
        params["transport_gripper_command"] = 1.0
        params["object_xy_push_steps"] = min(int(params.get("object_xy_push_steps") or 0), 6)
        params["object_xy_contact_seek_steps"] = min(int(params.get("object_xy_contact_seek_steps") or 0), 6)
        params["release_settle_steps"] = max(int(params.get("release_settle_steps") or 0), 24 + 4 * attempt_index)
        params["release_gripper_command"] = float(params.get("release_gripper_command") if params.get("release_gripper_command") is not None else 0.0)
        params["post_release_retreat_offset"] = [0.0, 0.0, 0.0]
        params["post_release_retreat_steps"] = 0
        params["post_release_retreat_min_distance"] = 0.0
        params["post_release_retreat_max_steps"] = 0
    elif selected_strategy == "regrasp_before_place":
        params["contact_guard_enabled"] = True
        params["contact_guard_recover_steps"] = max(int(params.get("contact_guard_recover_steps") or 12), 24 + 8 * attempt_index)
        params["object_xy_contact_seek_steps"] = max(int(params.get("object_xy_contact_seek_steps") or 0), 8 + 4 * attempt_index)
        params["object_xy_push_steps"] = max(int(params.get("object_xy_push_steps") or 0), 16 + 6 * attempt_index)
    else:
        params["object_xy_push_steps"] = min(int(params.get("object_xy_push_steps") or 0), 8)
        params["release_only_when_ready"] = True
    if near_goal_xy_refine:
        params["base_xy_deadband"] = min(float(params.get("base_xy_deadband") or 0.025), 0.025)
        params["base_max_delta"] = min(float(params.get("base_max_delta") or 0.18), 0.18)
    return params


def _vector(obs: dict[str, Any], key: str) -> Any | None:
    return _coerce_vector_value(_lookup_observation_value(obs, key))


def _delta_for_robocasa_action_frame(delta: Any, obs: dict[str, Any], frame: str) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project test dependency.
        return delta

    delta_array = np.asarray(delta, dtype=np.float32)
    normalized = str(frame or "world").strip().lower()
    if normalized in {"", "world", "global"}:
        return delta_array
    if normalized not in {"base", "robot_base", "mobilebase0_base"}:
        return delta_array
    quat = _lookup_observation_value(obs, "robot0_base_quat")
    if quat is None:
        return delta_array
    rotation = _quat_xyzw_to_rotation_matrix(quat)
    if rotation is None:
        return delta_array
    return rotation.T @ delta_array


def _quat_xyzw_to_rotation_matrix(quat: Any) -> Any | None:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project test dependency.
        return None

    q = np.asarray(quat, dtype=np.float32).reshape(-1)
    if q.shape[0] < 4:
        return None
    x, y, z, w = [float(value) for value in q[:4]]
    norm = float(np.linalg.norm([x, y, z, w]))
    if norm <= 1e-8:
        return None
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _lookup_observation_value(obs: Any, key: str) -> Any | None:
    if not isinstance(obs, dict):
        return None
    if key in obs:
        return obs[key]
    for nested_key in ("observation", "obs", "state", "proprio", "proprioception", "robot_state"):
        nested = obs.get(nested_key)
        if isinstance(nested, dict):
            value = _lookup_observation_value(nested, key)
            if value is not None:
                return value
    return None


def _coerce_vector_value(value: Any) -> Any | None:
    try:
        import numpy as np
    except Exception:
        return None

    if value is None:
        return None
    if isinstance(value, dict):
        for xyz_keys in (("x", "y", "z"), ("X", "Y", "Z")):
            if all(axis in value for axis in xyz_keys):
                try:
                    return np.asarray([value[axis] for axis in xyz_keys], dtype=np.float32)
                except (TypeError, ValueError):
                    pass
        for nested_key in (
            "position",
            "pos",
            "xyz",
            "translation",
            "translation_world",
            "world_position",
            "state",
            "obs",
            "value",
            "values",
            "array",
            "data",
        ):
            nested = value.get(nested_key)
            vector = _coerce_vector_value(nested)
            if vector is not None:
                return vector
        if len(value) == 1:
            vector = _coerce_vector_value(next(iter(value.values())))
            if vector is not None:
                return vector
        for nested_key, nested in value.items():
            lowered = str(nested_key).lower()
            if any(token in lowered for token in ("image", "depth", "seg", "mask", "rgb")):
                continue
            vector = _coerce_vector_value(nested)
            if vector is not None:
                return vector
        return None
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.shape[0] < 3:
        return None
    return array[:3]


def _coerce_vector_list(value: Any) -> list[float] | None:
    vector = _coerce_vector_value(value)
    if vector is None:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    array = np.asarray(vector, dtype=np.float64).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return [float(item) for item in array[:3]]


def _nearest_robocasa_visual_instance(
    instances_by_camera: dict[str, list[JsonDict]],
    *,
    entity_name: str | None,
    world_position: list[float],
    max_world_distance: float | None = None,
) -> JsonDict | None:
    """Select the nearest observed native visual instance to a caller point."""

    try:
        import numpy as np
    except Exception:
        return None
    target = np.asarray(world_position, dtype=np.float64).reshape(-1)
    if target.size < 3 or not np.all(np.isfinite(target[:3])):
        return None
    records: list[JsonDict] = []
    for camera_name, instances in sorted(instances_by_camera.items()):
        if not isinstance(instances, list):
            continue
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            if entity_name is not None and instance.get("entity_name") != entity_name:
                continue
            point = _coerce_vector_list(instance.get("native_world_position"))
            if point is None:
                continue
            distance = float(np.linalg.norm(np.asarray(point, dtype=np.float64)[:3] - target[:3]))
            if max_world_distance is not None and distance > float(max_world_distance):
                continue
            records.append(
                {
                    "camera_name": str(camera_name),
                    "segmentation_id": int(instance["segmentation_id"]),
                    "entity_name": instance.get("entity_name"),
                    "entity_kind": instance.get("entity_kind"),
                    "button_name": instance.get("button_name"),
                    "native_world_position": point,
                    "distance_to_requested_world_position": distance,
                    "pixel_count": int(instance.get("pixel_count", 0) or 0),
                }
            )
    if not records:
        return None
    records.sort(key=lambda item: (float(item["distance_to_requested_world_position"]), -int(item["pixel_count"])))
    return records[0]


def _target_position(obs: dict[str, Any], *, target_name: str | None, explicit_position: Any) -> Any | None:
    if explicit_position is not None:
        return _vector({"target": explicit_position}, "target")
    if target_name:
        direct = _vector(obs, f"{target_name}_pos")
        if direct is not None:
            return direct
    return _vector(obs, "obj_pos")


def _distance(obs: Any, *, target_name: str | None, explicit_position: Any, offset: Any | None) -> float | None:
    if not isinstance(obs, dict):
        return None
    try:
        import numpy as np
    except Exception:
        return None

    eef = _vector(obs, "robot0_eef_pos")
    target = _target_position(obs, target_name=target_name, explicit_position=explicit_position)
    if eef is None or target is None:
        return None
    return float(np.linalg.norm((target + (offset if offset is not None else np.zeros(3, dtype=np.float32))) - eef))


def _object_to_eef_distance(obs: Any, *, object_name: Any) -> float | None:
    if not isinstance(obs, dict) or not isinstance(object_name, str):
        return None
    try:
        import numpy as np
    except Exception:
        return None

    eef = _vector(obs, "robot0_eef_pos")
    object_position = _target_position(obs, target_name=object_name, explicit_position=None)
    if eef is None or object_position is None:
        return None
    return float(np.linalg.norm(object_position - eef))


def _object_to_target_distance(
    obs: Any,
    *,
    object_name: Any,
    target_name: str | None,
    explicit_target_position: Any,
    offset: Any | None,
) -> float | None:
    if not isinstance(obs, dict) or not isinstance(object_name, str):
        return None
    try:
        import numpy as np
    except Exception:
        return None

    object_position = _target_position(obs, target_name=object_name, explicit_position=None)
    target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
    if object_position is None or target_position is None:
        return None
    offset_vector = _coerce_vector_value(offset)
    if offset_vector is None:
        offset_vector = np.zeros(3, dtype=np.float32)
    return float(np.linalg.norm((target_position + offset_vector) - object_position))


def _object_to_target_xy_distance(
    obs: Any,
    *,
    object_name: Any,
    target_name: str | None,
    explicit_target_position: Any,
    offset: Any | None,
) -> float | None:
    if not isinstance(obs, dict) or not isinstance(object_name, str):
        return None
    try:
        import numpy as np
    except Exception:
        return None

    object_position = _target_position(obs, target_name=object_name, explicit_position=None)
    target_position = _target_position(obs, target_name=target_name, explicit_position=explicit_target_position)
    if object_position is None or target_position is None:
        return None
    offset_vector = _coerce_vector_value(offset)
    if offset_vector is None:
        offset_vector = np.zeros(3, dtype=np.float32)
    return float(np.linalg.norm((target_position[:2] + offset_vector[:2]) - object_position[:2]))


def _activate_robocasa_asset_cache(asset_cache_dir: str | None) -> str | None:
    if not asset_cache_dir:
        return None
    import robocasa.models

    src_root = Path(robocasa.models.assets_root).resolve()
    dst_root = Path(asset_cache_dir).expanduser().resolve()
    if src_root == dst_root:
        return str(dst_root)
    _ensure_asset_mirror(src_root, dst_root)
    robocasa.models.assets_root = str(dst_root)

    for module_name in (
        "robocasa.models.objects.kitchen_object_utils",
        "robocasa.models.objects.kitchen_objects",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "BASE_ASSET_ZOO_PATH"):
            setattr(module, "BASE_ASSET_ZOO_PATH", str(dst_root / "objects"))
        if module is not None and hasattr(module, "OBJ_CATEGORIES"):
            _rewrite_object_registry_paths(module.OBJ_CATEGORIES, src_root / "objects", dst_root / "objects")
    return str(dst_root)


def _configure_external_numba_cache() -> str:
    """Keep RoboSuite's lazy Numba compilation outside the sealed prefix."""

    configured = os.environ.get("NUMBA_CACHE_DIR")
    if configured:
        return configured
    cache_dir = Path(tempfile.gettempdir()) / f"agentic-embodied-arena-numba-{os.getuid()}"
    os.environ["NUMBA_CACHE_DIR"] = str(cache_dir)
    return str(cache_dir)


def _ensure_asset_mirror(src_root: Path, dst_root: Path) -> None:
    if not src_root.exists():
        raise RuntimeError(f"RoboCasa asset source does not exist: {src_root}")
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "README.md",
        "arenas",
        "box_links",
        "fixtures",
        "generative_textures",
        "groot_dataset_assets",
        "novel_instructions",
        "scenes",
        "textures",
    ):
        _mirror_child(src_root / name, dst_root / name)
    objects_src = src_root / "objects"
    objects_dst = dst_root / "objects"
    objects_dst.mkdir(exist_ok=True)
    _mirror_child(objects_src / "README.md", objects_dst / "README.md")
    _mirror_child(objects_src / "lightwheel", objects_dst / "lightwheel")
    # RoboCasa and RoboCasa365 select different Objaverse categories at reset
    # time (for example ``mug`` and ``hot_dog``).  Mirroring only the smoke
    # task's category leaves the second suite pointing back at the sealed
    # source tree, where upstream XML path normalization then fails with EROFS.
    _mirror_child(objects_src / "objaverse", objects_dst / "objaverse")


def _mirror_child(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(
            src,
            dst,
            copy_function=_link_or_copy_file,
            symlinks=True,
            dirs_exist_ok=True,
        )
    else:
        _link_or_copy_file(src, dst)


def _rewrite_object_registry_paths(registry: Any, src_objects_root: Path, dst_objects_root: Path) -> None:
    src_prefix = str(src_objects_root)
    dst_prefix = str(dst_objects_root)
    if isinstance(registry, dict):
        values = registry.values()
    else:
        values = registry
    for value in values:
        if isinstance(value, dict):
            _rewrite_object_registry_paths(value, src_objects_root, dst_objects_root)
            continue
        paths = getattr(value, "mjcf_paths", None)
        if isinstance(paths, list):
            rewritten = [
                path.replace(src_prefix, dst_prefix, 1)
                if isinstance(path, str)
                else path
                for path in paths
            ]
            # RoboCasa constructs its object registry during package import. A
            # source checkout intentionally kept free of downloaded assets
            # therefore produces empty registries before the external cache is
            # activated. Hydrate the ordinary ``<registry>/<category>`` layout
            # from the pinned cache without modifying the source checkout.
            if not rewritten:
                registry_type = getattr(value, "reg_type", None)
                category = getattr(value, "name", None)
                if isinstance(registry_type, str) and isinstance(category, str):
                    category_root = dst_objects_root / registry_type / category
                    excluded = set(getattr(value, "exclude", ()) or ())
                    if category_root.is_dir():
                        rewritten = [
                            str(model_xml)
                            for model_xml in sorted(category_root.glob("*/model.xml"))
                            if model_xml.parent.name not in excluded
                        ]
            value.mjcf_paths = rewritten


def _link_or_copy_file(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> str:
    destination = Path(dst)
    if destination.exists():
        return str(destination)
    # RoboCasa normalizes asset references by rewriting MJCF/XML files while
    # constructing an environment.  A hard link would make those writes reach
    # the sealed canonical asset inode, so mutable metadata must be copied.
    if Path(src).suffix.lower() == ".xml":
        shutil.copy2(src, dst)
        return str(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return str(dst)


def _make_direct_robosuite_env(config: RoboCasaRuntimeConfig, seed: int | None) -> Any:
    if importlib.util.find_spec("robosuite") is None:
        raise RuntimeError("RoboCasa direct state-only runtime requires `robosuite`.")
    import robosuite
    from robosuite.controllers import load_composite_controller_config

    env_name = config.env_id.removeprefix("robocasa/")
    split_kwargs: JsonDict
    if config.split == "target":
        split_kwargs = {
            "obj_instance_split": "target",
            "layout_and_style_ids": list(zip(range(1, 11), range(1, 11))),
        }
    elif config.split == "pretrain":
        split_kwargs = {"obj_instance_split": "pretrain", "layout_ids": -2, "style_ids": -2}
    elif config.split == "all":
        split_kwargs = {"obj_instance_split": None, "layout_ids": -3, "style_ids": -3}
    else:
        split_kwargs = {}

    controller_config = load_composite_controller_config(
        controller=None,
        robot=config.robot if isinstance(config.robot, str) else config.robot[0],
    )
    env_kwargs = dict(config.env_kwargs)
    camera_segmentations = env_kwargs.pop("camera_segmentations", None)
    kwargs = {
        "env_name": env_name,
        "robots": config.robot,
        "controller_configs": controller_config,
        "camera_names": config.camera_names,
        "camera_widths": config.camera_widths,
        "camera_heights": config.camera_heights,
        "has_renderer": config.render_onscreen,
        "has_offscreen_renderer": bool(config.camera_names),
        "ignore_done": True,
        "use_object_obs": True,
        "use_camera_obs": bool(config.camera_names),
        "camera_depths": config.camera_depths,
        "seed": seed,
        **split_kwargs,
        **env_kwargs,
    }
    env = robosuite.make(**kwargs)
    if camera_segmentations is not None:
        setattr(env, "_agent_runtime_camera_segmentations", camera_segmentations)
    return env


def _inject_direct_robocasa_segmentation(env: Any, obs: Any, config: RoboCasaRuntimeConfig) -> Any:
    """Render native MuJoCo segmentation for RoboCasa constructors that omit this kwarg."""

    requested = config.env_kwargs.get("camera_segmentations")
    if env is None or not isinstance(obs, dict) or requested is None:
        return obs
    modes = [str(requested)] if isinstance(requested, str) else [str(item) for item in requested]
    if not modes:
        return obs
    mode = modes[0]
    if mode not in {"instance", "class", "element"}:
        return obs
    unwrapped = _unwrap_env(env)
    sim = getattr(unwrapped, "sim", None)
    task_model = getattr(unwrapped, "model", None)
    if sim is None or task_model is None or not callable(getattr(sim, "render", None)):
        return obs
    if mode == "instance":
        names = list((getattr(task_model, "instances_to_ids", None) or {}).keys())
        geom_to_group = getattr(task_model, "geom_ids_to_instances", None) or {}
    elif mode == "class":
        names = list((getattr(task_model, "classes_to_ids", None) or {}).keys())
        geom_to_group = getattr(task_model, "geom_ids_to_classes", None) or {}
    else:
        names = []
        geom_to_group = {}
    name_to_id = {str(name): index for index, name in enumerate(names)}
    geom_mapping = {int(geom_id): name_to_id[str(name)] for geom_id, name in geom_to_group.items() if str(name) in name_to_id}
    augmented = dict(obs)
    for camera_name in config.camera_names:
        key = f"{camera_name}_segmentation_{mode}"
        if key in augmented:
            continue
        try:
            rendered = np.asarray(
                sim.render(
                    camera_name=camera_name,
                    width=int(config.camera_widths),
                    height=int(config.camera_heights),
                    depth=False,
                    segmentation=True,
                )
            )
        except Exception:
            continue
        if rendered.ndim != 3 or rendered.shape[-1] < 2:
            continue
        geom_ids = rendered[::-1, :, 1].astype(np.int64, copy=False)
        if mode == "element":
            segmentation = geom_ids.astype(np.int32, copy=False)
        else:
            segmentation = np.fromiter(
                (geom_mapping.get(int(value), -1) + 1 for value in geom_ids.flat),
                dtype=np.int32,
                count=geom_ids.size,
            ).reshape(geom_ids.shape)
        augmented[key] = segmentation[..., None]
    return augmented


def _direct_camera_segmentations(segmentations: Any, camera_count: int) -> list[list[str] | None]:
    if segmentations is None:
        return [None] * max(0, camera_count)
    if isinstance(segmentations, str):
        requested = [segmentations]
    elif isinstance(segmentations, (list, tuple)):
        requested = [str(value) for value in segmentations if value is not None]
    else:
        requested = [str(segmentations)]
    return [list(requested) for _ in range(max(0, camera_count))]


def _unwrap_env(env: Any) -> Any:
    candidates = _env_candidates(env)
    return candidates[-1] if candidates else env


def _pose_to_list(obj: Any) -> list[float] | None:
    pose = getattr(obj, "pose", None)
    if pose is None:
        return None
    p = getattr(pose, "p", None)
    q = getattr(pose, "q", None)
    if p is not None and q is not None:
        return _flatten_numeric_list(_to_builtin(p)) + _flatten_numeric_list(_to_builtin(q))
    as_list = _to_builtin(pose)
    if isinstance(as_list, list):
        return _flatten_numeric_list(as_list)
    return None


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _flatten_numeric_list(value: Any) -> list[float]:
    out: list[float] = []
    if isinstance(value, list):
        for item in value:
            out.extend(_flatten_numeric_list(item))
    elif isinstance(value, (int, float, bool)):
        out.append(float(value))
    return out
