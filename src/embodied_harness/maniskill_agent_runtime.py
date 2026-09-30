from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass, field
import importlib.util
import inspect
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .backend import EmbodiedBackend
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]
_SAPIEN_VULKAN_LOADER: Any | None = None


@dataclass(slots=True)
class ManiSkillRuntimeConfig:
    """Runtime configuration without a built-in benchmark case or answer prior."""

    env_id: str = ""
    task_instruction: str = ""
    obs_mode: str = "rgb+depth+segmentation"
    control_mode: str | None = None
    render_mode: str | None = None
    num_envs: int = 1
    live: bool = True
    sim_backend: str = "physx_cpu"
    render_backend: str | None = "gpu"
    env_kwargs: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class GroundingCandidate:
    """Geometry-only candidate retained for import compatibility."""

    label: str
    score: float
    pose_world: list[float] | None = None
    bbox_xyxy: list[int] | None = None
    camera_uid: str | None = None
    segmentation_id: int | None = None
    mask_artifact: str | None = None
    evidence: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


class ManiSkillAgentRuntimeBackend(EmbodiedBackend):
    """Prior-free ManiSkill visual runtime for coding-agent composition.

    The public surface contains native observations, calibration, generic
    segmentation/depth geometry, and generic controller actions. It contains
    no object-role aliases, target selector, task skill, fixed pose, or action
    sequence. Official success remains harness-only through :meth:`verify`.
    """

    def __init__(self, config: ManiSkillRuntimeConfig | None = None, env_factory: EnvFactory | None = None) -> None:
        self.config = config or ManiSkillRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._last_terminated: Any = False
        self._last_truncated: Any = False
        self._observation_generation = 0
        self._no_render_visual_fallback_active = False
        self._vulkan_loader_path: str | None = None
        self._pool_task: tuple[str, int, str] | None = None
        self._episode_return = 0.0
        self._native_steps = 0

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        import mani_skill.envs  # noqa: F401
        from mani_skill.utils.registration import REGISTERED_ENVS

        task = str(coordinate.get("task_id") or "")
        seed = coordinate.get("seed")
        spec = REGISTERED_ENVS.get(task)
        if (spec is None or task == "Empty-v1"
                or not spec.cls.__module__.startswith("mani_skill.envs.tasks.")
                or coordinate.get("variation") != task):
            raise ValueError("ManiSkill coordinate must name an original task environment")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("ManiSkill seed must be an integer in [0, 2**32)")
        description = inspect.getdoc(spec.cls) or task
        # Publish the native task description without private evaluator state.
        description = description.split("**Task Description:**", 1)[-1].split("**Randomizations:**", 1)[0].strip()
        self._pool_task = (task, seed, description)
        return {"bound": True, "mode": "native_task_environment", "env_id": task,
                "seed": seed, "native_max_episode_steps": spec.max_episode_steps,
                "actual_task_id": f"maniskill:{task}:seed_{seed}"}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_task is not None:
            selected, selected_seed, instruction = self._pool_task
            if seed is not None and seed != selected_seed:
                raise ValueError("ManiSkill reset seed differs from the bound coordinate")
            seed = selected_seed
            task_id = f"maniskill:{selected}:seed_{seed}"
            overrides.update(env_id=selected, task_instruction=instruction, control_mode=None,
                             env_kwargs={})
        runtime_config = self._merged_config(overrides)
        if not runtime_config.env_id:
            raise ValueError("ManiSkill env_id must be supplied by the benchmark caller.")
        if not runtime_config.task_instruction:
            raise ValueError("ManiSkill task_instruction must be supplied by the benchmark case.")
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:maniskill:live_runtime",
            instruction=runtime_config.task_instruction,
            goal={"env_id": runtime_config.env_id, "success_source": "harness_only"},
            budgets={"primitive_calls": 64, "verifier_calls": 4},
            tags=["w4", "maniskill", "visual", "prior_free", "live" if runtime_config.live else "dry"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "maniskill",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "native_rgb_depth_segmentation": True,
                    "camera_calibration": True,
                    "segmentation_id_geometry": True,
                    "generic_controller_actions": True,
                    "task_or_case_prior": False,
                    "object_role_names": False,
                    "fixed_bbox_or_pose": False,
                    "prearranged_action_sequence": False,
                    "private_success_exposed": False,
                },
            },
        )
        self._last_info = {}
        self._episode_return = 0.0
        self._native_steps = 0
        self._last_terminated = False
        self._last_truncated = False
        self._observation_generation = 0
        if runtime_config.live:
            self._env = self._make_env(runtime_config)
            self._last_obs, self._last_info = _split_reset_result(self._env.reset(seed=seed))
            horizon = getattr(getattr(self._env, "spec", None), "max_episode_steps", None)
            self._task_spec.metadata["native_evaluation_protocol"] = {
                "max_episode_steps": horizon,
                "reports_success": any(key in self._last_info for key in ("success", "is_success")),
                "episode_return": "Sum of native rewards; continue until native termination or the native time limit.",
                "control_loop": "Use Python loops for repeated action/state feedback within a code cell; obey the harness call budget.",
            }
        else:
            self._env = None
            self._last_obs = {}
        self.record_event("reset", {"task": self._task_spec.to_dict(), "seed": seed, "runtime": self.runtime_available()})
        return self._task_spec

    def observe(self) -> Observation:
        self._require_reset()
        public_state = _public_state_observation(self._last_obs)
        data = {
            "runtime": self.runtime_available(),
            "observation_summary": summarize_observation(public_state),
            "goal_xyz": _extract_public_goal_xyz(public_state),
            "visual_runtime": summarize_visual_runtime(self._last_obs),
        }
        obs = Observation(step=len(self.get_trace().events), data=data, metadata={"benchmark_id": "maniskill"})
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_maniskill_state",
                "L1",
                {"prompt": "str|None", "agent_context": "dict|None", "include_raw": "bool"},
                {"state_summary": "dict", "goal_xyz": "list[float]|None", "raw_observation": "dict|None"},
                (
                    "Return the current public observation without interpreting task roles. "
                    "When the upstream observation exposes a public goal/target position, goal_xyz is normalized to a 3-float list."
                ),
            ),
            self._primitive_card(
                "observe_maniskill_visual",
                "L1",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_uid": "str|None", "include_raw": "bool"},
                {"sensor_data": "dict", "camera_calibration": "dict", "instances": "dict", "visual_runtime": "dict"},
                "Return native RGB/depth/segmentation, camera calibration, and geometry-only instance measurements.",
            ),
            self._primitive_card(
                "inspect_maniskill_visual_readiness",
                "L1",
                {"camera_uid": "str|None"},
                {"visual_ready": "bool", "modalities_by_camera": "dict", "calibration_by_camera": "dict", "blockers": "list"},
                "Report modality and calibration availability without suggesting a target or action sequence.",
            ),
            self._primitive_card(
                "list_maniskill_instances",
                "L2",
                {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_uid": "str|None", "min_pixel_count": "int"},
                {"instances": "dict[str,list[dict]]", "evidence_id": "str"},
                (
                    "Enumerate all segmentation IDs and measured pixel/depth geometry; no semantic target is selected. "
                    "After selecting a candidate, call inspect_maniskill_instance for the exact segmentation_id before acting."
                ),
            ),
            self._primitive_card(
                "inspect_maniskill_instance",
                "L2",
                {"segmentation_id": "int", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "camera_uid": "str|None"},
                {"measurements": "list[dict]", "evidence_id": "str"},
                (
                    "Measure one caller-selected segmentation ID using only current depth and camera calibration. "
                    "Pass the returned evidence_id to downstream motion or gripper primitives."
                ),
            ),
            self._primitive_card(
                "detect_maniskill_color_regions",
                "L2",
                {
                    "reference_rgb": "list[float]",
                    "tolerance": "float",
                    "camera_uid": "str|None",
                    "min_pixel_count": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                {"regions": "dict[str,list[dict]]", "evidence_id": "str"},
                "Detect connected RGB regions for a caller-supplied color and measure them with native depth/calibration.",
            ),
            self._primitive_card(
                "apply_maniskill_action",
                "L3",
                {"action": "array", "repeat": "int", "agent_context": "dict|None", "evidence_ids": "list[str]|None"},
                {"executed": "bool", "steps": "int", "action": "array", "termination": "dict"},
                "Apply a caller-authored action to the configured ManiSkill controller.",
            ),
            self._primitive_card(
                "move_maniskill_tcp_delta",
                "L3",
                {"delta_xyz": "list[float]", "gripper": "float", "repeat": "int", "agent_context": "dict|None", "evidence_ids": "list[str]|None"},
                {"executed": "bool", "steps": "int", "actions": "list"},
                "Apply a caller-authored Cartesian delta and gripper command; performs no approach, grasp, or placement plan.",
            ),
            self._primitive_card(
                "move_maniskill_tcp_to",
                "L3",
                {
                    "target_xyz": "list[float]",
                    "gripper": "float",
                    "max_steps": "int",
                    "gain": "float",
                    "tolerance": "float",
                    "agent_context": "dict|None",
                    "evidence_ids": "list[str]|None",
                },
                {"executed": "bool", "steps": "int", "target_xyz": "list[float]", "final_tcp_xyz": "list[float]"},
                (
                    "Move the TCP toward a caller-supplied world-space point with caller-controlled convergence parameters. "
                    "The target_xyz should be derived from inspect_maniskill_instance measurements or observe_maniskill_state goal_xyz."
                ),
                preconditions=[
                    "Call observe_maniskill_state to get public state and goal_xyz when available.",
                    "Call list_maniskill_instances and inspect_maniskill_instance before object-directed motion.",
                    "Call record_maniskill_evidence with the selected target/action rationale before motion.",
                    "Pass evidence_ids from inspect_maniskill_instance or list_maniskill_instances.",
                ],
            ),
            self._primitive_card(
                "set_maniskill_gripper",
                "L3",
                {"command": "float", "repeat": "int", "agent_context": "dict|None", "evidence_ids": "list[str]|None"},
                {"executed": "bool", "steps": "int", "command": "float"},
                "Apply a caller-supplied gripper command while holding Cartesian translation at zero.",
                preconditions=[
                    "Pass evidence_ids from the inspected object when the gripper command is object-directed.",
                ],
            ),
            self._primitive_card(
                "observe_maniskill_control_state",
                "L1",
                {"agent_context": "dict|None"},
                {"tcp_pose": "dict|None", "qpos": "array|None", "qvel": "array|None"},
                "Return public robot control state for closed-loop action composition.",
            ),
            self._primitive_card(
                "observe_maniskill_termination",
                "L1",
                {"agent_context": "dict|None"},
                {"terminated": "bool|list", "truncated": "bool|list", "episode_done": "bool"},
                "Return only public Gymnasium termination flags, never private task success.",
            ),
            self._primitive_card(
                "record_maniskill_evidence",
                "L1",
                {"key": "str", "value": "any"},
                {"artifact_id": "str"},
                "Store caller-selected evidence in the episode trace before executing object-directed actions.",
            ),
        ]
        if self._pool_task is not None and self.config.control_mode != "pd_ee_delta_pos":
            cards = [card for card in cards if card.name not in {
                "move_maniskill_tcp_delta", "move_maniskill_tcp_to", "set_maniskill_gripper"}]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        allowed = {card.name for card in self.list_primitives()}
        if name not in allowed:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            result = handler(**kwargs) if handler else PrimitiveResult(name=name, ok=False, error="missing_handler")
        self.record_event("primitive_call", {"name": name, "kwargs": _to_builtin(kwargs), "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported ManiSkill verification scope: {scope}")
        else:
            success = _extract_success(self._last_info)
            has_success = any(key in self._last_info for key in ("success", "is_success"))
            result = VerificationResult(
                ok=bool(success),
                scope="task",
                message=("ManiSkill task success reported by env info" if success else
                         "ManiSkill env has not reported task success" if has_success else
                         "Native reward task; report episode return without inventing a binary success threshold"),
                metrics={**({"success": float(bool(success))} if has_success else {}),
                         "native_episode_return": self._episode_return, "native_steps": self._native_steps},
                metadata={"info_summary": _to_builtin(self._last_info),
                          "native_episode_done": self._termination_output()["episode_done"],
                          "native_success_defined": has_success},
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
        space = getattr(self._env, "action_space", None)
        agent = getattr(_unwrap_env(self._env), "agent", None) if self._env is not None else None
        return {
            "live": self.config.live,
            "env_created": self._env is not None,
            "mani_skill_importable": importlib.util.find_spec("mani_skill") is not None,
            "gymnasium_importable": importlib.util.find_spec("gymnasium") is not None,
            "no_render_visual_fallback_active": self._no_render_visual_fallback_active,
            "sapien_vulkan_loader": self._vulkan_loader_path,
            "native_control_mode": getattr(agent, "control_mode", self.config.control_mode),
            "native_action_space": ({"shape": list(space.shape), "low": _to_builtin(space.low),
                                     "high": _to_builtin(space.high)} if space is not None and hasattr(space, "low") else None),
        }

    def _merged_config(self, overrides: JsonDict) -> ManiSkillRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return ManiSkillRuntimeConfig(**data)

    def _make_env(self, config: ManiSkillRuntimeConfig) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.env_id, config.to_dict())
        if importlib.util.find_spec("mani_skill") is None or importlib.util.find_spec("gymnasium") is None:
            raise RuntimeError("ManiSkill live runtime requires `mani_skill` and `gymnasium`.")
        self._vulkan_loader_path = _preload_sapien_vulkan_loader()
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        _install_maniskill_pci_render_backend_parser_patch()
        kwargs = dict(config.env_kwargs)
        kwargs.update(
            obs_mode=config.obs_mode,
            control_mode=config.control_mode,
            num_envs=config.num_envs,
            sim_backend=config.sim_backend,
            render_backend=config.render_backend,
        )
        if config.render_mode is not None:
            kwargs["render_mode"] = config.render_mode
        return gym.make(config.env_id, **kwargs)

    def _primitive_observe_maniskill_state(
        self, prompt: str | None = None, agent_context: JsonDict | None = None, include_raw: bool = False
    ) -> PrimitiveResult:
        output: JsonDict = {
            "prompt": prompt,
            "agent_context": agent_context or {},
            "runtime": self.runtime_available(),
            "state_summary": summarize_observation(_public_state_observation(self._last_obs)),
            "goal_xyz": _extract_public_goal_xyz(_public_state_observation(self._last_obs)),
        }
        if include_raw:
            output["raw_observation"] = _to_builtin(_public_state_observation(self._last_obs))
        return PrimitiveResult(name="observe_maniskill_state", ok=True, output=output)

    def _primitive_observe_maniskill_visual(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        camera_uid: str | None = None,
        include_raw: bool = True,
    ) -> PrimitiveResult:
        sensor_data = extract_sensor_data(self._last_obs, camera_uid=camera_uid, include_raw=include_raw)
        calibration = extract_camera_calibration(self._last_obs, camera_uid=camera_uid, include_raw=include_raw)
        instances = extract_sensor_segmentation_instances(self._last_obs, camera_uid=camera_uid)
        visual_runtime = summarize_visual_runtime(self._last_obs, camera_uid=camera_uid)
        artifact_id = f"maniskill:visual:{len(self.get_trace().artifacts)}"
        self.get_trace().add_artifact(
            artifact_id,
            {
                "kind": "visual_grounding",
                "source": "native_observation",
                "observation_generation": self._observation_generation,
                "sensor_data": sensor_data,
                "camera_calibration": calibration,
                "instances": instances,
            },
        )
        return PrimitiveResult(
            name="observe_maniskill_visual",
            ok=bool(visual_runtime["visual_ready"]),
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "sensor_data": sensor_data,
                "camera_calibration": calibration,
                "instances": instances,
                "visual_runtime": visual_runtime,
                "evidence_id": artifact_id,
            },
            artifacts=[artifact_id],
            error=None if visual_runtime["visual_ready"] else "visual_observation_unavailable",
        )

    def _primitive_inspect_maniskill_visual_readiness(self, camera_uid: str | None = None) -> PrimitiveResult:
        runtime = summarize_visual_runtime(self._last_obs, camera_uid=camera_uid)
        calibration = extract_camera_calibration(self._last_obs, camera_uid=camera_uid, include_raw=False)
        calibration_by_camera = {
            uid: sorted(params.get("parameters", {})) for uid, params in calibration.items()
        }
        blockers: list[JsonDict] = []
        if not runtime["visual_ready"]:
            blockers.append({"code": "sensor_data_missing"})
        for modality in ("rgb", "depth", "segmentation"):
            if not runtime[f"has_{modality}"]:
                blockers.append({"code": f"{modality}_missing"})
        if not calibration:
            blockers.append({"code": "camera_calibration_missing"})
        output = {
            **runtime,
            "calibration_by_camera": calibration_by_camera,
            "geometry_ready": bool(runtime["has_depth"] and runtime["has_segmentation"] and calibration),
            "blockers": blockers,
        }
        return PrimitiveResult(name="inspect_maniskill_visual_readiness", ok=not blockers, output=output, error=None if not blockers else "visual_not_ready")

    def _primitive_list_maniskill_instances(
        self,
        camera_uid: str | None = None,
        min_pixel_count: int = 1,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        instances = extract_sensor_segmentation_instances(
            self._last_obs, camera_uid=camera_uid, min_pixel_count=min_pixel_count
        )
        evidence_id = f"maniskill:instances:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "observation_generation": self._observation_generation,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "instances": instances,
        }
        if instances:
            self.get_trace().add_artifact(evidence_id, payload)
        return PrimitiveResult(
            name="list_maniskill_instances",
            ok=bool(instances),
            output={**payload, "camera_uids": sorted(instances), "evidence_id": evidence_id if instances else None},
            artifacts=[evidence_id] if instances else [],
            error=None if instances else "segmentation_unavailable",
        )

    def _primitive_inspect_maniskill_instance(
        self,
        segmentation_id: int,
        camera_uid: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        all_instances = extract_sensor_segmentation_instances(self._last_obs, camera_uid=camera_uid)
        measurements = [
            instance
            for instances in all_instances.values()
            for instance in instances
            if int(instance["segmentation_id"]) == int(segmentation_id)
        ]
        evidence_id = f"maniskill:instance:{int(segmentation_id)}:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "observation_generation": self._observation_generation,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "segmentation_id": int(segmentation_id),
            "measurements": measurements,
        }
        if measurements:
            self.get_trace().add_artifact(evidence_id, payload)
        return PrimitiveResult(
            name="inspect_maniskill_instance",
            ok=bool(measurements),
            output={**payload, "evidence_id": evidence_id if measurements else None},
            artifacts=[evidence_id] if measurements else [],
            error=None if measurements else "segmentation_id_not_visible",
        )

    def _primitive_detect_maniskill_color_regions(
        self,
        reference_rgb: list[float],
        tolerance: float,
        camera_uid: str | None = None,
        min_pixel_count: int = 1,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if len(reference_rgb) != 3 or tolerance < 0 or min_pixel_count < 1:
            return PrimitiveResult(name="detect_maniskill_color_regions", ok=False, error="invalid_color_arguments")
        regions = detect_color_regions(
            self._last_obs,
            reference_rgb=reference_rgb,
            tolerance=tolerance,
            camera_uid=camera_uid,
            min_pixel_count=min_pixel_count,
        )
        evidence_id = f"maniskill:regions:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "observation_generation": self._observation_generation,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "reference_rgb": [float(value) for value in reference_rgb],
            "tolerance": float(tolerance),
            "regions": regions,
        }
        if regions:
            self.get_trace().add_artifact(evidence_id, payload)
        return PrimitiveResult(
            name="detect_maniskill_color_regions",
            ok=bool(regions),
            output={**payload, "evidence_id": evidence_id if regions else None},
            artifacts=[evidence_id] if regions else [],
            error=None if regions else "rgb_unavailable",
        )

    def _primitive_apply_maniskill_action(
        self,
        action: Any,
        repeat: int = 1,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        evidence_refs, evidence_error = self._resolve_visual_evidence(evidence_ids)
        if evidence_error is not None:
            return PrimitiveResult(name="apply_maniskill_action", ok=False, error=evidence_error)
        if self._env is None:
            return PrimitiveResult(name="apply_maniskill_action", ok=False, error="live_env_missing")
        if repeat < 1:
            return PrimitiveResult(name="apply_maniskill_action", ok=False, error="repeat_must_be_positive")
        steps = 0
        try:
            for _ in range(int(repeat)):
                self._step_action(action)
                steps += 1
                if self._termination_output()["episode_done"]:
                    break
        except Exception as exc:
            return PrimitiveResult(name="apply_maniskill_action", ok=False, error=f"action_failed: {exc}")
        return PrimitiveResult(
            name="apply_maniskill_action",
            ok=True,
            output={
                "executed": True,
                "steps": steps,
                "action": _to_builtin(action),
                "agent_context": agent_context or {},
                "evidence_ids": evidence_refs,
                "termination": self._termination_output(),
                "observation_summary": summarize_observation(_public_state_observation(self._last_obs)),
                "visual_runtime": summarize_visual_runtime(self._last_obs),
            },
        )

    def _primitive_move_maniskill_tcp_delta(
        self,
        delta_xyz: list[float],
        gripper: float,
        repeat: int = 1,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        if self.config.control_mode != "pd_ee_delta_pos":
            return PrimitiveResult(name="move_maniskill_tcp_delta", ok=False, error="control_mode_must_be_pd_ee_delta_pos")
        if len(delta_xyz) != 3:
            return PrimitiveResult(name="move_maniskill_tcp_delta", ok=False, error="delta_xyz_must_have_three_values")
        action = [float(delta_xyz[0]), float(delta_xyz[1]), float(delta_xyz[2]), float(gripper)]
        result = self._primitive_apply_maniskill_action(
            action=action, repeat=repeat, agent_context=agent_context, evidence_ids=evidence_ids
        )
        return PrimitiveResult(
            name="move_maniskill_tcp_delta",
            ok=result.ok,
            output={**result.output, "delta_xyz": action[:3], "gripper": action[3]},
            error=result.error,
        )

    def _primitive_move_maniskill_tcp_to(
        self,
        target_xyz: list[float],
        gripper: float,
        max_steps: int = 40,
        gain: float = 12.0,
        tolerance: float = 0.006,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        import numpy as np

        if self.config.control_mode != "pd_ee_delta_pos":
            return PrimitiveResult(name="move_maniskill_tcp_to", ok=False, error="control_mode_must_be_pd_ee_delta_pos")
        if len(target_xyz) != 3 or max_steps < 1 or gain <= 0 or tolerance <= 0:
            return PrimitiveResult(name="move_maniskill_tcp_to", ok=False, error="invalid_motion_arguments")
        evidence_refs, evidence_error = self._resolve_visual_evidence(evidence_ids)
        if evidence_error is not None:
            return PrimitiveResult(name="move_maniskill_tcp_to", ok=False, error=evidence_error)
        target = np.asarray(target_xyz, dtype=float)
        steps = 0
        try:
            for _ in range(int(max_steps)):
                current = np.asarray(self._tcp_position(), dtype=float)
                delta = target - current
                if float(np.linalg.norm(delta)) <= float(tolerance):
                    break
                command = np.clip(delta * float(gain), -1.0, 1.0)
                self._step_action([float(command[0]), float(command[1]), float(command[2]), float(gripper)])
                steps += 1
                if self._termination_output()["episode_done"]:
                    break
            final_tcp = self._tcp_position()
        except Exception as exc:
            return PrimitiveResult(name="move_maniskill_tcp_to", ok=False, error=f"motion_failed: {exc}")
        return PrimitiveResult(
            name="move_maniskill_tcp_to",
            ok=True,
            output={
                "executed": True,
                "steps": steps,
                "target_xyz": [float(value) for value in target],
                "final_tcp_xyz": final_tcp,
                "position_error": float(np.linalg.norm(target - np.asarray(final_tcp, dtype=float))),
                "gripper": float(gripper),
                "gain": float(gain),
                "tolerance": float(tolerance),
                "agent_context": agent_context or {},
                "evidence_ids": evidence_refs,
                "termination": self._termination_output(),
            },
        )

    def _primitive_set_maniskill_gripper(
        self,
        command: float,
        repeat: int = 1,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        result = self._primitive_move_maniskill_tcp_delta(
            delta_xyz=[0.0, 0.0, 0.0],
            gripper=float(command),
            repeat=repeat,
            agent_context=agent_context,
            evidence_ids=evidence_ids,
        )
        return PrimitiveResult(
            name="set_maniskill_gripper",
            ok=result.ok,
            output={**result.output, "command": float(command)},
            error=result.error,
        )

    def _primitive_observe_maniskill_control_state(
        self, agent_context: JsonDict | None = None
    ) -> PrimitiveResult:
        unwrapped = _unwrap_env(self._env)
        agent = getattr(unwrapped, "agent", None)
        robot = getattr(agent, "robot", None)
        tcp_pose = getattr(agent, "tcp_pose", None)
        qpos = robot.get_qpos() if robot is not None and hasattr(robot, "get_qpos") else None
        qvel = robot.get_qvel() if robot is not None and hasattr(robot, "get_qvel") else None
        return PrimitiveResult(
            name="observe_maniskill_control_state",
            ok=agent is not None,
            output={
                "tcp_pose": _pose_components(tcp_pose),
                "qpos": _to_builtin(qpos),
                "qvel": _to_builtin(qvel),
                "agent_context": agent_context or {},
            },
            error=None if agent is not None else "agent_control_state_unavailable",
        )

    def _primitive_observe_maniskill_termination(self, agent_context: JsonDict | None = None) -> PrimitiveResult:
        return PrimitiveResult(
            name="observe_maniskill_termination",
            ok=True,
            output={**self._termination_output(), "agent_context": agent_context or {}},
        )

    def _primitive_record_maniskill_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"maniskill:evidence:{key}:{len(self.get_trace().artifacts)}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value)})
        return PrimitiveResult(name="record_maniskill_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

    def _step_action(self, action: Any) -> None:
        if self._env is None:
            raise RuntimeError("ManiSkill environment is not initialized.")
        if self._termination_output()["episode_done"]:
            raise RuntimeError("Native ManiSkill episode has ended; no further actions are allowed")
        native_action = _coerce_maniskill_action(action)
        obs, reward, terminated, truncated, info = _split_step_result(self._env.step(native_action))
        import numpy as np
        self._episode_return += float(np.asarray(_to_builtin(reward)).reshape(-1)[0])
        self._native_steps += 1
        self._last_obs = obs
        self._last_info = info
        self._last_terminated = terminated
        self._last_truncated = truncated
        self._observation_generation += 1

    def _tcp_position(self) -> list[float]:
        agent = getattr(_unwrap_env(self._env), "agent", None)
        pose = getattr(agent, "tcp_pose", None)
        components = _pose_components(pose)
        position = components.get("p") if components else None
        if isinstance(position, list) and position and isinstance(position[0], list):
            position = position[0]
        if not isinstance(position, list) or len(position) < 3:
            raise RuntimeError("ManiSkill agent does not expose tcp_pose.p")
        return [float(value) for value in position[:3]]

    def _termination_output(self) -> JsonDict:
        terminated = _to_builtin(self._last_terminated)
        truncated = _to_builtin(self._last_truncated)
        return {
            "terminated": terminated,
            "truncated": truncated,
            "episode_done": _any_truthy(terminated) or _any_truthy(truncated),
        }

    def _resolve_visual_evidence(self, evidence_ids: list[str] | None) -> tuple[list[str], str | None]:
        refs = [str(value) for value in (evidence_ids or [])]
        for evidence_id in refs:
            payload = self.get_trace().artifacts.get(evidence_id)
            if not isinstance(payload, dict) or payload.get("kind") != "visual_grounding":
                return refs, f"visual_evidence_not_found: {evidence_id}"
            if payload.get("observation_generation") != self._observation_generation:
                return refs, f"visual_evidence_stale: {evidence_id}"
        return refs, None

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
            capability_tags=["w4", "maniskill", "prior_free"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=list(preconditions or []),
            cost={"primitive_calls": 1},
            failure_modes=["runtime_dependency_missing", "wrong_arguments", "observation_unavailable"],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def _preload_sapien_vulkan_loader() -> str | None:
    """Load SAPIEN's bundled Vulkan loader before SAPIEN opens the NVIDIA ICD."""

    global _SAPIEN_VULKAN_LOADER
    if not sys.platform.startswith("linux"):
        return None
    if _SAPIEN_VULKAN_LOADER is not None:
        return os.environ.get("SAPIEN_VULKAN_LIBRARY_PATH")

    configured = os.environ.get("SAPIEN_VULKAN_LIBRARY_PATH")
    candidates = [Path(configured)] if configured else []
    if not candidates:
        spec = importlib.util.find_spec("sapien")
        package_dirs = list(spec.submodule_search_locations or []) if spec is not None else []
        for package_dir in package_dirs:
            candidates.extend(sorted((Path(package_dir) / "vulkan_library").glob("libvulkan.so.*"), reverse=True))
    loader_path = next((path.resolve() for path in candidates if path.is_file()), None)
    if loader_path is None:
        return None

    try:
        _SAPIEN_VULKAN_LOADER = ctypes.CDLL(
            str(loader_path),
            mode=os.RTLD_NOW | os.RTLD_GLOBAL,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to preload SAPIEN Vulkan loader {loader_path}: {exc}") from exc
    os.environ["SAPIEN_VULKAN_LIBRARY_PATH"] = str(loader_path)
    return str(loader_path)


def _install_maniskill_pci_render_backend_parser_patch() -> None:
    """Keep SAPIEN PCI render device strings intact for ManiSkill 3.0.x."""

    try:
        backend_module = importlib.import_module("mani_skill.envs.utils.system.backend")
    except Exception:
        return
    if getattr(backend_module, "_embodied_pci_render_backend_parser_patch", False):
        return
    original_parse_backend_device_id = backend_module.parse_backend_device_id

    def parse_backend_device_id(backend: str) -> tuple[str, int | None]:
        if isinstance(backend, str) and backend.startswith("pci:"):
            return backend, None
        return original_parse_backend_device_id(backend)

    backend_module.parse_backend_device_id = parse_backend_device_id
    backend_module._embodied_pci_render_backend_parser_patch = True


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


def _public_state_observation(obs: Any) -> JsonDict:
    """Return native public state while filtering verifier-like fields."""

    if not isinstance(obs, dict):
        return {}
    output: JsonDict = {}
    if "agent" in obs:
        output["agent"] = obs["agent"]
    extra = obs.get("extra")
    if isinstance(extra, dict):
        verifier_fields = {
            "success",
            "is_success",
            "task_success",
            "reward",
            "checker",
            "check_success",
            "is_obj_placed",
            "task_complete",
        }
        public_extra = {
            str(key): value
            for key, value in extra.items()
            if str(key).lower() not in verifier_fields
        }
        if public_extra:
            output["extra"] = public_extra
    return output


def _extract_public_goal_xyz(public_obs: Any) -> list[float] | None:
    """Normalize a public target-like 3D coordinate when the simulator exposes one."""

    if not isinstance(public_obs, dict):
        return None
    candidate_keys = (
        "goal_xyz",
        "goal_position",
        "target_xyz",
        "target_position",
        "semantic_target_position",
        "goal_pos",
        "target_pos",
    )
    containers: list[Any] = [public_obs]
    extra = public_obs.get("extra")
    if isinstance(extra, dict):
        containers.insert(0, extra)
    for container in containers:
        if not isinstance(container, dict):
            continue
        for key in candidate_keys:
            if key not in container:
                continue
            xyz = _first_xyz(container[key])
            if xyz is not None:
                return xyz
    return None


def _first_xyz(value: Any) -> list[float] | None:
    builtin = _to_builtin(value)
    if isinstance(builtin, dict):
        for nested_key in ("p", "position", "xyz", "value"):
            xyz = _first_xyz(builtin.get(nested_key))
            if xyz is not None:
                return xyz
        return None
    if isinstance(builtin, (list, tuple)):
        if len(builtin) >= 3 and all(isinstance(item, (int, float)) for item in builtin[:3]):
            return [float(item) for item in builtin[:3]]
        for item in builtin:
            xyz = _first_xyz(item)
            if xyz is not None:
                return xyz
    return None


def extract_camera_summaries(obs: Any, camera_uid: str | None = None) -> dict[str, JsonDict]:
    sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    if not isinstance(sensor_data, dict):
        return {}
    return {
        str(uid): {str(key): summarize_observation(value) for key, value in payload.items()}
        for uid, payload in sensor_data.items()
        if (camera_uid is None or str(uid) == camera_uid) and isinstance(payload, dict)
    }


def extract_sensor_data(obs: Any, camera_uid: str | None = None, include_raw: bool = True) -> dict[str, JsonDict]:
    sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    if not isinstance(sensor_data, dict):
        return {}
    output: dict[str, JsonDict] = {}
    for uid, payload in sensor_data.items():
        if (camera_uid is not None and str(uid) != camera_uid) or not isinstance(payload, dict):
            continue
        output[str(uid)] = {
            "modalities": sorted(str(key) for key in payload),
            "summary": {str(key): summarize_observation(value) for key, value in payload.items()},
        }
        if include_raw:
            output[str(uid)]["raw"] = _to_builtin(payload)
    return output


def extract_camera_calibration(obs: Any, camera_uid: str | None = None, include_raw: bool = True) -> dict[str, JsonDict]:
    sensor_param = obs.get("sensor_param", {}) if isinstance(obs, dict) else {}
    if not isinstance(sensor_param, dict):
        return {}
    output: dict[str, JsonDict] = {}
    for uid, payload in sensor_param.items():
        if (camera_uid is not None and str(uid) != camera_uid) or not isinstance(payload, dict):
            continue
        entry: JsonDict = {
            "parameters": {str(key): summarize_observation(value) for key, value in payload.items()},
            "conventions": {
                "intrinsic_cv": "OpenCV camera intrinsic",
                "extrinsic_cv": "world-to-camera OpenCV transform",
                "cam2world_gl": "camera-to-world OpenGL transform",
            },
        }
        if include_raw:
            entry["raw"] = _to_builtin(payload)
        output[str(uid)] = entry
    return output


def summarize_visual_runtime(obs: Any, camera_uid: str | None = None) -> JsonDict:
    cameras = extract_camera_summaries(obs, camera_uid=camera_uid)
    modalities_by_camera = {uid: sorted(summary) for uid, summary in cameras.items()}
    calibration = extract_camera_calibration(obs, camera_uid=camera_uid, include_raw=False)
    complete = sorted(uid for uid, modalities in modalities_by_camera.items() if {"rgb", "depth", "segmentation"} <= set(modalities))
    return {
        "visual_ready": bool(cameras),
        "camera_uids": sorted(cameras),
        "modalities_by_camera": modalities_by_camera,
        "calibrated_camera_uids": sorted(calibration),
        "rgbd_segmentation_camera_uids": complete,
        "rgbd_segmentation_ready": bool(complete),
        "has_rgb": any("rgb" in modes for modes in modalities_by_camera.values()),
        "has_depth": any("depth" in modes for modes in modalities_by_camera.values()),
        "has_segmentation": any("segmentation" in modes for modes in modalities_by_camera.values()),
    }


def extract_sensor_segmentation_instances(
    obs: Any, camera_uid: str | None = None, min_pixel_count: int = 1
) -> dict[str, list[JsonDict]]:
    import numpy as np

    sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    sensor_param = obs.get("sensor_param", {}) if isinstance(obs, dict) else {}
    if not isinstance(sensor_data, dict):
        return {}
    output: dict[str, list[JsonDict]] = {}
    for uid, payload in sensor_data.items():
        uid = str(uid)
        if (camera_uid is not None and uid != camera_uid) or not isinstance(payload, dict) or "segmentation" not in payload:
            continue
        segmentation = _first_image(payload["segmentation"])
        if segmentation is None:
            continue
        depth = _first_image(payload.get("depth"))
        params = sensor_param.get(uid, {}) if isinstance(sensor_param, dict) else {}
        instances: list[JsonDict] = []
        for raw_id in np.unique(segmentation):
            seg_id = int(raw_id)
            if seg_id == 0:
                continue
            mask = segmentation == raw_id
            ys, xs = np.where(mask)
            if len(xs) < int(min_pixel_count):
                continue
            instance: JsonDict = {
                "camera_uid": uid,
                "segmentation_id": seg_id,
                "pixel_count": int(len(xs)),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid_uv": [float(xs.mean()), float(ys.mean())],
                "source": "native_segmentation",
            }
            rgb = _first_image(payload.get("rgb"))
            if rgb is not None and rgb.ndim == 3 and rgb.shape[-1] >= 3:
                pixels = np.asarray(rgb)[mask, :3]
                instance["rgb_mean"] = [float(value) for value in pixels.mean(axis=0)]
                instance["rgb_median"] = [float(value) for value in np.median(pixels, axis=0)]
            instance.update(_instance_depth_geometry(depth, mask, params, instance["centroid_uv"]))
            instances.append(instance)
        output[uid] = sorted(instances, key=lambda item: int(item["segmentation_id"]))
    return output


def detect_color_regions(
    obs: Any,
    reference_rgb: list[float],
    tolerance: float,
    camera_uid: str | None = None,
    min_pixel_count: int = 1,
) -> dict[str, list[JsonDict]]:
    import numpy as np

    sensor_data = obs.get("sensor_data", {}) if isinstance(obs, dict) else {}
    sensor_param = obs.get("sensor_param", {}) if isinstance(obs, dict) else {}
    output: dict[str, list[JsonDict]] = {}
    reference = np.asarray(reference_rgb, dtype=float)
    for uid, payload in sensor_data.items() if isinstance(sensor_data, dict) else []:
        uid = str(uid)
        if (camera_uid is not None and uid != camera_uid) or not isinstance(payload, dict):
            continue
        rgb = _first_image(payload.get("rgb"))
        if rgb is None or rgb.ndim != 3 or rgb.shape[-1] < 3:
            continue
        color = np.asarray(rgb[..., :3], dtype=float)
        mask = np.linalg.norm(color - reference, axis=-1) <= float(tolerance)
        depth = _first_image(payload.get("depth"))
        segmentation = _first_image(payload.get("segmentation"))
        params = sensor_param.get(uid, {}) if isinstance(sensor_param, dict) else {}
        regions: list[JsonDict] = []
        for component in _connected_components(mask):
            ys, xs = np.where(component)
            if len(xs) < int(min_pixel_count):
                continue
            pixels = color[component]
            region: JsonDict = {
                "camera_uid": uid,
                "pixel_count": int(len(xs)),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "centroid_uv": [float(xs.mean()), float(ys.mean())],
                "rgb_mean": [float(value) for value in pixels.mean(axis=0)],
                "source": "native_rgb_threshold",
            }
            if segmentation is not None:
                ids, counts = np.unique(segmentation[component], return_counts=True)
                region["segmentation_overlap"] = [
                    {"segmentation_id": int(seg_id), "pixel_count": int(count)}
                    for seg_id, count in zip(ids, counts)
                    if int(seg_id) != 0
                ]
            region.update(_instance_depth_geometry(depth, component, params, region["centroid_uv"]))
            regions.append(region)
        output[uid] = sorted(regions, key=lambda item: int(item["pixel_count"]), reverse=True)
    return output


def _connected_components(mask: Any) -> list[Any]:
    import numpy as np

    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    components: list[Any] = []
    height, width = mask.shape
    for y, x in zip(*np.where(mask & ~visited)):
        if visited[y, x]:
            continue
        stack = [(int(y), int(x))]
        visited[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        component = np.zeros(mask.shape, dtype=bool)
        ys, xs = zip(*pixels)
        component[list(ys), list(xs)] = True
        components.append(component)
    return components


def extract_segmentation_bboxes(
    obs: Any, seg_ids: list[int] | None = None, camera_uid: str | None = None
) -> list[JsonDict]:
    wanted = {int(value) for value in seg_ids} if seg_ids is not None else None
    return [
        instance
        for instances in extract_sensor_segmentation_instances(obs, camera_uid=camera_uid).values()
        for instance in instances
        if wanted is None or int(instance["segmentation_id"]) in wanted
    ]


def _instance_depth_geometry(depth: Any, mask: Any, params: Any, centroid_uv: list[float]) -> JsonDict:
    import numpy as np

    if depth is None:
        return {"depth": None, "centroid_camera_m": None, "centroid_world_m": None}
    depth_array = np.asarray(depth)
    if depth_array.ndim == 3 and depth_array.shape[-1] == 1:
        depth_array = depth_array[..., 0]
    values = depth_array[mask]
    values = values[np.isfinite(values) & (values > 0)]
    if values.size == 0:
        return {"depth": None, "centroid_camera_m": None, "centroid_world_m": None}
    scale = 0.001 if np.issubdtype(values.dtype, np.integer) or float(np.nanmedian(values)) > 20.0 else 1.0
    z = float(np.nanmedian(values)) * scale
    result: JsonDict = {
        "depth": {
            "unit": "meter",
            "source_unit_scale": scale,
            "min": float(np.nanmin(values)) * scale,
            "median": z,
            "max": float(np.nanmax(values)) * scale,
        },
        "centroid_camera_m": None,
        "centroid_world_m": None,
    }
    if not isinstance(params, dict) or "intrinsic_cv" not in params:
        return result
    intrinsic = _first_matrix(params["intrinsic_cv"], rows=3, cols=3)
    if intrinsic is None or intrinsic[0, 0] == 0 or intrinsic[1, 1] == 0:
        return result
    u, v = centroid_uv
    camera = np.asarray([(u - intrinsic[0, 2]) * z / intrinsic[0, 0], (v - intrinsic[1, 2]) * z / intrinsic[1, 1], z, 1.0])
    result["centroid_camera_m"] = [float(value) for value in camera[:3]]
    extrinsic = _extrinsic_homogeneous(params)
    if extrinsic is not None:
        try:
            camera_to_world = np.linalg.inv(extrinsic)
            world = camera_to_world @ camera
            result["centroid_world_m"] = [float(value) for value in world[:3]]
            ys, xs = np.where(mask)
            raw_depths = depth_array[mask].astype(float) * scale
            valid = np.isfinite(raw_depths) & (raw_depths > 0)
            xs = xs[valid]
            ys = ys[valid]
            raw_depths = raw_depths[valid]
            camera_points = np.stack(
                [
                    (xs - intrinsic[0, 2]) * raw_depths / intrinsic[0, 0],
                    (ys - intrinsic[1, 2]) * raw_depths / intrinsic[1, 1],
                    raw_depths,
                    np.ones_like(raw_depths),
                ],
                axis=0,
            )
            world_points = (camera_to_world @ camera_points)[:3].T
            result["world_bounds_m"] = {
                "min": [float(value) for value in world_points.min(axis=0)],
                "max": [float(value) for value in world_points.max(axis=0)],
                "center": [float(value) for value in ((world_points.min(axis=0) + world_points.max(axis=0)) / 2.0)],
            }
            result["world_median_m"] = [float(value) for value in np.median(world_points, axis=0)]
        except np.linalg.LinAlgError:
            pass
    return result


def _extrinsic_homogeneous(params: Any) -> Any | None:
    import numpy as np

    if not isinstance(params, dict) or "extrinsic_cv" not in params:
        return None
    extrinsic = _first_matrix(params["extrinsic_cv"], rows=4, cols=4)
    if extrinsic is not None:
        return extrinsic
    extrinsic_3x4 = _first_matrix(params["extrinsic_cv"], rows=3, cols=4)
    if extrinsic_3x4 is None:
        return None
    return np.vstack([extrinsic_3x4, [0.0, 0.0, 0.0, 1.0]])


def _first_image(value: Any) -> Any | None:
    if value is None:
        return None
    import numpy as np

    array = _to_numpy(value)
    if array is None:
        return None
    if array.ndim >= 4:
        array = array[0]
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return np.asarray(array)


def _first_matrix(value: Any, rows: int, cols: int) -> Any | None:
    import numpy as np

    array = _to_numpy(value)
    if array is None:
        return None
    while array.ndim > 2:
        array = array[0]
    if array.shape != (rows, cols):
        return None
    return np.asarray(array, dtype=float)


def _to_numpy(value: Any) -> Any | None:
    try:
        detached = value.detach() if hasattr(value, "detach") else value
        cpu_value = detached.cpu() if hasattr(detached, "cpu") else detached
        return cpu_value.numpy() if hasattr(cpu_value, "numpy") else __import__("numpy").asarray(cpu_value)
    except Exception:
        return None


def _split_reset_result(result: Any) -> tuple[Any, JsonDict]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, info if isinstance(info, dict) else {"raw_info": _to_builtin(info)}
    return result, {}


def _unwrap_env(env: Any) -> Any:
    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is None or unwrapped is current:
            break
        current = unwrapped
    return current


def _pose_components(pose: Any) -> JsonDict | None:
    if pose is None:
        return None
    output: JsonDict = {}
    for key in ("p", "q"):
        value = getattr(pose, key, None)
        if value is not None:
            output[key] = _to_builtin(value)
    return output or None


def _split_step_result(result: Any) -> tuple[Any, Any, Any, Any, JsonDict]:
    if not isinstance(result, tuple) or len(result) != 5:
        raise RuntimeError("Expected Gymnasium env.step to return (obs, reward, terminated, truncated, info).")
    obs, reward, terminated, truncated, info = result
    return obs, reward, terminated, truncated, info if isinstance(info, dict) else {"raw_info": _to_builtin(info)}


def _coerce_maniskill_action(action: Any) -> Any:
    """Convert JSON-friendly agent actions to Gymnasium-compatible containers."""

    import numpy as np

    if isinstance(action, dict):
        return {key: _coerce_maniskill_action(value) for key, value in action.items()}
    if isinstance(action, (list, tuple)):
        return np.asarray(action, dtype=np.float32)
    return action


def _extract_success(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    for key in ("success", "is_success"):
        if key in info:
            return _any_truthy(_to_builtin(info[key]))
    return False


def _any_truthy(value: Any) -> bool:
    if isinstance(value, dict):
        return any(_any_truthy(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_any_truthy(item) for item in value)
    try:
        return bool(value)
    except Exception:
        return False


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
