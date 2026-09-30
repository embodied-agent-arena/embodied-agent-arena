from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

from .robocasa_agent_runtime import (
    JsonDict,
    RoboCasaAgentRuntimeBackend,
    RoboCasaRuntimeConfig,
    _invoke_primitive_handler,
    extract_camera_summaries,
)
from .robocasa365_preflight import inspect_robocasa365_preflight
from .schemas import Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


@dataclass(slots=True)
class RoboCasa365RuntimeConfig(RoboCasaRuntimeConfig):
    """RoboCasa365 suite wrapper over the RoboCasa agent-native runtime."""

    env_id: str = "robocasa365/CoffeeSetupMug"
    split: str = "target"
    suite_variant: str = "robocasa365"
    dataset_root: str | None = None
    episode_id: str | None = None
    task_family: str | None = None
    package_probe_python: str | None = None
    camera_names: list[str] = field(
        default_factory=lambda: ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    )


_ALIAS_TO_BASE = {
    "observe_robocasa365_kitchen_state": "observe_robocasa_kitchen_state",
    "observe_robocasa365_rgbd": "observe_robocasa_rgbd",
    "ground_robocasa365_visual_target": "ground_robocasa_visual_target",
    "inspect_robocasa365_object": "inspect_robocasa_object",
    "locate_robocasa365_object": "locate_robocasa_object",
    "inspect_robocasa365_fixture": "inspect_robocasa_fixture",
    "locate_robocasa365_fixture": "locate_robocasa_fixture",
    "inspect_robocasa365_affordance": "inspect_robocasa_affordance",
    "inspect_robocasa365_button_contact_frame": "inspect_robocasa_button_contact_frame",
    "open_robocasa365_fixture": "open_robocasa_fixture",
    "close_robocasa365_fixture": "close_robocasa_fixture",
    "press_robocasa365_fixture_button": "press_robocasa_fixture_button",
    "refine_robocasa365_button_contact_search": "refine_robocasa_button_contact_search",
    "sweep_robocasa365_button_contact_candidates": "sweep_robocasa_button_contact_candidates",
    "settle_robocasa365_environment": "settle_robocasa_environment",
    "move_robocasa365_ee_to": "move_robocasa_ee_to",
    "grasp_robocasa365_object": "grasp_robocasa_object",
    "place_robocasa365_object_at": "place_robocasa_object_at",
    "inspect_robocasa365_transport_state": "inspect_robocasa_transport_state",
    "record_robocasa365_evidence": "record_robocasa_evidence",
}
_BASE_TO_ALIAS = {base: alias for alias, base in _ALIAS_TO_BASE.items()}


class RoboCasa365AgentRuntimeBackend(RoboCasaAgentRuntimeBackend):
    """Agent-native RoboCasa365 adapter.

    RoboCasa365 is treated as a suite expansion, not a new action space. The
    coding agent sees suite-specific primitives while the implementation reuses
    the validated RoboCasa visual, pose, grounding, and action-skill boundaries.
    """

    def __init__(self, config: RoboCasa365RuntimeConfig | None = None, **kwargs: Any) -> None:
        super().__init__(config=config or RoboCasa365RuntimeConfig(), **kwargs)

    @property
    def config(self) -> RoboCasa365RuntimeConfig:  # type: ignore[override]
        return self.__dict__["config"]

    @config.setter
    def config(self, value: RoboCasaRuntimeConfig) -> None:
        if not isinstance(value, RoboCasa365RuntimeConfig):
            value = RoboCasa365RuntimeConfig(**value.to_dict())
        self.__dict__["config"] = value

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        task = super().reset(task_id, seed=seed, config=config)
        task.source = "w7:robocasa365:agent_runtime"
        task.goal = {
            "suite_variant": self.config.suite_variant,
            "env_id": self.config.env_id,
            "split": self.config.split,
            "episode_id": self.config.episode_id,
            "task_family": self.config.task_family,
            "success_source": "robocasa365_harness_or_official_evaluator_only",
        }
        task.tags = ["w7", "robocasa365", "robocasa", "ai_native_runtime", "live" if self.config.live else "dry"]
        task.budgets = {"primitive_calls": 40, "verifier_calls": 4}
        task.metadata["benchmark_id"] = "robocasa365"
        task.metadata["suite_variant"] = self.config.suite_variant
        task.metadata["dataset_root"] = self.config.dataset_root
        task.metadata["agent_native_contract"] = {
            **task.metadata.get("agent_native_contract", {}),
            "suite_specific_primitive_aliases": True,
            "official_365_success_exposed_as_primitive": False,
            "uses_robocasa_visual_pose_action_core": True,
            "entity_selection_requires_exact_caller_name_or_id": True,
            "button_selection_requires_exact_caller_name_or_pose": True,
            "query_or_task_lexical_auto_selection": False,
            "runtime_recommendation_or_recovery_planner": False,
        }
        self.record_event("robocasa365_reset_contract", {"task": task.to_dict()})
        return task

    def observe(self) -> Observation:
        obs = super().observe()
        obs.metadata["benchmark_id"] = "robocasa365"
        obs.data["suite_variant"] = self.config.suite_variant
        obs.data["dataset_context"] = self._dataset_context()
        self.record_event("robocasa365_observe_contract", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        base_cards = super().list_primitives(level=None)
        base_by_name = {card.name: card for card in base_cards}
        alias_cards: list[PrimitiveCard] = [
            PrimitiveCard(
                name="get_robocasa365_task_context",
                capability_tags=["w7", "robocasa365", "task_context"],
                input_schema={"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"language_task": "dict", "dataset_context": "dict", "runtime": "dict"},
                cost={"primitive_calls": 1},
                failure_modes=["backend_not_reset"],
                abstraction_level="L1",
                leakage_risk="none",
                description="Return RoboCasa365 suite, split, episode, task-family, and runtime context.",
            )
        ]
        for alias, base_name in _ALIAS_TO_BASE.items():
            if base_name not in base_by_name:
                continue
            base_card = base_by_name[base_name]
            card = replace(
                base_card,
                name=alias,
                capability_tags=["w7", "robocasa365", *[tag for tag in base_card.capability_tags if tag != "w4"]],
                description=base_card.description.replace("RoboCasa", "RoboCasa365"),
            )
            if alias == "ground_robocasa365_visual_target":
                card.input_schema = {**card.input_schema, "button_name": "str|None"}
            elif alias == "press_robocasa365_fixture_button":
                card.input_schema = {
                    **card.input_schema,
                    "button_name": "str",
                    "base_delta_frame": "str",
                    "base_xy_deadband": "float|None",
                    "bind_gripper_contact_geometry": "bool",
                    "gripper_contact_surface_axis": "list[float]|None",
                }
            alias_cards.append(card)
        cards = alias_cards
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_robocasa365_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        if name == "get_robocasa365_task_context":
            result = _invoke_primitive_handler(self._primitive_get_robocasa365_task_context, kwargs)
        elif name in _ALIAS_TO_BASE or name in _BASE_TO_ALIAS:
            base_name = _ALIAS_TO_BASE.get(name, name)
            handler = getattr(self, f"_primitive_{base_name}", None)
            if handler is None:
                result = PrimitiveResult(name=name, ok=False, error=f"Missing handler for {base_name}")
            else:
                result = _invoke_primitive_handler(handler, kwargs)
            result.name = name
            _rewrite_robocasa365_primitive_names(result.output)
            result.output["suite_variant"] = self.config.suite_variant
            result.output["dataset_context"] = self._dataset_context()
            result.metadata["aliased_from"] = base_name
        else:
            result = PrimitiveResult(
                name=name,
                ok=False,
                error=f"Primitive {name!r} is not exposed by RoboCasa365AgentRuntimeBackend",
            )
        payload = {"name": name, "kwargs": kwargs, "result": result.to_dict()}
        self.record_event("primitive_call", payload)
        self.record_event("robocasa365_primitive_call", payload)
        artifact_id = f"robocasa365:diagnostic:{len(self.get_trace().artifacts)}"
        self.get_trace().add_artifact(artifact_id, {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        result.artifacts.append(artifact_id)
        from .robocasa_interface_repair import compact_public_result
        return compact_public_result(result)

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        result = super().verify(scope=scope, **kwargs)
        result.metadata["benchmark_id"] = "robocasa365"
        result.metadata["suite_variant"] = self.config.suite_variant
        result.metadata["official_365_evaluator_exposed_as_primitive"] = False
        return result

    def _primitive_ground_robocasa_visual_target(
        self,
        prompt: str | None = None,
        camera_name: str | None = None,
        segmentation_id: int | None = None,
        entity_name: str | None = None,
        button_name: str | None = None,
        world_position: list[float] | None = None,
        max_world_distance: float | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        if button_name is None:
            return super()._primitive_ground_robocasa_visual_target(
                prompt=prompt,
                camera_name=camera_name,
                segmentation_id=segmentation_id,
                entity_name=entity_name,
                world_position=world_position,
                max_world_distance=max_world_distance,
                query=query,
                agent_context=agent_context,
            )
        requested_button_name = str(button_name).strip()
        if not requested_button_name:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={"requested_button_name": button_name, "fabricated_evidence": False},
                error="requested_button_name_required",
            )
        public_position = self._public_named_button_position(entity_name, requested_button_name)
        requested_position = _finite_xyz(world_position)
        if public_position is None or requested_position is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "entity_name": entity_name,
                    "requested_button_name": requested_button_name,
                    "world_position": world_position,
                    "fabricated_evidence": False,
                },
                error="public_named_button_pose_required",
            )
        try:
            import numpy as np
        except Exception as exc:  # pragma: no cover - numpy is a runtime dependency.
            return PrimitiveResult(name="ground_robocasa_visual_target", ok=False, error=f"numpy_unavailable: {exc}")
        if not np.allclose(public_position, requested_position, rtol=0.0, atol=1e-5):
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "entity_name": entity_name,
                    "requested_button_name": requested_button_name,
                    "requested_world_position": requested_position,
                    "public_button_position": public_position,
                    "fabricated_evidence": False,
                },
                error="public_named_button_pose_mismatch",
            )
        if segmentation_id is not None:
            instances = self._robocasa_segmentation_instances(camera_name=camera_name)
            selected = next(
                (
                    instance
                    for camera_instances in instances.values()
                    for instance in camera_instances
                    if int(instance.get("segmentation_id", -1)) == int(segmentation_id)
                    and (camera_name is None or str(instance.get("camera_name", camera_name)) == camera_name)
                ),
                None,
            )
            observed_button_name = selected.get("button_name") if isinstance(selected, dict) else None
            if selected is not None and observed_button_name != requested_button_name:
                return PrimitiveResult(
                    name="ground_robocasa_visual_target",
                    ok=False,
                    output={
                        "entity_name": entity_name,
                        "segmentation_id": segmentation_id,
                        "requested_button_name": requested_button_name,
                        "observed_button_name": observed_button_name,
                        "instance": deepcopy(selected),
                        "fabricated_evidence": False,
                    },
                    error="visual_button_name_mismatch",
                )
            result = super()._primitive_ground_robocasa_visual_target(
                prompt=prompt,
                camera_name=camera_name,
                segmentation_id=segmentation_id,
                entity_name=entity_name,
                world_position=requested_position,
                max_world_distance=max_world_distance,
                query=query,
                agent_context=agent_context,
            )
            if result.ok:
                grounding = result.output.get("grounding")
                if isinstance(grounding, dict):
                    grounding["requested_button_name"] = requested_button_name
                    handle = result.output.get("evidence_handle")
                    if isinstance(handle, str) and handle in self._visual_grounding_handles:
                        self._visual_grounding_handles[handle]["requested_button_name"] = requested_button_name
            return result

        cameras = extract_camera_summaries(self._last_obs, camera_name=camera_name)
        selected_camera_name = next(
            (
                str(name)
                for name, camera in sorted(cameras.items())
                if isinstance(camera, dict) and {"rgb", "depth"}.issubset(camera)
            ),
            None,
        )
        if selected_camera_name is None:
            return PrimitiveResult(
                name="ground_robocasa_visual_target",
                ok=False,
                output={
                    "entity_name": entity_name,
                    "requested_button_name": requested_button_name,
                    "fabricated_evidence": False,
                },
                error="named_button_pose_requires_current_rgbd",
            )
        handle = (
            f"robocasa:visual:{self._observation_serial}:{selected_camera_name}:"
            f"public_button:{requested_button_name}:{len(self._visual_grounding_handles)}"
        )
        fixture_payload = self._fixtures.get(str(entity_name), {})
        grounding = {
            "prompt": prompt,
            "query": query,
            "agent_context": dict(agent_context or {}),
            "entity_name": entity_name,
            "camera_name": selected_camera_name,
            "segmentation_id": None,
            "instance": {
                "entity_name": entity_name,
                "entity_kind": "fixture_button",
                "button_name": requested_button_name,
                "binding_source": "public_named_button_pose_with_rgbd_observation",
            },
            "depth_statistics": None,
            "point_world": requested_position,
            "entity_kind": "fixture_button",
            "button_name": requested_button_name,
            "requested_button_name": requested_button_name,
            "native_geometry": deepcopy(fixture_payload),
            "requested_world_position": requested_position,
            "distance_to_requested_world_position": 0.0,
            "observation_serial": self._observation_serial,
            "source": "upstream_robocasa_rgb_depth_and_public_named_button_pose",
            "fallback_mode": "rgbd_public_named_button_pose_no_segmentation",
        }
        self._visual_grounding_handles[handle] = grounding
        self.get_trace().add_artifact(handle, deepcopy(grounding))
        return PrimitiveResult(
            name="ground_robocasa_visual_target",
            ok=True,
            output={
                "evidence_handle": handle,
                "grounding": grounding,
                "native_modalities": cameras[selected_camera_name],
            },
            artifacts=[handle],
        )

    def _public_named_button_position(
        self,
        fixture_name: str | None,
        button_name: str,
    ) -> list[float] | None:
        fixture = self._fixtures.get(str(fixture_name)) if fixture_name is not None else None
        sites = fixture.get("affordance_sites") if isinstance(fixture, dict) else None
        buttons = sites.get("start_buttons") if isinstance(sites, dict) else None
        return _finite_xyz(buttons.get(button_name)) if isinstance(buttons, dict) else None

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
        arm_delta_frame: str = "world",
        gripper_command: float = 1.0,
        motion_trace_limit: int = 5,
    ) -> PrimitiveResult:
        requested_button_name = str(button_name).strip() if button_name is not None else ""
        if not requested_button_name:
            return PrimitiveResult(
                name="press_robocasa_fixture_button",
                ok=False,
                output={
                    "fixture_name": fixture_name,
                    "requested_button_name": button_name,
                    "execution_status": "not_executed",
                },
                error="requested_button_name_required",
            )
        supplied_handles = [str(handle) for handle in evidence_handles or []]
        grounded_button_names = [
            self._visual_grounding_handles[handle].get("button_name")
            for handle in supplied_handles
            if handle in self._visual_grounding_handles
        ]
        if any(name != requested_button_name for name in grounded_button_names):
            return PrimitiveResult(
                name="press_robocasa_fixture_button",
                ok=False,
                output={
                    "fixture_name": fixture_name,
                    "requested_button_name": requested_button_name,
                    "grounded_button_names": grounded_button_names,
                    "evidence_handles": supplied_handles,
                    "execution_status": "not_executed",
                },
                error="visual_button_name_mismatch",
            )
        resolved_button_offset = button_offset
        if bind_gripper_contact_geometry and resolved_button_offset is None:
            contact_surface_axis = (
                gripper_contact_surface_axis
                if gripper_contact_surface_axis is not None
                else press_direction_vector
            )
            resolved_button_offset = _infer_eef_target_offset_for_gripper_contact(
                self._env, surface_axis=contact_surface_axis
            )
        return self._run_action_skill(
            "press_robocasa_fixture_button",
            fixture_name=fixture_name,
            evidence_handles=evidence_handles,
            button_name=requested_button_name,
            button_position=button_position,
            visual_binding_tolerance=visual_binding_tolerance,
            use_visual_button_anchor_position=use_visual_button_anchor_position,
            button_offset=resolved_button_offset,
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

    def _run_action_skill(self, skill_name: str, **kwargs: Any) -> PrimitiveResult:
        base_delta_frame = str(kwargs.get("base_delta_frame", "world")).strip().lower()
        use_adapter = (
            skill_name == "press_robocasa_fixture_button"
            and bool(kwargs.get("mobile_base_enabled", False))
            and base_delta_frame in {"base", "robot_base", "mobilebase0_base"}
            and self._env is not None
        )
        if not use_adapter:
            return super()._run_action_skill(skill_name, **kwargs)

        runtime_schema = self._runtime_action_schema()
        original_env = self._env
        self._env = _BaseDeltaFrameAdapter(
            original_env,
            observation=lambda: self._last_obs,
            runtime_action_schema=runtime_schema,
            target_position=kwargs.get("button_position"),
            base_xy_deadband=kwargs.get("base_xy_deadband"),
        )
        try:
            result = super()._run_action_skill(skill_name, **kwargs)
        finally:
            self._env = original_env
        result.output["base_delta_frame"] = base_delta_frame
        result.output["base_delta_frame_adapter_applied"] = True
        return result

    def _refresh_scene_registry(self) -> None:
        super()._refresh_scene_registry()
        if self._env is None:
            return
        env = getattr(self._env, "unwrapped", self._env)
        for attr in ("fixture_refs", "fixtures"):
            fixtures = getattr(env, attr, None)
            if not isinstance(fixtures, dict):
                continue
            for fixture_name, fixture in fixtures.items():
                payload = self._fixtures.get(str(fixture_name))
                if not isinstance(payload, dict):
                    continue
                buttons = _discover_fixture_button_geometries(env, fixture)
                if not buttons:
                    continue
                sites = payload.setdefault("affordance_sites", {})
                observed_buttons = sites.setdefault("start_buttons", {})
                for button_name, position in buttons.items():
                    observed_buttons.setdefault(button_name, position)

    def runtime_available(self) -> JsonDict:
        runtime = super().runtime_available()
        preflight = inspect_robocasa365_preflight(
            dataset_root=self.config.dataset_root,
            asset_cache_dir=self.config.asset_cache_dir,
            episode_hint=self.config.episode_id,
            task_family_hint=self.config.task_family,
            python=self.config.package_probe_python,
            require_dataset=False,
            require_episode=False,
        ).to_dict()
        runtime.update(
            {
                "benchmark_id": "robocasa365",
                "suite_variant": self.config.suite_variant,
                "dataset_root_set": bool(self.config.dataset_root),
                "episode_id": self.config.episode_id,
                "task_family": self.config.task_family,
                "package_probe_python": self.config.package_probe_python,
                "robocasa365_preflight": {
                    "ok": preflight["ok"],
                    "blockers": preflight["blockers"],
                    "warnings": preflight["warnings"],
                    "dataset": preflight["dataset"],
                    "assets": {
                        "asset_cache_detected": preflight["assets"]["asset_cache_detected"],
                        "asset_cache_dir_requested": preflight["assets"]["asset_cache_dir_requested"],
                    },
                    "runtime_contract": preflight["runtime_contract"],
                },
            }
        )
        return runtime

    def _merged_config(self, overrides: JsonDict) -> RoboCasa365RuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return RoboCasa365RuntimeConfig(**data)

    def _language_task(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> JsonDict:
        language_task = super()._language_task(prompt=prompt, query=query, agent_context=agent_context)
        language_task.update({"suite_variant": self.config.suite_variant, "dataset_context": self._dataset_context()})
        return language_task

    def _dataset_context(self) -> JsonDict:
        return {
            "suite_variant": self.config.suite_variant,
            "dataset_root": self.config.dataset_root,
            "episode_id": self.config.episode_id,
            "task_family": self.config.task_family,
            "split": self.config.split,
            "env_id": self.config.env_id,
        }

    def _primitive_get_robocasa365_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        runtime = self.runtime_available()
        robocasa365_preflight = runtime.get("robocasa365_preflight", {})
        dataset_preflight = (
            robocasa365_preflight.get("dataset", {})
            if isinstance(robocasa365_preflight, dict)
            else {}
        )
        return PrimitiveResult(
            name="get_robocasa365_task_context",
            ok=True,
            output={
                "language_task": self._language_task(prompt=prompt, query=query, agent_context=agent_context or {}),
                "dataset_context": deepcopy(self._dataset_context()),
                "agent_episode_spec": deepcopy(dataset_preflight.get("agent_episode_spec")),
                "episode_manifest_available": bool(dataset_preflight.get("episode_manifest_available")),
                "runtime": runtime,
                "verifier_boundary": {
                    "task_completion_claimed_by_primitive": False,
                    "harness_verify_required": True,
                    "official_365_success_exposed_as_primitive": False,
                },
            },
        )

def _rewrite_robocasa365_primitive_names(value: Any) -> None:
    if isinstance(value, dict):
        primitive = value.get("primitive")
        if isinstance(primitive, str) and primitive in _BASE_TO_ALIAS:
            value["primitive"] = _BASE_TO_ALIAS[primitive]
            value["aliased_from"] = primitive
        for item in value.values():
            _rewrite_robocasa365_primitive_names(item)
    elif isinstance(value, list):
        for item in value:
            _rewrite_robocasa365_primitive_names(item)


class _BaseDeltaFrameAdapter:
    """Rotate caller-requested world XY base commands into the observed base frame."""

    def __init__(
        self,
        env: Any,
        *,
        observation: Callable[[], Any],
        runtime_action_schema: JsonDict,
        target_position: Any = None,
        base_xy_deadband: Any = None,
    ) -> None:
        self._env = env
        self._observation = observation
        self._runtime_action_schema = runtime_action_schema
        self._target_position = target_position
        self._base_xy_deadband = base_xy_deadband

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def step(self, action: Any) -> Any:
        observation = self._observation()
        return self._env.step(
            _rotate_base_action_to_observed_frame(
                action,
                observation=observation,
                runtime_action_schema=self._runtime_action_schema,
                target_position=self._target_position,
                base_xy_deadband=self._base_xy_deadband,
            )
        )


def _rotate_base_action_to_observed_frame(
    action: Any,
    *,
    observation: Any,
    runtime_action_schema: JsonDict,
    target_position: Any = None,
    base_xy_deadband: Any = None,
) -> Any:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project test dependency.
        return action

    if not isinstance(observation, dict):
        return action
    quat = observation.get("robot0_base_quat")
    rotation = _quat_xyzw_rotation(quat)
    if rotation is None:
        return action
    stop_base = _inside_base_deadband(observation, target_position, base_xy_deadband)

    if isinstance(action, dict):
        converted = deepcopy(action)
        for key, value in converted.items():
            lowered = str(key).lower()
            if "base" in lowered and "mode" not in lowered:
                vector = np.asarray(value, dtype=np.float32).copy().reshape(-1)
                if vector.shape[0] >= 2:
                    if stop_base:
                        vector[:] = 0.0
                    else:
                        local = rotation.T @ np.asarray([vector[0], vector[1], 0.0], dtype=np.float32)
                        vector[:2] = local[:2]
                    converted[key] = vector.reshape(np.asarray(value).shape)
        return converted

    base_slice = runtime_action_schema.get("base_slice")
    if not isinstance(base_slice, (list, tuple)) or len(base_slice) != 2:
        return action
    start, end = int(base_slice[0]), int(base_slice[1])
    converted = np.asarray(action, dtype=np.float32).copy()
    flat = converted.reshape(-1)
    if start < 0 or end > flat.shape[0] or end - start < 2:
        return action
    if stop_base:
        flat[start:end] = 0.0
    else:
        local = rotation.T @ np.asarray([flat[start], flat[start + 1], 0.0], dtype=np.float32)
        flat[start : start + 2] = local[:2]
    return converted


def _inside_base_deadband(observation: JsonDict, target_position: Any, deadband: Any) -> bool:
    try:
        import numpy as np
        threshold = float(deadband)
        eef = np.asarray(observation.get("robot0_eef_pos"), dtype=np.float32).reshape(-1)
        target = np.asarray(target_position, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return False
    if threshold <= 0.0 or eef.shape[0] < 2 or target.shape[0] < 2:
        return False
    return float(np.linalg.norm(eef[:2] - target[:2])) <= threshold


def _quat_xyzw_rotation(quat: Any) -> Any | None:
    try:
        import numpy as np
    except Exception:  # pragma: no cover - numpy is a project test dependency.
        return None
    if quat is None:
        return None
    values = np.asarray(quat, dtype=np.float32).reshape(-1)
    if values.shape[0] < 4:
        return None
    x, y, z, w = [float(value) for value in values[:4]]
    norm = float(np.linalg.norm(values[:4]))
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


def _finite_xyz(value: Any) -> list[float] | None:
    try:
        import numpy as np

        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.shape[0] < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    return [float(item) for item in vector[:3]]


def _discover_fixture_button_geometries(env: Any, fixture: Any) -> JsonDict:
    sim = getattr(env, "sim", None)
    model = getattr(sim, "model", None)
    data = getattr(sim, "data", None)
    if model is None or data is None:
        return {}
    prefix = str(getattr(fixture, "naming_prefix", ""))
    fixture_name = str(getattr(fixture, "name", ""))
    candidates = {
        f"{prefix}start_button": "start_button",
        f"{prefix}power_button": "power_button",
        f"{fixture_name}_start_button": "start_button",
        f"{fixture_name}_power_button": "power_button",
    }
    buttons: JsonDict = {}
    for geom_name, button_name in candidates.items():
        if not geom_name:
            continue
        try:
            geom_id = int(model.geom_name2id(geom_name))
            position = data.geom_xpos[geom_id]
            values = [float(value) for value in position]
        except Exception:
            continue
        if len(values) == 3:
            buttons[button_name] = values
    return buttons


def _infer_eef_target_offset_for_gripper_contact(
    env: Any,
    *,
    surface_axis: Any = None,
) -> list[float] | None:
    try:
        import numpy as np
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
