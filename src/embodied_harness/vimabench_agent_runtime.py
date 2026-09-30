from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import importlib.util
import os
from pathlib import Path
import sys
from typing import Any, Callable

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]
EnvFactory = Callable[[str, dict[str, Any]], Any]
VIMABENCH_REQUIRED_LIVE_MODULES = ("vima_bench", "gym")


@dataclass(slots=True)
class VimaBenchRuntimeConfig:
    task_name: str = "visual_manipulation"
    partition: str | None = None
    modalities: list[str] = field(default_factory=lambda: ["rgb", "segm"])
    live: bool = True
    task_kwargs: JsonDict = field(default_factory=dict)
    env_kwargs: JsonDict = field(
        default_factory=lambda: {"display_debug_window": False}
    )

    def to_dict(self) -> JsonDict:
        return asdict(self)


class VimaBenchAgentRuntimeBackend(EmbodiedBackend):
    """AI-native VIMA-Bench runtime adapter.

    The live path constructs a real `vima_bench` environment. Agent-facing
    primitives expose multimodal prompt/scene observation, prompt/query/context
    grounding, and VIMA action-dict submission. Oracle policies and task checks
    are intentionally reserved for `verify()` and are not primitive cards.
    """

    def __init__(self, config: VimaBenchRuntimeConfig | None = None, env_factory: EnvFactory | None = None) -> None:
        self.config = config or VimaBenchRuntimeConfig()
        self._env_factory = env_factory
        self._env: Any | None = None
        self._last_obs: Any = None
        self._last_info: JsonDict = {}
        self._last_reward: float | None = None
        self._done: bool = False
        self._task_spec: TaskSpec | None = None
        self._trace: EpisodeTrace | None = None
        self._pool_task: tuple[str, str, int] | None = None

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        from vima_bench import PARTITION_TO_SPECS
        task_name = str(coordinate.get("task_id") or "")
        parts = str(coordinate.get("variation") or "").split("::")
        if (len(parts) != 2 or parts[1] != task_name
                or task_name not in PARTITION_TO_SPECS["test"].get(parts[0], {})):
            raise ValueError("VIMA task is absent from the selected native test partition")
        seed = coordinate.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("VIMA reset seed must be an integer in [0, 2**32)")
        self._pool_task = (task_name, parts[0], seed)
        return {"bound": True, "mode": "native_test_partition", "task_name": task_name,
                "partition": parts[0], "seed": seed}

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None) -> TaskSpec:
        overrides = dict(config or {})
        if self._pool_task is not None:
            task_name, partition, selected_seed = self._pool_task
            if seed != selected_seed:
                raise ValueError("VIMA reset seed differs from selected pool coordinate")
            overrides.update(task_name=task_name, partition=partition, task_kwargs={})
            task_id = f"vimabench:{partition}:{task_name}:seed_{seed}"
        runtime_config = self._merged_config(overrides)
        self.config = runtime_config
        self._trace = EpisodeTrace(task_id=task_id)
        self._task_spec = TaskSpec(
            task_id=task_id,
            source="w4:vimabench:live_runtime",
            instruction="Solve a VIMA-Bench tabletop task from multimodal prompt and RGB-D/segmentation observations.",
            goal={"task_name": runtime_config.task_name, "success_source": "vima_env_reward_or_info"},
            budgets={"primitive_calls": 40, "verifier_calls": 5},
            tags=["w4", "vimabench", "ai_native_runtime", "live" if runtime_config.live else "dry"],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "vimabench",
                "runtime_config": runtime_config.to_dict(),
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "grounding_accepts_prompt_query": True,
                    "prompt_placeholders_ground_segmentation_ids": True,
                    "oracle_exposed_as_primitive": False,
                    "success_check_exposed_as_primitive": False,
                    "mock_success_for_actions": False,
                },
            },
        )
        self._last_info = {}
        self._last_reward = None
        self._done = False
        if runtime_config.live:
            self._env = self._make_env(runtime_config, seed=seed)
            if seed is not None and hasattr(self._env, "seed"):
                self._env.seed(seed)
            reset_result = self._env.reset()
            self._last_obs, self._last_info = _split_reset_result(reset_result)
        else:
            self._env = None
            self._last_obs = {}
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
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "runtime": self.runtime_available(),
                "prompt": self._prompt_text(),
                "prompt_assets": summarize_prompt_assets(self._prompt_assets()),
                "prompt_asset_object_ids": self._prompt_asset_object_ids(),
                "observation_summary": summarize_observation(self._last_obs, redact_evaluation=True),
                "action_schema": summarize_action_schema(self._env),
            },
            metadata={"benchmark_id": "vimabench", "task_name": self.config.task_name},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            self._primitive_card(
                "observe_vima_prompt",
                "L1",
                input_schema={"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={
                    "prompt_text": "str|None",
                    "prompt_assets": "dict",
                    "prompt_asset_object_ids": "dict[str,int]",
                    "agent_context": "dict",
                    "action_schema": "dict",
                },
                description="Read VIMA's multimodal task prompt and prompt assets with optional agent query/context.",
            ),
            self._primitive_card(
                "observe_vima_scene",
                "L1",
                input_schema={"prompt": "str|None", "query": "str|None", "agent_context": "dict|None", "view": "str|None", "include_raw": "bool"},
                output_schema={
                    "rgb": "dict",
                    "depth": "dict",
                    "segm": "dict",
                    "segmentation_instances": "dict",
                    "ee": "dict|None",
                    "action_schema": "dict",
                },
                description="Summarize current RGB-D and segmentation tabletop observations from the real VIMA env.",
            ),
            self._primitive_card(
                "inspect_vima_instances",
                "L1",
                input_schema={
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "view": "str|None",
                    "min_pixel_count": "int",
                },
                output_schema={
                    "instances": "dict[str,list[dict]]",
                    "exact_prompt_bindings": "list[dict]",
                    "prompt_asset_keys": "list[str]",
                    "prompt_asset_object_ids": "dict[str,int]",
                    "action_schema": "dict",
                },
                description="List visible segmentation instances and only exact prompt-placeholder id bindings; the caller selects task entities.",
            ),
            self._primitive_card(
                "inspect_vima_instance",
                "L2",
                input_schema={
                    "segmentation_id": "int",
                    "view": "str|None",
                    "agent_context": "dict|None",
                    "prompt": "str|None",
                    "query": "str|None",
                },
                output_schema={"instance": "dict|None", "bbox_xyxy": "list[int]|None", "prompt_asset_key": "str|None", "evidence_id": "str|None"},
                description="Inspect one caller-selected segmentation instance without choosing its task role.",
            ),
            self._primitive_card(
                "build_vima_pick_place_action",
                "L3",
                input_schema={
                    "source_segmentation_id": "int",
                    "target_segmentation_id": "int",
                    "view": "str|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "evidence_ids": "list[str]|None",
                },
                output_schema={"action": "dict", "source": "dict", "target": "dict", "mapping": "dict"},
                description=(
                    "Convert caller-selected segmentation ids into one VIMA pick-place action candidate without selecting task entities. "
                    "VIMA pick-place actions are authored in the top/action view; omit view or pass view='top' even if earlier observation used another camera."
                ),
                preconditions=[
                    "Call observe_vima_prompt, observe_vima_scene, inspect_vima_instances, and inspect_vima_instance first.",
                    "Choose source_segmentation_id and target_segmentation_id from public prompt bindings or inspected instances.",
                    "Pass evidence_ids from selected source/target inspect_vima_instance calls.",
                    "Use view='top' or omit view for action construction because VIMA action coordinates are top/action-view coordinates.",
                ],
            ),
            self._primitive_card(
                "submit_vima_action",
                "L3",
                input_schema={
                    "pose0_position": "list[float]",
                    "pose0_rotation": "list[float]",
                    "pose1_position": "list[float]",
                    "pose1_rotation": "list[float]",
                    "agent_context": "dict|None",
                    "evidence_ids": "list[str]|None",
                },
                output_schema={"stepped": "bool", "observation_summary": "dict", "action_schema": "dict"},
                description="Submit a caller-authored VIMA action-space dict and return only post-action observation evidence.",
                preconditions=[
                    "Use action fields returned by build_vima_pick_place_action.",
                    "Pass evidence_ids from selected source/target visual inspections or the action plan.",
                ],
            ),
            self._primitive_card(
                "record_vima_evidence",
                "L1",
                input_schema={"key": "str", "value": "any"},
                output_schema={"artifact_id": "str"},
                description="Record agent-selected prompt, grounding, or action evidence in the trace.",
                preconditions=[
                    "Use only public outputs such as prompt_text, prompt_asset_object_ids, exact_prompt_bindings, evidence_id, and action fields.",
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
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by VimaBenchAgentRuntimeBackend")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            if handler is None:
                result = PrimitiveResult(name=name, ok=False, error=f"Missing handler for {name}")
            else:
                result = handler(**kwargs)
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported VIMA-Bench verification scope: {scope}")
        else:
            success = _extract_success(self._last_info, self._last_reward, self._done)
            result = VerificationResult(
                ok=success,
                scope="task",
                message="VIMA-Bench env reported task success" if success else "VIMA-Bench env has not reported task success",
                metrics={"success": float(success), "reward": float(self._last_reward or 0.0), "done": float(self._done)},
                metadata={"info_summary": _to_builtin(self._last_info)},
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
        live_modules = _vimabench_live_module_status()
        return {
            "live": self.config.live,
            "env_created": self._env is not None,
            "vima_bench_importable": "vima_bench" not in live_modules["missing_live_modules"],
            "repo_local_source_path": live_modules["repo_local_source_path"],
            "required_live_modules": live_modules["required_live_modules"],
            "missing_live_modules": live_modules["missing_live_modules"],
            "task_name": self.config.task_name,
            "partition": self.config.partition,
        }

    def _merged_config(self, overrides: JsonDict) -> VimaBenchRuntimeConfig:
        data = self.config.to_dict()
        data.update(overrides)
        return VimaBenchRuntimeConfig(**data)

    def _make_env(self, config: VimaBenchRuntimeConfig, seed: int | None = None) -> Any:
        if self._env_factory is not None:
            return self._env_factory(config.task_name, config.to_dict())
        live_modules = _vimabench_live_module_status()
        missing_modules = live_modules["missing_live_modules"]
        if missing_modules:
            source_hint = (
                f" Repo-local VIMA-Bench source path detected at {live_modules['repo_local_source_path']}."
                if live_modules["repo_local_source_path"]
                else ""
            )
            raise RuntimeError(
                "VIMA-Bench live runtime requires importable Python modules "
                f"{' and '.join(VIMABENCH_REQUIRED_LIVE_MODULES)}. "
                f"Missing Python modules: {', '.join(missing_modules)}.{source_hint}"
            )
        from vima_bench import PARTITION_TO_SPECS, make

        task_kwargs = deepcopy(config.task_kwargs)
        if config.partition:
            task_kwargs.update(PARTITION_TO_SPECS["test"][config.partition][config.task_name] or {})
        kwargs = dict(config.env_kwargs)
        kwargs.update({"modalities": config.modalities, "task_kwargs": task_kwargs})
        if seed is not None:
            kwargs.setdefault("seed", seed)
        return make(config.task_name, **kwargs)

    def _primitive_observe_vima_prompt(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        return PrimitiveResult(
            name="observe_vima_prompt",
            ok=True,
            output={
                "agent_prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "observation_ref": self._observation_ref(source="vima_prompt_observation"),
                "prompt_text": self._prompt_text(),
                "prompt_assets": summarize_prompt_assets(self._prompt_assets()),
                "prompt_asset_object_ids": self._prompt_asset_object_ids(),
                "action_schema": summarize_action_schema(self._env),
            },
        )

    def _primitive_observe_vima_scene(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        view: str | None = None,
        include_raw: bool = False,
    ) -> PrimitiveResult:
        scene = extract_scene_summaries(self._last_obs, view=view)
        if "depth" not in scene:
            rendered_depth = extract_rendered_depth_summaries(self._env, view=view)
            if rendered_depth:
                scene["depth"] = rendered_depth
        instances = extract_segmentation_instances(self._last_obs, view=view)
        if instances:
            scene["segmentation_instances"] = instances
        observation_ref = self._observation_ref(source="vima_scene_observation")
        artifacts: list[str] = []
        for key, payload in scene.items():
            artifact_id = f"vimabench:scene:{key}:{len(self.get_trace().artifacts)}"
            self.get_trace().add_artifact(artifact_id, {"observation_ref": observation_ref, "payload": payload})
            artifacts.append(artifact_id)
        output = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_ref": observation_ref,
            "action_schema": summarize_action_schema(self._env),
            **scene,
        }
        if include_raw:
            output["raw_scene"] = extract_scene_data(self._last_obs, self._env, view=view)
        return PrimitiveResult(
            name="observe_vima_scene",
            ok=bool(scene),
            output=output,
            artifacts=artifacts,
            error=None if scene else "no_scene_observation",
        )

    def _primitive_inspect_vima_instances(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        view: str | None = None,
        min_pixel_count: int = 1,
    ) -> PrimitiveResult:
        prompt_assets = summarize_prompt_assets(self._prompt_assets())
        prompt_asset_ids = self._prompt_asset_object_ids()
        instances = extract_segmentation_instances(self._last_obs, view=view, min_pixel_count=min_pixel_count)
        exact_bindings = bind_vima_instances_by_exact_object_id(instances, prompt_asset_ids)
        artifact_id = f"vimabench:visual_instances:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_ref": self._observation_ref(source="vima_visual_instances_observation"),
            "view": view,
            "min_pixel_count": min_pixel_count,
            "prompt_text": self._prompt_text(),
            "prompt_asset_keys": list(prompt_assets.keys()),
            "prompt_asset_object_ids": prompt_asset_ids,
            "instances": instances,
            "exact_prompt_bindings": exact_bindings,
            "action_schema": summarize_action_schema(self._env),
            "evidence_id": artifact_id,
        }
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="inspect_vima_instances",
            ok=bool(instances),
            output=payload,
            artifacts=[artifact_id],
            error=None if instances else "no_visible_segmentation_instances",
        )

    def _primitive_inspect_vima_instance(
        self,
        segmentation_id: int,
        view: str | None = None,
        agent_context: JsonDict | None = None,
        prompt: str | None = None,
        query: str | None = None,
    ) -> PrimitiveResult:
        instances = extract_segmentation_instances(self._last_obs, view=view, min_pixel_count=1)
        exact_bindings = bind_vima_instances_by_exact_object_id(instances, self._prompt_asset_object_ids())
        selected = next(
            (item for item in exact_bindings if int(item.get("segmentation_id", -1)) == int(segmentation_id)),
            None,
        )
        evidence_id = f"vimabench:instance:{int(segmentation_id)}:{len(self.get_trace().artifacts)}"
        payload = {
            "kind": "visual_grounding",
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "segmentation_id": int(segmentation_id),
            "view": view,
            "instance": selected,
            "bbox_xyxy": selected.get("bbox_xyxy") if selected else None,
            "prompt_asset_key": selected.get("prompt_asset_key") if selected else None,
            "observation_ref": self._observation_ref(source="vima_selected_instance_observation"),
        }
        if selected is not None:
            self.get_trace().add_artifact(evidence_id, payload)
        return PrimitiveResult(
            name="inspect_vima_instance",
            ok=selected is not None,
            output={**payload, "evidence_id": evidence_id if selected is not None else None},
            artifacts=[evidence_id] if selected is not None else [],
            error=None if selected is not None else "segmentation_instance_not_found",
        )

    def _primitive_build_vima_pick_place_action(
        self,
        source_segmentation_id: int,
        target_segmentation_id: int,
        view: str | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        evidence_refs, evidence_error = self._resolve_visual_evidence(evidence_ids)
        if evidence_error is not None:
            return PrimitiveResult(name="build_vima_pick_place_action", ok=False, error=evidence_error)
        requested_view = view
        action_view = "top"
        plan = plan_pick_place_action_from_segmentation(
            self._last_obs,
            self._env,
            source_segmentation_id=source_segmentation_id,
            target_segmentation_id=target_segmentation_id,
            view=action_view,
        )
        if plan is None:
            return PrimitiveResult(
                name="build_vima_pick_place_action",
                ok=False,
                output={
                    "source_segmentation_id": source_segmentation_id,
                    "target_segmentation_id": target_segmentation_id,
                    "requested_view": requested_view,
                    "used_action_view": action_view,
                    "prompt": prompt,
                    "query": query,
                    "agent_context": agent_context or {},
                    "evidence_ids": evidence_refs,
                    "observation_ref": self._observation_ref(source="vima_action_plan_observation"),
                    "segmentation_instances": extract_segmentation_instances(self._last_obs, view=action_view),
                },
                error="segmentation_ids_not_found",
            )
        artifact_id = f"vimabench:action_plan:{len(self.get_trace().artifacts)}"
        payload = {
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
            "observation_ref": self._observation_ref(source="vima_action_plan_observation"),
            "action_provenance": {
                "source": "agent_selected_segmentation_ids",
                "source_segmentation_id": source_segmentation_id,
                "target_segmentation_id": target_segmentation_id,
                "requested_view": requested_view,
                "used_action_view": action_view,
                "query": query,
                "agent_context": agent_context or {},
                "evidence_ids": evidence_refs,
            },
            **plan,
        }
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(
            name="build_vima_pick_place_action",
            ok=True,
            output=payload,
            artifacts=[artifact_id],
        )

    def _primitive_submit_vima_action(
        self,
        pose0_position: list[float],
        pose0_rotation: list[float],
        pose1_position: list[float],
        pose1_rotation: list[float],
        agent_context: JsonDict | None = None,
        evidence_ids: list[str] | None = None,
    ) -> PrimitiveResult:
        evidence_refs, evidence_error = self._resolve_visual_evidence(evidence_ids)
        if evidence_error is not None:
            return PrimitiveResult(name="submit_vima_action", ok=False, error=evidence_error)
        action = {
            "pose0_position": pose0_position,
            "pose0_rotation": pose0_rotation,
            "pose1_position": pose1_position,
            "pose1_rotation": pose1_rotation,
        }
        error = validate_vima_action(action)
        pre_action_observation_ref = self._observation_ref(source="vima_pre_action_observation")
        if error is not None:
            return PrimitiveResult(
                name="submit_vima_action",
                ok=False,
                output={
                    "action": action,
                    "agent_context": agent_context or {},
                    "pre_action_observation_ref": pre_action_observation_ref,
                    "action_provenance": {"source": "agent_supplied_action_dict", "validation": "failed", "evidence_ids": evidence_refs},
                },
                error=error,
            )
        if self._env is None:
            return PrimitiveResult(
                name="submit_vima_action",
                ok=False,
                output={
                    "action": action,
                    "agent_context": agent_context or {},
                    "pre_action_observation_ref": pre_action_observation_ref,
                    "action_provenance": {"source": "agent_supplied_action_dict", "validation": "schema_checked", "evidence_ids": evidence_refs},
                    "stepped": False,
                    "requires_live_env": True,
                    "action_schema": summarize_action_schema(self._env),
                },
                error="live_env_missing",
            )
        step_result = self._env.step(action)
        self._last_obs, reward, done, info = _split_step_result(step_result)
        self._last_reward = reward
        self._done = done
        self._last_info = info
        return PrimitiveResult(
            name="submit_vima_action",
            ok=True,
            output={
                "action": action,
                "agent_context": agent_context or {},
                "pre_action_observation_ref": pre_action_observation_ref,
                "post_action_observation_ref": self._observation_ref(source="vima_post_action_observation"),
                "action_provenance": {"source": "agent_supplied_action_dict", "validation": "schema_checked", "evidence_ids": evidence_refs},
                "stepped": True,
                "observation_summary": summarize_observation(self._last_obs, redact_evaluation=True),
                "action_schema": summarize_action_schema(self._env),
            },
        )

    def _primitive_record_vima_evidence(self, key: str, value: Any) -> PrimitiveResult:
        artifact_id = f"vimabench:evidence:{key}"
        self.get_trace().add_artifact(artifact_id, {"key": key, "value": _to_builtin(value)})
        return PrimitiveResult(name="record_vima_evidence", ok=True, output={"artifact_id": artifact_id}, artifacts=[artifact_id])

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
            capability_tags=["w4", "vimabench", "agent_native_runtime"],
            input_schema=input_schema,
            output_schema=output_schema,
            preconditions=list(preconditions or []),
            cost={"primitive_calls": 1},
            failure_modes=["backend_not_configured", "wrong_arguments", "runtime_dependency_missing"],
            abstraction_level=level,
            leakage_risk="none",
            description=description,
        )

    def _prompt_text(self) -> str | None:
        return _to_builtin(getattr(self._env, "prompt", None)) if self._env is not None else None

    def _prompt_assets(self) -> Any:
        return getattr(self._env, "prompt_assets", {}) if self._env is not None else {}

    def _observation_ref(self, source: str) -> JsonDict:
        return {
            "source": source,
            "trace_event_count": len(self.get_trace().events),
            "runtime_live": self.config.live,
            "env_created": self._env is not None,
            "observation_type": type(self._last_obs).__name__,
        }

    def _prompt_asset_object_ids(self) -> JsonDict:
        task = getattr(self._env, "task", None)
        placeholders = getattr(task, "placeholders", None)
        if not isinstance(placeholders, dict):
            return {}
        object_ids: JsonDict = {}
        for key, placeholder in placeholders.items():
            obj_id = getattr(placeholder, "obj_id", None)
            if isinstance(obj_id, int):
                object_ids[str(key)] = int(obj_id)
        return object_ids

    def _resolve_visual_evidence(self, evidence_ids: list[str] | None) -> tuple[list[str], str | None]:
        refs = [str(value) for value in (evidence_ids or [])]
        for evidence_id in refs:
            payload = self.get_trace().artifacts.get(evidence_id)
            if not isinstance(payload, dict) or payload.get("kind") != "visual_grounding":
                return refs, f"visual_evidence_not_found: {evidence_id}"
        return refs, None

    def _require_reset(self) -> None:
        if self._trace is None or self._task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def summarize_prompt_assets(assets: Any) -> JsonDict:
    if not isinstance(assets, dict):
        return {}
    return {str(key): summarize_observation(value) for key, value in assets.items()}


def _activate_repo_local_vimabench_source() -> str | None:
    configured = os.environ.get("VIMABENCH_SOURCE_DIR")
    project_paths = get_project_paths()
    candidates = [
        Path(configured).expanduser() if configured else None,
        project_paths.external_upstream("vimabench"),
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        source_dir = candidate.resolve()
        if not (source_dir / "vima_bench").is_dir():
            continue
        source_text = str(source_dir)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        return source_text
    return None


def _vimabench_live_module_status() -> JsonDict:
    source_path = _activate_repo_local_vimabench_source()
    missing_modules = [
        module_name for module_name in VIMABENCH_REQUIRED_LIVE_MODULES if importlib.util.find_spec(module_name) is None
    ]
    return {
        "repo_local_source_path": source_path,
        "required_live_modules": list(VIMABENCH_REQUIRED_LIVE_MODULES),
        "missing_live_modules": missing_modules,
    }


def summarize_action_schema(env: Any) -> JsonDict:
    action_space = getattr(env, "action_space", None)
    spaces = getattr(action_space, "spaces", None)
    if not isinstance(spaces, dict):
        return {
            "type": "dict",
            "fields": {
                "pose0_position": {"shape": [2], "dtype": "float", "description": "pick XY in tabletop bounds"},
                "pose0_rotation": {"shape": [4], "dtype": "float", "description": "pick quaternion XYZW"},
                "pose1_position": {"shape": [2], "dtype": "float", "description": "place XY in tabletop bounds"},
                "pose1_rotation": {"shape": [4], "dtype": "float", "description": "place quaternion XYZW"},
            },
        }
    return {
        "type": type(action_space).__name__,
        "fields": {str(key): summarize_space(space) for key, space in spaces.items()},
    }


def summarize_space(space: Any) -> JsonDict:
    summary: JsonDict = {"type": type(space).__name__}
    for attr in ("shape", "dtype", "low", "high"):
        if hasattr(space, attr):
            summary[attr] = _to_builtin(getattr(space, attr))
    return summary


def summarize_observation(obs: Any, *, redact_evaluation: bool = False) -> JsonDict:
    if isinstance(obs, dict):
        return {
            str(key): summarize_observation(value, redact_evaluation=redact_evaluation)
            for key, value in obs.items()
            if not redact_evaluation or not _is_private_evaluation_key(key)
        }
    shape = getattr(obs, "shape", None)
    dtype = getattr(obs, "dtype", None)
    if shape is not None:
        return {"type": type(obs).__name__, "shape": [int(dim) for dim in shape], "dtype": str(dtype)}
    if isinstance(obs, (list, tuple)):
        return {"type": type(obs).__name__, "length": len(obs)}
    return {"type": type(obs).__name__, "value": _to_builtin(obs)}


def _is_private_evaluation_key(key: Any) -> bool:
    normalized = str(key).lower()
    return any(token in normalized for token in ("reward", "done", "success", "checker"))


def extract_scene_summaries(obs: Any, view: str | None = None) -> JsonDict:
    if not isinstance(obs, dict):
        return {}
    output: JsonDict = {}
    for key in ("rgb", "depth", "segm"):
        payload = obs.get(key)
        if isinstance(payload, dict):
            selected = {str(uid): summarize_observation(value) for uid, value in payload.items() if view is None or uid == view}
            if selected:
                output[key] = selected
    if "ee" in obs:
        output["ee"] = summarize_observation(obs["ee"])
    return output


def extract_scene_data(obs: Any, env: Any, view: str | None = None) -> JsonDict:
    """Return current upstream camera arrays, rendering depth only when absent."""
    output: JsonDict = {}
    if isinstance(obs, dict):
        for key in ("rgb", "depth", "segm"):
            payload = obs.get(key)
            if isinstance(payload, dict):
                selected = {str(uid): _to_builtin(value) for uid, value in payload.items() if view is None or uid == view}
                if selected:
                    output[key] = selected
    if "depth" not in output and env is not None and hasattr(env, "render_camera"):
        cameras = getattr(env, "agent_cams", {})
        rendered: JsonDict = {}
        for uid, config in cameras.items() if isinstance(cameras, dict) else []:
            if view is not None and uid != view:
                continue
            try:
                _rgb, depth, _segm = env.render_camera(config)
            except Exception:  # noqa: BLE001 - one unavailable view must not hide other native views.
                continue
            rendered[str(uid)] = _to_builtin(depth)
        if rendered:
            output["depth"] = rendered
    return output


def extract_rendered_depth_summaries(env: Any, view: str | None = None) -> JsonDict:
    if env is None or not hasattr(env, "render_camera"):
        return {}
    agent_cams = getattr(env, "agent_cams", {})
    if not isinstance(agent_cams, dict):
        return {}
    output: JsonDict = {}
    for uid, config in agent_cams.items():
        if view is not None and uid != view:
            continue
        try:
            _, depth, _ = env.render_camera(config)
        except Exception as exc:  # pragma: no cover - depends on live renderer state.
            output[str(uid)] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
            continue
        output[str(uid)] = summarize_observation(depth)
    return output


def extract_segmentation_bboxes(obs: Any, segmentation_id: int, view: str | None = None) -> list[JsonDict]:
    import numpy as np

    if not isinstance(obs, dict) or not isinstance(obs.get("segm"), dict):
        return []
    bboxes: list[JsonDict] = []
    for uid, payload in obs["segm"].items():
        if view is not None and uid != view:
            continue
        segmentation = _to_numpy(payload)
        if segmentation is None:
            continue
        if segmentation.ndim >= 3:
            segmentation = segmentation[..., 0]
        ys, xs = np.where(segmentation == segmentation_id)
        if xs.size == 0 or ys.size == 0:
            continue
        bboxes.append(
            {
                "view": str(uid),
                "segmentation_id": int(segmentation_id),
                "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                "pixel_count": int(xs.size),
                "image_shape_hw": [int(segmentation.shape[0]), int(segmentation.shape[1])],
            }
        )
    return bboxes


def extract_segmentation_instances(obs: Any, view: str | None = None, min_pixel_count: int = 1) -> JsonDict:
    import numpy as np

    if not isinstance(obs, dict) or not isinstance(obs.get("segm"), dict):
        return {}
    output: JsonDict = {}
    for uid, payload in obs["segm"].items():
        if view is not None and uid != view:
            continue
        segmentation = _to_numpy(payload)
        if segmentation is None:
            continue
        if segmentation.ndim >= 3:
            segmentation = segmentation[..., 0]
        instances: list[JsonDict] = []
        for raw_id in np.unique(segmentation):
            segmentation_id = int(raw_id)
            if segmentation_id == 0:
                continue
            ys, xs = np.where(segmentation == segmentation_id)
            if xs.size < min_pixel_count:
                continue
            instances.append(
                {
                    "segmentation_id": segmentation_id,
                    "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    "center_xy": [float(xs.mean()), float(ys.mean())],
                    "pixel_count": int(xs.size),
                    "image_shape_hw": [int(segmentation.shape[0]), int(segmentation.shape[1])],
                }
            )
        instances.sort(key=lambda item: int(item["pixel_count"]), reverse=True)
        output[str(uid)] = instances
    return output


def bind_vima_instances_by_exact_object_id(
    instances: JsonDict,
    prompt_asset_ids: JsonDict | None = None,
) -> list[JsonDict]:
    asset_key_by_segmentation_id = {
        int(value): str(key)
        for key, value in (prompt_asset_ids or {}).items()
        if isinstance(value, int)
    }
    bound: list[JsonDict] = []
    for view_name, view_instances in instances.items():
        if not isinstance(view_instances, list):
            continue
        for instance in view_instances:
            segmentation_id = int(instance.get("segmentation_id", -1))
            bound.append(
                {
                    **instance,
                    "view": view_name,
                    "prompt_asset_key": asset_key_by_segmentation_id.get(segmentation_id),
                    "binding_source": (
                        "exact_prompt_placeholder_object_id"
                        if segmentation_id in asset_key_by_segmentation_id
                        else "unbound_observed_segmentation_id"
                    ),
                }
            )
    return bound


def plan_pick_place_action_from_segmentation(
    obs: Any,
    env: Any,
    source_segmentation_id: int,
    target_segmentation_id: int,
    view: str = "top",
) -> JsonDict | None:
    source_boxes = extract_segmentation_bboxes(obs, segmentation_id=source_segmentation_id, view=view)
    target_boxes = extract_segmentation_bboxes(obs, segmentation_id=target_segmentation_id, view=view)
    if not source_boxes or not target_boxes:
        return None
    source = dict(source_boxes[0])
    target = dict(target_boxes[0])
    source["center_xy"] = _bbox_center_xy(source["bbox_xyxy"])
    target["center_xy"] = _bbox_center_xy(target["bbox_xyxy"])
    pose0_position, mapping = _segmentation_center_to_action_xy(env, source["center_xy"], source.get("image_shape_hw"))
    pose1_position, _ = _segmentation_center_to_action_xy(env, target["center_xy"], target.get("image_shape_hw"))
    action = {
        "pose0_position": pose0_position,
        "pose0_rotation": [0.0, 0.0, 0.0, 1.0],
        "pose1_position": pose1_position,
        "pose1_rotation": [0.0, 0.0, 0.0, 1.0],
    }
    return {
        "action": action,
        "source": source,
        "target": target,
        "mapping": {
            **mapping,
            "view": view,
            "source_segmentation_id": int(source_segmentation_id),
            "target_segmentation_id": int(target_segmentation_id),
            "source_center_xy": source["center_xy"],
            "target_center_xy": target["center_xy"],
            "source": "linear_top_view_segmentation_to_vima_xy",
            "oracle_used": False,
        },
    }


def _bbox_center_xy(bbox_xyxy: list[int]) -> list[float]:
    x0, y0, x1, y1 = bbox_xyxy
    return [(float(x0) + float(x1)) / 2.0, (float(y0) + float(y1)) / 2.0]


def _segmentation_center_to_action_xy(env: Any, center_xy: list[float], image_shape_hw: list[int] | None = None) -> tuple[list[float], JsonDict]:
    low, high = _position_bounds_from_env(env)
    image_shape = image_shape_hw or _infer_top_image_shape(env) or [128, 256]
    height = max(float(image_shape[0]), 1.0)
    width = max(float(image_shape[1]), 1.0)
    center_x, center_y = center_xy
    x_value = float(low[0] + (center_y / height) * (high[0] - low[0]))
    y_value = float(low[1] + (center_x / width) * (high[1] - low[1]))
    x_value = min(max(x_value, float(low[0])), float(high[0]))
    y_value = min(max(y_value, float(low[1])), float(high[1]))
    return [x_value, y_value], {
        "position_bounds_low": [float(low[0]), float(low[1])],
        "position_bounds_high": [float(high[0]), float(high[1])],
        "image_shape_hw": [int(image_shape[0]), int(image_shape[1])],
    }


def _position_bounds_from_env(env: Any) -> tuple[Any, Any]:
    import numpy as np

    bounds = getattr(env, "position_bounds", None)
    if bounds is not None and hasattr(bounds, "low") and hasattr(bounds, "high"):
        return np.asarray(bounds.low, dtype=float), np.asarray(bounds.high, dtype=float)
    action_space = getattr(env, "action_space", None)
    spaces = getattr(action_space, "spaces", {})
    pose0_space = spaces.get("pose0_position") if isinstance(spaces, dict) else None
    if pose0_space is not None and hasattr(pose0_space, "low") and hasattr(pose0_space, "high"):
        return np.asarray(pose0_space.low, dtype=float), np.asarray(pose0_space.high, dtype=float)
    return np.asarray([0.25, -0.5], dtype=float), np.asarray([0.75, 0.5], dtype=float)


def _infer_top_image_shape(env: Any) -> list[int] | None:
    agent_cams = getattr(env, "agent_cams", None)
    if not isinstance(agent_cams, dict):
        return None
    top = agent_cams.get("top")
    if not isinstance(top, dict) or "image_size" not in top:
        return None
    image_size = top["image_size"]
    if isinstance(image_size, (list, tuple)) and len(image_size) == 2:
        return [int(image_size[0]), int(image_size[1])]
    return None


def validate_vima_action(action: JsonDict) -> str | None:
    expected = {
        "pose0_position": 2,
        "pose0_rotation": 4,
        "pose1_position": 2,
        "pose1_rotation": 4,
    }
    for key, length in expected.items():
        value = action.get(key)
        if not isinstance(value, list) or len(value) != length:
            return f"invalid_{key}"
        if not all(isinstance(item, (int, float)) for item in value):
            return f"non_numeric_{key}"
    return None


def _split_reset_result(reset_result: Any) -> tuple[Any, JsonDict]:
    if isinstance(reset_result, tuple) and len(reset_result) == 2:
        obs, info = reset_result
        return obs, dict(info or {})
    return reset_result, {}


def _split_step_result(step_result: Any) -> tuple[Any, float, bool, JsonDict]:
    if isinstance(step_result, tuple) and len(step_result) == 5:
        obs, reward, terminated, truncated, info = step_result
        return obs, float(reward), bool(terminated or truncated), dict(info or {})
    if isinstance(step_result, tuple) and len(step_result) == 4:
        obs, reward, done, info = step_result
        return obs, float(reward), bool(done), dict(info or {})
    raise ValueError("VIMA env.step must return Gym 4-tuple or Gymnasium 5-tuple.")


def _extract_success(info: JsonDict, reward: float | None, done: bool) -> bool:
    for key in ("success", "is_success", "task_success"):
        if key in info:
            return bool(_to_builtin(info[key]))
    return bool(done and reward is not None and reward > 0.0)


def _to_numpy(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is present in test env.
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        return np.asarray(value)
    except (TypeError, ValueError):
        return None


def _to_builtin(value: Any) -> Any:
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is present in test env.
        np = None  # type: ignore[assignment]
    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.dtype):
            return str(value)
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value
