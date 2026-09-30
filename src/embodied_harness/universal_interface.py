from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
import os
import re
from typing import Any

from .benchmark_catalog import BENCHMARK_CATALOG
from .backend import EmbodiedBackend
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


JsonDict = dict[str, Any]

UNIVERSAL_CONTRACT_SCHEMA_VERSION = "agentic-embodied-arena-universal-interface-contract/v2"
UNIVERSAL_BUDGET_ERROR_CODE = "budget_exhausted"

UNIVERSAL_INTERFACE_NAMES: tuple[str, ...] = (
    "task.context",
    "scene.observe",
    "entity.enumerate",
    "entity.inspect",
    "entity.locate",
    "geometry.measure",
    "affordance.inspect",
    "evidence.record",
    "action.prepare",
    "action.execute",
    "policy.invoke",
    "failure.diagnose",
    "progress.check",
)

# Match CALVIN's official single-subtask rollout budget. Short 24-step chunks
# repeatedly reset the recurrent MCIL policy state and do not reach the LED
# button from the official initial condition; the upstream evaluator allows a
# 360-step rollout for this subtask.
CALVIN_PUBLIC_LANGUAGE_SKILL_HORIZON = 360

UNIVERSAL_INTERFACE_ALIASES: dict[str, str] = {
    name.replace(".", "_"): name for name in UNIVERSAL_INTERFACE_NAMES
}

BENCHMARK_FAMILY_REGISTRY: tuple[dict[str, str], ...] = (
    {"benchmark_id": "maniskill", "family": "ManiSkill3", "adapter": "ManiSkillAgentRuntimeBackend"},
    {"benchmark_id": "vimabench", "family": "VIMA-Bench", "adapter": "VimaBenchAgentRuntimeBackend"},
    {"benchmark_id": "cliport", "family": "CLIPort", "adapter": "CLIPortAgentRuntimeBackend"},
    {"benchmark_id": "vlabench", "family": "VLABench", "adapter": "VLABenchAdapter"},
    {"benchmark_id": "robocasa", "family": "RoboCasa", "adapter": "RoboCasaAgentRuntimeBackend"},
    {"benchmark_id": "capx", "family": "CaP-X", "adapter": "CaPXComparatorAdapter"},
    {"benchmark_id": "rlbench", "family": "RLBench", "adapter": "RLBenchAgentRuntimeBackend"},
    {"benchmark_id": "calvin", "family": "CALVIN", "adapter": "CALVINAgentRuntimeBackend"},
    {"benchmark_id": "behavior1k", "family": "BEHAVIOR-1K", "adapter": "Behavior1KAgentRuntimeBackend"},
    {"benchmark_id": "robocasa365", "family": "RoboCasa365", "adapter": "RoboCasa365AgentRuntimeBackend"},
    {"benchmark_id": "robowits", "family": "RoboWits", "adapter": "RoboWitsAgentRuntimeBackend"},
    {"benchmark_id": "robotwin2", "family": "RoboTwin2", "adapter": "RoboTwin2AgentRuntimeBackend"},
    {"benchmark_id": "robodojo", "family": "RoboDojo", "adapter": "RoboDojoAgentRuntimeBackend"},
)

FORBIDDEN_AGENT_FIELDS: tuple[str, ...] = (
    "oracle",
    "checker",
    "expert",
    "demo",
    "reward",
    "success_function",
    "success_checker",
    "ground_truth",
)

ROBOCASA_PRESS_ACTION_ARGUMENT_KEYS: tuple[str, ...] = (
    "strategy",
    "horizon",
    "approach_steps",
    "approach_distance",
    "approach_stop_distance",
    "approach_patience_steps",
    "button_offset",
    "bind_gripper_contact_geometry",
    "gripper_contact_surface_axis",
    "tolerance",
    "press_depth",
    "max_press_depth",
    "press_contact_seek_steps",
    "press_contact_seek_depth",
    "press_direction_sign",
    "press_direction_vector",
    "press_steps",
    "hold_steps",
    "retreat_steps",
    "retreat_distance",
    "gain",
    "max_delta",
    "mobile_base_enabled",
    "mobile_base_active_phases",
    "base_gain",
    "base_max_delta",
    "base_command_sign",
    "base_mode_value",
    "arm_delta_frame",
    "gripper_command",
    "motion_trace_limit",
    "visual_binding_tolerance",
    "use_visual_button_anchor_position",
)

ROBOCASA_SWEEP_ACTION_ARGUMENT_KEYS: tuple[str, ...] = (
    "candidate_indices",
    "max_attempts",
    "stop_on_interaction",
    "move_to_previous_best",
    "prealign_steps",
    "prealign_gain",
    "prealign_max_delta",
    "strategy",
    "horizon",
    "approach_steps",
    "approach_distance",
    "approach_stop_distance",
    "approach_patience_steps",
    "bind_gripper_contact_geometry",
    "tolerance",
    "press_depth",
    "max_press_depth",
    "press_contact_seek_steps",
    "press_contact_seek_depth",
    "press_direction_sign",
    "press_steps",
    "hold_steps",
    "retreat_steps",
    "retreat_distance",
    "gain",
    "max_delta",
    "mobile_base_enabled",
    "mobile_base_active_phases",
    "base_gain",
    "base_max_delta",
    "base_command_sign",
    "base_mode_value",
    "base_delta_frame",
    "base_xy_deadband",
    "arm_delta_frame",
    "gripper_command",
    "motion_trace_limit",
    "visual_binding_tolerance",
    "use_visual_button_anchor_position",
)

ROBOCASA_MOVE_ACTION_ARGUMENT_KEYS: tuple[str, ...] = (
    "strategy",
    "horizon",
    "offset",
    "gain",
    "max_delta",
    "arm_delta_frame",
    "tolerance",
    "gripper_command",
    "avoidance_point",
    "min_distance_from_avoidance",
    "avoidance_tolerance",
    "stop_when_avoidance_reached",
)

ROBOCASA_GRASP_ACTION_ARGUMENT_KEYS: tuple[str, ...] = (
    "strategy",
    "horizon",
    "offset",
    "grasp_contact_steps",
    "grasp_contact_offset",
    "grasp_contact_tolerance",
    "force_grasp_contact_steps",
    "grasp_hold_steps",
    "gripper_command",
    "grasp_lift_delta",
    "contact_tolerance",
    "gain",
    "max_delta",
    "arm_delta_frame",
    "mobile_base_enabled",
    "base_gain",
    "base_max_delta",
    "base_command_sign",
    "base_mode_value",
    "base_xy_deadband",
    "motion_trace_limit",
)

ROBOCASA_PLACE_ACTION_ARGUMENT_KEYS: tuple[str, ...] = (
    "strategy",
    "horizon",
    "offset",
    "use_affordance_site",
    "affordance_site_name",
    "gripper_command",
    "transport_gripper_command",
    "release_gripper_command",
    "release_only_when_ready",
    "release_xy_tolerance",
    "release_requires_contact",
    "release_settle_steps",
    "post_release_retreat_offset",
    "post_release_retreat_steps",
    "post_release_retreat_min_distance",
    "post_release_retreat_max_steps",
    "tolerance",
    "object_relative_control",
    "object_error_gain",
    "object_error_clip",
    "object_xy_push_steps",
    "object_xy_push_align_steps",
    "object_xy_push_backoff",
    "object_xy_push_through",
    "object_xy_push_z_offset",
    "object_xy_push_reacquire_from_side",
    "object_xy_contact_seek_steps",
    "object_xy_contact_seek_backoff",
    "object_xy_contact_seek_z_offset",
    "object_xy_early_stop_enabled",
    "object_xy_stop_when_within",
    "object_xy_stop_requires_contact",
    "contact_guard_enabled",
    "contact_tolerance",
    "contact_guard_tolerance",
    "contact_guard_recover_steps",
    "contact_guard_offset",
    "gain",
    "max_delta",
    "arm_delta_frame",
    "mobile_base_enabled",
    "base_gain",
    "base_max_delta",
    "base_command_sign",
    "base_mode_value",
    "base_xy_deadband",
    "motion_trace_limit",
)

UNIVERSAL_HANDLE_CONTRACT: JsonDict = {
    "observation_handle": {
        "prefix": "obs:",
        "kind": "observation",
        "producer_interfaces": ["scene.observe"],
        "consumer_interfaces": [],
        "agent_visible_payload": "opaque handle plus sanitized public observation summary",
    },
    "entity_handle": {
        "prefix": "ent:",
        "kind": "entity",
        "producer_interfaces": ["entity.enumerate", "entity.inspect", "entity.locate"],
        "consumer_interfaces": ["entity.inspect", "entity.locate", "geometry.measure", "affordance.inspect"],
        "agent_visible_payload": "opaque handle plus public label/attributes evidence",
    },
    "evidence_handle": {
        "prefix": "ev:",
        "kind": "evidence",
        "producer_interfaces": [
            "scene.observe",
            "entity.enumerate",
            "entity.inspect",
            "entity.locate",
            "geometry.measure",
            "affordance.inspect",
            "evidence.record",
        ],
        "consumer_interfaces": ["evidence.record", "action.prepare", "policy.invoke"],
        "agent_visible_payload": "opaque evidence reference with sanitized source handle list",
    },
    "action_handle": {
        "prefix": "act:",
        "kind": "prepared_action",
        "producer_interfaces": ["action.prepare"],
        "consumer_interfaces": ["action.execute"],
        "agent_visible_payload": "opaque prepared action reference with native details hidden",
    },
    "execution_handle": {
        "prefix": "exec:",
        "kind": "execution",
        "producer_interfaces": ["action.execute"],
        "consumer_interfaces": ["failure.diagnose", "progress.check"],
        "agent_visible_payload": "opaque execution reference with public result summary",
    },
    "policy_handle": {
        "prefix": "policy:",
        "kind": "policy_invocation",
        "producer_interfaces": ["policy.invoke"],
        "consumer_interfaces": ["failure.diagnose", "progress.check"],
        "agent_visible_payload": "opaque policy invocation reference with public result summary",
    },
}

UNIVERSAL_EVIDENCE_CONTRACT: JsonDict = {
    "evidence_required_interfaces": ["action.prepare", "action.execute", "policy.invoke"],
    "evidence_handle_prefix": "ev:",
    "evidence_handle_consumers": ["evidence.record", "action.prepare", "policy.invoke"],
    "action_execution_requires_prepared_handle": True,
    "official_verifier_agent_callable": False,
    "reward_agent_callable": False,
    "oracle_agent_callable": False,
    "expert_trajectory_agent_callable": False,
    "hidden_harness_only_signals": [
        "official_verifier",
        "reward",
        "oracle",
        "success_checker",
        "expert_trajectory",
        "demo_replay",
    ],
    "forbidden_agent_fields": list(FORBIDDEN_AGENT_FIELDS),
    "native_low_level_parameters_hidden_until_execute": True,
    "low_level_parameters_disclosed_to_agent": False,
}

UNIVERSAL_BUDGET_CONTRACT: JsonDict = {
    "agent_visible": True,
    "response_location": "PrimitiveResult.metadata.budget",
    "context_location": "task.context.output.budget",
    "progress_location": "progress.check.output.progress.budget",
    "fields": ["limits", "used", "remaining", "exhausted"],
    "hard_counters": ["primitive_calls", "policy_calls", "verifier_calls"],
    "exhaustion_error_code": UNIVERSAL_BUDGET_ERROR_CODE,
    "official_verifier_details_visible": False,
}

_UNIVERSAL_INTERFACE_HANDLE_IO: dict[str, JsonDict] = {
    "task.context": {"handle_inputs": [], "handle_outputs": []},
    "scene.observe": {"handle_inputs": [], "handle_outputs": ["observation_handle", "evidence_handle"]},
    "entity.enumerate": {"handle_inputs": [], "handle_outputs": ["entities[].entity_handle", "evidence_handle"]},
    "entity.inspect": {"handle_inputs": ["entity"], "handle_outputs": ["entity_handle", "evidence_handle"]},
    "entity.locate": {"handle_inputs": ["entity"], "handle_outputs": ["entity_handle", "evidence_handle"]},
    "geometry.measure": {"handle_inputs": ["subjects"], "handle_outputs": ["evidence_handle"]},
    "affordance.inspect": {"handle_inputs": ["entity"], "handle_outputs": ["evidence_handle"]},
    "evidence.record": {"handle_inputs": ["source_handles"], "handle_outputs": ["evidence_handle"]},
    "action.prepare": {"handle_inputs": ["evidence_handles"], "handle_outputs": ["action_handle"]},
    "action.execute": {"handle_inputs": ["action_handle"], "handle_outputs": ["execution_handle"]},
    "policy.invoke": {"handle_inputs": ["evidence_handles"], "handle_outputs": ["policy_handle"]},
    "failure.diagnose": {"handle_inputs": ["handle"], "handle_outputs": []},
    "progress.check": {"handle_inputs": [], "handle_outputs": []},
}


@dataclass(frozen=True, slots=True)
class UniversalInterfaceSpec:
    name: str
    disclosure_level: str
    capability_tags: tuple[str, ...]
    summary: str
    input_schema: JsonDict = field(default_factory=dict)
    output_schema: JsonDict = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    side_effects: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    requires_evidence: bool = False

    def to_card(self, *, schema_visible: bool = True) -> PrimitiveCard:
        return PrimitiveCard(
            name=self.name,
            capability_tags=list(self.capability_tags),
            input_schema=deepcopy(self.input_schema) if schema_visible else {},
            output_schema=deepcopy(self.output_schema) if schema_visible else {},
            preconditions=list(self.preconditions) if schema_visible else [],
            side_effects=list(self.side_effects) if schema_visible else [],
            cost={"interface_calls": 1},
            failure_modes=(
                [*self.error_codes, UNIVERSAL_BUDGET_ERROR_CODE]
                if schema_visible
                else []
            ),
            abstraction_level=self.disclosure_level,
            leakage_risk="none",
            description=self.summary,
        )


@dataclass(frozen=True, slots=True)
class UniversalAdapterProfile:
    benchmark_id: str
    family: str
    adapter: str
    backend_kind: str
    context_primitives: tuple[str, ...] = ()
    observe_primitives: tuple[str, ...] = ()
    enumerate_primitives: tuple[str, ...] = ()
    inspect_primitives: tuple[str, ...] = ()
    locate_primitives: tuple[str, ...] = ()
    geometry_primitives: tuple[str, ...] = ()
    affordance_primitives: tuple[str, ...] = ()
    evidence_primitives: tuple[str, ...] = ()
    prepare_primitives: tuple[str, ...] = ()
    execute_primitives: tuple[str, ...] = ()
    policy_primitives: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def to_dict(self, *, include_internal_names: bool = True) -> JsonDict:
        payload = asdict(self)
        if include_internal_names:
            return payload
        mapped = self.mapped_interface_counts()
        return {
            "benchmark_id": self.benchmark_id,
            "family": self.family,
            "adapter": self.adapter,
            "backend_kind": self.backend_kind,
            "mapped_interface_counts": mapped,
            "native_names_hidden": True,
        }

    def mapped_interface_counts(self) -> JsonDict:
        return {
            "task.context": len(self.context_primitives),
            "scene.observe": len(self.observe_primitives),
            "entity.enumerate": len(self.enumerate_primitives),
            "entity.inspect": len(self.inspect_primitives),
            "entity.locate": len(self.locate_primitives),
            "geometry.measure": len(self.geometry_primitives),
            "affordance.inspect": len(self.affordance_primitives),
            "evidence.record": len(self.evidence_primitives),
            "action.prepare": len(self.prepare_primitives),
            "action.execute": len(self.execute_primitives),
            "policy.invoke": len(self.policy_primitives),
        }


UNIVERSAL_INTERFACE_SPECS: tuple[UniversalInterfaceSpec, ...] = (
    UniversalInterfaceSpec(
        name="task.context",
        disclosure_level="L1",
        capability_tags=("context", "planning"),
        summary="Return public task text, budgets, benchmark family, and high-level capability summaries.",
        input_schema={"query": "str|None", "agent_context": "dict|None"},
        output_schema={"task": "dict", "benchmark": "dict", "interfaces": "list[dict]"},
        error_codes=("task_not_reset",),
    ),
    UniversalInterfaceSpec(
        name="scene.observe",
        disclosure_level="L2",
        capability_tags=("perception", "vision", "state"),
        summary="Return native public observation evidence such as RGB/RGB-D/segmentation/state summaries.",
        input_schema={"view": "str|None", "modality": "str|None", "query": "str|None", "agent_context": "dict|None"},
        output_schema={"observation_handle": "str", "evidence_handle": "str", "observation": "dict"},
        error_codes=("observation_failed",),
    ),
    UniversalInterfaceSpec(
        name="entity.enumerate",
        disclosure_level="L2",
        capability_tags=("perception", "entities"),
        summary="Enumerate visible or public entities from the latest observation without task answers.",
        input_schema={"query": "str|None", "agent_context": "dict|None"},
        output_schema={"entities": "list[dict]", "evidence_handle": "str"},
        error_codes=("no_entities_visible",),
    ),
    UniversalInterfaceSpec(
        name="entity.inspect",
        disclosure_level="L2",
        capability_tags=("perception", "entities", "evidence"),
        summary="Inspect one selected entity and bind visual/state attributes to an evidence handle.",
        input_schema={"entity": "str|dict", "query": "str|None", "agent_context": "dict|None"},
        output_schema={"entity_handle": "str", "evidence_handle": "str", "attributes": "dict"},
        error_codes=("entity_not_found",),
    ),
    UniversalInterfaceSpec(
        name="entity.locate",
        disclosure_level="L2",
        capability_tags=("grounding", "pose", "mask", "bbox"),
        summary="Locate an entity and return a handle plus public bbox/mask/pose evidence when available.",
        input_schema={"entity": "str|dict", "query": "str|None", "agent_context": "dict|None"},
        output_schema={"entity_handle": "str", "evidence_handle": "str", "location": "dict"},
        error_codes=("entity_not_found", "location_unavailable"),
    ),
    UniversalInterfaceSpec(
        name="geometry.measure",
        disclosure_level="L2",
        capability_tags=("geometry", "spatial"),
        summary="Measure public distances, poses, sizes, or spatial relations between evidence-backed handles.",
        input_schema={"subjects": "list[str|dict]", "measurement": "str|None", "agent_context": "dict|None"},
        output_schema={"evidence_handle": "str", "measurement": "dict"},
        error_codes=("handle_not_found", "measurement_unavailable"),
    ),
    UniversalInterfaceSpec(
        name="affordance.inspect",
        disclosure_level="L2",
        capability_tags=("affordance", "planning"),
        summary="Report public action affordances such as grasp, press, place, toggle, navigate, or policy skill.",
        input_schema={"entity": "str|dict|None", "query": "str|None", "agent_context": "dict|None"},
        output_schema={"evidence_handle": "str", "affordances": "list[dict]"},
        error_codes=("affordance_unavailable",),
    ),
    UniversalInterfaceSpec(
        name="evidence.record",
        disclosure_level="L2",
        capability_tags=("evidence", "trace"),
        summary="Bind selected observation/entity/action reasoning into an auditable evidence handle.",
        input_schema={"key": "str", "value": "dict", "source_handles": "list[str]|None", "agent_context": "dict|None"},
        output_schema={"evidence_handle": "str", "source_handles": "list[str]"},
        error_codes=("invalid_evidence", "handle_not_found"),
    ),
    UniversalInterfaceSpec(
        name="action.prepare",
        disclosure_level="L3",
        capability_tags=("action", "compilation"),
        summary="Compile a semantic action into a benchmark-native action handle using evidence-backed handles.",
        input_schema={
            "action": "dict",
            "evidence_handles": "list[str]",
            "agent_context": "dict|None",
            "public_rationale": "str|None",
        },
        output_schema={"action_handle": "str", "compiled_action": "dict"},
        preconditions=("At least one evidence handle is required.",),
        error_codes=("evidence_required", "unsupported_action", "action_prepare_failed"),
        requires_evidence=True,
    ),
    UniversalInterfaceSpec(
        name="action.execute",
        disclosure_level="L4",
        capability_tags=("action", "execution"),
        summary="Execute a prepared action handle; low-level poses/control modes remain internal to the adapter.",
        input_schema={"action_handle": "str", "agent_context": "dict|None"},
        output_schema={"execution_handle": "str", "public_result": "dict"},
        preconditions=("action.prepare must create the action handle first.",),
        side_effects=("May step the native simulator or policy runtime.",),
        error_codes=("action_handle_unknown", "backend_primitive_failed", "evidence_required"),
        requires_evidence=True,
    ),
    UniversalInterfaceSpec(
        name="policy.invoke",
        disclosure_level="L3",
        capability_tags=("policy", "skill"),
        summary="Invoke a generic low-level policy skill with public prompt/context and evidence handles.",
        input_schema={"policy": "str", "inputs": "dict|None", "evidence_handles": "list[str]", "agent_context": "dict|None"},
        output_schema={"policy_handle": "str", "public_result": "dict"},
        preconditions=("Policy invocation must cite evidence handles.",),
        error_codes=("unsupported_policy", "evidence_required"),
        requires_evidence=True,
    ),
    UniversalInterfaceSpec(
        name="failure.diagnose",
        disclosure_level="L2",
        capability_tags=("recovery", "debug"),
        summary="Return public failure causes and repair suggestions from trace/action errors without hidden scoring data.",
        input_schema={"handle": "str|None", "agent_context": "dict|None"},
        output_schema={"diagnosis": "dict"},
        error_codes=("no_failure_available",),
    ),
    UniversalInterfaceSpec(
        name="progress.check",
        disclosure_level="L2",
        capability_tags=("progress", "state"),
        summary="Return public progress signals; official scoring remains hidden from the agent.",
        input_schema={"query": "str|None", "agent_context": "dict|None"},
        output_schema={"progress": "dict"},
        error_codes=("progress_unavailable",),
    ),
)

_SPEC_BY_NAME = {spec.name: spec for spec in UNIVERSAL_INTERFACE_SPECS}

_REGISTRY_BY_ID = {item["benchmark_id"]: item for item in BENCHMARK_FAMILY_REGISTRY}


def _profile(
    benchmark_id: str,
    *,
    backend_kind: str = "EmbodiedBackend",
    context: tuple[str, ...] = (),
    observe: tuple[str, ...] = (),
    enumerate_entities: tuple[str, ...] = (),
    inspect: tuple[str, ...] = (),
    locate: tuple[str, ...] = (),
    geometry: tuple[str, ...] = (),
    affordance: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    prepare: tuple[str, ...] = (),
    execute: tuple[str, ...] = (),
    policy: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> UniversalAdapterProfile:
    registry_item = _REGISTRY_BY_ID[benchmark_id]
    return UniversalAdapterProfile(
        benchmark_id=benchmark_id,
        family=registry_item["family"],
        adapter=registry_item["adapter"],
        backend_kind=backend_kind,
        context_primitives=context,
        observe_primitives=observe,
        enumerate_primitives=enumerate_entities,
        inspect_primitives=inspect,
        locate_primitives=locate,
        geometry_primitives=geometry,
        affordance_primitives=affordance,
        evidence_primitives=evidence,
        prepare_primitives=prepare,
        execute_primitives=execute,
        policy_primitives=policy,
        notes=notes,
    )


def _catalog_profile(
    benchmark_id: str,
    *,
    adapter: str,
    backend_kind: str = "EmbodiedBackend",
    context: tuple[str, ...] = (),
    observe: tuple[str, ...] = (),
    enumerate_entities: tuple[str, ...] = (),
    inspect: tuple[str, ...] = (),
    locate: tuple[str, ...] = (),
    geometry: tuple[str, ...] = (),
    affordance: tuple[str, ...] = (),
    evidence: tuple[str, ...] = (),
    prepare: tuple[str, ...] = (),
    execute: tuple[str, ...] = (),
    policy: tuple[str, ...] = (),
    notes: tuple[str, ...] = (),
) -> UniversalAdapterProfile:
    """Build a profile for a catalog target outside the legacy 13-operation registry."""

    entry = next(item for item in BENCHMARK_CATALOG if item.benchmark_id == benchmark_id)
    return UniversalAdapterProfile(
        benchmark_id=benchmark_id,
        family=entry.family,
        adapter=adapter,
        backend_kind=backend_kind,
        context_primitives=context,
        observe_primitives=observe,
        enumerate_primitives=enumerate_entities,
        inspect_primitives=inspect,
        locate_primitives=locate,
        geometry_primitives=geometry,
        affordance_primitives=affordance,
        evidence_primitives=evidence,
        prepare_primitives=prepare,
        execute_primitives=execute,
        policy_primitives=policy,
        notes=notes,
    )


UNIVERSAL_ADAPTER_PROFILES: dict[str, UniversalAdapterProfile] = {
    "maniskill": _profile(
        "maniskill",
        context=("observe_maniskill_state", "inspect_maniskill_visual_readiness"),
        observe=("observe_maniskill_state", "observe_maniskill_visual"),
        enumerate_entities=("list_maniskill_instances",),
        inspect=("inspect_maniskill_instance", "detect_maniskill_color_regions", "inspect_maniskill_visual_readiness"),
        locate=("locate_maniskill_actor", "inspect_maniskill_instance"),
        geometry=("inspect_maniskill_instance", "detect_maniskill_color_regions", "observe_maniskill_control_state"),
        affordance=("inspect_maniskill_visual_readiness", "observe_maniskill_control_state"),
        evidence=("record_maniskill_evidence", "record_w4_evidence"),
        prepare=("move_maniskill_tcp_to", "move_maniskill_tcp_delta", "apply_maniskill_action", "set_maniskill_gripper"),
        execute=("move_maniskill_tcp_to", "move_maniskill_tcp_delta", "apply_maniskill_action", "set_maniskill_gripper"),
        notes=("w4 smoke backend also exposes locate/grasp/place convenience hooks.",),
    ),
    "vimabench": _profile(
        "vimabench",
        context=("observe_vima_prompt",),
        observe=("observe_vima_prompt", "observe_vima_scene"),
        enumerate_entities=("inspect_vima_instances",),
        inspect=("inspect_vima_instance", "inspect_vima_instances"),
        locate=("inspect_vima_instance",),
        geometry=("inspect_vima_instance",),
        evidence=("record_vima_evidence", "record_w4_evidence"),
        prepare=("build_vima_pick_place_action",),
        execute=("submit_vima_action",),
    ),
    "cliport": _profile(
        "cliport",
        context=("get_cliport_task_language_goal",),
        observe=("observe_cliport_rgbd",),
        enumerate_entities=("inspect_cliport_instances",),
        inspect=("inspect_cliport_instance", "inspect_cliport_instances"),
        locate=("inspect_cliport_instance",),
        geometry=("inspect_cliport_instance", "inspect_cliport_instances"),
        evidence=("record_cliport_evidence", "record_w4_evidence"),
        prepare=("submit_cliport_pick_place_action",),
        execute=("submit_cliport_pick_place_action",),
    ),
    "vlabench": _profile(
        "vlabench",
        backend_kind="benchmark_local_runtime",
        context=("get_vlabench_instruction",),
        observe=("observe_vlabench_scene",),
        enumerate_entities=("observe_vlabench_scene",),
        inspect=("inspect_vlabench_visual_evidence", "locate_vlabench_entity"),
        locate=("locate_vlabench_entity",),
        geometry=("locate_vlabench_entity",),
        evidence=("record_vlabench_evidence", "ground_vlabench_visual_target", "record_w4_evidence"),
        prepare=("ground_vlabench_visual_target", "move_vlabench_ee_to", "grasp_vlabench_entity", "lift_vlabench_ee", "place_vlabench_entity_in", "execute_vlabench_skill"),
        execute=("move_vlabench_ee_to", "open_vlabench_gripper", "close_vlabench_gripper", "lift_vlabench_ee", "grasp_vlabench_entity", "place_vlabench_entity_in", "settle_vlabench_scene", "execute_vlabench_skill"),
        policy=("execute_vlabench_skill",),
        notes=("Adapter can wrap the benchmark-local VLABench runtime or the live smoke backend.",),
    ),
    "robocasa": _profile(
        "robocasa",
        context=("observe_robocasa_kitchen_state",),
        observe=("observe_robocasa_kitchen_state", "observe_robocasa_rgbd"),
        enumerate_entities=("inspect_robocasa_object", "inspect_robocasa_fixture"),
        inspect=("inspect_robocasa_object", "inspect_robocasa_fixture", "inspect_robocasa_affordance", "inspect_robocasa_button_contact_frame", "inspect_robocasa_transport_state"),
        locate=("locate_robocasa_object", "locate_robocasa_fixture", "ground_robocasa_visual_target", "inspect_robocasa_object"),
        geometry=("inspect_robocasa_affordance", "inspect_robocasa_button_contact_frame", "inspect_robocasa_transport_state"),
        affordance=("inspect_robocasa_affordance", "inspect_robocasa_button_contact_frame"),
        evidence=("record_robocasa_evidence", "record_w4_evidence"),
        prepare=("open_robocasa_fixture", "close_robocasa_fixture", "press_robocasa_fixture_button", "inspect_robocasa_button_contact_frame", "sweep_robocasa_button_contact_candidates", "move_robocasa_ee_to", "grasp_robocasa_object", "place_robocasa_object_at"),
        execute=("open_robocasa_fixture", "close_robocasa_fixture", "press_robocasa_fixture_button", "inspect_robocasa_button_contact_frame", "sweep_robocasa_button_contact_candidates", "settle_robocasa_environment", "move_robocasa_ee_to", "grasp_robocasa_object", "place_robocasa_object_at"),
    ),
    "capx": _profile(
        "capx",
        backend_kind="comparator_runtime",
        context=("get_capx_prompt", "get_capx_available_apis", "get_capx_api_trace", "build_capx_live_command_spec"),
        observe=("observe_capx_scene",),
        enumerate_entities=("enumerate_capx_objects",),
        inspect=("capx_get_object_pose", "capx_sample_grasp_pose"),
        locate=("capx_get_object_pose", "capx_sample_grasp_pose"),
        geometry=("compose_capx_geometry",),
        affordance=("capx_sample_grasp_pose",),
        evidence=("record_capx_evidence", "record_w4_evidence"),
        prepare=("capx_sample_grasp_pose", "capx_goto_pose", "capx_open_gripper", "capx_close_gripper", "submit_capx_action"),
        execute=("submit_capx_action", "capx_goto_pose", "capx_open_gripper", "capx_close_gripper"),
        notes=("Comparator runtime exposes API-level calls; live execution requires an injected upstream session.",),
    ),
    "rlbench": _profile(
        "rlbench",
        context=("observe_rlbench_scene",),
        observe=("observe_rlbench_scene", "inspect_rlbench_visual_evidence"),
        enumerate_entities=("inspect_rlbench_visual_evidence",),
        inspect=("inspect_rlbench_visual_evidence", "ground_rlbench_target"),
        locate=("ground_rlbench_target",),
        geometry=("inspect_rlbench_visual_evidence", "ground_rlbench_target"),
        affordance=("ground_rlbench_target",),
        evidence=("record_rlbench_evidence", "record_w4_evidence"),
        prepare=("move_rlbench_arm_to", "open_rlbench_gripper", "close_rlbench_gripper", "step_rlbench_action"),
        execute=("move_rlbench_arm_to", "open_rlbench_gripper", "close_rlbench_gripper", "step_rlbench_action"),
    ),
    "calvin": _profile(
        "calvin",
        context=("get_calvin_language_subgoal", "get_calvin_runtime_context"),
        observe=("observe_calvin_state", "observe_calvin_cameras"),
        enumerate_entities=("observe_calvin_state",),
        inspect=("observe_calvin_state", "observe_calvin_cameras"),
        locate=("observe_calvin_state",),
        geometry=("observe_calvin_state",),
        affordance=("get_calvin_language_subgoal",),
        evidence=("record_calvin_evidence", "record_w4_evidence"),
        prepare=("execute_calvin_language_skill", "submit_calvin_action"),
        execute=("execute_calvin_language_skill", "submit_calvin_action"),
        policy=("execute_calvin_language_skill",),
    ),
    "behavior1k": _profile(
        "behavior1k",
        context=("get_behavior1k_task_context",),
        observe=("observe_behavior1k_state", "inspect_behavior1k_visual_evidence"),
        enumerate_entities=("inspect_behavior1k_asset",),
        inspect=("inspect_behavior1k_asset", "inspect_behavior1k_visual_evidence", "inspect_behavior1k_object_state", "inspect_behavior1k_control", "inspect_behavior1k_contacts"),
        locate=("inspect_behavior1k_asset", "inspect_behavior1k_visual_evidence"),
        geometry=("inspect_behavior1k_visual_evidence", "inspect_behavior1k_control", "inspect_behavior1k_contacts"),
        affordance=("inspect_behavior1k_object_state", "inspect_behavior1k_contacts", "inspect_behavior1k_control"),
        evidence=("record_behavior1k_evidence",),
        prepare=("convert_behavior1k_grounding_to_controller_input", "execute_behavior1k_action_sequence", "execute_behavior1k_controller_command", "navigate_behavior1k_base_to_pose", "run_behavior1k_semantic_action"),
        execute=("submit_behavior1k_action", "step_behavior1k_action", "execute_behavior1k_action_sequence", "execute_behavior1k_controller_command", "navigate_behavior1k_base_to_pose", "run_behavior1k_semantic_action", "settle_behavior1k"),
        policy=("run_behavior1k_semantic_action",),
    ),
    "robocasa365": _profile(
        "robocasa365",
        context=("get_robocasa365_task_context",),
        observe=("observe_robocasa365_kitchen_state", "observe_robocasa365_rgbd"),
        enumerate_entities=("inspect_robocasa365_object", "inspect_robocasa365_fixture"),
        inspect=("inspect_robocasa365_object", "inspect_robocasa365_fixture", "inspect_robocasa365_affordance", "inspect_robocasa365_button_contact_frame", "inspect_robocasa365_transport_state"),
        locate=("locate_robocasa365_object", "locate_robocasa365_fixture", "ground_robocasa365_visual_target", "inspect_robocasa365_object"),
        geometry=("inspect_robocasa365_affordance", "inspect_robocasa365_button_contact_frame", "inspect_robocasa365_transport_state"),
        affordance=("inspect_robocasa365_affordance", "inspect_robocasa365_button_contact_frame"),
        evidence=("record_robocasa365_evidence",),
        prepare=("open_robocasa365_fixture", "close_robocasa365_fixture", "press_robocasa365_fixture_button", "inspect_robocasa365_button_contact_frame", "sweep_robocasa365_button_contact_candidates", "move_robocasa365_ee_to", "grasp_robocasa365_object", "place_robocasa365_object_at"),
        execute=("open_robocasa365_fixture", "close_robocasa365_fixture", "press_robocasa365_fixture_button", "inspect_robocasa365_button_contact_frame", "sweep_robocasa365_button_contact_candidates", "settle_robocasa365_environment", "move_robocasa365_ee_to", "grasp_robocasa365_object", "place_robocasa365_object_at"),
    ),
    "robowits": _profile(
        "robowits",
        context=("get_robowits_task_context",),
        observe=("observe_robowits_state", "inspect_robowits_live_observation", "inspect_robowits_camera_pixels"),
        enumerate_entities=("inspect_robowits_scene_or_tool", "inspect_robowits_scene_evidence", "inspect_robowits_object_poses"),
        inspect=("inspect_robowits_scene_or_tool", "inspect_robowits_scene_evidence", "inspect_robowits_live_observation", "inspect_robowits_camera_pixels", "inspect_robowits_object_poses", "inspect_robowits_grasp_state", "inspect_robowits_contact_stability"),
        locate=("locate_robowits_entity",),
        geometry=("locate_robowits_entity", "inspect_robowits_object_poses", "query_robowits_wrist_orientations", "query_robowits_motion", "inspect_robowits_contact_stability"),
        affordance=("inspect_robowits_grasp_state", "inspect_robowits_contact_stability", "query_robowits_motion"),
        evidence=("record_robowits_evidence",),
        prepare=("query_robowits_motion", "execute_robowits_ee_control"),
        execute=("execute_robowits_ee_control", "inspect_robowits_motion_outcome"),
    ),
    "robotwin2": _profile(
        "robotwin2",
        context=("observe_robotwin2_scene", "check_robotwin2_asset_readiness"),
        observe=("observe_robotwin2_scene", "observe_robotwin2_visual"),
        enumerate_entities=("locate_robotwin2_actor",),
        inspect=("locate_robotwin2_actor", "inspect_robotwin2_actor_points", "measure_robotwin2_actor_visual", "inspect_robotwin2_arm_pose"),
        locate=("locate_robotwin2_actor",),
        geometry=("inspect_robotwin2_actor_points", "measure_robotwin2_actor_visual", "inspect_robotwin2_arm_pose", "probe_robotwin2_motion_plan"),
        affordance=("inspect_robotwin2_actor_points", "probe_robotwin2_motion_plan"),
        evidence=("record_robotwin2_evidence",),
        prepare=("probe_robotwin2_motion_plan", "move_robotwin2_arm", "set_robotwin2_gripper", "submit_robotwin2_ee_action", "execute_robotwin2_actions"),
        execute=("move_robotwin2_arm", "set_robotwin2_gripper", "submit_robotwin2_ee_action", "execute_robotwin2_actions"),
        policy=("execute_robotwin2_actions",),
    ),
    "robodojo": _profile(
        "robodojo",
        context=("get_robodojo_robot_action_schema", "list_robodojo_policy_skills"),
        observe=("observe_robodojo_runtime_report", "observe_robodojo_visual"),
        enumerate_entities=("list_robodojo_policy_skills", "measure_robodojo_public_geometry"),
        inspect=("inspect_robodojo_policy_skill", "inspect_robodojo_collision_geometry", "measure_robodojo_contact_state"),
        locate=("measure_robodojo_public_geometry",),
        geometry=("measure_robodojo_public_geometry", "inspect_robodojo_collision_geometry", "measure_robodojo_contact_state"),
        affordance=("list_robodojo_policy_skills", "inspect_robodojo_policy_skill"),
        evidence=("record_robodojo_evidence",),
        prepare=("build_robodojo_joint_action", "compile_robodojo_ee_path", "compile_robodojo_contact_lift_actions"),
        execute=("run_robodojo_policy_skill", "submit_robodojo_low_level_actions"),
        policy=("run_robodojo_policy_skill",),
    ),
}


_POLICY_SKILL_PROFILES = {
    benchmark_id: _catalog_profile(
        benchmark_id,
        adapter="PolicySkillAgentRuntimeBackend",
        backend_kind="policy_skill_runtime",
        context=(f"get_{benchmark_id}_policy_context", f"probe_{benchmark_id}_policy_runtime"),
        observe=(f"inspect_{benchmark_id}_observation_contract",),
        enumerate_entities=(f"inspect_{benchmark_id}_observation_contract",),
        inspect=(f"inspect_{benchmark_id}_observation_contract",),
        affordance=(f"probe_{benchmark_id}_policy_runtime",),
        evidence=(f"score_or_record_{benchmark_id}_rollout_evidence",),
        prepare=(f"submit_{benchmark_id}_policy_action_chunk",),
        execute=(f"submit_{benchmark_id}_policy_action_chunk",),
        policy=(f"call_{benchmark_id}_policy_skill",),
        notes=(
            "Policy-as-skill target: contract conformance is scored here; official task success belongs to the downstream benchmark.",
        ),
    )
    for benchmark_id in ("openvla", "openpi", "lerobot", "octo")
}

OPENHANDS_ADAPTER_PROFILES: dict[str, UniversalAdapterProfile] = {
    **UNIVERSAL_ADAPTER_PROFILES,
    "mmsi_bench": _catalog_profile(
        "mmsi_bench",
        adapter="MMSISpatialBackend",
        backend_kind="diagnostic_runtime",
        context=("get_mmsi_task_context",),
        observe=("inspect_mmsi_image", "inspect_mmsi_pixels"),
        enumerate_entities=("inspect_mmsi_image",),
        inspect=("inspect_mmsi_image", "inspect_mmsi_pixels"),
        locate=("inspect_mmsi_image", "inspect_mmsi_pixels"),
        geometry=("estimate_mmsi_camera_motion", "compare_spatial_relation"),
        evidence=("record_spatial_evidence",),
        prepare=("submit_spatial_answer",),
        execute=("submit_spatial_answer",),
    ),
    "alfworld": _catalog_profile(
        "alfworld",
        adapter="ALFWorldAgentRuntimeBackend",
        backend_kind="text_interaction_runtime",
        context=("get_alfworld_task_context",),
        observe=("observe_alfworld_state", "list_alfworld_actions"),
        enumerate_entities=("observe_alfworld_state", "list_alfworld_actions"),
        inspect=("observe_alfworld_state", "list_alfworld_actions"),
        locate=("observe_alfworld_state", "list_alfworld_actions"),
        affordance=("list_alfworld_actions",),
        evidence=("record_alfworld_evidence",),
        prepare=("go_to", "take", "put", "open", "close", "toggle"),
        execute=("go_to", "take", "put", "open", "close", "toggle"),
    ),
    "scienceworld": _catalog_profile(
        "scienceworld",
        adapter="ScienceWorldAgentRuntimeBackend",
        backend_kind="text_interaction_runtime",
        context=("get_scienceworld_task_context",),
        observe=("observe_scienceworld_state", "list_scienceworld_actions"),
        enumerate_entities=("observe_scienceworld_state", "list_scienceworld_actions"),
        inspect=("observe_scienceworld_state", "focus_scienceworld_object"),
        locate=("observe_scienceworld_state", "list_scienceworld_actions"),
        affordance=("list_scienceworld_actions",),
        evidence=("record_scienceworld_evidence",),
        prepare=(
            "focus_scienceworld_object",
            "pick_up_scienceworld_object",
            "go_scienceworld_location",
            "move_scienceworld_object_to",
            "step_scienceworld_action",
        ),
        execute=(
            "focus_scienceworld_object",
            "pick_up_scienceworld_object",
            "go_scienceworld_location",
            "move_scienceworld_object_to",
            "step_scienceworld_action",
        ),
    ),
    "esi": _catalog_profile(
        "esi",
        adapter="SpatialDiagnosticBackend",
        backend_kind="diagnostic_runtime",
        context=("get_task_context",),
        observe=("inspect_view", "inspect_video_frames"),
        enumerate_entities=("detect_video_objects", "track_video_objects"),
        inspect=("inspect_view", "inspect_video_frames", "segment_video_objects"),
        locate=("detect_video_objects", "track_video_objects", "segment_video_objects"),
        geometry=("track_video_objects", "detect_video_objects"),
        evidence=("write_evidence",),
        prepare=("submit_answer",),
        execute=("submit_answer",),
        notes=("One catalog directory exposes both esi_spatial and vsi_bench runtime identities.",),
    ),
    "spatialclaw": _catalog_profile(
        "spatialclaw",
        adapter="SpatialClawBackend",
        backend_kind="diagnostic_runtime",
        context=("get_spatial_task_context",),
        observe=("inspect_spatial_observation", "inspect_spatial_view_set", "inspect_spatial_pixels"),
        enumerate_entities=("detect_spatial_image_objects", "segment_spatial_image_objects"),
        inspect=("inspect_spatial_observation", "inspect_spatial_pixels", "analyze_spatialclaw_visual_query"),
        locate=("detect_spatial_image_objects", "segment_spatial_image_objects"),
        geometry=("run_geometry_tool", "infer_mindcube_motion_from_views", "analyze_spatialclaw_visual_query"),
        evidence=("record_spatial_evidence",),
        prepare=("submit_spatial_answer",),
        execute=("submit_spatial_answer",),
    ),
    **_POLICY_SKILL_PROFILES,
}


def benchmark_family_registry() -> list[JsonDict]:
    return [dict(item) for item in BENCHMARK_FAMILY_REGISTRY]


def universal_interface_specs() -> list[UniversalInterfaceSpec]:
    return list(UNIVERSAL_INTERFACE_SPECS)


def universal_adapter_profile(benchmark_id: str) -> UniversalAdapterProfile:
    normalized = _normalize_benchmark_id(benchmark_id)
    return OPENHANDS_ADAPTER_PROFILES.get(
        normalized,
        UniversalAdapterProfile(
            benchmark_id=normalized,
            family=normalized,
            adapter="unknown",
            backend_kind="unknown",
        ),
    )


def universal_adapter_profiles(*, include_internal_names: bool = True) -> dict[str, JsonDict]:
    return {
        benchmark_id: profile.to_dict(include_internal_names=include_internal_names)
        for benchmark_id, profile in UNIVERSAL_ADAPTER_PROFILES.items()
    }


def openhands_adapter_profiles(*, include_internal_names: bool = True) -> dict[str, JsonDict]:
    """Return adapter profiles for every canonical OpenHands directory target."""

    return {
        benchmark_id: profile.to_dict(include_internal_names=include_internal_names)
        for benchmark_id, profile in OPENHANDS_ADAPTER_PROFILES.items()
    }


def universal_interface_contracts() -> list[JsonDict]:
    contracts: list[JsonDict] = []
    for spec in UNIVERSAL_INTERFACE_SPECS:
        handle_io = _UNIVERSAL_INTERFACE_HANDLE_IO[spec.name]
        contracts.append(
            {
                "name": spec.name,
                "agent_callable": True,
                "native_details_hidden": True,
                "disclosure_level": spec.disclosure_level,
                "capability_tags": list(spec.capability_tags),
                "summary": spec.summary,
                "input_schema": deepcopy(spec.input_schema),
                "output_schema": deepcopy(spec.output_schema),
                "handle_inputs": list(handle_io["handle_inputs"]),
                "handle_outputs": list(handle_io["handle_outputs"]),
                "requires_evidence": spec.requires_evidence,
                "preconditions": list(spec.preconditions),
                "side_effects": list(spec.side_effects),
                "error_codes": [*spec.error_codes, UNIVERSAL_BUDGET_ERROR_CODE],
            }
        )
    return contracts


def universal_error_code_registry() -> JsonDict:
    producer_by_code: dict[str, list[str]] = {}
    for spec in UNIVERSAL_INTERFACE_SPECS:
        for code in spec.error_codes:
            producer_by_code.setdefault(code, []).append(spec.name)
        producer_by_code.setdefault(UNIVERSAL_BUDGET_ERROR_CODE, []).append(spec.name)
    gateway_codes = (
        "unknown_interface",
        "missing_interface_handler",
        "gateway_exception",
        "verifier_budget_exhausted",
    )
    for code in gateway_codes:
        producer_by_code.setdefault(code, []).append("universal_gateway")
    by_code = {code: sorted(producers) for code, producers in sorted(producer_by_code.items())}
    return {
        "error_code_count": len(by_code),
        "by_code": by_code,
        "error_codes": [
            {"code": code, "producer_interfaces": producers}
            for code, producers in by_code.items()
        ],
    }


def universal_contract_manifest(*, include_internal_adapter_names: bool = True) -> JsonDict:
    return {
        "schema_version": UNIVERSAL_CONTRACT_SCHEMA_VERSION,
        "interface_count": len(UNIVERSAL_INTERFACE_NAMES),
        "interfaces": [spec.to_card(schema_visible=True).to_dict() for spec in UNIVERSAL_INTERFACE_SPECS],
        "interface_contracts": universal_interface_contracts(),
        "error_code_registry": universal_error_code_registry(),
        "handle_contract": deepcopy(UNIVERSAL_HANDLE_CONTRACT),
        "evidence_contract": deepcopy(UNIVERSAL_EVIDENCE_CONTRACT),
        "budget_contract": deepcopy(UNIVERSAL_BUDGET_CONTRACT),
        "benchmark_family_count": len(BENCHMARK_FAMILY_REGISTRY),
        "benchmark_families": benchmark_family_registry(),
        "adapter_profiles": universal_adapter_profiles(include_internal_names=include_internal_adapter_names),
        "disclosure_policy": {
            "l0": "Interface names and high-level capability summaries only.",
            "l1": "Selected interface parameter schemas.",
            "l2": "Observation, entity, geometry, affordance, and evidence handles.",
            "l3": "Prepared semantic actions or policy handles.",
            "l4": "Adapter consumes low-level action parameters only at execution time.",
            "l5": "Official scoring and hidden task internals remain harness-side.",
        },
    }


def universal_interface_cards(*, level: str | None = None, schema_visible: bool = True) -> list[PrimitiveCard]:
    cards = [spec.to_card(schema_visible=schema_visible) for spec in UNIVERSAL_INTERFACE_SPECS]
    if level == "L0":
        cards = [spec.to_card(schema_visible=False) for spec in UNIVERSAL_INTERFACE_SPECS]
    elif level is not None:
        cards = [card for card in cards if card.abstraction_level == level]
    return cards


class UniversalEmbodiedBackend(EmbodiedBackend):
    """Disclosure gateway that exposes one stable 13-interface surface.

    The wrapped benchmark backend keeps its native primitive names and official
    verifier. This gateway presents only high-level public capabilities to the
    coding agent and binds execution to evidence/action handles before touching
    native primitives.
    """

    def __init__(self, backend: EmbodiedBackend) -> None:
        self.backend = backend
        self._task: TaskSpec | None = None
        self._last_observation: Observation | None = None
        self._handles: dict[str, JsonDict] = {}
        self._actions: dict[str, JsonDict] = {}
        self._last_errors: list[JsonDict] = []
        self._budget_usage: dict[str, int] = {}
        self._counter = 0

    def reset(self, task_id: str, seed: int | None = None, config: JsonDict | None = None,
              *, budgets: dict[str, int] | None = None) -> TaskSpec:
        overrides = dict(budgets or {})
        for name, limit in overrides.items():
            if name not in {"primitive_calls", "verifier_calls"}:
                raise ValueError(f"Unsupported gateway budget: {name}")
            if type(limit) is not int or limit <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._task = self.backend.reset(task_id=task_id, seed=seed, config=config)
        self._task.budgets.update(overrides)
        self._last_observation = None
        self._handles = {}
        self._actions = {}
        self._last_errors = []
        self._budget_usage = {}
        self._counter = 0
        self.record_event(
            "universal_gateway_reset",
            {"interface_count": len(UNIVERSAL_INTERFACE_NAMES), "benchmark_id": self._benchmark_id()},
        )
        return self._public_task_spec(self._task)

    def observe(self) -> Observation:
        observation = self.backend.observe()
        sanitized = Observation(
            step=observation.step,
            data=_sanitize_public_observation_payload(observation.data),
            artifacts=list(observation.artifacts),
            metadata=_sanitize_public_observation_payload(observation.metadata),
        )
        self._last_observation = sanitized
        self.record_event("universal_gateway_observe", {"observation": sanitized.to_dict()})
        return sanitized

    def _latest_public_observation_data(self) -> JsonDict | None:
        if self._last_observation is None:
            return None
        if isinstance(self._last_observation.data, dict) and self._last_observation.data:
            return deepcopy(self._last_observation.data)
        observation = self._last_observation.to_dict()
        return observation if observation else None

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        cards = universal_interface_cards(level=level, schema_visible=level != "L0")
        self.record_event("universal_gateway_list_interfaces", {"level": level, "count": len(cards)})
        return cards

    def unavailable_interfaces(self) -> list[str]:
        """Return universal interfaces that have no direct route for this task."""

        profile = self._adapter_profile()
        if not profile.policy_primitives or self._benchmark_id() == "robotwin2":
            return ["policy.invoke"]
        return []

    def call_primitive(self, name: str | None = None, **kwargs: Any) -> PrimitiveResult:
        interface = self._normalize_interface(name or kwargs.pop("interface", None))
        if interface is None:
            return self._error("unknown", "unknown_interface", {"requested": name})
        handler = getattr(self, f"_call_{interface.replace('.', '_')}", None)
        if handler is None:
            return self._error(interface, "missing_interface_handler")
        budget_error = self._consume_interface_budget(interface)
        if budget_error is not None:
            self.record_event(
                "universal_gateway_call",
                {
                    "interface": interface,
                    "kwargs": _sanitize_agent_payload(kwargs),
                    "result": budget_error.to_dict(),
                },
            )
            return budget_error
        if interface in {"action.execute", "policy.invoke"}:
            self.record_event(
                "universal_gateway_call_started",
                {"interface": interface, "kwargs": _sanitize_agent_payload(kwargs)},
            )
        try:
            result = handler(**kwargs)
        except Exception as exc:  # noqa: BLE001 - gateway must keep agent sessions recoverable.
            result = self._error(interface, "gateway_exception", {"detail": f"{type(exc).__name__}: {exc}"})
        result.metadata = {
            **result.metadata,
            "budget": self._budget_snapshot(),
        }
        self.record_event(
            "universal_gateway_call",
            {"interface": interface, "kwargs": _sanitize_agent_payload(kwargs), "result": result.to_dict()},
        )
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        if not self._consume_named_budget("verifier_calls"):
            result = VerificationResult(
                ok=False,
                scope=scope,
                message="verifier call budget exhausted",
                metadata={
                    "error_code": "verifier_budget_exhausted",
                    "budget": self._budget_snapshot(),
                    "official_verifier_agent_callable": False,
                },
            )
            self.record_event(
                "universal_gateway_harness_verify",
                {"scope": scope, "ok": False, "error": "verifier_budget_exhausted"},
            )
            return result
        result = self.backend.verify(scope=scope, **kwargs)
        result.metadata = {**result.metadata, "budget": self._budget_snapshot()}
        self.record_event("universal_gateway_harness_verify", {"scope": scope, "ok": result.ok})
        return result

    def get_trace(self) -> EpisodeTrace:
        return self.backend.get_trace()

    def _budget_limit(self, name: str) -> int | None:
        if self._task is None:
            return None
        value = self._task.budgets.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            return None
        return int(value)

    def _consume_named_budget(self, name: str) -> bool:
        used = self._budget_usage.get(name, 0)
        limit = self._budget_limit(name)
        if limit is not None and used >= limit:
            return False
        self._budget_usage[name] = used + 1
        self._record_budget_metrics()
        return True

    def _consume_interface_budget(self, interface: str) -> PrimitiveResult | None:
        names = ["primitive_calls"]
        if interface == "policy.invoke":
            names.append("policy_calls")
        exhausted = [
            name
            for name in names
            if self._budget_limit(name) is not None
            and self._budget_usage.get(name, 0) >= int(self._budget_limit(name) or 0)
        ]
        if exhausted:
            return self._error(
                interface,
                UNIVERSAL_BUDGET_ERROR_CODE,
                {
                    "exhausted": exhausted,
                    "budget": self._budget_snapshot(),
                    "suggested_next_step": "stop issuing tools and return the latest public progress",
                },
            )
        for name in names:
            self._budget_usage[name] = self._budget_usage.get(name, 0) + 1
        self._record_budget_metrics()
        return None

    def _budget_snapshot(self) -> JsonDict:
        limits = (
            _sanitize_agent_payload(self._task.budgets)
            if self._task is not None
            else {}
        )
        remaining: JsonDict = {}
        for name, value in limits.items():
            if name not in {"primitive_calls", "policy_calls", "verifier_calls"}:
                remaining[name] = None
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                remaining[name] = None
                continue
            remaining[name] = max(0, int(value) - self._budget_usage.get(name, 0))
        exhausted = sorted(
            name
            for name, value in remaining.items()
            if value == 0 and name in {"primitive_calls", "policy_calls", "verifier_calls"}
        )
        return {
            "limits": limits,
            "used": dict(sorted(self._budget_usage.items())),
            "remaining": remaining,
            "exhausted": exhausted,
        }

    def _record_budget_metrics(self) -> None:
        if self._task is None:
            return
        try:
            self.get_trace().metrics["budget"] = self._budget_snapshot()
        except Exception:
            pass

    def list_task_ids(self) -> list[str]:
        if hasattr(self.backend, "list_task_ids"):
            return list(getattr(self.backend, "list_task_ids")())
        raise AttributeError("Wrapped backend does not expose list_task_ids().")

    def _call_task_context(self, query: str | None = None, agent_context: JsonDict | None = None) -> PrimitiveResult:
        self._require_task()
        native_cards = self.backend.list_primitives()
        calvin_policy_context = self._calvin_task_context_policy_call(agent_context=agent_context)
        payload = {
            "task": _sanitize_agent_payload(self._public_task_spec(self._task).to_dict()),
            "benchmark": self._benchmark_summary(),
            "budget": self._budget_snapshot(),
            "query": query,
            "agent_context": agent_context or {},
            "interfaces": [
                {
                    "name": spec.name,
                    "level": spec.disclosure_level,
                    "summary": spec.summary,
                    "capability_tags": list(spec.capability_tags),
                }
                for spec in UNIVERSAL_INTERFACE_SPECS
            ],
            "native_adapter": {
                "native_primitive_count": len(native_cards),
                "native_names_hidden": True,
                "adapter_profile": self._adapter_profile().to_dict(include_internal_names=False),
            },
        }
        unavailable_interfaces = self.unavailable_interfaces()
        if unavailable_interfaces:
            payload["unsupported_interfaces"] = unavailable_interfaces
        if self._benchmark_id() == "vimabench":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {
                    "type": "pick_place",
                    "source": "<entity_handle whose label/prompt_asset_key is dragged_obj_1>",
                    "target": "<entity_handle whose label/prompt_asset_key is base_obj>",
                },
                "public_rationale": (
                    "Use the exact public VIMA prompt-placeholder bindings returned by entity.enumerate; "
                    "the adapter converts their top-view segmentation masks into the native VIMA action. "
                    "Do not manually convert front-view pixels into action coordinates."
                ),
            }
        if self._benchmark_id() == "maniskill":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {"type": "control", "control": "<values matching the selected environment's native action space>", "repeat": 1},
                "public_rationale": "Use the selected task instruction, current observation and native controller action space to choose controls.",
            }
        if self._benchmark_id() == "capx":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {
                    "type": "pick_place",
                    "source": "<entity handle for the red/source cube>",
                    "target": "<entity handle for the green/support cube>",
                },
                "public_rationale": (
                    "Use scene.observe and entity.enumerate to bind the two caller-selected cubes, then execute "
                    "one semantic pick_place action. The adapter refreshes their public same-episode poses and "
                    "performs the open, approach, grasp, lift, stack, release, and settling sequence. Do not invent "
                    "or call benchmark-native API names."
                ),
            }
        if self._benchmark_id() == "capx" and self._task is not None and "native_config" in self._task.tags:
            api_info = self.backend.call_primitive("get_capx_available_apis")
            if api_info.ok:
                payload["public_native_api"] = _sanitize_agent_payload(api_info.output)
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {"type": "native_api", "api_action": "<original public function name>", "parameters": {}},
                "public_rationale": "Choose a documented original CaPX function and its original keyword arguments. Read its return value from action.execute output.native_api_result. Non-JSON native values stay in this episode: pass their native_value_handle back unchanged; small arrays also expose values for calculations.",
            }
        if self._benchmark_id() == "vlabench":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {"type": "pick_place", "source": "<source entity handle selected from the current instruction>", "target": "<destination entity handle selected from the current instruction>"},
                "public_rationale": "Ground both entities in the selected task's native observation; select the native SkillLib action appropriate to its instruction.",
            }
        if self._benchmark_id() in {"robocasa", "robocasa365"}:
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {
                    "type": "press",
                    "target": "<entity_handle for the instructed fixture>",
                    "button_name": "<public button name returned by fixture affordance evidence>",
                    "max_attempts": 16,
                    "candidate_limit": 16,
                    "mobile_base_enabled": True,
                    "mobile_base_active_phases": [
                        "button_approach",
                        "button_press",
                        "button_contact_seek",
                    ],
                    "base_delta_frame": "base",
                    "base_xy_deadband": 0.10,
                    "base_max_delta": 0.15,
                },
                "public_rationale": (
                    "Use scene.observe, entity.enumerate, and entity.locate for the instructed fixture, "
                    "then pass the fresh ev:* location handle to action.prepare. Keep the action semantic: "
                    "type=press, target=<fixture entity handle>, and button_name=<public affordance name>. "
                    "Do not copy native-looking skill names or nested parameter dictionaries from diagnostic "
                    "payloads; the adapter privately grounds the button and runs its contact-frame sweep."
                ),
            }
        if self._benchmark_id() == "robowits":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {
                    "type": "pick_place",
                    "source": "<entity handle for the next cube from public evidence>",
                    "target_position": "<desired object-center xyz derived from public target and bounds>",
                    "arm": "left",
                    "max_translation": 0.3,
                    "repeat_steps": 12,
                },
                "public_rationale": (
                    "RoboWits exposes no policy.invoke implementation. Preserve the ev:* handle returned by "
                    "scene.observe, then use the semantic pick_place action to execute a compact backend-native "
                    "approach, grasp, lift, transport, release, and retreat sequence. target_position is the desired "
                    "object center, not an end-effector pose. For the two-layer stack, derive from public bounds a "
                    "pair of base centers on the table, separated by about one cube width and centered around the "
                    "public marker; derive the apex center at the marker xy one cube height above the base centers. "
                    "Use fresh public entity evidence for each source. In this official EE_ABS stack task, retain "
                    "the public default left-arm wrist orientation for all three grasps; the right arm's initial "
                    "wrist orientation is not an interchangeable downward grasp pose. If the selected arm begins more than 0.3 m "
                    "from the source approach pose, first issue one bounded move_ee waypoint; otherwise do not expand "
                    "the composite into manual waypoints."
                ),
            }
        if self._benchmark_id() == "robotwin2":
            payload["suggested_action_template"] = {
                "interface": "action.prepare",
                "action": {
                    "type": "pick_place",
                    "source": "<entity handle for the instructed movable object>",
                    "target": "<entity handle for the instructed destination>",
                    "arm": "right",
                },
                "public_rationale": (
                    "RoboTwin2 exposes no policy.invoke implementation. Use scene.observe and entity.enumerate "
                    "to bind the instructed object and destination, preserve the fresh ev:* scene handle, and "
                    "prepare one semantic pick_place action. The adapter remeasures both caller-selected actors "
                    "from the official RGB-D/calibration stream before executing a bounded open, approach, grasp, "
                    "lift, transport, place, and release sequence. For place_empty_cup, select cup as source, "
                    "coaster as target, and the right arm. Do not manually copy object registry poses as gripper "
                    "poses or invent native primitive names."
                ),
            }
        if (os.getenv("EMBODIED_ARENA_POOL_ENTRY_ID")
                and self._benchmark_id() in {"robocasa", "robocasa365", "robowits", "robotwin2"}):
            # Default smoke tasks are not instructions for a selected native
            # episode (e.g. collecting screws is not a cube-stacking task).
            payload.pop("suggested_action_template", None)
            payload["action_guidance"] = (
                "Follow the selected episode's instruction and current observations. "
                "Use affordance.inspect to inspect available actions and parameters; "
                "choose controls for the current task and robot."
            )
        if calvin_policy_context is not None:
            payload.update(calvin_policy_context)
        return PrimitiveResult(name="task.context", ok=True, output=_sanitize_agent_payload(payload))

    def _call_scene_observe(
        self,
        view: str | None = None,
        modality: str | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        native_scene = self._call_benchmark_native_scene_observe(
            view=view,
            modality=modality,
            query=query,
            agent_context=agent_context,
        )
        if native_scene is not None:
            return native_scene
        observation = self.observe()
        handle = self._new_handle(
            "obs",
            {
                "kind": "observation",
                "benchmark_id": self._benchmark_id(),
                "view": view,
                "modality": modality,
                "query": query,
                "observation": observation.to_dict(),
                "agent_context": agent_context or {},
            },
        )
        evidence = self._new_handle("ev", {"kind": "scene_observation", "source_handle": handle})
        suggested_policy_call = self._suggested_policy_call_for_observation(
            observation_evidence_handle=evidence,
        )
        if suggested_policy_call is not None:
            self._handles[evidence]["suggested_policy_call"] = deepcopy(suggested_policy_call)
        output: JsonDict = {
            "observation_handle": handle,
            "evidence_handle": evidence,
            "observation": observation.to_dict(),
            "artifacts": list(observation.artifacts),
        }
        if suggested_policy_call is not None:
            output["suggested_next_interface"] = "policy.invoke"
            output["suggested_policy_call"] = suggested_policy_call
        return PrimitiveResult(
            name="scene.observe",
            ok=True,
            output=output,
            artifacts=[handle, evidence],
        )

    def _call_benchmark_native_scene_observe(
        self,
        *,
        view: str | None,
        modality: str | None,
        query: str | None,
        agent_context: JsonDict | None,
    ) -> PrimitiveResult | None:
        if self._benchmark_id() != "robodojo":
            return None
        names = self._available_native_names()
        if not {"observe_robodojo_runtime_report", "observe_robodojo_visual"}.intersection(names):
            return None

        native_results: list[PrimitiveResult] = []
        runtime_result: PrimitiveResult | None = None
        visual_result: PrimitiveResult | None = None
        if "observe_robodojo_runtime_report" in names:
            runtime_result = self.backend.call_primitive(
                "observe_robodojo_runtime_report",
                agent_context=agent_context,
            )
            native_results.append(runtime_result)
        if "observe_robodojo_visual" in names:
            visual_result = self.backend.call_primitive(
                "observe_robodojo_visual",
                query=query,
                agent_context=agent_context,
            )
            native_results.append(visual_result)

        observation_data: JsonDict | None = None
        selected_result = visual_result if visual_result is not None and visual_result.ok else runtime_result
        if visual_result is not None and isinstance(visual_result.output, dict):
            maybe_observation = visual_result.output.get("observation")
            if visual_result.ok and isinstance(maybe_observation, dict):
                observation_data = deepcopy(maybe_observation)
            elif selected_result is visual_result:
                observation_data = deepcopy(visual_result.output)
        if observation_data is None and runtime_result is not None and isinstance(runtime_result.output, dict):
            observation_data = deepcopy(runtime_result.output)
        if observation_data is None and selected_result is not None and isinstance(selected_result.output, dict):
            observation_data = deepcopy(selected_result.output)
        if observation_data is None or not _looks_like_robodojo_scene_observation(observation_data):
            return None

        native_artifacts: list[str] = []
        for result in native_results:
            native_artifacts.extend(result.artifacts)
        observation = Observation(
            step=len(self.get_trace().events),
            data=_sanitize_public_observation_payload(observation_data),
            artifacts=native_artifacts,
            metadata={
                "benchmark_id": self._benchmark_id(),
                "native_scene_observe": True,
                "view": view,
                "modality": modality,
                "query": query,
            },
        )
        self._last_observation = observation
        handle = self._new_handle(
            "obs",
            {
                "kind": "observation",
                "benchmark_id": self._benchmark_id(),
                "view": view,
                "modality": modality,
                "query": query,
                "observation": observation.to_dict(),
                "agent_context": agent_context or {},
                "native_observe_results": [_public_native_result(result) for result in native_results],
            },
        )
        evidence_payload: JsonDict = {
            "kind": "scene_observation",
            "source_handle": handle,
            "native_artifacts": native_artifacts,
        }
        if visual_result is not None and isinstance(visual_result.output, dict):
            evidence_payload["visual_evidence_refs"] = deepcopy(visual_result.output.get("evidence_refs") or [])
            evidence_payload["visual_observation_contract"] = deepcopy(
                visual_result.output.get("visual_observation_contract") or {}
            )
        evidence = self._new_handle("ev", evidence_payload)
        return PrimitiveResult(
            name="scene.observe",
            ok=bool(selected_result.ok if selected_result is not None else True),
            output={
                "observation_handle": handle,
                "evidence_handle": evidence,
                "observation": observation.to_dict(),
                "artifacts": native_artifacts,
                "native_scene_observe": True,
            },
            artifacts=[handle, evidence, *native_artifacts],
            error=None if selected_result is None or selected_result.ok else selected_result.error,
        )

    def _call_entity_enumerate(
        self,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        entities = self._enumerate_entities(query=query)
        evidence = self._new_handle(
            "ev",
            {"kind": "entity_list", "entities": entities, "query": query, "agent_context": agent_context or {}},
        )
        return PrimitiveResult(
            name="entity.enumerate",
            ok=True,
            output={"entities": entities, "evidence_handle": evidence},
            artifacts=[evidence],
        )

    def _call_entity_inspect(
        self,
        entity: str | JsonDict,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        resolved = self._resolve_entity(entity)
        if resolved is None:
            return self._error("entity.inspect", "entity_not_found", {"entity": entity})
        attrs = deepcopy(resolved.get("attributes", {}))
        attrs.update({"label": resolved.get("label"), "entity_id": resolved.get("entity_id")})
        handle = str(resolved.get("entity_handle") or resolved.get("handle"))
        evidence = self._new_handle(
            "ev",
            {
                "kind": "entity_inspection",
                "entity_handle": handle,
                "attributes": attrs,
                "query": query,
                "agent_context": agent_context or {},
            },
        )
        return PrimitiveResult(
            name="entity.inspect",
            ok=True,
            output={"entity_handle": handle, "evidence_handle": evidence, "attributes": _sanitize_agent_payload(attrs)},
            artifacts=[evidence],
        )

    def _call_entity_locate(
        self,
        entity: str | JsonDict,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        resolved = self._resolve_entity(entity)
        if resolved is None:
            return self._error("entity.locate", "entity_not_found", {"entity": entity})
        location = self._locate_entity_with_native_adapter(resolved, query=query, agent_context=agent_context)
        handle = str(resolved.get("entity_handle") or resolved.get("handle"))
        evidence = self._new_handle(
            "ev",
            {
                "kind": "entity_location",
                "entity_handle": handle,
                "location": location,
                "query": query,
                "agent_context": agent_context or {},
            },
        )
        return PrimitiveResult(
            name="entity.locate",
            ok=True,
            output={"entity_handle": handle, "evidence_handle": evidence, "location": _sanitize_agent_payload(location)},
            artifacts=[evidence],
        )

    def _call_geometry_measure(
        self,
        subjects: list[str | JsonDict],
        measurement: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        resolved_subjects = [self._resolve_handle_or_entity(subject) for subject in subjects]
        missing = [subject for subject, resolved in zip(subjects, resolved_subjects) if resolved is None]
        if missing:
            return self._error("geometry.measure", "handle_not_found", {"missing": missing})
        payload = {
            "measurement": measurement or "public_spatial_summary",
            "subjects": [_sanitize_agent_payload(item) for item in resolved_subjects if item],
            "relations": self._derive_public_relations([item for item in resolved_subjects if item]),
            "agent_context": agent_context or {},
        }
        adapter_measurements = self._profile_geometry_measurements(
            [item for item in resolved_subjects if item],
            measurement=measurement,
            agent_context=agent_context,
        )
        if adapter_measurements:
            payload["adapter_measurements"] = adapter_measurements
        evidence = self._new_handle("ev", {"kind": "geometry_measurement", **payload})
        return PrimitiveResult(name="geometry.measure", ok=True, output={"evidence_handle": evidence, "measurement": payload}, artifacts=[evidence])

    def _call_affordance_inspect(
        self,
        entity: str | JsonDict | None = None,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        resolved = self._resolve_entity(entity) if entity is not None else None
        affordances = self._infer_affordances(resolved, query=query)
        evidence = self._new_handle(
            "ev",
            {
                "kind": "affordance",
                "entity": resolved,
                "query": query,
                "affordances": affordances,
                "agent_context": agent_context or {},
            },
        )
        suggested_policy_call = self._suggested_policy_call_for_affordance(
            affordances,
            affordance_evidence_handle=evidence,
        )
        if suggested_policy_call is not None:
            self._handles[evidence]["suggested_policy_call"] = deepcopy(suggested_policy_call)
        output: JsonDict = {"evidence_handle": evidence, "affordances": affordances}
        if suggested_policy_call is not None:
            output["suggested_next_interface"] = "policy.invoke"
            output["suggested_policy_call"] = suggested_policy_call
        return PrimitiveResult(name="affordance.inspect", ok=True, output=output, artifacts=[evidence])

    def _calvin_task_context_policy_call(self, *, agent_context: JsonDict | None = None) -> JsonDict | None:
        if self._benchmark_id() != "calvin":
            return None
        if self._select_policy_primitive("language_skill") is None:
            return None
        subgoal = self._calvin_public_language_subgoal_from_context(agent_context=agent_context)
        if not subgoal:
            subgoal = self._latest_public_language_subgoal()
        if not subgoal:
            return None
        return {
            "public_language_subgoal": subgoal,
            "language_subgoal": subgoal,
            "suggested_next_interface": "policy.invoke",
            "suggested_policy_call": {
                "interface": "policy.invoke",
                "policy": "language_skill",
                "inputs": {"subgoal": subgoal, "horizon": CALVIN_PUBLIC_LANGUAGE_SKILL_HORIZON},
                "evidence_handles": [],
                "public_rationale": (
                    "CALVIN exposes a public language-conditioned policy skill through task.context; "
                    "this initial language-skill policy call can use evidence_handles=[] because the "
                    "public language_subgoal itself is the required planning evidence."
                ),
            },
        }

    def _calvin_public_language_subgoal_from_context(self, *, agent_context: JsonDict | None = None) -> str | None:
        if "get_calvin_language_subgoal" not in self._available_native_names():
            return None
        result = self.backend.call_primitive(
            "get_calvin_language_subgoal",
            agent_context=agent_context or {},
        )
        if not result.ok or not isinstance(result.output, dict):
            return None
        return _extract_public_language_subgoal(result.output)

    def _suggested_policy_call_for_observation(self, *, observation_evidence_handle: str) -> JsonDict | None:
        if self._benchmark_id() != "calvin":
            return None
        if self._select_policy_primitive("language_skill") is None:
            return None
        subgoal = self._latest_public_language_subgoal()
        if not subgoal:
            return None
        return {
            "interface": "policy.invoke",
            "policy": "language_skill",
            "inputs": {"subgoal": subgoal, "horizon": CALVIN_PUBLIC_LANGUAGE_SKILL_HORIZON},
            "evidence_handles": [observation_evidence_handle],
            "public_rationale": (
                "CALVIN exposes a public language-conditioned policy skill; the latest observation "
                "already provides the public language_subgoal and scene evidence needed for the first policy attempt."
            ),
        }

    def _suggested_policy_call_for_affordance(
        self,
        affordances: list[JsonDict],
        *,
        affordance_evidence_handle: str,
    ) -> JsonDict | None:
        if self._benchmark_id() != "calvin":
            return None
        has_policy_skill = any(
            item.get("available") is True and item.get("type") == "policy_skill"
            for item in affordances
            if isinstance(item, dict)
        )
        if not has_policy_skill:
            return None
        subgoal = self._latest_public_language_subgoal()
        if not subgoal:
            return None
        evidence_handles = [affordance_evidence_handle]
        latest_observation_evidence = self._latest_evidence_handle(kind="scene_observation")
        if latest_observation_evidence is not None:
            evidence_handles.append(latest_observation_evidence)
        return {
            "interface": "policy.invoke",
            "policy": "language_skill",
            "inputs": {"subgoal": subgoal, "horizon": CALVIN_PUBLIC_LANGUAGE_SKILL_HORIZON},
            "evidence_handles": _dedupe_strings(evidence_handles),
            "public_rationale": (
                "CALVIN exposes the language-conditioned policy as a public policy_skill; "
                "entity.enumerate may be empty because this benchmark executes from observation and language evidence."
            ),
        }

    def _latest_evidence_handle(self, *, kind: str) -> str | None:
        for handle, payload in reversed(list(self._handles.items())):
            if handle.startswith("ev:") and payload.get("kind") == kind:
                return handle
        return None

    def _latest_public_language_subgoal(self) -> str | None:
        for payload in reversed(list(self._handles.values())):
            observation = payload.get("observation") if isinstance(payload, dict) else None
            if not isinstance(observation, dict):
                continue
            data = observation.get("data")
            if not isinstance(data, dict):
                continue
            for key in ("language_subgoal", "subgoal", "instruction", "language"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if self._task is not None and self._benchmark_id() == "calvin":
            instruction = str(self._task.instruction or "").strip()
            return instruction or None
        return None

    def _call_evidence_record(
        self,
        key: str | None = None,
        value: JsonDict | None = None,
        source_handles: list[str] | str | None = None,
        agent_context: JsonDict | None = None,
        **kwargs: Any,
    ) -> PrimitiveResult:
        source_handles = _coerce_handle_list(
            source_handles,
            kwargs.pop("source_handle", None),
            kwargs.pop("handles", None),
            kwargs.pop("handle", None),
            kwargs.pop("observation_handle", None),
            kwargs.pop("entity_handle", None),
            kwargs.pop("evidence_handles", None),
            kwargs.pop("evidence_handle", None),
        )
        if key is None:
            key = str(kwargs.pop("name", None) or kwargs.pop("label", None) or "agent_evidence")
        key = str(key)
        if value is None:
            for alias in ("value", "evidence", "payload", "record", "rationale", "reasoning", "note"):
                if alias in kwargs:
                    raw_value = kwargs.pop(alias)
                    value = raw_value if isinstance(raw_value, dict) else {alias: raw_value}
                    break
        if value is None:
            value = dict(kwargs)
            kwargs = {}
        if not isinstance(value, dict):
            return self._error("evidence.record", "invalid_evidence", {"key": key})
        missing = [handle for handle in source_handles or [] if handle not in self._handles]
        if missing:
            return self._error("evidence.record", "handle_not_found", {"missing": missing})
        payload = {
            "kind": "agent_evidence",
            "key": key,
            "value": _sanitize_agent_payload(value),
            "source_handles": list(source_handles or []),
            "agent_context": agent_context or {},
        }
        native_result = self._call_first_available(
            self._profile_candidates(
                "evidence",
                ("record_w4_evidence", "record_maniskill_evidence", "record_robocasa_evidence", "record_spatial_evidence"),
            ),
            key=key,
            value=payload,
            agent_context=agent_context,
        )
        if native_result is not None and native_result.ok:
            payload["native_artifacts"] = list(native_result.artifacts)
        handle = self._new_handle("ev", payload)
        return PrimitiveResult(
            name="evidence.record",
            ok=True,
            output={"evidence_handle": handle, "source_handles": list(source_handles or [])},
            artifacts=[handle],
        )

    def _call_action_prepare(
        self,
        action: JsonDict,
        evidence_handles: list[str],
        agent_context: JsonDict | None = None,
        public_rationale: str | None = None,
    ) -> PrimitiveResult:
        missing = self._missing_evidence(evidence_handles)
        if missing:
            return self._error("action.prepare", "evidence_required", {"missing": missing})
        if not isinstance(action, dict):
            return self._error("action.prepare", "unsupported_action", {"action": repr(action)})
        compiled = self._compile_action(action=action, evidence_handles=evidence_handles, agent_context=agent_context)
        if compiled is None:
            return self._error("action.prepare", "unsupported_action", {"action": action, "benchmark_id": self._benchmark_id()})
        if public_rationale:
            compiled["public_rationale"] = str(public_rationale)
        action_handle = self._new_action_handle(compiled)
        return PrimitiveResult(
            name="action.prepare",
            ok=True,
            output={"action_handle": action_handle, "compiled_action": _public_compiled_action(compiled)},
            artifacts=[action_handle],
        )

    def _call_action_execute(
        self,
        action_handle: str,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        compiled = self._actions.get(action_handle)
        if compiled is None:
            return self._error("action.execute", "action_handle_unknown", {"action_handle": action_handle})
        evidence_handles = list(compiled.get("evidence_handles", []))
        missing = self._missing_evidence(evidence_handles)
        if missing:
            return self._error("action.execute", "evidence_required", {"missing": missing, "action_handle": action_handle})
        native_results: list[JsonDict] = []
        raw_results: list[PrimitiveResult] = []
        for step in compiled.get("native_steps", []):
            arguments = _resolve_step_arguments(deepcopy(step.get("arguments", {})), raw_results)
            if step["primitive"] == "__universal_robowits_feedback_place__":
                result = self._execute_robowits_feedback_place(
                    arguments,
                    previous_results=raw_results,
                )
            elif step["primitive"] == "__universal_robotwin2_pick_place__":
                result = self._execute_robotwin2_pick_place(arguments)
            elif step["primitive"] == "__universal_capx_pick_place__":
                result = self._execute_capx_pick_place(arguments)
            else:
                result = self.backend.call_primitive(step["primitive"], **arguments)
            raw_results.append(result)
            native_results.append(_public_native_result(result))
            if not result.ok:
                return self._error(
                    "action.execute",
                    "backend_primitive_failed",
                    {"action_handle": action_handle, "step": step["primitive"], "native_result": _public_native_result(result)},
                )
        execution_handle = self._new_handle(
            "exec",
            {
                "kind": "execution",
                "action_handle": action_handle,
                "native_results": native_results,
                "agent_context": agent_context or {},
            },
        )
        return PrimitiveResult(
            name="action.execute",
            ok=True,
            output={
                "execution_handle": execution_handle,
                **({"native_api_result": _sanitize_agent_payload(raw_results[-1].output["native_api_result"])}
                   if raw_results and "native_api_result" in raw_results[-1].output else {}),
                "public_result": {
                    "action_handle": action_handle,
                    "executed_step_count": len(native_results),
                    "native_details_hidden": True,
                },
            },
            artifacts=[execution_handle],
        )

    def _execute_capx_pick_place(self, arguments: JsonDict) -> PrimitiveResult:
        """Execute a caller-grounded CaP-X stack without exposing native APIs."""

        source = _clean_name(arguments.get("source"))
        target = _clean_name(arguments.get("target"))
        if not source or not target or _norm(source) == _norm(target):
            return PrimitiveResult(
                name="capx_semantic_pick_place",
                ok=False,
                output={"executed": False},
                error="capx_pick_place_requires_distinct_source_and_target",
            )

        enumerated = self.backend.call_primitive("enumerate_capx_objects")
        objects = (
            enumerated.output.get("objects")
            if isinstance(enumerated.output, dict)
            else None
        )
        if not enumerated.ok or not isinstance(objects, list):
            return PrimitiveResult(
                name="capx_semantic_pick_place",
                ok=False,
                output={"executed": False, "phase": "refresh_public_scene"},
                error=enumerated.error or "capx_public_objects_unavailable",
            )

        def select_object(name: str) -> JsonDict | None:
            normalized = _norm(name)
            canonical_alias = {
                "cubea": "primary",
                "red_cube": "primary",
                "source": "primary",
                "cubeb": "secondary",
                "green_cube": "secondary",
                "target": "secondary",
            }.get(normalized, normalized)
            preferred: JsonDict | None = None
            for item in objects:
                if not isinstance(item, dict):
                    continue
                aliases = {
                    _norm(item.get("entity_id")),
                    _norm(item.get("label")),
                    _norm(item.get("object_name")),
                    _norm(item.get("caller_selection")),
                }
                if canonical_alias in aliases:
                    return item
                if normalized in aliases:
                    preferred = item
            return preferred

        source_object = select_object(source)
        target_object = select_object(target)
        source_xyz = _pose_xyz(source_object)
        target_xyz = _pose_xyz(target_object)
        if source_xyz is None or target_xyz is None:
            return PrimitiveResult(
                name="capx_semantic_pick_place",
                ok=False,
                output={"executed": False, "phase": "ground_public_entities"},
                error="capx_selected_entity_pose_unavailable",
            )

        def extent_z(item: JsonDict | None) -> float:
            extent = item.get("extent") if isinstance(item, dict) else None
            values = _pose_xyz(extent)
            return max(0.005, values[2]) if values is not None else 0.04

        approach_clearance = max(
            0.08,
            min(
                0.25,
                _coerce_float(arguments.get("approach_clearance"), default=0.14),
            ),
        )
        lift_clearance = max(
            approach_clearance,
            min(
                0.35,
                _coerce_float(arguments.get("lift_clearance"), default=0.28),
            ),
        )
        stack_margin = max(
            0.0,
            min(
                0.02,
                _coerce_float(arguments.get("stack_margin"), default=0.005),
            ),
        )
        visual_evidence_refs = list(arguments.get("visual_evidence_refs") or [])
        quaternion = _coerce_pose_list(arguments.get("quaternion_wxyz")) or [
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        if len(quaternion) != 4:
            return PrimitiveResult(
                name="capx_semantic_pick_place",
                ok=False,
                output={"executed": False},
                error="capx_pick_place_quaternion_must_have_four_values",
            )
        lift_xyz = [source_xyz[0], source_xyz[1], source_xyz[2] + lift_clearance]
        place_xyz = [
            target_xyz[0],
            target_xyz[1],
            target_xyz[2]
            + 0.5 * (extent_z(source_object) + extent_z(target_object))
            + stack_margin,
        ]
        stages: list[tuple[str, str, JsonDict]] = [
            ("open", "capx_open_gripper", {}),
            (
                "approach_and_grasp_pose",
                "capx_goto_pose",
                {
                    "position": source_xyz,
                    "quaternion_wxyz": quaternion,
                    "z_approach": approach_clearance,
                    "visual_evidence_refs": visual_evidence_refs,
                },
            ),
            ("grasp", "capx_close_gripper", {}),
            ("grasp_settle", "capx_close_gripper", {}),
            (
                "lift",
                "capx_goto_pose",
                {
                    "position": lift_xyz,
                    "quaternion_wxyz": quaternion,
                    "z_approach": 0.0,
                    "visual_evidence_refs": visual_evidence_refs,
                },
            ),
            (
                "stack",
                "capx_goto_pose",
                {
                    "position": place_xyz,
                    "quaternion_wxyz": quaternion,
                    "z_approach": approach_clearance,
                    "visual_evidence_refs": visual_evidence_refs,
                },
            ),
            ("release", "capx_open_gripper", {}),
            ("release_settle_1", "capx_open_gripper", {}),
            ("release_settle_2", "capx_open_gripper", {}),
            ("release_settle_3", "capx_open_gripper", {}),
            (
                "retract",
                "capx_goto_pose",
                {
                    "position": [
                        place_xyz[0],
                        place_xyz[1],
                        place_xyz[2] + approach_clearance,
                    ],
                    "quaternion_wxyz": quaternion,
                    "z_approach": 0.0,
                    "visual_evidence_refs": visual_evidence_refs,
                },
            ),
        ]
        completed: list[str] = []
        for phase, primitive, kwargs in stages:
            result = self.backend.call_primitive(
                primitive,
                **kwargs,
                agent_context={
                    "semantic_action": "pick_place",
                    "phase": phase,
                    "uses_private_success": False,
                },
            )
            if not result.ok:
                return PrimitiveResult(
                    name="capx_semantic_pick_place",
                    ok=False,
                    output={
                        "executed": False,
                        "phase": phase,
                        "completed_stage_count": len(completed),
                    },
                    error=result.error or "capx_pick_place_stage_failed",
                )
            completed.append(phase)
        return PrimitiveResult(
            name="capx_semantic_pick_place",
            ok=True,
            output={
                "executed": True,
                "source": source,
                "target": target,
                "completed_stage_count": len(completed),
                "uses_private_success": False,
            },
        )

    def _execute_robotwin2_pick_place(self, arguments: JsonDict) -> PrimitiveResult:
        """Execute a generic RoboTwin2 pick/place from fresh public RGB-D measurements."""

        source = _clean_name(arguments.get("source"))
        target = _clean_name(arguments.get("target"))
        arm = _clean_name(arguments.get("arm") or "right")
        camera_name = _clean_name(arguments.get("camera_name") or "head_camera")
        if not source or not target:
            return PrimitiveResult(
                name="robotwin2_semantic_pick_place",
                ok=False,
                output={"executed": False},
                error="robotwin2_pick_place_requires_source_and_target",
            )
        if arm not in {"left", "right"}:
            return PrimitiveResult(
                name="robotwin2_semantic_pick_place",
                ok=False,
                output={"executed": False, "arm": arm},
                error="robotwin2_arm_must_be_left_or_right",
            )

        native_names = {card.name for card in self.backend.list_primitives()}
        if "execute_robotwin2_pick_place" in native_names:
            native_handles: list[str] = []
            for actor_label, phase in ((source, "source"), (target, "target")):
                measured = self.backend.call_primitive(
                    "measure_robotwin2_actor_visual",
                    actor_label=actor_label,
                    camera_name=camera_name,
                    prompt=(
                        f"Measure caller-selected {actor_label} for semantic "
                        f"pick_place {phase} grounding."
                    ),
                    agent_context={
                        "semantic_action": "pick_place",
                        "phase": f"{phase}_grounding",
                        "uses_private_success": False,
                    },
                )
                handles = [
                    str(handle)
                    for handle in (measured.output.get("evidence_handles") or [])
                    if isinstance(handle, str) and handle.startswith("robotwin2:visual:")
                ]
                if not measured.ok or not handles:
                    return PrimitiveResult(
                        name="robotwin2_semantic_pick_place",
                        ok=False,
                        output={"executed": False, "phase": f"{phase}_grounding"},
                        error=(
                            measured.error
                            or "robotwin2_visual_measurement_did_not_issue_evidence"
                        ),
                    )
                native_handles.extend(handles)
            return self.backend.call_primitive(
                "execute_robotwin2_pick_place",
                source_actor_label=source,
                target_actor_label=target,
                arm_tag=arm,
                evidence_handles=native_handles,
                pre_grasp_distance=arguments.get("pre_grasp_distance", 0.1),
                lift_distance=arguments.get("lift_distance", 0.08),
                pre_place_distance=arguments.get("pre_place_distance", 0.05),
                agent_context={
                    "semantic_action": "pick_place",
                    "uses_private_success": False,
                },
            )

        quaternion = arguments.get("quaternion_wxyz") or [0.5, -0.5, 0.5, 0.5]
        target_approach_tolerance = max(
            0.01,
            min(
                0.08,
                _coerce_float(arguments.get("target_approach_tolerance"), default=0.04),
            ),
        )
        stages: list[tuple[str, str, JsonDict]] = [
            ("open", source, {"operation": "open_gripper", "pos": 1.0}),
            (
                "source_approach",
                source,
                {
                    "operation": "move_to_pose",
                    "measurement_offset": [0.04, 0.0, 0.30],
                },
            ),
            (
                "source_descend",
                source,
                {
                    "operation": "move_to_pose",
                    "measurement_offset": [0.04, 0.0, 0.20],
                },
            ),
            ("grasp", source, {"operation": "close_gripper", "pos": 0.0}),
            (
                "lift",
                source,
                {"operation": "move_by_displacement", "delta": [0.0, 0.0, 0.10], "move_axis": "world"},
            ),
            (
                "target_approach",
                target,
                {
                    "operation": "move_to_pose",
                    "measurement_offset": [0.055, 0.007, 0.32],
                    "route": {
                        "strategy": "vertical_clearance",
                        "clearance": 0.12,
                        "max_upward_margin": 0.04,
                        "recovery_xy_step": 0.03,
                    },
                },
            ),
            (
                "target_descend",
                target,
                {
                    "operation": "move_to_pose",
                    "measurement_offset": [0.055, 0.007, 0.22],
                },
            ),
            ("release", target, {"operation": "open_gripper", "pos": 1.0}),
        ]
        summaries: list[JsonDict] = []
        for phase, actor_label, action in stages:
            measured = self.backend.call_primitive(
                "measure_robotwin2_actor_visual",
                actor_label=actor_label,
                camera_name=camera_name,
                prompt=f"Measure caller-selected {actor_label} for semantic pick_place phase {phase}.",
                agent_context={
                    "semantic_action": "pick_place",
                    "phase": phase,
                    "uses_private_success": False,
                },
            )
            native_handles = [
                str(handle)
                for handle in (measured.output.get("evidence_handles") or [])
                if isinstance(handle, str) and handle.startswith("robotwin2:visual:")
            ]
            if not measured.ok or not native_handles:
                return PrimitiveResult(
                    name="robotwin2_semantic_pick_place",
                    ok=False,
                    output={"executed": False, "phase": phase, "completed_stages": summaries},
                    error=measured.error or "robotwin2_visual_measurement_did_not_issue_evidence",
                )
            native_action = deepcopy(action)
            offset = native_action.pop("measurement_offset", None)
            resolved_target_xyz = None
            if offset is not None:
                native_action["target_pose"] = {
                    "evidence_handle": native_handles[0],
                    "field": "centroid_world",
                    "offset": offset,
                    "quaternion_wxyz": quaternion,
                }
                centroid = _pose_xyz(measured.output.get("centroid_world"))
                offset_xyz = _pose_xyz(offset)
                if centroid is not None and offset_xyz is not None:
                    resolved_target_xyz = [
                        centroid[index] + offset_xyz[index] for index in range(3)
                    ]
            executed = self.backend.call_primitive(
                "execute_robotwin2_actions",
                **{arm: native_action},
                evidence_handles=native_handles,
                agent_context={
                    "semantic_action": "pick_place",
                    "phase": phase,
                    "uses_private_success": False,
                },
            )
            accepted_near_target = False
            near_target_distance = None
            if not executed.ok and phase == "target_approach" and resolved_target_xyz is not None:
                arm_pose = self.backend.call_primitive(
                    "inspect_robotwin2_arm_pose",
                    arm_tag=arm,
                    agent_context={
                        "semantic_action": "pick_place",
                        "phase": "target_approach_recovery_check",
                        "uses_private_success": False,
                    },
                )
                current_xyz = _pose_xyz(arm_pose.output.get("pose_world")) if arm_pose.ok else None
                if current_xyz is not None:
                    near_target_distance = math.dist(current_xyz, resolved_target_xyz)
                    accepted_near_target = near_target_distance <= target_approach_tolerance
            summaries.append(
                {
                    "phase": phase,
                    "ok": executed.ok or accepted_near_target,
                    "error": None if accepted_near_target else executed.error,
                    "backend_move_reported_failure": bool(accepted_near_target),
                    "near_target_distance": near_target_distance,
                }
            )
            if accepted_near_target:
                continue
            if not executed.ok:
                return PrimitiveResult(
                    name="robotwin2_semantic_pick_place",
                    ok=False,
                    output={"executed": False, "phase": phase, "completed_stages": summaries},
                    error=executed.error or "robotwin2_pick_place_stage_failed",
                )
        return PrimitiveResult(
            name="robotwin2_semantic_pick_place",
            ok=True,
            output={
                "executed": True,
                "source": source,
                "target": target,
                "arm": arm,
                "completed_stage_count": len(summaries),
                "uses_private_success": False,
            },
        )

    def _execute_robowits_feedback_place(
        self,
        arguments: JsonDict,
        *,
        previous_results: list[PrimitiveResult],
    ) -> PrimitiveResult:
        """Align a held RoboWits object from public state before release.

        The semantic action supplies the desired object-center position.  A
        held cube is not rigidly coincident with the commanded end-effector
        pose, so a fixed place waypoint is insufficient.  This controller
        measures only public object/EE state, applies bounded corrections, and
        never reads reward, task success, or verifier state.
        """

        object_name = str(arguments.get("object_name") or "").strip()
        desired = _pose_xyz(arguments.get("target_position"))
        if not object_name or desired is None:
            return PrimitiveResult(
                name="robowits_feedback_place",
                ok=False,
                output={"executed": False},
                error="invalid_feedback_place_arguments",
            )
        requested_arm = _clean_name(arguments.get("arm") or "auto") or "auto"
        prior_arm = None
        if previous_results:
            prior_arm = _clean_name(previous_results[-1].output.get("arm"))
        arm = requested_arm if requested_arm in {"left", "right"} else prior_arm
        if arm not in {"left", "right"}:
            arm = "left" if float(desired[1]) >= 0.0 else "right"
        axis_angle = arguments.get("axis_angle")
        closed_width = max(
            0.0,
            min(0.1, _coerce_float(arguments.get("closed_gripper_width"), default=0.012)),
        )
        open_width = max(
            0.0,
            min(0.1, _coerce_float(arguments.get("open_gripper_width"), default=0.085)),
        )
        repeat_steps = max(8, min(16, _coerce_int(arguments.get("repeat_steps")) or 12))
        max_translation = max(
            0.005,
            min(0.3, _coerce_float(arguments.get("max_translation"), default=0.3)),
        )
        lift_height = max(
            0.08,
            min(0.23, _coerce_float(arguments.get("lift_height"), default=0.18)),
        )
        context = {
            "semantic_phase": "feedback_place",
            "source_entity": object_name,
            "uses_private_success": False,
        }
        controls: list[JsonDict] = []

        def observe(phase: str) -> tuple[PrimitiveResult, list[float] | None, list[float] | None, list[str]]:
            result = self.backend.call_primitive(
                "observe_robowits_state",
                names=[object_name],
                query=phase,
                context={**context, "feedback_phase": phase},
            )
            output = result.output or {}
            object_record = (
                ((output.get("object_state") or {}).get("object_poses") or {}).get(object_name)
                if isinstance(output, dict)
                else None
            )
            object_center = (
                _pose_xyz(object_record.get("center") or object_record.get("pos"))
                if isinstance(object_record, dict)
                else None
            )
            ee_values = (output.get("ee_state") or {}).get("ee_state_world") if isinstance(output, dict) else None
            ee_position = None
            if isinstance(ee_values, list) and len(ee_values) >= 6:
                start = 3 if arm == "left" else 0
                ee_position = _pose_xyz(ee_values[start : start + 3])
            handles = [
                str(handle)
                for handle in (output.get("evidence_handles") or [])
                if isinstance(handle, str) and handle.startswith("robowits:visual:")
            ]
            return result, object_center, ee_position, handles

        def control(
            phase: str,
            position: list[float],
            *,
            gripper_width: float,
            handles: list[str],
            phase_repeat_steps: int,
            hold_steps: int,
        ) -> PrimitiveResult:
            result = self.backend.call_primitive(
                "execute_robowits_ee_control",
                target_position=[float(value) for value in position[:3]],
                arm=arm,
                axis_angle=axis_angle,
                gripper_width=gripper_width,
                repeat_steps=phase_repeat_steps,
                hold_steps=hold_steps,
                max_translation=max_translation,
                observe_names=[object_name],
                evidence_handles=handles,
                context={**context, "feedback_phase": phase},
            )
            controls.append({"phase": phase, "ok": result.ok, "error": result.error})
            return result

        state, object_center, ee_position, handles = observe("alignment_0")
        if not state.ok or object_center is None or ee_position is None or not handles:
            return PrimitiveResult(
                name="robowits_feedback_place",
                ok=False,
                output={"executed": False, "public_state_available": False},
                error="robowits_public_feedback_state_unavailable",
            )

        tolerance = 0.005
        release_tolerance = 0.02
        error_norm = float("inf")
        for iteration in range(3):
            error = [float(desired[index]) - float(object_center[index]) for index in range(3)]
            error_norm = math.sqrt(sum(value * value for value in error))
            if error_norm <= tolerance:
                break
            scale = min(1.0, max_translation / error_norm) if error_norm > 0.0 else 0.0
            corrected_ee = [
                float(ee_position[index]) + error[index] * scale for index in range(3)
            ]
            aligned = control(
                f"align_{iteration}",
                corrected_ee,
                gripper_width=closed_width,
                handles=handles,
                phase_repeat_steps=repeat_steps,
                hold_steps=4,
            )
            if not aligned.ok:
                return PrimitiveResult(
                    name="robowits_feedback_place",
                    ok=False,
                    output={"executed": False, "controls": controls},
                    error=aligned.error or "robowits_feedback_alignment_failed",
                )
            state, object_center, ee_position, handles = observe(f"alignment_{iteration + 1}")
            if not state.ok or object_center is None or ee_position is None or not handles:
                return PrimitiveResult(
                    name="robowits_feedback_place",
                    ok=False,
                    output={"executed": False, "controls": controls},
                    error="robowits_public_feedback_state_unavailable",
                )

        error = [float(desired[index]) - float(object_center[index]) for index in range(3)]
        error_norm = math.sqrt(sum(value * value for value in error))
        if error_norm > release_tolerance:
            return PrimitiveResult(
                name="robowits_feedback_place",
                ok=False,
                output={"executed": False, "pre_release_error_norm": error_norm, "controls": controls},
                error="robowits_feedback_not_aligned",
            )
        released = control(
            "release",
            ee_position,
            gripper_width=open_width,
            handles=handles,
            phase_repeat_steps=repeat_steps,
            hold_steps=8,
        )
        if not released.ok:
            return PrimitiveResult(
                name="robowits_feedback_place",
                ok=False,
                output={"executed": False, "controls": controls},
                error=released.error or "robowits_feedback_release_failed",
            )
        post_state, post_center, post_ee, post_handles = observe("post_release")
        if not post_state.ok or post_center is None or post_ee is None or not post_handles:
            return PrimitiveResult(
                name="robowits_feedback_place",
                ok=False,
                output={"executed": False, "controls": controls},
                error="robowits_public_feedback_state_unavailable",
            )
        post_error = [float(desired[index]) - float(post_center[index]) for index in range(3)]
        post_error_norm = math.sqrt(sum(value * value for value in post_error))
        retreated = control(
            "retreat",
            [float(desired[0]), float(desired[1]), float(desired[2]) + lift_height],
            gripper_width=open_width,
            handles=post_handles,
            phase_repeat_steps=repeat_steps,
            hold_steps=0,
        )
        ok = bool(retreated.ok and post_error_norm <= release_tolerance)
        return PrimitiveResult(
            name="robowits_feedback_place",
            ok=ok,
            output={
                "executed": ok,
                "arm": arm,
                "pre_release_error_norm": error_norm,
                "post_release_error_norm": post_error_norm,
                "controls": controls,
                "uses_private_success": False,
            },
            error=None if ok else (retreated.error or "robowits_feedback_post_release_drift"),
        )

    def _call_policy_invoke(
        self,
        policy: str,
        inputs: JsonDict | None = None,
        evidence_handles: list[str] | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        evidence_handles = evidence_handles or []
        native_name = self._select_policy_primitive(policy)
        if native_name is None:
            return self._error("policy.invoke", "unsupported_policy", {"policy": policy, "benchmark_id": self._benchmark_id()})
        missing = self._missing_evidence(evidence_handles)
        if missing and not self._allows_public_context_only_policy_invoke(
            native_name=native_name,
            inputs=inputs or {},
            evidence_handles=evidence_handles,
        ):
            return self._error("policy.invoke", "evidence_required", {"missing": missing})
        native_args = self._policy_arguments_for_native_primitive(
            native_name,
            policy=policy,
            inputs=inputs or {},
            evidence_handles=evidence_handles,
            agent_context=agent_context,
        )
        result = self.backend.call_primitive(native_name, **native_args)
        public_result = _public_native_result(result)
        if not result.ok:
            self._last_errors.append(
                _sanitize_agent_payload(
                    {
                        "interface": "policy.invoke",
                        "code": result.error or "backend_primitive_failed",
                        "native_result": public_result,
                    }
                )
            )
        handle = self._new_handle(
            "policy",
            {"kind": "policy", "policy": policy, "native_result": public_result, "evidence_handles": evidence_handles},
        )
        return PrimitiveResult(
            name="policy.invoke",
            ok=result.ok,
            output={"policy_handle": handle, "public_result": public_result},
            artifacts=[handle],
            error=result.error,
        )

    def _allows_public_context_only_policy_invoke(
        self,
        *,
        native_name: str,
        inputs: JsonDict,
        evidence_handles: list[str],
    ) -> bool:
        if evidence_handles:
            return False
        if self._benchmark_id() != "calvin" or native_name != "execute_calvin_language_skill":
            return False
        return _extract_public_language_subgoal(inputs) is not None

    def _call_failure_diagnose(
        self,
        handle: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        last_error = deepcopy(self._last_errors[-1]) if self._last_errors else None
        public_failure = _public_failure_classification(last_error)
        related = self._handles.get(handle or "") or self._actions.get(handle or "")
        diagnosis = {
            "related_handle": _public_failure_related_handle(related) if related else None,
            "failure_category": public_failure["category"],
            "public_blocker": public_failure["blocker"],
            "suggested_next_step": public_failure["suggested_next_step"],
            "public_repair_options": public_failure["public_repair_options"] or [
                "refresh scene.observe",
                "rerun entity.locate for the target handle",
                "record missing evidence before preparing an action",
                "prepare a smaller semantic action",
            ],
            "agent_context": agent_context or {},
        }
        return PrimitiveResult(name="failure.diagnose", ok=True, output={"diagnosis": diagnosis})

    def _call_progress_check(
        self,
        query: str | None = None,
        agent_context: JsonDict | None = None,
    ) -> PrimitiveResult:
        trace = self.get_trace()
        last_error = deepcopy(self._last_errors[-1]) if self._last_errors else None
        public_failure = _public_failure_classification(last_error)
        progress = {
            "benchmark_id": self._benchmark_id(),
            "query": query,
            "event_count": len(trace.events),
            "evidence_handle_count": sum(1 for item in self._handles.values() if item.get("kind", "").startswith(("agent_evidence", "entity", "scene", "geometry", "affordance"))),
            "prepared_action_count": len(self._actions),
            "last_failure_category": public_failure["category"],
            "last_public_blocker": public_failure["blocker"],
            "budget": self._budget_snapshot(),
            "official_scoring_hidden": True,
            "agent_context": agent_context or {},
        }
        return PrimitiveResult(name="progress.check", ok=True, output={"progress": _sanitize_agent_payload(progress)})

    def _compile_action(
        self,
        *,
        action: JsonDict,
        evidence_handles: list[str],
        agent_context: JsonDict | None,
    ) -> JsonDict | None:
        action = _normalize_action_payload(action)
        action_agent_context = action.get("agent_context") if isinstance(action.get("agent_context"), dict) else {}
        effective_agent_context = {**action_agent_context, **(agent_context or {})}
        benchmark_id = self._benchmark_id()
        verb = _norm(action.get("type") or action.get("verb") or action.get("action") or "")
        obj = self._entity_name_from_action_value(
            action.get("source")
            or action.get("source_entity")
            or action.get("object")
            or action.get("entity")
            or action.get("target_object")
        )
        target = self._entity_name_from_action_value(action.get("target") or action.get("destination") or action.get("fixture"))
        relation = _clean_name(action.get("relation") or "on")
        if action.get("native_primitive") and isinstance(action.get("native_primitive"), str):
            return None

        native_steps = self._compile_profile_driven_action(
            verb=verb,
            obj=obj,
            target=target,
            relation=relation,
            action=action,
            evidence_handles=evidence_handles,
            agent_context=effective_agent_context,
        )

        if not native_steps and benchmark_id == "vimabench" and verb in {"pick_place", "place"} and obj and target:
            native_steps = [
                {"primitive": "pick_vima_object", "arguments": {"object_name": obj}},
                {"primitive": "place_vima_object", "arguments": {"object_name": obj, "target_name": target}},
            ]
        elif not native_steps and benchmark_id == "cliport" and verb in {"pick_place", "place"} and obj and target:
            native_steps = [
                {"primitive": "pick_cliport_object", "arguments": {"object_name": obj}},
                {"primitive": "place_cliport_object", "arguments": {"object_name": obj, "target_name": target}},
            ]
        elif not native_steps and benchmark_id == "vlabench":
            skill = _clean_name(action.get("skill") or action.get("policy") or verb)
            native_steps = [
                {
                    "primitive": "execute_vlabench_skill",
                    "arguments": {"skill_name": skill, "target_name": target or obj or "target", "horizon": int(action.get("horizon", 10))},
                }
            ]
        elif not native_steps and benchmark_id in {"robocasa", "robocasa365"} and target:
            prefix = "robocasa365" if benchmark_id == "robocasa365" else "robocasa"
            primitive = f"close_{prefix}_fixture" if verb in {"close", "press", "toggle", "turn_on"} else f"open_{prefix}_fixture"
            native_steps = [{"primitive": primitive, "arguments": {"fixture_name": target}}]
        elif not native_steps and benchmark_id == "capx":
            pose = action.get("pose") or self._pose_from_evidence(evidence_handles, obj)
            if pose is None:
                return None
            goto_arguments = self._capx_goto_pose_arguments(
                action=action,
                pose=pose,
                evidence_handles=evidence_handles,
                entity_id=obj,
                agent_context=effective_agent_context,
            )
            if goto_arguments is None:
                return None
            native_steps = [{"primitive": "capx_goto_pose", "arguments": goto_arguments}]
        elif not native_steps and benchmark_id == "rlbench":
            native_steps = [
                {"primitive": "move_rlbench_arm_to", "arguments": {"target_name": target or obj or "target_handle", "strategy": "rlbench_pose_action"}},
            ]
            if verb in {"reach_close", "close", "grasp", "pick", "pick_place"} or action.get("gripper") == "closed":
                native_steps.append({"primitive": "close_rlbench_gripper", "arguments": {}})
        elif not native_steps and benchmark_id == "calvin":
            state = _clean_name(action.get("state") or "on")
            if verb in {"turn_on", "toggle", "switch"}:
                native_steps = [{"primitive": "toggle_calvin_light", "arguments": {"state": state}}]
            else:
                subgoal = _clean_name(
                    action.get("subgoal")
                    or action.get("language_goal")
                    or action.get("instruction")
                    or action.get("language")
                    or "turn on the light"
                )
                native_steps = [
                    {
                        "primitive": "execute_calvin_language_skill",
                        "arguments": {"subgoal": subgoal, "horizon": int(action.get("horizon", 1))},
                    }
                ]

        if not native_steps:
            native_steps = self._compile_generic_by_available_primitives(verb=verb, obj=obj, target=target, relation=relation, action=action)
        if not native_steps:
            return None
        return {
            "benchmark_id": benchmark_id,
            "semantic_action": _sanitize_agent_payload(action),
            "evidence_handles": list(evidence_handles),
            "native_steps": native_steps,
            "native_details_disclosed_to_agent": False,
            "agent_context": effective_agent_context,
        }

    def _compile_profile_driven_action(
        self,
        *,
        verb: str,
        obj: str | None,
        target: str | None,
        relation: str | None,
        action: JsonDict,
        evidence_handles: list[str],
        agent_context: JsonDict | None,
    ) -> list[JsonDict]:
        benchmark_id = self._benchmark_id()
        names = self._available_native_names()
        profile = self._adapter_profile()
        available_profile_primitives = set(profile.prepare_primitives) | set(profile.execute_primitives) | set(profile.policy_primitives)
        if not names.intersection(available_profile_primitives):
            return []

        prompt = _clean_name(action.get("prompt") or action.get("instruction") or (self._task.instruction if self._task else ""))
        query = _clean_name(action.get("query") or prompt)
        pose = action.get("pose") or action.get("target_pose") or action.get("target_position") or self._pose_from_evidence(evidence_handles, target or obj)
        xyz = _pose_xyz(pose)
        evidence_kwargs = {
            "evidence_handles": list(evidence_handles),
            "evidence_ids": list(evidence_handles),
            "evidence_refs": list(evidence_handles),
        }

        def has(name: str) -> bool:
            return name in names

        def step(name: str, arguments: JsonDict | None = None) -> JsonDict:
            return {"primitive": name, "arguments": _drop_none(arguments or {})}

        if benchmark_id in {"openvla", "openpi", "lerobot", "octo"}:
            submit_name = f"submit_{benchmark_id}_policy_action_chunk"
            action_chunk = action.get("action_chunk") or action.get("actions") or action.get("control")
            if has(submit_name) and isinstance(action_chunk, list):
                return [
                    step(
                        submit_name,
                        {
                            "action_chunk": action_chunk,
                            "target": target or obj,
                            "agent_context": agent_context,
                        },
                    )
                ]

        if benchmark_id == "maniskill":
            source_pose = self._extract_pose_from_payload(
                self._resolve_entity(obj) or {}
            )
            target_pose = self._extract_pose_from_payload(
                self._resolve_entity(target) or {}
            )
            source_xyz = _pose_xyz(source_pose)
            target_xyz = _pose_xyz(
                action.get("target_position")
                or action.get("target_pose")
                or action.get("goal_xyz")
                or target_pose
            )
            native_evidence_ids = self._native_visual_evidence_ids_from_handles(
                evidence_handles,
                prefixes=(
                    "maniskill:visual:",
                    "maniskill:instance:",
                    "maniskill:regions:",
                ),
            )
            first_step_evidence = {"evidence_ids": native_evidence_ids}
            if (
                verb in {"pick_place", "place", "grasp_place"}
                and obj
                and source_xyz is not None
                and target_xyz is not None
                and has("move_maniskill_tcp_to")
                and has("set_maniskill_gripper")
            ):
                approach_height = max(
                    0.02, _coerce_float(action.get("approach_height"), default=0.08)
                )
                lift_height = max(
                    approach_height,
                    _coerce_float(action.get("lift_height"), default=0.12),
                )
                source_above = [
                    source_xyz[0],
                    source_xyz[1],
                    source_xyz[2] + approach_height,
                ]
                source_lifted = [
                    source_xyz[0],
                    source_xyz[1],
                    source_xyz[2] + lift_height,
                ]
                target_above = [
                    target_xyz[0],
                    target_xyz[1],
                    target_xyz[2] + lift_height,
                ]
                context = {"agent_context": agent_context}
                return [
                    step(
                        "set_maniskill_gripper",
                        {
                            "command": 1.0,
                            "repeat": 5,
                            **first_step_evidence,
                            **context,
                        },
                    ),
                    step(
                        "move_maniskill_tcp_to",
                        {"target_xyz": source_above, "gripper": 1.0, **context},
                    ),
                    step(
                        "move_maniskill_tcp_to",
                        {"target_xyz": source_xyz, "gripper": 1.0, **context},
                    ),
                    step(
                        "set_maniskill_gripper",
                        {"command": -1.0, "repeat": 10, **context},
                    ),
                    step(
                        "move_maniskill_tcp_to",
                        {"target_xyz": source_lifted, "gripper": -1.0, **context},
                    ),
                    step(
                        "move_maniskill_tcp_to",
                        {"target_xyz": target_above, "gripper": -1.0, **context},
                    ),
                    step(
                        "move_maniskill_tcp_to",
                        {"target_xyz": target_xyz, "gripper": -1.0, **context},
                    ),
                    step(
                        "set_maniskill_gripper",
                        {"command": -1.0, "repeat": 1, **context},
                    ),
                ]
            grounded_xyz = target_xyz if target is not None else source_xyz
            grounded_xyz = _pose_xyz(
                action.get("pose")
                or action.get("target_pose")
                or action.get("target_position")
            ) or grounded_xyz or xyz
            maniskill_evidence = {"evidence_ids": native_evidence_ids}
            if has("move_maniskill_tcp_to") and grounded_xyz is not None and verb in {"move", "reach", "goto", "go_to", "approach", "place", "pick_place"}:
                return [step("move_maniskill_tcp_to", {"target_xyz": grounded_xyz, "gripper": _maniskill_gripper_value(action), **maniskill_evidence, "agent_context": agent_context})]
            if has("move_maniskill_tcp_delta") and action.get("delta_xyz"):
                return [step("move_maniskill_tcp_delta", {"delta_xyz": action.get("delta_xyz"), "gripper": _maniskill_gripper_value(action), **maniskill_evidence, "agent_context": agent_context})]
            if has("set_maniskill_gripper") and verb in {"open", "close", "gripper", "grasp", "release"}:
                return [step("set_maniskill_gripper", {"command": _maniskill_gripper_value(action, closed_default=verb in {"close", "grasp"}), **maniskill_evidence, "agent_context": agent_context})]
            if has("apply_maniskill_action") and action.get("control") is not None:
                return [step("apply_maniskill_action", {"action": action["control"], "repeat": int(action.get("repeat", 1)), **maniskill_evidence, "agent_context": agent_context})]

        if benchmark_id == "vimabench":
            source_id = self._entity_int_attribute(obj, "segmentation_id", "segm_id", "instance_id")
            target_id = self._entity_int_attribute(target, "segmentation_id", "segm_id", "instance_id")
            source_id = _coerce_int(action.get("source_segmentation_id")) if action.get("source_segmentation_id") is not None else source_id
            target_id = _coerce_int(action.get("target_segmentation_id")) if action.get("target_segmentation_id") is not None else target_id
            visual_evidence_ids = _coerce_handle_list(action.get("evidence_ids"), action.get("evidence_id"))
            visual_evidence_ids.extend(
                self._native_visual_evidence_ids_from_handles(
                    evidence_handles,
                    prefixes=("vimabench:instance:",),
                )
            )
            visual_evidence_ids = _dedupe_strings(visual_evidence_ids)
            vima_evidence_ids = visual_evidence_ids or evidence_handles
            direct_action = action.get("vima_action") if isinstance(action.get("vima_action"), dict) else action
            if has("submit_vima_action") and all(key in direct_action for key in ("pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation")):
                return [step("submit_vima_action", {key: direct_action[key] for key in ("pose0_position", "pose0_rotation", "pose1_position", "pose1_rotation")} | {"agent_context": agent_context, "evidence_ids": vima_evidence_ids})]
            if has("build_vima_pick_place_action") and has("submit_vima_action") and source_id is not None and target_id is not None:
                return [
                    step("build_vima_pick_place_action", {"source_segmentation_id": source_id, "target_segmentation_id": target_id, "prompt": prompt, "query": query, "agent_context": agent_context, "evidence_ids": vima_evidence_ids}),
                    step(
                        "submit_vima_action",
                        {
                            "pose0_position": {"$from_previous": ["action", "pose0_position"]},
                            "pose0_rotation": {"$from_previous": ["action", "pose0_rotation"]},
                            "pose1_position": {"$from_previous": ["action", "pose1_position"]},
                            "pose1_rotation": {"$from_previous": ["action", "pose1_rotation"]},
                            "agent_context": agent_context,
                            "evidence_ids": vima_evidence_ids,
                        },
                    ),
                ]
            if has("build_vima_pick_place_action") and source_id is not None and target_id is not None:
                return [step("build_vima_pick_place_action", {"source_segmentation_id": source_id, "target_segmentation_id": target_id, "prompt": prompt, "query": query, "agent_context": agent_context, "evidence_ids": vima_evidence_ids})]

        if benchmark_id == "cliport":
            source_selector = action.get("source") or action.get("source_entity") or action.get("object") or action.get("entity")
            target_selector = action.get("target") or action.get("destination") or action.get("fixture")
            source_object_id = _coerce_int(action.get("source_object_id")) or self._entity_int_attribute(obj, "object_id", "id")
            target_object_id = _coerce_int(action.get("target_object_id")) or self._entity_int_attribute(target, "object_id", "id")
            evidence_object_ids = self._object_ids_from_evidence_handles(evidence_handles)
            if verb in {"pick", "grasp"} and source_object_id is None and target_object_id is not None:
                source_object_id, target_object_id = target_object_id, None
            if source_object_id is None and evidence_object_ids:
                source_object_id = evidence_object_ids[0]
            if target_object_id is None:
                target_object_id = next((item for item in evidence_object_ids if item != source_object_id), None)
            visual_evidence_ids = _coerce_handle_list(action.get("evidence_ids"), action.get("evidence_id"))
            visual_evidence_ids.extend(
                self._native_visual_evidence_ids_from_handles(
                    evidence_handles,
                    prefixes=("cliport:instance:", "cliport:instances:"),
                )
            )
            visual_evidence_ids = _dedupe_strings(visual_evidence_ids)
            pick_pose = action.get("pick_pose") or self._pose0_from_evidence_handles(
                evidence_handles,
                selector=source_selector,
                object_id=source_object_id,
            )
            composite_place_pose = self._cliport_composite_place_pose_from_evidence(
                evidence_handles,
                action=action,
                source_object_id=source_object_id,
                target_object_id=target_object_id,
            )
            place_pose = (
                action.get("place_pose")
                or composite_place_pose
                or self._pose0_from_evidence_handles(
                    evidence_handles,
                    selector=target_selector,
                    object_id=target_object_id,
                )
            )
            if has("submit_cliport_pick_place_action") and (
                pick_pose is not None
                or place_pose is not None
                or source_object_id is not None
                or target_object_id is not None
            ):
                return [
                    step(
                        "submit_cliport_pick_place_action",
                        {
                            "pick_pose": pick_pose,
                            "place_pose": place_pose,
                            "source_object_id": source_object_id,
                            "target_object_id": target_object_id,
                            "prompt": prompt,
                            "query": query,
                            "agent_context": agent_context,
                            "evidence_ids": visual_evidence_ids or evidence_handles,
                        },
                    )
                ]

        if benchmark_id == "vlabench":
            vlabench_candidate_evidence_handles = _coerce_handle_list(
                action.get("evidence_handles"),
                action.get("evidence_ids"),
                action.get("evidence_refs"),
                action.get("evidence_handle"),
            )
            vlabench_candidate_evidence_handles.extend(evidence_handles)
            vlabench_visual_evidence_ids: list[str] = []
            vlabench_visual_evidence_ids.extend(
                self._native_visual_evidence_ids_from_handles(
                    vlabench_candidate_evidence_handles,
                    prefixes=("vlabench:visual:",),
                )
            )
            vlabench_source_evidence_handles = self._source_evidence_leaf_handles(
                vlabench_candidate_evidence_handles,
                evidence_prefix=f"ev:{benchmark_id}:",
                preferred_kinds=("entity_location", "entity_inspection", "geometry_measurement", "affordance"),
            )
            vlabench_native_visual_handles = _dedupe_strings(vlabench_visual_evidence_ids)
            vlabench_execution_evidence_handles = _dedupe_strings(vlabench_native_visual_handles or vlabench_source_evidence_handles)
            if vlabench_execution_evidence_handles:
                evidence_kwargs = {
                    "evidence_handles": vlabench_execution_evidence_handles,
                    "evidence_ids": vlabench_execution_evidence_handles,
                    "evidence_refs": vlabench_execution_evidence_handles,
                }

            def vlabench_entity_pose(entity_id: str | None, fallback: Any = None) -> list[float] | None:
                resolved = self._resolve_entity(entity_id) if entity_id else None
                return (
                    _pose_xyz(fallback)
                    or self._pose_from_evidence(vlabench_candidate_evidence_handles, entity_id)
                    or self._extract_pose_from_payload(resolved or {}, min_length=3)
                )

            def vlabench_entity_segmentation_id(entity_id: str | None, explicit_keys: tuple[str, ...] = ()) -> int | None:
                if not entity_id:
                    return None
                for key in explicit_keys:
                    if key in action:
                        coerced = _coerce_first_int(action.get(key))
                        if coerced is not None:
                            return coerced
                resolved = self._resolve_entity(entity_id)
                if isinstance(resolved, dict):
                    coerced = self._int_attribute_from_payload(
                        resolved,
                        keys=("segmentation_id", "segm_id", "instance_id", "segmentation_ids"),
                        entity_name=entity_id,
                    )
                    if coerced is not None:
                        return coerced
                for payload in self._iter_handle_graph_payloads(vlabench_candidate_evidence_handles):
                    coerced = self._int_attribute_from_payload(
                        payload,
                        keys=("segmentation_id", "segm_id", "instance_id", "segmentation_ids"),
                        entity_name=entity_id,
                    )
                    if coerced is not None:
                        return coerced
                return None

            def with_vlabench_visual_grounding(
                base_steps: list[JsonDict],
                bindings: list[tuple[str | None, list[float] | None, tuple[str, ...]]],
            ) -> list[JsonDict]:
                if vlabench_native_visual_handles:
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = vlabench_native_visual_handles
                    return base_steps
                if not has("ground_vlabench_visual_target"):
                    fallback_handles = vlabench_execution_evidence_handles or list(evidence_handles)
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = fallback_handles
                    return base_steps
                grounding_steps: list[JsonDict] = []
                seen_entities: set[str] = set()
                for entity_name, world_position, explicit_keys in bindings:
                    if not entity_name:
                        continue
                    entity_key = _norm(entity_name)
                    if entity_key in seen_entities:
                        continue
                    segmentation_id = vlabench_entity_segmentation_id(entity_name, explicit_keys=explicit_keys)
                    if segmentation_id is None:
                        continue
                    seen_entities.add(entity_key)
                    grounding_steps.append(
                        step(
                            "ground_vlabench_visual_target",
                            {
                                "prompt": prompt or f"Ground VLABench entity {entity_name}.",
                                "query": query or entity_name,
                                "entity_name": entity_name,
                                "segmentation_id": segmentation_id,
                                "world_position": world_position,
                                "agent_context": agent_context,
                                "require_pcd": action.get("require_pcd", True),
                            },
                        )
                    )
                if not grounding_steps:
                    fallback_handles = [] if has("ground_vlabench_visual_target") else (vlabench_execution_evidence_handles or list(evidence_handles))
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = fallback_handles
                    return base_steps
                resolved_evidence = [
                    {"$from_step": index, "path": ["evidence_handle"]}
                    for index in range(len(grounding_steps))
                ]
                for base_step in base_steps:
                    base_step.setdefault("arguments", {})["evidence_handles"] = resolved_evidence
                return [*grounding_steps, *base_steps]

            if has("open_vlabench_gripper") and verb in {"open", "release"}:
                return [step("open_vlabench_gripper", {"agent_context": agent_context, "horizon": int(action.get("horizon", 10)), **evidence_kwargs})]
            if has("close_vlabench_gripper") and verb in {"close", "grasp"} and not obj:
                return [step("close_vlabench_gripper", {"agent_context": agent_context, "horizon": int(action.get("horizon", 10)), **evidence_kwargs})]
            if has("grasp_vlabench_entity") and verb in {"grasp", "pick"} and obj:
                target_position = xyz or vlabench_entity_pose(obj, pose)
                return with_vlabench_visual_grounding(
                    [
                        step(
                            "grasp_vlabench_entity",
                            {
                                "entity_name": obj,
                                "target_position": target_position,
                                "target_quat": action.get("target_quat"),
                                "target_euler": action.get("target_euler"),
                                "max_n_substep": int(action.get("max_n_substep", 2)),
                                "agent_context": agent_context,
                            },
                        )
                    ],
                    [(obj, target_position, ("source_segmentation_id", "object_segmentation_id", "segmentation_id"))],
                )
            if has("lift_vlabench_ee") and verb in {"lift", "raise"}:
                lift_entity = obj or target
                target_position = xyz or vlabench_entity_pose(lift_entity, pose)
                return with_vlabench_visual_grounding(
                    [
                        step(
                            "lift_vlabench_ee",
                            {
                                "lift_height": action.get("lift_height", action.get("height")),
                                "target_position": target_position,
                                "target_quat": action.get("target_quat"),
                                "target_euler": action.get("target_euler"),
                                "gripper_state": action.get("gripper_state"),
                                "agent_context": agent_context,
                            },
                        )
                    ],
                    [(lift_entity, target_position, ("source_segmentation_id", "object_segmentation_id", "segmentation_id"))],
                )
            if (
                verb == "pick_place"
                and obj
                and target
                and has("grasp_vlabench_entity")
                and has("lift_vlabench_ee")
                and has("place_vlabench_entity_in")
            ):
                source_position = vlabench_entity_pose(obj, pose)
                target_position = vlabench_entity_pose(target, pose)
                base_steps = [
                    step(
                        "grasp_vlabench_entity",
                        {
                            "entity_name": obj,
                            "target_position": source_position,
                            "target_quat": action.get("source_quat") or action.get("target_quat"),
                            "target_euler": action.get("source_euler") or action.get("target_euler"),
                            "max_n_substep": int(action.get("max_n_substep", 2)),
                            "agent_context": agent_context,
                        },
                    ),
                    step(
                        "lift_vlabench_ee",
                        {
                            "lift_height": action.get("lift_height", action.get("height", 0.3)),
                            "gripper_state": action.get("gripper_state", [0.0, 0.0]),
                            "agent_context": agent_context,
                        },
                    ),
                    step(
                        "place_vlabench_entity_in",
                        {
                            "entity_name": obj,
                            "target_name": target,
                            "container_name": target,
                            "target_position": target_position,
                            "target_quat": action.get("target_quat"),
                            "target_euler": action.get("target_euler"),
                            "use_native_place_point": True,
                            "agent_context": agent_context,
                        },
                    ),
                ]
                return with_vlabench_visual_grounding(
                    base_steps,
                    [
                        (obj, source_position, ("source_segmentation_id", "object_segmentation_id")),
                        (target, target_position, ("target_segmentation_id", "container_segmentation_id")),
                    ],
                )
            if has("place_vlabench_entity_in") and verb in {"place", "put", "pick_place"} and obj and target:
                target_position = xyz or vlabench_entity_pose(target, pose)
                return with_vlabench_visual_grounding(
                    [
                        step(
                            "place_vlabench_entity_in",
                            {
                                "entity_name": obj,
                                "target_name": target,
                                "container_name": target,
                                "target_position": target_position,
                                "target_quat": action.get("target_quat"),
                                "target_euler": action.get("target_euler"),
                                "use_native_place_point": True,
                                "agent_context": agent_context,
                            },
                        )
                    ],
                    [
                        (obj, vlabench_entity_pose(obj), ("source_segmentation_id", "object_segmentation_id")),
                        (target, target_position, ("target_segmentation_id", "container_segmentation_id", "segmentation_id")),
                    ],
                )
            if has("place_vlabench_entity_in") and verb in {"place", "put", "pick_place"} and target:
                target_position = xyz or vlabench_entity_pose(target, pose)
                return with_vlabench_visual_grounding(
                    [
                        step(
                            "place_vlabench_entity_in",
                            {
                                "target_name": target,
                                "container_name": target,
                                "target_position": target_position,
                                "target_quat": action.get("target_quat"),
                                "target_euler": action.get("target_euler"),
                                "use_native_place_point": True,
                                "agent_context": agent_context,
                            },
                        )
                    ],
                    [(target, target_position, ("target_segmentation_id", "container_segmentation_id", "segmentation_id"))],
                )
            if has("move_vlabench_ee_to") and (target or xyz is not None):
                target_name = target or obj
                target_position = xyz or vlabench_entity_pose(target_name, pose)
                return [
                    *with_vlabench_visual_grounding(
                        [
                            step(
                                "move_vlabench_ee_to",
                                {
                                    "target_name": target_name,
                                    "target_position": target_position,
                                    "offset": action.get("offset"),
                                    "target_site": action.get("target_site", "xpos"),
                                    "target_quat": action.get("target_quat"),
                                    "target_euler": action.get("target_euler"),
                                    "gripper_state": action.get("gripper_state"),
                                    "agent_context": agent_context,
                                    "horizon": int(action.get("horizon", 10)),
                                },
                            )
                        ],
                        [(target_name, target_position, ("target_segmentation_id", "object_segmentation_id", "segmentation_id"))],
                    )
                ]
            if has("settle_vlabench_scene") and verb in {"settle", "settle_scene"}:
                return [step("settle_vlabench_scene", {"agent_context": agent_context, "horizon": int(action.get("horizon", 10)), **evidence_kwargs})]
            if has("execute_vlabench_skill"):
                return [step("execute_vlabench_skill", {"skill_name": action.get("skill") or verb or "noop_step", "target_name": target or obj, "agent_context": agent_context, "horizon": int(action.get("horizon", 10)), **evidence_kwargs})]

        if benchmark_id in {"robocasa", "robocasa365"}:
            prefix = "robocasa365" if benchmark_id == "robocasa365" else "robocasa"
            native_visual_handles = _coerce_handle_list(
                action.get("evidence_handles"),
                action.get("evidence_ids"),
                action.get("evidence_refs"),
                action.get("evidence_handle"),
                action.get("native_visual_evidence_handles"),
                action.get("visual_evidence_handles"),
            )
            native_visual_handles.extend(
                self._native_visual_evidence_ids_from_handles(
                    evidence_handles,
                    prefixes=("robocasa:visual:", "robocasa365:visual:"),
                )
            )
            native_visual_handles = _dedupe_strings(native_visual_handles)

            def entity_pose(entity_id: str | None, fallback: Any = None) -> list[float] | None:
                resolved = self._resolve_entity(entity_id) if entity_id else None
                return (
                    _pose_xyz(fallback)
                    or self._pose_from_evidence(evidence_handles, entity_id)
                    or self._extract_pose_from_payload(resolved or {}, min_length=3)
                )

            button_name = _clean_name(action.get("button_name") or action.get("button"))
            button_position = _pose_xyz(action.get("button_position"))
            if button_position is None and (target or obj):
                inferred_button = self._robocasa_button_from_entity(target or obj, requested_button=button_name)
                if inferred_button is not None:
                    inferred_name, inferred_position = inferred_button
                    button_name = button_name or inferred_name
                    button_position = inferred_position

            def action_kwargs(keys: tuple[str, ...]) -> JsonDict:
                return {key: action[key] for key in keys if key in action}

            def with_visual_grounding(
                base_steps: list[JsonDict],
                bindings: list[tuple[str | None, list[float] | None]],
            ) -> list[JsonDict]:
                if native_visual_handles:
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = native_visual_handles
                    return base_steps
                ground_primitive = f"ground_{prefix}_visual_target"
                if not has(ground_primitive):
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = list(evidence_handles)
                    return base_steps
                grounding_steps: list[JsonDict] = []
                for entity_name, world_position in bindings:
                    if not entity_name or world_position is None:
                        continue
                    if any(
                        step_item.get("arguments", {}).get("entity_name") == entity_name
                        for step_item in grounding_steps
                    ):
                        continue
                    grounding_steps.append(
                        step(
                            ground_primitive,
                            {
                                "prompt": prompt,
                                "query": query,
                                "entity_name": entity_name,
                                "world_position": world_position,
                                "agent_context": agent_context,
                            },
                        )
                    )
                if not grounding_steps:
                    for base_step in base_steps:
                        base_step.setdefault("arguments", {})["evidence_handles"] = list(evidence_handles)
                    return base_steps
                resolved_evidence = [
                    {"$from_step": index, "path": ["evidence_handle"]}
                    for index in range(len(grounding_steps))
                ]
                for base_step in base_steps:
                    base_step.setdefault("arguments", {})["evidence_handles"] = resolved_evidence
                return [*grounding_steps, *base_steps]

            fixture = target or obj
            if fixture and verb in {"open", "close", "press", "toggle", "turn_on", "turn_off"}:
                primitive = (
                    f"press_{prefix}_fixture_button"
                    if verb in {"press", "toggle", "turn_on", "turn_off"}
                    else f"{verb}_{prefix}_fixture"
                )
                inspect_contact_primitive = f"inspect_{prefix}_button_contact_frame"
                sweep_contact_primitive = f"sweep_{prefix}_button_contact_candidates"
                if has(primitive):
                    if primitive == f"press_{prefix}_fixture_button" and has(inspect_contact_primitive) and has(sweep_contact_primitive):
                        configured_max_attempts = _coerce_int(action.get("max_attempts")) or 16
                        configured_candidate_limit = _coerce_int(action.get("candidate_limit")) or max(16, configured_max_attempts)
                        sweep_kwargs = action_kwargs(ROBOCASA_SWEEP_ACTION_ARGUMENT_KEYS)
                        sweep_kwargs.setdefault("max_attempts", configured_max_attempts)
                        # The strict operation cases use PandaOmron.  Give semantic press
                        # actions the same coarse mobile-base approach as the canonical
                        # smoke runner, then latch to arm-only control near the fixture.
                        sweep_kwargs.setdefault("mobile_base_enabled", True)
                        sweep_kwargs.setdefault(
                            "mobile_base_active_phases",
                            ["button_approach", "button_press", "button_contact_seek"],
                        )
                        sweep_kwargs.setdefault("base_delta_frame", "base")
                        sweep_kwargs.setdefault("base_xy_deadband", 0.10)
                        sweep_kwargs.setdefault("base_max_delta", 0.15)
                        # Official RoboCasa button tasks require the gripper to
                        # clear the button after contact (for example,
                        # StartCoffeeMachine checks gripper_button_far). Preserve
                        # the successful fixture state while finishing the
                        # semantic press with an arm-only retreat.
                        sweep_kwargs.setdefault("retreat_steps", 60)
                        sweep_kwargs.setdefault("retreat_distance", 0.22)
                        return with_visual_grounding(
                            [
                                step(
                                    inspect_contact_primitive,
                                    {
                                        "fixture_name": fixture,
                                        "button_name": button_name,
                                        "button_position": button_position,
                                        "visual_binding_tolerance": action.get("visual_binding_tolerance"),
                                        "candidate_limit": configured_candidate_limit,
                                        "prompt": prompt,
                                        "query": query,
                                        "agent_context": agent_context,
                                    },
                                ),
                                step(
                                    sweep_contact_primitive,
                                    {
                                        "contact_frame_candidates": {"$from_previous": ["contact_frame_candidates"]},
                                        "fixture_name": fixture,
                                        "button_name": button_name,
                                        "button_position": button_position,
                                        "visual_binding_tolerance": action.get("visual_binding_tolerance"),
                                        "prompt": prompt,
                                        "query": query,
                                        "agent_context": agent_context,
                                        **sweep_kwargs,
                                    },
                                ),
                            ],
                            [(fixture, button_position or entity_pose(fixture, pose))],
                        )
                    return with_visual_grounding(
                        [
                            step(
                                primitive,
                                {
                                    "fixture_name": fixture,
                                    "button_name": button_name,
                                    "button_position": button_position,
                                    "prompt": prompt,
                                    "query": query,
                                    "agent_context": agent_context,
                                    **action_kwargs(ROBOCASA_PRESS_ACTION_ARGUMENT_KEYS),
                                },
                            )
                        ],
                        [(fixture, button_position or entity_pose(fixture, pose))],
                    )
            if has(f"grasp_{prefix}_object") and verb in {"grasp", "pick"} and obj:
                return with_visual_grounding(
                    [
                        step(
                            f"grasp_{prefix}_object",
                            {
                                "object_name": obj,
                                "prompt": prompt,
                                "query": query,
                                "agent_context": agent_context,
                                **action_kwargs(ROBOCASA_GRASP_ACTION_ARGUMENT_KEYS),
                            },
                        )
                    ],
                    [(obj, entity_pose(obj, pose))],
                )
            if has(f"place_{prefix}_object_at") and verb in {"place", "put", "pick_place"} and obj:
                return with_visual_grounding(
                    [
                        step(
                            f"place_{prefix}_object_at",
                            {
                                "object_name": obj,
                                "target_name": target,
                                "target_position": xyz or entity_pose(target, pose),
                                "relation": relation or "at",
                                "prompt": prompt,
                                "query": query,
                                "agent_context": agent_context,
                                **action_kwargs(ROBOCASA_PLACE_ACTION_ARGUMENT_KEYS),
                            },
                        )
                    ],
                    [(obj, entity_pose(obj)), (target, xyz or entity_pose(target, pose))],
                )
            if has(f"move_{prefix}_ee_to") and (target or xyz is not None):
                entity_name = target or obj
                target_position = xyz or entity_pose(entity_name, pose)
                return with_visual_grounding(
                    [
                        step(
                            f"move_{prefix}_ee_to",
                            {
                                "target_name": entity_name,
                                "target_position": target_position,
                                "prompt": prompt,
                                "query": query,
                                "agent_context": agent_context,
                                **action_kwargs(ROBOCASA_MOVE_ACTION_ARGUMENT_KEYS),
                            },
                        )
                    ],
                    [(entity_name, target_position)],
                )

        if benchmark_id == "capx":
            if (
                verb in {"pick_place", "place", "grasp_place"}
                and obj
                and target
                and _norm(obj) != _norm(target)
            ):
                return [
                    step(
                        "__universal_capx_pick_place__",
                        {
                            "source": obj,
                            "target": target,
                            "approach_clearance": action.get(
                                "approach_clearance",
                                action.get("clearance", 0.14),
                            ),
                            "lift_clearance": action.get("lift_clearance", 0.28),
                            "stack_margin": action.get("stack_margin", 0.005),
                            "quaternion_wxyz": action.get("quaternion_wxyz"),
                            "visual_evidence_refs": self._capx_visual_evidence_refs_from_handles(
                                evidence_handles
                            ),
                        },
                    )
                ]
            capx_submit_args = self._capx_submit_action_arguments(
                action=action,
                verb=verb,
                pose=pose,
                evidence_handles=evidence_handles,
                entity_id=target or obj,
                agent_context=agent_context,
            )
            explicit_submit = any(
                key in action
                for key in (
                    "capx_action",
                    "api_action",
                    "primitive_action",
                    "parameters",
                    "prefer_submit_capx_action",
                )
            )
            if has("submit_capx_action") and capx_submit_args is not None and explicit_submit:
                return [step("submit_capx_action", capx_submit_args)]
            if has("capx_sample_grasp_pose") and verb in {"grasp", "pick"} and obj:
                return [step("capx_sample_grasp_pose", {"object_name": obj, "query": query, "agent_context": agent_context})]
            capx_motion_requested = (
                pose is not None
                or action.get("position") is not None
                or action.get("target_position") is not None
                or verb in {"move", "reach", "goto", "go_to", "goto_pose", "approach", "place", "pick_place"}
            )
            if has("capx_goto_pose") and capx_motion_requested:
                goto_arguments = self._capx_goto_pose_arguments(
                    action=action,
                    pose=pose,
                    evidence_handles=evidence_handles,
                    entity_id=target or obj,
                    agent_context=agent_context,
                )
                if goto_arguments is not None:
                    return [step("capx_goto_pose", goto_arguments)]
            if has("capx_close_gripper") and verb in {"close", "grasp"}:
                return [step("capx_close_gripper", {"agent_context": agent_context})]
            if has("capx_open_gripper") and verb in {"open", "release"}:
                return [step("capx_open_gripper", {"agent_context": agent_context})]
            if has("submit_capx_action") and capx_submit_args is not None:
                return [step("submit_capx_action", capx_submit_args)]

        if benchmark_id == "rlbench":
            if has("step_rlbench_action") and action.get("control") is not None:
                return [step("step_rlbench_action", {"action": action["control"], "agent_context": agent_context, "evidence_refs": evidence_handles})]
            rlbench_pose = self._complete_pose_from_evidence(pose, evidence_handles, target or obj, min_length=7)
            steps: list[JsonDict] = []
            if has("move_rlbench_arm_to") and (target or obj or pose is not None):
                steps.append(
                    step(
                        "move_rlbench_arm_to",
                        {
                            "target_name": target or obj,
                            "target_pose": rlbench_pose,
                            "strategy": "rlbench_pose_action",
                            "agent_context": agent_context,
                            "evidence_refs": evidence_handles,
                        },
                    )
                )
            if has("close_rlbench_gripper") and (verb in {"reach_close", "close", "grasp", "pick", "pick_place"} or action.get("gripper") == "closed"):
                steps.append(step("close_rlbench_gripper", {"agent_context": agent_context, "evidence_refs": evidence_handles}))
            if has("open_rlbench_gripper") and verb in {"open", "release"}:
                steps.append(step("open_rlbench_gripper", {"agent_context": agent_context, "evidence_refs": evidence_handles}))
            if steps:
                return steps

        if benchmark_id == "calvin":
            if has("submit_calvin_action") and action.get("control") is not None:
                return [step("submit_calvin_action", {"action": action["control"], "agent_context": agent_context, "evidence_refs": evidence_handles})]
            if has("execute_calvin_language_skill"):
                subgoal = (
                    action.get("subgoal")
                    or action.get("language_goal")
                    or action.get("instruction")
                    or action.get("language")
                    or prompt
                    or verb
                )
                return [step("execute_calvin_language_skill", {"subgoal": subgoal, "horizon": int(action.get("horizon", 1)), "agent_context": agent_context, "evidence_refs": evidence_handles})]

        if benchmark_id == "behavior1k":
            behavior_evidence_handles = self._native_visual_evidence_ids_from_handles(
                evidence_handles, prefixes=("behavior1k:",)
            )
            if has("execute_behavior1k_action_sequence") and isinstance(action.get("actions"), list):
                return [step("execute_behavior1k_action_sequence", {"actions": action["actions"], "prompt": prompt, "query": query, "evidence_handles": behavior_evidence_handles, "agent_context": agent_context})]
            if has("execute_behavior1k_controller_command") and action.get("controller_command") is not None:
                controller_command = action["controller_command"]
                commands = controller_command.get("commands") if isinstance(controller_command, dict) and "commands" in controller_command else controller_command
                return [
                    step(
                        "execute_behavior1k_controller_command",
                        {
                            "robot_name": _clean_name(action.get("robot_name") or action.get("robot") or "robot_runtime_id"),
                            "commands": commands,
                            "repeat": int(action.get("repeat", 1)),
                            "prompt": prompt,
                            "query": query,
                            "evidence_handles": behavior_evidence_handles,
                            "agent_context": agent_context,
                        },
                    )
                ]
            behavior_navigation_requested = (
                action.get("target_pose_xyyaw") is not None
                or action.get("base_pose_xyyaw") is not None
                or action.get("xyyaw") is not None
                or action.get("target_base_pose") is not None
                or action.get("base_pose") is not None
                or verb in {"navigate", "move_base", "base_move", "goto", "go_to", "approach"}
            )
            behavior_target_xyyaw = None
            if behavior_navigation_requested:
                behavior_target_xyyaw = _coerce_pose_list(
                    action.get("target_pose_xyyaw")
                    or action.get("base_pose_xyyaw")
                    or action.get("xyyaw")
                )
            if behavior_navigation_requested and behavior_target_xyyaw is None:
                base_xyz = _pose_xyz(action.get("target_base_pose") or action.get("base_pose") or pose)
                if base_xyz is not None:
                    behavior_target_xyyaw = [
                        base_xyz[0],
                        base_xyz[1],
                        _coerce_float(action.get("target_yaw", action.get("yaw", 0.0)), default=0.0),
                    ]
            if has("navigate_behavior1k_base_to_pose") and behavior_target_xyyaw is not None and len(behavior_target_xyyaw) >= 3:
                return [
                    step(
                        "navigate_behavior1k_base_to_pose",
                        {
                            "robot_name": _clean_name(action.get("robot_name") or action.get("robot") or "robot_runtime_id"),
                            "base_controller_name": _clean_name(action.get("base_controller_name") or action.get("controller_name") or action.get("controller") or "base"),
                            "target_pose_xyyaw": behavior_target_xyyaw[:3],
                            "max_steps": action.get("max_steps"),
                            "distance_tolerance": action.get("distance_tolerance"),
                            "yaw_tolerance": action.get("yaw_tolerance"),
                            "linear_gain": action.get("linear_gain"),
                            "angular_gain": action.get("angular_gain"),
                            "linear_limit": action.get("linear_limit"),
                            "angular_limit": action.get("angular_limit"),
                            "turn_in_place_yaw_threshold": action.get("turn_in_place_yaw_threshold"),
                            "command_mode": action.get("command_mode"),
                            "repeat_per_command": action.get("repeat_per_command"),
                            "prompt": prompt,
                            "query": query,
                            "evidence_handles": behavior_evidence_handles,
                            "agent_context": agent_context,
                        },
                    )
                ]
            if has("run_behavior1k_semantic_action"):
                return [
                    step(
                        "run_behavior1k_semantic_action",
                        {
                            "semantic_action": _clean_name(action.get("semantic_action") or action.get("action") or verb or "interact"),
                            "asset_name": _clean_name(action.get("asset_name") or obj or target or "target"),
                            "secondary_asset_name": _clean_name(action.get("secondary_asset_name") or action.get("secondary_target") or target),
                            "robot_name": _clean_name(action.get("robot_name") or action.get("robot")),
                            "attempts": action.get("attempts"),
                            "max_steps": action.get("max_steps"),
                            "evidence_handles": behavior_evidence_handles,
                            "agent_context": agent_context,
                        },
                    )
                ]
            if has("submit_behavior1k_action") and action.get("control") is not None:
                return [step("submit_behavior1k_action", {"action": action["control"], "prompt": prompt, "query": query, "evidence_handles": behavior_evidence_handles, "agent_context": agent_context})]

        if benchmark_id == "robowits":
            control_payload = action.get("control") if isinstance(action.get("control"), dict) else {}
            robowits_target_position = _pose_xyz(
                action.get("target_position")
                or action.get("position")
                or control_payload.get("target_position")
                or control_payload.get("position")
                or pose
            )
            robowits_arm = _clean_name(action.get("arm") or control_payload.get("arm") or "auto")
            robowits_axis_angle = action.get("axis_angle", control_payload.get("axis_angle"))
            robowits_visual_evidence_handles = [
                handle
                for handle in _coerce_handle_list(
                    action.get("visual_evidence_handles"),
                    action.get("native_evidence_handles"),
                    control_payload.get("visual_evidence_handles"),
                    control_payload.get("native_evidence_handles"),
                )
                if handle.startswith("robowits:visual:")
            ]
            robowits_visual_evidence_handles.extend(
                self._native_visual_evidence_ids_from_handles(
                    evidence_handles,
                    prefixes=("robowits:visual:",),
                )
            )
            robowits_visual_evidence_handles = _dedupe_strings(robowits_visual_evidence_handles)
            if has("execute_robowits_ee_control") and verb == "pick_place" and obj and robowits_target_position is not None:
                resolved_source = self._resolve_entity(obj)
                live_source_position = None
                live_source = self._call_if_available(
                    "observe_robowits_state",
                    names=[obj],
                    query="semantic_pick_place_source",
                    context={
                        **(agent_context or {}),
                        "semantic_phase": "source_refresh",
                        "source_entity": obj,
                    },
                )
                if live_source is not None and live_source.ok and isinstance(live_source.output, dict):
                    object_state = live_source.output.get("object_state")
                    object_poses = object_state.get("object_poses") if isinstance(object_state, dict) else None
                    source_record = object_poses.get(obj) if isinstance(object_poses, dict) else None
                    if isinstance(source_record, dict):
                        live_source_position = _pose_xyz(source_record.get("center") or source_record.get("pos"))
                source_position = (
                    _pose_xyz(action.get("source_position"))
                    or live_source_position
                    or self._extract_pose_from_payload(resolved_source or {}, min_length=3)
                    or self._pose_from_evidence(evidence_handles, obj)
                )
                if source_position is not None:
                    source_half_height = 0.025
                    source_attrs = resolved_source.get("attributes") if isinstance(resolved_source, dict) else None
                    source_bounds = source_attrs.get("bounds") if isinstance(source_attrs, dict) else None
                    if (
                        isinstance(source_bounds, list)
                        and len(source_bounds) >= 2
                        and (lower := _pose_xyz(source_bounds[0])) is not None
                        and (upper := _pose_xyz(source_bounds[1])) is not None
                    ):
                        source_half_height = max(0.005, min(0.08, abs(float(upper[2]) - float(lower[2])) / 2.0))
                    grasp_offset = max(
                        0.0,
                        min(
                            0.08,
                            _coerce_float(
                                action.get("grasp_height_offset"),
                                default=0.0,
                            ),
                        ),
                    )
                    lift_height = max(
                        grasp_offset + 0.08,
                        min(
                            0.23,
                            _coerce_float(
                                action.get("lift_height"),
                                default=max(0.18, source_half_height * 4.0),
                            ),
                        ),
                    )
                    repeat_steps = max(12, min(16, _coerce_int(action.get("repeat_steps")) or 12))
                    max_translation = max(
                        0.005,
                        min(0.3, _coerce_float(action.get("max_translation"), default=0.3)),
                    )
                    open_width = max(0.0, min(0.1, _coerce_float(action.get("open_gripper_width"), default=0.085)))
                    closed_width = max(0.0, min(0.1, _coerce_float(action.get("closed_gripper_width"), default=0.012)))

                    def robowits_pick_place_step(
                        phase: str,
                        position: list[float],
                        *,
                        gripper_width: float,
                        phase_repeat_steps: int | None = None,
                        hold_steps: int = 0,
                    ) -> JsonDict:
                        return step(
                            "execute_robowits_ee_control",
                            {
                                "target_position": [float(value) for value in position[:3]],
                                "arm": robowits_arm,
                                "axis_angle": robowits_axis_angle,
                                "gripper_width": gripper_width,
                                "repeat_steps": phase_repeat_steps or repeat_steps,
                                "hold_steps": hold_steps,
                                "max_translation": max_translation,
                                "observe_names": [obj],
                                "evidence_handles": robowits_visual_evidence_handles,
                                "context": {**(agent_context or {}), "semantic_phase": phase, "source_entity": obj},
                            },
                        )

                    source_grasp = [
                        float(source_position[0]),
                        float(source_position[1]),
                        float(source_position[2]) + grasp_offset,
                    ]
                    source_lift = [source_grasp[0], source_grasp[1], float(source_position[2]) + lift_height]
                    target_grasp = [
                        float(robowits_target_position[0]),
                        float(robowits_target_position[1]),
                        float(robowits_target_position[2]) + grasp_offset,
                    ]
                    target_lift = [target_grasp[0], target_grasp[1], float(robowits_target_position[2]) + lift_height]
                    refresh_visual = step(
                        "inspect_robowits_live_observation",
                        {
                            "prompt": "Refresh native RGB evidence for this semantic pick-and-place action.",
                            "query": f"pick_place:{obj}",
                            "context": {
                                **(agent_context or {}),
                                "semantic_phase": "visual_refresh",
                                "source_entity": obj,
                            },
                        },
                    )
                    fresh_visual_handles = {
                        "$from_step": 0,
                        "path": ["evidence_handles"],
                    }
                    original_visual_handles = robowits_visual_evidence_handles
                    robowits_visual_evidence_handles = fresh_visual_handles
                    return [
                        refresh_visual,
                        robowits_pick_place_step("approach", source_lift, gripper_width=open_width),
                        robowits_pick_place_step("descend", source_grasp, gripper_width=open_width),
                        robowits_pick_place_step(
                            "grasp",
                            source_grasp,
                            gripper_width=closed_width,
                            phase_repeat_steps=repeat_steps,
                            hold_steps=16,
                        ),
                        robowits_pick_place_step("lift", source_lift, gripper_width=closed_width),
                        robowits_pick_place_step("transport", target_lift, gripper_width=closed_width),
                        step(
                            "__universal_robowits_feedback_place__",
                            {
                                "object_name": obj,
                                "target_position": target_grasp,
                                "arm": robowits_arm,
                                "axis_angle": robowits_axis_angle,
                                "closed_gripper_width": closed_width,
                                "open_gripper_width": open_width,
                                "repeat_steps": repeat_steps,
                                "max_translation": max_translation,
                                "lift_height": lift_height,
                                "initial_visual_evidence_handles": original_visual_handles,
                            },
                        ),
                    ]
            if has("execute_robowits_ee_control") and robowits_target_position is not None:
                return [
                    step(
                        "execute_robowits_ee_control",
                        {
                            "target_position": robowits_target_position,
                            "arm": robowits_arm,
                            "axis_angle": robowits_axis_angle,
                            "gripper_width": action.get("gripper_width", control_payload.get("gripper_width")),
                            "repeat_steps": action.get("repeat_steps", control_payload.get("repeat_steps")),
                            "hold_steps": action.get("hold_steps", control_payload.get("hold_steps")),
                            "max_translation": action.get("max_translation", control_payload.get("max_translation")),
                            "observe_names": action.get("observe_names", control_payload.get("observe_names")),
                            "evidence_handles": robowits_visual_evidence_handles,
                            "context": agent_context or {},
                        },
                    )
                ]
            if has("query_robowits_motion") and robowits_target_position is not None:
                return [
                    step(
                        "query_robowits_motion",
                        {
                            "target_position": robowits_target_position,
                            "arm": robowits_arm,
                            "axis_angle": robowits_axis_angle,
                            "ignored_names": action.get("ignored_names", control_payload.get("ignored_names")),
                            "clearance_radius": action.get("clearance_radius", control_payload.get("clearance_radius")),
                            "samples": action.get("samples", control_payload.get("samples")),
                            "evidence_handles": robowits_visual_evidence_handles,
                            "context": agent_context or {},
                        },
                    )
                ]

        if benchmark_id == "robotwin2":
            arm_tag = _clean_name(action.get("arm") or action.get("arm_tag") or "right")
            if verb in {"pick_place", "place", "grasp_place"} and obj and target:
                return [
                    step(
                        "__universal_robotwin2_pick_place__",
                        {
                            "source": obj,
                            "target": target,
                            "arm": arm_tag,
                            "camera_name": action.get("camera_name", "head_camera"),
                            "quaternion_wxyz": action.get("quaternion_wxyz"),
                        },
                    )
                ]
            native_visual_handles = _dedupe_strings(
                [
                    *_coerce_handle_list(
                        action.get("evidence_handles"),
                        action.get("native_visual_evidence_handles"),
                        action.get("visual_evidence_handles"),
                    ),
                    *self._native_visual_evidence_ids_from_handles(
                        evidence_handles,
                        prefixes=("robotwin2:visual:",),
                    ),
                ]
            )

            def with_robotwin2_visual_evidence(base_step: JsonDict) -> list[JsonDict]:
                if native_visual_handles:
                    base_step.setdefault("arguments", {})["evidence_handles"] = native_visual_handles
                    return [base_step]
                if has("observe_robotwin2_visual"):
                    base_step.setdefault("arguments", {})["evidence_handles"] = {
                        "$from_step": 0,
                        "path": ["evidence_handles"],
                    }
                    return [
                        step(
                            "observe_robotwin2_visual",
                            {
                                "query": "refresh_visual_evidence_before_action",
                                "agent_context": agent_context,
                            },
                        ),
                        base_step,
                    ]
                base_step.setdefault("arguments", {})["evidence_handles"] = list(evidence_handles)
                return [base_step]

            if has("execute_robotwin2_actions") and (action.get("left") is not None or action.get("right") is not None):
                return with_robotwin2_visual_evidence(
                    step(
                        "execute_robotwin2_actions",
                        {"left": action.get("left"), "right": action.get("right"), "agent_context": agent_context},
                    )
                )
            if has("submit_robotwin2_ee_action") and action.get("control") is not None:
                return with_robotwin2_visual_evidence(
                    step(
                        "submit_robotwin2_ee_action",
                        {"action": action["control"], "action_type": action.get("action_type"), "agent_context": agent_context},
                    )
                )
            if has("set_robotwin2_gripper") and verb in {"open", "close", "grasp", "release", "gripper"}:
                command = "close" if verb in {"close", "grasp"} else "open"
                return with_robotwin2_visual_evidence(
                    step(
                        "set_robotwin2_gripper",
                        {"arm_tag": arm_tag, "command": action.get("command") or command, "pos": action.get("pos"), "agent_context": agent_context},
                    )
                )
            if has("move_robotwin2_arm") and (pose is not None or action.get("delta") is not None):
                return with_robotwin2_visual_evidence(
                    step(
                        "move_robotwin2_arm",
                        {"arm_tag": arm_tag, "target_pose": pose, "delta": action.get("delta"), "agent_context": agent_context},
                    )
                )
            if has("probe_robotwin2_motion_plan") and pose is not None:
                return with_robotwin2_visual_evidence(
                    step(
                        "probe_robotwin2_motion_plan",
                        {"arm_tag": arm_tag, "target_pose": pose, "agent_context": agent_context},
                    )
                )

        if benchmark_id == "robodojo":
            if has("submit_robodojo_low_level_actions") and action.get("actions") is not None:
                return [step("submit_robodojo_low_level_actions", {"actions": action.get("actions"), "agent_context": agent_context, "evidence_refs": evidence_handles})]
            if has("compile_robodojo_contact_lift_actions") and has("submit_robodojo_low_level_actions") and verb in {
                "contact_lift",
                "pick_lift",
            }:
                object_label = _clean_name(action.get("object_label") or obj)
                arm = _clean_name(action.get("arm"))
                if not object_label or arm not in {"left", "right"}:
                    return []
                compile_arguments: JsonDict = {
                    "runtime_state": self._latest_public_observation_data(),
                    "object_label": object_label,
                    "arm": arm,
                    "approach_axis": action.get("approach_axis"),
                    "jaw_axis": action.get("jaw_axis"),
                    "agent_context": agent_context,
                }
                for key in (
                    "lift_distance",
                    "lift_vector",
                    "lift_axis",
                    "precontact_distance",
                    "contact_depth",
                    "contact_position_offset",
                    "open_gripper_opening",
                    "grasp_gripper_opening",
                    "grasp_compression",
                    "precontact_repeat_steps",
                    "contact_repeat_steps",
                    "grasp_repeat_steps",
                    "lift_repeat_steps",
                    "max_cartesian_step",
                ):
                    if action.get(key) is not None:
                        compile_arguments[key] = action[key]
                native_evidence_ids = self._native_visual_evidence_ids_from_handles(
                    evidence_handles,
                    prefixes=("robodojo:visual:",),
                )
                return [
                    step("compile_robodojo_contact_lift_actions", compile_arguments),
                    step(
                        "submit_robodojo_low_level_actions",
                        {
                            "actions": {"$from_step": 0, "path": ["low_level_actions"]},
                            "reobserve_after_each_action": bool(action.get("reobserve_after_each_action", False)),
                            "control_horizon_frames": action.get("control_horizon_frames"),
                            "execution_mode": action.get("execution_mode", "point_ik"),
                            "planner_arm": action.get("planner_arm") or (
                                arm if action.get("execution_mode") == "curobo_trajectory" else None
                            ),
                            "agent_context": agent_context,
                            "evidence_refs": native_evidence_ids,
                        },
                    ),
                ]
            if has("run_robodojo_policy_skill"):
                return [
                    step(
                        "run_robodojo_policy_skill",
                        {
                            "observation": action.get("observation") or self._latest_public_observation_data(),
                            "agent_prompt": action.get("prompt") or prompt,
                            "policy_name": action.get("policy", "ACT"),
                            "max_actions": int(action.get("max_actions", action.get("max_steps", 8))),
                            "agent_context": agent_context,
                            "evidence_refs": evidence_handles,
                        },
                    )
                ]

        return []

    def _compile_generic_by_available_primitives(
        self,
        *,
        verb: str,
        obj: str | None,
        target: str | None,
        relation: str,
        action: JsonDict,
    ) -> list[JsonDict]:
        names = {card.name for card in self.backend.list_primitives()}
        if "run_robodojo_policy_skill" in names:
            return [
                {
                    "primitive": "run_robodojo_policy_skill",
                    "arguments": {
                        "policy_name": action.get("policy", "ACT"),
                        "agent_prompt": action.get("prompt", self._task.instruction if self._task else ""),
                        "agent_context": action.get("agent_context", {}),
                        "max_steps": int(action.get("max_steps", 8)),
                    },
                }
            ]
        if "execute_calvin_language_skill" in names:
            subgoal = action.get("subgoal") or action.get("language_goal") or action.get("instruction") or action.get("language") or verb
            return [{"primitive": "execute_calvin_language_skill", "arguments": {"subgoal": subgoal, "horizon": int(action.get("horizon", 1))}}]
        if "submit_calvin_action" in names and action.get("control"):
            return [{"primitive": "submit_calvin_action", "arguments": {"action": action["control"]}}]
        if "run_robowits_policy_skill" in names:
            return [{"primitive": "run_robowits_policy_skill", "arguments": {"skill": verb, "target": target or obj}}]
        if (
            self._benchmark_id() == "maniskill"
            and verb in {"pick_place", "place", "grasp_place"}
            and obj
            and target
            and {"grasp_maniskill_actor", "place_maniskill_actor_on"}.issubset(names)
        ):
            return [
                {
                    "primitive": "grasp_maniskill_actor",
                    "arguments": {"actor_name": obj},
                },
                {
                    "primitive": "place_maniskill_actor_on",
                    "arguments": {"actor_name": obj, "target_name": target},
                },
            ]
        if {"pick", "place"}.issubset(names) and obj and target:
            return [
                {"primitive": "pick", "arguments": {"object_id": obj}},
                {"primitive": "place", "arguments": {"object_id": obj, "target_id": target, "relation": relation}},
            ]
        return []

    def _enumerate_entities(self, query: str | None = None) -> list[JsonDict]:
        source = self._last_observation.data if self._last_observation else {}
        if not isinstance(source, dict):
            source = {}
        task_state = deepcopy(self._task.initial_state if self._task else {})
        merged = _deep_merge(task_state, source.get("state", {}) if isinstance(source.get("state"), dict) else {})
        entities: list[JsonDict] = []

        def append_public_registry(registry: Any, *, source_name: str, default_kind: str) -> None:
            if isinstance(registry, dict):
                for entity_id, attrs in registry.items():
                    entity_id = str(entity_id)
                    if isinstance(attrs, dict):
                        payload = deepcopy(attrs)
                        payload.setdefault("source", source_name)
                        payload.setdefault("kind", payload.get("kind") or default_kind)
                        payload.setdefault("name", entity_id)
                        label = _clean_name(payload.get("label") or payload.get("name") or entity_id)
                        entities.append(self._entity_payload(entity_id, label or entity_id, payload))
                    elif _looks_like_pose(attrs):
                        entities.append(
                            self._entity_payload(
                                entity_id,
                                entity_id,
                                {"pose": attrs, "source": source_name, "kind": default_kind},
                            )
                        )
            elif isinstance(registry, list):
                for item in registry:
                    if isinstance(item, dict):
                        entity_id = _clean_name(
                            item.get("entity_id")
                            or item.get("id")
                            or item.get("name")
                            or item.get("object_name")
                            or item.get("fixture_name")
                        )
                        if not entity_id:
                            continue
                        payload = deepcopy(item)
                        payload.setdefault("source", source_name)
                        payload.setdefault("kind", payload.get("kind") or default_kind)
                        entities.append(
                            self._entity_payload(
                                entity_id,
                                _clean_name(payload.get("label") or payload.get("name") or entity_id) or entity_id,
                                payload,
                            )
                        )
                    elif isinstance(item, str):
                        entities.append(
                            self._entity_payload(
                                item,
                                item,
                                {"source": source_name, "kind": default_kind},
                            )
                        )
        objects = merged.get("objects")
        if isinstance(objects, dict):
            for entity_id, attrs in objects.items():
                if isinstance(attrs, dict):
                    entities.append(self._entity_payload(entity_id, attrs.get("label") or entity_id, attrs))
        elif isinstance(objects, list):
            for entity_id in objects:
                entities.append(self._entity_payload(str(entity_id), str(entity_id), {}))
        object_poses = merged.get("object_poses")
        if isinstance(object_poses, dict):
            for entity_id, pose in object_poses.items():
                attrs = pose if isinstance(pose, dict) else {"pose": pose}
                entities.append(self._entity_payload(str(entity_id), str(entity_id), attrs))
        for key in ("object", "target", "subgoal"):
            value = merged.get(key)
            if isinstance(value, str):
                entities.append(self._entity_payload(value, value, {"source": key}))
        actor_poses = merged.get("actor_poses")
        if isinstance(actor_poses, dict):
            for entity_id, pose in actor_poses.items():
                entities.append(self._entity_payload(str(entity_id), str(entity_id), {"pose": pose}))
        scene_targets = merged.get("scene_targets")
        if isinstance(scene_targets, list):
            for target in scene_targets:
                entities.append(self._entity_payload(str(target), str(target), {"source": "scene_targets"}))
        append_public_registry(source.get("objects"), source_name="observation.objects", default_kind="object")
        append_public_registry(source.get("fixtures"), source_name="observation.fixtures", default_kind="fixture")
        if self._benchmark_id() == "maniskill":
            goal_xyz = _pose_xyz(source.get("goal_xyz"))
            if goal_xyz is not None:
                entities.append(
                    self._entity_payload(
                        "public_goal_position",
                        "public goal position",
                        {
                            "kind": "target",
                            "position": goal_xyz,
                            "source": "observation.goal_xyz",
                        },
                    )
                )
        pose_evidence = source.get("pose_evidence")
        if isinstance(pose_evidence, dict):
            append_public_registry(
                pose_evidence.get("objects"),
                source_name="observation.pose_evidence.objects",
                default_kind="object",
            )
            append_public_registry(
                pose_evidence.get("fixtures"),
                source_name="observation.pose_evidence.fixtures",
                default_kind="fixture",
            )
        entities.extend(self._enumerate_entities_from_native_profile(query=query))
        entities = _dedupe_entities(entities)
        if query:
            filtered = [entity for entity in entities if _norm(query) in _norm(entity.get("label", "")) or _norm(query) in _norm(entity.get("entity_id", ""))]
            if filtered:
                entities = filtered
        return entities

    def _enumerate_entities_from_native_profile(self, query: str | None = None) -> list[JsonDict]:
        entities: list[JsonDict] = []
        for primitive in self._adapter_profile().enumerate_primitives:
            arguments = self._arguments_for_native_primitive(
                primitive,
                entity_id=query or "",
                resolved={},
                query=query,
                agent_context=None,
            )
            result = self._call_if_available(primitive, **arguments)
            if result is None or not result.ok:
                continue
            entities.extend(self._entities_from_public_native_output(result.output, source=primitive))
            if entities:
                break
        return entities

    def _entities_from_public_native_output(self, output: JsonDict, *, source: str) -> list[JsonDict]:
        if not isinstance(output, dict):
            return []
        parent_evidence_id = output.get("evidence_id") if isinstance(output.get("evidence_id"), str) else None
        candidates: list[Any] = []
        grounding_evidence = output.get("object_grounding_evidence")
        if isinstance(grounding_evidence, dict):
            entity_evidence = grounding_evidence.get("entity_evidence")
            if isinstance(entity_evidence, (list, dict)):
                candidates.append(entity_evidence)
        visual_inspection = output.get("visual_inspection")
        if isinstance(visual_inspection, dict):
            visual_candidates = visual_inspection.get("entity_visual_candidates")
            if isinstance(visual_candidates, (list, dict)):
                candidates.append(visual_candidates)
        for field_name in ("entities", "objects", "instances", "exact_prompt_bindings"):
            value = output.get(field_name)
            if isinstance(value, (list, dict)):
                candidates.append(value)
        if not candidates:
            candidates.append(output)

        entities: list[JsonDict] = []
        for candidate in candidates:
            if isinstance(candidate, dict):
                for key, value in candidate.items():
                    if isinstance(value, dict):
                        attrs = deepcopy(value)
                        entity_id = _clean_name(attrs.get("entity_id") or attrs.get("id") or attrs.get("name") or key)
                        label = _clean_name(attrs.get("label") or attrs.get("name") or entity_id)
                        if entity_id and _looks_like_entity_attrs(attrs):
                            attrs.setdefault("source", source)
                            if parent_evidence_id:
                                attrs.setdefault("parent_evidence_id", parent_evidence_id)
                            entity_payload = self._entity_payload(entity_id, label or entity_id, attrs)
                            entities.append(entity_payload)
                            entities.extend(self._surface_region_entities_from_attrs(entity_payload, attrs, source=source, parent_evidence_id=parent_evidence_id))
                    elif isinstance(value, list):
                        for item_index, item in enumerate(value):
                            if not isinstance(item, dict):
                                continue
                            attrs = deepcopy(item)
                            attrs.setdefault("collection", key)
                            if "view" not in attrs:
                                attrs["view"] = key
                            entity_id = _clean_name(
                                attrs.get("entity_id")
                                or attrs.get("id")
                                or attrs.get("name")
                                or attrs.get("prompt_asset_key")
                                or attrs.get("caller_selection")
                                or attrs.get("object_name")
                                or attrs.get("target_name")
                            )
                            if not entity_id and attrs.get("segmentation_id") is not None:
                                entity_id = _clean_name(f"{key}:segmentation:{attrs.get('segmentation_id')}")
                            label = _clean_name(attrs.get("label") or attrs.get("name") or attrs.get("prompt_asset_key") or entity_id)
                            if entity_id and _looks_like_entity_attrs(attrs):
                                attrs.setdefault("source", source)
                                if parent_evidence_id:
                                    attrs.setdefault("parent_evidence_id", parent_evidence_id)
                                entity_payload = self._entity_payload(entity_id, label or entity_id, attrs)
                                entities.append(entity_payload)
                                entities.extend(self._surface_region_entities_from_attrs(entity_payload, attrs, source=source, parent_evidence_id=parent_evidence_id))
                    elif _looks_like_pose(value):
                        entities.append(self._entity_payload(str(key), str(key), {"pose": value, "source": source}))
            elif isinstance(candidate, list):
                for index, value in enumerate(candidate):
                    if isinstance(value, dict):
                        attrs = deepcopy(value)
                        entity_id = _clean_name(
                            attrs.get("entity_id")
                            or attrs.get("id")
                            or attrs.get("name")
                            or attrs.get("prompt_asset_key")
                            or attrs.get("caller_selection")
                            or attrs.get("object_name")
                            or attrs.get("target_name")
                        )
                        if not entity_id and attrs.get("segmentation_id") is not None:
                            entity_id = _clean_name(f"{source}:segmentation:{attrs.get('segmentation_id')}")
                        label = _clean_name(attrs.get("label") or attrs.get("name") or entity_id)
                        if entity_id and _looks_like_entity_attrs(attrs):
                            attrs.setdefault("source", source)
                            if parent_evidence_id:
                                attrs.setdefault("parent_evidence_id", parent_evidence_id)
                            entity_payload = self._entity_payload(entity_id, label or entity_id, attrs)
                            entities.append(entity_payload)
                            entities.extend(self._surface_region_entities_from_attrs(entity_payload, attrs, source=source, parent_evidence_id=parent_evidence_id))
                        elif _looks_like_entity_attrs(attrs):
                            generated_id = f"{source}:entity:{index}"
                            attrs.setdefault("source", source)
                            if parent_evidence_id:
                                attrs.setdefault("parent_evidence_id", parent_evidence_id)
                            entity_payload = self._entity_payload(generated_id, generated_id, attrs)
                            entities.append(entity_payload)
                            entities.extend(self._surface_region_entities_from_attrs(entity_payload, attrs, source=source, parent_evidence_id=parent_evidence_id))
                    elif isinstance(value, str):
                        entities.append(self._entity_payload(value, value, {"source": source}))
        return entities

    def _entity_payload(self, entity_id: str, label: str, attributes: JsonDict) -> JsonDict:
        existing = next((key for key, item in self._handles.items() if item.get("kind") == "entity" and item.get("entity_id") == entity_id), None)
        handle = existing or self._new_handle(
            "ent",
            {"kind": "entity", "entity_id": entity_id, "label": label, "attributes": _sanitize_agent_payload(attributes)},
        )
        return {"entity_handle": handle, "entity_id": entity_id, "label": label, "attributes": _sanitize_agent_payload(attributes)}

    def _surface_region_entities_from_attrs(
        self,
        parent_entity: JsonDict,
        attrs: JsonDict,
        *,
        source: str,
        parent_evidence_id: str | None,
    ) -> list[JsonDict]:
        regions = attrs.get("surface_regions")
        if not isinstance(regions, list):
            return []
        parent_entity_id = str(parent_entity.get("entity_id") or parent_entity.get("label") or "surface")
        parent_handle = str(parent_entity.get("entity_handle") or "")
        entities: list[JsonDict] = []
        for index, region in enumerate(regions):
            if not isinstance(region, dict):
                continue
            label = _clean_name(region.get("label") or region.get("region_label") or f"surface region {index}")
            region_id = _clean_name(region.get("entity_id") or f"{parent_entity_id}:{region.get('region_label') or index}")
            region_attrs = deepcopy(region)
            region_attrs.setdefault("source", f"{source}:surface_region")
            region_attrs.setdefault("virtual_entity", True)
            region_attrs.setdefault("parent_entity_handle", parent_handle)
            region_attrs.setdefault("parent_entity_id", parent_entity_id)
            if parent_evidence_id:
                region_attrs.setdefault("parent_evidence_id", parent_evidence_id)
            entities.append(self._entity_payload(region_id, label or region_id, region_attrs))
        return entities

    def _resolve_entity(self, entity: str | JsonDict | None) -> JsonDict | None:
        if entity is None:
            return None
        selector: JsonDict = {}
        if isinstance(entity, dict):
            handle = entity.get("entity_handle") or entity.get("handle")
            if isinstance(handle, str) and handle in self._handles:
                return deepcopy(self._handles[handle])
            selector = {str(key): value for key, value in entity.items() if value is not None}
            attrs = selector.pop("attributes", None)
            if isinstance(attrs, dict):
                selector.update({str(key): value for key, value in attrs.items() if value is not None})
            entity_id = (
                entity.get("entity_id")
                or entity.get("id")
                or entity.get("label")
                or entity.get("name")
                or entity.get("target_name")
                or entity.get("prompt_asset_key")
            )
        else:
            if entity in self._handles:
                return deepcopy(self._handles[entity])
            entity_id = entity
        entity_id_norm = _norm(str(entity_id)) if entity_id is not None else ""
        for item in self._enumerate_entities():
            if _norm(item.get("entity_id", "")) == entity_id_norm or _norm(item.get("label", "")) == entity_id_norm:
                return deepcopy(self._handles[item["entity_handle"]])
            if selector and _entity_matches_selector(item, selector):
                return deepcopy(self._handles[item["entity_handle"]])
        return None

    def _resolve_handle_or_entity(self, value: str | JsonDict) -> JsonDict | None:
        if isinstance(value, str) and value in self._handles:
            return deepcopy(self._handles[value])
        return self._resolve_entity(value)

    def _locate_entity_with_native_adapter(
        self,
        resolved: JsonDict,
        *,
        query: str | None,
        agent_context: JsonDict | None,
    ) -> JsonDict:
        entity_id = str(resolved.get("entity_id") or resolved.get("label"))
        benchmark_id = self._benchmark_id()
        attrs = deepcopy(resolved.get("attributes", {}))
        if isinstance(attrs, dict) and attrs.get("virtual_entity") and attrs.get("pose0") is not None:
            return {
                "source": "public_entity_surface_region",
                "public": _sanitize_agent_payload(attrs),
                "pose0": _sanitize_agent_payload(attrs.get("pose0")),
                "evidence_id": attrs.get("parent_evidence_id"),
            }
        native_calls: list[tuple[str, JsonDict]] = [
            (primitive, self._arguments_for_native_primitive(primitive, entity_id=entity_id, resolved=resolved, query=query, agent_context=agent_context))
            for primitive in self._adapter_profile().locate_primitives
        ]
        native_calls.extend([
            ("locate_maniskill_actor", {"actor_name": entity_id}),
            ("inspect_maniskill_instance", {"actor_name": entity_id}),
            ("ground_rlbench_target", {"target_name": entity_id, "query": query}),
            ("inspect_robocasa_object", {"object_name": entity_id}),
            ("capx_get_object_pose", {"object_name": entity_id, "query": query, "agent_context": agent_context}),
        ])
        if benchmark_id == "capx":
            native_calls.insert(0, ("capx_get_object_pose", {"object_name": entity_id, "query": query, "agent_context": agent_context}))
        for primitive, arguments in native_calls:
            if not arguments:
                continue
            result = self._call_if_available(primitive, **arguments)
            if result is not None and result.ok:
                output = _sanitize_agent_payload(result.output)
                if "pose" in output:
                    return {"pose": output["pose"], "source": primitive}
                return {"source": primitive, "public": output}
        if "pose" in attrs:
            return {"pose": attrs["pose"], "source": "public_entity_attributes"}
        if "position" in attrs:
            location = {"position": attrs["position"], "source": "public_entity_attributes"}
            if "quaternion_wxyz" in attrs:
                location["quaternion_wxyz"] = attrs["quaternion_wxyz"]
            return location
        if "pose0" in attrs:
            return {"pose0": attrs["pose0"], "source": "public_entity_attributes", "public": _sanitize_agent_payload(attrs)}
        return {"source": "entity_attributes", "public": _sanitize_agent_payload(attrs)}

    def _infer_affordances(self, resolved: JsonDict | None, query: str | None = None) -> list[JsonDict]:
        names = {card.name for card in self.backend.list_primitives()}
        affordances: list[JsonDict] = []
        attrs = resolved.get("attributes") if isinstance((resolved or {}).get("attributes"), dict) else {}
        geometry_tags = [str(tag) for tag in attrs.get("geometry_tags") or []] if isinstance(attrs, dict) else []
        affordance_hints = [str(hint) for hint in attrs.get("affordance_hints") or []] if isinstance(attrs, dict) else []
        if geometry_tags or affordance_hints:
            affordances.append(
                {
                    "type": "public_geometry",
                    "available": True,
                    "geometry_tags": geometry_tags,
                    "affordance_hints": affordance_hints,
                    "grasp_candidate": "grasp_candidate" in affordance_hints or "block_like" in geometry_tags,
                    "place_support_candidate": "place_support_candidate" in affordance_hints
                    or bool({"thin_surface", "container_like", "block_like"}.intersection(geometry_tags)),
                }
            )
        patterns = {
            "grasp": ("grasp", "pick", "close_gripper"),
            "place": ("place", "put", "goto_pose"),
            "press": ("press", "toggle", "close_robocasa_fixture"),
            "open": ("open",),
            "close": ("close",),
            "policy_skill": ("policy", "skill", "execute_", "run_"),
            "navigate": ("navigate", "move", "goto"),
        }
        for label, tokens in patterns.items():
            matching = [name for name in names if any(token in name for token in tokens)]
            if matching:
                affordances.append({"type": label, "available": True, "native_primitive_count": len(matching)})
        adapter_affordances = self._profile_affordance_measurements(resolved, query=query)
        if adapter_affordances:
            affordances.append(
                {
                    "type": "adapter_profile",
                    "available": True,
                    "native_details_hidden": True,
                    "measurements": adapter_affordances,
                }
            )
        if resolved is not None:
            affordances.append({"type": "inspect", "available": True, "entity_handle": resolved.get("entity_handle") or resolved.get("handle")})
        if query:
            query_norm = _norm(query)
            affordances = [item for item in affordances if query_norm in _norm(item["type"])] or affordances
        return affordances

    def _profile_geometry_measurements(
        self,
        subjects: list[JsonDict],
        *,
        measurement: str | None,
        agent_context: JsonDict | None,
    ) -> list[JsonDict]:
        if not subjects:
            return []
        subject = subjects[0]
        entity_id = str(subject.get("entity_id") or subject.get("label") or subject.get("entity_handle") or "")
        measurements: list[JsonDict] = []
        for primitive in self._adapter_profile().geometry_primitives:
            arguments = self._arguments_for_native_primitive(
                primitive,
                entity_id=entity_id,
                resolved=subject,
                query=measurement,
                agent_context=agent_context,
            )
            result = self._call_if_available(primitive, **arguments)
            if result is None:
                continue
            measurements.append(_public_adapter_probe_result(result))
            if len(measurements) >= 3:
                break
        return measurements

    def _profile_affordance_measurements(self, resolved: JsonDict | None, query: str | None = None) -> list[JsonDict]:
        entity_id = str((resolved or {}).get("entity_id") or (resolved or {}).get("label") or "")
        measurements: list[JsonDict] = []
        for primitive in self._adapter_profile().affordance_primitives:
            arguments = self._arguments_for_native_primitive(
                primitive,
                entity_id=entity_id,
                resolved=resolved or {},
                query=query,
                agent_context=None,
            )
            result = self._call_if_available(primitive, **arguments)
            if result is None:
                continue
            measurements.append(_public_adapter_probe_result(result))
            if len(measurements) >= 3:
                break
        return measurements

    def _derive_public_relations(self, subjects: list[JsonDict]) -> list[JsonDict]:
        relations: list[JsonDict] = []
        if len(subjects) >= 2:
            left, right = subjects[0], subjects[1]
            relations.append(
                {
                    "subject": left.get("entity_id") or left.get("entity_handle"),
                    "object": right.get("entity_id") or right.get("entity_handle"),
                    "relation": "relative_pose_available" if _extract_pose(left) and _extract_pose(right) else "public_relation_unresolved",
                }
            )
        return relations

    def _pose_from_evidence(self, evidence_handles: list[str], entity_id: str | None, *, min_length: int = 3) -> list[float] | None:
        for payload in self._iter_handle_graph_payloads(evidence_handles):
            pose = self._extract_pose_from_payload(payload, min_length=min_length)
            if pose is not None:
                return pose
        if entity_id:
            resolved = self._resolve_entity(entity_id)
            pose = self._extract_pose_from_payload(resolved or {}, min_length=min_length)
            if pose:
                return pose
        return None

    def _complete_pose_from_evidence(
        self,
        pose: Any,
        evidence_handles: list[str],
        entity_id: str | None,
        *,
        min_length: int,
    ) -> Any:
        explicit = _coerce_pose_list(pose)
        if explicit is not None and len(explicit) >= min_length:
            return explicit
        explicit_from_payload = self._extract_pose_from_payload(pose, min_length=min_length)
        if explicit_from_payload is not None:
            return explicit_from_payload
        evidence_pose = self._pose_from_evidence(evidence_handles, entity_id, min_length=min_length)
        if evidence_pose is None:
            return pose
        xyz = _pose_xyz(pose)
        if xyz is not None and len(evidence_pose) >= min_length:
            return [*xyz, *evidence_pose[3:min_length]]
        return evidence_pose

    def _extract_pose_from_payload(self, payload: Any, *, min_length: int = 3) -> list[float] | None:
        if isinstance(payload, dict):
            for key in (
                "target_pose",
                "pose",
                "pose_world",
                "eef_pose",
                "gripper_pose",
                "world_target",
                "xpos",
                "position",
                "pos",
                "target_position",
                "center",
                "pose0",
                "centroid_world_m",
                "world_median_m",
            ):
                pose = _coerce_pose_list(payload.get(key))
                if pose is not None and len(pose) >= min_length:
                    return pose
            for key in (
                "location",
                "value",
                "attributes",
                "entity",
                "selected",
                "public",
                "bbox_world",
                "world_bounds_m",
            ):
                pose = self._extract_pose_from_payload(payload.get(key), min_length=min_length)
                if pose is not None:
                    return pose
        elif isinstance(payload, (list, tuple)):
            pose = _coerce_pose_list(payload)
            if pose is not None and len(pose) >= min_length:
                return pose
        return None

    def _int_attribute_from_payload(
        self,
        payload: Any,
        *,
        keys: tuple[str, ...],
        entity_name: str | None = None,
    ) -> int | None:
        entity_norm = _norm(entity_name or "")

        def mentions_entity(value: Any) -> bool:
            if not entity_norm:
                return True
            if isinstance(value, dict):
                for key in (
                    "entity_name",
                    "selected_entity",
                    "entity_id",
                    "label",
                    "name",
                    "object_name",
                    "target_name",
                    "container_name",
                ):
                    item = value.get(key)
                    if isinstance(item, str) and _norm(item) == entity_norm:
                        return True
                for nested_key in ("entity", "selected", "public", "attributes", "location"):
                    if mentions_entity(value.get(nested_key)):
                        return True
            if isinstance(value, (list, tuple)):
                return any(mentions_entity(item) for item in value)
            return False

        def search(value: Any) -> int | None:
            if isinstance(value, dict):
                if mentions_entity(value):
                    for key in keys:
                        if key in value:
                            coerced = _coerce_first_int(value.get(key))
                            if coerced is not None:
                                return coerced
                for item in value.values():
                    coerced = search(item)
                    if coerced is not None:
                        return coerced
            elif isinstance(value, (list, tuple)):
                for item in value:
                    coerced = search(item)
                    if coerced is not None:
                        return coerced
            return None

        return search(payload)

    def _native_visual_evidence_ids_from_handles(self, handles: list[str], *, prefixes: tuple[str, ...]) -> list[str]:
        evidence_ids: list[str] = []

        def collect(value: Any) -> None:
            if isinstance(value, str):
                if value.startswith(prefixes):
                    evidence_ids.append(value)
                return
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"artifact_id", "evidence_id", "evidence_handle"}:
                        collect(item)
                    elif key == "native_artifacts":
                        collect(item)
                    else:
                        collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        for payload in self._iter_handle_graph_payloads(handles):
            collect(payload)
        return _dedupe_strings(evidence_ids)

    def _source_evidence_leaf_handles(
        self,
        handles: list[str],
        *,
        evidence_prefix: str,
        preferred_kinds: tuple[str, ...] = (),
    ) -> list[str]:
        leaves: list[str] = []
        seen: set[str] = set()

        def visit(handle: str) -> None:
            if handle in seen:
                return
            seen.add(handle)
            payload = self._handles.get(handle)
            if isinstance(payload, dict):
                source_handles = [
                    str(item)
                    for item in payload.get("source_handles", [])
                    if isinstance(item, str)
                ]
                if payload.get("kind") == "agent_evidence" and source_handles:
                    for source_handle in source_handles:
                        visit(source_handle)
                    return
            if handle.startswith(evidence_prefix):
                leaves.append(handle)

        for handle in handles:
            if isinstance(handle, str):
                visit(handle)

        if preferred_kinds:
            preferred = [
                handle
                for handle in leaves
                if str((self._handles.get(handle) or {}).get("kind") or "") in preferred_kinds
            ]
            if preferred:
                return _dedupe_strings(preferred)
        return _dedupe_strings(leaves)

    def _robocasa_button_from_entity(
        self,
        entity_id: str | None,
        *,
        requested_button: str | None = None,
    ) -> tuple[str, list[float]] | None:
        resolved = self._resolve_entity(entity_id) if entity_id else None
        attrs = resolved.get("attributes") if isinstance(resolved, dict) else None
        if not isinstance(attrs, dict):
            return None
        affordance_sites = attrs.get("affordance_sites")
        start_buttons = affordance_sites.get("start_buttons") if isinstance(affordance_sites, dict) else None
        if not isinstance(start_buttons, dict) or not start_buttons:
            return None
        button_candidates = [requested_button] if requested_button else []
        button_candidates.extend(sorted(str(key) for key in start_buttons))
        for button_name in button_candidates:
            if not button_name or button_name not in start_buttons:
                continue
            position = _pose_xyz(start_buttons.get(button_name))
            if position is not None:
                return button_name, position
        return None

    def _capx_goto_pose_arguments(
        self,
        *,
        action: JsonDict,
        pose: Any,
        evidence_handles: list[str],
        entity_id: str | None,
        agent_context: JsonDict | None,
    ) -> JsonDict | None:
        parameters = action.get("parameters") if isinstance(action.get("parameters"), dict) else {}
        position = _pose_xyz(
            action.get("position")
            or action.get("target_position")
            or parameters.get("position")
            or pose
        )
        if position is None:
            return None
        quaternion_wxyz = (
            _capx_quaternion_wxyz(action.get("quaternion_wxyz"))
            or _capx_quaternion_wxyz(action.get("target_quat"))
            or _capx_quaternion_wxyz(action.get("orientation"))
            or _capx_quaternion_wxyz(parameters.get("quaternion_wxyz"))
            or _capx_quaternion_wxyz(parameters.get("quaternion"))
            or _capx_quaternion_wxyz(pose)
            or self._capx_quaternion_from_evidence(evidence_handles, entity_id)
            or [1.0, 0.0, 0.0, 0.0]
        )
        z_approach = _coerce_float(
            action.get("z_approach", parameters.get("z_approach", 0.0)),
            default=0.0,
        )
        return {
            "position": position,
            "quaternion_wxyz": quaternion_wxyz,
            "z_approach": z_approach,
            "visual_evidence_refs": self._capx_visual_evidence_refs_from_handles(evidence_handles),
            "agent_context": agent_context,
        }

    def _capx_submit_action_arguments(
        self,
        *,
        action: JsonDict,
        verb: str,
        pose: Any,
        evidence_handles: list[str],
        entity_id: str | None,
        agent_context: JsonDict | None,
    ) -> JsonDict | None:
        action_name = _norm(action.get("capx_action") or action.get("api_action") or action.get("primitive_action") or "")
        if action.get("type") == "native_api" and action_name and self._task is not None and "native_config" in self._task.tags:
            parameters = action.get("parameters", {})
            if not isinstance(parameters, dict):
                return None
            return {"action": action_name, "parameters": parameters, "native_api": True}
        if not action_name:
            candidate = _norm(action.get("action") or action.get("type") or action.get("verb") or verb)
            if candidate in {"goto", "go_to"}:
                action_name = "goto_pose"
            elif candidate in {"open", "release"}:
                action_name = "open_gripper"
            elif candidate in {"close", "grasp"}:
                action_name = "close_gripper"
            elif candidate in {"goto_pose", "open_gripper", "close_gripper"}:
                action_name = candidate
        if action_name not in {"goto_pose", "open_gripper", "close_gripper"}:
            return None
        parameters: JsonDict = {}
        if action_name == "goto_pose":
            goto_arguments = self._capx_goto_pose_arguments(
                action=action,
                pose=pose,
                evidence_handles=evidence_handles,
                entity_id=entity_id,
                agent_context=agent_context,
            )
            if goto_arguments is None:
                return None
            parameters = {
                "position": goto_arguments["position"],
                "quaternion_wxyz": goto_arguments["quaternion_wxyz"],
                "z_approach": goto_arguments["z_approach"],
            }
        return {
            "arm": str(action.get("arm") or "default"),
            "action": action_name,
            "parameters": parameters,
            "visual_evidence_refs": self._capx_visual_evidence_refs_from_handles(evidence_handles),
            "agent_context": agent_context,
        }

    def _capx_quaternion_from_evidence(self, handles: list[str], entity_id: str | None) -> list[float] | None:
        def find(value: Any) -> list[float] | None:
            quaternion = _capx_quaternion_wxyz(value)
            if quaternion is not None:
                return quaternion
            if isinstance(value, dict):
                for item in value.values():
                    found = find(item)
                    if found is not None:
                        return found
            if isinstance(value, (list, tuple)):
                for item in value:
                    found = find(item)
                    if found is not None:
                        return found
            return None

        for payload in self._iter_handle_graph_payloads(handles):
            quaternion = find(payload)
            if quaternion is not None:
                return quaternion
        if entity_id:
            resolved = self._resolve_entity(entity_id)
            quaternion = find(resolved or {})
            if quaternion is not None:
                return quaternion
        return None

    def _capx_visual_evidence_refs_from_handles(self, handles: list[str]) -> list[JsonDict]:
        refs: list[JsonDict] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                source_path = value.get("source_path")
                if isinstance(source_path, str) and source_path and ("modality" in value or "shape" in value):
                    ref: JsonDict = {
                        "kind": value.get("kind", "capx_live_visual_frame"),
                        "source_path": source_path,
                    }
                    for key in ("modality", "shape", "dtype"):
                        if key in value:
                            ref[key] = deepcopy(value[key])
                    refs.append(_sanitize_agent_payload(ref))
                for key, item in value.items():
                    if key == "data" and isinstance(source_path, str) and source_path:
                        continue
                    collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        for payload in self._iter_handle_graph_payloads(handles):
            collect(payload)

        deduped: list[JsonDict] = []
        seen: set[tuple[str, str, tuple[Any, ...]]] = set()
        for ref in refs:
            shape = ref.get("shape")
            shape_key = tuple(shape) if isinstance(shape, list) else ()
            key = (str(ref.get("source_path")), str(ref.get("modality")), shape_key)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ref)
        return deduped

    def _object_ids_from_evidence_handles(self, handles: list[str]) -> list[int]:
        object_ids: list[int] = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"object_id", "source_object_id", "target_object_id"}:
                        coerced = _coerce_int(item)
                        if coerced is not None:
                            object_ids.append(coerced)
                    else:
                        collect(item)
                return
            if isinstance(value, (list, tuple, set)):
                for item in value:
                    collect(item)

        for payload in self._iter_handle_graph_payloads(handles):
            collect(payload)
        deduped: list[int] = []
        seen: set[int] = set()
        for object_id in object_ids:
            if object_id in seen:
                continue
            seen.add(object_id)
            deduped.append(object_id)
        return deduped

    def _pose0_from_evidence_handles(
        self,
        handles: list[str],
        *,
        selector: Any,
        object_id: int | None,
    ) -> list[Any] | None:
        selector_handles = set(_coerce_handle_list(selector))
        selector_entity_id = _clean_name(selector)
        if selector_handles:
            resolved = self._resolve_entity(next(iter(selector_handles)))
            if resolved is not None:
                selector_entity_id = _clean_name(resolved.get("entity_id") or resolved.get("label"))

        def pose0_from(value: Any) -> list[Any] | None:
            if isinstance(value, dict):
                pose0 = value.get("pose0")
                if isinstance(pose0, list) and len(pose0) == 2:
                    return deepcopy(pose0)
                public = value.get("public")
                if isinstance(public, dict):
                    found = pose0_from(public)
                    if found is not None:
                        return found
                instance = value.get("instance")
                if isinstance(instance, dict):
                    found = pose0_from(instance)
                    if found is not None:
                        return found
                location = value.get("location")
                if isinstance(location, dict):
                    found = pose0_from(location)
                    if found is not None:
                        return found
            return None

        def matches(value: Any) -> bool:
            if not isinstance(value, dict):
                return False
            if selector_handles:
                for key in ("entity_handle", "handle", "parent_entity_handle"):
                    item = value.get(key)
                    if isinstance(item, str) and item in selector_handles:
                        return True
            if selector_entity_id:
                for key in ("entity_id", "label", "parent_entity_id", "region_label"):
                    item = value.get(key)
                    if item is not None and _norm(str(item)) == _norm(selector_entity_id):
                        return True
            if object_id is not None:
                for key in ("object_id", "parent_object_id", "source_object_id", "target_object_id"):
                    if _coerce_int(value.get(key)) == object_id:
                        return True
            return False

        fallback: list[Any] | None = None
        for payload in self._iter_handle_graph_payloads(handles):
            pending: list[Any] = [payload]
            seen: set[int] = set()
            while pending:
                value = pending.pop(0)
                value_id = id(value)
                if value_id in seen:
                    continue
                seen.add(value_id)
                if isinstance(value, dict):
                    found = pose0_from(value)
                    if found is not None:
                        fallback = fallback or found
                        if matches(value):
                            return found
                    pending.extend(value.values())
                elif isinstance(value, (list, tuple)):
                    pending.extend(value)
        return fallback

    def _cliport_composite_place_pose_from_evidence(
        self,
        handles: list[str],
        *,
        action: JsonDict | None = None,
        source_object_id: int | None,
        target_object_id: int | None,
    ) -> list[Any] | None:
        supports: dict[int, tuple[list[Any], list[float] | None]] = {}
        excluded_object_ids = {source_object_id} if source_object_id is not None else set()

        source_handles: set[str] = set()
        if action is not None:
            for key in ("source", "source_entity", "object", "entity"):
                source_handles.update(self._entity_handles_from_value(action.get(key)))
        for handle in source_handles:
            attributes = self._entity_attributes_from_handle(handle)
            if attributes:
                for key in ("object_id", "parent_object_id"):
                    object_id = _coerce_int(attributes.get(key))
                    if object_id is not None:
                        excluded_object_ids.add(object_id)

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                object_id = _coerce_int(value.get("object_id")) or _coerce_int(value.get("parent_object_id"))
                if object_id is not None and object_id not in excluded_object_ids:
                    pose0 = value.get("pose0")
                    if isinstance(pose0, list) and len(pose0) == 2:
                        dimensions = value.get("dimensions")
                        dims = [float(item) for item in dimensions[:3]] if isinstance(dimensions, list) and len(dimensions) >= 3 else None
                        tags = value.get("geometry_tags")
                        hints = value.get("affordance_hints")
                        is_support = (
                            not isinstance(tags, list)
                            or "block_like" in tags
                            or "stackable_candidate" in (hints if isinstance(hints, list) else [])
                        )
                        if is_support:
                            supports.setdefault(object_id, (deepcopy(pose0), dims))
                for item in value.values():
                    collect(item)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    collect(item)

        candidate_handles: list[str] = []
        if action is not None:
            for key in ("target", "destination", "fixture", "target_entity", "target_entities"):
                candidate_handles.extend(self._entity_handles_from_value(action.get(key)))
        for payload in self._iter_handle_graph_payloads(handles):
            candidate_handles.extend(self._entity_handles_from_value(payload))
        for handle in _dedupe_strings(candidate_handles):
            if handle in source_handles:
                continue
            current_payload = self._cliport_current_entity_payload(handle)
            if current_payload is not None:
                collect(current_payload)

        for payload in self._iter_handle_graph_payloads(handles):
            collect(payload)

        if len(supports) < 2:
            return None

        selected = list(supports.values())[:2]
        positions: list[list[float]] = []
        heights: list[float] = []
        for pose0, dims in selected:
            xyz = _pose_xyz(pose0[0])
            if xyz is None:
                return None
            positions.append([float(xyz[0]), float(xyz[1]), float(xyz[2])])
            heights.append(float(dims[2]) if dims and len(dims) >= 3 else 0.04)

        center = [
            sum(item[0] for item in positions) / len(positions),
            sum(item[1] for item in positions) / len(positions),
            max(item[2] + height + max(height * 0.25, 0.005) for item, height in zip(positions, heights)),
        ]
        quat = deepcopy(selected[0][0][1]) if isinstance(selected[0][0][1], list) and len(selected[0][0][1]) == 4 else [0.0, 0.0, 0.0, 1.0]
        return [center, quat]

    def _entity_handles_from_value(self, value: Any) -> list[str]:
        handles: list[str] = []

        def add(handle: str) -> None:
            if handle.startswith("ent:") and handle in self._handles:
                handles.append(handle)

        def walk(item: Any) -> None:
            if isinstance(item, str):
                add(item)
                for match in re.findall(r"ent:[A-Za-z0-9:_-]+", item):
                    add(match)
                return
            if isinstance(item, dict):
                for nested in item.values():
                    walk(nested)
                return
            if isinstance(item, (list, tuple, set)):
                for nested in item:
                    walk(nested)

        walk(value)
        return _dedupe_strings(handles)

    def _entity_attributes_from_handle(self, handle: str) -> JsonDict | None:
        resolved = self._resolve_entity(handle)
        if not isinstance(resolved, dict):
            return None
        attributes = resolved.get("attributes")
        return attributes if isinstance(attributes, dict) else None

    def _cliport_current_entity_payload(self, handle: str) -> JsonDict | None:
        attributes = self._entity_attributes_from_handle(handle)
        if attributes is None:
            return None
        if attributes.get("virtual_entity"):
            return deepcopy(attributes)

        object_id = _coerce_int(attributes.get("object_id")) or _coerce_int(attributes.get("parent_object_id"))
        if object_id is None:
            return deepcopy(attributes)

        result = self._call_if_available("inspect_cliport_instance", object_id=object_id)
        if result is None or not result.ok or not isinstance(result.output, dict):
            return deepcopy(attributes)

        output = result.output
        instance = output.get("instance")
        if not isinstance(instance, dict):
            public = output.get("public")
            if isinstance(public, dict):
                instance = public.get("instance")
        if not isinstance(instance, dict):
            instance = output

        payload = deepcopy(instance)
        payload.setdefault("object_id", object_id)
        for key in ("dimensions", "geometry_tags", "affordance_hints", "dominant_color_name", "dominant_rgb"):
            if key not in payload and key in attributes:
                payload[key] = deepcopy(attributes[key])
        return payload

    def _iter_handle_graph_payloads(self, handles: list[str]):
        pending = list(handles)
        seen: set[str] = set()
        while pending:
            handle = pending.pop(0)
            if handle in seen:
                continue
            seen.add(handle)
            payload = self._handles.get(handle)
            if not isinstance(payload, dict):
                continue
            yield payload
            for key in ("source_handles", "evidence_handles"):
                value = payload.get(key)
                if isinstance(value, list):
                    pending.extend(item for item in value if isinstance(item, str))
            for key in ("source_handle", "evidence_handle", "entity_handle", "handle"):
                value = payload.get(key)
                if isinstance(value, str) and value not in seen and value != handle:
                    pending.append(value)

    def _select_policy_primitive(self, policy: str) -> str | None:
        if "policy.invoke" in self.unavailable_interfaces():
            return None
        names = {card.name for card in self.backend.list_primitives()}
        policy_norm = _norm(policy)
        for preferred in self._adapter_profile().policy_primitives:
            if preferred in names and (not policy_norm or policy_norm in _norm(preferred) or len(self._adapter_profile().policy_primitives) == 1):
                return preferred
        for name in names:
            if "policy" in name and policy_norm in _norm(name):
                return name
        for preferred in ("run_robodojo_policy_skill", "execute_vlabench_skill", "execute_calvin_language_skill"):
            if preferred in names:
                return preferred
        return None

    def _policy_arguments_for_native_primitive(
        self,
        native_name: str,
        *,
        policy: str,
        inputs: JsonDict,
        evidence_handles: list[str],
        agent_context: JsonDict | None,
    ) -> JsonDict:
        arguments = deepcopy(inputs)
        policy_name = _clean_name(policy)
        arguments.setdefault("evidence_handles", list(evidence_handles))
        arguments.setdefault("evidence_refs", list(evidence_handles))
        if native_name == "execute_vlabench_skill":
            arguments["skill_name"] = _clean_name(arguments.get("skill_name") or arguments.pop("skill", None) or policy_name or "noop_step")
            target = (
                arguments.get("target_name")
                or arguments.get("target")
                or arguments.get("object")
                or arguments.get("entity")
            )
            if target is not None:
                arguments["target_name"] = self._entity_name_from_action_value(target)
            arguments["horizon"] = int(arguments.get("horizon", 10))
        elif native_name == "execute_calvin_language_skill":
            language_goal_alias = arguments.pop("language_goal", None)
            instruction_alias = arguments.pop("instruction", None)
            language_alias = arguments.pop("language", None)
            arguments["subgoal"] = _clean_name(
                arguments.get("subgoal")
                or language_goal_alias
                or instruction_alias
                or language_alias
                or policy_name
            )
            arguments["horizon"] = int(arguments.get("horizon", 1))
            arguments.pop("evidence_handles", None)
        elif native_name == "run_behavior1k_semantic_action":
            target = (
                arguments.pop("target", None)
                or arguments.pop("object", None)
                or arguments.pop("entity", None)
            )
            desired_state = arguments.pop("desired_state", arguments.pop("state", None))
            explicit_semantic_action = arguments.get(
                "semantic_action"
            ) or arguments.pop("action", None)
            semantic_action = _clean_name(explicit_semantic_action or policy_name)
            if isinstance(desired_state, bool) and (
                explicit_semantic_action is None
                or "toggle" in _norm(semantic_action)
            ):
                semantic_action = "toggle_on" if desired_state else "toggle_off"
            arguments["semantic_action"] = semantic_action
            arguments["asset_name"] = _clean_name(
                arguments.get("asset_name")
                or self._entity_name_from_action_value(target)
                or "target"
            )
            arguments["evidence_handles"] = self._native_visual_evidence_ids_from_handles(
                evidence_handles, prefixes=("behavior1k:",)
            )
            arguments.pop("evidence_refs", None)
            arguments.pop("navigate_if_needed", None)
        elif native_name == "run_robodojo_policy_skill":
            generic_policy_names = {"", "policy", "skill", "default"}
            if _norm(policy_name or "") not in generic_policy_names:
                arguments.setdefault("policy_name", policy_name)
            arguments["max_actions"] = int(arguments.get("max_actions", arguments.get("horizon", 8)))
        elif native_name.startswith("call_") and native_name.endswith("_policy_skill"):
            # The universal gateway retains evidence lineage in the returned policy
            # handle. W5 policy runtimes intentionally accept only public policy
            # inputs, so do not inject gateway-only evidence aliases into them.
            arguments.pop("evidence_handles", None)
            arguments.pop("evidence_refs", None)
        arguments["agent_context"] = agent_context or arguments.get("agent_context") or {}
        return _drop_none(arguments)

    def _call_first_available(self, names: tuple[str, ...], **kwargs: Any) -> PrimitiveResult | None:
        for name in names:
            result = self._call_if_available(name, **kwargs)
            if result is not None:
                return result
        return None

    def _call_if_available(self, name: str, **kwargs: Any) -> PrimitiveResult | None:
        if name not in {card.name for card in self.backend.list_primitives()}:
            return None
        try:
            return self.backend.call_primitive(name, **kwargs)
        except TypeError:
            cleaned = {key: value for key, value in kwargs.items() if value is not None}
            try:
                return self.backend.call_primitive(name, **cleaned)
            except TypeError:
                return None

    def _missing_evidence(self, handles: list[str]) -> list[str]:
        if not handles:
            return ["<none>"]
        return [handle for handle in handles if handle not in self._handles or not handle.startswith("ev:")]

    def _new_handle(self, prefix: str, payload: JsonDict) -> str:
        self._counter += 1
        handle = f"{prefix}:{self._benchmark_id()}:{self._counter:04d}"
        stored = deepcopy(payload)
        stored.setdefault("handle", handle)
        self._handles[handle] = stored
        self.get_trace().add_artifact(handle, _sanitize_agent_payload(stored))
        return handle

    def _new_action_handle(self, payload: JsonDict) -> str:
        self._counter += 1
        handle = f"act:{self._benchmark_id()}:{self._counter:04d}"
        stored = deepcopy(payload)
        stored["handle"] = handle
        self._actions[handle] = stored
        self.get_trace().add_artifact(handle, _sanitize_agent_payload(stored))
        return handle

    def _error(self, name: str, code: str, payload: JsonDict | None = None) -> PrimitiveResult:
        detail = {"interface": name, "code": code, **(payload or {})}
        self._last_errors.append(_sanitize_agent_payload(detail))
        return PrimitiveResult(name=name, ok=False, output={"error_code": code}, error=code, metadata=_sanitize_agent_payload(payload or {}))

    def _benchmark_id(self) -> str:
        if self._task is not None:
            raw = self._task.metadata.get("benchmark_id") or self._task.source.split(":", 1)[-1]
            return _normalize_benchmark_id(str(raw))
        return "unknown"

    def _benchmark_summary(self) -> JsonDict:
        benchmark_id = self._benchmark_id()
        match = next((item for item in BENCHMARK_FAMILY_REGISTRY if item["benchmark_id"] == benchmark_id), None)
        summary = dict(match or {"benchmark_id": benchmark_id, "family": benchmark_id, "adapter": "unknown"})
        summary["adapter_profile"] = self._adapter_profile().to_dict(include_internal_names=False)
        return summary

    def _adapter_profile(self) -> UniversalAdapterProfile:
        return universal_adapter_profile(self._benchmark_id())

    def _profile_candidates(self, category: str, fallback: tuple[str, ...] = ()) -> tuple[str, ...]:
        profile = self._adapter_profile()
        values = tuple(getattr(profile, f"{category}_primitives", ()))
        return _dedupe_tuple((*values, *fallback))

    def _available_native_names(self) -> set[str]:
        return {card.name for card in self.backend.list_primitives()}

    def _entity_name_from_action_value(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, dict):
            handle = value.get("entity_handle") or value.get("handle")
            if isinstance(handle, str):
                resolved = self._resolve_entity(handle)
                if resolved is not None:
                    return _clean_name(resolved.get("entity_id") or resolved.get("label"))
            resolved = self._resolve_entity(value)
            if resolved is not None:
                return _clean_name(resolved.get("entity_id") or resolved.get("label"))
            return _clean_name(
                value.get("entity_id") or value.get("label") or value.get("name") or value.get("id") or value.get("prompt_asset_key")
            )
        if isinstance(value, str) and value in self._handles:
            resolved = self._resolve_entity(value)
            if resolved is not None:
                direct_name = _clean_name(resolved.get("entity_id") or resolved.get("label"))
                if direct_name:
                    return direct_name
            # Agents often bind a located entity into a later evidence.record
            # handle and then use that evidence handle as the semantic action
            # source. Resolve through the evidence graph so the compiler keeps
            # the source entity instead of falling back to a one-step motion.
            for payload in self._iter_handle_graph_payloads([value]):
                if payload.get("kind") != "entity":
                    continue
                graph_name = _clean_name(payload.get("entity_id") or payload.get("label"))
                if graph_name:
                    return graph_name
        return _clean_name(value)

    def _entity_int_attribute(self, entity: str | None, *keys: str) -> int | None:
        if entity is None:
            return None
        resolved = self._resolve_entity(entity)
        if resolved is None:
            return None
        return self._int_attribute_from_payload(resolved, keys=tuple(keys))

    def _arguments_for_native_primitive(
        self,
        primitive: str,
        *,
        entity_id: str,
        resolved: JsonDict,
        query: str | None,
        agent_context: JsonDict | None,
    ) -> JsonDict:
        attrs = resolved.get("attributes") if isinstance(resolved.get("attributes"), dict) else {}
        segmentation_id = attrs.get("segmentation_id") or attrs.get("segm_id") or attrs.get("instance_id")
        object_id = attrs.get("object_id") or attrs.get("id")
        common = {"prompt": query, "query": query, "agent_context": agent_context}
        if primitive in {"locate_maniskill_actor", "grasp_maniskill_actor", "place_maniskill_actor_on"}:
            return {"actor_name": entity_id}
        if primitive == "inspect_maniskill_instance":
            coerced = _coerce_int(segmentation_id)
            return {"segmentation_id": coerced, **common} if coerced is not None else {}
        if primitive == "inspect_vima_instance":
            coerced = _coerce_int(segmentation_id)
            return {"segmentation_id": coerced, **common} if coerced is not None else {}
        if primitive == "inspect_cliport_instance":
            coerced = _coerce_int(object_id)
            return {"object_id": coerced, **common} if coerced is not None else {}
        if "robocasa" in primitive and "fixture" in primitive:
            return {"fixture_name": entity_id, **common}
        if "robocasa" in primitive and ("object" in primitive or "visual_target" in primitive):
            return {"object_name": entity_id, "entity_name": entity_id, **common}
        if primitive == "ground_rlbench_target":
            return {"target_name": entity_id, "query": query, "agent_context": agent_context}
        if primitive.startswith("capx_"):
            return {"object_name": entity_id, "query": query, "agent_context": agent_context}
        if primitive == "locate_vlabench_entity":
            return {"entity_name": entity_id, "query": query, "agent_context": agent_context}
        if primitive.startswith("inspect_behavior1k_"):
            if primitive == "inspect_behavior1k_visual_evidence":
                return {"entity_name": entity_id, "query": query, "agent_context": agent_context}
            return {"asset_name": entity_id, "query": query, "agent_context": agent_context}
        if primitive == "locate_robowits_entity":
            return {"entity_name": entity_id, "query": query, "context": agent_context or {}}
        if primitive == "locate_robotwin2_actor":
            return {"actor_name": entity_id, "query": query, "agent_context": agent_context}
        if primitive == "measure_robodojo_public_geometry":
            return {"query": entity_id or query, "agent_context": agent_context}
        if primitive in {"observe_calvin_state", "observe_robotwin2_scene", "observe_rlbench_scene"}:
            return {"query": query, "agent_context": agent_context}
        return {"query": query or entity_id, "agent_context": agent_context}

    def _public_task_spec(self, task: TaskSpec | None) -> TaskSpec:
        if task is None:
            raise RuntimeError("Call reset() before using the universal gateway.")
        metadata = _public_task_metadata(task.metadata)
        metadata["official_scoring_hidden"] = True
        return TaskSpec(
            task_id=task.task_id,
            source=task.source,
            instruction=task.instruction,
            goal=_sanitize_agent_payload(task.goal),
            initial_state=_public_initial_state_summary(task.initial_state),
            budgets=_sanitize_agent_payload(task.budgets),
            tags=list(task.tags),
            allowed_primitive_levels=["L0", "L1", "L2", "L3", "L4"],
            metadata=metadata,
        )

    def _normalize_interface(self, name: str | None) -> str | None:
        if not name:
            return None
        cleaned = str(name).strip()
        if cleaned in _SPEC_BY_NAME:
            return cleaned
        return UNIVERSAL_INTERFACE_ALIASES.get(cleaned)

    def _require_task(self) -> None:
        if self._task is None:
            raise RuntimeError("Call reset() before using the universal gateway.")


def _public_native_result(result: PrimitiveResult) -> JsonDict:
    return {
        "name": result.name,
        "ok": result.ok,
        "error": result.error,
        "output": _sanitize_agent_payload(result.output),
        "artifacts": list(result.artifacts),
    }


def _public_failure_classification(last_error: Any) -> JsonDict:
    texts = [text.lower() for text in _collect_public_failure_strings(last_error)]
    joined = "\n".join(texts)
    if not texts:
        return {
            "category": None,
            "blocker": None,
            "suggested_next_step": None,
            "public_repair_options": [],
        }
    if "policy_server_unreachable" in joined:
        blocker = _first_matching_public_text(texts, "policy_server_unreachable")
        return {
            "category": "external_policy_server_unreachable",
            "blocker": blocker,
            "suggested_next_step": "start or bind the benchmark policy websocket server, then retry action.execute",
            "public_repair_options": [
                "start the benchmark policy server on the reported host/port",
                "check same-node socket permissions for loopback websocket connections",
                "rerun inspect policy skill before action.execute",
            ],
        }
    if "translation_exceeds_agent_supplied_control_bound" in joined:
        blocker = _first_matching_public_text(
            texts, "translation_exceeds_agent_supplied_control_bound"
        )
        return {
            "category": "motion_bound_exceeded",
            "blocker": blocker,
            "suggested_next_step": (
                "prepare a closer target_position waypoint or explicitly set max_translation from the public "
                "distance evidence, never above 0.3 metres"
            ),
            "public_repair_options": [
                "use target_position rather than target for a numeric xyz waypoint",
                "split the motion into shorter public-evidence-derived waypoints",
                "set max_translation at least to the reported translation_distance and at most 0.3 metres",
            ],
        }
    if "requires_live_api_session" in joined or "requires_upstream_runner" in joined:
        blocker = _first_matching_public_text(texts, "requires_live_api_session") or _first_matching_public_text(
            texts, "requires_upstream_runner"
        )
        return {
            "category": "upstream_live_session_required",
            "blocker": blocker,
            "suggested_next_step": "run the upstream benchmark runner or inject its live API object before executing this action",
            "public_repair_options": [
                "launch the upstream benchmark runner for this case",
                "inject the live API/session object into the adapter before action.execute",
                "use build command spec or runtime-profile recheck to verify the live runner boundary",
            ],
        }
    if "live_env_step_unavailable" in joined or "requires_live_env" in joined:
        blocker = _first_matching_public_text(texts, "live_env_step_unavailable") or _first_matching_public_text(
            texts, "requires_live_env"
        )
        return {
            "category": "upstream_live_session_required",
            "blocker": blocker,
            "suggested_next_step": "run the benchmark live environment/session before executing this action",
            "public_repair_options": [
                "launch the benchmark live environment for this case",
                "inject the live env/session object into the adapter before action.execute",
                "rerun the runtime-profile recheck to verify live env step readiness",
            ],
        }
    if "api_server_unreachable" in joined or "_port_closed" in joined:
        blocker = _first_matching_public_text(texts, "api_server_unreachable") or _first_matching_public_text(texts, "_port_closed")
        return {
            "category": "benchmark_sidecar_unreachable",
            "blocker": blocker,
            "suggested_next_step": "start the configured benchmark sidecar services on their public host/port",
            "public_repair_options": [
                "start the configured sidecar services",
                "rerun the sidecar preflight for closed ports",
                "retry the live runner after sidecar readiness is true",
            ],
        }
    if (
        "visual_evidence_handles_required" in joined
        or "visual_grounding_evidence_required" in joined
        or "native_three_view_rgb_observation_required" in joined
        or "three_view_rgb_observation_required" in joined
    ):
        blocker = (
            _first_matching_public_text(texts, "visual_evidence_handles_required")
            or ("visual_grounding_evidence_required" if "visual_grounding_evidence_required" in joined else None)
            or _first_matching_public_text(texts, "native_three_view_rgb_observation_required")
            or _first_matching_public_text(texts, "three_view_rgb_observation_required")
        )
        return {
            "category": "visual_evidence_required",
            "blocker": blocker,
            "suggested_next_step": "bind benchmark-native visual evidence handles before executing this action",
            "public_repair_options": [
                "refresh scene.observe with native visual modalities available",
                "rerun entity.locate and evidence.record to bind visual provenance handles",
                "retry action.prepare after evidence handles include native visual evidence",
            ],
        }
    if (
        "official_mcil_backend_not_ready" in joined
        or "policy_runtime_contract" in joined
        or "policy_backend_not_configured" in joined
        or "runtime_dependency" in joined
        or "runtime_missing" in joined
        or "incompatible_versions" in joined
        or "required_min_versions" in joined
        or "version:torch" in joined
    ):
        blocker = (
            ("policy_backend_not_configured" if "policy_backend_not_configured" in joined else None)
            or _first_matching_public_text(texts, "official_mcil_backend_not_ready")
            or _first_matching_public_text(texts, "version:")
            or _first_matching_public_text(texts, "policy_runtime_contract")
            or _first_matching_public_text(texts, "runtime_dependency")
        )
        return {
            "category": "runtime_dependency_missing",
            "blocker": blocker,
            "suggested_next_step": "select or repair the benchmark policy/runtime Python environment before retrying action.execute",
            "public_repair_options": [
                "rerun runtime profile detection for this benchmark",
                "install or select the required policy runtime dependencies",
                "retry the same high-level action-chain after policy runtime readiness is true",
            ],
        }
    if "no nvidia driver" in joined or "cuda" in joined or "failed to find a supported physical device" in joined:
        return {
            "category": "gpu_or_render_device_unavailable",
            "blocker": _first_matching_public_text(texts, "nvidia") or _first_matching_public_text(texts, "cuda"),
            "suggested_next_step": "rerun this runtime profile on a node with the required GPU/render device",
            "public_repair_options": [
                "select a GPU/render-capable node for this benchmark",
                "rerun the runtime profile recheck on that node",
                "keep interface/action schema unchanged until simulator startup succeeds",
            ],
        }
    if "egl" in joined or "osmesa" in joined or "xvfb" in joined or "display" in joined:
        return {
            "category": "headless_display_unavailable",
            "blocker": _first_matching_public_text(texts, "egl") or _first_matching_public_text(texts, "display"),
            "suggested_next_step": "rerun with a working headless display/render backend",
            "public_repair_options": [
                "verify EGL/OSMesa/Xvfb readiness for this runtime profile",
                "rerun the case on a node with the required display socket or render backend",
                "do not change task planning until native observation can start",
            ],
        }
    if "no module named" in joined or "missing_python_modules" in joined:
        return {
            "category": "runtime_dependency_missing",
            "blocker": _first_matching_public_text(texts, "no module named") or _first_matching_public_text(texts, "missing_python_modules"),
            "suggested_next_step": "select or repair the benchmark runtime Python environment",
            "public_repair_options": [
                "rerun runtime profile detection for this benchmark",
                "install the missing package in the benchmark-specific environment",
                "avoid changing the universal interface schema for dependency-only blockers",
            ],
        }
    return {
        "category": "interface_or_backend_call_failed",
        "blocker": _first_public_error_text(texts),
        "suggested_next_step": "refresh public observation/evidence and retry a smaller grounded action",
        "public_repair_options": [],
    }


def _collect_public_failure_strings(value: Any) -> list[str]:
    strings: list[str] = []
    if isinstance(value, str):
        if value.strip():
            strings.append(value.strip())
        return strings
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text in {"error", "code", "blocker"} and isinstance(item, str) and item.strip():
                strings.append(item.strip())
            elif key_text.endswith("blockers") and isinstance(item, list):
                strings.extend(str(entry).strip() for entry in item if str(entry).strip())
            else:
                strings.extend(_collect_public_failure_strings(item))
        return strings
    if isinstance(value, list):
        for item in value:
            strings.extend(_collect_public_failure_strings(item))
    return strings


def _first_matching_public_text(texts: list[str], needle: str) -> str | None:
    needle = needle.lower()
    for text in texts:
        if needle in text:
            return text
    return None


def _first_public_error_text(texts: list[str]) -> str | None:
    return texts[0] if texts else None


def _hidden_native_primitive_names() -> tuple[str, ...]:
    names: set[str] = set()
    for profile in OPENHANDS_ADAPTER_PROFILES.values():
        for field_name in (
            "context_primitives",
            "observe_primitives",
            "enumerate_primitives",
            "inspect_primitives",
            "locate_primitives",
            "geometry_primitives",
            "affordance_primitives",
            "evidence_primitives",
            "prepare_primitives",
            "execute_primitives",
            "policy_primitives",
        ):
            names.update(str(name) for name in getattr(profile, field_name))
    return tuple(sorted(names, key=len, reverse=True))


def _sanitize_public_text(text: str) -> str:
    sanitized = text
    sanitized = re.sub(
        r"visual_grounding_evidence_required:.*",
        (
            "visual_grounding_evidence_required: native visual evidence is unavailable or not bound; "
            "refresh scene.observe, entity.locate, and evidence.record before action.prepare/action.execute"
        ),
        sanitized,
        flags=re.IGNORECASE,
    )
    sanitized = re.sub(
        r"No VLABench policy-as-skill backend is configured[^.]*\.",
        "policy_backend_not_configured: selected policy skill is unavailable in the current benchmark runtime.",
        sanitized,
        flags=re.IGNORECASE,
    )
    for primitive_name in _hidden_native_primitive_names():
        sanitized = re.sub(
            rf"(?<![A-Za-z0-9_]){re.escape(primitive_name)}(?![A-Za-z0-9_])",
            "<hidden_native_step>",
            sanitized,
        )
    return sanitized


def _public_adapter_probe_result(result: PrimitiveResult) -> JsonDict:
    return {
        "ok": result.ok,
        "error": result.error,
        "output": _sanitize_agent_payload(result.output),
        "artifact_count": len(result.artifacts),
        "native_details_hidden": True,
    }


def _public_compiled_action(compiled: JsonDict) -> JsonDict:
    return _sanitize_agent_payload(
        {
            "benchmark_id": compiled.get("benchmark_id"),
            "semantic_action": compiled.get("semantic_action", {}),
            "evidence_handles": list(compiled.get("evidence_handles", [])),
            "prepared_step_count": len(compiled.get("native_steps", [])),
            "native_details_hidden": True,
            "native_details_disclosed_to_agent": False,
            "agent_context": compiled.get("agent_context", {}),
        }
    )


def _resolve_step_arguments(value: Any, previous_results: list[PrimitiveResult]) -> Any:
    if isinstance(value, dict):
        if "$from_previous" in value:
            if not previous_results:
                return None
            path = value["$from_previous"]
            return _value_at_path(previous_results[-1].output, path if isinstance(path, list) else [path])
        if "$from_step" in value:
            index = int(value.get("$from_step", -1))
            path = value.get("path", [])
            if not previous_results:
                return None
            try:
                source = previous_results[index].output
            except IndexError:
                return None
            return _value_at_path(source, path if isinstance(path, list) else [path])
        return {key: _resolve_step_arguments(item, previous_results) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_step_arguments(item, previous_results) for item in value]
    return value


def _value_at_path(value: Any, path: list[Any]) -> Any:
    current = value
    for item in path:
        if isinstance(current, dict):
            current = current.get(item)
        elif isinstance(current, list) and isinstance(item, int):
            try:
                current = current[item]
            except IndexError:
                return None
        else:
            return None
    return current


def _drop_none(value: JsonDict) -> JsonDict:
    return {key: item for key, item in value.items() if item is not None}


def _coerce_handle_list(*values: Any) -> list[str]:
    handles: list[str] = []

    def visit(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                handles.append(cleaned)
            return
        if isinstance(value, dict):
            for key in ("handle", "source_handle", "evidence_handle", "entity_handle", "observation_handle"):
                visit(value.get(key))
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                visit(item)

    for value in values:
        visit(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for handle in handles:
        if handle in seen:
            continue
        seen.add(handle)
        deduped.append(handle)
    return deduped


def _dedupe_strings(values: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def _pose_xyz(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in ("target_position", "position", "xyz", "pose", "pose_world", "world_target", "xpos", "center"):
            xyz = _pose_xyz(value.get(key))
            if xyz is not None:
                return xyz
        for key in ("bbox_world",):
            xyz = _pose_xyz(value.get(key))
            if xyz is not None:
                return xyz
        return None
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None


def _capx_quaternion_wxyz(value: Any) -> list[float] | None:
    if isinstance(value, dict):
        for key in (
            "quaternion_wxyz",
            "target_quat",
            "quaternion",
            "quat",
            "orientation",
            "rotation",
            "pose",
            "pose_world",
            "target_pose",
            "grasp_pose",
            "attributes",
            "entity",
            "location",
            "value",
        ):
            quaternion = _capx_quaternion_wxyz(value.get(key))
            if quaternion is not None:
                return quaternion
        return None
    if isinstance(value, (list, tuple)):
        try:
            if len(value) >= 7:
                return [float(item) for item in value[3:7]]
            if len(value) == 4:
                return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _coerce_pose_list(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    return None


def _coerce_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _gripper_value(action: JsonDict, *, closed_default: bool = False) -> float:
    raw = action.get("gripper")
    if isinstance(raw, str):
        normalized = _norm(raw)
        if normalized in {"close", "closed", "grasp"}:
            return 1.0
        if normalized in {"open", "release"}:
            return -1.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if action.get("close_gripper") is True or closed_default:
        return 1.0
    return -1.0


def _maniskill_gripper_value(action: JsonDict, *, closed_default: bool = False) -> float:
    """Map semantic gripper state to ManiSkill Panda's normalized controller."""
    raw = action.get("gripper")
    if isinstance(raw, str):
        normalized = _norm(raw)
        if normalized in {"close", "closed", "grasp"}:
            return -1.0
        if normalized in {"open", "release"}:
            return 1.0
    if isinstance(raw, (int, float)):
        return float(raw)
    if action.get("close_gripper") is True or closed_default:
        return -1.0
    return 1.0


_LARGE_AGENT_ARRAY_ELEMENT_LIMIT = 4096
_VISUAL_ARRAY_PATH_FRAGMENTS = (
    "cam_",
    "camera",
    "color",
    "depth",
    "frame",
    "image",
    "mask",
    "pixel",
    "point_cloud",
    "pointcloud",
    "rgb",
    "segmentation",
    "visual",
    "vision",
)


def _large_visual_array_summary(value: Any, path: tuple[str, ...]) -> JsonDict | None:
    if not any(
        fragment in component.lower()
        for component in path
        for fragment in _VISUAL_ARRAY_PATH_FRAGMENTS
    ):
        return None
    shape = getattr(value, "shape", None)
    size = getattr(value, "size", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and size is not None:
        try:
            dimensions = [int(dimension) for dimension in shape]
            element_count = int(size)
        except (TypeError, ValueError):
            return None
    else:
        dimensions = _nested_sequence_shape(value)
        if not dimensions:
            return None
        element_count = math.prod(dimensions)
    if element_count <= _LARGE_AGENT_ARRAY_ELEMENT_LIMIT:
        return None
    return {
        "array_content_omitted": True,
        "shape": dimensions,
        "dtype": str(dtype or _nested_sequence_scalar_type(value)),
        "element_count": element_count,
    }


def _nested_sequence_shape(value: Any) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    dimensions: list[int] = []
    current: Any = value
    while isinstance(current, (list, tuple)):
        if not current:
            return None
        dimensions.append(len(current))
        expected_child_length = (
            len(current[0]) if isinstance(current[0], (list, tuple)) else None
        )
        sample_indexes = {0, len(current) // 2, len(current) - 1}
        for index in sample_indexes:
            child = current[index]
            child_length = len(child) if isinstance(child, (list, tuple)) else None
            if child_length != expected_child_length:
                return None
        current = current[0]
    return dimensions


def _nested_sequence_scalar_type(value: Any) -> str:
    current = value
    while isinstance(current, (list, tuple)) and current:
        current = current[0]
    return type(current).__name__


def _sanitize_agent_payload(value: Any, _path: tuple[str, ...] = ()) -> Any:
    array_summary = _large_visual_array_summary(value, _path)
    if array_summary is not None:
        return array_summary
    if isinstance(value, dict):
        sanitized: JsonDict = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_forbidden_field(key_text):
                continue
            sanitized[key_text] = _sanitize_agent_payload(item, (*_path, key_text))
        return sanitized
    if isinstance(value, list):
        return [_sanitize_agent_payload(item, _path) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_agent_payload(item, _path) for item in value]
    if hasattr(value, "tolist"):
        return _sanitize_agent_payload(value.tolist(), _path)
    if hasattr(value, "item"):
        try:
            return _sanitize_agent_payload(value.item(), _path)
        except (TypeError, ValueError):
            pass
    if isinstance(value, str):
        return _sanitize_public_text(value)
    return value


def _public_failure_related_handle(value: JsonDict) -> JsonDict:
    """Summarize a retained handle without exposing compiled or low-level actions."""

    public: JsonDict = {
        "kind": _sanitize_agent_payload(value.get("kind") or "handle"),
        "implementation_details_hidden": True,
    }
    for key in (
        "action_handle",
        "entity_handle",
        "evidence_handle",
        "observation_handle",
        "policy_handle",
        "source_handle",
    ):
        if value.get(key) is not None:
            public[key] = _sanitize_agent_payload(value[key])
    for key in ("evidence_handles", "source_handles"):
        if isinstance(value.get(key), list):
            public[key] = _sanitize_agent_payload(value[key])
    implementation_results = value.get("native_results")
    if isinstance(implementation_results, list):
        public["executed_step_count"] = len(implementation_results)
        public["all_steps_accepted"] = all(
            isinstance(result, dict) and result.get("ok") is True
            for result in implementation_results
        )
        errors = [
            _sanitize_agent_payload(result.get("error"))
            for result in implementation_results
            if isinstance(result, dict) and result.get("error")
        ]
        if errors:
            public["public_step_errors"] = errors
    return public


def _public_task_metadata(metadata: JsonDict) -> JsonDict:
    allowed = {
        "benchmark_id",
        "task_name",
        "task_id",
        "env_id",
        "sequence_id",
        "mode",
        "partition",
        "suite",
    }
    public: JsonDict = {}
    for key in allowed:
        if key in metadata:
            public[key] = _sanitize_agent_payload(metadata[key])
    benchmark_id = metadata.get("benchmark_id")
    if benchmark_id is not None:
        public["benchmark_id"] = _normalize_benchmark_id(str(benchmark_id))
    return public


def _public_initial_state_summary(initial_state: JsonDict) -> JsonDict:
    if not isinstance(initial_state, dict):
        return {"state_available": bool(initial_state)}
    keys = sorted(str(key) for key in initial_state if not _is_low_level_context_key(str(key)))
    summary: JsonDict = {
        "state_available": bool(initial_state),
        "state_keys": keys,
        "low_level_state_hidden": True,
    }
    for field_name in ("objects", "object_poses", "entities", "instances"):
        value = initial_state.get(field_name)
        if isinstance(value, dict):
            summary[f"{field_name}_count"] = len(value)
        elif isinstance(value, list):
            summary[f"{field_name}_count"] = len(value)
    return summary


def _sanitize_public_observation_payload(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: JsonDict = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_forbidden_field(key_text) or _is_low_level_observation_key(key_text):
                continue
            sanitized[key_text] = _sanitize_public_observation_payload(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_public_observation_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_public_observation_payload(item) for item in value]
    if isinstance(value, str):
        return _sanitize_public_text(value)
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            normalized_shape = [int(item) for item in shape]
        except Exception:
            normalized_shape = [str(item) for item in shape]
        if len(normalized_shape) <= 1 and hasattr(value, "tolist"):
            try:
                listed = value.tolist()
                if not isinstance(listed, list) or len(listed) <= 64:
                    return _sanitize_public_observation_payload(listed)
            except Exception:
                pass
        return {
            "shape": normalized_shape,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
            "native_array_values_hidden": True,
            "raw_artifact_evidence_required": True,
        }
    return value


def _looks_like_robodojo_scene_observation(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(
        key in value
        for key in (
            "public_runtime_state",
            "public_runtime_state_after_action",
            "observation",
            "visual_observation_contract",
            "rgb_summaries",
            "raw_frame_refs",
            "evidence_refs",
        )
    ):
        return True
    return "vision" in value or "vision_summary" in value


def _is_low_level_observation_key(key: str) -> bool:
    key_norm = key.lower()
    blocked = {
        "available_apis",
        "api_inventory",
        "api_trace",
        "config",
        "debug_command",
        "install_command",
        "live_smoke_command",
        "native_primitive",
        "primitive_cards",
        "raw_api",
    }
    return key_norm in blocked or key_norm.endswith("_api") or key_norm.endswith("_apis")


def _is_low_level_context_key(key: str) -> bool:
    key_norm = key.lower()
    blocked_tokens = {
        "api",
        "command",
        "config",
        "controller",
        "demo",
        "expert",
        "oracle",
        "path",
        "primitive",
        "recipe",
        "reward",
        "success",
        "verifier",
        "waypoint",
    }
    return any(token in key_norm for token in blocked_tokens)


def _is_forbidden_field(key: str) -> bool:
    key_norm = key.lower()
    return any(token in key_norm for token in FORBIDDEN_AGENT_FIELDS)


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _normalize_benchmark_id(value: str) -> str:
    normalized = str(value).lower().replace("-", "_")
    aliases = {
        "mmsi": "mmsi_bench",
        "mmsi_bench": "mmsi_bench",
        "vima": "vimabench",
        "vimabench": "vimabench",
        "vima_bench": "vimabench",
        "behaviour1k": "behavior1k",
        "behavior_1k": "behavior1k",
        "behavior1k": "behavior1k",
        "robocasa_365": "robocasa365",
        "robo_casa365": "robocasa365",
        "robotwin_2": "robotwin2",
        "robo_twin2": "robotwin2",
        "robodojo": "robodojo",
        "robo_dojo": "robodojo",
        "esi_spatial": "esi",
        "vsi": "esi",
        "vsi_bench": "esi",
    }
    return aliases.get(normalized, normalized.replace("_", ""))


def _clean_name(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_public_language_subgoal(payload: JsonDict) -> str | None:
    for key in ("language_subgoal", "subgoal", "instruction", "language"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _normalize_action_payload(action: JsonDict) -> JsonDict:
    normalized = deepcopy(action)

    def semantic_action_alias(value: Any) -> Any:
        if value is None:
            return None
        action_norm = _norm(value)
        aliases = {
            "pick_and_place": "pick_place",
            "pick_place": "pick_place",
            "pick_n_place": "pick_place",
            "place_entity_in": "place",
            "place_entity": "place",
            "put_entity_in": "place",
            "put_object_in": "place",
            "put": "place",
            "grasp_entity": "grasp",
            "pick_entity": "grasp",
            "pick_object": "grasp",
            "pick_up": "grasp",
            "motion_call": "move",
            "move_ee_to_entity": "move",
            "hover_above_entity": "move",
        }
        return aliases.get(action_norm, value)

    action_type = (
        normalized.get("action_type")
        or normalized.get("actionType")
        or normalized.get("semantic_action")
    )
    if action_type is not None:
        normalized.setdefault("type", semantic_action_alias(action_type))
    if normalized.get("type") is not None:
        normalized["type"] = semantic_action_alias(normalized.get("type"))
    elif normalized.get("verb") is not None:
        normalized["type"] = semantic_action_alias(normalized.get("verb"))
    elif isinstance(normalized.get("action"), str):
        normalized["type"] = semantic_action_alias(normalized.get("action"))
    top_level_skill_name = _clean_name(
        normalized.get("skill_name") or normalized.get("skillName") or normalized.get("skill")
    )
    if top_level_skill_name is not None:
        normalized.setdefault("skill", top_level_skill_name)
        if _norm(normalized.get("type") or "") in {"", "skill_call"}:
            normalized["type"] = semantic_action_alias(top_level_skill_name)
    action_agent_context = normalized.get("agent_context") if isinstance(normalized.get("agent_context"), dict) else {}
    source_entity = (
        normalized.get("source_entity")
        or normalized.get("sourceEntity")
        or normalized.get("source_entity_name")
        or normalized.get("sourceObject")
        or normalized.get("source_object")
        or normalized.get("object_name")
        or action_agent_context.get("source_entity")
        or action_agent_context.get("source_object")
        or action_agent_context.get("object_name")
    )
    if source_entity is not None:
        normalized.setdefault("source", source_entity)
        normalized.setdefault("object", source_entity)
    entity_name = normalized.get("entity_name") or normalized.get("entityName")
    if entity_name is not None:
        normalized.setdefault("entity", entity_name)
        normalized.setdefault("object", entity_name)
    target_entity = (
        normalized.get("target_entity")
        or normalized.get("targetEntity")
        or normalized.get("target_entity_name")
        or normalized.get("targetEntityName")
        or normalized.get("target_container")
        or normalized.get("target_container_name")
        or normalized.get("targetContainerName")
        or normalized.get("target_name")
        or normalized.get("targetName")
        or normalized.get("container_name")
        or normalized.get("container")
        or action_agent_context.get("target_entity")
        or action_agent_context.get("target_container")
        or action_agent_context.get("container_name")
        or action_agent_context.get("container")
    )
    if target_entity is not None:
        normalized.setdefault("target", target_entity)
        normalized.setdefault("destination", target_entity)

    motion_call = normalized.get("motion_call")
    if isinstance(motion_call, dict):
        for key, value in motion_call.items():
            normalized.setdefault(str(key), value)
        normalized.setdefault("type", motion_call.get("type") or motion_call.get("verb") or "move")
        target_name = motion_call.get("target_name")
        if target_name is not None:
            normalized.setdefault("target", target_name)
        target_position = motion_call.get("target_position")
        if target_position is not None:
            normalized.setdefault("target_position", target_position)
        if motion_call.get("target_site") == "place":
            normalized.setdefault("type", "place")

    skill_call = normalized.get("skill_call")
    if isinstance(skill_call, dict):
        for key, value in skill_call.items():
            normalized.setdefault(str(key), value)
        skill_name = _clean_name(skill_call.get("skill_name") or skill_call.get("skill"))
        if skill_name is not None:
            normalized.setdefault("skill", skill_name)
        target_name = skill_call.get("target_name")
        if target_name is not None:
            normalized.setdefault("target", target_name)
        if isinstance(skill_call.get("agent_context"), dict):
            normalized.setdefault("agent_context", skill_call.get("agent_context"))
        skill_norm = _norm(skill_name or "")
        if skill_norm in {"grasp_entity", "pick_entity", "pick", "grasp"}:
            normalized.setdefault("type", "grasp")
            if target_name is not None:
                normalized.setdefault("object", target_name)
        elif skill_norm in {"place_entity_in", "place_entity", "place", "put"}:
            normalized.setdefault("type", "place")
        elif skill_norm in {"open_gripper", "open"}:
            normalized.setdefault("type", "open")
        elif skill_norm in {"close_gripper", "close"}:
            normalized.setdefault("type", "close")
        elif skill_norm in {"settle_scene", "settle"}:
            normalized.setdefault("type", "settle")
    return normalized


def _looks_like_pose(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) in {2, 3, 4, 6, 7, 10}
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def _looks_like_entity_attrs(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    entity_keys = {
        "bbox",
        "bbox_xyxy",
        "bounding_box",
        "caller_selection",
        "entity_id",
        "id",
        "instance_id",
        "label",
        "mask",
        "name",
        "object_id",
        "object_name",
        "observation_path",
        "pose",
        "pose_world",
        "position",
        "prompt_asset_key",
        "segmentation_id",
        "target_name",
        "world_target",
        "xpos",
    }
    return any(key in value for key in entity_keys)


def _entity_matches_selector(entity: JsonDict, selector: JsonDict) -> bool:
    attrs = entity.get("attributes") if isinstance(entity.get("attributes"), dict) else {}
    for key, expected in selector.items():
        if key in {"entity_handle", "handle", "attributes"} or expected is None:
            continue
        candidate = entity.get(key)
        if candidate is None and isinstance(attrs, dict):
            candidate = attrs.get(key)
        if candidate is None:
            return False
        expected_int = _coerce_int(expected)
        candidate_int = _coerce_int(candidate)
        if expected_int is not None or candidate_int is not None:
            if expected_int != candidate_int:
                return False
            continue
        if _norm(str(candidate)) != _norm(str(expected)):
            return False
    return True


def _dedupe_entities(entities: list[JsonDict]) -> list[JsonDict]:
    seen: set[str] = set()
    deduped: list[JsonDict] = []
    for entity in entities:
        key = _norm(entity.get("entity_id") or entity.get("label") or entity.get("entity_handle"))
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
    return deduped


def _dedupe_tuple(values: tuple[str, ...]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            deduped.append(value)
    return tuple(deduped)


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _coerce_first_int(value: Any) -> int | None:
    coerced = _coerce_int(value)
    if coerced is not None:
        return coerced
    if isinstance(value, (list, tuple)):
        for item in value:
            coerced = _coerce_first_int(item)
            if coerced is not None:
                return coerced
    if isinstance(value, dict):
        for key in ("segmentation_id", "segm_id", "instance_id", "id"):
            if key in value:
                coerced = _coerce_first_int(value.get(key))
                if coerced is not None:
                    return coerced
    return None


def _extract_pose(payload: JsonDict) -> list[float] | None:
    for key in ("pose", "pose_world", "position"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return list(value)
    attrs = payload.get("attributes")
    if isinstance(attrs, dict):
        return _extract_pose(attrs)
    location = payload.get("location")
    if isinstance(location, dict):
        return _extract_pose(location)
    return None


def _deep_merge(left: JsonDict, right: JsonDict) -> JsonDict:
    result = deepcopy(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def assert_universal_contract() -> None:
    names = [spec.name for spec in UNIVERSAL_INTERFACE_SPECS]
    if tuple(names) != UNIVERSAL_INTERFACE_NAMES:
        raise AssertionError("Universal interface registry order does not match canonical names.")
    if set(_UNIVERSAL_INTERFACE_HANDLE_IO) != set(UNIVERSAL_INTERFACE_NAMES):
        missing = sorted(set(UNIVERSAL_INTERFACE_NAMES) - set(_UNIVERSAL_INTERFACE_HANDLE_IO))
        extra = sorted(set(_UNIVERSAL_INTERFACE_HANDLE_IO) - set(UNIVERSAL_INTERFACE_NAMES))
        raise AssertionError(f"Handle IO contract must match interface names; missing={missing}, extra={extra}.")
    evidence_required = {
        spec.name for spec in UNIVERSAL_INTERFACE_SPECS if spec.requires_evidence
    }
    contract_required = set(UNIVERSAL_EVIDENCE_CONTRACT["evidence_required_interfaces"])
    if evidence_required != contract_required:
        raise AssertionError(
            "Evidence-required interface contract must match interface specs; "
            f"spec={sorted(evidence_required)}, contract={sorted(contract_required)}."
        )
    handle_producers = {
        interface
        for handle_contract in UNIVERSAL_HANDLE_CONTRACT.values()
        for interface in handle_contract["producer_interfaces"]
    }
    handle_consumers = {
        interface
        for handle_contract in UNIVERSAL_HANDLE_CONTRACT.values()
        for interface in handle_contract["consumer_interfaces"]
    }
    unknown_handle_interfaces = (handle_producers | handle_consumers) - set(UNIVERSAL_INTERFACE_NAMES)
    if unknown_handle_interfaces:
        raise AssertionError(f"Handle contract references unknown interfaces: {sorted(unknown_handle_interfaces)}.")
    schema_text = repr([spec.input_schema for spec in UNIVERSAL_INTERFACE_SPECS] + [spec.output_schema for spec in UNIVERSAL_INTERFACE_SPECS]).lower()
    leaked_fields = [field for field in FORBIDDEN_AGENT_FIELDS if field in schema_text]
    if leaked_fields:
        raise AssertionError(f"Universal interface schema leaks forbidden fields: {sorted(leaked_fields)}.")
    if len(BENCHMARK_FAMILY_REGISTRY) != 13:
        raise AssertionError("Benchmark family registry must contain exactly 13 active families.")
    if len(set(item["benchmark_id"] for item in BENCHMARK_FAMILY_REGISTRY)) != 13:
        raise AssertionError("Benchmark family ids must be unique.")
    registry_ids = {item["benchmark_id"] for item in BENCHMARK_FAMILY_REGISTRY}
    operation_profile_ids = set(UNIVERSAL_ADAPTER_PROFILES)
    if operation_profile_ids != registry_ids:
        missing = sorted(registry_ids - operation_profile_ids)
        extra = sorted(operation_profile_ids - registry_ids)
        raise AssertionError(f"Adapter profiles must match benchmark registry; missing={missing}, extra={extra}.")
    catalog_ids = {item.benchmark_id for item in BENCHMARK_CATALOG}
    openhands_profile_ids = set(OPENHANDS_ADAPTER_PROFILES)
    if openhands_profile_ids != catalog_ids:
        missing = sorted(catalog_ids - openhands_profile_ids)
        extra = sorted(openhands_profile_ids - catalog_ids)
        raise AssertionError(
            f"OpenHands adapter profiles must match the canonical catalog; missing={missing}, extra={extra}."
        )


assert_universal_contract()
