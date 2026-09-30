from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import asdict, dataclass, field
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable

import numpy as np

from .api_agent_runner import (
    APIModelConfig,
    ChatClient,
    OpenAICompatibleChatClient,
    extract_python_code,
    load_api_model_config,
)
from .alfworld_agent_runtime import ALFWorldAgentRuntimeBackend, ALFWorldRuntimeConfig
from .backend import EmbodiedBackend
from .behavior1k_agent_runtime import (
    Behavior1KAgentRuntimeBackend,
    Behavior1KRuntimeConfig,
    build_behavior1k_full_bddl_env_config,
)
from .calvin_agent_runtime import CALVINAgentRuntimeBackend, CALVINRuntimeConfig
from .vlabench_agent_runtime import VLABenchSameEpisodeOfficialBinding
from .capx_reset_asset import CAPX_RESET_ASSET
from .capx_comparator_runtime import (
    CapXLiveSession,
    _capx_observation_modalities,
    _public_capx_observation,
    build_context as build_capx_context,
    call_capx_comparator_primitive,
    list_capx_comparator_primitives,
    open_capx_live_session,
)
from .cliport_agent_runtime import CLIPortAgentRuntimeBackend, CLIPortRuntimeConfig
from .maniskill_agent_runtime import (
    ManiSkillAgentRuntimeBackend,
    ManiSkillRuntimeConfig,
)
from .paths import get_project_paths, resolve_project_root
from .policy_skill_agent_runtime import (
    PolicySkillAgentRuntimeBackend,
    PolicySkillRuntimeConfig,
)
from .rlbench_agent_runtime import RLBenchAgentRuntimeBackend, RLBenchRuntimeConfig
from .robodojo_agent_runtime import RoboDojoAgentRuntimeBackend, RoboDojoRuntimeConfig
from .robotwin2_agent_runtime import (
    RoboTwin2AgentRuntimeBackend,
    RoboTwin2RuntimeConfig,
)
from .robowits_agent_runtime import RoboWitsAgentRuntimeBackend, RoboWitsRuntimeConfig
from .robocasa_agent_runtime import (
    RoboCasaAgentRuntimeBackend,
    RoboCasaRuntimeConfig,
    _sample_action_spec as _sample_robocasa_action_spec,
    _split_step_result as _split_robocasa_step_result,
    _to_builtin as _robocasa_to_builtin,
    summarize_observation as summarize_robocasa_observation,
)
from .robocasa365_agent_runtime import (
    RoboCasa365AgentRuntimeBackend,
    RoboCasa365RuntimeConfig,
)
from .runner import ExecutionResult, StatefulCodeRunner
from .scienceworld_agent_runtime import (
    ScienceWorldAgentRuntimeBackend,
    ScienceWorldRuntimeConfig,
)
from .schemas import (
    EpisodeTrace,
    Observation,
    PrimitiveCard,
    PrimitiveResult,
    TaskSpec,
    VerificationResult,
)
from .spatial_backend import SpatialDiagnosticBackend, SpatialSample
from .spatialclaw_adapter import SpatialClawAgentNativeBackend, load_spatialclaw_samples
from .spatial_mmsi_adapter import MMSISpatialBackend
from .vimabench_agent_runtime import (
    VimaBenchAgentRuntimeBackend,
    VimaBenchRuntimeConfig,
)
from .w4_benchmark_backend import W4BenchmarkBackend


BackendFactory = Callable[[], EmbodiedBackend]
QWEN_PROMPT_CONTRACT = "current-interface-qwen-prompt-v2"


_AGENT_PRIVATE_KEY_TOKENS = {
    "answer_key",
    "checker",
    "demo",
    "expert",
    "gold",
    "ground_truth",
    "groundtruth",
    "oracle",
    "replay",
    "reward",
    "success_label",
    "success_source",
    "task_progress",
    "verifier",
}
_AGENT_PRIVATE_EXACT_KEYS = {
    "_check_success",
    "is_success",
    "official_success",
    "success",
    "task_success",
}
_AGENT_LOCAL_PATH_TOKENS = {
    "asset_cache_dir",
    "cache_dir",
    "dataset_path",
    "dataset_root",
    "local_path",
    "repo_path",
    "repo_root",
    "report_path",
    "source_path",
}
_AGENT_LOCAL_PATH_SUFFIXES = ("_dir", "_directory", "_file", "_path", "_root")
_AGENT_LOCAL_PATH_PREFIXES = ("/home/", "/hpc", "/mnt/", "/opt/", "/tmp/", "file://")
_AGENT_VISIBLE_SEQUENCE_LIMIT = 64


@dataclass(slots=True)
class LiveAgentSmokeCase:
    case_id: str
    benchmark_id: str
    task_id: str
    objective: str
    required_primitives: list[str]
    required_ok_primitives: list[str]
    backend_factory: BackendFactory
    seed: int | None = None
    reset_config: dict[str, Any] = field(default_factory=dict)
    action_hint: str = ""
    readiness_tier: str = "boundary_ready"
    counts_toward_official_success: bool = False
    code_timeout_seconds: float | None = None
    harness_only_verifier: bool = False


@dataclass(slots=True)
class LiveAgentSmokeResult:
    case_id: str
    benchmark_id: str
    model: str
    prompt_contract: str
    interface_fingerprint: str
    readiness_tier: str
    counts_toward_official_success: bool
    harness_only_verifier: bool
    agent_smoke_success: bool
    benchmark_task_success: bool
    official_gate_success: bool
    execution_ok: bool
    called_primitives: list[str]
    missing_required_primitives: list[str]
    failed_required_primitives: list[str]
    code: str
    execution_error: str | None = None
    verifier_message: str = ""
    trace_event_count: int = 0
    artifact_count: int = 0
    primitive_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _agent_visible_task(task: dict[str, Any]) -> dict[str, Any]:
    """Expose only task identity and instruction, never evaluator metadata."""

    return {
        key: _agent_visible_value(task[key])
        for key in ("task_id", "source", "instruction", "tags")
        if key in task
    }


def _agent_visible_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Keep public sensor/state data while removing private gates and local paths."""

    visible: dict[str, Any] = {}
    for key in ("step", "data", "metadata"):
        if key in observation:
            visible[key] = _agent_visible_value(observation[key])
    artifacts = observation.get("artifacts")
    if isinstance(artifacts, list):
        visible["artifact_count"] = len(artifacts)
    return visible


def _agent_visible_value(value: Any) -> Any:
    if isinstance(value, dict):
        visible: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            key = str(raw_key)
            normalized = key.strip().lower()
            if normalized in _AGENT_PRIVATE_EXACT_KEYS:
                continue
            if any(token in normalized for token in _AGENT_PRIVATE_KEY_TOKENS):
                continue
            if any(token in normalized for token in _AGENT_LOCAL_PATH_TOKENS):
                continue
            if normalized.endswith(_AGENT_LOCAL_PATH_SUFFIXES):
                continue
            visible[key] = _agent_visible_value(raw_value)
        return visible
    if isinstance(value, (list, tuple)):
        items = [
            _agent_visible_value(item) for item in value[:_AGENT_VISIBLE_SEQUENCE_LIMIT]
        ]
        if len(value) > _AGENT_VISIBLE_SEQUENCE_LIMIT:
            items.append(
                {"truncated_item_count": len(value) - _AGENT_VISIBLE_SEQUENCE_LIMIT}
            )
        return items
    if isinstance(value, str):
        lowered = value.strip().lower()
        if any(prefix in lowered for prefix in _AGENT_LOCAL_PATH_PREFIXES):
            return "<local-path-redacted>"
    return value


def _interface_fingerprint(case: LiveAgentSmokeCase, cards: list[PrimitiveCard]) -> str:
    payload = {
        "benchmark_id": case.benchmark_id,
        "case_id": case.case_id,
        "harness_only_verifier": case.harness_only_verifier,
        "primitive_cards": [card.to_dict() for card in cards],
        "prompt_contract": QWEN_PROMPT_CONTRACT,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _agent_code_contract_error(
    code: str,
    cards: list[PrimitiveCard],
    *,
    harness_only_verifier: bool,
) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"agent_code_syntax_error: {exc}"

    if harness_only_verifier and any(
        isinstance(node, ast.Name) and node.id == "backend" for node in ast.walk(tree)
    ):
        return (
            "agent_code_contract_error: backend is not available in a harness-only case"
        )

    primitive_names = {card.name for card in cards}
    defined_functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    allowed_bare_calls = (
        set(StatefulCodeRunner.SAFE_BUILTINS) | primitive_names | defined_functions
    )
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        primitive_name: str | None = None
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            if node.func.value.id == "primitives":
                primitive_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            if node.func.id not in allowed_bare_calls:
                return (
                    f"agent_code_contract_error: unknown bare callable {node.func.id!r}"
                )
            if node.func.id in primitive_names:
                primitive_name = node.func.id
        if primitive_name is None:
            continue
        if primitive_name not in primitive_names:
            return f"agent_code_contract_error: primitive {primitive_name!r} is not in the current interface"
        if node.args:
            return f"agent_code_contract_error: primitive {primitive_name!r} must use keyword arguments only"
        if any(keyword.arg is None for keyword in node.keywords):
            return f"agent_code_contract_error: primitive {primitive_name!r} may not receive **kwargs"
    return None


def _default_calvin_root() -> Path:
    return get_project_paths().external_upstream("calvin")


def _default_calvin_policy_root() -> Path:
    return (
        get_project_paths().external_assets("calvin")
        / "policy"
        / "D_D_static_rgb_baseline"
    )


def _calvin_noop_policy(subgoal: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        "subgoal_success": False,
        "metadata": {
            "backend": "calvin_noop_policy_step",
            "subgoal": subgoal,
            "official_task_success_claimed": False,
            "payload_keys": sorted(str(key) for key in payload),
        },
    }


def _default_robocasa_asset_cache() -> Path:
    return get_project_paths().external_assets("robocasa")


def _default_robocasa365_dataset_root(
    task_family: str = "TurnOnMicrowave", date: str = "20250813"
) -> Path:
    override = os.environ.get("ROBOCASA365_DATASET_ROOT")
    if override:
        return Path(override)
    return (
        get_project_paths().external_root
        / "robocasa365/datasets/v1.0/target/atomic"
        / task_family
        / date
        / "lerobot"
    )


def _default_robocasa365_probe_python() -> str:
    return os.environ.get("ROBOCASA365_PYTHON", sys.executable)


def _default_robowits_repo() -> Path:
    return get_project_paths().external_upstream("robowits")


def _default_robotwin2_repo() -> Path:
    return get_project_paths().external_upstream("robotwin2")


def _default_robodojo_repo() -> Path:
    return get_project_paths().external_upstream("robodojo")


def _spatialclaw_geometry_fixture_samples() -> list[SpatialSample]:
    return [
        SpatialSample(
            sample_id="qwen_geometry_fixture_0",
            source="spatialclaw:geometry_fixture",
            instruction=(
                "Use the geometry tool to decide which point is closer to the origin. "
                "Answer A if point A is closer; otherwise answer B."
            ),
            answer="A",
            category="geometry_tool_scaffold",
            views=[
                {
                    "view_id": "geometry_frame_0",
                    "artifact": "spatialclaw://fixture/geometry_frame_0",
                    "media_type": "geometry_descriptor",
                    "origin": [0.0, 0.0],
                    "points": {"A": [1.0, 1.0], "B": [4.0, 4.0]},
                    "observable_evidence_note": "Point coordinates are observable fixture evidence, not an answer key.",
                }
            ],
            choices=["A", "B"],
            facts=[],
            metadata={
                "benchmark_family": "SpatialClaw",
                "adapter_target": "NVlabs/SpatialClaw",
                "fixture_only": True,
                "live_erqa_success": False,
                "oracle_leakage_level": "none",
            },
        )
    ]


def _spatialclaw_erqa_backend() -> SpatialClawAgentNativeBackend:
    samples, _smoke = load_spatialclaw_samples(
        repo_path=get_project_paths().external_upstream("spatialclaw"),
        benchmark="erqa",
        limit=1,
    )
    return SpatialClawAgentNativeBackend(samples=samples)


def _spatialclaw_benchmark_backend(benchmark: str) -> SpatialClawAgentNativeBackend:
    samples, _smoke = load_spatialclaw_samples(
        repo_path=get_project_paths().external_upstream("spatialclaw"),
        benchmark=benchmark,
        limit=1,
    )
    return SpatialClawAgentNativeBackend(samples=samples)


def _prepare_robocasa_state_only_runtime() -> None:
    cache_dir = get_project_paths().external_environment("robocasa") / "cache/numba"
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache_dir))


def _robocasa_state_delta_skill_executor(
    env: Any, skill_name: str, kwargs: dict[str, Any]
) -> PrimitiveResult:
    if skill_name not in {"move_robocasa_ee_to", "manipulate_robocasa_object"}:
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                "execution_status": "not_executed",
                "requires_motion_backend": True,
            },
            error="state_delta_skill_unsupported",
        )
    action, low, high = _robocasa_zero_action(env)
    if action is None:
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                "execution_status": "not_executed",
                "requires_motion_backend": True,
            },
            error="action_space_unavailable",
        )
    obs = _robocasa_read_observation(env)
    if obs is None:
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                "execution_status": "not_executed",
                "requires_motion_backend": True,
            },
            error="observation_unavailable",
        )
    target_name = kwargs.get("target_name") or kwargs.get("object_name") or "obj"
    target_position = _robocasa_target_position(
        obs, target_name=target_name, explicit_position=kwargs.get("target_position")
    )
    eef_position = _robocasa_vector(obs, "robot0_eef_pos")
    if target_position is None or eef_position is None:
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                "execution_status": "not_executed",
                "requires_motion_backend": True,
                "motion_backend": "robocasa_state_delta_motion",
                "available_state_keys": sorted(str(key) for key in obs)
                if isinstance(obs, dict)
                else [],
            },
            error="state_grounding_unavailable",
        )

    offset = _robocasa_vector(
        {
            "offset": kwargs.get("offset")
            or (
                [0.0, 0.0, 0.12]
                if skill_name == "manipulate_robocasa_object"
                else [0.0, 0.0, 0.0]
            )
        },
        "offset",
    )
    target_with_offset = target_position + (
        offset if offset is not None else np.zeros(3, dtype=np.float32)
    )
    distance_before = float(np.linalg.norm(target_with_offset - eef_position))
    horizon = max(1, int(kwargs.get("horizon", 20)))
    max_delta = float(kwargs.get("max_delta", 0.3))
    gain = float(kwargs.get("gain", 2.0))
    tolerance = float(kwargs.get("tolerance", 0.06))
    gripper_command = float(kwargs.get("gripper_command", 1.0))
    step_records: list[dict[str, Any]] = []
    latest_reward: Any = None
    latest_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    for step_index in range(horizon):
        eef_position = _robocasa_vector(obs, "robot0_eef_pos")
        target_position = _robocasa_target_position(
            obs,
            target_name=target_name,
            explicit_position=kwargs.get("target_position"),
        )
        if eef_position is None or target_position is None:
            break
        target_with_offset = target_position + (
            offset if offset is not None else np.zeros(3, dtype=np.float32)
        )
        delta = target_with_offset - eef_position
        action = np.asarray(action, dtype=np.float32)
        action[...] = 0.0
        action[:3] = np.clip(delta * gain, -max_delta, max_delta)
        if action.shape[0] > 0:
            action[-1] = gripper_command
        bounded_action = (
            np.clip(action, low, high)
            if low is not None and high is not None
            else action
        )
        step_result = env.step(bounded_action)
        obs, latest_reward, terminated, truncated, latest_info = (
            _split_robocasa_step_result(step_result)
        )
        distance = _robocasa_distance(
            obs,
            target_name=target_name,
            explicit_position=kwargs.get("target_position"),
            offset=offset,
        )
        step_records.append(
            {
                "step_index": step_index,
                "distance_to_target": distance,
                "action_summary": summarize_robocasa_observation(bounded_action),
            }
        )
        if distance is not None and distance <= tolerance:
            break
        if terminated or truncated:
            break

    distance_after = _robocasa_distance(
        obs,
        target_name=target_name,
        explicit_position=kwargs.get("target_position"),
        offset=offset,
    )
    moved = distance_after is not None and distance_after < distance_before
    filtered_info_keys = [
        key
        for key in sorted(str(key) for key in dict(latest_info or {}))
        if "success" not in key.lower()
    ]
    return PrimitiveResult(
        name=skill_name,
        ok=bool(moved and step_records),
        output={
            **kwargs,
            "execution_status": "stepped" if step_records else "not_executed",
            "requires_motion_backend": False,
            "motion_backend": "robocasa_state_delta_motion",
            "motion_status": "distance_reduced" if moved else "no_distance_reduction",
            "moved": bool(moved),
            "target_name": target_name,
            "target_position": _robocasa_to_builtin(target_with_offset.tolist()),
            "distance_before": distance_before,
            "distance_after": distance_after,
            "steps": len(step_records),
            "official_task_completion_claimed": False,
            "step_summary": {
                "reward": _robocasa_to_builtin(latest_reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info_keys": filtered_info_keys,
                "observation_summary": summarize_robocasa_observation(obs),
                "motion_trace": step_records[-5:],
            },
        },
    )


def _robocasa_state_only_step_skill_executor(
    env: Any, skill_name: str, kwargs: dict[str, Any]
) -> PrimitiveResult:
    if skill_name in {"move_robocasa_ee_to", "manipulate_robocasa_object"}:
        return _robocasa_state_delta_skill_executor(env, skill_name, kwargs)
    action_space = getattr(env, "action_space", None)
    action_spec = getattr(env, "action_spec", None)
    if action_spec is not None:
        action = _sample_robocasa_action_spec(action_spec)
    elif action_space is not None and hasattr(action_space, "sample"):
        action = action_space.sample()
    else:
        return PrimitiveResult(
            name=skill_name,
            ok=False,
            output={
                **kwargs,
                "execution_status": "not_executed",
                "requires_motion_backend": True,
            },
            error="action_space_unavailable",
        )
    step_result = env.step(action)
    obs, reward, terminated, truncated, info = _split_robocasa_step_result(step_result)
    return PrimitiveResult(
        name=skill_name,
        ok=True,
        output={
            **kwargs,
            "execution_status": "stepped",
            "requires_motion_backend": False,
            "motion_backend": "robocasa_state_only_step_skill",
            "official_task_success_claimed": False,
            "step_summary": {
                "action_summary": summarize_robocasa_observation(action),
                "reward": _robocasa_to_builtin(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info_keys": sorted(str(key) for key in dict(info or {})),
                "observation_summary": summarize_robocasa_observation(obs),
            },
        },
    )


def _robocasa_read_observation(env: Any) -> dict[str, Any] | None:
    for candidate in (env, getattr(env, "unwrapped", getattr(env, "env", env))):
        for method_name in ("_get_observations", "_get_obs"):
            method = getattr(candidate, method_name, None)
            if callable(method):
                try:
                    obs = method()
                except Exception:
                    continue
                return obs if isinstance(obs, dict) else None
    return None


def _robocasa_zero_action(
    env: Any,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    action_spec = getattr(env, "action_spec", None)
    if isinstance(action_spec, tuple) and len(action_spec) == 2:
        low = np.asarray(action_spec[0], dtype=np.float32)
        high = np.asarray(action_spec[1], dtype=np.float32)
        return np.clip(np.zeros_like(low, dtype=np.float32), low, high), low, high
    action_space = getattr(env, "action_space", None)
    if action_space is not None and hasattr(action_space, "sample"):
        sample = np.asarray(action_space.sample(), dtype=np.float32)
        low = np.asarray(
            getattr(action_space, "low", np.full_like(sample, -1.0)), dtype=np.float32
        )
        high = np.asarray(
            getattr(action_space, "high", np.full_like(sample, 1.0)), dtype=np.float32
        )
        return np.clip(np.zeros_like(sample, dtype=np.float32), low, high), low, high
    return None, None, None


def _robocasa_vector(obs: dict[str, Any], key: str) -> np.ndarray | None:
    value = obs.get(key)
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.shape[0] < 3:
        return None
    return array[:3]


def _robocasa_target_position(
    obs: dict[str, Any],
    *,
    target_name: str | None,
    explicit_position: Any,
) -> np.ndarray | None:
    if explicit_position is not None:
        return _robocasa_vector({"target": explicit_position}, "target")
    if target_name:
        direct = _robocasa_vector(obs, f"{target_name}_pos")
        if direct is not None:
            return direct
    return _robocasa_vector(obs, "obj_pos")


def _robocasa_distance(
    obs: dict[str, Any],
    *,
    target_name: str | None,
    explicit_position: Any,
    offset: np.ndarray | None,
) -> float | None:
    eef = _robocasa_vector(obs, "robot0_eef_pos")
    target = _robocasa_target_position(
        obs, target_name=target_name, explicit_position=explicit_position
    )
    if eef is None or target is None:
        return None
    return float(
        np.linalg.norm(
            (target + (offset if offset is not None else np.zeros(3, dtype=np.float32)))
            - eef
        )
    )


def _policy_skill_contract_backend(benchmark_id: str) -> PolicySkillAgentRuntimeBackend:
    return PolicySkillAgentRuntimeBackend(
        config=PolicySkillRuntimeConfig(
            benchmark_id=benchmark_id,
            policy_id=f"{benchmark_id}-policy-skill-contract",
            action_horizon=4,
            observation_contract={"required_fields": ["image", "state", "instruction"]},
            live=True,
        )
    )


def built_in_live_cases(case_ids: list[str] | None = None) -> list[LiveAgentSmokeCase]:
    cases = {
        **{
            f"{benchmark_id}_policy_skill_contract": LiveAgentSmokeCase(
                case_id=f"{benchmark_id}_policy_skill_contract",
                benchmark_id=benchmark_id,
                task_id=f"w5_{benchmark_id}_policy_skill_contract",
                objective=(
                    f"Exercise the public {benchmark_id} policy-as-skill contract: inspect task/policy context and the "
                    "observation schema, record evidence derived from those public outputs, invoke the policy, and submit "
                    "the returned action chunk. The harness will audit contract progress; this case does not claim "
                    "standalone or official benchmark success."
                ),
                required_primitives=[
                    f"get_{benchmark_id}_policy_context",
                    f"inspect_{benchmark_id}_observation_contract",
                    f"score_or_record_{benchmark_id}_rollout_evidence",
                    f"call_{benchmark_id}_policy_skill",
                    f"submit_{benchmark_id}_policy_action_chunk",
                ],
                required_ok_primitives=[
                    f"get_{benchmark_id}_policy_context",
                    f"inspect_{benchmark_id}_observation_contract",
                    f"score_or_record_{benchmark_id}_rollout_evidence",
                    f"call_{benchmark_id}_policy_skill",
                    f"submit_{benchmark_id}_policy_action_chunk",
                ],
                backend_factory=lambda benchmark_id=benchmark_id: _policy_skill_contract_backend(
                    benchmark_id
                ),
                readiness_tier="policy_skill_contract",
                counts_toward_official_success=False,
                harness_only_verifier=True,
                action_hint=(
                    "Derive all policy-call inputs and evidence from the visible task, policy context, and observation "
                    "contract. Submit exactly the action_chunk returned by the policy primitive. Do not invent a "
                    "task-specific action, inspect evaluator state, or claim official task success."
                ),
            )
            for benchmark_id in ("openvla", "openpi", "lerobot", "octo")
        },
        "alfworld_pick_knife_to_sidetable": LiveAgentSmokeCase(
            case_id="alfworld_pick_knife_to_sidetable",
            benchmark_id="alfworld",
            task_id="qwen_live_alfworld_pick_knife_to_sidetable",
            objective=(
                "Read the ALFWorld household task and admissible-action evidence, infer the required route, "
                "record the plan, then solve the official TextWorld task through typed action primitives."
            ),
            required_primitives=[
                "get_alfworld_task_context",
                "observe_alfworld_state",
                "list_alfworld_actions",
                "go_to",
                "take",
                "put",
                "record_alfworld_evidence",
            ],
            required_ok_primitives=[
                "get_alfworld_task_context",
                "observe_alfworld_state",
                "list_alfworld_actions",
                "go_to",
                "take",
                "put",
                "record_alfworld_evidence",
            ],
            backend_factory=lambda: ALFWorldAgentRuntimeBackend(
                ALFWorldRuntimeConfig(live=True, max_steps=80, num_eval_games=1)
            ),
            seed=0,
            action_hint=(
                "Call get_alfworld_task_context(prompt=..., query='current ALFWorld task', agent_context=...). "
                "Then call observe_alfworld_state(include_actions=True, query='visible locations and valid actions', agent_context=...) "
                "and list_alfworld_actions(max_actions=80, agent_context=...) to inspect the current admissible commands. "
                "Derive each typed navigation and object-interaction argument from the current instruction, observation, and "
                "admissible-action evidence, refreshing public state after actions when needed. Record the chosen evidence "
                "and rationale with record_alfworld_evidence(key='plan', value=..., agent_context=...). "
                "Do not call raw env.step, expert trajectories, gold action sequences, oracle/checker/debug APIs, demos, or replay."
            ),
        ),
        "vimabench_visual_manipulation": LiveAgentSmokeCase(
            case_id="vimabench_visual_manipulation",
            benchmark_id="vimabench",
            task_id="qwen_live_vimabench_visual_manipulation",
            objective=(
                "Inspect the official VIMA prompt, scene, and visible segmentation instances, identify the source and "
                "target from exposed evidence, build one pick-place action from those IDs, record the evidence, and submit it."
            ),
            required_primitives=[
                "observe_vima_prompt",
                "observe_vima_scene",
                "inspect_vima_instances",
                "inspect_vima_instance",
                "build_vima_pick_place_action",
                "record_vima_evidence",
                "submit_vima_action",
            ],
            required_ok_primitives=[
                "observe_vima_prompt",
                "observe_vima_scene",
                "inspect_vima_instances",
                "inspect_vima_instance",
                "build_vima_pick_place_action",
                "record_vima_evidence",
                "submit_vima_action",
            ],
            backend_factory=lambda: VimaBenchAgentRuntimeBackend(
                VimaBenchRuntimeConfig(task_name="visual_manipulation", live=True)
            ),
            seed=0,
            action_hint=(
                "Read observe_vima_prompt and observe_vima_scene, then enumerate inspect_vima_instances. Compare the prompt "
                "tokens with each inspect_vima_instance result and derive the source_segmentation_id and target_segmentation_id "
                "from that exposed evidence. Pass those derived IDs to build_vima_pick_place_action, record the observations, IDs, "
                "and returned poses with record_vima_evidence, and pass the returned pose fields to submit_vima_action. "
                "Do not hardcode entity IDs, poses, or an action sequence, and do not access evaluator or verifier state."
            ),
        ),
        "cliport_stack_pyramid": LiveAgentSmokeCase(
            case_id="cliport_stack_pyramid",
            benchmark_id="cliport",
            task_id="qwen_live_cliport_stack_pyramid",
            objective=(
                "Read the CLIPort language goal, inspect RGB-D evidence, ground an object, "
                "stage a pick, place it back through the real env.step action path, and record evidence."
            ),
            required_primitives=[
                "observe_cliport_rgbd",
                "get_cliport_task_language_goal",
                "locate_cliport_object",
                "pick_cliport_object",
                "place_cliport_object",
                "record_cliport_evidence",
            ],
            required_ok_primitives=[
                "observe_cliport_rgbd",
                "get_cliport_task_language_goal",
                "pick_cliport_object",
                "place_cliport_object",
                "record_cliport_evidence",
            ],
            backend_factory=lambda: CLIPortAgentRuntimeBackend(
                CLIPortRuntimeConfig(
                    task_name="stack-block-pyramid-seq-seen-colors",
                    mode="test",
                    live=True,
                )
            ),
            seed=10001,
            action_hint=(
                "If the language goal is ambiguous, use object_ref=None and query='the object mentioned by the language goal'. "
                "Use the same query for pick and place so the smoke executes one real pick-place env.step. "
                "Official reward/done success is not required for this smoke."
            ),
        ),
        "cliport_place_red_in_green": LiveAgentSmokeCase(
            case_id="cliport_place_red_in_green",
            benchmark_id="cliport",
            task_id="qwen_live_cliport_place_red_in_green",
            objective=(
                "Read the CLIPort language goal, inspect RGB-D/segmentation evidence, "
                "ground the visible red block and green bowl, then solve the task with one real pick-place env.step."
            ),
            required_primitives=[
                "observe_cliport_rgbd",
                "get_cliport_task_language_goal",
                "inspect_cliport_instances",
                "inspect_cliport_instance",
                "submit_cliport_pick_place_action",
                "record_cliport_evidence",
            ],
            required_ok_primitives=[
                "observe_cliport_rgbd",
                "get_cliport_task_language_goal",
                "inspect_cliport_instances",
                "inspect_cliport_instance",
                "submit_cliport_pick_place_action",
                "record_cliport_evidence",
            ],
            backend_factory=lambda: CLIPortAgentRuntimeBackend(
                CLIPortRuntimeConfig(
                    task_name="place-red-in-green", mode="test", live=True
                )
            ),
            seed=10004,
            action_hint=(
                "Read get_cliport_task_language_goal and observe_cliport_rgbd, enumerate inspect_cliport_instances, and inspect "
                "candidate IDs with inspect_cliport_instance. Derive source_object_id and target_object_id from the current language "
                "and visible instance evidence, record that derivation with record_cliport_evidence, and call "
                "submit_cliport_pick_place_action with the derived IDs or poses. Do not hardcode object IDs, poses, or a fixed action, "
                "and do not access evaluator or verifier state."
            ),
        ),
        "vlabench_select_toy": LiveAgentSmokeCase(
            case_id="vlabench_select_toy",
            benchmark_id="vlabench",
            task_id="qwen_live_vlabench_select_toy",
            objective=(
                "Read the VLABench instruction, observe real state-only object evidence, "
                "ground the hawkeye entity, record a compact motion plan, then move the real robot end effector "
                "above hawkeye through upstream IK and env.step(action)."
            ),
            required_primitives=[
                "get_vlabench_instruction",
                "observe_vlabench_scene",
                "locate_vlabench_entity",
                "record_vlabench_evidence",
                "move_vlabench_ee_to",
            ],
            required_ok_primitives=[
                "get_vlabench_instruction",
                "observe_vlabench_scene",
                "locate_vlabench_entity",
                "record_vlabench_evidence",
                "move_vlabench_ee_to",
            ],
            backend_factory=lambda: VLABenchAgentSmokeBackend(
                exposed_primitives={
                    "get_vlabench_instruction",
                    "observe_vlabench_scene",
                    "resolve_vlabench_instruction_targets",
                    "locate_vlabench_entity",
                    "record_vlabench_evidence",
                    "move_vlabench_ee_to",
                    "grasp_vlabench_entity",
                    "lift_vlabench_ee",
                    "place_vlabench_entity_in",
                }
            ),
            seed=2,
            action_hint=(
                "Use get_vlabench_instruction and observe_vlabench_scene first. Then call "
                "locate_vlabench_entity(entity_name='hawkeye', query='object to put into the giftbox', ...). "
                "Record a compact plan with the selected entity pose. Finally call "
                "move_vlabench_ee_to(target_name='hawkeye', offset=[0.0, 0.0, 0.18], target_site='xpos', "
                "gripper_state=0.04, horizon=20, prompt=..., query='move above hawkeye', agent_context=...). "
                "This proves a real VLABench IK plus env.step(action) operation; official task success is not claimed. "
                "Do not call raw env.step, task oracle, checker, expert skills, demos, replay, get_task_progress, or get_expert_skill_sequence."
            ),
        ),
        "vlabench_select_toy_skilllib": LiveAgentSmokeCase(
            case_id="vlabench_select_toy_skilllib",
            benchmark_id="vlabench",
            task_id="qwen_live_vlabench_select_toy_skilllib",
            objective=(
                "Read the VLABench select_toy instruction and live object evidence, ground the toy and gift box, "
                "then solve the task through public SkillLib grasp/lift/place primitives backed by real env.step actions."
            ),
            required_primitives=[
                "get_vlabench_instruction",
                "observe_vlabench_scene",
                "inspect_vlabench_visual_evidence",
                "locate_vlabench_entity",
                "ground_vlabench_visual_target",
                "grasp_vlabench_entity",
                "lift_vlabench_ee",
                "place_vlabench_entity_in",
            ],
            required_ok_primitives=[
                "get_vlabench_instruction",
                "observe_vlabench_scene",
                "inspect_vlabench_visual_evidence",
                "locate_vlabench_entity",
                "ground_vlabench_visual_target",
                "grasp_vlabench_entity",
                "lift_vlabench_ee",
                "place_vlabench_entity_in",
            ],
            backend_factory=lambda: VLABenchAgentSmokeBackend(),
            reset_config={
                "episode_config_file": str(
                    resolve_project_root()
                    / "benchmarks/operation/vlabench/select_toy_minimal_episode_config.json"
                )
            },
            seed=2,
            code_timeout_seconds=300.0,
            action_hint=(
                "Read the instruction, inspect native RGB/depth/segmentation evidence, and derive entity bindings from the "
                "returned visual candidates. Pass evidence_handle values from visual grounding into every motion primitive via "
                "evidence_handles. Select motion parameters from current observations rather than embedding a task solution. "
                "Do not hardcode entity names, poses, lift distances, or a waypoint sequence, and "
                "do not access task progress, expert sequences, evaluator state, or verifier state."
            ),
        ),
        "mmsi_spatial_smoke": LiveAgentSmokeCase(
            case_id="mmsi_spatial_smoke",
            benchmark_id="mmsi_bench",
            task_id="spatial:mmsi_bench:mmsi_0",
            objective=(
                "Read the MMSI multi-image spatial question, inspect both available image artifacts, "
                "record agent-derived spatial evidence, compare the requested relation, then submit one answer choice."
            ),
            required_primitives=[
                "get_mmsi_task_context",
                "inspect_mmsi_image",
                "inspect_mmsi_pixels",
                "estimate_mmsi_camera_motion",
                "compare_spatial_relation",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_mmsi_task_context",
                "inspect_mmsi_image",
                "inspect_mmsi_pixels",
                "estimate_mmsi_camera_motion",
                "compare_spatial_relation",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: MMSISpatialBackend(
                data_path="benchmarks/non_operation/mmsi_bench/mmsi_smoke.jsonl"
            ),
            action_hint=(
                "Use ctx = primitives.get_mmsi_task_context(...).output, loop over ctx['views'] and call "
                "inspect_mmsi_image(view_id=..., query=ctx['instruction'], context=...). "
                "For at least one materialized view, call inspect_mmsi_pixels(view_id=..., bbox=None, "
                "query='read global image evidence for the spatial question', context=...) and record the returned "
                "image width/height and region statistics as evidence. Then call estimate_mmsi_camera_motion("
                "source_view_id=ctx['views'][0]['view_id'], target_view_id=ctx['views'][1]['view_id'], "
                "query=ctx['instruction'], choices=ctx['choices'], context=...) and use output['recommended_choice'] plus "
                "output['choice_scores'] as the primary camera-motion evidence. Submit exactly one choice label such as A/B/C/D. "
                "Do not submit an answer before using the real pixel and optical-flow primitives; do not call answer keys, oracle/checker internals, evaluator scripts, demos, replay, or raw dataset labels."
            ),
        ),
        "mmsi_camera_rotation_real_pixels": LiveAgentSmokeCase(
            case_id="mmsi_camera_rotation_real_pixels",
            benchmark_id="mmsi_bench",
            task_id="spatial:mmsi_bench:mmsi_1",
            objective=(
                "Read the second real MMSI-Bench local manifest sample, inspect both materialized first-person images, "
                "use pixel statistics and camera-motion estimation as evidence, then submit one non-oracle answer choice."
            ),
            required_primitives=[
                "get_mmsi_task_context",
                "inspect_mmsi_image",
                "inspect_mmsi_pixels",
                "estimate_mmsi_camera_motion",
                "compare_spatial_relation",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_mmsi_task_context",
                "inspect_mmsi_image",
                "inspect_mmsi_pixels",
                "estimate_mmsi_camera_motion",
                "compare_spatial_relation",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: MMSISpatialBackend(
                data_path="benchmarks/non_operation/mmsi_bench/mmsi_smoke.jsonl"
            ),
            action_hint=(
                "Use ctx = primitives.get_mmsi_task_context(...).output for task mmsi_1, inspect every ctx['views'] item with "
                "inspect_mmsi_image, then call inspect_mmsi_pixels on at least one view. Call estimate_mmsi_camera_motion with "
                "the two view ids, query=ctx['instruction'], and choices=ctx['choices']; record the motion interpretation and "
                "choice_scores with record_spatial_evidence before submit_spatial_answer. Do not use answer keys, thought fields, "
                "oracle/checker internals, evaluator scripts, demos, replay, or raw dataset labels."
            ),
        ),
        "esi_spatial_fixture_smoke": LiveAgentSmokeCase(
            case_id="esi_spatial_fixture_smoke",
            benchmark_id="esi_spatial",
            task_id="spatial:vsi_bench_style:fixture_vsi_style_0",
            objective=(
                "Read the ESI/VSI-style egocentric spatial question, inspect the exposed view descriptor, "
                "record the count evidence, then submit the derived short answer."
            ),
            required_primitives=[
                "get_task_context",
                "inspect_view",
                "write_evidence",
                "submit_answer",
            ],
            required_ok_primitives=[
                "get_task_context",
                "inspect_view",
                "write_evidence",
                "submit_answer",
            ],
            backend_factory=lambda: SpatialDiagnosticBackend(
                data_path="benchmarks/non_operation/esi/spatial_tasks.jsonl",
                sample_limit=1,
            ),
            action_hint=(
                "Use ctx = primitives.get_task_context(prompt=..., query='current spatial counting task', "
                "agent_context=...).output. Take view_id from ctx['views'][0]['view_id'], then call "
                "inspect_view(view_id=view_id, query='count visible table objects', agent_context=...). "
                "The returned view descriptor exposes observable objects and a spatial_summary but no answer key. "
                "Count table entries from view['objects'], record a compact evidence dict with write_evidence, "
                "then submit the count as a numeral string via submit_answer. Do not call oracle/checker/success/raw dataset labels."
            ),
        ),
        "vsi_real_video_frame_boundary": LiveAgentSmokeCase(
            case_id="vsi_real_video_frame_boundary",
            benchmark_id="vsi_bench",
            task_id="spatial:vsi_bench:vsi_0",
            objective=(
                "Read a real VSI-Bench Arkitscenes egocentric video question, inspect the local mp4 descriptor and "
                "sample real video frames, run local open-vocabulary detection/tracking and SAM2 box-prompted "
                "segmentation, record the visual evidence packet, then submit the agent's best non-oracle count."
            ),
            required_primitives=[
                "get_task_context",
                "inspect_view",
                "inspect_video_frames",
                "preflight_video_grounding_model",
                "preflight_video_segmentation_model",
                "track_video_objects",
                "segment_video_objects",
                "write_evidence",
                "submit_answer",
            ],
            required_ok_primitives=[
                "get_task_context",
                "inspect_view",
                "inspect_video_frames",
                "preflight_video_grounding_model",
                "preflight_video_segmentation_model",
                "track_video_objects",
                "segment_video_objects",
                "write_evidence",
                "submit_answer",
            ],
            backend_factory=lambda: SpatialDiagnosticBackend(
                data_path="benchmarks/non_operation/esi/vsi_debiased_local_video_smoke.jsonl",
                sample_limit=1,
            ),
            action_hint=(
                "Use ctx = primitives.get_task_context(prompt=..., query='current VSI video question', agent_context=...).output. "
                "Take view_id = ctx['views'][0]['view_id']; call inspect_view(view_id=view_id, query=ctx['instruction'], "
                "agent_context=...). Then call inspect_video_frames(view_id=view_id, sample_count=3, query='sample real mp4 frames "
                "for object counting evidence', agent_context=...). Then call preflight_video_grounding_model(model_hint='owl_vit', "
                "agent_context=...) and preflight_video_segmentation_model(model_hint='sam2', agent_context=...). Then call "
                "track_video_objects(view_id=view_id, query_labels=['table'], detector_backend='owl_vit', sample_count=8, "
                "score_threshold=0.02, max_detections_per_frame=4, query=ctx['instruction'], agent_context=...). Then call "
                "segment_video_objects(view_id=view_id, query_labels=['table'], detector_backend='owl_vit', segmentation_backend='sam2', "
                "sample_count=8, score_threshold=0.02, max_detections_per_frame=4, max_segments=4, query=ctx['instruction'], "
                "agent_context=...). "
                "Record video['width'], video['height'], video['frame_count'], sampled frame indices, frame RGB stats, "
                "detector name, tracks, segments, mask areas, mask bboxes, and label_counts with write_evidence. Submit the "
                "mask_object_count_estimate from segment_video_objects as a numeral string with submit_answer. This case proves "
                "real local video + DET + SAM2 SEG primitive access; do not call "
                "answer keys, checker, oracle labels, dataset ground_truth, demos, replay, or evaluator internals."
            ),
        ),
        "behavior1k_preflight_contract": LiveAgentSmokeCase(
            case_id="behavior1k_preflight_contract",
            benchmark_id="behavior1k",
            task_id="qwen_live_behavior1k_preflight_contract",
            objective=(
                "Read the BEHAVIOR-1K / OmniGibson task context, observe available runtime and state evidence, "
                "then record an explicit preflight evidence packet. This proves the agent-native contract while "
                "keeping official OmniGibson task success harness-side and blocked until the full simulator/assets stack is configured."
            ),
            required_primitives=[
                "get_behavior1k_task_context",
                "observe_behavior1k_state",
                "record_behavior1k_evidence",
            ],
            required_ok_primitives=[
                "get_behavior1k_task_context",
                "observe_behavior1k_state",
                "record_behavior1k_evidence",
            ],
            backend_factory=lambda: Behavior1KAgentRuntimeBackend(
                Behavior1KRuntimeConfig(task_name="turning_on_radio", live=False)
            ),
            action_hint=(
                "Use get_behavior1k_task_context(prompt=..., query='current BDDL/OmniGibson task context', agent_context=...). "
                "Then call observe_behavior1k_state(prompt=..., query='available dry-runtime observation evidence', "
                "agent_context=..., include_raw=False). Record a compact evidence dict with record_behavior1k_evidence("
                "key='preflight_contract', value={...}, agent_context=...). In live=False dry mode there are no scene assets, "
                "so do not call inspect_behavior1k_asset. Do not call success predicates, BDDL checker, evaluator, demos, replay, or raw simulator APIs. "
                "benchmark_task_success is expected to remain false until the full OmniGibson/BEHAVIOR stack is live."
            ),
        ),
        "behavior1k_full_bddl_toggle": LiveAgentSmokeCase(
            case_id="behavior1k_full_bddl_toggle",
            benchmark_id="behavior1k",
            task_id="behavior1k_turning_on_radio_full_bddl_toggle_qwen",
            objective=(
                "Solve one official BEHAVIOR-1K turning_on_radio episode in the full OmniGibson BDDL environment. "
                "Use native RGB-D/segmentation and public asset-state evidence to ground the relevant receiver, derive "
                "the action payload from exposed observations, and leave official task success exclusively to the harness verifier."
            ),
            required_primitives=[
                "get_behavior1k_task_context",
                "observe_behavior1k_state",
                "inspect_behavior1k_visual_evidence",
                "inspect_behavior1k_asset",
                "inspect_behavior1k_object_state",
                "inspect_behavior1k_contacts",
                "record_behavior1k_evidence",
                "submit_behavior1k_action",
            ],
            required_ok_primitives=[
                "get_behavior1k_task_context",
                "observe_behavior1k_state",
                "inspect_behavior1k_visual_evidence",
                "inspect_behavior1k_asset",
                "inspect_behavior1k_object_state",
                "inspect_behavior1k_contacts",
                "record_behavior1k_evidence",
                "submit_behavior1k_action",
            ],
            backend_factory=lambda: Behavior1KAgentRuntimeBackend(
                Behavior1KRuntimeConfig(
                    task_name="turning_on_radio",
                    env_config=build_behavior1k_full_bddl_env_config(
                        activity_name="turning_on_radio",
                        scene_model="house_double_floor_lower",
                        image_size=64,
                        # Physical toggle may need one public-pose replan when
                        # the movable support shifts during base navigation.
                        # Keep the episode horizon aligned with the semantic
                        # action budget so that recovery is not truncated by
                        # the environment at the old 300-step boundary.
                        max_steps=512,
                    ),
                    live=True,
                )
            ),
            code_timeout_seconds=900.0,
            readiness_tier="official_success_candidate",
            counts_toward_official_success=True,
            harness_only_verifier=True,
            action_hint=(
                "Read get_behavior1k_task_context, observe_behavior1k_state, and inspect_behavior1k_visual_evidence. Use "
                "inspect_behavior1k_asset to derive the relevant runtime asset name, then inspect its public state and contacts "
                "with inspect_behavior1k_object_state and inspect_behavior1k_contacts. Construct the submit_behavior1k_action "
                "payload from the current task schema and those observations, and record the grounding and action rationale with "
                "record_behavior1k_evidence. Do not hardcode an asset ID, pose, state mutation, or action sequence, and do not "
                "access BDDL predicates, evaluator state, or verifier state."
            ),
        ),
        "spatialclaw_geometry_fixture": LiveAgentSmokeCase(
            case_id="spatialclaw_geometry_fixture",
            benchmark_id="spatialclaw",
            task_id="spatial:spatialclaw_geometry_fixture:qwen_geometry_fixture_0",
            objective=(
                "Read a SpatialClaw-style spatial task, inspect the observation descriptor, run deterministic geometry tools, "
                "record the distance evidence, then submit the derived answer. This is a contract fixture for the agent-native "
                "SpatialClaw runtime, not a live ERQA dataset completion."
            ),
            required_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "run_geometry_tool",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "run_geometry_tool",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: SpatialClawAgentNativeBackend(
                samples=_spatialclaw_geometry_fixture_samples()
            ),
            action_hint=(
                "Use ctx = primitives.get_spatial_task_context(prompt=..., query='current SpatialClaw geometry task', "
                "agent_context=...).output. Use view_id = ctx['views'][0]['view_id'], then inspect_spatial_observation("
                "view_id=view_id, prompt=..., query='observable points and origin', agent_context=...). The inspected view contains "
                "origin=[0,0], points['A']=[1,1], and points['B']=[4,4]. Call run_geometry_tool twice with tool='distance_2d': "
                "inputs={'p1':[1.0,1.0], 'p2':[0.0,0.0]} and inputs={'p1':[4.0,4.0], 'p2':[0.0,0.0]}. "
                "Record the two distances and selected answer with record_spatial_evidence, then submit_spatial_answer(answer='A', ...). "
                "Do not call answer keys, oracle/checker internals, evaluator scripts, demos, replay, or raw dataset labels."
            ),
        ),
        "spatialclaw_erqa_real_visual": LiveAgentSmokeCase(
            case_id="spatialclaw_erqa_real_visual",
            benchmark_id="spatialclaw",
            task_id="spatial:spatialclaw_erqa:ERQA_1",
            objective=(
                "Load a real SpatialClaw ERQA sample through the upstream loader, inspect the local image observation "
                "and pixel evidence, record a visual reasoning packet, then submit the agent's best answer choice. "
                "This is the real ERQA visual-data path; task success remains verifier-side and is not assumed."
            ),
            required_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "analyze_spatialclaw_visual_query",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "analyze_spatialclaw_visual_query",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=_spatialclaw_erqa_backend,
            action_hint=(
                "Use ctx = primitives.get_spatial_task_context(prompt=..., query='real ERQA visual question', "
                "agent_context=...).output and read ctx['instruction'] plus ctx['views']. Select the first available view_id, "
                "call inspect_spatial_observation(view_id=..., prompt=..., query='available local image artifact', agent_context=...), "
                "then call inspect_spatial_pixels(view_id=..., bbox=None or a concrete bbox, prompt=..., "
                "query='RGB pixel evidence for the ERQA scene', agent_context=...). Also call "
                "analyze_spatialclaw_visual_query(view_id=..., prompt=..., query=ctx['instruction'], choices=ctx.get('choices'), "
                "agent_context=...) to obtain annotation/trajectory evidence and choice_scores. Record the image artifact, "
                "dimensions, pixel statistics, annotation_summary, choice_scores, and your non-oracle answer rationale with "
                "record_spatial_evidence. Submit the analysis output['recommended_choice'] when present; otherwise submit exactly "
                "one answer letter such as A/B/C/D. Do not call answer keys, evaluator internals, oracle/checker APIs, demos, replay, or raw dataset labels."
            ),
        ),
        "spatialclaw_spbench_det_relation": LiveAgentSmokeCase(
            case_id="spatialclaw_spbench_det_relation",
            benchmark_id="spatialclaw",
            task_id="spatial:spatialclaw_spbench:SI_1",
            objective=(
                "Load a real SpatialClaw SPBench image task, inspect the local image, detect the named objects "
                "with the local open-vocabulary detector, compute the camera-perspective bbox relation, record the "
                "visible evidence, then submit the matching multiple-choice answer."
            ),
            required_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "detect_spatial_image_objects",
                "run_geometry_tool",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "detect_spatial_image_objects",
                "run_geometry_tool",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: _spatialclaw_benchmark_backend("spbench"),
            code_timeout_seconds=300.0,
            action_hint=(
                "Use ctx = primitives.get_spatial_task_context(prompt=..., query='real SPBench spatial relation question', "
                "agent_context=...).output. Select the first view_id and call inspect_spatial_observation and "
                "inspect_spatial_pixels(bbox=None). The instruction asks whether the window is to the refrigerator's "
                "left-front, left-back, right-front, or right-back. Call detect_spatial_image_objects(view_id=view_id, "
                "query_labels=['window','refrigerator','fridge'], score_threshold=0.02, max_detections=20, "
                "detector_backend='owl_vit', ...). Pick the highest-scoring window bbox and refrigerator/fridge bbox from "
                "the returned detections using det['bbox_xyxy']; the DET primitive does not return bbox/box/bounding_box "
                "aliases, so do not invent fallback boxes. Call run_geometry_tool(tool='bbox_spatial_relation', "
                "inputs={'subject_bbox': window_det['bbox_xyxy'], 'reference_bbox': refrigerator_det['bbox_xyxy'], "
                "'image_size': [image_width, image_height]}, ...). "
                "Map compound_relation right-front/left-front/right-back/left-back to the visible ctx['choices'] label, "
                "record bbox/relation evidence with record_spatial_evidence, then submit exactly the bare answer letter "
                "such as 'A', 'B', 'C', or 'D' with no colon or explanatory suffix. "
                "Do not import modules, read files, call answer keys, evaluator internals, oracle/checker APIs, demos, replay, or raw dataset labels."
            ),
        ),
        "spatialclaw_mindcube_view_motion": LiveAgentSmokeCase(
            case_id="spatialclaw_mindcube_view_motion",
            benchmark_id="spatialclaw",
            task_id="spatial:spatialclaw_mindcube:among_group002_q0_1_1",
            objective=(
                "Load a real SpatialClaw MindCube two-view task, inspect public ordered view metadata, infer the "
                "camera-motion descriptor from those public view labels, record the choice scores, then submit the "
                "matching multiple-choice answer."
            ),
            required_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_view_set",
                "infer_mindcube_motion_from_views",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_view_set",
                "infer_mindcube_motion_from_views",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: _spatialclaw_benchmark_backend("mindcube"),
            action_hint=(
                "Use ctx = primitives.get_spatial_task_context(prompt=..., query='real MindCube two-view motion question', "
                "agent_context=...).output. Build view_ids from ctx['views'][0]['view_id'] and ctx['views'][1]['view_id']. "
                "Call inspect_spatial_view_set(view_ids=view_ids, prompt=..., query=ctx['instruction'], agent_context=...) "
                "to expose public view_filename/view_label metadata. Then call infer_mindcube_motion_from_views("
                "view_ids=view_ids, choices=ctx['choices'], prompt=..., query=ctx['instruction'], agent_context=...). "
                "Use output['recommended_choice'] as the answer when present. Record parsed views, motion, choice_scores, "
                "and selected answer with record_spatial_evidence, then submit_spatial_answer(answer=...). "
                "Do not call answer keys, evaluator internals, oracle/checker APIs, demos, replay, or raw dataset labels."
            ),
        ),
        "spatialclaw_omnispatial_annotation_motion": LiveAgentSmokeCase(
            case_id="spatialclaw_omnispatial_annotation_motion",
            benchmark_id="spatialclaw",
            task_id="spatial:spatialclaw_omnispatial:Dynamic_Reasoning_Motion_Analysis_0_0",
            objective=(
                "Load a real SpatialClaw OmniSpatial motion-analysis image, inspect the local RGB artifact, "
                "use the agent-visible colored marker/trajectory analysis primitive, record visual evidence, "
                "then submit the matching multiple-choice answer."
            ),
            required_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "analyze_spatialclaw_visual_query",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            required_ok_primitives=[
                "get_spatial_task_context",
                "inspect_spatial_observation",
                "inspect_spatial_pixels",
                "analyze_spatialclaw_visual_query",
                "record_spatial_evidence",
                "submit_spatial_answer",
            ],
            backend_factory=lambda: _spatialclaw_benchmark_backend("omnispatial"),
            code_timeout_seconds=240.0,
            action_hint=(
                "Use ctx = primitives.get_spatial_task_context(prompt=..., query='real OmniSpatial motion-analysis question', "
                "agent_context=...).output. Select ctx['views'][0]['view_id']; call inspect_spatial_observation and "
                "inspect_spatial_pixels(bbox=None) to ground the real local image. Then call "
                "analyze_spatialclaw_visual_query(view_id=view_id, prompt=..., query=ctx['instruction'], "
                "choices=ctx.get('choices'), agent_context=...) and inspect annotation_summary: the primitive extracts "
                "agent-visible colored trajectory/marker masks from the image and returns choice_scores plus recommended_choice. "
                "Record image size, detected annotation names, marker/trajectory summaries, choice_scores, and rationale with "
                "record_spatial_evidence. Submit exactly analysis['recommended_choice'] as a bare answer letter. This is an "
                "OmniSpatial annotation-motion boundary case, not an Omni3D metric-depth solution. Do not call answer keys, "
                "evaluator internals, oracle/checker APIs, demos, replay, or raw dataset labels."
            ),
        ),
        "scienceworld_find_non_living_thing": LiveAgentSmokeCase(
            case_id="scienceworld_find_non_living_thing",
            benchmark_id="scienceworld",
            task_id="qwen_live_scienceworld_find_non_living_thing",
            objective=(
                "Read the ScienceWorld text task and valid-action evidence, identify a visible non-living object, "
                "record the action plan, then solve the task through typed text-action primitives backed by real env.step calls."
            ),
            required_primitives=[
                "get_scienceworld_task_context",
                "observe_scienceworld_state",
                "list_scienceworld_actions",
                "focus_scienceworld_object",
                "pick_up_scienceworld_object",
                "go_scienceworld_location",
                "move_scienceworld_object_to",
                "record_scienceworld_evidence",
            ],
            required_ok_primitives=[
                "get_scienceworld_task_context",
                "observe_scienceworld_state",
                "list_scienceworld_actions",
                "focus_scienceworld_object",
                "pick_up_scienceworld_object",
                "go_scienceworld_location",
                "move_scienceworld_object_to",
                "record_scienceworld_evidence",
            ],
            backend_factory=lambda: ScienceWorldAgentRuntimeBackend(
                ScienceWorldRuntimeConfig(
                    task_name="find-non-living-thing",
                    variation_idx=0,
                    simplification="easy",
                    env_step_limit=50,
                    live=True,
                )
            ),
            action_hint=(
                "Use get_scienceworld_task_context(prompt=..., query='current ScienceWorld task', agent_context=...). "
                "Then call observe_scienceworld_state(include_actions=True, query='visible objects and valid actions', agent_context=...). "
                "For the requested find-non-living-thing task, do not assume which visible object or destination is correct. "
                "Use list_scienceworld_actions with filters derived from the current instruction and observation to inspect legal actions. "
                "Derive every object, destination, location, and action order from current public evidence, refreshing state and valid actions "
                "after each environment transition. Record a compact evidence-backed plan with record_scienceworld_evidence, then use the "
                "typed focus, pickup, navigation, and move primitives only when their corresponding legal actions are exposed. "
                "Do not call get_gold_action_sequence, oracle/debug APIs, run history labels, or direct env.step; use the typed primitives."
            ),
        ),
        "maniskill_pickcube_pd_ee_delta_pos": LiveAgentSmokeCase(
            case_id="maniskill_pickcube_pd_ee_delta_pos",
            benchmark_id="maniskill",
            task_id="qwen_live_maniskill_pickcube_pd_ee_delta_pos",
            objective=(
                "Solve the official ManiSkill PickCube instruction by deriving visible instance IDs, geometry, and low-level "
                "control actions from the exposed state and visual observations."
            ),
            required_primitives=[
                "observe_maniskill_state",
                "observe_maniskill_visual",
                "list_maniskill_instances",
                "inspect_maniskill_instance",
                "observe_maniskill_control_state",
                "move_maniskill_tcp_to",
                "set_maniskill_gripper",
                "record_maniskill_evidence",
            ],
            required_ok_primitives=[
                "observe_maniskill_state",
                "observe_maniskill_visual",
                "list_maniskill_instances",
                "inspect_maniskill_instance",
                "observe_maniskill_control_state",
                "move_maniskill_tcp_to",
                "set_maniskill_gripper",
                "record_maniskill_evidence",
            ],
            backend_factory=lambda: ManiSkillAgentRuntimeBackend(
                ManiSkillRuntimeConfig(
                    env_id="PickCube-v1",
                    task_instruction="Pick up the cube and move it to the target position.",
                    obs_mode="rgb+depth+segmentation",
                    control_mode="pd_ee_delta_pos",
                    live=True,
                    sim_backend="physx_cpu",
                    render_backend=os.getenv(
                        "AGENTIC_EMBODIED_ARENA_MANISKILL_RENDER_BACKEND", "gpu"
                    ),
                )
            ),
            seed=0,
            action_hint=(
                "Read observe_maniskill_state, observe_maniskill_visual, and observe_maniskill_control_state. Enumerate "
                "list_maniskill_instances and inspect candidate segmentation IDs with inspect_maniskill_instance. Derive the "
                "task entities and target_xyz values from those observations, then compose move_maniskill_tcp_to and "
                "set_maniskill_gripper calls and record the evidence with record_maniskill_evidence. Do not hardcode IDs, poses, "
                "gripper timing, or waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "calvin_policy_hook_noop_step": LiveAgentSmokeCase(
            case_id="calvin_policy_hook_noop_step",
            benchmark_id="calvin",
            task_id="qwen_live_calvin_policy_hook_noop_step",
            objective=(
                "Read the CALVIN language subgoal, inspect state and camera evidence, record a compact policy plan, "
                "then execute one policy-backed CALVIN language skill step through the real env.step path."
            ),
            required_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "execute_calvin_language_skill",
            ],
            required_ok_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "execute_calvin_language_skill",
            ],
            backend_factory=lambda: CALVINAgentRuntimeBackend(
                CALVINRuntimeConfig(
                    sequence_id="debug_language_sequence",
                    calvin_root=str(_default_calvin_root()),
                    dataset_root=str(
                        _default_calvin_root() / "dataset/calvin_debug_dataset/training"
                    ),
                    live=True,
                    show_gui=False,
                    use_egl=False,
                    policy_backend="calvin_noop_policy_step",
                ),
                policy=_calvin_noop_policy,
            ),
            seed=0,
            action_hint=(
                "Use observe_calvin_state(prompt=..., agent_context=...) and observe_calvin_cameras(query=..., agent_context=...). "
                "Call get_calvin_language_subgoal(agent_context=...) and get_calvin_runtime_context(prompt=..., query=..., agent_context=...). "
                "Record a compact plan with record_calvin_evidence. Then call "
                "execute_calvin_language_skill(subgoal=subgoal['subgoal'], horizon=1, prompt=..., query=..., agent_context=..., grounding=...). "
                "This proves the policy hook and real CALVIN env.step path; official task success is not claimed. "
                "Do not call raw env.step, task oracle, checker, demos, replay, dataset success labels, or harness_noop_step."
            ),
        ),
        "calvin_native_turn_off_led": LiveAgentSmokeCase(
            case_id="calvin_native_turn_off_led",
            benchmark_id="calvin",
            task_id="native_calvin_turn_off_led",
            objective=(
                "Turn off the CALVIN table LED by issuing agent-authored 7D relative Cartesian actions from current "
                "state and camera observations. The official task predicate remains private to the harness verifier."
            ),
            required_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "submit_calvin_action",
            ],
            required_ok_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "submit_calvin_action",
            ],
            backend_factory=lambda: CALVINAgentRuntimeBackend(
                CALVINRuntimeConfig(
                    sequence_id="native_turn_off_led",
                    calvin_root=str(_default_calvin_root()),
                    dataset_root=str(
                        get_project_paths().external_assets("calvin") / "dataset"
                    ),
                    live=True,
                    show_gui=False,
                    use_egl=False,
                    policy_backend="calvin_native_agent",
                    native_task_key="turn_off_led",
                )
            ),
            seed=0,
            action_hint=(
                "Read task/context and fresh state plus static/gripper cameras. Record compact evidence, then use "
                "action.execute with action={'control': [dx, dy, dz, droll, dpitch, dyaw, gripper]} to submit one "
                "bounded 7D relative Cartesian action at a time. Re-observe and adapt after short motion bursts. "
                "No policy checkpoint, demonstration replay, task checker, or hidden verifier is available to the agent."
            ),
            readiness_tier="official_solved_candidate",
            counts_toward_official_success=True,
            code_timeout_seconds=420.0,
        ),
        "calvin_official_mcil_turn_off_led": LiveAgentSmokeCase(
            case_id="calvin_official_mcil_turn_off_led",
            benchmark_id="calvin",
            task_id="qwen_live_calvin_official_mcil_turn_off_led",
            objective=(
                "Execute the official CALVIN language instruction using only current state, camera, language, and runtime "
                "context evidence; official success remains private to the harness verifier."
            ),
            required_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "execute_calvin_language_skill",
            ],
            required_ok_primitives=[
                "observe_calvin_state",
                "observe_calvin_cameras",
                "get_calvin_language_subgoal",
                "get_calvin_runtime_context",
                "record_calvin_evidence",
                "execute_calvin_language_skill",
            ],
            backend_factory=lambda: CALVINAgentRuntimeBackend(
                CALVINRuntimeConfig(
                    sequence_id="task_D_D_language_sequence",
                    calvin_root=str(_default_calvin_root()),
                    dataset_root=str(
                        _default_calvin_root() / "dataset/task_D_D/training"
                    ),
                    live=True,
                    show_gui=False,
                    use_egl=False,
                    policy_backend="calvin_official_mcil",
                    policy_train_folder=str(_default_calvin_policy_root()),
                    policy_checkpoint=str(
                        _default_calvin_policy_root() / "mcil_baseline.ckpt"
                    ),
                    policy_dataset_path=str(
                        _default_calvin_root() / "dataset/task_D_D"
                    ),
                    policy_device_id=0,
                    official_eval_sequence_index=1,
                    official_eval_num_workers=4,
                )
            ),
            seed=0,
            action_hint=(
                "Read observe_calvin_state, observe_calvin_cameras, get_calvin_language_subgoal, and "
                "get_calvin_runtime_context. Derive the grounding dictionary and execution horizon from the returned language, "
                "observation, and runtime budget evidence. Record that derivation with record_calvin_evidence, then call "
                "execute_calvin_language_skill with the returned subgoal and derived arguments. Do not hardcode a subgoal, entity, "
                "horizon, or action sequence, and do not access dataset labels, evaluator state, or verifier state."
            ),
            readiness_tier="official_solved_candidate",
            counts_toward_official_success=True,
            code_timeout_seconds=420.0,
        ),
        "robocasa_state_delta_motion": LiveAgentSmokeCase(
            case_id="robocasa_state_delta_motion",
            benchmark_id="robocasa",
            task_id="qwen_live_robocasa_state_delta_motion",
            objective=(
                "Inspect the RoboCasa kitchen state, locate the mug object, record a compact action plan, "
                "then move the robot end effector toward the grounded mug through real robosuite/RoboCasa steps."
            ),
            required_primitives=[
                "observe_robocasa_kitchen_state",
                "locate_robocasa_object",
                "record_robocasa_evidence",
                "move_robocasa_ee_to",
            ],
            required_ok_primitives=[
                "observe_robocasa_kitchen_state",
                "locate_robocasa_object",
                "record_robocasa_evidence",
                "move_robocasa_ee_to",
            ],
            backend_factory=lambda: (
                _prepare_robocasa_state_only_runtime()
                or RoboCasaAgentRuntimeBackend(
                    RoboCasaRuntimeConfig(
                        env_id="robocasa/CoffeeSetupMug",
                        split="pretrain",
                        robot="PandaOmron",
                        camera_names=[],
                        camera_depths=False,
                        live=True,
                        use_gymnasium_wrapper=False,
                        asset_cache_dir=str(_default_robocasa_asset_cache()),
                        motion_backend="robocasa_state_delta_motion",
                    ),
                    skill_executor=_robocasa_state_delta_skill_executor,
                )
            ),
            seed=0,
            action_hint=(
                "This is a direct robosuite state-only RoboCasa case: camera/RGB-D observations are disabled. "
                "Use observe_robocasa_kitchen_state(prompt=..., query='mug object state', agent_context=...). "
                "Then call locate_robocasa_object(object_name='obj', query='the mug object', agent_context=...). "
                "Record a compact plan with record_robocasa_evidence. Finally call move_robocasa_ee_to("
                "target_name='obj', offset=[0.0, 0.0, 0.12], horizon=20, strategy='robocasa_state_delta_motion', "
                "prompt=..., query=..., agent_context=...). This executes real env.step calls through a state-grounded motion primitive; "
                "official task success is not claimed. Do not call raw env.step, run_minimal_rollout, oracle, checker, demos, replay, or _check_success."
            ),
        ),
        "robocasa_rgbd_grasp_place": LiveAgentSmokeCase(
            case_id="robocasa_rgbd_grasp_place",
            benchmark_id="robocasa",
            task_id="qwen_live_robocasa_rgbd_grasp_place",
            objective=(
                "Inspect real RoboCasa RGB-D and state evidence for CoffeeSetupMug, ground the mug and coffee-machine "
                "dispenser from task language, record the subgoal plan, then execute grasp and place primitives through "
                "real robosuite/RoboCasa env.step calls."
            ),
            required_primitives=[
                "observe_robocasa_kitchen_state",
                "observe_robocasa_rgbd",
                "locate_robocasa_object",
                "locate_robocasa_fixture",
                "record_robocasa_evidence",
                "grasp_robocasa_object",
                "place_robocasa_object_at",
            ],
            required_ok_primitives=[
                "observe_robocasa_kitchen_state",
                "observe_robocasa_rgbd",
                "locate_robocasa_object",
                "locate_robocasa_fixture",
                "record_robocasa_evidence",
                "grasp_robocasa_object",
                "place_robocasa_object_at",
            ],
            backend_factory=lambda: (
                _prepare_robocasa_state_only_runtime()
                or RoboCasaAgentRuntimeBackend(
                    RoboCasaRuntimeConfig(
                        env_id="robocasa/CoffeeSetupMug",
                        split="pretrain",
                        robot="PandaOmron",
                        live=True,
                        use_gymnasium_wrapper=True,
                        asset_cache_dir=str(_default_robocasa_asset_cache()),
                        motion_backend="robocasa_state_delta_motion",
                    )
                )
            ),
            seed=0,
            action_hint=(
                "Use observe_robocasa_kitchen_state(prompt=..., query='mug and coffee machine task state', agent_context=...) "
                "and observe_robocasa_rgbd(prompt=..., query='RGB-D views for mug and dispenser grounding', agent_context=...). "
                "Call locate_robocasa_object(object_name=None, query='the mug named in the benchmark task', agent_context=...) and "
                "locate_robocasa_fixture(fixture_name=None, query='coffee machine dispenser target', agent_context=...). "
                "Record the selected object/fixture and visual/runtime evidence. For the live CoffeeSetupMug case, use the internal "
                "RoboCasa object name object_name='obj' for action primitives after grounding the task referent as mug. "
                "Call grasp_robocasa_object(object_name='obj', offset=[0.0,0.0,0.03], horizon=20, strategy='robocasa_state_delta_motion', ...) "
                "then place_robocasa_object_at(object_name='obj', target_name='coffeemachine_main_group_1', relation='under', "
                "horizon=30, strategy='robocasa_state_delta_motion', ...). Do not call raw env.step, run_minimal_rollout, "
                "oracle, checker, demos, replay, or _check_success; official task success is verifier-side."
            ),
        ),
        "robocasa_start_coffee_machine_button": LiveAgentSmokeCase(
            case_id="robocasa_start_coffee_machine_button",
            benchmark_id="robocasa",
            task_id="qwen_live_robocasa_start_coffee_machine_button",
            objective=(
                "Inspect the real RoboCasa StartCoffeeMachine RGB-D/state evidence, ground the coffee-machine "
                "start button, record the plan, then press the button through the public fixture-button "
                "primitive until the harness-side RoboCasa verifier reports official task success."
            ),
            required_primitives=[
                "observe_robocasa_kitchen_state",
                "observe_robocasa_rgbd",
                "inspect_robocasa_fixture",
                "locate_robocasa_fixture",
                "inspect_robocasa_affordance",
                "record_robocasa_evidence",
                "press_robocasa_fixture_button",
            ],
            required_ok_primitives=[
                "observe_robocasa_kitchen_state",
                "observe_robocasa_rgbd",
                "inspect_robocasa_fixture",
                "locate_robocasa_fixture",
                "inspect_robocasa_affordance",
                "record_robocasa_evidence",
                "press_robocasa_fixture_button",
            ],
            backend_factory=lambda: (
                _prepare_robocasa_state_only_runtime()
                or RoboCasaAgentRuntimeBackend(
                    RoboCasaRuntimeConfig(
                        env_id="robocasa/StartCoffeeMachine",
                        split="pretrain",
                        robot="PandaOmron",
                        live=True,
                        use_gymnasium_wrapper=False,
                        camera_names=[
                            "robot0_agentview_left",
                            "robot0_agentview_right",
                            "robot0_eye_in_hand",
                        ],
                        camera_depths=True,
                        asset_cache_dir=str(_default_robocasa_asset_cache()),
                        motion_backend="robocasa_state_delta_motion",
                    )
                )
            ),
            seed=0,
            code_timeout_seconds=180.0,
            action_hint=(
                "Read observe_robocasa_kitchen_state and observe_robocasa_rgbd, then use locate_robocasa_fixture and "
                "inspect_robocasa_fixture on candidates returned by the observations. Inspect the selected control with "
                "inspect_robocasa_affordance. Derive fixture_name, button_name or button_position, controller parameters, and "
                "press timing from that public evidence and the primitive schema; record them with record_robocasa_evidence before "
                "calling press_robocasa_fixture_button. Do not hardcode fixture IDs, button IDs, offsets, controller profiles, or "
                "waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "robocasa365_turn_on_microwave_button": LiveAgentSmokeCase(
            case_id="robocasa365_turn_on_microwave_button",
            benchmark_id="robocasa365",
            task_id="qwen_live_robocasa365_turn_on_microwave_button",
            objective=(
                "Inspect the real RoboCasa365 TurnOnMicrowave task context and RGB-D/state evidence, ground the "
                "microwave fixture and available button affordance, record the plan, then press the button through "
                "the suite-specific fixture-button primitive. The final qwen-generated code-file replay reaches the "
                "harness-side official verifier through public primitives only; private checker/success fields remain "
                "outside the agent interface."
            ),
            required_primitives=[
                "get_robocasa365_task_context",
                "observe_robocasa365_kitchen_state",
                "observe_robocasa365_rgbd",
                "inspect_robocasa365_fixture",
                "locate_robocasa365_fixture",
                "inspect_robocasa365_affordance",
                "record_robocasa365_evidence",
                "press_robocasa365_fixture_button",
            ],
            required_ok_primitives=[
                "get_robocasa365_task_context",
                "observe_robocasa365_kitchen_state",
                "observe_robocasa365_rgbd",
                "inspect_robocasa365_fixture",
                "locate_robocasa365_fixture",
                "inspect_robocasa365_affordance",
                "record_robocasa365_evidence",
                "press_robocasa365_fixture_button",
            ],
            backend_factory=lambda: (
                _prepare_robocasa_state_only_runtime()
                or RoboCasa365AgentRuntimeBackend(
                    RoboCasa365RuntimeConfig(
                        env_id="robocasa/TurnOnMicrowave",
                        split="target",
                        robot="PandaOmron",
                        live=True,
                        use_gymnasium_wrapper=False,
                        camera_names=[
                            "robot0_agentview_left",
                            "robot0_agentview_right",
                            "robot0_eye_in_hand",
                        ],
                        camera_depths=True,
                        asset_cache_dir=str(_default_robocasa_asset_cache()),
                        motion_backend="robocasa_state_delta_motion",
                        dataset_root=str(
                            _default_robocasa365_dataset_root(
                                "TurnOnMicrowave", "20250813"
                            )
                        ),
                        episode_id="episode_000000",
                        task_family="TurnOnMicrowave",
                        package_probe_python=_default_robocasa365_probe_python(),
                    )
                )
            ),
            seed=0,
            code_timeout_seconds=240.0,
            readiness_tier="official_success",
            counts_toward_official_success=True,
            action_hint=(
                "Read get_robocasa365_task_context, observe_robocasa365_kitchen_state, and observe_robocasa365_rgbd. Use "
                "locate_robocasa365_fixture and inspect_robocasa365_fixture on candidates present in those observations, then "
                "inspect the selected control with inspect_robocasa365_affordance. Derive fixture_name, button_name or position, "
                "motion parameters, and timing from the exposed episode evidence and primitive schema. Record the derivation with "
                "record_robocasa365_evidence and call press_robocasa365_fixture_button with those values. Do not hardcode fixture "
                "IDs, button IDs, offsets, controller profiles, or waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "capx_current_interface_live": LiveAgentSmokeCase(
            case_id="capx_current_interface_live",
            benchmark_id="capx",
            task_id="qwen_current_interface_capx_cube_stack",
            objective=(
                "Pick up the red cube, stack it on the green cube, and release it in the current CaP-X scene. "
                "Use only the task goal, current public observation, native visual evidence, and general CaP-X primitives."
            ),
            required_primitives=[
                "observe_capx_scene",
                "enumerate_capx_objects",
                "capx_get_object_pose",
                "capx_sample_grasp_pose",
                "compose_capx_geometry",
                "record_capx_evidence",
                "submit_capx_action",
            ],
            required_ok_primitives=[
                "observe_capx_scene",
                "enumerate_capx_objects",
                "capx_get_object_pose",
                "capx_sample_grasp_pose",
                "compose_capx_geometry",
                "record_capx_evidence",
                "submit_capx_action",
            ],
            backend_factory=lambda: CapXCurrentInterfaceSnapshotBackend(),
            seed=0,
            action_hint=(
                "Derive object descriptions, poses, extents, grasp orientation, clearances, action parameters, and ordering "
                "from the literal goal and public primitive outputs; do not consume the upstream task recipe."
            ),
            readiness_tier="official_success",
            counts_toward_official_success=True,
            harness_only_verifier=True,
        ),
        "capx_comparator_alias_boundary": LiveAgentSmokeCase(
            case_id="capx_comparator_alias_boundary",
            benchmark_id="capx",
            task_id="w4_capx_comparator_trace",
            objective="Exercise the legacy CaP-X comparator scaffold boundary without claiming live task success.",
            required_primitives=[
                "get_capx_prompt",
                "get_capx_available_apis",
                "capx_get_object_pose",
                "capx_sample_grasp_pose",
                "record_capx_evidence",
                "capx_goto_pose",
                "capx_close_gripper",
            ],
            required_ok_primitives=[
                "get_capx_prompt",
                "get_capx_available_apis",
                "capx_get_object_pose",
                "capx_sample_grasp_pose",
                "record_capx_evidence",
                "capx_goto_pose",
                "capx_close_gripper",
            ],
            backend_factory=CapXComparatorSmokeBackend,
            action_hint=(
                "Derive object names from the current prompt and API inventory, inspect their public poses and grasp samples, "
                "express intermediate waypoints in agent code, call record_capx_evidence for the derivation, then exercise "
                "only the exposed comparator aliases."
            ),
        ),
        "robotwin2_place_empty_cup": LiveAgentSmokeCase(
            case_id="robotwin2_place_empty_cup",
            benchmark_id="robotwin2",
            task_id="qwen_live_robotwin2_place_empty_cup",
            objective=(
                "Solve the official RoboTwin2 place_empty_cup task by deriving actors, arm choice, poses, gripper commands, "
                "and end-effector actions from exposed scene and RGB-D evidence."
            ),
            required_primitives=[
                "observe_robotwin2_scene",
                "observe_robotwin2_visual",
                "check_robotwin2_asset_readiness",
                "locate_robotwin2_actor",
                "set_robotwin2_gripper",
                "move_robotwin2_arm",
                "submit_robotwin2_ee_action",
                "record_robotwin2_evidence",
            ],
            required_ok_primitives=[
                "observe_robotwin2_scene",
                "observe_robotwin2_visual",
                "check_robotwin2_asset_readiness",
                "locate_robotwin2_actor",
                "set_robotwin2_gripper",
                "move_robotwin2_arm",
                "submit_robotwin2_ee_action",
                "record_robotwin2_evidence",
            ],
            backend_factory=lambda: RoboTwin2AgentRuntimeBackend(
                RoboTwin2RuntimeConfig(
                    repo_path=str(_default_robotwin2_repo()),
                    assets_path=str(get_project_paths().external_assets("robotwin2")),
                    task_name="place_empty_cup",
                    task_config="demo_clean",
                    live=True,
                    render_freq=0,
                    action_type="ee",
                )
            ),
            seed=0,
            code_timeout_seconds=300.0,
            action_hint=(
                "Read check_robotwin2_asset_readiness, observe_robotwin2_scene, and observe_robotwin2_visual. Use "
                "locate_robotwin2_actor only to inspect actor candidates named by the official instruction. Derive the arm, target "
                "poses or deltas, gripper commands, and full end-effector action vector from current robot and actor observations. "
                "Execute only those derived values through move_robotwin2_arm, set_robotwin2_gripper, and "
                "submit_robotwin2_ee_action, and record the evidence with record_robotwin2_evidence. Do not hardcode actor IDs, "
                "arm selection, points, poses, action vectors, or waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "rlbench_reach_target_step": LiveAgentSmokeCase(
            case_id="rlbench_reach_target_step",
            benchmark_id="rlbench",
            task_id="qwen_live_rlbench_reach_target_step",
            objective=(
                "Read the RLBench ReachTarget language and observation evidence, ground the target, "
                "record the agent's action rationale, then execute one real RLBench task.step action "
                "through the EndEffectorPoseViaPlanning pose-action path."
            ),
            required_primitives=[
                "observe_rlbench_scene",
                "inspect_rlbench_visual_evidence",
                "ground_rlbench_target",
                "record_rlbench_evidence",
                "move_rlbench_arm_to",
            ],
            required_ok_primitives=[
                "observe_rlbench_scene",
                "inspect_rlbench_visual_evidence",
                "ground_rlbench_target",
                "record_rlbench_evidence",
                "move_rlbench_arm_to",
            ],
            backend_factory=lambda: RLBenchAgentRuntimeBackend(
                RLBenchRuntimeConfig(
                    task_name="ReachTarget",
                    action_mode="MoveArmThenGripper(EndEffectorPoseViaPlanning, Discrete)",
                    live=True,
                    coppeliasim_root=os.environ.get("COPPELIASIM_ROOT"),
                    headless=True,
                    action_skill_backend="rlbench_pose_action",
                )
            ),
            readiness_tier="official_success",
            counts_toward_official_success=True,
            harness_only_verifier=True,
            seed=0,
            action_hint=(
                "Use observe_rlbench_scene and inspect_rlbench_visual_evidence first, then ground the target from "
                "public language and pose evidence. Record a compact plan with record_rlbench_evidence. Derive the "
                "7D target_pose from returned public evidence and call move_rlbench_arm_to; the adapter will execute "
                "the real CoppeliaSim/RLBench task.step path through rlbench_pose_action without exposing verifier state."
            ),
        ),
        "rlbench_reach_target_motion": LiveAgentSmokeCase(
            case_id="rlbench_reach_target_motion",
            benchmark_id="rlbench",
            task_id="qwen_live_rlbench_reach_target_motion",
            objective=(
                "Read the RLBench ReachTarget language and observation evidence, ground the target, "
                "record the agent's motion rationale, then move the arm to the target using the real "
                "EndEffectorPoseViaPlanning action path."
            ),
            required_primitives=[
                "observe_rlbench_scene",
                "inspect_rlbench_visual_evidence",
                "ground_rlbench_target",
                "record_rlbench_evidence",
                "move_rlbench_arm_to",
            ],
            required_ok_primitives=[
                "observe_rlbench_scene",
                "inspect_rlbench_visual_evidence",
                "ground_rlbench_target",
                "record_rlbench_evidence",
                "move_rlbench_arm_to",
            ],
            backend_factory=lambda: RLBenchAgentRuntimeBackend(
                RLBenchRuntimeConfig(
                    task_name="ReachTarget",
                    action_mode="MoveArmThenGripper(EndEffectorPoseViaPlanning, Discrete)",
                    live=True,
                    coppeliasim_root=os.environ.get("COPPELIASIM_ROOT"),
                    headless=True,
                    action_skill_backend="rlbench_pose_action",
                )
            ),
            seed=0,
            action_hint=(
                "Read observe_rlbench_scene and inspect_rlbench_visual_evidence, then call ground_rlbench_target without a "
                "fixed target name and compare its returned candidates with the official language instruction. Derive target_pose "
                "from the exposed visual or low-dimensional observation, record that grounding with record_rlbench_evidence, and "
                "pass the derived pose to move_rlbench_arm_to. Do not hardcode object names, poses, strategies, or waypoints, and "
                "do not access demos, evaluator state, or verifier state."
            ),
        ),
        "robodojo_stack_bowls_source_boundary": LiveAgentSmokeCase(
            case_id="robodojo_stack_bowls_source_boundary",
            benchmark_id="robodojo",
            task_id="qwen_live_robodojo_stack_bowls_source_boundary",
            objective=(
                "Read a real RoboDojo runtime report, inspect public object/robot geometry and gripper collision geometry, "
                "derive end-effector path points from those observations, compile them into schema-valid low-level actions, record evidence, "
                "and submit those actions without accessing rewards, demos, leaderboard summaries, or private verifier code."
            ),
            required_primitives=[
                "observe_robodojo_runtime_report",
                "get_robodojo_robot_action_schema",
                "measure_robodojo_public_geometry",
                "inspect_robodojo_collision_geometry",
                "compile_robodojo_ee_path",
                "record_robodojo_evidence",
                "submit_robodojo_low_level_actions",
            ],
            required_ok_primitives=[
                "observe_robodojo_runtime_report",
                "get_robodojo_robot_action_schema",
                "measure_robodojo_public_geometry",
                "inspect_robodojo_collision_geometry",
                "compile_robodojo_ee_path",
                "record_robodojo_evidence",
                "submit_robodojo_low_level_actions",
            ],
            backend_factory=lambda: RoboDojoAgentRuntimeBackend(
                RoboDojoRuntimeConfig(
                    repo_path=str(_default_robodojo_repo()),
                    task_name="stack_bowls",
                    env_cfg="arx_x5",
                    live=False,
                    env_kwargs={
                        "latest_live_report_path": str(
                            resolve_project_root()
                            / "reports/robodojo_current_runtime_visual_boundary_9988837.json"
                        )
                    },
                )
            ),
            action_hint=(
                "Read observe_robodojo_runtime_report and get_robodojo_robot_action_schema. Derive object labels and an arm from "
                "the public runtime state, inspect them with measure_robodojo_public_geometry and "
                "inspect_robodojo_collision_geometry, and construct every position, orientation, gripper value, and repeat count "
                "for compile_robodojo_ee_path from those measurements. Record the derivation with record_robodojo_evidence and "
                "submit the compiled actions with submit_robodojo_low_level_actions. Do not hardcode labels, arm selection, poses, "
                "contact frames, action vectors, or waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "robodojo_general_pickup_current_interface": LiveAgentSmokeCase(
            case_id="robodojo_general_pickup_current_interface",
            benchmark_id="robodojo",
            task_id="qwen_current_interface_robodojo_general_pickup",
            objective=(
                "Solve the RoboDojo general_pickup instruction from the benchmark-native three-view RGB observation, "
                "public scene/robot geometry, and public URDF collision geometry. Select the object, arm, contact axes, "
                "lift displacement, gripper openings, and low-level actions from those observations, then submit them."
            ),
            required_primitives=[
                "observe_robodojo_runtime_report",
                "observe_robodojo_visual",
                "get_robodojo_robot_action_schema",
                "measure_robodojo_public_geometry",
                "inspect_robodojo_collision_geometry",
                "compile_robodojo_contact_lift_actions",
                "record_robodojo_evidence",
                "submit_robodojo_low_level_actions",
            ],
            required_ok_primitives=[
                "observe_robodojo_runtime_report",
                "observe_robodojo_visual",
                "get_robodojo_robot_action_schema",
                "measure_robodojo_public_geometry",
                "inspect_robodojo_collision_geometry",
                "compile_robodojo_contact_lift_actions",
                "record_robodojo_evidence",
                "submit_robodojo_low_level_actions",
            ],
            backend_factory=lambda: RoboDojoAgentRuntimeBackend(
                RoboDojoRuntimeConfig(
                    repo_path=str(_default_robodojo_repo()),
                    task_name="general_pickup",
                    env_cfg="arx_x5",
                    # This case is part of the strict official-success set.  A
                    # source-boundary backend can replay an old report, but it
                    # cannot create the same-episode Isaac proxy required by
                    # the live official verifier.
                    live=True,
                    env_kwargs={
                        "live_start_timeout_seconds": 1200.0,
                        "latest_live_report_path": os.getenv(
                            "ROBODOJO_OPENHANDS_RUNTIME_REPORT",
                            str(
                                resolve_project_root()
                                / "reports/robodojo_general_pickup_current_interface_official_visual_9988945.json"
                            ),
                        ),
                    },
                )
            ),
            action_hint=(
                "Use only the current instruction, native RGB evidence refs, public object/robot poses, and public collision "
                "geometry. Derive every compiler parameter and submit the compiled actions with the same visual evidence refs."
            ),
            readiness_tier="official_replay_candidate",
            counts_toward_official_success=False,
            harness_only_verifier=True,
        ),
        "robowits_eval_json_boundary": LiveAgentSmokeCase(
            case_id="robowits_eval_json_boundary",
            benchmark_id="robowits",
            task_id="01",
            objective=(
                "Read the official RoboWits task context and live observations, derive relevant entities, poses, arm motion, "
                "and gripper control from exposed evidence, execute the derived control, and record its observed outcome."
            ),
            required_primitives=[
                "get_robowits_task_context",
                "observe_robowits_state",
                "inspect_robowits_scene_or_tool",
                "inspect_robowits_live_observation",
                "inspect_robowits_object_poses",
                "query_robowits_motion",
                "execute_robowits_ee_control",
                "inspect_robowits_motion_outcome",
                "record_robowits_evidence",
            ],
            required_ok_primitives=[
                "get_robowits_task_context",
                "observe_robowits_state",
                "inspect_robowits_scene_or_tool",
                "inspect_robowits_live_observation",
                "inspect_robowits_object_poses",
                "query_robowits_motion",
                "execute_robowits_ee_control",
                "inspect_robowits_motion_outcome",
                "record_robowits_evidence",
            ],
            backend_factory=lambda: RoboWitsAgentRuntimeBackend(
                RoboWitsRuntimeConfig(
                    repo_path=str(_default_robowits_repo()),
                    task_id="01",
                    dataset_split="eval_dataset_50",
                    episode_index=0,
                    live=False,
                )
            ),
            action_hint=(
                "Read get_robowits_task_context, observe_robowits_state, inspect_robowits_scene_or_tool, and "
                "inspect_robowits_live_observation. Derive entity names from the current instruction and inspect their poses with "
                "inspect_robowits_object_poses. Derive target_position, arm, and orientation from those observations; validate the "
                "candidate with query_robowits_motion, execute it through execute_robowits_ee_control, and inspect the result with "
                "inspect_robowits_motion_outcome. Record the evidence with record_robowits_evidence. Do not hardcode entity names, "
                "arm selection, poses, orientations, gripper widths, or waypoints, and do not access evaluator or verifier state."
            ),
        ),
        "robowits_stack_cube_official": LiveAgentSmokeCase(
            case_id="robowits_stack_cube_official",
            benchmark_id="robowits",
            task_id="14_06",
            objective=(
                "Solve the current RoboWits stacking task from its official instruction and live Genesis RGB/state evidence. "
                "Derive all entity choices, poses, arm controls, gripper commands, and replanning decisions from public "
                "observations, then stop when the public episode lifecycle terminates."
            ),
            required_primitives=[
                "get_robowits_task_context",
                "observe_robowits_state",
                "inspect_robowits_scene_or_tool",
                "inspect_robowits_live_observation",
                "inspect_robowits_object_poses",
                "query_robowits_motion",
                "execute_robowits_ee_control",
                "inspect_robowits_motion_outcome",
                "record_robowits_evidence",
            ],
            required_ok_primitives=[
                "get_robowits_task_context",
                "observe_robowits_state",
                "inspect_robowits_scene_or_tool",
                "inspect_robowits_live_observation",
                "inspect_robowits_object_poses",
                "query_robowits_motion",
                "execute_robowits_ee_control",
                "inspect_robowits_motion_outcome",
                "record_robowits_evidence",
            ],
            backend_factory=lambda: RoboWitsAgentRuntimeBackend(
                RoboWitsRuntimeConfig(
                    repo_path=str(_default_robowits_repo()),
                    task_id="14_06",
                    dataset_split="eval_dataset_50",
                    episode_index=0,
                    live=True,
                    device=os.environ.get("ROBOWITS_DEVICE", "cuda"),
                    genesis_backend=os.environ.get("GENESIS_BACKEND", "cuda"),
                    # Quadrants' graph solver wheel does not carry SM90
                    # graph-control fatbins. Genesis' official monolithic
                    # constraint solver is portable across the H200 path.
                    genesis_constraint_solver=os.environ.get(
                        "EMBODIED_ARENA_GENESIS_CONSTRAINT_SOLVER", "monolithic"
                    ),
                    # A complete public stacking solution expands to roughly
                    # 420 bounded EE control steps.  The upstream default of
                    # 100 truncates the episode after the first placement and
                    # prevents the official checker from ever becoming true.
                    env_kwargs={"show_viewer": False, "max_episode_steps": 800},
                )
            ),
            seed=3,
            code_timeout_seconds=900.0,
            action_hint=(
                "Use the official task context and native live observation primitives first. Derive every selected entity, "
                "target position, arm, orientation, gripper width, duration, and any recovery from current visual/state evidence. "
                "Pass visual evidence handles to motion primitives, inspect outcomes between controls, and record the derivation. "
                "Do not hardcode object names, poses, waypoints, controller profiles, or a task recipe, and do not access "
                "reward, success, evaluator, verifier, demos, or replay."
            ),
        ),
    }
    official_case_ids = [
        "alfworld_pick_knife_to_sidetable",
        "vimabench_visual_manipulation",
        "cliport_place_red_in_green",
        "cliport_stack_pyramid",
        "vlabench_select_toy_skilllib",
        "scienceworld_find_non_living_thing",
        "maniskill_pickcube_pd_ee_delta_pos",
        "robocasa_start_coffee_machine_button",
        "capx_current_interface_live",
        "rlbench_reach_target_motion",
        "rlbench_reach_target_step",
        "calvin_official_mcil_turn_off_led",
        "robocasa365_turn_on_microwave_button",
        "robotwin2_place_empty_cup",
        "mmsi_spatial_smoke",
        "spatialclaw_erqa_real_visual",
        "behavior1k_full_bddl_toggle",
        "robowits_stack_cube_official",
        "robodojo_general_pickup_current_interface",
    ]
    active_operation_case_ids = {
        "maniskill_pickcube_pd_ee_delta_pos",
        "vimabench_visual_manipulation",
        "cliport_place_red_in_green",
        "vlabench_select_toy_skilllib",
        "robocasa_start_coffee_machine_button",
        "capx_current_interface_live",
        "rlbench_reach_target_motion",
        "calvin_official_mcil_turn_off_led",
        "behavior1k_full_bddl_toggle",
        "robocasa365_turn_on_microwave_button",
        "robowits_eval_json_boundary",
        "robowits_stack_cube_official",
        "robotwin2_place_empty_cup",
        "robodojo_stack_bowls_source_boundary",
        "robodojo_general_pickup_current_interface",
    }
    for case_id in official_case_ids:
        if case_id in cases:
            cases[case_id].readiness_tier = "official_success"
            cases[case_id].counts_toward_official_success = True
            cases[case_id].harness_only_verifier = True
    for case_id in active_operation_case_ids:
        cases[case_id].harness_only_verifier = True
    for case_id, case in cases.items():
        if (
            case_id not in official_case_ids
            and not case.counts_toward_official_success
            and case.readiness_tier != "policy_skill_contract"
        ):
            case.readiness_tier = "boundary_ready"
            case.counts_toward_official_success = False

    groups = {
        "all": list(cases),
        "official_solved": official_case_ids,
        "final_ready": official_case_ids,
        "policy_skill_contract": [
            case_id
            for case_id, case in cases.items()
            if case.readiness_tier == "policy_skill_contract"
        ],
        "boundary_ready": [
            case_id
            for case_id, case in cases.items()
            if case.readiness_tier == "boundary_ready"
        ],
    }
    if not case_ids:
        case_ids = ["all"]
    expanded_case_ids: list[str] = []
    for case_id in case_ids:
        expanded_case_ids.extend(groups.get(case_id, [case_id]))
    deduped_case_ids = list(dict.fromkeys(expanded_case_ids))
    if deduped_case_ids == groups["all"]:
        return list(cases.values())
    unknown = [case_id for case_id in deduped_case_ids if case_id not in cases]
    if unknown:
        raise KeyError(f"Unknown live agent smoke case ids: {unknown}")
    return [cases[case_id] for case_id in deduped_case_ids]


class CapXComparatorSmokeBackend(W4BenchmarkBackend):
    """W4 CaP-X comparator scaffold with live task success kept separate."""

    def reset(
        self,
        task_id: str,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> TaskSpec:
        return super().reset("w4_capx_comparator_trace", seed=seed, config=config)

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        super().verify(scope=scope, **kwargs)
        result = VerificationResult(
            ok=False,
            scope=scope,
            message="CaP-X comparator scaffold executed; official task success requires the upstream CaP-X runner or an injected live public API session.",
            metrics={"agent_scaffold_executed": 1.0, "official_success": 0.0},
        )
        self.record_event("verifier_call", {"benchmark_id": "capx", **result.to_dict()})
        return result


class _CapXSnapshotObservationEnv:
    def __init__(self, observation: dict[str, Any]) -> None:
        self._observation = deepcopy(observation)

    def get_observation(self) -> dict[str, Any]:
        return deepcopy(self._observation)


class _CapXSnapshotLiveAPI:
    """Replay the public CaP-X API surface against one audited reset observation."""

    def __init__(self, observation_env: _CapXSnapshotObservationEnv) -> None:
        self._env = observation_env
        self.action_log: list[dict[str, Any]] = []

    def get_object_pose(
        self, object_name: str, return_bbox_extent: bool = False
    ) -> dict[str, Any]:
        pose = self._pose_for_object(object_name)
        if return_bbox_extent:
            pose["bbox_extent"] = pose.get("extent", [0.04, 0.04, 0.04])
        return pose

    def sample_grasp_pose(self, object_name: str) -> dict[str, Any]:
        pose = self._pose_for_object(object_name)
        return {
            "position": pose["position"],
            "quaternion_wxyz": pose["quaternion_wxyz"],
            "source_object": pose["source_object"],
            "replay_boundary": "public_reset_observation",
        }

    def goto_pose(
        self,
        position: list[float],
        quaternion_wxyz: list[float],
        *,
        z_approach: float = 0.0,
    ) -> dict[str, Any]:
        return self._record_action(
            "goto_pose",
            {
                "position": [float(item) for item in position],
                "quaternion_wxyz": [float(item) for item in quaternion_wxyz],
                "z_approach": float(z_approach),
            },
        )

    def open_gripper(self) -> dict[str, Any]:
        return self._record_action("open_gripper", {})

    def close_gripper(self) -> dict[str, Any]:
        return self._record_action("close_gripper", {})

    def _record_action(self, action: str, parameters: dict[str, Any]) -> dict[str, Any]:
        record = {
            "action": action,
            "parameters": deepcopy(parameters),
            "executed": True,
            "execution_boundary": "audited_capx_current_interface_replay",
            "official_verifier_exposed": False,
        }
        self.action_log.append(record)
        return deepcopy(record)

    def _pose_for_object(self, object_name: str) -> dict[str, Any]:
        observation = self._env.get_observation()
        query = _capx_snapshot_slug(object_name)
        aliases = {
            "cubea": ["cubeA", "cubeA_pos", "primary"],
            "redcube": ["cubeA", "cubeA_pos", "primary"],
            "source": ["cubeA", "cubeA_pos", "primary"],
            "primary": ["cubeA", "cubeA_pos", "primary"],
            "cubeb": ["cubeB", "cubeB_pos", "secondary"],
            "greencube": ["cubeB", "cubeB_pos", "secondary"],
            "target": ["cubeB", "cubeB_pos", "secondary"],
            "secondary": ["cubeB", "cubeB_pos", "secondary"],
        }
        candidates = aliases.get(query, [object_name, query])
        for candidate in candidates:
            pose = self._pose_from_observation(observation, candidate)
            if pose is not None:
                return pose
        raise KeyError(
            f"object pose not found in public reset observation: {object_name}"
        )

    def _pose_from_observation(
        self, observation: dict[str, Any], candidate: str
    ) -> dict[str, Any] | None:
        cube_poses = observation.get("cube_poses")
        if isinstance(cube_poses, dict) and candidate in cube_poses:
            pose = _capx_snapshot_pose(cube_poses[candidate])
            if pose is not None:
                pose["source_object"] = candidate
                return pose
        if candidate.endswith("_pos"):
            base = candidate[: -len("_pos")]
        else:
            base = candidate
        position = (
            observation.get(f"{base}_pos") if isinstance(observation, dict) else None
        )
        quaternion = (
            observation.get(f"{base}_quat") if isinstance(observation, dict) else None
        )
        if position is None and candidate in observation:
            position = observation.get(candidate)
        pose = _capx_snapshot_pose(
            [*(position or []), *(quaternion or [0.0, 0.0, 1.0, 0.0])]
        )
        if pose is None:
            return None
        pose["source_object"] = base
        return pose


def _capx_snapshot_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _capx_snapshot_pose(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) not in {3, 7, 10}:
        return None
    if not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in value
    ):
        return None
    position = [float(item) for item in value[:3]]
    quaternion = (
        [float(item) for item in value[3:7]]
        if len(value) >= 7
        else [0.0, 0.0, 1.0, 0.0]
    )
    pose: dict[str, Any] = {
        "position": position,
        "quaternion_wxyz": quaternion,
        "pose": [*position, *quaternion],
        "extent": [0.04, 0.04, 0.04],
    }
    if len(value) == 10:
        pose["extent"] = [float(item) for item in value[7:10]]
    return pose


class CapXCurrentInterfaceSnapshotBackend(EmbodiedBackend):
    """Agent view and evaluator binding for one live upstream CaP-X episode."""

    def __init__(
        self,
        reset_asset_path: str | Path | None = None,
        capx_root: str | Path | None = None,
        config_path: str | Path | None = None,
        session_factory: Callable[..., CapXLiveSession] = open_capx_live_session,
    ) -> None:
        repo_root = resolve_project_root()
        self.repository_root = repo_root
        self.reset_asset_path = Path(
            reset_asset_path
            or os.getenv("CAPX_CURRENT_INTERFACE_RESET_ASSET")
            or repo_root / CAPX_RESET_ASSET
        )
        self.capx_root = Path(
            capx_root
            or os.getenv("CAPX_ROOT")
            or get_project_paths().external_upstream("capx")
        )
        self.config_path = Path(
            config_path
            or os.getenv("CAPX_CONFIG")
            or get_project_paths().external_environment("capx") / "runtime_config.yaml"
        )
        self.trace: EpisodeTrace | None = None
        self.task: TaskSpec | None = None
        self.context: Any | None = None
        self._session_factory = session_factory
        self._live_session: CapXLiveSession | None = None
        self.evaluator_owned_capx_environment: Any | None = None
        self._native_api_name: str | None = None
        self._selected_config: Path | None = None
        self._selected_seed: int | None = None
        self._native_values: dict[str, Any] = {}

    def bind_pool_coordinate(self, coordinate: dict[str, Any]) -> dict[str, Any]:
        from .capx_comparator_runtime import load_config
        variation = str(coordinate.get("variation") or "")
        config_root = self.capx_root
        if os.environ.get('ARENA_REPORTING_BENCHMARK') == 'capx_libero_pro':
            config_root = Path(os.environ['CAPX_CASE_CONFIG_ROOT'])
        source = (config_root / variation).resolve()
        if not variation.startswith("env_configs/") or not source.is_relative_to((config_root / "env_configs").resolve()):
            raise ValueError("CaPX pool coordinate must name an original env_configs file")
        cfg = load_config(source)["env"]["cfg"]
        low = cfg.get("low_level")
        activity = low.get("activity_name") if isinstance(low, dict) else None
        if cfg.get("privileged", False):
            raise ValueError("Privileged CaPX configs are excluded")
        if isinstance(low, dict) and low.get('_target_') == 'capx.envs.simulators.libero.FrankaLiberoEnv':
            suite, index = low.get('suite_name'), low.get('task_id')
            if (cfg.get('apis') != ['FrankaLiberoApi'] or low.get('privileged') is not False
                    or coordinate.get('task_id') != f'libero::{suite}::{index}'):
                raise ValueError('CaPX LIBERO config/task/API mismatch')
            selected_api = 'FrankaLiberoApi'
        elif activity:
            if coordinate.get("task_id") != f"behavior::{activity}" or cfg.get("apis") != ["R1ProControlApi"]:
                raise ValueError("CaPX BEHAVIOR config/task mismatch")
            selected_api = "R1ProControlApi"
        else:
            allowed = {"cube_lifting": "FrankaControlApi", "cube_stack": "FrankaControlApi",
                       "cube_restack": "FrankaControlApi", "spill_wipe": "FrankaControlSpillWipeApi"}
            family = Path(variation).parts[1]
            if (family not in allowed or coordinate.get("task_id") != f"robosuite::{family}"
                    or cfg.get("apis") != [allowed[family]]
                    or not isinstance(low, str) or "robosuite" not in low):
                raise ValueError("CaPX Robosuite config/task/API mismatch")
            selected_api = allowed[family]
        seed = coordinate.get("seed")
        if type(seed) is not int or seed < 0:
            raise ValueError("CaPX instance seed must be a non-negative integer")
        self._selected_config = source
        self._selected_seed = seed
        self._native_api_name = selected_api
        self.config_path = source
        return {"bound": True, "mode": "native_capx_config", "task_id": coordinate["task_id"],
                "config_file": str(source), "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "api_name": self._native_api_name, "seed": seed}

    def reset(
        self,
        task_id: str,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> TaskSpec:
        _ = config
        project_paths = get_project_paths()
        if (
            self.capx_root.resolve()
            != project_paths.external_upstream("capx").resolve()
            or self.config_path.resolve()
            != (self._selected_config or (
                project_paths.external_environment("capx") / "runtime_config.yaml"
            )).resolve()
        ):
            raise ValueError("CaP-X reset source/config path mismatch")
        if self._selected_config is not None and seed != self._selected_seed:
            raise ValueError("CaPX reset seed differs from selected native instance")
        self.evaluator_owned_capx_environment = None
        self._native_values.clear()
        if self._live_session is not None:
            self._live_session.close()
        self.context = build_capx_context(
            self.capx_root, self.config_path, "live_same_episode"
        )
        self._live_session = self._session_factory(
            self.capx_root,
            self.config_path,
            seed=seed,
            **({"api_name": self._native_api_name} if self._native_api_name else {}),
        )
        self.evaluator_owned_capx_environment = self._live_session.low_level_environment
        if self._native_api_name == "R1ProControlApi":
            from .capx_comparator_runtime import refresh_capx_live_service_readiness
            refresh_capx_live_service_readiness(self.context, self._live_session)
        self.trace = EpisodeTrace(task_id=task_id)
        self.task = TaskSpec(
            task_id=task_id,
            source=(f"capx:{self.config_path.relative_to(self.capx_root)}"
                    if self.config_path.is_relative_to(self.capx_root)
                    else f"capx-derived:{self.config_path}") if self._selected_config else "capx:franka_robosuite_cube_stack",
            instruction=self.context.prompt if self._selected_config else "Pick up the red cube, stack it on top of the green cube, and release it.",
            tags=["capx", "rgbd", self.context.route if self._selected_config else "cube_stack", "native_config" if self._selected_config else "current_interface"],
            budgets={"primitive_calls": 24, "verifier_calls": 3},
            metadata={"benchmark_id": "capx", "seed": seed},
        )
        self.record_event("reset", {"task": self.task.to_dict()})
        return self.task

    def observe(self) -> Observation:
        if self.trace is None:
            raise RuntimeError("reset first")
        raw_observation = self._public_live_observation()
        public_observation = _public_capx_observation(raw_observation)
        modality = _capx_observation_modalities(raw_observation)
        visual_refs = modality["visual_evidence_refs"]
        observation = Observation(
            step=len(self.trace.events),
            data={
                "instruction": self.task.instruction if self.task else None,
                "public_observation": public_observation,
                "native_visual": {
                    "modalities": sorted({str(ref["modality"]) for ref in visual_refs}),
                    "evidence_refs": visual_refs,
                    "real_artifacts_observed": bool(visual_refs),
                },
            },
            metadata={"benchmark_id": "capx", "real_environment_reset": True},
        )
        self.record_event("observe", observation.to_dict())
        return observation

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        if self.context is None:
            raise RuntimeError("reset first")
        cards = list_capx_comparator_primitives(self.context)
        if self._native_api_name:
            # Motion signatures differ from Franka; document only this live API.
            cards = [card for card in cards if not card.name.startswith("capx_")]
            from .capx_comparator_runtime import capx_native_api_schema
            schema = capx_native_api_schema(self._live_session.live_api)
            for card in cards:
                if card.name == "submit_capx_action":
                    card.description = "Call an original CaPX functions() entry with its documented parameters. " + json.dumps(schema)
                    card.input_schema = {"action": "original public function name", "parameters": "dict of original keyword arguments", "native_api": "true"}
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        if self.context is None:
            raise RuntimeError("reset first")
        if self._live_session is None:
            raise RuntimeError("reset first")
        if self._native_api_name and name == "get_capx_available_apis":
            from .capx_comparator_runtime import capx_native_api_schema
            return PrimitiveResult(name=name, ok=True, output={"public_api": capx_native_api_schema(self._live_session.live_api), "prompt": self.context.prompt})
        if self._native_api_name and name == "submit_capx_action":
            from .capx_comparator_runtime import call_capx_native_api
            action = str(kwargs.get("action") or "")
            if action in {"get_object_pose", "sample_grasp_pose", "locate_object_for_grasp", "get_object_3d_points_and_masks_from_language", "segment_sam3_text_prompt", "segment_sam3_point_prompt", "get_sam3_mask", "find_object_base_rotate", "find_object_torso_rotate"}:
                self._live_session.ensure_perception_sidecars(
                    require_contact_graspnet=action == "sample_grasp_pose"
                    and not getattr(self._live_session, "serial_grasp_services", False))
            with self._live_session.activate_runtime_workdir():
                result = call_capx_native_api(self._live_session.live_api, action, kwargs.get("parameters", {}), self._native_values)
            self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
            return result
        if name in {"capx_get_object_pose", "capx_sample_grasp_pose"}:
            ensure_sidecars = getattr(
                self._live_session,
                "ensure_perception_sidecars",
                None,
            )
            if (
                callable(ensure_sidecars)
                and getattr(self._live_session, "capx_root", None) is not None
            ):
                ensure_sidecars(
                    require_contact_graspnet=name == "capx_sample_grasp_pose"
                    and not getattr(self._live_session, "serial_grasp_services", False)
                )
        with self._live_session.activate_runtime_workdir():
            result = call_capx_comparator_primitive(
                self.context,
                name,
                live_api=self._live_session.live_api,
                live_env=self._live_session.low_level_environment,
                **kwargs,
            )
        self.record_event(
            "primitive_call",
            {"name": name, "kwargs": kwargs, "result": result.to_dict()},
        )
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        _ = kwargs
        evaluator = self.evaluator_owned_capx_environment
        common_metadata = {
            "source": "evaluator_owned_live_same_episode",
            "official_scoring_hidden_from_agent": True,
            "same_episode_official_object_bound": evaluator is not None,
        }
        if evaluator is None:
            result = VerificationResult(
                ok=False,
                scope=scope,
                message="Official CaP-X evaluator is unavailable; reset the live episode first.",
                metrics={"official_success": 0.0},
                metadata={**common_metadata, "verifier_available": False},
            )
        else:
            reward_fn = getattr(evaluator, "compute_reward", None)
            completed_fn = getattr(evaluator, "task_completed", None)
            if not callable(reward_fn) and not callable(completed_fn):
                result = VerificationResult(
                    ok=False,
                    scope=scope,
                    message="Official CaP-X low-level success methods are unavailable.",
                    metrics={"official_success": 0.0},
                    metadata={**common_metadata, "verifier_available": False},
                )
            else:
                try:
                    # Native TaskMetric normalizes time by control steps. Before
                    # the first control step only the original success check is defined.
                    reward_ready = not any(
                        type(metric).__name__ == "TaskMetric" and metric.timesteps == 0
                        for metric in getattr(evaluator, "metrics", ())
                    )
                    reward = float(reward_fn()) if callable(reward_fn) and reward_ready else None
                    completed = (
                        bool(completed_fn())
                        if callable(completed_fn)
                        else bool(reward == 1.0)
                    )
                    result = VerificationResult(
                        ok=completed,
                        scope=scope,
                        message=(
                            "Official CaP-X task_completed check passed."
                            if completed
                            else "Official CaP-X task_completed check did not pass."
                        ),
                        metrics={
                            "official_success": float(completed),
                            **(
                                {"official_reward": reward}
                                if reward is not None
                                else {}
                            ),
                        },
                        metadata={
                            **common_metadata,
                            "verifier_available": True,
                            "official_reward_available": reward is not None,
                            **({"official_reward_pending_reason": "no_native_control_steps"}
                               if not reward_ready else {}),
                            "official_method": (
                                "low_level.task_completed"
                                if callable(completed_fn)
                                else "low_level.compute_reward_equals_one"
                            ),
                        },
                    )
                except Exception as exc:  # noqa: BLE001 - verifier failures are evidence.
                    result = VerificationResult(
                        ok=False,
                        scope=scope,
                        message=f"Official CaP-X verifier failed: {type(exc).__name__}: {exc}",
                        metrics={"official_success": 0.0},
                        metadata={
                            **common_metadata,
                            "verifier_available": True,
                            "verifier_error_type": type(exc).__name__,
                        },
                    )
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self.trace is None:
            raise RuntimeError("reset first")
        return self.trace

    def close(self) -> None:
        if self._live_session is not None:
            self._live_session.close()
        self._live_session = None
        self.evaluator_owned_capx_environment = None

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self.get_trace().add_event(event_type, len(self.get_trace().events), payload)

    def _public_live_observation(self) -> dict[str, Any]:
        if self._live_session is None:
            raw = None
        else:
            with self._live_session.activate_runtime_workdir():
                raw = self._live_session.low_level_environment.get_observation()
        if not isinstance(raw, dict):
            raise RuntimeError("live CaP-X observation is unavailable")
        return deepcopy(raw)


class VLABenchAgentSmokeBackend(EmbodiedBackend):
    """EmbodiedBackend wrapper around the VLABench smoke runtime.

    VLABench lives under a benchmark directory whose path contains a hyphen, so
    this wrapper loads the existing runtime module by file path and presents the
    standard harness surface to `LiveAPIAgentSmokeRunner`.
    """

    def __init__(
        self,
        vlabench_root: str | None = None,
        mujoco_gl: str | None = None,
        exposed_primitives: set[str] | None = None,
    ) -> None:
        self.repo_root = resolve_project_root()
        self.runtime_module = _load_vlabench_runtime_module(self.repo_root)
        self.vlabench_root = vlabench_root
        self.mujoco_gl = mujoco_gl or os.environ.get("MUJOCO_GL", "egl")
        self._exposed_primitives = (
            set(exposed_primitives) if exposed_primitives is not None else None
        )
        self.runtime: Any | None = None
        self.trace: EpisodeTrace | None = None
        self.task_spec: TaskSpec | None = None
        self._last_observation: dict[str, Any] | None = None
        self._last_skill: dict[str, Any] | None = None
        self._last_motion: dict[str, Any] | None = None
        self._last_verification: dict[str, Any] | None = None
        self.vlabench_official_episode: VLABenchSameEpisodeOfficialBinding | None = None
        self._pool_episode: tuple[str, int, str, dict[str, Any]] | None = None

    def bind_pool_coordinate(self, coordinate: dict[str, Any]) -> dict[str, Any]:
        task_name = str(coordinate.get("task_id") or "")
        parts = str(coordinate.get("variation") or "").split("::")
        root = get_project_paths().external_upstream("vlabench")
        # Track filenames are taken from the pinned evaluation directory;
        # only the five explicitly included numbered tracks are eligible.
        allowed = {p.stem for p in (root / "VLABench/configs/evaluation/tracks").glob("*.json")
                   if p.stem.startswith(tuple(f"track_{n}_" for n in (1, 2, 3, 4, 6)))}
        if (len(parts) != 3 or parts[0] not in allowed or parts[1] != task_name
                or not parts[2].startswith("episode_") or not parts[2][8:].isdigit()):
            raise ValueError("Invalid selected VLABench track/task/episode coordinate")
        index = int(parts[2][8:])
        seed = coordinate.get("seed")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("VLABench seed must be an integer in [0, 2**32)")
        path = root / "VLABench/configs/evaluation/tracks" / f"{parts[0]}.json"
        episodes = json.loads(path.read_text(encoding="utf-8")).get(task_name, [])
        if not 0 <= index < len(episodes):
            raise ValueError("Selected VLABench task/episode is absent from the native track")
        episode = episodes[index]
        if not isinstance(episode, dict) or not isinstance(episode.get("task"), dict):
            raise ValueError("Selected VLABench episode lacks native task configuration")
        identity = f"vlabench:{parts[0]}:{task_name}:{index}:seed_{seed}"
        self._pool_episode = (task_name, seed, identity, episode)
        return {"bound": True, "mode": "native_track_episode", "task_name": task_name,
                "track": parts[0], "episode_index": index, "seed": seed,
                "actual_task_id": identity, "config_file": str(path)}

    def reset(
        self,
        task_id: str,
        seed: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> TaskSpec:
        config = dict(config or {})
        if self._pool_episode is not None:
            selected_name, selected_seed, task_id, _ = self._pool_episode
            if seed != selected_seed:
                raise ValueError("VLABench reset seed differs from selected pool coordinate")
            config["task_name"] = selected_name
        self.trace = EpisodeTrace(task_id=task_id)
        self._last_observation = None
        self._last_skill = None
        self._last_motion = None
        self._last_verification = None
        self.vlabench_official_episode = None
        self.task_spec = TaskSpec(
            task_id=task_id,
            source="w4:vlabench:live_runtime",
            instruction="Solve a VLABench manipulation smoke task by reading instruction, grounding a live entity, and moving the robot end effector through simulator actions.",
            goal={
                "task_name": config.get("task_name", "select_toy"),
                "operation": "follow the current instruction with visually grounded motion primitives",
            },
            budgets={"primitive_calls": 48, "verifier_calls": 4},
            tags=[
                "w4",
                "vlabench",
                "ai_native_runtime",
                "live",
                "state_grounded_motion",
            ],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata={
                "benchmark_id": "vlabench",
                "agent_native_contract": {
                    "primitives_accept_agent_context": True,
                    "raw_observation_exposed": False,
                    "oracle_checker_demo_replay_exposed": False,
                    "reset_owned_by_adapter": True,
                },
            },
        )
        episode_config_file = Path(
            config.get("episode_config_file")
            or self.repo_root
            / "benchmarks/operation/vlabench/select_toy_minimal_episode_config.json"
        )
        request = self.runtime_module.AgentRequest(
            agent_prompt="Use instruction and state evidence before selecting a VLABench simulator skill.",
            query="Which object evidence is available for the next action?",
            context={"caller": "live_api_agent_smoke", "task_id": task_id},
        )
        mujoco_gl = str(config.get("mujoco_gl", self.mujoco_gl))
        disable_depth_to_cloud = bool(
            config.get(
                "disable_depth_to_cloud",
                mujoco_gl.lower() in {"disable", "disabled", "off"},
            )
        )
        self.runtime = self.runtime_module.VLABenchRuntime(
            vlabench_root=config.get("vlabench_root") or self.vlabench_root,
            mujoco_gl=mujoco_gl,
            disable_depth_to_cloud=disable_depth_to_cloud,
            camera_observation_fallback=bool(
                config.get("camera_observation_fallback", True)
            ),
            visual_artifact_root=config.get("visual_artifact_root")
            or os.environ.get("VLABENCH_VISUAL_ARTIFACT_DIR")
            or self.repo_root / "artifacts" / "runtime" / "vlabench" / "visual",
        )
        reset = self.runtime.reset_vlabench_task(
            config.get("task_name", "select_toy"),
            request,
            seed=seed,
            episode_config=(deepcopy(self._pool_episode[3]) if self._pool_episode is not None
                            else self.runtime_module.load_episode_config(str(episode_config_file))),
            require_live=True,
        )
        self.vlabench_official_episode = VLABenchSameEpisodeOfficialBinding(
            self.runtime.env
        )
        self.record_event("reset", {"task": self.task_spec.to_dict(), "reset": reset})
        return self.task_spec

    def observe(self) -> Observation:
        self._require_reset()
        request = self.runtime_module.AgentRequest(
            agent_prompt="Summarize VLABench state-only evidence for a coding agent.",
            query="What object state evidence is available?",
            context={"plan_step": "initial_observe"},
        )
        self._last_observation = self.runtime.observe_vlabench_scene(request)
        visual_request = self.runtime_module.AgentRequest(
            agent_prompt="Inspect the current native camera observation for agent planning.",
            query="Which visible instances can be grounded for subsequent actions?",
            context={"plan_step": "initial_visual_observe"},
        )
        visual_inspection = self.runtime.inspect_vlabench_visual_evidence(
            request=visual_request,
            require_pcd=True,
        )
        artifact_paths = [
            str(item["rgb_path"])
            for item in visual_inspection.get("artifacts", [])
            if isinstance(item, dict) and item.get("rgb_path")
        ]
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "instruction": self._last_observation.get("instruction"),
                "visual_evidence": self._last_observation.get("visual_evidence"),
                "visual_inspection": visual_inspection,
                "object_grounding_evidence": self._last_observation.get(
                    "object_grounding_evidence"
                ),
                "action_schema": self._last_observation.get("action_schema"),
            },
            artifacts=artifact_paths,
            metadata={
                "benchmark_id": "vlabench",
                "state_only": False,
                "native_visual_required": True,
            },
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        self._require_reset()
        cards = [
            PrimitiveCard(
                name="get_vlabench_instruction",
                capability_tags=["w4", "vlabench", "agent_native_runtime"],
                input_schema={"agent_context": "dict|None"},
                output_schema={"instruction": "str|None"},
                abstraction_level="L1",
                description="Return the current VLABench task instruction.",
            ),
            PrimitiveCard(
                name="observe_vlabench_scene",
                capability_tags=["w4", "vlabench", "agent_native_runtime"],
                input_schema={
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "visual_evidence": "dict",
                    "object_grounding_evidence": "dict",
                    "action_schema": "dict",
                },
                abstraction_level="L1",
                description="Observe the live VLABench instruction, scene registry, and current visual-modality summary.",
            ),
            PrimitiveCard(
                name="inspect_vlabench_visual_evidence",
                capability_tags=[
                    "w4",
                    "vlabench",
                    "agent_native_runtime",
                    "vision",
                    "rgbd",
                    "segmentation",
                ],
                input_schema={
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "require_pcd": "bool",
                },
                output_schema={
                    "visual_evidence": "dict",
                    "entity_visual_candidates": "list[dict]",
                    "observation_serial": "int",
                },
                abstraction_level="L1",
                description="Inspect native VLABench RGB, depth, segmentation, and point-cloud evidence without selecting a target.",
            ),
            PrimitiveCard(
                name="ground_vlabench_visual_target",
                capability_tags=[
                    "w4",
                    "vlabench",
                    "agent_native_runtime",
                    "vision",
                    "grounding",
                ],
                input_schema={
                    "prompt": "str",
                    "segmentation_id": "int",
                    "entity_name": "str|None",
                    "camera_index": "int|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                    "require_pcd": "bool",
                },
                output_schema={
                    "evidence_handle": "str",
                    "grounding": "dict",
                    "visual_evidence": "dict",
                },
                abstraction_level="L2",
                description="Bind an agent-selected native segmentation instance to an action-valid visual evidence handle.",
            ),
            PrimitiveCard(
                name="locate_vlabench_entity",
                capability_tags=["w4", "vlabench", "agent_native_runtime", "grounding"],
                input_schema={
                    "entity_name": "str|None",
                    "query": "str|None",
                    "prompt": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "selected_entity": "str|None",
                    "available_entities": "list[str]",
                    "entity": "dict|None",
                    "evidence_handle": "str|None",
                },
                abstraction_level="L2",
                description="Locate an exact entity in native RGB-D segmentation and return its same-frame action evidence handle.",
            ),
            PrimitiveCard(
                name="move_vlabench_ee_to",
                capability_tags=["w4", "vlabench", "agent_native_runtime", "motion"],
                input_schema={
                    "target_name": "str|None",
                    "target_position": "list[float]|None",
                    "evidence_handles": "list[str]",
                    "offset": "list[float]|None",
                    "target_site": "xpos|grasp|place",
                    "target_quat": "list[float]|None",
                    "target_euler": "list[float]|None",
                    "gripper_state": "float|list[float]|None",
                    "horizon": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "moved": "bool",
                    "ik_solved": "bool",
                    "distance_to_target": "float",
                    "target_position": "list[float]",
                },
                abstraction_level="L3",
                description="Move the live robot end effector via VLABench IK and env.step(action).",
            ),
            PrimitiveCard(
                name="open_vlabench_gripper",
                capability_tags=["w4", "vlabench", "agent_native_runtime", "motion"],
                input_schema={
                    "horizon": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={"stepped": "bool", "gripper_state": "list[float]"},
                abstraction_level="L2",
                description="Open the live VLABench gripper through env.step(action).",
            ),
            PrimitiveCard(
                name="close_vlabench_gripper",
                capability_tags=["w4", "vlabench", "agent_native_runtime", "motion"],
                input_schema={
                    "horizon": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={"stepped": "bool", "gripper_state": "list[float]"},
                abstraction_level="L2",
                description="Close the live VLABench gripper through env.step(action).",
            ),
            PrimitiveCard(
                name="execute_vlabench_skill",
                capability_tags=["w4", "vlabench", "agent_native_runtime"],
                input_schema={
                    "skill_name": "str",
                    "target_name": "str|None",
                    "horizon": "int",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "stepped": "bool",
                    "steps": "int",
                    "timestep": "dict|None",
                },
                abstraction_level="L3",
                description="Execute a configured VLABench skill through the upstream simulator step path.",
            ),
            PrimitiveCard(
                name="grasp_vlabench_entity",
                capability_tags=[
                    "w4",
                    "vlabench",
                    "agent_native_runtime",
                    "motion",
                    "skilllib",
                ],
                input_schema={
                    "entity_name": "str",
                    "evidence_handles": "list[str]",
                    "target_position": "list[float]|None",
                    "target_quat": "list[float]|None",
                    "target_euler": "list[float]|None",
                    "max_n_substep": "int",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "stepped": "bool",
                    "stage_ok": "bool",
                    "task_done_seen": "bool",
                    "waypoint_count": "int",
                },
                abstraction_level="L3",
                description="Run the public VLABench SkillLib.pick controller for a grounded entity through env.step(action).",
            ),
            PrimitiveCard(
                name="lift_vlabench_ee",
                capability_tags=[
                    "w4",
                    "vlabench",
                    "agent_native_runtime",
                    "motion",
                    "skilllib",
                ],
                input_schema={
                    "lift_height": "float",
                    "evidence_handles": "list[str]",
                    "target_position": "list[float]|None",
                    "target_quat": "list[float]|None",
                    "target_euler": "list[float]|None",
                    "gripper_state": "float|list[float]|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "stepped": "bool",
                    "stage_ok": "bool",
                    "task_done_seen": "bool",
                    "waypoint_count": "int",
                },
                abstraction_level="L3",
                description="Run the public VLABench SkillLib.lift controller through env.step(action).",
            ),
            PrimitiveCard(
                name="place_vlabench_entity_in",
                capability_tags=[
                    "w4",
                    "vlabench",
                    "agent_native_runtime",
                    "motion",
                    "skilllib",
                ],
                input_schema={
                    "container_name": "str",
                    "evidence_handles": "list[str]",
                    "target_position": "list[float]|None",
                    "target_quat": "list[float]|None",
                    "target_euler": "list[float]|None",
                    "use_native_place_point": "bool",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "stepped": "bool",
                    "stage_ok": "bool",
                    "task_done_seen": "bool",
                    "waypoint_count": "int",
                },
                abstraction_level="L3",
                description="Run the public VLABench SkillLib.place controller for a grounded container through env.step(action).",
            ),
            PrimitiveCard(
                name="settle_vlabench_scene",
                capability_tags=["w4", "vlabench", "agent_native_runtime", "motion"],
                input_schema={
                    "horizon": "int",
                    "hold_gripper_state": "float|list[float]|None",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "stepped": "bool",
                    "steps": "int",
                    "grasped_obj_name": "str|None",
                },
                abstraction_level="L3",
                description="Advance the live VLABench simulator after a manipulation step and return observation evidence.",
            ),
            PrimitiveCard(
                name="record_vlabench_evidence",
                capability_tags=["w4", "vlabench", "agent_native_runtime"],
                input_schema={"key": "str", "value": "any"},
                output_schema={"artifact_id": "str"},
                abstraction_level="L1",
                description="Record agent-selected VLABench evidence in the episode trace.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        if self._exposed_primitives is not None:
            cards = [card for card in cards if card.name in self._exposed_primitives]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_reset()
        allowed = {card.name for card in self.list_primitives()}
        if name not in allowed:
            result = PrimitiveResult(
                name=name,
                ok=False,
                error=f"Primitive {name!r} is not exposed by VLABenchAgentSmokeBackend",
            )
        elif name == "get_vlabench_instruction":
            result = PrimitiveResult(
                name=name,
                ok=True,
                output={
                    "instruction": self.runtime.get_vlabench_instruction(),
                    "agent_context": kwargs.get("agent_context") or {},
                },
            )
        elif name == "observe_vlabench_scene":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.observe_vlabench_scene(request)
            output.pop("raw_observation", None)
            self._last_observation = output
            result = PrimitiveResult(name=name, ok=True, output=output)
        elif name == "inspect_vlabench_visual_evidence":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.inspect_vlabench_visual_evidence(
                request=request,
                require_pcd=bool(kwargs.get("require_pcd", True)),
            )
            result = PrimitiveResult(
                name=name, ok=bool(output.get("ok")), output=output
            )
        elif name == "ground_vlabench_visual_target":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.ground_vlabench_visual_target(
                prompt=kwargs.get("prompt") or "",
                segmentation_id=int(kwargs["segmentation_id"]),
                entity_name=kwargs.get("entity_name"),
                camera_index=kwargs.get("camera_index"),
                request=request,
                require_pcd=bool(kwargs.get("require_pcd", True)),
            )
            result = PrimitiveResult(
                name=name, ok=bool(output.get("ok")), output=output
            )
        elif name == "resolve_vlabench_instruction_targets":
            output = self._resolve_instruction_targets(
                prompt=kwargs.get("prompt"),
                query=kwargs.get("query"),
                agent_context=kwargs.get("agent_context") or {},
            )
            result = PrimitiveResult(
                name=name,
                ok=bool(output.get("source_entity") and output.get("target_container")),
                output=output,
            )
        elif name == "locate_vlabench_entity":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.locate_vlabench_entity(
                entity_name=kwargs.get("entity_name"),
                request=request,
                query=kwargs.get("query"),
            )
            result = PrimitiveResult(
                name=name, ok=bool(output.get("selected_entity")), output=output
            )
        elif name == "move_vlabench_ee_to":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.move_vlabench_ee_to(
                target_name=kwargs.get("target_name"),
                target_position=kwargs.get("target_position"),
                evidence_handles=kwargs.get("evidence_handles"),
                offset=kwargs.get("offset"),
                target_site=kwargs.get("target_site", "xpos"),
                target_quat=kwargs.get("target_quat"),
                target_euler=kwargs.get("target_euler"),
                gripper_state=kwargs.get("gripper_state"),
                request=request,
                horizon=int(kwargs.get("horizon", 20)),
            )
            self._last_motion = output
            result = PrimitiveResult(
                name=name, ok=bool(output.get("moved")), output=output
            )
        elif name in {"open_vlabench_gripper", "close_vlabench_gripper"}:
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            method = getattr(self.runtime, name)
            output = method(request=request, horizon=int(kwargs.get("horizon", 10)))
            self._last_skill = output
            result = PrimitiveResult(
                name=name, ok=bool(output.get("stepped")), output=output
            )
        elif name == "execute_vlabench_skill":
            request = self.runtime_module.AgentRequest(
                agent_prompt="Execute the selected VLABench skill.",
                query=kwargs.get("target_name"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.execute_vlabench_skill(
                skill_name=kwargs["skill_name"],
                target_name=kwargs.get("target_name"),
                request=request,
                horizon=int(kwargs.get("horizon", 1)),
            )
            self._last_skill = output
            result = PrimitiveResult(
                name=name, ok=bool(output.get("stepped")), output=output
            )
        elif name == "grasp_vlabench_entity":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            entity_name = self._resolve_entity_argument(
                kwargs["entity_name"], role="source"
            )
            output = self.runtime.grasp_vlabench_entity(
                entity_name=entity_name,
                request=request,
                evidence_handles=kwargs.get("evidence_handles"),
                target_position=kwargs.get("target_position"),
                target_quat=kwargs.get("target_quat"),
                target_euler=kwargs.get("target_euler"),
                max_n_substep=int(kwargs.get("max_n_substep", 2)),
            )
            self._last_skill = output
            result = PrimitiveResult(
                name=name,
                ok=bool(output.get("stepped") and output.get("stage_ok")),
                output=output,
            )
        elif name == "lift_vlabench_ee":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.lift_vlabench_ee(
                request=request,
                evidence_handles=kwargs.get("evidence_handles"),
                lift_height=float(kwargs.get("lift_height", 0.3)),
                target_position=kwargs.get("target_position"),
                target_quat=kwargs.get("target_quat"),
                target_euler=kwargs.get("target_euler"),
                gripper_state=kwargs.get("gripper_state"),
            )
            self._last_skill = output
            result = PrimitiveResult(
                name=name,
                # Upstream trajectory generation continues after a lift whose
                # final Cartesian error misses SkillLib's 1 cm diagnostic
                # tolerance.  The official task checker, not this intermediate
                # controller diagnostic, decides whether the manipulation
                # succeeded.  Treat a non-empty live trajectory as an executed
                # primitive so a composite can proceed to the official place
                # skill while preserving stage_ok in the output for auditing.
                ok=bool(output.get("stepped")),
                output=output,
            )
        elif name == "place_vlabench_entity_in":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            container_name = self._resolve_entity_argument(
                kwargs["container_name"], role="target"
            )
            output = self.runtime.place_vlabench_entity_in(
                container_name=container_name,
                request=request,
                evidence_handles=kwargs.get("evidence_handles"),
                target_position=kwargs.get("target_position"),
                target_quat=kwargs.get("target_quat"),
                target_euler=kwargs.get("target_euler"),
                use_native_place_point=bool(
                    kwargs.get("use_native_place_point", False)
                ),
            )
            self._last_skill = output
            result = PrimitiveResult(
                name=name,
                ok=bool(output.get("stepped") and output.get("stage_ok")),
                output=output,
            )
        elif name == "settle_vlabench_scene":
            request = self.runtime_module.AgentRequest(
                agent_prompt=kwargs.get("prompt") or "",
                query=kwargs.get("query"),
                context=kwargs.get("agent_context") or {},
            )
            output = self.runtime.settle_vlabench_scene(
                request=request,
                horizon=int(kwargs.get("horizon", 40)),
                hold_gripper_state=kwargs.get("hold_gripper_state"),
            )
            self._last_skill = output
            result = PrimitiveResult(
                name=name, ok=bool(output.get("stepped")), output=output
            )
        else:
            artifact_id = f"vlabench:evidence:{kwargs['key']}"
            self.get_trace().add_artifact(
                artifact_id, {"key": kwargs["key"], "value": kwargs.get("value")}
            )
            result = PrimitiveResult(
                name=name,
                ok=True,
                output={"artifact_id": artifact_id},
                artifacts=[artifact_id],
            )
        self.record_event(
            "primitive_call",
            {"name": name, "kwargs": kwargs, "result": result.to_dict()},
        )
        return result

    def _resolve_instruction_targets(
        self,
        *,
        prompt: str | None,
        query: str | None,
        agent_context: dict[str, Any],
    ) -> dict[str, Any]:
        observation = self._last_observation or self.runtime.observe_vlabench_scene(
            self.runtime_module.AgentRequest(
                agent_prompt=prompt or "",
                query=query,
                context=agent_context,
            )
        )
        self._last_observation = observation
        instruction = str(
            observation.get("instruction")
            or self.runtime.get_vlabench_instruction()
            or ""
        )
        available_entities = self._available_vlabench_entities(observation)
        by_lower = {name.lower(): name for name in available_entities}
        source_entity: str | None = None
        target_container: str | None = None
        match = re.search(
            r"\bput\s+the\s+(.+?)\s+into\s+the\s+([A-Za-z0-9_ -]+)",
            instruction,
            flags=re.IGNORECASE,
        )
        if match:
            source_text = match.group(1).strip().lower().replace(" ", "_")
            target_text = match.group(2).strip().lower().replace(" ", "_")
            source_entity = by_lower.get(source_text)
            target_container = by_lower.get(target_text)
        if source_entity is None:
            source_entity = next(
                (name for name in available_entities if "giftbox" not in name.lower()),
                None,
            )
        if target_container is None:
            target_container = next(
                (name for name in available_entities if "giftbox" in name.lower()), None
            )
        return {
            "instruction": instruction or None,
            "source_entity": source_entity,
            "target_container": target_container,
            "available_entities": available_entities,
            "agent_context": agent_context,
            "resolution_source": "live_instruction_regex_plus_entity_registry",
        }

    def _resolve_entity_argument(self, value: Any, *, role: str) -> str:
        text = str(value or "").strip()
        if text.lower() not in {
            "",
            "auto",
            "instruction",
            "from-instruction",
            "from_instruction",
        }:
            return text
        resolved = self._resolve_instruction_targets(
            prompt=None, query=f"resolve {role}", agent_context={}
        )
        key = "source_entity" if role == "source" else "target_container"
        entity = resolved.get(key)
        if not entity:
            raise RuntimeError(
                f"Unable to resolve VLABench {role} entity from live instruction: {resolved}"
            )
        return str(entity)

    @staticmethod
    def _available_vlabench_entities(observation: dict[str, Any]) -> list[str]:
        evidence = observation.get("object_grounding_evidence")
        if not isinstance(evidence, dict):
            return []
        entities = evidence.get("entity_evidence")
        if not isinstance(entities, dict):
            return []
        return sorted(str(name) for name in entities)

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_reset()
        official = self.runtime.verify_vlabench_task()
        self._last_verification = official
        stepped = bool(self._last_skill and self._last_skill.get("stepped"))
        moved = bool(self._last_motion and self._last_motion.get("moved"))
        distance = (
            float(self._last_motion.get("distance_to_target", 999.0))
            if self._last_motion
            else 999.0
        )
        result = VerificationResult(
            ok=bool(official.get("ok")),
            scope=scope,
            message=str(official.get("message") or "VLABench verifier completed."),
            metrics={
                "smoke_step_executed": float(stepped or moved),
                "motion_moved": float(moved),
                "ee_distance_to_target": distance,
                "official_success": float(bool(official.get("ok"))),
                **dict(official.get("metrics") or {}),
            },
            metadata=dict(official.get("metadata") or {}),
        )
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self.trace is None:
            raise RuntimeError("Call reset() before using the backend.")
        return self.trace

    def close(self) -> None:
        if self.vlabench_official_episode is not None:
            self.vlabench_official_episode.mark_closed()
        if self.runtime is not None:
            self.runtime.close()
        self.runtime = None

    def _require_reset(self) -> None:
        if self.runtime is None or self.task_spec is None:
            raise RuntimeError("Call reset() before using the backend.")


def _load_vlabench_runtime_module(repo_root: Path) -> Any:
    del repo_root
    return importlib.import_module(".vlabench_runtime", package=__package__)


class LiveAPIAgentSmokeRunner:
    def __init__(
        self,
        config: APIModelConfig,
        client: ChatClient | None = None,
        max_turns: int = 3,
        code_timeout_seconds: float = 20.0,
        replay_code: str | None = None,
        expected_interface_fingerprint: str | None = None,
    ) -> None:
        self.config = config
        self.client = client
        self.max_turns = max_turns
        self.code_timeout_seconds = code_timeout_seconds
        self.replay_code = replay_code
        self.expected_interface_fingerprint = expected_interface_fingerprint

    def run_cases(self, cases: list[LiveAgentSmokeCase]) -> dict[str, Any]:
        results = [self.run_case(case) for case in cases]
        official_results = [
            result for result in results if result.counts_toward_official_success
        ]
        boundary_results = [
            result for result in results if not result.counts_toward_official_success
        ]
        return {
            "model": self.config.model,
            "prompt_contract": QWEN_PROMPT_CONTRACT,
            "case_count": len(results),
            "agent_smoke_success_count": sum(
                1 for result in results if result.agent_smoke_success
            ),
            "benchmark_task_success_count": sum(
                1 for result in results if result.benchmark_task_success
            ),
            "official_case_count": len(official_results),
            "official_success_count": sum(
                1 for result in official_results if result.official_gate_success
            ),
            "boundary_case_count": len(boundary_results),
            "boundary_ready_count": sum(
                1 for result in boundary_results if result.agent_smoke_success
            ),
            "results": [result.to_dict() for result in results],
        }

    def run_case(self, case: LiveAgentSmokeCase) -> LiveAgentSmokeResult:
        backend: EmbodiedBackend | None = None
        code = ""
        interface_fingerprint = ""
        try:
            backend = case.backend_factory()
            task = backend.reset(case.task_id, seed=case.seed, config=case.reset_config)
            observation = backend.observe()
            cards = backend.list_primitives()
            interface_fingerprint = _interface_fingerprint(case, cards)
            if (
                self.expected_interface_fingerprint is not None
                and self.expected_interface_fingerprint != interface_fingerprint
            ):
                raise ValueError(
                    "interface_fingerprint_mismatch: generated code is not bound to the current primitive interface"
                )
            messages = [
                {
                    "role": "system",
                    "content": _system_prompt(
                        harness_only_verifier=case.harness_only_verifier
                    ),
                },
                {
                    "role": "user",
                    "content": _case_prompt(
                        case,
                        _agent_visible_task(task.to_dict()),
                        _agent_visible_observation(observation.to_dict()),
                        cards,
                    ),
                },
            ]
            runner = StatefulCodeRunner(
                backend,
                timeout_seconds=case.code_timeout_seconds or self.code_timeout_seconds,
            )
            runner.bind_task(task)
            if case.harness_only_verifier:
                runner.globals.pop("backend", None)
            primitive_facade = runner.globals["primitives"]
            for card in cards:
                runner.globals.setdefault(
                    card.name, getattr(primitive_facade, card.name)
                )
            execution = ExecutionResult(ok=False, error="not attempted")
            if self.replay_code is not None:
                code = self.replay_code
                contract_error = _agent_code_contract_error(
                    code,
                    cards,
                    harness_only_verifier=case.harness_only_verifier,
                )
                execution = (
                    ExecutionResult(ok=False, error=contract_error)
                    if contract_error
                    else runner.execute(code)
                )
            else:
                for turn_index in range(1, self.max_turns + 1):
                    raw = self._complete(messages)
                    code = extract_python_code(raw)
                    contract_error = _agent_code_contract_error(
                        code,
                        cards,
                        harness_only_verifier=case.harness_only_verifier,
                    )
                    execution = (
                        ExecutionResult(ok=False, error=contract_error)
                        if contract_error
                        else runner.execute(code)
                    )
                    verdict = _trace_verdict(backend, case)
                    if (
                        execution.ok
                        and not verdict["missing_required_primitives"]
                        and not verdict["failed_required_primitives"]
                    ):
                        break
                    messages.append({"role": "assistant", "content": raw})
                    messages.append(
                        {
                            "role": "user",
                            "content": _repair_prompt(turn_index, execution, verdict),
                        }
                    )
            verifier = backend.verify(scope="task")
            verdict = _trace_verdict(backend, case)
            agent_smoke_success = (
                execution.ok
                and not verdict["missing_required_primitives"]
                and not verdict["failed_required_primitives"]
            )
            official_gate_success = bool(
                case.counts_toward_official_success
                and agent_smoke_success
                and verifier.ok
            )
            trace = backend.get_trace()
            return LiveAgentSmokeResult(
                case_id=case.case_id,
                benchmark_id=case.benchmark_id,
                model=self.config.model,
                prompt_contract=QWEN_PROMPT_CONTRACT,
                interface_fingerprint=interface_fingerprint,
                readiness_tier=case.readiness_tier,
                counts_toward_official_success=case.counts_toward_official_success,
                harness_only_verifier=case.harness_only_verifier,
                agent_smoke_success=agent_smoke_success,
                benchmark_task_success=verifier.ok,
                official_gate_success=official_gate_success,
                execution_ok=execution.ok,
                called_primitives=verdict["called_primitives"],
                missing_required_primitives=verdict["missing_required_primitives"],
                failed_required_primitives=verdict["failed_required_primitives"],
                code=code,
                execution_error=execution.error,
                verifier_message=verifier.message,
                trace_event_count=len(trace.events),
                artifact_count=len(trace.artifacts),
                primitive_evidence=_primitive_evidence_digest(trace, case),
            )
        except Exception as exc:
            trace = None
            verdict = {
                "called_primitives": [],
                "missing_required_primitives": list(case.required_primitives),
                "failed_required_primitives": list(case.required_ok_primitives),
            }
            if backend is not None:
                try:
                    trace = backend.get_trace()
                    verdict = _trace_verdict(backend, case)
                except Exception:
                    trace = None
            return LiveAgentSmokeResult(
                case_id=case.case_id,
                benchmark_id=case.benchmark_id,
                model=self.config.model,
                prompt_contract=QWEN_PROMPT_CONTRACT,
                interface_fingerprint=interface_fingerprint,
                readiness_tier=case.readiness_tier,
                counts_toward_official_success=case.counts_toward_official_success,
                harness_only_verifier=case.harness_only_verifier,
                agent_smoke_success=False,
                benchmark_task_success=False,
                official_gate_success=False,
                execution_ok=False,
                called_primitives=verdict["called_primitives"],
                missing_required_primitives=verdict["missing_required_primitives"],
                failed_required_primitives=verdict["failed_required_primitives"],
                code=code,
                execution_error=f"{type(exc).__name__}: {exc}",
                verifier_message="case failed before official verifier",
                trace_event_count=len(trace.events) if trace is not None else 0,
                artifact_count=len(trace.artifacts) if trace is not None else 0,
                primitive_evidence=_primitive_evidence_digest(trace, case)
                if trace is not None
                else [],
            )
        finally:
            close = getattr(backend, "close", None) if backend is not None else None
            if callable(close):
                close()

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            if not self.config.api_key or not self.config.base_url:
                raise ValueError(
                    "A real API run needs api_key/base_url or an injected test client."
                )
            self.client = OpenAICompatibleChatClient(
                api_key=self.config.api_key, base_url=self.config.base_url
            )
        return self.client.complete(
            messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )


def collect_case_prompt_snapshot(case: LiveAgentSmokeCase) -> dict[str, Any]:
    """Collect the exact agent-visible prompt from one real benchmark reset."""

    backend: EmbodiedBackend | None = None
    try:
        backend = case.backend_factory()
        task = backend.reset(case.task_id, seed=case.seed, config=case.reset_config)
        observation = backend.observe()
        cards = backend.list_primitives()
        interface_fingerprint = _interface_fingerprint(case, cards)
        visible_task = _agent_visible_task(task.to_dict())
        visible_observation = _agent_visible_observation(observation.to_dict())
        messages = [
            {
                "role": "system",
                "content": _system_prompt(
                    harness_only_verifier=case.harness_only_verifier
                ),
            },
            {
                "role": "user",
                "content": _case_prompt(case, visible_task, visible_observation, cards),
            },
        ]
        return {
            "snapshot_contract": QWEN_PROMPT_CONTRACT,
            "interface_fingerprint": interface_fingerprint,
            "case_id": case.case_id,
            "benchmark_id": case.benchmark_id,
            "task_id": case.task_id,
            "seed": case.seed,
            "readiness_tier": case.readiness_tier,
            "counts_toward_official_success": case.counts_toward_official_success,
            "harness_only_verifier": case.harness_only_verifier,
            "task": visible_task,
            "initial_observation": visible_observation,
            "primitive_cards": [card.to_dict() for card in cards],
            "messages": messages,
        }
    finally:
        close = getattr(backend, "close", None) if backend is not None else None
        if callable(close):
            close()


def _system_prompt(*, harness_only_verifier: bool = False) -> str:
    verifier_rule = (
        "The official verifier is harness-only: backend is not available to your code. "
        "Do not call backend, verify, checker, success predicates, evaluator internals, demos, or replay. "
        "Store the last action PrimitiveResult in result."
        if harness_only_verifier
        else "Finish with: result = backend.verify(scope='task')."
    )
    return (
        "You are a coding agent controlling a real embodied benchmark through Python primitives. "
        "Return only executable Python code. Do not include prose. Do not import modules, open files, "
        "or use unsafe calls. Use only primitives.<name>(keyword=value, ...) calls plus simple Python data handling. "
        "Every primitive call must use keyword arguments. Primitive calls return PrimitiveResult objects; "
        "read data with result.output['key'] or result['key']. Inspect observations before choosing actions, derive "
        "arguments from returned public evidence, and preserve any evidence_handle values required by downstream "
        "primitive input schemas. When the published cards include a task-language/context primitive, call it before "
        "grounding. When the task is visual and cards include scene/RGBD observation plus single-instance inspection, "
        "observe the scene first and inspect selected source/target instances individually before acting. "
        "Record a compact evidence packet before acting. "
        "Never use variable names or agent_context keys named oracle, answer_key, gold, checker, success_label, demo, or replay. "
        "The official benchmark verifier may remain false after a minimal smoke action; still complete the requested primitive chain. "
        f"{verifier_rule}"
    )


def _case_prompt(
    case: LiveAgentSmokeCase,
    task: dict[str, Any],
    observation: dict[str, Any],
    primitives: list[PrimitiveCard],
) -> str:
    primitive_lines = "\n".join(
        (
            f"- {card.name}: description={card.description or 'No additional description.'}; "
            f"capabilities={card.capability_tags}; inputs={card.input_schema}; outputs={card.output_schema}; "
            f"preconditions={card.preconditions}; failure_modes={card.failure_modes}"
        )
        for card in primitives
    )
    final_rule = (
        "Finish with: result = <the last action primitive result>. Do not call backend or any verifier."
        if case.harness_only_verifier
        else "Finish with: result = backend.verify(scope='task')."
    )
    return (
        f"Case id: {case.case_id}\n"
        f"Objective: {case.objective}\n\n"
        f"Task spec JSON:\n{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Initial observation JSON:\n{json.dumps(observation, ensure_ascii=False, sort_keys=True)}\n\n"
        "Available primitives:\n"
        f"{primitive_lines}\n\n"
        "Write one code cell that autonomously selects and composes the available primitives needed to solve the task. "
        "Use prompt/query/agent_context fields when their schemas allow them so observations can be conditioned on your "
        "current reasoning context. Do not assume hidden entity IDs, poses, action vectors, waypoints, controller parameters, "
        "or task recipes; derive them from the task and public primitive outputs. Re-observe or inspect intermediate results "
        "when needed, and handle an unsuccessful PrimitiveResult by adapting with other public primitives. "
        f"{final_rule}"
    )


def _repair_prompt(
    turn_index: int, execution: ExecutionResult, verdict: dict[str, list[str]]
) -> str:
    contract_satisfied = (
        not verdict["missing_required_primitives"]
        and not verdict["failed_required_primitives"]
    )
    return (
        f"Turn {turn_index} did not satisfy the task contract. Write corrected executable Python code only.\n"
        f"Execution ok: {execution.ok}\n"
        f"Execution error: {execution.error or ''}\n"
        f"Task contract satisfied: {contract_satisfied}\n"
        "Re-read the task, public observation, and primitive schemas. Select any missing observation, grounding, evidence, "
        "or action capabilities yourself; adapt arguments from PrimitiveResult outputs. Use primitives.<name>(keyword=value, ...) "
        "with keyword arguments only, and do not call backend or hidden evaluators."
    )


def _trace_verdict(
    backend: EmbodiedBackend, case: LiveAgentSmokeCase
) -> dict[str, list[str]]:
    trace = backend.get_trace()
    called: list[str] = []
    ok_by_name: dict[str, bool] = {}
    for event in trace.events:
        if event.event_type != "primitive_call":
            continue
        name = str(event.payload.get("name", ""))
        if not name:
            continue
        called.append(name)
        result = event.payload.get("result", {})
        if isinstance(result, dict):
            ok_by_name[name] = ok_by_name.get(name, False) or bool(result.get("ok"))
    missing = [name for name in case.required_primitives if name not in called]
    failed = [
        name for name in case.required_ok_primitives if not ok_by_name.get(name, False)
    ]
    return {
        "called_primitives": called,
        "missing_required_primitives": missing,
        "failed_required_primitives": failed,
    }


def _primitive_evidence_digest(
    trace: EpisodeTrace, case: LiveAgentSmokeCase
) -> list[dict[str, Any]]:
    evidence_keys = {
        "execution_status",
        "motion_backend",
        "motion_status",
        "moved",
        "distance_before",
        "distance_after",
        "distance_to_target",
        "steps",
        "ik_solved",
        "stage_ok",
        "task_done_seen",
        "waypoint_count",
        "observation_count",
        "official_task_completion_claimed",
        "artifact_id",
        "button_contact_seen",
        "state_changed",
        "fixture_state_before",
        "fixture_state_after",
        "best_distance",
        "button_offset",
        "arm_delta_frame",
        "press_direction_sign",
        "source_object_id",
        "target_object_id",
        "evidence_ids",
        "stepped",
        "action",
        "action_schema",
        "pose0",
        "pose1",
    }
    digest: list[dict[str, Any]] = []
    required = set(case.required_primitives)
    for event in trace.events:
        if event.event_type != "primitive_call":
            continue
        name = str(event.payload.get("name", ""))
        result = event.payload.get("result", {})
        if not isinstance(result, dict):
            continue
        result_ok = bool(result.get("ok"))
        if name not in required and result_ok:
            continue
        output = result.get("output", {})
        if not isinstance(output, dict):
            output = {}
        selected = {key: output[key] for key in evidence_keys if key in output}
        if not result_ok and not selected:
            selected = _compact_public_primitive_output(output)
        error = result.get("error")
        if error and "error" not in selected:
            selected["error"] = str(error)[:500]
        if not selected:
            continue
        digest.append({"name": name, "ok": result_ok, "output": selected})
    return digest


def _compact_public_primitive_output(value: Any, *, depth: int = 0) -> Any:
    if depth >= 4:
        return "<truncated>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        if value.startswith("file://") or Path(value).is_absolute():
            return "<local-path-redacted>"
        return value[:500]
    if isinstance(value, (list, tuple)):
        return [
            _compact_public_primitive_output(item, depth=depth + 1)
            for item in list(value)[:12]
        ]
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 24:
                compact["<truncated_key_count>"] = len(value) - index
                break
            compact[str(key)[:120]] = _compact_public_primitive_output(
                item, depth=depth + 1
            )
        return compact
    return repr(value)[:500]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run qwen/API coding-agent smoke over live embodied benchmark primitives."
    )
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="qwen3.5-27b")
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Case id to run. Repeat or use all.",
    )
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--code-timeout-seconds", type=float, default=20.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument(
        "--code-file",
        default=None,
        help="Replay qwen-generated Python code from a file instead of calling the API.",
    )
    parser.add_argument(
        "--code-source-report",
        default=None,
        help="Original qwen runner report containing the replayed code; recorded for provenance.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--collect-prompt-only", action="store_true")
    parser.add_argument("--prompt-output", default=None)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    cases = built_in_live_cases(args.case or ["all"])
    if args.collect_prompt_only:
        if len(cases) != 1:
            parser.error("--collect-prompt-only requires exactly one --case")
        if not args.prompt_output:
            parser.error("--collect-prompt-only requires --prompt-output")
        snapshot = collect_case_prompt_snapshot(cases[0])
        snapshot_text = json.dumps(
            snapshot, ensure_ascii=False, indent=args.indent, sort_keys=True
        )
        Path(args.prompt_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.prompt_output).write_text(snapshot_text + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"prompt_output": args.prompt_output, "case_id": cases[0].case_id},
                indent=2,
            )
        )
        return 0

    config = load_api_model_config(args.env_file, model=args.model)
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens
    code_path = Path(args.code_file).resolve() if args.code_file else None
    source_report_path = (
        Path(args.code_source_report).resolve() if args.code_source_report else None
    )
    if source_report_path is not None and code_path is None:
        parser.error("--code-source-report requires --code-file")
    expected_interface_fingerprint: str | None = None
    if source_report_path is not None:
        source_payload = json.loads(source_report_path.read_text(encoding="utf-8"))
        expected_interface_fingerprint = str(
            source_payload.get("interface_fingerprint") or ""
        )
        if not expected_interface_fingerprint:
            source_results = source_payload.get("results")
            if isinstance(source_results, list):
                matching_source_results = [
                    item
                    for item in source_results
                    if isinstance(item, dict)
                    and item.get("case_id") == cases[0].case_id
                ]
                if len(matching_source_results) == 1:
                    expected_interface_fingerprint = str(
                        matching_source_results[0].get("interface_fingerprint") or ""
                    )
        if not expected_interface_fingerprint:
            parser.error(
                "--code-source-report must bind code to an interface_fingerprint"
            )
    report = LiveAPIAgentSmokeRunner(
        config=config,
        max_turns=args.max_turns,
        code_timeout_seconds=args.code_timeout_seconds,
        replay_code=code_path.read_text(encoding="utf-8") if code_path else None,
        expected_interface_fingerprint=expected_interface_fingerprint,
    ).run_cases(cases)
    report["code_execution_mode"] = (
        "qwen_code_replay" if code_path else "qwen_api_online"
    )
    if code_path is not None:
        report["code_file"] = str(code_path)
        report["code_file_sha256"] = hashlib.sha256(code_path.read_bytes()).hexdigest()
    if source_report_path is not None:
        report["code_source_report"] = str(source_report_path)
        report["code_source_report_sha256"] = hashlib.sha256(
            source_report_path.read_bytes()
        ).hexdigest()
    text = json.dumps(report, ensure_ascii=False, indent=args.indent, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["agent_smoke_success_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
