from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import socket
import sys
from typing import Any, Callable, Protocol
from urllib.parse import urlparse
from uuid import uuid4

import numpy as np


JsonDict = dict[str, Any]

ROBODOJO_RGB_VIEWS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
ROBODOJO_THREE_VIEW_RGB_POLICIES = frozenset({"ACT", "Pi_05"})
ROBODOJO_DUAL_ARM_JOINT_ACTION_SCHEMA = {
    "left_arm_joint_state": 6,
    "left_ee_joint_state": 1,
    "right_arm_joint_state": 6,
    "right_ee_joint_state": 1,
}
_ROBODOJO_RGB_ALIASES = {
    "cam_head": ("cam_head", "cam_high", "head_camera", "top_camera"),
    "cam_left_wrist": ("cam_left_wrist", "left_camera", "left_wrist", "wrist_left"),
    "cam_right_wrist": ("cam_right_wrist", "right_camera", "right_wrist", "wrist_right"),
}


class XPolicyLabClient(Protocol):
    def call(self, func_name: str | None = None, obs: Any = None, **kwargs: Any) -> Any: ...


ActionExecutor = Callable[[JsonDict], Any]
ObservationProvider = Callable[[], JsonDict]
ActionValidator = Callable[[list[JsonDict]], list[JsonDict]]


@dataclass(slots=True)
class RoboDojoPolicySkillConfig:
    repo_path: str
    policy_name: str = "Pi_05"
    checkpoint_name: str = ""
    env_cfg: str = "arx_x5"
    action_type: str = "joint"
    server_url: str = "ws://127.0.0.1:6000"
    evaluation_id: str = "agentic-embodied-arena"

    def to_dict(self) -> JsonDict:
        return asdict(self)


def list_robodojo_policy_adapters(repo_path: str | Path) -> list[JsonDict]:
    root = Path(repo_path).expanduser().resolve()
    policy_root = root / "XPolicyLab" / "policy"
    adapters: list[JsonDict] = []
    if not policy_root.is_dir():
        return adapters
    for adapter_dir in sorted(path for path in policy_root.iterdir() if path.is_dir()):
        required = {
            "deploy_config": adapter_dir / "deploy.yml",
            "model": adapter_dir / "model.py",
            "deploy": adapter_dir / "deploy.py",
            "server_launcher": adapter_dir / "setup_eval_policy_server.sh",
            "client_launcher": adapter_dir / "setup_eval_env_client.sh",
        }
        if not any(path.exists() for path in required.values()):
            continue
        checkpoints = _checkpoint_dirs(adapter_dir)
        adapters.append(
            {
                "name": adapter_dir.name,
                "adapter_dir": str(adapter_dir),
                "files": {name: str(path) for name, path in required.items()},
                "adapter_ready": all(path.is_file() for path in required.values()),
                "checkpoint_count": len(checkpoints),
                "checkpoint_paths": [str(path) for path in checkpoints[:20]],
                "dummy_policy": adapter_dir.name == "demo_policy",
                "operation_semantics": "zero_action_wiring_only" if adapter_dir.name == "demo_policy" else "learned_policy",
            }
        )
    return adapters


def inspect_robodojo_policy_skill(config: RoboDojoPolicySkillConfig) -> JsonDict:
    root = Path(config.repo_path).expanduser().resolve()
    xpolicy_root = root / "XPolicyLab"
    adapter_dir = xpolicy_root / "policy" / config.policy_name
    expected_files = {
        "deploy_config": adapter_dir / "deploy.yml",
        "model": adapter_dir / "model.py",
        "deploy": adapter_dir / "deploy.py",
        "server_launcher": adapter_dir / "setup_eval_policy_server.sh",
        "client_launcher": adapter_dir / "setup_eval_env_client.sh",
        "same_machine_launcher": adapter_dir / "eval.sh",
    }
    checkpoint_path, checkpoint_candidates = resolve_robodojo_policy_checkpoint(
        adapter_dir,
        checkpoint_name=config.checkpoint_name,
        env_cfg=config.env_cfg,
        action_type=config.action_type,
    )
    deploy_config = _load_simple_yaml(expected_files["deploy_config"])
    policy_runtime = _policy_runtime_path(adapter_dir)
    policy_runtime_ready = any(
        candidate.is_file()
        for candidate in (policy_runtime / "bin/activate", policy_runtime / ".venv/bin/activate")
    )
    dummy_policy = config.policy_name == "demo_policy"
    adapter_ready = all(
        expected_files[name].is_file()
        for name in ("deploy_config", "model", "deploy", "server_launcher", "client_launcher")
    )
    normalization_contract = _normalization_asset_contract(
        checkpoint_path,
        policy_name=config.policy_name,
        deploy_config=deploy_config,
    )
    checkpoint_ready = (
        checkpoint_path is not None
        and _checkpoint_payload_ready(checkpoint_path, policy_name=config.policy_name)
        and bool(normalization_contract["ready"])
    )
    blockers: list[str] = []
    if not adapter_ready:
        blockers.append(f"policy_adapter_incomplete:{config.policy_name}")
    if not checkpoint_ready:
        blockers.append(
            f"policy_checkpoint_missing_or_incomplete:{config.policy_name}:{config.env_cfg}:{config.action_type}"
        )
    if not normalization_contract["ready"]:
        blockers.extend(normalization_contract["blockers"])
    if dummy_policy:
        blockers.append("demo_policy_is_zero_action_wiring")
    policy_server_endpoint = inspect_robodojo_policy_server_endpoint(config.server_url)
    execution_blockers = list(blockers)
    if not policy_server_endpoint["reachable"]:
        execution_blockers.append(policy_server_endpoint["blocker"])
    operation_ready = bool(adapter_ready and policy_runtime_ready and checkpoint_ready and not dummy_policy)
    return {
        "policy_name": config.policy_name,
        "official_boundary": "RoboDojo evaluation plus XPolicyLab policy adapter",
        "repo_path": str(root),
        "xpolicylab_path": str(xpolicy_root),
        "adapter_dir": str(adapter_dir),
        "expected_files": {name: str(path) for name, path in expected_files.items()},
        "missing_files": [str(path) for path in expected_files.values() if not path.is_file()],
        "adapter_ready": adapter_ready,
        "policy_runtime_path": str(policy_runtime),
        "policy_runtime_ready": policy_runtime_ready,
        "checkpoint_requested": config.checkpoint_name or None,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "checkpoint_candidates": [str(path) for path in checkpoint_candidates[:20]],
        "checkpoint_ready": checkpoint_ready,
        "checkpoint_contract": {
            "train_config_name": deploy_config.get("train_config_name"),
            "normalization_asset_id": deploy_config.get("repo_id"),
            "expected_checkpoint_step": deploy_config.get("checkpoint_num"),
            "generic_base_is_operation_ready": False,
        },
        "normalization_contract": normalization_contract,
        "server_url": config.server_url,
        "policy_server_endpoint": policy_server_endpoint,
        "policy_server_ready": bool(policy_server_endpoint["reachable"]),
        "dummy_policy": dummy_policy,
        "operation_semantics": "zero_action_wiring_only" if dummy_policy else "learned_policy",
        "operation_ready": operation_ready,
        "execution_ready": bool(operation_ready and policy_server_endpoint["reachable"]),
        "blockers": blockers,
        "execution_blockers": execution_blockers,
        "prompt_field": "instruction",
        "runtime_guidance_field": "agent_context.agent_prompt",
        "observation_contract": {
            "public_only": True,
            "required_by_policy": [
                "state or robot state fields",
                "vision.cam_head.color|rgb",
                "vision.cam_left_wrist.color|rgb",
                "vision.cam_right_wrist.color|rgb",
                "instruction",
            ],
            "three_view_rgb_policies": sorted(ROBODOJO_THREE_VIEW_RGB_POLICIES),
            "rgb_views": list(ROBODOJO_RGB_VIEWS),
            "summary_only_visual_input_accepted": False,
        },
        "success_verifier_exposed": False,
    }


def inspect_robodojo_policy_server_endpoint(server_url: str, *, timeout_s: float = 0.2) -> JsonDict:
    """Check only public websocket endpoint reachability; no policy call is made."""

    normalized_url = server_url.strip()
    if "://" not in normalized_url:
        normalized_url = f"ws://{normalized_url}"
    parsed = urlparse(normalized_url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    port = parsed.port
    if port is None and scheme in {"ws", "wss"}:
        port = 443 if scheme == "wss" else 80
    payload: JsonDict = {
        "server_url": server_url,
        "normalized_url": normalized_url,
        "scheme": scheme or None,
        "host": host,
        "port": port,
        "timeout_s": timeout_s,
        "check_attempted": False,
        "reachable": False,
        "transport_only_check": True,
        "websocket_handshake_performed": False,
        "policy_rpc_performed": False,
        "loopback": host in {"localhost", "127.0.0.1", "::1"},
        "error_type": None,
        "error_errno": None,
        "error": None,
        "blocker": "policy_server_unreachable:unknown",
    }
    if scheme not in {"ws", "wss"}:
        payload["blocker"] = f"policy_server_url_scheme_unsupported:{scheme or '<empty>'}"
        return payload
    if not host or port is None:
        payload["blocker"] = "policy_server_url_host_or_port_missing"
        return payload
    payload["check_attempted"] = True
    try:
        with socket.create_connection((host, int(port)), timeout=timeout_s):
            pass
    except OSError as exc:
        payload["error_type"] = type(exc).__name__
        payload["error_errno"] = getattr(exc, "errno", None)
        payload["error"] = str(exc)
        errno_slug = f"errno_{exc.errno}" if getattr(exc, "errno", None) is not None else type(exc).__name__
        payload["blocker"] = f"policy_server_unreachable:{errno_slug}"
        return payload
    payload["reachable"] = True
    payload["blocker"] = None
    return payload


def resolve_robodojo_policy_checkpoint(
    adapter_dir: str | Path,
    *,
    checkpoint_name: str = "",
    env_cfg: str = "arx_x5",
    action_type: str = "joint",
) -> tuple[Path | None, list[Path]]:
    adapter = Path(adapter_dir).expanduser().resolve()
    candidates = _checkpoint_dirs(adapter)
    requested = checkpoint_name.strip()
    if not requested:
        return (candidates[0] if len(candidates) == 1 else None), candidates

    direct = Path(requested).expanduser()
    requested_paths = [direct]
    if not direct.is_absolute():
        requested_paths.extend(
            [
                adapter / "checkpoints" / requested,
                adapter / "checkpoints" / f"RoboDojo-{requested}-{env_cfg}-{action_type}-0",
            ]
        )
    for path in requested_paths:
        resolved = path.resolve()
        if resolved.is_dir():
            return resolved, candidates
    by_name = [path for path in candidates if path.name == requested or requested in path.name]
    return (by_name[0] if len(by_name) == 1 else None), candidates


def create_robodojo_policy_client(
    config: RoboDojoPolicySkillConfig,
    *,
    trial_id: str | None = None,
    action_case_id: str | None = None,
) -> XPolicyLabClient:
    xpolicy_root = Path(config.repo_path).expanduser().resolve() / "XPolicyLab"
    if not xpolicy_root.is_dir():
        raise FileNotFoundError(f"XPolicyLab not found: {xpolicy_root}")
    adapter_dir = xpolicy_root / "policy" / config.policy_name
    _prepend_robodojo_import_paths([xpolicy_root, *_policy_runtime_site_packages(_policy_runtime_path(adapter_dir))])
    try:
        from client_server.ws.model_client import WsModelClient
    except Exception as exc:  # pragma: no cover - depends on the live policy environment
        raise RuntimeError(f"XPolicyLab websocket client import failed: {type(exc).__name__}:{exc}") from exc
    return WsModelClient(
        url=config.server_url,
        evaluation_id=config.evaluation_id,
        trial_id=trial_id or f"coding-agent-{uuid4().hex[:12]}",
        action_case_id=action_case_id,
    )


def run_robodojo_policy_skill(
    config: RoboDojoPolicySkillConfig,
    *,
    observation: JsonDict,
    agent_prompt: str | None = None,
    agent_context: JsonDict | None = None,
    max_actions: int = 8,
    reset_policy: bool = False,
    client: XPolicyLabClient | None = None,
    action_validator: ActionValidator | None = None,
    action_executor: ActionExecutor | None = None,
    observation_provider: ObservationProvider | None = None,
    evidence_refs: list[Any] | None = None,
) -> JsonDict:
    if not isinstance(observation, dict) or not observation:
        raise ValueError("a non-empty public RoboDojo observation is required")
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")

    own_client = client is None
    policy_client = client or create_robodojo_policy_client(config)
    policy_observation = prepare_robodojo_policy_observation(
        observation,
        agent_prompt=agent_prompt,
        agent_context=agent_context,
        policy_name=config.policy_name,
        evidence_refs=evidence_refs,
    )
    observation_evidence_refs = deepcopy(policy_observation["evidence_refs"])
    try:
        if reset_policy:
            policy_client.call(func_name="reset", obs={"agent_context": agent_context or {}})
        policy_client.call(func_name="update_obs", obs=policy_observation)
        raw_actions = policy_client.call(func_name="get_action")
        actions = normalize_robodojo_policy_actions(raw_actions, max_actions=max_actions)
        if action_validator is not None:
            actions = action_validator(actions)

        execution_results: list[Any] = []
        observations_after_action: list[JsonDict] = []
        execution_rejected = False
        if action_executor is not None:
            for action in actions:
                execution_result = _jsonable(action_executor(action))
                execution_results.append(execution_result)
                if observation_provider is not None:
                    observed = _jsonable(observation_provider())
                    if isinstance(observed, dict):
                        observations_after_action.append(observed)
                        policy_client.call(
                            func_name="update_obs",
                            obs=prepare_robodojo_policy_observation(
                                observed,
                                agent_prompt=agent_prompt,
                                agent_context=agent_context,
                                policy_name=config.policy_name,
                                evidence_refs=evidence_refs,
                            ),
                        )
                if not _policy_action_execution_ok(execution_result):
                    execution_rejected = True
                    break

        execution_complete = action_executor is None or len(execution_results) == len(actions)
        execution_ok = action_executor is None or (execution_complete and not execution_rejected)

        return {
            "policy_name": config.policy_name,
            "checkpoint_name": config.checkpoint_name or None,
            "server_url": config.server_url,
            "agent_prompt_applied": agent_prompt is not None,
            "agent_context": _jsonable(agent_context or {}),
            "action_count": len(actions),
            "actions": _jsonable(actions),
            "env_step_executed": action_executor is not None,
            "executed_action_count": len(execution_results),
            "execution_complete": execution_complete,
            "execution_ok": execution_ok,
            "execution_results": execution_results,
            "observations_after_action": observations_after_action,
            "observation_evidence_refs": observation_evidence_refs,
            "action_chunk_evidence_refs": deepcopy(observation_evidence_refs),
            "visual_observation_contract": deepcopy(policy_observation["visual_observation_contract"]),
            "official_success_claimed": False,
            "verifier_exposed": False,
            "skill_contract": {
                "policy_selectable_by_agent": True,
                "prompt_selectable_by_agent": True,
                "action_budget_selectable_by_agent": True,
                "task_recipe_embedded": False,
                "fixed_object_or_pose_embedded": False,
            },
        }
    finally:
        if own_client and hasattr(policy_client, "close"):
            policy_client.close()  # type: ignore[attr-defined]


def _policy_action_execution_ok(value: Any) -> bool:
    """Read only generic controller acceptance, never task-success fields."""

    if not isinstance(value, dict):
        return value is not False
    for key in ("ok", "accepted"):
        if key in value and value[key] is False:
            return False
    return True


def prepare_robodojo_policy_observation(
    observation: JsonDict,
    *,
    agent_prompt: str | None,
    agent_context: JsonDict | None,
    policy_name: str = "Pi_05",
    evidence_refs: list[Any] | None = None,
) -> JsonDict:
    public = deepcopy(observation)
    if "public_runtime_state" in public and isinstance(public["public_runtime_state"], dict):
        public = deepcopy(public["public_runtime_state"])
    nested = public.get("observation")
    if isinstance(nested, dict):
        flattened = deepcopy(nested)
        for key in ("robot", "scene", "env_summary"):
            if key in public and key not in flattened:
                flattened[key] = deepcopy(public[key])
        public = flattened
    policy_agent_context = deepcopy(agent_context or {})
    if agent_prompt is not None:
        prompt = agent_prompt.strip()
        if not prompt:
            raise ValueError("agent_prompt cannot be empty")
        policy_agent_context["agent_prompt"] = prompt
    if policy_name in ROBODOJO_THREE_VIEW_RGB_POLICIES:
        visual_contract, visual_refs = inspect_robodojo_rgb_observation(public, policy_name=policy_name)
        if not visual_contract["ready"]:
            missing = ",".join(visual_contract["missing_views"])
            invalid = ",".join(visual_contract["invalid_views"])
            details = ";".join(part for part in (f"missing={missing}" if missing else "", f"invalid={invalid}" if invalid else "") if part)
            policy_slug = policy_name.lower().replace("_", "")
            raise ValueError(f"{policy_slug}_three_view_rgb_observation_required:{details}")
        _coerce_robodojo_policy_rgb_frames(public, visual_contract)
        public["visual_observation_contract"] = visual_contract
        public["evidence_refs"] = _normalize_evidence_refs([*visual_refs, *(evidence_refs or [])])
    else:
        public["visual_observation_contract"] = {
            "policy_name": policy_name,
            "ready": None,
            "contract": "policy_specific_not_validated_by_pi05_contract",
        }
        public["evidence_refs"] = _normalize_evidence_refs(evidence_refs or [])
    public["agent_context"] = _jsonable(policy_agent_context)
    return public


def inspect_robodojo_rgb_observation(
    observation: JsonDict,
    *,
    policy_name: str = "Pi_05",
) -> tuple[JsonDict, list[JsonDict]]:
    """Validate and reference the native three-view RGB payload without fabricating frames."""

    vision = observation.get("vision")
    if not isinstance(vision, dict):
        vision = {}
    views: dict[str, JsonDict] = {}
    refs: list[JsonDict] = []
    missing: list[str] = []
    invalid: list[str] = []
    for canonical_name in ROBODOJO_RGB_VIEWS:
        source_name, image_key, frame = _find_rgb_frame(vision, canonical_name)
        if source_name is None:
            missing.append(canonical_name)
            continue
        frame_summary = _rgb_frame_summary(frame)
        if not frame_summary["raw_frame"]:
            invalid.append(canonical_name)
            continue
        source_path = f"vision.{source_name}" + (f".{image_key}" if image_key else "")
        views[canonical_name] = {"source_path": source_path, **frame_summary}
        refs.append(
            {
                "kind": "robodojo_rgb_frame",
                "camera": canonical_name,
                "source_path": source_path,
                "shape": frame_summary.get("shape"),
                "dtype": frame_summary.get("dtype"),
            }
        )
    return (
        {
            "policy_name": policy_name,
            "contract": "native_three_view_rgb",
            "required_views": list(ROBODOJO_RGB_VIEWS),
            "ready": not missing and not invalid,
            "missing_views": missing,
            "invalid_views": invalid,
            "views": views,
            "summary_only_visual_input_accepted": False,
            "frames_fabricated": False,
        },
        refs,
    )


def _find_rgb_frame(vision: JsonDict, canonical_name: str) -> tuple[str | None, str | None, Any]:
    for candidate in _ROBODOJO_RGB_ALIASES[canonical_name]:
        if candidate not in vision:
            continue
        camera = vision[candidate]
        if isinstance(camera, dict):
            for image_key in ("color", "rgb"):
                if image_key in camera:
                    return candidate, image_key, camera[image_key]
            return candidate, None, camera
        return candidate, None, camera
    return None, None, None


def _coerce_robodojo_policy_rgb_frames(observation: JsonDict, visual_contract: JsonDict) -> None:
    views = visual_contract.get("views")
    if not isinstance(views, dict):
        return
    vision = observation.get("vision")
    if not isinstance(vision, dict):
        return
    for canonical_name, view in views.items():
        if not isinstance(view, dict):
            continue
        source_path = view.get("source_path")
        if not isinstance(source_path, str) or not source_path:
            continue
        try:
            frame = _value_at_path(observation, source_path)
            coerced = _coerce_rgb_frame_array(frame)
            _set_value_at_path(observation, source_path, coerced)
            camera = vision.setdefault(str(canonical_name), {})
            if isinstance(camera, dict):
                camera["color"] = coerced
        except (KeyError, TypeError, ValueError):
            continue


def _coerce_rgb_frame_array(frame: Any) -> Any:
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return frame
    array = frame if isinstance(frame, np.ndarray) else np.asarray(frame)
    if array.ndim != 3:
        return frame
    if array.shape[-1] != 3 and array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] != 3:
        return frame
    if array.dtype.kind == "f":
        finite = array[np.isfinite(array)]
        if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255)
    if array.dtype != np.uint8:
        array = array.astype(np.uint8, copy=False)
    return np.ascontiguousarray(array)


def _value_at_path(value: JsonDict, source_path: str) -> Any:
    current: Any = value
    for part in source_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(source_path)
        current = current[part]
    return current


def _set_value_at_path(value: JsonDict, source_path: str, replacement: Any) -> None:
    current: Any = value
    parts = source_path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise KeyError(source_path)
        current = current[part]
    if not isinstance(current, dict):
        raise KeyError(source_path)
    current[parts[-1]] = replacement


def _rgb_frame_summary(frame: Any) -> JsonDict:
    if isinstance(frame, (bytes, bytearray, memoryview)):
        return {
            "raw_frame": len(frame) > 0,
            "shape": None,
            "dtype": "uint8",
            "encoding": "compressed_or_encoded_bytes",
        }
    shape = getattr(frame, "shape", None)
    if shape is None:
        shape = _nested_shape(frame)
    try:
        dimensions = [int(item) for item in shape] if shape is not None else []
    except (TypeError, ValueError):
        dimensions = []
    channels = dimensions[-1] if len(dimensions) == 3 and dimensions[-1] in {1, 3, 4} else (
        dimensions[0] if len(dimensions) == 3 and dimensions[0] in {1, 3, 4} else None
    )
    return {
        "raw_frame": bool(len(dimensions) == 3 and channels == 3 and all(item > 0 for item in dimensions)),
        "shape": dimensions or None,
        "dtype": str(getattr(frame, "dtype", type(frame).__name__)),
        "encoding": "array",
    }


def _nested_shape(value: Any) -> list[int] | None:
    shape: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        shape.append(len(current))
        if not current:
            break
        current = current[0]
    return shape or None


def _normalize_evidence_refs(evidence_refs: list[Any]) -> list[JsonDict]:
    normalized: list[JsonDict] = []
    for ref in evidence_refs:
        if isinstance(ref, str) and ref.strip():
            normalized.append({"artifact_id": ref.strip()})
        elif isinstance(ref, dict) and ref:
            normalized.append({str(key): _jsonable(value) for key, value in ref.items()})
        else:
            raise ValueError("evidence_refs must contain non-empty strings or mappings")
    return normalized


def normalize_robodojo_policy_actions(raw_actions: Any, *, max_actions: int) -> list[JsonDict]:
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        raise TypeError(f"policy returned {type(raw_actions).__name__}, expected action dict list")
    actions: list[JsonDict] = []
    for index, action in enumerate(raw_actions[:max_actions]):
        if not isinstance(action, dict) or not action:
            raise ValueError(f"policy action {index} must be a non-empty dict")
        keys = {str(key) for key in action}
        expected_keys = set(ROBODOJO_DUAL_ARM_JOINT_ACTION_SCHEMA)
        missing = sorted(expected_keys - keys)
        extra = sorted(keys - expected_keys)
        if missing or extra:
            raise ValueError(
                f"policy action {index} must match the complete 6/1/6/1 joint schema; "
                f"missing={missing}; extra={extra}"
            )
        normalized: JsonDict = {}
        for key, expected_dim in ROBODOJO_DUAL_ARM_JOINT_ACTION_SCHEMA.items():
            value = action[key]
            if isinstance(value, (str, bytes, bytearray)):
                raise TypeError(f"policy action {index} field {key} must be a numeric vector")
            try:
                vector = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"policy action {index} field {key} must be a numeric vector") from exc
            if vector.ndim != 1 or vector.shape[0] != expected_dim:
                raise ValueError(
                    f"policy action {index} field {key} must have dimension {expected_dim}, "
                    f"got shape {tuple(vector.shape)}"
                )
            if not np.isfinite(vector).all():
                raise ValueError(f"policy action {index} field {key} must contain only finite values")
            normalized[key] = vector.astype(float, copy=False).tolist()
        actions.append(normalized)
    if not actions:
        raise ValueError("policy returned an empty action chunk")
    return actions


def _normalization_asset_contract(
    checkpoint_path: Path | None,
    *,
    policy_name: str,
    deploy_config: JsonDict,
) -> JsonDict:
    """Fail closed when a Pi policy's configured normalization asset is not selected exactly."""

    if policy_name not in {"Pi_0", "Pi_05", "Pi_0_Fast"}:
        return {"required": False, "ready": True, "blockers": []}
    asset_id = str(deploy_config.get("repo_id") or "").strip()
    blockers: list[str] = []
    if not asset_id:
        blockers.append("normalization_asset_id_missing_from_deploy_config")
    model_roots = [] if checkpoint_path is None else [
        checkpoint_path,
        *(child for child in checkpoint_path.iterdir() if child.is_dir()),
    ]
    matching_assets = [root / "assets" / asset_id for root in model_roots if asset_id and (root / "assets" / asset_id).is_dir()]
    asset_path = matching_assets[0] if len(matching_assets) == 1 else None
    stats_path = asset_path / "norm_stats.json" if asset_path is not None else None
    stats: Any = None
    if stats_path is None or not stats_path.is_file():
        blockers.append(f"normalization_asset_missing_or_mismatched:{asset_id or '<unset>'}")
    else:
        try:
            stats = json.loads(stats_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            blockers.append(f"normalization_asset_invalid_json:{asset_id}")
    if stats is not None and not _normalization_stats_have_quantiles(stats):
        blockers.append(f"normalization_asset_quantiles_missing:{asset_id}")
    return {
        "required": True,
        "ready": not blockers,
        "asset_id": asset_id or None,
        "asset_path": str(asset_path) if asset_path is not None else None,
        "stats_path": str(stats_path) if stats_path is not None else None,
        "quantiles_required": True,
        "fallback_allowed": False,
        "blockers": blockers,
    }


def _normalization_stats_have_quantiles(value: Any) -> bool:
    leaves: list[dict[str, Any]] = []

    def visit(item: Any) -> None:
        if not isinstance(item, dict):
            return
        if "mean" in item or "std" in item or "q01" in item or "q99" in item:
            leaves.append(item)
            return
        for child in item.values():
            visit(child)

    visit(value)
    return bool(leaves) and all(leaf.get("q01") is not None and leaf.get("q99") is not None for leaf in leaves)


def _policy_runtime_path(adapter_dir: Path) -> Path:
    deploy = _load_simple_yaml(adapter_dir / "deploy.yml")
    raw = str(deploy.get("policy_uv_env_path") or "").strip()
    if not raw:
        return adapter_dir
    path = Path(raw).expanduser()
    return path.resolve() if path.is_absolute() else (adapter_dir / path).resolve()


def _policy_runtime_site_packages(policy_runtime: Path) -> list[Path]:
    paths: list[Path] = []

    def add_site_packages(root: Path) -> None:
        lib = root / "lib"
        if not lib.is_dir():
            return
        for candidate in sorted(lib.glob("python*/site-packages")):
            if candidate.is_dir() and candidate not in paths:
                paths.append(candidate)

    runtime_roots = [policy_runtime, policy_runtime / ".venv"]
    for root in list(runtime_roots):
        python_executable = root / "bin" / "python"
        if python_executable.exists():
            resolved = python_executable.resolve()
            if resolved.name.startswith("python") and len(resolved.parents) >= 2:
                runtime_roots.append(resolved.parents[1])

    for root in runtime_roots:
        add_site_packages(root)
    for root in runtime_roots:
        lib = root / "lib"
        if not lib.is_dir():
            continue
        for candidate in sorted(lib.glob("python*/site-packages/*/lib/python*/site-packages")):
            if candidate.is_dir() and candidate not in paths:
                paths.append(candidate)
    return paths


def _prepend_robodojo_import_paths(paths: list[Path]) -> None:
    resolved_paths = [item.resolve() for item in paths if item.exists()]
    for path in reversed(resolved_paths):
        text = str(path)
        if text in sys.path:
            sys.path.remove(text)
        sys.path.insert(0, text)


def _checkpoint_dirs(adapter_dir: Path) -> list[Path]:
    root = adapter_dir / "checkpoints"
    if not root.is_dir():
        return []
    return sorted(path.resolve() for path in root.iterdir() if path.is_dir())


def _checkpoint_payload_ready(path: Path, *, policy_name: str) -> bool:
    if not path.is_dir():
        return False
    roots = [path, *(child for child in path.iterdir() if child.is_dir())]
    if policy_name in {"Pi_0", "Pi_05", "Pi_0_Fast"}:
        return any(_pi_checkpoint_payload_ready(root) for root in roots)
    if policy_name == "ACT":
        required = (path / "policy_last.ckpt", path / "dataset_stats.pkl")
        return all(item.is_file() and item.stat().st_size > 0 for item in required)
    return any(item.is_file() for item in path.rglob("*"))


def _pi_checkpoint_payload_ready(root: Path) -> bool:
    if not (root / "assets").exists():
        return False
    if (root / "params").exists():
        return True
    model_path = root / "model.safetensors"
    if not model_path.is_file() or model_path.stat().st_size <= 0:
        return False
    parts_dir = root / ".parts"
    complete_marker = root / "model.safetensors.complete.json"
    if not parts_dir.exists():
        return True
    if not complete_marker.is_file():
        return False
    try:
        marker = json.loads(complete_marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    expected_size = marker.get("expected_size")
    if expected_size is None:
        return False
    try:
        return model_path.stat().st_size == int(expected_size)
    except (TypeError, ValueError, OSError):
        return False


def _load_simple_yaml(path: Path) -> JsonDict:
    if not path.is_file():
        return {}
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception:
        result: JsonDict = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip().strip("\"'")
        return result


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)
