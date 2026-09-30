"""Public, evaluator-only live official predicate binding for native workers.

This module is intentionally lightweight enough for the strict Agent Server
capsule.  It contains no campaign evidence writer and exposes no agent tool.
The native JSONL worker calls it only through the bridge's authenticated,
one-shot coordinator socket while the original backend instance is alive.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import inspect
from pathlib import Path
from types import MappingProxyType
from typing import Any

from embodied_harness.integrity import IntegrityError, sha256_file, sha256_json


IDENTITY_FIELDS = (
    "evaluation_campaign_id",
    "run_id",
    "task_id",
    "family_id",
    "case_id",
    "episode_id",
    "attempt_id",
    "attempt_index",
    "coordinator_nonce",
)


@dataclass(frozen=True)
class LiveOfficialSpec:
    operation_id: str
    benchmark_id: str
    catalog_symbol: str
    source_sha256: str
    class_qualname: str
    attribute_name: str
    result_shape: str
    success_key: str | None
    official_object_paths: tuple[tuple[str | int, ...], ...]


def _spec(
    operation_id: str,
    benchmark_id: str,
    symbol: str,
    digest: str,
    shape: str,
    paths: tuple[tuple[str | int, ...], ...],
    success_key: str | None = None,
) -> LiveOfficialSpec:
    qualified = symbol.split(":", 1)[1]
    class_name, attribute = qualified.rsplit(".", 1)
    return LiveOfficialSpec(
        operation_id=operation_id,
        benchmark_id=benchmark_id,
        catalog_symbol=symbol,
        source_sha256=digest,
        class_qualname=class_name,
        attribute_name=attribute,
        result_shape=shape,
        success_key=success_key,
        official_object_paths=paths,
    )


LIVE_OFFICIAL_SPECS = MappingProxyType(
    {
        item.operation_id: item
        for item in (
            _spec("behavior1k_full_bddl_toggle", "behavior1k", "OmniGibson/omnigibson/tasks/task_base.py:BaseTask.success", "6614585f68aee9c012265a91bd0dc6413bf65961bc9d641cec46dae721a61298", "bool", (("_env", "task"), ("_env", "unwrapped", "task"))),
            _spec("capx_current_interface_live", "capx", "capx/envs/simulators/robosuite_cubes.py:FrankaRobosuiteCubesLowLevel.task_completed", "a53b8d44c2285967fd9911affda52c625e55146bc564a96a87f5f93d2fd50232", "bool", (("evaluator_owned_capx_environment",),)),
            _spec("cliport_place_red_in_green", "cliport", "cliport/tasks/task.py:Task.done", "5dd28eefa22925f949edf575d3b85084111c9ca74674c92ca8d5debf5d9d4109", "bool", (("_task",),)),
            _spec("maniskill_pickcube_pd_ee_delta_pos", "maniskill", "mani_skill/envs/tasks/tabletop/pick_cube.py:PickCubeEnv.evaluate", "53bfed0229cd4de2c7e363e7f3c531d5c5c84803c28e1345a2aea1a4972f3168", "mapping", (("_env", "unwrapped"),), "success"),
            _spec("rlbench_reach_target_motion", "rlbench", "rlbench/backend/task.py:Task.success", "02789876d0242afebbb282a9394d519d3af5d0e852e1ffd3b1f7abc9f95a5831", "first_bool", (("_task", "_task"),)),
            _spec("robocasa_start_coffee_machine_button", "robocasa", "robocasa/environments/kitchen/atomic/kitchen_coffee.py:StartCoffeeMachine._check_success", "f6bf57df9c9b83ec74cd7d371587298037f60e76a3fcf7495dc8c2d34797746c", "bool", (("_env",), ("_env", "env"), ("_env", "env", "env"), ("_env", "env", "env", "env"))),
            _spec("robocasa365_turn_on_microwave_button", "robocasa365", "robocasa/environments/kitchen/atomic/kitchen_microwave.py:MicrowavePressButton._check_success", "acd9aa8ab1b55118923d3835388defaf21b28059919a5f7806e43866db868add", "bool", (("_env",), ("_env", "env"), ("_env", "env", "env"), ("_env", "env", "env", "env"))),
            _spec("robodojo_general_pickup_current_interface", "robodojo", "task/RoboDojo/tasks/general_pickup.py:GeneralPickupCommon.run_reward", "1a9eef7fe518f9d2c6805ad1a8f97c1045fa83726f84b9c63f67ffdfabbc40a7", "robodojo_reward_state", (("isaac_subprocess_live_general_pickup",),)),
            _spec("robotwin2_place_empty_cup", "robotwin2", "envs/place_empty_cup.py:place_empty_cup.check_success", "424bbd61f990ebfffb283eb152d17103c2a990edd20f68229203a96a2fd4d581", "bool", (("_env",),)),
            _spec("robowits_stack_cube_official", "robowits", "gs_gym/envs/robowits/14_stack_cubes.py:StackCubesEnv._check_success", "8f2b61c160e90a54a920333304a8be37e18f4ab6b54e81c3c10c3354618a7268", "bool", (("_env", "envs", 0),)),
            _spec("vimabench_visual_manipulation", "vimabench", "vima_bench/tasks/task_suite/instruction_following/simple_manipulation.py:SimpleManipulation.check_success", "bede3935ed72a4d5a051f5a0c42df356039f07a1b860565fd2f7acf353386ece", "first_bool", (("_env", "task"),)),
            _spec("vlabench_select_toy_skilllib", "vlabench", "VLABench/tasks/dm_task.py:LM4ManipBaseTask.should_terminate_episode", "5e0011bf019ac2393a51560eb46750c0319a27c6b36074a3bcf4a6bb04b9e109", "bool", (("vlabench_official_episode", "official_object"),)),
        )
    }
)


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {field: value.get(field) for field in IDENTITY_FIELDS}
    if any(item is None or item == "" for item in result.values()):
        raise IntegrityError("live official request identity is incomplete")
    if type(result["attempt_index"]) is not int or result["attempt_index"] < 1:
        raise IntegrityError("live official attempt_index must be positive")
    return result


def _extract_object(native: Any, spec: LiveOfficialSpec) -> Any:
    matches: dict[int, Any] = {}
    for path in spec.official_object_paths:
        candidate = native
        for attribute in path:
            if isinstance(attribute, int):
                candidate = candidate[attribute] if isinstance(candidate, (list, tuple)) and len(candidate) > attribute else None
            else:
                candidate = getattr(candidate, attribute, None)
            if candidate is None:
                break
        if candidate is not None and any(base.__qualname__ == spec.class_qualname for base in type(candidate).__mro__):
            matches[id(candidate)] = candidate
    if len(matches) != 1:
        raise IntegrityError(
            f"live official object mismatch for {spec.operation_id}: expected exactly one {spec.class_qualname}"
        )
    return next(iter(matches.values()))


def _official_scalar(value: Any) -> Any:
    module = type(value).__module__.split(".", 1)[0]
    item = getattr(value, "item", None)
    if module in {"numpy", "torch"} and callable(item):
        try:
            return item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise IntegrityError(
                "live official verifier returned a non-scalar value"
            ) from exc
    return value


def _flag(backend: Any, attribute: str) -> bool:
    values = getattr(backend, attribute, None)
    try:
        return bool(_official_scalar(values[0]))
    except (IndexError, KeyError, TypeError):
        return False


def _decode(spec: LiveOfficialSpec, result: Any, backend: Any) -> tuple[bool, dict[str, Any]]:
    payload: dict[str, Any] = {"result_shape": spec.result_shape}
    scalar_result = _official_scalar(result)
    first_result = (
        _official_scalar(result[0])
        if isinstance(result, (tuple, list)) and result
        else None
    )
    if spec.result_shape == "bool" and type(scalar_result) is bool:
        success = scalar_result
        payload["value"] = scalar_result
    elif spec.result_shape == "first_bool" and type(first_result) is bool:
        success = first_result
        payload.update(first=first_result, length=len(result))
    elif spec.result_shape == "mapping" and isinstance(result, Mapping):
        value = _official_scalar(result.get(spec.success_key))
        if type(value) is not bool:
            raise IntegrityError("live official mapping success value is not boolean")
        success = value
        payload.update(success_key=spec.success_key, success_value=value, keys=sorted(key for key in result if isinstance(key, str)))
    elif spec.result_shape == "robodojo_reward_state" and result is None:
        is_episode_end = getattr(backend, "is_episode_end", None)
        episode_end = bool(_official_scalar(is_episode_end())) if callable(is_episode_end) else None
        get_score = getattr(backend, "get_score", None)
        if callable(get_score):
            get_score()
        success_flag = _flag(backend, "success")
        end_flag = _flag(backend, "end_flag")
        success = success_flag and (end_flag or episode_end is None)
        payload.update(run_reward_returned_none=True, success_flag_env0=success_flag, end_flag_env0=end_flag)
    else:
        raise IntegrityError("live official verifier returned an unsupported result shape")
    payload["success"] = success
    sha256_json(payload)
    return success, payload


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _success_semantics(spec: LiveOfficialSpec) -> str:
    if spec.success_key is not None:
        return f"{spec.result_shape}:{spec.success_key}"
    return spec.result_shape


def _capture_robodojo_proxy(
    *,
    native_backend: Any,
    spec: LiveOfficialSpec,
    identity: Mapping[str, Any],
    environment_digest: str,
    transcript_digest: str,
) -> dict[str, Any]:
    proxy = getattr(native_backend, "isaac_subprocess_live_general_pickup", None)
    if (
        proxy is None
        or type(proxy).__module__ != "embodied_harness.robodojo_agent_runtime"
        or type(proxy).__qualname__ != "RoboDojoIsaacLiveOfficialProxy"
    ):
        raise IntegrityError("RoboDojo same-episode Isaac proxy is unavailable")
    capture = getattr(proxy, "capture_live_official", None)
    if not callable(capture):
        raise IntegrityError("RoboDojo live official proxy is invalid")
    receipt = capture(
        {
            "identity": dict(identity),
            "environment_digest": environment_digest,
            "transcript_digest": transcript_digest,
        }
    )
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("schema_version")
        != "agentic-embodied-arena/robodojo-live-official-receipt/v1"
        or receipt.get("operation_id") != spec.operation_id
        or receipt.get("official_symbol") != spec.catalog_symbol
        or receipt.get("official_source_sha256") != spec.source_sha256
        or not isinstance(receipt.get("episode_instance_nonce"), str)
        or not receipt.get("episode_instance_nonce")
        or not isinstance(receipt.get("submitted_actions_digest"), str)
        or len(receipt["submitted_actions_digest"]) != 64
    ):
        raise IntegrityError("RoboDojo live official receipt binding is invalid")
    request_binding = receipt.get("request_binding")
    if (
        not isinstance(request_binding, Mapping)
        or request_binding.get("identity") != identity
        or request_binding.get("environment_digest") != environment_digest
        or request_binding.get("transcript_digest") != transcript_digest
        or sha256_json(request_binding) != receipt.get("request_binding_digest")
    ):
        raise IntegrityError("RoboDojo live official request binding is invalid")
    probe_runtime = receipt.get("probe_runtime")
    if not isinstance(probe_runtime, Mapping) or any(
        type(probe_runtime.get(field)) is not int or probe_runtime[field] < 1
        for field in ("pid", "sid", "starttime_ticks")
    ):
        raise IntegrityError("RoboDojo live official probe identity is invalid")
    if not isinstance(probe_runtime.get("mount_namespace"), str):
        raise IntegrityError("RoboDojo live official probe namespace is invalid")
    result = receipt.get("official_result")
    if (
        not isinstance(result, Mapping)
        or result.get("source") != "upstream_official_result"
        or type(result.get("success")) is not bool
        or result.get("success_semantics") != "robodojo_reward_state"
        or result.get("episode_state_before") != "live"
        or result.get("episode_state_after") != "live"
        or type(result.get("started_at_unix")) not in {int, float}
        or type(result.get("finished_at_unix")) not in {int, float}
        or result["finished_at_unix"] < result["started_at_unix"]
    ):
        raise IntegrityError("RoboDojo live official result is invalid")

    def iso_timestamp(value: float) -> str:
        return datetime.fromtimestamp(value, timezone.utc).isoformat().replace(
            "+00:00", "Z"
        )

    return {
        "operation_id": spec.operation_id,
        "benchmark_id": spec.benchmark_id,
        "evaluation_mode": "live_in_episode",
        "catalog_symbol": spec.catalog_symbol,
        "official_source_digest": spec.source_sha256,
        "environment_digest": environment_digest,
        "transcript_digest": transcript_digest,
        "identity": dict(identity),
        "call_arguments_digest": sha256_json({"args": [], "kwargs": {}}),
        "success": result["success"],
        "source": "upstream_official_result",
        "success_semantics": "robodojo_reward_state",
        "started_at": iso_timestamp(float(result["started_at_unix"])),
        "finished_at": iso_timestamp(float(result["finished_at_unix"])),
        "episode_state_before": "live",
        "episode_state_after": "live",
        "stdout": "",
        "stderr": "",
        "official_payload": {
            "same_episode_proxy_receipt": dict(receipt),
            "success": result["success"],
            "reward_state": result.get("payload"),
        },
    }


def capture_live_official_result(
    *,
    native_backend: Any,
    operation_id: str,
    identity: Mapping[str, Any],
    environment_digest: str,
    transcript_digest: str,
) -> dict[str, Any]:
    """Call the fixed official descriptor on the original native backend graph."""

    spec = LIVE_OFFICIAL_SPECS.get(operation_id)
    if spec is None:
        raise IntegrityError("post-hoc or unknown operation rejects live official capture")
    normalized_identity = _identity(identity)
    if normalized_identity["case_id"] != operation_id:
        raise IntegrityError("live official operation identity mismatch")
    for label, digest in (("environment", environment_digest), ("transcript", transcript_digest)):
        if not isinstance(digest, str) or len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise IntegrityError(f"live official {label} digest is invalid")
    if operation_id == "robodojo_general_pickup_current_interface":
        return _capture_robodojo_proxy(
            native_backend=native_backend,
            spec=spec,
            identity=normalized_identity,
            environment_digest=environment_digest,
            transcript_digest=transcript_digest,
        )
    official = _extract_object(native_backend, spec)
    attested_class = next((base for base in type(official).__mro__ if base.__qualname__ == spec.class_qualname), None)
    if attested_class is None:
        raise IntegrityError("live official class is unavailable")
    descriptor = inspect.getattr_static(attested_class, spec.attribute_name, None)
    source_descriptor = descriptor.fget if isinstance(descriptor, property) else descriptor
    if isinstance(source_descriptor, (staticmethod, classmethod)):
        source_descriptor = source_descriptor.__func__
    source_name = inspect.getsourcefile(source_descriptor)
    if descriptor is None or source_name is None:
        raise IntegrityError("live official source descriptor is unavailable")
    source = Path(source_name).resolve()
    if source.is_symlink() or not source.is_file() or sha256_file(source) != spec.source_sha256:
        raise IntegrityError("live official source digest mismatch")
    if inspect.getattr_static(type(official), spec.attribute_name, None) is not descriptor:
        raise IntegrityError("live official backend overrides the source-locked descriptor")
    target = descriptor.__get__(official, type(official)) if hasattr(descriptor, "__get__") else descriptor
    stdout = io.StringIO()
    stderr = io.StringIO()
    started_at = _timestamp()
    call_arguments: dict[str, Any] = {"args": [], "kwargs": {}}
    with redirect_stdout(stdout), redirect_stderr(stderr):
        if operation_id == "vlabench_select_toy_skilllib":
            binding = getattr(native_backend, "vlabench_official_episode", None)
            if binding is None or getattr(binding, "official_object", None) is not official:
                raise IntegrityError("VLABench live official episode binding is unavailable")
            receipt = binding.capture()
            if (
                not isinstance(receipt, Mapping)
                or receipt.get("schema_version")
                != "agentic-embodied-arena/vlabench-same-episode-official/v1"
                or receipt.get("catalog_symbol") != spec.catalog_symbol
                or receipt.get("official_source_digest") != spec.source_sha256
                or receipt.get("same_episode") is not True
                or receipt.get("episode_state_before") != "live"
                or receipt.get("episode_state_after") != "live"
                or type(receipt.get("success")) is not bool
            ):
                raise IntegrityError("VLABench same-episode official receipt is invalid")
            call_arguments = dict(receipt.get("call_arguments") or {})
            if set(call_arguments) != {"args", "kwargs"}:
                raise IntegrityError("VLABench official call arguments are invalid")
            success = bool(receipt["success"])
            payload = {"success": success, "binding_receipt": dict(receipt)}
        else:
            if not callable(target):
                result = target
            else:
                signature = inspect.signature(target)
                required = [
                    parameter
                    for parameter in signature.parameters.values()
                    if parameter.default is inspect.Parameter.empty
                    and parameter.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                ]
                if required:
                    raise IntegrityError("live official predicate requires replay arguments")
                result = target()
            success, payload = _decode(spec, result, official)
    finished_at = _timestamp()
    return {
        "operation_id": operation_id,
        "benchmark_id": spec.benchmark_id,
        "evaluation_mode": "live_in_episode",
        "catalog_symbol": spec.catalog_symbol,
        "official_source_digest": spec.source_sha256,
        "environment_digest": environment_digest,
        "transcript_digest": transcript_digest,
        "identity": normalized_identity,
        "call_arguments_digest": sha256_json(call_arguments),
        "success": success,
        "source": "upstream_official_result",
        "success_semantics": _success_semantics(spec),
        "started_at": started_at,
        "finished_at": finished_at,
        "episode_state_before": "live",
        "episode_state_after": "live",
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "official_payload": payload,
    }


__all__ = ["IDENTITY_FIELDS", "LIVE_OFFICIAL_SPECS", "LiveOfficialSpec", "capture_live_official_result"]
