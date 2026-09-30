"""Canonical, success-free reset assets for the native CaP-X case."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from embodied_harness.integrity import IntegrityError, load_json_object, sha256_file


CAPX_RESET_ASSET_SCHEMA_VERSION = "agentic-embodied-arena/capx-live-reset-asset/v1"
CAPX_RAW_RESET_SCHEMA_VERSION = "agentic-embodied-arena/capx-raw-live-reset/v1"
CAPX_CASE_ID = "capx_current_interface_live"
CAPX_SEED = 0
CAPX_LOCKED_REVISION = "53e9966d7a8e2fa7494676772bccc35280f5c0ed"
CAPX_OFFICIAL_SOURCE_SHA256 = (
    "a53b8d44c2285967fd9911affda52c625e55146bc564a96a87f5f93d2fd50232"
)
CAPX_RUNTIME_CONFIG = "external/environments/capx/runtime_config.yaml"
CAPX_OFFICIAL_SOURCE = (
    "external/upstreams/capx/capx/envs/simulators/robosuite_cubes.py"
)
CAPX_RESET_ASSET = "external/assets/capx/current_interface_reset.json"
_FORBIDDEN_KEYS = {
    "full_prompt",
    "official_success",
    "official_verification",
    "reward",
    "success",
    "task_completed",
}


def _assert_no_private_result_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        forbidden = _FORBIDDEN_KEYS.intersection(value)
        if forbidden:
            raise IntegrityError(
                f"CaP-X reset asset contains forbidden result fields: {sorted(forbidden)}"
            )
        for item in value.values():
            _assert_no_private_result_fields(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_private_result_fields(item)


def validate_capx_reset_asset(
    path: Path | str,
    *,
    repository_root: Path | str,
    expected_seed: int | None,
) -> dict[str, Any]:
    """Load and bind a canonical reset asset to the locked source/config/seed."""

    root = Path(repository_root).resolve(strict=True)
    asset_path = Path(path).resolve(strict=True)
    if asset_path.is_symlink() or not asset_path.is_file():
        raise IntegrityError("CaP-X reset asset must be a regular file")
    if asset_path.stat().st_mode & 0o222:
        raise IntegrityError("CaP-X reset asset must be immutable")
    asset = load_json_object(asset_path, require_canonical=True)
    if set(asset) != {
        "benchmark_id",
        "capture_report_sha256",
        "case_id",
        "reset_observation",
        "runtime_config",
        "schema_version",
        "seed",
        "source",
    }:
        raise IntegrityError("CaP-X reset asset fields are invalid")
    if (
        asset.get("schema_version") != CAPX_RESET_ASSET_SCHEMA_VERSION
        or asset.get("benchmark_id") != "capx"
        or asset.get("case_id") != CAPX_CASE_ID
    ):
        raise IntegrityError("CaP-X reset asset identity mismatch")
    seed = asset.get("seed")
    if type(seed) is not int or seed != CAPX_SEED or expected_seed not in {None, seed}:
        raise IntegrityError("CaP-X reset asset seed mismatch")
    config = asset.get("runtime_config")
    source = asset.get("source")
    if not isinstance(config, Mapping) or set(config) != {"path", "sha256"}:
        raise IntegrityError("CaP-X reset asset runtime config binding is invalid")
    if not isinstance(source, Mapping) or set(source) != {
        "official_source_path",
        "official_source_sha256",
        "revision",
    }:
        raise IntegrityError("CaP-X reset asset source binding is invalid")
    config_path = root / CAPX_RUNTIME_CONFIG
    official_path = root / CAPX_OFFICIAL_SOURCE
    if (
        config.get("path") != CAPX_RUNTIME_CONFIG
        or not config_path.is_file()
        or sha256_file(config_path) != config.get("sha256")
    ):
        raise IntegrityError("CaP-X reset asset runtime config mismatch")
    if (
        source.get("revision") != CAPX_LOCKED_REVISION
        or source.get("official_source_path") != CAPX_OFFICIAL_SOURCE
        or source.get("official_source_sha256") != CAPX_OFFICIAL_SOURCE_SHA256
        or not official_path.is_file()
        or sha256_file(official_path) != CAPX_OFFICIAL_SOURCE_SHA256
    ):
        raise IntegrityError("CaP-X reset asset official source mismatch")
    observation = asset.get("reset_observation")
    if not isinstance(observation, Mapping) or not {
        "cubeA_pos",
        "cubeA_quat",
        "cubeB_pos",
        "cubeB_quat",
        "cube_poses",
    }.issubset(observation):
        raise IntegrityError("CaP-X reset asset observation is incomplete")
    _assert_no_private_result_fields(asset)
    return asset


__all__ = [
    "CAPX_CASE_ID",
    "CAPX_LOCKED_REVISION",
    "CAPX_OFFICIAL_SOURCE",
    "CAPX_OFFICIAL_SOURCE_SHA256",
    "CAPX_RAW_RESET_SCHEMA_VERSION",
    "CAPX_RESET_ASSET",
    "CAPX_RESET_ASSET_SCHEMA_VERSION",
    "CAPX_RUNTIME_CONFIG",
    "CAPX_SEED",
    "validate_capx_reset_asset",
]
