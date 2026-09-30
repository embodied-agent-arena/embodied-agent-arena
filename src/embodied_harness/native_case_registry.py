"""OpenHands-free case and pinned-runtime resolution for the native harness."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .backend import EmbodiedBackend
from .paths import (
    EXTERNAL_ROOT_ENV,
    PROJECT_ROOT_ENV,
    get_project_paths,
    resolve_project_root,
)
from .runtime_terms import verify_receipt
from .subprocess_backend_bridge import (
    NATIVE_XVFB_ENV,
    NATIVE_XVFB_SCRIPT_ENV,
    SubprocessBackendBridge,
)

NATIVE_PYTHON_ENV = "EMBODIED_ARENA_NATIVE_PYTHON"
NATIVE_PYTHON_PREFIX = "EMBODIED_ARENA_NATIVE_PYTHON_"
NATIVE_BRIDGE_TIMEOUT_ENV = "EMBODIED_ARENA_NATIVE_TIMEOUT_SECONDS"
NATIVE_POLICY_TIMEOUT_ENV = "EMBODIED_ARENA_NATIVE_POLICY_TIMEOUT_SECONDS"
RUNTIME_TERMS_RECEIPT_ENV = "EMBODIED_ARENA_RUNTIME_TERMS_RECEIPT"
ENVIRONMENT_LOCK = Path("configs/operation_environment_lock.json")

_MODEL_SECRET_PREFIXES = (
    "LLM_",
    "OPENAI_",
    "DASHSCOPE_",
    "OPENROUTER_",
    "ANTHROPIC_",
)
_SECRET_MARKERS = ("API_KEY", "TOKEN", "PASSWORD", "CREDENTIAL", "SECRET")
_RUNTIME_COMPATIBILITY_ALIASES = {"robocasa365": "robocasa"}
_LICENSE_GATED_BENCHMARKS = frozenset({"behavior1k", "robodojo"})
_NATIVE_UNUSED_ASSET_IDS = {
    "behavior1k": {
        # The historical lock describes the former full-dataset/container
        # route.  The native route below binds a selected-case closure and the
        # host-compatible Kit file that were proven live on Ubuntu 22.04.
        "behavior_dataset",
        "omnigibson_minimal_kit",
        "ubuntu20_libgthread_compat",
        "ubuntu20_libglib_compat",
        "ubuntu20_libpcre_compat",
        "ubuntu20_libx11_compat",
        "ubuntu20_libxcb_compat",
        "ubuntu20_libxau_compat",
        "ubuntu20_libxdmcp_compat",
        "ubuntu20_libbsd_compat",
        "ubuntu20_libsm_compat",
        "ubuntu20_libice_compat",
        "ubuntu20_libxext_compat",
        "ubuntu20_libuuid_compat",
        "ubuntu20_libgl_compat",
        "ubuntu20_libglx_compat",
        "ubuntu20_libgldispatch_compat",
        "ubuntu20_libglu_compat",
        "ubuntu20_libxt_compat",
        "ubuntu22_runtime_image",
        "pinned_apptainer_cli",
        "pinned_apptainer_config",
    },
    "calvin": {
        # The evaluated policy is now the coding agent itself.  The native
        # route needs only the official simulator config and hidden task
        # predicate, not the historical MCIL checkpoint or full task dataset.
        "official_validation_runtime_config",
        "official_mcil_policy",
    },
    "capx": {
        # The native backend now performs a real same-episode upstream reset;
        # the old serialized reset is retained only for release provenance.
        "current_interface_live_reset",
    },
    "rlbench": {
        # The native launcher uses the host's pinned Xvfb package and the Qt
        # libraries bundled with CoppeliaSim. These legacy copied directories
        # were only needed by the former container/runtime-image path.
        "xvfb_runtime",
        "qt_xcb_compat",
    },
    "robodojo": {"task_robot_and_layout_assets", "ubuntu22_runtime_image"},
}
_NATIVE_ADDITIONAL_ASSETS: dict[str, tuple[dict[str, str], ...]] = {
    "behavior1k": (
        {
            "asset_id": "turning_on_radio_scene_closure",
            "path": "external/assets/behavior1k/datasets/behavior-1k-assets",
            "kind": "directory",
            "sha256": "b7603d2a1d5b46b8702b6556942a58c5dd7d876dbc704e566cc2058c3a5f98f9",
        },
        {
            "asset_id": "omnigibson_robot_assets_native",
            "path": "external/assets/behavior1k/datasets/omnigibson-robot-assets",
            "kind": "directory",
            "sha256": "0d86d7c78ea1b071d81b4383a3a4327a161f43b5b0521c1473065ed694aeb704",
        },
        {
            "asset_id": "turning_on_radio_challenge_template",
            "path": (
                "external/assets/behavior1k/datasets/2025-challenge-task-instances/"
                "scenes/house_double_floor_lower/json/"
                "house_double_floor_lower_task_turning_on_radio_0_0_template.json"
            ),
            "kind": "file",
            "sha256": "50f0a0de0b47121532fa19fe1665cb60eb40c5f2d9a2df2befe4d6df49293030",
        },
        {
            "asset_id": "omnigibson_minimal_kit_native_ubuntu22",
            "path": (
                "external/assets/behavior1k/runtime/"
                "omnigibson_4_5_0_no_flowusd_no_xr.kit"
            ),
            "kind": "file",
            "sha256": "557ff858dab99a90241b978225456f5cbc074e72aff662b6997f99f676bb9498",
        },
    ),
    "calvin": (
        {
            "asset_id": "native_validation_runtime_config",
            "path": "external/assets/calvin/dataset/.hydra/merged_config.yaml",
            "kind": "file",
            "sha256": "ceb01532a97de8e4da0ead08e50918c0a1d6992c3acc252c4ef0e7545270de23",
        },
    ),
    "capx": (
        {
            "asset_id": "sam3_checkpoint",
            "path": (
                "external/assets/capx/sam3-hf/hub/models--facebook--sam3/"
                "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
            ),
            "kind": "file",
            "sha256": "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e",
        },
        {
            "asset_id": "sam3_config",
            "path": (
                "external/assets/capx/sam3-hf/hub/models--facebook--sam3/"
                "snapshots/3c879f39826c281e95690f02c7821c4de09afae7/config.json"
            ),
            "kind": "file",
            "sha256": "4616385e4b21f2e5e22c875b65679185cbccfa95de42542b9166f7dc3d57160f",
        },
        {
            "asset_id": "contact_graspnet_checkpoint",
            "path": (
                "external/upstreams/capx/capx/third_party/contact_graspnet_pytorch/"
                "checkpoints/contact_graspnet/checkpoints/model.pt"
            ),
            "kind": "file",
            "sha256": "39fc3439d5814043ba64e0715c127c2e8aca6ea376c22614e11af1bc89317762",
        },
        {
            "asset_id": "native_runtime_config",
            "path": "external/environments/capx/runtime_config.yaml",
            "kind": "file",
            "sha256": "ae2331f05ef89f20c7a67e9c4ffd5db16c3f0e700c32cc62c0eb81cc27f097e0",
        },
    ),
    "robodojo": (
        {
            "asset_id": "representative_nine_case_closure",
            "path": "external/assets/robodojo",
            "kind": "directory",
            "sha256": "d0ed00b0ffbc648bf99d2e027d8f33b917a84857b6f6904d04a56f79f8dab95b",
        },
    ),
}
_NATIVE_RUNTIME_LOCK_OVERRIDES: dict[str, tuple[str, ...]] = {
    # The active CALVIN lane is a small uv/pip environment.  Its checkpoint-
    # free lock is generated by materialize_calvin_native_runtime.py; the
    # historical conda+MCIL locks remain untouched for release provenance.
    "calvin": ("external/environments/calvin/requirements-native.lock",),
}


class NativeRuntimeUnavailable(RuntimeError):
    """The selected case has no executable benchmark-native Python."""


@dataclass(frozen=True, slots=True)
class NativeCaseSpec:
    """Small declarative case record used by the native control plane.

    Backend construction is deliberately lazy.  The campaign process can load
    and schedule every case without importing NumPy or any simulator adapter;
    the selected adapter is imported only inside its isolated worker.
    """

    case_id: str
    benchmark_id: str
    task_id: str
    objective: str
    seed: int | None
    split: str
    reset_config: dict[str, Any]
    code_timeout_seconds: float | None
    readiness_tier: str
    counts_toward_official_success: bool
    harness_only_verifier: bool

    def backend_factory(self) -> EmbodiedBackend:
        # Keep the legacy case implementation behind the subprocess boundary.
        # Importing it in the campaign controller would eagerly import every
        # benchmark adapter and defeat the lightweight native path.
        from .live_api_agent_smoke import built_in_live_cases

        matches = built_in_live_cases([self.case_id])
        if len(matches) != 1:
            raise KeyError(
                f"Expected one backend factory for {self.case_id!r}, got {len(matches)}"
            )
        return matches[0].backend_factory()


@dataclass(frozen=True, slots=True)
class NativeRuntimeBinding:
    benchmark_id: str
    python_executable: Path
    declared_python: str
    source_paths: tuple[Path, ...]
    asset_paths: tuple[Path, ...]
    environment: dict[str, str]
    uses_subprocess: bool = True

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "python_executable": str(self.python_executable),
            "declared_python": self.declared_python,
            "source_paths": [str(path) for path in self.source_paths],
            "asset_paths": [str(path) for path in self.asset_paths],
            "uses_subprocess": self.uses_subprocess,
        }


_CASE_TIMEOUT_SECONDS = {
    "behavior1k_full_bddl_toggle": 900.0,
    "calvin_native_turn_off_led": 420.0,
    "calvin_official_mcil_turn_off_led": 420.0,
    "robocasa_start_coffee_machine_button": 180.0,
    "robocasa365_turn_on_microwave_button": 240.0,
    "robotwin2_place_empty_cup": 300.0,
    "robowits_stack_cube_official": 900.0,
    "vlabench_select_toy_skilllib": 300.0,
}
_CASE_READINESS_OVERRIDES = {
    "calvin_native_turn_off_led": "official_solved_candidate",
    "calvin_official_mcil_turn_off_led": "official_solved_candidate",
    "capx_current_interface_live": "official_success",
    "robodojo_general_pickup_current_interface": "official_replay_candidate",
}
_NON_COUNTING_CASES = {
    "robodojo_general_pickup_current_interface",
}


def _native_case_catalog() -> dict[str, NativeCaseSpec]:
    operation_root = resolve_project_root() / "benchmarks/operation"
    result: dict[str, NativeCaseSpec] = {}
    for case_path in sorted(operation_root.glob("*/cases.yaml")):
        try:
            document = json.loads(case_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise NativeRuntimeUnavailable(
                f"Cannot read native case declaration {case_path}: {exc}"
            ) from exc
        benchmark_id = str(document.get("benchmark_id") or case_path.parent.name)
        for row in document.get("cases", []):
            if not isinstance(row, dict):
                continue
            agent_visible = dict(row.get("agent_visible") or {})
            case_id = str(row.get("case_id") or "")
            task_id = str(agent_visible.get("task_id") or "")
            if not case_id or not task_id:
                raise NativeRuntimeUnavailable(
                    f"Native case declaration {case_path} needs case_id and "
                    "agent_visible.task_id"
                )
            if case_id in result:
                raise NativeRuntimeUnavailable(
                    f"Duplicate native case declaration: {case_id!r}"
                )
            reset_config: dict[str, Any] = {}
            if case_id == "vlabench_select_toy_skilllib":
                reset_config["episode_config_file"] = str(
                    resolve_project_root()
                    / "benchmarks/operation/vlabench/select_toy_minimal_episode_config.json"
                )
            result[case_id] = NativeCaseSpec(
                case_id=case_id,
                benchmark_id=benchmark_id,
                task_id=task_id,
                objective=str(agent_visible.get("objective") or ""),
                seed=(int(row["seed"]) if row.get("seed") is not None else None),
                split=str(row.get("split") or "default"),
                reset_config=reset_config,
                code_timeout_seconds=_CASE_TIMEOUT_SECONDS.get(case_id),
                readiness_tier=_CASE_READINESS_OVERRIDES.get(
                    case_id, "official_success"
                ),
                counts_toward_official_success=case_id not in _NON_COUNTING_CASES,
                harness_only_verifier=True,
            )
    return result


def built_in_native_cases(
    case_ids: list[str] | None = None,
) -> list[NativeCaseSpec]:
    """Load native cases from the canonical JSON-compatible YAML catalog."""

    catalog = _native_case_catalog()
    if case_ids is None:
        return list(catalog.values())
    unknown = [case_id for case_id in case_ids if case_id not in catalog]
    if unknown:
        raise KeyError(f"Unknown native case ids: {unknown}")
    return [catalog[case_id] for case_id in case_ids]


def load_native_case(case_id: str) -> NativeCaseSpec:
    cases = built_in_native_cases([case_id])
    if len(cases) != 1:
        raise KeyError(f"Expected one native case for {case_id!r}, got {len(cases)}")
    return cases[0]


def _operation_row(benchmark_id: str) -> dict[str, Any]:
    root = resolve_project_root()
    lock_path = root / ENVIRONMENT_LOCK
    try:
        document = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeRuntimeUnavailable(
            f"Cannot read pinned environment declaration {lock_path}: {exc}"
        ) from exc
    for row in document.get("benchmarks", []):
        if isinstance(row, dict) and row.get("benchmark_id") == benchmark_id:
            return row
    raise NativeRuntimeUnavailable(
        f"No pinned environment declaration for benchmark {benchmark_id!r}"
    )


def _declared_path(relative: str) -> Path:
    paths = get_project_paths()
    value = Path(relative)
    if value.is_absolute():
        return value
    parts = value.parts
    if parts and parts[0] == "external":
        return paths.external_root.joinpath(*parts[1:])
    return paths.project_root / value


def native_asset_declarations(benchmark_id: str) -> list[dict[str, Any]]:
    """Return the assets consumed by the active native route.

    The historical operation lock remains immutable because released runtime
    packs bind its digest.  Small route-specific closures can add assets here;
    their own declaration files are bound into the native content receipt.
    """

    declarations = [
        dict(item)
        for item in _operation_row(benchmark_id).get("assets", [])
        if isinstance(item, dict)
        and item.get("path")
        and str(item.get("asset_id") or "")
        not in _NATIVE_UNUSED_ASSET_IDS.get(benchmark_id, set())
    ]
    declarations.extend(
        dict(item) for item in _NATIVE_ADDITIONAL_ASSETS.get(benchmark_id, ())
    )
    return declarations


def native_runtime_lock_declarations(benchmark_id: str) -> tuple[Path, ...]:
    """Return rebuild-lock files used by the active native route."""

    override = _NATIVE_RUNTIME_LOCK_OVERRIDES.get(benchmark_id)
    if override is not None:
        return tuple(_declared_path(value) for value in override)
    environment = dict(_operation_row(benchmark_id).get("environment") or {})
    return tuple(
        _declared_path(str(environment[name]))
        for name in ("lock_file", "pip_overlay_lock")
        if environment.get(name)
    )


def _python_candidates(benchmark_id: str, declared: str) -> list[Path]:
    key = benchmark_id.upper().replace("-", "_")
    explicit = os.environ.get(f"{NATIVE_PYTHON_PREFIX}{key}") or os.environ.get(
        NATIVE_PYTHON_ENV
    )
    declared_path = _declared_path(declared)
    prefix = get_project_paths().external_environment(benchmark_id)
    alias = _RUNTIME_COMPATIBILITY_ALIASES.get(benchmark_id)
    alias_prefix = get_project_paths().external_environment(alias) if alias else None
    values = [
        *(Path(explicit).expanduser() for _ in (0,) if explicit),
        declared_path,
        prefix / "bin/python",
        prefix / "bin/python3",
        prefix / "bin/python3.10",
        prefix / "bin/python3.11",
        prefix / "base/bin/python",
        prefix / "base/bin/python3.10",
        prefix / "base/bin/python3.11",
        *(
            [
                alias_prefix / "bin/python",
                alias_prefix / "bin/python3",
                alias_prefix / "bin/python3.10",
                alias_prefix / "bin/python3.11",
            ]
            if alias_prefix is not None
            else []
        ),
    ]
    unique: list[Path] = []
    for path in values:
        # Preserve the venv/Conda launcher path. Resolving its ``python``
        # symlink to /usr/bin/python would bypass pyvenv.cfg and site-packages.
        absolute = path.absolute()
        if absolute not in unique:
            unique.append(absolute)
    return unique


def _is_secret_name(name: str) -> bool:
    upper = name.upper()
    return upper.startswith(_MODEL_SECRET_PREFIXES) or any(
        marker in upper for marker in _SECRET_MARKERS
    )


def _sanitized_parent_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    source = os.environ if environ is None else environ
    return {
        str(name): str(value)
        for name, value in source.items()
        if not _is_secret_name(str(name))
    }


def _runtime_authorization_environment(
    benchmark_id: str,
    *,
    project_root: Path,
    external_root: Path,
) -> dict[str, str]:
    """Return Isaac EULA flags only after a current local receipt verifies."""

    if benchmark_id not in _LICENSE_GATED_BENCHMARKS:
        return {}
    configured = os.environ.get(RUNTIME_TERMS_RECEIPT_ENV)
    receipt_path = (
        Path(configured).expanduser()
        if configured
        else project_root / ".arena/local-authorization/runtime-terms-v1.json"
    )
    if configured and not receipt_path.is_absolute():
        receipt_path = project_root / receipt_path
    result = verify_receipt(
        receipt_path,
        required_scopes=(benchmark_id,),
        external_root=external_root,
    )
    if not result.get("ok"):
        errors = ", ".join(str(item) for item in result.get("errors", []))
        raise NativeRuntimeUnavailable(
            f"Runtime terms receipt rejected for {benchmark_id!r}: "
            f"{result.get('path')} ({errors or 'unknown verification error'}). "
            "Run scripts/confirm_runtime_terms.py interactively."
        )
    if benchmark_id == "behavior1k":
        key_path = external_root / "assets/behavior1k/datasets/omnigibson.key"
        try:
            valid_key_file = (
                key_path.is_file()
                and not key_path.is_symlink()
                and key_path.stat().st_size in {44, 45}
            )
        except OSError:
            valid_key_file = False
        if not valid_key_file:
            raise NativeRuntimeUnavailable(
                "The user-provided OmniGibson key is missing or malformed at "
                f"{key_path}; the harness never prints, hashes, or copies this secret."
            )
    return {
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
        "EMBODIED_ARENA_RUNTIME_TERMS_RECEIPT_SHA256": str(result["receipt_sha256"]),
    }


def _source_paths(benchmark_id: str) -> tuple[Path, ...]:
    paths = get_project_paths()
    root = paths.external_upstream(benchmark_id)
    candidates = [root]
    if benchmark_id == "behavior1k":
        candidates.extend((root / "OmniGibson", root / "bddl3"))
    elif benchmark_id == "robodojo":
        candidates.extend(
            (
                root / "XPolicyLab",
                root / "third_party/IsaacLab",
                root / "third_party/curobo",
            )
        )
    elif benchmark_id == "robotwin2":
        candidates.append(root / "third_party/curobo")
    elif benchmark_id == "capx":
        if os.environ.get('ARENA_REPORTING_BENCHMARK') == 'capx_libero_pro':
            for name in ('CAPX_LIBERO_ROOT', 'CAPX_LIBERO_ROBOSUITE_ROOT'):
                candidate = Path(os.environ[name]).resolve()
                if not candidate.is_dir():
                    raise NativeRuntimeUnavailable(f'Missing isolated LIBERO source: {candidate}')
                candidates.append(candidate)
        else:
            candidates.append(paths.external_assets("capx") / "robosuite")
    elif benchmark_id == "cliport":
        # The representative environment uses PyBullet only. Upstream imports
        # training-only torch/kornia modules at package import time, so keep
        # the documented import-only stubs ahead of the checkout rather than
        # installing a multi-gigabyte policy-training stack in every worker.
        candidates.insert(
            0,
            paths.project_root / "src/embodied_harness/cliport_runtime_stubs",
        )
    return tuple(path for path in candidates if path.exists())


def _runtime_environment(
    benchmark_id: str,
    *,
    source_paths: tuple[Path, ...],
    asset_paths: tuple[Path, ...],
) -> dict[str, str]:
    paths = get_project_paths()
    env = _sanitized_parent_environment()
    python_paths = [paths.project_root / "src", *source_paths]
    inherited_pythonpath = env.get("PYTHONPATH")
    if inherited_pythonpath:
        python_paths.extend(
            Path(item) for item in inherited_pythonpath.split(os.pathsep) if item
        )
    env.update(
        {
            PROJECT_ROOT_ENV: str(paths.project_root),
            EXTERNAL_ROOT_ENV: str(paths.external_root),
            "PYTHONPATH": os.pathsep.join(
                dict.fromkeys(str(path) for path in python_paths)
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(
        _runtime_authorization_environment(
            benchmark_id,
            project_root=paths.project_root,
            external_root=paths.external_root,
        )
    )
    cache_root = paths.artifact_root / "runtime-cache" / benchmark_id
    cache_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    env.setdefault("NUMBA_CACHE_DIR", str(cache_root / "numba"))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(cache_root / "torch_extensions"))

    assets = paths.external_assets(benchmark_id)
    if benchmark_id == "behavior1k":
        env.setdefault("BEHAVIOR1K_DATASET_PATH", str(assets / "datasets"))
        env.setdefault("BEHAVIOR_DATASET_PATH", str(assets / "datasets"))
        env.setdefault("OMNIGIBSON_DATASET_PATH", str(assets / "datasets"))
        env.setdefault("OMNIGIBSON_DATA_PATH", str(assets / "datasets"))
        env.setdefault("OMNIGIBSON_ASSET_PATH", str(assets / "datasets"))
        env.setdefault(
            "BEHAVIOR1K_MINIMAL_KIT_PATH",
            str(assets / "runtime/omnigibson_4_5_0_no_flowusd_no_xr.kit"),
        )
        env.setdefault("BEHAVIOR1K_RUNTIME_CACHE_ROOT", str(cache_root))
        env.setdefault(
            "OMNIGIBSON_APPDATA_PATH", str(cache_root / "omnigibson-appdata")
        )
        env.setdefault(
            "BEHAVIOR1K_OMNIGIBSON_ROOT",
            str(paths.external_upstream(benchmark_id) / "OmniGibson"),
        )
        env.setdefault(
            "BEHAVIOR1K_BDDL_ROOT", str(paths.external_upstream(benchmark_id) / "bddl3")
        )
        env["VK_ICD_FILENAMES"] = str(assets / "runtime/nvidia_icd.json")
        env["VK_DRIVER_FILES"] = str(assets / "runtime/nvidia_icd.json")
        env.setdefault("BEHAVIOR1K_MINIMAL_KIT_NO_FLOWUSD", "1")
        env.setdefault("BEHAVIOR1K_ONLINE_OBJECT_SAMPLING", "0")
        env.setdefault("OMNIGIBSON_HEADLESS", "True")
    elif benchmark_id == "calvin":
        env.setdefault("CALVIN_ROOT", str(paths.external_upstream("calvin")))
        env.setdefault(
            "CALVIN_DATASET_ROOT", str(paths.external_assets("calvin") / "dataset")
        )
        env.setdefault(
            "CALVIN_ASSET_DATA_ROOT",
            str(paths.external_upstream("calvin") / "calvin_env/data"),
        )
    elif benchmark_id == "capx":
        env.setdefault("CAPX_ROOT", str(paths.external_upstream("capx")))
        env.setdefault(
            "CAPX_CURRENT_INTERFACE_RESET_ASSET",
            str(assets / "current_interface_reset.json"),
        )
        env.setdefault(
            "CAPX_CONFIG",
            str(paths.external_environment("capx") / "runtime_config.yaml"),
        )
        env.setdefault(
            "AGENTIC_EMBODIED_ARENA_CAPX_ASSET_PYROKI_PANDA_DESCRIPTION",
            str(assets / "pyroki_panda_description"),
        )
        env.setdefault("HF_HOME", str(assets / "sam3-hf"))
        env.setdefault("HF_HUB_OFFLINE", "1")
        env.setdefault(
            "ROBOT_DESCRIPTIONS_CACHE",
            str(assets / "robot-descriptions-cache"),
        )
        env.setdefault("MUJOCO_GL", "egl")
    elif benchmark_id == "robodojo":
        env.setdefault("ROBODOJO_ROOT", str(paths.external_upstream("robodojo")))
        env.setdefault("ROBODOJO_ASSETS_ROOT", str(assets))
        # Isaac/Warp generated caches are not reliable on object-store-backed
        # workspaces. Keep all writable JIT/runtime state on the node-local
        # filesystem while preserving an explicit caller override.
        local_cache_base = Path(
            env.get("EMBODIED_ARENA_LOCAL_CACHE_ROOT") or env.get("TMPDIR") or "/tmp"
        )
        local_cache = local_cache_base / "agentic-embodied-arena-runtime-cache/robodojo"
        local_cache.mkdir(parents=True, exist_ok=True)
        if "XDG_CACHE_HOME" not in os.environ:
            env["XDG_CACHE_HOME"] = str(local_cache / "xdg")
        if "NUMBA_CACHE_DIR" not in os.environ:
            env["NUMBA_CACHE_DIR"] = str(local_cache / "numba")
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            env["TORCH_EXTENSIONS_DIR"] = str(local_cache / "torch_extensions")
        env.setdefault("WARP_CACHE_PATH", str(local_cache / "warp"))

        xdg_runtime = local_cache / "xdg-runtime"
        xdg_runtime.mkdir(parents=True, exist_ok=True)
        xdg_runtime.chmod(0o700)
        if "XDG_RUNTIME_DIR" not in os.environ:
            env["XDG_RUNTIME_DIR"] = str(xdg_runtime)
        implicit_layer_dir = local_cache / "empty-vulkan-implicit-layer"
        implicit_layer_dir.mkdir(parents=True, exist_ok=True)
        env.setdefault("VK_IMPLICIT_LAYER_PATH", str(implicit_layer_dir))
        env.setdefault("DISABLE_LAYER_NV_OPTIMUS_1", "1")
        env.setdefault("DISABLE_VK_LAYER_MESA_device_select", "1")
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        # This virtualized GPU node does not expose a system Vulkan ICD, while
        # the sealed NVIDIA manifest resolves the physical device correctly.
        # Keep the verified manifest as the default and retain an explicit
        # escape hatch for hosts whose container runtime injects its own ICD.
        disable_custom_vulkan = env.get(
            "ROBODOJO_DISABLE_CUSTOM_VULKAN_ICD", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if disable_custom_vulkan:
            env.pop("VK_ICD_FILENAMES", None)
            env.pop("VK_DRIVER_FILES", None)
        else:
            vulkan_manifest = assets / "runtime/nvidia_icd.json"
            env["VK_ICD_FILENAMES"] = str(vulkan_manifest)
            env["VK_DRIVER_FILES"] = str(vulkan_manifest)

        configured_cuda = env.get("ROBODOJO_CUDA_HOME") or env.get("CUDA_HOME")
        cuda_candidates = [
            *(Path(configured_cuda).expanduser() for _ in (0,) if configured_cuda),
            Path("/usr/local/cuda"),
        ]
        cuda_home = next(
            (
                candidate
                for candidate in cuda_candidates
                if (candidate / "bin/nvcc").is_file()
                and (candidate / "include/cuda_runtime_api.h").is_file()
            ),
            None,
        )
        if cuda_home is not None:
            env["CUDA_HOME"] = str(cuda_home)
            env.setdefault("CUDACXX", str(cuda_home / "bin/nvcc"))
        env.setdefault("MAX_JOBS", "4")
        env.setdefault(
            "AGENTIC_EMBODIED_ARENA_NATIVE_PYTHON_ROBODOJO",
            str(paths.external_environment("robodojo") / "bin/python"),
        )
    elif benchmark_id == "cliport":
        env.setdefault("CLIPORT_ROOT", str(paths.external_upstream("cliport")))
        env.setdefault("CLIPORT_SOURCE_DIR", str(paths.external_upstream("cliport")))
    elif benchmark_id == "maniskill":
        env.setdefault("AGENTIC_EMBODIED_ARENA_MANISKILL_RENDER_BACKEND", "gpu")
    elif benchmark_id == "rlbench":
        coppelia_candidates = [
            assets / "CoppeliaSim_Edu_V4_1_0_Ubuntu20_04",
            assets / "CoppeliaSim",
        ]
        coppelia = next(
            (path for path in coppelia_candidates if path.exists()),
            coppelia_candidates[0],
        )
        env.setdefault("COPPELIASIM_ROOT", str(coppelia))
        env.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(coppelia))
        library_path = env.get("LD_LIBRARY_PATH")
        if str(coppelia) not in (library_path or "").split(os.pathsep):
            env["LD_LIBRARY_PATH"] = os.pathsep.join(
                item for item in (str(coppelia), library_path) if item
            )
        env[NATIVE_XVFB_ENV] = "1"
        env[NATIVE_XVFB_SCRIPT_ENV] = str(
            paths.project_root / "scripts/run_with_xvfb.sh"
        )
    elif benchmark_id in {"robocasa", "robocasa365", "vlabench"}:
        env.setdefault("MUJOCO_GL", "egl")
        env.pop("MUJOCO_EGL_DEVICE_ID", None)
        env.pop("EGL_DEVICE_ID", None)
        if benchmark_id in {"robocasa", "robocasa365"}:
            env.setdefault(
                "ROBOCASA_ASSET_CACHE_DIR",
                str(paths.external_assets("robocasa")),
            )
    elif benchmark_id == "robotwin2":
        env.setdefault("ROBOTWIN2_ASSETS_ROOT", str(assets))
        # Warp/NVRTC PCH writes fail on some object-store-backed workspaces.
        # Keep generated caches on the node-local filesystem while leaving
        # explicitly supplied cache locations untouched.
        local_cache_base = Path(
            env.get("EMBODIED_ARENA_LOCAL_CACHE_ROOT") or env.get("TMPDIR") or "/tmp"
        )
        local_cache = (
            local_cache_base / "agentic-embodied-arena-runtime-cache/robotwin2"
        )
        local_cache.mkdir(parents=True, exist_ok=True)
        if "XDG_CACHE_HOME" not in os.environ:
            env["XDG_CACHE_HOME"] = str(local_cache / "xdg")
        env.setdefault("WARP_CACHE_PATH", str(local_cache / "warp"))
        if "TORCH_EXTENSIONS_DIR" not in os.environ:
            env["TORCH_EXTENSIONS_DIR"] = str(local_cache / "torch_extensions")
        env.setdefault("MPLCONFIGDIR", str(local_cache / "matplotlib"))

        # SAPIEN checks both GLVND directories without first checking whether
        # each directory exists. Bind its sealed, runtime-local manifests so
        # a minimal host image does not fail on a missing /etc/glvnd tree.
        runtime_prefix = paths.external_environment("robotwin2")
        sapien_roots = (
            runtime_prefix / "lib/python3.10/site-packages/sapien/vulkan_library",
            runtime_prefix / "base/lib/python3.10/site-packages/sapien/vulkan_library",
        )
        sapien_root = next(
            (
                candidate
                for candidate in sapien_roots
                if (candidate / "nvidia_icd.json").is_file()
                and (candidate / "10_nvidia.json").is_file()
            ),
            None,
        )
        if sapien_root is not None:
            vulkan_manifest = sapien_root / "nvidia_icd.json"
            env.setdefault("VK_ICD_FILENAMES", str(vulkan_manifest))
            env.setdefault("VK_DRIVER_FILES", str(vulkan_manifest))
            env.setdefault(
                "__EGL_VENDOR_LIBRARY_FILENAMES",
                str(sapien_root / "10_nvidia.json"),
            )
        env.setdefault("PYGLET_HEADLESS", "true")
        env.setdefault("PYGLET_HEADLESS_DEVICE", "0")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("EGL_PLATFORM", "surfaceless")

        # cuRobo may JIT extensions during reset. Prefer an explicitly pinned
        # toolkit, then the standard host CUDA locations used by the existing
        # RoboTwin2 launchers.
        configured_cuda = env.get("ROBOTWIN2_CUDA_HOME") or env.get("CUDA_HOME")
        cuda_candidates = [
            *(Path(configured_cuda).expanduser() for _ in (0,) if configured_cuda),
            Path("/usr/local/cuda-12.2"),
            Path("/usr/local/cuda"),
        ]
        cuda_home = next(
            (
                candidate
                for candidate in cuda_candidates
                if (candidate / "bin/nvcc").is_file()
                and (candidate / "include/cuda_runtime_api.h").is_file()
            ),
            None,
        )
        if cuda_home is not None:
            env["CUDA_HOME"] = str(cuda_home)
            env.setdefault("CUDACXX", str(cuda_home / "bin/nvcc"))
            cuda_bin = str(cuda_home / "bin")
            path_parts = [
                item for item in env.get("PATH", "").split(os.pathsep) if item
            ]
            if cuda_bin not in path_parts:
                env["PATH"] = os.pathsep.join([cuda_bin, *path_parts])
            cuda_lib = cuda_home / "lib64"
            if cuda_lib.is_dir():
                library_parts = [
                    item
                    for item in env.get("LD_LIBRARY_PATH", "").split(os.pathsep)
                    if item
                ]
                if str(cuda_lib) not in library_parts:
                    env["LD_LIBRARY_PATH"] = os.pathsep.join(
                        [str(cuda_lib), *library_parts]
                    )
        env.setdefault("MAX_JOBS", "4")
    elif benchmark_id == "robowits":
        env.setdefault("ROBOWITS_DEVICE", "cuda")
        env.setdefault("GENESIS_BACKEND", "cuda")
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        egl_vendor_config = paths.project_root / "configs/nvidia-egl-vendor.json"
        if egl_vendor_config.is_file():
            env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", str(egl_vendor_config))
        env.setdefault("EMBODIED_ARENA_GENESIS_CONSTRAINT_SOLVER", "monolithic")
        env.setdefault("GS_GYM_ASSET_PATHS", str(assets))
    elif benchmark_id == "vimabench":
        env.setdefault(
            "VIMABENCH_SOURCE_DIR", str(paths.external_upstream("vimabench"))
        )

    # Keep the declaration useful to launch diagnostics even when an optional
    # asset is missing; the backend will return the precise missing path.
    for index, path in enumerate(asset_paths):
        env[f"EMBODIED_ARENA_ASSET_{index}"] = str(path)
    return env


def resolve_native_runtime(
    benchmark_id: str, *, require_assets: bool = True
) -> NativeRuntimeBinding:
    row = _operation_row(benchmark_id)
    environment = dict(row.get("environment") or {})
    declared = str(environment.get("python_executable") or "")
    candidates = _python_candidates(benchmark_id, declared)
    executable = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if executable is None:
        tried = ", ".join(str(path) for path in candidates)
        raise NativeRuntimeUnavailable(
            f"Pinned runtime for {benchmark_id!r} is unavailable; tried: {tried}"
        )
    sources = _source_paths(benchmark_id)
    asset_declarations = native_asset_declarations(benchmark_id)
    assets = tuple(_declared_path(str(item["path"])) for item in asset_declarations)
    if require_assets:
        missing = []
        for item, path in zip(asset_declarations, assets, strict=True):
            expected_kind = str(item.get("kind") or "directory")
            exists = path.is_file() if expected_kind == "file" else path.is_dir()
            if not exists:
                missing.append(f"{item.get('asset_id')}={path}")
        if missing:
            raise NativeRuntimeUnavailable(
                f"Required native assets for {benchmark_id!r} are unavailable: "
                + ", ".join(missing)
            )
    return NativeRuntimeBinding(
        benchmark_id=benchmark_id,
        python_executable=executable,
        declared_python=declared,
        source_paths=sources,
        asset_paths=assets,
        environment=_runtime_environment(
            benchmark_id, source_paths=sources, asset_paths=assets
        ),
    )


def make_native_backend(
    case: Any,
    *,
    in_process: bool = False,
    timeout_seconds: float | None = None,
) -> tuple[EmbodiedBackend, NativeRuntimeBinding | None]:
    """Instantiate a case directly or through its isolated pinned runtime."""

    if in_process:
        return case.backend_factory(), None
    binding = resolve_native_runtime(str(case.benchmark_id))
    try:
        bridge_timeout = float(
            timeout_seconds
            if timeout_seconds is not None
            else os.environ.get(NATIVE_BRIDGE_TIMEOUT_ENV, "900")
        )
    except (TypeError, ValueError):
        bridge_timeout = 900.0
    try:
        policy_timeout = float(os.environ.get(NATIVE_POLICY_TIMEOUT_ENV, "0")) or None
    except ValueError:
        policy_timeout = None
    # ``inherit_environment`` is intentionally false for the benchmark
    # worker, so forward only the non-secret frozen pool coordinates.  This
    # keeps the native ABI isolated while allowing an adapter that implements
    # the optional pool hook (or reads its own POOL_* variables) to bind the
    # selected task/episode.  Model credentials and unrelated host state stay
    # outside the child process.
    pool_environment = {
        key: value
        for key, value in os.environ.items()
        if key.startswith("EMBODIED_ARENA_POOL_")
    }
    worker_environment = {**binding.environment, **pool_environment}
    backend = SubprocessBackendBridge(
        case_id=str(case.case_id),
        python_executable=str(binding.python_executable),
        worker_script=resolve_project_root() / "scripts/native_backend_jsonl_worker.py",
        cwd=resolve_project_root(),
        env=worker_environment,
        inherit_environment=False,
        timeout_seconds=bridge_timeout,
        policy_timeout_seconds=policy_timeout,
    )
    return backend, binding


def current_python_binding(benchmark_id: str) -> NativeRuntimeBinding:
    """Small diagnostic binding for explicit in-process smoke tests."""

    return NativeRuntimeBinding(
        benchmark_id=benchmark_id,
        python_executable=Path(sys.executable).resolve(),
        declared_python=sys.executable,
        source_paths=(),
        asset_paths=(),
        environment={},
        uses_subprocess=False,
    )
