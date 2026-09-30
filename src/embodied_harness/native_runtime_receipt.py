"""Content receipts for benchmark-native runtimes and case asset closures.

The native harness intentionally does not depend on the historical container
stack.  A receipt therefore binds the Python prefix that is actually launched,
its installed-package inventory, pinned source revisions, and the assets used
by the representative case.  Full-content hashing is optional because several
benchmark trees are tens of gigabytes; a presence-only receipt never claims to
be a content seal.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from .native_case_registry import (
    native_asset_declarations,
    native_runtime_lock_declarations,
    resolve_native_runtime,
)
from .paths import get_project_paths


NATIVE_RUNTIME_RECEIPT_SCHEMA = "agentic-embodied-arena/native-runtime-receipt/v1"
_IGNORED_TREE_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache"})
_IGNORED_TREE_SUFFIXES = (".pyc", ".pyo")
_SECRET_TREE_NAMES = frozenset({"omnigibson.key"})
_INTERPRETER_PROBE = r"""
import importlib.metadata as metadata
import json
import platform
import sys
import sysconfig

packages = []
for distribution in metadata.distributions():
    name = str(distribution.metadata.get("Name") or "").strip()
    if not name:
        continue
    direct_url = None
    try:
        raw_direct_url = distribution.read_text("direct_url.json")
        if raw_direct_url:
            direct_url = json.loads(raw_direct_url)
    except Exception:
        direct_url = None
    packages.append({
        "name": name,
        "normalized_name": name.lower().replace("_", "-").replace(".", "-"),
        "version": str(distribution.version),
        "direct_url": direct_url,
    })
packages.sort(key=lambda item: (item["normalized_name"], item["version"], json.dumps(item["direct_url"], sort_keys=True)))
print(json.dumps({
    "implementation": platform.python_implementation(),
    "version": platform.python_version(),
    "version_detail": sys.version,
    "executable": sys.executable,
    "prefix": sys.prefix,
    "base_prefix": sys.base_prefix,
    "cache_tag": getattr(sys.implementation, "cache_tag", None),
    "soabi": sysconfig.get_config_var("SOABI"),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": packages,
}, ensure_ascii=False, sort_keys=True))
"""


class NativeRuntimeReceiptError(RuntimeError):
    """A native runtime could not be measured or its receipt is invalid."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision(
    path: Path, *, ignore_submodule_dirty: bool = False
) -> tuple[str | None, bool | None]:
    if not path.is_dir():
        return None, None
    revision = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )
    if revision.returncode:
        return None, None
    status_command = [
        "git",
        "-C",
        str(path),
        "status",
        "--porcelain",
        "--untracked-files=no",
    ]
    if ignore_submodule_dirty:
        status_command.append("--ignore-submodules=dirty")
    status_result = subprocess.run(
        status_command,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    dirty = None if status_result.returncode else bool(status_result.stdout.strip())
    return revision.stdout.strip(), dirty


def _git_diff_sha256(path: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(path), "diff", "--binary", "--no-ext-diff", "--no-color"],
        check=False,
        capture_output=True,
        timeout=60,
    )
    return _sha256_bytes(completed.stdout) if completed.returncode == 0 else None


def _git_untracked_paths(path: Path) -> list[str] | None:
    completed = subprocess.run(
        [
            "git",
            "-C",
            str(path),
            "ls-files",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        check=False,
        capture_output=True,
        timeout=30,
    )
    if completed.returncode:
        return None
    return sorted(
        item.decode("utf-8", errors="surrogateescape")
        for item in completed.stdout.split(b"\0")
        if item
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeRuntimeReceiptError(
            f"cannot read JSON object {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise NativeRuntimeReceiptError(f"expected JSON object: {path}")
    return value


def _operation_row(benchmark_id: str) -> dict[str, Any]:
    paths = get_project_paths()
    document = _load_json(
        paths.project_root / "configs/operation_environment_lock.json"
    )
    row = next(
        (
            item
            for item in document.get("benchmarks", [])
            if isinstance(item, dict) and item.get("benchmark_id") == benchmark_id
        ),
        None,
    )
    if row is None:
        raise NativeRuntimeReceiptError(
            f"benchmark {benchmark_id!r} is absent from operation_environment_lock.json"
        )
    return row


def _source_rows(benchmark_id: str) -> list[dict[str, Any]]:
    paths = get_project_paths()
    document = _load_json(paths.project_root / "external/sources.lock.yaml")
    declaration = next(
        (
            item
            for item in document.get("sources", [])
            if isinstance(item, dict) and item.get("benchmark_id") == benchmark_id
        ),
        None,
    )
    if declaration is None:
        raise NativeRuntimeReceiptError(
            f"benchmark {benchmark_id!r} is absent from sources.lock.yaml"
        )
    checkout_root = paths.external_root / str(declaration.get("checkout") or "")
    operation = _operation_row(benchmark_id)
    source_patches = [
        item for item in operation.get("source_patches", []) if isinstance(item, dict)
    ]
    rows: list[dict[str, Any]] = []
    for repository in declaration.get("repositories", []):
        if not isinstance(repository, dict):
            continue
        checkout = checkout_root / str(repository.get("checkout") or ".")
        patch_checkouts = [
            _declared_path(str(item.get("checkout") or ""))
            for item in source_patches
            if item.get("checkout")
        ]
        # The declared-patch row below checks revision, complete working diff,
        # patch-file digest and untracked files. Use it for a patched checkout
        # itself, as well as for patched submodules, without also demanding a
        # clean copy of that same checkout.
        own_patch = next((item for item in source_patches
                          if item.get("checkout") and
                          _declared_path(str(item["checkout"])) == checkout), None)
        if own_patch is not None:
            if own_patch.get("revision") != repository.get("revision"):
                raise NativeRuntimeReceiptError(
                    f"declared patch revision differs from source lock: {checkout}"
                )
            continue
        ignore_submodule_dirty = any(
            patch_checkout != checkout and patch_checkout.is_relative_to(checkout)
            for patch_checkout in patch_checkouts
        )
        actual, dirty = _git_revision(
            checkout, ignore_submodule_dirty=ignore_submodule_dirty
        )
        expected = str(repository.get("revision") or "")
        rows.append(
            {
                "repository_id": repository.get("repository_id"),
                "checkout": str(checkout),
                "expected_revision": expected,
                "actual_revision": actual,
                "revision_matches": actual == expected,
                "tracked_files_dirty": dirty,
                "ignore_submodule_dirty": ignore_submodule_dirty,
                "working_tree_matches_declaration": dirty is False,
            }
        )
    for source_patch in source_patches:
        checkout = _declared_path(str(source_patch.get("checkout") or ""))
        patch_path = paths.project_root / str(source_patch.get("patch") or "")
        expected_revision = str(source_patch.get("revision") or "")
        expected_patch_sha256 = str(source_patch.get("sha256") or "")
        actual_revision, dirty = _git_revision(checkout)
        actual_patch_sha256 = _git_diff_sha256(checkout)
        untracked_paths = _git_untracked_paths(checkout)
        patch_file_sha256 = _sha256_file(patch_path) if patch_path.is_file() else None
        patch_matches = bool(
            dirty is True
            and actual_revision == expected_revision
            and actual_patch_sha256 == expected_patch_sha256
            and patch_file_sha256 == expected_patch_sha256
            and untracked_paths == []
        )
        rows.append(
            {
                "repository_id": f"declared_patch:{checkout.name}",
                "checkout": str(checkout),
                "expected_revision": expected_revision,
                "actual_revision": actual_revision,
                "revision_matches": actual_revision == expected_revision,
                "tracked_files_dirty": dirty,
                "untracked_paths": untracked_paths,
                "ignore_submodule_dirty": False,
                "working_tree_matches_declaration": patch_matches,
                "declared_patch": {
                    "path": str(patch_path),
                    "expected_sha256": expected_patch_sha256,
                    "file_sha256": patch_file_sha256,
                    "working_diff_sha256": actual_patch_sha256,
                    "matches": patch_matches,
                },
            }
        )
    return rows


def _declared_path(value: str) -> Path:
    paths = get_project_paths()
    relative = Path(value)
    if relative.is_absolute():
        return relative
    if relative.parts and relative.parts[0] == "external":
        return paths.external_root.joinpath(*relative.parts[1:])
    return paths.project_root / relative


def _native_asset_rows(benchmark_id: str) -> list[dict[str, Any]]:
    binding = resolve_native_runtime(benchmark_id, require_assets=False)
    binding_paths = list(binding.asset_paths)
    remaining = list(binding_paths)
    rows: list[dict[str, Any]] = []
    for item in native_asset_declarations(benchmark_id):
        path = _declared_path(str(item["path"]))
        if path not in remaining:
            continue
        remaining.remove(path)
        kind = str(item.get("kind") or "directory")
        exists = path.is_file() if kind == "file" else path.is_dir()
        rows.append(
            {
                "asset_id": str(item.get("asset_id") or ""),
                "path": str(path),
                "kind": kind,
                "exists": exists,
                "declared_sha256": item.get("sha256"),
            }
        )
    if remaining:
        raise NativeRuntimeReceiptError(
            "runtime binding contains assets absent from the operation declaration: "
            + ", ".join(str(path) for path in remaining)
        )
    if benchmark_id == "capx":
        group = os.environ.get("ARENA_REPORTING_BENCHMARK", "")
        extra = {}
        if group == "capx_libero_pro":
            for key in ("CAPX_LIBERO_ROOT", "CAPX_LIBERO_ROBOSUITE_ROOT", "CAPX_CASE_CONFIG_ROOT", "LIBERO_CONFIG_PATH"):
                if os.environ.get(key): extra[key] = Path(os.environ[key])
            if os.environ.get("CAPX_SIDECAR_PYTHON"):
                extra["shared_capx_environment"] = Path(os.environ["CAPX_SIDECAR_PYTHON"]).parent.parent
            description = Path.home() / ".cache/robot_descriptions/example-robot-data"
            if description.is_dir(): extra["libero_ik_robot_description"] = description
        elif group == "capx_behavior1k" and os.environ.get("OMNIGIBSON_DATA_PATH"):
            extra["behavior_dataset"] = Path(os.environ["OMNIGIBSON_DATA_PATH"])
        for asset_id, path in extra.items():
            rows.append(dict(asset_id=asset_id, path=str(path), kind="directory",
                             exists=path.is_dir(), declared_sha256=None))
    return rows


def _interpreter_inventory(python: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(python), "-I", "-c", _INTERPRETER_PROBE],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if completed.returncode:
        raise NativeRuntimeReceiptError(
            f"interpreter inventory failed for {python}: {completed.stderr[-4000:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise NativeRuntimeReceiptError(
            f"interpreter inventory returned invalid JSON for {python}: {exc}"
        ) from exc
    if not isinstance(value, dict) or not isinstance(value.get("packages"), list):
        raise NativeRuntimeReceiptError(
            f"interpreter inventory is incomplete for {python}"
        )
    return value


def _conda_inventory(prefix: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((prefix / "conda-meta").glob("*.json")):
        value = _load_json(path)
        rows.append(
            {
                "name": value.get("name"),
                "version": value.get("version"),
                "build": value.get("build"),
                "build_number": value.get("build_number"),
                "channel": value.get("channel"),
                "subdir": value.get("subdir"),
                "url": value.get("url"),
                "md5": value.get("md5"),
                "sha256": value.get("sha256"),
                "record_sha256": _sha256_file(path),
            }
        )
    return rows


def _should_ignore(relative: Path) -> bool:
    return bool(
        any(part in _IGNORED_TREE_NAMES for part in relative.parts)
        or relative.name.endswith(_IGNORED_TREE_SUFFIXES)
        or relative.name in _SECRET_TREE_NAMES
    )


def _tree_entries(root: Path) -> Iterable[dict[str, Any]]:
    """Yield stable content records without following symlinks."""

    if root.is_symlink():
        root = root.resolve(strict=True)
    if root.is_file():
        info = root.stat()
        yield {
            "path": "",
            "type": "file",
            "mode": stat.S_IMODE(info.st_mode),
            "size": info.st_size,
            "sha256": _sha256_file(root),
        }
        return
    if not root.is_dir():
        raise NativeRuntimeReceiptError(f"content root is missing: {root}")
    stack = [root]
    while stack:
        directory = stack.pop()
        children = sorted(directory.iterdir(), key=lambda item: item.name, reverse=True)
        directories: list[Path] = []
        for path in reversed(children):
            relative = path.relative_to(root)
            if _should_ignore(relative):
                continue
            info = path.lstat()
            common = {
                "path": relative.as_posix(),
                "mode": stat.S_IMODE(info.st_mode),
            }
            if stat.S_ISLNK(info.st_mode):
                yield {
                    **common,
                    "type": "symlink",
                    "target": os.readlink(path),
                }
            elif stat.S_ISREG(info.st_mode):
                yield {
                    **common,
                    "type": "file",
                    "size": info.st_size,
                    "sha256": _sha256_file(path),
                }
            elif stat.S_ISDIR(info.st_mode):
                yield {**common, "type": "directory"}
                directories.append(path)
            else:
                raise NativeRuntimeReceiptError(
                    f"content tree contains unsupported special file: {path}"
                )
        stack.extend(reversed(directories))


def content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    byte_count = 0
    for entry in _tree_entries(path):
        digest.update(_canonical_bytes(entry))
        count += 1
        byte_count += int(entry.get("size") or 0)
    return {
        "sha256": digest.hexdigest(),
        "entry_count": count,
        "file_bytes": byte_count,
        "ignored_directory_names": sorted(_IGNORED_TREE_NAMES),
        "ignored_file_suffixes": list(_IGNORED_TREE_SUFFIXES),
        "excluded_secret_file_names": sorted(_SECRET_TREE_NAMES),
    }


def _root_stat(path: Path) -> dict[str, Any] | None:
    try:
        info = path.lstat()
    except OSError:
        return None
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "symlink_target": os.readlink(path) if path.is_symlink() else None,
    }


def _freeze_lines(packages: Iterable[Mapping[str, Any]]) -> list[str]:
    lines = []
    for package in packages:
        name = str(package.get("name") or "")
        version = str(package.get("version") or "")
        if name and version:
            lines.append(f"{name}=={version}")
    return sorted(lines, key=str.lower)


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(payload)
        temporary = Path(stream.name)
    os.replace(temporary, path)


def capture_native_runtime_receipt(
    benchmark_id: str,
    *,
    output_root: Path | None = None,
    hash_runtime: bool = False,
    hash_assets: bool = False,
    identity_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture one benchmark runtime and optionally seal its full content."""

    paths = get_project_paths()
    binding = resolve_native_runtime(benchmark_id, require_assets=False)
    python = binding.python_executable
    interpreter = _interpreter_inventory(python)
    prefix = Path(str(interpreter["prefix"]))
    base_prefix = Path(str(interpreter["base_prefix"]))
    resolved_python = python.resolve(strict=True)
    source_rows = _source_rows(benchmark_id)
    asset_rows = _native_asset_rows(benchmark_id)
    rebuild_locks = [
        {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for path in native_runtime_lock_declarations(benchmark_id)
    ]

    def measured_identity(path: Path) -> dict[str, Any]:
        key = str(path.resolve(strict=True))
        if identity_cache is not None and key in identity_cache:
            return dict(identity_cache[key])
        identity = content_identity(path)
        if identity_cache is not None:
            identity_cache[key] = dict(identity)
        return identity

    for asset in asset_rows:
        asset_path = Path(str(asset["path"]))
        asset["root_stat"] = _root_stat(asset_path)
        asset["content_identity"] = (
            measured_identity(asset_path) if hash_assets and asset["exists"] else None
        )
    runtime_identity = measured_identity(prefix) if hash_runtime else None
    base_runtime_identity = (
        measured_identity(base_prefix)
        if hash_runtime and base_prefix.resolve() != prefix.resolve()
        else runtime_identity
    )
    conda_packages = _conda_inventory(prefix)
    packages = list(interpreter["packages"])
    package_inventory_sha256 = _sha256_bytes(_canonical_bytes(packages))
    conda_inventory_sha256 = _sha256_bytes(_canonical_bytes(conda_packages))
    destination = (
        Path(output_root)
        if output_root is not None
        else paths.artifact_root / "native-runtime-receipts"
    ) / benchmark_id
    freeze_payload = (
        "# Installed-version snapshot; source checkouts are bound in receipt.json.\n"
        + "\n".join(_freeze_lines(packages))
        + "\n"
    ).encode("utf-8")
    freeze_path = destination / "requirements.freeze.txt"
    _write_atomic(freeze_path, freeze_payload)
    declaration_paths = [
        paths.project_root / "configs/operation_environment_lock.json",
        paths.project_root / "configs/operation_runtime_matrix.json",
        paths.project_root / "external/sources.lock.yaml",
        paths.project_root / "benchmarks/operation" / benchmark_id / "benchmark.yaml",
        paths.project_root / "benchmarks/operation" / benchmark_id / "cases.yaml",
        paths.project_root / "src/embodied_harness/native_case_registry.py",
        paths.project_root / "src/embodied_harness/native_runtime_receipt.py",
        paths.project_root / "src/embodied_harness/runtime_terms.py",
        paths.project_root / "src/embodied_harness/native_agent_loop.py",
        paths.project_root / "src/embodied_harness/pool_coordinate_bridge.py",
        paths.project_root / "src/embodied_harness/native_prompting.py",
        paths.project_root / "src/embodied_harness/subprocess_backend_bridge.py",
        paths.project_root / "src/embodied_harness/universal_interface.py",
        paths.project_root / "src/embodied_harness/live_api_agent_smoke.py",
        paths.project_root / "scripts/native_backend_jsonl_worker.py",
        paths.project_root / "scripts/run_benchmark_case.py",
        paths.project_root / "src/embodied_harness/case_study_recording.py",
    ]
    route_closure = {
        "behavior1k": paths.project_root / "scripts/materialize_behavior1k_assets.py",
        "calvin": paths.project_root / "scripts/materialize_calvin_native_runtime.py",
        "capx": paths.project_root / "configs/capx_minimal_closure.json",
        "robodojo": paths.project_root / "configs/robodojo_case_closure.json",
    }.get(benchmark_id)
    if route_closure is not None:
        declaration_paths.append(route_closure)
    backend_sources = {
        "behavior1k": ("behavior1k_agent_runtime.py",),
        "capx": ("capx_comparator_runtime.py", "behavior1k_agent_runtime.py",
                 "capx_checked_manipulation.py", "capx_navigation_guard.py", "capx_carry_navigation.py",
                 "capx_planner_collision_policy.py"),
        "calvin": ("calvin_agent_runtime.py",),
        "cliport": ("cliport_agent_runtime.py",),
        "maniskill": ("maniskill_agent_runtime.py",),
        "rlbench": ("rlbench_agent_runtime.py",),
        "robocasa": ("robocasa_agent_runtime.py",),
        "robocasa365": (
            "robocasa_agent_runtime.py",
            "robocasa365_agent_runtime.py",
        ),
        "robodojo": ("robodojo_agent_runtime.py",),
        "robotwin2": ("robotwin2_agent_runtime.py",),
        "robowits": ("robowits_agent_runtime.py",),
        "vimabench": ("vimabench_agent_runtime.py",),
        "vlabench": ("vlabench_runtime.py",),
    }.get(benchmark_id, ())
    for backend_source in backend_sources:
        declaration_paths.append(
            paths.project_root / "src/embodied_harness" / backend_source
        )
    if benchmark_id == "calvin":
        declaration_paths.append(
            paths.project_root / "scripts/calvin_native_runtime_probe.py"
        )
    for source_patch in _operation_row(benchmark_id).get("source_patches", []):
        if isinstance(source_patch, dict) and source_patch.get("patch"):
            declaration_paths.append(paths.project_root / str(source_patch["patch"]))
    declaration_paths = list(dict.fromkeys(declaration_paths))
    # Bind the actual visual loop, action/recording code and budget entry point.
    # Previously these new modules could change without invalidating a receipt.
    declaration_paths.extend(paths.project_root / 'src/embodied_harness' / name for name in (
        'w4_rgb.py', 'stream_video.py', 'hq_recording.py', 'episode_loop.py',
        'capx_libero_support.py', 'native_episode_budget.py'))
    declaration_paths.append(paths.project_root / 'scripts/run_native_universal_case.py')
    receipt: dict[str, Any] = {
        "schema_version": NATIVE_RUNTIME_RECEIPT_SCHEMA,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": benchmark_id,
        "project_root": str(paths.project_root),
        "external_root": str(paths.external_root),
        "artifact_root": str(paths.artifact_root),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "runtime": {
            "declared_python": binding.declared_python,
            "python_executable": str(python),
            "resolved_python": str(resolved_python),
            "resolved_python_sha256": _sha256_file(resolved_python),
            "prefix": str(prefix),
            "prefix_root_stat": _root_stat(prefix),
            "content_identity": runtime_identity,
            "base_prefix": str(base_prefix),
            "base_prefix_root_stat": _root_stat(base_prefix),
            "base_content_identity": base_runtime_identity,
            "interpreter": {
                key: value for key, value in interpreter.items() if key != "packages"
            },
            "package_count": len(packages),
            "package_inventory_sha256": package_inventory_sha256,
            "packages": packages,
            "conda_package_count": len(conda_packages),
            "conda_inventory_sha256": conda_inventory_sha256,
            "conda_packages": conda_packages,
            "requirements_snapshot": {
                "path": str(freeze_path),
                "sha256": _sha256_bytes(freeze_payload),
                "rebuild_lock": False,
            },
            "rebuild_locks": rebuild_locks,
        },
        "sources": source_rows,
        "assets": asset_rows,
        "declarations": [
            {"path": str(path), "sha256": _sha256_file(path)}
            for path in declaration_paths
        ],
        "completeness": {
            "functional_runtime": True,
            "source_revisions_match": bool(source_rows)
            and all(row["revision_matches"] for row in source_rows),
            "tracked_sources_clean": all(
                row.get("working_tree_matches_declaration") is True
                for row in source_rows
            ),
            "required_assets_present": all(row["exists"] for row in asset_rows),
            "runtime_content_sealed": (
                runtime_identity is not None and base_runtime_identity is not None
            ),
            "asset_content_sealed": all(
                row["content_identity"] is not None for row in asset_rows
            ),
            "rebuild_lock_complete": bool(rebuild_locks)
            and all(row["exists"] for row in rebuild_locks),
        },
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    _write_atomic(destination / "receipt.json", _canonical_bytes(receipt))
    return receipt


def load_native_runtime_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path)
    if receipt.get("schema_version") != NATIVE_RUNTIME_RECEIPT_SCHEMA:
        raise NativeRuntimeReceiptError(f"unsupported native receipt schema: {path}")
    expected = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    actual = _sha256_bytes(_canonical_bytes(unsigned))
    if expected != actual:
        raise NativeRuntimeReceiptError(f"native receipt digest mismatch: {path}")
    return receipt


def validate_native_runtime_launch_receipt(
    benchmark_id: str,
) -> dict[str, Any]:
    """Validate the cheap launch-time portion of a native content receipt.

    This deliberately does not rehash multi-gigabyte runtime and asset trees;
    strict campaigns should run the full digest audit once before scheduling.
    It does bind every launch to a self-authenticating receipt, the actual
    interpreter binary, current package inventory, declarations, and clean
    source revisions in well under simulator startup time.
    """

    paths = get_project_paths()
    binding = resolve_native_runtime(benchmark_id, require_assets=True)
    receipt_root = Path(os.environ.get("EMBODIED_ARENA_NATIVE_RECEIPT_ROOT") or
                        paths.artifact_root / "native-runtime-receipts")
    if not receipt_root.is_absolute():
        raise NativeRuntimeReceiptError("Native receipt root must be absolute")
    receipt_path = receipt_root / benchmark_id / "receipt.json"
    receipt = load_native_runtime_receipt(receipt_path)
    errors: list[str] = []
    if receipt.get("benchmark_id") != benchmark_id:
        errors.append("benchmark identity mismatch")
    if (
        Path(str(receipt.get("external_root") or "")).resolve()
        != paths.external_root.resolve()
    ):
        errors.append("external root mismatch")
    completeness = receipt.get("completeness")
    required_flags = (
        "functional_runtime",
        "source_revisions_match",
        "tracked_sources_clean",
        "required_assets_present",
        "runtime_content_sealed",
        "asset_content_sealed",
    )
    if not isinstance(completeness, dict):
        errors.append("receipt completeness block is missing")
    else:
        errors.extend(
            f"receipt completeness flag is false: {name}"
            for name in required_flags
            if completeness.get(name) is not True
        )
    runtime = receipt.get("runtime")
    if not isinstance(runtime, dict):
        errors.append("runtime block is missing")
        runtime = {}
    receipt_python = Path(str(runtime.get("python_executable") or ""))
    if receipt_python.absolute() != binding.python_executable.absolute():
        errors.append("resolved launch interpreter path changed")
    target = Path(str(runtime.get("resolved_python") or ""))
    if not target.is_file():
        errors.append("resolved interpreter target is missing")
    elif _sha256_file(target) != runtime.get("resolved_python_sha256"):
        errors.append("resolved interpreter target digest changed")
    try:
        inventory = _interpreter_inventory(binding.python_executable)
    except NativeRuntimeReceiptError as exc:
        errors.append(str(exc))
    else:
        inventory_digest = _sha256_bytes(_canonical_bytes(inventory["packages"]))
        if inventory_digest != runtime.get("package_inventory_sha256"):
            errors.append("installed package inventory changed")
    rebuild_locks = runtime.get("rebuild_locks")
    if isinstance(rebuild_locks, list):
        for lock in rebuild_locks:
            if not isinstance(lock, dict):
                errors.append("runtime rebuild lock binding is malformed")
                continue
            lock_path = Path(str(lock.get("path") or ""))
            expected_exists = lock.get("exists")
            if expected_exists is True:
                if not lock_path.is_file():
                    errors.append(f"runtime rebuild lock is missing: {lock_path}")
                elif _sha256_file(lock_path) != lock.get("sha256"):
                    errors.append(f"runtime rebuild lock changed: {lock_path}")
            elif expected_exists is False:
                if lock_path.exists():
                    errors.append(
                        "runtime rebuild lock appeared after the content seal: "
                        f"{lock_path}"
                    )
            else:
                errors.append("runtime rebuild lock existence binding is malformed")
    sources = receipt.get("sources")
    if not isinstance(sources, list):
        errors.append("source bindings are missing")
    else:
        for source in sources:
            if not isinstance(source, dict):
                errors.append("source binding is malformed")
                continue
            checkout = Path(str(source.get("checkout") or ""))
            revision, dirty = _git_revision(
                checkout,
                ignore_submodule_dirty=bool(source.get("ignore_submodule_dirty")),
            )
            if revision != source.get("expected_revision"):
                errors.append(f"source revision changed: {checkout}")
            declared_patch = source.get("declared_patch")
            if isinstance(declared_patch, dict):
                patch_path = Path(str(declared_patch.get("path") or ""))
                expected_patch = declared_patch.get("expected_sha256")
                if not patch_path.is_file():
                    errors.append(f"declared source patch is missing: {patch_path}")
                elif _sha256_file(patch_path) != expected_patch:
                    errors.append(f"declared source patch changed: {patch_path}")
                if _git_diff_sha256(checkout) != expected_patch:
                    errors.append(f"source working diff changed: {checkout}")
                if _git_untracked_paths(checkout) != []:
                    errors.append(
                        f"declared source patch has untracked files: {checkout}"
                    )
                if dirty is not True:
                    errors.append(f"declared source patch is not applied: {checkout}")
            elif dirty is not False:
                errors.append(f"tracked source is dirty: {checkout}")
    declarations = receipt.get("declarations")
    if not isinstance(declarations, list):
        errors.append("declaration bindings are missing")
    else:
        for declaration in declarations:
            if not isinstance(declaration, dict):
                errors.append("declaration binding is malformed")
                continue
            declaration_path = Path(str(declaration.get("path") or ""))
            if not declaration_path.is_file():
                errors.append(f"declaration is missing: {declaration_path}")
            elif _sha256_file(declaration_path) != declaration.get("sha256"):
                errors.append(f"declaration digest changed: {declaration_path}")
    if errors:
        raise NativeRuntimeReceiptError(
            f"native launch receipt failed for {benchmark_id}: " + "; ".join(errors)
        )
    return {
        "schema_version": receipt["schema_version"],
        "benchmark_id": benchmark_id,
        "receipt_path": str(receipt_path),
        "receipt_sha256": receipt["receipt_sha256"],
        "runtime_content_sealed": True,
        "asset_content_sealed": True,
        "validation_mode": "launch_fast",
    }


__all__ = [
    "NATIVE_RUNTIME_RECEIPT_SCHEMA",
    "NativeRuntimeReceiptError",
    "capture_native_runtime_receipt",
    "content_identity",
    "load_native_runtime_receipt",
    "validate_native_runtime_launch_receipt",
]
