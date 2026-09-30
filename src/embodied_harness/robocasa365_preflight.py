from __future__ import annotations

import csv
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from .paths import get_project_paths


JsonDict = dict[str, Any]

_PROJECT_PATHS = get_project_paths()
DEFAULT_ROBOCASA365_REPO = _PROJECT_PATHS.external_upstream("robocasa")
DEFAULT_ROBOCASA365_DATASET_ROOT = _PROJECT_PATHS.external_assets("robocasa365")
DEFAULT_ROBOCASA365_ASSET_CANDIDATES = [
    _PROJECT_PATHS.external_assets("robocasa"),
    _PROJECT_PATHS.external_assets("robocasa365"),
    Path.home() / ".robocasa",
    Path.home() / ".robosuite",
]


@dataclass(slots=True)
class RoboCasa365PreflightReport:
    ok: bool
    repo: JsonDict
    packages: JsonDict
    assets: JsonDict
    dataset: JsonDict
    runtime_contract: JsonDict
    blockers: list[str]
    warnings: list[str]

    def to_dict(self) -> JsonDict:
        return {
            "ok": self.ok,
            "repo": self.repo,
            "packages": self.packages,
            "assets": self.assets,
            "dataset": self.dataset,
            "runtime_contract": self.runtime_contract,
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
        }


def inspect_robocasa365_preflight(
    repo_path: str | Path | None = None,
    dataset_root: str | Path | None = None,
    asset_cache_dir: str | Path | None = None,
    python: str | Path | None = None,
    episode_hint: str | None = None,
    task_family_hint: str | None = None,
    require_dataset: bool = False,
    require_episode: bool = False,
) -> RoboCasa365PreflightReport:
    """Inspect RoboCasa365 readiness without creating an env or touching verifiers.

    The smaller RoboCasa runtime can reuse the same robosuite/MuJoCo action
    space, but RoboCasa365 should not be counted as full-ready unless its suite
    expansion assets/dataset can be identified. This preflight makes that gate
    explicit and keeps task success private to the harness verifier.
    """

    resolved_python = _resolve_python(python)
    repo = _inspect_repo(_resolve_repo_path(repo_path), task_family_hint=task_family_hint)
    packages = _inspect_packages(resolved_python)
    assets = _inspect_assets(asset_cache_dir)
    dataset = _inspect_dataset(_resolve_dataset_root(dataset_root), episode_hint, task_family_hint, resolved_python)

    blockers: list[str] = []
    warnings: list[str] = []
    if not packages["robocasa"]["importable"] and not repo["exists"]:
        blockers.append("robocasa365_upstream_or_package_missing")
    if not packages["robosuite"]["importable"]:
        blockers.append("robosuite_package_missing")
    if not packages["mujoco"]["importable"]:
        blockers.append("mujoco_package_missing")
    if not packages["gymnasium"]["importable"]:
        blockers.append("gymnasium_package_missing")
    if repo["exists"] and not repo["task_registry_detected"]:
        warnings.append("robocasa365_task_registry_not_detected")
    if not repo["exists"] and packages["robocasa"]["importable"]:
        warnings.append("robocasa365_repo_missing_using_installed_package")
    if not assets["asset_cache_detected"]:
        blockers.append("robocasa_asset_cache_missing")
    if require_dataset and not dataset["dataset_root_exists"]:
        blockers.append("robocasa365_dataset_root_missing")
    elif not dataset["dataset_root_exists"]:
        warnings.append("robocasa365_dataset_root_missing_not_required")
    elif not dataset["episode_manifest_available"]:
        warnings.append("robocasa365_episode_manifest_not_detected_not_required")
    if require_episode and not dataset["episode_hint_matched"]:
        blockers.append("robocasa365_episode_hint_missing")
    if task_family_hint and not dataset["task_family_hint_matched"] and not repo["task_family_hint_matched"]:
        warnings.append("robocasa365_task_family_hint_not_matched")

    runtime_contract = {
        "benchmark_id": "robocasa365",
        "adapter": "embodied_harness.robocasa365_agent_runtime.RoboCasa365AgentRuntimeBackend",
        "agent_visible_success_checker": False,
        "agent_visible_demo_replay": False,
        "agent_visible_oracle": False,
        "uses_shared_robocasa_visual_pose_action_core": True,
        "full_ready_requires": [
            "robocasa/robosuite/mujoco/gymnasium importable",
            "RoboCasa/RoboCasa365 assets visible",
            "RoboCasa365 dataset or episode manifest visible",
            "one live episode with RGB-D/segmentation and action primitives",
            "harness-side official or task verifier success",
        ],
    }
    return RoboCasa365PreflightReport(
        ok=not blockers,
        repo=repo,
        packages=packages,
        assets=assets,
        dataset=dataset,
        runtime_contract=runtime_contract,
        blockers=blockers,
        warnings=warnings,
    )


def _resolve_repo_path(repo_path: str | Path | None) -> Path:
    value = repo_path or os.environ.get("ROBOCASA365_REPO") or os.environ.get("ROBOCASA_REPO") or DEFAULT_ROBOCASA365_REPO
    return Path(value).expanduser()


def _resolve_dataset_root(dataset_root: str | Path | None) -> Path:
    value = (
        dataset_root
        or os.environ.get("ROBOCASA365_DATASET_ROOT")
        or os.environ.get("ROBOCASA_DATASET_ROOT")
        or DEFAULT_ROBOCASA365_DATASET_ROOT
    )
    return Path(value).expanduser()


def _resolve_python(python: str | Path | None) -> Path | None:
    value = python or os.environ.get("ROBOCASA365_PYTHON") or os.environ.get("ROBOCASA_PYTHON")
    return Path(value).expanduser() if value else None


def _inspect_packages(python: Path | None = None) -> JsonDict:
    if python is not None:
        return _inspect_packages_with_python(python)
    packages = {}
    for package in ["robocasa", "robosuite", "mujoco", "gymnasium"]:
        spec = importlib.util.find_spec(package)
        packages[package] = {
            "importable": spec is not None,
            "origin": getattr(spec, "origin", None) if spec is not None else None,
            "search_locations": [str(path) for path in (spec.submodule_search_locations or [])] if spec is not None else [],
        }
    return packages


def _inspect_packages_with_python(python: Path) -> JsonDict:
    packages = {
        package: {
            "importable": False,
            "origin": None,
            "search_locations": [],
            "probe_python": str(python),
        }
        for package in ["robocasa", "robosuite", "mujoco", "gymnasium"]
    }
    if not python.exists() or not os.access(python, os.X_OK):
        for package in packages.values():
            package["probe_error"] = "python_not_executable"
        return packages
    probe = """
import importlib.util
import json

payload = {}
for package in ["robocasa", "robosuite", "mujoco", "gymnasium"]:
    spec = importlib.util.find_spec(package)
    payload[package] = {
        "importable": spec is not None,
        "origin": getattr(spec, "origin", None) if spec is not None else None,
        "search_locations": [str(path) for path in (spec.submodule_search_locations or [])] if spec is not None else [],
    }
print(json.dumps(payload, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired as exc:
        for package in packages.values():
            package["probe_error"] = "timeout"
            package["probe_stdout_tail"] = str(exc.stdout or "")[-1000:]
        return packages
    except OSError as exc:
        for package in packages.values():
            package["probe_error"] = f"{type(exc).__name__}: {exc}"
        return packages
    try:
        loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        for package in packages.values():
            package["probe_error"] = f"json_parse_failed: {exc}"
            package["probe_returncode"] = completed.returncode
            package["probe_stdout_tail"] = completed.stdout[-1000:]
            package["probe_stderr_tail"] = completed.stderr[-1000:]
        return packages
    for package, status in loaded.items():
        if package not in packages or not isinstance(status, dict):
            continue
        packages[package].update(status)
        packages[package]["probe_python"] = str(python)
        packages[package]["probe_returncode"] = completed.returncode
        packages[package]["probe_stderr_tail"] = completed.stderr[-1000:]
    return packages


def _inspect_repo(repo_path: Path, task_family_hint: str | None = None) -> JsonDict:
    task_files: list[Path] = []
    registry_candidates = [
        repo_path / "robocasa" / "environments" / "kitchen",
        repo_path / "robocasa" / "models" / "assets",
        repo_path / "robocasa" / "utils",
    ]
    if repo_path.exists():
        task_files = [
            path
            for path in (repo_path / "robocasa").rglob("*.py")
            if any(token in path.as_posix().lower() for token in ["kitchen", "task", "env"])
        ][:200]
    task_family_matches = _matching_paths(task_files, task_family_hint)
    return {
        "path": str(repo_path),
        "exists": repo_path.exists(),
        "is_dir": repo_path.is_dir(),
        "python_package_marker": (repo_path / "robocasa" / "__init__.py").is_file(),
        "registry_candidates": [
            {"path": str(path), "exists": path.exists(), "file_count": _count_files(path, limit=500)}
            for path in registry_candidates
        ],
        "task_registry_detected": any(path.exists() for path in registry_candidates) or bool(task_files),
        "task_file_sample": [str(path) for path in task_files[:20]],
        "task_file_count_sampled": len(task_files),
        "task_family_hint": task_family_hint,
        "task_family_hint_matched": bool(task_family_matches),
        "task_family_matches": [str(path) for path in task_family_matches[:20]],
    }


def _inspect_assets(asset_cache_dir: str | Path | None = None) -> JsonDict:
    explicit = Path(asset_cache_dir).expanduser() if asset_cache_dir else None
    env_candidates = [
        os.environ.get("ROBOCASA_ASSET_CACHE_DIR"),
        os.environ.get("ROBOCASA365_ASSET_CACHE_DIR"),
        os.environ.get("ROBOSUITE_ASSET_CACHE_DIR"),
    ]
    candidates: list[Path]
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [Path(value).expanduser() for value in env_candidates if value]
        candidates.extend(DEFAULT_ROBOCASA365_ASSET_CANDIDATES)
    deduped = list(dict.fromkeys(str(path) for path in candidates))
    entries = [_asset_candidate(Path(path)) for path in deduped]
    detected = [entry for entry in entries if entry["exists"] and entry["file_count_sampled"] > 0]
    return {
        "asset_cache_dir_requested": str(explicit) if explicit is not None else None,
        "asset_cache_detected": bool(detected),
        "candidates": entries,
    }


def _asset_candidate(path: Path) -> JsonDict:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir(),
        "file_count_sampled": _count_files(path, limit=500),
        "sample_files": [str(item) for item in _sample_files(path, limit=12)],
    }


def _inspect_dataset(
    dataset_root: Path,
    episode_hint: str | None,
    task_family_hint: str | None,
    registry_python: Path | None = None,
) -> JsonDict:
    manifest_patterns = ["*.json", "*.jsonl", "*.csv", "*.hdf5", "*.h5", "*.npz", "*.pkl", "*.yaml", "*.yml"]
    manifest_files: list[Path] = []
    if dataset_root.exists():
        for pattern in manifest_patterns:
            manifest_files.extend(dataset_root.rglob(pattern))
            if len(manifest_files) >= 500:
                break
    manifest_files = manifest_files[:500]
    episode_matches = _matching_paths(manifest_files, episode_hint)
    task_family_matches = _matching_paths(manifest_files, task_family_hint)
    lerobot_records = _sample_lerobot_episode_records(dataset_root, max_records=80)
    generic_records = _sample_episode_records(manifest_files, max_records=max(0, 80 - len(lerobot_records)))
    records = (lerobot_records + generic_records)[:80]
    registry = _inspect_robocasa_dataset_registry(registry_python, task_family_hint=task_family_hint)
    episode_record_matches = _matching_records(records, episode_hint)
    task_family_record_matches = _matching_records(records, task_family_hint)
    selected_record = _select_agent_episode_spec(records, episode_hint, task_family_hint)
    return {
        "dataset_root": str(dataset_root),
        "dataset_root_exists": dataset_root.exists(),
        "manifest_file_count_sampled": len(manifest_files),
        "manifest_file_sample": [str(path) for path in manifest_files[:20]],
        "episode_manifest_available": bool(records),
        "episode_record_count_sampled": len(records),
        "episode_record_sample": records[:8],
        "lerobot_meta_detected": bool(lerobot_records),
        "lerobot_episode_record_count_sampled": len(lerobot_records),
        "robocasa_dataset_registry": registry,
        "dataset_candidate_available": bool(registry.get("candidate_sample")),
        "agent_dataset_candidate": _select_registry_dataset_candidate(registry, task_family_hint),
        "episode_hint": episode_hint,
        "episode_hint_matched": bool(episode_matches or episode_record_matches),
        "episode_hint_matches": [str(path) for path in episode_matches[:20]],
        "episode_hint_record_matched": bool(episode_record_matches),
        "episode_hint_record_matches": episode_record_matches[:8],
        "task_family_hint": task_family_hint,
        "task_family_hint_matched": bool(
            task_family_matches or task_family_record_matches or registry.get("task_family_hint_matched")
        ),
        "task_family_matches": [str(path) for path in task_family_matches[:20]],
        "task_family_hint_record_matched": bool(task_family_record_matches),
        "task_family_hint_record_matches": task_family_record_matches[:8],
        "agent_episode_spec": selected_record,
    }


def _inspect_robocasa_dataset_registry(python: Path | None, task_family_hint: str | None = None) -> JsonDict:
    base = {
        "available": False,
        "probe_python": str(python) if python is not None else None,
        "soup_count": 0,
        "soup_keys_sample": [],
        "candidate_count_sampled": 0,
        "candidate_sample": [],
        "task_family_hint": task_family_hint,
        "task_family_hint_matched": False,
        "task_family_matches": [],
    }
    if python is None:
        base["probe_error"] = "python_not_configured"
        return base
    if not python.exists() or not os.access(python, os.X_OK):
        base["probe_error"] = "python_not_executable"
        return base
    probe = """
import json

payload = {
    "available": False,
    "soup_count": 0,
    "soup_keys_sample": [],
    "candidate_count_sampled": 0,
    "candidate_sample": [],
}
try:
    from robocasa.utils import dataset_registry as dr

    soups = getattr(dr, "DATASET_SOUP_REGISTRY", {}) or {}
    candidates = []
    for soup_name, entries in soups.items():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            compact = {
                "soup": soup_name,
                "task": entry.get("task"),
                "split": entry.get("split"),
                "source": entry.get("source"),
                "horizon": entry.get("horizon"),
                "filter_key": entry.get("filter_key"),
                "path": entry.get("path"),
            }
            candidates.append(compact)
            if len(candidates) >= 120:
                break
        if len(candidates) >= 120:
            break
    payload.update(
        {
            "available": True,
            "soup_count": len(soups),
            "soup_keys_sample": list(soups)[:40],
            "candidate_count_sampled": len(candidates),
            "candidate_sample": candidates,
        }
    )
except Exception as exc:
    payload["probe_error"] = f"{type(exc).__name__}: {exc}"
print(json.dumps(payload, sort_keys=True))
"""
    try:
        completed = subprocess.run(
            [str(python), "-c", probe],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=45,
        )
    except subprocess.TimeoutExpired as exc:
        base["probe_error"] = "timeout"
        base["probe_stdout_tail"] = str(exc.stdout or "")[-1000:]
        return base
    except OSError as exc:
        base["probe_error"] = f"{type(exc).__name__}: {exc}"
        return base
    try:
        loaded = json.loads(completed.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        base["probe_error"] = f"json_parse_failed: {exc}"
        base["probe_returncode"] = completed.returncode
        base["probe_stdout_tail"] = completed.stdout[-1000:]
        base["probe_stderr_tail"] = completed.stderr[-1000:]
        return base
    if isinstance(loaded, dict):
        base.update(loaded)
    base["probe_returncode"] = completed.returncode
    base["probe_stderr_tail"] = completed.stderr[-1000:]
    matches = _matching_records(base.get("candidate_sample", []), task_family_hint)
    base["task_family_hint_matched"] = bool(matches)
    base["task_family_matches"] = matches[:8]
    return base


def _select_registry_dataset_candidate(registry: JsonDict, task_family_hint: str | None) -> JsonDict | None:
    candidates = registry.get("candidate_sample")
    if not isinstance(candidates, list) or not candidates:
        return None
    matches = _matching_records([candidate for candidate in candidates if isinstance(candidate, dict)], task_family_hint)
    if matches:
        return matches[0]
    first = candidates[0]
    return first if isinstance(first, dict) else None


def _sample_lerobot_episode_records(dataset_root: Path, max_records: int = 80) -> list[JsonDict]:
    if max_records <= 0 or not dataset_root.exists():
        return []
    meta_dir = dataset_root / "meta"
    episodes_path = meta_dir / "episodes.jsonl"
    tasks_path = meta_dir / "tasks.jsonl"
    if not episodes_path.is_file():
        return []
    tasks = _read_lerobot_tasks(tasks_path)
    records: list[JsonDict] = []
    try:
        with episodes_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if len(records) >= max_records:
                    break
                line = line.strip()
                if not line:
                    continue
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    continue
                records.append(_lerobot_episode_record_from_mapping(loaded, episodes_path, tasks, dataset_root))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    return records


def _read_lerobot_tasks(tasks_path: Path) -> dict[int, str]:
    tasks: dict[int, str] = {}
    if not tasks_path.is_file():
        return tasks
    try:
        with tasks_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    continue
                index = loaded.get("task_index", loaded.get("index", loaded.get("id")))
                text = loaded.get("task", loaded.get("instruction", loaded.get("language_instruction")))
                if isinstance(index, int) and text not in (None, ""):
                    tasks[index] = str(text)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return tasks


def _lerobot_episode_record_from_mapping(
    record: JsonDict,
    path: Path,
    tasks: dict[int, str],
    dataset_root: Path,
) -> JsonDict:
    episode_index = record.get("episode_index", record.get("index"))
    task_indices = _as_int_list(record.get("tasks", record.get("task_index")))
    task_texts = [tasks[index] for index in task_indices if index in tasks]
    extras = _read_lerobot_episode_meta(dataset_root, episode_index)
    instruction = (
        _first_present(record, ["instruction", "language_instruction", "task", "task_description"])
        or _first_present(extras, ["instruction", "language_instruction", "task", "task_description", "description"])
        or "; ".join(task_texts)
        or None
    )
    task_family = _first_present(record, ["task_family", "task_type", "category"]) or _first_present(
        extras, ["task_family", "task_type", "category", "env_name", "env_id"]
    )
    if task_family is None and len(dataset_root.parts) >= 3:
        task_family = dataset_root.parts[-3]
    env_id = _first_present(record, ["env_id", "environment", "env"]) or _first_present(extras, ["env_id", "environment", "env"])
    if env_id is None and task_family:
        env_id = f"robocasa365/{task_family}"
    episode_id = _first_present(record, ["episode_id", "episode", "id"])
    if episode_id is None and isinstance(episode_index, int):
        episode_id = f"episode_{episode_index:06d}"
    compact = _compact_record({**extras, **record})
    return {
        "manifest_path": str(path),
        "episode_id": episode_id,
        "episode_index": episode_index,
        "task_family": task_family,
        "env_id": env_id,
        "instruction": instruction,
        "objects": _as_string_list(_first_present(extras, ["objects", "object_names", "target_objects", "objs"])),
        "fixtures": _as_string_list(_first_present(extras, ["fixtures", "fixture_names", "target_fixtures"])),
        "camera_names": _as_string_list(_first_present(extras, ["camera_names", "cameras"])),
        "lerobot_task_indices": task_indices,
        "lerobot_task_texts": task_texts,
        "raw_keys": sorted(str(key) for key in record.keys())[:40],
        "compact_record": compact,
    }


def _read_lerobot_episode_meta(dataset_root: Path, episode_index: Any) -> JsonDict:
    if not isinstance(episode_index, int):
        return {}
    candidates = [
        dataset_root / "extras" / f"episode_{episode_index:06d}" / "ep_meta.json",
        dataset_root / "extras" / f"episode_{episode_index}" / "ep_meta.json",
        dataset_root / "extra" / f"episode_{episode_index:06d}" / "ep_meta.json",
    ]
    for path in candidates:
        if not path.is_file() or path.stat().st_size > 5_000_000:
            continue
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(loaded, dict):
            return loaded
    return {}


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, int)]
    return []


def _sample_episode_records(manifest_files: list[Path], max_records: int = 80) -> list[JsonDict]:
    records: list[JsonDict] = []
    for path in manifest_files:
        if len(records) >= max_records:
            break
        records.extend(_read_manifest_records(path, max_records=max_records - len(records)))
    return records[:max_records]


def _read_manifest_records(path: Path, max_records: int) -> list[JsonDict]:
    suffix = path.suffix.lower()
    if max_records <= 0 or suffix not in {".json", ".jsonl", ".csv", ".yaml", ".yml"}:
        return []
    if not path.is_file() or path.stat().st_size > 5_000_000:
        return []
    try:
        if suffix == ".json":
            loaded = json.loads(path.read_text(encoding="utf-8"))
            return [_episode_record_from_mapping(record, path) for record in _iter_manifest_dicts(loaded)[:max_records]]
        if suffix == ".jsonl":
            records: list[JsonDict] = []
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if len(records) >= max_records:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    loaded = json.loads(line)
                    for record in _iter_manifest_dicts(loaded):
                        records.append(_episode_record_from_mapping(record, path))
                        if len(records) >= max_records:
                            break
            return records
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                return [
                    _episode_record_from_mapping(row, path)
                    for _, row in zip(range(max_records), csv.DictReader(handle), strict=False)
                ]
        return [_episode_record_from_mapping(record, path) for record in _iter_manifest_dicts(_load_yaml(path))[:max_records]]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, csv.Error):
        return []


def _load_yaml(path: Path) -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_manifest_dicts(loaded: Any) -> list[JsonDict]:
    if isinstance(loaded, list):
        return [item for item in loaded if isinstance(item, dict)]
    if not isinstance(loaded, dict):
        return []
    for key in ["episodes", "episode_specs", "tasks", "manifest", "data"]:
        value = loaded.get(key)
        if isinstance(value, list):
            records = [item for item in value if isinstance(item, dict)]
            if records:
                return records
    return [loaded]


def _episode_record_from_mapping(record: JsonDict, path: Path) -> JsonDict:
    compact = _compact_record(record)
    return {
        "manifest_path": str(path),
        "episode_id": _first_present(record, ["episode_id", "episode", "id", "uid", "traj_id", "demo_id"]),
        "task_family": _first_present(record, ["task_family", "family", "task_type", "suite_task", "category"]),
        "env_id": _first_present(record, ["env_id", "environment", "env", "task_id"]),
        "instruction": _first_present(
            record,
            ["instruction", "language_instruction", "language", "task_description", "description", "goal"],
        ),
        "objects": _as_string_list(_first_present(record, ["objects", "object_names", "target_objects", "objs"])),
        "fixtures": _as_string_list(_first_present(record, ["fixtures", "fixture_names", "target_fixtures"])),
        "camera_names": _as_string_list(_first_present(record, ["camera_names", "cameras"])),
        "raw_keys": sorted(str(key) for key in record.keys())[:40],
        "compact_record": compact,
    }


def _compact_record(record: JsonDict) -> JsonDict:
    keep = [
        "episode_id",
        "episode",
        "id",
        "task_family",
        "task_type",
        "env_id",
        "task_id",
        "instruction",
        "language_instruction",
        "task_description",
        "objects",
        "object_names",
        "target_objects",
        "fixtures",
        "fixture_names",
        "target_fixtures",
        "camera_names",
    ]
    compact: JsonDict = {}
    for key in keep:
        if key in record:
            compact[key] = _truncate_value(record[key])
    return compact


def _truncate_value(value: Any) -> Any:
    if isinstance(value, str):
        return value[:500]
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_truncate_value(item) for item in value[:20]]
    if isinstance(value, dict):
        return {str(key): _truncate_value(item) for key, item in list(value.items())[:20]}
    return str(value)[:500]


def _first_present(record: JsonDict, keys: list[str]) -> Any:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _as_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [str(key) for key in value.keys()]
    if isinstance(value, list):
        return [str(item) for item in value[:50]]
    return [str(value)]


def _matching_records(records: list[JsonDict], hint: str | None) -> list[JsonDict]:
    if not hint:
        return []
    needle = hint.lower()
    matches: list[JsonDict] = []
    for record in records:
        haystack = json.dumps(record, ensure_ascii=False, sort_keys=True).lower()
        if needle in haystack:
            matches.append(record)
    return matches


def _select_agent_episode_spec(
    records: list[JsonDict],
    episode_hint: str | None,
    task_family_hint: str | None,
) -> JsonDict | None:
    if not records:
        return None
    for hint in [episode_hint, task_family_hint]:
        matches = _matching_records(records, hint)
        if matches:
            return matches[0]
    return records[0]


def _matching_paths(paths: list[Path], hint: str | None) -> list[Path]:
    if not hint:
        return []
    needle = hint.lower()
    return [path for path in paths if needle in path.as_posix().lower()]


def _count_files(path: Path, limit: int = 500) -> int:
    if not path.exists() or not path.is_dir():
        return 0
    count = 0
    for item in path.rglob("*"):
        if item.is_file():
            count += 1
            if count >= limit:
                break
    return count


def _sample_files(path: Path, limit: int = 12) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []
    sample: list[Path] = []
    for item in path.rglob("*"):
        if item.is_file():
            sample.append(item)
            if len(sample) >= limit:
                break
    return sample
