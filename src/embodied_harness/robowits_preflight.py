from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from .robowits_agent_runtime import DEFAULT_ROBOWITS_REPO, run_robowits_preflight_probe

ROBOWITS_HF_ASSET_REPO = "XHRlyb2001/RoboWits_assets"
_ASSET_REF_RE = re.compile(r"get_asset_path\(\s*['\"]([^'\"]+)['\"]")


def inspect_robowits_asset_preflight(repo_path: str | Path = DEFAULT_ROBOWITS_REPO) -> dict[str, object]:
    """Inspect RoboWits heavy asset readiness without downloading assets or printing secrets."""

    repo = Path(repo_path)
    assets_dir = repo / "assets"
    metadata_path = assets_dir / "metadata.json"
    metadata = _load_asset_metadata(metadata_path)
    task_refs = _scan_robowits_task_asset_refs(repo)
    referenced_hf = sorted({ref for refs in task_refs.values() for ref in refs if ref.startswith("hf_assets/")})
    referenced_blenderkit = sorted(
        {
            parts[1]
            for refs in task_refs.values()
            for ref in refs
            if (parts := ref.split("/")) and len(parts) >= 3 and parts[0] == "blender_kit"
        }
    )
    free_ids = set(metadata.get("free") or [])
    full_plan_ids = set(metadata.get("full_plan") or [])
    required_free = sorted(set(referenced_blenderkit) & free_ids)
    required_full_plan = sorted(set(referenced_blenderkit) & full_plan_ids)
    required_unknown = sorted(set(referenced_blenderkit) - free_ids - full_plan_ids)
    missing_hf = [ref for ref in referenced_hf if not (assets_dir / ref).exists()]
    missing_blenderkit = [
        asset_id for asset_id in referenced_blenderkit if not (assets_dir / "blender_kit" / asset_id / "obj.glb").exists()
    ]
    missing_full_plan = [asset_id for asset_id in required_full_plan if asset_id in missing_blenderkit]
    blenderkit_independent_cases = _blenderkit_independent_cases(repo, task_refs, assets_dir)
    dataset_root = repo / "dataset" / "robowits"
    secret_configured = bool(os.environ.get("BLENDERKIT_API_KEY") or os.environ.get("BLENDERKIT_KEY"))
    blender_binary, blender_source, blender_configured_path = _resolve_blender_binary()

    blockers: list[dict[str, object]] = []
    if not repo.exists():
        blockers.append({"id": "repo_missing", "detail": str(repo)})
    if not (assets_dir / "setup_assets.sh").exists():
        blockers.append({"id": "setup_assets_script_missing", "detail": str(assets_dir / "setup_assets.sh")})
    if missing_hf:
        blockers.append({"id": "hf_assets_missing", "missing_count": len(missing_hf), "sample": missing_hf[:10]})
    if missing_blenderkit:
        blockers.append(
            {"id": "blenderkit_assets_missing", "missing_count": len(missing_blenderkit), "sample": missing_blenderkit[:10]}
        )
    if missing_full_plan:
        blockers.append(
            {
                "id": "blenderkit_full_plan_assets_missing",
                "missing_count": len(missing_full_plan),
                "sample": missing_full_plan[:10],
            }
        )
    if missing_blenderkit and not secret_configured:
        blockers.append({"id": "blenderkit_key_missing_for_download", "detail": "BLENDERKIT_API_KEY/BLENDERKIT_KEY not configured"})
    if missing_blenderkit and blender_binary is None:
        blockers.append(
            {
                "id": "blender_missing_for_preprocess",
                "detail": "blender is not on PATH and ROBOWITS_BLENDER_BINARY/BLENDER_BINARY is not executable",
            }
        )

    return {
        "repo_path": str(repo),
        "assets_dir": str(assets_dir),
        "download_sources": {
            "hf_dataset_repo": ROBOWITS_HF_ASSET_REPO,
            "setup_script": str(assets_dir / "setup_assets.sh"),
            "hf_download_command": "bash assets/setup_assets.sh --skip-blenderkit",
            "blenderkit_download_command_template": "bash assets/setup_assets.sh --skip-hf --api-key <redacted>",
            "blenderkit_downloader": str(assets_dir / "download_blenderkit_asset.py"),
            "blenderkit_key_in_argv": True,
            "secret_values_redacted": True,
        },
        "commands": {
            "hf": shutil.which("hf"),
            "blender": blender_binary,
            "blender_source": blender_source,
            "blender_configured_path": blender_configured_path,
        },
        "secrets": {
            "blenderkit_secret_configured": secret_configured,
            "secret_values_redacted": True,
        },
        "dataset_families": {
            "eval_dataset_50_json_count": _count_files(dataset_root / "eval_dataset_50", "*.json"),
            "eval_dataset_mutation_10_json_count": _count_files(dataset_root / "eval_dataset_mutation_10", "*.json"),
            "eval_dataset_mutation_10_particles_npz_count": _count_files(dataset_root / "eval_dataset_mutation_10", "*.npz"),
        },
        "task_code_families": _task_family_summary(repo),
        "blenderkit_independent_cases": blenderkit_independent_cases,
        "metadata": {
            "exists": metadata_path.exists(),
            "path": str(metadata_path),
            "free_count": len(metadata.get("free") or []),
            "full_plan_count": len(metadata.get("full_plan") or []),
        },
        "required_assets": {
            "hf_asset_paths": referenced_hf,
            "hf_asset_count": len(referenced_hf),
            "blenderkit_asset_ids": referenced_blenderkit,
            "blenderkit_asset_count": len(referenced_blenderkit),
            "blenderkit_free_ids": required_free,
            "blenderkit_free_count": len(required_free),
            "blenderkit_full_plan_ids": required_full_plan,
            "blenderkit_full_plan_count": len(required_full_plan),
            "blenderkit_unknown_ids": required_unknown,
            "blenderkit_unknown_count": len(required_unknown),
        },
        "installed_assets": {
            "hf_present_count": len(referenced_hf) - len(missing_hf),
            "hf_missing_count": len(missing_hf),
            "hf_missing_paths": missing_hf,
            "blenderkit_present_count": len(referenced_blenderkit) - len(missing_blenderkit),
            "blenderkit_missing_count": len(missing_blenderkit),
            "blenderkit_missing_ids": missing_blenderkit,
            "blenderkit_full_plan_missing_count": len(missing_full_plan),
            "blenderkit_full_plan_missing_ids": missing_full_plan,
        },
        "readiness": {
            "hf_assets_ready": bool(referenced_hf) and not missing_hf,
            "blenderkit_free_assets_ready": bool(required_free) and not any(asset_id in missing_blenderkit for asset_id in required_free),
            "blenderkit_full_plan_assets_ready": bool(required_full_plan) and not missing_full_plan,
            "full_family_assets_ready": bool(referenced_hf) and bool(referenced_blenderkit) and not missing_hf and not missing_blenderkit,
            "full_family_blocker": "blenderkit_full_plan_assets_missing" if missing_full_plan else None,
        },
        "blockers": blockers,
    }


def _resolve_blender_binary() -> tuple[str | None, str | None, str | None]:
    configured = os.environ.get("ROBOWITS_BLENDER_BINARY") or os.environ.get("BLENDER_BINARY")
    if configured:
        configured_path = Path(configured).expanduser()
        candidates = [configured_path]
        if configured_path.is_dir():
            candidates = [
                configured_path / "blender",
                configured_path / "blender.exe",
                configured_path / "blender" / "blender",
            ]
        for candidate in candidates:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate), "ROBOWITS_BLENDER_BINARY" if os.environ.get("ROBOWITS_BLENDER_BINARY") else "BLENDER_BINARY", configured
        return None, None, configured
    path_binary = shutil.which("blender")
    if path_binary:
        return path_binary, "PATH", None
    return None, None, None


def _load_asset_metadata(metadata_path: Path) -> dict[str, list[str]]:
    if not metadata_path.exists():
        return {"free": [], "full_plan": []}
    try:
        raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"free": [], "full_plan": []}
    return {
        "free": sorted(str(item) for item in raw.get("free", []) if item),
        "full_plan": sorted(str(item) for item in raw.get("full_plan", []) if item),
    }


def _scan_robowits_task_asset_refs(repo: Path) -> dict[str, list[str]]:
    task_root = repo / "gs_gym" / "envs" / "robowits"
    refs: dict[str, list[str]] = {}
    if not task_root.exists():
        return refs
    for path in sorted(task_root.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(errors="ignore")
        matches = sorted(set(_ASSET_REF_RE.findall(text)))
        if matches:
            refs[str(path.relative_to(task_root))] = matches
    return refs


def _blenderkit_independent_cases(repo: Path, task_refs: dict[str, list[str]], assets_dir: Path) -> dict[str, object]:
    task_root = repo / "gs_gym" / "envs" / "robowits"
    task_files = _robowits_task_files(task_root)
    cases: list[dict[str, object]] = []
    for path in task_files:
        rel = str(path.relative_to(task_root))
        refs = task_refs.get(rel, [])
        if any(ref.startswith("blender_kit/") for ref in refs):
            continue
        hf_refs = sorted(ref for ref in refs if ref.startswith("hf_assets/"))
        missing_hf_refs = [ref for ref in hf_refs if not (assets_dir / ref).exists()]
        stem = path.stem
        is_mutation = rel.startswith("mutation/")
        base_task_id = stem.split("_", 1)[0]
        cases.append(
            {
                "task_file": rel,
                "task_id": stem,
                "base_task_id": base_task_id,
                "family": "mutation" if is_mutation else "seed",
                "asset_refs": refs,
                "hf_asset_refs": hf_refs,
                "missing_hf_asset_refs": missing_hf_refs,
                "asset_mode": "hf_only" if hf_refs else "no_task_asset_refs",
                "runnable_with_current_hf_assets": not missing_hf_refs,
            }
        )
    ready_cases = [case for case in cases if case["runnable_with_current_hf_assets"]]
    return {
        "description": "Task files with no BlenderKit get_asset_path references; they can be used to smoke Genesis/LeRobot plumbing while BlenderKit assets or keys are unavailable.",
        "case_count": len(cases),
        "ready_case_count": len(ready_cases),
        "seed_task_ids": sorted(str(case["task_id"]) for case in cases if case["family"] == "seed"),
        "mutation_task_ids": sorted(str(case["task_id"]) for case in cases if case["family"] == "mutation"),
        "hf_only_task_ids": sorted(str(case["task_id"]) for case in cases if case["asset_mode"] == "hf_only"),
        "no_task_asset_ref_task_ids": sorted(str(case["task_id"]) for case in cases if case["asset_mode"] == "no_task_asset_refs"),
        "ready_task_ids": sorted(str(case["task_id"]) for case in ready_cases),
        "cases": cases,
    }


def _robowits_task_files(task_root: Path) -> list[Path]:
    if not task_root.exists():
        return []
    ignored = {"__init__.py", "robowits.py", "utils.py", "placement.py"}
    return sorted(
        path
        for path in task_root.rglob("*.py")
        if path.name not in ignored and not path.name.startswith("_")
    )


def _task_family_summary(repo: Path) -> dict[str, object]:
    task_root = repo / "gs_gym" / "envs" / "robowits"
    top_level = [path for path in task_root.glob("*.py") if path.name != "__init__.py"] if task_root.exists() else []
    mutation = [path for path in (task_root / "mutation").glob("*.py") if path.name != "__init__.py"] if task_root.exists() else []
    refs = _scan_robowits_task_asset_refs(repo)
    return {
        "core_task_file_count": len(top_level),
        "mutation_task_file_count": len(mutation),
        "task_files_with_asset_refs": len(refs),
        "sample_task_asset_refs": {key: refs[key] for key in sorted(refs)[:5]},
    }


def _count_files(path: Path, pattern: str) -> int:
    return len(list(path.glob(pattern))) if path.exists() else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run RoboWits / Genesis / gs_gym preflight checks.")
    parser.add_argument("--repo-path", default=str(DEFAULT_ROBOWITS_REPO))
    parser.add_argument("--dataset-split", default="eval_dataset_50")
    parser.add_argument("--task-id", default="01")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--create-env", action="store_true", help="Try creating and resetting a real RoboWits env.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    payload = run_robowits_preflight_probe(
        repo_path=args.repo_path,
        dataset_split=args.dataset_split,
        task_id=args.task_id,
        episode_index=args.episode_index,
        create_env=args.create_env,
    )
    payload["asset_families"] = inspect_robowits_asset_preflight(args.repo_path)
    text = json.dumps(payload, ensure_ascii=False, indent=args.indent, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)

    dataset_ok = bool((payload.get("dataset") or {}).get("loaded"))
    imports = payload.get("imports") or {}
    gs_gym_ok = bool((imports.get("gs_gym") or {}).get("importable"))
    env_ok = not args.create_env or bool((payload.get("env_probe") or {}).get("ok"))
    return 0 if dataset_ok and gs_gym_ok and env_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
