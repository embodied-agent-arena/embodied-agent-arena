from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

from .behavior1k_agent_runtime import ENV_PROBE_MARKER, official_version_probe, run_env_probe_inline, run_preflight_probe
from .paths import get_project_paths

JsonDict = dict[str, Any]

VISUAL_MODALITIES = {
    "rgb",
    "depth",
    "depth_linear",
    "seg_instance",
    "seg_semantic",
    "segmentation",
    "semantic_segmentation",
    "instance_segmentation",
    "point_cloud",
    "normal",
}

MINIMAL_SENSOR_SCENE_ENV_CONFIG: JsonDict = {
    "scene": {"type": "InteractiveTraversableScene", "scene_model": "Rs_int"},
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb", "depth", "seg_instance", "proprio"],
            "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 64, "image_height": 64}}},
        }
    ],
}

SMALL_OBJECT_SENSOR_SCENE_ENV_CONFIG: JsonDict = {
    "scene": {"type": "InteractiveTraversableScene", "scene_model": "gates_bedroom"},
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb", "depth", "seg_instance", "proprio"],
            "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 64, "image_height": 64}}},
        }
    ],
}

OFFICIAL_QUICK_OBJECT_SENSOR_SCENE_ENV_CONFIG: JsonDict = {
    "scene": {
        "type": "InteractiveTraversableScene",
        "scene_model": "gates_bedroom",
        "load_object_categories": ["floors", "walls", "ceilings", "coffee_table"],
    },
    "robots": [
        {
            "type": "Fetch",
            "obs_modalities": ["rgb", "depth", "seg_instance", "proprio"],
            "sensor_config": {"VisionSensor": {"sensor_kwargs": {"image_width": 64, "image_height": 64}}},
        }
    ],
}


def behavior1k_env_config_gate_from_env(raw: str | None = None) -> JsonDict:
    source = "BEHAVIOR1K_ENV_CONFIG_JSON" if raw is None else "provided"
    raw_config = os.getenv("BEHAVIOR1K_ENV_CONFIG_JSON") if raw is None else raw
    if not raw_config:
        return behavior1k_env_config_gate({"scene": {"type": "Scene"}}, source="default_empty_scene")
    try:
        loaded = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "source": source,
            "error": f"JSONDecodeError: {exc.msg}",
            "blockers": ["BEHAVIOR1K_ENV_CONFIG_JSON must be valid JSON"],
            "minimal_sensor_scene_env_config": MINIMAL_SENSOR_SCENE_ENV_CONFIG,
        }
    if not isinstance(loaded, dict):
        return {
            "ok": False,
            "source": source,
            "error": "BEHAVIOR1K_ENV_CONFIG_JSON must decode to a JSON object",
            "blockers": ["BEHAVIOR1K_ENV_CONFIG_JSON must decode to a JSON object"],
            "minimal_sensor_scene_env_config": MINIMAL_SENSOR_SCENE_ENV_CONFIG,
        }
    return behavior1k_env_config_gate(loaded, source=source)


def behavior1k_env_config_gate(config: JsonDict, *, source: str = "provided") -> JsonDict:
    scene = config.get("scene") if isinstance(config.get("scene"), dict) else {}
    scene_type = str(scene.get("type", "Scene"))
    scene_model = _first_string(scene, ("scene_model", "model", "name", "scene_file", "load_scene_model"))
    object_specs = _collect_object_specs(config)
    sensor_specs = _collect_sensor_specs(config)
    modalities = sorted(_collect_modalities(config))

    non_empty_scene_requested = bool(scene_model or object_specs or scene_type != "Scene")
    visual_sensor_requested = bool(sensor_specs or VISUAL_MODALITIES.intersection(modalities))
    ok = non_empty_scene_requested and visual_sensor_requested

    blockers: list[str] = []
    if not non_empty_scene_requested:
        blockers.append("empty_scene_config: set scene.type to an asset-backed scene, provide scene.scene_model, or add object specs")
    if not visual_sensor_requested:
        blockers.append("missing_visual_sensor_config: add robot obs_modalities including rgb/depth/seg_instance or a camera/sensor config")

    required_conditions = [
        "OmniGibson and BDDL importable in the selected BEHAVIOR1K_PYTHON interpreter",
        "Isaac/Kit can start on the host or Slurm GPU node without native crash",
        "BEHAVIOR / OmniGibson assets are installed and include the requested scene model and robot assets",
        "A render-capable sensor stack can produce at least one RGB/depth/segmentation observation",
    ]
    if scene_model:
        required_conditions.append(f"Requested scene model asset is present: {scene_model}")

    return {
        "ok": ok,
        "source": source,
        "goal": "non_empty_scene_with_agent_visible_sensor",
        "env_var": "BEHAVIOR1K_ENV_CONFIG_JSON",
        "scene": {
            "type": scene_type,
            "scene_model": scene_model,
            "non_empty_requested": non_empty_scene_requested,
            "object_spec_count": len(object_specs),
        },
        "sensors": {
            "visual_sensor_requested": visual_sensor_requested,
            "modalities": modalities,
            "sensor_spec_count": len(sensor_specs),
            "sensor_specs": sensor_specs[:8],
        },
        "blockers": blockers,
        "required_conditions": required_conditions,
        "minimal_sensor_scene_env_config": MINIMAL_SENSOR_SCENE_ENV_CONFIG,
        "small_object_sensor_scene_env_config": SMALL_OBJECT_SENSOR_SCENE_ENV_CONFIG,
        "official_quick_object_sensor_scene_env_config": OFFICIAL_QUICK_OBJECT_SENSOR_SCENE_ENV_CONFIG,
    }


def behavior1k_asset_gate_from_env(config: JsonDict | None = None, data_root: str | Path | None = None) -> JsonDict:
    config = config or _config_from_env_or_default()
    scene = config.get("scene") if isinstance(config.get("scene"), dict) else {}
    scene_model = _first_string(scene, ("scene_model", "model", "name", "scene_file", "load_scene_model"))
    resolved_root = Path(data_root) if data_root is not None else _default_omnigibson_data_root()
    behavior_assets, asset_layout = _resolve_behavior_asset_root(resolved_root)
    legacy_behavior_assets = resolved_root / "behavior-1k-assets"
    robot_assets = _default_omnigibson_asset_root(resolved_root)
    scene_dir = behavior_assets / "scenes"
    scene_json_dir = scene_dir / scene_model / "json" if scene_model else None
    scene_info = _scene_asset_summary(scene_json_dir)
    metadata_dir = behavior_assets / "metadata"
    robot_types = _collect_robot_types(config)
    robot_asset_checks = _robot_asset_checks(robot_assets, robot_types)

    blockers: list[str] = []
    if not resolved_root.exists():
        blockers.append("omnigibson_data_root_missing")
    if not robot_assets.exists():
        blockers.append("omnigibson_robot_assets_missing")
    if not behavior_assets.exists():
        blockers.append("behavior1k_assets_missing")
    if not scene_dir.exists():
        blockers.append("behavior1k_scene_assets_missing")
    if scene_json_dir is not None and not scene_json_dir.exists():
        blockers.append(f"behavior1k_scene_model_json_missing:{scene_model}")
    for robot_name, checks in robot_asset_checks.items():
        if not checks["ok"]:
            blockers.append(f"omnigibson_robot_model_missing:{robot_name}")

    return {
        "ok": not blockers,
        "goal": "behavior1k_runtime_assets_available_before_isaac_start",
        "env_var": "OMNIGIBSON_DATA_PATH",
        "source": "OMNIGIBSON_DATA_PATH" if os.getenv("OMNIGIBSON_DATA_PATH") else "repo_default",
        "data_root": str(resolved_root),
        "asset_layout": asset_layout,
        "requested_scene_model": scene_model,
        "requested_robot_types": robot_types,
        "paths": {
            "behavior_assets": str(behavior_assets),
            "legacy_behavior_assets": str(legacy_behavior_assets),
            "robot_assets": str(robot_assets),
            "scene_dir": str(scene_dir),
            "scene_json_dir": str(scene_json_dir) if scene_json_dir is not None else None,
            "metadata_dir": str(metadata_dir),
        },
        "robot_asset_checks": robot_asset_checks,
        "scene_asset_summary": scene_info,
        "exists": {
            "data_root": resolved_root.exists(),
            "behavior_assets": behavior_assets.exists(),
            "robot_assets": robot_assets.exists(),
            "scene_dir": scene_dir.exists(),
            "scene_json_dir": scene_json_dir.exists() if scene_json_dir is not None else None,
            "metadata_dir": metadata_dir.exists(),
        },
        "blockers": blockers,
        "repair_hint": (
            "Install official BEHAVIOR runtime assets so OMNIGIBSON_DATA_PATH points either to a flat v3.9-style "
            "directory containing scenes/, objects/, metadata/, and omnigibson-robot-assets/, or to a parent "
            "containing behavior-1k-assets/ plus omnigibson-robot-assets/."
        ),
    }


def _resolve_behavior_asset_root(data_root: Path) -> tuple[Path, str]:
    legacy = data_root / "behavior-1k-assets"
    if legacy.exists() or (legacy / "scenes").exists():
        return legacy, "nested_behavior_1k_assets"

    flat_markers = ("scenes", "metadata", "objects", "systems")
    if any((data_root / marker).exists() for marker in flat_markers):
        return data_root, "flat_behavior_assets"

    return legacy, "missing"


def _default_omnigibson_asset_root(data_root: Path) -> Path:
    configured = os.getenv("OMNIGIBSON_ASSET_PATH")
    if configured:
        return Path(configured).expanduser()
    repo_assets = get_project_paths().external_assets("behavior1k") / "og_assets_full"
    if (repo_assets / "models" / "fetch" / "fetch" / "fetch.usd").exists():
        return repo_assets
    return data_root / "omnigibson-robot-assets"


def _collect_object_specs(config: JsonDict) -> list[Any]:
    scene = config.get("scene") if isinstance(config.get("scene"), dict) else {}
    specs: list[Any] = []
    for key in ("objects", "load_object_categories", "not_load_object_categories", "include_object_categories"):
        value = scene.get(key)
        if value:
            specs.append({key: _shape_summary(value)})
    for key in ("objects", "object_instances"):
        value = config.get(key)
        if value:
            specs.append({key: _shape_summary(value)})
    return specs


def _collect_sensor_specs(config: JsonDict) -> list[JsonDict]:
    specs: list[JsonDict] = []
    for key in ("sensors", "sensor_config", "camera", "cameras", "render"):
        value = config.get(key)
        if value:
            specs.append({"path": key, "summary": _shape_summary(value)})
    robots = config.get("robots", [])
    if isinstance(robots, dict):
        robots = [robots]
    if isinstance(robots, list):
        for index, robot in enumerate(robots):
            if not isinstance(robot, dict):
                continue
            for key in ("obs_modalities", "sensor_config", "sensors", "camera", "cameras"):
                value = robot.get(key)
                if value:
                    specs.append({"path": f"robots[{index}].{key}", "summary": _shape_summary(value)})
    return specs


def _collect_robot_types(config: JsonDict) -> list[str]:
    robots = config.get("robots", [])
    if isinstance(robots, dict):
        robots = [robots]
    robot_types: list[str] = []
    if isinstance(robots, list):
        for robot in robots:
            if isinstance(robot, dict) and isinstance(robot.get("type"), str):
                robot_types.append(robot["type"])
    return sorted(set(robot_types))


def _robot_asset_checks(robot_assets: Path, robot_types: list[str]) -> JsonDict:
    checks: JsonDict = {}
    required_groups_by_robot = {
        "Fetch": [
            [
                robot_assets / "models" / "fetch" / "fetch" / "fetch.usd",
                robot_assets / "models" / "fetch" / "fetch.urdf",
                robot_assets / "models" / "fetch" / "fetch_descriptor.yaml",
            ],
            [
                robot_assets / "models" / "fetch" / "usd" / "fetch.usda",
                robot_assets / "models" / "fetch" / "urdf" / "fetch.urdf",
            ],
        ],
    }
    for robot_type in robot_types:
        required_groups = required_groups_by_robot.get(robot_type, [])
        if not required_groups:
            checks[robot_type] = {"ok": robot_assets.exists(), "required_paths": [], "missing_paths": []}
            continue
        missing_groups = [[str(path) for path in group if not path.exists()] for group in required_groups]
        ok = any(not missing for missing in missing_groups)
        checks[robot_type] = {
            "ok": ok,
            "required_paths": [[str(path) for path in group] for group in required_groups],
            "missing_paths": [] if ok else missing_groups[0],
            "accepted_layouts": ["download_package_fetch_usd", "omnigibson_robot_base_fetch_usda"],
        }
    return checks


def _scene_asset_summary(scene_json_dir: Path | None) -> JsonDict:
    if scene_json_dir is None:
        return {"available": False, "reason": "no_scene_model_requested"}
    if not scene_json_dir.exists():
        return {"available": False, "scene_json_dir": str(scene_json_dir), "reason": "scene_json_dir_missing"}
    candidates = sorted(scene_json_dir.glob("*.json"))
    if not candidates:
        return {"available": False, "scene_json_dir": str(scene_json_dir), "reason": "scene_json_missing"}
    scene_file = candidates[0]
    try:
        loaded = json.loads(scene_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"available": False, "scene_json_dir": str(scene_json_dir), "scene_file": str(scene_file), "reason": f"{type(exc).__name__}: {exc}"}
    objects_info = loaded.get("objects_info") if isinstance(loaded, dict) else {}
    init_info = objects_info.get("init_info") if isinstance(objects_info, dict) else {}
    categories: dict[str, int] = {}
    if isinstance(init_info, dict):
        for obj_info in init_info.values():
            if not isinstance(obj_info, dict):
                continue
            args = obj_info.get("args") if isinstance(obj_info.get("args"), dict) else {}
            category = str(args.get("category") or "object")
            categories[category] = categories.get(category, 0) + 1
    return {
        "available": True,
        "scene_json_dir": str(scene_json_dir),
        "scene_file": str(scene_file),
        "object_instance_count": len(init_info) if isinstance(init_info, dict) else None,
        "category_counts": dict(sorted(categories.items())),
        "smallest_official_full_scene_observed": "gates_bedroom" if scene_json_dir.parent.name == "gates_bedroom" else None,
        "official_partial_load_config_key": "scene.load_object_categories",
    }


def _collect_modalities(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, str):
        normalized = value.lower()
        if normalized in VISUAL_MODALITIES or normalized == "proprio":
            found.add(normalized)
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            key_str = str(key).lower()
            if key_str in VISUAL_MODALITIES:
                found.add(key_str)
            found.update(_collect_modalities(item))
    elif isinstance(value, (list, tuple, set)):
        for item in value:
            found.update(_collect_modalities(item))
    return found


def _first_string(payload: JsonDict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _config_from_env_or_default() -> JsonDict:
    raw_config = os.getenv("BEHAVIOR1K_ENV_CONFIG_JSON")
    if not raw_config:
        return {"scene": {"type": "Scene"}}
    try:
        loaded = json.loads(raw_config)
    except json.JSONDecodeError:
        return {"scene": {"type": "Scene"}}
    return loaded if isinstance(loaded, dict) else {"scene": {"type": "Scene"}}


def _default_omnigibson_data_root() -> Path:
    configured = os.getenv("OMNIGIBSON_DATA_PATH")
    if configured:
        return Path(configured).expanduser()
    return get_project_paths().external_assets("behavior1k") / "datasets"


def _shape_summary(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(key) for key in value)[:12], "length": len(value)}
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__, "value": str(value)[:120]}


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "False").lower() in ("1", "true", "t", "yes", "y")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run BEHAVIOR-1K / OmniGibson / BDDL preflight checks.")
    parser.add_argument("--create-env", action="store_true", help="Try creating the lightest configured OmniGibson environment.")
    parser.add_argument("--env-timeout-seconds", type=int, default=900, help="Timeout for the child env reset/action probe.")
    parser.add_argument("--probe-tags", action="store_true", help="Query upstream BEHAVIOR-1K tags with git ls-remote.")
    parser.add_argument(
        "--require-nonempty-sensor-config",
        action="store_true",
        help="Fail unless BEHAVIOR1K_ENV_CONFIG_JSON requests a non-empty scene and visual sensor config.",
    )
    parser.add_argument(
        "--require-process-ok",
        action="store_true",
        help="Fail strict readiness when the env-probe child writes a success marker but the native process exits nonzero.",
    )
    parser.add_argument("--env-probe-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default=None)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    if args.env_probe_child:
        payload = run_env_probe_inline()
        fast_exit_after_marker = _env_truthy("BEHAVIOR1K_ENV_PROBE_FAST_EXIT")
        payload["fast_exit_after_marker"] = bool(fast_exit_after_marker)
        sidecar_output = os.getenv("BEHAVIOR1K_ENV_PROBE_OUTPUT")
        if sidecar_output:
            try:
                output_path = Path(sidecar_output).expanduser()
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
            except OSError as exc:
                payload["sidecar_write_error"] = f"{type(exc).__name__}: {exc}"
        print(ENV_PROBE_MARKER + json.dumps(payload, ensure_ascii=False, sort_keys=True))
        status = 0 if payload.get("ok") else 1
        if fast_exit_after_marker:
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(status)
        return status

    env_config_gate = behavior1k_env_config_gate_from_env()
    asset_gate = behavior1k_asset_gate_from_env(_config_from_env_or_default())
    require_gate = args.require_nonempty_sensor_config or _env_truthy("BEHAVIOR1K_REQUIRE_NONEMPTY_SENSOR_CONFIG")
    require_process_ok = args.require_process_ok or _env_truthy("BEHAVIOR1K_REQUIRE_PROCESS_OK")
    should_create_env = args.create_env
    skipped_env_probe: JsonDict | None = None
    if args.create_env and require_gate and (not env_config_gate.get("ok") or not asset_gate.get("ok")):
        should_create_env = False
        skipped_env_probe = {
            "ok": False,
            "skipped": True,
            "reason": "pre_isaac_gate_failed",
            "env_config_gate_ok": bool(env_config_gate.get("ok")),
            "asset_gate_ok": bool(asset_gate.get("ok")),
            "blockers": list(env_config_gate.get("blockers", [])) + list(asset_gate.get("blockers", [])),
        }
    payload = run_preflight_probe(create_env=should_create_env, env_timeout_seconds=args.env_timeout_seconds)
    payload["env_config_gate"] = env_config_gate
    payload["asset_gate"] = asset_gate
    if skipped_env_probe is not None:
        payload["env_probe"] = skipped_env_probe
    if args.probe_tags:
        payload["upstream_tags"] = official_version_probe()
    payload["strict_process_ok_required"] = bool(require_process_ok)

    text = json.dumps(payload, ensure_ascii=False, indent=args.indent, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)

    imports = payload.get("imports", {})
    imports_ok = all(imports.get(name, {}).get("ok") for name in ("bddl", "omnigibson"))
    env_ok = not args.create_env or bool((payload.get("env_probe") or {}).get("ok"))
    process_ok = not args.create_env or not require_process_ok or bool((payload.get("env_probe") or {}).get("process_ok"))
    gate_ok = not require_gate or (bool(env_config_gate.get("ok")) and bool(asset_gate.get("ok")))
    return 0 if imports_ok and env_ok and process_ok and gate_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
