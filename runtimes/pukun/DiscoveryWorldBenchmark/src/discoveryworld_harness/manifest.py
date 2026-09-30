from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import HarnessConfig

MANIFEST_ID = "robench_discoveryworld_v1"


@dataclass
class TaskRecord:
    benchmark: str
    track: str
    task_id: str
    scenario_name: str
    difficulty: str
    seed: int
    task_type: str
    goal_text: str
    source: str = "official_discoveryworld"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEBUG_3_SPECS = [
    ("Small Skills: Pick and Place Test", "Normal", 0),
    ("Small Skills: Dialog Test", "Normal", 0),
    ("Small Skills: Doors Test", "Normal", 0),
]

DEBUG_10_SPECS = [
    ("Small Skills: Pick and Place Test", "Normal", 0),
    ("Small Skills: Pick and Give Test", "Normal", 0),
    ("Small Skills: Dialog Test", "Normal", 0),
    ("Small Skills: Doors Test", "Normal", 0),
    ("Small Skills: Doors with Keys Test", "Normal", 0),
    ("Small Skills: Navigation in a House Test", "Normal", 0),
    ("Small Skills: Instrument Measurement Test", "Normal", 0),
    ("Small Skills: Search Test", "Normal", 0),
    ("Small Skills: Moving Agents Test", "Normal", 0),
    ("Small Skills: Pick and Place Test", "Normal", 1),
]


def build_manifest(config: HarnessConfig, force: bool = False) -> dict[str, list[TaskRecord]]:
    import os
    if os.environ.get('EMBODIED_ARENA_W3_TASK_POOL'):
        return _load_manifest(Path(os.environ['EMBODIED_ARENA_W3_TASK_POOL']))
    config.ensure_dirs()
    manifest_path = config.manifest_dir / f"{MANIFEST_ID}.json"
    if manifest_path.exists() and not force:
        return _load_manifest(manifest_path)

    suites = {
        "debug_3": [_record(*spec) for spec in DEBUG_3_SPECS],
        "debug_10": [_record(*spec) for spec in DEBUG_10_SPECS],
        "full_v1": _full_v1_records(),
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({name: [task.to_dict() for task in tasks] for name, tasks in suites.items()}, f, indent=2)
    return suites


def select_suite(
    manifest: dict[str, list[TaskRecord]],
    suite: str,
    num_tasks: int | None = None,
    start_index: int | None = None,
    seed: int = 0,
) -> list[TaskRecord]:
    if suite not in manifest:
        raise ValueError(f"Unknown suite '{suite}'. Expected one of: {', '.join(sorted(manifest))}")
    tasks = list(manifest[suite])
    if start_index is not None and start_index < 0:
        raise ValueError("start_index must be >= 0")
    if start_index is not None:
        end_index = None if num_tasks is None else start_index + num_tasks
        return tasks[start_index:end_index]
    if num_tasks is not None:
        rng = random.Random(seed)
        if num_tasks < len(tasks):
            tasks = tasks[:]
            rng.shuffle(tasks)
            tasks = tasks[:num_tasks]
        else:
            tasks = tasks[:num_tasks]
    return tasks


def _load_manifest(path: Path) -> dict[str, list[TaskRecord]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {suite: [TaskRecord(**row) for row in rows] for suite, rows in raw.items()}


def _record(scenario_name: str, difficulty: str, seed: int) -> TaskRecord:
    task_type = _task_type(scenario_name)
    task_id = f"{_safe(scenario_name)}-{_safe(difficulty)}-s{seed}"
    goal = (
        f"DiscoveryWorld scenario '{scenario_name}' ({difficulty}), seed {seed}. "
        "The official taskProgress description is read after reset."
    )
    return TaskRecord(
        benchmark="DiscoveryWorld",
        track="text_json",
        task_id=task_id,
        scenario_name=scenario_name,
        difficulty=difficulty,
        seed=seed,
        task_type=task_type,
        goal_text=goal,
    )


def _full_v1_records() -> list[TaskRecord]:
    try:
        from discoveryworld.ScenarioMaker import SCENARIO_INFOS
    except Exception:
        return [_record(*spec) for spec in DEBUG_10_SPECS]

    records: list[TaskRecord] = []
    for scenario_name, info in sorted(SCENARIO_INFOS.items()):
        for difficulty in info.get("difficulty", []):
            for seed in range(5):
                records.append(_record(scenario_name, difficulty, seed))
    return records


def _task_type(scenario_name: str) -> str:
    name = scenario_name
    name = name.replace("Small Skills:", "smallskills")
    return _safe(name).lower()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
