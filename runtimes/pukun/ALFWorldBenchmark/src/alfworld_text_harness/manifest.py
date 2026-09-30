from __future__ import annotations

import json
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable

from .config import HarnessConfig, TASK_TYPE_ORDER

SOURCE_SPLITS = ("valid_seen", "valid_unseen")
BENCHMARK_NAME = "ALFWorld"
TRACK_NAME = "text"


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_type: str
    source_split: str
    gamefile: str
    traj_data: str
    goal_text: str | None = None

    @property
    def task_root(self) -> str:
        return str(Path(self.traj_data).parent)


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _goal_text(traj_data: dict, gamefile: Path) -> str | None:
    official_goal = _official_goal_text(gamefile)
    if official_goal:
        return official_goal

    anns = traj_data.get("turk_annotations", {}).get("anns", [])
    if anns and isinstance(anns, list):
        first = anns[0]
        if isinstance(first, dict):
            return first.get("task_desc")
    return None


def _official_goal_text(gamefile: Path) -> str | None:
    game_data = _read_json(gamefile)
    grammar = game_data.get("grammar")
    if not isinstance(grammar, str):
        return None

    match = re.search(r'"task"\s*:\s*\[\s*\{\s*"rhs"\s*:\s*"([^"]+)"', grammar, re.DOTALL)
    if not match:
        return None
    return match.group(1).strip()


def _is_supported_task(root: Path, files: set[str], task_type: str, gamefile: Path) -> bool:
    if "traj_data.json" not in files:
        return False
    if "game.tw-pddl" not in files:
        return False
    if "movable" in str(root) or "Sliced" in str(root):
        return False
    if task_type not in TASK_TYPE_ORDER:
        return False
    if not gamefile.exists():
        return False
    game_data = _read_json(gamefile)
    return bool(game_data.get("solvable", False))


def build_manifest(config: HarnessConfig, force: bool = False) -> list[TaskRecord]:
    if config.manifest_path.exists() and not force:
        return rebase_manifest_paths(load_manifest(config.manifest_path), config)

    records: list[TaskRecord] = []
    data_root = config.data_dir / "json_2.1.1"
    for source_split in SOURCE_SPLITS:
        source_dir = data_root / source_split
        for root_str, _dirs, file_names in sorted(source_dir.walk() if hasattr(source_dir, "walk") else []):
            # Path.walk exists on Python 3.12; this branch is kept for completeness.
            root = Path(root_str)
            files = set(file_names)
            traj_path = root / "traj_data.json"
            gamefile = root / "game.tw-pddl"
            if "traj_data.json" not in files:
                continue
            traj_data = _read_json(traj_path)
            task_type = traj_data.get("task_type", "")
            if not _is_supported_task(root, files, task_type, gamefile):
                continue
            rel_task = root.relative_to(source_dir).as_posix()
            records.append(
                TaskRecord(
                    task_id=f"{source_split}/{rel_task}",
                    task_type=task_type,
                    source_split=source_split,
                    gamefile=str(gamefile),
                    traj_data=str(traj_path),
                    goal_text=_goal_text(traj_data, gamefile),
                )
            )

    if not records:
        # Python 3.11 fallback because Path.walk is not available.
        records = _build_manifest_with_os_walk(config)

    records = sorted(records, key=lambda r: (r.task_type, r.source_split, r.task_id))
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in records], f, indent=2, ensure_ascii=False)
    return records


def _build_manifest_with_os_walk(config: HarnessConfig) -> list[TaskRecord]:
    import os

    records: list[TaskRecord] = []
    data_root = config.data_dir / "json_2.1.1"
    for source_split in SOURCE_SPLITS:
        source_dir = data_root / source_split
        for root_str, _dirs, file_names in os.walk(source_dir):
            root = Path(root_str)
            files = set(file_names)
            traj_path = root / "traj_data.json"
            gamefile = root / "game.tw-pddl"
            if "traj_data.json" not in files:
                continue
            traj_data = _read_json(traj_path)
            task_type = traj_data.get("task_type", "")
            if not _is_supported_task(root, files, task_type, gamefile):
                continue
            rel_task = root.relative_to(source_dir).as_posix()
            records.append(
                TaskRecord(
                    task_id=f"{source_split}/{rel_task}",
                    task_type=task_type,
                    source_split=source_split,
                    gamefile=str(gamefile),
                    traj_data=str(traj_path),
                    goal_text=_goal_text(traj_data, gamefile),
                )
            )
    return records


def load_manifest(path: Path) -> list[TaskRecord]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [TaskRecord(**row) for row in rows]


def rebase_manifest_paths(records: Iterable[TaskRecord], config: HarnessConfig) -> list[TaskRecord]:
    """Map cached manifest paths onto the active ALFWORLD_DATA root when possible."""
    data_root = config.data_dir / "json_2.1.1"
    rebased: list[TaskRecord] = []
    for record in records:
        task_root = data_root / record.task_id
        gamefile = task_root / "game.tw-pddl"
        traj_data = task_root / "traj_data.json"
        if gamefile.exists() and traj_data.exists():
            rebased.append(replace(record, gamefile=str(gamefile), traj_data=str(traj_data)))
        else:
            rebased.append(record)
    return rebased


def select_suite(
    records: Iterable[TaskRecord],
    suite: str,
    num_tasks: int | None = None,
    seed: int = 42,
    start_index: int = 0,
) -> list[TaskRecord]:
    if start_index < 0:
        raise ValueError("start_index must be >= 0")
    records = list(records)
    if suite not in {"debug_10", "mini_30", "full", "full_v1"}:
        raise ValueError(f"Unknown suite '{suite}'. Expected one of: debug_10, mini_30, full_v1, full")

    if suite in {"full", "full_v1"}:
        selected = records
    elif suite == "mini_30":
        selected = _balanced_sample(records, target_count=30, seed=seed)
    else:
        selected = _balanced_sample(records, target_count=10, seed=seed)

    if start_index:
        selected = selected[start_index:]
    if num_tasks is not None:
        selected = selected[:num_tasks]
    return selected


def _balanced_sample(records: list[TaskRecord], target_count: int, seed: int) -> list[TaskRecord]:
    rng = random.Random(seed)
    by_type: dict[str, list[TaskRecord]] = {task_type: [] for task_type in TASK_TYPE_ORDER}
    for record in records:
        by_type.setdefault(record.task_type, []).append(record)
    for items in by_type.values():
        items.sort(key=lambda item: item.task_id)
        rng.shuffle(items)

    selected: list[TaskRecord] = []
    while len(selected) < target_count:
        made_progress = False
        for task_type in TASK_TYPE_ORDER:
            items = by_type.get(task_type, [])
            if items and len(selected) < target_count:
                selected.append(items.pop())
                made_progress = True
        if not made_progress:
            break
    return selected
