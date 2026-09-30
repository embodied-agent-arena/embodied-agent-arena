from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .config import HarnessConfig


DEBUG_TASK_ORDER = [
    "boil",
    "melt",
    "freeze",
    "use-thermometer",
    "find-living-thing",
    "find-non-living-thing",
    "find-plant",
    "find-animal",
    "chemistry-mix",
    "lifespan-longest-lived",
]


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_name: str
    task_type: str
    variation_idx: int
    simplification: str
    goal_text: str
    source_split: str = "official_variation_0"


def build_manifest(config: HarnessConfig, force: bool = False) -> list[TaskRecord]:
    from task_pool import load_selected_pool
    selected = load_selected_pool()
    if selected is not None:
        return [TaskRecord(**row) for row in selected]
    if config.manifest_path.exists() and not force:
        return load_manifest(config.manifest_path)

    tasks_path = config.scienceworld_root / "scienceworld" / "tasks.json"
    with tasks_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)

    metadata_by_name = {row["task_name"]: row for row in rows}
    records = _official_variation_records(metadata_by_name, config)
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with config.manifest_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(record) for record in records], f, indent=2, ensure_ascii=False)
    return records


def _official_variation_records(metadata_by_name: dict[str, dict[str, str]], config: HarnessConfig) -> list[TaskRecord]:
    from scienceworld import ScienceWorldEnv

    records: list[TaskRecord] = []
    task_names = sorted(metadata_by_name)
    for task_name in task_names:
        env = ScienceWorldEnv(task_name, None, envStepLimit=1)
        try:
            split_variations = {
                "official_train": env.get_variations_train(),
                "official_dev": env.get_variations_dev(),
                "official_test": env.get_variations_test(),
            }
        finally:
            env.close()

        row = metadata_by_name[task_name]
        goal_text = f"{row.get('topic', '').strip()}: {row.get('task', '').strip()}".strip(": ")
        for source_split, variations in split_variations.items():
            for variation_idx in variations:
                records.append(
                    TaskRecord(
                        task_id=f"{task_name}-v{variation_idx}-{_simplification_slug(config.simplification)}",
                        task_name=task_name,
                        task_type=task_name,
                        variation_idx=int(variation_idx),
                        simplification=config.simplification,
                        goal_text=goal_text,
                        source_split=source_split,
                    )
                )
    return sorted(records, key=lambda record: (record.task_name, record.variation_idx, record.source_split))


def load_manifest(path: Path) -> list[TaskRecord]:
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return [TaskRecord(**row) for row in rows]


def select_suite(
    records: Iterable[TaskRecord],
    suite: str,
    start_index: int = 0,
    num_tasks: int | None = None,
    seed: int = 42,
) -> list[TaskRecord]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    records = list(records)
    valid_suites = {"debug_3", "debug_10", "variation0_v1", "train_v1", "dev_v1", "test_v1", "full_v1"}
    if suite not in valid_suites:
        raise ValueError(f"Unknown suite '{suite}'. Expected: {', '.join(sorted(valid_suites))}")

    if suite == "full_v1":
        selected = records
    elif suite == "variation0_v1":
        selected = [record for record in records if record.variation_idx == 0]
    elif suite in {"train_v1", "dev_v1", "test_v1"}:
        split = "official_" + suite.replace("_v1", "")
        selected = [record for record in records if record.source_split == split]
    else:
        target = 3 if suite == "debug_3" else 10
        by_name = {}
        for record in records:
            if record.variation_idx == 0:
                by_name.setdefault(record.task_name, record)
        selected = [by_name[name] for name in DEBUG_TASK_ORDER if name in by_name][:target]
        if len(selected) < target:
            remaining = [record for record in records if record.variation_idx == 0 and record not in selected]
            rng = random.Random(seed)
            rng.shuffle(remaining)
            selected.extend(remaining[: target - len(selected)])

    if start_index:
        selected = selected[start_index:]
        if not selected:
            raise ValueError(f"start_index {start_index} is outside suite '{suite}' with no tasks remaining")
    if num_tasks is not None:
        selected = selected[:num_tasks]
    return selected


def _simplification_slug(simplification: str) -> str:
    return simplification.replace(",", "+") if simplification else "standard"
