from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


MANIFEST_ID = "robench_alfred_official_data_v1"

TASK_TYPE_ORDER = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
    "pick_and_place_with_movable_recep",
]


@dataclass(frozen=True)
class AlfredOfficialTaskRecord:
    benchmark: str
    track: str
    task_id: str
    task_type: str
    source_split: str
    traj_path: str
    goal_text: str
    floor_plan: str
    scene_num: int
    pddl_params: dict[str, Any]
    init_action: dict[str, Any]
    high_level_actions: list[dict[str, Any]] = field(default_factory=list)
    step_by_step_instructions: list[str] = field(default_factory=list)
    source: str = "official_alfred_traj_data"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_context(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "track": self.track,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "source_split": self.source_split,
            "goal_text": self.goal_text,
            "step_by_step_instructions": list(self.step_by_step_instructions),
            "floor_plan": self.floor_plan,
        }


PRIVATE_MANIFEST_FIELDS = {
    "traj_path",
    "pddl_params",
    "init_action",
    "high_level_actions",
}


def public_task_context(record: AlfredOfficialTaskRecord | dict[str, Any]) -> dict[str, Any]:
    if isinstance(record, AlfredOfficialTaskRecord):
        return record.to_public_context()
    return {
        "benchmark": record.get("benchmark"),
        "track": record.get("track"),
        "task_id": record.get("task_id"),
        "task_type": record.get("task_type"),
        "source_split": record.get("source_split"),
        "goal_text": record.get("goal_text"),
        "step_by_step_instructions": _public_step_by_step_instructions(record),
        "floor_plan": record.get("floor_plan"),
    }


def build_manifest(
    data_root: Path,
    output_dir: Path,
    *,
    force: bool = False,
    seed: int = 0,
) -> dict[str, list[AlfredOfficialTaskRecord]]:
    from task_pool import load_selected_pool
    selected = load_selected_pool()
    if selected is not None:
        return {suite:[AlfredOfficialTaskRecord(**row) for row in rows] for suite,rows in selected.items()}
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"{MANIFEST_ID}.json"
    if manifest_path.exists() and not force:
        return load_manifest(manifest_path)

    tasks = discover_tasks(data_root)
    eval_tasks = [task for task in tasks if task.source_split in {"valid_seen", "valid_unseen"}]
    suites = {
        "official_debug_5": select_balanced_subset(tasks, 5, seed=seed),
        "official_debug_10": select_balanced_subset(tasks, 10, seed=seed),
        "official_eval_v1": eval_tasks,
        "full_v1": tasks,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump({name: [task.to_dict() for task in rows] for name, rows in suites.items()}, f, indent=2)
    return suites


def load_manifest(path: Path) -> dict[str, list[AlfredOfficialTaskRecord]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {suite: [AlfredOfficialTaskRecord(**row) for row in rows] for suite, rows in raw.items()}


def discover_tasks(data_root: Path) -> list[AlfredOfficialTaskRecord]:
    if not data_root.exists():
        raise FileNotFoundError(f"ALFRED data root not found: {data_root}")
    task_paths = sorted(data_root.glob("*/*/trial_*/traj_data.json"))
    records = [_record_from_traj(path, data_root) for path in task_paths]
    return sorted(records, key=lambda row: (row.source_split, row.task_type, row.task_id, row.traj_path))


def select_balanced_subset(
    tasks: list[AlfredOfficialTaskRecord],
    count: int,
    *,
    seed: int = 0,
) -> list[AlfredOfficialTaskRecord]:
    by_type: dict[str, list[AlfredOfficialTaskRecord]] = {task_type: [] for task_type in TASK_TYPE_ORDER}
    for task in tasks:
        by_type.setdefault(task.task_type, []).append(task)
    for rows in by_type.values():
        rows.sort(key=lambda row: (row.source_split != "valid_seen", row.traj_path))

    selected: list[AlfredOfficialTaskRecord] = []
    while len(selected) < count:
        progressed = False
        for task_type in TASK_TYPE_ORDER:
            rows = by_type.get(task_type, [])
            if rows:
                selected.append(rows.pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    if len(selected) < count:
        remaining = [task for task in tasks if task not in selected]
        rng = random.Random(seed)
        rng.shuffle(remaining)
        selected.extend(remaining[: count - len(selected)])
    return selected


def summarize_suites(suites: dict[str, list[AlfredOfficialTaskRecord]]) -> dict[str, Any]:
    return {
        name: {
            "tasks": len(rows),
            "by_split": _count_by(rows, "source_split"),
            "by_task_type": _count_by(rows, "task_type"),
            "sample_task_ids": [row.task_id for row in rows[:5]],
        }
        for name, rows in suites.items()
    }


def _record_from_traj(path: Path, data_root: Path) -> AlfredOfficialTaskRecord:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    scene = data.get("scene") or {}
    annotations = ((data.get("turk_annotations") or {}).get("anns") or [{}])
    plan = data.get("plan") or {}
    high_pddl = plan.get("high_pddl") or []
    return AlfredOfficialTaskRecord(
        benchmark="ALFRED",
        track="official_visual",
        task_id=str(data.get("task_id") or path.parent.name),
        task_type=str(data.get("task_type") or path.parent.parent.name.split("-")[0]),
        source_split=path.relative_to(data_root).parts[0],
        traj_path=str(path),
        goal_text=str(annotations[0].get("task_desc") or _templated_goal(data.get("pddl_params") or {})),
        floor_plan=str(scene.get("floor_plan") or f"FloorPlan{scene.get('scene_num')}"),
        scene_num=int(scene.get("scene_num") or _scene_num_from_floor_plan(scene.get("floor_plan"))),
        pddl_params=dict(data.get("pddl_params") or {}),
        init_action=dict(scene.get("init_action") or {}),
        high_level_actions=[
            {
                "high_idx": item.get("high_idx"),
                "action": (item.get("discrete_action") or {}).get("action"),
                "args": (item.get("discrete_action") or {}).get("args", []),
            }
            for item in high_pddl
        ],
        step_by_step_instructions=_annotation_high_descs(annotations),
    )


def _public_step_by_step_instructions(record: dict[str, Any]) -> list[str]:
    existing = record.get("step_by_step_instructions")
    if isinstance(existing, list) and existing:
        return [str(item) for item in existing if str(item).strip()]
    traj_path = record.get("traj_path")
    if not traj_path:
        return []
    try:
        with Path(str(traj_path)).open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    annotations = ((data.get("turk_annotations") or {}).get("anns") or [{}])
    return _annotation_high_descs(annotations)


def _annotation_high_descs(annotations: list[dict[str, Any]]) -> list[str]:
    if not annotations:
        return []
    high_descs = annotations[0].get("high_descs") or []
    return [str(item).strip() for item in high_descs if str(item).strip()]


def _templated_goal(params: dict[str, Any]) -> str:
    obj = params.get("object_target") or "object"
    parent = params.get("parent_target") or "receptacle"
    return f"Complete the ALFRED task involving {obj} and {parent}."


def _scene_num_from_floor_plan(value: Any) -> int:
    text = str(value or "")
    digits = "".join(char for char in text if char.isdigit())
    return int(digits or 0)


def _count_by(rows: list[AlfredOfficialTaskRecord], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(getattr(row, field_name))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
