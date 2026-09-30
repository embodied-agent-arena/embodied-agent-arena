from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .official_dataset import load_official_task_specs


MANIFEST_ID = "robench_virtualhome_symbolic_v1"
PRIVATE_MANIFEST_ID = f"{MANIFEST_ID}_private"


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    task_type: str
    goal_text: str
    graph_name: str
    graph: dict[str, Any]
    expected_plan: list[str]
    action_space: list[str]
    success_predicate: dict[str, Any]
    suite_tags: list[str]
    source: str = "local_virtualhome_evolving_graph_debug"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "benchmark": "VirtualHome",
            "track": "symbolic_evolving_graph",
            "task_id": self.task_id,
            "task_type": self.task_type,
            "goal_text": self.goal_text,
            "graph_name": self.graph_name,
            "source": self.source,
            "suite_tags": list(self.suite_tags),
            "success_predicate_kind": self.success_predicate.get("kind"),
            "action_count": len(self.action_space),
        }


def build_manifest(config: HarnessConfig, force: bool = False) -> dict[str, Any]:
    from task_pool import load_selected_pool
    selected = load_selected_pool()
    if selected is not None:
        return selected
    manifest_dir = config.outputs_dir / "manifests"
    public_path = manifest_dir / f"{MANIFEST_ID}.json"
    private_path = manifest_dir / f"{PRIVATE_MANIFEST_ID}.json"
    if public_path.exists() and private_path.exists() and not force:
        with private_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    tasks = _tasks()
    official_specs, official_report = load_official_task_specs(
        root_dir=config.root_dir,
        virtualhome_repo=config.virtualhome_repo,
        limit=config.official_task_limit,
        max_program_steps=config.max_steps,
    )
    official_tasks = [
        TaskRecord(
            task_id=spec.task_id,
            task_type=spec.task_type,
            goal_text=spec.goal_text,
            graph_name=spec.graph_name,
            graph=spec.graph,
            expected_plan=spec.expected_plan,
            action_space=spec.action_space,
            success_predicate=spec.success_predicate,
            suite_tags=["official_debug_10", "official_mini_30", "official_compatible_full_v1"],
            source=spec.source,
        )
        for spec in official_specs
    ]
    private_suites = {
        "debug_5": [task.to_dict() for task in tasks],
        "mini_5": [task.to_dict() for task in tasks],
    }
    public_suites = {
        "debug_5": [task.to_public_dict() for task in tasks],
        "mini_5": [task.to_public_dict() for task in tasks],
    }
    if official_tasks:
        private_suites["official_debug_10"] = [task.to_dict() for task in official_tasks[:10]]
        private_suites["official_mini_30"] = [task.to_dict() for task in official_tasks[:30]]
        private_suites["official_compatible_full_v1"] = [task.to_dict() for task in official_tasks]
        public_suites["official_debug_10"] = [task.to_public_dict() for task in official_tasks[:10]]
        public_suites["official_mini_30"] = [task.to_public_dict() for task in official_tasks[:30]]
        public_suites["official_compatible_full_v1"] = [task.to_public_dict() for task in official_tasks]
    private_manifest = {
        "manifest_id": PRIVATE_MANIFEST_ID,
        "benchmark": "VirtualHome",
        "track": "symbolic_evolving_graph",
        "visibility": "private_runtime",
        "source_repo": "https://github.com/xavierpuigf/virtualhome",
        "source_runtime": "VirtualHome Evolving Graph",
        "suites": private_suites,
        "task_count": len(tasks) + len(official_tasks),
        "official_dataset_report": official_report,
        "notes": [
            "This is a local deterministic symbolic suite for harness bring-up.",
            "If official_* suites are present, they are compatible subsets from the official VirtualHome program dataset.",
            "It is not VirtualHome Unity/video evaluation and not a full official benchmark split.",
            "This private runtime manifest contains verifier predicates and expected plans for harness checks.",
            f"The public manifest is {public_path.name}.",
        ],
    }
    public_manifest = {
        "manifest_id": MANIFEST_ID,
        "benchmark": "VirtualHome",
        "track": "symbolic_evolving_graph",
        "visibility": "public_agent_metadata",
        "source_repo": "https://github.com/xavierpuigf/virtualhome",
        "source_runtime": "VirtualHome Evolving Graph",
        "suites": public_suites,
        "task_count": len(tasks) + len(official_tasks),
        "official_dataset_status": official_report.get("status"),
        "official_compatible_task_count": len(official_tasks),
        "official_compatible_full_suite": "official_compatible_full_v1" if official_tasks else None,
        "notes": [
            "Public manifest for agent-facing task metadata.",
            "No expected plans, full success predicates, or initial graph states are included.",
            "This is not VirtualHome Unity/video evaluation and not a full official benchmark split.",
            "Official suites, when present, are compatible Evolving Graph subsets filtered by replay checks.",
            "Set VIRTUALHOME_OFFICIAL_LIMIT=0 and force-manifest to scan the full official source dataset for all compatible symbolic tasks.",
        ],
    }
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with private_path.open("w", encoding="utf-8") as f:
        json.dump(private_manifest, f, indent=2, ensure_ascii=False)
    with public_path.open("w", encoding="utf-8") as f:
        json.dump(public_manifest, f, indent=2, ensure_ascii=False)
    return private_manifest


def select_suite(
    manifest: dict[str, Any],
    suite: str,
    start_index: int = 0,
    num_tasks: int | None = None,
) -> list[TaskRecord]:
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    rows = list((manifest.get("suites") or {}).get(suite) or [])
    if not rows:
        raise ValueError(f"Unknown or empty suite: {suite}")
    if start_index:
        rows = rows[start_index:]
        if not rows:
            raise ValueError(f"start_index {start_index} is outside suite '{suite}' with no tasks remaining")
    if num_tasks is not None:
        rows = rows[:num_tasks]
    return [TaskRecord(**row) for row in rows]


def _tasks() -> list[TaskRecord]:
    return [
        TaskRecord(
            task_id="vh_open_fridge",
            task_type="open_object",
            goal_text="Open the fridge in the symbolic kitchen.",
            graph_name="debug_kitchen",
            graph=_debug_kitchen_graph(fridge_open=False, lamp_on=False),
            expected_plan=["[Find] <fridge> (3)", "[Open] <fridge> (3)"],
            action_space=_action_space(),
            success_predicate={"kind": "object_state", "object_id": 3, "state": "OPEN", "expected": True},
            suite_tags=["debug_5", "mini_5"],
        ),
        TaskRecord(
            task_id="vh_plate_in_fridge",
            task_type="put_in",
            goal_text="Put the plate inside the fridge.",
            graph_name="debug_kitchen",
            graph=_debug_kitchen_graph(fridge_open=False, lamp_on=False),
            expected_plan=[
                "[Find] <fridge> (3)",
                "[Open] <fridge> (3)",
                "[Find] <plate> (4)",
                "[Grab] <plate> (4)",
                "[PutIn] <plate> (4) <fridge> (3)",
            ],
            action_space=_action_space(),
            success_predicate={"kind": "relation", "from_id": 4, "relation": "INSIDE", "to_id": 3},
            suite_tags=["debug_5", "mini_5"],
        ),
        TaskRecord(
            task_id="vh_switch_on_lamp",
            task_type="switch_on",
            goal_text="Switch on the lamp.",
            graph_name="debug_kitchen",
            graph=_debug_kitchen_graph(fridge_open=False, lamp_on=False),
            expected_plan=["[Find] <lamp> (5)", "[SwitchOn] <lamp> (5)"],
            action_space=_action_space(),
            success_predicate={"kind": "object_state", "object_id": 5, "state": "ON", "expected": True},
            suite_tags=["debug_5", "mini_5"],
        ),
        TaskRecord(
            task_id="vh_close_fridge",
            task_type="close_object",
            goal_text="Close the fridge, which starts open.",
            graph_name="debug_kitchen",
            graph=_debug_kitchen_graph(fridge_open=True, lamp_on=False),
            expected_plan=["[Find] <fridge> (3)", "[Close] <fridge> (3)"],
            action_space=_action_space(),
            success_predicate={"kind": "object_state", "object_id": 3, "state": "CLOSED", "expected": True},
            suite_tags=["debug_5", "mini_5"],
        ),
        TaskRecord(
            task_id="vh_mug_on_table",
            task_type="put_on",
            goal_text="Put the mug on the table.",
            graph_name="debug_kitchen",
            graph=_debug_kitchen_graph(fridge_open=False, lamp_on=False),
            expected_plan=[
                "[Find] <mug> (6)",
                "[Grab] <mug> (6)",
                "[Find] <table> (7)",
                "[PutBack] <mug> (6) <table> (7)",
            ],
            action_space=_action_space(),
            success_predicate={"kind": "relation", "from_id": 6, "relation": "ON", "to_id": 7},
            suite_tags=["debug_5", "mini_5"],
        ),
    ]


def _debug_kitchen_graph(*, fridge_open: bool, lamp_on: bool) -> dict[str, Any]:
    fridge_state = "OPEN" if fridge_open else "CLOSED"
    lamp_state = "ON" if lamp_on else "OFF"
    return {
        "nodes": [
            {"id": 1, "class_name": "kitchen", "category": "Rooms", "properties": [], "states": []},
            {"id": 2, "class_name": "character", "category": "", "properties": [], "states": []},
            {
                "id": 3,
                "class_name": "fridge",
                "category": "Appliances",
                "properties": ["CAN_OPEN", "CONTAINERS"],
                "states": [fridge_state],
            },
            {"id": 4, "class_name": "plate", "category": "Props", "properties": ["GRABBABLE", "RECIPIENT"], "states": []},
            {"id": 5, "class_name": "lamp", "category": "Props", "properties": ["HAS_SWITCH"], "states": [lamp_state]},
            {"id": 6, "class_name": "mug", "category": "Props", "properties": ["GRABBABLE", "RECIPIENT"], "states": []},
            {"id": 7, "class_name": "table", "category": "Furniture", "properties": ["SURFACES"], "states": []},
        ],
        "edges": [
            {"from_id": 2, "relation_type": "INSIDE", "to_id": 1},
            {"from_id": 3, "relation_type": "INSIDE", "to_id": 1},
            {"from_id": 4, "relation_type": "INSIDE", "to_id": 1},
            {"from_id": 5, "relation_type": "INSIDE", "to_id": 1},
            {"from_id": 6, "relation_type": "INSIDE", "to_id": 1},
            {"from_id": 7, "relation_type": "INSIDE", "to_id": 1},
            *_close_edges(2, [3, 4, 5, 6, 7]),
        ],
    }


def _close_edges(character_id: int, object_ids: list[int]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for object_id in object_ids:
        edges.append({"from_id": character_id, "relation_type": "CLOSE", "to_id": object_id})
        edges.append({"from_id": object_id, "relation_type": "CLOSE", "to_id": character_id})
    return edges


def _action_space() -> list[str]:
    return [
        "[Find] <fridge> (3)",
        "[Open] <fridge> (3)",
        "[Close] <fridge> (3)",
        "[Find] <plate> (4)",
        "[Grab] <plate> (4)",
        "[PutIn] <plate> (4) <fridge> (3)",
        "[Find] <lamp> (5)",
        "[SwitchOn] <lamp> (5)",
        "[SwitchOff] <lamp> (5)",
        "[Find] <mug> (6)",
        "[Grab] <mug> (6)",
        "[Find] <table> (7)",
        "[PutBack] <mug> (6) <table> (7)",
    ]
