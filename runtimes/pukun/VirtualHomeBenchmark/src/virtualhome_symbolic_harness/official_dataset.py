from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


DATASET_DIR_NAME = "programs_processed_precond_nograb_morepreconds"
IGNORED_DELTA_RELATIONS = {"BETWEEN", "CLOSE", "FACING"}


@dataclass(frozen=True)
class OfficialTaskSpec:
    task_id: str
    task_type: str
    goal_text: str
    graph_name: str
    graph: dict[str, Any]
    expected_plan: list[str]
    action_space: list[str]
    success_predicate: dict[str, Any]
    source: str


def load_official_task_specs(
    *,
    root_dir: Path,
    virtualhome_repo: Path,
    limit: int | None = 30,
    max_program_steps: int = 8,
) -> tuple[list[OfficialTaskSpec], dict[str, Any]]:
    dataset_root = find_official_dataset_root(root_dir, virtualhome_repo)
    report: dict[str, Any] = {
        "dataset_root": str(dataset_root) if dataset_root else None,
        "requested_limit": limit if limit is not None else "all",
        "max_program_steps": max_program_steps,
        "scanned_files": 0,
        "accepted_tasks": 0,
        "rejection_counts": {},
        "rejection_examples": [],
    }
    if dataset_root is None:
        report["status"] = "missing_dataset"
        return [], report

    executable_root = dataset_root / "executable_programs"
    if not executable_root.exists():
        report["status"] = "missing_executable_programs"
        return [], report

    specs: list[OfficialTaskSpec] = []
    for program_path in sorted(executable_root.glob("**/*.txt")):
        report["scanned_files"] += 1
        spec, reject_reason = _build_task_spec(
            dataset_root=dataset_root,
            executable_root=executable_root,
            program_path=program_path,
            virtualhome_repo=virtualhome_repo,
            max_program_steps=max_program_steps,
        )
        if spec is None:
            _record_rejection(report, reject_reason or "unknown", program_path)
            continue
        specs.append(spec)
        if limit is not None and len(specs) >= limit:
            break

    action_pool = sorted({action for spec in specs for action in spec.expected_plan})
    specs = [replace(spec, action_space=action_pool) for spec in specs]
    report["accepted_tasks"] = len(specs)
    report["candidate_action_count"] = len(action_pool)
    report["status"] = "ok" if specs else "no_compatible_tasks"
    return specs, report


def find_official_dataset_root(root_dir: Path, virtualhome_repo: Path) -> Path | None:
    configured = os.environ.get("VIRTUALHOME_DATA_ROOT")
    if configured:
        path = Path(configured).expanduser().resolve()
        return path if path.is_dir() else None
    candidates = [
        root_dir / "data" / "official" / DATASET_DIR_NAME,
        root_dir / "data" / DATASET_DIR_NAME,
        virtualhome_repo / "virtualhome" / "dataset" / DATASET_DIR_NAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def build_graph_delta(init_graph: dict[str, Any], final_graph: dict[str, Any]) -> dict[str, Any]:
    init_states = _node_states(init_graph)
    final_states = _node_states(final_graph)
    required_node_states: list[dict[str, Any]] = []
    for node_id, states in sorted(final_states.items()):
        for state in sorted(states - init_states.get(node_id, set())):
            required_node_states.append({"object_id": node_id, "state": state, "expected": True})

    init_edges = _edge_set(init_graph)
    final_edges = _edge_set(final_graph)
    required_edges = [
        {"from_id": from_id, "relation": relation, "to_id": to_id, "expected": True}
        for from_id, relation, to_id in sorted(final_edges - init_edges)
        if relation not in IGNORED_DELTA_RELATIONS
    ]
    return {
        "kind": "graph_delta",
        "required_node_states": required_node_states,
        "required_edges": required_edges,
        "delta_counts": {
            "required_node_states": len(required_node_states),
            "required_edges": len(required_edges),
        },
    }


def evaluate_graph_delta_dict(state_graph: dict[str, Any], predicate: dict[str, Any]) -> tuple[bool, str]:
    node_states = _node_states(state_graph)
    edge_set = _edge_set(state_graph)
    missing: list[str] = []
    for row in predicate.get("required_node_states", []):
        node_id = int(row["object_id"])
        state = str(row["state"]).upper()
        expected = bool(row.get("expected", True))
        present = state in node_states.get(node_id, set())
        if present is not expected:
            missing.append(f"state:{node_id}:{state}:expected_{expected}")
    for row in predicate.get("required_edges", []):
        edge = (int(row["from_id"]), str(row["relation"]).upper(), int(row["to_id"]))
        expected = bool(row.get("expected", True))
        present = edge in edge_set
        if present is not expected:
            missing.append(f"edge:{edge[0]}:{edge[1]}:{edge[2]}:expected_{expected}")
    if missing:
        return False, "graph_delta_missing:" + ",".join(missing[:10])
    return True, "graph_delta_matched"


def _build_task_spec(
    *,
    dataset_root: Path,
    executable_root: Path,
    program_path: Path,
    virtualhome_repo: Path,
    max_program_steps: int,
) -> tuple[OfficialTaskSpec | None, str | None]:
    title, description, plan = _parse_program_file(program_path)
    if not plan:
        return None, "empty_program"
    if len(plan) > max_program_steps:
        return None, "program_too_long"

    graph_path = _matching_graph_path(dataset_root, executable_root, program_path)
    if graph_path is None:
        return None, "missing_init_final_graph"

    try:
        with graph_path.open("r", encoding="utf-8") as f:
            graphs = json.load(f)
        init_graph = _normalize_graph(graphs["init_graph"])
        final_graph = _normalize_graph(graphs["final_graph"])
        predicate = build_graph_delta(init_graph, final_graph)
    except Exception:
        return None, "invalid_graph_json"

    if not predicate["required_node_states"] and not predicate["required_edges"]:
        return None, "empty_graph_delta"

    ok, reason = _replay_plan_matches_delta(init_graph, plan, predicate, virtualhome_repo)
    if not ok:
        return None, f"replay_failed:{reason}"

    rel = program_path.relative_to(executable_root)
    task_id = "official_" + "_".join(rel.with_suffix("").parts)
    task_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id)
    graph_name = rel.parts[0] if rel.parts else "official_graph"
    return (
        OfficialTaskSpec(
            task_id=task_id,
            task_type=_task_type_from_plan(plan),
            goal_text=_goal_text(title, description),
            graph_name=graph_name,
            graph=init_graph,
            expected_plan=plan,
            action_space=[],
            success_predicate=predicate,
            source="official_virtualhome_programs_compatible_subset",
        ),
        None,
    )


def _parse_program_file(path: Path) -> tuple[str, str, list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [line.strip() for line in lines if line.strip()]
    metadata = [line for line in nonempty if "[" not in line]
    title = metadata[0] if metadata else path.stem
    description = metadata[1] if len(metadata) > 1 else ""
    plan = []
    for line in lines:
        if "[" not in line:
            continue
        action = line[line.index("[") :].strip()
        plan.append(_normalize_program_line(action))
    return title, description, plan


def _normalize_program_line(line: str) -> str:
    line = re.sub(r"\((?:\d+\.)?(\d+)\)", r"(\1)", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line


def _matching_graph_path(dataset_root: Path, executable_root: Path, program_path: Path) -> Path | None:
    rel = program_path.relative_to(executable_root).with_suffix(".json")
    for folder in ("init_and_final_graph", "init_and_final_graphs"):
        candidate = dataset_root / folder / rel
        if candidate.exists():
            return candidate
    return None


def _normalize_graph(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = []
    for node in graph.get("nodes", []):
        row = dict(node)
        row["properties"] = [str(item).upper() for item in row.get("properties", [])]
        row["states"] = [str(item).upper() for item in row.get("states", [])]
        nodes.append(row)
    edges = []
    for edge in graph.get("edges", []):
        relation = edge.get("relation_type", edge.get("relation"))
        edges.append(
            {
                "from_id": int(edge["from_id"]),
                "relation_type": str(relation).upper(),
                "to_id": int(edge["to_id"]),
            }
        )
    return {"nodes": nodes, "edges": edges}


def _replay_plan_matches_delta(
    init_graph: dict[str, Any],
    plan: list[str],
    predicate: dict[str, Any],
    virtualhome_repo: Path,
) -> tuple[bool, str]:
    try:
        _ensure_virtualhome_paths(virtualhome_repo)
        from evolving_graph.environment import EnvironmentGraph, EnvironmentState
        from evolving_graph.execution import ScriptExecutor
        from evolving_graph.scripts import read_script_from_list_string

        graph = EnvironmentGraph(init_graph)
        executor = ScriptExecutor(graph, name_equivalence={})
        state = EnvironmentState(graph, {}, instance_selection=True)
        for action_line in plan:
            script = read_script_from_list_string([action_line])
            valid, next_state = executor.execute_one_step(script, state)
            if not valid:
                return False, "invalid_action"
            state = next_state
        success, reason = evaluate_graph_delta_dict(state.to_dict(), predicate)
        return success, reason
    except Exception as exc:
        return False, type(exc).__name__


def _ensure_virtualhome_paths(virtualhome_repo: Path) -> None:
    for path in (virtualhome_repo, virtualhome_repo / "virtualhome" / "simulation"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def _goal_text(title: str, description: str) -> str:
    text = title.strip()
    if description.strip() and description.strip() != text:
        text = f"{text}: {description.strip()}"
    return text


def _task_type_from_plan(plan: list[str]) -> str:
    semantic_actions: list[str] = []
    for line in plan:
        match = re.match(r"\[(\w+)\]", line.strip())
        if not match:
            continue
        action = match.group(1).lower()
        if action not in {"find", "walk", "lookat", "turnto"}:
            semantic_actions.append(action)
    return semantic_actions[-1] if semantic_actions else "navigation_or_observation"


def _node_states(graph: dict[str, Any]) -> dict[int, set[str]]:
    result: dict[int, set[str]] = {}
    for node in graph.get("nodes", []):
        result[int(node["id"])] = {str(state).upper() for state in node.get("states", [])}
    return result


def _edge_set(graph: dict[str, Any]) -> set[tuple[int, str, int]]:
    edges = set()
    for edge in graph.get("edges", []):
        relation = edge.get("relation_type", edge.get("relation"))
        edges.add((int(edge["from_id"]), str(relation).upper(), int(edge["to_id"])))
    return edges


def _record_rejection(report: dict[str, Any], reason: str, path: Path) -> None:
    counts = report["rejection_counts"]
    counts[reason] = counts.get(reason, 0) + 1
    if len(report["rejection_examples"]) < 20:
        report["rejection_examples"].append({"reason": reason, "path": str(path)})
