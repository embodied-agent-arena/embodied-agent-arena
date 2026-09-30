from __future__ import annotations

import sys
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord


@dataclass(frozen=True)
class StepResult:
    action_line: str
    valid_action: bool
    observation_before: dict[str, Any]
    observation_after: dict[str, Any]
    verification: dict[str, Any]
    error: str | None


class VirtualHomeSymbolicBackend:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.task: TaskRecord | None = None
        self.executor: Any | None = None
        self.state: Any | None = None
        self.graph_state_list: list[dict[str, Any]] = []

    def reset_task(self, task: TaskRecord) -> dict[str, Any]:
        ensure_virtualhome_paths(self.config)
        from evolving_graph.environment import EnvironmentGraph, EnvironmentState
        from evolving_graph.execution import ScriptExecutor

        self.task = task
        graph = EnvironmentGraph(task.graph)
        self.executor = ScriptExecutor(graph, name_equivalence={})
        self.state = EnvironmentState(graph, {}, instance_selection=True)
        self.graph_state_list = [self.state.to_dict()]
        return self.observe()

    def observe(self) -> dict[str, Any]:
        self._require_state()
        return summarize_state(self.state)

    def step(self, action_line: str, evidence: dict[str, Any]) -> StepResult:
        self._require_state()
        from evolving_graph.scripts import read_script_from_list_string

        observation_before = self.observe()
        error = None
        valid = False
        try:
            script = read_script_from_list_string([action_line])
            valid, next_state = self.executor.execute_one_step(script, self.state)
            if valid:
                self.state = next_state
                self.graph_state_list.append(self.state.to_dict())
            else:
                error = self.executor.info.get_error_string() or "action_not_executable"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        verification = self.check_success(evidence)
        return StepResult(
            action_line=action_line,
            valid_action=bool(valid),
            observation_before=observation_before,
            observation_after=self.observe(),
            verification=verification,
            error=error,
        )

    def validate_step(self, action_line: str, evidence: dict[str, Any] | None = None) -> StepResult:
        self._require_state()
        from evolving_graph.scripts import read_script_from_list_string

        observation_before = self.observe()
        error = None
        valid = False
        observation_after = observation_before
        try:
            state_copy = copy.deepcopy(self.state)
            script = read_script_from_list_string([action_line])
            valid, next_state = self.executor.execute_one_step(script, state_copy)
            if valid:
                observation_after = summarize_state(next_state)
            else:
                error = self.executor.info.get_error_string() or "action_not_executable"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        verification = self.check_success(evidence or {})
        return StepResult(
            action_line=action_line,
            valid_action=bool(valid),
            observation_before=observation_before,
            observation_after=observation_after,
            verification=verification,
            error=error,
        )

    def check_success(self, evidence: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.task is None or self.state is None:
            return {"success": False, "completed": False, "score": 0.0, "predicate_kind": "none", "reason": "not_started"}
        success, _raw_reason = evaluate_predicate(self.state, self.task.success_predicate)
        return {
            "success": success,
            "completed": success,
            "score": 1.0 if success else 0.0,
            "predicate_kind": self.task.success_predicate.get("kind"),
            "reason": "satisfied" if success else "not_satisfied",
        }

    def _require_state(self) -> None:
        if self.task is None or self.executor is None or self.state is None:
            raise RuntimeError("VirtualHomeSymbolicBackend.reset_task must be called first.")


def ensure_virtualhome_paths(config: HarnessConfig) -> None:
    repo = config.virtualhome_repo
    simulation = repo / "virtualhome" / "simulation"
    for path in (repo, simulation):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def summarize_state(state: Any, scope: str | None = None) -> dict[str, Any]:
    query = str(scope or "").lower().strip()
    nodes = []
    for node in state.get_nodes():
        row = {
            "id": node.id,
            "class_name": node.class_name,
            "category": node.category,
            "properties": sorted(prop.name for prop in node.properties),
            "states": sorted(item.name for item in node.states),
        }
        if not query or query in str(node.id).lower() or query in node.class_name.lower():
            nodes.append(row)
    edges = sorted(
        {
            (edge["from_id"], edge["relation_type"], edge["to_id"])
            for edge in state.to_dict().get("edges", [])
        }
    )
    edge_rows = [
        {"from_id": from_id, "relation": relation, "to_id": to_id}
        for from_id, relation, to_id in edges
        if not query or query in str(from_id).lower() or query in str(to_id).lower()
    ]
    return {"nodes": nodes, "edges": edge_rows}


def evaluate_predicate(state: Any, predicate: dict[str, Any]) -> tuple[bool, str]:
    from evolving_graph.environment import ExistsRelation, NodeInstance, NodeInstanceFilter
    from evolving_graph.execution import Relation, State

    kind = predicate.get("kind")
    if kind == "graph_delta":
        from .official_dataset import evaluate_graph_delta_dict

        return evaluate_graph_delta_dict(state.to_dict(), predicate)
    if kind == "object_state":
        node = state.get_node(int(predicate["object_id"]))
        expected_state = State[str(predicate["state"]).upper()]
        has_state = expected_state in node.states
        expected = bool(predicate.get("expected", True))
        success = has_state is expected
        return success, f"object_{node.id}_{expected_state.name}_{'matched' if success else 'not_matched'}"
    if kind == "relation":
        from_node = state.get_node(int(predicate["from_id"]))
        to_node = state.get_node(int(predicate["to_id"]))
        relation = Relation[str(predicate["relation"]).upper()]
        success = state.evaluate(ExistsRelation(NodeInstance(from_node), relation, NodeInstanceFilter(to_node)))
        return success, f"relation_{from_node.id}_{relation.name}_{to_node.id}_{'present' if success else 'missing'}"
    return False, f"unknown_predicate_kind:{kind}"
