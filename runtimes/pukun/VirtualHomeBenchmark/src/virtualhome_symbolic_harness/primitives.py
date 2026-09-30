from __future__ import annotations

import re
from typing import Any

from .backend import StepResult, VirtualHomeSymbolicBackend
from .manifest import TaskRecord
from .primitive_cards import primitive_cards
from .trace import TraceWriter


FAMILIES = {
    "get_task_context": "CTX",
    "list_actions": "CTX",
    "list_executable_actions": "STATE",
    "query_symbolic_state": "STATE",
    "validate_program_step": "STATE",
    "explain_action_preconditions": "STATE",
    "execute_program_step": "HACT",
    "write_evidence": "EVD",
    "read_evidence": "EVD",
    "check_activity_success": "VERIFY",
    "agent_code_generated": "CTX",
    "code_execution_started": "CTX",
    "code_turn_feedback_written": "CTX",
    "code_execution_finished": "VERIFY",
    "side_effect_client_observed": "CTX",
}


class VirtualHomeSymbolicPrimitives:
    def __init__(
        self,
        backend: VirtualHomeSymbolicBackend,
        trace: TraceWriter,
        task: TaskRecord,
        suite: str,
        max_env_steps: int,
    ):
        self.backend = backend
        self.trace = trace
        self.task = task
        self.suite = suite
        self.max_env_steps = max_env_steps
        self.evidence: dict[str, Any] = {}
        self.metrics = {
            "env_steps": 0,
            "invalid_action_count": 0,
            "code_exception_count": 0,
            "code_timeout_count": 0,
        }

    def get_task_context(self) -> dict[str, Any]:
        context = {
            "benchmark": "VirtualHome",
            "track": "symbolic_evolving_graph",
            "benchmark_suite": self.suite,
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "goal_text": self.task.goal_text,
            "graph_name": self.task.graph_name,
            "success_predicate_kind": self.task.success_predicate.get("kind"),
            "max_steps": self.max_env_steps,
            "note": "Use one-step primitives only. The expected plan is not exposed.",
        }
        self._record_observation("get_task_context", {"result": context})
        return context

    def list_actions(self, query: str | None = None, limit: int = 80) -> list[str]:
        actions = list(self.task.action_space)
        if query:
            query_text = str(query).lower().strip()
            actions = [action for action in actions if query_text in action.lower()]
        bounded_limit = max(1, min(int(limit or 80), 200))
        result = actions[:bounded_limit]
        self._record_observation(
            "list_actions",
            {
                "query": query,
                "limit": bounded_limit,
                "returned_count": len(result),
                "total_matching_count": len(actions),
                "actions": result,
            },
        )
        return result

    def query_symbolic_state(self, scope: str | None = None) -> dict[str, Any]:
        result = self.backend.observe() if scope is None else __import__(
            "virtualhome_symbolic_harness.backend", fromlist=["summarize_state"]
        ).summarize_state(self.backend.state, scope)
        self._record_observation("query_symbolic_state", {"scope": scope, "result": result})
        return result

    def list_executable_actions(self, query: str | None = None, limit: int = 40) -> list[str]:
        candidate_limit = max(1, min(int(limit or 40), 100))
        query_text = str(query or "").lower().strip()
        candidates = [
            action
            for action in self.task.action_space
            if not query_text or query_text in action.lower()
        ]
        executable: list[str] = []
        checked_count = 0
        for action in candidates:
            if len(executable) >= candidate_limit:
                break
            checked_count += 1
            validation = self.backend.validate_step(str(action), self.evidence)
            if validation.valid_action:
                executable.append(str(action))
        self._record_observation(
            "list_executable_actions",
            {
                "query": query,
                "limit": candidate_limit,
                "checked_count": checked_count,
                "total_matching_count": len(candidates),
                "returned_count": len(executable),
                "actions": executable,
            },
        )
        return executable

    def validate_program_step(self, action_line: str) -> dict[str, Any]:
        result = self.backend.validate_step(str(action_line), self.evidence)
        payload = {
            "valid_action": result.valid_action,
            "valid": result.valid_action,
            "error": result.error,
            "verification": result.verification,
            "state_after": compact_state_for_action(result.action_line, result.observation_after),
        }
        self._record_observation(
            "validate_program_step",
            {"action_line": str(action_line), "result": payload},
        )
        return payload

    def explain_action_preconditions(self, action_line: str) -> dict[str, Any]:
        observation = self.backend.observe()
        validation = self.backend.validate_step(str(action_line), self.evidence)
        result = {
            "action_line": str(action_line),
            "valid_action": validation.valid_action,
            "error": validation.error,
            "current_relevant_state": compact_state_for_action(str(action_line), observation),
            "note": (
                "VirtualHome preconditions are enforced by the Evolving Graph executor. "
                "Use query_symbolic_state(scope) to inspect object ids/classes mentioned here."
            ),
        }
        self._record_observation(
            "explain_action_preconditions",
            {"action_line": str(action_line), "result": result},
        )
        return result

    def execute_program_step(self, action_line: str) -> dict[str, Any]:
        if self.metrics["env_steps"] >= self.max_env_steps:
            observation = self.backend.observe()
            verification = self.backend.check_success(self.evidence)
            error = f"primitive step budget exceeded: max_env_steps={self.max_env_steps}"
            self.trace.record_event(
                self.task,
                {
                    "benchmark_suite": self.suite,
                    "primitive": "execute_program_step",
                    "canonical_family": FAMILIES["execute_program_step"],
                    "side_effect": True,
                    "arguments": {"action_line": str(action_line)},
                    "valid_action": False,
                    "observation_before": observation,
                    "observation_after": observation,
                    "verification": verification,
                    "error": error,
                    "evidence": dict(self.evidence),
                },
            )
            return {
                "valid_action": False,
                "valid": False,
                "error": error,
                "verification": verification,
                "state_after": compact_state_for_action(str(action_line), observation),
            }
        result = self.backend.step(str(action_line), self.evidence)
        self.metrics["env_steps"] += 1
        if not result.valid_action:
            self.metrics["invalid_action_count"] += 1
        self._record_step("execute_program_step", {"action_line": str(action_line)}, result)
        return {
            "valid_action": result.valid_action,
            "valid": result.valid_action,
            "error": result.error,
            "verification": result.verification,
            "state_after": compact_state_for_action(result.action_line, result.observation_after),
        }

    def write_evidence(self, key: str, value: Any) -> dict[str, Any]:
        self.evidence[str(key)] = value
        result = {"key": str(key), "value": value}
        self._record_observation("write_evidence", {"result": result, "evidence": dict(self.evidence)})
        return result

    def read_evidence(self) -> dict[str, Any]:
        evidence = dict(self.evidence)
        self._record_observation("read_evidence", {"evidence": evidence})
        return evidence

    def check_activity_success(self) -> dict[str, Any]:
        result = self.backend.check_success(self.evidence)
        self._record_observation("check_activity_success", {"verification": result, "evidence": dict(self.evidence)})
        return result

    def record_harness_event(self, event_name: str, payload: dict[str, Any], side_effect: bool = False) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": event_name,
                "canonical_family": FAMILIES[event_name],
                "side_effect": side_effect,
                **payload,
            },
        )

    def _record_observation(self, primitive: str, payload: dict[str, Any]) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": primitive,
                "canonical_family": FAMILIES[primitive],
                "side_effect": False,
                **payload,
            },
        )

    def _record_step(self, primitive: str, arguments: dict[str, Any], result: StepResult) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": primitive,
                "canonical_family": FAMILIES[primitive],
                "side_effect": True,
                "arguments": arguments,
                "valid_action": result.valid_action,
                "observation_before": result.observation_before,
                "observation_after": result.observation_after,
                "verification": result.verification,
                "error": result.error,
                "evidence": dict(self.evidence),
            },
        )

    def cards_for_prompt(self) -> list[dict[str, Any]]:
        return primitive_cards()


def compact_state_for_action(action_line: str, observation: dict[str, Any]) -> dict[str, Any]:
    ids = {int(value) for value in re.findall(r"\((\d+)\)", str(action_line))}
    nodes = observation.get("nodes") or []
    edges = observation.get("edges") or []
    character_ids = {
        int(node["id"])
        for node in nodes
        if str(node.get("class_name") or "").lower() == "character" and isinstance(node.get("id"), int)
    }
    relevant_ids = ids | character_ids
    relevant_nodes = [node for node in nodes if node.get("id") in relevant_ids]
    relevant_edges = [
        edge
        for edge in edges
        if edge.get("from_id") in relevant_ids or edge.get("to_id") in relevant_ids
    ][:80]
    return {
        "action_object_ids": sorted(ids),
        "character_ids": sorted(character_ids),
        "nodes": relevant_nodes,
        "edges": relevant_edges,
        "edge_limit": 80,
    }
