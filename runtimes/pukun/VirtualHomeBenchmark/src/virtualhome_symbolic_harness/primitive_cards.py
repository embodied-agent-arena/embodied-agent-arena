from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PrimitiveCard:
    name: str
    signature: str
    canonical_family: str
    description: str
    arguments: dict[str, str]
    returns: str
    side_effect: str
    leakage_level: str
    limitations: list[str]
    backend_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CARDS = [
    PrimitiveCard(
        "get_task_context",
        "get_task_context()",
        "CTX",
        "Return task id, goal text, suite, action budget, and success predicate kind.",
        {},
        "dict",
        "none",
        "L1",
        ["Does not expose the expected plan."],
        "local manifest public fields",
    ),
    PrimitiveCard(
        "list_actions",
        "list_actions(query=None, limit=80)",
        "CTX",
        "List candidate one-step VirtualHome program actions, optionally filtered by text.",
        {"query": "Optional substring such as 'keyboard', 'chair', or 'home_office'.", "limit": "Maximum returned actions, capped at 200."},
        "list[str]",
        "none",
        "L1",
        ["This is a bounded candidate pool, not an ordered gold plan.", "Use query/limit instead of printing huge action lists."],
        "suite action candidate pool",
    ),
    PrimitiveCard(
        "list_executable_actions",
        "list_executable_actions(query=None, limit=40)",
        "STATE",
        "List candidate program actions that are currently executable under the Evolving Graph preconditions.",
        {"query": "Optional substring such as an object class/id.", "limit": "Maximum executable actions returned, capped at 100."},
        "list[str]",
        "none",
        "L2/L3 precondition probe",
        [
            "This is not a gold plan and is not ordered by task relevance.",
            "It dry-runs candidates on a state copy and does not change the benchmark state.",
        ],
        "VirtualHome ScriptExecutor.execute_one_step on copied state",
    ),
    PrimitiveCard(
        "query_symbolic_state",
        "query_symbolic_state(scope=None)",
        "STATE",
        "Return compact symbolic nodes and relations from the current Evolving Graph state.",
        {"scope": "Optional object class/id substring."},
        "dict",
        "none",
        "L2 symbolic state",
        ["No Unity pixels or hidden gold program are returned."],
        "VirtualHome Evolving Graph state",
    ),
    PrimitiveCard(
        "validate_program_step",
        "validate_program_step(action_line)",
        "STATE",
        "Dry-run one VirtualHome program line against the current symbolic state and return validity/error without state mutation.",
        {"action_line": "One candidate line, usually copied from list_actions() or list_executable_actions()."},
        "dict with valid_action/error/verifier summary and compact after-state when valid",
        "none",
        "L2/L3 precondition probe",
        ["Does not execute the action in the real benchmark state.", "A valid single step is not necessarily task-solving."],
        "VirtualHome ScriptExecutor.execute_one_step on copied state",
    ),
    PrimitiveCard(
        "explain_action_preconditions",
        "explain_action_preconditions(action_line)",
        "STATE",
        "Return validation plus the relevant current graph nodes/relations for the objects mentioned in an action line.",
        {"action_line": "One candidate VirtualHome action line."},
        "dict",
        "none",
        "L2 symbolic state",
        ["Uses executor error text and current graph context; it does not reveal hidden expert programs."],
        "VirtualHome Evolving Graph state + executor error",
    ),
    PrimitiveCard(
        "execute_program_step",
        "execute_program_step(action_line)",
        "HACT",
        "Execute exactly one VirtualHome program line, such as '[Open] <fridge> (3)'.",
        {"action_line": "One supported VirtualHome action line."},
        "dict with valid_action/valid, error, verifier summary, and compact after-state around the character/action object",
        "one symbolic transition",
        "L3",
        ["Does not execute multi-step plans automatically.", "Must be called from solve.py, never from python -c or temporary scripts."],
        "VirtualHome ScriptExecutor.execute_one_step",
    ),
    PrimitiveCard(
        "write_evidence",
        "write_evidence(key, value)",
        "EVD",
        "Store trace-local evidence for later inspection.",
        {"key": "Evidence key.", "value": "JSON-serializable value."},
        "dict",
        "none",
        "L1",
        ["Does not change symbolic state."],
        "harness memory",
    ),
    PrimitiveCard(
        "read_evidence",
        "read_evidence()",
        "EVD",
        "Read trace-local evidence accumulated by the generated code.",
        {},
        "dict",
        "none",
        "L1",
        ["Current task only."],
        "harness memory",
    ),
    PrimitiveCard(
        "check_activity_success",
        "check_activity_success()",
        "VERIFY",
        "Return safe success/completion/score for the current symbolic task.",
        {},
        "dict",
        "none",
        "L1 verifier summary",
        ["Does not reveal the expected plan."],
        "local symbolic predicate verifier",
    ),
]


def primitive_cards() -> list[dict[str, Any]]:
    return [card.to_dict() for card in CARDS]


def get_primitive_cards() -> list[dict[str, Any]]:
    return primitive_cards()
