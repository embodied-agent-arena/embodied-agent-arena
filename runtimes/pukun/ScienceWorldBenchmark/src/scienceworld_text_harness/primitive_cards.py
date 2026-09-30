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
    side_effect: str
    returns: str
    backend_source: str
    leakage_level: str = "L1/L3"
    limitations: list[str] | None = None


SOLVE_PY_ONLY_LIMITATION = (
    "OpenHands workspace policy: call this primitive only from solve.py; do not "
    "import primitive_api from python -c, stdin Python, heredocs, notebooks, or temporary scripts."
)


CARDS = [
    PrimitiveCard(
        name="get_task_context",
        signature="get_task_context()",
        canonical_family="CTX",
        description="Return benchmark, task id/name, task description, variation, simplification, and step budget.",
        arguments={},
        side_effect="none",
        returns="dict",
        backend_source="TaskRecord + ScienceWorld taskdescription()",
        limitations=["Does not expose gold paths or hidden simulator state."],
    ),
    PrimitiveCard(
        name="observe_text_world",
        signature="observe_text_world()",
        canonical_family="CTX",
        description="Return the latest ScienceWorld textual observation.",
        arguments={},
        side_effect="none",
        returns="str",
        backend_source="ScienceWorldEnv.step/reset observation cache",
        limitations=["Only returns the currently revealed text observation."],
    ),
    PrimitiveCard(
        name="list_actions",
        signature="list_actions()",
        canonical_family="CTX",
        description="Return current grounded valid actions from ScienceWorld info['valid'].",
        arguments={},
        side_effect="none",
        returns="list[str]",
        backend_source="ScienceWorldEnv get_valid_action_object_combinations via info['valid']",
        limitations=["Returns grounded actions, not template actions such as 'go OBJ'."],
    ),
    PrimitiveCard(
        name="inspect_current_state",
        signature="inspect_current_state(query=None, limit=80)",
        canonical_family="STATE",
        description="Return a compact observed-only summary of the current room text, inventory text, and current grounded action groups.",
        arguments={
            "query": "Optional string or list of strings used to surface current actions containing any query term.",
            "limit": "Maximum actions per group / query-action list.",
        },
        side_effect="none",
        returns="dict with current_location, visible_entries, exits_or_doors, inventory_entries, query_actions, substance_query_terms, substance_candidates, action_groups, recent_actions, loop_warnings, valid_action_count, boundary",
        backend_source="Current observation text + inventory text + ScienceWorld valid action list + RoBench recent action trace",
        leakage_level="L2 observed text + L3 grounded action metadata",
        limitations=[
            "Observation-only helper; it does not move, manipulate, score, plan, or choose an action.",
            "Only summarizes the current public text observation, current inventory text, current valid actions, and actions already attempted by this wrapper.",
            "Substance candidates are parsed from visible text such as 'substance called water' and matched only against current valid actions.",
            "When query=None, substance candidates may be filtered by public goal terms such as water/unknown substance; this is not a hidden solution.",
            "Loop warnings are generic anti-repetition hints from the attempted action trace, not task-specific solutions.",
            "Does not expose hidden objects, hidden room contents, gold paths, future states, or a correct action sequence.",
        ],
    ),
    PrimitiveCard(
        name="filter_actions",
        signature="filter_actions(include=None, exclude=None, startswith=None, limit=80, exclude_failed=True)",
        canonical_family="CTX",
        description="Return current grounded valid actions filtered by simple text terms, optionally hiding actions that already failed in this task.",
        arguments={
            "include": "Optional substring/list of substrings that returned actions must contain; may also be substance_candidate dict(s) from inspect_current_state.",
            "exclude": "Optional substring or list of substrings to filter out.",
            "startswith": "Optional action prefix.",
            "limit": "Maximum returned actions.",
            "exclude_failed": "When True, suppress actions recorded by step_text_action as repeated invalid/parser-no-match failures.",
        },
        side_effect="none",
        returns="list[str]",
        backend_source="Current ScienceWorld info['valid'] list",
        limitations=[
            "Only filters current valid actions; it does not execute, fuzzy-match, navigate, or choose a plan.",
            "When include is substance_candidate dict(s), it restricts to that candidate's current matching_actions.",
            "Every returned action still must be passed exactly to step_text_action(action).",
            "Use exclude_failed=False only when deliberately rechecking a previously failed action after the state changed.",
        ],
    ),
    PrimitiveCard(
        name="list_recent_failures",
        signature="list_recent_failures(limit=20)",
        canonical_family="STATE",
        description="Return actions that failed in this task, with failure counts and last error summaries.",
        arguments={"limit": "Maximum failure records to return."},
        side_effect="none",
        returns="list[dict]",
        backend_source="RoBench wrapper failure memory",
        limitations=[
            "Only reports failures observed through this primitive facade in the current task.",
            "Does not reveal any hidden correct action sequence.",
        ],
    ),
    PrimitiveCard(
        name="get_score_state",
        signature="get_score_state()",
        canonical_family="VERIFY",
        description="Return the current verifier summary, wrapper metrics, and failure-memory count.",
        arguments={},
        side_effect="none",
        returns="dict",
        backend_source="ScienceWorld verifier + RoBench wrapper metrics",
        limitations=["Verifier summary only; it does not expose hidden solutions or future rewards."],
    ),
    PrimitiveCard(
        name="step_text_action",
        signature="step_text_action(action)",
        canonical_family="HACT",
        description="Execute exactly one current grounded ScienceWorld action.",
        arguments={"action": "Exact action string copied from list_actions() or filter_actions()."},
        side_effect="one native text action",
        returns="Compact StepResult dict with valid_action/error/observation_after/verification and bounded action counts/sample.",
        backend_source="ScienceWorldEnv.step(action)",
        limitations=[
            "Action must be copied from list_actions(); no fuzzy matching or planning.",
            "The OpenHands facade returns a compact result; call list_actions() or filter_actions() for the current full valid action set.",
            "Do not print full result/action-list objects into stdout; keep evidence compact.",
            "Some valid ScienceWorld actions, including unrelated 'focus on ...' actions, can immediately fail or terminate a task.",
            "After each side-effecting action, call check_success() and stop if it reports done or a terminal score.",
            "Must be called from python solve.py; ad-hoc python -c snippets and temporary scripts are rejected for side-effecting actions.",
            "After one solve.py process has issued side-effect actions, rerunning solve.py cannot continue manipulating the same backend session.",
        ],
    ),
    PrimitiveCard(
        name="look",
        signature="look()",
        canonical_family="STATE",
        description="Return the current room/world description as a free ScienceWorld action.",
        arguments={},
        side_effect="none",
        returns="str",
        backend_source="ScienceWorldEnv.look()",
        limitations=["Observation-only helper; it does not move, manipulate, or reveal hidden state."],
    ),
    PrimitiveCard(
        name="inventory",
        signature="inventory()",
        canonical_family="STATE",
        description="Return the agent inventory as a free ScienceWorld action.",
        arguments={},
        side_effect="none",
        returns="str",
        backend_source="ScienceWorldEnv.inventory()",
        limitations=["Only reports currently carried objects; it does not expose object locations elsewhere."],
    ),
    PrimitiveCard(
        name="write_evidence",
        signature="write_evidence(key, value)",
        canonical_family="EVD",
        description="Record agent evidence in trace-local memory.",
        arguments={"key": "Evidence key.", "value": "JSON-serializable value."},
        side_effect="none",
        returns="dict",
        backend_source="RoBench trace memory",
        limitations=["Trace-local memory only; it cannot alter the ScienceWorld environment or verifier state."],
    ),
    PrimitiveCard(
        name="read_evidence",
        signature="read_evidence()",
        canonical_family="EVD",
        description="Return evidence previously written by the generated code.",
        arguments={},
        side_effect="none",
        returns="dict",
        backend_source="RoBench trace memory",
        limitations=["Returns only evidence previously written by this generated code run."],
    ),
    PrimitiveCard(
        name="check_success",
        signature="check_success()",
        canonical_family="VERIFY",
        description="Return the current native ScienceWorld verifier state.",
        arguments={},
        side_effect="none",
        returns="dict with success, score, done, reward, source",
        backend_source="ScienceWorldEnv info['score']/done",
        limitations=["Returns verifier summary only; it does not reveal a gold action sequence or hidden solution."],
    ),
]


def get_primitive_cards() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for card in CARDS:
        payload = asdict(card)
        limitations = list(payload.get("limitations") or [])
        if SOLVE_PY_ONLY_LIMITATION not in limitations:
            limitations.append(SOLVE_PY_ONLY_LIMITATION)
        payload["limitations"] = limitations
        cards.append(payload)
    return cards


def render_primitive_cards_for_prompt() -> str:
    lines = []
    for card in get_primitive_cards():
        args = ", ".join(f"{name}: {desc}" for name, desc in card["arguments"].items()) or "none"
        lines.append(
            f"- {card['signature']} [{card['canonical_family']}] side_effect={card['side_effect']}: "
            f"{card['description']} Args: {args}. Returns: {card['returns']}. "
            f"Leakage: {card['leakage_level']}. Backend: {card['backend_source']}. "
            f"Limitations: {card['limitations'] or []}"
        )
    return "\n".join(lines)
