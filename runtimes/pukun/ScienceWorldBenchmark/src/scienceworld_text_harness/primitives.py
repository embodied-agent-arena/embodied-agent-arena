from __future__ import annotations

import os
import re
from typing import Any

from .backend import StepResult, invalid_step_result
from .manifest import TaskRecord
from .state_machine import ScienceWorldTextStateMachine
from .trace import TraceWriter


FAMILIES = {
    "get_task_context": "CTX",
    "observe_text_world": "CTX",
    "list_actions": "CTX",
    "inspect_current_state": "STATE",
    "filter_actions": "CTX",
    "list_recent_failures": "STATE",
    "get_score_state": "VERIFY",
    "step_text_action": "HACT",
    "look": "STATE",
    "inventory": "STATE",
    "check_success": "VERIFY",
    "write_evidence": "EVD",
    "read_evidence": "EVD",
    "agent_code_generated": "CTX",
    "code_execution_started": "CTX",
    "code_turn_feedback_written": "CTX",
    "code_execution_finished": "VERIFY",
    "side_effect_client_locked": "CTX",
    "side_effect_client_observed": "CTX",
}


class ScienceWorldTextPrimitives:
    def __init__(
        self,
        state_machine: ScienceWorldTextStateMachine,
        trace: TraceWriter,
        task: TaskRecord,
        suite: str,
    ):
        self.state_machine = state_machine
        self.trace = trace
        self.task = task
        self.suite = suite
        self.evidence: dict[str, Any] = {}
        self.failed_actions: dict[str, dict[str, Any]] = {}
        self.recent_actions: list[dict[str, Any]] = []
        self.failure_event_counter = 0
        self.repeated_failure_limit = _repeated_failure_limit()
        self.metrics = {
            "env_steps": 0,
            "invalid_action_count": 0,
            "parser_no_match_count": 0,
            "repeated_failure_fuse_count": 0,
        }

    def get_task_context(self) -> dict[str, Any]:
        context = {
            "benchmark": "ScienceWorld",
            "track": "text",
            "benchmark_suite": self.suite,
            "task_id": self.task.task_id,
            "task_name": self.task.task_name,
            "task_type": self.task.task_type,
            "variation_idx": self.task.variation_idx,
            "simplification": self.task.simplification,
            "goal_text": self.state_machine.task_description or self.task.goal_text,
            "max_steps": self.state_machine.max_steps,
        }
        self._record_observation("get_task_context", {"result": context})
        return context

    def observe_text_world(self) -> str:
        observation = self.state_machine.backend.observe_text_world()
        self._record_observation("observe_text_world", {"observation": observation})
        return observation

    def list_actions(self) -> list[str]:
        actions = self.state_machine.backend.list_actions()
        self._record_observation("list_actions", {"valid_actions": _trace_safe_actions(actions)})
        return actions

    def inspect_current_state(self, query: Any = None, limit: int = 80) -> dict[str, Any]:
        observation = self.state_machine.backend.observe_text_world()
        room_description = self.state_machine.backend.look()
        inventory = self.state_machine.backend.inventory()
        actions = self.state_machine.backend.list_actions()
        bounded_limit = max(1, min(int(limit), 200))
        query_terms = _normalize_terms(query)
        substance_query_terms = query_terms or _goal_query_terms(self.state_machine.task_description or self.task.goal_text)
        current_location = (
            _extract_current_location(observation)
            or _extract_current_location(room_description)
            or _last_known_location(self.recent_actions)
        )
        room_text = room_description if current_location else observation
        visible_entries = _extract_visible_entries(room_text, limit=60)
        result = {
            "current_location": current_location,
            "visible_entries": visible_entries,
            "exits_or_doors": _extract_exits_or_doors(room_text, limit=40),
            "inventory_entries": _extract_inventory_entries(inventory, limit=40),
            "query_terms": query_terms,
            "query_actions": _query_actions(actions, query_terms, bounded_limit),
            "substance_query_terms": substance_query_terms,
            "substance_candidates": _substance_candidates(visible_entries, actions, substance_query_terms, bounded_limit),
            "action_groups": _action_groups(actions, bounded_limit),
            "recent_actions": _recent_action_summaries(self.recent_actions, limit=10),
            "loop_warnings": _loop_warnings(self.recent_actions, current_location),
            "valid_action_count": len(actions),
            "boundary": (
                "Observed-only summary from the current text observation, inventory text, "
                "current grounded valid actions, and this wrapper's own recent action trace. "
                "No hidden solution, future state, or gold path."
            ),
        }
        self._record_observation(
            "inspect_current_state",
            {
                "query": query_terms,
                "result": _trace_safe_value(result),
            },
        )
        return result

    def filter_actions(
        self,
        include: Any = None,
        exclude: Any = None,
        startswith: str | None = None,
        limit: int = 80,
        exclude_failed: bool = True,
        incl: Any = None,
        contains: Any = None,
    ) -> list[str]:
        actions = self.state_machine.backend.list_actions()
        include = include if include is not None else (incl if incl is not None else contains)
        candidate_actions = _candidate_matching_actions(include)
        current_action_set = set(actions)
        actions_to_scan = [action for action in candidate_actions if action in current_action_set] if candidate_actions else actions
        include_terms = [] if candidate_actions else _normalize_terms(include)
        exclude_terms = _normalize_terms(exclude)
        prefix = str(startswith or "").lower()
        matches: list[str] = []
        excluded_failed = 0
        for action in actions_to_scan:
            lowered = action.lower()
            if exclude_failed and _action_key(action) in self.failed_actions:
                excluded_failed += 1
                continue
            if prefix and not lowered.startswith(prefix):
                continue
            if include_terms and not all(term in lowered for term in include_terms):
                continue
            if exclude_terms and any(term in lowered for term in exclude_terms):
                continue
            matches.append(action)
        bounded_limit = max(1, min(int(limit), 200))
        result = matches[:bounded_limit]
        self._record_observation(
            "filter_actions",
            {
                "query": {
                    "include": include_terms,
                    "candidate_action_allowlist_count": len(candidate_actions),
                    "exclude": exclude_terms,
                    "startswith": startswith,
                    "limit": bounded_limit,
                    "exclude_failed": bool(exclude_failed),
                },
                "valid_actions": _trace_safe_actions(result),
                "total_matches": len(matches),
                "excluded_failed_count": excluded_failed,
            },
        )
        return result

    def list_recent_failures(self, limit: int = 20) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 100))
        failures = sorted(
            self.failed_actions.values(),
            key=lambda item: int(item.get("last_seen_order") or 0),
            reverse=True,
        )[:bounded_limit]
        result = [dict(item) for item in failures]
        self._record_observation(
            "list_recent_failures",
            {
                "limit": bounded_limit,
                "failure_count": len(self.failed_actions),
                "failures": result,
            },
        )
        return result

    def get_score_state(self) -> dict[str, Any]:
        verification = self.state_machine.backend.check_success().to_dict()
        result = {
            "verification": verification,
            "metrics": dict(self.metrics),
            "failed_action_count": len(self.failed_actions),
            "state": self.state_machine.state.value,
        }
        self._record_observation("get_score_state", {"result": result})
        return result

    def step_text_action(self, action: str) -> StepResult:
        before_verification = self.state_machine.backend.check_success()
        before_score = before_verification.score
        before_location = _extract_current_location(self.state_machine.backend.observe_text_world()) or _last_known_location(
            self.recent_actions
        )
        failure = self.failed_actions.get(_action_key(action))
        if failure and int(failure.get("count") or 0) >= self.repeated_failure_limit:
            result = invalid_step_result(
                action=action,
                observation=self.state_machine.backend.observe_text_world(),
                valid_actions=self.state_machine.backend.list_actions(),
                verification=self.state_machine.backend.check_success(),
                error={
                    "kind": "repeated_failed_action",
                    "message": (
                        "This action already failed repeatedly in the current task. "
                        "Call list_recent_failures() and filter_actions(exclude_failed=True) before retrying."
                    ),
                    "previous_failure": dict(failure),
                },
            )
            self.metrics["repeated_failure_fuse_count"] += 1
            self._remember_recent_action(action, result, before_score=before_score, before_location=before_location)
            self._record_step("step_text_action", {"action": action}, result, native_action=None)
            return result

        result = self.state_machine.execute_action(action)
        self._remember_recent_action(action, result, before_score=before_score, before_location=before_location)
        if result.valid_action:
            self.metrics["env_steps"] += 1
            self._forget_failed_action(action)
        elif result.error and result.error.get("kind") == "invalid_action":
            self.metrics["invalid_action_count"] += 1
            self._remember_failed_action(action, result)
        elif result.error and result.error.get("kind") == "parser_no_match":
            self.metrics["parser_no_match_count"] += 1
            self._remember_failed_action(action, result)
        elif result.error:
            self._remember_failed_action(action, result)
        self._record_step("step_text_action", {"action": action}, result, native_action=action)
        return result

    def look(self) -> str:
        result = self.state_machine.backend.look()
        self._record_observation("look", {"result": result})
        return result

    def inventory(self) -> str:
        result = self.state_machine.backend.inventory()
        self._record_observation("inventory", {"result": result})
        return result

    def check_success(self) -> dict[str, Any]:
        verification = self.state_machine.backend.check_success()
        result = verification.to_dict()
        self._record_observation("check_success", {"verification": result})
        return result

    def write_evidence(self, key: str, value: Any) -> dict[str, Any]:
        self.evidence[key] = value
        result = {"key": key, "value": value}
        self._record_observation(
            "write_evidence",
            {
                "result": {"key": key, "value": _trace_safe_value(value)},
                "evidence": _trace_safe_evidence(self.evidence),
            },
        )
        return result

    def read_evidence(self) -> dict[str, Any]:
        evidence = dict(self.evidence)
        self._record_observation("read_evidence", {"evidence": _trace_safe_evidence(evidence)})
        return evidence

    def record_harness_event(self, event_name: str, payload: dict[str, Any], side_effect: bool = False) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": event_name,
                "canonical_family": FAMILIES[event_name],
                "state": self.state_machine.state.value,
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
                "state": self.state_machine.state.value,
                "side_effect": False,
                **payload,
            },
        )

    def _record_step(
        self,
        primitive: str,
        arguments: dict[str, Any],
        result: StepResult,
        native_action: str | None,
    ) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": primitive,
                "canonical_family": FAMILIES[primitive],
                "state": self.state_machine.state.value,
                "side_effect": True,
                "arguments": arguments,
                "native_action": native_action,
                "valid_action": result.valid_action,
                "observation_before": result.observation_before,
                "observation_after": result.observation_after,
                "valid_actions_before": _trace_safe_actions(result.valid_actions_before),
                "valid_actions_after": _trace_safe_actions(result.valid_actions_after),
                "score": result.score,
                "done": result.done,
                "success": result.success,
                "error": result.error,
                "evidence": _trace_safe_evidence(self.evidence),
            },
        )

    def _remember_failed_action(self, action: str, result: StepResult) -> None:
        key = _action_key(action)
        self.failure_event_counter += 1
        previous = self.failed_actions.get(key, {})
        self.failed_actions[key] = {
            "action": str(action),
            "count": int(previous.get("count") or 0) + 1,
            "last_error": result.error,
            "last_observation": _trace_safe_value(result.observation_after),
            "last_seen_order": self.failure_event_counter,
        }

    def _forget_failed_action(self, action: str) -> None:
        self.failed_actions.pop(_action_key(action), None)

    def _remember_recent_action(
        self,
        action: str,
        result: StepResult,
        *,
        before_score: float,
        before_location: str | None,
    ) -> None:
        score_after = float(result.score)
        self.recent_actions.append(
            {
                "action": str(action),
                "valid_action": bool(result.valid_action),
                "score_before": float(before_score),
                "score_after": score_after,
                "score_delta": round(score_after - float(before_score), 4),
                "location_before": before_location,
                "location_after": _extract_current_location(result.observation_after)
                or (_navigation_target(str(action)) if _is_navigation_action(str(action)) else before_location),
                "done": bool(result.done),
                "success": bool(result.success),
                "error_kind": (result.error or {}).get("kind") if result.error else None,
            }
        )
        self.recent_actions = self.recent_actions[-20:]


def _trace_safe_evidence(evidence: dict[str, Any]) -> dict[str, Any]:
    items = list(evidence.items())
    max_items = _trace_max_items()
    result: dict[str, Any] = {
        "_trace_evidence_key_count": len(items),
        "_trace_evidence_keys": [str(key) for key, _value in items[:max_items]],
    }
    for key, value in items[:max_items]:
        result[str(key)] = _trace_evidence_leaf(value)
    if len(items) > max_items:
        result["_trace_omitted_keys"] = len(items) - max_items
    return result


def _normalize_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.lower()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).lower() for item in value if str(item).strip()]
    return [str(value).lower()]


def _candidate_matching_actions(value: Any) -> list[str]:
    actions: list[str] = []
    if isinstance(value, dict):
        for action in value.get("matching_actions") or []:
            if isinstance(action, str):
                actions.append(action)
        return actions
    if isinstance(value, (list, tuple, set)):
        for item in value:
            actions.extend(_candidate_matching_actions(item))
    return list(dict.fromkeys(actions))


def _goal_query_terms(goal_text: str) -> list[str]:
    goal = str(goal_text or "").lower()
    terms: list[str] = []
    for term in (
        "water",
        "ice",
        "unknown substance",
        "substance b",
        "sodium chloride",
        "soap",
        "wood",
        "red paint",
        "blue paint",
        "yellow paint",
        "paint",
    ):
        if term in goal:
            terms.append(term)
    return terms


def _extract_current_location(observation: str) -> str | None:
    text = str(observation or "")
    for pattern in (
        r"This room is called the ([^.]+)\.",
        r"This outside location is called the ([^.]+)\.",
        r"This location is called the ([^.]+)\.",
        r"You (?:teleport|move|go|travel|walk) to the ([^.]+)\.",
        r"You (?:teleport|move|go|travel|walk) to ([^.]+)\.",
        r"You are in the ([^.]+)\.",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _normalize_location_name(match.group(1))
    return None


def _extract_visible_entries(observation: str, limit: int) -> list[str]:
    entries: list[str] = []
    in_visible_block = False
    for raw_line in str(observation or "").splitlines():
        line = raw_line.strip()
        if "you see:" in line.lower() and not line.lower().startswith("you also see"):
            in_visible_block = True
            continue
        if line.lower().startswith("you also see"):
            in_visible_block = False
            continue
        if not in_visible_block or not line:
            continue
        cleaned = line.rstrip(".")
        if cleaned.lower() == "the agent":
            continue
        entries.append(cleaned)
        if len(entries) >= limit:
            break
    return entries


def _extract_exits_or_doors(observation: str, limit: int) -> list[str]:
    entries: list[str] = []
    in_exit_block = False
    for raw_line in str(observation or "").splitlines():
        line = raw_line.strip()
        if line.lower().startswith("you also see"):
            in_exit_block = True
            continue
        if not in_exit_block or not line:
            continue
        entries.append(line.rstrip("."))
        if len(entries) >= limit:
            break
    return entries


def _extract_inventory_entries(inventory: str, limit: int) -> list[str]:
    entries: list[str] = []
    for raw_line in str(inventory or "").splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("in your inventory"):
            continue
        entries.append(line.rstrip("."))
        if len(entries) >= limit:
            break
    return entries


def _query_actions(actions: list[str], query_terms: list[str], limit: int) -> list[str]:
    if not query_terms:
        return []
    matches = [
        action
        for action in actions
        if any(term in action.lower() for term in query_terms)
    ]
    return matches[:limit]


def _substance_candidates(
    visible_entries: list[str],
    actions: list[str],
    query_terms: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for entry in visible_entries:
        lowered_entry = entry.lower()
        for substance_name in _substance_names(lowered_entry):
            if query_terms and not any(term in substance_name or substance_name in term for term in query_terms):
                continue
            container = _substance_container(lowered_entry, substance_name)
            action_terms = [substance_name]
            if container:
                action_terms.extend([container, f"substance in {container}"])
            matches = _matching_substance_actions(actions, action_terms, limit)
            candidates.append(
                {
                    "substance": substance_name,
                    "container": container,
                    "visible_entry": entry,
                    "matching_actions": matches,
                }
            )
            if len(candidates) >= 20:
                return candidates
    return candidates


def _substance_names(entry: str) -> list[str]:
    names: list[str] = []
    for match in re.finditer(r"substance called ([a-z0-9 _-]+)", entry, flags=re.IGNORECASE):
        name = re.split(r"[,.)]", match.group(1), maxsplit=1)[0].strip()
        if name and name not in names:
            names.append(name)
    return names


def _substance_container(entry: str, substance_name: str) -> str | None:
    escaped = re.escape(substance_name)
    patterns = (
        rf"in the ([^:()]+) is:.*substance called {escaped}",
        rf"(?:a|an|the) ([^()]+) \(containing .*substance called {escaped}",
    )
    for pattern in patterns:
        match = re.search(pattern, entry, flags=re.IGNORECASE)
        if match:
            return _normalize_object_name(match.group(1))
    return None


def _matching_substance_actions(actions: list[str], action_terms: list[str], limit: int) -> list[str]:
    normalized_terms = [_normalize_object_name(term) for term in action_terms if term]
    term_tiers = [
        [term for term in normalized_terms if term.startswith("substance in ")],
        [term for term in normalized_terms if not term.startswith("substance in ") and term.startswith("substance")],
        [term for term in normalized_terms if not term.startswith("substance")],
    ]
    prioritized: list[str] = []
    for prefix in ("focus on ", "look at ", "move ", "pour ", "dunk ", "mix ", "use "):
        for terms in term_tiers:
            for action in actions:
                lowered = action.lower()
                if not lowered.startswith(prefix):
                    continue
                if any(term and term in lowered for term in terms):
                    prioritized.append(action)
    for terms in term_tiers:
        for action in actions:
            lowered = action.lower()
            if any(term and term in lowered for term in terms):
                prioritized.append(action)
    return list(dict.fromkeys(prioritized))[:limit]


def _action_groups(actions: list[str], limit: int) -> dict[str, list[str]]:
    groups = {
        "focus": _actions_with_prefix(actions, "focus on ", limit),
        "focus_objects": _focus_object_actions(actions, limit),
        "go": _actions_with_prefix(actions, "go to ", limit),
        "teleport": _actions_with_prefix(actions, "teleport to ", limit),
        "move": _actions_with_prefix(actions, "move ", limit),
        "pick_up": _actions_with_prefix(actions, "pick up ", limit),
        "put": _actions_with_prefix(actions, "put ", limit),
        "activate": _actions_with_prefix(actions, "activate ", limit),
        "deactivate": _actions_with_prefix(actions, "deactivate ", limit),
        "open": _actions_with_prefix(actions, "open ", limit),
        "close": _actions_with_prefix(actions, "close ", limit),
        "dunk": _actions_with_prefix(actions, "dunk ", limit),
        "pour": _actions_with_prefix(actions, "pour ", limit),
        "mix": _actions_with_prefix(actions, "mix ", limit),
        "use": _actions_with_prefix(actions, "use ", limit),
    }
    groups["other_sample"] = [
        action
        for action in actions
        if not any(action.startswith(prefix) for prefix in _ACTION_GROUP_PREFIXES)
    ][:limit]
    return {name: values for name, values in groups.items() if values}


def _recent_action_summaries(recent_actions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return [dict(item) for item in recent_actions[-max(1, int(limit)):]]


def _loop_warnings(recent_actions: list[dict[str, Any]], current_location: str | None) -> list[str]:
    warnings: list[str] = []
    if not recent_actions:
        return warnings

    last = recent_actions[-1]
    last_action = str(last.get("action") or "")
    if last.get("done"):
        warnings.append("The last action ended the episode; stop acting and only call check_success().")
    if _is_navigation_action(last_action):
        location_before = last.get("location_before")
        location_after = last.get("location_after")
        if location_after and location_after == location_before and float(last.get("score_delta") or 0.0) <= 0.0:
            warnings.append("The last navigation action did not change room or score; inspect state before retrying navigation.")
        if current_location and _navigation_target(last_action) == current_location.lower():
            warnings.append("The last navigation target is already the current_location; do not repeat that same go/teleport action.")

    repeated_count = 1
    for item in reversed(recent_actions[:-1]):
        if str(item.get("action") or "") != last_action:
            break
        repeated_count += 1
    if repeated_count >= 2:
        warnings.append(f"The same action was repeated {repeated_count} times in a row; choose a different grounded action.")

    no_progress = [
        item
        for item in recent_actions[-5:]
        if bool(item.get("valid_action")) and not bool(item.get("success")) and float(item.get("score_delta") or 0.0) <= 0.0
    ]
    if len(no_progress) >= 4:
        warnings.append("Most recent valid actions did not improve score; switch tactic instead of continuing the same scan.")
    return warnings


def _last_known_location(recent_actions: list[dict[str, Any]]) -> str | None:
    for item in reversed(recent_actions):
        location = item.get("location_after")
        if location:
            return str(location)
    return None


def _actions_with_prefix(actions: list[str], prefix: str, limit: int) -> list[str]:
    return [action for action in actions if action.startswith(prefix)][:limit]


def _focus_object_actions(actions: list[str], limit: int) -> list[str]:
    excluded_targets = {
        "agent",
        "air",
        "inventory",
        "hallway",
        "kitchen",
        "bathroom",
        "bedroom",
        "greenhouse",
        "living room",
        "art studio",
        "workshop",
        "foundry",
        "outside",
    }
    result = []
    for action in actions:
        if not action.startswith("focus on "):
            continue
        target = action[len("focus on ") :].strip().lower()
        if target in excluded_targets or "door" in target:
            continue
        result.append(action)
        if len(result) >= limit:
            break
    return result


def _is_navigation_action(action: str) -> bool:
    return action.startswith("go to ") or action.startswith("teleport to ")


def _navigation_target(action: str) -> str | None:
    lowered = action.lower()
    for prefix in ("go to ", "teleport to "):
        if lowered.startswith(prefix):
            return _normalize_location_name(lowered[len(prefix):])
    return None


def _normalize_location_name(value: str) -> str:
    lowered = " ".join(str(value).strip().lower().split())
    for prefix in ("the ",):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
    return lowered


def _normalize_object_name(value: str) -> str:
    lowered = " ".join(str(value).strip().lower().split())
    for prefix in ("a ", "an ", "the "):
        if lowered.startswith(prefix):
            lowered = lowered[len(prefix):]
    return lowered.rstrip(".,:;")


_ACTION_GROUP_PREFIXES = (
    "focus on ",
    "go to ",
    "teleport to ",
    "move ",
    "pick up ",
    "put ",
    "activate ",
    "deactivate ",
    "open ",
    "close ",
    "dunk ",
    "pour ",
    "mix ",
    "use ",
)


def _action_key(action: Any) -> str:
    return " ".join(str(action).lower().split())


def _trace_evidence_leaf(value: Any) -> dict[str, Any]:
    try:
        text = repr(_trace_safe_value(value))
    except Exception:
        text = repr(value)
    max_chars = _trace_max_chars()
    return {
        "type": type(value).__name__,
        "chars": len(text),
        "preview": text[:max_chars],
        "truncated": len(text) > max_chars,
    }


def _trace_safe_actions(actions: list[str]) -> dict[str, Any]:
    max_items = _trace_max_action_items()
    return {
        "count": len(actions),
        "sample": actions[:max_items],
        "omitted": max(0, len(actions) - max_items),
    }


def _trace_safe_value(value: Any, depth: int = 0) -> Any:
    max_chars = _trace_max_chars()
    max_items = _trace_max_items()
    if value is None or isinstance(value, bool | int | float):
        return value
    if isinstance(value, str):
        return _truncate_string(value, max_chars)
    if depth >= 4:
        return _truncate_string(repr(value), max_chars)
    if isinstance(value, dict):
        items = list(value.items())
        result: dict[str, Any] = {}
        for key, item in items[:max_items]:
            result[str(key)] = _trace_safe_value(item, depth + 1)
        if len(items) > max_items:
            result["_trace_omitted_keys"] = len(items) - max_items
        return result
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        result = [_trace_safe_value(item, depth + 1) for item in items[:max_items]]
        if len(items) > max_items:
            result.append({"_trace_omitted_items": len(items) - max_items})
        return result
    return _truncate_string(repr(value), max_chars)


def _truncate_string(value: str, max_chars: int) -> Any:
    if len(value) <= max_chars:
        return value
    return {
        "_trace_truncated": True,
        "type": "str",
        "chars": len(value),
        "preview": value[:max_chars],
    }


def _trace_max_chars() -> int:
    raw = os.environ.get("SCIENCEWORLD_TRACE_EVIDENCE_MAX_CHARS", "500").strip()
    try:
        return max(100, int(raw))
    except ValueError:
        return 500


def _trace_max_items() -> int:
    raw = os.environ.get("SCIENCEWORLD_TRACE_EVIDENCE_MAX_ITEMS", "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def _trace_max_action_items() -> int:
    raw = os.environ.get("SCIENCEWORLD_TRACE_ACTION_SAMPLE_SIZE", "25").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 25


def _repeated_failure_limit() -> int:
    raw = os.environ.get("SCIENCEWORLD_REPEATED_FAILURE_LIMIT", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2
