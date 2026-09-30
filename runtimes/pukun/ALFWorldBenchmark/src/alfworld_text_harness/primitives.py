from __future__ import annotations

from typing import Any

from .backend import StepResult
from .manifest import TaskRecord
from .state_machine import ALFWorldTextStateMachine
from .trace import TraceWriter

FAMILIES = {
    "get_task_context": "CTX",
    "observe_text_state": "CTX",
    "list_actions": "CTX",
    "match_actions": "CTX",
    "step_text_action": "HACT",
    "examine_object": "PER",
    "look": "NAV",
    "inventory": "STATE",
    "go_to": "NAV",
    "open_object": "HACT",
    "close_object": "HACT",
    "pickup_object": "HACT",
    "place_object": "HACT",
    "toggle_object": "HACT",
    "clean_object": "HACT",
    "heat_object": "HACT",
    "cool_object": "HACT",
    "check_success": "VERIFY",
    "write_evidence": "EVD",
    "read_evidence": "EVD",
    "agent_code_generated": "CTX",
    "code_execution_started": "CTX",
    "code_execution_finished": "VERIFY",
    "code_turn_feedback_written": "CTX",
    "side_effect_client_observed": "CTX",
    "side_effect_client_locked": "CTX",
}


class ALFWorldTextPrimitives:
    def __init__(
        self,
        state_machine: ALFWorldTextStateMachine,
        trace: TraceWriter,
        task: TaskRecord,
        suite: str,
    ):
        self.state_machine = state_machine
        self.trace = trace
        self.task = task
        self.suite = suite
        self.evidence: dict[str, Any] = {}
        self.metrics = {
            "env_steps": 0,
            "invalid_action_count": 0,
            "wrapper_no_match_count": 0,
            "wrapper_ambiguity_count": 0,
        }

    def get_task_context(self) -> dict[str, Any]:
        context = {
            "benchmark": "ALFWorld",
            "track": "text",
            "benchmark_suite": self.suite,
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "goal_text": self.task.goal_text,
            "max_steps": self.state_machine.max_steps,
            "max_env_steps": self.state_machine.max_steps,
            "step_budget": self.state_machine.max_steps,
            "env_steps_used": self.state_machine.step_count,
            "remaining_steps": max(0, self.state_machine.max_steps - self.state_machine.step_count),
        }
        self._record_observation("get_task_context", {"result": context})
        return context

    def observe_text_state(self) -> str:
        observation = self.state_machine.backend.observe_text_state()
        self._record_observation("observe_text_state", {"observation": observation})
        return observation

    def list_actions(self) -> list[str]:
        actions = self.state_machine.backend.list_actions()
        self._record_observation("list_actions", {"admissible_actions": actions})
        return actions

    def match_actions(
        self,
        intent: str | None = None,
        object_name: str | None = None,
        receptacle_name: str | None = None,
        include: Any = None,
        limit: int = 20,
    ) -> dict[str, Any]:
        actions = self.state_machine.backend.list_actions()
        prefixes = _intent_prefixes(intent)
        include_terms = _normalize_terms(include)
        matches: list[dict[str, Any]] = []
        for action in actions:
            normalized = _norm(action)
            score = 0
            reasons: list[str] = []
            if prefixes:
                if not normalized.startswith(prefixes):
                    continue
                score += 3
                reasons.append("intent_prefix")
            if object_name:
                if not _contains(normalized, object_name):
                    continue
                score += 2
                reasons.append("object_name")
            if receptacle_name:
                if not _contains(normalized, receptacle_name):
                    continue
                score += 2
                reasons.append("receptacle_name")
            if include_terms:
                if not all(term in normalized for term in include_terms):
                    continue
                score += len(include_terms)
                reasons.append("include_terms")
            matches.append({"action": action, "score": score, "reasons": reasons})
        matches.sort(key=lambda item: (-int(item["score"]), str(item["action"])))
        bounded_limit = max(1, min(int(limit or 20), 100))
        result = {
            "query": {
                "intent": intent,
                "object_name": object_name,
                "receptacle_name": receptacle_name,
                "include": include_terms,
                "limit": bounded_limit,
            },
            "total_matches": len(matches),
            "matches": matches[:bounded_limit],
            "note": "Copy one returned action exactly into step_text_action(action), or call a thin wrapper such as go_to/pickup_object.",
        }
        self._record_observation("match_actions", {"result": result})
        return result

    def step_text_action(self, action: str) -> StepResult:
        return self._execute_native("step_text_action", action, {"action": action})

    def examine_object(self, name: str) -> StepResult:
        return self._wrapper("examine_object", {"name": name}, lambda a: a.startswith("examine ") and _contains(a, name))

    def look(self) -> StepResult:
        return self._execute_native("look", "look", {})

    def inventory(self) -> StepResult:
        return self._execute_native("inventory", "inventory", {})

    def go_to(self, name: str) -> StepResult:
        return self._wrapper("go_to", {"name": name}, lambda a: a.startswith("go to ") and _contains(a, name))

    def open_object(self, name: str) -> StepResult:
        return self._wrapper("open_object", {"name": name}, lambda a: a.startswith("open ") and _contains(a, name))

    def close_object(self, name: str) -> StepResult:
        return self._wrapper("close_object", {"name": name}, lambda a: a.startswith("close ") and _contains(a, name))

    def pickup_object(self, name: str) -> StepResult:
        return self._wrapper("pickup_object", {"name": name}, lambda a: a.startswith("take ") and _contains(a, name))

    def place_object(self, obj: str, receptacle: str) -> StepResult:
        prefixes = ("put ", "move ", "place ")
        return self._wrapper(
            "place_object",
            {"obj": obj, "receptacle": receptacle},
            lambda a: a.startswith(prefixes) and _contains(a, obj) and _contains(a, receptacle),
        )

    def toggle_object(self, name: str) -> StepResult:
        prefixes = ("use ", "toggle ", "turn on ", "turn off ")
        return self._wrapper(
            "toggle_object",
            {"name": name},
            lambda a: a.startswith(prefixes) and _contains(a, name),
        )

    def clean_object(self, obj: str) -> StepResult:
        return self._wrapper(
            "clean_object",
            {"obj": obj},
            lambda a: (a.startswith("clean ") or a.startswith("wash ")) and _contains(a, obj),
        )

    def heat_object(self, obj: str) -> StepResult:
        return self._wrapper("heat_object", {"obj": obj}, lambda a: a.startswith("heat ") and _contains(a, obj))

    def cool_object(self, obj: str) -> StepResult:
        return self._wrapper("cool_object", {"obj": obj}, lambda a: a.startswith("cool ") and _contains(a, obj))

    def check_success(self) -> dict[str, Any]:
        verification = self.state_machine.backend.check_success()
        result = verification.to_dict()
        self._record_observation("check_success", {"verification": result})
        return result

    def write_evidence(self, key: str, value: Any) -> dict[str, Any]:
        self.evidence[key] = value
        result = {"key": key, "value": value}
        self._record_observation("write_evidence", {"result": result, "evidence": dict(self.evidence)})
        return result

    def read_evidence(self) -> dict[str, Any]:
        evidence = dict(self.evidence)
        self._record_observation("read_evidence", {"evidence": evidence})
        return evidence

    def record_harness_event(self, event_name: str, payload: dict[str, Any], side_effect: bool = False) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": event_name,
                "canonical_family": FAMILIES.get(event_name, "CTX"),
                "state": self.state_machine.state.value,
                "side_effect": side_effect,
                **payload,
            },
        )

    def _wrapper(self, primitive: str, arguments: dict[str, Any], predicate) -> StepResult:
        actions = self.state_machine.backend.list_actions()
        matches = [action for action in actions if predicate(_norm(action))]
        if len(matches) == 0:
            self.metrics["wrapper_no_match_count"] += 1
            result = self.state_machine._invalid(None, "no_match", "No admissible command matched wrapper.", [])
            self._record_step(primitive, arguments, result, native_action=None, candidates=[])
            return result
        if len(matches) > 1:
            self.metrics["wrapper_ambiguity_count"] += 1
            result = self.state_machine._invalid(None, "ambiguous", "Multiple admissible commands matched wrapper.", matches)
            self._record_step(primitive, arguments, result, native_action=None, candidates=matches)
            return result
        return self._execute_native(primitive, matches[0], arguments)

    def _execute_native(self, primitive: str, native_action: str, arguments: dict[str, Any]) -> StepResult:
        result = self.state_machine.execute_action(native_action)
        if result.valid_action:
            self.metrics["env_steps"] += 1
        elif result.error and result.error.get("kind") == "invalid_action":
            self.metrics["invalid_action_count"] += 1
        self._record_step(primitive, arguments, result, native_action=native_action)
        return result

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
        candidates: list[str] | None = None,
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
                "candidates": candidates,
                "observation_before": result.observation_before,
                "observation_after": result.observation_after,
                "admissible_actions_before": result.admissible_actions_before,
                "admissible_actions_after": result.admissible_actions_after,
                "score": result.score,
                "done": result.done,
                "success": result.success,
                "error": result.error,
                "evidence": dict(self.evidence),
            },
        )


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def _contains(action: str, query: str) -> bool:
    action = _norm(action)
    return all(part in action for part in _norm(query).split())


def _normalize_terms(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [_norm(value)]
    if isinstance(value, (list, tuple, set)):
        return [_norm(item) for item in value if str(item).strip()]
    return [_norm(value)]


def _intent_prefixes(intent: str | None) -> tuple[str, ...]:
    key = _norm(intent or "")
    aliases = {
        "go": ("go to ",),
        "go to": ("go to ",),
        "navigate": ("go to ",),
        "examine": ("examine ",),
        "look": ("look",),
        "inventory": ("inventory",),
        "open": ("open ",),
        "close": ("close ",),
        "pickup": ("take ",),
        "pick up": ("take ",),
        "take": ("take ",),
        "place": ("put ", "move ", "place "),
        "put": ("put ", "move ", "place "),
        "toggle": ("use ", "toggle ", "turn on ", "turn off "),
        "use": ("use ", "toggle ", "turn on ", "turn off "),
        "clean": ("clean ", "wash "),
        "wash": ("clean ", "wash "),
        "heat": ("heat ",),
        "cool": ("cool ",),
    }
    return aliases.get(key, ()) if key else ()
