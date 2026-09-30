from __future__ import annotations

from typing import Any

from .backend import DiscoveryWorldBackend, StepResult, current_objects
from .manifest import TaskRecord
from .trace import TraceWriter

FORBIDDEN_EVIDENCE_KEYS = {
    "DiscoveryWorldAPI",
    "api",
    "backend",
    "criticalHypotheses",
    "criticalQuestions",
    "oracle",
    "oracle_action",
    "oracle_plan",
    "scorecard",
    "worldHistory",
    "world_state",
}

FORBIDDEN_EVIDENCE_SUBSTRINGS = {
    "expert_plan",
    "ground_truth",
    "hidden_state",
    "oracle",
    "scorecard",
}

FAMILIES = {
    "get_task_context": "CTX",
    "observe_world": "CTX",
    "list_known_actions": "CTX",
    "get_action_schema": "CTX",
    "list_accessible_objects": "STATE",
    "list_nearby_objects": "STATE",
    "list_inventory": "STATE",
    "validate_action_call": "STATE",
    "list_teleport_locations": "NAV",
    "move_direction": "NAV",
    "rotate_direction": "NAV",
    "teleport_to_location": "NAV",
    "pickup_object": "HACT",
    "drop_object": "HACT",
    "put_object": "HACT",
    "open_object": "HACT",
    "close_object": "HACT",
    "activate_object": "HACT",
    "deactivate_object": "HACT",
    "use_object": "HACT",
    "read_object": "STATE",
    "eat_object": "HACT",
    "wait": "STATE",
    "talk_to": "DIALOG",
    "choose_dialog_option": "DIALOG",
    "write_evidence": "EVD",
    "read_evidence": "EVD",
    "check_success": "VERIFY",
    "agent_code_generated": "CTX",
    "code_execution_started": "CTX",
    "code_turn_feedback_written": "CTX",
    "code_execution_finished": "VERIFY",
    "side_effect_client_observed": "CTX",
    "side_effect_client_locked": "CTX",
}

EXPOSED_NATIVE_ACTIONS = {
    "PICKUP",
    "DROP",
    "PUT",
    "OPEN",
    "CLOSE",
    "ACTIVATE",
    "DEACTIVATE",
    "TALK",
    "EAT",
    "READ",
    "USE",
    "MOVE_DIRECTION",
    "ROTATE_DIRECTION",
    "TELEPORT_TO_LOCATION",
}

DIRECTIONS = {"north", "east", "south", "west"}


class DiscoveryWorldPrimitives:
    def __init__(
        self,
        backend: DiscoveryWorldBackend,
        trace: TraceWriter,
        task: TaskRecord,
        suite: str,
    ):
        self.backend = backend
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
            "benchmark": "DiscoveryWorld",
            "track": "text_json",
            "benchmark_suite": self.suite,
            "task_id": self.task.task_id,
            "scenario_name": self.task.scenario_name,
            "difficulty": self.task.difficulty,
            "seed": self.task.seed,
            "task_type": self.task.task_type,
            "goal_text": self.task.goal_text,
        }
        self._record_observation("get_task_context", {"result": context})
        return context

    def observe_world(self) -> dict[str, Any]:
        observation = self.backend.observe()
        self._record_observation("observe_world", {"observation": observation})
        return observation

    def list_known_actions(self) -> dict[str, Any]:
        actions = {
            name: spec
            for name, spec in self.backend.list_known_actions(limited=False).items()
            if name in EXPOSED_NATIVE_ACTIONS
        }
        self._record_observation("list_known_actions", {"known_actions": actions})
        return actions

    def get_action_schema(self) -> dict[str, Any]:
        schema = {
            "directions": sorted(DIRECTIONS),
            "object_reference": "Use a uuid from list_accessible_objects/list_inventory, or a unique visible object name.",
            "primitives": {
                "move_direction": {
                    "arguments": {"direction": "north|east|south|west"},
                    "native_action": {"action": "MOVE_DIRECTION", "arg1": "<direction string>"},
                    "side_effect": "one grid move + tick",
                },
                "rotate_direction": {
                    "arguments": {"direction": "north|east|south|west"},
                    "native_action": {"action": "ROTATE_DIRECTION", "arg1": "<direction string>"},
                    "side_effect": "one rotate + tick",
                },
                "teleport_to_location": {
                    "arguments": {"location_name": "exact key from list_teleport_locations()"},
                    "native_action": {"action": "TELEPORT_TO_LOCATION", "arg1": "<location_name>"},
                    "side_effect": "one native teleport + tick",
                },
                "pickup_object": {"arguments": {"obj": "accessible object ref"}, "native_action": "PICKUP"},
                "drop_object": {"arguments": {"obj": "inventory object ref"}, "native_action": "DROP"},
                "put_object": {
                    "arguments": {"obj": "inventory/current object ref", "target": "accessible target ref"},
                    "native_action": "PUT",
                },
                "open_object": {"arguments": {"obj": "accessible object ref"}, "native_action": "OPEN"},
                "close_object": {"arguments": {"obj": "accessible object ref"}, "native_action": "CLOSE"},
                "activate_object": {"arguments": {"obj": "accessible object ref"}, "native_action": "ACTIVATE"},
                "deactivate_object": {"arguments": {"obj": "accessible object ref"}, "native_action": "DEACTIVATE"},
                "use_object": {
                    "arguments": {"obj": "inventory/current object ref", "target": "inventory/current target ref"},
                    "native_action": "USE",
                },
                "read_object": {"arguments": {"obj": "accessible/inventory object ref"}, "native_action": "READ"},
                "eat_object": {"arguments": {"obj": "accessible/inventory object ref"}, "native_action": "EAT"},
                "talk_to": {"arguments": {"agent": "accessible agent ref"}, "native_action": "TALK"},
                "choose_dialog_option": {"arguments": {"option_index": "integer dialog option"}, "native_action": "dialog choice"},
                "wait": {"arguments": {}, "native_action": "tick only"},
            },
            "recommended_before_side_effect": "Call validate_action_call(...) when unsure about argument shape or object ambiguity.",
        }
        self._record_observation("get_action_schema", {"schema": schema})
        return schema

    def list_accessible_objects(self) -> list[dict[str, Any]]:
        objects = list(self.backend.observe().get("accessibleEnvironmentObjects", []))
        self._record_observation("list_accessible_objects", {"objects": objects})
        return objects

    def list_nearby_objects(self, query: str | None = None, max_distance: int | float | None = None) -> list[dict[str, Any]]:
        nearby = (self.backend.observe().get("nearbyObjects") or {}).get("objects") or {}
        results: list[dict[str, Any]] = []
        for direction, objects in nearby.items():
            for obj in objects:
                item = dict(obj)
                item["direction"] = direction
                if max_distance is not None:
                    distance = item.get("distance")
                    try:
                        if float(distance) > float(max_distance):
                            continue
                    except (TypeError, ValueError):
                        continue
                if query and not _matches_object(item, query):
                    continue
                results.append(item)
        self._record_observation(
            "list_nearby_objects",
            {"query": query, "max_distance": max_distance, "objects": results},
        )
        return results

    def list_inventory(self) -> list[dict[str, Any]]:
        objects = list(self.backend.observe().get("inventoryObjects", []))
        self._record_observation("list_inventory", {"objects": objects})
        return objects

    def list_teleport_locations(self) -> dict[str, Any]:
        locations = self.backend.list_teleport_locations()
        self._record_observation("list_teleport_locations", {"locations": locations})
        return locations

    def validate_action_call(
        self,
        primitive_name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        primitive = _norm(str(primitive_name)).replace(" ", "_")
        args = dict(arguments or {})
        args.update(kwargs)
        result = self._validate_action_call(primitive, args)
        self._record_observation(
            "validate_action_call",
            {"primitive_name": primitive_name, "arguments": args, "result": result},
        )
        return result

    def move_direction(self, direction: str) -> StepResult:
        direction = _norm(direction)
        if direction not in DIRECTIONS:
            return self._invalid("move_direction", {"direction": direction}, "invalid_direction", f"Invalid direction: {direction}")
        return self._execute("move_direction", {"action": "MOVE_DIRECTION", "arg1": direction}, {"direction": direction})

    def rotate_direction(self, direction: str) -> StepResult:
        direction = _norm(direction)
        if direction not in DIRECTIONS:
            return self._invalid("rotate_direction", {"direction": direction}, "invalid_direction", f"Invalid direction: {direction}")
        return self._execute("rotate_direction", {"action": "ROTATE_DIRECTION", "arg1": direction}, {"direction": direction})

    def teleport_to_location(self, location_name: str) -> StepResult:
        locations = self.backend.list_teleport_locations()
        if location_name not in locations:
            return self._invalid(
                "teleport_to_location",
                {"location_name": location_name},
                "no_match",
                "No exact teleport location matched.",
                candidates=sorted(locations),
            )
        return self._execute(
            "teleport_to_location",
            {"action": "TELEPORT_TO_LOCATION", "arg1": location_name},
            {"location_name": location_name},
        )

    def pickup_object(self, obj: Any) -> StepResult:
        return self._object_action("pickup_object", "PICKUP", obj, scopes={"accessibleEnvironmentObjects"})

    def drop_object(self, obj: Any) -> StepResult:
        return self._object_action("drop_object", "DROP", obj, scopes={"inventoryObjects"})

    def put_object(self, obj: Any, target: Any) -> StepResult:
        return self._two_object_action(
            "put_object",
            "PUT",
            obj,
            target,
            scopes1={"inventoryObjects", "accessibleEnvironmentObjects"},
            scopes2={"accessibleEnvironmentObjects"},
        )

    def open_object(self, obj: Any) -> StepResult:
        return self._object_action("open_object", "OPEN", obj, scopes={"accessibleEnvironmentObjects"})

    def close_object(self, obj: Any) -> StepResult:
        return self._object_action("close_object", "CLOSE", obj, scopes={"accessibleEnvironmentObjects"})

    def activate_object(self, obj: Any) -> StepResult:
        return self._object_action("activate_object", "ACTIVATE", obj, scopes={"accessibleEnvironmentObjects"})

    def deactivate_object(self, obj: Any) -> StepResult:
        return self._object_action("deactivate_object", "DEACTIVATE", obj, scopes={"accessibleEnvironmentObjects"})

    def use_object(self, obj: Any, target: Any) -> StepResult:
        return self._two_object_action(
            "use_object",
            "USE",
            obj,
            target,
            scopes1={"inventoryObjects", "accessibleEnvironmentObjects"},
            scopes2={"inventoryObjects", "accessibleEnvironmentObjects"},
        )

    def read_object(self, obj: Any) -> StepResult:
        return self._object_action(
            "read_object",
            "READ",
            obj,
            scopes={"accessibleEnvironmentObjects", "inventoryObjects"},
        )

    def eat_object(self, obj: Any) -> StepResult:
        return self._object_action(
            "eat_object",
            "EAT",
            obj,
            scopes={"accessibleEnvironmentObjects", "inventoryObjects"},
        )

    def wait(self) -> StepResult:
        result = self.backend.tick()
        self.metrics["env_steps"] += 1
        if not result.valid_action:
            self.metrics["invalid_action_count"] += 1
        self._record_step("wait", {}, result, native_action_json=None)
        return result

    def talk_to(self, agent: Any) -> StepResult:
        return self._object_action("talk_to", "TALK", agent, scopes={"accessibleEnvironmentObjects"})

    def choose_dialog_option(self, option_index: int) -> StepResult:
        try:
            option = int(option_index)
        except Exception:
            return self._invalid(
                "choose_dialog_option",
                {"option_index": option_index},
                "invalid_option",
                "Dialog option must be an integer.",
            )
        return self._execute(
            "choose_dialog_option",
            {"chosen_dialog_option_int": option},
            {"option_index": option},
        )

    def check_success(self) -> dict[str, Any]:
        result = self.backend.check_success().to_agent_dict()
        self._record_observation("check_success", {"verification": result})
        return result

    def write_evidence(self, key: str, value: Any) -> dict[str, Any]:
        requested_key = str(key)
        safe_key = _safe_evidence_key(requested_key)
        safe_value = _sanitize_evidence_value(value)
        self.evidence[safe_key] = safe_value
        result = {
            "key": safe_key,
            "value": safe_value,
            "key_was_sanitized": safe_key != requested_key,
        }
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
                "canonical_family": FAMILIES[event_name],
                "side_effect": side_effect,
                **payload,
            },
        )

    def _object_action(self, primitive: str, action: str, obj: Any, scopes: set[str]) -> StepResult:
        resolved = self._resolve_object(obj, scopes)
        if "error" in resolved:
            return self._invalid(primitive, {"obj": obj}, resolved["error"], resolved["message"], resolved.get("candidates"))
        return self._execute(primitive, {"action": action, "arg1": resolved["uuid"]}, {"obj": obj})

    def _two_object_action(
        self,
        primitive: str,
        action: str,
        obj: Any,
        target: Any,
        scopes1: set[str],
        scopes2: set[str],
    ) -> StepResult:
        first = self._resolve_object(obj, scopes1)
        if "error" in first:
            return self._invalid(primitive, {"obj": obj, "target": target}, first["error"], first["message"], first.get("candidates"))
        second = self._resolve_object(target, scopes2)
        if "error" in second:
            return self._invalid(primitive, {"obj": obj, "target": target}, second["error"], second["message"], second.get("candidates"))
        return self._execute(
            primitive,
            {"action": action, "arg1": first["uuid"], "arg2": second["uuid"]},
            {"obj": obj, "target": target},
        )

    def _resolve_object(self, ref: Any, scopes: set[str]) -> dict[str, Any]:
        objects = [obj for obj in current_objects(self.backend.observe()) if obj.get("scope") in scopes]
        if isinstance(ref, int) or (isinstance(ref, str) and ref.strip().isdigit()):
            uuid = int(ref)
            matches = [obj for obj in objects if obj.get("uuid") == uuid]
            if len(matches) == 1:
                return {"uuid": uuid, "object": matches[0]}
            return {"error": "no_match", "message": f"No current object with uuid {uuid}.", "candidates": objects}

        query = _norm(str(ref))
        exact = [
            obj
            for obj in objects
            if _norm(str(obj.get("name", ""))) == query or _norm(str(obj.get("description", ""))) == query
        ]
        candidates = exact or [
            obj
            for obj in objects
            if query in _norm(str(obj.get("name", ""))) or query in _norm(str(obj.get("description", "")))
        ]
        unique_by_uuid = {obj.get("uuid"): obj for obj in candidates if obj.get("uuid") is not None}
        if len(unique_by_uuid) == 1:
            obj = next(iter(unique_by_uuid.values()))
            return {"uuid": obj["uuid"], "object": obj}
        if len(unique_by_uuid) > 1:
            return {"error": "ambiguous", "message": f"Multiple objects matched '{ref}'.", "candidates": list(unique_by_uuid.values())}
        return {"error": "no_match", "message": f"No current object matched '{ref}'.", "candidates": objects}

    def _execute(self, primitive: str, action_json: dict[str, Any], arguments: dict[str, Any]) -> StepResult:
        result = self.backend.perform_action(action_json)
        self.metrics["env_steps"] += 1
        if not result.valid_action:
            self.metrics["invalid_action_count"] += 1
        self._record_step(primitive, arguments, result, native_action_json=action_json)
        return result

    def _invalid(
        self,
        primitive: str,
        arguments: dict[str, Any],
        kind: str,
        message: str,
        candidates: Any = None,
    ) -> StepResult:
        if kind == "ambiguous":
            self.metrics["wrapper_ambiguity_count"] += 1
        elif kind == "no_match":
            self.metrics["wrapper_no_match_count"] += 1
        error = {"kind": kind, "message": message, "candidates": candidates}
        result = self.backend.invalid_result(error)
        self._record_step(primitive, arguments, result, native_action_json=None)
        return result

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

    def _validate_action_call(self, primitive: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if primitive in {"move_direction", "rotate_direction"}:
            direction = _norm(str(arguments.get("direction", "")))
            if direction not in DIRECTIONS:
                return {
                    "ok": False,
                    "error": "invalid_direction",
                    "message": "Direction must be one of north, east, south, west.",
                    "expected": {"direction": sorted(DIRECTIONS)},
                    "received": arguments,
                }
            native = "MOVE_DIRECTION" if primitive == "move_direction" else "ROTATE_DIRECTION"
            return {"ok": True, "native_action_json": {"action": native, "arg1": direction}}

        if primitive == "teleport_to_location":
            location = str(arguments.get("location_name", ""))
            locations = self.backend.list_teleport_locations()
            if location not in locations:
                return {
                    "ok": False,
                    "error": "no_match",
                    "message": "No exact teleport location matched.",
                    "candidates": sorted(locations),
                    "received": arguments,
                }
            return {"ok": True, "native_action_json": {"action": "TELEPORT_TO_LOCATION", "arg1": location}}

        one_object_specs = {
            "pickup_object": ("PICKUP", {"accessibleEnvironmentObjects"}, "obj"),
            "drop_object": ("DROP", {"inventoryObjects"}, "obj"),
            "open_object": ("OPEN", {"accessibleEnvironmentObjects"}, "obj"),
            "close_object": ("CLOSE", {"accessibleEnvironmentObjects"}, "obj"),
            "activate_object": ("ACTIVATE", {"accessibleEnvironmentObjects"}, "obj"),
            "deactivate_object": ("DEACTIVATE", {"accessibleEnvironmentObjects"}, "obj"),
            "read_object": ("READ", {"accessibleEnvironmentObjects", "inventoryObjects"}, "obj"),
            "eat_object": ("EAT", {"accessibleEnvironmentObjects", "inventoryObjects"}, "obj"),
            "talk_to": ("TALK", {"accessibleEnvironmentObjects"}, "agent"),
        }
        if primitive in one_object_specs:
            native, scopes, arg_name = one_object_specs[primitive]
            resolved = self._resolve_object(arguments.get(arg_name), scopes)
            if "error" in resolved:
                return {"ok": False, **resolved, "received": arguments}
            return {"ok": True, "native_action_json": {"action": native, "arg1": resolved["uuid"]}, "resolved": resolved["object"]}

        two_object_specs = {
            "put_object": (
                "PUT",
                {"inventoryObjects", "accessibleEnvironmentObjects"},
                {"accessibleEnvironmentObjects"},
            ),
            "use_object": (
                "USE",
                {"inventoryObjects", "accessibleEnvironmentObjects"},
                {"inventoryObjects", "accessibleEnvironmentObjects"},
            ),
        }
        if primitive in two_object_specs:
            native, first_scopes, second_scopes = two_object_specs[primitive]
            first = self._resolve_object(arguments.get("obj"), first_scopes)
            if "error" in first:
                return {"ok": False, "arg": "obj", **first, "received": arguments}
            second = self._resolve_object(arguments.get("target"), second_scopes)
            if "error" in second:
                return {"ok": False, "arg": "target", **second, "received": arguments}
            return {
                "ok": True,
                "native_action_json": {"action": native, "arg1": first["uuid"], "arg2": second["uuid"]},
                "resolved": {"obj": first["object"], "target": second["object"]},
            }

        if primitive == "choose_dialog_option":
            try:
                option = int(arguments.get("option_index"))
            except Exception:
                return {"ok": False, "error": "invalid_option", "message": "option_index must be an integer."}
            return {"ok": True, "native_action_json": {"chosen_dialog_option_int": option}}

        if primitive == "wait":
            return {"ok": True, "native_action_json": None}

        return {
            "ok": False,
            "error": "unknown_primitive",
            "message": f"{primitive!r} is not a DiscoveryWorld action primitive.",
            "available": sorted(get_action_primitive_names()),
        }

    def _record_step(
        self,
        primitive: str,
        arguments: dict[str, Any],
        result: StepResult,
        native_action_json: dict[str, Any] | None,
    ) -> None:
        self.trace.record_event(
            self.task,
            {
                "benchmark_suite": self.suite,
                "primitive": primitive,
                "canonical_family": FAMILIES[primitive],
                "side_effect": True,
                "arguments": arguments,
                "native_action_json": native_action_json,
                "action_response": result.action_response,
                "tick_response": result.tick_response,
                "valid_action": result.valid_action,
                "teleport_used": result.teleport_used,
                "observation_before": result.observation_before,
                "observation_after": result.observation_after,
                "verifier_result": self.backend.verification.to_trace_dict(),
                "success": result.success,
                "completed": result.completed,
                "score": result.score,
                "error": result.error,
                "evidence": dict(self.evidence),
            },
        )


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("_", " ").split())


def _matches_object(obj: dict[str, Any], query: Any) -> bool:
    needle = _norm(str(query))
    return (
        needle in _norm(str(obj.get("uuid", "")))
        or needle in _norm(str(obj.get("name", "")))
        or needle in _norm(str(obj.get("description", "")))
    )


def get_action_primitive_names() -> set[str]:
    return {
        "activate_object",
        "choose_dialog_option",
        "close_object",
        "deactivate_object",
        "drop_object",
        "eat_object",
        "move_direction",
        "open_object",
        "pickup_object",
        "put_object",
        "read_object",
        "rotate_direction",
        "talk_to",
        "teleport_to_location",
        "use_object",
        "wait",
    }


def _safe_evidence_key(key: str) -> str:
    lower = key.lower()
    if key in FORBIDDEN_EVIDENCE_KEYS or any(token in lower for token in FORBIDDEN_EVIDENCE_SUBSTRINGS):
        safe = "agent_note"
        for char in lower:
            safe += char if char.isalnum() else "_"
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_")
    return key


def _sanitize_evidence_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _safe_evidence_key(str(key)): _sanitize_evidence_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_evidence_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_evidence_value(item) for item in value]
    return value
