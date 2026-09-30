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


PRIMITIVE_CARDS = [
    PrimitiveCard(
        "get_task_context",
        "get_task_context()",
        "CTX",
        "Return benchmark metadata, scenario, difficulty, seed, official goal text, and step budget.",
        {},
        "dict",
        "none",
        "L1",
        ["Does not expose scorecard internals, world history, or hidden state."],
        "manifest + current taskProgress description",
    ),
    PrimitiveCard(
        "observe_world",
        "observe_world()",
        "CTX",
        "Return the latest safe JSON observation summary.",
        {},
        "dict",
        "none",
        "L1",
        ["Vision base64 and full world object lists are not returned."],
        "DiscoveryWorldAPI.getAgentObservation",
    ),
    PrimitiveCard(
        "list_known_actions",
        "list_known_actions()",
        "CTX",
        "Return native action descriptions supported by the v1 primitive facade.",
        {},
        "dict",
        "none",
        "L1",
        ["Does not include TELEPORT_TO_OBJECT or first-round Discovery Feed actions."],
        "DiscoveryWorldAPI.listKnownActions",
    ),
    PrimitiveCard(
        "get_action_schema",
        "get_action_schema()",
        "CTX",
        "Return the primitive-level argument schema and native payload shape for exposed DiscoveryWorld actions.",
        {},
        "dict",
        "none",
        "L1",
        ["Schema only; it does not execute or reveal a plan."],
        "RoBench primitive facade schema",
    ),
    PrimitiveCard(
        "list_accessible_objects",
        "list_accessible_objects()",
        "STATE",
        "List currently accessible objects that can be interacted with.",
        {},
        "list[dict]",
        "none",
        "L1",
        ["Object names may be ambiguous; use uuid when multiple objects share a name."],
        "observation.ui.accessibleEnvironmentObjects",
    ),
    PrimitiveCard(
        "list_nearby_objects",
        "list_nearby_objects(query=None, max_distance=None)",
        "STATE",
        "List nearby objects grouped into a flat direction-aware view for navigation and search.",
        {"query": "Optional name/description/uuid substring.", "max_distance": "Optional maximum nearby distance."},
        "list[dict]",
        "none",
        "L1/L2",
        ["Nearby does not mean accessible; call list_accessible_objects() before manipulation."],
        "observation.ui.nearbyObjects",
    ),
    PrimitiveCard(
        "list_inventory",
        "list_inventory()",
        "STATE",
        "List objects currently in inventory.",
        {},
        "list[dict]",
        "none",
        "L1",
        ["Only returns inventory visible to the agent."],
        "observation.ui.inventoryObjects",
    ),
    PrimitiveCard(
        "validate_action_call",
        "validate_action_call(primitive_name, arguments=None, **kwargs)",
        "STATE",
        "Dry-run wrapper argument validation and object resolution before taking a side-effecting action.",
        {
            "primitive_name": "Primitive name such as move_direction, pickup_object, or put_object.",
            "arguments": "Optional dict of arguments.",
            "**kwargs": "Alternative keyword arguments, for example direction='north'.",
        },
        "dict with ok/error/native_action_json/resolved candidates",
        "none",
        "L1/L2",
        [
            "Does not call the simulator or tick time.",
            "A valid wrapper payload can still fail in the native simulator because of dynamic preconditions.",
        ],
        "RoBench primitive facade resolver",
    ),
    PrimitiveCard(
        "list_teleport_locations",
        "list_teleport_locations()",
        "NAV",
        "List official named teleport locations for efficient smoke/navigation.",
        {},
        "dict",
        "none",
        "L3",
        ["Teleport is an allowed v1 navigation helper and must be trace-marked when used."],
        "DiscoveryWorldAPI.listTeleportLocationsDict",
    ),
    PrimitiveCard(
        "move_direction",
        "move_direction(direction)",
        "NAV",
        "Move one tile north, east, south, or west.",
        {"direction": "One of north, east, south, west."},
        "StepResult",
        "one native action + tick",
        "L3",
        ["Does not path-find or retry.", "direction is a string, not an object uuid; use validate_action_call before execution if unsure."],
        "MOVE_DIRECTION",
    ),
    PrimitiveCard(
        "rotate_direction",
        "rotate_direction(direction)",
        "NAV",
        "Rotate to face north, east, south, or west.",
        {"direction": "One of north, east, south, west."},
        "StepResult",
        "one native action + tick",
        "L3",
        ["Does not move."],
        "ROTATE_DIRECTION",
    ),
    PrimitiveCard(
        "teleport_to_location",
        "teleport_to_location(location_name)",
        "NAV",
        "Teleport to one official named location.",
        {"location_name": "Name from list_teleport_locations()."},
        "StepResult",
        "one native action + tick",
        "L3",
        ["No object teleport. Invalid names fail without guessing."],
        "TELEPORT_TO_LOCATION",
    ),
    PrimitiveCard("pickup_object", "pickup_object(obj)", "HACT", "Pick up one accessible object.", {"obj": "uuid or unique visible name."}, "StepResult", "one native action + tick", "L3", ["No search or navigation."], "PICKUP"),
    PrimitiveCard("drop_object", "drop_object(obj)", "HACT", "Drop one inventory object.", {"obj": "uuid or unique inventory name."}, "StepResult", "one native action + tick", "L3", ["Does not choose placement."], "DROP"),
    PrimitiveCard("put_object", "put_object(obj, target)", "HACT", "Put an inventory object in/on another accessible object or give it to an agent.", {"obj": "uuid or unique inventory name.", "target": "uuid or unique accessible target."}, "StepResult", "one native action + tick", "L3", ["No navigation or search."], "PUT"),
    PrimitiveCard("open_object", "open_object(obj)", "HACT", "Open one accessible object.", {"obj": "uuid or unique visible name."}, "StepResult", "one native action + tick", "L3", ["Fails if object is not accessible/openable."], "OPEN"),
    PrimitiveCard("close_object", "close_object(obj)", "HACT", "Close one accessible object.", {"obj": "uuid or unique visible name."}, "StepResult", "one native action + tick", "L3", ["Fails if object is not accessible/closeable."], "CLOSE"),
    PrimitiveCard("activate_object", "activate_object(obj)", "HACT", "Activate one accessible object.", {"obj": "uuid or unique visible name."}, "StepResult", "one native action + tick", "L3", ["No automatic tool selection."], "ACTIVATE"),
    PrimitiveCard("deactivate_object", "deactivate_object(obj)", "HACT", "Deactivate one accessible object.", {"obj": "uuid or unique visible name."}, "StepResult", "one native action + tick", "L3", ["No automatic tool selection."], "DEACTIVATE"),
    PrimitiveCard("use_object", "use_object(obj, target)", "HACT", "Use one object on another object.", {"obj": "uuid or unique visible/inventory name.", "target": "uuid or unique visible/inventory name."}, "StepResult", "one native action + tick", "L3", ["Both args are required by native USE."], "USE"),
    PrimitiveCard("read_object", "read_object(obj)", "STATE", "Read one accessible or inventory object.", {"obj": "uuid or unique name."}, "StepResult", "one native action + tick", "L3", ["Only reads objects the environment allows reading."], "READ"),
    PrimitiveCard("eat_object", "eat_object(obj)", "HACT", "Eat one inventory or accessible object.", {"obj": "uuid or unique name."}, "StepResult", "one native action + tick", "L3", ["No health/goal interpretation."], "EAT"),
    PrimitiveCard("wait", "wait()", "STATE", "Advance the world by one tick without a native action.", {}, "StepResult", "tick only", "L3", ["Useful for moving agents; still consumes budget."], "tick"),
    PrimitiveCard("talk_to", "talk_to(agent)", "DIALOG", "Talk to one accessible agent.", {"agent": "uuid or unique visible agent name."}, "StepResult", "one native action + tick", "L3", ["Dialog choices require choose_dialog_option()."], "TALK"),
    PrimitiveCard("choose_dialog_option", "choose_dialog_option(option_index)", "DIALOG", "Choose a numbered dialog option while in dialog mode.", {"option_index": "1-based integer option index from observation['dialog_box']['dialogOptions']."}, "StepResult with valid_action, error, success, completed, score, verification", "one dialog action + tick", "L3", ["Only valid while observation.dialog_box.is_in_dialog is true."], "chosen_dialog_option_int"),
    PrimitiveCard("write_evidence", "write_evidence(key, value)", "EVD", "Store trace-local evidence.", {"key": "Evidence key.", "value": "JSON-serializable value."}, "dict", "none", "L1", ["Does not alter the DiscoveryWorld environment.", "Evidence keys that look like private/oracle/backend fields are renamed before trace storage."], "harness memory"),
    PrimitiveCard("read_evidence", "read_evidence()", "EVD", "Read evidence written in this task run.", {}, "dict", "none", "L1", ["Only current run evidence."], "harness memory"),
    PrimitiveCard("check_success", "check_success()", "VERIFY", "Return safe native verifier summary.", {}, "dict", "none", "L1 verifier summary", ["Does not expose scoreCard, criticalHypotheses, or associated notes."], "getTaskScorecard + areTasksComplete"),
]


def get_primitive_cards() -> list[dict[str, Any]]:
    return [card.to_dict() for card in PRIMITIVE_CARDS]


def render_primitive_cards_for_prompt() -> str:
    sections: list[str] = []
    for card in PRIMITIVE_CARDS:
        args = ", ".join(f"{name}: {desc}" for name, desc in card.arguments.items()) or "none"
        limits = " ".join(f"- {item}" for item in card.limitations)
        sections.append(
            "\n".join(
                [
                    f"{card.signature}",
                    f"  family: {card.canonical_family}",
                    f"  description: {card.description}",
                    f"  args: {args}",
                    f"  returns: {card.returns}",
                    f"  side_effect: {card.side_effect}",
                    f"  leakage_level: {card.leakage_level}",
                    f"  backend_source: {card.backend_source}",
                    f"  limitations: {limits}",
                ]
            )
        )
    return "\n\n".join(sections)
