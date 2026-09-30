from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import time
from typing import Any


JsonDict = dict[str, Any]


def _now() -> float:
    return round(time.time(), 6)


@dataclass(slots=True)
class TaskSpec:
    task_id: str
    source: str
    instruction: str
    goal: JsonDict = field(default_factory=dict)
    initial_state: JsonDict = field(default_factory=dict)
    budgets: JsonDict = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    allowed_primitive_levels: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: JsonDict) -> "TaskSpec":
        return cls(
            task_id=str(data["task_id"]),
            source=str(data.get("source", "unknown")),
            instruction=str(data.get("instruction", "")),
            goal=dict(data.get("goal", {})),
            initial_state=dict(data.get("initial_state", {})),
            budgets=dict(data.get("budgets", {})),
            tags=list(data.get("tags", [])),
            allowed_primitive_levels=list(data.get("allowed_primitive_levels", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(slots=True)
class Observation:
    step: int
    data: JsonDict
    artifacts: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class PrimitiveCard:
    name: str
    capability_tags: list[str]
    input_schema: JsonDict = field(default_factory=dict)
    output_schema: JsonDict = field(default_factory=dict)
    preconditions: list[str] = field(default_factory=list)
    side_effects: list[str] = field(default_factory=list)
    cost: JsonDict = field(default_factory=dict)
    failure_modes: list[str] = field(default_factory=list)
    abstraction_level: str = "L1"
    leakage_risk: str = "none"
    description: str = ""

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class PrimitiveResult:
    name: str
    ok: bool
    output: JsonDict = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    error: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)

    def __getitem__(self, key: str) -> Any:
        return self.output[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.output.get(key, default)

    def __contains__(self, key: object) -> bool:
        return key in self.output

    def __getattr__(self, name: str) -> Any:
        if name in self.output:
            return self.output[name]
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")


@dataclass(slots=True)
class VerificationResult:
    ok: bool
    scope: str
    message: str
    metrics: JsonDict = field(default_factory=dict)
    leaked_fields: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class TraceEvent:
    event_type: str
    step: int
    payload: JsonDict = field(default_factory=dict)
    timestamp: float = field(default_factory=_now)

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(slots=True)
class EpisodeTrace:
    task_id: str
    events: list[TraceEvent] = field(default_factory=list)
    artifacts: dict[str, JsonDict] = field(default_factory=dict)
    metrics: JsonDict = field(default_factory=dict)
    final_status: str | None = None

    def add_event(self, event_type: str, step: int, payload: JsonDict | None = None) -> TraceEvent:
        event = TraceEvent(event_type=event_type, step=step, payload=payload or {})
        self.events.append(event)
        return event

    def add_artifact(self, artifact_id: str, payload: JsonDict) -> None:
        self.artifacts[artifact_id] = payload

    def to_dict(self) -> JsonDict:
        return {
            "task_id": self.task_id,
            "events": [event.to_dict() for event in self.events],
            "artifacts": self.artifacts,
            "metrics": self.metrics,
            "final_status": self.final_status,
        }

    def to_jsonl(self) -> str:
        lines = [json.dumps(event.to_dict(), sort_keys=True) for event in self.events]
        footer = {
            "event_type": "trace_footer",
            "task_id": self.task_id,
            "final_status": self.final_status,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
        }
        lines.append(json.dumps(footer, sort_keys=True))
        return "\n".join(lines)
