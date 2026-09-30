from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRunResult:
    stopped_reason: str
    steps_attempted: int
    metrics: dict[str, Any] = field(default_factory=dict)
