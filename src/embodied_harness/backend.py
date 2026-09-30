from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


class EmbodiedBackend(ABC):
    """Agent-facing backend contract shared by all benchmark adapters."""

    @abstractmethod
    def reset(self, task_id: str, seed: int | None = None, config: dict[str, Any] | None = None) -> TaskSpec:
        raise NotImplementedError

    @abstractmethod
    def observe(self) -> Observation:
        raise NotImplementedError

    @abstractmethod
    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        raise NotImplementedError

    @abstractmethod
    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        raise NotImplementedError

    @abstractmethod
    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        raise NotImplementedError

    @abstractmethod
    def get_trace(self) -> EpisodeTrace:
        raise NotImplementedError

    def record_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        trace = self.get_trace()
        step = len(trace.events)
        trace.add_event(event_type, step, payload or {})
