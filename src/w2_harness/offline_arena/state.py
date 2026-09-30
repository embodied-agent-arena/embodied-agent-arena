"""Lifecycle and durable state helpers for the CPU-only offline arena."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


CHECKPOINT_SCHEMA_VERSION = "w2-offline-arena-checkpoint-v1.0"


class EpisodeLifecycleState(str, Enum):
    """Fail-closed states for one offline evidence episode."""

    CREATED = "CREATED"
    ACTIVE = "ACTIVE"
    SUBMITTED = "SUBMITTED"
    EVALUATED = "EVALUATED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    CANCELED = "CANCELED"
    CLOSED = "CLOSED"

    @property
    def failed(self) -> bool:
        return self in {
            EpisodeLifecycleState.FAILED,
            EpisodeLifecycleState.BLOCKED,
            EpisodeLifecycleState.CANCELED,
        }

    @property
    def terminal(self) -> bool:
        return self in {
            EpisodeLifecycleState.SUBMITTED,
            EpisodeLifecycleState.EVALUATED,
            EpisodeLifecycleState.CLOSED,
        } or self.failed


_ALLOWED_TRANSITIONS: dict[EpisodeLifecycleState, frozenset[EpisodeLifecycleState]] = {
    EpisodeLifecycleState.CREATED: frozenset(
        {
            EpisodeLifecycleState.ACTIVE,
            EpisodeLifecycleState.FAILED,
            EpisodeLifecycleState.BLOCKED,
            EpisodeLifecycleState.CANCELED,
            EpisodeLifecycleState.CLOSED,
        }
    ),
    EpisodeLifecycleState.ACTIVE: frozenset(
        {
            EpisodeLifecycleState.SUBMITTED,
            EpisodeLifecycleState.FAILED,
            EpisodeLifecycleState.BLOCKED,
            EpisodeLifecycleState.CANCELED,
            EpisodeLifecycleState.CLOSED,
        }
    ),
    EpisodeLifecycleState.SUBMITTED: frozenset(
        {
            EpisodeLifecycleState.EVALUATED,
            EpisodeLifecycleState.FAILED,
            EpisodeLifecycleState.CANCELED,
            EpisodeLifecycleState.CLOSED,
        }
    ),
    EpisodeLifecycleState.EVALUATED: frozenset({EpisodeLifecycleState.CLOSED}),
    EpisodeLifecycleState.FAILED: frozenset({EpisodeLifecycleState.CLOSED}),
    EpisodeLifecycleState.BLOCKED: frozenset({EpisodeLifecycleState.CLOSED}),
    EpisodeLifecycleState.CANCELED: frozenset({EpisodeLifecycleState.CLOSED}),
    EpisodeLifecycleState.CLOSED: frozenset(),
}


class StateTransitionError(RuntimeError):
    """The requested lifecycle transition is not legal."""


class AtomicCheckpointError(RuntimeError):
    """A checkpoint could not be committed or verified."""


@dataclass
class EpisodeLifecycle:
    """Small state machine with a monotonic public revision."""

    state: EpisodeLifecycleState = EpisodeLifecycleState.CREATED
    revision: int = 0
    failure_code: str | None = None

    def require(self, *allowed: EpisodeLifecycleState) -> None:
        if self.state not in allowed:
            expected = ", ".join(value.value for value in allowed)
            raise StateTransitionError(
                f"operation requires state {expected}; current state is {self.state.value}"
            )

    def transition(
        self,
        target: EpisodeLifecycleState,
        *,
        failure_code: str | None = None,
    ) -> None:
        target = (
            target
            if isinstance(target, EpisodeLifecycleState)
            else EpisodeLifecycleState(target)
        )
        if target not in _ALLOWED_TRANSITIONS[self.state]:
            raise StateTransitionError(
                f"transition {self.state.value} -> {target.value} is not allowed"
            )
        if target.failed:
            if not isinstance(failure_code, str) or not failure_code.strip():
                raise StateTransitionError("failure transitions require a failure_code")
            self.failure_code = failure_code.strip()
        elif failure_code is not None:
            raise StateTransitionError("successful transitions cannot set failure_code")
        self.state = target
        self.revision += 1

    def touch(self) -> None:
        """Advance an in-state revision after one successful active action."""

        self.require(EpisodeLifecycleState.ACTIVE)
        self.revision += 1

    def checkpoint_failed(self) -> None:
        """Record durability failure even if the intended transition was terminal."""

        if self.state is EpisodeLifecycleState.FAILED:
            return
        self.state = EpisodeLifecycleState.FAILED
        self.failure_code = "AtomicCheckpointError"
        self.revision += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "revision": self.revision,
            "failure_code": self.failure_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EpisodeLifecycle":
        if set(value) != {"state", "revision", "failure_code"}:
            raise StateTransitionError("checkpoint lifecycle fields are not canonical")
        try:
            state = EpisodeLifecycleState(value["state"])
        except (TypeError, ValueError) as exc:
            raise StateTransitionError("checkpoint contains an unknown state") from exc
        revision = value["revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise StateTransitionError("checkpoint revision must be non-negative")
        failure_code = value["failure_code"]
        if state.failed:
            if not isinstance(failure_code, str) or not failure_code.strip():
                raise StateTransitionError("failed checkpoint state needs failure_code")
        elif failure_code is not None:
            raise StateTransitionError("non-failed checkpoint cannot carry failure_code")
        return cls(state=state, revision=revision, failure_code=failure_code)


@dataclass(frozen=True)
class EpisodeState:
    """Portable persistent state for one CPU-offline episode."""

    episode_id: str
    benchmark_id: str
    sample_id: str
    profile: str
    revision: int
    revealed_assets: tuple[str, ...] = ()
    derived_assets: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    action_history: tuple[dict[str, Any], ...] = ()
    model_call_count: int = 0
    environment_action_count: int = 0
    submission_status: str = "not_submitted"
    remaining_budget: dict[str, int] | None = None
    checkpoint_hash: str | None = None

    def __post_init__(self) -> None:
        for name in ("episode_id", "benchmark_id", "sample_id", "profile"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise StateTransitionError(f"{name} must be non-empty")
        for name in ("revision", "model_call_count", "environment_action_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise StateTransitionError(f"{name} must be non-negative")
        for name in ("revealed_assets", "derived_assets", "evidence_refs"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise StateTransitionError(f"{name} must be a sequence")
            normalized = tuple(str(item) for item in value)
            if any(not item for item in normalized) or len(set(normalized)) != len(normalized):
                raise StateTransitionError(f"{name} must contain unique identifiers")
            object.__setattr__(self, name, normalized)
        if isinstance(self.action_history, (str, bytes)) or not isinstance(
            self.action_history, Sequence
        ):
            raise StateTransitionError("action_history must be a sequence")
        history = tuple(dict(item) for item in self.action_history)
        object.__setattr__(self, "action_history", history)
        if self.submission_status not in {
            "not_submitted",
            "submitted",
            "evaluated",
            "failed",
            "blocked",
            "canceled",
        }:
            raise StateTransitionError("submission_status is invalid")
        budget = {} if self.remaining_budget is None else dict(self.remaining_budget)
        if any(
            not isinstance(key, str)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in budget.items()
        ):
            raise StateTransitionError("remaining_budget must contain non-negative counters")
        object.__setattr__(self, "remaining_budget", budget)
        if self.checkpoint_hash is not None and (
            not isinstance(self.checkpoint_hash, str)
            or len(self.checkpoint_hash) != 64
            or any(char not in "0123456789abcdef" for char in self.checkpoint_hash)
        ):
            raise StateTransitionError("checkpoint_hash must be a lowercase SHA-256")

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "benchmark_id": self.benchmark_id,
            "sample_id": self.sample_id,
            "profile": self.profile,
            "revision": self.revision,
            "revealed_assets": list(self.revealed_assets),
            "derived_assets": list(self.derived_assets),
            "evidence_refs": list(self.evidence_refs),
            "action_history": [dict(item) for item in self.action_history],
            "model_call_count": self.model_call_count,
            "environment_action_count": self.environment_action_count,
            "submission_status": self.submission_status,
            "remaining_budget": dict(self.remaining_budget or {}),
            "checkpoint_hash": self.checkpoint_hash,
        }


def canonical_sha256(value: Any) -> str:
    """Hash finite JSON with the repository's canonical ordering convention."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AtomicCheckpointError("checkpoint payload must be finite JSON") from exc
    return hashlib.sha256(payload).hexdigest()


class AtomicCheckpointStore:
    """Integrity-check and atomically replace one deterministic JSON checkpoint."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AtomicCheckpointError(
                f"checkpoint directory creation failed: {type(exc).__name__}"
            ) from None
        if self.path.exists() and not self.path.is_file():
            raise AtomicCheckpointError("checkpoint path must identify a file")

    def save(self, payload: Mapping[str, Any]) -> Path:
        if not isinstance(payload, Mapping):
            raise AtomicCheckpointError("checkpoint payload must be an object")
        normalized = self._json_copy(payload)
        envelope = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "digest_algorithm": "sha256",
            "payload_sha256": canonical_sha256(normalized),
            "payload": normalized,
        }
        serialized = json.dumps(
            envelope,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        ) + "\n"
        temporary = self.path.parent / (
            f".{self.path.name}.{uuid.uuid4().hex}.tmp"
        )
        descriptor: int | None = None
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                descriptor = None
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            self._fsync_parent()
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise AtomicCheckpointError(
                f"atomic checkpoint commit failed: {type(exc).__name__}"
            ) from None
        return self.path

    def load(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise AtomicCheckpointError("checkpoint does not exist") from None
        except (OSError, json.JSONDecodeError):
            raise AtomicCheckpointError("checkpoint is unreadable or invalid JSON") from None
        required = {
            "schema_version",
            "digest_algorithm",
            "payload_sha256",
            "payload",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise AtomicCheckpointError("checkpoint envelope fields are not canonical")
        if (
            value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
            or value.get("digest_algorithm") != "sha256"
        ):
            raise AtomicCheckpointError("unsupported checkpoint envelope")
        payload = value.get("payload")
        if not isinstance(payload, Mapping):
            raise AtomicCheckpointError("checkpoint payload must be an object")
        if value.get("payload_sha256") != canonical_sha256(payload):
            raise AtomicCheckpointError("checkpoint payload digest mismatch")
        return self._json_copy(payload)

    @staticmethod
    def _json_copy(value: Any) -> Any:
        try:
            return json.loads(
                json.dumps(
                    value,
                    allow_nan=False,
                    ensure_ascii=True,
                    sort_keys=True,
                )
            )
        except (TypeError, ValueError) as exc:
            raise AtomicCheckpointError(
                "checkpoint payload must contain finite JSON values"
            ) from exc

    def _fsync_parent(self) -> None:
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        descriptor = os.open(self.path.parent, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "AtomicCheckpointError",
    "AtomicCheckpointStore",
    "EpisodeLifecycle",
    "EpisodeLifecycleState",
    "EpisodeState",
    "StateTransitionError",
    "canonical_sha256",
]
