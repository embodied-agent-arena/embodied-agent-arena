"""Shared turn/deadline policy; adapters own interpreters, actions and verifiers."""
from __future__ import annotations

import subprocess
import time
from typing import Any


def is_timeout(error: BaseException) -> bool:
    while error is not None:
        if isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
            return True
        error = error.__cause__
    return False


class EpisodeLoop:
    """One instance per episode/attempt, used with an already persistent runner.

    Each adapter supplies native success/terminal flags after execution and
    feedback. Ordinary code errors allow repair; timeouts end the episode.
    """

    def __init__(self, max_turns: int, timeout_seconds: float, *, replay: bool = False,
                 clock=time.monotonic):
        if max_turns < 1 or timeout_seconds <= 0:
            raise ValueError("episode turn and time limits must be positive")
        self.max_turns = max_turns
        self.timeout_seconds = timeout_seconds
        self.replay = replay
        self.clock = clock
        self.started = clock()
        self.turn = 0
        self.stop_reason: str | None = None
        self.budget_exhausted: list[str] = []
        self.turns: list[dict[str, Any]] = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_reason is not None:
            raise StopIteration
        if self.turn != len(self.turns):
            raise RuntimeError("finish_turn must record feedback before the next turn")
        self.turn += 1
        return self.turn

    def phase_timeout(self, limit: float) -> float:
        remaining = self.timeout_seconds - (self.clock() - self.started)
        if remaining <= 0:
            raise TimeoutError("Global trial deadline exhausted")
        return min(limit, remaining)

    def finish_turn(self, *, execution_ok: bool = False, success: bool = False,
                    terminal: bool = False, timed_out: bool = False,
                    budget_exhausted=(), error: str | None = None,
                    stage: str = "execution") -> None:
        timed_out = timed_out or self.clock() - self.started >= self.timeout_seconds
        reason = None
        if timed_out:
            reason = "timeout"
        elif success:
            reason = "success"
        elif budget_exhausted:
            reason = "budget_exhausted"
        elif terminal:
            reason = "terminal"
        elif stage != "execution" and error:
            reason = "agent_error"
        elif self.turn >= self.max_turns:
            reason = "replay_complete" if self.replay else "budget_exhausted"
            if not self.replay:
                budget_exhausted = ["max_agent_iterations"]
        self.stop_reason = reason
        self.budget_exhausted = list(dict.fromkeys(budget_exhausted))
        self.turns.append(dict(turn=self.turn, stage=stage, execution_ok=execution_ok,
                               success=success and not timed_out, terminal=terminal,
                               timed_out=timed_out, error=error, stop_reason=reason,
                               elapsed_seconds=round(self.clock() - self.started, 6)))

    def report(self) -> dict[str, Any]:
        return dict(schema_version="embodied-episode-loop/v1", same_episode=True,
                    persistent_python_state=True, max_turns=self.max_turns,
                    timeout_seconds=self.timeout_seconds, turns_started=self.turn,
                    stop_reason=self.stop_reason, budget_exhausted=self.budget_exhausted,
                    turns=self.turns)
