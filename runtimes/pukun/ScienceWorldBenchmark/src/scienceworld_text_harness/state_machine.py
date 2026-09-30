from __future__ import annotations

from enum import Enum
from typing import Any

from .backend import ScienceWorldTextBackend, StepResult, TaskState, invalid_step_result
from .manifest import TaskRecord


class HarnessState(str, Enum):
    UNINITIALIZED = "UNINITIALIZED"
    RESETTING = "RESETTING"
    OBSERVED = "OBSERVED"
    VALIDATING_ACTION = "VALIDATING_ACTION"
    STEPPING_ENV = "STEPPING_ENV"
    VERIFYING = "VERIFYING"
    TERMINATED_SUCCESS = "TERMINATED_SUCCESS"
    TERMINATED_FAILURE = "TERMINATED_FAILURE"
    ERROR = "ERROR"


class ScienceWorldTextStateMachine:
    def __init__(self, backend: ScienceWorldTextBackend, max_steps: int):
        self.backend = backend
        self.max_steps = max_steps
        self.state = HarnessState.UNINITIALIZED
        self.step_count = 0
        self.task_description = ""

    def reset_task(self, task: TaskRecord) -> TaskState:
        self.state = HarnessState.RESETTING
        self.step_count = 0
        task_state = self.backend.reset_task(task)
        self.task_description = task_state.task_description
        self.state = HarnessState.OBSERVED
        return task_state

    def execute_action(self, action: str) -> StepResult:
        if self.state in {HarnessState.TERMINATED_SUCCESS, HarnessState.TERMINATED_FAILURE}:
            return self._invalid(action, "terminal_state", "Cannot step after terminal state.")
        if self.state == HarnessState.UNINITIALIZED:
            return self._invalid(action, "not_reset", "reset_task must be called before step.")

        self.state = HarnessState.VALIDATING_ACTION
        actions = self.backend.list_actions()
        if action not in actions:
            self.state = HarnessState.OBSERVED
            return self._invalid(
                action,
                "invalid_action",
                "Action is not in the current ScienceWorld valid action list.",
                candidates=actions,
            )

        self.state = HarnessState.STEPPING_ENV
        try:
            result = self.backend.step_native_action(action)
        except Exception as exc:  # pragma: no cover - defensive boundary.
            self.state = HarnessState.ERROR
            return self._invalid(action, "env_error", str(exc), candidates=actions)

        self.step_count += 1
        self.state = HarnessState.VERIFYING
        if result.success:
            self.state = HarnessState.TERMINATED_SUCCESS
        elif result.done or self.step_count >= self.max_steps:
            self.state = HarnessState.TERMINATED_FAILURE
        else:
            self.state = HarnessState.OBSERVED
        return result

    def _invalid(
        self,
        action: str | None,
        kind: str,
        message: str,
        candidates: list[str] | None = None,
    ) -> StepResult:
        error: dict[str, Any] = {"kind": kind, "message": message}
        if candidates is not None:
            error["candidates"] = candidates
        return invalid_step_result(
            action=action,
            observation=self.backend.observe_text_world(),
            valid_actions=self.backend.list_actions(),
            verification=self.backend.check_success(),
            error=error,
        )

    @property
    def terminal(self) -> bool:
        return self.state in {HarnessState.TERMINATED_SUCCESS, HarnessState.TERMINATED_FAILURE}
