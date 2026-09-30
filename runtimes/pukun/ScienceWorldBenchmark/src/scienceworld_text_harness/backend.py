from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scienceworld import ScienceWorldEnv

from .config import HarnessConfig
from .manifest import TaskRecord
from .verifier import ScienceWorldNativeVerifier, VerificationResult


@dataclass(frozen=True)
class TaskState:
    task: TaskRecord
    observation: str
    valid_actions: list[str]
    verification: VerificationResult
    task_description: str


@dataclass(frozen=True)
class StepResult:
    action: str | None
    observation_before: str
    observation_after: str
    valid_actions_before: list[str]
    valid_actions_after: list[str]
    verification: VerificationResult
    valid_action: bool
    error: dict[str, Any] | None = None

    @property
    def success(self) -> bool:
        return self.verification.success

    @property
    def done(self) -> bool:
        return self.verification.done

    @property
    def score(self) -> float:
        return self.verification.score


class ScienceWorldTextBackend:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.env: ScienceWorldEnv | None = None
        self.task: TaskRecord | None = None
        self.observation = ""
        self.valid_actions: list[str] = []
        self.task_description = ""
        self.verification = VerificationResult(success=False, score=0.0, done=False)

    def reset_task(self, task: TaskRecord) -> TaskState:
        self.close()
        self.task = task
        self.env = ScienceWorldEnv("", None, envStepLimit=self.config.max_steps)
        self.env.load(task.task_name, task.variation_idx, task.simplification)
        observation, info = self.env.reset()
        self.observation = observation
        self.task_description = self.env.get_task_description()
        self.valid_actions = _valid_actions_from_info(info)
        self.verification = ScienceWorldNativeVerifier.from_reset(info)
        return TaskState(
            task=task,
            observation=self.observation,
            valid_actions=list(self.valid_actions),
            verification=self.verification,
            task_description=self.task_description,
        )

    def observe_text_world(self) -> str:
        return self.observation

    def list_actions(self) -> list[str]:
        return list(self.valid_actions)

    def look(self) -> str:
        if self.env is None:
            raise RuntimeError("ScienceWorldTextBackend.reset_task must be called before look.")
        return self.env.look()

    def inventory(self) -> str:
        if self.env is None:
            raise RuntimeError("ScienceWorldTextBackend.reset_task must be called before inventory.")
        return self.env.inventory()

    def step_native_action(self, action: str) -> StepResult:
        if self.env is None:
            raise RuntimeError("ScienceWorldTextBackend.reset_task must be called before step.")
        observation_before = self.observation
        actions_before = list(self.valid_actions)
        observation, reward, done, info = self.env.step(action)
        self.observation = observation
        self.valid_actions = _valid_actions_from_info(info)
        self.verification = ScienceWorldNativeVerifier.from_step(reward, done, info)
        error = None
        valid_action = True
        if _looks_like_parser_no_match(self.observation):
            error = {
                "kind": "parser_no_match",
                "message": "ScienceWorld listed the action as valid, but env.step returned a parser no-match observation.",
            }
            valid_action = False
        return StepResult(
            action=action,
            observation_before=observation_before,
            observation_after=self.observation,
            valid_actions_before=actions_before,
            valid_actions_after=list(self.valid_actions),
            verification=self.verification,
            valid_action=valid_action,
            error=error,
        )

    def check_success(self) -> VerificationResult:
        return self.verification

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
        self.env = None


def invalid_step_result(
    action: str | None,
    observation: str,
    valid_actions: list[str],
    verification: VerificationResult,
    error: dict[str, Any],
) -> StepResult:
    return StepResult(
        action=action,
        observation_before=observation,
        observation_after=observation,
        valid_actions_before=list(valid_actions),
        valid_actions_after=list(valid_actions),
        verification=verification,
        valid_action=False,
        error=error,
    )


def _valid_actions_from_info(info: dict[str, Any]) -> list[str]:
    actions = [str(action) for action in info.get("valid", [])]
    return sorted(dict.fromkeys(actions))


def _looks_like_parser_no_match(observation: str) -> bool:
    return observation.strip().lower().startswith("no known action matches that input")
