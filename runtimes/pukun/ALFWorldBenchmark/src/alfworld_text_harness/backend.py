from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import textworld
import textworld.gym

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

from .config import HarnessConfig
from .manifest import TaskRecord
from .verifier import ALFWorldNativeVerifier, VerificationResult


@dataclass(frozen=True)
class TaskState:
    task: TaskRecord
    observation: str
    admissible_actions: list[str]
    verification: VerificationResult


@dataclass(frozen=True)
class StepResult:
    action: str | None
    observation_before: str
    observation_after: str
    admissible_actions_before: list[str]
    admissible_actions_after: list[str]
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


class ALFWorldTextBackend:
    def __init__(self, config: HarnessConfig, max_steps: int | None = None):
        self.config = config
        self.max_steps = max_steps or config.max_steps
        self.env = None
        self.task: TaskRecord | None = None
        self.observation = ""
        self.admissible_actions: list[str] = []
        self.verification = VerificationResult(success=False, score=0.0, done=False)

    def reset_task(self, task: TaskRecord) -> TaskState:
        self.close()
        self.task = task
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_games(
            [task.gamefile],
            request_infos,
            batch_size=1,
            asynchronous=False,
            max_episode_steps=self.max_steps,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos],
        )
        self.env = textworld.gym.make(env_id)
        obs, info = self.env.reset()
        self.observation = _first(obs, "")
        self.admissible_actions = list(_first(info.get("admissible_commands", [[]]), []))
        self.verification = ALFWorldNativeVerifier.from_reset(info)
        return TaskState(
            task=task,
            observation=self.observation,
            admissible_actions=list(self.admissible_actions),
            verification=self.verification,
        )

    def observe_text_state(self) -> str:
        return self.observation

    def list_actions(self) -> list[str]:
        return list(self.admissible_actions)

    def step_native_action(self, action: str) -> StepResult:
        if self.env is None:
            raise RuntimeError("ALFWorldTextBackend.reset_task must be called before step.")
        observation_before = self.observation
        actions_before = list(self.admissible_actions)
        obs, scores, dones, infos = self.env.step([action])
        self.observation = _first(obs, "")
        self.admissible_actions = list(_first(infos.get("admissible_commands", [[]]), []))
        self.verification = ALFWorldNativeVerifier.from_step(scores, dones, infos)
        return StepResult(
            action=action,
            observation_before=observation_before,
            observation_after=self.observation,
            admissible_actions_before=actions_before,
            admissible_actions_after=list(self.admissible_actions),
            verification=self.verification,
            valid_action=True,
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
    admissible_actions: list[str],
    verification: VerificationResult,
    error: dict[str, Any],
) -> StepResult:
    return StepResult(
        action=action,
        observation_before=observation,
        observation_after=observation,
        admissible_actions_before=list(admissible_actions),
        admissible_actions_after=list(admissible_actions),
        verification=verification,
        valid_action=False,
        error=error,
    )


def _first(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return default if value is None else value
