from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from pathlib import Path

from .primitives import ALFWorldTextPrimitives


@dataclass
class AgentRunResult:
    stopped_reason: str
    steps_attempted: int
    metrics: dict[str, Any] = field(default_factory=dict)


class RandomActionBaseline:
    name = "random"

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def run(self, primitives: ALFWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        steps = 0
        primitives.get_task_context()
        primitives.observe_text_state()
        while steps < max_steps:
            status = primitives.check_success()
            if status["success"]:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if status["done"]:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)
            actions = primitives.list_actions()
            if not actions:
                return AgentRunResult(stopped_reason="no_actions", steps_attempted=steps)
            action = self.rng.choice(actions)
            result = primitives.step_text_action(action)
            steps += 1
            if result.success:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if result.done:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)
        return AgentRunResult(stopped_reason="max_steps", steps_attempted=steps)


def get_agent(name: str, seed: int = 42, root_dir: Path | None = None):
    if name == "random":
        return RandomActionBaseline(seed=seed)
    if name in {"deepseek_v4", "deepseek_v4_code"}:
        if root_dir is None:
            raise ValueError(f"root_dir is required for {name} agent.")
        from .deepseek_agent import DeepSeekV4ActionAgent, DeepSeekV4CodeAgent

        if name == "deepseek_v4":
            return DeepSeekV4ActionAgent(root_dir=root_dir)
        return DeepSeekV4CodeAgent(root_dir=root_dir)
    raise ValueError(f"Unknown agent '{name}'. Expected: random, deepseek_v4, deepseek_v4_code")
