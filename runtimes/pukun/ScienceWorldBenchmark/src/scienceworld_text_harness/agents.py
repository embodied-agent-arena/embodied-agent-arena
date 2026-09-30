from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .primitives import ScienceWorldTextPrimitives


@dataclass(frozen=True)
class AgentRunResult:
    stopped_reason: str
    steps_attempted: int
    metrics: dict[str, Any] = field(default_factory=dict)


class RandomActionAgent:
    name = "random"

    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def run(self, primitives: ScienceWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        for step in range(max_steps):
            status = primitives.check_success()
            if status["success"]:
                return AgentRunResult(stopped_reason="success", steps_attempted=step)
            if status["done"]:
                return AgentRunResult(stopped_reason="done", steps_attempted=step)

            actions = [action for action in primitives.list_actions() if action != "reset task"]
            if not actions:
                return AgentRunResult(stopped_reason="no_actions", steps_attempted=step)
            result = primitives.step_text_action(self.rng.choice(actions))
            if result.success:
                return AgentRunResult(stopped_reason="success", steps_attempted=step + 1)
            if result.done:
                return AgentRunResult(stopped_reason="done", steps_attempted=step + 1)
        return AgentRunResult(stopped_reason="max_steps", steps_attempted=max_steps)


class ScriptedSmokeAgent:
    name = "scripted_smoke"

    def run(self, primitives: ScienceWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        primitives.write_evidence("task_context", primitives.get_task_context())
        primitives.write_evidence("initial_look", primitives.look())
        primitives.write_evidence("initial_inventory", primitives.inventory())
        steps = 0
        preferred_exact = ("look around", "inventory")
        preferred_prefixes = ("go to ", "open ", "look at ")
        while steps < min(max_steps, 5):
            actions = [action for action in primitives.list_actions() if action != "reset task"]
            chosen = next((action for action in preferred_exact if action in actions), None)
            for prefix in preferred_prefixes:
                if chosen:
                    break
                chosen = next((action for action in actions if action.startswith(prefix)), None)
                if chosen:
                    break
            chosen = chosen or (actions[0] if actions else None)
            if not chosen:
                return AgentRunResult(stopped_reason="no_actions", steps_attempted=steps)
            result = primitives.step_text_action(chosen)
            steps += 1
            if result.success:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if result.done:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)
        return AgentRunResult(stopped_reason="script_complete", steps_attempted=steps)


def get_agent(name: str, seed: int, root_dir: Path):
    if name == "random":
        return RandomActionAgent(seed)
    if name == "scripted_smoke":
        return ScriptedSmokeAgent()
    if name in {"deepseek_v4", "deepseek_v4_code"}:
        from .deepseek_agent import DeepSeekV4ActionAgent, DeepSeekV4CodeAgent

        if name == "deepseek_v4":
            return DeepSeekV4ActionAgent(root_dir=root_dir)
        return DeepSeekV4CodeAgent(root_dir=root_dir)
    raise ValueError("Unknown agent '{}'. Expected: random, scripted_smoke, deepseek_v4, deepseek_v4_code".format(name))
