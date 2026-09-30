from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .primitives import DiscoveryWorldPrimitives


@dataclass
class AgentRunResult:
    stopped_reason: str
    steps_attempted: int
    metrics: dict[str, Any] = field(default_factory=dict)


class RandomBaseline:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def run(self, primitives: DiscoveryWorldPrimitives, max_steps: int) -> AgentRunResult:
        primitives.get_task_context()
        primitives.observe_world()
        steps = 0
        while steps < max_steps:
            status = primitives.check_success()
            if status["success"]:
                return AgentRunResult("success", steps)
            if status["completed"]:
                return AgentRunResult("completed", steps)
            action = self._choose_primitive(primitives)
            result = action()
            steps += 1
            if result.success:
                return AgentRunResult("success", steps)
            if result.completed:
                return AgentRunResult("completed", steps)
        return AgentRunResult("max_steps", steps)

    def _choose_primitive(self, primitives: DiscoveryWorldPrimitives):
        observation = primitives.observe_world()
        choices = []
        directions = observation.get("agentLocation", {}).get("directions_you_can_move", [])
        for direction in directions:
            choices.append(lambda d=direction: primitives.move_direction(d))
        for direction in ("north", "east", "south", "west"):
            choices.append(lambda d=direction: primitives.rotate_direction(d))
        locations = list(primitives.list_teleport_locations().keys())
        for location in locations:
            choices.append(lambda loc=location: primitives.teleport_to_location(loc))
        accessible = primitives.list_accessible_objects()
        for obj in accessible[:5]:
            ref = obj.get("uuid")
            choices.extend(
                [
                    lambda r=ref: primitives.pickup_object(r),
                    lambda r=ref: primitives.open_object(r),
                    lambda r=ref: primitives.close_object(r),
                    lambda r=ref: primitives.read_object(r),
                    lambda r=ref: primitives.talk_to(r),
                ]
            )
        if not choices:
            return primitives.wait
        return self.rng.choice(choices)


def get_agent(name: str, seed: int = 0, root_dir: Path | None = None):
    if name == "random":
        return RandomBaseline(seed=seed)
    if name in {"deepseek_v3_code", "deepseek_v4_code"}:
        if root_dir is None:
            raise ValueError(f"root_dir is required for {name}.")
        from .deepseek_agent import DeepSeekV3CodeAgent

        return DeepSeekV3CodeAgent(root_dir=root_dir, agent_name=name)
    raise ValueError("Unknown agent '%s'. Expected: random, deepseek_v3_code" % name)
