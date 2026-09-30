from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .code_executor import execute_agent_code
from .primitives import VirtualHomeSymbolicPrimitives


@dataclass(frozen=True)
class AgentResult:
    stopped_reason: str
    metrics: dict[str, Any]


class OracleAgent:
    def run(self, primitives: VirtualHomeSymbolicPrimitives, max_steps: int) -> AgentResult:
        for action in primitives.task.expected_plan[:max_steps]:
            primitives.execute_program_step(action)
            if primitives.check_activity_success()["success"]:
                return AgentResult("success", {})
        return AgentResult("success" if primitives.check_activity_success()["success"] else "max_steps", {})


class RandomAgent:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def run(self, primitives: VirtualHomeSymbolicPrimitives, max_steps: int) -> AgentResult:
        actions = primitives.list_actions()
        for _ in range(max_steps):
            primitives.execute_program_step(self.rng.choice(actions))
            if primitives.check_activity_success()["success"]:
                return AgentResult("success", {})
        return AgentResult("max_steps", {})


class OracleCodeAgent:
    def run(self, primitives: VirtualHomeSymbolicPrimitives, max_steps: int) -> AgentResult:
        code = "\n".join(
            [
                "ctx = get_task_context()",
                "state = query_symbolic_state()",
                *[f"execute_program_step({action!r})" for action in primitives.task.expected_plan[:max_steps]],
                "write_evidence('final_state', query_symbolic_state())",
                "check_activity_success()",
            ]
        )
        primitives.record_harness_event(
            "agent_code_generated",
            {
                "agent": "oracle_code",
                "executed_code": code,
                "primitive_cards": primitives.cards_for_prompt(),
            },
        )
        result = execute_agent_code(primitives, code)
        metrics = result.metrics | {"code_attempts": 1}
        return AgentResult("success" if result.final_success else "code_exception" if result.exception else "max_steps", metrics)


def get_agent(name: str, seed: int = 0):
    if name == "oracle":
        return OracleAgent()
    if name == "random":
        return RandomAgent(seed)
    if name == "oracle_code":
        return OracleCodeAgent()
    if name == "deepseek_v4_code":
        from .deepseek_agent import DeepSeekV4CodeAgent

        return DeepSeekV4CodeAgent()
    raise ValueError(f"Unknown agent: {name}. Expected: oracle, oracle_code, random, deepseek_v4_code")
