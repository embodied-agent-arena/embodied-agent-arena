from __future__ import annotations

from collections import defaultdict
from typing import Any

from .agents import get_agent
from .backend import ALFWorldTextBackend
from .config import HarnessConfig, get_config
from .manifest import TaskRecord, build_manifest, select_suite
from .primitives import ALFWorldTextPrimitives
from .state_machine import ALFWorldTextStateMachine
from .trace import TraceWriter


class BenchmarkRunner:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or get_config()

    def run(
        self,
        agent_name: str,
        suite: str,
        num_tasks: int | None = None,
        seed: int | None = None,
        force_manifest: bool = False,
        max_agent_steps: int | None = None,
        start_index: int = 0,
    ) -> dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        manifest = build_manifest(self.config, force=force_manifest)
        suite_task_count = len(select_suite(manifest, suite=suite, seed=seed))
        tasks = select_suite(manifest, suite=suite, num_tasks=num_tasks, seed=seed, start_index=start_index)
        trace = TraceWriter(self.config)
        task_rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            task_rows.append(self._run_task(task, suite, agent_name, seed + start_index + index, trace, max_agent_steps))
        summary = self._summarize(agent_name, suite, task_rows, trace.run_id, start_index, suite_task_count)
        trace.write_summary(summary, task_rows)
        return summary

    def _run_task(
        self,
        task: TaskRecord,
        suite: str,
        agent_name: str,
        seed: int,
        trace: TraceWriter,
        max_agent_steps: int | None,
    ) -> dict[str, Any]:
        backend = ALFWorldTextBackend(self.config, max_steps=self.config.max_steps)
        machine = ALFWorldTextStateMachine(backend, max_steps=self.config.max_steps)
        stopped_reason = "exception"
        exception = None
        try:
            agent = get_agent(agent_name, seed=seed, root_dir=self.config.root_dir)
            machine.reset_task(task)
            primitives = ALFWorldTextPrimitives(machine, trace, task, suite)
            agent_result = agent.run(primitives, max_steps=max_agent_steps or self.config.max_steps)
            stopped_reason = agent_result.stopped_reason
            final = backend.check_success()
            metrics = _merge_metrics(machine.step_count, primitives.metrics, agent_result.metrics)
        except Exception as exc:  # pragma: no cover - command boundary.
            exception = str(exc)
            final = backend.check_success()
            metrics = _merge_metrics(machine.step_count, {})
        finally:
            backend.close()

        return {
            "agent": agent_name,
            "suite": suite,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "source_split": task.source_split,
            "goal_text": task.goal_text,
            "gamefile": task.gamefile,
            "success": final.success,
            "score": final.score,
            "done": final.done,
            "stopped_reason": stopped_reason,
            "exception": exception,
            **metrics,
        }

    def _summarize(
        self,
        agent_name: str,
        suite: str,
        task_rows: list[dict[str, Any]],
        run_id: str,
        start_index: int,
        suite_task_count: int,
    ) -> dict[str, Any]:
        total = len(task_rows)
        successes = sum(1 for row in task_rows if row["success"])
        by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"tasks": 0, "successes": 0})
        for row in task_rows:
            bucket = by_type[row["task_type"]]
            bucket["tasks"] += 1
            bucket["successes"] += int(bool(row["success"]))
        success_by_task_type = {
            task_type: {
                **values,
                "success_rate": values["successes"] / values["tasks"] if values["tasks"] else 0.0,
            }
            for task_type, values in sorted(by_type.items())
        }

        return {
            "run_id": run_id,
            "benchmark": "ALFWorld",
            "track": "text",
            "agent": agent_name,
            "suite": suite,
            "num_tasks": total,
            "suite_task_count": suite_task_count,
            "start_index": start_index,
            "end_index_exclusive": start_index + total,
            "is_full_suite_run": start_index == 0 and total == suite_task_count,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "avg_steps": _avg(row["env_steps"] for row in task_rows),
            "avg_score": _avg(row["score"] for row in task_rows),
            "invalid_action_count": sum(row["invalid_action_count"] for row in task_rows),
            "wrapper_no_match_count": sum(row["wrapper_no_match_count"] for row in task_rows),
            "wrapper_ambiguity_count": sum(row["wrapper_ambiguity_count"] for row in task_rows),
            "avg_code_attempts": _avg(row["code_attempts"] for row in task_rows),
            "code_exception_count": sum(row["code_exception_count"] for row in task_rows),
            "code_timeout_count": sum(row["code_timeout_count"] for row in task_rows),
            "code_primitive_budget_count": sum(row["code_primitive_budget_count"] for row in task_rows),
            "timeout_count": sum(
                1
                for row in task_rows
                if row["stopped_reason"] == "max_steps"
                or (not row["success"] and row["env_steps"] >= self.config.max_steps)
            ),
            "exception_count": sum(1 for row in task_rows if row["exception"]),
            "success_by_task_type": success_by_task_type,
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "task_type": row["task_type"],
                    "source_split": row["source_split"],
                    "gamefile": row["gamefile"],
                    "goal_text": row["goal_text"],
                    "success": row["success"],
                    "env_steps": row["env_steps"],
                    "stopped_reason": row["stopped_reason"],
                    "exception": row["exception"],
                    "code_attempts": row["code_attempts"],
                }
                for row in task_rows
            ],
        }


def _merge_metrics(env_steps: int, *sources: dict[str, Any]) -> dict[str, Any]:
    metrics = {
        "env_steps": env_steps,
        "invalid_action_count": 0,
        "wrapper_no_match_count": 0,
        "wrapper_ambiguity_count": 0,
        "code_attempts": 0,
        "code_exception_count": 0,
        "code_timeout_count": 0,
        "code_primitive_budget_count": 0,
    }
    for source in sources:
        metrics.update(source)
    return metrics


def _avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
