from __future__ import annotations

from collections import defaultdict
from typing import Any

from .agents import get_agent
from .backend import VirtualHomeSymbolicBackend
from .config import HarnessConfig, get_config
from .manifest import TaskRecord, build_manifest, select_suite
from .primitives import VirtualHomeSymbolicPrimitives
from .trace import TraceWriter


class BenchmarkRunner:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or get_config()

    def run(
        self,
        agent_name: str,
        suite: str,
        start_index: int = 0,
        num_tasks: int | None = None,
        seed: int | None = None,
        force_manifest: bool = False,
        max_agent_steps: int | None = None,
    ) -> dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        manifest = build_manifest(self.config, force=force_manifest)
        tasks = select_suite(manifest, suite=suite, start_index=start_index, num_tasks=num_tasks)
        trace = TraceWriter(self.config)
        rows = []
        for index, task in enumerate(tasks):
            rows.append(
                self._run_task(task, suite, agent_name, seed + index, trace, max_agent_steps or self.config.max_steps)
            )
            partial_summary = self._summarize(agent_name, suite, rows, trace.run_id, start_index)
            partial_summary["requested_tasks"] = len(tasks)
            partial_summary["completed_tasks"] = len(rows)
            trace.write_partial_summary(partial_summary, rows)
        summary = self._summarize(agent_name, suite, rows, trace.run_id, start_index)
        summary["requested_tasks"] = len(tasks)
        summary["completed_tasks"] = len(rows)
        trace.write_summary(summary, rows)
        return summary

    def _run_task(
        self,
        task: TaskRecord,
        suite: str,
        agent_name: str,
        seed: int,
        trace: TraceWriter,
        max_agent_steps: int,
    ) -> dict[str, Any]:
        backend = VirtualHomeSymbolicBackend(self.config)
        exception = None
        stopped_reason = "exception"
        verification = {"success": False, "completed": False, "score": 0.0, "reason": "not_started"}
        metrics = _empty_metrics()
        try:
            backend.reset_task(task)
            primitives = VirtualHomeSymbolicPrimitives(backend, trace, task, suite, max_agent_steps)
            result = get_agent(agent_name, seed).run(primitives, max_agent_steps)
            stopped_reason = result.stopped_reason
            verification = backend.check_success(primitives.evidence)
            metrics = _merge_metrics(primitives.metrics, result.metrics)
        except Exception as exc:
            exception = f"{type(exc).__name__}: {exc}"
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "goal_text": task.goal_text,
            "success": bool(verification.get("success")),
            "score": float(verification.get("score") or 0.0),
            "stopped_reason": stopped_reason,
            "exception": exception,
            **metrics,
        }

    def _summarize(
        self,
        agent_name: str,
        suite: str,
        rows: list[dict[str, Any]],
        run_id: str,
        start_index: int,
    ) -> dict[str, Any]:
        total = len(rows)
        successes = sum(1 for row in rows if row["success"])
        by_type: dict[str, dict[str, Any]] = defaultdict(lambda: {"tasks": 0, "successes": 0})
        for row in rows:
            bucket = by_type[row["task_type"]]
            bucket["tasks"] += 1
            bucket["successes"] += int(bool(row["success"]))
        return {
            "run_id": run_id,
            "benchmark": "VirtualHome",
            "track": "symbolic_evolving_graph",
            "agent": agent_name,
            "suite": suite,
            "start_index": start_index,
            "end_index_exclusive": start_index + total,
            "num_tasks": total,
            "successes": successes,
            "success_rate": successes / total if total else 0.0,
            "avg_steps": _avg(row["env_steps"] for row in rows),
            "avg_score": _avg(row["score"] for row in rows),
            "avg_code_attempts": _avg(row["code_attempts"] for row in rows),
            "invalid_action_count": sum(row["invalid_action_count"] for row in rows),
            "code_exception_count": sum(row["code_exception_count"] for row in rows),
            "code_timeout_count": sum(row["code_timeout_count"] for row in rows),
            "timeout_count": sum(1 for row in rows if row["stopped_reason"] == "max_steps"),
            "exception_count": sum(1 for row in rows if row["exception"]),
            "success_by_task_type": {
                key: {
                    **value,
                    "success_rate": value["successes"] / value["tasks"] if value["tasks"] else 0.0,
                }
                for key, value in sorted(by_type.items())
            },
        }


def _empty_metrics() -> dict[str, int]:
    return {
        "env_steps": 0,
        "invalid_action_count": 0,
        "code_attempts": 0,
        "code_exception_count": 0,
        "code_timeout_count": 0,
    }


def _merge_metrics(*sources: dict[str, Any]) -> dict[str, Any]:
    merged = _empty_metrics()
    for source in sources:
        merged.update(source)
    return merged


def _avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
