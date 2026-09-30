from __future__ import annotations

from collections import defaultdict
import contextlib
import signal
from typing import Any

from .agents import get_agent
from .backend import DiscoveryWorldBackend
from .config import HarnessConfig, get_config
from .manifest import TaskRecord, build_manifest, select_suite
from .primitives import DiscoveryWorldPrimitives
from .trace import TraceWriter
from .verifier import empty_verification


class TaskTimeoutError(TimeoutError):
    pass


class BenchmarkRunner:
    def __init__(self, config: HarnessConfig | None = None):
        self.config = config or get_config()

    def run(
        self,
        agent_name: str,
        suite: str,
        num_tasks: int | None = None,
        start_index: int | None = None,
        seed: int | None = None,
        force_manifest: bool = False,
        max_agent_steps: int | None = None,
    ) -> dict[str, Any]:
        seed = self.config.seed if seed is None else seed
        manifest = build_manifest(self.config, force=force_manifest)
        suite_task_count = len(manifest[suite])
        tasks = select_suite(manifest, suite=suite, num_tasks=num_tasks, start_index=start_index, seed=seed)
        trace = TraceWriter(self.config)
        task_rows: list[dict[str, Any]] = []
        for index, task in enumerate(tasks):
            task_rows.append(self._run_task(task, suite, agent_name, seed + index, trace, max_agent_steps))
            partial_summary = self._summarize(
                agent_name,
                suite,
                task_rows,
                trace.run_id,
                suite_task_count,
                start_index or 0,
                requested_task_count=len(tasks),
            )
            trace.write_summary(partial_summary, task_rows)
        summary = self._summarize(
            agent_name,
            suite,
            task_rows,
            trace.run_id,
            suite_task_count,
            start_index or 0,
            requested_task_count=len(tasks),
        )
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
        backend = DiscoveryWorldBackend(self.config)
        stopped_reason = "exception"
        exception = None
        final = empty_verification()
        metrics = _merge_metrics(0, {})
        try:
            with _time_limit(self.config.task_timeout_seconds):
                agent = get_agent(agent_name, seed=seed, root_dir=self.config.root_dir)
                backend.reset_task(task)
                primitives = DiscoveryWorldPrimitives(backend, trace, task, suite)
                agent_result = agent.run(primitives, max_steps=max_agent_steps or self.config.max_steps)
                stopped_reason = agent_result.stopped_reason
                final = backend.check_success()
                metrics = _merge_metrics(primitives.metrics["env_steps"], primitives.metrics, agent_result.metrics)
        except TaskTimeoutError as exc:
            stopped_reason = "task_timeout"
            exception = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            message = str(exc)
            if "DEEPSEEK_API_KEY is not set" in message:
                stopped_reason = "blocked_missing_deepseek_api_key"
                exception = None
            else:
                exception = message
            try:
                final = backend.check_success()
            except Exception:
                final = empty_verification()
        finally:
            backend.close()

        return {
            "agent": agent_name,
            "suite": suite,
            "task_id": task.task_id,
            "scenario_name": task.scenario_name,
            "difficulty": task.difficulty,
            "seed": task.seed,
            "task_type": task.task_type,
            "goal_text": task.goal_text,
            "success": final.success,
            "score": final.score_normalized,
            "completed": final.completed,
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
        suite_task_count: int,
        start_index: int,
        requested_task_count: int | None = None,
    ) -> dict[str, Any]:
        total = len(task_rows)
        end_index_exclusive = start_index + total
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
            "benchmark": "DiscoveryWorld",
            "track": "text_json",
            "agent": agent_name,
            "suite": suite,
            "suite_task_count": suite_task_count,
            "start_index": start_index,
            "end_index_exclusive": end_index_exclusive,
            "requested_task_count": requested_task_count if requested_task_count is not None else total,
            "completed_task_count": total,
            "is_full_suite_run": start_index == 0 and end_index_exclusive == suite_task_count,
            "num_tasks": total,
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
                if row["stopped_reason"] in {"max_steps", "code_timeout", "task_timeout", "primitive_call_budget"}
            ),
            "blocked_count": sum(1 for row in task_rows if row["stopped_reason"].startswith("blocked_")),
            "exception_count": sum(1 for row in task_rows if row["exception"]),
            "success_by_task_type": success_by_task_type,
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "scenario_name": row["scenario_name"],
                    "difficulty": row["difficulty"],
                    "seed": row["seed"],
                    "task_type": row["task_type"],
                    "goal_text": row["goal_text"],
                    "success": row["success"],
                    "env_steps": row["env_steps"],
                    "score": row["score"],
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
    metrics["env_steps"] = env_steps
    return metrics


def _avg(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


@contextlib.contextmanager
def _time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def handler(_signum, _frame):
        raise TaskTimeoutError(f"Task timed out after {seconds} seconds.")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        signal.signal(signal.SIGALRM, old_handler)
