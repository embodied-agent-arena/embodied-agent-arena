"""Same-episode official-success binding for the live VLABench backend.

The pinned upstream ``Evaluator.evaluate_single_episode`` cannot adopt an
already-running environment: it always constructs, resets, and closes its own
environment.  VLABench's composer task hook is the predicate that makes the
evaluator's ``env.step(...).last()`` successful, so the live backend binds that
official task object and its physics from the agent-operated episode instead.
"""

from __future__ import annotations

from typing import Any


VLABENCH_OFFICIAL_SUCCESS_SYMBOL = (
    "VLABench/tasks/dm_task.py:LM4ManipBaseTask.should_terminate_episode"
)
VLABENCH_OFFICIAL_SUCCESS_SOURCE_SHA256 = (
    "5e0011bf019ac2393a51560eb46750c0319a27c6b36074a3bcf4a6bb04b9e109"
)
VLABENCH_SAME_EPISODE_RECEIPT_SCHEMA = (
    "agentic-embodied-arena/vlabench-same-episode-official/v1"
)


class VLABenchSameEpisodeOfficialBinding:
    """One-shot binding of the official task predicate to the live episode."""

    def __init__(self, env: Any) -> None:
        task = getattr(env, "task", None)
        physics = getattr(env, "physics", None)
        if task is None or physics is None:
            raise RuntimeError("VLABench live environment lacks task/physics")
        self._env = env
        self._task = task
        self._physics = physics
        self._captured = False
        self._closed = False

    @property
    def official_object(self) -> Any:
        return self._task

    @property
    def call_arguments(self) -> tuple[Any, ...]:
        return (self._physics,)

    @property
    def call_keyword_arguments(self) -> dict[str, Any]:
        return {}

    def capture(self) -> dict[str, Any]:
        """Invoke the source-locked predicate on the still-open exact episode."""

        if self._captured:
            raise RuntimeError("VLABench official episode already captured")
        if self._closed or bool(getattr(self._env, "closed", False)):
            raise RuntimeError("VLABench official episode is already closed")
        if getattr(self._env, "task", None) is not self._task:
            raise RuntimeError("VLABench official task identity changed")
        if getattr(self._env, "physics", None) is not self._physics:
            raise RuntimeError("VLABench official physics identity changed")

        predicate = getattr(self._task, "should_terminate_episode", None)
        if not callable(predicate):
            raise RuntimeError("VLABench official success predicate is unavailable")
        result = predicate(self._physics)
        if type(result) is not bool:
            item = getattr(result, "item", None)
            result = item() if callable(item) else result
        if type(result) is not bool:
            raise RuntimeError("VLABench official success predicate was not boolean")
        self._captured = True
        return {
            "schema_version": VLABENCH_SAME_EPISODE_RECEIPT_SCHEMA,
            "catalog_symbol": VLABENCH_OFFICIAL_SUCCESS_SYMBOL,
            "official_source_digest": VLABENCH_OFFICIAL_SUCCESS_SOURCE_SHA256,
            "result_shape": "bool",
            "success": result,
            "same_episode": True,
            "episode_state_before": "live",
            "episode_state_after": "live",
            "call_arguments": {"args": ["bound_live_physics"], "kwargs": {}},
        }

    def mark_closed(self) -> None:
        self._closed = True


__all__ = [
    "VLABENCH_OFFICIAL_SUCCESS_SOURCE_SHA256",
    "VLABENCH_OFFICIAL_SUCCESS_SYMBOL",
    "VLABENCH_SAME_EPISODE_RECEIPT_SCHEMA",
    "VLABenchSameEpisodeOfficialBinding",
]
