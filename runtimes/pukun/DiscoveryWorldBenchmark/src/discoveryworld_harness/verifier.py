from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    completed: bool
    score_normalized: float
    source: str = "discoveryworld_native"
    tasks: list[dict[str, Any]] = field(default_factory=list)

    def to_agent_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "completed": self.completed,
            "scoreNormalized": self.score_normalized,
            "source": self.source,
        }

    def to_trace_dict(self) -> dict[str, Any]:
        return asdict(self)


class DiscoveryWorldNativeVerifier:
    @staticmethod
    def from_api(api: Any) -> VerificationResult:
        scorecard = api.getTaskScorecard() or []
        completed = bool(api.areTasksComplete())
        task_summaries: list[dict[str, Any]] = []
        scores: list[float] = []
        successes: list[bool] = []
        for item in scorecard:
            score = float(item.get("scoreNormalized") or 0.0)
            scores.append(score)
            successes.append(bool(item.get("completedSuccessfully")))
            task_summaries.append(
                {
                    "taskName": item.get("taskName"),
                    "taskDescription": item.get("taskDescription"),
                    "completed": bool(item.get("completed")),
                    "completedSuccessfully": bool(item.get("completedSuccessfully")),
                    "scoreNormalized": score,
                }
            )
        success = bool(successes) and all(successes)
        avg_score = sum(scores) / len(scores) if scores else 0.0
        return VerificationResult(
            success=success,
            completed=completed,
            score_normalized=avg_score,
            tasks=task_summaries,
        )


def empty_verification() -> VerificationResult:
    return VerificationResult(success=False, completed=False, score_normalized=0.0)
