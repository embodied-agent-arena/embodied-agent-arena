from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score: float
    done: bool
    reward: float = 0.0
    source: str = "scienceworld_native"

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "score": self.score,
            "done": self.done,
            "reward": self.reward,
            "source": self.source,
        }


class ScienceWorldNativeVerifier:
    @staticmethod
    def from_reset(info: dict[str, Any]) -> VerificationResult:
        return ScienceWorldNativeVerifier.from_info(info, done=False, reward=0.0)

    @staticmethod
    def from_step(reward: float, done: bool, info: dict[str, Any]) -> VerificationResult:
        return ScienceWorldNativeVerifier.from_info(info, done=done, reward=reward)

    @staticmethod
    def from_info(info: dict[str, Any], done: bool, reward: float) -> VerificationResult:
        score = float(info.get("score", 0.0))
        return VerificationResult(
            success=score >= 100.0,
            score=score,
            done=bool(done),
            reward=float(reward),
        )
