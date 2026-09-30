from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score: float
    done: bool
    source: str = "alfworld_native"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ALFWorldNativeVerifier:
    @staticmethod
    def from_reset(info: dict[str, Any]) -> VerificationResult:
        won = _first(info.get("won", [False]), default=False)
        return VerificationResult(success=bool(won), score=0.0, done=False)

    @staticmethod
    def from_step(scores: list[Any], dones: list[Any], infos: dict[str, Any]) -> VerificationResult:
        score = _first(scores, default=0.0)
        done = _first(dones, default=False)
        won = _first(infos.get("won", [False]), default=False)
        return VerificationResult(success=bool(won), score=float(score), done=bool(done))


def _first(value: Any, default: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return default if value is None else value
