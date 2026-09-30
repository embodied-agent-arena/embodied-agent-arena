"""Post-submission-only deterministic evaluator for the offline arena."""

from __future__ import annotations

import hashlib
import json
import math
import re
import string
from collections.abc import Callable, Mapping
from typing import Any

from .contracts import EvaluationResult, ParsedSubmission, PublicTask
from .state import EpisodeLifecycleState


EVALUATOR_ID = "offline_arena.diagnostic_exact_match"
EVALUATOR_VERSION = "1.0.0"
_IMPLEMENTATION_HASH = hashlib.sha256(
    b"w2_harness.offline_arena.PrivateEvaluator:1.0.0"
).hexdigest()
_REFERENCE_KEYS = (
    "answer",
    "correct_answer",
    "expected_answer",
    "ground_truth",
    "reference_answer",
    "target_answer",
)


class PrivateEvaluationError(RuntimeError):
    """The private evaluator could not produce a safe diagnostic result."""


class ReferenceAccessError(PrivateEvaluationError):
    """Reference access was attempted outside the post-submission boundary."""


def _normalize_text(value: Any) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    text = str(value).strip().casefold()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text)


def _extract_choice(value: Any, choice_count: int) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return ""
    text = str(value).strip()
    allowed_count = max(1, min(choice_count or 5, 26))
    allowed = {chr(ord("A") + index) for index in range(allowed_count)}
    boxed = re.search(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if boxed and boxed.group(1).upper() in allowed:
        return boxed.group(1).upper()
    for pattern in (
        r"^\s*([A-Za-z])\s*$",
        r"\b([A-Za-z])[.):]",
        r"\(([A-Za-z])\)",
        r"\b(?:answer|choice|option)\s*(?:is|:|=)?\s*([A-Za-z])\b",
    ):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in allowed:
            return match.group(1).upper()
    return ""


def _first_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if not isinstance(value, str):
        return None
    match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", value)
    if match is None:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _extract_reference(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key in _REFERENCE_KEYS:
            if key in value:
                return value[key]
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return None


class PrivateEvaluator:
    """Own a sealed reference source and open it only for SUBMITTED episodes."""

    def __init__(
        self,
        reference: Any | Callable[[], Any],
        *,
        evaluator_id: str = EVALUATOR_ID,
    ) -> None:
        if not isinstance(evaluator_id, str) or not evaluator_id.strip():
            raise ValueError("evaluator_id must be non-empty")
        self._evaluator_id = evaluator_id.strip()
        if callable(reference):
            self.__reference_provider: Callable[[], Any] = reference
        else:
            self.__reference_provider = lambda value=reference: value
        self.__reference_access_count = 0
        self.__discarded = False
        self.__evaluated_episodes: set[str] = set()

    @property
    def evaluator_id(self) -> str:
        return self._evaluator_id

    @property
    def reference_access_count(self) -> int:
        """Expose only an audit count, never the reference or its digest."""

        return self.__reference_access_count

    def evaluate(
        self,
        submission: ParsedSubmission,
        *,
        state: EpisodeLifecycleState,
        task: PublicTask,
        episode_id: str,
    ) -> EvaluationResult:
        if state is not EpisodeLifecycleState.SUBMITTED:
            raise ReferenceAccessError(
                "private evaluation requires SUBMITTED episode state"
            )
        if not isinstance(submission, ParsedSubmission):
            raise PrivateEvaluationError("submission must be ParsedSubmission")
        if not isinstance(task, PublicTask):
            raise PrivateEvaluationError("task must be PublicTask")
        if not isinstance(episode_id, str) or not episode_id:
            raise PrivateEvaluationError("episode_id must be non-empty")
        if episode_id in self.__evaluated_episodes:
            raise PrivateEvaluationError("episode was already evaluated")
        if self.__discarded:
            raise PrivateEvaluationError("private reference was discarded")

        self.__reference_access_count += 1
        try:
            source = self.__reference_provider()
        except Exception:
            raise PrivateEvaluationError("private reference provider failed") from None
        reference = _extract_reference(source)
        result = self._compare(
            submission=submission,
            reference=reference,
            task=task,
            episode_id=episode_id,
        )
        self.__evaluated_episodes.add(episode_id)
        return result

    def discard_reference(self) -> None:
        """Release the provider after evaluation or environment close."""

        self.__reference_provider = lambda: None
        self.__discarded = True

    def _compare(
        self,
        *,
        submission: ParsedSubmission,
        reference: Any,
        task: PublicTask,
        episode_id: str,
    ) -> EvaluationResult:
        config = {
            "answer_format": task.answer_format,
            "choice_count": len(task.choices),
            "metric_family": "deterministic_exact",
        }
        provenance = {
            "evaluator_id": self.evaluator_id,
            "evaluator_version": EVALUATOR_VERSION,
            "implementation_hash": _IMPLEMENTATION_HASH,
            "config_hash": hashlib.sha256(
                json.dumps(
                    config,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "official_or_diagnostic": "diagnostic",
            "denominator_policy": "reference_available_submitted_rows_only",
            # A hash of a small answer space can disclose the label by enumeration.
            "reference_source_hash": None,
        }
        if reference is None:
            return EvaluationResult(
                episode_id=episode_id,
                benchmark_id=task.benchmark_id,
                sample_id=task.sample_id,
                metric="reference_unavailable",
                score=None,
                passed=None,
                prediction_normalized=_normalize_text(submission.answer),
                task_outcome="inconclusive",
                evidence_refs=submission.evidence_refs,
                score_kind="unscored",
                denominator_eligible=False,
                notes="private reference unavailable; no score produced",
                evaluator_provenance=provenance,
            )

        answer_type = task.answer_format.casefold()
        if answer_type in {"multiple_choice", "choice", "choice_exact"}:
            prediction = _extract_choice(submission.answer, len(task.choices))
            expected = _extract_choice(reference, len(task.choices))
            passed = bool(prediction and expected and prediction == expected)
            metric = "choice_exact"
        elif answer_type in {"numeric", "number", "numeric_exact"}:
            prediction_number = _first_number(submission.answer)
            expected_number = _first_number(reference)
            prediction = (
                "" if prediction_number is None else format(prediction_number, ".12g")
            )
            passed = (
                prediction_number is not None
                and expected_number is not None
                and math.isclose(
                    prediction_number,
                    expected_number,
                    rel_tol=1e-6,
                    abs_tol=1e-6,
                )
            )
            metric = "numeric_exact"
        else:
            prediction = _normalize_text(submission.answer)
            expected = _normalize_text(reference)
            passed = bool(prediction and expected and prediction == expected)
            metric = "normalized_exact_match"

        return EvaluationResult(
            episode_id=episode_id,
            benchmark_id=task.benchmark_id,
            sample_id=task.sample_id,
            metric=metric,
            score=1.0 if passed else 0.0,
            passed=passed,
            prediction_normalized=prediction,
            task_outcome="correct" if passed else "incorrect",
            evidence_refs=submission.evidence_refs,
            notes="deterministic CPU diagnostic; not an official benchmark score",
            evaluator_provenance=provenance,
        )


OfflineEvaluator = PrivateEvaluator


__all__ = [
    "EVALUATOR_ID",
    "EVALUATOR_VERSION",
    "OfflineEvaluator",
    "PrivateEvaluationError",
    "PrivateEvaluator",
    "ReferenceAccessError",
]
