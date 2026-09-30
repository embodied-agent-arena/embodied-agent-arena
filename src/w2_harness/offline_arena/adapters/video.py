"""CPU-only VSI-Bench adapter for the offline evidence arena.

The adapter treats the checked-in annotation as private/public split data and
never exposes ``ground_truth`` while an episode is active.  Video decoding is
optional.  When a materialized MP4 cannot be decoded, the deterministic
``materialized_frames/<dataset>/<scene>/frame_NN.jpg`` set is the authoritative
offline observation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import string
from typing import Any, Mapping, Protocol, Sequence

from PIL import Image

from .base import (
    ACTION_POLICY_SCHEMA_VERSION,
    ASSET_CATALOG_SCHEMA_VERSION,
    EVALUATION_SCHEMA_VERSION,
    MEDIA_PLAN_SCHEMA_VERSION,
    OBSERVATION_SCHEMA_VERSION,
    PUBLIC_TASK_SCHEMA_VERSION,
    AdapterError,
    AdapterSample,
    BaseAdapter,
    Choice,
    DataUnavailableError,
    InvalidSubmissionError,
    ParsedSubmission,
    SampleNotFoundError,
    normalize_mcq_answer,
    register_adapter,
)


_DATASETS = frozenset({"arkitscenes", "scannet", "scannetpp"})
_FRAME_NAME = re.compile(r"^frame_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_-]+$")
_FROZEN_MINIMUM = 3
_RAW_SAMPLE_SIZE = 3
_RAW_WINDOW_LIMIT = 8
_MISSING = object()


class VSIBenchError(AdapterError):
    """Base error for an unusable VSI-Bench offline operation."""


class VSIBenchDataError(VSIBenchError, DataUnavailableError):
    """Raised when annotation or media data violates the adapter contract."""

    def __init__(self, reason: str, *, path: Path | str | None = None) -> None:
        DataUnavailableError.__init__(
            self,
            "VSI-Bench",
            reason,
            path=path,
        )


class VSIBenchActionError(VSIBenchError):
    """Raised when a media action cannot be satisfied without guessing."""


class VSIBenchSubmissionError(InvalidSubmissionError):
    """Raised when an answer submission is malformed."""


class VideoReader(Protocol):
    """Small injectable CPU decoder boundary used only for raw MP4 access."""

    def probe(self, path: Path) -> Mapping[str, Any] | None: ...

    def extract(self, path: Path, frame_index: int, output: Path) -> None: ...


class _OpenCVVideoReader:
    """Best-effort CPU reader; an unavailable codec is a normal fallback."""

    def probe(self, path: Path) -> Mapping[str, Any] | None:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return None
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                return None
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_value = float(capture.get(cv2.CAP_PROP_FPS))
            if frame_count <= 0 or width <= 0 or height <= 0:
                return None
            # Opening a container is not enough: require one actual decoded frame.
            ok, _ = capture.read()
            if not ok:
                return None
            fps = fps_value if math.isfinite(fps_value) and fps_value > 0 else None
            return {
                "frame_count": frame_count,
                "width": width,
                "height": height,
                "fps": fps,
            }
        finally:
            capture.release()

    def extract(self, path: Path, frame_index: int, output: Path) -> None:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise VSIBenchDataError("CPU video decoder is unavailable") from exc
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise VSIBenchDataError("raw VSI video is not decodable")
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise VSIBenchDataError("raw VSI frame decode failed")
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
            if not cv2.imwrite(str(temporary), frame):
                raise VSIBenchDataError("decoded VSI frame could not be written")
            temporary.replace(output)
        finally:
            capture.release()


@dataclass(frozen=True)
class _PublicRow:
    sample_id: str
    dataset: str
    scene_name: str
    question_type: str
    question: str
    choices: tuple[str, ...]
    answer_format: str


@dataclass(frozen=True)
class _PrivateRow:
    ground_truth: str


@dataclass(frozen=True)
class _Frame:
    asset_id: str
    path: Path
    sha256: str
    width: int
    height: int
    frozen_ordinal: int | None
    source_frame_index: int | None
    timestamp_seconds: float | None
    frame_index_status: str
    timestamp_status: str
    selection_position: str
    source_kind: str


@dataclass(frozen=True)
class _SceneMedia:
    dataset: str
    scene_name: str
    mode: str
    video_path: Path | None
    frozen_paths: tuple[Path, ...]
    frame_count: int | None
    fps: float | None
    width: int | None
    height: int | None
    raw_video_status: str


@dataclass(frozen=True)
class _LoadedSample:
    sample: AdapterSample
    row: _PublicRow
    media: _SceneMedia
    frames: tuple[_Frame, ...]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _asset_id(
    *,
    dataset: str,
    scene_name: str,
    source_kind: str,
    position: int | str,
    content_sha256: str,
) -> str:
    identity = {
        "benchmark_id": "VSI-Bench",
        "dataset": dataset,
        "scene_name": scene_name,
        "source_kind": source_kind,
        "position": position,
        "content_sha256": content_sha256,
    }
    return "asset_" + _canonical_hash(identity)[:32]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _uniform_indices(
    frame_count: int,
    count: int = _RAW_SAMPLE_SIZE,
) -> tuple[int, ...]:
    if frame_count <= 0:
        return ()
    if frame_count <= count:
        return tuple(range(frame_count))
    return tuple(
        round(position * (frame_count - 1) / (count - 1))
        for position in range(count)
    )


def _position_label(index: int, count: int) -> str:
    if count == 3:
        return ("first", "middle", "last")[index]
    return f"uniform_{index:02d}_of_{count:02d}"


def _choice_letter(value: Any, choice_count: int) -> str | None:
    text = "" if value is None else str(value).strip()
    match = re.match(r"^\s*([A-Z])(?:\s*[.:)]|\s*$)", text, re.IGNORECASE)
    if not match:
        return None
    letter = match.group(1).upper()
    return letter if ord(letter) - ord("A") < choice_count else None


def _finite_number(value: Any) -> float | None:
    match = re.search(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
        "" if value is None else str(value),
    )
    if not match:
        return None
    try:
        number = float(match.group(0))
    except ValueError:
        return None
    return number if math.isfinite(number) else None


@register_adapter
class VSIBenchAdapter(BaseAdapter):
    """Unified, state-independent adapter for the executable VSI subset."""

    benchmark_id = "VSI-Bench"
    source_runner_id = "VSI-Bench"
    package_id = "vsi_bench"
    result_benchmark_id = "VSI-Bench"
    aliases = ("VSI", "VSI-Bench-U")
    score_scope = "diagnostic"
    default_observation_policy = "selective_video_evidence"
    observation_policy = default_observation_policy
    initial_asset_limit = 1

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        cache_root: Path | str | None = None,
        video_reader: VideoReader | None = None,
        raw_video_root: Path | str | None = None,
        raw_sample_size: int = _RAW_SAMPLE_SIZE,
        sample_allowlist: Sequence[str | int] | None = None,
    ) -> None:
        super().__init__(data_root, cache_root=cache_root)
        if (
            isinstance(raw_sample_size, bool)
            or not isinstance(raw_sample_size, int)
            or not 1 <= raw_sample_size <= _RAW_WINDOW_LIMIT
        ):
            raise ValueError(
                f"raw_sample_size must be an integer from 1 to {_RAW_WINDOW_LIMIT}"
            )
        self.video_reader = video_reader or _OpenCVVideoReader()
        self.raw_video_root = (
            None
            if raw_video_root is None
            else Path(raw_video_root).expanduser().resolve()
        )
        if self.raw_video_root is not None and not self.raw_video_root.is_dir():
            raise ValueError("raw_video_root must be an existing directory")
        self.raw_sample_size = raw_sample_size
        self.sample_allowlist = (
            None
            if sample_allowlist is None
            else frozenset(str(value) for value in sample_allowlist)
        )
        if self.sample_allowlist is not None and (
            not self.sample_allowlist
            or any(not value.isdigit() for value in self.sample_allowlist)
        ):
            raise ValueError("sample_allowlist must contain numeric VSI sample IDs")
        self._public_rows: dict[str, _PublicRow] = {}
        self._private_rows: dict[str, _PrivateRow] = {}
        self._eligible_ids: tuple[str, ...] = ()
        self._scene_media: dict[tuple[str, str], _SceneMedia | None] = {}
        self._frame_cache: dict[tuple[str, str, int], _Frame] = {}
        self._asset_paths: dict[str, Path] = {}
        self._loaded_samples: dict[str, _LoadedSample] = {}
        self._load_index()

    # Shared BaseAdapter surface ---------------------------------------------
    def load_sample(
        self,
        sample_id: str | int | Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        self._ensure_open()
        if isinstance(sample_id, Mapping):
            task_id = str(sample_id.get("task_id") or "")
            if task_id in self._loaded_samples:
                return self._loaded_samples[task_id].sample
            for key in ("source_sample_id", "sample_id", "id", "index"):
                if sample_id.get(key) is not None:
                    sample_id = str(sample_id[key])
                    break
            else:
                raise SampleNotFoundError("VSI sample mapping has no source ID")
        source_id = (
            self._eligible_ids[0]
            if sample_id is None and self._eligible_ids
            else str(sample_id)
        )
        if not source_id:
            raise VSIBenchDataError("VSI executable denominator is empty")
        existing_task_id = self._source_to_task.get(source_id)
        if existing_task_id in self._loaded_samples:
            return self._loaded_samples[existing_task_id].sample

        row, media = self._eligible_row(source_id)
        frames = self._full_context_frames(media)
        media = self._scene_media[(row.dataset, row.scene_name)] or media
        choices: list[Choice] = []
        for index, raw in enumerate(row.choices):
            match = re.match(r"^([A-Z])\s*[.:)]\s*(.+)$", raw, re.IGNORECASE)
            choices.append(
                Choice(
                    label=(
                        match.group(1).upper()
                        if match is not None
                        else string.ascii_uppercase[index]
                    ),
                    text=match.group(2).strip() if match is not None else raw,
                )
            )
        task_id = "task_" + _canonical_hash(
            {"benchmark_id": self.benchmark_id, "source_sample_id": source_id}
        )[:32]
        sample = AdapterSample(
            task_id=task_id,
            source_sample_id=source_id,
            prompt=row.question,
            choices=tuple(choices),
            asset_ids=tuple(frame.asset_id for frame in frames),
            metadata={
                "dataset": row.dataset,
                "scene_name": row.scene_name,
                "question_type": row.question_type,
                "answer_format": row.answer_format,
                "media_mode": media.mode,
                "raw_video_status": media.raw_video_status,
                "selection_basis": "scene_media_only",
                "source_frame_index_status": (
                    "decoder_reported"
                    if media.mode == "raw_video"
                    else "unknown_not_persisted"
                ),
                "source_timestamp_status": (
                    "derived_from_decoder_fps"
                    if media.mode == "raw_video" and media.fps is not None
                    else (
                        "unknown_decoder_fps"
                        if media.mode == "raw_video"
                        else "unknown_not_persisted"
                    )
                ),
            },
        )
        self._loaded_samples[task_id] = _LoadedSample(
            sample=sample,
            row=row,
            media=media,
            frames=frames,
        )
        self._source_to_task[source_id] = task_id
        self._current_task_id = task_id
        return sample

    def build_public_task(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        return {
            "schema_version": PUBLIC_TASK_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "prompt": state.sample.prompt,
            "choices": [choice.to_dict() for choice in state.sample.choices],
            "answer_type": state.row.answer_format,
            "asset_ids": list(state.sample.asset_ids),
        }

    def build_asset_catalog(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        return {
            "schema_version": ASSET_CATALOG_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "assets": [
                self._base_asset_record(frame, index, include_path=False)
                for index, frame in enumerate(state.frames)
            ],
        }

    def build_action_policy(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        frame_semantics = (
            "source_frame_index"
            if state.media.mode == "raw_video"
            else "frozen_frame_ordinal"
        )
        return {
            "schema_version": ACTION_POLICY_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "policy": self.observation_policy,
            "allowed_actions": ["inspect", "submit_answer"],
            "inspect": {
                "required_arguments": ["operation"],
                "operations": ["GET_FRAME", "GET_FRAME_WINDOW"],
                "asset_ids": list(state.sample.asset_ids),
                "frame_index_semantics": frame_semantics,
                "timestamp_policy": (
                    "known_decoder_fps"
                    if state.media.mode == "raw_video" and state.media.fps is not None
                    else (
                        "unavailable_unknown_decoder_fps"
                        if state.media.mode == "raw_video"
                        else "unavailable_unknown_not_persisted"
                    )
                ),
                "get_frame_window_end": "inclusive",
                "max_calls": 4,
                "read_only": True,
            },
            "turn_limit": 5,
        }

    def build_full_context_media_plan(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        return {
            "schema_version": MEDIA_PLAN_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "strategy": "deterministic_full_context_frame_sample",
            "selection_basis": "scene_media_only",
            "media": [
                self._base_asset_record(frame, index, include_path=True)
                for index, frame in enumerate(state.frames)
            ],
        }

    def build_selective_initial_observation(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        preview = self._contact_sheet(state.media, state.frames)
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "observation_policy": self.observation_policy,
            "observed_asset_ids": [preview.asset_id],
            "remaining_asset_ids": list(state.sample.asset_ids),
            "observation_complete": False,
            "preview": self._base_asset_record(preview, 0, include_path=True),
            "next_media_operations": ["GET_FRAME", "GET_FRAME_WINDOW"],
            "selection_basis": "scene_media_only",
        }

    def build_submission_schema(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._loaded_state(sample)
        answer: dict[str, Any]
        if state.sample.choices:
            answer = {
                "type": "string",
                "enum": [choice.label for choice in state.sample.choices],
            }
        else:
            answer = {
                "oneOf": [
                    {"type": "number"},
                    {
                        "type": "string",
                        "pattern": r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*$",
                    },
                ]
            }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "VSI-Bench diagnostic submission",
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {
                "answer": answer,
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "uniqueItems": True,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
        }

    def parse_submission(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | Any,
        submission: Any = _MISSING,
    ) -> ParsedSubmission:
        if submission is _MISSING:
            raw = sample
            state = self._loaded_state(None)
        else:
            raw = submission
            state = self._loaded_state(sample)  # type: ignore[arg-type]
        if isinstance(raw, Mapping):
            allowed = {"answer", "evidence_refs", "confidence"}
            if set(raw) - allowed:
                raise VSIBenchSubmissionError("submission contains unsupported fields")
            answer = raw.get("answer")
            raw_refs = raw.get("evidence_refs", ())
            confidence = raw.get("confidence", _MISSING)
        else:
            answer = raw
            raw_refs = ()
            confidence = _MISSING
        if confidence is not _MISSING and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise VSIBenchSubmissionError(
                "confidence must be a finite number between zero and one"
            )
        if isinstance(raw_refs, (str, bytes)) or not isinstance(raw_refs, Sequence):
            raise VSIBenchSubmissionError("evidence_refs must be an array")
        if any(not isinstance(value, str) for value in raw_refs):
            raise VSIBenchSubmissionError("evidence_refs entries must be strings")
        evidence_refs = tuple(value.strip() for value in raw_refs)
        if any(not value for value in evidence_refs) or len(evidence_refs) != len(
            set(evidence_refs)
        ):
            raise VSIBenchSubmissionError(
                "evidence_refs must be unique non-empty strings"
            )
        if state.sample.choices:
            normalized = normalize_mcq_answer(answer, state.sample.choices)
            issue = "unrecognized_multiple_choice"
        else:
            number = _finite_number(answer)
            normalized = None if number is None else format(number, ".12g")
            issue = "unrecognized_numeric_answer"
        try:
            raw_payload = json.dumps(
                raw,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            )
        except (TypeError, ValueError):
            raw_payload = repr(raw)
        return ParsedSubmission(
            task_id=state.sample.task_id,
            answer=normalized,
            valid=normalized is not None,
            issues=() if normalized is not None else (issue,),
            raw_sha256=hashlib.sha256(raw_payload.encode("utf-8")).hexdigest(),
            _adapter_capability=self._submission_capability,
        )

    def evaluate_private(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | ParsedSubmission,
        submission: ParsedSubmission | object = _MISSING,
    ) -> dict[str, Any]:
        if submission is _MISSING:
            parsed = sample
            state = self._loaded_state(None)
        else:
            parsed = submission
            state = self._loaded_state(sample)  # type: ignore[arg-type]
        if not isinstance(parsed, ParsedSubmission):
            raise VSIBenchSubmissionError(
                "evaluate_private requires the result of parse_submission"
            )
        if parsed._adapter_capability is not self._submission_capability:
            raise VSIBenchSubmissionError(
                "submission belongs to another adapter instance"
            )
        if parsed.task_id != state.sample.task_id:
            raise VSIBenchSubmissionError("submission and sample task IDs differ")

        # Private truth is consulted only after the capability checks above.
        truth = self._private_rows[state.row.sample_id].ground_truth
        if state.sample.choices:
            expected = _choice_letter(truth, len(state.sample.choices))
            metric = "multiple_choice_accuracy"
            correct = bool(parsed.valid and parsed.answer == expected)
        else:
            expected_number = _finite_number(truth)
            predicted_number = _finite_number(parsed.answer)
            metric = "numeric_exact"
            correct = bool(
                parsed.valid
                and expected_number is not None
                and predicted_number is not None
                and predicted_number == expected_number
            )
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "metric": metric,
            "score_scope": "diagnostic",
            "submission_valid": parsed.valid,
            "normalized_answer": parsed.answer,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "official_score": None,
        }

    def _loaded_state(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None,
    ) -> _LoadedSample:
        self._ensure_open()
        if sample is None:
            task_id = self._current_task_id
        elif isinstance(sample, AdapterSample):
            task_id = sample.task_id
        elif isinstance(sample, Mapping):
            task_id = str(sample.get("task_id") or "")
        else:
            value = str(sample)
            task_id = (
                value
                if value in self._loaded_samples
                else self._source_to_task.get(value)
            )
        if not task_id or task_id not in self._loaded_samples:
            raise SampleNotFoundError("VSI sample is not loaded by this adapter")
        return self._loaded_samples[task_id]

    def _base_asset_record(
        self,
        frame: _Frame,
        sequence_index: int,
        *,
        include_path: bool,
    ) -> dict[str, Any]:
        value = {
            "asset_id": frame.asset_id,
            "uri": f"asset://{frame.asset_id}",
            "media_type": "image",
            "mime_type": (
                "image/png"
                if frame.path.suffix.casefold() == ".png"
                else "image/jpeg"
            ),
            "sequence_index": sequence_index,
            "byte_size": frame.path.stat().st_size,
            "content_sha256": frame.sha256,
            "frame": self._frame_metadata(frame),
        }
        if include_path:
            value["local_path"] = str(frame.path)
        return value

    def _close_resources(self) -> None:
        self._loaded_samples.clear()
        self._frame_cache.clear()
        self._asset_paths.clear()

    # Component/schema adapter surface ---------------------------------------
    def discover_samples(self) -> tuple[str, ...]:
        """Return the deterministic diagnostic denominator, sorted by sample id."""

        self._ensure_open()
        return self._eligible_ids

    def sample_status(self, sample_id: str | int) -> str:
        key = str(sample_id)
        if key in self._eligible_ids:
            return "ready"
        if key in self._public_rows:
            return "data_missing"
        raise SampleNotFoundError("VSI sample is outside the released annotation")

    def denominator_report(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "released_rows": len(self._public_rows),
            "executable_denominator": len(self._eligible_ids),
            "data_missing": len(self._public_rows) - len(self._eligible_ids),
            "other_excluded": 0,
            "denominator_policy": (
                "materialized_rows_with_readable_video_or_frozen_frames"
            ),
            "score_scope": "diagnostic",
            "official_evaluator_used": False,
            "official_score": None,
        }

    def load_public_task(self, sample_id: str | int) -> dict[str, Any]:
        row, media = self._eligible_row(sample_id)
        frames = self._full_context_frames(media)
        media = self._scene_media[(row.dataset, row.scene_name)] or media
        return {
            "benchmark_id": self.benchmark_id,
            "sample_id": row.sample_id,
            "question": row.question,
            "choices": list(row.choices),
            "answer_format": row.answer_format,
            "public_metadata": {
                "dataset": row.dataset,
                "scene_name": row.scene_name,
                "question_type": row.question_type,
                "split": "test",
                "media_mode": media.mode,
                "raw_video_status": media.raw_video_status,
                "full_context_frame_count": len(frames),
                "frame_selection": "deterministic_first_middle_last_or_uniform",
                "selection_basis": "scene_media_only",
                "source_frame_index_status": (
                    "decoder_reported"
                    if media.mode == "raw_video"
                    else "unknown_not_persisted"
                ),
                "source_timestamp_status": (
                    "derived_from_decoder_fps"
                    if media.mode == "raw_video" and media.fps is not None
                    else (
                        "unknown_decoder_fps"
                        if media.mode == "raw_video"
                        else "unknown_not_persisted"
                    )
                ),
            },
            "asset_ids": [frame.asset_id for frame in frames],
            "observation_policy": self.default_observation_policy,
            "category": row.question_type,
            "submission_schema_id": (
                "w2.answer.choice.v1"
                if row.choices
                else "w2.answer.numeric.v1"
            ),
        }

    def materialize_sample(self, sample_id: str | int) -> dict[str, Any]:
        sample = self.load_sample(sample_id)
        return {
            "public_task": self.load_public_task(sample.source_sample_id),
            "assets": list(self.list_assets(sample.source_sample_id)),
            "execution_status": "ready",
            "denominator_eligible": True,
            "score_scope": "diagnostic",
            "official_score": None,
        }

    materialize = materialize_sample
    materialize_task = materialize_sample

    def get_public_task(self, sample_id: str | int) -> dict[str, Any]:
        return self.load_public_task(sample_id)

    def list_assets(self, sample_id: str | int) -> tuple[dict[str, Any], ...]:
        _, media = self._eligible_row(sample_id)
        return tuple(
            self._frame_descriptor(frame)
            for frame in self._full_context_frames(media)
        )

    def initial_observation_plan(
        self,
        sample_id: str | int,
        *,
        profile: str = "w2_light",
    ) -> dict[str, Any]:
        _, media = self._eligible_row(sample_id)
        frames = self._full_context_frames(media)
        if profile == "direct":
            return {
                "policy": "deterministic_full_context_frame_sample",
                "selection_basis": "scene_media_only",
                "visible_assets": [self._frame_descriptor(frame) for frame in frames],
                "available_actions": [],
            }
        if profile != "w2_light":
            raise VSIBenchActionError("unsupported offline harness profile")
        contact_sheet = self._contact_sheet(media, frames)
        return {
            "policy": "selective_contact_sheet_then_frame_access",
            "selection_basis": "scene_media_only",
            "visible_assets": [self._frame_descriptor(contact_sheet)],
            "available_actions": ["GET_FRAME", "GET_FRAME_WINDOW"],
            "preview_frame_asset_ids": [frame.asset_id for frame in frames],
        }

    def action_policy(
        self,
        sample_id: str | int | None = None,
        *,
        profile: str = "w2_light",
        state_revision: int = 0,
    ) -> dict[str, Any]:
        if sample_id is None:
            if self._current_task_id in self._loaded_samples:
                sample_id = self._loaded_samples[self._current_task_id].row.sample_id
            elif self._eligible_ids:
                sample_id = self._eligible_ids[0]
            else:
                raise VSIBenchDataError("VSI executable denominator is empty")
        self._eligible_row(sample_id)
        if profile not in {"direct", "w2_light"}:
            raise VSIBenchActionError("unsupported offline harness profile")
        if not _is_int(state_revision) or state_revision < 0:
            raise VSIBenchActionError("state_revision must be a non-negative integer")
        allowed = ["GET_TASK_CONTEXT", "LIST_ASSETS", "SUBMIT"]
        if profile == "w2_light":
            allowed[2:2] = ["GET_FRAME", "GET_FRAME_WINDOW"]
        return {
            "allowed_actions": allowed,
            "action_schemas": {
                "GET_FRAME": {
                    "one_of": ["frame_index", "timestamp_seconds"],
                    "frame_index_semantics": (
                        "source frame index for decoded video; "
                        "frozen ordinal for fallback"
                    ),
                    "timestamp_policy": "requires_known_decoder_fps",
                },
                "GET_FRAME_WINDOW": {
                    "required": ["start_frame_index", "end_frame_index"],
                    "end_inclusive": True,
                    "maximum_frames": _RAW_WINDOW_LIMIT,
                    "frozen_fallback_maximum": "available_frozen_frame_count",
                },
            },
            "media_operation_ceiling": 4 if profile == "w2_light" else 0,
            "duplicate_action_policy": "content_identity_cache",
            "current_profile": profile,
            "state_revision": state_revision,
        }

    get_action_policy = action_policy

    def execute_action(
        self,
        sample_id: str | int,
        action: str | Mapping[str, Any],
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        row, media = self._eligible_row(sample_id)
        action_name, values = self._action_envelope(action, arguments)
        if action_name == "GET_TASK_CONTEXT":
            if values:
                raise VSIBenchActionError("GET_TASK_CONTEXT takes no arguments")
            return {"action": action_name, "task": self.load_public_task(row.sample_id)}
        if action_name == "LIST_ASSETS":
            if values:
                raise VSIBenchActionError("LIST_ASSETS takes no arguments")
            return {
                "action": action_name,
                "assets": list(self.list_assets(row.sample_id)),
            }
        if action_name == "GET_FRAME":
            frame = self._get_frame_action(media, values)
            return {
                "action": action_name,
                "asset": self._frame_descriptor(frame),
                "frame": self._frame_metadata(frame),
                "selection_basis": "agent_request_only",
            }
        if action_name == "GET_FRAME_WINDOW":
            frames = self._get_window_action(media, values)
            return {
                "action": action_name,
                "assets": [self._frame_descriptor(frame) for frame in frames],
                "frames": [self._frame_metadata(frame) for frame in frames],
                "selection_basis": "agent_request_only",
                "end_inclusive": True,
            }
        raise VSIBenchActionError("unsupported VSI-Bench action")

    def evaluate_submission(
        self,
        sample_id: str | int,
        submission: ParsedSubmission,
    ) -> dict[str, Any]:
        """Run the private diagnostic only after a parsed submission exists."""

        loaded = self.load_sample(sample_id)
        evaluation = self.evaluate_private(loaded, submission)
        valid = bool(evaluation["submission_valid"])
        correct = bool(evaluation["correct"])
        return {
            "execution_success": True,
            "contract_valid": True,
            "submission_valid": valid,
            "task_outcome": (
                "correct" if correct else "incorrect" if valid else "not_evaluated"
            ),
            "score_kind": "diagnostic",
            "official_score": None,
            "denominator_eligible": True,
            "error_type": None if valid else "invalid_answer_format",
            "blocked_reason": None,
        }

    def submit(self, sample_id: str | int, raw_submission: Any) -> dict[str, Any]:
        loaded = self.load_sample(sample_id)
        parsed = self.parse_submission(loaded, raw_submission)
        return self.evaluate_submission(sample_id, parsed)

    # Compatibility aliases kept deliberately thin for the shared harness.
    def list_sample_ids(self) -> tuple[str, ...]:
        return self.discover_samples()

    sample_ids = list_sample_ids

    def load_task(self, sample_id: str | int) -> dict[str, Any]:
        return self.load_public_task(sample_id)

    def get_initial_observation(
        self, sample_id: str | int, *, profile: str = "w2_light"
    ) -> dict[str, Any]:
        return self.initial_observation_plan(sample_id, profile=profile)

    def handle_action(
        self,
        sample_id: str | int,
        action: str | Mapping[str, Any],
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.execute_action(sample_id, action, arguments)

    def evaluate(
        self, sample_id: str | int, submission: ParsedSubmission
    ) -> dict[str, Any]:
        return self.evaluate_submission(sample_id, submission)

    def resolve_asset_path(self, asset_id: str) -> Path:
        """Resolve runtime media internally without putting paths in public tasks."""

        try:
            return self._asset_paths[asset_id]
        except KeyError as exc:
            raise VSIBenchActionError("unknown or unrevealed asset id") from exc

    resolve_asset = resolve_asset_path

    def read_asset(self, asset_id: str) -> bytes:
        return self.resolve_asset_path(asset_id).read_bytes()

    # Data loading ------------------------------------------------------------
    def _load_index(self) -> None:
        annotation = self.data_root / "dataset" / "test.jsonl"
        if not self.data_root.is_dir() or not annotation.is_file():
            raise VSIBenchDataError("VSI-Bench dataset/test.jsonl is unavailable")
        try:
            annotation.resolve().relative_to(self.data_root)
        except ValueError as exc:
            raise VSIBenchDataError("VSI-Bench annotation escapes data_root") from exc

        scene_keys: set[tuple[str, str]] = set()
        try:
            with annotation.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise VSIBenchDataError(
                            f"invalid VSI annotation JSON at line {line_number}"
                        ) from exc
                    public, private = self._parse_row(raw, line_number)
                    if public.sample_id in self._public_rows:
                        raise VSIBenchDataError("duplicate VSI sample id")
                    self._public_rows[public.sample_id] = public
                    self._private_rows[public.sample_id] = private
                    if (
                        self.sample_allowlist is None
                        or public.sample_id in self.sample_allowlist
                    ):
                        scene_keys.add((public.dataset, public.scene_name))
        except (OSError, UnicodeError) as exc:
            raise VSIBenchDataError("VSI annotation cannot be read") from exc
        if not self._public_rows:
            raise VSIBenchDataError("VSI annotation is empty")

        for key in sorted(scene_keys):
            self._scene_media[key] = self._inspect_scene(*key)
        eligible = [
            sample_id
            for sample_id, row in self._public_rows.items()
            if (
                (self.sample_allowlist is None or sample_id in self.sample_allowlist)
                and self._scene_media.get((row.dataset, row.scene_name)) is not None
            )
        ]
        if self.sample_allowlist is not None and set(eligible) != set(
            self.sample_allowlist
        ):
            missing = len(self.sample_allowlist - set(eligible))
            raise VSIBenchDataError(
                f"VSI sample allowlist contains {missing} unavailable sample(s)"
            )
        self._eligible_ids = tuple(
            sorted(eligible, key=lambda value: (int(value), value))
        )

    def _parse_row(
        self, raw: Any, line_number: int
    ) -> tuple[_PublicRow, _PrivateRow]:
        if not isinstance(raw, Mapping):
            raise VSIBenchDataError(f"VSI row {line_number} must be an object")
        required = {
            "id",
            "dataset",
            "scene_name",
            "question_type",
            "question",
            "ground_truth",
            "options",
        }
        if not required.issubset(raw):
            raise VSIBenchDataError(f"VSI row {line_number} lacks required fields")
        sample_id = str(raw["id"])
        if not sample_id.isdigit():
            raise VSIBenchDataError("VSI sample id must be a non-negative integer")
        dataset = str(raw["dataset"]).casefold()
        scene_name = str(raw["scene_name"])
        if dataset not in _DATASETS:
            raise VSIBenchDataError("unknown VSI source dataset")
        if not _SAFE_COMPONENT.fullmatch(scene_name):
            raise VSIBenchDataError("unsafe VSI scene name")
        question_type = str(raw["question_type"]).strip()
        question = str(raw["question"]).strip()
        ground_truth = str(raw["ground_truth"]).strip()
        if not question_type or not question or not ground_truth:
            raise VSIBenchDataError(
                "VSI task text and private answer must be non-empty"
            )
        options = raw["options"]
        if options is None:
            choices: tuple[str, ...] = ()
            if _finite_number(ground_truth) is None:
                raise VSIBenchDataError("open VSI answers must be numeric")
            answer_format = "numeric"
        elif (
            isinstance(options, list)
            and 1 < len(options) <= len(string.ascii_uppercase)
        ):
            normalized: list[str] = []
            for index, value in enumerate(options):
                if not isinstance(value, str):
                    raise VSIBenchDataError("VSI choices must be strings")
                choice = value.strip()
                if not choice:
                    raise VSIBenchDataError("VSI choices must be non-empty")
                expected_label = string.ascii_uppercase[index]
                match = re.match(
                    r"^([A-Z])\s*[.:)]\s*(.+)$",
                    choice,
                    re.IGNORECASE,
                )
                if match is not None:
                    if match.group(1).upper() != expected_label:
                        raise VSIBenchDataError(
                            "VSI choice labels must match option order"
                        )
                    choice = f"{expected_label}. {match.group(2).strip()}"
                else:
                    choice = f"{expected_label}. {choice}"
                normalized.append(choice)
            choices = tuple(normalized)
            if _choice_letter(ground_truth, len(choices)) is None:
                raise VSIBenchDataError("VSI choice answer is outside the option set")
            answer_format = "multiple_choice"
        else:
            raise VSIBenchDataError("VSI options must be null or an option array")
        return (
            _PublicRow(
                sample_id=sample_id,
                dataset=dataset,
                scene_name=scene_name,
                question_type=question_type,
                question=question,
                choices=choices,
                answer_format=answer_format,
            ),
            _PrivateRow(ground_truth=ground_truth),
        )

    def _inspect_scene(self, dataset: str, scene_name: str) -> _SceneMedia | None:
        video = self._source_video_path(dataset, scene_name)
        raw_video_status = "raw_video_absent_frozen_frames_authoritative"
        if video.is_file():
            raw_video_status = "decode_unavailable_frozen_frames_authoritative"
            try:
                probe = self.video_reader.probe(video)
            except Exception:  # Decoder failures are expected; frozen data is next.
                probe = None
            if probe is not None:
                try:
                    frame_count = int(probe["frame_count"])
                    width = int(probe["width"])
                    height = int(probe["height"])
                    fps_raw = probe.get("fps")
                    fps = None if fps_raw is None else float(fps_raw)
                except (KeyError, TypeError, ValueError):
                    probe = None
                else:
                    if frame_count > 0 and width > 0 and height > 0 and (
                        fps is None or (math.isfinite(fps) and fps > 0)
                    ):
                        candidate = _SceneMedia(
                            dataset=dataset,
                            scene_name=scene_name,
                            mode="raw_video",
                            video_path=video,
                            frozen_paths=self._frozen_paths(dataset, scene_name),
                            frame_count=frame_count,
                            fps=fps,
                            width=width,
                            height=height,
                            raw_video_status="cpu_decodable",
                        )
                        indices = _uniform_indices(
                            frame_count, count=self.raw_sample_size
                        )
                        try:
                            bulk_extract = getattr(
                                self.video_reader, "extract_many", None
                            )
                            if callable(bulk_extract):
                                outputs = tuple(
                                    self._raw_frame_path(candidate, frame_index)
                                    for frame_index in indices
                                )
                                missing = tuple(
                                    (frame_index, output)
                                    for frame_index, output in zip(
                                        indices, outputs, strict=True
                                    )
                                    if not output.is_file()
                                )
                                if missing:
                                    bulk_extract(
                                        candidate.video_path,
                                        tuple(item[0] for item in missing),
                                        tuple(item[1] for item in missing),
                                    )
                            for slot, frame_index in enumerate(indices):
                                self._raw_frame(
                                    candidate,
                                    frame_index,
                                    _position_label(slot, len(indices)),
                                )
                        except VSIBenchDataError:
                            raw_video_status = (
                                "decode_failed_frozen_frames_authoritative"
                            )
                        else:
                            return candidate

        frozen = self._frozen_paths(dataset, scene_name)
        if len(frozen) < _FROZEN_MINIMUM:
            return None
        return _SceneMedia(
            dataset=dataset,
            scene_name=scene_name,
            mode="frozen_frames",
            video_path=video if video.is_file() else None,
            frozen_paths=frozen,
            frame_count=None,
            fps=None,
            width=None,
            height=None,
            raw_video_status=raw_video_status,
        )

    def _source_video_path(self, dataset: str, scene_name: str) -> Path:
        if self.raw_video_root is not None:
            return (
                self.raw_video_root
                / dataset
                / dataset
                / f"{scene_name}.mp4"
            )
        return self.data_root / "materialized_videos" / dataset / f"{scene_name}.mp4"

    def _frozen_paths(self, dataset: str, scene_name: str) -> tuple[Path, ...]:
        directory = self.data_root / "materialized_frames" / dataset / scene_name
        if not directory.is_dir():
            return ()
        indexed: dict[int, Path] = {}
        for path in sorted(directory.iterdir()):
            if not path.is_file():
                continue
            match = _FRAME_NAME.fullmatch(path.name)
            if match is None:
                continue
            ordinal = int(match.group(1))
            if ordinal in indexed:
                return ()
            indexed[ordinal] = path
        if not indexed or sorted(indexed) != list(range(len(indexed))):
            return ()
        paths = tuple(indexed[index] for index in range(len(indexed)))
        try:
            for path in paths:
                self._inspect_image(path)
        except VSIBenchDataError:
            return ()
        return paths

    @staticmethod
    def _inspect_image(path: Path) -> tuple[int, int]:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
                image.load()
        except (OSError, ValueError) as exc:
            raise VSIBenchDataError("VSI materialized frame is unreadable") from exc
        if width <= 0 or height <= 0:
            raise VSIBenchDataError("VSI materialized frame has invalid dimensions")
        return width, height

    # Media plans and actions -------------------------------------------------
    def _eligible_row(self, sample_id: str | int) -> tuple[_PublicRow, _SceneMedia]:
        key = str(sample_id)
        try:
            row = self._public_rows[key]
        except KeyError as exc:
            raise SampleNotFoundError("unknown VSI sample id") from exc
        if self.sample_allowlist is not None and key not in self.sample_allowlist:
            raise SampleNotFoundError("VSI sample is outside the configured scope")
        media = self._scene_media.get((row.dataset, row.scene_name))
        if media is None:
            raise VSIBenchDataError(
                "VSI sample lacks readable video or complete frozen frames"
            )
        return row, media

    def _full_context_frames(self, media: _SceneMedia) -> tuple[_Frame, ...]:
        if media.mode == "frozen_frames":
            return tuple(
                self._frozen_frame(media, index)
                for index in range(len(media.frozen_paths))
            )
        assert media.frame_count is not None
        indices = _uniform_indices(
            media.frame_count, count=self.raw_sample_size
        )
        try:
            return tuple(
                self._raw_frame(media, index, _position_label(slot, len(indices)))
                for slot, index in enumerate(indices)
            )
        except VSIBenchDataError:
            if len(media.frozen_paths) < _FROZEN_MINIMUM:
                raise
            fallback = _SceneMedia(
                dataset=media.dataset,
                scene_name=media.scene_name,
                mode="frozen_frames",
                video_path=media.video_path,
                frozen_paths=media.frozen_paths,
                frame_count=None,
                fps=None,
                width=None,
                height=None,
                raw_video_status="decode_failed_frozen_frames_authoritative",
            )
            self._scene_media[(media.dataset, media.scene_name)] = fallback
            return tuple(
                self._frozen_frame(fallback, index)
                for index in range(len(fallback.frozen_paths))
            )

    def _frozen_frame(self, media: _SceneMedia, ordinal: int) -> _Frame:
        key = (media.dataset, media.scene_name, ordinal)
        cached = self._frame_cache.get(key)
        if cached is not None and cached.frozen_ordinal is not None:
            return cached
        try:
            path = media.frozen_paths[ordinal]
        except IndexError as exc:
            raise VSIBenchActionError("frozen frame index is out of range") from exc
        width, height = self._inspect_image(path)
        sha256 = _sha256_file(path)
        frame = _Frame(
            asset_id=_asset_id(
                dataset=media.dataset,
                scene_name=media.scene_name,
                source_kind="frozen_frame",
                position=ordinal,
                content_sha256=sha256,
            ),
            path=path,
            sha256=sha256,
            width=width,
            height=height,
            frozen_ordinal=ordinal,
            source_frame_index=None,
            timestamp_seconds=None,
            frame_index_status="unknown_not_persisted",
            timestamp_status="unknown_not_persisted",
            selection_position=_position_label(ordinal, len(media.frozen_paths)),
            source_kind="benchmark_materialized_frame",
        )
        self._frame_cache[key] = frame
        self._asset_paths[frame.asset_id] = path
        return frame

    def _raw_frame(
        self, media: _SceneMedia, frame_index: int, selection_position: str
    ) -> _Frame:
        if media.video_path is None or media.frame_count is None:
            raise VSIBenchDataError("raw VSI video metadata is unavailable")
        if frame_index < 0 or frame_index >= media.frame_count:
            raise VSIBenchActionError("source frame index is out of range")
        key = (media.dataset, media.scene_name, frame_index)
        cached = self._frame_cache.get(key)
        if cached is not None and cached.source_frame_index is not None:
            return cached
        path = self._raw_frame_path(media, frame_index)
        if not path.is_file():
            try:
                self.video_reader.extract(media.video_path, frame_index, path)
            except VSIBenchDataError:
                raise
            except Exception as exc:
                raise VSIBenchDataError("raw VSI frame decode failed") from exc
        width, height = self._inspect_image(path)
        sha256 = _sha256_file(path)
        timestamp = None if media.fps is None else frame_index / media.fps
        frame = _Frame(
            asset_id=_asset_id(
                dataset=media.dataset,
                scene_name=media.scene_name,
                source_kind="decoded_video_frame",
                position=frame_index,
                content_sha256=sha256,
            ),
            path=path,
            sha256=sha256,
            width=width,
            height=height,
            frozen_ordinal=None,
            source_frame_index=frame_index,
            timestamp_seconds=timestamp,
            frame_index_status="decoder_reported",
            timestamp_status=(
                "derived_from_decoder_fps"
                if timestamp is not None
                else "unknown_decoder_fps"
            ),
            selection_position=selection_position,
            source_kind="cpu_derived",
        )
        self._frame_cache[key] = frame
        self._asset_paths[frame.asset_id] = path
        return frame

    def _raw_frame_path(self, media: _SceneMedia, frame_index: int) -> Path:
        return (
            self._cache_root()
            / "vsi_bench"
            / media.dataset
            / media.scene_name
            / f"source_frame_{frame_index:08d}.jpg"
        )

    def _contact_sheet(
        self, media: _SceneMedia, frames: Sequence[_Frame]
    ) -> _Frame:
        source_hash = _canonical_hash([frame.sha256 for frame in frames])
        path = (
            self._cache_root()
            / "vsi_bench"
            / media.dataset
            / media.scene_name
            / f"contact_sheet_{source_hash[:16]}.png"
        )
        if not path.is_file():
            images: list[Image.Image] = []
            try:
                for frame in frames:
                    with Image.open(frame.path) as image:
                        converted = image.convert("RGB")
                        converted.thumbnail((320, 240), Image.Resampling.LANCZOS)
                        images.append(converted.copy())
                gap = 2
                width = sum(image.width for image in images) + gap * (len(images) - 1)
                height = max(image.height for image in images)
                sheet = Image.new("RGB", (width, height), color=(0, 0, 0))
                left = 0
                for image in images:
                    top = (height - image.height) // 2
                    sheet.paste(image, (left, top))
                    left += image.width + gap
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.tmp")
                sheet.save(temporary, format="PNG", compress_level=9)
                temporary.replace(path)
            finally:
                for image in images:
                    image.close()
        width, height = self._inspect_image(path)
        sha256 = _sha256_file(path)
        frame = _Frame(
            asset_id=_asset_id(
                dataset=media.dataset,
                scene_name=media.scene_name,
                source_kind="contact_sheet",
                position=source_hash,
                content_sha256=sha256,
            ),
            path=path,
            sha256=sha256,
            width=width,
            height=height,
            frozen_ordinal=None,
            source_frame_index=None,
            timestamp_seconds=None,
            frame_index_status="not_applicable_composite",
            timestamp_status="not_applicable_composite",
            selection_position="deterministic_full_context_preview",
            source_kind="cpu_derived",
        )
        self._asset_paths[frame.asset_id] = path
        return frame

    def _get_frame_action(
        self, media: _SceneMedia, values: Mapping[str, Any]
    ) -> _Frame:
        allowed = {"frame_index", "timestamp_seconds"}
        if set(values) - allowed or len(set(values) & allowed) != 1:
            raise VSIBenchActionError(
                "GET_FRAME requires exactly one frame_index or timestamp_seconds"
            )
        if "timestamp_seconds" in values:
            timestamp = values["timestamp_seconds"]
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise VSIBenchActionError("timestamp_seconds must be numeric")
            timestamp = float(timestamp)
            if not math.isfinite(timestamp) or timestamp < 0:
                raise VSIBenchActionError(
                    "timestamp_seconds must be finite and non-negative"
                )
            if (
                media.mode != "raw_video"
                or media.fps is None
                or media.frame_count is None
            ):
                raise VSIBenchActionError(
                    "source timestamp is unknown for materialized frozen frames"
                )
            frame_index = round(timestamp * media.fps)
            if frame_index >= media.frame_count:
                raise VSIBenchActionError("timestamp_seconds is out of range")
            return self._raw_frame(media, frame_index, "agent_timestamp_request")
        frame_index = values["frame_index"]
        if not _is_int(frame_index) or frame_index < 0:
            raise VSIBenchActionError("frame_index must be a non-negative integer")
        if media.mode == "raw_video":
            return self._raw_frame(media, frame_index, "agent_frame_request")
        return self._frozen_frame(media, frame_index)

    def _get_window_action(
        self, media: _SceneMedia, values: Mapping[str, Any]
    ) -> tuple[_Frame, ...]:
        required = {"start_frame_index", "end_frame_index"}
        if set(values) != required:
            raise VSIBenchActionError(
                "GET_FRAME_WINDOW requires start_frame_index and end_frame_index"
            )
        start = values["start_frame_index"]
        end = values["end_frame_index"]
        if not _is_int(start) or not _is_int(end) or start < 0 or end < start:
            raise VSIBenchActionError("invalid inclusive frame window")
        if media.mode == "raw_video":
            if media.frame_count is None or end >= media.frame_count:
                raise VSIBenchActionError("source frame window is out of range")
        elif end >= len(media.frozen_paths):
            raise VSIBenchActionError("frozen frame window is out of range")
        count = end - start + 1
        maximum = (
            _RAW_WINDOW_LIMIT
            if media.mode == "raw_video"
            else len(media.frozen_paths)
        )
        if count > maximum:
            raise VSIBenchActionError("requested frame window exceeds the policy limit")
        if media.mode == "raw_video":
            return tuple(
                self._raw_frame(media, index, "agent_frame_window")
                for index in range(start, end + 1)
            )
        return tuple(
            self._frozen_frame(media, index)
            for index in range(start, end + 1)
        )

    @staticmethod
    def _action_envelope(
        action: str | Mapping[str, Any], arguments: Mapping[str, Any] | None
    ) -> tuple[str, Mapping[str, Any]]:
        if isinstance(action, Mapping):
            if arguments is not None or set(action) != {"action", "arguments"}:
                raise VSIBenchActionError("invalid action envelope")
            name = action.get("action")
            values = action.get("arguments")
        else:
            name = action
            values = {} if arguments is None else arguments
        if not isinstance(name, str) or not isinstance(values, Mapping):
            raise VSIBenchActionError("action and arguments have invalid types")
        return name, values

    @staticmethod
    def _frame_metadata(frame: _Frame) -> dict[str, Any]:
        return {
            "frozen_frame_ordinal": frame.frozen_ordinal,
            "source_frame_index": frame.source_frame_index,
            "timestamp_seconds": frame.timestamp_seconds,
            "source_frame_index_status": frame.frame_index_status,
            "source_timestamp_status": frame.timestamp_status,
            "selection_position": frame.selection_position,
        }

    def _frame_descriptor(self, frame: _Frame) -> dict[str, Any]:
        metadata = self._frame_metadata(frame)
        metadata.update(
            {
                "selection_basis": "scene_media_only",
                "runtime_path_exposed": False,
            }
        )
        return {
            "asset_id": frame.asset_id,
            "modality": "derived_image",
            "source_kind": frame.source_kind,
            "width": frame.width,
            "height": frame.height,
            "frame_count": 1,
            "duration": None,
            "public": True,
            "source_media_hash": frame.sha256,
            "metadata": metadata,
        }

__all__ = [
    "VSIBenchActionError",
    "VSIBenchAdapter",
    "VSIBenchDataError",
    "VSIBenchError",
    "VSIBenchSubmissionError",
    "VideoReader",
]
