"""CPU-only adapters for the static-image W2 source benchmarks.

The adapters in this module intentionally expose only public task text and
benchmark-provided image assets.  References and source paths stay in the
adapter process and are consulted only by the post-submission evaluator.
None of the actions below claims to reconstruct depth, pose, or 3D geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlsplit
import zipfile

from PIL import Image, UnidentifiedImageError

from w2_harness.evaluators.bop_ask_v1 import BOPASKEvaluatorV1
from w2_harness.evaluators.task_router_v2 import TaskTypeEvaluatorRouterV2

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
    InvalidSubmissionError,
    ParsedSubmission,
    SampleNotFoundError,
    normalize_mcq_answer,
    register_adapter,
)


STATIC_IMAGE_ACTIONS = (
    "GET_TASK_CONTEXT",
    "LIST_ASSETS",
    "QUERY_STATE",
    "OPEN_ASSET",
    "CROP_REGION",
    "ZOOM_REGION",
    "RECORD_EVIDENCE",
    "LIST_EVIDENCE",
    "SUBMIT",
)

_BOP_SOURCE_FILES = (
    ("core_handal", "core/bopask-test-handal.json"),
    ("core_hope", "core/bopask-test-hope.json"),
    ("core_ycbv", "core/bopask-test-ycbv.json"),
    ("lab_home", "lab/bopask-test-home.json"),
)
_BOP_SUPPORTED_FAMILIES = {
    ("depth_relative", "closer"),
    ("depth_relative", "farther"),
    ("spatial_reasoning", "relative_position"),
}
_IMAGE_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
_IMAGE_MIME = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
_MISSING = object()


class OfflineStaticAdapterError(AdapterError):
    """Base error for a fail-closed static adapter operation."""


class AdapterDataError(OfflineStaticAdapterError):
    """The declared benchmark layout is absent or malformed."""


class UnknownSampleError(SampleNotFoundError, OfflineStaticAdapterError):
    """A sample is not part of the adapter's executable denominator."""


class DataMissingError(UnknownSampleError):
    """A released sample is excluded because a truly required asset is absent."""


class SubmissionParseError(InvalidSubmissionError, OfflineStaticAdapterError):
    """A submission cannot be normalized without guessing its answer."""


@dataclass(frozen=True)
class _ImagePayload:
    role: str
    payload: bytes


@dataclass(frozen=True)
class _Record:
    sample_id: str
    question: str
    choices: tuple[str, ...]
    answer_format: str
    category: str
    public_metadata: Mapping[str, Any]
    private_answer: Any
    source: Any


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                value,
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise OfflineStaticAdapterError("adapter values must be JSON-safe") from exc


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterDataError(f"{field} must be a non-empty string")
    return value.strip()


def _import_parquet() -> Any:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise AdapterDataError("pyarrow is required to read benchmark parquet") from exc
    return parquet


def _parquet_row(path: Path, index: int, columns: Sequence[str]) -> dict[str, Any]:
    parquet = _import_parquet()
    source = parquet.ParquetFile(path)
    offset = 0
    for group_index in range(source.num_row_groups):
        rows = source.metadata.row_group(group_index).num_rows
        if index < offset + rows:
            local_index = index - offset
            values = source.read_row_group(group_index, columns=list(columns))
            return dict(values.slice(local_index, 1).to_pylist()[0])
        offset += rows
    raise AdapterDataError(f"parquet row index is out of range: {index}")


def _image_info(payload: bytes) -> tuple[int, int, str]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            width, height = image.size
            image_format = str(image.format or "").upper()
            image.verify()
    except (OSError, UnidentifiedImageError) as exc:
        raise AdapterDataError("benchmark image is unreadable") from exc
    if width < 1 or height < 1:
        raise AdapterDataError("benchmark image dimensions are invalid")
    return width, height, image_format


def _safe_child(root: Path, relative_value: Any) -> Path:
    relative = Path(str(relative_value or "").replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise AdapterDataError("benchmark media path is unsafe")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AdapterDataError("benchmark media path escapes its dataset") from exc
    return candidate


def _strip_image_token(value: Any) -> str:
    text = _nonempty(value, "question")
    return re.sub(r"^\s*<image>\s*", "", text, flags=re.IGNORECASE).strip()


class StaticImageBenchmarkAdapter(BaseAdapter):
    """Unified read-only CPU adapter for benchmark-provided static images."""

    benchmark_id = ""
    package_id = ""
    aliases: tuple[str, ...] = ()
    score_scope = "diagnostic"
    denominator_policy = ""
    observation_policy = ""
    media_operation_ceiling = 4
    initial_asset_limit = 1

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        cache_root: Path | str | None = None,
        materialization_root: Path | str | None = None,
    ) -> None:
        if cache_root is not None and materialization_root is not None:
            if Path(cache_root).expanduser().resolve() != Path(
                materialization_root
            ).expanduser().resolve():
                raise AdapterDataError(
                    "cache_root and materialization_root must identify the same directory"
                )
        selected_cache = (
            materialization_root if materialization_root is not None else cache_root
        )
        super().__init__(data_root, cache_root=selected_cache)
        self.materialization_root = self._cache_root()
        self._records: dict[str, _Record] = {}
        self._excluded: dict[str, str] = {}
        self._released_count = 0
        self._asset_paths: dict[str, Path] = {}
        self._asset_formats: dict[str, str] = {}
        self._materialized: dict[str, dict[str, Any]] = {}
        self._loaded_records: dict[str, _Record] = {}
        self._loaded_samples: dict[str, AdapterSample] = {}
        self._sample_assets: dict[str, set[str]] = {}
        self._derived_descriptors: dict[str, dict[str, dict[str, Any]]] = {}
        self._action_cache: dict[tuple[str, str], dict[str, Any]] = {}
        self._media_operations: dict[str, int] = {}
        self._evidence: dict[str, list[dict[str, Any]]] = {}
        self._state_revisions: dict[str, int] = {}
        self._load_records()
        if not self._records and not self._excluded:
            raise AdapterDataError(f"{self.benchmark_id} contains no released rows")

    def _load_records(self) -> None:
        raise NotImplementedError

    def _image_payloads(self, record: _Record) -> Sequence[_ImagePayload]:
        raise NotImplementedError

    def _evaluate_record(self, record: _Record, prediction: Any) -> Any:
        router = TaskTypeEvaluatorRouterV2()
        metadata = dict(record.public_metadata)
        if record.answer_format == "normalized_point_list":
            metadata.update(
                {
                    "metric_task_type": "coordinate",
                    "coordinate_match_mode": "set",
                    "coordinate_min": 0.0,
                    "coordinate_max": 1.0,
                }
            )
        return router.evaluate(
            prediction,
            record.private_answer,
            answer_type=(
                "multiple_choice"
                if record.choices
                else "short_text"
            ),
            evaluator="choice_exact" if record.choices else "normalized_match",
            benchmark=self.benchmark_id,
            choices=record.choices,
            question=record.question,
            metadata=metadata,
        )

    def _add_record(self, record: _Record) -> None:
        if record.sample_id in self._records or record.sample_id in self._excluded:
            raise AdapterDataError(f"duplicate benchmark sample ID: {record.sample_id}")
        self._records[record.sample_id] = record

    def _exclude(self, sample_id: str, reason: str) -> None:
        if sample_id in self._records or sample_id in self._excluded:
            raise AdapterDataError(f"duplicate benchmark sample ID: {sample_id}")
        self._excluded[sample_id] = reason

    def discover_samples(self) -> tuple[str, ...]:
        """Return only rows eligible for this adapter's declared denominator."""

        self._ensure_open()
        return tuple(self._records)

    list_sample_ids = discover_samples
    sample_ids = discover_samples

    def sample_status(self, sample_id: str) -> str:
        self._ensure_open()
        if sample_id in self._records:
            return "ready"
        if sample_id in self._excluded:
            return self._excluded[sample_id]
        raise UnknownSampleError("sample is not present in the released scope")

    def denominator_report(self) -> dict[str, Any]:
        self._ensure_open()
        data_missing = sum(value == "data_missing" for value in self._excluded.values())
        return {
            "benchmark_id": self.benchmark_id,
            "released_rows": self._released_count,
            "executable_denominator": len(self._records),
            "data_missing": data_missing,
            "other_excluded": len(self._excluded) - data_missing,
            "denominator_policy": self.denominator_policy,
            "score_scope": self.score_scope,
            "official_evaluator_used": False,
            "official_score": None,
        }

    def _record(self, sample_id: str) -> _Record:
        if sample_id in self._excluded:
            if self._excluded[sample_id] == "data_missing":
                raise DataMissingError(
                    "sample is excluded from the executable denominator: data_missing"
                )
            raise UnknownSampleError("sample is excluded from the executable denominator")
        try:
            return self._records[sample_id]
        except KeyError as exc:
            raise UnknownSampleError("sample is not present in the released scope") from exc

    @staticmethod
    def _task_id(sample_id: str, benchmark_id: str) -> str:
        digest = _sha256(
            _canonical_bytes(
                {"benchmark_id": benchmark_id, "source_sample_id": sample_id}
            )
        )
        return f"task_{digest[:32]}"

    @staticmethod
    def _choices(values: Sequence[str]) -> tuple[Choice, ...]:
        output: list[Choice] = []
        for index, value in enumerate(values):
            match = re.match(r"^([A-Z])[.:)]\s*(.+)$", value, flags=re.IGNORECASE)
            output.append(
                Choice(
                    label=match.group(1) if match else chr(ord("A") + index),
                    text=match.group(2) if match else value,
                )
            )
        return tuple(output)

    def _store_asset(
        self,
        payload: bytes,
        *,
        source_kind: str = "benchmark_public",
        identity: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]]:
        width, height, image_format = _image_info(payload)
        digest = _sha256(payload)
        opaque_digest = _sha256(
            _canonical_bytes(
                {
                    "domain": "w2-offline-static-asset-v1",
                    "content_sha256": digest,
                    "identity": dict(identity or {"kind": "benchmark_public"}),
                }
            )
        )
        asset_id = f"asset_{opaque_digest}"
        suffix = _IMAGE_SUFFIX.get(image_format, ".img")
        destination = self.materialization_root / f"{asset_id}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if not destination.is_file() or _sha256(destination.read_bytes()) != digest:
                raise AdapterDataError("materialized asset cache failed integrity check")
        else:
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
        self._asset_paths[asset_id] = destination
        self._asset_formats[asset_id] = image_format
        descriptor = {
            "asset_id": asset_id,
            "modality": "image",
            "source_kind": source_kind,
            "width": width,
            "height": height,
            "frame_count": 1,
            "duration": None,
            "public": True,
            "source_media_hash": digest,
            "metadata": _json_copy(metadata or {}),
        }
        return asset_id, descriptor

    def materialize_sample(self, sample_id: str) -> dict[str, Any]:
        """Materialize one real source row without exposing its private sidecar."""

        self._ensure_open()
        sample_id = str(sample_id)
        record = self._record(sample_id)
        cached = self._materialized.get(sample_id)
        if cached is not None:
            return _json_copy(cached)

        descriptors_by_id: dict[str, dict[str, Any]] = {}
        roles_by_id: dict[str, set[str]] = {}
        for image in self._image_payloads(record):
            asset_id, descriptor = self._store_asset(image.payload)
            descriptors_by_id.setdefault(asset_id, descriptor)
            roles_by_id.setdefault(asset_id, set()).add(image.role)
        if not descriptors_by_id:
            raise AdapterDataError("executable sample has no readable public image")
        for asset_id, descriptor in descriptors_by_id.items():
            descriptor["metadata"] = {
                "roles": sorted(roles_by_id[asset_id]),
                "pixel_claim": "benchmark_provided_image_only",
            }

        asset_ids = list(descriptors_by_id)
        self._sample_assets[sample_id] = set(asset_ids)
        public_task = {
            "benchmark_id": self.benchmark_id,
            "sample_id": record.sample_id,
            "question": record.question,
            "choices": list(record.choices),
            "answer_format": record.answer_format,
            "public_metadata": _json_copy(record.public_metadata),
            "asset_ids": asset_ids,
            "observation_policy": self.observation_policy,
            "category": record.category,
            "submission_schema_id": "w2.offline.static_submission.v1",
        }
        result = {
            "public_task": public_task,
            "assets": list(descriptors_by_id.values()),
            "execution_status": "ready",
            "denominator_eligible": True,
            "score_scope": self.score_scope,
            "official_score": None,
        }
        self._materialized[sample_id] = result
        return _json_copy(result)

    materialize = materialize_sample
    materialize_task = materialize_sample

    def load_sample(
        self,
        sample_id: str | int | Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        self._ensure_open()
        if isinstance(sample_id, AdapterSample):
            task_id = sample_id.task_id
            if task_id in self._loaded_samples:
                return self._loaded_samples[task_id]
            sample_id = sample_id.source_sample_id
        elif isinstance(sample_id, Mapping):
            task_id = str(sample_id.get("task_id") or "")
            if task_id in self._loaded_samples:
                return self._loaded_samples[task_id]
            for key in ("source_sample_id", "sample_id", "id", "index"):
                if sample_id.get(key) is not None:
                    sample_id = str(sample_id[key])
                    break
            else:
                raise UnknownSampleError("sample mapping has no source ID")
        source_id = next(iter(self._records), "") if sample_id is None else str(sample_id)
        if not source_id:
            raise UnknownSampleError("executable denominator is empty")
        existing = self._source_to_task.get(source_id)
        if existing in self._loaded_samples:
            return self._loaded_samples[existing]
        materialized = self.materialize_sample(source_id)
        record = self._record(source_id)
        task_id = self._task_id(source_id, self.benchmark_id)
        sample = AdapterSample(
            task_id=task_id,
            source_sample_id=source_id,
            prompt=record.question,
            choices=self._choices(record.choices),
            asset_ids=tuple(materialized["public_task"]["asset_ids"]),
            metadata={
                **_json_copy(record.public_metadata),
                "answer_format": record.answer_format,
                "category": record.category,
            },
        )
        self._loaded_samples[task_id] = sample
        self._loaded_records[task_id] = record
        self._source_to_task[source_id] = task_id
        self._current_task_id = task_id
        return sample

    def _loaded_state(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | int | None,
    ) -> tuple[AdapterSample, _Record]:
        self._ensure_open()
        if sample is None:
            task_id = self._current_task_id
        elif isinstance(sample, AdapterSample):
            task_id = sample.task_id
        elif isinstance(sample, Mapping):
            task_id = str(sample.get("task_id") or "")
            if not task_id:
                loaded = self.load_sample(sample)
                task_id = loaded.task_id
        else:
            value = str(sample)
            task_id = value if value in self._loaded_samples else self._source_to_task.get(value)
            if task_id is None:
                task_id = self.load_sample(value).task_id
        if not task_id or task_id not in self._loaded_samples:
            raise UnknownSampleError("sample is not loaded by this adapter")
        return self._loaded_samples[task_id], self._loaded_records[task_id]

    def get_public_task(self, sample_id: str | int) -> dict[str, Any]:
        return self.materialize_sample(str(sample_id))["public_task"]

    load_public_task = get_public_task
    load_task = get_public_task

    def list_assets(self, sample_id: str | int) -> tuple[dict[str, Any], ...]:
        return tuple(self.materialize_sample(str(sample_id))["assets"])

    def _component_descriptor(self, sample_id: str, asset_id: str) -> dict[str, Any]:
        if asset_id not in self._sample_assets.get(sample_id, set()):
            raise OfflineStaticAdapterError("asset ID does not belong to this sample")
        for descriptor in self.materialize_sample(sample_id)["assets"]:
            if descriptor["asset_id"] == asset_id:
                return descriptor
        try:
            return _json_copy(self._derived_descriptors[sample_id][asset_id])
        except KeyError as exc:
            raise OfflineStaticAdapterError("asset descriptor is unavailable") from exc

    def _transport_descriptor(
        self,
        sample_id: str,
        asset_id: str,
        sequence_index: int,
    ) -> dict[str, Any]:
        descriptor = self._component_descriptor(sample_id, asset_id)
        image_format = self._asset_formats[asset_id]
        path = self.resolve_asset(asset_id)
        return {
            "asset_id": asset_id,
            "media_type": "image",
            "mime_type": _IMAGE_MIME.get(image_format, "application/octet-stream"),
            "sequence_index": sequence_index,
            "byte_size": path.stat().st_size,
            "content_sha256": descriptor["source_media_hash"],
            "local_path": str(path),
        }

    def build_public_task(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, record = self._loaded_state(sample)
        return {
            "schema_version": PUBLIC_TASK_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "prompt": loaded.prompt,
            "choices": [choice.to_dict() for choice in loaded.choices],
            "answer_type": record.answer_format,
            "asset_ids": list(loaded.asset_ids),
        }

    def build_asset_catalog(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, _ = self._loaded_state(sample)
        assets = []
        for index, asset_id in enumerate(loaded.asset_ids):
            value = self._transport_descriptor(
                loaded.source_sample_id, asset_id, index
            )
            value.pop("local_path")
            value["uri"] = f"asset://{asset_id}"
            value["metadata"] = self._component_descriptor(
                loaded.source_sample_id, asset_id
            )["metadata"]
            assets.append(value)
        return {
            "schema_version": ASSET_CATALOG_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "assets": assets,
        }

    def build_action_policy(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, _ = self._loaded_state(sample)
        return {
            "schema_version": ACTION_POLICY_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "policy": self.observation_policy,
            "allowed_actions": ["inspect", "submit_answer"],
            "inspect": {
                "required_arguments": ["operation", "asset_id"],
                "operations": ["OPEN_ASSET", "CROP_REGION", "ZOOM_REGION"],
                "asset_ids": list(loaded.asset_ids),
                "coordinate_semantics": "integer_image_pixels",
                "max_calls": self.media_operation_ceiling,
                "read_only": True,
            },
            "turn_limit": self.media_operation_ceiling + 1,
        }

    def build_full_context_media_plan(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, _ = self._loaded_state(sample)
        return {
            "schema_version": MEDIA_PLAN_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "strategy": "all_benchmark_provided_static_images",
            "media": [
                self._transport_descriptor(loaded.source_sample_id, asset_id, index)
                for index, asset_id in enumerate(loaded.asset_ids)
            ],
        }

    def build_selective_initial_observation(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, _ = self._loaded_state(sample)
        visible_count = min(self.initial_asset_limit, len(loaded.asset_ids))
        observed = loaded.asset_ids[:visible_count]
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "observation_policy": self.observation_policy,
            "observed_asset_ids": list(observed),
            "remaining_asset_ids": list(loaded.asset_ids[visible_count:]),
            "observation_complete": visible_count == len(loaded.asset_ids),
            "media": [
                self._transport_descriptor(loaded.source_sample_id, asset_id, index)
                for index, asset_id in enumerate(observed)
            ],
            "next_media_operations": ["OPEN_ASSET", "CROP_REGION", "ZOOM_REGION"],
        }

    def build_submission_schema(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded, _ = self._loaded_state(sample)
        answer: dict[str, Any]
        if loaded.choices:
            answer = {
                "type": "string",
                "enum": [choice.label for choice in loaded.choices],
            }
        else:
            answer = {
                "oneOf": [
                    {"type": "string", "minLength": 1},
                    {"type": "array"},
                    {"type": "object"},
                    {"type": "number"},
                ]
            }
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"{self.benchmark_id} {self.score_scope} submission",
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {"answer": answer},
        }

    def action_policy(
        self,
        sample_id: str | int | None = None,
        *,
        profile: str = "w2_light",
        state_revision: int | None = None,
    ) -> dict[str, Any]:
        if profile not in {"direct", "w2_light"}:
            raise OfflineStaticAdapterError("static adapter profile is unsupported")
        source_id = None if sample_id is None else str(sample_id)
        if source_id is not None:
            self.load_sample(source_id)
        revision = (
            self._state_revisions.get(source_id, 0)
            if state_revision is None
            else state_revision
        )
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise OfflineStaticAdapterError("state revision must be non-negative")
        allowed = list(STATIC_IMAGE_ACTIONS)
        if profile == "direct":
            allowed = ["GET_TASK_CONTEXT", "LIST_ASSETS", "SUBMIT"]
        return {
            "allowed_actions": allowed,
            "action_schemas": {
                "GET_TASK_CONTEXT": {"required": []},
                "LIST_ASSETS": {"required": []},
                "QUERY_STATE": {"required": []},
                "OPEN_ASSET": {"required": ["asset_id"]},
                "CROP_REGION": {
                    "required": ["asset_id", "x", "y", "width", "height"],
                    "coordinate_semantics": "integer_image_pixels",
                },
                "ZOOM_REGION": {
                    "required": ["asset_id", "x", "y", "width", "height"],
                    "coordinate_semantics": "integer_image_pixels",
                    "resize_semantics": "nearest_neighbor_to_source_canvas",
                },
                "RECORD_EVIDENCE": {"required": ["asset_id", "description"]},
                "LIST_EVIDENCE": {"required": []},
                "SUBMIT": {"required": ["answer"]},
            },
            "media_operation_ceiling": self.media_operation_ceiling,
            "duplicate_action_policy": "content_identity_cache",
            "current_profile": profile,
            "state_revision": revision,
        }

    get_action_policy = action_policy

    def initial_observation_plan(
        self,
        sample_id: str | int,
        *,
        profile: str = "w2_light",
    ) -> dict[str, Any]:
        source_id = str(sample_id)
        assets = list(self.list_assets(source_id))
        if profile == "direct":
            return {
                "policy": "all_benchmark_provided_static_images",
                "visible_assets": assets,
                "available_actions": [],
            }
        if profile != "w2_light":
            raise OfflineStaticAdapterError("static adapter profile is unsupported")
        return {
            "policy": "initial_static_image_then_read_only_image_actions",
            "visible_assets": assets[: self.initial_asset_limit],
            "remaining_asset_ids": [
                value["asset_id"] for value in assets[self.initial_asset_limit :]
            ],
            "available_actions": ["OPEN_ASSET", "CROP_REGION", "ZOOM_REGION"],
        }

    get_initial_observation = initial_observation_plan

    @staticmethod
    def _action_envelope(
        action: str | Mapping[str, Any],
        arguments: Mapping[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        if isinstance(action, Mapping):
            if arguments is not None:
                raise OfflineStaticAdapterError("action arguments were supplied twice")
            allowed = {"action", "type", "arguments"}
            if set(action) - allowed:
                raise OfflineStaticAdapterError("action envelope has unsupported fields")
            name = action.get("action", action.get("type"))
            values = action.get("arguments", {})
        else:
            name = action
            values = {} if arguments is None else arguments
        if not isinstance(name, str) or not name.strip():
            raise OfflineStaticAdapterError("action name is required")
        if not isinstance(values, Mapping):
            raise OfflineStaticAdapterError("action arguments must be an object")
        return name.strip().upper(), dict(values)

    @staticmethod
    def _require_fields(
        values: Mapping[str, Any], required: set[str]
    ) -> None:
        if set(values) != required:
            raise OfflineStaticAdapterError(
                "action arguments must match the declared schema exactly"
            )

    def _consume_media_operation(
        self,
        sample_id: str,
        action: str,
        values: Mapping[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        cache_key = _sha256(_canonical_bytes({"action": action, "arguments": values}))
        cached = self._action_cache.get((sample_id, cache_key))
        if cached is not None:
            return cache_key, _json_copy(cached)
        used = self._media_operations.get(sample_id, 0)
        if used >= self.media_operation_ceiling:
            raise OfflineStaticAdapterError("static media operation ceiling exceeded")
        self._media_operations[sample_id] = used + 1
        self._state_revisions[sample_id] = self._state_revisions.get(sample_id, 0) + 1
        return cache_key, None

    def _transform_region(
        self,
        sample_id: str,
        operation: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        self._require_fields(values, {"asset_id", "x", "y", "width", "height"})
        asset_id = str(values["asset_id"])
        source_descriptor = self._component_descriptor(sample_id, asset_id)
        coordinates: list[int] = []
        for field in ("x", "y", "width", "height"):
            value = values[field]
            if isinstance(value, bool) or not isinstance(value, int):
                raise OfflineStaticAdapterError("region coordinates must be integers")
            coordinates.append(value)
        x, y, width, height = coordinates
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            raise OfflineStaticAdapterError("region must have positive in-bounds extent")
        if x + width > source_descriptor["width"] or y + height > source_descriptor["height"]:
            raise OfflineStaticAdapterError("region exceeds image bounds")
        try:
            with Image.open(self.resolve_asset(asset_id)) as source:
                output = source.crop((x, y, x + width, y + height))
                if operation == "ZOOM_REGION":
                    output = output.resize(source.size, resample=Image.Resampling.NEAREST)
                encoded = io.BytesIO()
                output.save(encoded, format="PNG")
        except (OSError, UnidentifiedImageError) as exc:
            raise OfflineStaticAdapterError("static image transform failed") from exc
        box = {"x": x, "y": y, "width": width, "height": height}
        derived_id, descriptor = self._store_asset(
            encoded.getvalue(),
            source_kind="cpu_image_transform",
            identity={
                "parent_asset_id": asset_id,
                "operation": operation,
                "source_box": box,
            },
            metadata={
                "parent_asset_id": asset_id,
                "operation": operation,
                "source_box": box,
                "pixel_claim": "deterministic_2d_pixel_transform_only",
            },
        )
        self._sample_assets.setdefault(sample_id, set()).add(derived_id)
        self._derived_descriptors.setdefault(sample_id, {})[derived_id] = descriptor
        return descriptor

    def execute_action(
        self,
        sample_id: str | int,
        action: str | Mapping[str, Any],
        arguments: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_id = str(sample_id)
        self.load_sample(source_id)
        action_name, values = self._action_envelope(action, arguments)
        if action_name == "GET_TASK_CONTEXT":
            self._require_fields(values, set())
            return {"action": action_name, "task": self.load_public_task(source_id)}
        if action_name == "LIST_ASSETS":
            self._require_fields(values, set())
            return {"action": action_name, "assets": list(self.list_assets(source_id))}
        if action_name == "QUERY_STATE":
            self._require_fields(values, set())
            return {
                "action": action_name,
                "state_revision": self._state_revisions.get(source_id, 0),
                "media_operations_used": self._media_operations.get(source_id, 0),
                "media_operation_ceiling": self.media_operation_ceiling,
                "evidence_count": len(self._evidence.get(source_id, [])),
            }
        if action_name in {"OPEN_ASSET", "CROP_REGION", "ZOOM_REGION"}:
            if action_name == "OPEN_ASSET":
                self._require_fields(values, {"asset_id"})
            cache_key, cached = self._consume_media_operation(
                source_id, action_name, values
            )
            if cached is not None:
                return cached
            if action_name == "OPEN_ASSET":
                descriptor = self._component_descriptor(
                    source_id, str(values["asset_id"])
                )
            else:
                descriptor = self._transform_region(source_id, action_name, values)
            result = {"action": action_name, "asset": descriptor}
            self._action_cache[(source_id, cache_key)] = _json_copy(result)
            return _json_copy(result)
        if action_name == "RECORD_EVIDENCE":
            self._require_fields(values, {"asset_id", "description"})
            asset_id = str(values["asset_id"])
            self._component_descriptor(source_id, asset_id)
            description = str(values["description"]).strip()
            if not description:
                raise OfflineStaticAdapterError("evidence description must be non-empty")
            evidence_id = "evidence_" + _sha256(
                _canonical_bytes(
                    {
                        "sample_id": source_id,
                        "asset_id": asset_id,
                        "description": description,
                    }
                )
            )[:32]
            item = {
                "evidence_id": evidence_id,
                "asset_id": asset_id,
                "description": description,
            }
            evidence = self._evidence.setdefault(source_id, [])
            if all(existing["evidence_id"] != evidence_id for existing in evidence):
                evidence.append(item)
                self._state_revisions[source_id] = self._state_revisions.get(source_id, 0) + 1
            return {"action": action_name, "evidence": _json_copy(item)}
        if action_name == "LIST_EVIDENCE":
            self._require_fields(values, set())
            return {
                "action": action_name,
                "evidence": _json_copy(self._evidence.get(source_id, [])),
            }
        if action_name == "SUBMIT":
            known = {
                item["evidence_id"] for item in self._evidence.get(source_id, [])
            }
            result = self.submit(source_id, values, known_evidence_ids=known)
            return {"action": action_name, "terminal": True, "result": result}
        raise OfflineStaticAdapterError("unsupported static image action")

    handle_action = execute_action

    def resolve_asset(self, asset_id: str) -> Path:
        """Resolve an opaque ID inside the private harness process."""

        self._ensure_open()
        try:
            return self._asset_paths[asset_id]
        except KeyError as exc:
            raise OfflineStaticAdapterError(
                "unknown or not-yet-materialized asset ID"
            ) from exc

    resolve_asset_path = resolve_asset

    def read_asset(self, asset_id: str) -> bytes:
        return self.resolve_asset(asset_id).read_bytes()

    def parse_submission(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | Any,
        submission: Any = _MISSING,
        *,
        known_evidence_ids: Iterable[str] | None = None,
    ) -> ParsedSubmission:
        """Normalize a submission without consulting the private annotation."""

        if submission is _MISSING:
            raw_submission = sample
            loaded, _ = self._loaded_state(None)
        else:
            raw_submission = submission
            loaded, _ = self._loaded_state(sample)  # type: ignore[arg-type]
        if isinstance(raw_submission, Mapping):
            try:
                raw_for_hash = _canonical_bytes(raw_submission)
            except (TypeError, ValueError) as exc:
                raise SubmissionParseError("submission must be finite JSON") from exc
            payload: Any = dict(raw_submission)
        elif isinstance(raw_submission, bytes):
            raw_for_hash = bytes(raw_submission)
            try:
                text = raw_submission.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SubmissionParseError("submission must be UTF-8") from exc
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                payload = {"answer": text.strip()}
        elif isinstance(raw_submission, str):
            raw_for_hash = raw_submission.encode("utf-8")
            try:
                payload = json.loads(raw_submission)
            except json.JSONDecodeError:
                payload = {"answer": raw_submission.strip()}
        else:
            try:
                raw_for_hash = _canonical_bytes(raw_submission)
            except (TypeError, ValueError) as exc:
                raise SubmissionParseError(
                    "submission must be JSON, text, or an object"
                ) from exc
            payload = {"answer": raw_submission}
        if not isinstance(payload, Mapping):
            payload = {"answer": payload}
        allowed = {"answer", "evidence_refs", "confidence"}
        if set(payload) - allowed:
            raise SubmissionParseError("submission contains unsupported fields")
        if "answer" not in payload or payload["answer"] is None:
            raise SubmissionParseError("submission answer is required")

        raw_evidence = payload.get("evidence_refs", [])
        if not isinstance(raw_evidence, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_evidence
        ):
            raise SubmissionParseError("evidence_refs must be a list of IDs")
        evidence_refs = [item.strip() for item in raw_evidence]
        if len(evidence_refs) != len(set(evidence_refs)):
            raise SubmissionParseError("evidence_refs must be unique")
        if known_evidence_ids is not None:
            known = set(known_evidence_ids)
            if any(item not in known for item in evidence_refs):
                raise SubmissionParseError("submission cites unknown evidence")
        confidence = payload.get("confidence")
        if confidence is not None:
            if isinstance(confidence, bool):
                raise SubmissionParseError("confidence must be numeric")
            try:
                confidence = float(confidence)
            except (TypeError, ValueError) as exc:
                raise SubmissionParseError("confidence must be numeric") from exc
            if not 0.0 <= confidence <= 1.0:
                raise SubmissionParseError("confidence must be between zero and one")

        try:
            answer_value = _json_copy(payload["answer"])
        except OfflineStaticAdapterError as exc:
            raise SubmissionParseError("submission answer must be finite JSON") from exc
        if loaded.choices:
            normalized = normalize_mcq_answer(answer_value, loaded.choices)
            issue = "unrecognized_multiple_choice"
        else:
            if isinstance(answer_value, str):
                normalized = answer_value.strip() or None
            else:
                normalized = _canonical_bytes(answer_value).decode("utf-8")
            issue = "empty_answer"
        return ParsedSubmission(
            task_id=loaded.task_id,
            answer=normalized,
            valid=normalized is not None,
            issues=() if normalized is not None else (issue,),
            raw_sha256=_sha256(raw_for_hash),
            _adapter_capability=self._submission_capability,
        )

    def evaluate_private(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | ParsedSubmission,
        submission: ParsedSubmission | object = _MISSING,
    ) -> dict[str, Any]:
        if submission is _MISSING:
            parsed = sample
            loaded, record = self._loaded_state(None)
        else:
            parsed = submission
            loaded, record = self._loaded_state(sample)  # type: ignore[arg-type]
        if not isinstance(parsed, ParsedSubmission):
            raise SubmissionParseError(
                "evaluate_private requires the result of parse_submission"
            )
        if parsed._adapter_capability is not self._submission_capability:
            raise SubmissionParseError("submission belongs to another adapter instance")
        if parsed.task_id != loaded.task_id:
            raise SubmissionParseError("submission and sample task IDs differ")
        if not parsed.valid:
            return {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "task_id": loaded.task_id,
                "metric": "not_evaluated_invalid_submission",
                "score_scope": self.score_scope,
                "submission_valid": False,
                "normalized_answer": parsed.answer,
                "evaluated": False,
                "correct": False,
                "score": 0.0,
                "official_score": None,
            }

        # This is the only path that consults _Record.private_answer.
        evaluation = self._evaluate_record(record, parsed.answer)
        passed = evaluation.passed
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "metric": evaluation.metric,
            "score_scope": self.score_scope,
            "submission_valid": True,
            "normalized_answer": parsed.answer,
            "evaluated": passed is not None,
            "correct": bool(passed),
            "score": float(evaluation.score or 0.0),
            "official_score": None,
        }

    def evaluate_submission(
        self,
        sample_id: str | int,
        submission: ParsedSubmission,
    ) -> dict[str, Any]:
        """Project the private subset check into the canonical result fields."""

        loaded = self.load_sample(sample_id)
        evaluation = self.evaluate_private(loaded, submission)
        valid = bool(evaluation["submission_valid"])
        evaluated = bool(evaluation["evaluated"])
        correct = bool(evaluation["correct"])
        return {
            "execution_success": True,
            "contract_valid": True,
            "submission_valid": valid,
            "task_outcome": (
                "correct"
                if correct
                else "incorrect"
                if valid and evaluated
                else "not_evaluated"
            ),
            "score_kind": self.score_scope,
            "official_score": None,
            "denominator_eligible": True,
            "error_type": (
                None
                if correct
                else "model_answer_error"
                if valid and evaluated
                else "invalid_answer_format"
                if not valid
                else "diagnostic_evaluator_inconclusive"
            ),
            "blocked_reason": None,
        }

    evaluate = evaluate_submission

    def submit(
        self,
        sample_id: str | int,
        raw_submission: Any,
        *,
        known_evidence_ids: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        loaded = self.load_sample(sample_id)
        parsed = self.parse_submission(
            loaded,
            raw_submission,
            known_evidence_ids=known_evidence_ids,
        )
        return self.evaluate_submission(sample_id, parsed)

    def _close_resources(self) -> None:
        self._records.clear()
        self._excluded.clear()
        self._asset_paths.clear()
        self._asset_formats.clear()
        self._materialized.clear()
        self._loaded_records.clear()
        self._loaded_samples.clear()
        self._sample_assets.clear()
        self._derived_descriptors.clear()
        self._action_cache.clear()
        self._media_operations.clear()
        self._evidence.clear()
        self._state_revisions.clear()


@register_adapter
class ThreeDSRBenchAdapter(StaticImageBenchmarkAdapter):
    """3DSRBench v1 static-image diagnostic adapter."""

    benchmark_id = "3DSRBench"
    package_id = "3dsrbench"
    aliases = ("3DSR", "3DSRBench-v1")
    score_scope = "diagnostic"
    denominator_policy = "materialized_rows_with_readable_public_image"
    observation_policy = "single_image_full_context"

    def _load_records(self) -> None:
        dataset = self.data_root / "dataset"
        annotations = dataset / "3dsrbench_v1-00000-of-00001.parquet"
        archive = dataset / "coco_images.zip"
        if not annotations.is_file() or not archive.is_file():
            raise AdapterDataError("3DSRBench parquet or COCO image archive is missing")
        parquet = _import_parquet()
        columns = [
            "index",
            "question",
            "A",
            "B",
            "C",
            "D",
            "answer",
            "category",
            "image_source",
            "image_url",
        ]
        rows = parquet.read_table(annotations, columns=columns).to_pylist()
        with zipfile.ZipFile(archive) as source:
            members_by_name: dict[str, list[str]] = {}
            for info in source.infolist():
                if info.is_dir():
                    continue
                members_by_name.setdefault(Path(info.filename).name, []).append(
                    info.filename
                )
        self._archive = archive
        for raw in rows:
            sample_id = _nonempty(raw.get("index"), "3DSRBench index")
            basename = Path(urlsplit(_nonempty(raw.get("image_url"), "image_url")).path).name
            candidates = members_by_name.get(basename, [])
            if len(candidates) != 1:
                self._exclude(sample_id, "data_missing")
                continue
            choices = tuple(
                f"{label}. {str(raw[label]).strip()}"
                for label in ("A", "B", "C", "D")
                if raw.get(label) is not None and str(raw[label]).strip()
            )
            if len(choices) < 2:
                raise AdapterDataError("3DSRBench row has fewer than two choices")
            self._add_record(
                _Record(
                    sample_id=sample_id,
                    question=_nonempty(raw.get("question"), "question"),
                    choices=choices,
                    answer_format="multiple_choice",
                    category=_nonempty(raw.get("category"), "category"),
                    public_metadata={
                        "official_split": "benchmark_v1",
                        "category": str(raw["category"]),
                        "image_source": str(raw.get("image_source") or "unknown"),
                    },
                    private_answer=_nonempty(raw.get("answer"), "answer"),
                    source=candidates[0],
                )
            )
        self._released_count = len(rows)

    def _image_payloads(self, record: _Record) -> Sequence[_ImagePayload]:
        try:
            with zipfile.ZipFile(self._archive) as source:
                payload = source.read(str(record.source))
        except (KeyError, OSError, zipfile.BadZipFile) as exc:
            raise AdapterDataError("3DSRBench image archive member is unreadable") from exc
        return (_ImagePayload("rgb", payload),)


@register_adapter
class RoboSpatialHomeAdapter(StaticImageBenchmarkAdapter):
    """Self-contained RoboSpatial-Home subset adapter.

    Only the three public Home parquet splits are inspected.  Broader
    RoboSpatial/EmbodiedScan assets are neither discovered nor accepted.
    """

    benchmark_id = "RoboSpatial-Home"
    package_id = "robospatial"
    aliases = ("RoboSpatial", "robospatial-home")
    score_scope = "subset"
    denominator_policy = "self_contained_home_rows_with_readable_public_image"
    observation_policy = "single_image_home_subset"

    def _load_records(self) -> None:
        dataset = self.data_root / "dataset"
        readme = dataset / "README.md"
        if readme.is_file() and "pretty_name: robospatial-home" not in readme.read_text(
            encoding="utf-8"
        ).casefold():
            raise AdapterDataError("RoboSpatial data root is not the public Home release")
        parquet = _import_parquet()
        total = 0
        for split in ("compatibility", "configuration", "context"):
            path = dataset / "data" / f"{split}-00000-of-00001.parquet"
            if not path.is_file():
                raise AdapterDataError(f"RoboSpatial-Home split is missing: {split}")
            rows = parquet.read_table(
                path,
                columns=["category", "question", "answer"],
            ).to_pylist()
            for index, raw in enumerate(rows):
                category = _nonempty(raw.get("category"), "category").casefold()
                if category != split:
                    raise AdapterDataError("RoboSpatial-Home category/split mismatch")
                sample_id = f"{split}_{index:04d}"
                is_context = split == "context"
                self._add_record(
                    _Record(
                        sample_id=sample_id,
                        question=_nonempty(raw.get("question"), "question"),
                        choices=("A. Yes", "B. No") if not is_context else (),
                        answer_format=(
                            "normalized_point_list" if is_context else "yes_no"
                        ),
                        category=split,
                        public_metadata={
                            "official_split": split,
                            "dataset_scope": "RoboSpatial-Home",
                            "coordinate_space": (
                                "normalized_image" if is_context else "not_applicable"
                            ),
                        },
                        private_answer=_nonempty(raw.get("answer"), "answer"),
                        source=(path, index),
                    )
                )
            total += len(rows)
        self._released_count = total

    def _image_payloads(self, record: _Record) -> Sequence[_ImagePayload]:
        path, index = record.source
        raw = _parquet_row(path, index, ("img", "depth_image", "mask"))
        payloads: list[_ImagePayload] = []
        for field, role in (
            ("img", "rgb"),
            ("depth_image", "provided_depth_map"),
            ("mask", "provided_mask"),
        ):
            value = raw.get(field)
            payload = value.get("bytes") if isinstance(value, Mapping) else None
            if payload:
                payloads.append(_ImagePayload(role, bytes(payload)))
            elif field == "img":
                raise AdapterDataError("RoboSpatial-Home RGB payload is missing")
        return tuple(payloads)


@register_adapter
class BOPASKAdapter(StaticImageBenchmarkAdapter):
    """Validated 351-row binary subset over the released core/lab files."""

    benchmark_id = "BOP-ASK"
    package_id = "bop_ask"
    aliases = ("BOPASK",)
    score_scope = "subset"
    denominator_policy = (
        "validated_binary_rows_excluding_data_missing_and_unscored_nonbinary"
    )
    observation_policy = "single_image_validated_subset"

    def _load_records(self) -> None:
        dataset = self.data_root / "dataset"
        total = 0
        for split, relative in _BOP_SOURCE_FILES:
            path = dataset / relative
            if not path.is_file():
                raise AdapterDataError(f"BOP-ASK annotation file is missing: {relative}")
            try:
                rows = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise AdapterDataError("BOP-ASK annotation JSON is unreadable") from exc
            if not isinstance(rows, list):
                raise AdapterDataError("BOP-ASK annotation root must be a list")
            for index, raw in enumerate(rows):
                if not isinstance(raw, Mapping):
                    raise AdapterDataError("BOP-ASK row must be an object")
                sample_id = f"{split}_{index:04d}"
                question_type = _nonempty(raw.get("question_type"), "question_type").casefold()
                subtype = _nonempty(raw.get("question_subtype"), "question_subtype").casefold()
                if (question_type, subtype) not in _BOP_SUPPORTED_FAMILIES:
                    self._exclude(sample_id, "unsupported_task_semantics")
                    continue
                messages = raw.get("messages")
                if not isinstance(messages, list):
                    raise AdapterDataError("BOP-ASK messages must be a list")
                user_messages = [
                    item.get("content")
                    for item in messages
                    if isinstance(item, Mapping) and item.get("role") == "user"
                ]
                answer_messages = [
                    item.get("content")
                    for item in messages
                    if isinstance(item, Mapping) and item.get("role") == "assistant"
                ]
                if len(user_messages) != 1 or len(answer_messages) != 1:
                    raise AdapterDataError("BOP-ASK row must have one user/assistant turn")

                locators: list[tuple[str, Path]] = []
                missing_required = False
                for field, role, required in (
                    ("images", "rgb", True),
                    (
                        "depths",
                        "provided_depth_map",
                        question_type == "depth_relative",
                    ),
                    ("masks", "provided_mask", False),
                ):
                    values = raw.get(field) or []
                    if not isinstance(values, list):
                        raise AdapterDataError(f"BOP-ASK {field} must be a list")
                    if required and not values:
                        missing_required = True
                    for value in values:
                        candidate = _safe_child(dataset, value)
                        if candidate.is_file():
                            locators.append((role, candidate))
                        elif required:
                            missing_required = True
                if missing_required:
                    self._exclude(sample_id, "data_missing")
                    continue

                is_yes_no = question_type in {"depth_relative", "spatial_reasoning"}
                answer_format = {
                    ("grasp", "2dplane"): "grasp_points",
                    ("object_rearrangement", "point_wise"): "object_markers",
                    ("pose", "2dbbox"): "bbox_points",
                    ("trajectory", "2d"): "trajectory_points",
                }.get((question_type, subtype), "yes_no")
                self._add_record(
                    _Record(
                        sample_id=sample_id,
                        question=_strip_image_token(user_messages[0]),
                        choices=("A. yes", "B. no") if is_yes_no else (),
                        answer_format=answer_format,
                        category=f"{question_type}:{subtype}",
                        public_metadata={
                            "official_split": split,
                            "question_type": question_type,
                            "question_subtype": subtype,
                            "coordinate_frame": (
                                "image_pixels" if not is_yes_no else "not_applicable"
                            ),
                        },
                        private_answer=_nonempty(answer_messages[0], "answer"),
                        source=tuple(locators),
                    )
                )
            total += len(rows)
        self._released_count = total

    def _image_payloads(self, record: _Record) -> Sequence[_ImagePayload]:
        payloads: list[_ImagePayload] = []
        for role, path in record.source:
            try:
                payloads.append(_ImagePayload(role, path.read_bytes()))
            except OSError as exc:
                raise AdapterDataError("BOP-ASK public image became unreadable") from exc
        return tuple(payloads)

    def _evaluate_record(self, record: _Record, prediction: Any) -> Any:
        return BOPASKEvaluatorV1().evaluate(
            prediction,
            record.private_answer,
            answer_type="short_text",
            evaluator="bop_ask_v1",
            choices=record.choices,
            question=record.question,
            metadata=dict(record.public_metadata),
        )


__all__ = [
    "AdapterDataError",
    "BOPASKAdapter",
    "DataMissingError",
    "OfflineStaticAdapterError",
    "RoboSpatialHomeAdapter",
    "STATIC_IMAGE_ACTIONS",
    "StaticImageBenchmarkAdapter",
    "SubmissionParseError",
    "ThreeDSRBenchAdapter",
    "UnknownSampleError",
]
