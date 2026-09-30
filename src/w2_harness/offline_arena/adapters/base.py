"""Unified contracts for read-only CPU benchmark adapters.

Adapters own the private annotation state.  Objects returned to a model-facing
caller contain questions, choices, and opaque asset identifiers, but never a
reference answer or an annotation path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


PUBLIC_TASK_SCHEMA_VERSION = "w2-offline-public-task-v1.0"
ASSET_CATALOG_SCHEMA_VERSION = "w2-offline-asset-catalog-v1.0"
ACTION_POLICY_SCHEMA_VERSION = "w2-offline-action-policy-v1.0"
MEDIA_PLAN_SCHEMA_VERSION = "w2-offline-media-plan-v1.0"
OBSERVATION_SCHEMA_VERSION = "w2-offline-observation-v1.0"
EVALUATION_SCHEMA_VERSION = "w2-offline-private-evaluation-v1.0"

_OPAQUE_ID_VERSION = "w2-offline-opaque-id-v1"
_MISSING = object()
_PRIVATE_METADATA_KEYS = frozenset(
    {
        "answer",
        "correct_answer",
        "expected_answer",
        "ground_truth",
        "gt_answer",
        "hidden_label",
        "private_reference",
        "private_truth",
        "rationale",
        "reference_answer",
        "target_answer",
        "thought",
    }
)
_PRIVATE_METADATA_KEY_PARTS = ("ground_truth", "reference")


class AdapterError(RuntimeError):
    """Base class for deterministic adapter failures."""


class DataUnavailableError(AdapterError):
    """Required local benchmark data is absent or unreadable."""

    def __init__(
        self,
        benchmark_id: str,
        reason: str,
        *,
        path: Path | str | None = None,
    ) -> None:
        self.benchmark_id = benchmark_id
        self.reason = reason
        self.path = None if path is None else Path(path)
        suffix = "" if self.path is None else f" ({self.path})"
        super().__init__(f"{benchmark_id}: {reason}{suffix}")


# A descriptive alias makes it easy for callers to catch typed data absence.
DataAbsenceError = DataUnavailableError


class SampleNotFoundError(AdapterError):
    """The annotation source exists, but it has no requested sample."""


class InvalidSubmissionError(AdapterError):
    """A private evaluation was requested without a parsed submission."""


class AdapterClosedError(AdapterError):
    """An operation was attempted after adapter resources were closed."""


class AdapterRegistrationError(AdapterError):
    """Adapter registration is duplicate or malformed."""


def _json_copy(value: Any, label: str) -> Any:
    try:
        return json.loads(
            json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise AdapterError(f"{label} must contain finite JSON values") from exc


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AdapterError(f"{label} must be a non-empty string")
    return value.strip()


def _opaque_id(kind: str, namespace: str, source_id: str, ordinal: int | None = None) -> str:
    fields = [_OPAQUE_ID_VERSION, kind, namespace, source_id]
    if ordinal is not None:
        fields.append(str(ordinal))
    digest = hashlib.sha256("\0".join(fields).encode("utf-8")).hexdigest()[:32]
    return f"{kind}_{digest}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.strip().split()).casefold()


@dataclass(frozen=True)
class Choice:
    """One visible multiple-choice option."""

    label: str
    text: str

    def __post_init__(self) -> None:
        label = str(self.label).strip().upper()
        text = str(self.text).strip()
        if not re.fullmatch(r"[A-Z]", label):
            raise AdapterError("choice labels must be one uppercase letter")
        if not text:
            raise AdapterError("choice text must be non-empty")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "text", text)

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "text": self.text}


MultipleChoiceOption = Choice


@dataclass(frozen=True)
class AdapterSample(Mapping[str, Any]):
    """Loaded sample handle with no annotation truth or local paths."""

    task_id: str
    source_sample_id: str
    prompt: str
    choices: tuple[Choice, ...]
    asset_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def sample_id(self) -> str:
        return self.source_sample_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "source_sample_id": self.source_sample_id,
            "prompt": self.prompt,
            "choices": [choice.to_dict() for choice in self.choices],
            "asset_ids": list(self.asset_ids),
            "metadata": _json_copy(self.metadata, "sample metadata"),
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class ParsedSubmission(Mapping[str, Any]):
    """Reference-free normalized submission produced by one adapter instance."""

    task_id: str
    answer: str | None
    valid: bool
    issues: tuple[str, ...]
    raw_sha256: str
    _adapter_capability: object = field(repr=False, compare=False)

    @property
    def normalized_answer(self) -> str | None:
        return self.answer

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "answer": self.answer,
            "valid": self.valid,
            "issues": list(self.issues),
            "raw_sha256": self.raw_sha256,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def _coerce_choices(choices: Sequence[Choice | Mapping[str, Any] | str]) -> tuple[Choice, ...]:
    output: list[Choice] = []
    for index, value in enumerate(choices):
        if isinstance(value, Choice):
            choice = value
        elif isinstance(value, Mapping):
            choice = Choice(label=str(value.get("label", "")), text=str(value.get("text", "")))
        else:
            text = str(value).strip()
            match = re.match(r"^([A-Z])[.:)]\s*(.+)$", text, flags=re.DOTALL)
            if match is None:
                choice = Choice(label=chr(65 + index), text=text)
            else:
                choice = Choice(label=match.group(1), text=match.group(2))
        output.append(choice)
    if len(output) < 2:
        raise AdapterError("multiple-choice samples require at least two choices")
    labels = [choice.label for choice in output]
    if len(labels) != len(set(labels)):
        raise AdapterError("multiple-choice labels must be unique")
    return tuple(output)


def _answer_value(raw: Any) -> Any:
    if isinstance(raw, ParsedSubmission):
        return raw.answer
    if isinstance(raw, Mapping):
        for key in ("answer", "final_answer", "prediction", "output"):
            if key in raw:
                return raw[key]
        return None
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if text.startswith("{") or text.startswith('"'):
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if decoded != raw:
                return _answer_value(decoded)
    return raw


def normalize_mcq_answer(
    raw: Any,
    choices: Sequence[Choice | Mapping[str, Any] | str],
) -> str | None:
    """Normalize a model MCQ response without consulting the private label."""

    normalized_choices = _coerce_choices(choices)
    labels = {choice.label for choice in normalized_choices}
    value = _answer_value(raw)
    if value is None or isinstance(value, (list, tuple, set, dict)):
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None

    direct = re.fullmatch(r"[\[(]?\s*([A-Za-z])\s*[\])\].:]?", text)
    if direct is not None and direct.group(1).upper() in labels:
        return direct.group(1).upper()

    boxed = re.fullmatch(r"\\boxed\{\s*([A-Za-z])\s*\}", text)
    if boxed is not None and boxed.group(1).upper() in labels:
        return boxed.group(1).upper()

    labelled = re.search(
        r"(?:^|\b)(?:final\s+answer|answer|choice|option)\s*"
        r"(?:is\s*)?(?::|=)?\s*[\[(]?([A-Za-z])[\])\].:]?(?:\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if labelled is not None and labelled.group(1).upper() in labels:
        return labelled.group(1).upper()

    candidate = _normalized_text(text)
    matches: list[str] = []
    for choice in normalized_choices:
        variants = {
            _normalized_text(choice.text),
            _normalized_text(f"{choice.label}. {choice.text}"),
            _normalized_text(f"{choice.label}: {choice.text}"),
            _normalized_text(f"{choice.label}) {choice.text}"),
        }
        if candidate in variants:
            matches.append(choice.label)
    return matches[0] if len(set(matches)) == 1 else None


normalize_multiple_choice = normalize_mcq_answer


def _data_layout_candidates(root: Path, package_id: str) -> tuple[Path, ...]:
    return (
        root,
        root / package_id,
        root / "full" / package_id,
        root / "data" / "full" / package_id,
        root / "runtime" / "data" / "full" / package_id,
    )


def _git_link_roots(anchor: Path) -> tuple[Path, ...]:
    roots: list[Path] = []
    for parent in (anchor, *anchor.parents):
        dot_git = parent / ".git"
        if dot_git.is_dir():
            roots.extend((parent, parent.parent))
            break
        if not dot_git.is_file():
            continue
        try:
            prefix, raw_git_dir = dot_git.read_text(encoding="utf-8").strip().split(":", 1)
            if prefix.strip() != "gitdir":
                break
            git_dir = Path(raw_git_dir.strip())
            if not git_dir.is_absolute():
                git_dir = (parent / git_dir).resolve()
            common_file = git_dir / "commondir"
            common_dir = (
                (git_dir / common_file.read_text(encoding="utf-8").strip()).resolve()
                if common_file.is_file()
                else git_dir
            )
        except (OSError, ValueError):
            break
        primary_worktree = common_dir.parent
        roots.extend((parent, primary_worktree, primary_worktree.parent))
        break
    return tuple(roots)


def discover_data_root(package_id: str, *, benchmark_id: str | None = None) -> Path:
    """Find ``runtime/data/full/<package>`` without reading project secrets."""

    package = _nonempty(package_id, "package_id")
    benchmark = benchmark_id or package
    anchors = [Path.cwd(), Path(__file__).resolve()]
    configured: list[Path] = []
    for name in ("W2_DATA_ROOT", "W2_RUNTIME_ROOT", "W2_PROJECT_ROOT"):
        value = os.environ.get(name)
        if value:
            configured.append(Path(value).expanduser())

    roots: list[Path] = configured
    for anchor in anchors:
        roots.extend((anchor, *anchor.parents))
        roots.extend(_git_link_roots(anchor))
    seen: set[Path] = set()
    for root in roots:
        for candidate in _data_layout_candidates(root.resolve(), package):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.is_dir() and (candidate / "dataset").is_dir():
                return candidate
    raise DataUnavailableError(
        benchmark,
        f"local runtime/data/full/{package} layout was not found",
    )


def resolve_data_root(
    data_root: Path | str | None,
    *,
    package_id: str,
    benchmark_id: str,
) -> Path:
    if data_root is None:
        return discover_data_root(package_id, benchmark_id=benchmark_id)
    root = Path(data_root).expanduser().resolve()
    for candidate in _data_layout_candidates(root, package_id):
        if candidate.is_dir() and (candidate / "dataset").is_dir():
            return candidate.resolve()
    return root


@dataclass(frozen=True)
class _AssetRecord:
    asset_id: str
    path: Path
    mime_type: str
    sequence_index: int
    byte_size: int
    content_sha256: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "uri": f"asset://{self.asset_id}",
            "media_type": "image",
            "mime_type": self.mime_type,
            "sequence_index": self.sequence_index,
            "byte_size": self.byte_size,
            "content_sha256": self.content_sha256,
        }

    def transport_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "media_type": "image",
            "mime_type": self.mime_type,
            "sequence_index": self.sequence_index,
            "byte_size": self.byte_size,
            "content_sha256": self.content_sha256,
            "local_path": str(self.path),
        }


@dataclass(frozen=True)
class _PrivateSampleState:
    sample: AdapterSample
    reference_label: str
    assets: tuple[_AssetRecord, ...]


@runtime_checkable
class Adapter(Protocol):
    """One benchmark adapter consumed identically by direct and light harnesses."""

    benchmark_id: str
    package_id: str

    def load_sample(self, sample_id: str | int | Mapping[str, Any] | None = None) -> AdapterSample:
        ...

    def build_public_task(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def build_asset_catalog(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def build_action_policy(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def build_full_context_media_plan(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def build_selective_initial_observation(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def build_submission_schema(self, sample: AdapterSample | None = None) -> dict[str, Any]:
        ...

    def parse_submission(self, sample: AdapterSample, submission: Any) -> ParsedSubmission:
        ...

    def evaluate_private(self, sample: AdapterSample, submission: ParsedSubmission) -> dict[str, Any]:
        ...

    def close(self) -> None:
        ...


class BaseAdapter(ABC):
    """Common boundary, media, submission, and lifecycle implementation."""

    benchmark_id = ""
    package_id = ""
    aliases: tuple[str, ...] = ()
    observation_policy = "selective_multiview"
    initial_asset_limit = 1

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        cache_root: Path | str | None = None,
    ) -> None:
        if not self.benchmark_id or not self.package_id:
            raise AdapterError("adapter benchmark_id and package_id must be declared")
        self.data_root = resolve_data_root(
            data_root,
            package_id=self.package_id,
            benchmark_id=self.benchmark_id,
        )
        self._configured_cache_root = (
            None if cache_root is None else Path(cache_root).expanduser().resolve()
        )
        self._temporary_cache: tempfile.TemporaryDirectory[str] | None = None
        self._states: dict[str, _PrivateSampleState] = {}
        self._source_to_task: dict[str, str] = {}
        self._current_task_id: str | None = None
        self._submission_capability = object()
        self._closed = False

    @abstractmethod
    def load_sample(
        self,
        sample_id: str | int | Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        """Load one real source row and its readable media."""

    def _ensure_open(self) -> None:
        if self._closed:
            raise AdapterClosedError(f"{self.benchmark_id} adapter is closed")

    def _cache_root(self) -> Path:
        self._ensure_open()
        if self._configured_cache_root is not None:
            root = self._configured_cache_root
        else:
            if self._temporary_cache is None:
                self._temporary_cache = tempfile.TemporaryDirectory(
                    prefix=f"w2-{self.package_id}-"
                )
            root = Path(self._temporary_cache.name)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _materialize_bytes(
        self,
        *,
        source_sample_id: str,
        sequence_index: int,
        suffix: str,
        payload: bytes,
    ) -> Path:
        if not payload:
            raise DataUnavailableError(
                self.benchmark_id,
                f"empty media payload for sample {source_sample_id}",
            )
        task_id = _opaque_id("task", self.benchmark_id, source_sample_id)
        asset_id = _opaque_id(
            "asset", self.benchmark_id, source_sample_id, sequence_index
        )
        normalized_suffix = suffix.casefold() if suffix.startswith(".") else f".{suffix.casefold()}"
        output = self._cache_root() / task_id / f"{asset_id}{normalized_suffix}"
        if output.is_file() and output.stat().st_size == len(payload):
            return output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f"{output.name}.tmp.{os.getpid()}")
        temporary.write_bytes(payload)
        temporary.replace(output)
        return output.resolve()

    def _store_sample(
        self,
        *,
        source_sample_id: str,
        prompt: str,
        choices: Sequence[Choice | Mapping[str, Any] | str],
        reference_label: Any,
        media: Sequence[tuple[Path, str]],
        metadata: Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        self._ensure_open()
        source_id = _nonempty(str(source_sample_id), "source sample ID")
        public_prompt = _nonempty(prompt, "sample prompt")
        normalized_choices = _coerce_choices(choices)
        labels = {choice.label for choice in normalized_choices}
        expected = str(reference_label or "").strip().upper()
        if expected not in labels:
            raise AdapterError(
                f"{self.benchmark_id} sample {source_id} has an invalid private label"
            )
        safe_metadata = _json_copy(metadata or {}, "sample metadata")
        if not isinstance(safe_metadata, dict):
            raise AdapterError("sample metadata must be an object")
        for key in safe_metadata:
            normalized_key = str(key).casefold().replace("-", "_")
            if normalized_key in _PRIVATE_METADATA_KEYS or any(
                fragment in normalized_key
                for fragment in _PRIVATE_METADATA_KEY_PARTS
            ):
                raise AdapterError(f"private annotation field cannot enter metadata: {key}")

        task_id = _opaque_id("task", self.benchmark_id, source_id)
        assets: list[_AssetRecord] = []
        for index, (raw_path, mime_type) in enumerate(media):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file() or path.stat().st_size <= 0:
                raise DataUnavailableError(
                    self.benchmark_id,
                    f"sample {source_id} has unreadable view {index}",
                    path=path,
                )
            assets.append(
                _AssetRecord(
                    asset_id=_opaque_id("asset", self.benchmark_id, source_id, index),
                    path=path,
                    mime_type=_nonempty(mime_type, "asset MIME type"),
                    sequence_index=index,
                    byte_size=path.stat().st_size,
                    content_sha256=_sha256_file(path),
                )
            )
        if not assets:
            raise DataUnavailableError(
                self.benchmark_id,
                f"sample {source_id} has no readable views",
            )

        sample = AdapterSample(
            task_id=task_id,
            source_sample_id=source_id,
            prompt=public_prompt,
            choices=normalized_choices,
            asset_ids=tuple(asset.asset_id for asset in assets),
            metadata=safe_metadata,
        )
        self._states[task_id] = _PrivateSampleState(
            sample=sample,
            reference_label=expected,
            assets=tuple(assets),
        )
        self._source_to_task[source_id] = task_id
        self._current_task_id = task_id
        return sample

    def _coerce_sample(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None,
    ) -> AdapterSample:
        self._ensure_open()
        if sample is None:
            task_id = self._current_task_id
        elif isinstance(sample, AdapterSample):
            task_id = sample.task_id
        elif isinstance(sample, Mapping):
            task_id = str(sample.get("task_id") or "")
        else:
            value = str(sample)
            task_id = value if value in self._states else self._source_to_task.get(value)
        if not task_id or task_id not in self._states:
            raise SampleNotFoundError("sample is not loaded by this adapter instance")
        return self._states[task_id].sample

    def _state(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None,
    ) -> _PrivateSampleState:
        loaded = self._coerce_sample(sample)
        return self._states[loaded.task_id]

    def build_public_task(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded = self._coerce_sample(sample)
        return {
            "schema_version": PUBLIC_TASK_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "prompt": loaded.prompt,
            "choices": [choice.to_dict() for choice in loaded.choices],
            "answer_type": "multiple_choice",
            "asset_ids": list(loaded.asset_ids),
        }

    def load_public_task(self, sample_id: str | int) -> dict[str, Any]:
        """Project a loaded row onto the current public-only typed task."""

        loaded = self.load_sample(sample_id)
        source_id = loaded.source_sample_id
        portable_sample_id = (
            source_id
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:+-]{0,191}", source_id)
            else loaded.task_id
        )
        metadata = _json_copy(loaded.metadata, "sample metadata")
        category = str(
            metadata.get("category")
            or metadata.get("question_type")
            or metadata.get("task_type")
            or "unspecified"
        ).strip()
        return {
            "benchmark_id": self.benchmark_id,
            "sample_id": portable_sample_id,
            "question": loaded.prompt,
            "choices": [choice.to_dict() for choice in loaded.choices],
            "answer_format": "multiple_choice",
            "public_metadata": metadata,
            "asset_ids": list(loaded.asset_ids),
            "observation_policy": self.observation_policy,
            "category": category or "unspecified",
            "submission_schema_id": "w2.answer.choice.v1",
        }

    get_public_task = load_public_task

    def list_assets(self, sample_id: str | int) -> tuple[dict[str, Any], ...]:
        """Return canonical descriptors while retaining local paths privately."""

        loaded = self.load_sample(sample_id)
        state = self._state(loaded)
        return tuple(
            {
                "asset_id": asset.asset_id,
                "modality": "image",
                "source_kind": "view",
                "width": None,
                "height": None,
                "frame_count": 1,
                "duration": None,
                "public": True,
                "source_media_hash": asset.content_sha256,
                "metadata": {
                    "mime_type": asset.mime_type,
                    "sequence_index": asset.sequence_index,
                },
            }
            for asset in state.assets
        )

    def resolve_asset_path(self, asset_id: str) -> Path:
        """Resolve an opaque public ID only inside the private adapter process."""

        self._ensure_open()
        for state in self._states.values():
            for asset in state.assets:
                if asset.asset_id == asset_id:
                    return asset.path
        raise SampleNotFoundError("asset ID is not materialized by this adapter")

    resolve_asset = resolve_asset_path

    def materialize_sample(self, sample_id: str | int) -> dict[str, Any]:
        loaded = self.load_sample(sample_id)
        return {
            "public_task": self.load_public_task(loaded.source_sample_id),
            "assets": list(self.list_assets(loaded.source_sample_id)),
            "execution_status": "ready",
            "denominator_eligible": True,
            "score_scope": "diagnostic",
            "official_score": None,
        }

    materialize = materialize_sample
    materialize_task = materialize_sample

    def build_asset_catalog(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._state(sample)
        return {
            "schema_version": ASSET_CATALOG_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "assets": [asset.public_dict() for asset in state.assets],
        }

    def build_action_policy(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded = self._coerce_sample(sample)
        return {
            "schema_version": ACTION_POLICY_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "policy": self.observation_policy,
            "allowed_actions": ["inspect", "submit_answer"],
            "inspect": {
                "required_arguments": ["asset_id"],
                "asset_ids": list(loaded.asset_ids),
                "max_calls": len(loaded.asset_ids),
                "read_only": True,
            },
            "turn_limit": len(loaded.asset_ids) + 1,
        }

    def build_full_context_media_plan(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        state = self._state(sample)
        return {
            "schema_version": MEDIA_PLAN_SCHEMA_VERSION,
            "task_id": state.sample.task_id,
            "strategy": "full_context",
            "media": [asset.transport_dict() for asset in state.assets],
        }

    def build_selective_initial_observation(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded = self._coerce_sample(sample)
        visible_count = min(max(1, self.initial_asset_limit), len(loaded.asset_ids))
        observed = loaded.asset_ids[:visible_count]
        return {
            "schema_version": OBSERVATION_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "observation_policy": self.observation_policy,
            "observed_asset_ids": list(observed),
            "remaining_asset_ids": list(loaded.asset_ids[visible_count:]),
            "observation_complete": visible_count == len(loaded.asset_ids),
        }

    def build_submission_schema(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | None = None,
    ) -> dict[str, Any]:
        loaded = self._coerce_sample(sample)
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Multiple-choice submission",
            "type": "object",
            "additionalProperties": False,
            "required": ["answer"],
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": [choice.label for choice in loaded.choices],
                }
            },
        }

    def parse_submission(
        self,
        sample: AdapterSample | Mapping[str, Any] | str | Any,
        submission: Any = _MISSING,
    ) -> ParsedSubmission:
        if submission is _MISSING:
            raw = sample
            loaded = self._coerce_sample(None)
        else:
            raw = submission
            loaded = self._coerce_sample(sample)
        normalized = normalize_mcq_answer(raw, loaded.choices)
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
            task_id=loaded.task_id,
            answer=normalized,
            valid=normalized is not None,
            issues=() if normalized is not None else ("unrecognized_multiple_choice",),
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
            loaded = self._coerce_sample(None)
        else:
            parsed = submission
            loaded = self._coerce_sample(sample)  # type: ignore[arg-type]
        if not isinstance(parsed, ParsedSubmission):
            raise InvalidSubmissionError(
                "evaluate_private requires the result of parse_submission"
            )
        if parsed._adapter_capability is not self._submission_capability:
            raise InvalidSubmissionError("submission belongs to another adapter instance")
        if parsed.task_id != loaded.task_id:
            raise InvalidSubmissionError("submission and sample task IDs differ")

        # This is the only public method that reads the annotation truth, and it
        # does so only after validating a submission created by parse_submission.
        expected = self._states[loaded.task_id].reference_label
        correct = bool(parsed.valid and parsed.answer == expected)
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "task_id": loaded.task_id,
            "metric": "multiple_choice_accuracy",
            "score_scope": "diagnostic",
            "submission_valid": parsed.valid,
            "normalized_answer": parsed.answer,
            "correct": correct,
            "score": 1.0 if correct else 0.0,
            "official_score": None,
        }

    def _close_resources(self) -> None:
        """Subclass hook for archive/table handles."""

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._close_resources()
        finally:
            self._states.clear()
            self._source_to_task.clear()
            self._current_task_id = None
            if self._temporary_cache is not None:
                self._temporary_cache.cleanup()
                self._temporary_cache = None
            self._closed = True

    def __enter__(self) -> "BaseAdapter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


ADAPTER_REGISTRY: dict[str, type[BaseAdapter]] = {}
_ADAPTER_ALIASES: dict[str, str] = {}


def _adapter_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")


def register_adapter(adapter_type: type[BaseAdapter]) -> type[BaseAdapter]:
    """Register benchmark selection outside all model-facing task logic."""

    if not isinstance(adapter_type, type) or not issubclass(adapter_type, BaseAdapter):
        raise AdapterRegistrationError("registered adapters must extend BaseAdapter")
    canonical = _nonempty(adapter_type.benchmark_id, "adapter benchmark_id")
    existing = ADAPTER_REGISTRY.get(canonical)
    if existing is not None and existing is not adapter_type:
        raise AdapterRegistrationError(f"duplicate adapter for {canonical}")
    ADAPTER_REGISTRY[canonical] = adapter_type
    for name in (canonical, adapter_type.package_id, *adapter_type.aliases):
        key = _adapter_key(name)
        owner = _ADAPTER_ALIASES.get(key)
        if owner is not None and owner != canonical:
            raise AdapterRegistrationError(f"duplicate adapter alias: {name}")
        _ADAPTER_ALIASES[key] = canonical
    return adapter_type


def get_adapter_class(benchmark_id: str) -> type[BaseAdapter]:
    try:
        canonical = _ADAPTER_ALIASES[_adapter_key(benchmark_id)]
        return ADAPTER_REGISTRY[canonical]
    except KeyError as exc:
        raise AdapterRegistrationError(f"no offline adapter registered for {benchmark_id}") from exc


def create_adapter(benchmark_id: str, **kwargs: Any) -> BaseAdapter:
    return get_adapter_class(benchmark_id)(**kwargs)


def registered_adapters() -> tuple[str, ...]:
    return tuple(sorted(ADAPTER_REGISTRY))
