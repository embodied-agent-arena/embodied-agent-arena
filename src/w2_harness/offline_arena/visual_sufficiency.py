"""Typed visual-first modality sufficiency contracts.

This module is deliberately independent of the historical active-visual
six-arm implementation.  It contains no provider or runtime integration and
derives modality policy only from public asset descriptors.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any


VISUAL_FIRST_SEMANTIC = True
VISUAL_FIRST = VISUAL_FIRST_SEMANTIC

E1_MINIMAL_VISUAL_DIRECT = "E1_MINIMAL_VISUAL_DIRECT"
E2_MODALITY_SUFFICIENT_DIRECT = "E2_MODALITY_SUFFICIENT_DIRECT"
E3_ACTIVE_MODALITY_SUFFICIENT = "E3_ACTIVE_MODALITY_SUFFICIENT"

V1_SINGLE_CENTER_FRAME = "V1_SINGLE_CENTER_FRAME"
V3_DUPLICATED_CENTER_FRAME = "V3_DUPLICATED_CENTER_FRAME"
V3_UNIFORM_ORDERED = "V3_UNIFORM_ORDERED"
V3_UNIFORM_SHUFFLED = "V3_UNIFORM_SHUFFLED"
V7_UNIFORM_ORDERED = "V7_UNIFORM_ORDERED"

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,191}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_OR_SENSITIVE_KEY_PARTS = (
    "answer",
    "authorization",
    "credential",
    "ground_truth",
    "groundtruth",
    "local_path",
    "password",
    "private_reference",
    "reference_answer",
    "secret",
    "target_answer",
    "timestamp",
    "token",
)
_MISSING = object()


class VisualSufficiencyContractError(ValueError):
    """A visual sufficiency value is ambiguous, unsafe, or inconsistent."""


class VisualModality(str, Enum):
    """Benchmark-neutral visual modality families."""

    STATIC = "static"
    MULTI_VIEW = "multi_view"
    VIDEO = "video"


class BundlePolicy(str, Enum):
    """Stable policies which affect evidence-bundle validation."""

    MODALITY_EVIDENCE = "modality_evidence"
    V1_SINGLE_CENTER_FRAME = V1_SINGLE_CENTER_FRAME
    V3_DUPLICATED_CENTER_FRAME = V3_DUPLICATED_CENTER_FRAME
    V3_UNIFORM_ORDERED = V3_UNIFORM_ORDERED
    V3_UNIFORM_SHUFFLED = V3_UNIFORM_SHUFFLED
    V7_UNIFORM_ORDERED = V7_UNIFORM_ORDERED


def canonical_sha256(value: Any) -> str:
    """Return a canonical finite-JSON SHA-256 digest."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VisualSufficiencyContractError(
            "canonical values must contain finite JSON"
        ) from exc
    return hashlib.sha256(payload).hexdigest()


def _normalized_key(value: Any) -> str:
    return str(value).strip().casefold().replace("-", "_").replace(" ", "_")


def _safe_json(value: Any, label: str) -> Any:
    """Copy finite public JSON and reject private/audit-bearing keys."""

    def check(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                normalized = _normalized_key(key)
                if any(part in normalized for part in _PRIVATE_OR_SENSITIVE_KEY_PARTS):
                    raise VisualSufficiencyContractError(
                        f"{label} contains a private or sensitive field"
                    )
                check(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                check(child)

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(child) for child in item]
        return item

    check(value)
    try:
        serialized = json.dumps(
            thaw(value),
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
        )
        return copy.deepcopy(json.loads(serialized))
    except (TypeError, ValueError) as exc:
        raise VisualSufficiencyContractError(
            f"{label} must contain finite JSON"
        ) from exc


def _private_json(value: Any, label: str) -> Any:
    """Copy finite audit-only JSON without interpreting its private fields."""

    def thaw(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): thaw(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [thaw(child) for child in item]
        return item

    try:
        return json.loads(
            json.dumps(
                thaw(value),
                allow_nan=False,
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    except (TypeError, ValueError) as exc:
        raise VisualSufficiencyContractError(
            f"{label} must contain finite JSON"
        ) from exc


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value.strip()) is None:
        raise VisualSufficiencyContractError(
            f"{label} must be a portable non-empty identifier"
        )
    return value.strip()


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    lower = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < lower:
        qualifier = "non-negative" if allow_zero else "positive"
        raise VisualSufficiencyContractError(f"{label} must be a {qualifier} integer")
    return value


def _finite_float(
    value: Any,
    label: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise VisualSufficiencyContractError(f"{label} must be finite")
    result = float(value)
    if result < minimum or (maximum is not None and result > maximum):
        raise VisualSufficiencyContractError(f"{label} is outside its allowed range")
    return result


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise VisualSufficiencyContractError(f"{label} must be a sequence")
    return tuple(value)


def _strictly_increasing(values: Sequence[int]) -> bool:
    return all(left < right for left, right in zip(values, values[1:]))


def _coerce_modality(value: VisualModality | str) -> VisualModality:
    if isinstance(value, VisualModality):
        return value
    normalized = _normalized_key(value)
    aliases = {
        "image": VisualModality.STATIC,
        "single_image": VisualModality.STATIC,
        "static": VisualModality.STATIC,
        "multi_image": VisualModality.MULTI_VIEW,
        "multi_view": VisualModality.MULTI_VIEW,
        "multiview": VisualModality.MULTI_VIEW,
        "video": VisualModality.VIDEO,
        "video_frame": VisualModality.VIDEO,
        "video_frames": VisualModality.VIDEO,
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise VisualSufficiencyContractError("unsupported visual modality") from exc


def _coerce_bundle_policy(value: BundlePolicy | str) -> BundlePolicy:
    if isinstance(value, BundlePolicy):
        return value
    aliases = {
        "standard": BundlePolicy.MODALITY_EVIDENCE,
        "duplicated_frame_control": BundlePolicy.V3_DUPLICATED_CENTER_FRAME,
    }
    normalized = str(value).strip()
    if normalized in aliases:
        return aliases[normalized]
    try:
        return BundlePolicy(normalized)
    except (TypeError, ValueError) as exc:
        raise VisualSufficiencyContractError("unsupported bundle policy") from exc


@dataclass(frozen=True)
class VisualFirstArmDefinition:
    """One stable visual-first experiment arm."""

    arm_id: str
    harness_profile: str
    delivery_mode: str
    evidence_requirement: str
    active_acquisition: bool
    submit_requires_sufficiency: bool
    visual_first: bool = VISUAL_FIRST_SEMANTIC

    def __post_init__(self) -> None:
        arm_id = _identifier(self.arm_id, "arm_id")
        expected = {
            E1_MINIMAL_VISUAL_DIRECT: (
                "direct",
                "direct_visual_payload",
                "one_visual",
                False,
                False,
            ),
            E2_MODALITY_SUFFICIENT_DIRECT: (
                "direct",
                "direct_visual_payload",
                "modality_sufficient",
                False,
                True,
            ),
            E3_ACTIVE_MODALITY_SUFFICIENT: (
                "w2_light",
                "active_visual_acquisition",
                "modality_sufficient",
                True,
                True,
            ),
        }
        if arm_id not in expected:
            raise VisualSufficiencyContractError("unknown visual-first arm")
        actual = (
            self.harness_profile,
            self.delivery_mode,
            self.evidence_requirement,
            self.active_acquisition,
            self.submit_requires_sufficiency,
        )
        if actual != expected[arm_id]:
            raise VisualSufficiencyContractError(
                "visual-first arm fields do not match the stable definition"
            )
        if self.visual_first is not True:
            raise VisualSufficiencyContractError("visual-first semantics must be true")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "harness_profile": self.harness_profile,
            "delivery_mode": self.delivery_mode,
            "evidence_requirement": self.evidence_requirement,
            "active_acquisition": self.active_acquisition,
            "submit_requires_sufficiency": self.submit_requires_sufficiency,
            "visual_first": self.visual_first,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_dict())


VISUAL_FIRST_ARM_DEFINITIONS = (
    VisualFirstArmDefinition(
        arm_id=E1_MINIMAL_VISUAL_DIRECT,
        harness_profile="direct",
        delivery_mode="direct_visual_payload",
        evidence_requirement="one_visual",
        active_acquisition=False,
        submit_requires_sufficiency=False,
    ),
    VisualFirstArmDefinition(
        arm_id=E2_MODALITY_SUFFICIENT_DIRECT,
        harness_profile="direct",
        delivery_mode="direct_visual_payload",
        evidence_requirement="modality_sufficient",
        active_acquisition=False,
        submit_requires_sufficiency=True,
    ),
    VisualFirstArmDefinition(
        arm_id=E3_ACTIVE_MODALITY_SUFFICIENT,
        harness_profile="w2_light",
        delivery_mode="active_visual_acquisition",
        evidence_requirement="modality_sufficient",
        active_acquisition=True,
        submit_requires_sufficiency=True,
    ),
)
VISUAL_FIRST_ARMS = MappingProxyType(
    {definition.arm_id: definition for definition in VISUAL_FIRST_ARM_DEFINITIONS}
)


@dataclass(frozen=True)
class VSIFrameArmDefinition:
    """One stable VSI frame-presentation control."""

    arm_id: str
    presented_frame_count: int
    distinct_frame_count: int
    selection_policy: str
    ordered: bool
    minimum_temporal_span_ratio: float
    duplicated_frame_control: bool = False
    conditional: bool = False

    def __post_init__(self) -> None:
        _identifier(self.arm_id, "VSI arm_id")
        presented = _positive_int(
            self.presented_frame_count, "presented_frame_count"
        )
        distinct = _positive_int(self.distinct_frame_count, "distinct_frame_count")
        if distinct > presented:
            raise VisualSufficiencyContractError(
                "distinct_frame_count cannot exceed presented_frame_count"
            )
        _identifier(self.selection_policy, "selection_policy")
        span = _finite_float(
            self.minimum_temporal_span_ratio,
            "minimum_temporal_span_ratio",
            maximum=1.0,
        )
        object.__setattr__(self, "minimum_temporal_span_ratio", span)
        for name in ("ordered", "duplicated_frame_control", "conditional"):
            if not isinstance(getattr(self, name), bool):
                raise VisualSufficiencyContractError(f"{name} must be boolean")
        if self.duplicated_frame_control and not (
            presented == 3 and distinct == 1 and span == 0.0
        ):
            raise VisualSufficiencyContractError(
                "duplicated-frame control must present one frame three times"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "presented_frame_count": self.presented_frame_count,
            "distinct_frame_count": self.distinct_frame_count,
            "selection_policy": self.selection_policy,
            "ordered": self.ordered,
            "minimum_temporal_span_ratio": self.minimum_temporal_span_ratio,
            "duplicated_frame_control": self.duplicated_frame_control,
            "conditional": self.conditional,
        }

    @property
    def canonical_hash(self) -> str:
        return canonical_sha256(self.to_dict())


VSI_FRAME_ARM_DEFINITIONS = (
    VSIFrameArmDefinition(
        arm_id=V1_SINGLE_CENTER_FRAME,
        presented_frame_count=1,
        distinct_frame_count=1,
        selection_policy="single_center_frame",
        ordered=True,
        minimum_temporal_span_ratio=0.0,
    ),
    VSIFrameArmDefinition(
        arm_id=V3_DUPLICATED_CENTER_FRAME,
        presented_frame_count=3,
        distinct_frame_count=1,
        selection_policy="duplicated_center_frame",
        ordered=False,
        minimum_temporal_span_ratio=0.0,
        duplicated_frame_control=True,
    ),
    VSIFrameArmDefinition(
        arm_id=V3_UNIFORM_ORDERED,
        presented_frame_count=3,
        distinct_frame_count=3,
        selection_policy="uniform_temporal_sample",
        ordered=True,
        minimum_temporal_span_ratio=0.60,
    ),
    VSIFrameArmDefinition(
        arm_id=V3_UNIFORM_SHUFFLED,
        presented_frame_count=3,
        distinct_frame_count=3,
        selection_policy="uniform_temporal_sample_shuffled",
        ordered=False,
        minimum_temporal_span_ratio=0.60,
    ),
)
VSI_ARM_DEFINITIONS = VSI_FRAME_ARM_DEFINITIONS
VSI_FRAME_ARMS = MappingProxyType(
    {definition.arm_id: definition for definition in VSI_FRAME_ARM_DEFINITIONS}
)


def conditional_v7_definition(
    *, payload_frame_capacity: int | None
) -> VSIFrameArmDefinition | None:
    """Return V7 only when a caller proves capacity for seven payload frames."""

    if payload_frame_capacity is None:
        return None
    capacity = _positive_int(payload_frame_capacity, "payload_frame_capacity")
    if capacity < 7:
        return None
    return VSIFrameArmDefinition(
        arm_id=V7_UNIFORM_ORDERED,
        presented_frame_count=7,
        distinct_frame_count=7,
        selection_policy="uniform_temporal_sample",
        ordered=True,
        minimum_temporal_span_ratio=0.60,
        conditional=True,
    )


@dataclass(frozen=True)
class ModalityEvidenceBundle:
    """Ordered answer-bearing media evidence with audit-private timestamps."""

    modality: VisualModality | str
    source_asset_ids: tuple[str, ...]
    source_media_hashes: tuple[str, ...]
    operation: str
    ordered: bool
    distinct_count: int
    presented_count: int
    temporal_span_ratio: float | None
    frame_indices: tuple[int, ...]
    private_audit_timestamps: tuple[str | float, ...] = field(
        default=(), repr=False, compare=False
    )
    private_provenance: Mapping[str, Any] = field(
        default_factory=dict, repr=False, compare=False
    )
    sequence_positions: tuple[int, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    payload_bytes: int = 0
    bundle_policy: BundlePolicy | str = BundlePolicy.MODALITY_EVIDENCE
    safe_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        modality = _coerce_modality(self.modality)
        object.__setattr__(self, "modality", modality)
        operation = _identifier(str(self.operation).upper(), "operation")
        object.__setattr__(self, "operation", operation)
        if not isinstance(self.ordered, bool):
            raise VisualSufficiencyContractError("ordered must be boolean")

        asset_ids = tuple(
            _identifier(item, "source_asset_ids")
            for item in _sequence(self.source_asset_ids, "source_asset_ids")
        )
        media_hashes = _sequence(self.source_media_hashes, "source_media_hashes")
        if not all(
            isinstance(item, str) and _SHA256.fullmatch(item)
            for item in media_hashes
        ):
            raise VisualSufficiencyContractError(
                "source_media_hashes must contain lowercase SHA-256 values"
            )
        media_hashes = tuple(media_hashes)
        object.__setattr__(self, "source_asset_ids", asset_ids)
        object.__setattr__(self, "source_media_hashes", media_hashes)

        presented = _positive_int(self.presented_count, "presented_count")
        distinct = _positive_int(self.distinct_count, "distinct_count")
        if distinct > presented:
            raise VisualSufficiencyContractError(
                "distinct_count cannot exceed presented_count"
            )
        if len(asset_ids) != presented or len(media_hashes) != presented:
            raise VisualSufficiencyContractError(
                "ordered source identities must match presented_count"
            )

        positions = tuple(
            _positive_int(item, "sequence_positions", allow_zero=True)
            for item in _sequence(self.sequence_positions, "sequence_positions")
        )
        if len(positions) != presented or len(set(positions)) != presented:
            raise VisualSufficiencyContractError(
                "sequence_positions must uniquely cover every presentation"
            )
        object.__setattr__(self, "sequence_positions", positions)

        indices = tuple(
            _positive_int(item, "frame_indices", allow_zero=True)
            for item in _sequence(self.frame_indices, "frame_indices")
        )
        if modality is VisualModality.VIDEO:
            if len(indices) != presented:
                raise VisualSufficiencyContractError(
                    "video bundles require one frame index per presentation"
                )
        elif indices:
            raise VisualSufficiencyContractError(
                "frame_indices are only valid for video evidence"
            )
        object.__setattr__(self, "frame_indices", indices)

        timestamps = _sequence(
            self.private_audit_timestamps, "private_audit_timestamps"
        )
        if len(timestamps) not in {0, presented}:
            raise VisualSufficiencyContractError(
                "private_audit_timestamps must be empty or presentation-aligned"
            )
        normalized_timestamps: list[str | float] = []
        for value in timestamps:
            if isinstance(value, str):
                if not value.strip():
                    raise VisualSufficiencyContractError(
                        "private audit timestamps must not be empty"
                    )
                normalized_timestamps.append(value.strip())
            else:
                normalized_timestamps.append(
                    _finite_float(value, "private audit timestamp")
                )
        object.__setattr__(
            self, "private_audit_timestamps", tuple(normalized_timestamps)
        )
        private_provenance = _private_json(
            self.private_provenance, "private_provenance"
        )
        if not isinstance(private_provenance, dict):
            raise VisualSufficiencyContractError(
                "private_provenance must be an audit-only object"
            )
        object.__setattr__(
            self, "private_provenance", MappingProxyType(private_provenance)
        )

        refs = tuple(
            _identifier(item, "evidence_refs")
            for item in _sequence(self.evidence_refs, "evidence_refs")
        )
        if len(refs) not in {0, presented}:
            raise VisualSufficiencyContractError(
                "evidence refs must be absent before observation or presentation-aligned"
            )
        object.__setattr__(self, "evidence_refs", refs)

        payload_bytes = _positive_int(
            self.payload_bytes, "payload_bytes", allow_zero=operation == "LIST_ASSETS"
        )
        object.__setattr__(self, "payload_bytes", payload_bytes)
        policy = _coerce_bundle_policy(self.bundle_policy)
        object.__setattr__(self, "bundle_policy", policy)

        if self.temporal_span_ratio is None:
            span = None
        else:
            span = _finite_float(
                self.temporal_span_ratio,
                "temporal_span_ratio",
                maximum=1.0,
            )
        if modality is VisualModality.VIDEO:
            if span is None:
                raise VisualSufficiencyContractError(
                    "video bundles require temporal_span_ratio"
                )
        elif span not in {None, 0.0}:
            raise VisualSufficiencyContractError(
                "temporal_span_ratio is only meaningful for video evidence"
            )
        object.__setattr__(self, "temporal_span_ratio", span)

        safe = _safe_json(self.safe_provenance, "safe_provenance")
        if not isinstance(safe, dict):
            raise VisualSufficiencyContractError("safe_provenance must be an object")
        object.__setattr__(self, "safe_provenance", MappingProxyType(safe))

        duplicate_control = policy is BundlePolicy.V3_DUPLICATED_CENTER_FRAME
        self._validate_distinctness(duplicate_control)
        self._validate_policy_shape(policy)
        if self.ordered:
            if not _strictly_increasing(positions):
                raise VisualSufficiencyContractError(
                    "ordered bundles require increasing sequence positions"
                )
            if indices and not _strictly_increasing(indices):
                raise VisualSufficiencyContractError(
                    "ordered video bundles require increasing frame indices"
                )

    def _validate_distinctness(self, duplicate_control: bool) -> None:
        presented = self.presented_count
        if duplicate_control:
            if self.modality is not VisualModality.VIDEO:
                raise VisualSufficiencyContractError(
                    "duplicated-frame control is valid only for video"
                )
            if not (
                presented == 3
                and self.distinct_count == 1
                and len(set(self.frame_indices)) == 1
                and len(set(self.source_media_hashes)) == 1
            ):
                raise VisualSufficiencyContractError(
                    "duplicated-frame control must repeat one frame three times"
                )
            return

        if self.modality is VisualModality.MULTI_VIEW:
            if len(set(self.source_asset_ids)) != presented or len(
                set(self.source_media_hashes)
            ) != presented:
                raise VisualSufficiencyContractError(
                    "duplicate views require an explicit duplicated-frame control"
                )
            computed = presented
        elif self.modality is VisualModality.VIDEO:
            if len(set(self.frame_indices)) != presented or len(
                set(self.source_media_hashes)
            ) != presented:
                raise VisualSufficiencyContractError(
                    "repeated video frames require the duplicated-frame control"
                )
            computed = presented
        else:
            computed = min(
                len(set(self.source_asset_ids)), len(set(self.source_media_hashes))
            )
        if self.distinct_count != computed:
            raise VisualSufficiencyContractError(
                "distinct_count does not match source evidence identity"
            )

    def _validate_policy_shape(self, policy: BundlePolicy) -> None:
        if policy is BundlePolicy.V1_SINGLE_CENTER_FRAME and not (
            self.modality is VisualModality.VIDEO
            and self.presented_count == self.distinct_count == 1
            and self.temporal_span_ratio == 0.0
        ):
            raise VisualSufficiencyContractError("V1 bundle shape is invalid")
        if policy is BundlePolicy.V3_DUPLICATED_CENTER_FRAME and not (
            self.ordered is False and self.temporal_span_ratio == 0.0
        ):
            raise VisualSufficiencyContractError("V3 duplicate control is invalid")
        if policy in {
            BundlePolicy.V3_UNIFORM_ORDERED,
            BundlePolicy.V3_UNIFORM_SHUFFLED,
        }:
            expected_order = policy is BundlePolicy.V3_UNIFORM_ORDERED
            if not (
                self.modality is VisualModality.VIDEO
                and self.presented_count == self.distinct_count == 3
                and self.ordered is expected_order
                and self.temporal_span_ratio is not None
                and self.temporal_span_ratio >= 0.60
            ):
                raise VisualSufficiencyContractError(
                    "V3 uniform bundle shape is invalid"
                )
        if policy is BundlePolicy.V7_UNIFORM_ORDERED and not (
            self.modality is VisualModality.VIDEO
            and self.presented_count == self.distinct_count == 7
            and self.ordered is True
            and self.temporal_span_ratio is not None
            and self.temporal_span_ratio >= 0.60
        ):
            raise VisualSufficiencyContractError("V7 uniform bundle shape is invalid")

    def _identity_projection(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "source_asset_ids": list(self.source_asset_ids),
            "source_media_hashes": list(self.source_media_hashes),
            "operation": self.operation,
            "ordered": self.ordered,
            "distinct_count": self.distinct_count,
            "presented_count": self.presented_count,
            "temporal_span_ratio": self.temporal_span_ratio,
            "frame_indices": list(self.frame_indices),
            "sequence_positions": list(self.sequence_positions),
            "evidence_refs": list(self.evidence_refs),
            "payload_bytes": self.payload_bytes,
            "bundle_policy": self.bundle_policy.value,
            "safe_provenance": copy.deepcopy(dict(self.safe_provenance)),
        }

    @property
    def canonical_hash(self) -> str:
        """Stable identity excludes audit-clock values but preserves list order."""

        return canonical_sha256(self._identity_projection())

    def to_model_visible_dict(self) -> dict[str, Any]:
        # Do not reveal condition, true frame order/index, temporal coverage,
        # source hashes, or private identity commitments to the model.
        return {
            "modality": self.modality.value,
            "operation": self.operation,
            "source_asset_ids": list(self.source_asset_ids),
            "presented_count": self.presented_count,
            "distinct_count": self.distinct_count,
            "sequence_positions": list(self.sequence_positions),
            "evidence_refs": list(self.evidence_refs),
            "payload_bytes": self.payload_bytes,
        }

    def model_visible_projection(self) -> dict[str, Any]:
        return self.to_model_visible_dict()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical public audit projection, never a model payload."""

        projection = self._identity_projection()
        projection["canonical_hash"] = self.canonical_hash
        return projection

    def to_audit_dict(self) -> dict[str, Any]:
        projection = self.to_dict()
        projection["private_audit_timestamps"] = list(
            self.private_audit_timestamps
        )
        projection["private_provenance"] = copy.deepcopy(
            dict(self.private_provenance)
        )
        return projection

    @classmethod
    def from_media_summary(
        cls, summary: Mapping[str, Any]
    ) -> "ModalityEvidenceBundle":
        """Construct from one explicit media summary without adapter imports."""

        if not isinstance(summary, Mapping):
            raise VisualSufficiencyContractError("media summary must be an object")
        required = {
            "modality",
            "source_asset_ids",
            "source_media_hashes",
            "operation",
            "ordered",
            "distinct_count",
            "presented_count",
            "temporal_span_ratio",
            "frame_indices",
            "sequence_positions",
            "evidence_refs",
            "payload_bytes",
            "bundle_policy",
        }
        optional = {
            "canonical_hash",
            "private_audit_timestamps",
            "private_provenance",
            "safe_provenance",
        }
        if not required <= set(summary) or set(summary) - required - optional:
            raise VisualSufficiencyContractError(
                "media summary fields are not canonical"
            )
        bundle = cls(
            modality=summary["modality"],
            source_asset_ids=summary["source_asset_ids"],
            source_media_hashes=summary["source_media_hashes"],
            operation=summary["operation"],
            ordered=summary["ordered"],
            distinct_count=summary["distinct_count"],
            presented_count=summary["presented_count"],
            temporal_span_ratio=summary["temporal_span_ratio"],
            frame_indices=summary["frame_indices"],
            private_audit_timestamps=summary.get("private_audit_timestamps", ()),
            private_provenance=summary.get("private_provenance", {}),
            sequence_positions=summary["sequence_positions"],
            evidence_refs=summary["evidence_refs"],
            payload_bytes=summary["payload_bytes"],
            bundle_policy=summary["bundle_policy"],
            safe_provenance=summary.get("safe_provenance", {}),
        )
        supplied_hash = summary.get("canonical_hash")
        if supplied_hash is not None and supplied_hash != bundle.canonical_hash:
            raise VisualSufficiencyContractError(
                "media summary canonical_hash does not match its content"
            )
        return bundle

    @classmethod
    def from_observed_evidence(
        cls,
        records: Sequence[Any],
        *,
        modality: VisualModality | str,
        operation: str,
        ordered: bool,
        temporal_span_ratio: float | None,
        payload_bytes: int,
        sequence_positions: Sequence[int] | None = None,
        private_audit_timestamps: Sequence[str | float] = (),
        private_provenance: Mapping[str, Any] | None = None,
        bundle_policy: BundlePolicy | str = BundlePolicy.MODALITY_EVIDENCE,
        safe_provenance: Mapping[str, Any] | None = None,
    ) -> "ModalityEvidenceBundle":
        """Bind structurally typed EvidenceRecord-like values into one bundle.

        Only public evidence identity fields are inspected. Record provenance,
        benchmark identity, outcomes, and references are neither consulted nor
        copied; callers may attach audit material explicitly as private
        provenance.
        """

        observed = _sequence(records, "records")
        if not observed:
            raise VisualSufficiencyContractError(
                "at least one observed evidence record is required"
            )
        normalized_modality = _coerce_modality(modality)
        positions = (
            tuple(range(len(observed)))
            if sequence_positions is None
            else tuple(sequence_positions)
        )
        asset_ids: list[str] = []
        media_hashes: list[str] = []
        evidence_refs: list[str] = []
        frame_indices: list[int] = []
        for record in observed:
            asset_ids.append(_record_field(record, "asset_id"))
            media_hashes.append(_record_field(record, "media_hash"))
            evidence_refs.append(_record_field(record, "evidence_id"))
            if normalized_modality is VisualModality.VIDEO:
                frame = _record_field(record, "frame", default=None)
                direct_index = _record_field(record, "frame_index", default=None)
                if direct_index is None and isinstance(frame, Mapping):
                    direct_index = frame.get("frame_index")
                if direct_index is None:
                    raise VisualSufficiencyContractError(
                        "video evidence records require public frame_index"
                    )
                frame_indices.append(direct_index)
        policy = _coerce_bundle_policy(bundle_policy)
        if policy is BundlePolicy.V3_DUPLICATED_CENTER_FRAME:
            distinct_count = 1
        elif normalized_modality is VisualModality.VIDEO:
            distinct_count = min(
                len(set(frame_indices)), len(set(media_hashes))
            )
        else:
            distinct_count = min(len(set(asset_ids)), len(set(media_hashes)))
        return cls(
            modality=normalized_modality,
            source_asset_ids=tuple(asset_ids),
            source_media_hashes=tuple(media_hashes),
            operation=operation,
            ordered=ordered,
            distinct_count=distinct_count,
            presented_count=len(observed),
            temporal_span_ratio=temporal_span_ratio,
            frame_indices=tuple(frame_indices),
            private_audit_timestamps=tuple(private_audit_timestamps),
            private_provenance=private_provenance or {},
            sequence_positions=positions,
            evidence_refs=tuple(evidence_refs),
            payload_bytes=payload_bytes,
            bundle_policy=policy,
            safe_provenance=safe_provenance or {},
        )


def _record_field(record: Any, name: str, default: Any = _MISSING) -> Any:
    if isinstance(record, Mapping):
        value = record.get(name, _MISSING)
    else:
        value = getattr(record, name, _MISSING)
    if value is _MISSING and default is not _MISSING:
        return default
    if value is _MISSING or value is None:
        raise VisualSufficiencyContractError(
            f"observed evidence record lacks public {name}"
        )
    return value


def _descriptor_value(descriptor: Any, field_name: str, default: Any = _MISSING) -> Any:
    if isinstance(descriptor, Mapping):
        value = descriptor.get(field_name, _MISSING)
    else:
        value = getattr(descriptor, field_name, _MISSING)
    if value is _MISSING:
        if default is _MISSING:
            raise VisualSufficiencyContractError(
                f"public asset descriptor lacks {field_name}"
            )
        return default
    return value


def _descriptor_marker(descriptor: Any) -> tuple[bool, bool]:
    modality = _normalized_key(_descriptor_value(descriptor, "modality"))
    source_kind = _normalized_key(_descriptor_value(descriptor, "source_kind"))
    metadata = _descriptor_value(descriptor, "metadata", {})
    if not isinstance(metadata, Mapping):
        raise VisualSufficiencyContractError("public asset metadata must be an object")
    metadata_keys = {_normalized_key(key) for key in metadata}
    frame_count = _descriptor_value(descriptor, "frame_count", None)
    duration = _descriptor_value(descriptor, "duration", None)
    frame_marker = (
        modality in {"video", "video_frame", "video_frames"}
        or "frame" in source_kind
        or "video" in source_kind
        or (
            isinstance(frame_count, int)
            and not isinstance(frame_count, bool)
            and frame_count > 1
        )
        or (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and float(duration) > 0.0
        )
        or bool(
            metadata_keys
            & {
                "frame_index",
                "frozen_frame_ordinal",
                "source_frame_index",
                "timestamp_seconds",
            }
        )
    )
    view_marker = (
        modality in {"multi_view", "multiview"}
        or source_kind == "view"
        or source_kind.endswith("_view")
        or source_kind.startswith("view_")
    )
    return frame_marker, view_marker


@dataclass(frozen=True)
class PublicAssetModalitySummary:
    """The only public descriptor facts used to build a requirement."""

    modality: VisualModality
    asset_count: int
    frame_count: int | None
    duration: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality.value,
            "asset_count": self.asset_count,
            "frame_count": self.frame_count,
            "duration": self.duration,
        }


def summarize_public_asset_descriptors(
    asset_descriptors: Sequence[Any],
) -> PublicAssetModalitySummary:
    """Infer modality without consulting benchmark or sample names."""

    descriptors = _sequence(asset_descriptors, "asset_descriptors")
    if not descriptors:
        raise VisualSufficiencyContractError("at least one public asset is required")
    frame_markers: list[bool] = []
    view_markers: list[bool] = []
    frame_counts: list[int] = []
    durations: list[float] = []
    asset_ids: list[str] = []
    for descriptor in descriptors:
        if _descriptor_value(descriptor, "public") is not True:
            raise VisualSufficiencyContractError(
                "modality inference accepts only public asset descriptors"
            )
        asset_id = _identifier(
            _descriptor_value(descriptor, "asset_id"), "asset.asset_id"
        )
        asset_ids.append(asset_id)
        frame_marker, view_marker = _descriptor_marker(descriptor)
        frame_markers.append(frame_marker)
        view_markers.append(view_marker)
        frame_count = _descriptor_value(descriptor, "frame_count", None)
        if frame_count is not None:
            frame_counts.append(_positive_int(frame_count, "asset.frame_count"))
        duration = _descriptor_value(descriptor, "duration", None)
        if duration is not None:
            durations.append(_finite_float(duration, "asset.duration"))
    if len(set(asset_ids)) != len(asset_ids):
        raise VisualSufficiencyContractError("public asset IDs must be unique")
    if any(frame_markers) and any(view_markers):
        raise VisualSufficiencyContractError(
            "public descriptors mix video-frame and multi-view semantics"
        )
    if any(frame_markers):
        if not all(frame_markers):
            raise VisualSufficiencyContractError(
                "public descriptors mix video and non-video assets"
            )
        modality = VisualModality.VIDEO
        raw_frame_counts = [count for count in frame_counts if count > 1]
        frame_count = max(raw_frame_counts) if raw_frame_counts else len(descriptors)
    elif any(view_markers):
        if not all(view_markers):
            raise VisualSufficiencyContractError(
                "public descriptors mix view and non-view assets"
            )
        modality = VisualModality.MULTI_VIEW
        frame_count = len(descriptors)
    else:
        modality = (
            VisualModality.MULTI_VIEW
            if len(descriptors) > 1
            else VisualModality.STATIC
        )
        frame_count = (
            len(descriptors)
            if modality is VisualModality.MULTI_VIEW
            else max(frame_counts) if frame_counts else None
        )
    duration = max(durations) if durations else None
    return PublicAssetModalitySummary(
        modality=modality,
        asset_count=len(descriptors),
        frame_count=frame_count,
        duration=duration,
    )


def infer_modality_from_public_assets(
    asset_descriptors: Sequence[Any],
) -> VisualModality:
    return summarize_public_asset_descriptors(asset_descriptors).modality


infer_visual_modality = infer_modality_from_public_assets


def _normalize_public_requirement(value: Any) -> Any:
    if value is None:
        return "modality_sufficient"
    if isinstance(value, bool):
        return "ordered_modality_sufficient" if value else "modality_sufficient"
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized or len(normalized) > 512:
            raise VisualSufficiencyContractError(
                "public_requirement must be concise public text"
            )
        return normalized
    safe = _safe_json(value, "public_requirement")
    if not isinstance(safe, dict):
        raise VisualSufficiencyContractError(
            "public_requirement must be text, boolean, or an object"
        )
    return MappingProxyType(safe)


def _explicit_order_signal(value: Any) -> bool | None:
    if isinstance(value, Mapping):
        for key in ("ordered", "temporal_order_required", "sequence_order_required"):
            if key in value:
                raw = value[key]
                if not isinstance(raw, bool):
                    raise VisualSufficiencyContractError(
                        "public ordered requirement must be boolean"
                    )
                return raw
        return None
    normalized = _normalized_key(value)
    if normalized in {"unordered", "order_irrelevant", "modality_sufficient"}:
        return False
    if any(
        token in normalized
        for token in ("ordered", "temporal_order", "sequence_order", "chronolog")
    ):
        return True
    return None


def _order_required(public_requirement: Any, category: str | None) -> bool:
    explicit = _explicit_order_signal(public_requirement)
    if explicit is not None:
        return explicit
    normalized_category = _normalized_key(category or "")
    return any(
        token in normalized_category
        for token in (
            "before_after",
            "chronolog",
            "motion",
            "movement",
            "sequence_order",
            "temporal",
            "temporal_order",
            "trajectory",
        )
    )


@dataclass(frozen=True)
class EvidenceSufficiencyRequirement:
    """A requirement derived exclusively from public modality facts."""

    modality: VisualModality | str
    asset_count: int
    frame_count: int | None
    duration: float | None
    public_requirement: Any = "modality_sufficient"
    category: str | None = None
    requirement_id: str = field(init=False)
    required_asset_count: int = field(init=False)
    required_frame_count: int = field(init=False)
    minimum_temporal_span_ratio: float = field(init=False)
    ordered_required: bool = field(init=False)
    allowed_actions: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        modality = _coerce_modality(self.modality)
        asset_count = _positive_int(self.asset_count, "asset_count")
        frame_count = self.frame_count
        if frame_count is not None:
            frame_count = _positive_int(frame_count, "frame_count")
        duration = self.duration
        if duration is not None:
            duration = _finite_float(duration, "duration")
        public_requirement = _normalize_public_requirement(self.public_requirement)
        category = self.category
        if category is not None:
            if (
                not isinstance(category, str)
                or not category.strip()
                or len(category) > 512
            ):
                raise VisualSufficiencyContractError(
                    "category must be concise public text or null"
                )
            category = category.strip()

        if modality is VisualModality.STATIC:
            required_assets = 1
            required_frames = 0
            span = 0.0
            allowed = ("OPEN_ASSET", "CROP_REGION", "ZOOM_REGION")
        elif modality is VisualModality.MULTI_VIEW:
            if asset_count > 4:
                raise VisualSufficiencyContractError(
                    "multi-view sufficiency supports all public views only up to four"
                )
            required_assets = asset_count
            required_frames = 0
            span = 0.0
            allowed = ("GET_VIEW", "COMPOSE_ASSETS", "CROP_REGION")
        else:
            required_assets = 0
            required_frames = 3
            span = 0.60
            allowed = ("GET_FRAME", "GET_FRAME_WINDOW", "CROP_REGION")

        ordered_required = _order_required(public_requirement, category)
        public_identity = (
            dict(public_requirement)
            if isinstance(public_requirement, Mapping)
            else public_requirement
        )
        identity = {
            "modality": modality.value,
            "asset_count": asset_count,
            "frame_count": frame_count,
            "duration": duration,
            "public_requirement": public_identity,
            "category": category,
            "required_asset_count": required_assets,
            "required_frame_count": required_frames,
            "minimum_temporal_span_ratio": span,
            "ordered_required": ordered_required,
        }
        requirement_id = f"visual-sufficiency-{canonical_sha256(identity)[:24]}"
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "asset_count", asset_count)
        object.__setattr__(self, "frame_count", frame_count)
        object.__setattr__(self, "duration", duration)
        object.__setattr__(self, "public_requirement", public_requirement)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "requirement_id", requirement_id)
        object.__setattr__(self, "required_asset_count", required_assets)
        object.__setattr__(self, "required_frame_count", required_frames)
        object.__setattr__(self, "minimum_temporal_span_ratio", span)
        object.__setattr__(self, "ordered_required", ordered_required)
        object.__setattr__(self, "allowed_actions", allowed)

    @classmethod
    def from_public_asset_descriptors(
        cls,
        asset_descriptors: Sequence[Any],
        *,
        public_requirement: Any = "modality_sufficient",
        category: str | None = None,
    ) -> "EvidenceSufficiencyRequirement":
        summary = summarize_public_asset_descriptors(asset_descriptors)
        return cls(
            modality=summary.modality,
            asset_count=summary.asset_count,
            frame_count=summary.frame_count,
            duration=summary.duration,
            public_requirement=public_requirement,
            category=category,
        )

    def to_dict(self) -> dict[str, Any]:
        public_requirement = (
            dict(self.public_requirement)
            if isinstance(self.public_requirement, Mapping)
            else self.public_requirement
        )
        return {
            "requirement_id": self.requirement_id,
            "modality": self.modality.value,
            "asset_count": self.asset_count,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "public_requirement": copy.deepcopy(public_requirement),
            "category": self.category,
            "required_asset_count": self.required_asset_count,
            "required_frame_count": self.required_frame_count,
            "minimum_temporal_span_ratio": self.minimum_temporal_span_ratio,
            "ordered_required": self.ordered_required,
            "allowed_actions": list(self.allowed_actions),
        }

    @property
    def canonical_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("requirement_id")
        return canonical_sha256(payload)

    def evaluate(
        self,
        bundles: Sequence[ModalityEvidenceBundle | Mapping[str, Any]],
    ) -> "EvidenceSufficiencyStatus":
        normalized = tuple(
            bundle
            if isinstance(bundle, ModalityEvidenceBundle)
            else ModalityEvidenceBundle.from_media_summary(bundle)
            for bundle in bundles
        )
        return evaluate_evidence_sufficiency(self, normalized)

    def evaluate_observed_evidence(
        self,
        records: Sequence[Any],
        *,
        operation: str,
        ordered: bool,
        temporal_span_ratio: float | None,
        payload_bytes: int,
        sequence_positions: Sequence[int] | None = None,
        private_audit_timestamps: Sequence[str | float] = (),
        private_provenance: Mapping[str, Any] | None = None,
        bundle_policy: BundlePolicy | str = BundlePolicy.MODALITY_EVIDENCE,
        safe_provenance: Mapping[str, Any] | None = None,
    ) -> "EvidenceSufficiencyStatus":
        """Evaluate structurally typed observed evidence in one integration call."""

        bundle = ModalityEvidenceBundle.from_observed_evidence(
            records,
            modality=self.modality,
            operation=operation,
            ordered=ordered,
            temporal_span_ratio=temporal_span_ratio,
            payload_bytes=payload_bytes,
            sequence_positions=sequence_positions,
            private_audit_timestamps=private_audit_timestamps,
            private_provenance=private_provenance,
            bundle_policy=bundle_policy,
            safe_provenance=safe_provenance,
        )
        return evaluate_evidence_sufficiency(self, (bundle,))


def requirement_from_public_assets(
    asset_descriptors: Sequence[Any],
    *,
    public_requirement: Any = "modality_sufficient",
    category: str | None = None,
) -> EvidenceSufficiencyRequirement:
    return EvidenceSufficiencyRequirement.from_public_asset_descriptors(
        asset_descriptors,
        public_requirement=public_requirement,
        category=category,
    )


@dataclass(frozen=True)
class EvidenceSufficiencyFeedback:
    """Reference-free correction feedback with an exact public field set."""

    requirement_id: str
    modality: str
    required_evidence: Mapping[str, Any]
    current_evidence: Mapping[str, Any]
    missing_evidence: tuple[str, ...]
    allowed_actions: tuple[str, ...]
    empty_reference_disclosure: bool = False
    retryable_with_new_model_turn: bool = True

    def __post_init__(self) -> None:
        _identifier(self.requirement_id, "requirement_id")
        modality = _coerce_modality(self.modality).value
        object.__setattr__(self, "modality", modality)
        required = _safe_json(self.required_evidence, "required_evidence")
        current = _safe_json(self.current_evidence, "current_evidence")
        if not isinstance(required, dict) or not isinstance(current, dict):
            raise VisualSufficiencyContractError(
                "evidence summaries must be public objects"
            )
        missing = tuple(
            _identifier(item, "missing_evidence")
            for item in _sequence(self.missing_evidence, "missing_evidence")
        )
        actions = tuple(
            _identifier(item, "allowed_actions")
            for item in _sequence(self.allowed_actions, "allowed_actions")
        )
        if "LIST_ASSETS" in actions:
            raise VisualSufficiencyContractError(
                "LIST_ASSETS cannot correct visual evidence insufficiency"
            )
        if self.empty_reference_disclosure is not False:
            raise VisualSufficiencyContractError(
                "feedback must not disclose empty or valid evidence references"
            )
        if self.retryable_with_new_model_turn is not True:
            raise VisualSufficiencyContractError(
                "insufficiency feedback must allow a new model turn"
            )
        object.__setattr__(self, "required_evidence", MappingProxyType(required))
        object.__setattr__(self, "current_evidence", MappingProxyType(current))
        object.__setattr__(self, "missing_evidence", missing)
        object.__setattr__(self, "allowed_actions", actions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "modality": self.modality,
            "required_evidence": copy.deepcopy(dict(self.required_evidence)),
            "current_evidence": copy.deepcopy(dict(self.current_evidence)),
            "missing_evidence": list(self.missing_evidence),
            "allowed_actions": list(self.allowed_actions),
            "empty_reference_disclosure": self.empty_reference_disclosure,
            "retryable_with_new_model_turn": self.retryable_with_new_model_turn,
        }


@dataclass(frozen=True)
class EvidenceSufficiencyStatus:
    """Count-only evaluation status; source identities never enter this object."""

    requirement_id: str
    modality: VisualModality
    satisfied: bool
    modality_sufficient: bool
    ordered_sufficient: bool
    distinct_asset_count: int
    distinct_frame_count: int
    presented_count: int
    temporal_span_ratio: float | None
    relevant_bundle_count: int
    ignored_list_assets_count: int
    required_evidence: Mapping[str, Any]
    current_evidence: Mapping[str, Any]
    missing_evidence: tuple[str, ...]
    feedback: EvidenceSufficiencyFeedback | None

    def __post_init__(self) -> None:
        required = _safe_json(self.required_evidence, "required_evidence")
        current = _safe_json(self.current_evidence, "current_evidence")
        if not isinstance(required, dict) or not isinstance(current, dict):
            raise VisualSufficiencyContractError(
                "status evidence summaries must be public objects"
            )
        object.__setattr__(self, "required_evidence", MappingProxyType(required))
        object.__setattr__(self, "current_evidence", MappingProxyType(current))

    @property
    def status(self) -> str:
        return "sufficient" if self.satisfied else "insufficient"

    @property
    def content_sufficient(self) -> bool:
        return self.modality_sufficient

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "modality": self.modality.value,
            "status": self.status,
            "satisfied": self.satisfied,
            "content_sufficient": self.content_sufficient,
            "modality_sufficient": self.modality_sufficient,
            "ordered_sufficient": self.ordered_sufficient,
            "distinct_asset_count": self.distinct_asset_count,
            "distinct_frame_count": self.distinct_frame_count,
            "presented_count": self.presented_count,
            "temporal_span_ratio": self.temporal_span_ratio,
            "relevant_bundle_count": self.relevant_bundle_count,
            "ignored_list_assets_count": self.ignored_list_assets_count,
            "required_evidence": copy.deepcopy(dict(self.required_evidence)),
            "current_evidence": copy.deepcopy(dict(self.current_evidence)),
            "missing_evidence": list(self.missing_evidence),
            "feedback": None if self.feedback is None else self.feedback.to_dict(),
        }


EvidenceSufficiencyEvaluation = EvidenceSufficiencyStatus


def _temporal_span(
    requirement: EvidenceSufficiencyRequirement,
    bundles: Sequence[ModalityEvidenceBundle],
    frame_indices: Sequence[int],
) -> float:
    numeric_timestamps = [
        float(value)
        for bundle in bundles
        for value in bundle.private_audit_timestamps
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if (
        requirement.duration is not None
        and requirement.duration > 0.0
        and len(numeric_timestamps) >= 2
    ):
        return min(
            1.0,
            (max(numeric_timestamps) - min(numeric_timestamps))
            / requirement.duration,
        )
    if (
        requirement.frame_count is not None
        and requirement.frame_count > 1
        and len(frame_indices) >= 2
        and max(frame_indices) < requirement.frame_count
    ):
        return (max(frame_indices) - min(frame_indices)) / (
            requirement.frame_count - 1
        )
    declared = [
        bundle.temporal_span_ratio
        for bundle in bundles
        if bundle.temporal_span_ratio is not None
    ]
    return max(declared, default=0.0)


def _required_evidence_summary(
    requirement: EvidenceSufficiencyRequirement,
) -> dict[str, Any]:
    if requirement.modality is VisualModality.STATIC:
        return {"distinct_asset_count": 1}
    if requirement.modality is VisualModality.MULTI_VIEW:
        return {"distinct_asset_count": requirement.required_asset_count}
    return {
        "distinct_frame_count": requirement.required_frame_count,
        "minimum_temporal_span_ratio": requirement.minimum_temporal_span_ratio,
        "ordered_required": requirement.ordered_required,
    }


def _current_evidence_summary(
    requirement: EvidenceSufficiencyRequirement,
    *,
    distinct_asset_count: int,
    distinct_frame_count: int,
    temporal_span_ratio: float | None,
    ordered_sufficient: bool,
) -> dict[str, Any]:
    if requirement.modality is VisualModality.STATIC:
        return {"distinct_asset_count": distinct_asset_count}
    if requirement.modality is VisualModality.MULTI_VIEW:
        return {"distinct_asset_count": distinct_asset_count}
    return {
        "distinct_frame_count": distinct_frame_count,
        "temporal_span_ratio": temporal_span_ratio,
        "ordered": ordered_sufficient,
    }


def evaluate_evidence_sufficiency(
    requirement: EvidenceSufficiencyRequirement,
    bundles: Sequence[ModalityEvidenceBundle],
) -> EvidenceSufficiencyStatus:
    """Evaluate evidence counts, temporal coverage, and order independently."""

    if not isinstance(requirement, EvidenceSufficiencyRequirement):
        raise VisualSufficiencyContractError(
            "requirement must be EvidenceSufficiencyRequirement"
        )
    raw_bundles = _sequence(bundles, "bundles")
    if not all(isinstance(bundle, ModalityEvidenceBundle) for bundle in raw_bundles):
        raise VisualSufficiencyContractError(
            "bundles must contain ModalityEvidenceBundle values"
        )
    ignored = sum(bundle.operation == "LIST_ASSETS" for bundle in raw_bundles)
    relevant = tuple(
        bundle for bundle in raw_bundles if bundle.operation != "LIST_ASSETS"
    )
    if any(bundle.modality is not requirement.modality for bundle in relevant):
        raise VisualSufficiencyContractError(
            "answer-bearing evidence modality does not match the requirement"
        )

    asset_ids = [item for bundle in relevant for item in bundle.source_asset_ids]
    media_hashes = [
        item for bundle in relevant for item in bundle.source_media_hashes
    ]
    distinct_asset_count = min(len(set(asset_ids)), len(set(media_hashes)))
    frame_indices = [item for bundle in relevant for item in bundle.frame_indices]
    distinct_frame_count = (
        min(len(set(frame_indices)), len(set(media_hashes)))
        if frame_indices
        else len(set(media_hashes))
        if requirement.modality is VisualModality.VIDEO
        else 0
    )
    presented_count = sum(bundle.presented_count for bundle in relevant)

    if requirement.modality is VisualModality.VIDEO:
        span = _temporal_span(requirement, relevant, frame_indices)
        if len(frame_indices) <= 1:
            ordered_sufficient = True
        else:
            ordered_sufficient = all(bundle.ordered for bundle in relevant) and (
                _strictly_increasing(frame_indices)
            )
        modality_sufficient = (
            distinct_frame_count >= requirement.required_frame_count
            and span >= requirement.minimum_temporal_span_ratio
        )
    elif requirement.modality is VisualModality.MULTI_VIEW:
        span = None
        ordered_sufficient = True
        modality_sufficient = (
            distinct_asset_count >= requirement.required_asset_count
        )
    else:
        span = None
        ordered_sufficient = True
        modality_sufficient = distinct_asset_count >= 1

    missing: list[str] = []
    if requirement.modality is VisualModality.STATIC and not modality_sufficient:
        missing.append("visual_asset")
    elif requirement.modality is VisualModality.MULTI_VIEW and not modality_sufficient:
        missing.append("distinct_views")
    elif requirement.modality is VisualModality.VIDEO:
        if distinct_frame_count < requirement.required_frame_count:
            missing.append("distinct_video_frames")
        if span < requirement.minimum_temporal_span_ratio:
            missing.append("temporal_span")
    if requirement.ordered_required and not ordered_sufficient:
        missing.append("ordered_sequence")
    satisfied = modality_sufficient and (
        ordered_sufficient or not requirement.ordered_required
    )

    required_summary = _required_evidence_summary(requirement)
    current_summary = _current_evidence_summary(
        requirement,
        distinct_asset_count=distinct_asset_count,
        distinct_frame_count=distinct_frame_count,
        temporal_span_ratio=span,
        ordered_sufficient=ordered_sufficient,
    )
    feedback = None
    if not satisfied:
        feedback = EvidenceSufficiencyFeedback(
            requirement_id=requirement.requirement_id,
            modality=requirement.modality.value,
            required_evidence=required_summary,
            current_evidence=current_summary,
            missing_evidence=tuple(missing),
            allowed_actions=requirement.allowed_actions,
        )
    return EvidenceSufficiencyStatus(
        requirement_id=requirement.requirement_id,
        modality=requirement.modality,
        satisfied=satisfied,
        modality_sufficient=modality_sufficient,
        ordered_sufficient=ordered_sufficient,
        distinct_asset_count=distinct_asset_count,
        distinct_frame_count=distinct_frame_count,
        presented_count=presented_count,
        temporal_span_ratio=span,
        relevant_bundle_count=len(relevant),
        ignored_list_assets_count=ignored,
        required_evidence=required_summary,
        current_evidence=current_summary,
        missing_evidence=tuple(missing),
        feedback=feedback,
    )


evaluate_sufficiency = evaluate_evidence_sufficiency


__all__ = [
    "BundlePolicy",
    "E1_MINIMAL_VISUAL_DIRECT",
    "E2_MODALITY_SUFFICIENT_DIRECT",
    "E3_ACTIVE_MODALITY_SUFFICIENT",
    "EvidenceSufficiencyEvaluation",
    "EvidenceSufficiencyFeedback",
    "EvidenceSufficiencyRequirement",
    "EvidenceSufficiencyStatus",
    "ModalityEvidenceBundle",
    "PublicAssetModalitySummary",
    "V1_SINGLE_CENTER_FRAME",
    "V3_DUPLICATED_CENTER_FRAME",
    "V3_UNIFORM_ORDERED",
    "V3_UNIFORM_SHUFFLED",
    "V7_UNIFORM_ORDERED",
    "VISUAL_FIRST",
    "VISUAL_FIRST_ARM_DEFINITIONS",
    "VISUAL_FIRST_ARMS",
    "VISUAL_FIRST_SEMANTIC",
    "VSI_ARM_DEFINITIONS",
    "VSI_FRAME_ARM_DEFINITIONS",
    "VSI_FRAME_ARMS",
    "VSIFrameArmDefinition",
    "VisualFirstArmDefinition",
    "VisualModality",
    "VisualSufficiencyContractError",
    "canonical_sha256",
    "conditional_v7_definition",
    "evaluate_evidence_sufficiency",
    "evaluate_sufficiency",
    "infer_modality_from_public_assets",
    "infer_visual_modality",
    "requirement_from_public_assets",
    "summarize_public_asset_descriptors",
]
