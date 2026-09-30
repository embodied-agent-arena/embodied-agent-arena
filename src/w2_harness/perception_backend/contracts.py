"""Portable contracts for optional, local-only perception estimates.

This module deliberately depends only on the Python standard library.  In
particular, importing it must not initialize any model framework.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


PERCEPTION_SCHEMA_VERSION = "w2-portable-perception-v1.0"
PILOT_SCHEMA_VERSION = "w2-portable-perception-pilot-v1.0"
ORIGINAL_XYXY_COORDINATE_SYSTEM = "pixel_xyxy_original"
DEPTH_COORDINATE_FRAME = "camera_image"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+-]{0,191}$")
_DEVICE = re.compile(r"^(?:cpu|mps|cuda(?::[0-9]+)?)$")
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


class PerceptionError(RuntimeError):
    """Base class carrying a stable, non-sensitive failure code."""

    reason_code = "perception_error"


class PerceptionInputError(PerceptionError, ValueError):
    reason_code = "invalid_input"


class PerceptionContractError(PerceptionError, ValueError):
    reason_code = "contract_error"


class BackendPreparationError(PerceptionError):
    reason_code = "backend_prepare_failed"


class BackendNotPreparedError(PerceptionError):
    reason_code = "backend_not_prepared"


class BackendInferenceError(PerceptionError):
    reason_code = "backend_inference_failed"


class BackendOutputError(PerceptionError):
    reason_code = "invalid_backend_output"


class PerceptionStatus(str, Enum):
    SUCCESS = "success"
    NO_DETECTION = "no_detection"
    AMBIGUOUS = "ambiguous"
    FAILED = "failed"


def _normalized_key(value: Any) -> str:
    return str(value).casefold().replace("-", "_").replace(" ", "_")


def _contains_sensitive_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = _normalized_key(key)
            if any(part in normalized for part in _SENSITIVE_KEY_PARTS):
                return True
            if _contains_sensitive_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_sensitive_key(item) for item in value)
    return False


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite, credential-free JSON in a deterministic form."""

    if _contains_sensitive_key(value):
        raise PerceptionContractError("output contains a credential-bearing field")
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PerceptionContractError(
            "output must contain only finite JSON values"
        ) from exc


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def validate_sha256(value: Any, field_name: str = "sha256") -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise PerceptionInputError(f"{field_name} must be a lowercase SHA-256")
    return value


def validate_device(value: Any) -> str:
    if not isinstance(value, str) or _DEVICE.fullmatch(value.strip()) is None:
        raise PerceptionInputError("device must be cpu, mps, cuda, or cuda:<index>")
    return value.strip()


def _identifier(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value.strip()) is None:
        raise PerceptionContractError(
            f"{field_name} must be a portable non-empty identifier"
        )
    return value.strip()


def _text(value: Any, field_name: str, *, maximum: int = 1024) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerceptionContractError(f"{field_name} must be non-empty text")
    normalized = " ".join(value.split())
    if len(normalized) > maximum:
        raise PerceptionContractError(
            f"{field_name} must be at most {maximum} characters"
        )
    return normalized


def normalize_prompts(prompts: Sequence[str] | None) -> tuple[str, ...]:
    if prompts is None:
        return ()
    if isinstance(prompts, (str, bytes, bytearray)) or not isinstance(
        prompts, Sequence
    ):
        raise PerceptionInputError("prompts must be a sequence of strings")
    normalized = tuple(_text(item, "prompt", maximum=256) for item in prompts)
    if not normalized:
        raise PerceptionInputError("prompts must not be empty when supplied")
    folded = [item.casefold() for item in normalized]
    if len(set(folded)) != len(folded):
        raise PerceptionInputError("prompts must be unique ignoring case")
    return normalized


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError, OverflowError):
            raise PerceptionContractError(f"{field_name} must be numeric") from None
    result = float(value)
    if not math.isfinite(result):
        raise PerceptionContractError(f"{field_name} must be finite")
    return result


@dataclass(frozen=True)
class ImageDimensions:
    width: int
    height: int

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise PerceptionContractError(
                    f"original_dimensions.{name} must be a positive integer"
                )

    def to_dict(self) -> dict[str, int]:
        return {"width": self.width, "height": self.height}


@dataclass(frozen=True)
class SourceBinding:
    source_index: int
    source_media_hash: str
    original_dimensions: ImageDimensions
    view_id: str | None = None
    frame_index: int | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise PerceptionContractError("source_index must be a non-negative integer")
        validate_sha256(self.source_media_hash, "source_media_hash")
        if not isinstance(self.original_dimensions, ImageDimensions):
            raise PerceptionContractError(
                "original_dimensions must use ImageDimensions"
            )
        if self.view_id is not None:
            object.__setattr__(self, "view_id", _identifier(self.view_id, "view_id"))
        if self.frame_index is not None and (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise PerceptionContractError(
                "frame_index must be a non-negative integer or null"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "source_media_hash": self.source_media_hash,
            "original_dimensions": self.original_dimensions.to_dict(),
            "view_id": self.view_id,
            "frame_index": self.frame_index,
        }


@dataclass(frozen=True)
class LocalImageSource:
    """One verified local image; filesystem location is never serialized."""

    path: Path
    binding: SourceBinding

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser()
        if not path.is_file():
            raise PerceptionInputError("image must be an existing local file")
        object.__setattr__(self, "path", path.resolve())
        if not isinstance(self.binding, SourceBinding):
            raise PerceptionContractError("binding must use SourceBinding")

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        source_index: int = 0,
        expected_media_hash: str | None = None,
        view_id: str | None = None,
        frame_index: int | None = None,
        max_bytes: int = 100 * 1024 * 1024,
    ) -> "LocalImageSource":
        local_path = Path(path).expanduser()
        if not local_path.is_file():
            raise PerceptionInputError("image must be an existing local file")
        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise PerceptionInputError("max_bytes must be a positive integer")
        if local_path.stat().st_size > max_bytes:
            raise PerceptionInputError("image exceeds the configured byte limit")
        payload = local_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if expected_media_hash is not None:
            expected = validate_sha256(expected_media_hash, "expected_media_hash")
            if digest != expected:
                raise PerceptionInputError(
                    "image bytes do not match expected_media_hash"
                )
        try:
            from PIL import Image

            with Image.open(io.BytesIO(payload)) as image:
                width, height = image.size
                image.verify()
        except Exception as exc:
            raise PerceptionInputError("image is not a decodable local image") from exc
        return cls(
            path=local_path,
            binding=SourceBinding(
                source_index=source_index,
                source_media_hash=digest,
                original_dimensions=ImageDimensions(width=width, height=height),
                view_id=view_id,
                frame_index=frame_index,
            ),
        )

    def open_rgb(self) -> Any:
        """Load a detached PIL RGB image only when inference is requested."""

        try:
            from PIL import Image

            payload = self.path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != self.binding.source_media_hash:
                raise PerceptionInputError("image changed after source binding")
            with Image.open(io.BytesIO(payload)) as image:
                return image.convert("RGB").copy()
        except PerceptionInputError:
            raise
        except Exception as exc:
            raise PerceptionInputError(
                "image could not be decoded for inference"
            ) from exc

    def verify_unchanged(self) -> None:
        current = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if current != self.binding.source_media_hash:
            raise PerceptionInputError("image changed after source binding")

    def to_dict(self) -> dict[str, Any]:
        return self.binding.to_dict()


# A concise public alias for callers and tests.
ImageSource = LocalImageSource


@dataclass(frozen=True)
class BoundingBoxXYXY:
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    original_dimensions: ImageDimensions
    coordinate_system: str = ORIGINAL_XYXY_COORDINATE_SYSTEM

    def __post_init__(self) -> None:
        if self.coordinate_system != ORIGINAL_XYXY_COORDINATE_SYSTEM:
            raise PerceptionContractError(
                "bbox coordinate_system must identify original-image xyxy pixels"
            )
        if not isinstance(self.original_dimensions, ImageDimensions):
            raise PerceptionContractError(
                "bbox original_dimensions must use ImageDimensions"
            )
        values = tuple(
            _finite_number(getattr(self, name), f"bbox.{name}")
            for name in ("x_min", "y_min", "x_max", "y_max")
        )
        x_min, y_min, x_max, y_max = values
        if x_min < 0 or y_min < 0:
            raise PerceptionContractError("bbox coordinates must not be negative")
        if x_min >= x_max or y_min >= y_max:
            raise PerceptionContractError("bbox must have positive area")
        if x_max > self.original_dimensions.width:
            raise PerceptionContractError("bbox exceeds original image width")
        if y_max > self.original_dimensions.height:
            raise PerceptionContractError("bbox exceeds original image height")
        for name, value in zip(
            ("x_min", "y_min", "x_max", "y_max"), values
        ):
            object.__setattr__(self, name, value)

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x_min, self.y_min, self.x_max, self.y_max

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "xyxy",
            "coordinate_system": self.coordinate_system,
            "values": list(self.xyxy),
            "original_dimensions": self.original_dimensions.to_dict(),
        }


@dataclass(frozen=True)
class LetterboxTransform:
    """Exact inverse mapping from padded inference pixels to source pixels."""

    original_dimensions: ImageDimensions
    inference_dimensions: ImageDimensions
    scale: float
    pad_left: float
    pad_top: float

    def __post_init__(self) -> None:
        if not isinstance(self.original_dimensions, ImageDimensions) or not isinstance(
            self.inference_dimensions, ImageDimensions
        ):
            raise PerceptionContractError(
                "letterbox dimensions must use ImageDimensions"
            )
        scale = _finite_number(self.scale, "letterbox.scale")
        pad_left = _finite_number(self.pad_left, "letterbox.pad_left")
        pad_top = _finite_number(self.pad_top, "letterbox.pad_top")
        if scale <= 0 or pad_left < 0 or pad_top < 0:
            raise PerceptionContractError("letterbox scale/padding is invalid")
        scaled_width = self.original_dimensions.width * scale
        scaled_height = self.original_dimensions.height * scale
        if not math.isclose(
            scaled_width + 2 * pad_left,
            self.inference_dimensions.width,
            abs_tol=1.0,
        ):
            raise PerceptionContractError("letterbox width geometry is inconsistent")
        if not math.isclose(
            scaled_height + 2 * pad_top,
            self.inference_dimensions.height,
            abs_tol=1.0,
        ):
            raise PerceptionContractError("letterbox height geometry is inconsistent")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "pad_left", pad_left)
        object.__setattr__(self, "pad_top", pad_top)

    @classmethod
    def centered(
        cls,
        *,
        original_dimensions: ImageDimensions,
        inference_dimensions: ImageDimensions,
    ) -> "LetterboxTransform":
        scale = min(
            inference_dimensions.width / original_dimensions.width,
            inference_dimensions.height / original_dimensions.height,
        )
        return cls(
            original_dimensions=original_dimensions,
            inference_dimensions=inference_dimensions,
            scale=scale,
            pad_left=(inference_dimensions.width - original_dimensions.width * scale)
            / 2.0,
            pad_top=(inference_dimensions.height - original_dimensions.height * scale)
            / 2.0,
        )

    def to_original(self, xyxy: Sequence[Any]) -> BoundingBoxXYXY:
        if isinstance(xyxy, (str, bytes, bytearray)) or len(xyxy) != 4:
            raise PerceptionContractError("letterbox bbox must contain four values")
        values = tuple(
            _finite_number(value, "letterbox bbox coordinate") for value in xyxy
        )
        x_min, y_min, x_max, y_max = values
        if x_min < 0 or y_min < 0:
            raise PerceptionContractError(
                "letterbox bbox coordinates must not be negative"
            )
        if x_max > self.inference_dimensions.width:
            raise PerceptionContractError("letterbox bbox exceeds inference width")
        if y_max > self.inference_dimensions.height:
            raise PerceptionContractError("letterbox bbox exceeds inference height")
        return BoundingBoxXYXY(
            x_min=(x_min - self.pad_left) / self.scale,
            y_min=(y_min - self.pad_top) / self.scale,
            x_max=(x_max - self.pad_left) / self.scale,
            y_max=(y_max - self.pad_top) / self.scale,
            original_dimensions=self.original_dimensions,
        )


def letterbox_xyxy_to_original(
    xyxy: Sequence[Any], transform: LetterboxTransform
) -> BoundingBoxXYXY:
    if not isinstance(transform, LetterboxTransform):
        raise PerceptionContractError("transform must use LetterboxTransform")
    return transform.to_original(xyxy)


@dataclass(frozen=True)
class ArtifactReference:
    kind: str
    representation: str
    sha256: str
    shape: tuple[int, ...]
    dtype: str
    reference: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _identifier(self.kind, "artifact.kind"))
        object.__setattr__(
            self,
            "representation",
            _identifier(self.representation, "artifact.representation"),
        )
        validate_sha256(self.sha256, "artifact.sha256")
        if not isinstance(self.shape, tuple) or not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in self.shape
        ):
            raise PerceptionContractError(
                "artifact.shape must contain positive integer dimensions"
            )
        object.__setattr__(self, "dtype", _identifier(self.dtype, "artifact.dtype"))
        if self.reference is not None:
            if not isinstance(self.reference, str) or not self.reference.strip():
                raise PerceptionContractError(
                    "artifact.reference must be relative text or null"
                )
            reference = self.reference.strip().replace("\\", "/")
            pure = PurePosixPath(reference)
            if pure.is_absolute() or ".." in pure.parts:
                raise PerceptionContractError(
                    "artifact.reference must be a safe relative path"
                )
            object.__setattr__(self, "reference", reference)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "representation": self.representation,
            "sha256": self.sha256,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "reference": self.reference,
        }


@dataclass(frozen=True)
class BackendIdentity:
    backend_id: str
    implementation: str
    version: str
    model_sha256: str
    local_files_only: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "backend_id", _identifier(self.backend_id, "backend_id")
        )
        object.__setattr__(
            self,
            "implementation",
            _identifier(self.implementation, "backend.implementation"),
        )
        object.__setattr__(self, "version", _text(self.version, "backend.version"))
        validate_sha256(self.model_sha256, "backend.model_sha256")
        if self.local_files_only is not True:
            raise PerceptionContractError(
                "portable perception backends must be local-files-only"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_id": self.backend_id,
            "implementation": self.implementation,
            "version": self.version,
            "model_sha256": self.model_sha256,
            "local_files_only": True,
        }


@dataclass(frozen=True)
class FailureInfo:
    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _identifier(self.code, "failure.code"))
        object.__setattr__(
            self, "message", _text(self.message, "failure.message", maximum=512)
        )
        if not isinstance(self.retryable, bool):
            raise PerceptionContractError("failure.retryable must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
        }


def _confidence(value: Any) -> float | None:
    if value is None:
        return None
    result = _finite_number(value, "confidence")
    if not 0.0 <= result <= 1.0:
        raise PerceptionContractError("confidence must be in [0, 1]")
    return result


@dataclass(frozen=True)
class DetectionEstimate:
    source_index: int
    source_media_hash: str
    class_name: str
    prompt: str | None
    bbox: BoundingBoxXYXY
    confidence: float | None = None
    mask: ArtifactReference | None = None
    ambiguity_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise PerceptionContractError("detection source_index is invalid")
        validate_sha256(self.source_media_hash, "detection.source_media_hash")
        object.__setattr__(
            self, "class_name", _text(self.class_name, "detection.class", maximum=256)
        )
        if self.prompt is not None:
            object.__setattr__(
                self, "prompt", _text(self.prompt, "detection.prompt", maximum=256)
            )
        if not isinstance(self.bbox, BoundingBoxXYXY):
            raise PerceptionContractError("detection.bbox must use BoundingBoxXYXY")
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        if self.mask is not None:
            if not isinstance(self.mask, ArtifactReference) or self.mask.kind != "mask":
                raise PerceptionContractError(
                    "detection.mask must be a mask ArtifactReference"
                )
            if self.mask.shape[-2:] != (
                self.bbox.original_dimensions.height,
                self.bbox.original_dimensions.width,
            ):
                raise PerceptionContractError(
                    "detection mask must use original image dimensions"
                )
        if self.ambiguity_reason is not None:
            object.__setattr__(
                self,
                "ambiguity_reason",
                _identifier(self.ambiguity_reason, "ambiguity_reason"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_index": self.source_index,
            "source_media_hash": self.source_media_hash,
            "class": self.class_name,
            "prompt": self.prompt,
            "bbox": self.bbox.to_dict(),
            "mask": None if self.mask is None else self.mask.to_dict(),
            "confidence": self.confidence,
            "ambiguity_reason": self.ambiguity_reason,
        }


def _safe_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise PerceptionContractError("provenance must be an object")
    copied = json.loads(canonical_json_bytes(dict(value)).decode("utf-8"))
    if copied.get("cross_view_identity_inferred") is not False:
        raise PerceptionContractError(
            "provenance must state cross_view_identity_inferred=false"
        )
    if copied.get("semantic_completion_performed") is not False:
        raise PerceptionContractError(
            "provenance must state semantic_completion_performed=false"
        )
    return copied


@dataclass(frozen=True)
class DetectionSegmentationResult:
    status: PerceptionStatus | str
    sources: tuple[SourceBinding, ...]
    prompts: tuple[str, ...]
    detections: tuple[DetectionEstimate, ...]
    backend: BackendIdentity
    provenance: dict[str, Any]
    ambiguity_reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    failure: FailureInfo | None = None
    schema_version: str = field(default=PERCEPTION_SCHEMA_VERSION, init=False)
    result_kind: str = field(default="detection_segmentation", init=False)

    def __post_init__(self) -> None:
        try:
            status = PerceptionStatus(self.status)
        except (TypeError, ValueError):
            raise PerceptionContractError("unsupported perception status") from None
        object.__setattr__(self, "status", status)
        if not isinstance(self.sources, tuple) or not self.sources or not all(
            isinstance(item, SourceBinding) for item in self.sources
        ):
            raise PerceptionContractError("sources must contain SourceBinding values")
        indexes = [item.source_index for item in self.sources]
        if len(set(indexes)) != len(indexes):
            raise PerceptionContractError("source indexes must be unique")
        prompts = normalize_prompts(self.prompts) if self.prompts else ()
        object.__setattr__(self, "prompts", prompts)
        if not isinstance(self.detections, tuple) or not all(
            isinstance(item, DetectionEstimate) for item in self.detections
        ):
            raise PerceptionContractError(
                "detections must contain DetectionEstimate values"
            )
        source_by_index = {item.source_index: item for item in self.sources}
        for detection in self.detections:
            source = source_by_index.get(detection.source_index)
            if source is None:
                raise PerceptionContractError("detection references an unknown source")
            if detection.source_media_hash != source.source_media_hash:
                raise PerceptionContractError("detection source hash mismatch")
            if detection.bbox.original_dimensions != source.original_dimensions:
                raise PerceptionContractError(
                    "detection dimensions do not match its source"
                )
        if not isinstance(self.backend, BackendIdentity):
            raise PerceptionContractError("backend must use BackendIdentity")
        object.__setattr__(self, "provenance", _safe_provenance(self.provenance))
        ambiguity_reasons = tuple(
            _identifier(item, "ambiguity_reasons") for item in self.ambiguity_reasons
        )
        limitations = tuple(
            _identifier(item, "limitations") for item in self.limitations
        )
        object.__setattr__(self, "ambiguity_reasons", ambiguity_reasons)
        object.__setattr__(self, "limitations", limitations)
        if status is PerceptionStatus.SUCCESS and not self.detections:
            raise PerceptionContractError("success requires at least one detection")
        if status is PerceptionStatus.NO_DETECTION and self.detections:
            raise PerceptionContractError("no_detection cannot contain detections")
        if status is PerceptionStatus.AMBIGUOUS and not ambiguity_reasons:
            raise PerceptionContractError("ambiguous requires reason codes")
        if status is PerceptionStatus.FAILED:
            if self.failure is None or self.detections:
                raise PerceptionContractError(
                    "failed requires failure details and no detections"
                )
        elif self.failure is not None:
            raise PerceptionContractError("only failed results may contain failure")

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "result_kind": self.result_kind,
            "status": self.status.value,
            "sources": [item.to_dict() for item in self.sources],
            "prompts": list(self.prompts),
            "detections": [item.to_dict() for item in self.detections],
            "backend": self.backend.to_dict(),
            "provenance": self.provenance,
            "ambiguity_reasons": list(self.ambiguity_reasons),
            "limitations": list(self.limitations),
            "failure": None if self.failure is None else self.failure.to_dict(),
        }
        canonical_json_bytes(value)
        return value


@dataclass(frozen=True)
class DepthEstimate:
    source_index: int
    source_media_hash: str
    original_dimensions: ImageDimensions
    representation: ArtifactReference
    valid_mask: ArtifactReference
    valid_pixel_count: int
    is_relative: bool = True
    is_metric: bool = False
    units: str = "unitless"
    coordinate_frame: str = DEPTH_COORDINATE_FRAME
    limitations: tuple[str, ...] = (
        "monocular_relative_depth",
        "scale_and_shift_ambiguous",
        "not_metric_distance",
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_index, bool)
            or not isinstance(self.source_index, int)
            or self.source_index < 0
        ):
            raise PerceptionContractError("depth source_index is invalid")
        validate_sha256(self.source_media_hash, "depth.source_media_hash")
        if not isinstance(self.original_dimensions, ImageDimensions):
            raise PerceptionContractError(
                "depth original_dimensions must use ImageDimensions"
            )
        expected_shape = (
            self.original_dimensions.height,
            self.original_dimensions.width,
        )
        if (
            not isinstance(self.representation, ArtifactReference)
            or self.representation.kind != "depth"
            or self.representation.shape != expected_shape
        ):
            raise PerceptionContractError(
                "depth representation must match original image dimensions"
            )
        if (
            not isinstance(self.valid_mask, ArtifactReference)
            or self.valid_mask.kind != "valid_mask"
            or self.valid_mask.shape != expected_shape
        ):
            raise PerceptionContractError(
                "depth valid mask must match original image dimensions"
            )
        pixel_count = expected_shape[0] * expected_shape[1]
        if (
            isinstance(self.valid_pixel_count, bool)
            or not isinstance(self.valid_pixel_count, int)
            or not 0 <= self.valid_pixel_count <= pixel_count
        ):
            raise PerceptionContractError("valid_pixel_count is out of range")
        if self.is_relative is not True or self.is_metric is not False:
            raise PerceptionContractError(
                "Depth Anything V2 Small output must remain relative, never metric"
            )
        if self.units != "unitless":
            raise PerceptionContractError(
                "relative Depth Anything V2 Small output must be unitless"
            )
        if self.coordinate_frame != DEPTH_COORDINATE_FRAME:
            raise PerceptionContractError("unsupported depth coordinate frame")
        limitations = tuple(
            _identifier(item, "depth.limitations") for item in self.limitations
        )
        required = {
            "monocular_relative_depth",
            "scale_and_shift_ambiguous",
            "not_metric_distance",
        }
        if not required.issubset(limitations):
            raise PerceptionContractError(
                "depth limitations must preserve relative-depth caveats"
            )
        object.__setattr__(self, "limitations", limitations)

    def to_dict(self) -> dict[str, Any]:
        total = self.original_dimensions.width * self.original_dimensions.height
        return {
            "source_index": self.source_index,
            "source_media_hash": self.source_media_hash,
            "original_dimensions": self.original_dimensions.to_dict(),
            "representation": self.representation.to_dict(),
            "is_relative": True,
            "is_metric": False,
            "units": "unitless",
            "coordinate_frame": self.coordinate_frame,
            "valid_mask": self.valid_mask.to_dict(),
            "valid_pixel_count": self.valid_pixel_count,
            "valid_fraction": self.valid_pixel_count / total,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True)
class DepthResult:
    status: PerceptionStatus | str
    sources: tuple[SourceBinding, ...]
    estimates: tuple[DepthEstimate, ...]
    backend: BackendIdentity
    provenance: dict[str, Any]
    failure: FailureInfo | None = None
    schema_version: str = field(default=PERCEPTION_SCHEMA_VERSION, init=False)
    result_kind: str = field(default="relative_depth", init=False)

    def __post_init__(self) -> None:
        try:
            status = PerceptionStatus(self.status)
        except (TypeError, ValueError):
            raise PerceptionContractError("unsupported depth status") from None
        if status not in {PerceptionStatus.SUCCESS, PerceptionStatus.FAILED}:
            raise PerceptionContractError("depth status must be success or failed")
        object.__setattr__(self, "status", status)
        if not isinstance(self.sources, tuple) or not self.sources or not all(
            isinstance(item, SourceBinding) for item in self.sources
        ):
            raise PerceptionContractError("depth sources are invalid")
        if len({item.source_index for item in self.sources}) != len(self.sources):
            raise PerceptionContractError("depth source indexes must be unique")
        if not isinstance(self.estimates, tuple) or not all(
            isinstance(item, DepthEstimate) for item in self.estimates
        ):
            raise PerceptionContractError("depth estimates are invalid")
        expected = {
            (item.source_index, item.source_media_hash, item.original_dimensions)
            for item in self.sources
        }
        actual = {
            (item.source_index, item.source_media_hash, item.original_dimensions)
            for item in self.estimates
        }
        if status is PerceptionStatus.SUCCESS and actual != expected:
            raise PerceptionContractError(
                "successful depth result must bind every source exactly once"
            )
        if status is PerceptionStatus.FAILED:
            if self.failure is None or self.estimates:
                raise PerceptionContractError(
                    "failed depth result requires failure details and no estimates"
                )
        elif self.failure is not None:
            raise PerceptionContractError("only failed depth may contain failure")
        if not isinstance(self.backend, BackendIdentity):
            raise PerceptionContractError("depth backend identity is invalid")
        object.__setattr__(self, "provenance", _safe_provenance(self.provenance))

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "result_kind": self.result_kind,
            "status": self.status.value,
            "sources": [item.to_dict() for item in self.sources],
            "estimates": [item.to_dict() for item in self.estimates],
            "backend": self.backend.to_dict(),
            "provenance": self.provenance,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }
        canonical_json_bytes(value)
        return value


PerceptionResult = DetectionSegmentationResult | DepthResult


__all__ = [
    "DEPTH_COORDINATE_FRAME",
    "ORIGINAL_XYXY_COORDINATE_SYSTEM",
    "PERCEPTION_SCHEMA_VERSION",
    "PILOT_SCHEMA_VERSION",
    "ArtifactReference",
    "BackendIdentity",
    "BackendInferenceError",
    "BackendNotPreparedError",
    "BackendOutputError",
    "BackendPreparationError",
    "BoundingBoxXYXY",
    "DepthEstimate",
    "DepthResult",
    "DetectionEstimate",
    "DetectionSegmentationResult",
    "FailureInfo",
    "ImageDimensions",
    "ImageSource",
    "LetterboxTransform",
    "LocalImageSource",
    "PerceptionContractError",
    "PerceptionError",
    "PerceptionInputError",
    "PerceptionResult",
    "PerceptionStatus",
    "SourceBinding",
    "canonical_json_bytes",
    "canonical_sha256",
    "letterbox_xyxy_to_original",
    "normalize_prompts",
    "validate_device",
    "validate_sha256",
]
