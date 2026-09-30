"""Lazy, local-only Ultralytics YOLOE detection/segmentation backend."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from .artifacts import (
    ArtifactStore,
    array_to_python,
    encode_original_polygon_mask,
    fingerprint_local_path,
)
from .contracts import (
    ArtifactReference,
    BackendIdentity,
    BackendNotPreparedError,
    BackendOutputError,
    BackendPreparationError,
    BoundingBoxXYXY,
    DetectionEstimate,
    DetectionSegmentationResult,
    FailureInfo,
    LocalImageSource,
    PerceptionContractError,
    PerceptionError,
    PerceptionInputError,
    PerceptionStatus,
    SourceBinding,
    normalize_prompts,
    validate_device,
)


YOLOE_BACKEND_ID = "ultralytics_yoloe_local"
_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".torchscript"}


@contextmanager
def _ultralytics_offline_environment() -> Iterator[None]:
    """Disable Ultralytics network probes and dependency auto-installation."""

    overrides = {
        "YOLO_OFFLINE": "true",
        "YOLO_AUTOINSTALL": "false",
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _package_version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _safe_sources(
    sources: Sequence[LocalImageSource],
) -> tuple[LocalImageSource, ...]:
    if isinstance(sources, (str, bytes, bytearray)) or not isinstance(
        sources, Sequence
    ):
        raise PerceptionInputError("sources must be a sequence of local images")
    normalized = tuple(sources)
    if not normalized or not all(
        isinstance(item, LocalImageSource) for item in normalized
    ):
        raise PerceptionInputError("sources must contain LocalImageSource values")
    indexes = [item.binding.source_index for item in normalized]
    if len(set(indexes)) != len(indexes):
        raise PerceptionInputError("source indexes must be unique")
    return normalized


def _model_names(value: Any) -> dict[int, str]:
    names = getattr(value, "names", None)
    if isinstance(names, Mapping):
        items = names.items()
    elif isinstance(names, Sequence) and not isinstance(
        names, (str, bytes, bytearray)
    ):
        items = enumerate(names)
    else:
        nested = getattr(value, "model", None)
        names = getattr(nested, "names", None)
        if isinstance(names, Mapping):
            items = names.items()
        elif isinstance(names, Sequence) and not isinstance(
            names, (str, bytes, bytearray)
        ):
            items = enumerate(names)
        else:
            raise BackendPreparationError(
                "local YOLOE model does not expose a class vocabulary"
            )
    result: dict[int, str] = {}
    for raw_id, raw_name in items:
        try:
            class_id = int(raw_id)
        except (TypeError, ValueError, OverflowError):
            raise BackendPreparationError(
                "local YOLOE class vocabulary has an invalid class id"
            ) from None
        if isinstance(raw_id, bool) or class_id < 0:
            raise BackendPreparationError(
                "local YOLOE class vocabulary has an invalid class id"
            )
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise BackendPreparationError(
                "local YOLOE class vocabulary has an invalid class name"
            )
        result[class_id] = " ".join(raw_name.split())
    if not result:
        raise BackendPreparationError("local YOLOE class vocabulary is empty")
    return result


def _prompt_binding(
    names: Mapping[int, str], prompts: tuple[str, ...]
) -> tuple[dict[int, str], tuple[int, ...], tuple[str, ...], str]:
    if not prompts:
        return (
            {class_id: name for class_id, name in names.items()},
            tuple(sorted(names)),
            (),
            "embedded_model_vocabulary",
        )
    by_name: dict[str, list[int]] = {}
    for class_id, name in names.items():
        by_name.setdefault(name.casefold(), []).append(class_id)
    bindings: dict[int, str] = {}
    selected: list[int] = []
    ambiguity: set[str] = set()
    for prompt in prompts:
        matches = by_name.get(prompt.casefold(), [])
        if not matches:
            raise BackendPreparationError(
                "requested prompt is absent from the prepared local vocabulary"
            )
        if len(matches) > 1:
            ambiguity.add("duplicate_prepared_class_name")
        for class_id in matches:
            bindings[class_id] = prompt
            selected.append(class_id)
    return (
        bindings,
        tuple(sorted(set(selected))),
        tuple(sorted(ambiguity)),
        "prepared_prompt_embeddings",
    )


class UltralyticsYOLOEBackend:
    """Run YOLOE only after an explicit, offline ``prepare`` call.

    Dynamic ``set_classes`` is intentionally not used: current Ultralytics
    versions may download/install the text encoder on first use.  Custom text
    prompts therefore require a locally prepared prompt-embedding ``.npz`` or
    weights whose embedded vocabulary already contains those exact prompts.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cpu",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        artifact_store: ArtifactStore | None = None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        local_model = Path(model_path).expanduser()
        if not local_model.is_file():
            raise PerceptionInputError("YOLOE model must be an existing local file")
        if local_model.suffix.casefold() not in _MODEL_SUFFIXES:
            raise PerceptionInputError("YOLOE local model format is unsupported")
        self.model_path = local_model.resolve()
        self.device = validate_device(device)
        self.confidence_threshold = _threshold(
            confidence_threshold, "confidence_threshold"
        )
        self.iou_threshold = _threshold(iou_threshold, "iou_threshold")
        if artifact_store is not None and not isinstance(
            artifact_store, ArtifactStore
        ):
            raise PerceptionInputError("artifact_store must use ArtifactStore")
        if model_factory is not None and not callable(model_factory):
            raise PerceptionInputError("model_factory must be callable")
        self.artifact_store = artifact_store
        self._model_factory = model_factory
        self._model: Any = None
        self._identity: BackendIdentity | None = None
        self._names: dict[int, str] = {}
        self._prompts: tuple[str, ...] = ()
        self._class_prompts: dict[int, str] = {}
        self._selected_class_ids: tuple[int, ...] = ()
        self._preparation_ambiguity: tuple[str, ...] = ()
        self._prompt_binding_mode = "unprepared"
        self._prompt_embeddings_sha256: str | None = None
        self._lock = threading.Lock()

    @property
    def prepared(self) -> bool:
        return self._model is not None and self._identity is not None

    @property
    def identity(self) -> BackendIdentity:
        if self._identity is None:
            raise BackendNotPreparedError("YOLOE backend has not been prepared")
        return self._identity

    def prepare(
        self,
        *,
        prompts: Sequence[str] | None = None,
        prompt_embeddings_path: str | Path | None = None,
    ) -> BackendIdentity:
        normalized_prompts = normalize_prompts(prompts)
        embedding_path: Path | None = None
        if prompt_embeddings_path is not None:
            embedding_path = Path(prompt_embeddings_path).expanduser()
            if (
                not embedding_path.is_file()
                or embedding_path.suffix.casefold() != ".npz"
            ):
                raise PerceptionInputError(
                    "prompt embeddings must be an existing local .npz file"
                )
            embedding_path = embedding_path.resolve()
        try:
            with _ultralytics_offline_environment():
                if self._model_factory is None:
                    # Optional heavyweight import: never reached by core imports.
                    from ultralytics import YOLOE

                    model = YOLOE(str(self.model_path), verbose=False)
                else:
                    model = self._model_factory(str(self.model_path))
                if embedding_path is not None:
                    loader = getattr(model, "load_prompt_embeddings", None)
                    if not callable(loader):
                        raise BackendPreparationError(
                            "installed Ultralytics lacks local prompt embedding support"
                        )
                    loader(str(embedding_path))
                names = _model_names(model)
                bindings, selected, ambiguity, mode = _prompt_binding(
                    names, normalized_prompts
                )
                mover = getattr(model, "to", None)
                if callable(mover):
                    moved = mover(self.device)
                    if moved is not None:
                        model = moved
                evaluator = getattr(model, "eval", None)
                if callable(evaluator):
                    evaluator()
        except ImportError:
            raise BackendPreparationError(
                "optional Ultralytics dependencies are not installed"
            ) from None
        except PerceptionError:
            raise
        except Exception:
            raise BackendPreparationError(
                "local YOLOE model preparation failed"
            ) from None

        identity = BackendIdentity(
            backend_id=YOLOE_BACKEND_ID,
            implementation="ultralytics.YOLOE",
            version=_package_version("ultralytics"),
            model_sha256=fingerprint_local_path(self.model_path),
        )
        self._model = model
        self._identity = identity
        self._names = names
        self._prompts = normalized_prompts
        self._class_prompts = bindings
        self._selected_class_ids = selected
        self._preparation_ambiguity = ambiguity
        self._prompt_binding_mode = mode
        self._prompt_embeddings_sha256 = (
            None
            if embedding_path is None
            else fingerprint_local_path(embedding_path)
        )
        return identity

    def infer(
        self,
        sources: Sequence[LocalImageSource],
    ) -> DetectionSegmentationResult:
        normalized_sources = _safe_sources(sources)
        if not self.prepared:
            raise BackendNotPreparedError("YOLOE backend has not been prepared")
        for source in normalized_sources:
            source.verify_unchanged()
        images = [source.open_rgb() for source in normalized_sources]
        kwargs: dict[str, Any] = {
            "source": images,
            "stream": False,
            "device": self.device,
            "conf": self.confidence_threshold,
            "iou": self.iou_threshold,
            "verbose": False,
            "save": False,
            "show": False,
        }
        if self._prompts:
            kwargs["classes"] = list(self._selected_class_ids)
        provenance = self._provenance()
        try:
            with self._lock, _ultralytics_offline_environment():
                results = list(self._model.predict(**kwargs))
            return parse_yoloe_results(
                results,
                normalized_sources,
                backend=self.identity,
                class_names=self._names,
                class_prompts=self._class_prompts,
                prompts=self._prompts,
                artifact_store=self.artifact_store,
                preparation_ambiguity=self._preparation_ambiguity,
                provenance=provenance,
            )
        except (PerceptionInputError, BackendNotPreparedError):
            raise
        except (PerceptionContractError, BackendOutputError):
            return DetectionSegmentationResult(
                status=PerceptionStatus.FAILED,
                sources=tuple(item.binding for item in normalized_sources),
                prompts=self._prompts,
                detections=(),
                backend=self.identity,
                provenance=provenance,
                limitations=("estimate_only", "no_cross_view_identity"),
                failure=FailureInfo(
                    code="invalid_backend_output",
                    message="YOLOE returned output outside the portable contract",
                ),
            )
        except Exception:
            return DetectionSegmentationResult(
                status=PerceptionStatus.FAILED,
                sources=tuple(item.binding for item in normalized_sources),
                prompts=self._prompts,
                detections=(),
                backend=self.identity,
                provenance=provenance,
                limitations=("estimate_only", "no_cross_view_identity"),
                failure=FailureInfo(
                    code="backend_inference_failed",
                    message="local YOLOE inference failed",
                ),
            )

    def _provenance(self) -> dict[str, Any]:
        return {
            "local_files_only": True,
            "network_access": "disabled",
            "automatic_download": False,
            "prompt_binding_mode": self._prompt_binding_mode,
            "prompt_embeddings_sha256": self._prompt_embeddings_sha256,
            "coordinate_output": "original_image_pixels",
            "confidence_source": "ultralytics_boxes_conf_or_null",
            "ultralytics_operation": "predict",
            "tracking_used": False,
            "cross_view_identity_inferred": False,
            "semantic_completion_performed": False,
        }


def _threshold(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise PerceptionInputError(f"{field_name} must be in [0, 1]")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        raise PerceptionInputError(f"{field_name} must be in [0, 1]") from None
    if not 0.0 <= result <= 1.0:
        raise PerceptionInputError(f"{field_name} must be in [0, 1]")
    return result


def _sequence(value: Any, field_name: str) -> list[Any]:
    converted = array_to_python(value)
    if isinstance(converted, (str, bytes, bytearray)) or not isinstance(
        converted, Sequence
    ):
        raise BackendOutputError(f"{field_name} must be an array")
    return list(converted)


def _class_id(value: Any) -> int:
    if isinstance(value, bool):
        raise BackendOutputError("YOLOE class id must be an integer")
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        raise BackendOutputError("YOLOE class id must be an integer") from None
    integer = int(converted)
    if converted != integer or integer < 0:
        raise BackendOutputError("YOLOE class id must be a non-negative integer")
    return integer


def _original_shape(result: Any, source: LocalImageSource) -> None:
    shape = getattr(result, "orig_shape", None)
    if shape is None:
        raise BackendOutputError("YOLOE result lacks original image dimensions")
    values = _sequence(shape, "YOLOE orig_shape")
    if len(values) != 2:
        raise BackendOutputError("YOLOE orig_shape must be height,width")
    try:
        observed = (int(values[0]), int(values[1]))
    except (TypeError, ValueError, OverflowError):
        raise BackendOutputError("YOLOE orig_shape is invalid") from None
    expected = (
        source.binding.original_dimensions.height,
        source.binding.original_dimensions.width,
    )
    if observed != expected:
        raise BackendOutputError(
            "YOLOE result dimensions do not match the bound source"
        )


def _mask_reference(
    raw_polygon: Any,
    *,
    source: LocalImageSource,
    artifact_store: ArtifactStore | None,
) -> ArtifactReference:
    dimensions = source.binding.original_dimensions
    payload = encode_original_polygon_mask(
        raw_polygon,
        width=dimensions.width,
        height=dimensions.height,
    )
    if artifact_store is not None:
        return artifact_store.write(
            payload,
            kind="mask",
            representation="polygon_xy_original",
            shape=(dimensions.height, dimensions.width),
            dtype="float32",
            suffix=".json",
        )
    return ArtifactReference(
        kind="mask",
        representation="polygon_xy_original",
        sha256=hashlib.sha256(payload).hexdigest(),
        shape=(dimensions.height, dimensions.width),
        dtype="float32",
        reference=None,
    )


def parse_yoloe_results(
    results: Sequence[Any],
    sources: Sequence[LocalImageSource],
    *,
    backend: BackendIdentity,
    class_names: Mapping[int, str],
    class_prompts: Mapping[int, str],
    prompts: Sequence[str] = (),
    artifact_store: ArtifactStore | None = None,
    preparation_ambiguity: Sequence[str] = (),
    provenance: Mapping[str, Any] | None = None,
) -> DetectionSegmentationResult:
    """Validate framework output and bind every estimate to one source."""

    normalized_sources = _safe_sources(sources)
    raw_results = list(results)
    if len(raw_results) != len(normalized_sources):
        raise BackendOutputError("YOLOE result count does not match source count")
    detections: list[DetectionEstimate] = []
    ambiguity_reasons = set(preparation_ambiguity)
    for result, source in zip(raw_results, normalized_sources):
        _original_shape(result, source)
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            rows: list[Any] = []
            classes: list[Any] = []
            confidences: list[Any] | None = None
        else:
            rows = _sequence(getattr(boxes, "xyxy", None), "YOLOE boxes.xyxy")
            classes = _sequence(getattr(boxes, "cls", None), "YOLOE boxes.cls")
            raw_confidence = getattr(boxes, "conf", None)
            confidences = (
                None
                if raw_confidence is None
                else _sequence(raw_confidence, "YOLOE boxes.conf")
            )
        if len(classes) != len(rows):
            raise BackendOutputError("YOLOE class and bbox counts differ")
        if confidences is not None and len(confidences) != len(rows):
            raise BackendOutputError("YOLOE confidence and bbox counts differ")

        masks = getattr(result, "masks", None)
        raw_polygons: list[Any] | None = None
        if masks is not None:
            xy = getattr(masks, "xy", None)
            if xy is not None:
                raw_polygons = _sequence(xy, "YOLOE masks.xy")
                if len(raw_polygons) != len(rows):
                    ambiguity_reasons.add("mask_detection_count_mismatch")
                    raw_polygons = None

        for index, raw_bbox in enumerate(rows):
            bbox_values = _sequence(raw_bbox, "YOLOE bbox")
            if len(bbox_values) != 4:
                raise BackendOutputError("YOLOE bbox must contain xyxy")
            class_id = _class_id(classes[index])
            class_name = class_names.get(class_id)
            if class_name is None:
                raise BackendOutputError("YOLOE class id is absent from vocabulary")
            prompt = class_prompts.get(class_id)
            ambiguity_reason = None
            if prompt is None:
                ambiguity_reason = "unbound_class_prompt"
                ambiguity_reasons.add(ambiguity_reason)
            mask = (
                None
                if raw_polygons is None
                else _mask_reference(
                    raw_polygons[index],
                    source=source,
                    artifact_store=artifact_store,
                )
            )
            detections.append(
                DetectionEstimate(
                    source_index=source.binding.source_index,
                    source_media_hash=source.binding.source_media_hash,
                    class_name=class_name,
                    prompt=prompt,
                    bbox=BoundingBoxXYXY(
                        x_min=bbox_values[0],
                        y_min=bbox_values[1],
                        x_max=bbox_values[2],
                        y_max=bbox_values[3],
                        original_dimensions=source.binding.original_dimensions,
                    ),
                    confidence=(
                        None if confidences is None else confidences[index]
                    ),
                    mask=mask,
                    ambiguity_reason=ambiguity_reason,
                )
            )
    detections.sort(
        key=lambda item: (
            item.source_index,
            item.class_name.casefold(),
            item.bbox.xyxy,
            -1.0 if item.confidence is None else item.confidence,
        )
    )
    if ambiguity_reasons:
        status = PerceptionStatus.AMBIGUOUS
    elif detections:
        status = PerceptionStatus.SUCCESS
    else:
        status = PerceptionStatus.NO_DETECTION
    safe_provenance = {
        "local_files_only": True,
        "network_access": "disabled",
        "automatic_download": False,
        "coordinate_output": "original_image_pixels",
        "confidence_source": "ultralytics_boxes_conf_or_null",
        "tracking_used": False,
        "cross_view_identity_inferred": False,
        "semantic_completion_performed": False,
        **dict(provenance or {}),
    }
    return DetectionSegmentationResult(
        status=status,
        sources=tuple(item.binding for item in normalized_sources),
        prompts=tuple(prompts),
        detections=tuple(detections),
        backend=backend,
        provenance=safe_provenance,
        ambiguity_reasons=tuple(sorted(ambiguity_reasons)),
        limitations=("estimate_only", "no_cross_view_identity"),
    )


__all__ = [
    "YOLOE_BACKEND_ID",
    "UltralyticsYOLOEBackend",
    "parse_yoloe_results",
]
