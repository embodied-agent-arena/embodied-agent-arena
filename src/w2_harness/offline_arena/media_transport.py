"""Deterministic CPU-only media transport for the offline W2 Arena.

The transport is deliberately independent of model and agent runtimes.  It
turns trusted local benchmark media into bounded, content-addressed payloads
that can be shared by the Direct and W2-Light harness profiles.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import io
import json
import math
import mimetypes
import os
from pathlib import Path
import re
import tempfile
import threading
from typing import Any, Callable
import warnings

from PIL import Image, ImageOps


OPEN_ASSET = "OPEN_ASSET"
GET_VIEW = "GET_VIEW"
GET_FRAME = "GET_FRAME"
GET_FRAME_WINDOW = "GET_FRAME_WINDOW"
CROP_REGION = "CROP_REGION"
ZOOM_REGION = "ZOOM_REGION"
COMPOSE_ASSETS = "COMPOSE_ASSETS"

MEDIA_HELPERS = frozenset(
    {
        OPEN_ASSET,
        GET_VIEW,
        GET_FRAME,
        GET_FRAME_WINDOW,
        CROP_REGION,
        ZOOM_REGION,
        COMPOSE_ASSETS,
    }
)
MEDIA_TRANSPORT_SCHEMA_VERSION = "w2-media-transport-record-v1.0"

DEFAULT_MAX_SOURCE_BYTES = 1024 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 50_000_000
DEFAULT_MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_CONTEXT_PAYLOAD_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_SIZE = (1600, 1600)
DEFAULT_JPEG_QUALITY = 90
DEFAULT_MAX_FRAME_WINDOW = 16
DEFAULT_MAX_COMPOSE_ASSETS = 16

_CACHE_SCHEMA_VERSION = "w2-media-transport-cache-v1.0"
_ENCODER_CONTRACT = "pillow-jpeg-q90-444-or-png-z9-v1"
_DATA_URL = re.compile(
    r"\Adata:([^;,]+);base64,([A-Za-z0-9+/]*={0,2})\Z", re.ASCII
)
_SHA256 = re.compile(r"\A[0-9a-f]{64}\Z", re.ASCII)
_WINDOWS_ABSOLUTE = re.compile(r"\A[A-Za-z]:[\\/]", re.ASCII)
_FORMAT_TO_MIME = {
    "BMP": "image/bmp",
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
_IMAGE_MIMES = frozenset(_FORMAT_TO_MIME.values())
_VIDEO_SUFFIX_TO_MIME = {
    ".avi": "video/x-msvideo",
    ".m4v": "video/x-m4v",
    ".mkv": "video/x-matroska",
    ".mov": "video/quicktime",
    ".mp4": "video/mp4",
    ".webm": "video/webm",
}
_OPERATION_PREFIX = {
    OPEN_ASSET: "asset",
    GET_FRAME: "frame-asset",
    CROP_REGION: "crop",
    ZOOM_REGION: "zoom",
    COMPOSE_ASSETS: "composite",
}


class MediaTransportError(ValueError):
    """Base error for a fail-closed media operation."""


class MediaSourceError(MediaTransportError):
    """The source reference or declared source metadata is invalid."""


class MediaDecodeError(MediaTransportError):
    """The source cannot be decoded under the supported CPU contract."""


class MediaPayloadLimitError(MediaTransportError):
    """A generated model payload exceeds its configured hard limit."""


class MediaFrameError(MediaTransportError):
    """A video frame request is invalid or cannot be decoded."""


@dataclass(frozen=True)
class MediaTransportRecord(Mapping[str, Any]):
    """One sanitized model payload plus deterministic transport provenance."""

    operation: str
    asset_id: str
    media_kind: str
    mime_type: str
    width: int
    height: int
    byte_length: int
    payload_bytes: int
    source_sha256: str
    content_sha256: str
    derivation_sha256: str
    parent_asset_ids: tuple[str, ...] = ()
    data_url: str | None = None
    cache_hit: bool = False
    view_id: str | None = None
    frame_id: str | None = None
    frame_index: int | None = None
    timestamp_ms: int | None = None
    frame_count: int | None = None
    duration_ms: int | None = None
    fps_numerator: int | None = None
    fps_denominator: int | None = None
    schema_version: str = MEDIA_TRANSPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEDIA_TRANSPORT_SCHEMA_VERSION:
            raise MediaTransportError("unsupported media transport record schema")
        if self.operation not in MEDIA_HELPERS:
            raise MediaTransportError("unknown media transport operation")
        if self.media_kind not in {"rgb", "mask", "diagram", "video"}:
            raise MediaTransportError("record media_kind is unsupported")
        expected_mime = {
            "rgb": "image/jpeg",
            "mask": "image/png",
            "diagram": "image/png",
        }.get(self.media_kind)
        if expected_mime is not None and self.mime_type != expected_mime:
            raise MediaTransportError("record media kind and MIME are inconsistent")
        if self.media_kind == "video" and not self.mime_type.startswith("video/"):
            raise MediaTransportError("video record MIME is invalid")
        if not _safe_public_id(self.asset_id):
            raise MediaTransportError("asset_id must be an opaque identifier")
        if self.view_id is not None and not _safe_public_id(self.view_id):
            raise MediaTransportError("view_id must be an opaque identifier")
        if self.frame_id is not None and not _safe_public_id(self.frame_id):
            raise MediaTransportError("frame_id must be an opaque identifier")
        for digest in (
            self.source_sha256,
            self.content_sha256,
            self.derivation_sha256,
        ):
            if _SHA256.fullmatch(digest) is None:
                raise MediaTransportError("transport hashes must be lowercase SHA-256")
        if self.width < 1 or self.height < 1:
            raise MediaTransportError("transport dimensions must be positive")
        if self.byte_length < 1 or self.payload_bytes < 0:
            raise MediaTransportError("transport byte lengths are invalid")
        if self.data_url is None and self.payload_bytes != 0:
            raise MediaTransportError("payload_bytes requires a model payload")
        if self.data_url is not None:
            match = _DATA_URL.fullmatch(self.data_url)
            if match is None or match.group(1) != self.mime_type:
                raise MediaTransportError("record payload MIME is inconsistent")
            if len(self.data_url.encode("ascii")) != self.payload_bytes:
                raise MediaTransportError("record payload byte count is inconsistent")
            try:
                raw = base64.b64decode(match.group(2), validate=True)
            except (binascii.Error, ValueError):
                raise MediaTransportError("record payload base64 is invalid") from None
            if len(raw) != self.byte_length:
                raise MediaTransportError("record byte_length is inconsistent")
            if hashlib.sha256(raw).hexdigest() != self.content_sha256:
                raise MediaTransportError("record content hash is inconsistent")
        if any(not _safe_public_id(value) for value in self.parent_asset_ids):
            raise MediaTransportError("parent asset IDs must be opaque")
        if self.frame_index is not None and self.frame_index < 0:
            raise MediaTransportError("frame_index must be non-negative")
        if self.timestamp_ms is not None and self.timestamp_ms < 0:
            raise MediaTransportError("timestamp_ms must be non-negative")
        if self.frame_count is not None and self.frame_count < 1:
            raise MediaTransportError("frame_count must be positive")
        if (
            self.frame_index is not None
            and self.frame_count is not None
            and self.frame_index >= self.frame_count
        ):
            raise MediaTransportError("frame_index must be less than frame_count")
        if (self.fps_numerator is None) != (self.fps_denominator is None):
            raise MediaTransportError("FPS numerator and denominator must be paired")
        if self.fps_numerator is not None:
            denominator = self.fps_denominator
            if denominator is None or self.fps_numerator < 1 or denominator < 1:
                raise MediaTransportError("record FPS must be positive")

    @property
    def sha256(self) -> str:
        """Alias for callers that use a generic payload hash field."""

        return self.content_sha256

    @property
    def output_sha256(self) -> str:
        return self.content_sha256

    @property
    def payload_size_bytes(self) -> int:
        return self.payload_bytes

    @property
    def payload(self) -> str | None:
        return self.data_url

    @property
    def cache_key(self) -> str:
        return self.derivation_sha256

    @property
    def transport_bytes(self) -> int:
        return self.byte_length

    def to_dict(
        self,
        *,
        include_payload: bool = True,
        include_runtime: bool = True,
    ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "asset_id": self.asset_id,
            "media_kind": self.media_kind,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "byte_length": self.byte_length,
            "payload_bytes": self.payload_bytes,
            "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256,
            "derivation_sha256": self.derivation_sha256,
            "parent_asset_ids": list(self.parent_asset_ids),
            "view_id": self.view_id,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "timestamp_ms": self.timestamp_ms,
            "frame_count": self.frame_count,
            "duration_ms": self.duration_ms,
            "fps_numerator": self.fps_numerator,
            "fps_denominator": self.fps_denominator,
        }
        if include_payload:
            value["data_url"] = self.data_url
        if include_runtime:
            value["cache_hit"] = self.cache_hit
        _assert_model_visible(value)
        return value

    def to_model_visible_dict(self) -> dict[str, Any]:
        """Return the profile-neutral representation shown to either harness."""

        return self.to_dict(include_payload=True, include_runtime=False)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


@dataclass(frozen=True)
class _SourceSpec:
    source: Any
    media_kind: str = "auto"
    declared_mime: str | None = None
    expected_sha256: str | None = None


@dataclass(frozen=True)
class _DecodedImage:
    image: Image.Image
    source_sha256: str
    source_mime: str


@dataclass(frozen=True)
class _VideoSource:
    path: Path
    source_sha256: str
    mime_type: str
    byte_length: int
    width: int
    height: int
    frame_count: int
    fps: Fraction
    file_signature: tuple[int, int]
    record: MediaTransportRecord


class MediaTransport:
    """One deterministic transport shared by Direct and W2-Light.

    ``source_root`` is optional.  When supplied, every path is resolved beneath
    it.  Paths are never copied into records, cache metadata, or exceptions.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        *,
        source_root: str | Path | None = None,
        sources: Mapping[str, Any] | None = None,
        views: Mapping[str, Any] | None = None,
        max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
        max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
        max_context_payload_bytes: int = DEFAULT_MAX_CONTEXT_PAYLOAD_BYTES,
        default_max_size: int | Sequence[int] = DEFAULT_MAX_SIZE,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        max_frame_window: int = DEFAULT_MAX_FRAME_WINDOW,
        max_compose_assets: int = DEFAULT_MAX_COMPOSE_ASSETS,
    ) -> None:
        self.max_source_bytes = _positive_int(
            max_source_bytes, "max_source_bytes"
        )
        self.max_image_pixels = _positive_int(
            max_image_pixels, "max_image_pixels"
        )
        self.max_payload_bytes = _positive_int(
            max_payload_bytes, "max_payload_bytes"
        )
        self.max_context_payload_bytes = _positive_int(
            max_context_payload_bytes, "max_context_payload_bytes"
        )
        self.default_max_size = _size_pair(
            default_max_size, "default_max_size"
        )
        if (
            isinstance(jpeg_quality, bool)
            or not isinstance(jpeg_quality, int)
            or not 1 <= jpeg_quality <= 95
        ):
            raise MediaTransportError("jpeg_quality must be an integer from 1 to 95")
        self.jpeg_quality = jpeg_quality
        self.max_frame_window = _positive_int(
            max_frame_window, "max_frame_window"
        )
        self.max_compose_assets = _positive_int(
            max_compose_assets, "max_compose_assets"
        )

        self.source_root = (
            None if source_root is None else Path(source_root).resolve()
        )
        if self.source_root is not None and not self.source_root.is_dir():
            raise MediaSourceError("source_root must be an existing directory")
        self.cache_dir = None if cache_dir is None else Path(cache_dir).resolve()
        if self.cache_dir is not None:
            (self.cache_dir / "derived").mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._source_specs: dict[str, _SourceSpec] = {}
        self._view_specs: dict[str, _SourceSpec] = {}
        self._records: dict[str, MediaTransportRecord] = {}
        self._raw_assets: dict[str, bytes] = {}
        self._videos: dict[str, _VideoSource] = {}
        self._videos_by_hash: dict[str, _VideoSource] = {}
        self._memory_cache: dict[str, tuple[bytes, dict[str, Any]]] = {}

        for source_id, source in (sources or {}).items():
            self.register_source(source_id, source)
        for view_id, source in (views or {}).items():
            self.register_view(view_id, source)

    @property
    def helper_names(self) -> tuple[str, ...]:
        return tuple(sorted(MEDIA_HELPERS))

    def register_source(
        self,
        source_id: str,
        source: Any,
        *,
        media_kind: str = "auto",
        declared_mime: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        source_id = _public_id(source_id, "source_id")
        spec = _source_spec(
            source,
            media_kind=media_kind,
            declared_mime=declared_mime,
            expected_sha256=expected_sha256,
        )
        with self._lock:
            if source_id in self._source_specs or source_id in self._view_specs:
                raise MediaSourceError("duplicate opaque source identifier")
            self._source_specs[source_id] = spec

    def register_view(
        self,
        view_id: str,
        source: Any,
        *,
        media_kind: str = "auto",
        declared_mime: str | None = None,
        expected_sha256: str | None = None,
    ) -> None:
        view_id = _public_id(view_id, "view_id")
        spec = _source_spec(
            source,
            media_kind=media_kind,
            declared_mime=declared_mime,
            expected_sha256=expected_sha256,
        )
        with self._lock:
            if view_id in self._view_specs or view_id in self._source_specs:
                raise MediaSourceError("duplicate opaque source identifier")
            self._view_specs[view_id] = spec

    def open_asset(
        self,
        source: Any,
        *,
        media_kind: str = "auto",
        kind: str | None = None,
        declared_mime: str | None = None,
        expected_sha256: str | None = None,
        max_size: int | Sequence[int] | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> MediaTransportRecord:
        """Validate and open one image or video without exposing its path."""

        if isinstance(source, MediaTransportRecord):
            return source
        if isinstance(source, Mapping):
            known = source.get("asset_id")
            if isinstance(known, str) and known in self._records:
                return self._records[known]
            if "source" in source:
                return self.open_asset(
                    source["source"],
                    media_kind=str(source.get("media_kind", media_kind)),
                    declared_mime=source.get("declared_mime", declared_mime),
                    expected_sha256=source.get(
                        "expected_sha256", expected_sha256
                    ),
                    max_size=source.get("max_size", max_size),
                    max_width=max_width,
                    max_height=max_height,
                )
            if isinstance(source.get("data_url"), str):
                source_record = source
                expected_sha256 = source_record.get(
                    "content_sha256", expected_sha256
                )
                if media_kind == "auto" and isinstance(
                    source_record.get("media_kind"), str
                ):
                    media_kind = source_record["media_kind"]
                source = source_record["data_url"]
            else:
                raise MediaSourceError("asset object has no usable public media")

        source, spec = self._resolve_registered_source(source)
        if spec is not None:
            if media_kind == "auto":
                media_kind = spec.media_kind
            declared_mime = declared_mime or spec.declared_mime
            expected_sha256 = expected_sha256 or spec.expected_sha256
        if kind is not None:
            if (
                media_kind != "auto"
                and _normalize_kind(media_kind) != _normalize_kind(kind)
            ):
                raise MediaSourceError("media_kind and kind disagree")
            media_kind = kind
        normalized_kind = _normalize_kind(media_kind, source=source)
        if normalized_kind == "video":
            return self._open_video(
                source,
                declared_mime=declared_mime,
                expected_sha256=expected_sha256,
            ).record

        target_size = self._output_size(
            max_size=max_size,
            max_width=max_width,
            max_height=max_height,
        )
        decoded = self._decode_image(
            source,
            declared_mime=declared_mime,
            expected_sha256=expected_sha256,
        )

        def build() -> Image.Image:
            prepared = _prepare_image(decoded.image, normalized_kind)
            return _resize_to_fit(
                prepared,
                target_size,
                allow_upscale=False,
                media_kind=normalized_kind,
            )

        return self._materialize_image(
            operation=OPEN_ASSET,
            media_kind=normalized_kind,
            source_sha256=decoded.source_sha256,
            parent_asset_ids=(),
            derivation={
                "source_sha256": decoded.source_sha256,
                "media_kind": normalized_kind,
                "max_size": list(target_size),
            },
            image_factory=build,
        )

    def get_view(
        self,
        view_id: str | Path,
        *,
        source: Any | None = None,
        **open_options: Any,
    ) -> MediaTransportRecord:
        """Open one registered view, preserving only its opaque view ID."""

        raw_view_id = str(view_id)
        if source is None and raw_view_id in self._view_specs:
            spec = self._view_specs[raw_view_id]
            source = spec.source
            open_options.setdefault("media_kind", spec.media_kind)
            open_options.setdefault("declared_mime", spec.declared_mime)
            open_options.setdefault("expected_sha256", spec.expected_sha256)
            public_view_id = _public_id(raw_view_id, "view_id")
        elif source is not None:
            public_view_id = _public_id(raw_view_id, "view_id")
        else:
            source = view_id
            public_view_id = None
        record = self.open_asset(source, **open_options)
        if record.media_kind == "video":
            raise MediaSourceError("GET_VIEW requires a decoded image asset")
        if public_view_id is None:
            public_view_id = f"view-{record.asset_id.rsplit('-', 1)[-1]}"
        return replace(record, operation=GET_VIEW, view_id=public_view_id)

    def get_frame(
        self,
        source: Any,
        frame_index: int | None = None,
        *,
        timestamp_ms: int | None = None,
        max_size: int | Sequence[int] | None = None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> MediaTransportRecord:
        """Decode one exact video frame with stable identity and timestamp."""

        video = self._as_video(source)
        if frame_index is None:
            if timestamp_ms is None:
                raise MediaFrameError("GET_FRAME requires frame_index or timestamp_ms")
            timestamp_ms = _nonnegative_int(timestamp_ms, "timestamp_ms")
            if (
                video.record.duration_ms is not None
                and timestamp_ms > video.record.duration_ms
            ):
                raise MediaFrameError("timestamp_ms is outside the decoded video range")
            frame_index = _frame_for_timestamp(timestamp_ms, video.fps)
        elif timestamp_ms is not None:
            raise MediaFrameError("frame_index and timestamp_ms are mutually exclusive")
        frame_index = _nonnegative_int(frame_index, "frame_index")
        if frame_index >= video.frame_count:
            raise MediaFrameError("frame_index is outside the decoded video range")
        canonical_timestamp = _timestamp_ms(frame_index, video.fps)
        target_size = self._output_size(
            max_size=max_size,
            max_width=max_width,
            max_height=max_height,
        )
        frame_identity = _canonical_sha256(
            {
                "source_sha256": video.source_sha256,
                "frame_index": frame_index,
                "timestamp_ms": canonical_timestamp,
            }
        )
        frame_id = f"frame-{frame_identity[:32]}"

        def build() -> Image.Image:
            image = self._decode_video_frame(video, frame_index)
            return _resize_to_fit(
                image.convert("RGB"),
                target_size,
                allow_upscale=False,
                media_kind="rgb",
            )

        return self._materialize_image(
            operation=GET_FRAME,
            media_kind="rgb",
            source_sha256=video.source_sha256,
            parent_asset_ids=(video.record.asset_id,),
            derivation={
                "source_sha256": video.source_sha256,
                "frame_index": frame_index,
                "timestamp_ms": canonical_timestamp,
                "max_size": list(target_size),
            },
            image_factory=build,
            frame_id=frame_id,
            frame_index=frame_index,
            timestamp_ms=canonical_timestamp,
            frame_count=video.frame_count,
            duration_ms=video.record.duration_ms,
            fps_numerator=video.fps.numerator,
            fps_denominator=video.fps.denominator,
        )

    def get_frame_window(
        self,
        source: Any,
        start_frame: int = 0,
        end_frame: int | None = None,
        *,
        step: int = 1,
        count: int | None = None,
        frame_indices: Sequence[int] | None = None,
        max_size: int | Sequence[int] | None = None,
    ) -> list[MediaTransportRecord]:
        """Return an ordered, bounded frame window; ``end_frame`` is exclusive."""

        video = self._as_video(source)
        if frame_indices is not None:
            if (
                end_frame is not None
                or count is not None
                or start_frame != 0
                or step != 1
            ):
                raise MediaFrameError(
                    "frame_indices cannot be combined with window parameters"
                )
            if isinstance(frame_indices, (str, bytes)) or not isinstance(
                frame_indices, Sequence
            ):
                raise MediaFrameError("frame_indices must be a sequence")
            if len(frame_indices) > self.max_frame_window:
                raise MediaFrameError(
                    "frame window exceeds the configured frame limit"
                )
            indices = [
                _nonnegative_int(value, "frame_index") for value in frame_indices
            ]
            if not indices:
                raise MediaFrameError("frame_indices must not be empty")
            if indices != sorted(indices) or len(set(indices)) != len(indices):
                raise MediaFrameError("frame_indices must be strictly increasing")
        else:
            start = _nonnegative_int(start_frame, "start_frame")
            stride = _positive_int(step, "step")
            if end_frame is not None and count is not None:
                raise MediaFrameError("end_frame and count are mutually exclusive")
            if end_frame is not None:
                stop = _nonnegative_int(end_frame, "end_frame")
                if stop <= start:
                    raise MediaFrameError("end_frame must be greater than start_frame")
                if stop > video.frame_count:
                    raise MediaFrameError(
                        "frame window is outside the decoded video range"
                    )
                frame_range = range(start, stop, stride)
                if len(frame_range) > self.max_frame_window:
                    raise MediaFrameError(
                        "frame window exceeds the configured frame limit"
                    )
                indices = list(frame_range)
            else:
                requested = 3 if count is None else _positive_int(count, "count")
                if requested > self.max_frame_window:
                    raise MediaFrameError(
                        "frame window exceeds the configured frame limit"
                    )
                indices = [start + offset * stride for offset in range(requested)]
        if len(indices) > self.max_frame_window:
            raise MediaFrameError("frame window exceeds the configured frame limit")
        if any(index >= video.frame_count for index in indices):
            raise MediaFrameError("frame window is outside the decoded video range")
        return [
            self.get_frame(video.record, index, max_size=max_size)
            for index in indices
        ]

    def crop_region(
        self,
        asset: Any,
        region: Mapping[str, Any] | Sequence[float],
        *,
        coordinate_space: str = "normalized_0_1",
        bbox_format: str = "xyxy",
        max_size: int | Sequence[int] | None = None,
    ) -> MediaTransportRecord:
        """Crop a strict pixel or normalized region from a transported image."""

        parent, raw = self._resolve_image_asset(asset)
        image = self._decode_transport_image(raw, parent)
        pixel_box = _pixel_box(
            region,
            image.size,
            coordinate_space=coordinate_space,
            bbox_format=bbox_format,
        )
        target_size = self._output_size(max_size=max_size)

        def build() -> Image.Image:
            cropped = image.crop(pixel_box)
            return _resize_to_fit(
                cropped,
                target_size,
                allow_upscale=False,
                media_kind=parent.media_kind,
            )

        return self._materialize_image(
            operation=CROP_REGION,
            media_kind=parent.media_kind,
            source_sha256=parent.source_sha256,
            parent_asset_ids=(parent.asset_id,),
            derivation={
                "parent_content_sha256": parent.content_sha256,
                "pixel_box": list(pixel_box),
                "max_size": list(target_size),
            },
            image_factory=build,
            view_id=parent.view_id,
            frame_id=parent.frame_id,
            frame_index=parent.frame_index,
            timestamp_ms=parent.timestamp_ms,
            frame_count=parent.frame_count,
            duration_ms=parent.duration_ms,
            fps_numerator=parent.fps_numerator,
            fps_denominator=parent.fps_denominator,
        )

    def zoom_region(
        self,
        asset: Any,
        region: Mapping[str, Any] | Sequence[float] | None = None,
        *,
        zoom_factor: float = 2.0,
        factor: float | None = None,
        center: Sequence[float] | None = None,
        coordinate_space: str = "normalized_0_1",
        bbox_format: str = "xyxy",
        max_size: int | Sequence[int] | None = None,
    ) -> MediaTransportRecord:
        """Crop and deterministically enlarge a region within the output cap."""

        if factor is not None:
            if zoom_factor != 2.0:
                raise MediaTransportError(
                    "factor and zoom_factor are mutually exclusive"
                )
            zoom_factor = factor
        if (
            isinstance(zoom_factor, bool)
            or not isinstance(zoom_factor, (int, float))
            or not math.isfinite(float(zoom_factor))
            or float(zoom_factor) < 1.0
        ):
            raise MediaTransportError("zoom_factor must be finite and at least 1")
        zoom_factor = float(zoom_factor)
        parent, raw = self._resolve_image_asset(asset)
        image = self._decode_transport_image(raw, parent)
        if region is None:
            pixel_box = _center_zoom_box(
                image.size,
                center=center,
                zoom_factor=zoom_factor,
                coordinate_space=coordinate_space,
            )
        else:
            if center is not None:
                raise MediaTransportError(
                    "center cannot be combined with an explicit region"
                )
            pixel_box = _pixel_box(
                region,
                image.size,
                coordinate_space=coordinate_space,
                bbox_format=bbox_format,
            )
        cap = self._output_size(max_size=max_size)
        crop_width = pixel_box[2] - pixel_box[0]
        crop_height = pixel_box[3] - pixel_box[1]
        desired = (
            max(1, _round_half_up(crop_width * zoom_factor)),
            max(1, _round_half_up(crop_height * zoom_factor)),
        )
        target_size = (min(cap[0], desired[0]), min(cap[1], desired[1]))

        def build() -> Image.Image:
            cropped = image.crop(pixel_box)
            return _resize_to_fit(
                cropped,
                target_size,
                allow_upscale=True,
                media_kind=parent.media_kind,
            )

        return self._materialize_image(
            operation=ZOOM_REGION,
            media_kind=parent.media_kind,
            source_sha256=parent.source_sha256,
            parent_asset_ids=(parent.asset_id,),
            derivation={
                "parent_content_sha256": parent.content_sha256,
                "pixel_box": list(pixel_box),
                "zoom_factor": format(zoom_factor, ".12g"),
                "target_size": list(target_size),
            },
            image_factory=build,
            view_id=parent.view_id,
            frame_id=parent.frame_id,
            frame_index=parent.frame_index,
            timestamp_ms=parent.timestamp_ms,
            frame_count=parent.frame_count,
            duration_ms=parent.duration_ms,
            fps_numerator=parent.fps_numerator,
            fps_denominator=parent.fps_denominator,
        )

    def compose_assets(
        self,
        assets: Sequence[Any],
        *,
        layout: str = "horizontal",
        columns: int | None = None,
        gap: int = 0,
        background: Sequence[int] = (255, 255, 255),
        max_size: int | Sequence[int] | None = None,
    ) -> MediaTransportRecord:
        """Compose image assets into one deterministic PNG diagram."""

        if isinstance(assets, (str, bytes)) or not isinstance(assets, Sequence):
            raise MediaTransportError("assets must be a sequence")
        if not assets:
            raise MediaTransportError("COMPOSE_ASSETS requires at least one asset")
        if len(assets) > self.max_compose_assets:
            raise MediaTransportError("composition exceeds the configured asset limit")
        layout = str(layout).strip().lower()
        if layout not in {"horizontal", "vertical", "grid"}:
            raise MediaTransportError("layout must be horizontal, vertical, or grid")
        gap = _nonnegative_int(gap, "gap")
        background_rgb = _rgb(background)
        resolved = [self._resolve_image_asset(asset) for asset in assets]
        parents = [value[0] for value in resolved]
        images = [
            _prepare_image(
                self._decode_transport_image(raw, record), "rgb"
            )
            for record, raw in resolved
        ]
        if columns is None:
            columns = math.ceil(math.sqrt(len(images))) if layout == "grid" else 1
        columns = _positive_int(columns, "columns")
        if layout != "grid" and columns != 1:
            raise MediaTransportError("columns is only configurable for grid layout")
        target_size = self._output_size(max_size=max_size)
        canvas_size, placements = _composition_geometry(
            [image.size for image in images],
            layout=layout,
            columns=columns,
            gap=gap,
        )
        if canvas_size[0] * canvas_size[1] > self.max_image_pixels:
            raise MediaTransportError(
                "composed image exceeds the configured pixel limit"
            )

        def build() -> Image.Image:
            canvas = Image.new("RGB", canvas_size, background_rgb)
            for image, position in zip(images, placements):
                canvas.paste(image, position)
            return _resize_to_fit(
                canvas,
                target_size,
                allow_upscale=False,
                media_kind="diagram",
            )

        source_sha256 = _canonical_sha256(
            {"source_sha256": [record.source_sha256 for record in parents]}
        )
        return self._materialize_image(
            operation=COMPOSE_ASSETS,
            media_kind="diagram",
            source_sha256=source_sha256,
            parent_asset_ids=tuple(record.asset_id for record in parents),
            derivation={
                "parent_content_sha256": [
                    record.content_sha256 for record in parents
                ],
                "layout": layout,
                "columns": columns,
                "gap": gap,
                "background": list(background_rgb),
                "max_size": list(target_size),
            },
            image_factory=build,
        )

    def full_context_records(self, assets: Any) -> list[MediaTransportRecord]:
        """Materialize canonical full media context before profile projection."""

        values: list[Any]
        if isinstance(assets, Mapping) and not {
            "asset_id",
            "data_url",
            "source",
        }.intersection(assets):
            values = [
                {"view_id": str(view_id), "source": source}
                for view_id, source in sorted(
                    assets.items(), key=lambda item: str(item[0])
                )
            ]
        elif isinstance(assets, Sequence) and not isinstance(
            assets, (str, bytes, bytearray, memoryview)
        ):
            values = list(assets)
        else:
            values = [assets]

        records: list[MediaTransportRecord] = []
        for value in values:
            if isinstance(value, Mapping) and "view_id" in value and "source" in value:
                options = {
                    key: value[key]
                    for key in (
                        "media_kind",
                        "declared_mime",
                        "expected_sha256",
                        "max_size",
                    )
                    if key in value
                }
                record = self.get_view(
                    str(value["view_id"]), source=value["source"], **options
                )
            else:
                record = self.open_asset(value)
            if record.media_kind == "video":
                if record.frame_count is None:
                    raise MediaFrameError("video descriptor has no frame count")
                indices = _representative_indices(record.frame_count)
                records.extend(
                    self.get_frame(record, index) for index in indices
                )
            else:
                records.append(record)
        total = sum(record.payload_bytes for record in records)
        if total > self.max_context_payload_bytes:
            raise MediaPayloadLimitError(
                "full media context exceeds the configured payload limit"
            )
        return records

    def full_context(
        self,
        assets: Any,
        *,
        profile_id: str,
    ) -> list[dict[str, Any]]:
        """Build identical full-context media for Direct or W2-Light."""

        profile = str(profile_id).strip().lower().replace("-", "_")
        if profile not in {"direct", "w2_light"}:
            raise MediaTransportError("unsupported full-context harness profile")
        return [
            record.to_model_visible_dict()
            for record in self.full_context_records(assets)
        ]

    def direct_full_context(self, assets: Any) -> list[dict[str, Any]]:
        return self.full_context(assets, profile_id="direct")

    def light_full_context(self, assets: Any) -> list[dict[str, Any]]:
        return self.full_context(assets, profile_id="w2_light")

    def w2_light_full_context(self, assets: Any) -> list[dict[str, Any]]:
        return self.light_full_context(assets)

    def context_for_profile(
        self, profile_id: str, assets: Any
    ) -> list[dict[str, Any]]:
        return self.full_context(assets, profile_id=profile_id)

    def execute(
        self,
        helper: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> MediaTransportRecord | list[MediaTransportRecord]:
        """Dispatch one strict helper call without model/API dependencies."""

        name = str(helper).strip().upper()
        if name not in MEDIA_HELPERS:
            raise MediaTransportError("unknown media helper")
        if arguments is not None and not isinstance(arguments, Mapping):
            raise MediaTransportError("helper arguments must be an object")
        payload = dict(arguments or {})
        duplicate = set(payload).intersection(kwargs)
        if duplicate:
            raise MediaTransportError("duplicate helper arguments")
        payload.update(kwargs)
        aliases: dict[str, str] = {}
        if name == OPEN_ASSET:
            aliases = {
                "asset": "source",
                "asset_ref": "source",
                "source_ref": "source",
            }
        elif name == GET_VIEW:
            aliases = {"view": "view_id", "source_ref": "source"}
        elif name in {GET_FRAME, GET_FRAME_WINDOW}:
            aliases = {
                "asset": "source",
                "asset_ref": "source",
                "source_ref": "source",
                "video": "source",
                "video_ref": "source",
            }
            if name == GET_FRAME:
                aliases["index"] = "frame_index"
            else:
                aliases.update(
                    {
                        "start": "start_frame",
                        "start_index": "start_frame",
                        "end": "end_frame",
                        "end_index": "end_frame",
                        "stop": "end_frame",
                        "stride": "step",
                        "indices": "frame_indices",
                        "frame_count": "count",
                        "window_size": "count",
                    }
                )
        elif name in {CROP_REGION, ZOOM_REGION}:
            aliases = {
                "asset_ref": "asset",
                "source": "asset",
                "source_ref": "asset",
                "bbox": "region",
            }
        elif name == COMPOSE_ASSETS:
            aliases = {"asset_refs": "assets", "records": "assets"}
        for old, new in aliases.items():
            if old in payload:
                if new in payload:
                    raise MediaTransportError("duplicate helper argument aliases")
                payload[new] = payload.pop(old)
        dispatch: dict[str, Callable[..., Any]] = {
            OPEN_ASSET: self.open_asset,
            GET_VIEW: self.get_view,
            GET_FRAME: self.get_frame,
            GET_FRAME_WINDOW: self.get_frame_window,
            CROP_REGION: self.crop_region,
            ZOOM_REGION: self.zoom_region,
            COMPOSE_ASSETS: self.compose_assets,
        }
        try:
            return dispatch[name](**payload)
        except TypeError:
            raise MediaTransportError("media helper arguments are invalid") from None

    def execute_model_visible(
        self,
        helper: str,
        arguments: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        result = self.execute(helper, arguments, **kwargs)
        if isinstance(result, list):
            return [record.to_model_visible_dict() for record in result]
        return result.to_model_visible_dict()

    # Uppercase methods mirror the JSON action names used by W2-Light.
    def OPEN_ASSET(  # noqa: N802
        self, source: Any, **kwargs: Any
    ) -> MediaTransportRecord:
        result = self.execute(OPEN_ASSET, {"source": source}, **kwargs)
        assert isinstance(result, MediaTransportRecord)
        return result

    def GET_VIEW(  # noqa: N802
        self, view_id: str | Path, **kwargs: Any
    ) -> MediaTransportRecord:
        result = self.execute(GET_VIEW, {"view_id": view_id}, **kwargs)
        assert isinstance(result, MediaTransportRecord)
        return result

    def GET_FRAME(  # noqa: N802
        self,
        source: Any,
        frame_index: int | None = None,
        **kwargs: Any,
    ) -> MediaTransportRecord:
        payload: dict[str, Any] = {"source": source}
        if frame_index is not None:
            payload["frame_index"] = frame_index
        result = self.execute(GET_FRAME, payload, **kwargs)
        assert isinstance(result, MediaTransportRecord)
        return result

    def GET_FRAME_WINDOW(  # noqa: N802
        self, source: Any, **kwargs: Any
    ) -> list[MediaTransportRecord]:
        result = self.execute(GET_FRAME_WINDOW, {"source": source}, **kwargs)
        assert isinstance(result, list)
        return result

    def CROP_REGION(  # noqa: N802
        self, asset: Any, region: Any, **kwargs: Any
    ) -> MediaTransportRecord:
        result = self.execute(
            CROP_REGION, {"asset": asset, "region": region}, **kwargs
        )
        assert isinstance(result, MediaTransportRecord)
        return result

    def ZOOM_REGION(  # noqa: N802
        self, asset: Any, region: Any = None, **kwargs: Any
    ) -> MediaTransportRecord:
        payload = {"asset": asset}
        if region is not None:
            payload["region"] = region
        result = self.execute(ZOOM_REGION, payload, **kwargs)
        assert isinstance(result, MediaTransportRecord)
        return result

    def COMPOSE_ASSETS(  # noqa: N802
        self, assets: Sequence[Any], **kwargs: Any
    ) -> MediaTransportRecord:
        result = self.execute(COMPOSE_ASSETS, {"assets": assets}, **kwargs)
        assert isinstance(result, MediaTransportRecord)
        return result

    def _resolve_registered_source(
        self, source: Any
    ) -> tuple[Any, _SourceSpec | None]:
        if isinstance(source, str):
            spec = self._source_specs.get(source)
            if spec is not None:
                return spec.source, spec
            spec = self._view_specs.get(source)
            if spec is not None:
                return spec.source, spec
            if source in self._records:
                return self._records[source], None
        return source, None

    def _decode_image(
        self,
        source: Any,
        *,
        declared_mime: str | None,
        expected_sha256: str | None,
    ) -> _DecodedImage:
        if isinstance(source, MediaTransportRecord):
            record, raw = self._resolve_image_asset(source)
            image = self._decode_transport_image(raw, record)
            return _DecodedImage(image, record.source_sha256, record.mime_type)
        raw, data_url_mime = self._read_image_source(source)
        source_sha256 = hashlib.sha256(raw).hexdigest()
        _check_expected_hash(expected_sha256, source_sha256)
        normalized_declared = _normalize_declared_mime(declared_mime, image=True)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as probe:
                    source_format = str(probe.format or "").upper()
                    source_mime = _FORMAT_TO_MIME.get(source_format)
                    width, height = probe.size
                    frame_count = int(getattr(probe, "n_frames", 1))
                    probe.verify()
                if source_mime not in _IMAGE_MIMES:
                    raise MediaDecodeError("decoded image format is unsupported")
                if frame_count != 1:
                    raise MediaDecodeError("animated image sources are unsupported")
                if width < 1 or height < 1 or width * height > self.max_image_pixels:
                    raise MediaDecodeError("decoded image dimensions exceed limits")
                with Image.open(io.BytesIO(raw)) as decoded:
                    image = ImageOps.exif_transpose(decoded)
                    image.load()
                    image = image.copy()
        except MediaTransportError:
            raise
        except Exception:
            raise MediaDecodeError(
                "image source is corrupt or cannot be decoded"
            ) from None
        for supplied in (data_url_mime, normalized_declared):
            if supplied is not None and supplied != source_mime:
                raise MediaSourceError(
                    "declared image MIME does not match decoded image format"
                )
        if image.width * image.height > self.max_image_pixels:
            raise MediaDecodeError("oriented image dimensions exceed limits")
        return _DecodedImage(image, source_sha256, source_mime)

    def _read_image_source(self, source: Any) -> tuple[bytes, str | None]:
        data_url_mime: str | None = None
        if isinstance(source, str) and source.startswith("data:"):
            match = _DATA_URL.fullmatch(source)
            if match is None:
                raise MediaSourceError("image data URL must use strict base64 encoding")
            data_url_mime = _normalize_declared_mime(match.group(1), image=True)
            encoded = match.group(2)
            maximum = ((self.max_source_bytes + 2) // 3) * 4
            if len(encoded) > maximum:
                raise MediaSourceError("image source exceeds the configured byte limit")
            try:
                raw = base64.b64decode(encoded, validate=True)
            except (binascii.Error, ValueError):
                raise MediaSourceError(
                    "image data URL contains invalid base64"
                ) from None
        elif isinstance(source, (bytes, bytearray, memoryview)):
            raw = bytes(source)
        elif isinstance(source, (str, Path)):
            path = self._validated_path(source)
            size = path.stat().st_size
            if size < 1 or size > self.max_source_bytes:
                raise MediaSourceError("media source byte length is outside limits")
            try:
                raw = path.read_bytes()
            except OSError:
                raise MediaSourceError("media source cannot be read") from None
        else:
            raise MediaSourceError(
                "image source must be local bytes, data URL, or local path"
            )
        if not raw:
            raise MediaSourceError("image source must not be empty")
        if len(raw) > self.max_source_bytes:
            raise MediaSourceError("image source exceeds the configured byte limit")
        return raw, data_url_mime

    def _validated_path(self, source: str | Path) -> Path:
        text = str(source)
        if text.startswith(("http://", "https://", "ftp://")):
            raise MediaSourceError("remote media sources are forbidden")
        candidate = Path(source)
        if not candidate.is_absolute() and self.source_root is not None:
            candidate = self.source_root / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            raise MediaSourceError("local media source does not exist") from None
        if self.source_root is not None:
            try:
                resolved.relative_to(self.source_root)
            except ValueError:
                raise MediaSourceError("media source is outside source_root") from None
        if not resolved.is_file():
            raise MediaSourceError("local media source must be a regular file")
        return resolved

    def _open_video(
        self,
        source: Any,
        *,
        declared_mime: str | None,
        expected_sha256: str | None,
    ) -> _VideoSource:
        if isinstance(source, MediaTransportRecord):
            if source.media_kind != "video" or source.asset_id not in self._videos:
                raise MediaSourceError(
                    "video descriptor is not owned by this transport"
                )
            return self._videos[source.asset_id]
        if isinstance(source, str) and source in self._videos:
            return self._videos[source]
        if not isinstance(source, (str, Path)):
            raise MediaSourceError("video sources must be local files")
        path = self._validated_path(source)
        initial_signature = _file_signature(path)
        size = initial_signature[0]
        if size < 1 or size > self.max_source_bytes:
            raise MediaSourceError("video source byte length is outside limits")
        source_sha256 = _file_sha256(path)
        _check_expected_hash(expected_sha256, source_sha256)
        with self._lock:
            existing = self._videos_by_hash.get(source_sha256)
            if existing is not None:
                return replace(
                    existing,
                    record=replace(existing.record, cache_hit=True),
                )

        mime_type = _VIDEO_SUFFIX_TO_MIME.get(path.suffix.lower())
        if mime_type is None:
            guessed, _ = mimetypes.guess_type(path.name)
            if not isinstance(guessed, str) or not guessed.startswith("video/"):
                raise MediaSourceError("video container type is unsupported")
            mime_type = guessed
        normalized_declared = _normalize_declared_mime(declared_mime, image=False)
        if normalized_declared is not None and normalized_declared != mime_type:
            raise MediaSourceError("declared video MIME does not match its container")
        cv2 = _opencv()
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise MediaDecodeError(
                    "video source cannot be opened by the CPU decoder"
                )
            frame_count = _capture_int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = _capture_int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = _capture_int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_value = float(capture.get(cv2.CAP_PROP_FPS))
            if (
                frame_count < 1
                or width < 1
                or height < 1
                or width * height > self.max_image_pixels
                or not math.isfinite(fps_value)
                or fps_value <= 0.0
            ):
                raise MediaDecodeError("video metadata is incomplete or outside limits")
            ok, first = capture.read()
            if not ok or first is None or getattr(first, "size", 0) == 0:
                raise MediaDecodeError("video source has no decodable first frame")
        finally:
            capture.release()
        if _file_signature(path) != initial_signature:
            raise MediaSourceError("video source changed while it was decoded")
        fps = Fraction(str(fps_value)).limit_denominator(1_000_000)
        duration_ms = _timestamp_ms(frame_count - 1, fps)
        derivation_sha256 = _canonical_sha256(
            {
                "operation": OPEN_ASSET,
                "media_kind": "video",
                "source_sha256": source_sha256,
                "frame_count": frame_count,
                "fps": [fps.numerator, fps.denominator],
            }
        )
        asset_id = f"video-{source_sha256[:32]}"
        record = MediaTransportRecord(
            operation=OPEN_ASSET,
            asset_id=asset_id,
            media_kind="video",
            mime_type=mime_type,
            width=width,
            height=height,
            byte_length=size,
            payload_bytes=0,
            source_sha256=source_sha256,
            content_sha256=source_sha256,
            derivation_sha256=derivation_sha256,
            cache_hit=False,
            frame_count=frame_count,
            duration_ms=duration_ms,
            fps_numerator=fps.numerator,
            fps_denominator=fps.denominator,
        )
        video = _VideoSource(
            path=path,
            source_sha256=source_sha256,
            mime_type=mime_type,
            byte_length=size,
            width=width,
            height=height,
            frame_count=frame_count,
            fps=fps,
            file_signature=initial_signature,
            record=record,
        )
        with self._lock:
            self._videos[asset_id] = video
            self._videos_by_hash[source_sha256] = video
            self._records[asset_id] = record
        return video

    def _as_video(self, source: Any) -> _VideoSource:
        if isinstance(source, str) and source in self._records:
            source = self._records[source]
        if isinstance(source, MediaTransportRecord):
            video = self._videos.get(source.asset_id)
            if source.media_kind != "video" or video is None:
                raise MediaSourceError("GET_FRAME requires a local video descriptor")
            return video
        source, spec = self._resolve_registered_source(source)
        return self._open_video(
            source,
            declared_mime=None if spec is None else spec.declared_mime,
            expected_sha256=None if spec is None else spec.expected_sha256,
        )

    def _decode_video_frame(
        self, video: _VideoSource, frame_index: int
    ) -> Image.Image:
        if _file_signature(video.path) != video.file_signature:
            raise MediaFrameError("video source changed after it was opened")
        cv2 = _opencv()
        capture = cv2.VideoCapture(str(video.path))
        try:
            if not capture.isOpened():
                raise MediaFrameError("video source cannot be reopened")
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None or getattr(frame, "size", 0) == 0:
                raise MediaFrameError("requested video frame cannot be decoded")
            next_position = float(capture.get(cv2.CAP_PROP_POS_FRAMES))
            if (
                math.isfinite(next_position)
                and next_position > 0.0
                and abs(next_position - (frame_index + 1)) > 0.5
            ):
                raise MediaFrameError(
                    "video decoder did not return the requested frame identity"
                )
            if len(frame.shape) == 2:
                rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif frame.shape[2] == 4:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.load()
            if _file_signature(video.path) != video.file_signature:
                raise MediaFrameError("video source changed while a frame was decoded")
            return image
        except MediaTransportError:
            raise
        except Exception:
            raise MediaFrameError("requested video frame cannot be decoded") from None
        finally:
            capture.release()

    def _resolve_image_asset(
        self, asset: Any
    ) -> tuple[MediaTransportRecord, bytes]:
        if isinstance(asset, str) and asset in self._records:
            record = self._records[asset]
        elif isinstance(asset, MediaTransportRecord):
            record = asset
        elif isinstance(asset, Mapping):
            asset_id = asset.get("asset_id")
            if isinstance(asset_id, str) and asset_id in self._records:
                record = self._records[asset_id]
            elif isinstance(asset.get("data_url"), str):
                record = self.open_asset(
                    asset["data_url"],
                    media_kind=str(asset.get("media_kind", "auto")),
                    expected_sha256=asset.get("content_sha256"),
                )
            else:
                raise MediaSourceError("unknown transported image asset")
        else:
            record = self.open_asset(asset)
        if record.media_kind == "video" or record.data_url is None:
            raise MediaSourceError("operation requires a decoded image asset")
        raw = self._raw_assets.get(record.asset_id)
        if raw is None:
            match = _DATA_URL.fullmatch(record.data_url)
            if match is None:
                raise MediaDecodeError("transported image payload is malformed")
            try:
                raw = base64.b64decode(match.group(2), validate=True)
            except (binascii.Error, ValueError):
                raise MediaDecodeError(
                    "transported image payload is malformed"
                ) from None
        if hashlib.sha256(raw).hexdigest() != record.content_sha256:
            raise MediaDecodeError("transported image content hash mismatch")
        return record, raw

    def _decode_transport_image(
        self, raw: bytes, record: MediaTransportRecord
    ) -> Image.Image:
        try:
            with Image.open(io.BytesIO(raw)) as decoded:
                if decoded.size != (record.width, record.height):
                    raise MediaDecodeError("transported image dimensions mismatch")
                decoded.load()
                return decoded.copy()
        except MediaTransportError:
            raise
        except Exception:
            raise MediaDecodeError("transported image cannot be decoded") from None

    def _materialize_image(
        self,
        *,
        operation: str,
        media_kind: str,
        source_sha256: str,
        parent_asset_ids: tuple[str, ...],
        derivation: Mapping[str, Any],
        image_factory: Callable[[], Image.Image],
        view_id: str | None = None,
        frame_id: str | None = None,
        frame_index: int | None = None,
        timestamp_ms: int | None = None,
        frame_count: int | None = None,
        duration_ms: int | None = None,
        fps_numerator: int | None = None,
        fps_denominator: int | None = None,
    ) -> MediaTransportRecord:
        mime_type = "image/jpeg" if media_kind == "rgb" else "image/png"
        derivation_sha256 = _canonical_sha256(
            {
                "encoder_contract": _ENCODER_CONTRACT,
                "jpeg_quality": self.jpeg_quality,
                "operation": operation,
                "output_mime": mime_type,
                "derivation": derivation,
            }
        )
        with self._lock:
            cached = self._read_cache(derivation_sha256, mime_type)
            cache_hit = cached is not None
            if cached is None:
                image = image_factory()
                if image.width * image.height > self.max_image_pixels:
                    raise MediaTransportError(
                        "derived image exceeds the configured pixel limit"
                    )
                raw = self._encode_image(image, media_kind)
                width, height = image.size
                content_sha256 = hashlib.sha256(raw).hexdigest()
                data_url = _data_url(raw, mime_type)
                self._check_payload(data_url)
                metadata = {
                    "schema_version": _CACHE_SCHEMA_VERSION,
                    "derivation_sha256": derivation_sha256,
                    "content_sha256": content_sha256,
                    "mime_type": mime_type,
                    "width": width,
                    "height": height,
                    "byte_length": len(raw),
                }
                self._write_cache(derivation_sha256, raw, metadata)
            else:
                raw, metadata = cached
                width = int(metadata["width"])
                height = int(metadata["height"])
                content_sha256 = str(metadata["content_sha256"])
                data_url = _data_url(raw, mime_type)
                self._check_payload(data_url)

            prefix = _OPERATION_PREFIX[operation]
            asset_id = f"{prefix}-{derivation_sha256[:32]}"
            record = MediaTransportRecord(
                operation=operation,
                asset_id=asset_id,
                media_kind=media_kind,
                mime_type=mime_type,
                width=width,
                height=height,
                byte_length=len(raw),
                payload_bytes=len(data_url.encode("ascii")),
                source_sha256=source_sha256,
                content_sha256=content_sha256,
                derivation_sha256=derivation_sha256,
                parent_asset_ids=parent_asset_ids,
                data_url=data_url,
                cache_hit=cache_hit,
                view_id=view_id,
                frame_id=frame_id,
                frame_index=frame_index,
                timestamp_ms=timestamp_ms,
                frame_count=frame_count,
                duration_ms=duration_ms,
                fps_numerator=fps_numerator,
                fps_denominator=fps_denominator,
            )
            self._records[asset_id] = record
            self._raw_assets[asset_id] = raw
            return record

    def _encode_image(self, image: Image.Image, media_kind: str) -> bytes:
        stream = io.BytesIO()
        if media_kind == "rgb":
            rgb = _prepare_image(image, "rgb")
            rgb.save(
                stream,
                format="JPEG",
                quality=self.jpeg_quality,
                subsampling=0,
                optimize=False,
                progressive=False,
            )
        else:
            png = _prepare_image(image, media_kind)
            png.save(
                stream,
                format="PNG",
                optimize=False,
                compress_level=9,
            )
        raw = stream.getvalue()
        if not raw:
            raise MediaDecodeError("derived image encoder returned no bytes")
        return raw

    def _check_payload(self, data_url: str) -> None:
        if len(data_url.encode("ascii")) > self.max_payload_bytes:
            raise MediaPayloadLimitError(
                "media payload exceeds the configured byte limit"
            )

    def _output_size(
        self,
        *,
        max_size: int | Sequence[int] | None,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> tuple[int, int]:
        size = (
            self.default_max_size
            if max_size is None
            else _size_pair(max_size, "max_size")
        )
        if max_width is not None:
            size = (_positive_int(max_width, "max_width"), size[1])
        if max_height is not None:
            size = (size[0], _positive_int(max_height, "max_height"))
        return size

    def _cache_paths(self, key: str, mime_type: str) -> tuple[Path, Path]:
        if self.cache_dir is None:
            raise MediaTransportError("persistent cache is not configured")
        suffix = ".jpg" if mime_type == "image/jpeg" else ".png"
        base = self.cache_dir / "derived" / key
        return base.with_suffix(suffix), base.with_suffix(".json")

    def _read_cache(
        self, key: str, mime_type: str
    ) -> tuple[bytes, dict[str, Any]] | None:
        memory = self._memory_cache.get(key)
        if memory is not None and self._valid_cache_entry(
            key, mime_type, memory[0], memory[1]
        ):
            return memory
        if self.cache_dir is None:
            return None
        data_path, metadata_path = self._cache_paths(key, mime_type)
        try:
            raw = data_path.read_bytes()
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(metadata, dict) or not self._valid_cache_entry(
            key, mime_type, raw, metadata
        ):
            return None
        self._memory_cache[key] = (raw, metadata)
        return raw, metadata

    def _valid_cache_entry(
        self,
        key: str,
        mime_type: str,
        raw: bytes,
        metadata: Mapping[str, Any],
    ) -> bool:
        if (
            metadata.get("schema_version") != _CACHE_SCHEMA_VERSION
            or metadata.get("derivation_sha256") != key
            or metadata.get("mime_type") != mime_type
            or metadata.get("byte_length") != len(raw)
            or metadata.get("content_sha256")
            != hashlib.sha256(raw).hexdigest()
        ):
            return False
        try:
            with Image.open(io.BytesIO(raw)) as image:
                expected_format = "JPEG" if mime_type == "image/jpeg" else "PNG"
                if str(image.format or "").upper() != expected_format:
                    return False
                if [image.width, image.height] != [
                    metadata.get("width"),
                    metadata.get("height"),
                ]:
                    return False
                image.verify()
        except Exception:
            return False
        return True

    def _write_cache(
        self, key: str, raw: bytes, metadata: dict[str, Any]
    ) -> None:
        self._memory_cache[key] = (raw, dict(metadata))
        if self.cache_dir is None:
            return
        data_path, metadata_path = self._cache_paths(
            key, str(metadata["mime_type"])
        )
        _atomic_write(data_path, raw)
        metadata_bytes = (
            json.dumps(
                metadata,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        _atomic_write(metadata_path, metadata_bytes)


def _source_spec(
    source: Any,
    *,
    media_kind: str,
    declared_mime: str | None,
    expected_sha256: str | None,
) -> _SourceSpec:
    if isinstance(source, Mapping) and "source" in source:
        media_kind = str(source.get("media_kind", media_kind))
        declared_mime = source.get("declared_mime", declared_mime)
        expected_sha256 = source.get("expected_sha256", expected_sha256)
        source = source["source"]
    _normalize_kind(media_kind, source=source)
    if declared_mime is not None and not isinstance(declared_mime, str):
        raise MediaSourceError("declared_mime must be a string")
    if expected_sha256 is not None:
        _validated_sha256(expected_sha256, "expected_sha256")
    return _SourceSpec(source, media_kind, declared_mime, expected_sha256)


def _normalize_kind(value: Any, *, source: Any = None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MediaSourceError("media_kind must be a non-empty string")
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "image": "rgb",
        "photo": "rgb",
        "photograph": "rgb",
        "rgb_image": "rgb",
        "segmentation": "mask",
        "segmentation_mask": "mask",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized == "auto":
        if isinstance(source, (str, Path)) and not str(source).startswith("data:"):
            if Path(source).suffix.lower() in _VIDEO_SUFFIX_TO_MIME:
                return "video"
        return "rgb"
    if normalized not in {"rgb", "mask", "diagram", "video"}:
        raise MediaSourceError("media_kind is unsupported")
    return normalized


def _prepare_image(image: Image.Image, media_kind: str) -> Image.Image:
    if media_kind == "rgb":
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image.convert("RGB")
    if media_kind == "mask":
        return image.convert("L")
    if media_kind == "diagram":
        if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
            return image.convert("RGBA")
        return image.convert("RGB")
    raise MediaTransportError("image encoder received an invalid media kind")


def _resize_to_fit(
    image: Image.Image,
    bounds: tuple[int, int],
    *,
    allow_upscale: bool,
    media_kind: str,
) -> Image.Image:
    target = _fit_size(image.size, bounds, allow_upscale=allow_upscale)
    if target == image.size:
        return image.copy()
    resampling = getattr(Image, "Resampling", Image)
    method = resampling.NEAREST if media_kind == "mask" else resampling.LANCZOS
    return image.resize(target, resample=method)


def _fit_size(
    source: tuple[int, int],
    bounds: tuple[int, int],
    *,
    allow_upscale: bool,
) -> tuple[int, int]:
    width, height = source
    max_width, max_height = bounds
    if not allow_upscale and width <= max_width and height <= max_height:
        return width, height
    if width * max_height >= height * max_width:
        target_width = max_width
        target_height = max(1, (height * max_width + width // 2) // width)
    else:
        target_height = max_height
        target_width = max(1, (width * max_height + height // 2) // height)
    if not allow_upscale:
        target_width = min(width, target_width)
        target_height = min(height, target_height)
    return target_width, target_height


def _pixel_box(
    region: Mapping[str, Any] | Sequence[float],
    image_size: tuple[int, int],
    *,
    coordinate_space: str,
    bbox_format: str,
) -> tuple[int, int, int, int]:
    bbox_format = str(bbox_format).strip().lower()
    if isinstance(region, Mapping):
        if {"x_min", "y_min", "x_max", "y_max"} <= set(region):
            values = [
                region["x_min"],
                region["y_min"],
                region["x_max"],
                region["y_max"],
            ]
            bbox_format = "xyxy"
        elif {"left", "top", "right", "bottom"} <= set(region):
            values = [
                region["left"],
                region["top"],
                region["right"],
                region["bottom"],
            ]
            bbox_format = "xyxy"
        elif {"x", "y", "width", "height"} <= set(region):
            values = [
                region["x"],
                region["y"],
                region["width"],
                region["height"],
            ]
            bbox_format = "xywh"
        else:
            raise MediaTransportError("region object has no supported bounding box")
    elif isinstance(region, Sequence) and not isinstance(region, (str, bytes)):
        if len(region) != 4:
            raise MediaTransportError("region must contain four coordinates")
        values = list(region)
    else:
        raise MediaTransportError("region must be an object or four coordinates")
    numbers: list[float] = []
    for value in values:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise MediaTransportError("region coordinates must be finite numbers")
        numbers.append(float(value))
    if bbox_format == "xywh":
        left, top, width, height = numbers
        right, bottom = left + width, top + height
    elif bbox_format == "xyxy":
        left, top, right, bottom = numbers
    else:
        raise MediaTransportError("bbox_format must be xyxy or xywh")
    image_width, image_height = image_size
    space = str(coordinate_space).strip().lower()
    if space in {"normalized", "normalized_0_1"}:
        if not (0.0 <= left < right <= 1.0 and 0.0 <= top < bottom <= 1.0):
            raise MediaTransportError("normalized region is outside image bounds")
        pixel_left = math.floor(left * image_width)
        pixel_top = math.floor(top * image_height)
        pixel_right = math.ceil(right * image_width)
        pixel_bottom = math.ceil(bottom * image_height)
    elif space in {"pixel", "pixels"}:
        if not (
            0.0 <= left < right <= image_width
            and 0.0 <= top < bottom <= image_height
        ):
            raise MediaTransportError("pixel region is outside image bounds")
        if any(not value.is_integer() for value in (left, top, right, bottom)):
            raise MediaTransportError("pixel region coordinates must be integers")
        pixel_left, pixel_top, pixel_right, pixel_bottom = (
            int(left),
            int(top),
            int(right),
            int(bottom),
        )
    else:
        raise MediaTransportError("coordinate_space is unsupported")
    pixel_right = min(image_width, max(pixel_left + 1, pixel_right))
    pixel_bottom = min(image_height, max(pixel_top + 1, pixel_bottom))
    return pixel_left, pixel_top, pixel_right, pixel_bottom


def _center_zoom_box(
    image_size: tuple[int, int],
    *,
    center: Sequence[float] | None,
    zoom_factor: float,
    coordinate_space: str,
) -> tuple[int, int, int, int]:
    width, height = image_size
    if center is None:
        center_x, center_y = width / 2.0, height / 2.0
    else:
        if isinstance(center, (str, bytes)) or len(center) != 2:
            raise MediaTransportError("center must contain two coordinates")
        raw_x, raw_y = center
        if (
            isinstance(raw_x, bool)
            or isinstance(raw_y, bool)
            or not isinstance(raw_x, (int, float))
            or not isinstance(raw_y, (int, float))
            or not math.isfinite(float(raw_x))
            or not math.isfinite(float(raw_y))
        ):
            raise MediaTransportError("center coordinates must be finite numbers")
        center_x, center_y = float(raw_x), float(raw_y)
        space = str(coordinate_space).strip().lower()
        if space in {"normalized", "normalized_0_1"}:
            if not 0.0 <= center_x <= 1.0 or not 0.0 <= center_y <= 1.0:
                raise MediaTransportError("normalized center is outside image bounds")
            center_x *= width
            center_y *= height
        elif space not in {"pixel", "pixels"}:
            raise MediaTransportError("coordinate_space is unsupported")
    crop_width = max(1, min(width, _round_half_up(width / zoom_factor)))
    crop_height = max(1, min(height, _round_half_up(height / zoom_factor)))
    left = min(width - crop_width, max(0, _round_half_up(center_x - crop_width / 2)))
    top = min(height - crop_height, max(0, _round_half_up(center_y - crop_height / 2)))
    return left, top, left + crop_width, top + crop_height


def _composition_geometry(
    sizes: Sequence[tuple[int, int]],
    *,
    layout: str,
    columns: int,
    gap: int,
) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    if layout == "horizontal":
        canvas = (
            sum(width for width, _ in sizes) + gap * (len(sizes) - 1),
            max(height for _, height in sizes),
        )
        positions: list[tuple[int, int]] = []
        cursor = 0
        for width, height in sizes:
            positions.append((cursor, (canvas[1] - height) // 2))
            cursor += width + gap
        return canvas, positions
    if layout == "vertical":
        canvas = (
            max(width for width, _ in sizes),
            sum(height for _, height in sizes) + gap * (len(sizes) - 1),
        )
        positions = []
        cursor = 0
        for width, height in sizes:
            positions.append(((canvas[0] - width) // 2, cursor))
            cursor += height + gap
        return canvas, positions
    columns = min(columns, len(sizes))
    rows = math.ceil(len(sizes) / columns)
    cell_width = max(width for width, _ in sizes)
    cell_height = max(height for _, height in sizes)
    canvas = (
        columns * cell_width + gap * (columns - 1),
        rows * cell_height + gap * (rows - 1),
    )
    positions = []
    for index, (width, height) in enumerate(sizes):
        row, column = divmod(index, columns)
        positions.append(
            (
                column * (cell_width + gap) + (cell_width - width) // 2,
                row * (cell_height + gap) + (cell_height - height) // 2,
            )
        )
    return canvas, positions


def _representative_indices(frame_count: int) -> tuple[int, ...]:
    if frame_count == 1:
        return (0,)
    if frame_count == 2:
        return (0, 1)
    return (0, frame_count // 2, frame_count - 1)


def _timestamp_ms(frame_index: int, fps: Fraction) -> int:
    numerator = frame_index * 1000 * fps.denominator
    return (2 * numerator + fps.numerator) // (2 * fps.numerator)


def _frame_for_timestamp(timestamp_ms: int, fps: Fraction) -> int:
    numerator = timestamp_ms * fps.numerator
    denominator = 1000 * fps.denominator
    return (2 * numerator + denominator) // (2 * denominator)


def _capture_int(value: Any) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(number) or number < 0:
        return 0
    return int(round(number))


def _opencv() -> Any:
    try:
        import cv2
    except ImportError:
        raise MediaDecodeError(
            "OpenCV is required for CPU video frame decoding"
        ) from None
    return cv2


def _normalize_declared_mime(value: Any, *, image: bool) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise MediaSourceError("declared MIME must be a non-empty string")
    normalized = value.strip().lower()
    if normalized == "image/jpg":
        normalized = "image/jpeg"
    if image and normalized not in _IMAGE_MIMES:
        raise MediaSourceError("declared image MIME is unsupported")
    if not image and not normalized.startswith("video/"):
        raise MediaSourceError("declared video MIME is unsupported")
    return normalized


def _check_expected_hash(expected: str | None, actual: str) -> None:
    if expected is None:
        return
    if _validated_sha256(expected, "expected_sha256") != actual:
        raise MediaSourceError("media source does not match expected_sha256")


def _validated_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MediaSourceError(f"{field_name} must be lowercase SHA-256")
    return value


def _canonical_sha256(value: Any) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise MediaTransportError("media derivation values must be JSON-safe") from None
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        raise MediaSourceError("media source cannot be read") from None
    return digest.hexdigest()


def _file_signature(path: Path) -> tuple[int, int]:
    try:
        metadata = path.stat()
    except OSError:
        raise MediaSourceError("media source cannot be inspected") from None
    return metadata.st_size, metadata.st_mtime_ns


def _data_url(raw: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MediaTransportError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MediaTransportError(f"{field_name} must be a non-negative integer")
    return value


def _size_pair(value: int | Sequence[int], field_name: str) -> tuple[int, int]:
    if isinstance(value, int) and not isinstance(value, bool):
        width = height = _positive_int(value, field_name)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 2:
            raise MediaTransportError(f"{field_name} must contain width and height")
        width = _positive_int(value[0], f"{field_name}.width")
        height = _positive_int(value[1], f"{field_name}.height")
    else:
        raise MediaTransportError(f"{field_name} must be an integer or size pair")
    return width, height


def _rgb(value: Sequence[int]) -> tuple[int, int, int]:
    if (
        isinstance(value, (str, bytes))
        or not isinstance(value, Sequence)
        or len(value) != 3
    ):
        raise MediaTransportError("background must contain three RGB values")
    result: list[int] = []
    for channel in value:
        if (
            isinstance(channel, bool)
            or not isinstance(channel, int)
            or not 0 <= channel <= 255
        ):
            raise MediaTransportError(
                "background RGB values must be integers from 0 to 255"
            )
        result.append(channel)
    return result[0], result[1], result[2]


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def _public_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not _safe_public_id(value.strip()):
        raise MediaSourceError(f"{field_name} must be an opaque identifier")
    return value.strip()


def _safe_public_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 160
        and value not in {".", ".."}
        and not _absolute_like(value)
        and "/" not in value
        and "\\" not in value
        and all(ord(character) >= 32 for character in value)
    )


def _absolute_like(value: str) -> bool:
    return value.startswith(("/", "\\\\")) or _WINDOWS_ABSOLUTE.match(value) is not None


def _assert_model_visible(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if "path" in str(key).casefold():
                raise MediaTransportError("model-visible records cannot contain paths")
            _assert_model_visible(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            _assert_model_visible(item)
    elif (
        isinstance(value, str)
        and not value.startswith("data:")
        and _absolute_like(value)
    ):
        raise MediaTransportError("model-visible records cannot contain absolute paths")


__all__ = [
    "COMPOSE_ASSETS",
    "CROP_REGION",
    "GET_FRAME",
    "GET_FRAME_WINDOW",
    "GET_VIEW",
    "MEDIA_HELPERS",
    "MEDIA_TRANSPORT_SCHEMA_VERSION",
    "OPEN_ASSET",
    "ZOOM_REGION",
    "MediaDecodeError",
    "MediaFrameError",
    "MediaPayloadLimitError",
    "MediaSourceError",
    "MediaTransport",
    "MediaTransportError",
    "MediaTransportRecord",
]
