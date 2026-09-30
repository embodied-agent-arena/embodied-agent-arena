"""Deterministic local artifact encoding for optional perception outputs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

from .contracts import (
    ArtifactReference,
    BackendOutputError,
    PerceptionInputError,
    canonical_json_bytes,
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_local_path(path: str | Path) -> str:
    """Fingerprint a local file or a complete local model directory."""

    local = Path(path).expanduser()
    if local.is_file():
        return sha256_file(local)
    if not local.is_dir():
        raise PerceptionInputError("model must be an existing local file or directory")
    root = local.resolve()
    manifest: list[dict[str, str | int]] = []
    for candidate in sorted(root.rglob("*")):
        if not candidate.is_file():
            continue
        # Hugging Face snapshot directories commonly contain symlinks into the
        # adjacent local blob cache.  Hash the resolved bytes while retaining
        # the stable path visible beneath the caller-supplied directory.
        relative = candidate.relative_to(root).as_posix()
        manifest.append(
            {
                "path": relative,
                "sha256": sha256_file(candidate),
                "size": candidate.stat().st_size,
            }
        )
    if not manifest:
        raise PerceptionInputError("local model directory contains no files")
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


@dataclass(frozen=True)
class ArtifactStore:
    """Write immutable content-addressed files beneath one explicit directory."""

    root: Path
    reference_prefix: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser().resolve()
        object.__setattr__(self, "root", root)
        if self.reference_prefix is not None:
            prefix = self.reference_prefix.strip().replace("\\", "/")
            pure = PurePosixPath(prefix)
            if not prefix or pure.is_absolute() or ".." in pure.parts:
                raise PerceptionInputError(
                    "artifact reference_prefix must be a safe relative path"
                )
            object.__setattr__(self, "reference_prefix", prefix.rstrip("/"))

    def write(
        self,
        payload: bytes,
        *,
        kind: str,
        representation: str,
        shape: tuple[int, ...],
        dtype: str,
        suffix: str,
    ) -> ArtifactReference:
        if not isinstance(payload, bytes):
            raise PerceptionInputError("artifact payload must be bytes")
        if not suffix.startswith(".") or "/" in suffix or "\\" in suffix:
            raise PerceptionInputError("artifact suffix is invalid")
        digest = hashlib.sha256(payload).hexdigest()
        filename = f"{kind}-{digest}{suffix}"
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self.root / filename
        if destination.exists():
            if sha256_file(destination) != digest:
                raise BackendOutputError("artifact hash collision detected")
        else:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=self.root
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, destination)
            finally:
                try:
                    Path(temporary_name).unlink()
                except FileNotFoundError:
                    pass
        reference = (
            filename
            if self.reference_prefix is None
            else f"{self.reference_prefix}/{filename}"
        )
        return ArtifactReference(
            kind=kind,
            representation=representation,
            sha256=digest,
            shape=shape,
            dtype=dtype,
            reference=reference,
        )


def array_to_python(value: Any) -> Any:
    """Convert tensor/ndarray-like objects without importing their frameworks."""

    current = value
    for method_name in ("detach", "cpu"):
        method = getattr(current, method_name, None)
        if callable(method):
            current = method()
    method = getattr(current, "tolist", None)
    if callable(method):
        current = method()
    return current


def rows_2d(value: Any, *, field_name: str) -> tuple[tuple[Any, ...], ...]:
    converted = array_to_python(value)
    if isinstance(converted, (str, bytes, bytearray)) or not isinstance(
        converted, Sequence
    ):
        raise BackendOutputError(f"{field_name} must be a two-dimensional array")
    rows: list[tuple[Any, ...]] = []
    for row in converted:
        if isinstance(row, (str, bytes, bytearray)) or not isinstance(row, Sequence):
            raise BackendOutputError(f"{field_name} must be a two-dimensional array")
        rows.append(tuple(row))
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise BackendOutputError(f"{field_name} must be a non-empty rectangular array")
    return tuple(rows)


def encode_relative_depth(
    value: Any,
) -> tuple[bytes, bytes, tuple[int, int], int]:
    """Encode raw relative output as f32le and a byte-per-pixel valid mask.

    Non-finite or float32-overflowing cells are zero-filled in the depth artifact
    and marked invalid.  The valid mask therefore remains the authority.
    """

    rows = rows_2d(value, field_name="predicted_depth")
    depth = bytearray()
    valid = bytearray()
    valid_count = 0
    for row in rows:
        for raw in row:
            try:
                number = float(raw)
                packed = struct.pack("<f", number)
                finite = math.isfinite(number)
            except (TypeError, ValueError, OverflowError, struct.error):
                finite = False
                packed = struct.pack("<f", 0.0)
            if finite:
                valid.append(1)
                valid_count += 1
                depth.extend(packed)
            else:
                valid.append(0)
                depth.extend(struct.pack("<f", 0.0))
    return bytes(depth), bytes(valid), (len(rows), len(rows[0])), valid_count


def _point(value: Any) -> tuple[float, float]:
    converted = array_to_python(value)
    if isinstance(converted, (str, bytes, bytearray)) or not isinstance(
        converted, Sequence
    ) or len(converted) != 2:
        raise BackendOutputError("mask polygon point must contain x and y")
    try:
        x, y = float(converted[0]), float(converted[1])
    except (TypeError, ValueError, OverflowError):
        raise BackendOutputError("mask polygon coordinates must be numeric") from None
    if not math.isfinite(x) or not math.isfinite(y):
        raise BackendOutputError("mask polygon coordinates must be finite")
    return x, y


def encode_original_polygon_mask(
    polygons: Any,
    *,
    width: int,
    height: int,
) -> bytes:
    """Encode one or more original-pixel polygons as canonical JSON."""

    converted = array_to_python(polygons)
    if isinstance(converted, (str, bytes, bytearray)) or not isinstance(
        converted, Sequence
    ):
        raise BackendOutputError("mask polygons must be a sequence")
    # Ultralytics exposes one Nx2 array per instance.  Accept a direct Nx2
    # polygon and a sequence of polygons so fake arrays can exercise both forms.
    candidates: Iterable[Any]
    if converted and _looks_like_point(converted[0]):
        candidates = (converted,)
    else:
        candidates = converted
    normalized: list[list[list[float]]] = []
    for polygon in candidates:
        polygon_value = array_to_python(polygon)
        if isinstance(polygon_value, (str, bytes, bytearray)) or not isinstance(
            polygon_value, Sequence
        ):
            raise BackendOutputError("mask polygon must be a sequence of points")
        points: list[list[float]] = []
        for raw_point in polygon_value:
            x, y = _point(raw_point)
            if x < 0 or y < 0 or x > width or y > height:
                raise BackendOutputError(
                    "mask polygon exceeds original image coordinates"
                )
            points.append([x, y])
        if len(points) < 3:
            raise BackendOutputError("mask polygon must contain at least three points")
        normalized.append(points)
    if not normalized:
        raise BackendOutputError("mask polygon payload is empty")
    return canonical_json_bytes(
        {
            "coordinate_system": "pixel_polygon_original",
            "original_dimensions": {"width": width, "height": height},
            "polygons": normalized,
        }
    )


def _looks_like_point(value: Any) -> bool:
    converted = array_to_python(value)
    return (
        isinstance(converted, Sequence)
        and not isinstance(converted, (str, bytes, bytearray))
        and len(converted) == 2
        and all(not isinstance(item, Sequence) for item in converted)
    )


def write_canonical_json(path: str | Path, value: Any) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        try:
            Path(temporary_name).unlink()
        except FileNotFoundError:
            pass


__all__ = [
    "ArtifactStore",
    "array_to_python",
    "encode_original_polygon_mask",
    "encode_relative_depth",
    "fingerprint_local_path",
    "rows_2d",
    "sha256_file",
    "write_canonical_json",
]
