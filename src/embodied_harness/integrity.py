"""Shared canonical hashing and immutable JSON persistence primitives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


class IntegrityError(ValueError):
    """Raised when evidence is malformed, mutable, or digest-inconsistent."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the single canonical byte representation used by all digests."""

    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise IntegrityError(f"value is not canonical-JSON serializable: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path | str) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def parse_json_object(
    data: bytes | str,
    *,
    label: str = "JSON evidence",
    require_canonical: bool = False,
) -> dict[str, Any]:
    raw = data.encode("utf-8") if isinstance(data, str) else data
    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IntegrityError(f"invalid {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntegrityError(f"{label} must be an object")
    if require_canonical and raw != canonical_json_bytes(payload):
        raise IntegrityError(f"{label} is not canonical JSON")
    return payload


def load_json_object(path: Path | str, *, require_canonical: bool = False) -> dict[str, Any]:
    source = Path(path)
    try:
        data = source.read_bytes()
    except FileNotFoundError as exc:
        raise IntegrityError(f"missing JSON evidence: {source}") from exc
    return parse_json_object(data, label=f"JSON evidence {source}", require_canonical=require_canonical)


def immutable_write_json(path: Path | str, payload: dict[str, Any]) -> str:
    """Create one immutable JSON object and fail if the target already exists."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_json_bytes(payload)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o444)
    except FileExistsError as exc:
        raise IntegrityError(f"immutable evidence already exists: {target}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        try:
            target.unlink()
        except OSError:
            pass
        raise
    try:
        directory_fd = os.open(target.parent, os.O_RDONLY)
    except OSError:
        directory_fd = None
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    return sha256_bytes(data)


__all__ = [
    "IntegrityError",
    "canonical_json_bytes",
    "immutable_write_json",
    "load_json_object",
    "parse_json_object",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
]
