"""CPU-only real-data adapters for MMSI-Bench and MindCube."""

from __future__ import annotations

import ast
import base64
import binascii
import csv
import json
import re
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from .base import (
    AdapterError,
    AdapterSample,
    BaseAdapter,
    Choice,
    DataUnavailableError,
    SampleNotFoundError,
    register_adapter,
)


_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".webp"})


def _source_id(value: str | int | Mapping[str, Any] | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in ("source_sample_id", "sample_id", "id", "index"):
            candidate = value.get(key)
            if candidate is not None and str(candidate).strip():
                return str(candidate).strip()
        raise SampleNotFoundError("sample selector has no source sample ID")
    candidate = str(value).strip()
    if not candidate:
        raise SampleNotFoundError("sample ID must be non-empty")
    return candidate


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(piece) if piece.isdigit() else piece.casefold()
        for piece in re.split(r"(\d+)", value)
    )


def _ordered_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path.resolve()
            for path in directory.iterdir()
            if path.is_file()
            and path.suffix.casefold() in _IMAGE_SUFFIXES
            and path.stat().st_size > 0
        ),
        key=lambda path: _natural_key(path.name),
    )


def _mime_for_suffix(suffix: str) -> str:
    normalized = suffix.casefold()
    if normalized in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if normalized == ".png":
        return "image/png"
    if normalized == ".webp":
        return "image/webp"
    if normalized == ".bmp":
        return "image/bmp"
    raise DataUnavailableError("multi-view", f"unsupported image suffix: {suffix}")


def _payload_format(payload: bytes) -> tuple[str, str]:
    if payload.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        return ".webp", "image/webp"
    if payload.startswith(b"BM"):
        return ".bmp", "image/bmp"
    raise AdapterError("embedded media is not a supported image")


def _split_mmsi_question(question: str) -> tuple[str, tuple[Choice, ...]]:
    if "Options:" not in question:
        raise AdapterError("MMSI question has no Options section")
    prompt, option_text = question.split("Options:", 1)
    matches = re.findall(
        r"(?:^|[;,]\s*)([A-Z]):\s*(.*?)(?=(?:[;,]\s*[A-Z]:)|$)",
        option_text.strip(),
        flags=re.DOTALL,
    )
    choices = tuple(Choice(label=label, text=text.strip()) for label, text in matches)
    if len(choices) < 2:
        raise AdapterError("MMSI question options are malformed")
    return prompt.strip(), choices


def _split_mindcube_question(question: str) -> tuple[str, tuple[Choice, ...]]:
    markers = list(re.finditer(r"(?:^|\s)([A-D])\.\s+", question))
    if len(markers) < 2:
        raise AdapterError("MindCube question options are malformed")
    choices: list[Choice] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(question)
        choices.append(Choice(label=marker.group(1), text=question[start:end].strip()))
    return question[: markers[0].start()].strip(), tuple(choices)


def _usable_materialized_views(
    root: Path,
    sample_id: str,
    expected_count: int,
) -> list[tuple[Path, str]]:
    files = _ordered_images(root / "materialized_media" / sample_id)
    if len(files) != expected_count:
        return []
    return [(path, _mime_for_suffix(path.suffix)) for path in files]


@register_adapter
class MMSIBenchAdapter(BaseAdapter):
    """Read MMSI parquet/TSV annotations and embedded multi-view images."""

    benchmark_id = "MMSI-Bench"
    package_id = "mmsi_bench"
    aliases = ("MMSI", "mmsi")
    observation_policy = "selective_multiview"
    initial_asset_limit = 2

    def discover_samples(self) -> tuple[str, ...]:
        """Return source IDs without projecting private annotation fields."""

        self._ensure_open()
        parquet_path = self.data_root / "dataset" / "MMSI_Bench.parquet"
        if parquet_path.is_file():
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pq.read_table(parquet_path, columns=["id"])
                values = {
                    str(value.as_py()).strip()
                    for value in table.column("id")
                    if value.as_py() is not None and str(value.as_py()).strip()
                }
                if values:
                    return tuple(sorted(values, key=_natural_key))
            except ImportError:
                pass
            except (OSError, ValueError, pa.ArrowException) as exc:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MMSI parquet ID column is unreadable",
                    path=parquet_path,
                ) from exc

        tsv_path = self.data_root / "dataset" / "MMSI_bench.tsv"
        if tsv_path.is_file():
            try:
                with self._open_tsv(tsv_path) as rows:
                    values = {
                        str(row.get("index") or "").strip()
                        for row in rows
                        if str(row.get("index") or "").strip()
                    }
            except (OSError, csv.Error) as exc:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MMSI TSV ID column is unreadable",
                    path=tsv_path,
                ) from exc
            if values:
                return tuple(sorted(values, key=_natural_key))
        raise DataUnavailableError(
            self.benchmark_id,
            "no MMSI annotation IDs are available",
            path=self.data_root / "dataset",
        )

    list_sample_ids = discover_samples
    sample_ids = discover_samples

    def _default_sample_id(self) -> str:
        media_root = self.data_root / "materialized_media"
        if media_root.is_dir():
            candidates = sorted(
                (
                    path.name
                    for path in media_root.iterdir()
                    if path.is_dir() and _ordered_images(path)
                ),
                key=_natural_key,
            )
            if candidates:
                return candidates[0]

        parquet_path = self.data_root / "dataset" / "MMSI_Bench.parquet"
        if parquet_path.is_file():
            try:
                import pyarrow as pa
                import pyarrow.parquet as pq

                table = pq.read_table(parquet_path, columns=["id"])
                if table.num_rows:
                    return str(table.column("id")[0].as_py())
            except ImportError:
                pass
            except (OSError, ValueError, pa.ArrowException) as exc:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MMSI parquet ID column is unreadable",
                    path=parquet_path,
                ) from exc

        tsv_path = self.data_root / "dataset" / "MMSI_bench.tsv"
        if tsv_path.is_file():
            with self._open_tsv(tsv_path) as rows:
                try:
                    row = next(rows)
                except StopIteration:
                    pass
                else:
                    return str(row.get("index") or "").strip()
        raise DataUnavailableError(
            self.benchmark_id,
            "no MMSI annotation rows are available",
            path=self.data_root,
        )

    @staticmethod
    def _open_tsv(path: Path) -> Any:
        try:
            csv.field_size_limit(sys.maxsize)
        except OverflowError:
            csv.field_size_limit(2**31 - 1)
        handle = path.open("r", encoding="utf-8", newline="")

        class _Rows:
            def __enter__(self) -> csv.DictReader:
                return csv.DictReader(handle, delimiter="\t")

            def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
                handle.close()

        return _Rows()

    def _parquet_row(self, sample_id: str) -> dict[str, Any] | None:
        path = self.data_root / "dataset" / "MMSI_Bench.parquet"
        if not path.is_file():
            return None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            return None
        try:
            selector: int | str = int(sample_id) if re.fullmatch(r"-?\d+", sample_id) else sample_id
            table = pq.read_table(
                path,
                columns=["id", "images", "question_type", "question", "answer", "difficulty"],
                filters=[("id", "=", selector)],
            )
        except (OSError, ValueError, TypeError, pa.ArrowException) as exc:
            raise DataUnavailableError(
                self.benchmark_id,
                "MMSI parquet is unreadable",
                path=path,
            ) from exc
        if table.num_rows == 0:
            return None
        if table.num_rows != 1:
            raise AdapterError(f"duplicate MMSI source ID: {sample_id}")
        return table.to_pylist()[0]

    def _tsv_row(self, sample_id: str) -> dict[str, Any] | None:
        path = self.data_root / "dataset" / "MMSI_bench.tsv"
        if not path.is_file():
            return None
        try:
            with self._open_tsv(path) as rows:
                raw = next(
                    (row for row in rows if str(row.get("index") or "").strip() == sample_id),
                    None,
                )
        except (OSError, csv.Error) as exc:
            raise DataUnavailableError(
                self.benchmark_id,
                "MMSI TSV is unreadable",
                path=path,
            ) from exc
        if raw is None:
            return None
        try:
            encoded_images = ast.literal_eval(str(raw.get("image") or ""))
            if not isinstance(encoded_images, list) or not all(
                isinstance(value, str) for value in encoded_images
            ):
                raise ValueError("image field is not a string list")
            images = [base64.b64decode(value, validate=True) for value in encoded_images]
        except (ValueError, SyntaxError, binascii.Error) as exc:
            raise DataUnavailableError(
                self.benchmark_id,
                f"MMSI TSV media is malformed for sample {sample_id}",
                path=path,
            ) from exc
        return {
            "id": sample_id,
            "images": images,
            "question_type": raw.get("category"),
            "question": raw.get("question"),
            "answer": raw.get("answer"),
            "difficulty": None,
        }

    def _load_row(self, sample_id: str) -> dict[str, Any]:
        if not self.data_root.is_dir():
            raise DataUnavailableError(
                self.benchmark_id,
                "MMSI data root is absent",
                path=self.data_root,
            )
        row = self._parquet_row(sample_id)
        if row is None:
            row = self._tsv_row(sample_id)
        if row is None:
            annotation_exists = any(
                path.is_file()
                for path in (
                    self.data_root / "dataset" / "MMSI_Bench.parquet",
                    self.data_root / "dataset" / "MMSI_bench.tsv",
                )
            )
            if not annotation_exists:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MMSI parquet and TSV annotations are absent",
                    path=self.data_root / "dataset",
                )
            raise SampleNotFoundError(f"MMSI sample does not exist: {sample_id}")
        return row

    def _media(self, sample_id: str, images: Sequence[Any]) -> list[tuple[Path, str]]:
        existing = _usable_materialized_views(self.data_root, sample_id, len(images))
        if existing:
            return existing
        output: list[tuple[Path, str]] = []
        for index, value in enumerate(images):
            if not isinstance(value, (bytes, bytearray, memoryview)):
                raise DataUnavailableError(
                    self.benchmark_id,
                    f"MMSI sample {sample_id} has a non-binary image",
                )
            payload = bytes(value)
            suffix, mime_type = _payload_format(payload)
            output.append(
                (
                    self._materialize_bytes(
                        source_sample_id=sample_id,
                        sequence_index=index,
                        suffix=suffix,
                        payload=payload,
                    ),
                    mime_type,
                )
            )
        return output

    def load_sample(
        self,
        sample_id: str | int | Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        self._ensure_open()
        source_id = _source_id(sample_id) or self._default_sample_id()
        row = self._load_row(source_id)
        images = row.get("images") or []
        if not isinstance(images, Sequence) or isinstance(images, (str, bytes, bytearray)):
            raise DataUnavailableError(
                self.benchmark_id,
                f"MMSI sample {source_id} has no image sequence",
            )
        prompt, choices = _split_mmsi_question(str(row.get("question") or ""))
        metadata = {
            "question_type": row.get("question_type"),
            "difficulty": row.get("difficulty"),
            "view_count": len(images),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return self._store_sample(
            source_sample_id=source_id,
            prompt=prompt,
            choices=choices,
            reference_label=row.get("answer"),
            media=self._media(source_id, images),
            metadata=metadata,
        )


@register_adapter
class MindCubeAdapter(BaseAdapter):
    """Read MindCube JSONL annotations and materialize selected zip views."""

    benchmark_id = "MindCube"
    package_id = "mindcube"
    aliases = ("mind_cube",)
    observation_policy = "selective_limited_views"
    initial_asset_limit = 1

    def discover_samples(self) -> tuple[str, ...]:
        """Return the deterministic source denominator from the private index."""

        return tuple(sorted(self._load_rows(), key=_natural_key))

    list_sample_ids = discover_samples
    sample_ids = discover_samples

    def __init__(
        self,
        data_root: Path | str | None = None,
        *,
        cache_root: Path | str | None = None,
    ) -> None:
        super().__init__(data_root, cache_root=cache_root)
        self._rows: dict[str, dict[str, Any]] | None = None
        self._annotation_path: Path | None = None
        self._archive: zipfile.ZipFile | None = None

    def _zip(self) -> zipfile.ZipFile:
        if self._archive is not None:
            return self._archive
        path = self.data_root / "dataset" / "data.zip"
        if not path.is_file():
            raise DataUnavailableError(
                self.benchmark_id,
                "MindCube data.zip is absent",
                path=path,
            )
        try:
            self._archive = zipfile.ZipFile(path, "r")
        except (OSError, zipfile.BadZipFile) as exc:
            raise DataUnavailableError(
                self.benchmark_id,
                "MindCube data.zip is unreadable",
                path=path,
            ) from exc
        return self._archive

    def _annotation_candidates(self) -> tuple[Path, ...]:
        return (
            self.data_root / "extracted" / "data" / "raw" / "MindCube.jsonl",
            self.data_root / "extracted" / "data" / "data" / "raw" / "MindCube.jsonl",
            self.data_root / "dataset" / "data" / "raw" / "MindCube.jsonl",
        )

    def _load_rows(self) -> dict[str, dict[str, Any]]:
        self._ensure_open()
        if self._rows is not None:
            return self._rows
        if not self.data_root.is_dir():
            raise DataUnavailableError(
                self.benchmark_id,
                "MindCube data root is absent",
                path=self.data_root,
            )

        handle: Any
        annotation_label: str
        for candidate in self._annotation_candidates():
            if candidate.is_file():
                try:
                    handle = candidate.open("r", encoding="utf-8")
                except OSError as exc:
                    raise DataUnavailableError(
                        self.benchmark_id,
                        "MindCube extracted annotation is unreadable",
                        path=candidate,
                    ) from exc
                self._annotation_path = candidate.resolve()
                annotation_label = str(candidate)
                break
        else:
            archive = self._zip()
            members = sorted(
                name
                for name in archive.namelist()
                if name.endswith("/raw/MindCube.jsonl") or name == "data/raw/MindCube.jsonl"
            )
            if not members:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MindCube.jsonl is absent from data.zip",
                    path=self.data_root / "dataset" / "data.zip",
                )
            annotation_label = f"data.zip::{members[0]}"
            try:
                handle = archive.open(members[0], "r")
            except (KeyError, OSError, zipfile.BadZipFile) as exc:
                raise DataUnavailableError(
                    self.benchmark_id,
                    "MindCube zip annotation is unreadable",
                ) from exc

        rows: dict[str, dict[str, Any]] = {}
        try:
            for line_number, raw_line in enumerate(handle, start=1):
                if isinstance(raw_line, bytes):
                    raw_line = raw_line.decode("utf-8")
                if not raw_line.strip():
                    continue
                value = json.loads(raw_line)
                if not isinstance(value, dict) or not str(value.get("id") or "").strip():
                    raise ValueError(f"invalid row {line_number}")
                source_id = str(value["id"]).strip()
                if source_id in rows:
                    raise ValueError(f"duplicate sample ID {source_id}")
                rows[source_id] = {
                    "id": source_id,
                    "category": value.get("category"),
                    "type": value.get("type"),
                    "question": value.get("question"),
                    "images": value.get("images"),
                    "gt_answer": value.get("gt_answer"),
                }
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            raise DataUnavailableError(
                self.benchmark_id,
                f"MindCube annotation is malformed: {annotation_label}",
            ) from exc
        finally:
            handle.close()
        if not rows:
            raise DataUnavailableError(
                self.benchmark_id,
                "MindCube annotation has no rows",
            )
        self._rows = rows
        return rows

    def _default_sample_id(self, rows: Mapping[str, Any]) -> str:
        media_root = self.data_root / "materialized_media"
        if media_root.is_dir():
            candidates = sorted(
                (
                    path.name
                    for path in media_root.iterdir()
                    if path.is_dir() and path.name in rows and _ordered_images(path)
                ),
                key=_natural_key,
            )
            if candidates:
                return candidates[0]
        return next(iter(rows))

    def _safe_relative_image(self, raw: Any, sample_id: str) -> PurePosixPath:
        value = str(raw or "").replace("\\", "/")
        relative = PurePosixPath(value)
        if not value or relative.is_absolute() or ".." in relative.parts:
            raise DataUnavailableError(
                self.benchmark_id,
                f"MindCube sample {sample_id} has an unsafe image path",
            )
        return relative

    def _extracted_media_roots(self) -> tuple[Path, ...]:
        roots: list[Path] = []
        if self._annotation_path is not None:
            roots.append(self._annotation_path.parents[1])
        roots.extend(
            (
                self.data_root / "extracted" / "data",
                self.data_root / "extracted" / "data" / "data",
            )
        )
        unique: list[Path] = []
        for root in roots:
            resolved = root.resolve()
            if resolved not in unique:
                unique.append(resolved)
        return tuple(unique)

    def _from_extracted(
        self,
        images: Sequence[PurePosixPath],
    ) -> list[tuple[Path, str]]:
        for root in self._extracted_media_roots():
            output: list[tuple[Path, str]] = []
            for relative in images:
                path = (root / Path(*relative.parts)).resolve()
                try:
                    path.relative_to(root)
                except ValueError:
                    output = []
                    break
                if not path.is_file() or path.stat().st_size <= 0:
                    output = []
                    break
                output.append((path, _mime_for_suffix(path.suffix)))
            if len(output) == len(images):
                return output
        return []

    def _from_zip(
        self,
        sample_id: str,
        images: Sequence[PurePosixPath],
    ) -> list[tuple[Path, str]]:
        archive = self._zip()
        names = set(archive.namelist())
        output: list[tuple[Path, str]] = []
        for index, relative in enumerate(images):
            candidates = (
                PurePosixPath("data", *relative.parts).as_posix(),
                relative.as_posix(),
            )
            member = next((name for name in candidates if name in names), None)
            if member is None:
                raise DataUnavailableError(
                    self.benchmark_id,
                    f"MindCube sample {sample_id} media is absent from data.zip",
                )
            try:
                payload = archive.read(member)
            except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
                raise DataUnavailableError(
                    self.benchmark_id,
                    f"MindCube sample {sample_id} media is unreadable",
                ) from exc
            suffix = PurePosixPath(member).suffix.casefold()
            mime_type = _mime_for_suffix(suffix)
            output.append(
                (
                    self._materialize_bytes(
                        source_sample_id=sample_id,
                        sequence_index=index,
                        suffix=suffix,
                        payload=payload,
                    ),
                    mime_type,
                )
            )
        return output

    def _media(self, sample_id: str, raw_images: Sequence[Any]) -> list[tuple[Path, str]]:
        materialized = _usable_materialized_views(
            self.data_root, sample_id, len(raw_images)
        )
        if materialized:
            return materialized
        images = [self._safe_relative_image(value, sample_id) for value in raw_images]
        extracted = self._from_extracted(images)
        return extracted if extracted else self._from_zip(sample_id, images)

    def load_sample(
        self,
        sample_id: str | int | Mapping[str, Any] | None = None,
    ) -> AdapterSample:
        self._ensure_open()
        rows = self._load_rows()
        source_id = _source_id(sample_id) or self._default_sample_id(rows)
        row = rows.get(source_id)
        if row is None:
            raise SampleNotFoundError(f"MindCube sample does not exist: {source_id}")
        raw_images = row.get("images") or []
        if not isinstance(raw_images, Sequence) or isinstance(
            raw_images, (str, bytes, bytearray)
        ) or not raw_images:
            raise DataUnavailableError(
                self.benchmark_id,
                f"MindCube sample {source_id} has no image sequence",
            )
        prompt, choices = _split_mindcube_question(str(row.get("question") or ""))
        metadata = {
            "category": row.get("category"),
            "type": row.get("type"),
            "view_count": len(raw_images),
        }
        metadata = {key: value for key, value in metadata.items() if value is not None}
        return self._store_sample(
            source_sample_id=source_id,
            prompt=prompt,
            choices=choices,
            reference_label=row.get("gt_answer"),
            media=self._media(source_id, raw_images),
            metadata=metadata,
        )

    def _close_resources(self) -> None:
        if self._archive is not None:
            self._archive.close()
            self._archive = None
        self._rows = None
        self._annotation_path = None


__all__ = ["MMSIBenchAdapter", "MindCubeAdapter"]
