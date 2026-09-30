from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from .spatial_backend import SpatialSample
from .spatial_vsi_adapter import write_spatial_samples_jsonl


MMSI_BENCH_HF_DATASET = "RunsenXu/MMSI-Bench"
MMSI_BENCH_PARQUET = "MMSI_Bench.parquet"


@dataclass(slots=True)
class MMSIBenchAssetPreflight:
    manifest_path: str
    image_root: str
    manifest_exists: bool
    sample_count: int
    view_count: int
    materialized_view_count: int
    missing_artifacts: list[str]
    blockers: list[str]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "image_root": self.image_root,
            "manifest_exists": self.manifest_exists,
            "sample_count": self.sample_count,
            "view_count": self.view_count,
            "materialized_view_count": self.materialized_view_count,
            "missing_artifacts": list(self.missing_artifacts),
            "ready": self.ready,
            "blockers": list(self.blockers),
            "data_source": {
                "dataset": MMSI_BENCH_HF_DATASET,
                "file": MMSI_BENCH_PARQUET,
                "local_manifest_contains_harness_side_answers": True,
                "agent_visible_answer_key": False,
            },
        }


def inspect_mmsi_asset_preflight(
    *,
    manifest_path: str | Path = "benchmarks/non_operation/mmsi_bench/mmsi_smoke.jsonl",
    image_root: str | Path = "benchmarks/non_operation/mmsi_bench/images",
) -> MMSIBenchAssetPreflight:
    """Check the local real MMSI manifest and materialized image artifacts."""

    manifest = Path(manifest_path)
    root = Path(image_root)
    blockers: list[str] = []
    missing_artifacts: list[str] = []
    sample_count = 0
    view_count = 0
    materialized_count = 0
    manifest_exists = manifest.is_file()
    if not manifest_exists:
        blockers.append(f"mmsi_manifest_not_found:{manifest}")
    else:
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                blockers.append(f"mmsi_manifest_json_decode_failed:{manifest}:{line_number}:{exc.msg}")
                continue
            sample_count += 1
            for view in record.get("views", []) or []:
                view_count += 1
                artifact = Path(str(view.get("artifact", "")))
                if artifact.is_file():
                    materialized_count += 1
                else:
                    missing_artifacts.append(str(view.get("artifact", "")))
        if sample_count == 0:
            blockers.append(f"mmsi_manifest_empty:{manifest}")
        if missing_artifacts:
            blockers.append(f"mmsi_image_artifacts_missing:{len(missing_artifacts)}")

    return MMSIBenchAssetPreflight(
        manifest_path=str(manifest),
        image_root=str(root),
        manifest_exists=manifest_exists,
        sample_count=sample_count,
        view_count=view_count,
        materialized_view_count=materialized_count,
        missing_artifacts=missing_artifacts,
        blockers=blockers,
    )


def load_mmsi_bench_samples(
    *,
    parquet_path: str | Path | None = None,
    split: str = "test",
    limit: int | None = None,
    image_output_dir: str | Path | None = None,
    local_dir: str | Path | None = None,
) -> list[SpatialSample]:
    """Load real MMSI-Bench rows and materialize image artifacts.

    MMSI-Bench stores each task's multi-image payload in a Hugging Face
    parquet file. Answers and thoughts are retained only on the returned
    `SpatialSample` for harness-side verification; agent-visible primitives
    expose question text, choices, view ids, and image artifact paths.
    """

    resolved_parquet = Path(parquet_path) if parquet_path is not None else _download_mmsi_parquet(local_dir=local_dir)
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install pandas and pyarrow to load MMSI-Bench parquet rows.") from exc

    df = pd.read_parquet(resolved_parquet)
    if split != "test":
        raise ValueError("MMSI-Bench currently exposes only the test split.")

    image_root = Path(image_output_dir) if image_output_dir is not None else resolved_parquet.parent / "images"
    image_root.mkdir(parents=True, exist_ok=True)

    samples: list[SpatialSample] = []
    for _, row in df.head(limit if limit is not None else len(df)).iterrows():
        samples.append(mmsi_record_to_spatial_sample(dict(row), image_output_dir=image_root))
    return samples


def load_mmsi_bench_samples_from_rows_api(
    *,
    split: str = "test",
    offset: int = 0,
    limit: int = 2,
    image_output_dir: str | Path = "benchmarks/non_operation/mmsi_bench/images",
) -> list[SpatialSample]:
    """Load a tiny real MMSI-Bench slice through Hugging Face datasets-server."""

    try:
        import requests
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install requests to load MMSI-Bench rows through the datasets-server API.") from exc

    response = requests.get(
        "https://datasets-server.huggingface.co/rows",
        params={"dataset": MMSI_BENCH_HF_DATASET, "config": "default", "split": split, "offset": offset, "length": limit},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    samples = []
    for row_wrapper in payload.get("rows", []):
        row = dict(row_wrapper.get("row", {}))
        samples.append(mmsi_record_to_spatial_sample(row, image_output_dir=image_output_dir))
    return samples


def mmsi_record_to_spatial_sample(record: dict[str, Any], *, image_output_dir: str | Path) -> SpatialSample:
    native_id = str(_first_present(record, "id", "index", default="unknown"))
    question = str(_first_present(record, "question", "prompt", default=""))
    question_type = str(_first_present(record, "question_type", "category", default="unknown"))
    answer = str(_first_present(record, "answer", default=""))
    views = _materialize_mmsi_images(record.get("images", record.get("image")), sample_id=native_id, image_output_dir=Path(image_output_dir))

    return SpatialSample(
        sample_id=f"mmsi_{native_id}",
        source="mmsi_bench",
        instruction=question,
        answer=answer,
        category=question_type,
        choices=_extract_choice_labels(question),
        views=views,
        facts=[],
        metadata={
            "benchmark_family": "multi_image_spatial",
            "adapter_target": "MMSI-Bench",
            "hf_dataset": MMSI_BENCH_HF_DATASET,
            "hf_file": MMSI_BENCH_PARQUET,
            "split": "test",
            "native_id": native_id,
            "question_type": question_type,
            "difficulty": _json_safe(record.get("difficulty")),
            "mean_normed_duration_seconds": _json_safe(record.get("mean_normed_duration_seconds")),
            "image_count": len(views),
            "oracle_leakage_level": "none",
            "hidden_harness_field_count": 2,
            "label_visibility": "harness_side_verifier_only",
        },
    )


class MMSISpatialBackend:
    """MMSI-specific primitive facade over the generic spatial verifier."""

    def __init__(self, samples: list[SpatialSample] | None = None, data_path: str | Path | None = None, sample_limit: int | None = None) -> None:
        from .spatial_backend import SpatialDiagnosticBackend

        if data_path is None:
            self._backend = SpatialDiagnosticBackend(samples=samples, sample_limit=sample_limit)
        else:
            self._backend = SpatialDiagnosticBackend(samples=samples, data_path=data_path, sample_limit=sample_limit)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._backend, name)

    def list_primitives(self, level: str | None = None) -> list[Any]:
        from .schemas import PrimitiveCard

        cards = [
            PrimitiveCard(
                name="get_mmsi_task_context",
                capability_tags=["task", "metadata", "mmsi", "spatial"],
                input_schema={"prompt": "str|None", "query": "str|None", "context": "dict|None"},
                output_schema={"instruction": "str", "choices": "list[str]", "views": "list[dict]", "evidence": "dict"},
                abstraction_level="L1",
                description="Return MMSI task context and image references without answer-key or reasoning fields.",
            ),
            PrimitiveCard(
                name="inspect_mmsi_image",
                capability_tags=["vision", "image", "mmsi", "spatial"],
                input_schema={"query": "str|None", "view_id": "str", "context": "dict|None"},
                output_schema={"view": "dict", "artifact_status": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect one MMSI image reference and report whether the real dataset artifact is available.",
            ),
            PrimitiveCard(
                name="inspect_mmsi_pixels",
                capability_tags=["vision", "image", "pixels", "bbox", "mmsi", "spatial"],
                input_schema={"query": "str|None", "view_id": "str", "bbox": "list[int]|None", "context": "dict|None"},
                output_schema={"image": "dict", "regions": "list[dict]", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect a materialized MMSI image artifact and return auditable pixel and bbox region statistics.",
            ),
            PrimitiveCard(
                name="estimate_mmsi_camera_motion",
                capability_tags=["vision", "optical_flow", "camera_motion", "mmsi", "spatial"],
                input_schema={
                    "source_view_id": "str",
                    "target_view_id": "str",
                    "query": "str|None",
                    "choices": "list[str]|None",
                    "context": "dict|None",
                },
                output_schema={
                    "motion": "dict",
                    "choice_scores": "list[dict]",
                    "recommended_choice": "str|None",
                    "evidence": "dict",
                },
                abstraction_level="L2",
                description="Estimate egocentric camera motion from two real MMSI image artifacts using local optical-flow/keypoint evidence.",
            ),
            PrimitiveCard(
                name="compare_spatial_relation",
                capability_tags=["spatial_relation", "evidence", "mmsi"],
                input_schema={"query": "str", "context": "dict|None"},
                output_schema={"relation_request": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Record a spatial relation comparison query over visible MMSI context without oracle access.",
            ),
            PrimitiveCard(
                name="record_spatial_evidence",
                capability_tags=["evidence", "memory"],
                input_schema={"key": "str", "value": "object", "context": "dict|None"},
                output_schema={"artifact_id": "str", "evidence": "dict"},
                abstraction_level="L2",
                description="Attach agent-derived MMSI spatial evidence to the episode trace.",
            ),
            PrimitiveCard(
                name="submit_spatial_answer",
                capability_tags=["commit", "answer"],
                input_schema={"answer": "str", "context": "dict|None"},
                output_schema={"submission": "dict", "evidence": "dict"},
                abstraction_level="L3",
                description="Commit an MMSI answer for harness-side verifier scoring.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self._backend.record_event("list_primitives", {"level": level, "count": len(cards), "facade": "mmsi"})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> Any:
        if name == "get_mmsi_task_context":
            result = self._backend.call_primitive(
                "get_task_context",
                prompt=kwargs.get("prompt"),
                query=kwargs.get("query"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "inspect_mmsi_image":
            result = self._backend.call_primitive(
                "inspect_view",
                view_id=kwargs["view_id"],
                query=kwargs.get("query"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            if result.ok:
                view = result.output.get("view", {})
                artifact = Path(str(view.get("artifact", "")))
                result.output["artifact_status"] = {
                    "artifact": str(view.get("artifact", "")),
                    "exists": artifact.is_file(),
                    "blocker": None if artifact.is_file() else "dataset image artifact is not materialized at this path",
                }
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "inspect_mmsi_pixels":
            result = self._backend.call_primitive(
                "inspect_view",
                view_id=kwargs["view_id"],
                query=kwargs.get("query"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            if result.ok:
                view = result.output.get("view", {})
                pixel_result = _inspect_materialized_image_pixels(
                    view=view,
                    bbox=kwargs.get("bbox"),
                    query=kwargs.get("query"),
                    agent_context=kwargs.get("context") or kwargs.get("agent_context"),
                    evidence=result.output.get("evidence", {}),
                )
                result.ok = pixel_result["ok"]
                result.error = pixel_result.get("error")
                result.output = {key: value for key, value in pixel_result.items() if key not in {"ok", "error"}}
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "estimate_mmsi_camera_motion":
            source = self._backend.call_primitive(
                "inspect_view",
                view_id=kwargs["source_view_id"],
                query=kwargs.get("query"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            target = self._backend.call_primitive(
                "inspect_view",
                view_id=kwargs["target_view_id"],
                query=kwargs.get("query"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            if not source.ok:
                return self._record_facade_primitive(name, kwargs, source)
            if not target.ok:
                return self._record_facade_primitive(name, kwargs, target)
            from .schemas import PrimitiveResult

            output = _estimate_mmsi_camera_motion(
                source_view=source.output.get("view", {}),
                target_view=target.output.get("view", {}),
                query=kwargs.get("query"),
                choices=kwargs.get("choices"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
                evidence=source.output.get("evidence", {}),
            )
            result = PrimitiveResult(
                name=name,
                ok=bool(output.pop("ok")),
                output=output,
                error=output.get("error"),
            )
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "compare_spatial_relation":
            query = str(kwargs.get("query", ""))
            result = self._backend.call_primitive(
                "write_evidence",
                key=f"relation_query_{len(self._backend.get_trace().artifacts) + 1}",
                value={"query": query, "context": kwargs.get("context") or kwargs.get("agent_context") or {}},
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            if result.ok:
                result.name = name
                result.output["relation_request"] = {
                    "query": query,
                    "context": kwargs.get("context") or kwargs.get("agent_context") or {},
                    "oracle_access": False,
                }
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "record_spatial_evidence":
            result = self._backend.call_primitive(
                "write_evidence",
                key=kwargs["key"],
                value=kwargs.get("value"),
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            return self._record_facade_primitive(name, kwargs, result)
        elif name == "submit_spatial_answer":
            result = self._backend.call_primitive(
                "submit_answer",
                answer=kwargs["answer"],
                agent_context=kwargs.get("context") or kwargs.get("agent_context"),
            )
            return self._record_facade_primitive(name, kwargs, result)
        else:
            from .schemas import PrimitiveResult

            result = PrimitiveResult(name=name, ok=False, error=f"Unknown MMSI primitive: {name}")
            return self._record_facade_primitive(name, kwargs, result)

    def _record_facade_primitive(self, name: str, kwargs: dict[str, Any], result: Any) -> Any:
        result.name = name
        self._backend.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result


def write_mmsi_samples_jsonl(samples: Iterable[SpatialSample], path: str | Path) -> None:
    write_spatial_samples_jsonl(samples, path)


def _download_mmsi_parquet(*, local_dir: str | Path | None = None) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        raise RuntimeError("Install huggingface_hub to download MMSI-Bench from Hugging Face.") from exc

    target_dir = Path(local_dir) if local_dir is not None else Path("benchmarks/non_operation/mmsi_bench/cache/hf")
    return Path(
        hf_hub_download(
            repo_id=MMSI_BENCH_HF_DATASET,
            repo_type="dataset",
            filename=MMSI_BENCH_PARQUET,
            local_dir=target_dir,
        )
    )


def _materialize_mmsi_images(images: Any, *, sample_id: str, image_output_dir: Path) -> list[dict[str, Any]]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    if images is None:
        payloads = []
    elif isinstance(images, (bytes, bytearray, str, dict)):
        payloads = [images]
    else:
        try:
            payloads = list(images)
        except TypeError:
            payloads = [images]
    views: list[dict[str, Any]] = []
    for index, payload in enumerate(payloads):
        suffix = ".jpg"
        image_path = image_output_dir / f"{sample_id}_{index}{suffix}"
        raw = _extract_image_bytes(payload)
        if raw is not None:
            image_path.write_bytes(raw)
            artifact = str(image_path)
            status = "materialized"
        else:
            artifact = f"mmsi://{sample_id}/{index}"
            status = "unmaterialized"
        views.append(
            {
                "view_id": f"mmsi:{sample_id}:{index}",
                "artifact": artifact,
                "media_type": "image",
                "dataset": "MMSI-Bench",
                "image_index": index,
                "artifact_status": status,
            }
        )
    return views


def _extract_image_bytes(payload: Any) -> bytes | None:
    if isinstance(payload, bytes):
        return payload
    if isinstance(payload, bytearray):
        return bytes(payload)
    if isinstance(payload, dict):
        for key in ("bytes", "data"):
            raw = payload.get(key)
            if isinstance(raw, bytes):
                return raw
            if isinstance(raw, bytearray):
                return bytes(raw)
        src = payload.get("src")
        if isinstance(src, str) and src:
            try:
                import requests

                response = requests.get(src, timeout=60)
                response.raise_for_status()
                return response.content
            except Exception:
                return None
    return None


def _inspect_materialized_image_pixels(
    *,
    view: dict[str, Any],
    bbox: Any,
    query: str | None,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    artifact = Path(str(view.get("artifact", "")))
    if not artifact.is_file():
        return {
            "ok": False,
            "error": f"image_artifact_not_found:{artifact}",
            "image": {"artifact": str(artifact), "exists": False},
            "regions": [],
            "evidence": evidence,
        }

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is optional outside live image inspection.
        return {"ok": False, "error": f"pillow_unavailable:{exc}", "image": {"artifact": str(artifact), "exists": True}, "regions": [], "evidence": evidence}

    try:
        with Image.open(artifact) as image:
            image.load()
            rgb = image.convert("RGB")
            width, height = rgb.size
            try:
                requested_bbox = _normalize_bbox(bbox, width=width, height=height)
            except (TypeError, ValueError) as exc:
                return {
                    "ok": False,
                    "error": f"invalid_bbox:{exc}",
                    "image": {"artifact": str(artifact), "exists": True, "width": width, "height": height},
                    "regions": [],
                    "evidence": evidence,
                }
            full_region = _summarize_region(rgb, name="full_image", bbox=(0, 0, width, height))
            regions = [full_region]
            if requested_bbox != (0, 0, width, height):
                regions.append(_summarize_region(rgb, name="requested_bbox", bbox=requested_bbox))
            return {
                "ok": True,
                "query": query,
                "agent_context": agent_context or {},
                "image": {
                    "view_id": view.get("view_id"),
                    "artifact": str(artifact),
                    "exists": True,
                    "format": image.format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "media_type": view.get("media_type"),
                },
                "bbox": list(requested_bbox),
                "regions": regions,
                "evidence": dict(evidence)
                | {
                    "pixel_inspection": True,
                    "artifact": str(artifact),
                    "bbox": list(requested_bbox),
                    "query": query,
                    "agent_context": agent_context or {},
                },
            }
    except Exception as exc:
        return {"ok": False, "error": f"image_decode_failed:{type(exc).__name__}:{exc}", "image": {"artifact": str(artifact), "exists": True}, "regions": [], "evidence": evidence}


def _normalize_bbox(bbox: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
    if bbox is None:
        return (0, 0, width, height)
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be a four-item [x0, y0, x1, y1] list")
    x0, y0, x1, y1 = (int(value) for value in bbox)
    x0 = max(0, min(width, x0))
    x1 = max(0, min(width, x1))
    y0 = max(0, min(height, y0))
    y1 = max(0, min(height, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"bbox has empty area after clipping: {[x0, y0, x1, y1]}")
    return (x0, y0, x1, y1)


def _summarize_region(image: Any, *, name: str, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    from PIL import ImageStat

    region = image.crop(bbox)
    stat = ImageStat.Stat(region)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return {
        "name": name,
        "bbox": list(bbox),
        "width": width,
        "height": height,
        "area_pixels": width * height,
        "channels": ["R", "G", "B"],
        "mean_rgb": [round(value, 3) for value in stat.mean],
        "extrema_rgb": [[int(low), int(high)] for low, high in stat.extrema],
    }


def _estimate_mmsi_camera_motion(
    *,
    source_view: dict[str, Any],
    target_view: dict[str, Any],
    query: str | None,
    choices: Any,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    source_path = Path(str(source_view.get("artifact", "")))
    target_path = Path(str(target_view.get("artifact", "")))
    if not source_path.is_file() or not target_path.is_file():
        missing = [str(path) for path in (source_path, target_path) if not path.is_file()]
        return {
            "ok": False,
            "error": f"mmsi_motion_image_artifact_not_found:{missing}",
            "motion": {"source_artifact": str(source_path), "target_artifact": str(target_path)},
            "choice_scores": [],
            "recommended_choice": None,
            "evidence": evidence,
        }
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional runtime dependency.
        return {
            "ok": False,
            "error": f"opencv_unavailable:{exc}",
            "motion": {"source_artifact": str(source_path), "target_artifact": str(target_path)},
            "choice_scores": [],
            "recommended_choice": None,
            "evidence": evidence,
        }

    image_a = cv2.imread(str(source_path), cv2.IMREAD_GRAYSCALE)
    image_b = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)
    if image_a is None or image_b is None:
        return {
            "ok": False,
            "error": "mmsi_motion_image_decode_failed",
            "motion": {"source_artifact": str(source_path), "target_artifact": str(target_path)},
            "choice_scores": [],
            "recommended_choice": None,
            "evidence": evidence,
        }

    keypoints = cv2.goodFeaturesToTrack(image_a, maxCorners=400, qualityLevel=0.01, minDistance=6)
    flow_vectors: Any = np.empty((0, 2), dtype=np.float32)
    tracked_count = 0
    if keypoints is not None:
        tracked, status, _err = cv2.calcOpticalFlowPyrLK(image_a, image_b, keypoints, None)
        if tracked is not None and status is not None:
            source_points = keypoints[status[:, 0] == 1].reshape(-1, 2)
            target_points = tracked[status[:, 0] == 1].reshape(-1, 2)
            flow_vectors = target_points - source_points
            tracked_count = int(flow_vectors.shape[0])

    median_flow = [0.0, 0.0]
    mean_flow = [0.0, 0.0]
    radial_median = 0.0
    if tracked_count:
        median_flow = [round(float(v), 3) for v in np.median(flow_vectors, axis=0)]
        mean_flow = [round(float(v), 3) for v in np.mean(flow_vectors, axis=0)]
        height, width = image_a.shape[:2]
        center = np.asarray([width / 2.0, height / 2.0], dtype=np.float32)
        source_points = keypoints[status[:, 0] == 1].reshape(-1, 2)
        radial = np.sum((source_points - center) * flow_vectors, axis=1) / (np.linalg.norm(source_points - center, axis=1) + 1e-6)
        radial_median = round(float(np.median(radial)), 3)

    orb_flow = _estimate_orb_feature_flow(cv2, np, image_a, image_b)
    interpretation_flow = orb_flow["median_image_flow_xy"] if orb_flow["match_count"] >= 5 else median_flow

    motion = {
        "source_view_id": source_view.get("view_id"),
        "target_view_id": target_view.get("view_id"),
        "source_artifact": str(source_path),
        "target_artifact": str(target_path),
        "tracked_feature_count": tracked_count,
        "median_image_flow_xy": median_flow,
        "mean_image_flow_xy": mean_flow,
        "orb_match_count": orb_flow["match_count"],
        "orb_median_image_flow_xy": orb_flow["median_image_flow_xy"],
        "radial_flow_median": radial_median,
        "interpretation": _interpret_mmsi_flow(query=query, median_flow=interpretation_flow, mean_flow=mean_flow, radial_median=radial_median),
    }
    choice_scores = _score_mmsi_motion_choices(choices or [], motion)
    recommended = choice_scores[0]["label"] if choice_scores else None
    return {
        "ok": tracked_count > 0,
        "query": query,
        "agent_context": agent_context or {},
        "motion": motion,
        "choice_scores": choice_scores,
        "recommended_choice": recommended,
        "evidence": dict(evidence)
        | {
            "primitive": "estimate_mmsi_camera_motion",
            "query": query,
            "agent_context": agent_context or {},
            "source_view_id": source_view.get("view_id"),
            "target_view_id": target_view.get("view_id"),
            "tracked_feature_count": tracked_count,
        },
    }


def _interpret_mmsi_flow(
    *,
    query: str | None,
    median_flow: list[float],
    mean_flow: list[float],
    radial_median: float,
) -> dict[str, Any]:
    question = str(query or "").lower()
    flow_x = float(median_flow[0] if median_flow else 0.0)
    flow_y = float(median_flow[1] if len(median_flow) > 1 else 0.0)
    mean_x = float(mean_flow[0] if mean_flow else 0.0)
    if "rotating" in question or "turn" in question:
        # When the camera yaws left, static scene features move right in the image.
        yaw = "left" if mean_x > 1.0 else ("right" if mean_x < -1.0 else "unclear")
        return {"motion_type": "camera_rotation", "yaw_direction": yaw, "image_feature_shift": "right" if mean_x > 1.0 else "left" if mean_x < -1.0 else "near_static"}
    lateral = "right" if flow_x < -1.0 else ("left" if flow_x > 1.0 else "unclear")
    forward = radial_median < -2.0 or abs(flow_y) > 8.0
    return {
        "motion_type": "egocentric_translation",
        "translation_lateral": lateral,
        "translation_depth": "forward" if forward else "unclear_or_rotation",
        "image_feature_shift": "left" if flow_x < -1.0 else "right" if flow_x > 1.0 else "near_static",
    }


def _estimate_orb_feature_flow(cv2: Any, np: Any, image_a: Any, image_b: Any) -> dict[str, Any]:
    try:
        orb = cv2.ORB_create(2000)
        keypoints_a, desc_a = orb.detectAndCompute(image_a, None)
        keypoints_b, desc_b = orb.detectAndCompute(image_b, None)
        if desc_a is None or desc_b is None:
            return {"match_count": 0, "median_image_flow_xy": [0.0, 0.0]}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        raw_matches = matcher.knnMatch(desc_a, desc_b, k=2)
        good_matches = []
        for pair in raw_matches:
            if len(pair) != 2:
                continue
            first, second = pair
            if first.distance < 0.75 * second.distance:
                good_matches.append(first)
        if not good_matches:
            return {"match_count": 0, "median_image_flow_xy": [0.0, 0.0]}
        points_a = np.float32([keypoints_a[match.queryIdx].pt for match in good_matches])
        points_b = np.float32([keypoints_b[match.trainIdx].pt for match in good_matches])
        flow = points_b - points_a
        return {
            "match_count": int(len(good_matches)),
            "median_image_flow_xy": [round(float(v), 3) for v in np.median(flow, axis=0)],
        }
    except Exception:
        return {"match_count": 0, "median_image_flow_xy": [0.0, 0.0]}


def _score_mmsi_motion_choices(choices: Any, motion: dict[str, Any]) -> list[dict[str, Any]]:
    labels = _coerce_choice_labels(choices)
    interpretation = motion.get("interpretation", {})
    scored = []
    for label, text in labels:
        normalized = text.lower()
        score = 0.0
        reasons: list[str] = []
        if interpretation.get("motion_type") == "camera_rotation":
            yaw = interpretation.get("yaw_direction")
            if yaw and yaw in normalized:
                score += 2.0
                reasons.append(f"camera yaw estimated as {yaw}")
        else:
            lateral = interpretation.get("translation_lateral")
            depth = interpretation.get("translation_depth")
            if lateral and lateral in normalized:
                score += 1.0
                reasons.append(f"lateral ego motion estimated as {lateral}")
            if depth == "forward" and "forward" in normalized:
                score += 1.0
                reasons.append("radial/vertical flow is consistent with forward motion")
            if depth != "forward" and "back" in normalized:
                score += 0.25
        scored.append({"label": label, "text": text, "score": round(score, 3), "reasons": reasons})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _coerce_choice_labels(choices: Any) -> list[tuple[str, str]]:
    if not isinstance(choices, (list, tuple)):
        return []
    labels = []
    for index, choice in enumerate(choices):
        text = str(choice)
        match = re.match(r"\s*([A-Z])\s*[:\).\uff0e]\s*(.*)", text)
        if match:
            labels.append((match.group(1), match.group(2).strip()))
        else:
            labels.append((chr(ord("A") + index), text.strip()))
    return labels


def _extract_choice_labels(question: str) -> list[str]:
    import re

    labels = []
    pattern = r"(?:^|[\s,])([A-D])[\).\uff0e:]\s*(.*?)(?=(?:[\s,][A-D][\).\uff0e:]\s)|$)"
    for match in re.finditer(pattern, question, flags=re.S):
        labels.append(f"{match.group(1)}: {match.group(2).strip()}")
    return labels


def _first_present(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a SpatialDiagnosticBackend JSONL manifest from MMSI-Bench.")
    parser.add_argument("--parquet-path", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--image-output-dir", type=Path, default=Path("benchmarks/non_operation/mmsi_bench/images"))
    parser.add_argument("--local-dir", type=Path, default=Path("benchmarks/non_operation/mmsi_bench/cache/hf"))
    parser.add_argument("--rows-api", action="store_true", help="Load a tiny real slice through Hugging Face datasets-server instead of downloading the full parquet.")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.rows_api:
        samples = load_mmsi_bench_samples_from_rows_api(
            split=args.split,
            offset=args.offset,
            limit=args.limit,
            image_output_dir=args.image_output_dir,
        )
    else:
        samples = load_mmsi_bench_samples(
            parquet_path=args.parquet_path,
            split=args.split,
            limit=args.limit if args.limit > 0 else None,
            image_output_dir=args.image_output_dir,
            local_dir=args.local_dir,
        )
    write_mmsi_samples_jsonl(samples, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(samples), "image_output_dir": str(args.image_output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
