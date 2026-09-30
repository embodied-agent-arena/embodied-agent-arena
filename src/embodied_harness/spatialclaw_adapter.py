from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterator
from urllib.parse import unquote, urlparse

from .paths import get_project_paths
from .schemas import PrimitiveCard, PrimitiveResult
from .spatial_backend import (
    SpatialDiagnosticBackend,
    SpatialSample,
    _bbox_area,
    _bbox_center,
    _detect_rgb_frames_with_owl_vit,
    _detect_rgb_frames_with_torchvision,
    _is_open_vocabulary_detector_backend,
    _load_sam2_image_predictor,
    _normalize_detector_backend,
    _preflight_video_grounding_model,
    _preflight_video_segmentation_model,
    _summarize_mask_array,
)


DEFAULT_SPATIALCLAW_REPO = get_project_paths().external_upstream("spatialclaw")
SPATIALCLAW_UPSTREAM_COMMIT = "b8ff5f775062fb1268556b2911913927eb20b328"
SPATIALCLAW_ASSET_PATTERNS = {
    "erqa": ["*.parquet", "erqa.tfrecord"],
    "omni3d": ["annotations.json", "images/**/*.jpg", "images/**/*.png"],
    "omnispatial": ["data.json", "*/*.png", "*/*.jpg"],
    "spbench": ["SPBench-SI.parquet", "SPBench-MV.parquet", "*/*.jpg", "*/*.png"],
    "mindcube": [
        "MindCube.jsonl",
        "MindCube_tinybench.jsonl",
        "raw/MindCube.jsonl",
        "raw/MindCube_tinybench.jsonl",
        "data/raw/MindCube.jsonl",
        "data/raw/MindCube_tinybench.jsonl",
    ],
    "mmsi": ["MMSI_Bench.parquet", "images/*.jpg", "images/*.png"],
}
SPATIALCLAW_DATASET_DIRS = {
    "erqa": ("ERQA", "data"),
    "omni3d": ("Omni3D-Bench", None),
    "omnispatial": ("OmniSpatial", "OmniSpatial-test"),
    "spbench": ("SPBench", None),
    "mindcube": ("MindCube", None),
    "mmsi": ("MMSI-Bench", None),
}
SPATIALCLAW_PREFLIGHT_MATCHING_FILE_EXPORT_LIMIT = 50


@dataclass(slots=True)
class SpatialClawRuntimeSmoke:
    repo_path: str
    import_ok: bool
    registry_count: int
    registry_names: list[str]
    benchmark: str
    loader_class: str | None
    sample_count: int
    blocker: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": self.repo_path,
            "import_ok": self.import_ok,
            "registry_count": self.registry_count,
            "registry_names": list(self.registry_names),
            "benchmark": self.benchmark,
            "loader_class": self.loader_class,
            "sample_count": self.sample_count,
            "blocker": self.blocker,
        }


@dataclass(slots=True)
class SpatialClawAssetPreflight:
    repo_path: str
    benchmark: str
    data_root: str
    expected_dataset_dir: str
    expected_file_patterns: list[str]
    repo_exists: bool
    data_root_exists: bool
    dataset_dir_exists: bool
    matching_files: list[str]
    loader_import_ok: bool
    loader_class: str | None
    sample_count: int | None
    blockers: list[str]

    @property
    def ready(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        exported_matches = self.matching_files[:SPATIALCLAW_PREFLIGHT_MATCHING_FILE_EXPORT_LIMIT]
        return {
            "repo_path": self.repo_path,
            "benchmark": self.benchmark,
            "data_root": self.data_root,
            "expected_dataset_dir": self.expected_dataset_dir,
            "expected_file_patterns": list(self.expected_file_patterns),
            "repo_exists": self.repo_exists,
            "data_root_exists": self.data_root_exists,
            "dataset_dir_exists": self.dataset_dir_exists,
            "matching_file_count": len(self.matching_files),
            "matching_files": list(exported_matches),
            "matching_files_truncated": len(self.matching_files) > len(exported_matches),
            "loader_import_ok": self.loader_import_ok,
            "loader_class": self.loader_class,
            "sample_count": self.sample_count,
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


def inspect_spatialclaw_asset_preflight(
    *,
    repo_path: str | Path = DEFAULT_SPATIALCLAW_REPO,
    benchmark: str = "erqa",
    data_root: str | Path | None = None,
    run_loader: bool = True,
) -> SpatialClawAssetPreflight:
    """Check SpatialClaw benchmark assets without fabricating samples."""

    repo = Path(repo_path)
    root = Path(data_root) if data_root is not None else repo / "data"
    expected_dir = _spatialclaw_expected_dataset_dir(root, benchmark)
    patterns = list(SPATIALCLAW_ASSET_PATTERNS.get(benchmark.lower(), ["*.parquet", "*.tfrecord"]))
    matching_files: list[str] = []
    if expected_dir.is_dir():
        for pattern in patterns:
            matching_files.extend(str(path) for path in sorted(expected_dir.glob(pattern)) if path.is_file())

    blockers: list[str] = []
    repo_exists = repo.is_dir()
    data_root_exists = root.is_dir()
    dataset_dir_exists = expected_dir.is_dir()
    if not repo_exists:
        blockers.append(f"spatialclaw_repo_not_found:{repo}")
    if repo_exists and not data_root_exists:
        blockers.append(f"spatialclaw_data_root_not_found:{root}")
    if repo_exists and data_root_exists and not dataset_dir_exists:
        blockers.append(f"spatialclaw_dataset_dir_not_found:{expected_dir}")
    if repo_exists and dataset_dir_exists and not matching_files:
        blockers.append(f"spatialclaw_{benchmark.lower()}_assets_missing:{expected_dir}:expected {', '.join(patterns)}")

    loader_import_ok = False
    loader_class: str | None = None
    sample_count: int | None = None
    if run_loader and repo_exists:
        try:
            samples, smoke = load_spatialclaw_samples(repo_path=repo, benchmark=benchmark, data_root=root, limit=1)
            loader_import_ok = smoke.import_ok
            loader_class = smoke.loader_class
            sample_count = smoke.sample_count
            if not samples and smoke.blocker:
                blockers.append(f"spatialclaw_loader_no_samples:{smoke.blocker}")
        except Exception as exc:  # pragma: no cover - depends on optional upstream installation.
            blockers.append(f"spatialclaw_loader_failed:{type(exc).__name__}:{exc}")

    return SpatialClawAssetPreflight(
        repo_path=str(repo),
        benchmark=benchmark,
        data_root=str(root),
        expected_dataset_dir=str(expected_dir),
        expected_file_patterns=patterns,
        repo_exists=repo_exists,
        data_root_exists=data_root_exists,
        dataset_dir_exists=dataset_dir_exists,
        matching_files=matching_files,
        loader_import_ok=loader_import_ok,
        loader_class=loader_class,
        sample_count=sample_count,
        blockers=blockers,
    )


def load_spatialclaw_samples(
    *,
    repo_path: str | Path = DEFAULT_SPATIALCLAW_REPO,
    benchmark: str = "erqa",
    data_root: str | Path | None = None,
    limit: int | None = None,
    question_type: list[str] | None = None,
) -> tuple[list[SpatialSample], SpatialClawRuntimeSmoke]:
    """Load real SpatialClaw benchmark samples through the upstream loader.

    This function does not fabricate samples. If upstream imports work but the
    benchmark assets are absent, it returns an empty list plus an exact blocker.
    """

    repo = Path(repo_path)
    root = Path(data_root) if data_root is not None else repo / "data"
    with _spatialclaw_path(repo):
        factory_module = importlib.import_module("spatial_agent.evals.factory")
        registry = getattr(factory_module, "BENCHMARK_REGISTRY")
        factory = getattr(factory_module, "BenchmarkFactory")
        registry_names = sorted(str(name) for name in registry)
        bench = factory.create_benchmark(
            benchmark,
            data_root=str(root),
            question_type=question_type,
        )
        if bench is None:
            smoke = SpatialClawRuntimeSmoke(
                repo_path=str(repo),
                import_ok=True,
                registry_count=len(registry_names),
                registry_names=registry_names,
                benchmark=benchmark,
                loader_class=None,
                sample_count=0,
                blocker=f"benchmark '{benchmark}' is registered as a sentinel with no loader",
            )
            return [], smoke

        samples: list[SpatialSample] = []
        for index, native_sample in enumerate(bench):
            if limit is not None and index >= limit:
                break
            samples.append(spatialclaw_sample_to_spatial_sample(native_sample, benchmark=benchmark))

        blocker = None
        if not samples:
            expected_dir = Path(bench.data_path) / "data"
            patterns = SPATIALCLAW_ASSET_PATTERNS.get(benchmark.lower(), ["*.parquet", "*.tfrecord"])
            blocker = (
                f"SpatialClaw loader {type(bench).__name__} imported and instantiated, "
                f"but no samples were found under {expected_dir}; expected {', '.join(patterns)}."
            )
        smoke = SpatialClawRuntimeSmoke(
            repo_path=str(repo),
            import_ok=True,
            registry_count=len(registry_names),
            registry_names=registry_names,
            benchmark=benchmark,
            loader_class=type(bench).__name__,
            sample_count=len(samples),
            blocker=blocker,
        )
        return samples, smoke


def spatialclaw_sample_to_spatial_sample(sample: Any, *, benchmark: str) -> SpatialSample:
    views = []
    images = list(getattr(sample, "images", []) or [])
    for index, artifact in enumerate(images):
        view_metadata = _parse_spatialclaw_view_metadata(str(artifact))
        views.append(
            {
                "view_id": f"{benchmark}:{getattr(sample, 'sample_id', index)}:{index}",
                "artifact": str(artifact),
                "media_type": "image_or_frame",
                "frame_index": index,
                **view_metadata,
            }
        )
    image_groups = getattr(sample, "image_groups", None)
    if image_groups:
        for group_index, group in enumerate(image_groups):
            for frame_index, artifact in enumerate(group):
                view_metadata = _parse_spatialclaw_view_metadata(str(artifact))
                views.append(
                    {
                        "view_id": f"{benchmark}:{getattr(sample, 'sample_id', group_index)}:video{group_index}:{frame_index}",
                        "artifact": str(artifact),
                        "media_type": "video_frame",
                        "video_index": group_index,
                        "frame_index": frame_index,
                        **view_metadata,
                    }
                )

    native_public_metadata = _extract_public_native_metadata(sample)
    return SpatialSample(
        sample_id=str(getattr(sample, "sample_id", "unknown")),
        source=f"spatialclaw:{benchmark}",
        instruction=str(getattr(sample, "question", "")),
        answer=str(getattr(sample, "answer", "")),
        category=str(getattr(sample, "question_type", "unknown")),
        views=views,
        choices=_extract_spatialclaw_choices(sample),
        facts=[],
        metadata={
            "benchmark_family": "SpatialClaw",
            "adapter_target": "NVlabs/SpatialClaw",
            "upstream_commit": SPATIALCLAW_UPSTREAM_COMMIT,
            "native_loader_sample_class": type(sample).__name__,
            "answer_type": str(getattr(sample, "answer_type", "")),
            "oracle_leakage_level": "none",
        }
        | native_public_metadata,
    )


def _parse_spatialclaw_view_metadata(artifact: str) -> dict[str, Any]:
    path = Path(str(artifact))
    stem = path.stem
    orientation = None
    for label in ("front", "left", "right", "back"):
        if re.search(rf"(^|[_\-]){label}([_\-]|\d|$)", stem, flags=re.IGNORECASE):
            orientation = label
            break
    filename_index = None
    index_match = re.search(r"(\d+)(?!.*\d)", stem)
    if index_match:
        filename_index = int(index_match.group(1))
    return {
        "view_group": path.parent.name if path.parent.name else None,
        "view_filename": path.name,
        "view_label": orientation,
        "filename_index": filename_index,
    }


def _extract_public_native_metadata(sample: Any) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for attr in ("scene_name", "subset", "sample_type"):
        value = getattr(sample, attr, None)
        if value not in (None, ""):
            metadata[attr] = value
    category = getattr(sample, "category", None)
    if category not in (None, ""):
        metadata["native_category"] = category
    return metadata


def _extract_spatialclaw_choices(sample: Any) -> list[str]:
    choices = getattr(sample, "choices", None)
    if isinstance(choices, dict):
        return [f"{str(key).upper()}: {value}" for key, value in sorted(choices.items(), key=lambda item: str(item[0]))]
    if isinstance(choices, (list, tuple)) and choices:
        labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        normalized = []
        for index, choice in enumerate(choices):
            text = str(choice)
            if re.match(r"^[A-Z][).:]\s+", text, re.IGNORECASE):
                normalized.append(text)
            else:
                label = labels[index] if index < len(labels) else str(index)
                normalized.append(f"{label}: {text}")
        return normalized
    parsed_from_instruction = _parse_spatialclaw_choices(None, str(getattr(sample, "question", "")))
    return [f"{label}: {text}" for label, text in parsed_from_instruction]


class SpatialClawAgentNativeBackend(SpatialDiagnosticBackend):
    """Agent-native SpatialClaw adapter with answer-key-free primitives."""

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        cards = [
            PrimitiveCard(
                name="get_spatial_task_context",
                capability_tags=["task", "metadata", "spatialclaw"],
                input_schema={"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"instruction": "str", "choices": "list[str]", "views": "list[dict]", "evidence": "dict"},
                abstraction_level="L1",
                description="Return SpatialClaw task context without answer-key or evaluator fields.",
            ),
            PrimitiveCard(
                name="inspect_spatial_observation",
                capability_tags=["vision", "view_selection", "spatialclaw"],
                input_schema={"view_id": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"view": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect one SpatialClaw image, frame, or video observation descriptor.",
            ),
            PrimitiveCard(
                name="inspect_spatial_view_set",
                capability_tags=["vision", "multi_view", "metadata", "spatialclaw"],
                input_schema={"view_ids": "list[str]|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"views": "list[dict]", "view_groups": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect public multi-view file, order, and view-label metadata without exposing answer keys.",
            ),
            PrimitiveCard(
                name="inspect_spatial_pixels",
                capability_tags=["vision", "image", "pixels", "bbox", "spatialclaw"],
                input_schema={"view_id": "str", "bbox": "list[int]|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"image": "dict", "full_image": "dict", "requested_bbox": "dict|None", "regions": "list[dict]", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect a local SpatialClaw image artifact and return auditable pixel and optional bbox statistics.",
            ),
            PrimitiveCard(
                name="detect_spatial_image_objects",
                capability_tags=["vision", "image", "DET", "bbox", "open_vocab", "spatialclaw"],
                input_schema={
                    "view_id": "str",
                    "query_labels": "list[str]",
                    "score_threshold": "float",
                    "max_detections": "int",
                    "detector_backend": "str",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={"image": "dict", "detections": "list[dict]", "label_counts": "dict", "measurements": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Run local/open-vocabulary detection on a real SpatialClaw image and return bbox evidence for agent-side reasoning.",
            ),
            PrimitiveCard(
                name="segment_spatial_image_objects",
                capability_tags=["vision", "image", "DET", "SEG", "mask", "spatialclaw"],
                input_schema={
                    "view_id": "str",
                    "query_labels": "list[str]",
                    "score_threshold": "float",
                    "max_detections": "int",
                    "max_segments": "int",
                    "detector_backend": "str",
                    "segmentation_backend": "str",
                    "prompt": "str|None",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={"image": "dict", "detections": "list[dict]", "segments": "list[dict]", "measurements": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Use detector boxes as prompts for local SAM2 segmentation and return mask summaries only.",
            ),
            PrimitiveCard(
                name="analyze_spatialclaw_visual_query",
                capability_tags=["vision", "trajectory", "annotation", "spatialclaw"],
                input_schema={"view_id": "str", "prompt": "str|None", "query": "str|None", "choices": "list[str]|None", "agent_context": "dict|None"},
                output_schema={"annotation_summary": "dict", "choice_scores": "list[dict]", "recommended_choice": "str|None", "evidence": "dict"},
                abstraction_level="L2",
                description="Analyze visible SpatialClaw task annotations such as colored trajectories/markers and score answer choices from image evidence.",
            ),
            PrimitiveCard(
                name="infer_mindcube_motion_from_views",
                capability_tags=["vision", "multi_view", "motion", "spatialclaw", "mindcube"],
                input_schema={"view_ids": "list[str]|None", "choices": "list[str]|None", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"parsed_views": "list[dict]", "motion": "dict", "choice_scores": "list[dict]", "recommended_choice": "str|None", "evidence": "dict"},
                abstraction_level="L2",
                description="Infer a MindCube camera-motion descriptor from public ordered view labels and score visible choices.",
            ),
            PrimitiveCard(
                name="run_geometry_tool",
                capability_tags=["geometry", "measurement", "spatialclaw"],
                input_schema={"tool": "str", "inputs": "dict", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"tool": "str", "result": "object", "evidence": "dict"},
                abstraction_level="L2",
                description="Run a deterministic geometry helper on agent-provided measurements.",
            ),
            PrimitiveCard(
                name="record_spatial_evidence",
                capability_tags=["evidence", "memory", "spatialclaw"],
                input_schema={"key": "str", "value": "object", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"artifact_id": "str", "evidence": "dict"},
                abstraction_level="L2",
                description="Attach SpatialClaw reasoning evidence to the episode trace.",
            ),
            PrimitiveCard(
                name="submit_spatial_answer",
                capability_tags=["commit", "answer", "spatialclaw"],
                input_schema={"answer": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"submission": "dict", "evidence": "dict"},
                abstraction_level="L3",
                description="Commit the agent's final SpatialClaw answer.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards), "adapter": "spatialclaw"})
        return cards

    def _primitive_get_spatial_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        result = self._primitive_get_task_context(prompt=prompt, query=query, agent_context=agent_context)
        return PrimitiveResult(
            name="get_spatial_task_context",
            ok=result.ok,
            output=result.output,
            artifacts=result.artifacts,
            error=result.error,
            metadata=result.metadata,
        )

    def _primitive_inspect_spatial_observation(
        self,
        view_id: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        result = self._primitive_inspect_view(view_id=view_id, query=query, agent_context=agent_context)
        if result.ok:
            result.output["prompt"] = prompt
            result.output["evidence"]["prompt"] = prompt
        return PrimitiveResult(
            name="inspect_spatial_observation",
            ok=result.ok,
            output=result.output,
            artifacts=result.artifacts,
            error=result.error,
            metadata=result.metadata,
        )

    def _primitive_inspect_spatial_view_set(
        self,
        view_ids: list[str] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        selected = _select_spatialclaw_views(self._state.get("views", []), view_ids=view_ids)
        if view_ids and len(selected) != len(view_ids):
            known = {str(view.get("view_id")) for view in self._state.get("views", [])}
            missing = [str(view_id) for view_id in view_ids if str(view_id) not in known]
            return PrimitiveResult(
                name="inspect_spatial_view_set",
                ok=False,
                output={"requested_view_ids": view_ids, "missing_view_ids": missing},
                error=f"unknown_view_ids:{missing}",
            )
        public_views = [_public_view_descriptor(view) for view in selected]
        groups: dict[str, list[str]] = {}
        for view in public_views:
            group = str(view.get("view_group") or "ungrouped")
            groups.setdefault(group, []).append(str(view.get("view_id")))
        evidence = self._request_evidence(
            "inspect_spatial_view_set",
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            extra={"view_count": len(public_views), "view_groups": groups},
        )
        return PrimitiveResult(
            name="inspect_spatial_view_set",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "views": public_views,
                "view_groups": groups,
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_inspect_spatial_pixels(
        self,
        view_id: str,
        bbox: Any = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        descriptor = self._primitive_inspect_view(view_id=view_id, query=query, agent_context=agent_context)
        if not descriptor.ok:
            return PrimitiveResult(
                name="inspect_spatial_pixels",
                ok=False,
                output=descriptor.output,
                artifacts=descriptor.artifacts,
                error=descriptor.error,
                metadata=descriptor.metadata,
            )

        view = descriptor.output.get("view", {})
        evidence = dict(descriptor.output.get("evidence", {}))
        evidence["primitive"] = "inspect_spatial_pixels"
        evidence["prompt"] = prompt
        pixel_result = _inspect_spatialclaw_image_pixels(
            view=view,
            bbox=bbox,
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            evidence=evidence,
        )
        return PrimitiveResult(
            name="inspect_spatial_pixels",
            ok=bool(pixel_result.pop("ok")),
            output=pixel_result,
            artifacts=descriptor.artifacts,
            error=pixel_result.get("error"),
        )

    def _primitive_detect_spatial_image_objects(
        self,
        view_id: str,
        query_labels: list[str],
        score_threshold: float = 0.02,
        max_detections: int = 20,
        detector_backend: str = "owl_vit",
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        descriptor = self._primitive_inspect_view(view_id=view_id, query=query, agent_context=agent_context)
        if not descriptor.ok:
            return PrimitiveResult(
                name="detect_spatial_image_objects",
                ok=False,
                output=descriptor.output,
                artifacts=descriptor.artifacts,
                error=descriptor.error,
                metadata=descriptor.metadata,
            )
        view = descriptor.output.get("view", {})
        evidence = dict(descriptor.output.get("evidence", {}))
        evidence["primitive"] = "detect_spatial_image_objects"
        evidence["prompt"] = prompt
        detection = _detect_spatialclaw_image_objects(
            view=view,
            query_labels=query_labels,
            score_threshold=score_threshold,
            max_detections=max_detections,
            detector_backend=detector_backend,
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            evidence=evidence,
        )
        return PrimitiveResult(
            name="detect_spatial_image_objects",
            ok=bool(detection.pop("ok")),
            output=detection,
            artifacts=descriptor.artifacts,
            error=detection.get("error"),
        )

    def _primitive_segment_spatial_image_objects(
        self,
        view_id: str,
        query_labels: list[str],
        score_threshold: float = 0.02,
        max_detections: int = 12,
        max_segments: int = 8,
        detector_backend: str = "owl_vit",
        segmentation_backend: str = "sam2",
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        descriptor = self._primitive_inspect_view(view_id=view_id, query=query, agent_context=agent_context)
        if not descriptor.ok:
            return PrimitiveResult(
                name="segment_spatial_image_objects",
                ok=False,
                output=descriptor.output,
                artifacts=descriptor.artifacts,
                error=descriptor.error,
                metadata=descriptor.metadata,
            )
        view = descriptor.output.get("view", {})
        evidence = dict(descriptor.output.get("evidence", {}))
        evidence["primitive"] = "segment_spatial_image_objects"
        evidence["prompt"] = prompt
        segmentation = _segment_spatialclaw_image_objects(
            view=view,
            query_labels=query_labels,
            score_threshold=score_threshold,
            max_detections=max_detections,
            max_segments=max_segments,
            detector_backend=detector_backend,
            segmentation_backend=segmentation_backend,
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            evidence=evidence,
        )
        return PrimitiveResult(
            name="segment_spatial_image_objects",
            ok=bool(segmentation.pop("ok")),
            output=segmentation,
            artifacts=descriptor.artifacts,
            error=segmentation.get("error"),
        )

    def _primitive_analyze_spatialclaw_visual_query(
        self,
        view_id: str,
        prompt: str | None = None,
        query: str | None = None,
        choices: Any = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        descriptor = self._primitive_inspect_view(view_id=view_id, query=query, agent_context=agent_context)
        if not descriptor.ok:
            return PrimitiveResult(
                name="analyze_spatialclaw_visual_query",
                ok=False,
                output=descriptor.output,
                artifacts=descriptor.artifacts,
                error=descriptor.error,
                metadata=descriptor.metadata,
            )
        view = descriptor.output.get("view", {})
        evidence = dict(descriptor.output.get("evidence", {}))
        evidence["primitive"] = "analyze_spatialclaw_visual_query"
        evidence["prompt"] = prompt
        output = _analyze_spatialclaw_visual_query(
            view=view,
            instruction=self._sample.instruction if self._sample is not None else query,
            prompt=prompt,
            query=query,
            choices=choices,
            agent_context=agent_context,
            evidence=evidence,
        )
        return PrimitiveResult(
            name="analyze_spatialclaw_visual_query",
            ok=bool(output.pop("ok")),
            output=output,
            artifacts=descriptor.artifacts,
            error=output.get("error"),
        )

    def _primitive_infer_mindcube_motion_from_views(
        self,
        view_ids: list[str] | None = None,
        choices: Any = None,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        selected = _select_spatialclaw_views(self._state.get("views", []), view_ids=view_ids)
        if len(selected) < 2:
            return PrimitiveResult(
                name="infer_mindcube_motion_from_views",
                ok=False,
                output={"view_count": len(selected), "required_view_count": 2},
                error="mindcube_motion_requires_at_least_two_views",
            )
        parsed_views = [_public_view_descriptor(view) for view in selected]
        output = _infer_mindcube_motion_from_public_views(
            parsed_views=parsed_views,
            choices=choices if choices is not None else (self._sample.choices if self._sample is not None else []),
            prompt=prompt,
            query=query or (self._sample.instruction if self._sample is not None else None),
            agent_context=agent_context,
        )
        evidence = self._request_evidence(
            "infer_mindcube_motion_from_views",
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            extra={
                "view_count": len(parsed_views),
                "motion_descriptor": output["motion"].get("descriptor"),
                "recommended_choice": output.get("recommended_choice"),
            },
        )
        output["evidence"] = evidence
        return PrimitiveResult(
            name="infer_mindcube_motion_from_views",
            ok=bool(output.pop("ok")),
            output=output,
            artifacts=[evidence["artifact_id"]],
            error=output.get("error"),
        )

    def _primitive_run_geometry_tool(
        self,
        tool: str,
        inputs: dict[str, Any],
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        try:
            value = _run_geometry_tool(tool, inputs)
        except (KeyError, TypeError, ValueError) as exc:
            return PrimitiveResult(name="run_geometry_tool", ok=False, error=f"{type(exc).__name__}: {exc}")
        evidence = self._request_evidence(
            "run_geometry_tool",
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            extra={"tool": tool, "inputs": inputs, "result": value},
        )
        return PrimitiveResult(
            name="run_geometry_tool",
            ok=True,
            output={"tool": tool, "result": value, "prompt": prompt, "query": query, "agent_context": agent_context or {}, "evidence": evidence},
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_record_spatial_evidence(
        self,
        key: str,
        value: Any,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"spatialclaw_evidence:{key}"
        payload = {"key": key, "value": value, "prompt": prompt, "query": query, "agent_context": agent_context or {}}
        self._state.setdefault("evidence", {})[key] = value
        self.get_trace().add_artifact(artifact_id, payload)
        evidence = {
            "artifact_id": artifact_id,
            "source": "spatialclaw_agent_evidence",
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
        }
        return PrimitiveResult(
            name="record_spatial_evidence",
            ok=True,
            output={"artifact_id": artifact_id, "evidence": evidence},
            artifacts=[artifact_id],
        )

    def _primitive_submit_spatial_answer(
        self,
        answer: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        result = self._primitive_submit_answer(answer=answer, agent_context=agent_context)
        if result.ok:
            result.output["prompt"] = prompt
            result.output["query"] = query
            result.output["evidence"]["prompt"] = prompt
            result.output["evidence"]["query"] = query
        return PrimitiveResult(
            name="submit_spatial_answer",
            ok=result.ok,
            output=result.output,
            artifacts=result.artifacts,
            error=result.error,
            metadata=result.metadata,
        )


def _run_geometry_tool(tool: str, inputs: dict[str, Any]) -> Any:
    if tool == "distance_2d":
        p1 = inputs["p1"]
        p2 = inputs["p2"]
        return math.dist([float(p1[0]), float(p1[1])], [float(p2[0]), float(p2[1])])
    if tool == "bbox_center":
        x1, y1, x2, y2 = [float(v) for v in inputs["bbox"]]
        return {"x": (x1 + x2) / 2.0, "y": (y1 + y2) / 2.0}
    if tool == "compare_depth_order":
        a = float(inputs["a_depth"])
        b = float(inputs["b_depth"])
        if math.isclose(a, b):
            return "similar_depth"
        return "a_closer" if a < b else "b_closer"
    if tool == "bbox_spatial_relation":
        return _bbox_spatial_relation(
            subject_bbox=inputs["subject_bbox"],
            reference_bbox=inputs["reference_bbox"],
            image_size=inputs.get("image_size"),
        )
    if tool == "ratio":
        metric = str(inputs.get("metric", "value"))
        numerator = _geometry_numeric_value(inputs.get("numerator", inputs.get("a")), metric=metric)
        denominator = _geometry_numeric_value(inputs.get("denominator", inputs.get("b")), metric=metric)
        if math.isclose(denominator, 0.0):
            raise ValueError("ratio denominator is zero")
        return {
            "metric": metric,
            "numerator": round(float(numerator), 6),
            "denominator": round(float(denominator), 6),
            "ratio": round(float(numerator) / float(denominator), 6),
        }
    raise ValueError(f"unsupported geometry tool: {tool}")


def _select_spatialclaw_views(views: list[dict[str, Any]], view_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if not view_ids:
        return [dict(view) for view in views]
    wanted = [str(view_id) for view_id in view_ids]
    by_id = {str(view.get("view_id")): dict(view) for view in views}
    return [by_id[view_id] for view_id in wanted if view_id in by_id]


def _public_view_descriptor(view: dict[str, Any]) -> dict[str, Any]:
    public_keys = (
        "view_id",
        "artifact",
        "media_type",
        "frame_index",
        "video_index",
        "view_group",
        "view_filename",
        "view_label",
        "filename_index",
    )
    return {key: view.get(key) for key in public_keys if key in view}


def _detect_spatialclaw_image_objects(
    *,
    view: dict[str, Any],
    query_labels: list[str],
    score_threshold: float,
    max_detections: int,
    detector_backend: str,
    prompt: str | None,
    query: str | None,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    artifact_value = str(view.get("artifact", ""))
    artifact_path, path_blocker = _local_image_artifact_path(artifact_value)
    if path_blocker is not None or not artifact_path.is_file():
        blocker = path_blocker or f"image_artifact_not_found:{artifact_path}"
        return {
            "ok": False,
            "error": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": artifact_value, "exists": False},
            "detections": [],
            "label_counts": {},
            "measurements": {"object_count_estimate": 0},
            "evidence": dict(evidence)
            | {
                "detection": False,
                "blocker": blocker,
                "query": query,
                "agent_context": agent_context or {},
            },
        }

    backend = _normalize_detector_backend(detector_backend)
    if _is_open_vocabulary_detector_backend(backend):
        preflight = _preflight_video_grounding_model(model_hint=backend)
        if not preflight.get("ready"):
            blocker = "open_vocab_detector_not_ready:" + ",".join(str(item) for item in preflight.get("blockers", []))
            return {
                "ok": False,
                "error": blocker,
                "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": True},
                "detections": [],
                "label_counts": {},
                "measurements": {"object_count_estimate": 0},
                "evidence": dict(evidence)
                | {
                    "detection": False,
                    "blocker": blocker,
                    "detector_preflight": preflight,
                    "query": query,
                    "agent_context": agent_context or {},
                },
            }

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency.
        blocker = f"image_detection_dependency_unavailable:{exc}"
        return {
            "ok": False,
            "error": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": True},
            "detections": [],
            "label_counts": {},
            "measurements": {"object_count_estimate": 0},
            "evidence": dict(evidence) | {"detection": False, "blocker": blocker},
        }

    try:
        with Image.open(artifact_path) as image:
            rgb_image = image.convert("RGB")
            width, height = rgb_image.size
            rgb = np.asarray(rgb_image)
        frames = [{"index": 0, "rgb": rgb}]
        if _is_open_vocabulary_detector_backend(backend):
            detection_result = _detect_rgb_frames_with_owl_vit(
                frames=frames,
                query_labels=query_labels,
                score_threshold=float(score_threshold),
                max_detections_per_frame=int(max_detections),
                model_hint=backend,
            )
        else:
            detection_result = _detect_rgb_frames_with_torchvision(
                frames=frames,
                query_labels=query_labels,
                score_threshold=float(score_threshold),
                max_detections_per_frame=int(max_detections),
            )
    except Exception as exc:
        blocker = f"image_detection_failed:{type(exc).__name__}:{exc}"
        return {
            "ok": False,
            "error": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": True},
            "detections": [],
            "label_counts": {},
            "measurements": {"object_count_estimate": 0},
            "evidence": dict(evidence)
            | {
                "detection": False,
                "blocker": blocker,
                "query": query,
                "agent_context": agent_context or {},
            },
        }

    detections = [_summarize_spatial_detection(image_size=(width, height), detection=item) for item in detection_result["detections"]]
    return {
        "ok": True,
        "prompt": prompt,
        "query": query,
        "agent_context": agent_context or {},
        "image": {
            "view_id": view.get("view_id"),
            "artifact": str(artifact_path),
            "exists": True,
            "width": width,
            "height": height,
            "media_type": view.get("media_type"),
        },
        "detector": detection_result.get("detector", {}),
        "query_labels": detection_result.get("query_labels", []),
        "detections": detections,
        "label_counts": detection_result.get("label_counts", {}),
        "frame_summaries": detection_result.get("frame_summaries", {}),
        "measurements": {
            "object_count_estimate": len(detections),
            "detected_labels": sorted({str(item.get("label")) for item in detections}),
            "max_score": max((float(item.get("score", 0.0)) for item in detections), default=0.0),
        },
        "evidence": dict(evidence)
        | {
            "detection": True,
            "artifact": str(artifact_path),
            "query": query,
            "agent_context": agent_context or {},
            "detector_backend": backend,
            "query_labels": list(query_labels or []),
        },
    }


def _segment_spatialclaw_image_objects(
    *,
    view: dict[str, Any],
    query_labels: list[str],
    score_threshold: float,
    max_detections: int,
    max_segments: int,
    detector_backend: str,
    segmentation_backend: str,
    prompt: str | None,
    query: str | None,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    detection = _detect_spatialclaw_image_objects(
        view=view,
        query_labels=query_labels,
        score_threshold=score_threshold,
        max_detections=max_detections,
        detector_backend=detector_backend,
        prompt=prompt,
        query=query,
        agent_context=agent_context,
        evidence=evidence,
    )
    if not detection.get("ok"):
        detection["segments"] = []
        return detection

    preflight = _preflight_video_segmentation_model(model_hint=segmentation_backend)
    if not preflight.get("ready"):
        blocker = "segmentation_model_not_ready:" + ",".join(str(item) for item in preflight.get("blockers", []))
        return detection | {
            "ok": False,
            "error": blocker,
            "segments": [],
            "evidence": dict(detection.get("evidence", {}))
            | {
                "segmentation": False,
                "blocker": blocker,
                "segmentation_preflight": preflight,
            },
        }

    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency.
        blocker = f"image_segmentation_dependency_unavailable:{exc}"
        return detection | {"ok": False, "error": blocker, "segments": []}

    artifact_path = Path(str(detection["image"]["artifact"]))
    try:
        with Image.open(artifact_path) as image:
            rgb = np.asarray(image.convert("RGB"))
        predictor = _load_sam2_image_predictor(preflight)
        predictor.set_image(rgb)
        segments: list[dict[str, Any]] = []
        for index, item in enumerate(detection.get("detections", [])[: max(1, int(max_segments))]):
            bbox = [float(value) for value in item.get("bbox_xyxy", [])]
            if len(bbox) != 4:
                continue
            masks, scores, _logits = predictor.predict(box=np.asarray(bbox, dtype=np.float32), multimask_output=True)
            best_index = int(np.argmax(scores)) if len(scores) else 0
            mask_summary = _summarize_mask_array(masks[best_index] if len(masks) else np.zeros(rgb.shape[:2], dtype=bool))
            segments.append(
                {
                    "segment_id": f"{view.get('view_id')}:{index}",
                    "label": item.get("label"),
                    "score": item.get("score"),
                    "bbox_xyxy": item.get("bbox_xyxy"),
                    "mask_score": round(float(scores[best_index]), 6) if len(scores) else None,
                    **mask_summary,
                }
            )
    except Exception as exc:
        blocker = f"image_segmentation_failed:{type(exc).__name__}:{exc}"
        return detection | {
            "ok": False,
            "error": blocker,
            "segments": [],
            "evidence": dict(detection.get("evidence", {})) | {"segmentation": False, "blocker": blocker},
        }

    detection["segments"] = segments
    detection["measurements"] = dict(detection.get("measurements", {})) | {
        "segment_count": len(segments),
        "nonempty_segment_count": sum(1 for item in segments if item.get("mask_nonempty")),
    }
    detection["evidence"] = dict(detection.get("evidence", {})) | {
        "segmentation": True,
        "segmentation_backend": segmentation_backend,
        "max_segments": int(max_segments),
    }
    return detection


def _summarize_spatial_detection(*, image_size: tuple[int, int], detection: dict[str, Any]) -> dict[str, Any]:
    width, height = image_size
    bbox = [round(float(value), 3) for value in detection.get("bbox_xyxy", [])]
    center = _bbox_center(bbox) if len(bbox) == 4 else [None, None]
    area = _bbox_area(bbox) if len(bbox) == 4 else 0.0
    bbox_width = max(0.0, bbox[2] - bbox[0]) if len(bbox) == 4 else 0.0
    bbox_height = max(0.0, bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0
    return dict(detection) | {
        "bbox_xyxy": bbox,
        "bbox_width": round(float(bbox_width), 3),
        "bbox_height": round(float(bbox_height), 3),
        "bbox_area": round(float(area), 3),
        "bbox_area_fraction": round(float(area) / max(1.0, float(width * height)), 6),
        "center_xy": [round(float(center[0]), 3), round(float(center[1]), 3)] if None not in center else center,
    }


def _infer_mindcube_motion_from_public_views(
    *,
    parsed_views: list[dict[str, Any]],
    choices: Any,
    prompt: str | None,
    query: str | None,
    agent_context: dict[str, Any] | None,
) -> dict[str, Any]:
    first = parsed_views[0] if parsed_views else {}
    second = parsed_views[1] if len(parsed_views) > 1 else {}
    pair = (str(first.get("view_label") or "").lower(), str(second.get("view_label") or "").lower())
    mapping = {
        ("front", "left"): "diagonally forward and left",
        ("front", "right"): "diagonally forward and right",
        ("left", "front"): "diagonally forward and right",
        ("right", "front"): "diagonally forward and left",
        ("back", "left"): "diagonally backward and left",
        ("back", "right"): "diagonally backward and right",
        ("left", "back"): "diagonally backward and right",
        ("right", "back"): "diagonally backward and left",
        ("front", "back"): "directly backward",
        ("back", "front"): "directly forward",
        ("left", "right"): "directly right",
        ("right", "left"): "directly left",
    }
    descriptor = mapping.get(pair)
    if descriptor is None:
        first_index = first.get("filename_index")
        second_index = second.get("filename_index")
        if isinstance(first_index, int) and isinstance(second_index, int) and first_index != second_index:
            descriptor = "viewpoint changed, but public view labels are ambiguous"
        else:
            descriptor = "unknown"
    label_choices = _parse_spatialclaw_choices(choices, query)
    choice_scores = _score_mindcube_motion_choices(label_choices, descriptor=descriptor)
    recommended = choice_scores[0]["label"] if choice_scores and choice_scores[0]["score"] > 0 else None
    return {
        "ok": descriptor != "unknown",
        "prompt": prompt,
        "query": query,
        "agent_context": agent_context or {},
        "parsed_views": parsed_views,
        "motion": {
            "first_view_label": pair[0] or None,
            "second_view_label": pair[1] or None,
            "descriptor": descriptor,
            "public_signal": "ordered_view_filename_orientation",
        },
        "choice_scores": choice_scores,
        "recommended_choice": recommended,
    }


def _score_mindcube_motion_choices(choices: list[tuple[str, str]], *, descriptor: str) -> list[dict[str, Any]]:
    target_tokens = {token for token in re.split(r"[^a-z]+", descriptor.lower()) if token and token not in {"and"}}
    scored: list[dict[str, Any]] = []
    for label, text in choices:
        normalized = text.lower()
        choice_tokens = {token for token in re.split(r"[^a-z]+", normalized) if token and token not in {"and"}}
        score = float(len(target_tokens & choice_tokens))
        reasons = []
        if normalized.strip() == descriptor.lower():
            score += 10.0
            reasons.append("choice text exactly matches public view-label motion descriptor")
        elif target_tokens and target_tokens.issubset(choice_tokens):
            score += 4.0
            reasons.append("choice contains all descriptor direction tokens")
        elif score > 0:
            reasons.append("choice shares direction tokens with descriptor")
        scored.append({"label": label, "text": text, "score": round(score, 3), "reasons": reasons})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _bbox_spatial_relation(subject_bbox: Any, reference_bbox: Any, image_size: Any | None = None) -> dict[str, Any]:
    subject = _coerce_bbox(subject_bbox, name="subject_bbox")
    reference = _coerce_bbox(reference_bbox, name="reference_bbox")
    subject_center = _bbox_center(subject)
    reference_center = _bbox_center(reference)
    dx = float(subject_center[0]) - float(reference_center[0])
    dy = float(subject_center[1]) - float(reference_center[1])
    horizontal = "same-horizontal" if math.isclose(dx, 0.0, abs_tol=1e-6) else ("right" if dx > 0 else "left")
    depth = "same-depth" if math.isclose(dy, 0.0, abs_tol=1e-6) else ("front" if dy < 0 else "back")
    compound = f"{horizontal}-{depth}" if "same" not in horizontal and "same" not in depth else f"{horizontal}:{depth}"
    result: dict[str, Any] = {
        "subject_center_xy": [round(float(subject_center[0]), 3), round(float(subject_center[1]), 3)],
        "reference_center_xy": [round(float(reference_center[0]), 3), round(float(reference_center[1]), 3)],
        "delta_xy": [round(dx, 3), round(dy, 3)],
        "horizontal": horizontal,
        "depth": depth,
        "compound_relation": compound,
        "heuristic": "image_x_right_and_image_y_up_as_camera_front",
    }
    if image_size:
        result["image_size"] = [int(image_size[0]), int(image_size[1])]
        result["normalized_delta_xy"] = [
            round(dx / max(1.0, float(image_size[0])), 6),
            round(dy / max(1.0, float(image_size[1])), 6),
        ]
    return result


def _geometry_numeric_value(value: Any, *, metric: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        if metric in value:
            return float(value[metric])
        if "bbox_xyxy" in value:
            value = value["bbox_xyxy"]
        elif "bbox" in value:
            value = value["bbox"]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        bbox = _coerce_bbox(value, name="bbox")
        if metric == "width":
            return max(0.0, bbox[2] - bbox[0])
        if metric == "height":
            return max(0.0, bbox[3] - bbox[1])
        return _bbox_area(bbox)
    raise ValueError(f"cannot convert value to numeric metric '{metric}': {value!r}")


def _coerce_bbox(value: Any, *, name: str) -> list[float]:
    if isinstance(value, dict):
        if "bbox_xyxy" in value:
            value = value["bbox_xyxy"]
        elif "bbox" in value:
            value = value["bbox"]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{name} must be a four-item bbox")
    return [float(item) for item in value]


def _inspect_spatialclaw_image_pixels(
    *,
    view: dict[str, Any],
    bbox: Any,
    prompt: str | None,
    query: str | None,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    artifact_value = str(view.get("artifact", ""))
    artifact_path, path_blocker = _local_image_artifact_path(artifact_value)
    if path_blocker is not None:
        return {
            "ok": False,
            "error": path_blocker,
            "blocker": path_blocker,
            "image": {"view_id": view.get("view_id"), "artifact": artifact_value, "exists": False},
            "full_image": None,
            "requested_bbox": None,
            "regions": [],
            "evidence": dict(evidence)
            | {
                "pixel_inspection": False,
                "blocker": path_blocker,
                "artifact": artifact_value,
                "query": query,
                "agent_context": agent_context or {},
            },
        }

    if not artifact_path.is_file():
        blocker = f"image_artifact_not_found:{artifact_path}"
        return {
            "ok": False,
            "error": blocker,
            "blocker": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": False},
            "full_image": None,
            "requested_bbox": None,
            "regions": [],
            "evidence": dict(evidence)
            | {
                "pixel_inspection": False,
                "blocker": blocker,
                "artifact": str(artifact_path),
                "query": query,
                "agent_context": agent_context or {},
            },
        }

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - Pillow is optional outside image inspection.
        blocker = f"pillow_unavailable:{exc}"
        return {
            "ok": False,
            "error": blocker,
            "blocker": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": True},
            "full_image": None,
            "requested_bbox": None,
            "regions": [],
            "evidence": dict(evidence)
            | {
                "pixel_inspection": False,
                "blocker": blocker,
                "artifact": str(artifact_path),
                "query": query,
                "agent_context": agent_context or {},
            },
        }

    try:
        with Image.open(artifact_path) as image:
            image.load()
            rgb = image.convert("RGB")
            width, height = rgb.size
            try:
                requested_bbox_tuple = _normalize_spatialclaw_bbox(bbox, width=width, height=height)
            except (TypeError, ValueError) as exc:
                blocker = f"invalid_bbox:{exc}"
                return {
                    "ok": False,
                    "error": blocker,
                    "blocker": blocker,
                    "image": {
                        "view_id": view.get("view_id"),
                        "artifact": str(artifact_path),
                        "exists": True,
                        "format": image.format,
                        "mode": image.mode,
                        "width": width,
                        "height": height,
                        "media_type": view.get("media_type"),
                    },
                    "full_image": None,
                    "requested_bbox": None,
                    "regions": [],
                    "evidence": dict(evidence)
                    | {
                        "pixel_inspection": False,
                        "blocker": blocker,
                        "artifact": str(artifact_path),
                        "query": query,
                        "agent_context": agent_context or {},
                    },
                }

            full_image = _summarize_spatialclaw_region(rgb, name="full_image", bbox=(0, 0, width, height))
            requested_bbox = None
            regions = [full_image]
            if requested_bbox_tuple != (0, 0, width, height):
                requested_bbox = _summarize_spatialclaw_region(rgb, name="requested_bbox", bbox=requested_bbox_tuple)
                regions.append(requested_bbox)

            return {
                "ok": True,
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "image": {
                    "view_id": view.get("view_id"),
                    "artifact": str(artifact_path),
                    "exists": True,
                    "format": image.format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "media_type": view.get("media_type"),
                },
                "bbox": list(requested_bbox_tuple),
                "full_image": full_image,
                "requested_bbox": requested_bbox,
                "regions": regions,
                "evidence": dict(evidence)
                | {
                    "pixel_inspection": True,
                    "artifact": str(artifact_path),
                    "bbox": list(requested_bbox_tuple),
                    "query": query,
                    "agent_context": agent_context or {},
                },
            }
    except Exception as exc:
        blocker = f"image_decode_failed:{type(exc).__name__}:{exc}"
        return {
            "ok": False,
            "error": blocker,
            "blocker": blocker,
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "exists": True},
            "full_image": None,
            "requested_bbox": None,
            "regions": [],
            "evidence": dict(evidence)
            | {
                "pixel_inspection": False,
                "blocker": blocker,
                "artifact": str(artifact_path),
                "query": query,
                "agent_context": agent_context or {},
            },
        }


def _local_image_artifact_path(artifact: str) -> tuple[Path, str | None]:
    if not artifact:
        return Path(), "image_artifact_not_found:"
    parsed = urlparse(artifact)
    if parsed.scheme and parsed.scheme != "file":
        return Path(), f"non_local_image_artifact:{artifact}"
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            return Path(), f"non_local_image_artifact:{artifact}"
        return Path(unquote(parsed.path)), None
    return Path(artifact), None


def _analyze_spatialclaw_visual_query(
    *,
    view: dict[str, Any],
    instruction: str | None,
    prompt: str | None,
    query: str | None,
    choices: Any,
    agent_context: dict[str, Any] | None,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    artifact_value = str(view.get("artifact", ""))
    artifact_path, path_blocker = _local_image_artifact_path(artifact_value)
    if path_blocker is not None or not artifact_path.is_file():
        blocker = path_blocker or f"image_artifact_not_found:{artifact_path}"
        return {
            "ok": False,
            "error": blocker,
            "annotation_summary": {"artifact": artifact_value, "exists": False},
            "choice_scores": [],
            "recommended_choice": None,
            "evidence": dict(evidence)
            | {
                "annotation_analysis": False,
                "blocker": blocker,
                "query": query,
                "agent_context": agent_context or {},
            },
        }
    try:
        from PIL import Image
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional outside visual tests.
        return {
            "ok": False,
            "error": f"visual_analysis_dependency_unavailable:{exc}",
            "annotation_summary": {"artifact": str(artifact_path), "exists": True},
            "choice_scores": [],
            "recommended_choice": None,
            "evidence": dict(evidence),
        }

    with Image.open(artifact_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    masks = {
        "yellow_trajectory": (
            (rgb[:, :, 0] > 130)
            & (rgb[:, :, 1] > 100)
            & (rgb[:, :, 2] < 95)
            & ((rgb[:, :, 0].astype("int16") - rgb[:, :, 2].astype("int16")) > 55)
        ),
        "green_marker": (rgb[:, :, 1] > 100) & (rgb[:, :, 0] < 130) & (rgb[:, :, 2] < 130),
        "purple_marker": (rgb[:, :, 0] > 80) & (rgb[:, :, 2] > 95) & (rgb[:, :, 1] < 110),
    }
    annotations = {name: _summarize_boolean_mask(mask, width=width, height=height) for name, mask in masks.items()}
    trajectory = annotations["yellow_trajectory"]
    green = annotations["green_marker"]
    purple = annotations["purple_marker"]
    relation = _spatialclaw_annotation_relation(trajectory=trajectory, green=green, purple=purple, width=width, height=height)
    label_choices = _parse_spatialclaw_choices(choices, instruction)
    choice_scores = _score_spatialclaw_visual_choices(label_choices, relation=relation, instruction=instruction or query or "")
    recommended = choice_scores[0]["label"] if choice_scores else None
    return {
        "ok": bool(trajectory["pixel_count"] > 0),
        "prompt": prompt,
        "query": query,
        "agent_context": agent_context or {},
        "annotation_summary": {
            "image": {"view_id": view.get("view_id"), "artifact": str(artifact_path), "width": width, "height": height},
            "annotations": annotations,
            "relation": relation,
            "method": "deterministic_rgb_annotation_masks",
        },
        "choice_scores": choice_scores,
        "recommended_choice": recommended,
        "evidence": dict(evidence)
        | {
            "annotation_analysis": True,
            "artifact": str(artifact_path),
            "query": query,
            "agent_context": agent_context or {},
            "detected_annotation_names": [name for name, item in annotations.items() if item["pixel_count"] > 0],
        },
    }


def _summarize_boolean_mask(mask: Any, *, width: int, height: int) -> dict[str, Any]:
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return {"pixel_count": 0, "bbox": None, "center_xy": None, "normalized_center_xy": None}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    center = [round(float(xs.mean()), 3), round(float(ys.mean()), 3)]
    return {
        "pixel_count": int(len(xs)),
        "bbox": bbox,
        "center_xy": center,
        "normalized_center_xy": [round(center[0] / max(1, width), 4), round(center[1] / max(1, height), 4)],
    }


def _spatialclaw_annotation_relation(
    *,
    trajectory: dict[str, Any],
    green: dict[str, Any],
    purple: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    relation = {
        "trajectory_present": trajectory["pixel_count"] > 0,
        "green_marker_present": green["pixel_count"] > 0,
        "purple_marker_present": purple["pixel_count"] > 0,
        "trajectory_region": "unknown",
        "trajectory_connects_markers": False,
        "endpoint_interpretation": "unknown",
    }
    bbox = trajectory.get("bbox")
    if bbox:
        x0, y0, x1, y1 = bbox
        horizontal = "left" if (x0 + x1) / 2 < width / 3 else "right" if (x0 + x1) / 2 > width * 2 / 3 else "center"
        vertical = "top" if (y0 + y1) / 2 < height / 3 else "bottom" if (y0 + y1) / 2 > height * 2 / 3 else "middle"
        relation["trajectory_region"] = f"{vertical}_{horizontal}"
        relation["trajectory_bbox"] = bbox
    if green.get("center_xy") and purple.get("center_xy"):
        gx, gy = green["center_xy"]
        px, py = purple["center_xy"]
        relation["marker_vector_green_to_purple_xy"] = [round(float(px - gx), 3), round(float(py - gy), 3)]
        relation["trajectory_connects_markers"] = trajectory["pixel_count"] > 0
        relation["endpoint_interpretation"] = (
            "yellow path links the green object marker toward the purple target marker near the upper-left wooden-step area"
            if px < gx and py < gy
            else "yellow path links visible colored markers"
        )
    return relation


def _parse_spatialclaw_choices(choices: Any, instruction: str | None) -> list[tuple[str, str]]:
    if isinstance(choices, (list, tuple)) and choices:
        raw_choices = [str(choice) for choice in choices]
    else:
        text = str(instruction or "")
        raw_choices = [f"{label}. {body.strip()}" for label, body in re.findall(r"\b([A-D])\.\s*(.*?)(?=\s+[A-D]\.\s*|Please answer|$)", text)]
    parsed = []
    for index, raw in enumerate(raw_choices):
        match = re.match(r"\s*([A-Z])\s*[:\).\uff0e]\s*(.*)", raw)
        if match:
            parsed.append((match.group(1), match.group(2).strip()))
        else:
            parsed.append((chr(ord("A") + index), raw.strip()))
    return parsed


def _score_spatialclaw_visual_choices(
    choices: list[tuple[str, str]],
    *,
    relation: dict[str, Any],
    instruction: str,
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    endpoint = str(relation.get("endpoint_interpretation", "")).lower()
    connects_steps = "wooden-step" in endpoint or "wooden step" in endpoint or "wooden" in endpoint
    for label, text in choices:
        normalized = text.lower()
        score = 0.0
        reasons: list[str] = []
        if connects_steps and "wooden steps" in normalized:
            score += 2.0
            reasons.append("detected yellow trajectory ends toward the wooden-step target marker")
        if connects_steps and "puts" in normalized:
            score += 0.75
            reasons.append("trajectory connects object marker to placement target rather than only lifting")
        if "front of" in normalized:
            score -= 0.5
            reasons.append("annotation target is on/at the wooden steps rather than clearly in front")
        if "very top" in normalized:
            score -= 0.25
            reasons.append("detected target marker is on the step area, not specifically the very top")
        if "moves it up" in normalized or "moves it up" in instruction.lower() and "moves it up" in normalized:
            score -= 0.25
            reasons.append("yellow trajectory has a placement target, not just upward motion")
        scored.append({"label": label, "text": text, "score": round(score, 3), "reasons": reasons})
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def _spatialclaw_expected_dataset_dir(data_root: Path, benchmark: str) -> Path:
    dataset_dir, asset_subdir = SPATIALCLAW_DATASET_DIRS.get(benchmark.lower(), (benchmark.upper(), "data"))
    path = data_root / dataset_dir
    if asset_subdir:
        path = path / asset_subdir
    return path


def _normalize_spatialclaw_bbox(bbox: Any, *, width: int, height: int) -> tuple[int, int, int, int]:
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


def _summarize_spatialclaw_region(image: Any, *, name: str, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    from PIL import ImageStat

    region = image.crop(bbox)
    stat = ImageStat.Stat(region)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return {
        "name": name,
        "bbox": list(bbox),
        "shape": [height, width, 3],
        "width": width,
        "height": height,
        "area_pixels": width * height,
        "channels": ["R", "G", "B"],
        "mean_rgb": [round(value, 3) for value in stat.mean],
        "extrema_rgb": [[int(low), int(high)] for low, high in stat.extrema],
    }


@contextmanager
def _spatialclaw_path(repo_path: Path) -> Iterator[None]:
    repo = str(repo_path.resolve())
    inserted = False
    if repo not in sys.path:
        sys.path.insert(0, repo)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(repo)
            except ValueError:
                pass
