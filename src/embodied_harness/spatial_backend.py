from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from .backend import EmbodiedBackend
from .paths import get_project_paths
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


DEFAULT_SPATIAL_FIXTURE = Path("benchmarks/non_operation/esi/spatial_tasks.jsonl")
_SAM2_IMAGE_PREDICTOR_CACHE: dict[tuple[str, str, str, str], Any] = {}


@dataclass(slots=True)
class SpatialSample:
    sample_id: str
    source: str
    instruction: str
    answer: str
    category: str
    views: list[dict[str, Any]]
    choices: list[str]
    facts: list[dict[str, Any]]
    metadata: dict[str, Any]

    @property
    def task_id(self) -> str:
        safe_source = _safe_id(self.source)
        safe_sample = _safe_id(self.sample_id)
        return f"spatial:{safe_source}:{safe_sample}"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SpatialSample":
        return cls(
            sample_id=str(data["sample_id"]),
            source=str(data["source"]),
            instruction=str(data["instruction"]),
            answer=str(data["answer"]),
            category=str(data.get("category", "unknown")),
            views=list(data.get("views", [])),
            choices=[str(choice) for choice in data.get("choices", [])],
            facts=list(data.get("facts", [])),
            metadata=dict(data.get("metadata", {})),
        )


class SpatialDiagnosticLoader:
    def __init__(self, path: str | Path = DEFAULT_SPATIAL_FIXTURE) -> None:
        self.path = Path(path)

    def load(self, limit: int | None = None) -> list[SpatialSample]:
        samples: list[SpatialSample] = []
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if limit is not None and len(samples) >= limit:
                    break
                if not line.strip():
                    continue
                obj = json.loads(line)
                obj.setdefault("metadata", {})["jsonl_line"] = line_number
                samples.append(SpatialSample.from_dict(obj))
        return samples


class SpatialDiagnosticBackend(EmbodiedBackend):
    """M1B adapter contract for static/video/multi-view spatial diagnostics."""

    def __init__(
        self,
        samples: list[SpatialSample] | None = None,
        data_path: str | Path = DEFAULT_SPATIAL_FIXTURE,
        sample_limit: int | None = None,
    ) -> None:
        self.samples = samples or SpatialDiagnosticLoader(data_path).load(limit=sample_limit)
        self._sample_by_task_id = {sample.task_id: sample for sample in self.samples}
        self._sample: SpatialSample | None = None
        self._task: TaskSpec | None = None
        self._state: dict[str, Any] = {}
        self._trace: EpisodeTrace | None = None

    def list_task_ids(self, source: str | None = None, category: str | None = None) -> list[str]:
        ids = []
        for sample in self.samples:
            if source is not None and sample.source != source:
                continue
            if category is not None and sample.category != category:
                continue
            ids.append(sample.task_id)
        return ids

    def task_spec_from_sample(self, sample: SpatialSample) -> TaskSpec:
        return TaskSpec(
            task_id=sample.task_id,
            source=sample.source,
            instruction=sample.instruction,
            goal={
                "answer_type": "multiple_choice" if sample.choices else "short_answer",
                "choices": list(sample.choices),
                "category": sample.category,
            },
            initial_state={
                "views": list(sample.views),
                "evidence": {},
                "submission": None,
            },
            budgets={"primitive_calls": 10, "verifier_calls": 4, "view_budget": max(1, len(sample.views))},
            tags=["m1b", "spatial", sample.source, sample.category],
            allowed_primitive_levels=["L1", "L2", "L3"],
            metadata=dict(sample.metadata)
            | {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "oracle_leakage_level": str(sample.metadata.get("oracle_leakage_level", "none")),
                "agent_native_contract": {
                    "primitives_accept_prompt_query_or_agent_context": True,
                    "oracle_checker_success_primitives_exposed": False,
                    "answer_key_visibility": "harness_side_verifier_only",
                },
            },
        )

    def reset(self, task_id: str, seed: int | None = None, config: dict[str, Any] | None = None) -> TaskSpec:
        if task_id not in self._sample_by_task_id:
            raise KeyError(f"Unknown spatial task: {task_id}")
        self._sample = self._sample_by_task_id[task_id]
        self._task = self.task_spec_from_sample(self._sample)
        self._state = dict(self._task.initial_state)
        self._state["seed"] = seed
        self._state["config"] = config or {}
        self._trace = EpisodeTrace(task_id=task_id)
        self.record_event("reset", {"task": self._task.to_dict(), "seed": seed, "config": config or {}})
        return self._task

    def observe(self) -> Observation:
        self._require_task()
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "instruction": self._task.instruction if self._task else "",
                "source": self._sample.source if self._sample else "",
                "category": self._sample.category if self._sample else "",
                "choices": list(self._sample.choices if self._sample else []),
                "view_ids": [view.get("view_id") for view in self._state.get("views", [])],
                "previous_evidence_keys": sorted(self._state.get("evidence", {})),
                "has_submission": self._state.get("submission") is not None,
            },
            artifacts=[str(view.get("artifact", "")) for view in self._state.get("views", []) if view.get("artifact")],
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        cards = [
            PrimitiveCard(
                name="get_task_context",
                capability_tags=["task", "metadata", "spatial"],
                input_schema={"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"instruction": "str", "choices": "list[str]", "views": "list[dict]", "evidence": "dict"},
                abstraction_level="L1",
                description="Return spatial task context and request evidence without answer-key fields.",
            ),
            PrimitiveCard(
                name="inspect_view",
                capability_tags=["vision", "view_selection", "spatial"],
                input_schema={"view_id": "str", "query": "str|None", "agent_context": "dict|None"},
                output_schema={"view": "dict", "evidence": "dict"},
                abstraction_level="L2",
                description="Inspect one static frame, egocentric frame, or multi-view image descriptor.",
            ),
            PrimitiveCard(
                name="inspect_video_frames",
                capability_tags=["vision", "video", "frame_sampling", "spatial"],
                input_schema={
                    "view_id": "str",
                    "frame_indices": "list[int]|None",
                    "sample_count": "int",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "video": "dict",
                    "sampled_frames": "list[{index, width, height, mean_rgb, center_rgb, extrema_rgb}]",
                    "evidence": "dict",
                },
                abstraction_level="L2",
                description="Read a materialized local video artifact and return frame metadata plus RGB statistics.",
            ),
            PrimitiveCard(
                name="detect_video_objects",
                capability_tags=["vision", "video", "DET", "bbox", "object_counting", "spatial"],
                input_schema={
                    "view_id": "str",
                    "query_labels": "list[str]|None",
                    "frame_indices": "list[int]|None",
                    "sample_count": "int",
                    "score_threshold": "float",
                    "max_detections_per_frame": "int",
                    "detector_backend": "str",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "video": "dict",
                    "detections": "list[dict]",
                    "label_counts": "dict[str,int]",
                    "frame_summaries": "dict",
                    "evidence": "dict",
                },
                abstraction_level="L2",
                description="Run a local detector over sampled video frames and return non-oracle object labels, boxes, and counts.",
            ),
            PrimitiveCard(
                name="track_video_objects",
                capability_tags=["vision", "video", "DET", "tracking", "measurement", "object_counting", "spatial"],
                input_schema={
                    "view_id": "str",
                    "query_labels": "list[str]|None",
                    "frame_indices": "list[int]|None",
                    "sample_count": "int",
                    "score_threshold": "float",
                    "max_detections_per_frame": "int",
                    "detector_backend": "str",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "video": "dict",
                    "tracks": "list[dict]",
                    "measurements": "dict",
                    "evidence": "dict",
                },
                abstraction_level="L2",
                description="Group query-driven video detections into cross-frame tracks and bbox measurements without oracle labels.",
            ),
            PrimitiveCard(
                name="segment_video_objects",
                capability_tags=["vision", "video", "DET", "SEG", "mask", "object_counting", "spatial"],
                input_schema={
                    "view_id": "str",
                    "query_labels": "list[str]|None",
                    "frame_indices": "list[int]|None",
                    "sample_count": "int",
                    "score_threshold": "float",
                    "max_detections_per_frame": "int",
                    "detector_backend": "str",
                    "segmentation_backend": "str",
                    "max_segments": "int",
                    "query": "str|None",
                    "agent_context": "dict|None",
                },
                output_schema={
                    "video": "dict",
                    "detections": "list[dict]",
                    "segments": "list[{frame_index,label,score,bbox_xyxy,mask_bbox_xyxy,mask_area,mask_center_xy}]",
                    "measurements": "dict",
                    "evidence": "dict",
                },
                abstraction_level="L2",
                description="Use detector boxes as prompts for a local segmentation backend and return mask summaries for agent-side evidence.",
            ),
            PrimitiveCard(
                name="preflight_video_grounding_model",
                capability_tags=["vision", "video", "DET", "open_vocabulary", "preflight", "spatial"],
                input_schema={"model_hint": "str|None", "agent_context": "dict|None"},
                output_schema={"ready": "bool", "backend": "dict", "blockers": "list[str]", "evidence": "dict"},
                abstraction_level="L1",
                description="Check whether a stronger local/open-vocabulary video grounding model or service is available.",
            ),
            PrimitiveCard(
                name="preflight_video_segmentation_model",
                capability_tags=["vision", "video", "SEG", "mask", "preflight", "spatial"],
                input_schema={"model_hint": "str|None", "agent_context": "dict|None"},
                output_schema={"ready": "bool", "backend": "dict", "blockers": "list[str]", "evidence": "dict"},
                abstraction_level="L1",
                description="Check whether a local promptable video/image segmentation backend such as SAM2 is available.",
            ),
            PrimitiveCard(
                name="write_evidence",
                capability_tags=["evidence", "memory"],
                input_schema={"key": "str", "value": "object", "agent_context": "dict|None"},
                output_schema={"artifact_id": "str", "evidence": "dict"},
                abstraction_level="L2",
                description="Attach spatial evidence to the episode trace.",
            ),
            PrimitiveCard(
                name="submit_answer",
                capability_tags=["commit", "answer"],
                input_schema={"answer": "str", "agent_context": "dict|None"},
                output_schema={"submission": "dict", "evidence": "dict"},
                abstraction_level="L3",
                description="Commit a spatial answer.",
            ),
        ]
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards)})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        self._require_task()
        handler = getattr(self, f"_primitive_{name}", None)
        if handler is None:
            result = PrimitiveResult(name=name, ok=False, error=f"Unknown primitive: {name}")
        else:
            result = handler(**kwargs)
        self.record_event("primitive_call", {"name": name, "kwargs": kwargs, "result": result.to_dict()})
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        self._require_task()
        if scope == "evidence":
            count = len(self._state.get("evidence", {}))
            result = VerificationResult(
                ok=count > 0,
                scope=scope,
                message="spatial evidence recorded" if count else "no spatial evidence recorded",
                metrics={"evidence_count": count},
            )
        elif scope in {"submission", "task"}:
            result = self._verify_submission(scope=scope)
        else:
            result = VerificationResult(ok=False, scope=scope, message=f"Unknown verification scope: {scope}")
        if scope == "task":
            self.get_trace().final_status = "success" if result.ok else "failed"
        self.record_event("verifier_call", result.to_dict())
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace is None:
            raise RuntimeError("Backend has not been reset.")
        return self._trace

    def _primitive_get_task_context(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        evidence = self._request_evidence("get_task_context", prompt=prompt, query=query, agent_context=agent_context)
        return PrimitiveResult(
            name="get_task_context",
            ok=True,
            output={
                "prompt": prompt,
                "query": query,
                "agent_context": agent_context or {},
                "instruction": self._task.instruction if self._task else "",
                "source": self._sample.source if self._sample else "",
                "category": self._sample.category if self._sample else "",
                "choices": list(self._sample.choices if self._sample else []),
                "views": [
                    {
                        "view_id": view.get("view_id"),
                        "artifact": view.get("artifact"),
                        "timestamp": view.get("timestamp"),
                    }
                    for view in self._state.get("views", [])
                ],
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_inspect_view(
        self,
        view_id: str,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        for view in self._state.get("views", []):
            if str(view.get("view_id")) == str(view_id):
                evidence = self._request_evidence(
                    "inspect_view",
                    query=query,
                    agent_context=agent_context,
                    extra={"view_id": str(view_id), "artifact": str(view.get("artifact", ""))},
                )
                return PrimitiveResult(
                    name="inspect_view",
                    ok=True,
                    output={"query": query, "agent_context": agent_context or {}, "view": dict(view), "evidence": evidence},
                    artifacts=[evidence["artifact_id"]],
                )
        return PrimitiveResult(name="inspect_view", ok=False, error=f"unknown_view_id:{view_id}")

    def _primitive_inspect_video_frames(
        self,
        view_id: str,
        frame_indices: list[int] | None = None,
        sample_count: int = 3,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        view = self._find_view(view_id)
        if view is None:
            return PrimitiveResult(name="inspect_video_frames", ok=False, error=f"unknown_view_id:{view_id}")
        artifact = str(view.get("artifact") or "")
        path = Path(artifact)
        if not path.is_file():
            return PrimitiveResult(
                name="inspect_video_frames",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": False},
                error=f"video_artifact_missing:{artifact}",
            )
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return PrimitiveResult(
                name="inspect_video_frames",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"unsupported_video_artifact:{path.suffix}",
            )
        try:
            video = _inspect_video_file(path=path, frame_indices=frame_indices, sample_count=sample_count)
        except Exception as exc:  # pragma: no cover - depends on optional codec/runtime behavior.
            return PrimitiveResult(
                name="inspect_video_frames",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"video_frame_inspection_failed:{type(exc).__name__}:{exc}",
            )
        evidence = self._request_evidence(
            "inspect_video_frames",
            query=query,
            agent_context=agent_context,
            extra={
                "view_id": str(view_id),
                "artifact": artifact,
                "frame_count": video["metadata"]["frame_count"],
                "sampled_frame_indices": [frame["index"] for frame in video["sampled_frames"]],
                "video_frame_inspection": True,
            },
        )
        return PrimitiveResult(
            name="inspect_video_frames",
            ok=True,
            output={
                "query": query,
                "agent_context": agent_context or {},
                "view": dict(view),
                "video": video["metadata"],
                "sampled_frames": video["sampled_frames"],
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_detect_video_objects(
        self,
        view_id: str,
        query_labels: list[str] | None = None,
        frame_indices: list[int] | None = None,
        sample_count: int = 5,
        score_threshold: float = 0.35,
        max_detections_per_frame: int = 20,
        detector_backend: str = "torchvision_coco",
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        view = self._find_view(view_id)
        if view is None:
            return PrimitiveResult(name="detect_video_objects", ok=False, error=f"unknown_view_id:{view_id}")
        artifact = str(view.get("artifact") or "")
        path = Path(artifact)
        if not path.is_file():
            return PrimitiveResult(
                name="detect_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": False},
                error=f"video_artifact_missing:{artifact}",
            )
        if path.suffix.lower() not in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
            return PrimitiveResult(
                name="detect_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"unsupported_video_artifact:{path.suffix}",
            )
        try:
            detector = _detect_video_objects_from_file(
                path=path,
                frame_indices=frame_indices,
                sample_count=sample_count,
                query_labels=query_labels,
                score_threshold=score_threshold,
                max_detections_per_frame=max_detections_per_frame,
                detector_backend=detector_backend,
            )
        except Exception as exc:  # pragma: no cover - optional model/runtime behavior.
            return PrimitiveResult(
                name="detect_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"video_object_detection_failed:{type(exc).__name__}:{exc}",
            )
        evidence = self._request_evidence(
            "detect_video_objects",
            query=query,
            agent_context=agent_context,
            extra={
                "view_id": str(view_id),
                "artifact": artifact,
                "sampled_frame_indices": detector["sampled_frame_indices"],
                "query_labels": detector["query_labels"],
                "label_counts": detector["label_counts"],
                "video_object_detection": True,
                "detector_name": detector["detector"]["name"],
                "detector_backend": detector["detector"].get("backend", detector_backend),
            },
        )
        return PrimitiveResult(
            name="detect_video_objects",
            ok=True,
            output={
                "query": query,
                "agent_context": agent_context or {},
                "view": dict(view),
                "video": detector["video"],
                "detector": detector["detector"],
                "query_labels": detector["query_labels"],
                "sampled_frame_indices": detector["sampled_frame_indices"],
                "detections": detector["detections"],
                "label_counts": detector["label_counts"],
                "frame_summaries": detector["frame_summaries"],
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_track_video_objects(
        self,
        view_id: str,
        query_labels: list[str] | None = None,
        frame_indices: list[int] | None = None,
        sample_count: int = 8,
        score_threshold: float = 0.35,
        max_detections_per_frame: int = 20,
        detector_backend: str = "torchvision_coco",
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        view = self._find_view(view_id)
        if view is None:
            return PrimitiveResult(name="track_video_objects", ok=False, error=f"unknown_view_id:{view_id}")
        normalized_backend = _normalize_detector_backend(detector_backend)
        if _is_open_vocabulary_detector_backend(normalized_backend):
            preflight = _preflight_video_grounding_model(model_hint=detector_backend)
            if not preflight["ready"]:
                return PrimitiveResult(
                    name="track_video_objects",
                    ok=False,
                    output={
                        "view_id": view_id,
                        "detector_backend": detector_backend,
                        "preflight": preflight,
                    },
                    error="missing_video_grounding_model",
                )
        elif normalized_backend not in {"torchvision_coco", "coco"}:
            return PrimitiveResult(
                name="track_video_objects",
                ok=False,
                output={"view_id": view_id, "detector_backend": detector_backend},
                error=f"unsupported_detector_backend:{detector_backend}",
            )
        artifact = str(view.get("artifact") or "")
        path = Path(artifact)
        if not path.is_file():
            return PrimitiveResult(
                name="track_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": False},
                error=f"video_artifact_missing:{artifact}",
            )
        try:
            detector = _detect_video_objects_from_file(
                path=path,
                frame_indices=frame_indices,
                sample_count=sample_count,
                query_labels=query_labels,
                score_threshold=score_threshold,
                max_detections_per_frame=max_detections_per_frame,
                detector_backend=detector_backend,
            )
            tracking = _track_detections(detector["detections"], video=detector["video"])
        except Exception as exc:  # pragma: no cover - optional codec/model behavior.
            return PrimitiveResult(
                name="track_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"video_object_tracking_failed:{type(exc).__name__}:{exc}",
            )
        evidence = self._request_evidence(
            "track_video_objects",
            query=query,
            agent_context=agent_context,
            extra={
                "view_id": str(view_id),
                "artifact": artifact,
                "query_labels": detector["query_labels"],
                "sampled_frame_indices": detector["sampled_frame_indices"],
                "track_count": len(tracking["tracks"]),
                "measurement_summary": tracking["measurements"],
                "video_object_tracking": True,
                "detector_name": detector["detector"]["name"],
            },
        )
        return PrimitiveResult(
            name="track_video_objects",
            ok=True,
            output={
                "query": query,
                "agent_context": agent_context or {},
                "view": dict(view),
                "video": detector["video"],
                "detector": detector["detector"],
                "query_labels": detector["query_labels"],
                "sampled_frame_indices": detector["sampled_frame_indices"],
                "detections": detector["detections"],
                "tracks": tracking["tracks"],
                "measurements": tracking["measurements"],
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_segment_video_objects(
        self,
        view_id: str,
        query_labels: list[str] | None = None,
        frame_indices: list[int] | None = None,
        sample_count: int = 3,
        score_threshold: float = 0.05,
        max_detections_per_frame: int = 6,
        detector_backend: str = "owl_vit",
        segmentation_backend: str = "sam2",
        max_segments: int = 8,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        view = self._find_view(view_id)
        if view is None:
            return PrimitiveResult(name="segment_video_objects", ok=False, error=f"unknown_view_id:{view_id}")
        normalized_seg_backend = _normalize_detector_backend(segmentation_backend)
        if normalized_seg_backend not in {"sam2", "segment_anything_2"}:
            return PrimitiveResult(
                name="segment_video_objects",
                ok=False,
                output={"view_id": view_id, "segmentation_backend": segmentation_backend},
                error=f"unsupported_segmentation_backend:{segmentation_backend}",
            )
        seg_preflight = _preflight_video_segmentation_model(model_hint=segmentation_backend)
        if not seg_preflight["ready"]:
            return PrimitiveResult(
                name="segment_video_objects",
                ok=False,
                output={"view_id": view_id, "segmentation_backend": segmentation_backend, "preflight": seg_preflight},
                error="missing_video_segmentation_model",
            )
        normalized_det_backend = _normalize_detector_backend(detector_backend)
        if _is_open_vocabulary_detector_backend(normalized_det_backend):
            det_preflight = _preflight_video_grounding_model(model_hint=detector_backend)
            if not det_preflight["ready"]:
                return PrimitiveResult(
                    name="segment_video_objects",
                    ok=False,
                    output={"view_id": view_id, "detector_backend": detector_backend, "preflight": det_preflight},
                    error="missing_video_grounding_model",
                )
        artifact = str(view.get("artifact") or "")
        path = Path(artifact)
        if not path.is_file():
            return PrimitiveResult(
                name="segment_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": False},
                error=f"video_artifact_missing:{artifact}",
            )
        try:
            segmentation = _segment_video_objects_from_file(
                path=path,
                frame_indices=frame_indices,
                sample_count=sample_count,
                query_labels=query_labels,
                score_threshold=score_threshold,
                max_detections_per_frame=max_detections_per_frame,
                detector_backend=detector_backend,
                segmentation_backend=segmentation_backend,
                max_segments=max_segments,
            )
        except Exception as exc:  # pragma: no cover - optional model/runtime behavior.
            return PrimitiveResult(
                name="segment_video_objects",
                ok=False,
                output={"view_id": view_id, "artifact": artifact, "exists": True},
                error=f"video_object_segmentation_failed:{type(exc).__name__}:{exc}",
            )
        evidence = self._request_evidence(
            "segment_video_objects",
            query=query,
            agent_context=agent_context,
            extra={
                "view_id": str(view_id),
                "artifact": artifact,
                "query_labels": segmentation["query_labels"],
                "sampled_frame_indices": segmentation["sampled_frame_indices"],
                "segment_count": len(segmentation["segments"]),
                "measurement_summary": segmentation["measurements"],
                "video_object_segmentation": True,
                "detector_name": segmentation["detector"]["name"],
                "segmentation_backend": segmentation["segmenter"]["backend"],
            },
        )
        return PrimitiveResult(
            name="segment_video_objects",
            ok=True,
            output={
                "query": query,
                "agent_context": agent_context or {},
                "view": dict(view),
                "video": segmentation["video"],
                "detector": segmentation["detector"],
                "segmenter": segmentation["segmenter"],
                "query_labels": segmentation["query_labels"],
                "sampled_frame_indices": segmentation["sampled_frame_indices"],
                "detections": segmentation["detections"],
                "segments": segmentation["segments"],
                "measurements": segmentation["measurements"],
                "evidence": evidence,
            },
            artifacts=[evidence["artifact_id"]],
        )

    def _primitive_preflight_video_grounding_model(
        self,
        model_hint: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        preflight = _preflight_video_grounding_model(model_hint=model_hint)
        evidence = self._request_evidence(
            "preflight_video_grounding_model",
            agent_context=agent_context,
            extra={
                "model_hint": model_hint,
                "ready": preflight["ready"],
                "blockers": preflight["blockers"],
                "video_grounding_model_preflight": True,
            },
        )
        return PrimitiveResult(
            name="preflight_video_grounding_model",
            ok=bool(preflight["ready"]),
            output={**preflight, "agent_context": agent_context or {}, "evidence": evidence},
            artifacts=[evidence["artifact_id"]],
            error=None if preflight["ready"] else "missing_video_grounding_model",
        )

    def _primitive_preflight_video_segmentation_model(
        self,
        model_hint: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        preflight = _preflight_video_segmentation_model(model_hint=model_hint)
        evidence = self._request_evidence(
            "preflight_video_segmentation_model",
            agent_context=agent_context,
            extra={
                "model_hint": model_hint,
                "ready": preflight["ready"],
                "blockers": preflight["blockers"],
                "video_segmentation_model_preflight": True,
            },
        )
        return PrimitiveResult(
            name="preflight_video_segmentation_model",
            ok=bool(preflight["ready"]),
            output={**preflight, "agent_context": agent_context or {}, "evidence": evidence},
            artifacts=[evidence["artifact_id"]],
            error=None if preflight["ready"] else "missing_video_segmentation_model",
        )

    def _primitive_write_evidence(
        self,
        key: str,
        value: Any,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"spatial_evidence:{key}"
        payload = {"key": key, "value": value, "agent_context": agent_context or {}}
        self._state.setdefault("evidence", {})[key] = value
        self.get_trace().add_artifact(artifact_id, payload)
        evidence = {"artifact_id": artifact_id, "source": "agent_written_evidence", "agent_context": agent_context or {}}
        return PrimitiveResult(name="write_evidence", ok=True, output={"artifact_id": artifact_id, "evidence": evidence}, artifacts=[artifact_id])

    def _primitive_submit_answer(self, answer: str, agent_context: dict[str, Any] | None = None) -> PrimitiveResult:
        submission = {"answer": str(answer), "agent_context": agent_context or {}}
        self._state["submission"] = submission
        evidence = self._request_evidence(
            "submit_answer",
            agent_context=agent_context,
            extra={"answer_length": len(str(answer)), "evidence_keys": sorted(self._state.get("evidence", {}))},
        )
        return PrimitiveResult(name="submit_answer", ok=True, output={"submission": submission, "evidence": evidence}, artifacts=[evidence["artifact_id"]])

    def _verify_submission(self, scope: str) -> VerificationResult:
        submission = self._state.get("submission")
        if not submission:
            return VerificationResult(ok=False, scope=scope, message="no submission", metrics={"success": 0.0})
        expected = self._sample.answer if self._sample else ""
        submitted = str(submission.get("answer", ""))
        ok = _normalize_answer(submitted) == _normalize_answer(expected)
        return VerificationResult(
            ok=ok,
            scope=scope,
            message="answer matches spatial oracle" if ok else "answer does not match spatial oracle",
            metrics={"success": float(ok), "exact_match": float(ok)},
            leaked_fields=[],
            metadata={"harness_side_reference_answer_used": True, "agent_visible_answer_key": False},
        )

    def _require_task(self) -> None:
        if self._task is None or self._sample is None:
            raise RuntimeError("Call reset() before using the spatial backend.")

    def _request_evidence(
        self,
        primitive_name: str,
        *,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        artifact_id = f"spatial_request:{primitive_name}:{len(self.get_trace().artifacts) + 1}"
        evidence = {
            "artifact_id": artifact_id,
            "source": "spatial_backend_request_trace",
            "primitive": primitive_name,
            "prompt": prompt,
            "query": query,
            "agent_context": agent_context or {},
        }
        if extra:
            evidence.update(extra)
        self.get_trace().add_artifact(artifact_id, evidence)
        return evidence

    def _find_view(self, view_id: str) -> dict[str, Any] | None:
        for view in self._state.get("views", []):
            if str(view.get("view_id")) == str(view_id):
                return dict(view)
        return None


def _normalize_answer(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _inspect_video_file(*, path: Path, frame_indices: list[int] | None, sample_count: int) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("cv2 is required for local video frame inspection") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        selected_indices = _select_frame_indices(frame_count=frame_count, frame_indices=frame_indices, sample_count=sample_count)
        sampled_frames = []
        for index in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                sampled_frames.append({"index": int(index), "read_ok": False})
                continue
            sampled_frames.append(_summarize_video_frame(frame_bgr=frame_bgr, index=int(index)))
    finally:
        cap.release()
    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {
        "metadata": {
            "artifact_path": str(path),
            "exists": True,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "duration_seconds": round(duration_seconds, 6) if duration_seconds is not None else None,
            "sample_count": len(sampled_frames),
        },
        "sampled_frames": sampled_frames,
    }


def _select_frame_indices(*, frame_count: int, frame_indices: list[int] | None, sample_count: int) -> list[int]:
    if frame_indices:
        return sorted({max(0, min(frame_count - 1, int(index))) if frame_count > 0 else max(0, int(index)) for index in frame_indices})
    count = max(1, min(8, int(sample_count or 3)))
    if frame_count <= 1:
        return [0]
    if count == 1:
        return [0]
    span = frame_count - 1
    return sorted({round(i * span / (count - 1)) for i in range(count)})


def _summarize_video_frame(*, frame_bgr: Any, index: int) -> dict[str, Any]:
    rgb = frame_bgr[:, :, ::-1]
    height, width = rgb.shape[:2]
    flat = rgb.reshape(-1, 3)
    mean_rgb = [round(float(value), 3) for value in flat.mean(axis=0)]
    min_rgb = [int(value) for value in flat.min(axis=0)]
    max_rgb = [int(value) for value in flat.max(axis=0)]
    center = rgb[height // 2, width // 2]
    return {
        "index": index,
        "read_ok": True,
        "width": int(width),
        "height": int(height),
        "mean_rgb": mean_rgb,
        "center_rgb": [int(value) for value in center],
        "extrema_rgb": [[min_rgb[channel], max_rgb[channel]] for channel in range(3)],
    }


def _detect_video_objects_from_file(
    *,
    path: Path,
    frame_indices: list[int] | None,
    sample_count: int,
    query_labels: list[str] | None,
    score_threshold: float,
    max_detections_per_frame: int,
    detector_backend: str = "torchvision_coco",
) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("cv2 is required for local video object detection") from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    frames: list[dict[str, Any]] = []
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        selected_indices = _select_frame_indices(
            frame_count=frame_count,
            frame_indices=frame_indices,
            sample_count=sample_count,
        )
        for index in selected_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(index)))
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frames.append({"index": int(index), "rgb": frame_bgr[:, :, ::-1].copy()})
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"no readable sampled frames in video: {path}")

    backend = _normalize_detector_backend(detector_backend)
    if backend in {"torchvision_coco", "coco"}:
        detection = _detect_rgb_frames_with_torchvision(
            frames=frames,
            query_labels=query_labels,
            score_threshold=score_threshold,
            max_detections_per_frame=max_detections_per_frame,
        )
    elif _is_open_vocabulary_detector_backend(backend):
        detection = _detect_rgb_frames_with_owl_vit(
            frames=frames,
            query_labels=query_labels,
            score_threshold=score_threshold,
            max_detections_per_frame=max_detections_per_frame,
            model_hint=detector_backend,
        )
    else:
        raise RuntimeError(f"unsupported_detector_backend:{detector_backend}")
    duration_seconds = frame_count / fps if fps > 0 and frame_count > 0 else None
    return {
        "video": {
            "artifact_path": str(path),
            "exists": True,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "fps": round(fps, 6),
            "duration_seconds": round(duration_seconds, 6) if duration_seconds is not None else None,
        },
        **detection,
    }


def _segment_video_objects_from_file(
    *,
    path: Path,
    frame_indices: list[int] | None,
    sample_count: int,
    query_labels: list[str] | None,
    score_threshold: float,
    max_detections_per_frame: int,
    detector_backend: str,
    segmentation_backend: str,
    max_segments: int,
) -> dict[str, Any]:
    detector = _detect_video_objects_from_file(
        path=path,
        frame_indices=frame_indices,
        sample_count=sample_count,
        query_labels=query_labels,
        score_threshold=score_threshold,
        max_detections_per_frame=max_detections_per_frame,
        detector_backend=detector_backend,
    )
    preflight = _preflight_video_segmentation_model(model_hint=segmentation_backend)
    if not preflight.get("ready"):
        blockers = ",".join(str(item) for item in preflight.get("blockers", []))
        raise RuntimeError(f"sam2_segmentation_model_not_ready:{blockers}")
    frames = _read_video_rgb_frames_by_index(path, detector["sampled_frame_indices"])
    predictor = _load_sam2_image_predictor(preflight)
    try:
        import numpy as np
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("numpy and torch are required for SAM2 video segmentation") from exc
    from contextlib import nullcontext

    max_total_segments = max(1, int(max_segments or 8))
    segments: list[dict[str, Any]] = []
    detections_by_frame: dict[int, list[dict[str, Any]]] = {}
    for detection in detector["detections"]:
        if len(segments) + sum(len(items) for items in detections_by_frame.values()) >= max_total_segments:
            break
        frame_index = int(detection.get("frame_index", 0))
        detections_by_frame.setdefault(frame_index, []).append(detection)

    device = str(preflight.get("selected_device") or "cpu")
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device.startswith("cuda") and torch.cuda.is_available() else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        for frame_index in sorted(detections_by_frame):
            if len(segments) >= max_total_segments:
                break
            frame = frames.get(frame_index)
            if frame is None:
                continue
            predictor.set_image(frame)
            for detection in detections_by_frame[frame_index]:
                if len(segments) >= max_total_segments:
                    break
                bbox = [float(value) for value in detection.get("bbox_xyxy", [])]
                if len(bbox) != 4 or _bbox_area(bbox) <= 0:
                    continue
                masks, scores, _ = predictor.predict(
                    box=np.asarray(bbox, dtype=np.float32),
                    multimask_output=True,
                )
                scores_np = np.asarray(scores).reshape(-1) if scores is not None else np.asarray([])
                masks_np = np.asarray(masks)
                if masks_np.ndim == 2:
                    masks_np = masks_np[None, ...]
                if masks_np.size == 0:
                    continue
                best_index = int(scores_np.argmax()) if scores_np.size else 0
                best_index = min(best_index, masks_np.shape[0] - 1)
                mask_summary = _summarize_mask_array(masks_np[best_index])
                if not mask_summary["mask_nonempty"]:
                    continue
                segments.append(
                    {
                        "segment_id": f"segment_{len(segments) + 1}",
                        "frame_index": frame_index,
                        "label": str(detection.get("label", "")),
                        "detection_score": float(detection.get("score", 0.0)),
                        "segmentation_score": round(float(scores_np[best_index]), 6) if scores_np.size else None,
                        "bbox_xyxy": [round(value, 3) for value in bbox],
                        **mask_summary,
                    }
                )

    counts_by_label: dict[str, int] = {}
    for segment in segments:
        label = str(segment.get("label", ""))
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
    measurements = {
        "segment_count": len(segments),
        "mask_object_count_estimate": len(segments),
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "method": "detector_box_prompted_sam2_mask_summary",
        "max_segments": max_total_segments,
    }
    return {
        "video": detector["video"],
        "detector": detector["detector"],
        "segmenter": {
            "name": "facebook.sam2",
            "backend": "sam2",
            "model_ref": preflight.get("selected_checkpoint"),
            "config_ref": preflight.get("selected_config"),
            "repo_ref": preflight.get("selected_repo"),
            "device": device,
            "mask_payload": "summaries_only",
        },
        "query_labels": detector["query_labels"],
        "sampled_frame_indices": detector["sampled_frame_indices"],
        "detections": detector["detections"],
        "segments": segments,
        "measurements": measurements,
    }


def _read_video_rgb_frames_by_index(path: Path, frame_indices: list[int]) -> dict[int, Any]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("cv2 is required for local video frame segmentation") from exc
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"unable to open video: {path}")
    frames: dict[int, Any] = {}
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
            ok, frame_bgr = cap.read()
            if ok and frame_bgr is not None:
                frames[int(frame_index)] = frame_bgr[:, :, ::-1].copy()
    finally:
        cap.release()
    return frames


def _summarize_mask_array(mask: Any) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("numpy is required for mask summaries") from exc
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim >= 3:
        mask_bool = mask_bool.squeeze()
    if mask_bool.ndim != 2 or not mask_bool.any():
        return {"mask_nonempty": False, "mask_area": 0, "mask_bbox_xyxy": None, "mask_center_xy": None}
    ys, xs = np.where(mask_bool)
    left = int(xs.min())
    right = int(xs.max())
    top = int(ys.min())
    bottom = int(ys.max())
    return {
        "mask_nonempty": True,
        "mask_area": int(mask_bool.sum()),
        "mask_bbox_xyxy": [left, top, right, bottom],
        "mask_center_xy": [round(float(xs.mean()), 3), round(float(ys.mean()), 3)],
        "mask_shape_hw": [int(mask_bool.shape[0]), int(mask_bool.shape[1])],
    }


def _normalize_detector_backend(detector_backend: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(detector_backend or "torchvision_coco").lower()).strip("_")
    return normalized or "torchvision_coco"


def _is_open_vocabulary_detector_backend(detector_backend: str | None) -> bool:
    return _normalize_detector_backend(detector_backend) in {
        "owl_vit",
        "owlvit",
        "open_vocab",
        "open_vocabulary",
        "open_vocabulary_detector",
        "vsi_grounding",
    }


def _detect_rgb_frames_with_torchvision(
    *,
    frames: list[dict[str, Any]],
    query_labels: list[str] | None,
    score_threshold: float,
    max_detections_per_frame: int,
) -> dict[str, Any]:
    try:
        import torch
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
        from torchvision.transforms.functional import to_tensor
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("torch and torchvision are required for video object detection") from exc

    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    categories = list(weights.meta.get("categories", []))
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.eval()
    normalized_queries = [_normalize_label(label) for label in (query_labels or []) if str(label).strip()]
    tensors = [to_tensor(frame["rgb"]) for frame in frames]
    max_per_frame = max(1, int(max_detections_per_frame or 20))
    detections: list[dict[str, Any]] = []
    frame_summaries: dict[str, Any] = {}
    label_counts: dict[str, int] = {}
    with torch.no_grad():
        outputs = model(tensors)
    for frame, output in zip(frames, outputs, strict=False):
        frame_index = int(frame["index"])
        frame_detections: list[dict[str, Any]] = []
        boxes = output.get("boxes", [])
        labels = output.get("labels", [])
        scores = output.get("scores", [])
        candidate_detections: list[dict[str, Any]] = []
        for box, label_index, score in zip(boxes, labels, scores, strict=False):
            confidence = float(score)
            if confidence < float(score_threshold):
                continue
            label_id = int(label_index)
            label = categories[label_id] if 0 <= label_id < len(categories) else str(label_id)
            if normalized_queries and not _label_matches_queries(label, normalized_queries):
                continue
            bbox = [round(float(value), 3) for value in box.tolist()]
            candidate_detections.append(
                {
                    "frame_index": frame_index,
                    "label": str(label),
                    "score": round(confidence, 6),
                    "bbox_xyxy": bbox,
                }
            )
        frame_detections = _dedupe_frame_detections(candidate_detections, iou_threshold=0.55)[:max_per_frame]
        for item in frame_detections:
            detections.append(item)
            label_counts[str(item["label"])] = label_counts.get(str(item["label"]), 0) + 1
        frame_summaries[str(frame_index)] = {
            "candidate_count": len(candidate_detections),
            "detection_count": len(frame_detections),
            "labels": sorted({item["label"] for item in frame_detections}),
            "frame_nms_iou_threshold": 0.55,
        }
    return {
        "detector": {
            "name": "torchvision.fasterrcnn_resnet50_fpn_coco",
            "backend": "torchvision_coco",
            "score_threshold": float(score_threshold),
            "max_detections_per_frame": max_per_frame,
            "category_count": len(categories),
            "frame_nms_iou_threshold": 0.55,
        },
        "query_labels": query_labels or [],
        "sampled_frame_indices": [int(frame["index"]) for frame in frames],
        "detections": detections,
        "label_counts": dict(sorted(label_counts.items())),
        "frame_summaries": frame_summaries,
    }


def _detect_rgb_frames_with_owl_vit(
    *,
    frames: list[dict[str, Any]],
    query_labels: list[str] | None,
    score_threshold: float,
    max_detections_per_frame: int,
    model_hint: str | None = None,
) -> dict[str, Any]:
    queries = [str(label).strip() for label in (query_labels or []) if str(label).strip()]
    if not queries:
        raise RuntimeError("owl_vit_requires_query_labels")
    model_status = _preflight_video_grounding_model(model_hint=model_hint)
    model_ref = model_status.get("selected_model_ref")
    if not model_status.get("ready") or not model_ref:
        blockers = ",".join(str(item) for item in model_status.get("blockers", []))
        raise RuntimeError(f"owl_vit_model_not_ready:{blockers}")
    try:
        import torch
        from PIL import Image
        from transformers import OwlViTForObjectDetection, OwlViTProcessor
    except ImportError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError("torch, PIL, and transformers are required for OWL-ViT video detection") from exc

    processor = OwlViTProcessor.from_pretrained(str(model_ref), local_files_only=True)
    model = OwlViTForObjectDetection.from_pretrained(str(model_ref), local_files_only=True)
    model.eval()

    max_per_frame = max(1, int(max_detections_per_frame or 20))
    detections: list[dict[str, Any]] = []
    frame_summaries: dict[str, Any] = {}
    label_counts: dict[str, int] = {}
    with torch.no_grad():
        for frame in frames:
            frame_index = int(frame["index"])
            image = Image.fromarray(frame["rgb"])
            inputs = processor(text=[queries], images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1]], dtype=torch.float32)
            result = processor.post_process_object_detection(
                outputs=outputs,
                threshold=float(score_threshold),
                target_sizes=target_sizes,
            )[0]
            boxes = result.get("boxes", [])
            labels = result.get("labels", [])
            scores = result.get("scores", [])
            ranked = sorted(
                zip(boxes, labels, scores, strict=False),
                key=lambda item: float(item[2]),
                reverse=True,
            )
            candidate_detections: list[dict[str, Any]] = []
            for box, label_index, score in ranked:
                label_id = int(label_index)
                label = queries[label_id] if 0 <= label_id < len(queries) else str(label_id)
                confidence = float(score)
                bbox = [round(float(value), 3) for value in box.tolist()]
                candidate_detections.append(
                    {
                        "frame_index": frame_index,
                        "label": label,
                        "score": round(confidence, 6),
                        "bbox_xyxy": bbox,
                    }
                )
            frame_detections = _dedupe_frame_detections(
                candidate_detections,
                iou_threshold=0.55,
                compatible_labels_only=False,
            )[:max_per_frame]
            for item in frame_detections:
                detections.append(item)
                label_counts[str(item["label"])] = label_counts.get(str(item["label"]), 0) + 1
            frame_summaries[str(frame_index)] = {
                "candidate_count": len(candidate_detections),
                "detection_count": len(frame_detections),
                "labels": sorted({item["label"] for item in frame_detections}),
                "frame_nms_iou_threshold": 0.55,
                "frame_nms_class_agnostic": True,
            }
    return {
        "detector": {
            "name": "transformers.owlvit",
            "backend": "owl_vit",
            "model_ref": str(model_ref),
            "score_threshold": float(score_threshold),
            "max_detections_per_frame": max_per_frame,
            "category_count": len(queries),
            "local_files_only": True,
            "frame_nms_iou_threshold": 0.55,
            "frame_nms_class_agnostic": True,
        },
        "query_labels": queries,
        "sampled_frame_indices": [int(frame["index"]) for frame in frames],
        "detections": detections,
        "label_counts": dict(sorted(label_counts.items())),
        "frame_summaries": frame_summaries,
    }


def _dedupe_frame_detections(
    detections: list[dict[str, Any]],
    *,
    iou_threshold: float,
    compatible_labels_only: bool = True,
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        bbox = [float(value) for value in detection.get("bbox_xyxy", [])]
        if len(bbox) != 4:
            continue
        duplicate = False
        for kept_item in kept:
            kept_bbox = [float(value) for value in kept_item.get("bbox_xyxy", [])]
            if compatible_labels_only and not _labels_are_compatible(str(detection.get("label", "")), str(kept_item.get("label", ""))):
                continue
            if _bbox_iou(bbox, kept_bbox) >= float(iou_threshold):
                duplicate = True
                break
        if not duplicate:
            kept.append(detection)
    return kept


def _labels_are_compatible(first: str, second: str) -> bool:
    first_norm = _normalize_label(first)
    second_norm = _normalize_label(second)
    if first_norm == second_norm:
        return True
    return _label_matches_queries(first_norm, [second_norm]) or _label_matches_queries(second_norm, [first_norm])


def _deprecated_detect_rgb_frames_with_owl_vit(
    *,
    frames: list[dict[str, Any]],
    query_labels: list[str] | None,
    score_threshold: float,
    max_detections_per_frame: int,
    model_hint: str | None = None,
) -> dict[str, Any]:
    """Kept only to make accidental stale imports fail loudly during development."""
    raise RuntimeError("deprecated_owlvit_detector_stub")


def _normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", str(label).lower()).strip()
    if normalized.endswith("s"):
        normalized = normalized[:-1]
    return normalized


def _label_matches_queries(label: str, normalized_queries: list[str]) -> bool:
    normalized_label = _normalize_label(label)
    label_tokens = set(normalized_label.split())
    for query in normalized_queries:
        query_tokens = set(query.split())
        if query == normalized_label or query in normalized_label or normalized_label in query:
            return True
        if query_tokens and query_tokens.issubset(label_tokens):
            return True
    return False


def _track_detections(detections: list[dict[str, Any]], *, video: dict[str, Any], iou_threshold: float = 0.35) -> dict[str, Any]:
    tracks: list[dict[str, Any]] = []
    for detection in sorted(detections, key=lambda item: (int(item.get("frame_index", 0)), -float(item.get("score", 0.0)))):
        label = str(detection.get("label", ""))
        bbox = [float(value) for value in detection.get("bbox_xyxy", [])]
        if len(bbox) != 4:
            continue
        best_track: dict[str, Any] | None = None
        best_iou = 0.0
        for track in tracks:
            if track["label"] != label:
                continue
            iou = _bbox_iou(track["_last_bbox"], bbox)
            if iou > best_iou:
                best_iou = iou
                best_track = track
        if best_track is None or best_iou < iou_threshold:
            best_track = {
                "track_id": f"track_{len(tracks) + 1}",
                "label": label,
                "frame_indices": [],
                "detections": [],
                "_last_bbox": bbox,
            }
            tracks.append(best_track)
        best_track["frame_indices"].append(int(detection.get("frame_index", 0)))
        best_track["detections"].append(detection)
        best_track["_last_bbox"] = bbox

    public_tracks: list[dict[str, Any]] = []
    counts_by_label: dict[str, int] = {}
    for track in tracks:
        boxes = [[float(value) for value in item["bbox_xyxy"]] for item in track["detections"]]
        areas = [_bbox_area(box) for box in boxes]
        centers = [_bbox_center(box) for box in boxes]
        label = track["label"]
        counts_by_label[label] = counts_by_label.get(label, 0) + 1
        public_tracks.append(
            {
                "track_id": track["track_id"],
                "label": label,
                "frame_indices": track["frame_indices"],
                "observation_count": len(track["detections"]),
                "score_max": round(max(float(item.get("score", 0.0)) for item in track["detections"]), 6),
                "bbox_xyxy_first": [round(value, 3) for value in boxes[0]],
                "bbox_xyxy_last": [round(value, 3) for value in boxes[-1]],
                "center_xy_first": [round(value, 3) for value in centers[0]],
                "center_xy_last": [round(value, 3) for value in centers[-1]],
                "bbox_area_mean": round(sum(areas) / len(areas), 3) if areas else 0.0,
            }
        )

    measurements = {
        "track_count": len(public_tracks),
        "unique_object_count_estimate": len(public_tracks),
        "counts_by_label": dict(sorted(counts_by_label.items())),
        "video_width": video.get("width"),
        "video_height": video.get("height"),
        "method": "greedy_label_iou_bbox_tracking",
        "iou_threshold": float(iou_threshold),
    }
    return {"tracks": public_tracks, "measurements": measurements}


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_center(box: list[float]) -> list[float]:
    return [(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0]


def _bbox_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = _bbox_area([left, top, right, bottom])
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _preflight_video_grounding_model(model_hint: str | None = None) -> dict[str, Any]:
    import os
    import urllib.error
    import urllib.request

    model_assets = get_project_paths().external_assets("huggingface")
    env_names = [
        "VSI_GROUNDING_MODEL_PATH",
        "VSI_OWL_VIT_MODEL_PATH",
        "OWL_VIT_MODEL_PATH",
        "GROUNDING_DINO_MODEL_PATH",
    ]
    candidates = []
    hint = str(model_hint or "").strip()
    if hint and not _is_open_vocabulary_detector_backend(hint):
        hint_path = Path(hint).expanduser()
        candidates.append(_inspect_grounding_model_candidate("model_hint", hint, hint_path))
    for name in env_names:
        value = os.environ.get(name)
        if value:
            path = Path(value).expanduser()
            candidates.append(_inspect_grounding_model_candidate(name, value, path))
    if os.environ.get("VSI_DISABLE_DEFAULT_GROUNDING_MODEL", "").lower() not in {"1", "true", "yes"}:
        for default_path in [
            model_assets / "owlvit-base-patch32",
            model_assets / "hub/models--google--owlvit-base-patch32",
        ]:
            candidates.append(_inspect_grounding_model_candidate("repo_default", str(default_path), default_path))
    existing = [candidate for candidate in candidates if candidate["exists"]]
    ready_models = [candidate for candidate in candidates if candidate.get("ready")]

    service_url = os.environ.get("VSI_GROUNDING_SERVICE_URL", "http://127.0.0.1:8117/health")
    service: dict[str, Any] = {"url": service_url, "reachable": False}
    try:
        with urllib.request.urlopen(service_url, timeout=0.35) as response:
            service["reachable"] = 200 <= int(response.status) < 500
            service["status"] = int(response.status)
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        service["error"] = f"{type(exc).__name__}:{exc}"

    blockers: list[str] = []
    if not ready_models and not service["reachable"]:
        blockers.append("missing_open_vocabulary_video_grounding_model_or_service")
    if not ready_models:
        blockers.append("no_loadable_grounding_model_path")
    if existing and not ready_models:
        blockers.append("grounding_model_path_incomplete_or_missing_weights")
    if not service["reachable"]:
        blockers.append("grounding_service_unreachable")

    selected = ready_models[0] if ready_models else None
    return {
        "ready": bool(ready_models or service["reachable"]),
        "model_hint": model_hint,
        "selected_model_ref": selected.get("load_path") if selected else None,
        "backend": {
            "expected_capability": "open_vocabulary_video_detection_or_vlm_grounding",
            "checked_env_vars": env_names,
            "model_paths": candidates,
            "service": service,
        },
        "blockers": blockers,
        "official_ready_note": (
            "COCO detector/tracking can provide boundary evidence, but VSI/ESI official-ready counting requires "
            "a stronger local open-vocabulary detector, tracking model, or video-VLM service."
        ),
    }


def _preflight_video_segmentation_model(model_hint: str | None = None) -> dict[str, Any]:
    import importlib.util
    import os
    import sys

    repo_candidates: list[Path] = []
    for env_name in ["VSI_SAM2_REPO", "SAM2_REPO"]:
        value = os.environ.get(env_name)
        if value:
            repo_candidates.append(Path(value).expanduser())
    use_default_sam2 = os.environ.get("VSI_DISABLE_DEFAULT_SAM2_MODEL", "").lower() not in {"1", "true", "yes"}
    if use_default_sam2:
        repo_candidates.append(get_project_paths().external_upstream("sam2"))

    checkpoint_candidates: list[Path] = []
    for env_name in ["VSI_SAM2_CHECKPOINT", "SAM2_CHECKPOINT"]:
        value = os.environ.get(env_name)
        if value:
            checkpoint_candidates.append(Path(value).expanduser())
    if use_default_sam2:
        hf_home = Path(os.environ.get("HF_HOME", get_project_paths().external_assets("huggingface")))
        checkpoint_candidates.extend(
            sorted(
                (hf_home / "hub/models--facebook--sam2.1-hiera-large/snapshots").glob(
                    "*/sam2.1_hiera_large.pt"
                ),
                reverse=True,
            )
        )
        checkpoint_candidates.append(get_project_paths().external_assets("sam2") / "checkpoints/sam2.1_hiera_large.pt")

    config_candidates: list[Path | str] = []
    for env_name in ["VSI_SAM2_CONFIG", "SAM2_CONFIG"]:
        value = os.environ.get(env_name)
        if value:
            config_candidates.append(Path(value).expanduser() if "/" in value else value)
    if use_default_sam2:
        config_candidates.append("configs/sam2.1/sam2.1_hiera_l.yaml")

    repo_infos = [_inspect_sam2_repo_candidate(path) for path in repo_candidates]
    checkpoint_infos = [_inspect_sam2_file_candidate(path, role="checkpoint") for path in checkpoint_candidates]
    config_infos = [_inspect_sam2_config_candidate(item, repo_candidates=repo_candidates) for item in config_candidates]
    selected_repo = next((item for item in repo_infos if item["ready"]), None)
    selected_checkpoint = next((item for item in checkpoint_infos if item["ready"]), None)
    selected_config = next((item for item in config_infos if item["ready"]), None)

    import_ok = False
    import_error: str | None = None
    if selected_repo:
        repo_path = str(selected_repo["path"])
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
    try:
        import_ok = importlib.util.find_spec("sam2") is not None
        if import_ok:
            from sam2.build_sam import build_sam2 as _build_sam2  # noqa: F401
            from sam2.sam2_image_predictor import SAM2ImagePredictor as _SAM2ImagePredictor  # noqa: F401
    except Exception as exc:  # pragma: no cover - optional dependency behavior.
        import_ok = False
        import_error = f"{type(exc).__name__}:{exc}"

    try:
        import torch

        torch_ok = True
        torch_error = None
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - optional dependency behavior.
        torch_ok = False
        torch_error = f"{type(exc).__name__}:{exc}"
        cuda_available = False
    requested_device = os.environ.get("VSI_SAM2_DEVICE") or ("cuda" if cuda_available else "cpu")

    blockers: list[str] = []
    if not selected_repo:
        blockers.append("sam2_repo_missing")
    if not selected_checkpoint:
        blockers.append("sam2_checkpoint_missing")
    if not selected_config:
        blockers.append("sam2_config_missing")
    if not import_ok:
        blockers.append("sam2_python_import_failed")
    if not torch_ok:
        blockers.append("torch_unavailable_for_sam2")

    ready = bool(selected_repo and selected_checkpoint and selected_config and import_ok and torch_ok)
    return {
        "ready": ready,
        "model_hint": model_hint,
        "selected_repo": selected_repo.get("path") if selected_repo else None,
        "selected_checkpoint": selected_checkpoint.get("path") if selected_checkpoint else None,
        "selected_config": selected_config.get("config_ref") if selected_config else None,
        "selected_device": requested_device,
        "backend": {
            "expected_capability": "box_prompted_image_or_video_segmentation",
            "repo_candidates": repo_infos,
            "checkpoint_candidates": checkpoint_infos,
            "config_candidates": config_infos,
            "import_ok": import_ok,
            "import_error": import_error,
            "torch_ok": torch_ok,
            "torch_error": torch_error,
            "cuda_available": cuda_available,
            "weights_source": "local_huggingface_cache_or_repo_checkpoint",
        },
        "blockers": blockers,
        "official_ready_note": (
            "This gate only proves SAM2 can be imported and its local checkpoint/config are present. "
            "A benchmark is official-ready only after a public primitive chain solves a verifier case."
        ),
    }


def _inspect_sam2_repo_candidate(path: Path) -> dict[str, Any]:
    path = path.expanduser()
    return {
        "path": str(path),
        "exists": path.exists(),
        "has_package": (path / "sam2").is_dir(),
        "has_build_sam": (path / "sam2/build_sam.py").is_file(),
        "ready": bool((path / "sam2/build_sam.py").is_file() and (path / "sam2/sam2_image_predictor.py").is_file()),
    }


def _inspect_sam2_file_candidate(path: Path, *, role: str) -> dict[str, Any]:
    path = path.expanduser()
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "size_bytes": int(path.stat().st_size) if path.is_file() else None,
        "ready": bool(path.is_file() and path.stat().st_size > 0),
    }


def _inspect_sam2_config_candidate(config: Path | str, *, repo_candidates: list[Path]) -> dict[str, Any]:
    if isinstance(config, Path):
        config_path = config.expanduser()
        return {
            "config_ref": str(config_path),
            "exists": config_path.exists(),
            "is_file": config_path.is_file(),
            "ready": config_path.is_file(),
        }
    config_ref = str(config)
    exists_in_repo = any((repo.expanduser() / "sam2" / config_ref).is_file() for repo in repo_candidates)
    return {
        "config_ref": config_ref,
        "exists": exists_in_repo,
        "is_file": exists_in_repo,
        "ready": exists_in_repo,
    }


def _load_sam2_image_predictor(preflight: dict[str, Any]) -> Any:
    import os
    import sys

    repo = str(preflight.get("selected_repo") or "")
    checkpoint = str(preflight.get("selected_checkpoint") or "")
    config = str(preflight.get("selected_config") or "")
    device = str(preflight.get("selected_device") or os.environ.get("VSI_SAM2_DEVICE") or "cpu")
    if not repo or not checkpoint or not config:
        raise RuntimeError("sam2_preflight_missing_selected_paths")
    key = (repo, checkpoint, config, device)
    cached = _SAM2_IMAGE_PREDICTOR_CACHE.get(key)
    if cached is not None:
        return cached
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model = build_sam2(config, checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    _SAM2_IMAGE_PREDICTOR_CACHE[key] = predictor
    return predictor


def _inspect_grounding_model_candidate(source: str, value: str, path: Path) -> dict[str, Any]:
    path = path.expanduser()
    info: dict[str, Any] = {
        "source": source,
        "value": value,
        "path": str(path),
        "exists": path.exists(),
        "ready": False,
    }
    if not path.exists():
        if "/" in value and not value.startswith(("/", ".")):
            info["model_id"] = value
            info["blocker"] = "model_id_configured_but_not_local"
        return info
    load_path = path
    if path.is_dir() and not (path / "config.json").is_file():
        snapshots = path / "snapshots"
        if snapshots.is_dir():
            snapshot_dirs = sorted([item for item in snapshots.iterdir() if item.is_dir()], reverse=True)
            if snapshot_dirs:
                load_path = snapshot_dirs[0]
    has_config = (load_path / "config.json").is_file() if load_path.is_dir() else False
    weight_names = ["model.safetensors", "pytorch_model.bin", "tf_model.h5", "flax_model.msgpack"]
    weights = [name for name in weight_names if load_path.is_dir() and (load_path / name).is_file()]
    info.update(
        {
            "load_path": str(load_path),
            "has_config": has_config,
            "weight_files": weights,
            "ready": bool(has_config and weights),
        }
    )
    if has_config and not weights:
        info["blocker"] = "missing_weight_file"
    elif not has_config:
        info["blocker"] = "missing_config_json"
    return info
