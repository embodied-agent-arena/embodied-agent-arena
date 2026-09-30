from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from .spatial_backend import SpatialSample


VSI_BENCH_HF_DATASET = "nyu-visionx/VSI-Bench"


def inspect_vsi_visual_asset_preflight(
    *,
    samples: Sequence[SpatialSample] | None = None,
    manifest_path: str | Path | None = None,
    video_root: str | Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Check whether VSI samples expose local visual artifacts for agent use.

    Annotation-only `vsi://...` descriptors are useful for contract tests, but
    they are not a visual benchmark runtime. This preflight makes that boundary
    machine-readable before a qwen final gate is attempted.
    """

    selected_samples: list[SpatialSample]
    if samples is not None:
        selected_samples = list(samples)
    elif manifest_path is not None:
        selected_samples = _load_spatial_samples_from_jsonl(manifest_path, limit=limit)
    else:
        selected_samples = []

    if limit is not None:
        selected_samples = selected_samples[: max(0, limit)]

    records: list[dict[str, Any]] = []
    local_ready_count = 0
    uri_only_count = 0
    missing_count = 0
    for sample in selected_samples:
        sample_records: list[dict[str, Any]] = []
        sample_local_ready = False
        for view in sample.views:
            artifact = str(view.get("artifact") or "")
            materialized = False
            artifact_path = None
            blocker = None
            if artifact.startswith("vsi://"):
                uri_only_count += 1
                blocker = f"uri_only_visual_artifact:{artifact}"
                if video_root is not None:
                    dataset = str(view.get("dataset") or sample.metadata.get("source_dataset") or "")
                    scene_name = str(view.get("scene_name") or sample.metadata.get("scene_name") or "")
                    candidate = Path(video_root) / dataset / f"{scene_name}.mp4"
                    artifact_path = str(candidate)
                    if candidate.is_file():
                        materialized = True
                        blocker = None
                    else:
                        missing_count += 1
                        blocker = f"missing_video_file:{candidate}"
            else:
                candidate = Path(artifact)
                artifact_path = str(candidate)
                if candidate.is_file():
                    materialized = True
                else:
                    missing_count += 1
                    blocker = f"missing_visual_file:{candidate}"
            if materialized:
                local_ready_count += 1
                sample_local_ready = True
            sample_records.append(
                {
                    "view_id": view.get("view_id"),
                    "artifact": artifact,
                    "artifact_path": artifact_path,
                    "media_type": view.get("media_type"),
                    "materialized": materialized,
                    "blocker": blocker,
                }
            )
        records.append(
            {
                "sample_id": sample.sample_id,
                "source": sample.source,
                "category": sample.category,
                "scene_name": sample.metadata.get("scene_name"),
                "local_visual_ready": sample_local_ready,
                "views": sample_records,
            }
        )

    blockers: list[str] = []
    if not selected_samples:
        blockers.append("vsi_no_samples_loaded")
    if selected_samples and local_ready_count == 0:
        blockers.append("vsi_no_local_visual_artifacts")
    if missing_count:
        blockers.append(f"vsi_missing_visual_artifacts:{missing_count}")
    if uri_only_count and video_root is None:
        blockers.append(f"vsi_uri_only_artifacts_without_video_root:{uri_only_count}")

    return {
        "ready": bool(selected_samples) and local_ready_count > 0 and not blockers,
        "sample_count": len(selected_samples),
        "view_count": sum(len(record["views"]) for record in records),
        "local_ready_count": local_ready_count,
        "uri_only_count": uri_only_count,
        "missing_count": missing_count,
        "video_root": str(video_root) if video_root is not None else None,
        "records": records,
        "blockers": blockers,
        "agent_native_boundary": {
            "annotation_only_is_visual_ready": False,
            "requires_local_video_or_frame_artifacts": True,
            "oracle_answer_exposed_to_agent": False,
        },
    }


def vsi_record_to_spatial_sample(
    record: dict[str, Any],
    *,
    config: str = "full",
    include_oracle_fact: bool = False,
    video_root: str | Path | None = None,
) -> SpatialSample:
    """Convert one VSI-Bench annotation row into the M1B spatial schema.

    The answer key remains on the harness-side sample for verification, but
    oracle facts are omitted by default so coding-agent primitives cannot read
    the benchmark label through the environment interface.
    """

    native_id = str(_first_present(record, "id", "question_id", default="unknown"))
    dataset = str(_first_present(record, "dataset", "source_dataset", default="unknown"))
    scene_name = str(_first_present(record, "scene_name", "video", "video_id", default="unknown_scene"))
    question_type = str(_first_present(record, "question_type", "task_type", default="unknown"))
    question = str(_first_present(record, "question", "prompt", default=""))
    answer = str(_first_present(record, "ground_truth", "answer", "label", default=""))
    choices = normalize_vsi_options(record.get("options"))
    artifact = _video_artifact(dataset=dataset, scene_name=scene_name, video_root=video_root)

    facts = []
    if include_oracle_fact:
        facts.append(
            {
                "fact_id": "ground_truth",
                "text": "VSI-Bench ground truth answer for teacher-mode adapter smoke tests.",
                "answer": answer,
            }
        )

    return SpatialSample(
        sample_id=f"vsi_{native_id}",
        source="vsi_bench",
        instruction=question,
        answer=answer,
        category=question_type,
        choices=choices,
        views=[
            {
                "view_id": f"{dataset}:{scene_name}",
                "artifact": artifact,
                "media_type": "egocentric_video",
                "dataset": dataset,
                "scene_name": scene_name,
            }
        ],
        facts=facts,
        metadata={
            "benchmark_family": "video_spatial",
            "adapter_target": "VSI-Bench",
            "hf_dataset": VSI_BENCH_HF_DATASET,
            "hf_config": config,
            "native_id": native_id,
            "source_dataset": dataset,
            "scene_name": scene_name,
            "question_type": question_type,
            "pruned": bool(record.get("pruned", False)),
            "has_options": bool(choices),
            "oracle_leakage_level": "L4_spatial_fact" if include_oracle_fact else "none",
        },
    )


def load_vsi_bench_samples(
    *,
    config: str = "debiased",
    split: str = "test",
    limit: int | None = None,
    include_pruned: bool = True,
    include_oracle_fact: bool = False,
    streaming: bool = True,
    video_root: str | Path | None = None,
) -> list[SpatialSample]:
    """Load VSI-Bench annotations from HuggingFace and convert them.

    The video archives are not downloaded here. Each sample points to a stable
    `vsi://dataset/scene_name` artifact unless `video_root` is provided.
    """

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional environment.
        raise RuntimeError("Install the optional `datasets` package to load VSI-Bench from HuggingFace.") from exc

    dataset = load_dataset(VSI_BENCH_HF_DATASET, config, split=split, streaming=streaming)
    samples: list[SpatialSample] = []
    for record in dataset:
        if not include_pruned and bool(record.get("pruned", False)):
            continue
        samples.append(
            vsi_record_to_spatial_sample(
                dict(record),
                config=config,
                include_oracle_fact=include_oracle_fact,
                video_root=video_root,
            )
        )
        if limit is not None and len(samples) >= limit:
            break
    return samples


def write_spatial_samples_jsonl(samples: Iterable[SpatialSample], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(spatial_sample_to_record(sample), sort_keys=True) + "\n")


def spatial_sample_to_record(sample: SpatialSample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "source": sample.source,
        "instruction": sample.instruction,
        "answer": sample.answer,
        "category": sample.category,
        "views": list(sample.views),
        "choices": list(sample.choices),
        "facts": list(sample.facts),
        "metadata": dict(sample.metadata),
    }


def _load_spatial_samples_from_jsonl(path: str | Path, *, limit: int | None = None) -> list[SpatialSample]:
    samples: list[SpatialSample] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(samples) >= limit:
                break
            if not line.strip():
                continue
            samples.append(SpatialSample.from_dict(json.loads(line)))
    return samples


def normalize_vsi_options(options: Any) -> list[str]:
    if options in (None, ""):
        return []
    if isinstance(options, list):
        return [str(option) for option in options]
    if isinstance(options, tuple):
        return [str(option) for option in options]
    if isinstance(options, dict):
        return [f"{key}: {value}" for key, value in options.items()]
    if isinstance(options, str):
        stripped = options.strip()
        if not stripped:
            return []
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                return normalize_vsi_options(json.loads(stripped))
            except json.JSONDecodeError:
                pass
        separators = ["\n", " | ", ";"]
        for separator in separators:
            if separator in stripped:
                return [part.strip() for part in stripped.split(separator) if part.strip()]
        return [stripped]
    return [str(options)]


def _video_artifact(dataset: str, scene_name: str, video_root: str | Path | None) -> str:
    if video_root is None:
        return f"vsi://{dataset}/{scene_name}"
    return str(Path(video_root) / dataset / f"{scene_name}.mp4")


def _first_present(record: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return default


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a SpatialDiagnosticBackend JSONL manifest from VSI-Bench.")
    parser.add_argument("--config", default="debiased", choices=("full", "debiased", "pruned"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-pruned", action="store_true")
    parser.add_argument("--include-oracle-facts", action="store_true")
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--video-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    samples = load_vsi_bench_samples(
        config=args.config,
        split=args.split,
        limit=args.limit if args.limit > 0 else None,
        include_pruned=args.include_pruned,
        include_oracle_fact=args.include_oracle_facts,
        streaming=args.streaming,
        video_root=args.video_root,
    )
    write_spatial_samples_jsonl(samples, args.output)
    print(json.dumps({"output": str(args.output), "samples": len(samples), "config": args.config}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
