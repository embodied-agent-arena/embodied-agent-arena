#!/usr/bin/env python3
"""Download and materialize the W1 geometry pilot: MultiSPA-derived, InFlux val, Map-free val."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "outputs/materialized-w1"
MAPFREE_URLS = [
    "https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/val.zip",
    "https://storage.googleapis.com/niantic-lon-static/research/map-free-reloc/mapfree_val.zip",
]
INFLUX_REPO = "princeton-vl/InFlux-Real"
INFLUX_REVISION = "40e7a6974b04356f8f5d23ff8c1b9bbf5473d301"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows))


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_suffix(dest.suffix + ".partial")
    print(json.dumps({"event": "download", "url": url, "dest": str(dest)}), flush=True)
    with urllib.request.urlopen(url, timeout=600) as response, temporary.open("wb") as sink:
        shutil.copyfileobj(response, sink, 1024 * 1024)
    temporary.replace(dest)


def extract_frame(video: Path, dest: Path, frame_index: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vf", f"select=eq(n\\,{frame_index})",
        "-vframes", "1", str(dest),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    return completed.returncode == 0 and dest.is_file() and dest.stat().st_size > 0


def image_size(path: Path) -> tuple[int, int]:
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def ranked(items, *, key, seed=42, limit=None):
    scored = sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{key}:{item}".encode()).hexdigest())
    return scored if limit is None else scored[:limit]


def parse_mapfree_table(path: Path) -> dict[str, list[float]]:
    rows = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        rows[parts[0]] = [float(v) for v in parts[1:]]
    return rows


def quat_to_matrix(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def camera_center_mm(qw, qx, qy, qz, tx, ty, tz):
    rotation = quat_to_matrix(qw, qx, qy, qz)
    # Map-free: p_cam = R p_world + t; center C = -R^T t, in meters.
    cx = -(rotation[0][0] * tx + rotation[1][0] * ty + rotation[2][0] * tz)
    cy = -(rotation[0][1] * tx + rotation[1][1] * ty + rotation[2][1] * tz)
    cz = -(rotation[0][2] * tx + rotation[1][2] * ty + rotation[2][2] * tz)
    return [1000.0 * cx, 1000.0 * cy, 1000.0 * cz]


def geodesic_deg(qw, qx, qy, qz):
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    qw = max(-1.0, min(1.0, qw / n))
    return math.degrees(2 * math.acos(abs(qw)))


def copy_image(src: Path, dest: Path) -> tuple[int, int]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return image_size(dest)


def materialize_mapfree(root: Path, archive: Path, *, scenes=15, queries_per_scene=2, seed=42) -> list[dict]:
    extracted = root / "dataset"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        scene_names = sorted({
            Path(name).parts[1] if Path(name).parts[0] in {"val", "mapfree"} and len(Path(name).parts) > 1
            else Path(name).parts[0]
            for name in names if "/seq0/" in name or name.endswith("poses.txt")
        })
        scene_names = [name for name in scene_names if name.startswith("s")]
        selected = ranked(scene_names, key="mapfree-scene", seed=seed, limit=scenes)
        wanted = []
        for info in zf.infolist():
            parts = Path(info.filename).parts
            if any(part in selected for part in parts):
                wanted.append(info)
        print(json.dumps({"event": "extract", "benchmark": "mapfree", "scenes": selected}), flush=True)
        for info in wanted:
            target = extracted / info.filename
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            with zf.open(info) as source, target.open("wb") as sink:
                shutil.copyfileobj(source, sink, 1024 * 1024)
    cases = []
    scene_dirs = []
    for path in extracted.rglob("poses.txt"):
        if path.parent.name in selected or any(part in selected for part in path.parts):
            scene_dirs.append(path.parent)
    scene_dirs = sorted({path.resolve() for path in scene_dirs})
    for scene_dir in scene_dirs:
        poses = parse_mapfree_table(scene_dir / "poses.txt")
        intrinsics = parse_mapfree_table(scene_dir / "intrinsics.txt")
        ref_key = "seq0/frame_00000.jpg"
        if ref_key not in poses or ref_key not in intrinsics:
            continue
        queries = ranked(
            [key for key in poses if key.startswith("seq1/") and key in intrinsics],
            key=f"mapfree-q:{scene_dir.name}", seed=seed, limit=queries_per_scene,
        )
        ref_src = scene_dir / ref_key
        if not ref_src.is_file():
            continue
        ref_rel = Path("images") / scene_dir.name / "ref.jpg"
        copy_image(ref_src, root / ref_rel)
        ref_k = intrinsics[ref_key]
        for query_key in queries:
            query_src = scene_dir / query_key
            if not query_src.is_file():
                continue
            query_rel = Path("images") / scene_dir.name / (Path(query_key).stem + ".jpg")
            qw, qx, qy, qz, tx, ty, tz = poses[query_key]
            qk = intrinsics[query_key]
            rw, rh = copy_image(ref_src, root / ref_rel)
            qw_img, qh = copy_image(query_src, root / query_rel)
            sample_id = f"{scene_dir.name}_{Path(query_key).stem}"
            cases.append({
                "sample_id": sample_id,
                "benchmark": "mapfree",
                "task_family": "metric_relative_pose",
                "answer_type": "pose_wxyz_t",
                "units": "m",
                "question": (
                    "Estimate the metric relative pose of the query image with respect to the reference image. "
                    "The reference camera is identity. Output a world-to-camera quaternion and translation in meters: "
                    '{"qw":..., "qx":..., "qy":..., "qz":..., "tx":..., "ty":..., "tz":...}. '
                    "p_query = R(q) p_reference + t."
                ),
                "answer_schema": {"qw": "float", "qx": "float", "qy": "float", "qz": "float",
                                  "tx": "m", "ty": "m", "tz": "m"},
                "images": [
                    {"path": str(ref_rel), "role": "reference", "width": rw, "height": rh},
                    {"path": str(query_rel), "role": "query", "width": qw_img, "height": qh},
                ],
                "public_context": {
                    "intrinsics_reference_px": {"fx": ref_k[0], "fy": ref_k[1], "cx": ref_k[2], "cy": ref_k[3]},
                    "intrinsics_query_px": {"fx": qk[0], "fy": qk[1], "cx": qk[2], "cy": qk[3]},
                    "pose_convention": "mapfree_world_to_camera_query_from_reference",
                    "split": "val",
                },
                "gt": {"qw": qw, "qx": qx, "qy": qy, "qz": qz, "tx": tx, "ty": ty, "tz": tz},
            })
    return cases


def materialize_multispa_derived(mapfree_cases: list[dict], dest: Path) -> list[dict]:
    cases = []
    dest.mkdir(parents=True, exist_ok=True)
    for row in mapfree_cases:
        gt = row["gt"]
        center = camera_center_mm(gt["qw"], gt["qx"], gt["qy"], gt["qz"], gt["tx"], gt["ty"], gt["tz"])
        angle = geodesic_deg(gt["qw"], gt["qx"], gt["qy"], gt["qz"])
        distance = math.sqrt(sum(v * v for v in center))
        images = []
        for item in row["images"]:
            src = dest.parent / "mapfree" / item["path"]
            rel = Path("images") / row["sample_id"] / Path(item["path"]).name
            if src.is_file():
                copy_image(src, dest / rel)
                images.append({**item, "path": str(rel)})
        if len(images) != 2:
            continue
        families = [
            ("camera_translation_vector", "vector", "mm",
             "Compare these two images. The first is the reference camera. "
             "Give the camera-center displacement from image 1 to image 2 as [x, y, z] in millimeters. "
             "X is right, Y is down, Z is forward.",
             {"vector": center}),
            ("camera_translation_distance", "scalar", "mm",
             "Compare these two images. Estimate the camera-center travel distance in millimeters.",
             {"value": distance}),
            ("camera_rotation_angle", "scalar", "deg",
             "Compare these two images. Estimate the camera rotation magnitude in degrees.",
             {"value": angle}),
        ]
        family = families[int(hashlib.sha256(row["sample_id"].encode()).hexdigest(), 16) % 3]
        name, answer_type, units, question, truth = family
        cases.append({
            "sample_id": f"{row['sample_id']}_{name}",
            "benchmark": "multispa",
            "task_family": name,
            "answer_type": answer_type,
            "units": units,
            "question": question + " This is a MultiSPA-derived sample generated from Map-free val poses.",
            "answer_schema": {"value": units} if answer_type == "scalar" else {"vector": "[x,y,z] mm"},
            "images": images,
            "public_context": {
                "coordinate_frame": "X-right Y-down Z-forward",
                "source": "multispa_derived_from_mapfree_val",
            },
            "gt": truth,
        })
    return cases


def hf_dataset_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{INFLUX_REPO}/resolve/{INFLUX_REVISION}/{path}"


def select_influx_intrinsics(record: dict, *, allow_extrapolated=False):
    """Select finite calibration labels; extrapolation is an explicit protocol."""
    fields = ["intrinsics_gt"]
    if allow_extrapolated:
        fields.append("intrinsics_gt_extrapolated")
    for field in fields:
        k = record.get(field)
        if not isinstance(k, dict):
            continue
        values = [k.get(key) for key in ("fx", "fy", "cx", "cy")]
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               for v in values) and values[0] > 0 and values[1] > 0:
            return {key: float(k[key]) for key in ("fx", "fy", "cx", "cy")}, field
    return None, None


def materialize_influx(root: Path, *, videos=15, frames_per_video=2, seed=42,
                       allow_extrapolated=False) -> list[dict]:
    dataset = root / "dataset"
    dataset.mkdir(parents=True, exist_ok=True)
    split_path = dataset / "influx/video_frame_count_and_split_v1.json"
    gt_path = dataset / "influx/gt_validation_dict_v1.json"
    if not split_path.is_file():
        download(hf_dataset_url("influx/video_frame_count_and_split_v1.json"), split_path)
    if not gt_path.is_file():
        download(hf_dataset_url("influx/gt_validation_dict_v1.json"), gt_path)
    split = json.loads(split_path.read_text())
    gt = json.loads(gt_path.read_text())
    val_videos = []
    if isinstance(split, dict):
        for name, meta in split.items():
            label = meta.get("split") if isinstance(meta, dict) else None
            if label in {"val", "validation"} or name in gt:
                val_videos.append(name)
    if not val_videos:
        val_videos = list(gt)
    # Official split is "val" (70 videos). Keep walking the ranked pool so a
    # failed mp4 download does not shrink the pilot below the requested count.
    pool = ranked(sorted(set(val_videos) & set(gt)), key="influx-video", seed=seed)
    cases = []
    kept = 0
    for name in pool:
        if kept >= videos:
            break
        frames_gt = gt[name]
        eligible = {f: select_influx_intrinsics(record, allow_extrapolated=allow_extrapolated)
                    for f, record in frames_gt.items()}
        valid_ids = [f for f, (k, _) in eligible.items() if k is not None]
        if len(valid_ids) < frames_per_video:
            print(json.dumps({"event": "skip_invalid_gt_video", "video": name,
                              "valid_frames": len(valid_ids)}), flush=True)
            continue
        video_rel = f"influx/videos/{name}.mp4"
        video = dataset / video_rel
        try:
            if not video.is_file():
                download(hf_dataset_url(video_rel), video)
        except Exception as exc:
            print(json.dumps({"event": "skip_video", "video": name, "error": str(exc)}), flush=True)
            continue
        frame_ids = ranked(sorted(valid_ids, key=lambda x: int(x)), key=f"influx-f:{name}", seed=seed)
        video_cases = []
        for frame_id in frame_ids:
            k, gt_source = eligible[frame_id]
            dest = root / "images" / name / f"{int(frame_id):06d}.png"
            if not dest.exists() and not extract_frame(video, dest, int(frame_id)):
                print(json.dumps({"event": "frame_fail", "video": name, "frame": frame_id}), flush=True)
                continue
            width, height = image_size(dest)
            rel = dest.relative_to(root)
            video_cases.append({
                "sample_id": f"{name}_{int(frame_id):06d}",
                "benchmark": "influx",
                "task_family": "camera_intrinsics",
                "answer_type": "intrinsics",
                "units": "px",
                "question": (
                    "Estimate this frame's pinhole intrinsics in pixels. "
                    'Submit {"fx":..., "fy":..., "cx":..., "cy":...}. '
                    "The lens zoom/focus may differ from a default 35mm assumption."
                ),
                "answer_schema": {"fx": "px", "fy": "px", "cx": "px", "cy": "px"},
                "images": [{"path": str(rel), "role": "frame", "width": width, "height": height}],
                "public_context": {
                    "source": "influx_val",
                    "video": name,
                    "frame_index": int(frame_id),
                    "split": "validation",
                },
                "gt": {"intrinsics": {key: float(k[key]) for key in ("fx", "fy", "cx", "cy")}},
                "gt_provenance": {"field": gt_source, "revision": INFLUX_REVISION,
                                  "extrapolated": gt_source == "intrinsics_gt_extrapolated"},
            })
            if len(video_cases) == frames_per_video:
                cases.extend(video_cases)
                kept += 1
                break
    return cases


def download_mapfree_archive(dest: Path) -> Path:
    archive = dest / "val.zip"
    if archive.is_file() and archive.stat().st_size > 1_000_000:
        return archive
    last_error = None
    for url in MAPFREE_URLS:
        try:
            download(url, archive)
            return archive
        except Exception as exc:
            last_error = exc
            print(json.dumps({"event": "download_failed", "url": url, "error": str(exc)}), flush=True)
    raise RuntimeError(f"Map-free val.zip download failed: {last_error}")


def finalize(root: Path, benchmark: str, cases: list[dict], extra: dict) -> None:
    if not cases:
        raise RuntimeError(f"No {benchmark} cases materialized")
    write_jsonl(root / "catalog.jsonl", cases)
    files = [{"path": "catalog.jsonl", "sha256": sha256_file(root / "catalog.jsonl"),
              "bytes": (root / "catalog.jsonl").stat().st_size}]
    receipt = {"benchmark": benchmark, "case_count": len(cases), "files": files, **extra}
    write_json(root / "download_receipt.json", receipt)
    print(json.dumps({"benchmark": benchmark, "event": "ready", "cases": len(cases)}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--benchmarks", nargs="+", choices=["multispa", "influx", "mapfree"],
                   default=["multispa", "influx", "mapfree"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mapfree-scenes", type=int, default=15)
    p.add_argument("--influx-videos", type=int, default=15)
    p.add_argument("--influx-allow-extrapolated", action="store_true",
                   help="Explicitly include out-of-LUT extrapolated labels with provenance; default uses finite intrinsics_gt only")
    p.add_argument("--mapfree-archive", type=Path,
                   help="Local Map-free val.zip or sample.zip from the official license page")
    args = p.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    mapfree_cases = []
    if "mapfree" in args.benchmarks or "multispa" in args.benchmarks:
        archive = args.mapfree_archive
        if archive is None:
            archive = download_mapfree_archive(args.root / "mapfree")
        elif not archive.is_file():
            raise FileNotFoundError(f"Map-free archive missing: {archive}")
        mapfree_cases = materialize_mapfree(
            args.root / "mapfree", archive, scenes=args.mapfree_scenes, seed=args.seed)
        if "mapfree" in args.benchmarks:
            finalize(args.root / "mapfree", "mapfree", mapfree_cases,
                     {"repository": "nianticlabs/map-free-reloc", "split": "val",
                      "license": "Niantic Map-free non-commercial research"})
    if "multispa" in args.benchmarks:
        if not mapfree_cases:
            raise RuntimeError("MultiSPA-derived materialization requires Map-free val cases")
        cases = materialize_multispa_derived(mapfree_cases, args.root / "multispa")
        finalize(args.root / "multispa", "multispa", cases, {
            "label": "MultiSPA-derived",
            "source": "generated from Map-free val poses using MultiSPA camera-motion templates and official 20% L2 rule",
            "official_eval_set": "not published; regenerate from facebookresearch/Multi-SpatialMLLM if ScanNet/TAPVid3D become available",
        })
    if "influx" in args.benchmarks:
        cases = materialize_influx(args.root / "influx", videos=args.influx_videos, seed=args.seed,
                                  allow_extrapolated=args.influx_allow_extrapolated)
        finalize(args.root / "influx", "influx", cases, {
            "repository": INFLUX_REPO, "revision": INFLUX_REVISION, "split": "validation",
            "gt_policy": "interpolation_then_extrapolation" if args.influx_allow_extrapolated else "finite_interpolation_only",
        })


if __name__ == "__main__":
    raise SystemExit(main())
