#!/usr/bin/env python3
"""Episode-local YOLOE worker; reuse the installed ML Python without changing W2's venv."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    protocol = sys.stdout
    os.environ.update(YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", OMP_NUM_THREADS="2")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

    def send(value):
        protocol.write(json.dumps(value) + "\n")
        protocol.flush()

    try:
        started = time.monotonic()
        if hashlib.sha256(a.model.read_bytes()).hexdigest() != a.sha256:
            raise ValueError("YOLOE checkpoint SHA256 mismatch")
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from w2_harness.perception_backend.yoloe import UltralyticsYOLOEBackend
            from w2_harness.perception_backend.contracts import LocalImageSource
            torch.set_num_threads(2)
            backend = UltralyticsYOLOEBackend(a.model, device=a.device)
            identity = backend.prepare()
        send(dict(ok=True, backend=identity.to_dict(), load_seconds=time.monotonic() - started,
                  confidence_threshold=backend.confidence_threshold, iou_threshold=backend.iou_threshold,
                  max_detections_per_image=6, max_detections_total=48,
                  packages={name: importlib.metadata.version(name) for name in
                            ("torch", "torchvision", "ultralytics", "numpy", "opencv-python")}))
        for line in sys.stdin:
            started = time.monotonic()
            request = json.loads(line)
            # One image at a time bounds GPU memory independently of the public bundle size.
            source = LocalImageSource.from_path(request["path"], expected_media_hash=request["sha256"])
            with contextlib.redirect_stdout(sys.stderr):
                result = backend.infer([source])
            if result.status.value == "failed":
                raise RuntimeError(str(result.failure.to_dict()))
            detections = sorted(result.detections, key=lambda d: d.confidence or 0, reverse=True)
            send(dict(ok=True, status=result.status.value,
                      detections=[dict(label=d.class_name, confidence=round(d.confidence or 0, 4),
                                       bbox_xyxy=[round(v, 1) for v in d.bbox.xyxy]) for d in detections[:6]],
                      total_detections=len(detections), elapsed_seconds=time.monotonic() - started,
                      cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(a.device)
                      if a.device.startswith("cuda") else 0))
    except Exception as exc:
        send(dict(ok=False, error=f"{type(exc).__name__}: {exc}"))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
