"""Optional visual observations from W2's native YOLOE backend, with no extra agent actions."""
import hashlib
import json
import os
from pathlib import Path
import select
import subprocess
import time


class YOLOEObservation:
    def __init__(self, *, python, worker, model, sha256, device, outputs, timeout=120):
        self.outputs = Path(outputs)
        self.timeout = timeout
        self.cache = {}
        self.excluded_ids, self.excluded_hashes = set(), set()
        self.stats = dict(model=str(model), sha256=sha256, device=device,
                          inferred_images=0, cache_hits=0, skipped_images=0, inference_seconds=0,
                          cuda_peak_allocated_bytes=0)
        self.log = (self.outputs / "yoloe.stderr.log").open("w")
        self.process = None
        try:
            self.process = subprocess.Popen(
                [str(python), "-u", str(worker), "--model", str(model), "--sha256", sha256,
                 "--device", device], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.log, text=True, bufsize=1,
                env=dict(os.environ, YOLO_CONFIG_DIR=str(self.outputs / "yolo_config")))
            self.stats.update(self._receive())
        except Exception:
            self.close()
            raise

    def _receive(self):
        if not select.select([self.process.stdout], [], [], self.timeout)[0]:
            raise TimeoutError(f"YOLOE exceeded {self.timeout}s; see yoloe.stderr.log")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("YOLOE worker exited; see yoloe.stderr.log")
        value = json.loads(line)
        if not value.pop("ok", False):
            raise RuntimeError("YOLOE: " + value.get("error", "unknown failure"))
        return value

    def infer(self, path, sha256):
        self.process.stdin.write(json.dumps(dict(path=str(path), sha256=sha256)) + "\n")
        self.process.stdin.flush()
        return self._receive()

    def observe(self, descriptors, frames, visible_assets):
        started = time.monotonic()
        # Remember non-RGB ancestry across turns, including derived crops/composites.
        for asset in visible_assets:
            roles = asset.get("metadata", {}).get("roles", [])
            if any(role in roles for role in ("provided_depth_map", "provided_mask")):
                self.excluded_ids.add(asset["asset_id"])
                self.excluded_hashes.add(asset.get("source_media_hash"))
        for _ in range(len(descriptors) + 1):
            changed = False
            for d in descriptors:
                ids = {d.get("asset_id"), d.get("public_asset_id"), *d.get("parent_asset_ids", [])}
                if ids & self.excluded_ids or d.get("source_sha256") in self.excluded_hashes:
                    before = len(self.excluded_ids) + len(self.excluded_hashes)
                    self.excluded_ids.update({d.get("asset_id"), d.get("public_asset_id")} - {None})
                    self.excluded_hashes.update({d.get("content_sha256")} - {None})
                    changed |= before != len(self.excluded_ids) + len(self.excluded_hashes)
            if not changed:
                break
        images, candidates, seen, source_images = [], [], {}, []
        for index, d in enumerate(descriptors):
            entry = dict(image_index=index, width=d["width"], height=d["height"])
            images.append(entry)
            if d.get("asset_id") in self.excluded_ids or d.get("source_sha256") in self.excluded_hashes:
                entry["status"] = "skipped_non_rgb"
                self.stats["skipped_images"] += 1
                continue
            path = Path(frames) / f"{index:03d}.png"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            source_images.append(dict(image_index=index, png_sha256=digest,
                                      content_sha256=d.get("content_sha256"), asset_id=d.get("asset_id")))
            if digest in seen:
                entry.update(status="duplicate", same_detections_as=seen[digest])
                self.stats["cache_hits"] += 1
                continue
            seen[digest] = index
            if digest not in self.cache:
                result = self.infer(path, digest)
                self.cache[digest] = result
                self.stats["inferred_images"] += 1
                self.stats["inference_seconds"] += result["elapsed_seconds"]
                self.stats["cuda_peak_allocated_bytes"] = max(
                    self.stats["cuda_peak_allocated_bytes"], result["cuda_peak_allocated_bytes"])
            else:
                self.stats["cache_hits"] += 1
            result = self.cache[digest]
            entry.update(status=result["status"], total_detections=result["total_detections"])
            candidates.append((index, result["detections"]))
        # Round-robin keeps later video frames represented in the bounded text budget.
        detections = [dict(image_index=index, **values[rank]) for rank in range(6)
                      for index, values in candidates if rank < len(values)][:48]
        public = dict(model="YOLOE-11s-seg-pf", kind="estimated_visual_observation",
                      note="image_index is the zero-based attached image / frames/NNN.png index. "
                           "Boxes are xyxy pixels in that image, not world coordinates. "
                           "Labels may be wrong or incomplete; use original images. No cross-view identity inferred.",
                      images=images, detections=detections, max_detections_per_image=6,
                      max_detections_total=48)
        with (self.outputs / "yoloe_observations.jsonl").open("a") as f:
            f.write(json.dumps(dict(observation=public, stats=self.stats,
                                   source_images=source_images,
                                   feedback_seconds=time.monotonic() - started)) + "\n")
        return public

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        self.log.close()
