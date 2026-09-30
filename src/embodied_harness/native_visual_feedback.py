"""Attach only RGB explicitly returned by CLIPort's public observation primitive."""
from __future__ import annotations

import hashlib
from pathlib import Path


class PublicRGBFeedback:
    def __init__(self, directory: Path):
        self.directory = directory
        self.images: list[dict] = []
        self.observation_count = 0

    def clear(self):
        self.images = []

    def capture(self, name, result):
        # Never inspect backend state, verifier results, arbitrary files or depth.
        if name == "submit_cliport_pick_place_action":
            # Even a failed native action can have partially moved the scene.
            self.clear()
            return
        if name != "observe_cliport_rgbd" or not result.ok:
            return
        raw = result.output.get("raw_observation", {})
        colors = raw.get("color", []) if isinstance(raw, dict) else []
        if not colors:
            return
        import numpy as np
        from PIL import Image

        images, seen = [], set()
        for camera_index, pixels in enumerate(colors[:4]):
            array = np.asarray(pixels)
            if (array.ndim != 3 or array.shape[2] != 3 or
                    not 0 < array.shape[0] <= 4096 or not 0 < array.shape[1] <= 4096 or
                    array.dtype.kind not in "ui" or array.min() < 0 or array.max() > 255):
                raise ValueError("Public CLIPort RGB must be an HxWx3 byte-valued image")
            array = array.astype(np.uint8)
            digest = hashlib.sha256(str(array.shape).encode() + array.tobytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"{digest}.png"
            if not path.exists():
                Image.fromarray(array).save(path)
            images.append(dict(path=str(path), rgb_sha256=digest, camera_index=camera_index,
                               width=array.shape[1], height=array.shape[0],
                               source_primitive=name))
        self.images = images

    def trace_result(self, result):
        """Keep complete public RGBD once; trace references it instead of copying pixels."""
        import gzip
        import json

        payload = result.to_dict()
        output = payload["output"]
        raw = {"raw_observation": output.pop("raw_observation", None), "camera_raw": {}}
        for camera, observation in output.get("camera_observations", {}).items():
            if isinstance(observation, dict) and "raw" in observation:
                raw["camera_raw"][camera] = observation.pop("raw")
        if raw["raw_observation"] is None and not raw["camera_raw"]:
            return payload
        self.observation_count += 1
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"observation_{self.observation_count:04d}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as stream:
            json.dump(raw, stream, separators=(",", ":"))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output["public_raw_archive"] = {"path": str(path), "sha256": digest,
                                        "encoding": "gzip-json", "schema": "cliport-public-raw/v1"}
        output["model_rgb_files"] = list(self.images)
        return payload

    @property
    def paths(self):
        return [Path(item["path"]) for item in self.images]


CLIPORT_VISUAL_INSTRUCTIONS = (
    "For visual identification call observe_cliport_rgbd with include_raw=True, then yield. "
    "The harness attaches those public RGB camera images to the next model request in camera order. "
    "Do not print or convert pixel arrays into text. Print only compact task, object and action summaries. "
    "After an action, request a new public RGB observation at the end of the cell if another visual "
    "decision is needed. Latest public RGB remains attached during read-only inspection; an action "
    "invalidates it until you observe again. Use the documented public instance evidence to bind visible objects. "
    "For PickPlace tasks, pick_pose/place_pose are suction end-effector poses, while an inspected "
    "instance's pose0 is its object-body pose. Reuse the pose format, not an arbitrary object's wrist "
    "orientation. The native suction picker uses quaternion [0, 0, 0, 1] for a downward pick; tilted "
    "object quaternions can make the pre-pick motion unreachable. Choose the contact point and "
    "placement using the public geometry and images. Push tasks retain their native push convention. "
    "Use direct documented attributes such as result.ok and result.output; dynamic attribute access "
    "through getattr/hasattr is unavailable in the code sandbox."
)
