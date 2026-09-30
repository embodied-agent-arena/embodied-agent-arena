"""W1 open-loop geometry facade. Only public RGB and the task text leave the session."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from .open_loop_media import IMAGE_NOTE, clear_transport_frames, write_transport_frame
from .w1_scoring import parse_submission, score_case
from .w1_vlm_scoring import BENCHMARKS as VLM_BENCHMARKS
from .w1_output import public_question

BENCHMARKS = {
    "multispa": "MultiSPA",
    "influx": "InFlux",
    "mapfree": "Map-free Relocalization",
    **VLM_BENCHMARKS,
    "tracespatial_2d": "TraceSpatial-Bench 2D",
    "tracespatial_3d": "TraceSpatial-Bench 3D",
}


def load_catalog(data_root: Path) -> dict[str, dict]:
    catalog = data_root / "catalog.jsonl"
    if not catalog.is_file():
        raise FileNotFoundError(f"W1 catalog missing: {catalog}")
    cases = {}
    for line in catalog.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        sample_id = record.get("sample_id")
        if not sample_id:
            raise ValueError(f"Catalog row missing sample_id in {catalog}")
        if sample_id in cases:
            raise ValueError(f"Duplicate sample_id in {catalog}: {sample_id}")
        cases[sample_id] = record
    return cases


class W1Session:
    runtime_mode = "persistent_repl_code"

    def __init__(self, case, data_root, workspace, outputs, run_id, *, executor, max_images=32):
        self.case = dict(case)
        self.data_root = Path(data_root)
        self.workspace = Path(workspace)
        self.outputs = Path(outputs)
        self.trace = SimpleNamespace(run_id=run_id)
        self.executor = executor
        self.max_images = max_images
        self.turn = 0
        self.steps = 0
        self.owner_pid = None
        self.pending_media = False
        self.image_bundle = []
        self.terminal = False
        self.submitted = None
        self.evaluation = None
        self.max_env_steps = 8

    def event(self, value):
        with (self.outputs / "events.jsonl").open("a") as f:
            f.write(json.dumps(value, default=str) + "\n")

    def record_execution_started(self, **kwargs):
        self.event(dict(event="execution_started", **kwargs))

    def record_agent_code(self, path):
        self.turn += 1
        if self.executor != "probe" and self.pending_media:
            self.pending_media = False
        self.event(dict(event="code", turn=self.turn,
                        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))

    def _public_task(self):
        images = []
        for index, item in enumerate(self.case.get("images") or []):
            images.append({
                "image_index": index,
                "role": item.get("role"),
                "width": item.get("width"),
                "height": item.get("height"),
            })
        return {
            "benchmark": BENCHMARKS[self.case["benchmark"]],
            "task_family": self.case.get("task_family"),
            "question": (public_question(self.case["question"])
                         if self.case["benchmark"] in VLM_BENCHMARKS else self.case.get("question")),
            "answer_type": self.case.get("answer_type"),
            "answer_schema": self.case.get("answer_schema"),
            "units": self.case.get("units"),
            "public_context": self.case.get("public_context") or {},
            "images": images,
        }

    def _remaining(self):
        return {
            "environment_actions": max(0, self.max_env_steps - self.steps),
            "submission_remaining": 0 if self.submitted is not None else 1,
        }

    def observe(self):
        assets = []
        for index, item in enumerate(self.case.get("images") or []):
            assets.append({
                "asset_id": item.get("role") or f"image_{index}",
                "kind": "rgb",
                "metadata": {"role": item.get("role"), "width": item.get("width"),
                             "height": item.get("height")},
            })
        return {
            "observation_format": "w2_public_v2",
            "task_summary": self._public_task(),
            "remaining_budget": self._remaining(),
            "terminal": self.terminal,
            "lifecycle_state": "EVALUATED" if self.evaluation else ("SUBMITTED" if self.submitted else "ACTIVE"),
            "available_actions": ["SUBMIT"],
            "valid_submission_evidence_refs": [a["asset_id"] for a in assets],
            "visible_assets": assets,
            "transport_images": self.image_bundle,
            "answer_bearing_media_visible": bool(self.image_bundle),
            "image_note": IMAGE_NOTE,
        }

    def feedback_observation(self):
        if self.terminal:
            return self.observe()
        frames = self.workspace / "frames"
        frames.mkdir(exist_ok=True)
        clear_transport_frames(frames)
        from PIL import Image
        from .case_study_recording import publish_image
        bundle = []
        for index, item in enumerate(self.case.get("images") or []):
            if index >= self.max_images:
                raise RuntimeError(f"W1 unique image budget exceeded: {self.max_images}")
            source = self.data_root / item["path"]
            if not source.is_file():
                raise FileNotFoundError(f"W1 image missing: {source}")
            if item.get("sha256") and hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError("W1 frozen RGB hash mismatch")
            path, native, transport = write_transport_frame(source, frames, index)
            with Image.open(path) as img:
                rgb = img.convert("RGB")
            publish_image(rgb, f"w1.visible_image.{index}", source_path=path,
                          source_metadata={"role": item.get("role"), "path": item["path"],
                                           "native_width": native[0], "native_height": native[1]})
            catalog_w = item.get("width") or native[0]
            catalog_h = item.get("height") or native[1]
            bundle.append(dict(image_index=index, width=catalog_w, height=catalog_h,
                               transport_width=transport[0], transport_height=transport[1],
                               bindings=[dict(asset_id=item.get("role") or f"image_{index}",
                                              role=item.get("role"), path=str(path.relative_to(self.workspace)))]))
        self.pending_media = bool(bundle)
        self.image_bundle = bundle
        self.event(dict(event="public_observation_full", turn=self.turn, image_bundle=bundle))
        return self.observe()

    @staticmethod
    def model_stdout(stdout):
        if stdout and "w2_public_v2" in stdout:
            return "[Public observation is included in current_observation below.]"
        return (stdout or "")[-4000:]

    def call(self, name, args, kwargs, client):
        if args:
            raise ValueError("Use named arguments")
        pid = int(client.get("pid", 0))
        if pid <= 0 or Path(client.get("argv0", "")).name != "solve.py":
            raise ValueError("W1 primitives require the persistent Python worker")
        if self.owner_pid is None:
            self.owner_pid = pid
        if self.owner_pid != pid:
            raise ValueError("Episode belongs to another interpreter")
        if name == "observe":
            return self.observe()
        if name == "submit":
            return self.submit(kwargs)
        if name == "step":
            if kwargs.get("action") != "SUBMIT":
                raise ValueError("W1 only exposes SUBMIT")
            return self.submit(kwargs.get("arguments") or {})
        raise ValueError("Only observe, step, and submit are exposed; evaluate is private")

    def submit(self, kwargs):
        if self.executor != "probe" and not self.image_bundle:
            raise ValueError("Public images must be attached before submit")
        if (self.case["benchmark"] in VLM_BENCHMARKS and self.executor != "probe"
                and (self.turn < 2 or self.pending_media)):
            raise ValueError("First observe and yield; submit only after the model receives the images")
        if self.submitted is not None:
            raise ValueError("Answer already submitted")
        self.steps += 1
        self.submitted = parse_submission(kwargs.get("answer", kwargs))
        self.terminal = True
        self.evaluation = score_case(self.submitted, self.case)
        self.event(dict(event="submitted", prediction=self.submitted, evaluation=self.evaluation))
        return self.observe()

    def finish(self):
        if self.evaluation is None:
            return None
        if self.evaluation.get("evaluation_valid") is False:
            return {
                "passed": None, "task_outcome": "invalid_ground_truth",
                "official_score": None, "official_score_eligible": False,
                "submission_valid": True, "prediction": self.submitted,
                "metrics": self.evaluation,
            }
        if self.case["benchmark"] in VLM_BENCHMARKS:
            valid = self.evaluation.get("submission_valid", False)
            passed = self.evaluation.get("passed")
            return dict(passed=passed, submission_valid=valid, prediction=self.submitted,
                        official_score=None, official_score_eligible=False,
                        score_kind="local_diagnostic", metrics=self.evaluation,
                        task_outcome="invalid_prediction" if not valid else
                        "scored" if passed is None else "correct" if passed else "incorrect")
        return {
            "passed": bool(self.evaluation.get("passed")),
            "task_outcome": "correct" if self.evaluation.get("passed") else "incorrect",
            "official_score": self.evaluation.get("official_score"),
            "submission_valid": True,
            "prediction": self.submitted,
            "metrics": self.evaluation,
        }

    def close(self):
        return None
