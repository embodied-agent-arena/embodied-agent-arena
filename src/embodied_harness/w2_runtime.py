"""W2 environment facade for the existing Pukun persistent Python loop.

Only public observations/media cross the RPC boundary. The upstream adapter
owns actions, submission parsing, private references and final evaluation.
"""
import base64
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

BENCHMARKS = {
    "mmsi_bench": "MMSI-Bench", "mindcube": "MindCube", "vsi_bench": "VSI-Bench",
    "3dsrbench": "3DSRBench", "robospatial": "RoboSpatial-Home", "bop_ask": "BOP-ASK",
}


class W2Session:
    runtime_mode = "persistent_repl_code"

    def __init__(self, adapter, sample_id, workspace, outputs, run_id, *, executor, max_images=32,
                 perception=None):
        from w2_harness.offline_arena.adapters import build_offline_environment
        from w2_harness.offline_arena.media_transport import MediaTransport
        self.trace = SimpleNamespace(run_id=run_id)
        self.workspace, self.outputs = Path(workspace), Path(outputs)
        self.executor, self.max_images = executor, max_images
        self.turn, self.pending_media, self.steps = 0, False, 0
        self.owner_pid = None
        self.perception, self.last_perception = perception, None
        self.image_bundle = []
        self.environment = build_offline_environment(
            adapter, sample_id, harness_profile="w2_light", observation_policy="full_context",
            media_transport=MediaTransport(self.outputs / "media_cache"),
            checkpoint_path=self.outputs / "environment.checkpoint.json", event_sink=self.event,
        )
        self.environment.reset()

    def event(self, value):
        with (self.outputs / "events.jsonl").open("a") as f:
            f.write(json.dumps(value, default=str) + "\n")

    def record_execution_started(self, **kwargs):
        self.event(dict(event="execution_started", **kwargs))

    def record_agent_code(self, path):
        self.turn += 1
        # This callback runs after a successful model response, before execution.
        # Pending media were attached by the existing author in the previous turn.
        if self.executor != "probe":
            self.environment.record_model_call()
            if self.pending_media:
                self.environment.commit_model_visible_media(
                    request_id=f"{self.trace.run_id}:turn:{self.turn}", cache_hit=False)
                self.pending_media = False
        self.event(dict(event="code", turn=self.turn, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()))

    def observe(self):
        observation = self.environment.observe().to_dict()
        return self._public_observation(observation)

    def _public_observation(self, observation):
        # Full native records stay in the trace. Keep actionable IDs and task semantics here.
        audit_keys = {"source_media_hash", "content_sha256", "source_sha256", "derivation_sha256",
                      "evidence_binding_contract_hash", "runtime_path_exposed", "pixel_claim",
                      "selection_basis", "source_frame_index_status", "source_timestamp_status"}
        def clean(value):
            if isinstance(value, dict):
                return {k: clean(v) for k, v in value.items() if k not in audit_keys and v is not None}
            if isinstance(value, list):
                return [clean(v) for v in value]
            return value
        public = {"observation_format": "w2_public_v2"}
        for key in ("task_summary", "remaining_budget", "terminal", "lifecycle_state", "safe_error",
                    "available_actions", "valid_submission_evidence_refs", "action_result"):
            if key in observation:
                public[key] = clean(observation[key])
        public["transport_images"] = self.image_bundle
        if self.last_perception is not None:
            public["perception"] = self.last_perception
        primary = {image["bindings"][0]["asset_id"] for image in self.image_bundle}
        aliases = {b["asset_id"] for image in self.image_bundle for b in image["bindings"]} - primary
        public["visible_assets"] = clean([a for a in observation.get("visible_assets", [])
                                          if a["asset_id"] not in aliases])
        public["answer_bearing_media_visible"] = bool(self.image_bundle)
        public["image_note"] = ("Initial public images are attached automatically after the first cell. "
                                "Do not OPEN_ASSET/GET_VIEW an already attached image; it still costs a native action. "
                                "image_index identifies the attached image; bindings preserve aliases for identical pixels.")
        return public

    @staticmethod
    def model_stdout(stdout):
        """Remove only full observation echoes; preserve computations and other agent output."""
        if stdout and len(stdout) < 200000:
            try:
                value = json.loads(stdout)
                if isinstance(value, dict) and value.get("observation_format") == "w2_public_v2":
                    return "[Public observation is included in current_observation below.]"
            except (ValueError, RecursionError):
                pass
        lines = []
        for line in (stdout or "").splitlines():
            if len(line) < 200000 and "w2_public_v2" in line:
                try:
                    value = ast.literal_eval(line)
                    if isinstance(value, dict) and value.get("observation_format") == "w2_public_v2":
                        lines.append("[Public observation is included in current_observation below.]")
                        continue
                except (ValueError, SyntaxError, RecursionError):
                    pass
            lines.append(line)
        return "\n".join(lines)[-4000:]

    def feedback_observation(self):
        observation = self.environment.observe().to_dict()
        if observation["terminal"] or observation["lifecycle_state"] != "ACTIVE":
            return self._public_observation(observation)
        media = self.environment.model_media(observation)
        frames = self.workspace / "frames"
        frames.mkdir(exist_ok=True)
        # Fresh per-turn images give the shared author the exact current bundle.
        for path in frames.glob("*.png"):
            path.unlink()
        from PIL import Image
        import io
        descriptors, bundle, seen = [], [], {}
        assets = {a["asset_id"]: a for a in observation.get("visible_assets", [])}
        for item in media:
            header, payload = item["data_url"].split(",", 1)
            if not header.startswith("data:image/") or ";base64" not in header:
                raise ValueError("Expected native inline image media")
            with Image.open(io.BytesIO(base64.b64decode(payload, validate=True))) as img:
                rgb = img.convert("RGB")
                digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                binding = dict(asset_id=item.get("public_asset_id", item["asset_id"]),
                               evidence_ref=item["evidence_ref"])
                metadata = assets.get(binding["asset_id"], {}).get("metadata", {})
                frame_index = item.get("frame_index")
                if frame_index is None:
                    frame_index = metadata.get("source_frame_index", metadata.get("frame_index"))
                if frame_index is not None:
                    binding["frame_index"] = frame_index
                timestamp = item.get("timestamp_ms")
                if timestamp is None and metadata.get("timestamp_seconds") is not None:
                    timestamp = round(metadata["timestamp_seconds"] * 1000, 3)
                if timestamp is not None:
                    binding["timestamp_ms"] = timestamp
                if digest in seen:
                    if binding not in bundle[seen[digest]]["bindings"]:
                        bundle[seen[digest]]["bindings"].append(binding)
                    continue
                index = len(bundle)
                if index >= self.max_images:
                    self.environment.discard_provisional_model_visible_media("image_budget_exceeded")
                    raise RuntimeError(f"W2 unique image budget exceeded: {self.max_images}")
                seen[digest] = index
                path = frames / f"{index:03d}.png"
                rgb.save(path)
                from .case_study_recording import publish_image
                publish_image(rgb, f"w2.visible_image.{index}", source_path=path, source_metadata=binding)
                bundle.append(dict(image_index=index, width=rgb.width, height=rgb.height, bindings=[binding]))
            if path.stat().st_size > 10 * 1024 * 1024:
                raise ValueError("Public image exceeds shared author transport limit")
            descriptors.append({k: v for k, v in item.items() if k != "data_url"})
        self.pending_media = bool(media)
        # Project and later commit every native binding: deduplication changes transport only.
        public = self.environment.project_model_visible_media(observation, media)
        self.image_bundle = bundle
        if self.perception is not None:
            self.last_perception = self.perception.observe(descriptors, frames, public["visible_assets"])
        self.event(dict(event="public_observation_full", turn=self.turn, observation=public,
                        transport_images=[{k: v for k, v in item.items() if k != "data_url"} for item in media],
                        image_bundle=bundle, perception=self.last_perception))
        return self._public_observation(public)

    def call(self, name, args, kwargs, client):
        if args:
            raise ValueError("Use named arguments")
        pid = int(client.get("pid", 0))
        if pid <= 0 or Path(client.get("argv0", "")).name != "solve.py":
            raise ValueError("W2 primitives require the persistent Python worker")
        if self.owner_pid is None:
            self.owner_pid = pid
        if self.owner_pid != pid:
            raise ValueError("Episode belongs to another interpreter")
        if name == "observe":
            return self.observe()
        if name == "submit":
            if self.executor != "probe" and self.turn < 2:
                raise ValueError("First observe and yield to receive the actual images")
            result = self.environment.submit(kwargs)
        elif name == "step":
            if kwargs.get("action") == "SUBMIT":
                return self.call("submit", [], kwargs.get("arguments", {}), client)
            result = self.environment.step(kwargs)
        else:
            raise ValueError("Only observe, step, and submit are exposed; evaluate is private")
        self.steps = 8 - result.remaining_budget["environment_actions"]
        return self._public_observation(result.to_dict())

    def finish(self):
        if self.environment.state.value in {"SUBMITTED", "EVALUATED"}:
            return self.environment.evaluate().to_dict()
        return None

    def close(self):
        self.environment.close()
