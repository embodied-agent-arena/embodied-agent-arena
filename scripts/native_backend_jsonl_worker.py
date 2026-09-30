#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, TextIO

JsonDict = dict[str, Any]
_PROTOCOL_STDOUT: TextIO | None = None
LIVE_OFFICIAL_CAPTURE_SCHEMA = "agentic-embodied-arena/live-official-worker-capture/v1"


def _process_identity() -> JsonDict:
    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    tail = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": os.getpid(),
        "sid": os.getsid(0),
        "starttime_ticks": int(tail[19]),
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
    }


def _isolate_protocol_stdout() -> None:
    """Reserve the original stdout pipe for JSONL and send native noise to stderr."""
    global _PROTOCOL_STDOUT
    if _PROTOCOL_STDOUT is not None:
        return
    protocol_fd = os.dup(sys.stdout.fileno())
    _PROTOCOL_STDOUT = os.fdopen(
        protocol_fd,
        "w",
        encoding="utf-8",
        buffering=1,
        closefd=True,
    )
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())


def _jsonable(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    module_root = type(value).__module__.partition(".")[0]
    if module_root in {"jax", "jaxlib", "numpy", "torch"}:
        tolist = getattr(value, "tolist", None)
        if callable(tolist):
            return _jsonable(tolist())
        item = getattr(value, "item", None)
        if callable(item):
            return _jsonable(item())
    return value


def _make_backend(case_id: str):
    if case_id.startswith("vlabench"):
        from embodied_harness.native_resource_admission import admit_vlabench
        admit_vlabench()
    from embodied_harness.native_case_registry import load_native_case

    case = load_native_case(case_id)
    return case.backend_factory()


def _handle(
    backend: Any, method: str, args: JsonDict, state: JsonDict | None = None
) -> Any:
    state = state if state is not None else {}
    if method == "reset":
        result = backend.reset(
            task_id=str(args["task_id"]),
            seed=args.get("seed"),
            config=dict(args.get("config") or {}),
        )
        from embodied_harness.native_episode_budget import apply_native_episode_budget
        apply_native_episode_budget(backend, result)
        state.clear()
        state.update(
            episode_instance_nonce=secrets.token_hex(32),
            capture_consumed=False,
            reset_complete=True,
        )
        return result
    if method == "bind_pool_coordinate":
        hook = getattr(backend, "bind_pool_coordinate", None)
        if not callable(hook):
            return {
                "mode": "coordinate_only",
                "bound": False,
                "reason": "adapter_has_no_bind_pool_coordinate_hook",
            }
        result = hook(dict(args.get("coordinate") or {}))
        if isinstance(result, dict):
            return result
        return {"mode": "adapter_hook", "bound": True, "result": result}
    if method == "capture_rgb":
        if not state.get("reset_complete"):
            raise RuntimeError("RGB capture requires a live reset episode")
        from embodied_harness.w4_rgb import capture_rgb
        return capture_rgb(backend, **args)
    if method == "observe":
        return backend.observe()
    if method == "list_primitives":
        return backend.list_primitives(level=args.get("level"))
    if method == "call_primitive":
        return backend.call_primitive(
            str(args["name"]), **dict(args.get("kwargs") or {})
        )
    if method == "verify":
        return backend.verify(
            scope=str(args.get("scope") or "task"), **dict(args.get("kwargs") or {})
        )
    if method == "get_trace":
        trace = backend.get_trace()
        if args.get("metadata_only"):
            # Trace enrichment does not consume events; avoid serializing the
            # entire history after every primitive. Full trace requests remain
            # unchanged for callers that need the complete log.
            artifact_ids = args.get("artifact_ids")
            artifacts = trace.artifacts if artifact_ids is None else {
                key: trace.artifacts[key] for key in artifact_ids if key in trace.artifacts
            }
            return {"task_id": trace.task_id, "events": [], "artifacts": artifacts,
                    "metrics": trace.metrics, "final_status": trace.final_status}
        return trace
    if method == "get_live_episode_binding":
        if state.get("reset_complete") is not True:
            raise RuntimeError("live official binding requires reset")
        return {
            "schema_version": "agentic-embodied-arena/live-episode-binding/v1",
            "operation_id": args.get("operation_id"),
            "episode_instance_nonce": state["episode_instance_nonce"],
            "native_runtime": _process_identity(),
        }
    if method == "capture_live_official":
        if state.get("reset_complete") is not True:
            raise RuntimeError("live official capture requires reset")
        if state.get("capture_consumed") is True:
            raise RuntimeError("live official capture already consumed")
        expected_nonce = args.get("expected_episode_instance_nonce")
        if expected_nonce != state.get("episode_instance_nonce"):
            raise RuntimeError("live official episode instance mismatch")
        runtime = _process_identity()
        expected_runtime = args.get("expected_native_runtime")
        if expected_runtime is not None and expected_runtime != runtime:
            raise RuntimeError("live official native runtime identity mismatch")
        state["capture_consumed"] = True
        from embodied_harness.live_official_runtime import capture_live_official_result

        receipt = capture_live_official_result(
            native_backend=backend,
            operation_id=str(args.get("operation_id") or ""),
            identity=dict(args.get("identity") or {}),
            environment_digest=str(args.get("environment_digest") or ""),
            transcript_digest=str(args.get("transcript_digest") or ""),
        )
        return {
            "schema_version": LIVE_OFFICIAL_CAPTURE_SCHEMA,
            "operation_id": receipt["operation_id"],
            "case_id": receipt["identity"]["case_id"],
            "episode_instance_nonce": state["episode_instance_nonce"],
            "native_runtime": runtime,
            "transcript_digest": receipt["transcript_digest"],
            "request_identity": receipt["identity"],
            "environment_digest": receipt["environment_digest"],
            "official_result": {
                "source": receipt["source"],
                "success": receipt["success"],
                "success_semantics": receipt["success_semantics"],
                "started_at": receipt["started_at"],
                "finished_at": receipt["finished_at"],
                "episode_state_before": receipt["episode_state_before"],
                "episode_state_after": receipt["episode_state_after"],
                "stdout": receipt["stdout"],
                "stderr": receipt["stderr"],
                "payload": receipt["official_payload"],
            },
            "official_receipt": receipt,
        }
    if method == "close":
        # run_worker.finally drains video before closing the simulator.
        # OmniGibson shutdown can exit the process without Python finally.
        raise EOFError
    raise ValueError(f"Unknown worker method: {method}")


def _emit(payload: JsonDict) -> None:
    protocol_stdout = _PROTOCOL_STDOUT or sys.stdout
    protocol_stdout.write(
        json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    )
    protocol_stdout.flush()


def run_worker(case_id: str) -> int:
    backend = _make_backend(case_id)
    state: JsonDict = {}
    from embodied_harness.case_study_recording import capture_backend, install_native_taps
    install_native_taps(backend)
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                request_id = str(request.get("id") or "")
                result = _handle(
                    backend,
                    str(request.get("method") or ""),
                    dict(request.get("args") or {}),
                    state,
                )
            except EOFError:
                return 0
            except Exception as exc:  # noqa: BLE001 - bridge reports native failures.
                _emit(
                    {
                        "id": str(locals().get("request_id") or ""),
                        "ok": False,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                )
                continue
            if request.get("method") in {"reset", "observe", "call_primitive"}:
                capture_backend(backend, result)
            _emit({"id": request_id, "ok": True, "result": _jsonable(result)})
        return 0
    finally:
        from embodied_harness.stream_video import close_all
        close_all()
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="JSONL worker for benchmark-native EmbodiedBackend processes."
    )
    parser.add_argument("--case-id", required=True)
    args = parser.parse_args(argv)
    _isolate_protocol_stdout()
    return run_worker(args.case_id)


if __name__ == "__main__":
    raise SystemExit(main())
