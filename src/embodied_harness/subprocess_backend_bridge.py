from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import selectors
import socket
import subprocess
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend import EmbodiedBackend
from .schemas import (
    EpisodeTrace,
    Observation,
    PrimitiveCard,
    PrimitiveResult,
    TaskSpec,
    TraceEvent,
    VerificationResult,
)

JsonDict = dict[str, Any]
NATIVE_XVFB_ENV = "AGENTIC_EMBODIED_ARENA_NATIVE_XVFB"
NATIVE_XVFB_SCRIPT_ENV = "AGENTIC_EMBODIED_ARENA_NATIVE_XVFB_SCRIPT"
LIVE_OFFICIAL_SOCKET_ENV = "AGENTIC_EMBODIED_ARENA_LIVE_OFFICIAL_SOCKET"
LIVE_OFFICIAL_AUTH_SHA256_ENV = "AGENTIC_EMBODIED_ARENA_LIVE_OFFICIAL_AUTH_SHA256"
LIVE_OFFICIAL_BRIDGE_SCHEMA = "agentic-embodied-arena/live-official-bridge-capture/v1"


def _observation_from_dict(data: JsonDict) -> Observation:
    return Observation(
        step=int(data.get("step", 0)),
        data=dict(data.get("data") or {}),
        artifacts=list(data.get("artifacts") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _primitive_card_from_dict(data: JsonDict) -> PrimitiveCard:
    return PrimitiveCard(
        name=str(data["name"]),
        capability_tags=list(data.get("capability_tags") or []),
        input_schema=dict(data.get("input_schema") or {}),
        output_schema=dict(data.get("output_schema") or {}),
        preconditions=list(data.get("preconditions") or []),
        side_effects=list(data.get("side_effects") or []),
        cost=dict(data.get("cost") or {}),
        failure_modes=list(data.get("failure_modes") or []),
        abstraction_level=str(data.get("abstraction_level") or "L1"),
        leakage_risk=str(data.get("leakage_risk") or "none"),
        description=str(data.get("description") or ""),
    )


def _primitive_result_from_dict(data: JsonDict) -> PrimitiveResult:
    return PrimitiveResult(
        name=str(data.get("name") or ""),
        ok=bool(data.get("ok")),
        output=dict(data.get("output") or {}),
        artifacts=list(data.get("artifacts") or []),
        error=data.get("error"),
        metadata=dict(data.get("metadata") or {}),
    )


def _verification_result_from_dict(data: JsonDict) -> VerificationResult:
    return VerificationResult(
        ok=bool(data.get("ok")),
        scope=str(data.get("scope") or "task"),
        message=str(data.get("message") or ""),
        metrics=dict(data.get("metrics") or {}),
        leaked_fields=list(data.get("leaked_fields") or []),
        metadata=dict(data.get("metadata") or {}),
    )


def _trace_from_dict(data: JsonDict) -> EpisodeTrace:
    trace = EpisodeTrace(
        task_id=str(data.get("task_id") or ""),
        artifacts=dict(data.get("artifacts") or {}),
        metrics=dict(data.get("metrics") or {}),
        final_status=data.get("final_status"),
    )
    for event in data.get("events") or []:
        if not isinstance(event, dict):
            continue
        trace.events.append(
            TraceEvent(
                event_type=str(event.get("event_type") or "event"),
                step=int(event.get("step") or 0),
                payload=dict(event.get("payload") or {}),
                timestamp=float(event.get("timestamp") or 0.0),
            )
        )
    return trace


def _process_identity() -> JsonDict:
    stat = Path("/proc/self/stat").read_text(encoding="utf-8")
    tail = stat[stat.rfind(")") + 2 :].split()
    return {
        "pid": os.getpid(),
        "sid": os.getsid(0),
        "starttime_ticks": int(tail[19]),
        "mount_namespace": os.readlink("/proc/self/ns/mnt"),
    }


def _json_digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(slots=True)
class SubprocessBackendBridge(EmbodiedBackend):
    """Proxy a native benchmark backend through a JSONL worker process.

    This keeps the lightweight controller on its own Python while letting heavy
    simulators run inside their benchmark-native Python ABI.
    """

    case_id: str
    python_executable: str
    worker_script: str | Path
    cwd: str | Path
    env: dict[str, str] = field(default_factory=dict)
    inherit_environment: bool = True
    timeout_seconds: float = 900.0
    strict_episode: bool = False
    episode_lost: bool = field(default=False, init=False)
    policy_timeout_seconds: float | None = None
    policy_timeout_primitives: tuple[str, ...] = ("execute_calvin_language_skill",)
    _process: subprocess.Popen[str] | None = field(default=None, init=False, repr=False)
    _stderr_file: io.TextIOBase | None = field(default=None, init=False, repr=False)
    _noise_lines: list[str] = field(default_factory=list, init=False, repr=False)
    _trace_cache: EpisodeTrace | None = field(default=None, init=False, repr=False)
    _active_timeout_seconds: float | None = field(default=None, init=False, repr=False)
    _active_request_label: str = field(default="", init=False, repr=False)
    _last_reset_args: JsonDict | None = field(default=None, init=False, repr=False)
    _process_reset_applied: bool = field(default=False, init=False, repr=False)
    _episode_binding: JsonDict | None = field(default=None, init=False, repr=False)
    _request_lock: threading.RLock = field(
        default_factory=threading.RLock, init=False, repr=False
    )
    _live_socket: socket.socket | None = field(default=None, init=False, repr=False)
    _live_socket_thread: threading.Thread | None = field(
        default=None, init=False, repr=False
    )
    _live_socket_stop: threading.Event = field(
        default_factory=threading.Event, init=False, repr=False
    )
    _live_capture_consumed: bool = field(default=False, init=False, repr=False)

    def reset(
        self, task_id: str, seed: int | None = None, config: JsonDict | None = None
    ) -> TaskSpec:
        self._stop_live_official_socket()
        if self.strict_episode and self._process is not None and self._process.poll() is not None:
            self._mark_process_unusable(self._process)
        self.episode_lost = False  # Only an explicit reset may begin another episode.
        reset_args = {"task_id": task_id, "seed": seed, "config": dict(config or {})}
        payload = self._request("reset", reset_args)
        task = TaskSpec.from_dict(payload)
        self._trace_cache = EpisodeTrace(task_id=task.task_id)
        self._last_reset_args = dict(reset_args)
        self._process_reset_applied = True
        if (
            self.env.get(LIVE_OFFICIAL_SOCKET_ENV) is not None
            or self.env.get(LIVE_OFFICIAL_AUTH_SHA256_ENV) is not None
        ):
            self._episode_binding = dict(
                self._request(
                    "get_live_episode_binding", {"operation_id": self.case_id}
                )
            )
        self._live_capture_consumed = False
        self._start_live_official_socket_if_configured()
        return task

    def bind_pool_coordinate(self, coordinate: JsonDict) -> JsonDict:
        """Offer a frozen pool coordinate to an optional native adapter hook.

        The controller deliberately keeps the benchmark worker's environment
        isolated.  This small JSONL method is the explicit, non-secret seam
        for adapters that can map a pool task/episode to their native reset;
        adapters without the hook return ``coordinate_only`` from the worker.
        """

        payload = self._request(
            "bind_pool_coordinate", {"coordinate": dict(coordinate)}
        )
        return dict(payload) if isinstance(payload, dict) else {"result": payload}

    def capture_rgb(self, **kwargs):
        return self._request("capture_rgb", kwargs)

    def observe(self) -> Observation:
        return _observation_from_dict(self._request("observe", {}))

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        payload = self._request("list_primitives", {"level": level})
        return [_primitive_card_from_dict(item) for item in payload]

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        with self._request_context(
            request_label=f"call_primitive:{name}",
            timeout_seconds=self._timeout_for_call_primitive(name),
        ):
            payload = self._request("call_primitive", {"name": name, "kwargs": kwargs})
        result = _primitive_result_from_dict(payload)
        if result.artifacts:
            self._merge_remote_trace_metadata(artifact_ids=result.artifacts)
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        payload = self._request("verify", {"scope": scope, "kwargs": kwargs})
        result = _verification_result_from_dict(payload)
        self._merge_remote_trace_metadata()
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._trace_cache is None:
            self._trace_cache = _trace_from_dict(self._request("get_trace", {}))
        return self._trace_cache

    @property
    def noise_lines(self) -> list[str]:
        return list(self._noise_lines)

    def close(self) -> None:
        self._stop_live_official_socket()
        process = self._process
        if process is None:
            return
        graceful_close_sent = False
        try:
            self._send({"id": str(uuid.uuid4()), "method": "close", "args": {}})
            graceful_close_sent = True
        except Exception:
            pass
        if graceful_close_sent:
            try:
                # Streaming encoders drain queued frames and close containers
                # on worker exit. Do not terminate them after only ten seconds.
                process.wait(timeout=75.0 if os.environ.get("ARENA_CASE_STUDY_FORMAT") == "mp4" else 10.0)
            except subprocess.TimeoutExpired:
                pass
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5.0)
            except Exception:
                pass
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
            self._stderr_file = None
        self._process = None
        self._process_reset_applied = False
        self._episode_binding = None

    @property
    def live_episode_binding(self) -> JsonDict | None:
        return (
            dict(self._episode_binding) if self._episode_binding is not None else None
        )

    def capture_live_official(self, request: JsonDict) -> JsonDict:
        """Consume the one live capture opportunity on the original worker."""

        if (
            self._process is None
            or self._process.poll() is not None
            or not self._process_reset_applied
        ):
            raise RuntimeError(
                "live official capture requires the original open worker"
            )
        if self._live_capture_consumed:
            raise RuntimeError("live official capture already consumed")
        binding = self._episode_binding
        if not isinstance(binding, dict):
            raise RuntimeError("live official episode binding is unavailable")
        expected_nonce = request.get(
            "expected_episode_instance_nonce", binding.get("episode_instance_nonce")
        )
        if expected_nonce != binding.get("episode_instance_nonce"):
            raise RuntimeError("live official episode instance mismatch")
        expected_runtime = request.get("expected_native_runtime")
        if expected_runtime is not None and expected_runtime != binding.get(
            "native_runtime"
        ):
            raise RuntimeError("live official native runtime identity mismatch")
        bridge_runtime = _process_identity()
        expected_bridge_runtime = request.get("expected_runtime_identity")
        if (
            expected_bridge_runtime is not None
            and expected_bridge_runtime != bridge_runtime
        ):
            raise RuntimeError("live official bridge runtime identity mismatch")
        self._live_capture_consumed = True
        result = self._request(
            "capture_live_official",
            {
                "operation_id": self.case_id,
                "identity": dict(request.get("identity") or {}),
                "environment_digest": request.get("environment_digest"),
                "transcript_digest": request.get("transcript_digest"),
                "expected_episode_instance_nonce": expected_nonce,
                "expected_native_runtime": binding.get("native_runtime"),
            },
        )
        request_binding = {
            "identity": dict(request.get("identity") or {}),
            "environment_digest": request.get("environment_digest"),
            "transcript_digest": request.get("transcript_digest"),
            "expected_episode_instance_nonce": expected_nonce,
            "expected_native_runtime": binding.get("native_runtime"),
            "expected_runtime_identity": expected_bridge_runtime,
        }
        return {
            **dict(result),
            "bridge_schema_version": LIVE_OFFICIAL_BRIDGE_SCHEMA,
            "bridge_runtime": bridge_runtime,
            "request_binding": request_binding,
            "request_binding_digest": _json_digest(request_binding),
        }

    def _request(self, method: str, args: JsonDict) -> Any:
        with self._request_lock:
            request_id = str(uuid.uuid4())
            self._send({"id": request_id, "method": method, "args": args})
            return self._read_response(request_id, method)

    def _start_live_official_socket_if_configured(self) -> None:
        path_value = self.env.get(LIVE_OFFICIAL_SOCKET_ENV)
        auth_digest = self.env.get(LIVE_OFFICIAL_AUTH_SHA256_ENV)
        if path_value is None and auth_digest is None:
            return
        if not path_value or not auth_digest:
            raise RuntimeError(
                "live official socket path and authorization digest must be configured together"
            )
        path = Path(path_value)
        if not path.is_absolute():
            raise RuntimeError("live official socket path must be absolute")
        if len(os.fsencode(path)) > 107:
            raise RuntimeError("live official socket path exceeds the AF_UNIX limit")
        if len(auth_digest) != 64 or any(
            char not in "0123456789abcdef" for char in auth_digest
        ):
            raise RuntimeError("live official authorization digest is invalid")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise RuntimeError("live official socket path already exists")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(path))
        os.chmod(path, 0o600)
        listener.listen(1)
        listener.settimeout(0.2)
        self._live_socket_stop.clear()
        self._live_socket = listener
        self._live_socket_thread = threading.Thread(
            target=self._serve_live_official_socket,
            args=(path, auth_digest),
            name=f"live-official-{self.case_id}",
            daemon=True,
        )
        self._live_socket_thread.start()

    def _serve_live_official_socket(self, path: Path, auth_digest: str) -> None:
        listener = self._live_socket
        if listener is None:
            return
        try:
            while (
                not self._live_socket_stop.is_set() and not self._live_capture_consumed
            ):
                try:
                    connection, _ = listener.accept()
                except TimeoutError:
                    continue
                except OSError:
                    if self._live_socket_stop.is_set():
                        break
                    raise
                with connection:
                    try:
                        raw = b""
                        while b"\n" not in raw and len(raw) <= 1024 * 1024:
                            chunk = connection.recv(65536)
                            if not chunk:
                                break
                            raw += chunk
                        request = json.loads(raw.split(b"\n", 1)[0])
                        nonce = request.pop("authorization_nonce", None)
                        if not isinstance(nonce, str) or not hmac.compare_digest(
                            hashlib.sha256(nonce.encode("utf-8")).hexdigest(),
                            auth_digest,
                        ):
                            response = {"ok": False, "error": "authorization rejected"}
                        else:
                            response = {
                                "ok": True,
                                "result": self.capture_live_official(request),
                            }
                    except Exception as exc:  # noqa: BLE001 - private protocol must fail closed.
                        response = {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    connection.sendall(
                        json.dumps(
                            response, sort_keys=True, separators=(",", ":")
                        ).encode("utf-8")
                        + b"\n"
                    )
        finally:
            try:
                listener.close()
            except OSError:
                pass
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def _stop_live_official_socket(self) -> None:
        self._live_socket_stop.set()
        listener = self._live_socket
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread = self._live_socket_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1)
        self._live_socket = None
        self._live_socket_thread = None

    def _read_response(self, request_id: str, method: str) -> Any:
        while True:
            line = self._readline()
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                self._record_noise(line)
                continue
            if response.get("id") != request_id:
                self._record_noise(line)
                continue
            if response.get("ok") is not True:
                error = response.get("error") or {}
                error_type = error.get("type") if isinstance(error, dict) else ""
                message = (
                    error.get("message") if isinstance(error, dict) else str(error)
                )
                stderr_tail = self._stderr_tail()
                diagnostic = f"; stderr_tail={stderr_tail}" if stderr_tail else ""
                if error_type == "TypeError":
                    raise TypeError(
                        f"Native backend bridge {method} failed: {message}{diagnostic}"
                    )
                raise RuntimeError(
                    f"Native backend bridge {method} failed: {message}{diagnostic}"
                )
            return response.get("result")

    def _timeout_for_call_primitive(self, name: str) -> float | None:
        if name not in set(self.policy_timeout_primitives):
            return None
        if self.policy_timeout_seconds is None:
            return None
        try:
            timeout_seconds = float(self.policy_timeout_seconds)
        except (TypeError, ValueError):
            return None
        return timeout_seconds if timeout_seconds > 0 else None

    def _request_context(
        self, *, request_label: str, timeout_seconds: float | None
    ) -> "_BridgeRequestContext":
        return _BridgeRequestContext(
            self, request_label=request_label, timeout_seconds=timeout_seconds
        )

    def _send(self, payload: JsonDict) -> None:
        process = self._ensure_process()
        self._replay_reset_if_needed(process, payload)
        self._send_raw(process, payload)

    def _send_raw(self, process: subprocess.Popen[str], payload: JsonDict) -> None:
        if process.stdin is None:
            raise RuntimeError("Native backend bridge stdin is unavailable.")
        process.stdin.write(json.dumps(payload, sort_keys=True) + "\n")
        process.stdin.flush()

    def _replay_reset_if_needed(
        self, process: subprocess.Popen[str], payload: JsonDict
    ) -> None:
        if payload.get("method") == "reset":
            return
        if self._process_reset_applied or self._last_reset_args is None:
            return
        request_id = str(uuid.uuid4())
        reset_payload = {
            "id": request_id,
            "method": "reset",
            "args": dict(self._last_reset_args),
        }
        self._send_raw(process, reset_payload)
        with self._request_context(request_label="reset_replay", timeout_seconds=None):
            self._read_response(request_id, "reset_replay")
        self._process_reset_applied = True
        if self._trace_cache is None:
            self._trace_cache = EpisodeTrace(
                task_id=str(self._last_reset_args.get("task_id") or "")
            )

    def _readline(self) -> str:
        process = self._ensure_process()
        if process.stdout is None:
            raise RuntimeError("Native backend bridge stdout is unavailable.")
        selector = selectors.DefaultSelector()
        try:
            selector.register(process.stdout, selectors.EVENT_READ)
            effective_timeout_seconds = (
                self._active_timeout_seconds or self.timeout_seconds
            )
            ready = selector.select(effective_timeout_seconds)
        finally:
            selector.close()
        if not ready:
            stderr_tail = self._stderr_tail()
            try:
                self._terminate_unusable_process(process)
            except Exception:
                pass
            request_label = self._active_request_label or "native_backend_request"
            raise RuntimeError(
                "Native backend bridge timed out before response; "
                f"request={request_label}; timeout_seconds={effective_timeout_seconds}; stderr_tail={stderr_tail}"
            )
        line = process.stdout.readline()
        if not line:
            returncode = process.poll()
            if returncode is None:
                try:
                    returncode = process.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    pass
            stderr_tail = self._stderr_tail()
            self._mark_process_unusable(process)
            raise RuntimeError(
                "Native backend bridge exited before response; "
                f"returncode={returncode}; stderr_tail={stderr_tail}"
            )
        return line.strip()

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process
        if self._process is not None:
            self._mark_process_unusable(self._process)
        if self.strict_episode and self.episode_lost:
            raise RuntimeError("native_episode_lost: backend process exited; implicit episode reset is disabled")
        worker = str(Path(self.worker_script))
        command = [self.python_executable, worker, "--case-id", self.case_id]
        child_env = dict(os.environ) if self.inherit_environment else {}
        child_env.update(self.env)
        if child_env.get(NATIVE_XVFB_ENV) in {"1", "true", "TRUE", "yes", "on"}:
            xvfb_script = child_env.get(NATIVE_XVFB_SCRIPT_ENV) or str(
                Path(self.cwd) / "scripts" / "run_with_xvfb.sh"
            )
            command = ["bash", xvfb_script, *command]
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except Exception:
                pass
        recording_dir = os.environ.get("ARENA_CASE_STUDY_DIR")
        explicit_log_dir = os.environ.get("ARENA_NATIVE_LOG_DIR")
        if recording_dir or explicit_log_dir:
            log_dir = Path(explicit_log_dir) if explicit_log_dir else Path(recording_dir).parent / "native_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self._stderr_file = tempfile.NamedTemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace", delete=False,
                dir=log_dir, prefix="native-", suffix=".stderr.log"
            )
        else:
            self._stderr_file = tempfile.TemporaryFile(
                mode="w+t", encoding="utf-8", errors="replace"
            )
        self._process = subprocess.Popen(
            command,
            cwd=str(self.cwd),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            text=True,
            bufsize=1,
        )
        return self._process

    def _terminate_unusable_process(self, process: subprocess.Popen[str]) -> None:
        try:
            process.terminate()
        except Exception:
            pass
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=1)
            except Exception:
                kill = getattr(process, "kill", None)
                if callable(kill):
                    try:
                        kill()
                    except Exception:
                        pass
        self._mark_process_unusable(process)

    def _mark_process_unusable(self, process: subprocess.Popen[str]) -> None:
        if self._last_reset_args is not None:
            self.episode_lost = True
        self._stop_live_official_socket()
        if self._process is process:
            self._process = None
        self._process_reset_applied = False
        self._episode_binding = None

    def _stderr_tail(self) -> str:
        stderr_file = self._stderr_file
        if stderr_file is None:
            return ""
        try:
            stderr_file.flush()
            position = stderr_file.tell()
            stderr_file.seek(0)
            text = stderr_file.read()
            stderr_file.seek(position)
        except Exception:
            return ""
        if len(text) <= 16000:
            return text
        return text[:2000] + "\n...[native stderr truncated]...\n" + text[-14000:]

    def _record_noise(self, line: str) -> None:
        if not line:
            return
        self._noise_lines.append(line[:800])
        del self._noise_lines[:-20]

    def _merge_remote_trace_metadata(self, artifact_ids: list[str] | None = None) -> None:
        if self._trace_cache is None:
            return
        try:
            remote_trace = _trace_from_dict(self._request("get_trace", {
                "metadata_only": True, "artifact_ids": artifact_ids,
            }))
        except (
            Exception
        ) as exc:  # pragma: no cover - trace enrichment must not break live tool calls.
            self._record_noise(
                f"trace_metadata_merge_failed:{type(exc).__name__}:{exc}"
            )
            return
        self._trace_cache.artifacts.update(remote_trace.artifacts)
        self._trace_cache.metrics.update(remote_trace.metrics)
        if remote_trace.final_status is not None:
            self._trace_cache.final_status = remote_trace.final_status

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@dataclass(slots=True)
class _BridgeRequestContext:
    bridge: SubprocessBackendBridge
    request_label: str
    timeout_seconds: float | None
    _previous_timeout_seconds: float | None = field(
        default=None, init=False, repr=False
    )
    _previous_request_label: str = field(default="", init=False, repr=False)

    def __enter__(self) -> None:
        self._previous_timeout_seconds = self.bridge._active_timeout_seconds
        self._previous_request_label = self.bridge._active_request_label
        self.bridge._active_timeout_seconds = self.timeout_seconds
        self.bridge._active_request_label = self.request_label

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.bridge._active_timeout_seconds = self._previous_timeout_seconds
        self.bridge._active_request_label = self._previous_request_label
