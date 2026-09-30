#!/usr/bin/env python3
"""The single, persistent coding runtime used by every RoBench adapter.

The agent authors one ``cell.py`` at a time.  Cells execute in one Python
process, against one benchmark server/episode, and OpenHands turns resume one
conversation instead of starting a new headless session on every turn.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import warnings
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

import openhands_bridge as oh_bridge


RUNTIME_MODE = "persistent_repl_code"
RUNTIME_MODES = (RUNTIME_MODE,)
_RESULT_PREFIX = "__ROBENCH_CELL_RESULT__"
_READY_PREFIX = "__ROBENCH_REPL_READY__"
_CONVERSATION_ID_RE = re.compile(r"Conversation ID:\s*([A-Za-z0-9_-]+)")
_SAFE_ENV_NAMES = {
    "PATH",
    "LANG",
    "LANGUAGE",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "VIRTUAL_ENV",
    "CONDA_PREFIX",
    "PYTHONHOME",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "ROBENCH_PRIMITIVE_SERVER_URL",
    "ROBENCH_PRIMITIVE_SERVER_TOKEN",
    "ROBENCH_PRIMITIVE_SERVER_SOCKET",
}


def add_runtime_args(parser: argparse.ArgumentParser, *, default_max_code_turns: int = 4) -> None:
    if "--max-primitive-calls" not in parser._option_string_actions:
        parser.add_argument("--max-primitive-calls", type=int, default=None,
                            help="Total agent primitive RPC budget across code cells.")
    parser.add_argument(
        "--runtime-mode",
        choices=RUNTIME_MODES,
        default=RUNTIME_MODE,
        help=(
            "Persistent code-cell runtime: one resumable coding-agent conversation, "
            "one Python interpreter, and one benchmark episode."
        ),
    )
    parser.add_argument(
        "--max-code-turns",
        type=int,
        default=default_max_code_turns,
        help="Maximum authored code cells (probe always executes one cell).",
    )
    parser.add_argument(
        "--trial-timeout-seconds",
        type=int,
        default=0,
        help="Global wall-clock deadline. Zero derives a bounded deadline from the per-phase timeout.",
    )
    parser.add_argument(
        "--max-cell-bytes",
        type=int,
        default=256_000,
        help="Maximum size of one authored cell.py.",
    )
    parser.add_argument(
        "--max-cell-output-chars",
        type=int,
        default=20_000,
        help="Maximum combined stdout/stderr retained from one cell.",
    )
    parser.add_argument(
        "--repl-sandbox",
        choices=("auto", "required", "off"),
        default="auto",
        help="Filesystem/process sandbox for generated cells. auto uses bubblewrap when available.",
    )
    parser.add_argument(
        "--author-sandbox",
        choices=("auto", "required", "off"),
        default="auto",
        help="Filesystem sandbox for the OpenHands author process; network remains available for model calls.",
    )
    parser.add_argument("--repl-memory-mb", type=int, default=2048)
    parser.add_argument("--repl-max-open-files", type=int, default=256)
    parser.add_argument("--repl-max-file-mb", type=int, default=64)


def create_workspace(workspace_dir: Path, *, allowed_root: Path) -> Path:
    """Create a fresh workspace without ever deleting or reusing user data."""
    root = allowed_root.expanduser().resolve()
    target = workspace_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if target == root or root not in target.parents:
        raise ValueError(f"Workspace must be a strict child of {root}: {target}")
    if target.exists():
        raise FileExistsError(f"Refusing to reuse or delete existing workspace: {target}")
    target.mkdir(parents=True, exist_ok=False)
    return target


def build_cell_env(source_env: dict[str, str], *, runtime_dir: Path) -> dict[str, str]:
    """Build the secret-minimized environment visible to generated code."""
    env = {name: value for name, value in source_env.items() if name in _SAFE_ENV_NAMES and value}
    env.setdefault("PATH", os.defpath)
    env.setdefault("LANG", "C.UTF-8")
    private_home = runtime_dir / "home"
    private_tmp = runtime_dir / "tmp"
    private_home.mkdir(parents=True, exist_ok=True)
    private_tmp.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(private_home),
            "TMPDIR": str(private_tmp),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "ROBENCH_GENERATED_CODE": "1",
        }
    )
    return env


def build_author_env(
    source_env: dict[str, str],
    *,
    conversation_dir: Path,
    workspace_dir: Path,
    author_home: Path,
) -> dict[str, str]:
    """OpenHands gets its model config, but never the primitive capability token."""
    env = dict(source_env)
    env.pop("ROBENCH_PRIMITIVE_SERVER_URL", None)
    env.pop("ROBENCH_PRIMITIVE_SERVER_TOKEN", None)
    env["ROBENCH_AUTHOR_ONLY"] = "1"
    env["OPENHANDS_CONVERSATIONS_DIR"] = str(conversation_dir)
    env["OPENHANDS_WORK_DIR"] = str(workspace_dir)
    env["HOME"] = str(author_home)
    env["PERSISTENCE_DIR"] = str(author_home / ".openhands")
    env["TMPDIR"] = "/tmp"
    return env


class _LineReader:
    def __init__(self, pipe: Any):
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread = threading.Thread(target=self._read, args=(pipe,), daemon=True)
        self._thread.start()

    def _read(self, pipe: Any) -> None:
        try:
            for line in iter(pipe.readline, ""):
                self._queue.put(line)
        finally:
            self._queue.put(None)

    def get(self, timeout: float) -> str | None:
        return self._queue.get(timeout=max(0.01, timeout))


class PrimitiveSocketProxy:
    """Expose the loopback HTTP server through AF_UNIX for a networkless cell sandbox."""

    def __init__(self, server_url: str, socket_path: Path, *, upstream_token: str,
                 max_calls: int | None = None):
        match = re.fullmatch(r"http://(127\.0\.0\.1|localhost):(\d+)", server_url.rstrip("/"))
        if not match:
            raise ValueError(f"Primitive server must be loopback HTTP, got {server_url!r}")
        self.target = ("127.0.0.1", int(match.group(2)))
        self.socket_path = socket_path.resolve()
        self.upstream_token = upstream_token
        self.active_token = ""
        self.max_calls = max_calls
        self.call_count = 0
        self.budget_exceeded = False
        self._budget_lock = threading.Lock()
        if self.socket_path.exists():
            raise FileExistsError(f"Refusing to replace primitive socket: {self.socket_path}")
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self.listener.listen(16)
        self.listener.settimeout(0.5)
        self._closed = threading.Event()
        self._enabled = threading.Event()
        self._thread = threading.Thread(target=self._accept, daemon=True)
        self._thread.start()

    def _accept(self) -> None:
        while not self._closed.is_set():
            try:
                client, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            if not self._enabled.is_set():
                client.close()
                continue
            threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket) -> None:
        with client:
            client.settimeout(10)
            try:
                request = self._read_http_request(client)
            except (OSError, ValueError):
                return
            expected = f"Authorization: Bearer {self.active_token}".encode("utf-8")
            if not self._enabled.is_set() or expected not in request:
                client.sendall(b"HTTP/1.0 401 Unauthorized\r\nContent-Length: 0\r\n\r\n")
                return
            with self._budget_lock:
                if self.max_calls is not None and self.max_calls >= 0 and self.call_count >= self.max_calls:
                    self.budget_exceeded = True
                    body = b'{"ok":false,"error":"max_primitive_calls budget exhausted"}'
                    client.sendall(b"HTTP/1.0 429 Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: "
                                   + str(len(body)).encode() + b"\r\n\r\n" + body)
                    return
                self.call_count += 1
            upstream_auth = f"Authorization: Bearer {self.upstream_token}".encode("utf-8")
            request = request.replace(expected, upstream_auth, 1)
            try:
                upstream = socket.create_connection(self.target, timeout=10)
            except OSError:
                return
            with upstream:
                # Public primitive clients allow 120s; simulator work can exceed the 10s connect timeout.
                upstream.settimeout(120)
                upstream.sendall(request)
                while self._enabled.is_set():
                    try:
                        chunk = upstream.recv(65536)
                    except OSError:
                        return
                    if not chunk:
                        return
                    client.sendall(chunk)

    @staticmethod
    def _read_http_request(client: socket.socket) -> bytes:
        request = b""
        while b"\r\n\r\n" not in request:
            request += client.recv(65536)
            if not request or len(request) > 2_000_000:
                raise ValueError("invalid primitive request")
        header, _, body = request.partition(b"\r\n\r\n")
        content_length = 0
        for line in header.split(b"\r\n"):
            if line.lower().startswith(b"content-length:"):
                content_length = int(line.split(b":", 1)[1].strip())
                break
        while len(body) < content_length:
            body += client.recv(min(65536, content_length - len(body)))
        return header + b"\r\n\r\n" + body

    def enable(self, token: str) -> None:
        self.active_token = token
        self._enabled.set()

    def disable(self) -> None:
        self._enabled.clear()
        self.active_token = ""

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self.listener.close()
        self._thread.join(timeout=2)
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass


class PersistentPythonRepl:
    """A private JSON-line driver that executes cells in one global namespace."""

    def __init__(
        self,
        *,
        workspace_dir: Path,
        runtime_dir: Path,
        env: dict[str, str],
        sandbox_mode: str,
        memory_mb: int,
        max_open_files: int,
        max_file_mb: int,
        max_output_chars: int,
        startup_timeout: int,
        transport_socket: Path | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.runtime_dir = runtime_dir.resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=False)
        self.driver_path = self.runtime_dir / "solve.py"
        self.driver_path.write_text(_driver_source(), encoding="utf-8")
        self.sandbox_backend = _resolve_sandbox(sandbox_mode)
        process_env = dict(env)
        process_env.pop("ROBENCH_PRIMITIVE_SERVER_TOKEN", None)
        if self.sandbox_backend != "none":
            process_env["HOME"] = str(self.workspace_dir)
            process_env["TMPDIR"] = "/tmp"
        if transport_socket is not None:
            process_env["ROBENCH_PRIMITIVE_SERVER_SOCKET"] = (
                ".robench_primitive.sock" if self.sandbox_backend != "none" else str(transport_socket)
            )
        command = _build_repl_command(
            driver_path=self.driver_path,
            workspace_dir=self.workspace_dir,
            runtime_dir=self.runtime_dir,
            sandbox_backend=self.sandbox_backend,
            memory_mb=memory_mb,
            max_open_files=max_open_files,
            max_file_mb=max_file_mb,
            transport_socket=transport_socket,
        )
        self.command = command
        self.process = subprocess.Popen(
            command,
            cwd=self.workspace_dir,
            env=process_env,
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
            start_new_session=True,
        )
        assert self.process.stdout is not None
        assert self.process.stderr is not None
        self._stdout = _LineReader(self.process.stdout)
        self._stderr_tail = oh_bridge.TailBuffer(max_output_chars)
        self._stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self._stderr_thread.start()
        try:
            line = self._next_protocol_line(_READY_PREFIX, startup_timeout)
            ready = json.loads(line[len(_READY_PREFIX) :])
        except BaseException:
            _terminate_process_group(self.process)
            self._close_pipes()
            raise
        self.worker_pid = int(ready["pid"])
        self.max_output_chars = max_output_chars

    def _drain_stderr(self) -> None:
        assert self.process.stderr is not None
        for line in iter(self.process.stderr.readline, ""):
            self._stderr_tail.append(line)

    def _next_protocol_line(self, prefix: str, timeout_seconds: float) -> str:
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Persistent interpreter did not emit {prefix!r} in time")
            try:
                line = self._stdout.get(remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"Persistent interpreter did not emit {prefix!r} in time") from exc
            if line is None:
                raise RuntimeError(
                    "Persistent interpreter exited unexpectedly: " + self._stderr_tail.text()[-4000:]
                )
            if line.startswith(prefix):
                return line.rstrip("\n")

    def execute(
        self,
        cell_path: Path,
        *,
        timeout_seconds: int,
        primitive_token: str = "",
    ) -> tuple[subprocess.CompletedProcess[str], bool]:
        if self.process.poll() is not None:
            return subprocess.CompletedProcess(self.command, 1, "", "Persistent interpreter is not running."), False
        request = {
            "cell_path": str(cell_path.resolve()),
            "max_output_chars": self.max_output_chars,
            "primitive_token": primitive_token,
        }
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        try:
            line = self._next_protocol_line(_RESULT_PREFIX, timeout_seconds)
        except TimeoutError:
            self.close(force=True)
            return subprocess.CompletedProcess(
                self.command,
                124,
                "",
                f"Cell timed out after {timeout_seconds} seconds.",
            ), True
        except RuntimeError as exc:
            return subprocess.CompletedProcess(self.command, 1, "", str(exc)), False
        payload = json.loads(line[len(_RESULT_PREFIX) :])
        _terminate_descendants(self.worker_pid)
        stderr = str(payload.get("stderr", ""))
        driver_stderr = self._stderr_tail.text()
        if driver_stderr:
            stderr += "\n[repl driver stderr]\n" + driver_stderr[-4000:]
        return subprocess.CompletedProcess(
            self.command,
            int(payload.get("returncode", 1)),
            str(payload.get("stdout", "")),
            stderr,
        ), False

    def close(self, *, force: bool = False) -> None:
        if self.process.poll() is not None:
            self._close_pipes()
            return
        if not force:
            try:
                assert self.process.stdin is not None
                self.process.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
                self._close_pipes()
                return
            except (BrokenPipeError, subprocess.TimeoutExpired):
                pass
        _terminate_process_group(self.process)
        self._close_pipes()

    def _close_pipes(self) -> None:
        for pipe in (self.process.stdin, self.process.stdout, self.process.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()

    def __enter__(self) -> "PersistentPythonRepl":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


def run_persistent_repl_code(
    *,
    args: argparse.Namespace,
    session: Any,
    workspace_dir: Path,
    cell_path: Path,
    openhands_binary: str | None,
    env: dict[str, str],
    outputs_dir: Path,
    benchmark_label: str,
    feedback_writer: Callable[[int, int, subprocess.CompletedProcess[str], bool], dict[str, Any]],
) -> tuple[subprocess.CompletedProcess[str], bool, dict[str, Any], str, str, int]:
    if args.runtime_mode != RUNTIME_MODE:
        raise ValueError(f"Only --runtime-mode {RUNTIME_MODE} is supported")
    if args.executor == "openhands-headless" and openhands_binary is None:
        raise RuntimeError("OpenHands CLI executable was not resolved.")

    max_turns = 1 if args.executor == "probe" else max(1, int(args.max_code_turns))
    phase_timeout = max(1, int(args.code_timeout_seconds))
    global_timeout = int(args.trial_timeout_seconds or 0)
    if global_timeout <= 0:
        global_timeout = max_turns * phase_timeout * (2 if args.executor != "probe" else 1) + 30
    from codex_cell_author import _native_imports
    _native_imports()
    from embodied_harness.episode_loop import EpisodeLoop, is_timeout
    loop = EpisodeLoop(max_turns, global_timeout, replay=args.executor == "probe")

    runtime_root = outputs_dir / "repl_runtime" / session.trace.run_id
    runtime_root.mkdir(parents=True, exist_ok=False)
    conversation_dir = runtime_root / "openhands_conversations"
    conversation_dir.mkdir(parents=True, exist_ok=False)
    author_home = runtime_root / "author_home"
    author_home.mkdir(parents=True, exist_ok=False)
    cells_dir = workspace_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    repl_dir = runtime_root / "python_repl"

    cell_env = build_cell_env(env, runtime_dir=runtime_root)
    cell_secret_values = oh_bridge.collect_secret_values(
        {"ROBENCH_PRIMITIVE_SERVER_TOKEN": env.get("ROBENCH_PRIMITIVE_SERVER_TOKEN", "")}
    )
    author_env = build_author_env(
        env,
        conversation_dir=conversation_dir,
        workspace_dir=workspace_dir,
        author_home=author_home,
    )
    combined_stdout: list[str] = []
    combined_stderr: list[str] = []
    turn_logs: list[dict[str, Any]] = []
    last_completed = subprocess.CompletedProcess([], 1, "", "No cell was executed.")
    any_timed_out = False
    turns_started = 0
    conversation_id: str | None = None
    codex_author = None
    if args.executor in {"codex-exec", "cursor-exec", "openai-compatible"}:
        from codex_cell_author import CodexCellAuthor
        codex_author = CodexCellAuthor(args, workspace_dir)
        conversation_id = session.trace.run_id
        # The native Codex/Cursor client runs in its own read-only, empty directory
        # and rejects all tool events. It never receives the Pukun workspace.
        if args.executor == "openai-compatible":
            author_sandbox_backend = "remote-api-completion"
        elif args.executor == "cursor-exec":
            author_sandbox_backend = "cursor-read-only-completion"
        else:
            author_sandbox_backend = "codex-read-only-completion"
    else:
        author_sandbox_backend = _resolve_sandbox(args.author_sandbox, option_name="author-sandbox")

    primitive_url = cell_env.get("ROBENCH_PRIMITIVE_SERVER_URL")
    if not primitive_url:
        raise RuntimeError("ROBENCH_PRIMITIVE_SERVER_URL is required")
    primitive_upstream_token = env.get("ROBENCH_PRIMITIVE_SERVER_TOKEN", "")
    if not primitive_upstream_token:
        raise RuntimeError("ROBENCH_PRIMITIVE_SERVER_TOKEN is required")
    primitive_socket_path = Path("/tmp") / f"robench-{uuid.uuid4().hex}.sock"
    primitive_proxy = PrimitiveSocketProxy(
        primitive_url,
        primitive_socket_path,
        upstream_token=primitive_upstream_token,
        max_calls=getattr(args, "max_primitive_calls", None),
    )

    try:
        repl = PersistentPythonRepl(
            workspace_dir=workspace_dir,
            runtime_dir=repl_dir,
            env=cell_env,
            sandbox_mode=args.repl_sandbox,
            memory_mb=max(128, int(args.repl_memory_mb)),
            max_open_files=max(32, int(args.repl_max_open_files)),
            max_file_mb=max(1, int(args.repl_max_file_mb)),
            max_output_chars=max(1000, int(args.max_cell_output_chars)),
            startup_timeout=loop.phase_timeout(phase_timeout),
            transport_socket=primitive_socket_path,
        )
    except BaseException:
        primitive_proxy.close()
        raise
    try:
        for turn_index in loop:
            turns_started = turn_index
            turn_record: dict[str, Any] = {"turn": turn_index, "worker_pid": repl.worker_pid}
            if codex_author is not None:
                try:
                    code = codex_author.next_cell(timeout_seconds=loop.phase_timeout(phase_timeout), loop=loop)
                    cell_path.write_text(code, encoding="utf-8")
                    turn_record["model"] = codex_author.report()
                except Exception as exc:
                    last_completed = subprocess.CompletedProcess([], 1, "", f"{type(exc).__name__}: {exc}")
                    combined_stderr.append(last_completed.stderr)
                    turn_record.update(author_error=last_completed.stderr, model=codex_author.report())
                    any_timed_out = is_timeout(exc)
                    loop.finish_turn(stage="model_completion", error=last_completed.stderr,
                                     timed_out=any_timed_out,
                                     budget_exhausted=codex_author.report()["usage"]["budget_exhausted"])
                    turn_logs.append(turn_record)
                    break
            if args.executor == "openhands-headless":
                prompt_file = "task.md" if turn_index == 1 else write_continuation_prompt(
                    workspace_dir,
                    turn_index,
                    benchmark_label=benchmark_label,
                )
                author_command = build_openhands_prompt_command(
                    args=args,
                    openhands_binary=str(openhands_binary),
                    prompt_file=prompt_file,
                    conversation_id=conversation_id,
                )
                author_command = _wrap_author_command(
                    command=author_command,
                    openhands_binary=Path(str(openhands_binary)),
                    workspace_dir=workspace_dir,
                    conversation_dir=conversation_dir,
                    author_home=author_home,
                    sandbox_backend=author_sandbox_backend,
                )
                author_source = f"openhands_persistent_conversation_turn_{turn_index}"
                author_timeout = loop.phase_timeout(phase_timeout)
                session.record_execution_started(
                    command=author_command,
                    source=author_source,
                    solve_path=cell_path,
                    timeout_seconds=author_timeout,
                )
                author_paths = oh_bridge.build_execution_log_paths(
                    outputs_dir=outputs_dir,
                    run_id=session.trace.run_id,
                    executor_source=author_source,
                )
                author_completed, author_timed_out = oh_bridge.run_executor(
                    command=author_command,
                    workspace_dir=workspace_dir,
                    env=author_env,
                    timeout_seconds=author_timeout,
                    event_log_paths=author_paths,
                )
                author_paths = oh_bridge.persist_execution_logs(
                    outputs_dir=outputs_dir,
                    run_id=session.trace.run_id,
                    completed=author_completed,
                    executor_source=author_source,
                )
                combined_stdout.append(f"\n[turn {turn_index} author stdout]\n{author_completed.stdout or ''}")
                combined_stderr.append(f"\n[turn {turn_index} author stderr]\n{author_completed.stderr or ''}")
                turn_record.update(
                    {
                        "prompt_file": prompt_file,
                        "author_stdout": author_paths.get("stdout", ""),
                        "author_stderr": author_paths.get("stderr", ""),
                        "author_returncode": author_completed.returncode,
                        "author_timed_out": author_timed_out,
                    }
                )
                any_timed_out = any_timed_out or author_timed_out
                if author_completed.returncode != 0 or author_timed_out:
                    last_completed = author_completed
                    turn_logs.append(turn_record)
                    break
                author_output = (author_completed.stdout or "") + "\n" + (author_completed.stderr or "")
                try:
                    author_output += "\n" + Path(author_paths["stdout"]).read_text(
                        encoding="utf-8", errors="replace"
                    )
                except (KeyError, OSError):
                    pass
                discovered = _discover_conversation_id(conversation_dir, author_output)
                if conversation_id is None:
                    if discovered is None:
                        last_completed = subprocess.CompletedProcess(
                            author_command,
                            1,
                            author_completed.stdout,
                            (author_completed.stderr or "")
                            + "\nCould not resolve OpenHands conversation ID; refusing to start disconnected turns.",
                        )
                        turn_record["conversation_resume_error"] = True
                        turn_logs.append(turn_record)
                        break
                    conversation_id = discovered
                elif discovered is not None and discovered != conversation_id:
                    last_completed = subprocess.CompletedProcess(
                        author_command,
                        1,
                        author_completed.stdout,
                        f"OpenHands resumed a different conversation: {discovered} != {conversation_id}",
                    )
                    turn_record["conversation_resume_error"] = True
                    turn_logs.append(turn_record)
                    break

            if not cell_path.is_file():
                last_completed = subprocess.CompletedProcess([], 1, "", f"Missing authored cell: {cell_path.name}")
                turn_record["cell_missing"] = True
                turn_logs.append(turn_record)
                break
            cell_size = cell_path.stat().st_size
            if cell_size > int(args.max_cell_bytes):
                last_completed = subprocess.CompletedProcess(
                    [], 1, "", f"Authored cell is {cell_size} bytes; limit is {args.max_cell_bytes}."
                )
                turn_record["cell_too_large"] = cell_size
                turn_logs.append(turn_record)
                break

            archived_cell = cells_dir / f"turn_{turn_index:03d}.py"
            if archived_cell.exists():
                raise FileExistsError(f"Refusing to overwrite archived cell: {archived_cell}")
            shutil.copy2(cell_path, archived_cell)
            session.record_agent_code(archived_cell)
            cell_timeout = loop.phase_timeout(phase_timeout)
            final_source = f"persistent_python_repl_cell_{turn_index}"
            session.record_execution_started(
                command=[str(repl.driver_path), archived_cell.name],
                source=final_source,
                solve_path=archived_cell,
                timeout_seconds=cell_timeout,
            )
            cell_primitive_token = secrets.token_urlsafe(32)
            primitive_proxy.enable(cell_primitive_token)
            try:
                final_completed, final_timed_out = repl.execute(
                    archived_cell,
                    timeout_seconds=cell_timeout,
                    primitive_token=cell_primitive_token,
                )
            finally:
                primitive_proxy.disable()
            final_completed = oh_bridge.redact_completed_process(
                final_completed,
                cell_secret_values | {cell_primitive_token},
            )
            final_paths = oh_bridge.persist_execution_logs(
                outputs_dir=outputs_dir,
                run_id=session.trace.run_id,
                completed=final_completed,
                executor_source=final_source,
            )
            combined_stdout.append(f"\n[turn {turn_index} cell stdout]\n{final_completed.stdout or ''}")
            combined_stderr.append(f"\n[turn {turn_index} cell stderr]\n{final_completed.stderr or ''}")
            last_completed = final_completed
            any_timed_out = any_timed_out or final_timed_out
            feedback = feedback_writer(turn_index, max_turns, final_completed, final_timed_out)
            feedback["primitive_calls"] = primitive_proxy.call_count
            feedback["remaining_primitive_calls"] = (
                max(0, primitive_proxy.max_calls - primitive_proxy.call_count)
                if primitive_proxy.max_calls is not None and primitive_proxy.max_calls >= 0 else None)
            if codex_author is not None:
                codex_author.feedback(feedback)
            turn_record.update(
                {
                    "cell_file": str(archived_cell),
                    "cell_stdout": final_paths.get("stdout", ""),
                    "cell_stderr": final_paths.get("stderr", ""),
                    "cell_returncode": final_completed.returncode,
                    "cell_timed_out": final_timed_out,
                    "feedback_file": feedback.get("feedback_file", ""),
                    "feedback_json": feedback.get("feedback_json", ""),
                    "success": bool(feedback.get("success")),
                    "terminal": bool(feedback.get("terminal")),
                    "env_steps": feedback.get("env_steps"),
                    "score": feedback.get("score"),
                }
            )
            turn_logs.append(turn_record)
            loop.finish_turn(execution_ok=final_completed.returncode == 0,
                             success=bool(feedback.get("success")),
                             budget_exhausted=["max_primitive_calls"] if primitive_proxy.budget_exceeded else [],
                             terminal=bool(feedback.get("terminal")), timed_out=final_timed_out,
                             error=final_completed.stderr if final_completed.returncode else None)
            any_timed_out = any_timed_out or loop.stop_reason == "timeout"
            if loop.stop_reason:
                break
            if turn_index < max_turns:
                cell_path.write_text(_next_cell_stub(turn_index + 1), encoding="utf-8")
    except TimeoutError as exc:
        any_timed_out = True
        last_completed = subprocess.CompletedProcess([], 124, "", str(exc))
        loop.finish_turn(timed_out=True, error=str(exc))
    finally:
        repl.close()
        primitive_proxy.close()
        try:
            (workspace_dir / ".robench_primitive.sock").unlink()
        except FileNotFoundError:
            pass

    if loop.stop_reason is None:
        loop.finish_turn(stage="author", timed_out=any_timed_out,
                         error=last_completed.stderr or "Author failed to provide a runnable cell")
    event_log_paths: dict[str, Any] = {
        "episode_loop": loop.report(),
        "primitive_rpc_budget": {"max_calls": primitive_proxy.max_calls,
                                 "used_calls": primitive_proxy.call_count,
                                 "exceeded": primitive_proxy.budget_exceeded,
                                 "scope": "agent RPCs; native limits also apply; harness feedback excluded"},
        "turns": turn_logs,
        "runtime_mode": RUNTIME_MODE,
        "runtime_dir": str(runtime_root),
        "conversation_dir": str(conversation_dir),
        "conversation_id": conversation_id,
        "python_worker_pid": repl.worker_pid,
        "sandbox_backend": repl.sandbox_backend,
        "network_isolated": repl.sandbox_backend != "none",
        "author_sandbox_backend": author_sandbox_backend,
        "author_filesystem_isolated": codex_author is None and author_sandbox_backend != "none",
        "author_network_isolated": False,
        "global_timeout_seconds": global_timeout,
    }
    if codex_author is not None:
        event_log_paths["model"] = codex_author.report()
        (runtime_root / "model_transcript.json").write_text(
            json.dumps(codex_author.messages, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if turn_logs:
        event_log_paths.update(
            {
                "stdout": turn_logs[-1].get("author_stdout", ""),
                "stderr": turn_logs[-1].get("author_stderr", ""),
                "final_stdout": turn_logs[-1].get("cell_stdout", ""),
                "final_stderr": turn_logs[-1].get("cell_stderr", ""),
            }
        )
    combined = subprocess.CompletedProcess(
        last_completed.args,
        last_completed.returncode,
        "".join(combined_stdout) or (last_completed.stdout or ""),
        "".join(combined_stderr) or (last_completed.stderr or ""),
    )
    label = {"probe": "persistent_repl_probe", "codex-exec": "codex_persistent_repl", "cursor-exec": "cursor_persistent_repl", "openai-compatible": "api_persistent_repl"}.get(args.executor, "openhands_persistent_repl")
    return combined, any_timed_out, event_log_paths, label, RUNTIME_MODE, turns_started


def build_openhands_prompt_command(
    *,
    args: argparse.Namespace,
    openhands_binary: str,
    prompt_file: str,
    conversation_id: str | None,
) -> list[str]:
    command = [
        openhands_binary,
        "--headless",
        "--json",
        "--override-with-envs",
        "--exit-without-confirmation",
    ]
    if conversation_id:
        command.extend(["--resume", conversation_id])
    command.extend(["-f", prompt_file])
    command.extend(args.openhands_extra_arg or [])
    return command


def _wrap_author_command(
    *,
    command: list[str],
    openhands_binary: Path,
    workspace_dir: Path,
    conversation_dir: Path,
    author_home: Path,
    sandbox_backend: str,
) -> list[str]:
    if sandbox_backend == "none":
        return command
    resolved_binary = openhands_binary.resolve()
    wrapped = [
        sandbox_backend,
        "--die-with-parent",
        "--unshare-pid",
        "--unshare-ipc",
        "--unshare-uts",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    roots = _runtime_read_roots(Path(sys.executable).resolve())
    for extra in (
        Path("/etc/ssl"),
        Path("/etc/resolv.conf"),
        Path("/etc/hosts"),
        Path("/etc/nsswitch.conf"),
        Path("/etc/passwd"),
        Path("/etc/group"),
    ):
        if extra.exists():
            roots.append(extra)
    for path in _minimal_distinct_paths(roots):
        wrapped.extend(["--ro-bind", str(path), str(path)])
    if not any(resolved_binary == path or path in resolved_binary.parents for path in roots):
        wrapped.extend(["--ro-bind", str(resolved_binary), str(resolved_binary)])
    interpreter = _read_shebang_interpreter(resolved_binary)
    if interpreter is not None and not any(
        interpreter == path or path in interpreter.parents for path in roots
    ):
        interpreter_root = interpreter.parent.parent
        wrapped.extend(["--ro-bind", str(interpreter_root), str(interpreter_root)])
    wrapped.extend(["--bind", str(workspace_dir), str(workspace_dir)])
    wrapped.extend(["--bind", str(conversation_dir), str(conversation_dir)])
    wrapped.extend(["--bind", str(author_home), str(author_home)])
    wrapped.extend(["--chdir", str(workspace_dir), str(resolved_binary), *command[1:]])
    return wrapped


def write_continuation_prompt(workspace_dir: Path, turn_index: int, *, benchmark_label: str) -> str:
    prompt_path = workspace_dir / f"task_turn_{turn_index}.md"
    prompt_path.write_text(
        "\n".join(
            [
                f"# Continue The Same {benchmark_label} Episode",
                "",
                "This is the same coding-agent conversation, Python interpreter, and benchmark episode.",
                "Read `turn_feedback_latest.md` and earlier files in `cells/`.",
                "Write only the next natural code cell to `cell.py`; do not replay earlier cells.",
                "Imports, variables, functions, and client objects created by earlier cells still exist.",
                "Use only `primitive_api.py`; do not edit primitive cards or harness-private files.",
                "Keep the cell focused and print concise diagnostics or verifier output.",
                "",
                "The original instructions remain in `task.md`.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return prompt_path.name


def write_turn_feedback_files(
    *,
    session: Any,
    workspace_dir: Path,
    turn_index: int,
    max_turns: int,
    completed: subprocess.CompletedProcess[str],
    timed_out: bool,
    payload: dict[str, Any],
) -> dict[str, Any]:
    normalized = {
        "runtime_mode": getattr(session, "runtime_mode", RUNTIME_MODE),
        "turn": turn_index,
        "max_turns": max_turns,
        "returncode": completed.returncode,
        "timed_out": timed_out,
        "stdout_tail": (completed.stdout or "")[-6000:],
        "stderr_tail": (completed.stderr or "")[-6000:],
        **payload,
    }
    json_path = workspace_dir / f"turn_{turn_index}_feedback.json"
    json_path.write_text(json.dumps(_jsonable(normalized), indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = [
        "# Latest Turn Feedback",
        "",
        f"- runtime_mode: `{normalized['runtime_mode']}`",
        f"- turn: `{turn_index}/{max_turns}`",
        f"- returncode: `{completed.returncode}`",
        f"- timed_out: `{timed_out}`",
        f"- success: `{bool(normalized.get('success'))}`",
        f"- terminal: `{bool(normalized.get('terminal'))}`",
    ]
    for key in ("env_steps", "remaining_env_steps", "score"):
        if key in normalized:
            markdown.append(f"- {key}: `{normalized.get(key)}`")
    markdown.extend(["", "## Verifier", "", "```json"])
    markdown.append(json.dumps(_jsonable(normalized.get("verifier", {})), indent=2, ensure_ascii=False))
    markdown.extend(["```", ""])
    for title, key, fence in [
        ("Observation", "current_observation", "text"),
        ("Available Actions", "current_actions", "json"),
        ("Evidence", "evidence", "json"),
        ("Visual Evidence", "visual_evidence", "json"),
        ("Metrics", "metrics", "json"),
    ]:
        if key not in normalized:
            continue
        value = normalized.get(key)
        markdown.extend([f"## {title}", "", f"```{fence}"])
        markdown.append(
            json.dumps(_jsonable(value), indent=2, ensure_ascii=False)
            if fence == "json"
            else _truncate_text(value, 5000)
        )
        markdown.extend(["```", ""])
    markdown.extend(
        [
            "## Stdout Tail",
            "",
            "```text",
            (completed.stdout or "")[-4000:],
            "```",
            "",
            "## Stderr Tail",
            "",
            "```text",
            (completed.stderr or "")[-4000:],
            "```",
        ]
    )
    latest_path = workspace_dir / "turn_feedback_latest.md"
    latest_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    primitives = getattr(session, "primitives", None)
    if primitives is not None and hasattr(primitives, "record_harness_event"):
        primitives.record_harness_event(
            "code_turn_feedback_written",
            {
                "runtime_mode": normalized["runtime_mode"],
                "turn": turn_index,
                "feedback_file": latest_path.name,
                "feedback_json": json_path.name,
                "success": bool(normalized.get("success")),
                "terminal": bool(normalized.get("terminal")),
                "env_steps": normalized.get("env_steps"),
                "remaining_env_steps": normalized.get("remaining_env_steps"),
            },
            side_effect=False,
        )
    return {**normalized, "feedback_file": latest_path.name, "feedback_json": json_path.name}


def _resolve_sandbox(mode: str, *, option_name: str = "repl-sandbox") -> str:
    if mode == "off":
        return "none"
    binary = shutil.which("bwrap")
    if binary:
        probe = subprocess.run([binary, "--ro-bind", "/", "/", "--unshare-net", "--", "/bin/true"],
                               capture_output=True, timeout=5, check=False)
        if probe.returncode == 0:
            return binary
        if mode == "required":
            raise RuntimeError(f"--{option_name} required, but bubblewrap is not permitted: "
                               + probe.stderr.decode(errors="replace")[-400:])
        warnings.warn("bubblewrap is not permitted; auto uses the existing Python audit boundary, without OS network isolation", RuntimeWarning)
    if mode == "required":
        raise RuntimeError(f"--{option_name} required, but bubblewrap (bwrap) is unavailable")
    return "none"


def _build_repl_command(
    *,
    driver_path: Path,
    workspace_dir: Path,
    runtime_dir: Path,
    sandbox_backend: str,
    memory_mb: int,
    max_open_files: int,
    max_file_mb: int,
    transport_socket: Path | None,
) -> list[str]:
    # Keep the venv executable. resolve() follows pukun/bin/python to
    # /usr/bin/python3.10 and drops site-packages (numpy/libblas then break).
    python = Path(sys.executable)
    command = [
        shutil.which("prlimit") or "prlimit",
        f"--as={memory_mb * 1024 * 1024}",
        f"--nofile={max_open_files}",
        f"--fsize={max_file_mb * 1024 * 1024}",
        "--",
    ]
    if sandbox_backend == "none":
        return command + [str(python), "-u", str(driver_path)]
    bwrap = [
        sandbox_backend,
        "--die-with-parent",
        "--unshare-ipc",
        "--unshare-uts",
        "--unshare-net",
        "--dir",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
    ]
    for path in _runtime_read_roots(python):
        bwrap.extend(["--ro-bind", str(path), str(path)])
    bwrap.extend(["--bind", str(workspace_dir), str(workspace_dir)])
    bwrap.extend(["--bind", str(runtime_dir), str(runtime_dir)])
    if transport_socket is not None:
        bwrap.extend(
            [
                "--bind",
                str(transport_socket),
                str(workspace_dir / ".robench_primitive.sock"),
            ]
        )
    bwrap.extend(["--chdir", str(workspace_dir), str(python), "-u", str(driver_path)])
    return command + bwrap


def _symlink_chain(path: Path) -> list[Path]:
    hops: list[Path] = []
    current = Path(path)
    seen: set[str] = set()
    for _ in range(32):
        key = str(current)
        if key in seen:
            break
        seen.add(key)
        hops.append(current)
        try:
            if not current.is_symlink():
                break
            target = Path(os.readlink(current))
            if not target.is_absolute():
                target = current.parent / target
            current = target
        except OSError:
            break
    return hops


def _runtime_read_roots(python: Path) -> list[Path]:
    candidates = [Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc/ld.so.cache")]
    for value in (Path(sys.prefix), Path(sys.base_prefix), python.parent):
        candidates.append(value.resolve())
    # uv-managed venvs are often venv/bin/python -> cpython-X -> cpython-X.Y.Z.
    # resolve() keeps only the final prefix, so bwrap execvp of the venv path
    # gets ENOENT on the intermediate hop even though the file exists on disk.
    for hop in _symlink_chain(python):
        candidates.append(hop)
        candidates.append(hop.parent)
        if hop.parent.name == "bin":
            candidates.append(hop.parent.parent)
    return _minimal_distinct_paths(candidates)


def _minimal_distinct_paths(candidates: list[Path]) -> list[Path]:
    roots: list[Path] = []
    for candidate in candidates:
        if not candidate.exists() or any(candidate == item or item in candidate.parents for item in roots):
            continue
        roots = [item for item in roots if candidate not in item.parents]
        roots.append(candidate)
    return roots


def _read_shebang_interpreter(executable: Path) -> Path | None:
    try:
        with executable.open("rb") as source:
            first_line = source.readline(4096).decode("utf-8", errors="replace").strip()
    except OSError:
        return None
    if not first_line.startswith("#!"):
        return None
    first_word = first_line[2:].split(maxsplit=1)[0]
    if not first_word.startswith("/") or first_word == "/usr/bin/env":
        return None
    interpreter = Path(first_word).resolve()
    return interpreter if interpreter.exists() else None


def _discover_conversation_id(conversation_dir: Path, output: str) -> str | None:
    matches = _CONVERSATION_ID_RE.findall(output)
    if matches:
        return matches[-1]
    children = sorted(path.name for path in conversation_dir.iterdir() if path.is_dir())
    return children[0] if len(children) == 1 else None


def _remaining_seconds(deadline: float) -> int:
    remaining = int(deadline - time.monotonic())
    if remaining <= 0:
        raise TimeoutError("Global trial deadline exhausted")
    return remaining


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=2)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)


def _terminate_descendants(root_pid: int) -> None:
    """Remove subprocesses left behind by a completed cell while keeping its interpreter."""
    parent_by_pid: dict[int, int] = {}
    for stat_path in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = stat_path.read_text(encoding="utf-8")
            remainder = text[text.rfind(")") + 2 :].split()
            parent_by_pid[int(stat_path.parent.name)] = int(remainder[1])
        except (OSError, ValueError, IndexError):
            continue
    descendants: set[int] = set()
    frontier = {root_pid}
    while frontier:
        children = {pid for pid, parent in parent_by_pid.items() if parent in frontier}
        children -= descendants
        if not children:
            break
        descendants.update(children)
        frontier = children
    for sig in (signal.SIGTERM, signal.SIGKILL):
        for pid in sorted(descendants, reverse=True):
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass
        if sig == signal.SIGTERM and descendants:
            time.sleep(0.1)


def _next_cell_stub(turn_index: int) -> str:
    return (
        f'"""Code cell {turn_index}: continue from persistent globals and episode state."""\n'
        "# Read turn_feedback_latest.md, then write only the next incremental actions.\n"
    )


def _driver_source() -> str:
    return '''#!/usr/bin/env python3
import contextlib
import _thread
import io
import json
import os
import subprocess
import sys
import warnings
import threading
import traceback

READY = "__ROBENCH_REPL_READY__"
RESULT = "__ROBENCH_CELL_RESULT__"
GLOBALS = {"__name__": "__main__", "__file__": "solve.py"}
sys.path.insert(0, os.getcwd())

def _blocked_thread_start(*args, **kwargs):
    raise RuntimeError("Background threads are disabled in persistent RoBench cells.")

_thread.start_new_thread = _blocked_thread_start
threading._start_new_thread = _blocked_thread_start
_REAL_POPEN = subprocess.Popen
_CELL_CHILDREN = []

def _tracked_popen(*args, **kwargs):
    process = _REAL_POPEN(*args, **kwargs)
    _CELL_CHILDREN.append(process)
    return process

subprocess.Popen = _tracked_popen

def _cleanup_cell_children():
    for process in list(_CELL_CHILDREN):
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=.5)
    _CELL_CHILDREN.clear()

class CappedWriter(io.TextIOBase):
    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self.parts = []
        self.length = 0
        self.truncated = 0
    def write(self, value):
        value = str(value)
        available = max(0, self.limit - self.length)
        if available:
            kept = value[:available]
            self.parts.append(kept)
            self.length += len(kept)
        self.truncated += max(0, len(value) - available)
        return len(value)
    def getvalue(self):
        text = "".join(self.parts)
        if self.truncated:
            text += "\\n... [truncated %d chars]" % self.truncated
        return text

print(READY + json.dumps({"pid": os.getpid()}), flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get("shutdown"):
            break
        cell_path = request["cell_path"]
        limit = int(request.get("max_output_chars", 20000))
        stdout = CappedWriter(limit)
        stderr = CappedWriter(limit)
        returncode = 0
        try:
            code = open(cell_path, "r", encoding="utf-8").read()
            GLOBALS["__file__"] = "solve.py"
            primitive_token = str(request.get("primitive_token") or "")
            if primitive_token:
                os.environ["ROBENCH_PRIMITIVE_SERVER_TOKEN"] = primitive_token
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(code, cell_path, "exec"), GLOBALS, GLOBALS)
        except BaseException:
            returncode = 1
            traceback.print_exc(file=stderr)
        finally:
            os.environ.pop("ROBENCH_PRIMITIVE_SERVER_TOKEN", None)
            _cleanup_cell_children()
        payload = {"returncode": returncode, "stdout": stdout.getvalue(), "stderr": stderr.getvalue()}
    except BaseException:
        payload = {"returncode": 1, "stdout": "", "stderr": traceback.format_exc()}
    print(RESULT + json.dumps(payload, ensure_ascii=False), flush=True)
'''


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"
