#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
LOCAL_UV_TOOL_BIN = Path.home() / ".local" / "bin"
STREAM_TAIL_CHARS = 20000
SECRET_VALUE_PATTERNS = [
    re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(DEEPSEEK_API_KEY|LLM_API_KEY|OPENAI_API_KEY)\s*=\s*[^\s\"']{8,}"),
]


def add_executor_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--executor",
        choices=["probe", "openhands-headless"],
        default="probe",
        help="probe runs workspace solve.py directly; openhands-headless delegates agent loop to OpenHands CLI.",
    )
    parser.add_argument("--openhands-command", default="openhands", help="OpenHands CLI executable for openhands-headless.")
    parser.add_argument(
        "--openhands-extra-arg",
        action="append",
        default=[],
        help="Extra argument to append to the OpenHands CLI command. Repeat for multiple args.",
    )


def resolve_openhands_binary(value: str) -> str | None:
    if "/" in value:
        path = Path(value).expanduser()
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
        return None
    resolved = shutil.which(value)
    if resolved:
        return resolved
    local_uv_tool = LOCAL_UV_TOOL_BIN / value
    if local_uv_tool.exists() and os.access(local_uv_tool, os.X_OK):
        return str(local_uv_tool)
    return None


def missing_openhands_payload(benchmark: str, track: str, executor: str) -> dict[str, Any]:
    return {
        "benchmark": benchmark,
        "track": track,
        "status": "blocked_openhands_cli_missing",
        "executor": executor,
        "reason": "OpenHands CLI was not found. Install/configure OpenHands, or pass --openhands-command /absolute/path/to/openhands.",
        "expected_command_shape": "openhands --headless --json --override-with-envs --exit-without-confirmation -f task.md",
    }


def resolve_or_exit_openhands(args: argparse.Namespace, *, benchmark: str, track: str) -> str | None:
    if args.executor != "openhands-headless":
        return None
    binary = resolve_openhands_binary(args.openhands_command)
    if binary is None:
        print(json.dumps(missing_openhands_payload(benchmark, track, args.executor), indent=2, ensure_ascii=False))
        raise SystemExit(2)
    return binary


def build_executor_command(
    *,
    args: argparse.Namespace,
    solve_path: Path,
    openhands_binary: str | None,
) -> tuple[list[str], str, str]:
    if args.executor == "probe":
        return [sys.executable, solve_path.name], "openhands_workspace_probe", "robench_probe_subprocess"
    if args.executor == "openhands-headless":
        if openhands_binary is None:
            raise RuntimeError("OpenHands CLI executable was not resolved.")
        command = [
            openhands_binary,
            "--headless",
            "--json",
            "--override-with-envs",
            "--exit-without-confirmation",
            "-f",
            "task.md",
        ]
        command.extend(args.openhands_extra_arg or [])
        return command, "openhands_headless_cli", "openhands_headless_cli"
    raise ValueError(f"Unsupported executor: {args.executor}")


def is_secret_key_name(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in SECRET_KEY_HINTS)


def collect_secret_values(env: dict[str, str]) -> set[str]:
    return {
        value
        for key, value in env.items()
        if value and len(value) >= 8 and is_secret_key_name(key)
    }


def redact_text(text: str, secret_values: set[str]) -> str:
    redacted = text
    for value in sorted(secret_values, key=len, reverse=True):
        redacted = redacted.replace(value, "[REDACTED_SECRET]")
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def redact_completed_process(
    completed: subprocess.CompletedProcess[str],
    secret_values: set[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        completed.args,
        returncode=completed.returncode,
        stdout=redact_text(completed.stdout or "", secret_values),
        stderr=redact_text(completed.stderr or "", secret_values),
    )


class TailBuffer:
    def __init__(self, limit: int = STREAM_TAIL_CHARS):
        self.limit = limit
        self._text = ""
        self._lock = threading.Lock()

    def append(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._text += text
            if len(self._text) > self.limit:
                self._text = self._text[-self.limit :]

    def text(self) -> str:
        with self._lock:
            return self._text


def build_execution_log_paths(
    *,
    outputs_dir: Path,
    run_id: str,
    executor_source: str,
) -> dict[str, str]:
    log_dir = outputs_dir / "openhands_events" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "jsonl" if executor_source == "openhands_headless_cli" else "log"
    return {
        "stdout": str(log_dir / f"{executor_source}_stdout.{suffix}"),
        "stderr": str(log_dir / f"{executor_source}_stderr.log"),
    }


def stream_pipe_to_file(
    *,
    pipe: Any,
    path: Path,
    secret_values: set[str],
    tail: TailBuffer,
) -> None:
    with path.open("a", encoding="utf-8") as sink:
        for chunk in iter(pipe.readline, ""):
            redacted = redact_text(chunk, secret_values)
            sink.write(redacted)
            sink.flush()
            tail.append(redacted)


def append_stream_text(path: Path, text: str, secret_values: set[str], tail: TailBuffer) -> None:
    redacted = redact_text(text, secret_values)
    with path.open("a", encoding="utf-8") as sink:
        sink.write(redacted)
    tail.append(redacted)


def run_executor(
    *,
    command: list[str],
    workspace_dir: Path,
    env: dict[str, str],
    timeout_seconds: int,
    event_log_paths: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], bool]:
    secret_values = collect_secret_values(env)
    if event_log_paths:
        stdout_path = Path(event_log_paths["stdout"])
        stderr_path = Path(event_log_paths["stderr"])
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        stdout_tail = TailBuffer()
        stderr_tail = TailBuffer()
        process = subprocess.Popen(
            command,
            cwd=workspace_dir,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        threads = [
            threading.Thread(
                target=stream_pipe_to_file,
                kwargs={
                    "pipe": process.stdout,
                    "path": stdout_path,
                    "secret_values": secret_values,
                    "tail": stdout_tail,
                },
                daemon=True,
            ),
            threading.Thread(
                target=stream_pipe_to_file,
                kwargs={
                    "pipe": process.stderr,
                    "path": stderr_path,
                    "secret_values": secret_values,
                    "tail": stderr_tail,
                },
                daemon=True,
            ),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = 124
            process.wait()
        for thread in threads:
            thread.join(timeout=5)
        if timed_out:
            append_stream_text(
                stderr_path,
                f"\nTimed out after {timeout_seconds} seconds.",
                secret_values,
                stderr_tail,
            )
        completed = subprocess.CompletedProcess(
            command,
            returncode=returncode,
            stdout=stdout_tail.text(),
            stderr=stderr_tail.text(),
        )
        setattr(completed, "_robench_streamed_event_log_paths", event_log_paths)
        return completed, timed_out

    try:
        completed = subprocess.run(
                command,
                cwd=workspace_dir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
        )
        return redact_completed_process(completed, secret_values), False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        completed = subprocess.CompletedProcess(
                command,
                returncode=124,
                stdout=stdout,
                stderr=f"{stderr}\nTimed out after {timeout_seconds} seconds.",
        )
        return redact_completed_process(completed, secret_values), True


def persist_execution_logs(
    *,
    outputs_dir: Path,
    run_id: str,
    completed: subprocess.CompletedProcess[str],
    executor_source: str,
) -> dict[str, str]:
    streamed_paths = getattr(completed, "_robench_streamed_event_log_paths", None)
    if streamed_paths:
        return dict(streamed_paths)
    log_dir = outputs_dir / "openhands_events" / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    suffix = "jsonl" if executor_source == "openhands_headless_cli" else "log"
    stdout_path = log_dir / f"{executor_source}_stdout.{suffix}"
    stderr_path = log_dir / f"{executor_source}_stderr.log"
    stdout_path.write_text(completed.stdout or "", encoding="utf-8")
    stderr_path.write_text(completed.stderr or "", encoding="utf-8")
    return {
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
