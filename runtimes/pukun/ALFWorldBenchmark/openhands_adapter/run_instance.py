#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ADAPTER_DIR = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import openhands_bridge as oh_bridge
import persistent_repl_runtime as repl_runtime
from alfworld_text_harness.backend import ALFWorldTextBackend, StepResult
from alfworld_text_harness.config import HarnessConfig, get_config
from alfworld_text_harness.manifest import TaskRecord, build_manifest, select_suite
from alfworld_text_harness.primitives import ALFWorldTextPrimitives
from alfworld_text_harness.state_machine import ALFWorldTextStateMachine
from alfworld_text_harness.trace import TraceWriter


ALLOWED_PRIMITIVES = {
    "check_success",
    "clean_object",
    "close_object",
    "cool_object",
    "examine_object",
    "get_task_context",
    "go_to",
    "heat_object",
    "inventory",
    "list_actions",
    "look",
    "match_actions",
    "observe_text_state",
    "open_object",
    "pickup_object",
    "place_object",
    "read_evidence",
    "toggle_object",
    "write_evidence",
}
READ_ONLY_PRIMITIVES = {
    "check_success",
    "get_task_context",
    "list_actions",
    "match_actions",
    "observe_text_state",
    "read_evidence",
}
SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
SECRET_VALUE_PATTERNS = [
    re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?i)(DEEPSEEK_API_KEY|LLM_API_KEY|OPENAI_API_KEY)\s*=\s*[^\s\"']{8,}"),
]
LOCAL_UV_TOOL_BIN = Path.home() / ".local" / "bin"


class ALFWorldOpenHandsSession:
    def __init__(
        self,
        *,
        config: HarnessConfig,
        task: TaskRecord,
        suite: str,
        instance_id: str,
        max_env_steps: int,
        runtime_mode: str = repl_runtime.RUNTIME_MODE,
    ):
        self.config = config
        self.task = task
        self.suite = suite
        self.instance_id = instance_id
        self.max_env_steps = max_env_steps
        self.runtime_mode = runtime_mode
        self.backend = ALFWorldTextBackend(config, max_steps=max_env_steps)
        self.machine = ALFWorldTextStateMachine(self.backend, max_steps=max_env_steps)
        self.trace = TraceWriter(config)
        self.machine.reset_task(task)
        self.primitives = ALFWorldTextPrimitives(self.machine, self.trace, task, suite)
        self.side_effect_client_pid: int | None = None
        self.side_effect_client_pids: list[int] = []
        self.workspace_dir: Path | None = None

    def call(
        self,
        primitive: str,
        args: list[Any],
        kwargs: dict[str, Any],
        client: dict[str, Any] | None = None,
    ) -> Any:
        if primitive not in ALLOWED_PRIMITIVES:
            raise ValueError(f"Primitive '{primitive}' is not exposed in this OpenHands workspace.")
        if primitive not in READ_ONLY_PRIMITIVES:
            self._enforce_side_effect_client(client or {}, primitive)
        method = getattr(self.primitives, primitive)
        result = method(*args, **kwargs)
        if primitive not in READ_ONLY_PRIMITIVES:
            self._lock_side_effect_client_after_valid_action(client or {}, primitive, result)
        return _jsonable(result)

    def _enforce_side_effect_client(self, client: dict[str, Any], primitive: str) -> None:
        self._assert_workspace_facade_unchanged()
        entrypoint = Path(str(client.get("argv0") or "")).name
        pid = client.get("pid")
        if entrypoint != "solve.py":
            raise ValueError(
                f"ALFWorld side-effect primitive '{primitive}' must be called by running python solve.py. "
                "Ad-hoc python -c snippets, stdin Python, notebooks, or temporary scripts may inspect "
                "read-only primitives only; they may not manipulate the environment or write evidence."
            )
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("ALFWorld side-effect primitives require a valid solve.py process id.")
        self._assert_live_solve_process(pid, primitive)
        if self.side_effect_client_pid is not None and pid != self.side_effect_client_pid:
            raise ValueError(
                "This ALFWorld task already executed a real environment action from an earlier "
                "solve.py process. Do not rerun solve.py to keep manipulating the same backend "
                "session; inspect check_success() and finish."
            )

    def _lock_side_effect_client_after_valid_action(
        self,
        client: dict[str, Any],
        primitive: str,
        result: Any,
    ) -> None:
        pid = client.get("pid")
        entrypoint = Path(str(client.get("argv0") or "")).name
        if self.side_effect_client_pid is not None:
            return
        if not isinstance(result, StepResult) or not result.valid_action:
            return
        pid = client.get("pid")
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("ALFWorld side-effect primitives require a valid solve.py process id.")
        entrypoint = Path(str(client.get("argv0") or "")).name
        self.side_effect_client_pid = pid
        self.primitives.record_harness_event(
            "side_effect_client_locked",
            {
                "source": "primitive_server",
                "entrypoint": entrypoint,
                "client_pid": pid,
                "primitive": primitive,
                "runtime_mode": self.runtime_mode,
                "policy": "one persistent Python process owns valid environment action calls",
            },
            side_effect=False,
        )

    def _assert_workspace_facade_unchanged(self) -> None:
        if self.workspace_dir is None:
            return
        workspace_api = self.workspace_dir / "primitive_api.py"
        source_api = ADAPTER_DIR / "primitive_api.py"
        try:
            if workspace_api.read_bytes() != source_api.read_bytes():
                raise RuntimeError(
                    "Workspace primitive_api.py was modified. Only cell.py may be edited during an OpenHands run."
                )
        except OSError as exc:
            raise RuntimeError(f"Could not verify workspace primitive_api.py integrity: {exc}") from exc

    def _assert_live_solve_process(self, pid: int, primitive: str) -> None:
        try:
            completed = subprocess.run(
                ["ps", "-ww", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not verify solve.py process for primitive '{primitive}': {exc}") from exc
        command = (completed.stdout or "").strip()
        if completed.returncode != 0 or not command or "solve.py" not in command:
            raise RuntimeError(
                f"Side-effect primitive '{primitive}' requires a live python solve.py process; "
                f"pid {pid} command was {command or 'not found'}."
            )

    def record_agent_code(self, solve_path: Path) -> None:
        code = solve_path.read_text(encoding="utf-8")
        self.primitives.record_harness_event(
            "agent_code_generated",
            {
                "source": "openhands_workspace_code_cell",
                "code": code,
                "workspace_file": str(solve_path.name),
            },
            side_effect=False,
        )

    def record_execution_started(
        self,
        *,
        command: list[str],
        source: str,
        solve_path: Path,
        timeout_seconds: int,
    ) -> None:
        self.primitives.record_harness_event(
            "code_execution_started",
            {
                "source": source,
                "command": command,
                "workspace_file": str(solve_path.name),
                "max_env_steps": self.max_env_steps,
                "runtime_mode": self.runtime_mode,
                "timeout_seconds": timeout_seconds,
            },
            side_effect=False,
        )

    def record_execution_finished(
        self,
        completed: subprocess.CompletedProcess[str],
        *,
        source: str,
        timed_out: bool,
        event_log_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        final_verification = self.primitives.check_success()
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
        payload = {
            "source": source,
            "returncode": completed.returncode,
            "stdout": stdout[-20000:],
            "stderr": stderr[-20000:],
            "stdout_truncated": len(stdout) > 20000,
            "stderr_truncated": len(stderr) > 20000,
            "event_log_paths": event_log_paths or {},
            "exception": (
                f"Timed out while running {source}"
                if timed_out
                else None if completed.returncode == 0 else f"Agent process exited {completed.returncode}"
            ),
            "timed_out": timed_out,
            "primitive_budget_exceeded": False,
            "final_success": bool(final_verification.get("success")),
            "final_verification": final_verification,
        }
        self.primitives.record_harness_event("code_execution_finished", payload, side_effect=False)
        return payload

    def write_summary(
        self,
        *,
        completed: subprocess.CompletedProcess[str],
        execution_payload: dict[str, Any],
        workspace_dir: Path,
        suite_task_count: int,
        start_index: int,
        agent_label: str,
        executor_source: str,
        timed_out: bool,
        code_attempts: int = 1,
        event_log_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        final = self.backend.check_success()
        stopped_reason = "success" if final.success else "agent_exit_0"
        if timed_out:
            stopped_reason = "code_timeout"
        if completed.returncode != 0:
            stopped_reason = "agent_error"
        if timed_out:
            stopped_reason = "code_timeout"
        row = {
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "task_id": self.task.task_id,
            "task_type": self.task.task_type,
            "source_split": self.task.source_split,
            "goal_text": self.task.goal_text,
            "success": final.success,
            "score": final.score,
            "done": final.done,
            "stopped_reason": stopped_reason,
            "exception": execution_payload.get("exception"),
            "env_steps": self.machine.step_count,
            "invalid_action_count": self.primitives.metrics.get("invalid_action_count", 0),
            "wrapper_no_match_count": self.primitives.metrics.get("wrapper_no_match_count", 0),
            "wrapper_ambiguity_count": self.primitives.metrics.get("wrapper_ambiguity_count", 0),
            "code_attempts": code_attempts,
            "code_exception_count": int(completed.returncode != 0),
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": 0,
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
        }
        summary = {
            "run_id": self.trace.run_id,
            "benchmark": "ALFWorld",
            "track": "text",
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "num_tasks": 1,
            "suite_task_count": suite_task_count,
            "start_index": start_index,
            "end_index_exclusive": start_index + 1,
            "is_full_suite_run": False,
            "successes": int(final.success),
            "success_rate": 1.0 if final.success else 0.0,
            "success_by_task_type": _success_by_task_type(row["task_type"], row["success"]),
            "avg_steps": float(self.machine.step_count),
            "avg_score": float(final.score),
            "invalid_action_count": row["invalid_action_count"],
            "wrapper_no_match_count": row["wrapper_no_match_count"],
            "wrapper_ambiguity_count": row["wrapper_ambiguity_count"],
            "avg_code_attempts": float(code_attempts),
            "runtime_mode": self.runtime_mode,
            "code_exception_count": row["code_exception_count"],
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": 0,
            "timeout_count": int(timed_out),
            "exception_count": int(completed.returncode != 0),
            "requested_tasks": 1,
            "completed_tasks": 1,
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
            "boundary": (
                "OpenHands-style local workspace smoke. Backend/private task resources are served through "
                "a primitive facade and are not copied into the agent workspace."
            ),
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "task_type": row["task_type"],
                    "source_split": row["source_split"],
                    "goal_text": row["goal_text"],
                    "suite_index": start_index,
                    "success": row["success"],
                    "env_steps": row["env_steps"],
                    "score": row["score"],
                    "stopped_reason": row["stopped_reason"],
                    "exception": row["exception"],
                    "code_attempts": row["code_attempts"],
                }
            ],
        }
        self.trace.write_summary(summary, [row])
        return summary

    def close(self) -> None:
        self.backend.close()


class PrimitiveRequestHandler(BaseHTTPRequestHandler):
    server: "PrimitiveHTTPServer"

    def do_POST(self) -> None:
        if self.path != "/call":
            self._send_json(404, {"ok": False, "error": "unknown endpoint"})
            return
        if not secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {self.server.session_token}"):
            self._send_json(401, {"ok": False, "error": "unauthorized"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            primitive = str(payload.get("primitive"))
            args = list(payload.get("args") or [])
            kwargs = dict(payload.get("kwargs") or {})
            client = payload.get("client") if isinstance(payload.get("client"), dict) else {}
            with self.server.call_lock:
                result = self.server.session.call(primitive, args, kwargs, client)
            self._send_json(200, {"ok": True, "result": result})
        except Exception as exc:
            self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(_jsonable(payload), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class PrimitiveHTTPServer(ThreadingHTTPServer):
    def __init__(self, session: ALFWorldOpenHandsSession):
        super().__init__(("127.0.0.1", 0), PrimitiveRequestHandler)
        self.session = session
        self.session_token = secrets.token_urlsafe(32)
        self.call_lock = threading.RLock()


def write_persistent_feedback(
    *,
    session: ALFWorldOpenHandsSession,
    workspace_dir: Path,
    turn_index: int,
    max_turns: int,
    completed: subprocess.CompletedProcess[str],
    timed_out: bool,
) -> dict[str, Any]:
    verification = session.backend.check_success()
    return repl_runtime.write_turn_feedback_files(
        session=session,
        workspace_dir=workspace_dir,
        turn_index=turn_index,
        max_turns=max_turns,
        completed=completed,
        timed_out=timed_out,
        payload={
            "success": bool(verification.success),
            "terminal": bool(verification.done) or session.machine.terminal,
            "score": verification.score,
            "verifier": verification.to_dict(),
            "env_steps": session.machine.step_count,
            "remaining_env_steps": max(0, session.machine.max_steps - session.machine.step_count),
            "current_observation": session.backend.observe_text_state(),
            "current_actions": session.backend.list_actions()[:80],
            "evidence": session.primitives.evidence,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ALFWorld Text OpenHands-style instance.")
    parser.add_argument("--instance-id", default="alfworld_text_debug_0")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--task-index", type=int, default=None, help="Visual-track alias for --start-index.")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional visual manifest path.")
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--max-primitive-calls", type=int, default=None)
    parser.add_argument("--code-timeout-seconds", type=int, default=180)
    repl_runtime.add_runtime_args(parser)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--solution", type=Path, default=None, help="Optional initial cell.py to copy into the workspace.")
    parser.add_argument("--force-manifest", action="store_true")
    oh_bridge.add_executor_args(parser)
    args = parser.parse_args()

    instance = load_instance(args.instance_id)
    if instance.get("track") == "visual":
        from run_visual_instance import run_from_parent_args

        run_from_parent_args(args, instance)
        return

    openhands_binary = oh_bridge.resolve_or_exit_openhands(args, benchmark="ALFWorld", track="text")

    suite = args.suite or str(instance["suite"])
    start_index = args.start_index if args.start_index is not None else int(instance["start_index"])
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else int(instance["max_env_steps"])

    config = get_config()
    manifest = build_manifest(config, force=args.force_manifest)
    suite_task_count = len(select_suite(manifest, suite=suite, num_tasks=None, start_index=0))
    tasks = select_suite(manifest, suite=suite, num_tasks=1, start_index=start_index)
    if not tasks:
        raise RuntimeError(f"No ALFWorld task selected for suite={suite} start_index={start_index}.")
    task = tasks[0]

    session = ALFWorldOpenHandsSession(
        config=config,
        task=task,
        suite=suite,
        instance_id=args.instance_id,
        max_env_steps=max_env_steps,
        runtime_mode=args.runtime_mode,
    )
    workspace_dir = args.workspace_dir or config.outputs_dir / "openhands_workspaces" / session.trace.run_id / args.instance_id
    prepare_workspace(
        workspace_dir,
        instance=instance,
        task=task,
        suite=suite,
        instance_id=args.instance_id,
        max_env_steps=max_env_steps,
        solution=args.solution,
        executor=args.executor,
    )
    session.workspace_dir = workspace_dir
    cell_path = workspace_dir / "cell.py"

    server = PrimitiveHTTPServer(session)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    env = {
        **os.environ,
        "ROBENCH_PRIMITIVE_SERVER_URL": f"http://127.0.0.1:{server.server_port}",
        "ROBENCH_PRIMITIVE_SERVER_TOKEN": server.session_token,
    }
    try:
        completed, timed_out, event_log_paths, agent_label, executor_source, code_attempts = repl_runtime.run_persistent_repl_code(
            args=args,
            session=session,
            workspace_dir=workspace_dir,
            cell_path=cell_path,
            openhands_binary=openhands_binary,
            env=env,
            outputs_dir=config.outputs_dir,
            benchmark_label="ALFWorld Text",
            feedback_writer=lambda turn, max_turns, proc, timeout: write_persistent_feedback(
                session=session,
                workspace_dir=workspace_dir,
                turn_index=turn,
                max_turns=max_turns,
                completed=proc,
                timed_out=timeout,
            ),
        )
        execution_payload = session.record_execution_finished(
            completed,
            source=executor_source,
            timed_out=timed_out,
            event_log_paths=event_log_paths,
        )
        summary = session.write_summary(
            completed=completed,
            execution_payload=execution_payload,
            workspace_dir=workspace_dir,
            suite_task_count=suite_task_count,
            start_index=start_index,
            agent_label=agent_label,
            executor_source=executor_source,
            timed_out=timed_out,
            code_attempts=code_attempts,
            event_log_paths=event_log_paths,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        session.close()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def load_instance(instance_id: str) -> dict[str, Any]:
    with (ADAPTER_DIR / "instances.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("instance_id") == instance_id:
                return row
    raise KeyError(f"Unknown OpenHands adapter instance_id: {instance_id}")


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


def persist_execution_logs(
    *,
    config: HarnessConfig,
    run_id: str,
    completed: subprocess.CompletedProcess[str],
    executor_source: str,
) -> dict[str, str]:
    log_dir = config.outputs_dir / "openhands_events" / run_id
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


def prepare_workspace(
    workspace_dir: Path,
    *,
    instance: dict[str, Any],
    task: TaskRecord,
    suite: str,
    instance_id: str,
    max_env_steps: int,
    solution: Path | None,
    executor: str,
) -> None:
    repl_runtime.create_workspace(workspace_dir, allowed_root=ROOT / "outputs" / "openhands_workspaces")
    shutil.copy2(ADAPTER_DIR / "primitive_api.py", workspace_dir / "primitive_api.py")
    shutil.copy2(ADAPTER_DIR / "primitive_cards.md", workspace_dir / "primitive_cards.md")
    shutil.copy2(ADAPTER_DIR / "primitive_cards.json", workspace_dir / "primitive_cards.json")
    template = (ADAPTER_DIR / "task_template.md").read_text(encoding="utf-8")
    (workspace_dir / "task.md").write_text(
        template.format(
            benchmark=instance["benchmark"],
            track=instance["track"],
            suite=suite,
            instance_id=instance_id,
            task_id=task.task_id,
            task_type=task.task_type,
            source_split=task.source_split,
            goal_text=task.goal_text,
            max_env_steps=max_env_steps,
        ),
        encoding="utf-8",
    )
    if solution is not None:
        shutil.copy2(solution, workspace_dir / "cell.py")
    elif executor == "openhands-headless":
        (workspace_dir / "cell.py").write_text(STARTER_SOLVE_PY, encoding="utf-8")
    else:
        (workspace_dir / "cell.py").write_text(PROBE_SOLVE_PY, encoding="utf-8")


STARTER_SOLVE_PY = '''"""Write the next bounded code cell for the persistent RoBench session.

Use only primitive_api.py. The harness executes this cell and preserves its globals
for later cells. Do not execute it during authoring.
"""
'''


PROBE_SOLVE_PY = '''from primitive_api import (
    check_success,
    get_task_context,
    list_actions,
    match_actions,
    observe_text_state,
    read_evidence,
    write_evidence,
)

ctx = get_task_context()
obs = observe_text_state()
actions = list_actions()
pickup_matches = match_actions(intent="pickup", limit=10)
go_matches = match_actions(intent="go_to", limit=10)
write_evidence(
    "initial_probe",
    {
        "goal": ctx["goal_text"],
        "observation_head": obs[:500],
        "num_actions": len(actions),
        "action_sample": actions[:20],
        "pickup_matches": pickup_matches,
        "go_matches": go_matches,
    },
)
print("TASK:", ctx["task_id"])
print("GOAL:", ctx["goal_text"])
print("ACTIONS:", actions[:10])
print("EVIDENCE:", read_evidence())
print("VERIFY:", check_success())
'''


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        payload = asdict(value)
        for attr in ("success", "done", "score"):
            if hasattr(value, attr):
                payload[attr] = getattr(value, attr)
        return _jsonable(payload)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _success_by_task_type(task_type: str | None, success: bool) -> dict[str, dict[str, float | int]]:
    key = str(task_type or "unknown")
    return {
        key: {
            "tasks": 1,
            "successes": int(bool(success)),
            "success_rate": 1.0 if success else 0.0,
        }
    }


if __name__ == "__main__":
    main()
