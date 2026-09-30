#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
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

import persistent_repl_runtime as repl_runtime
from openhands_bridge import add_executor_args, build_execution_log_paths, build_executor_command, persist_execution_logs, resolve_or_exit_openhands, run_executor
from scienceworld_text_harness.backend import ScienceWorldTextBackend
from scienceworld_text_harness.code_executor import default_primitive_call_budget
from scienceworld_text_harness.config import HarnessConfig, get_config
from scienceworld_text_harness.manifest import TaskRecord, build_manifest, select_suite
from scienceworld_text_harness.primitives import ScienceWorldTextPrimitives
from scienceworld_text_harness.state_machine import ScienceWorldTextStateMachine
from scienceworld_text_harness.trace import TraceWriter


ALLOWED_PRIMITIVES = {
    "check_success",
    "filter_actions",
    "get_score_state",
    "get_task_context",
    "inspect_current_state",
    "inventory",
    "list_actions",
    "list_recent_failures",
    "look",
    "observe_text_world",
    "read_evidence",
    "step_text_action",
    "write_evidence",
}


class ScienceWorldOpenHandsSession:
    def __init__(
        self,
        *,
        config: HarnessConfig,
        task: TaskRecord,
        suite: str,
        instance_id: str,
        max_env_steps: int,
        max_primitive_calls: int | None,
        runtime_mode: str = repl_runtime.RUNTIME_MODE,
    ):
        self.config = config
        self.task = task
        self.suite = suite
        self.instance_id = instance_id
        self.max_env_steps = max_env_steps
        self.max_primitive_calls = max_primitive_calls
        self.primitive_call_count = 0
        self.primitive_budget_exceeded = False
        self.runtime_mode = runtime_mode
        self.backend = ScienceWorldTextBackend(config)
        self.machine = ScienceWorldTextStateMachine(self.backend, max_steps=max_env_steps)
        self.trace = TraceWriter(config)
        self.machine.reset_task(task)
        self.primitives = ScienceWorldTextPrimitives(self.machine, self.trace, task, suite)
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
        self._enforce_workspace_client(client or {}, primitive)
        self._count_primitive_call(primitive)
        method = getattr(self.primitives, primitive)
        result = _jsonable(method(*args, **kwargs))
        if primitive == "step_text_action":
            return _compact_step_result_for_agent(result)
        return result

    def _count_primitive_call(self, primitive: str) -> None:
        self.primitive_call_count += 1
        if self.max_primitive_calls is not None and self.primitive_call_count > self.max_primitive_calls:
            self.primitive_budget_exceeded = True
            raise RuntimeError(
                f"Exceeded max_primitive_calls={self.max_primitive_calls} while calling {primitive}. "
                "Generated code is likely looping without progress."
            )

    def _enforce_workspace_client(self, client: dict[str, Any], primitive: str) -> None:
        self._assert_workspace_facade_unchanged()
        entrypoint = Path(str(client.get("argv0") or "")).name
        pid = client.get("pid")
        if entrypoint != "solve.py":
            raise ValueError(
                "ScienceWorld primitive_api calls must be made by running python solve.py. "
                "Ad-hoc python -c snippets, stdin Python, notebooks, heredocs, or temporary "
                "scripts may not import/call primitive_api functions for this adapter."
            )
        if not isinstance(pid, int) or pid <= 0:
            raise ValueError("ScienceWorld primitive_api calls require a valid solve.py process id.")
        self._assert_live_solve_process(pid)
        if primitive != "step_text_action":
            return
        if self.side_effect_client_pid is None:
            self.side_effect_client_pid = pid
            self.primitives.record_harness_event(
                "side_effect_client_locked",
                {
                    "source": "primitive_server",
                    "entrypoint": entrypoint,
                    "client_pid": pid,
                    "runtime_mode": self.runtime_mode,
                    "policy": "one persistent Python process may issue side-effect environment actions",
                },
                side_effect=False,
            )
            return
        if pid != self.side_effect_client_pid:
            raise ValueError(
                "This ScienceWorld task is locked to its persistent Python worker; "
                "a different process cannot manipulate the same backend episode."
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

    def _assert_live_solve_process(self, pid: int) -> None:
        try:
            completed = subprocess.run(
                ["ps", "-ww", "-p", str(pid), "-o", "command="],
                check=False,
                capture_output=True,
                text=True,
                timeout=2,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"Could not verify solve.py process for ScienceWorld action: {exc}") from exc
        command = (completed.stdout or "").strip()
        if completed.returncode != 0 or not command or "solve.py" not in command:
            raise RuntimeError(
                "ScienceWorld side-effect actions require a live python solve.py process; "
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
                "max_primitive_calls": self.max_primitive_calls,
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
            "primitive_budget_exceeded": self.primitive_budget_exceeded,
            "primitive_call_count": self.primitive_call_count,
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
        if self.primitive_budget_exceeded:
            stopped_reason = "primitive_call_budget"
        if timed_out:
            stopped_reason = "code_timeout"
        row = {
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "task_id": self.task.task_id,
            "task_name": self.task.task_name,
            "task_type": self.task.task_type,
            "variation_idx": self.task.variation_idx,
            "simplification": self.task.simplification,
            "source_split": self.task.source_split,
            "goal_text": self.machine.task_description or self.task.goal_text,
            "success": final.success,
            "score": final.score,
            "done": final.done,
            "stopped_reason": stopped_reason,
            "exception": execution_payload.get("exception"),
            "env_steps": self.machine.step_count,
            "invalid_action_count": self.primitives.metrics.get("invalid_action_count", 0),
            "parser_no_match_count": self.primitives.metrics.get("parser_no_match_count", 0),
            "code_attempts": code_attempts,
            "code_exception_count": int(completed.returncode != 0),
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": int(self.primitive_budget_exceeded),
            "primitive_call_count": self.primitive_call_count,
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
        }
        summary = {
            "run_id": self.trace.run_id,
            "benchmark": "ScienceWorld",
            "track": "text",
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "suite_task_count": suite_task_count,
            "start_index": start_index,
            "end_index_exclusive": start_index + 1,
            "is_full_suite_run": False,
            "partial": False,
            "num_tasks": 1,
            "successes": int(final.success),
            "success_rate": 1.0 if final.success else 0.0,
            "success_by_task_type": _success_by_task_type(row["task_type"], row["success"]),
            "avg_steps": float(self.machine.step_count),
            "avg_score": float(final.score),
            "invalid_action_count": row["invalid_action_count"],
            "parser_no_match_count": row["parser_no_match_count"],
            "avg_code_attempts": float(code_attempts),
            "runtime_mode": self.runtime_mode,
            "code_exception_count": row["code_exception_count"],
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": row["code_primitive_budget_count"],
            "primitive_call_count": self.primitive_call_count,
            "timeout_count": int(timed_out),
            "blocked_count": 0,
            "exception_count": int(completed.returncode != 0),
            "requested_tasks": 1,
            "completed_tasks": 1,
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
            "boundary": (
                "OpenHands-style local workspace smoke. ScienceWorldEnv and private runtime objects are served "
                "through a primitive facade and are not copied into the agent workspace."
            ),
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "task_name": row["task_name"],
                    "task_type": row["task_type"],
                    "variation_idx": row["variation_idx"],
                    "simplification": row["simplification"],
                    "source_split": row["source_split"],
                    "goal_text": row["goal_text"],
                    "suite_index": start_index,
                    "success": row["success"],
                    "score": row["score"],
                    "env_steps": row["env_steps"],
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
    def __init__(self, session: ScienceWorldOpenHandsSession):
        super().__init__(("127.0.0.1", 0), PrimitiveRequestHandler)
        self.session = session
        self.session_token = secrets.token_urlsafe(32)
        self.call_lock = threading.RLock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ScienceWorld Text OpenHands-style instance.")
    parser.add_argument("--instance-id", default="scienceworld_text_debug_0")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--max-primitive-calls", type=int, default=None)
    parser.add_argument("--code-timeout-seconds", type=int, default=180)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--solution", type=Path, default=None, help="Optional initial cell.py to copy into the workspace.")
    parser.add_argument("--force-manifest", action="store_true")
    repl_runtime.add_runtime_args(parser)
    add_executor_args(parser)
    args = parser.parse_args()
    openhands_binary = resolve_or_exit_openhands(args, benchmark="ScienceWorld", track="text")

    instance = load_instance(args.instance_id)
    suite = args.suite or str(instance["suite"])
    start_index = args.start_index if args.start_index is not None else int(instance["start_index"])
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else int(instance["max_env_steps"])
    max_primitive_calls = (
        args.max_primitive_calls
        if args.max_primitive_calls is not None
        else default_primitive_call_budget(max_env_steps)
    )

    config = get_config()
    manifest = build_manifest(config, force=args.force_manifest)
    suite_task_count = len(select_suite(manifest, suite=suite, start_index=0, num_tasks=None))
    tasks = select_suite(manifest, suite=suite, start_index=start_index, num_tasks=1)
    if not tasks:
        raise RuntimeError(f"No ScienceWorld task selected for suite={suite} start_index={start_index}.")
    task = tasks[0]

    session = ScienceWorldOpenHandsSession(
        config=config,
        task=task,
        suite=suite,
        instance_id=args.instance_id,
        max_env_steps=max_env_steps,
        max_primitive_calls=max_primitive_calls,
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
        max_primitive_calls=max_primitive_calls,
        goal_text=session.machine.task_description or task.goal_text,
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
            benchmark_label="ScienceWorld Text",
            feedback_writer=lambda turn, max_turns, proc, timeout: write_persistent_turn_feedback(
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


def write_persistent_turn_feedback(
    *,
    session: ScienceWorldOpenHandsSession,
    workspace_dir: Path,
    turn_index: int,
    max_turns: int,
    completed: subprocess.CompletedProcess[str],
    timed_out: bool,
) -> dict[str, Any]:
    final = session.backend.check_success()
    verifier = _jsonable(final)
    try:
        observation = session.primitives.observe_text_world()
    except Exception as exc:
        observation = f"{type(exc).__name__}: {exc}"
    try:
        actions = session.primitives.list_actions()[:120]
    except Exception as exc:
        actions = [f"{type(exc).__name__}: {exc}"]
    payload = {
        "success": bool(getattr(final, "success", False)),
        "terminal": bool(getattr(final, "success", False) or getattr(final, "done", False) or session.machine.terminal),
        "score": getattr(final, "score", 0.0),
        "verifier": verifier,
        "current_observation": observation,
        "current_actions": actions,
        "env_steps": session.machine.step_count,
        "remaining_env_steps": max(0, session.machine.max_steps - session.machine.step_count),
        "metrics": session.primitives.metrics,
        "evidence": getattr(session.primitives, "evidence", {}),
    }
    return repl_runtime.write_turn_feedback_files(
        session=session,
        workspace_dir=workspace_dir,
        turn_index=turn_index,
        max_turns=max_turns,
        completed=completed,
        timed_out=timed_out,
        payload=payload,
    )


def load_instance(instance_id: str) -> dict[str, Any]:
    with (ADAPTER_DIR / "instances.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("instance_id") == instance_id:
                return row
    raise KeyError(f"Unknown OpenHands adapter instance_id: {instance_id}")


def prepare_workspace(
    workspace_dir: Path,
    *,
    instance: dict[str, Any],
    task: TaskRecord,
    suite: str,
    instance_id: str,
    max_env_steps: int,
    max_primitive_calls: int,
    goal_text: str,
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
            task_name=task.task_name,
            source_split=task.source_split,
            variation_idx=task.variation_idx,
            simplification=task.simplification,
            goal_text=goal_text,
            max_env_steps=max_env_steps,
            max_primitive_calls=max_primitive_calls,
            mode_instructions=build_mode_instructions(instance),
        ),
        encoding="utf-8",
    )
    if solution is not None:
        shutil.copy2(solution, workspace_dir / "cell.py")
    elif executor == "openhands-headless":
        (workspace_dir / "cell.py").write_text(STARTER_SOLVE_PY, encoding="utf-8")
    else:
        (workspace_dir / "cell.py").write_text(PROBE_SOLVE_PY, encoding="utf-8")


def build_mode_instructions(instance: dict[str, Any]) -> str:
    mode = str(instance.get("evaluation_mode") or "bounded_eval")
    if mode == "conformance_smoke":
        return (
            "This instance is a conformance smoke for the OpenHands-compatible harness, "
            "not a score-optimization run. Keep the provided `cell.py` as the intended "
            "small read-only probe; do not expand it into action exploration."
        )
    return (
        "This is a bounded evaluation. Write only the next `cell.py`; keep primitive calls "
        "and action selection inside harness-executed cells and avoid exhaustive action sweeps."
    )


STARTER_SOLVE_PY = '''"""Write the next bounded code cell for the persistent RoBench session.

Use only primitive_api.py. The harness executes this cell and preserves its globals
for later cells. Do not execute it during authoring.
"""
'''


PROBE_SOLVE_PY = '''from primitive_api import (
    check_success,
    filter_actions,
    get_score_state,
    get_task_context,
    list_actions,
    list_recent_failures,
    observe_text_world,
    read_evidence,
    write_evidence,
)

ctx = get_task_context()
obs = observe_text_world()
actions = list_actions()
movement = filter_actions(startswith="move", limit=10)
look_actions = filter_actions(include="look", limit=10)
failed_actions = list_recent_failures()
score_state = get_score_state()
write_evidence(
    "initial_probe",
    {
        "goal": ctx["goal_text"],
        "observation_head": obs[:500],
        "num_actions": len(actions),
        "action_sample": actions[:20],
        "movement_sample": movement,
        "look_action_sample": look_actions,
        "failed_actions": failed_actions,
        "score_state": score_state,
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
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _compact_step_result_for_agent(result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    verification = result.get("verification") if isinstance(result.get("verification"), dict) else {}
    actions_before = result.get("valid_actions_before")
    actions_after = result.get("valid_actions_after")
    actions_after_list = actions_after if isinstance(actions_after, list) else []
    compact = {
        "action": result.get("action"),
        "valid_action": bool(result.get("valid_action")),
        "error": result.get("error"),
        "observation_after": _truncate_text(result.get("observation_after"), 2200),
        "verification": verification,
        "success": bool(verification.get("success")),
        "done": bool(verification.get("done")),
        "score": verification.get("score"),
        "reward": verification.get("reward"),
        "valid_actions_before_count": len(actions_before) if isinstance(actions_before, list) else None,
        "valid_actions_after_count": len(actions_after_list) if isinstance(actions_after, list) else None,
        "valid_actions_after_sample": actions_after_list[:40],
        "note": "Compact StepResult. Use list_actions() or filter_actions() for the current full valid action set.",
    }
    return compact


def _truncate_text(value: Any, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... [truncated {len(text) - limit} chars]"


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
