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
from discoveryworld_harness.backend import DiscoveryWorldBackend
from discoveryworld_harness.config import HarnessConfig, get_config
from discoveryworld_harness.manifest import TaskRecord, build_manifest, select_suite
from discoveryworld_harness.primitives import DiscoveryWorldPrimitives
from discoveryworld_harness.trace import TraceWriter
from discoveryworld_harness.verifier import empty_verification


ALLOWED_PRIMITIVES = {
    "activate_object",
    "check_success",
    "choose_dialog_option",
    "close_object",
    "deactivate_object",
    "drop_object",
    "eat_object",
    "get_action_schema",
    "get_task_context",
    "list_accessible_objects",
    "list_inventory",
    "list_known_actions",
    "list_nearby_objects",
    "list_teleport_locations",
    "move_direction",
    "observe_world",
    "open_object",
    "pickup_object",
    "put_object",
    "read_evidence",
    "read_object",
    "rotate_direction",
    "talk_to",
    "teleport_to_location",
    "use_object",
    "validate_action_call",
    "wait",
    "write_evidence",
}

READ_ONLY_PRIMITIVES = {
    "check_success",
    "get_action_schema",
    "get_task_context",
    "list_accessible_objects",
    "list_inventory",
    "list_known_actions",
    "list_nearby_objects",
    "list_teleport_locations",
    "observe_world",
    "read_evidence",
    "validate_action_call",
    "write_evidence",
}


class DiscoveryWorldOpenHandsSession:
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
        self.backend = DiscoveryWorldBackend(config)
        self.trace = TraceWriter(config)
        self.backend.reset_task(task)
        self.primitives = DiscoveryWorldPrimitives(self.backend, self.trace, task, suite)
        self.side_effect_client_pid: int | None = None
        self.side_effect_client_pids: list[int] = []
        self.workspace_dir: Path | None = None

    def call(
        self,
        primitive: str,
        args: list[Any],
        kwargs: dict[str, Any],
        *,
        client: dict[str, Any] | None = None,
    ) -> Any:
        if primitive not in ALLOWED_PRIMITIVES:
            raise ValueError(f"Primitive '{primitive}' is not exposed in this OpenHands workspace.")
        self._enforce_side_effect_client(primitive, client)
        if self.primitives.metrics["env_steps"] >= self.max_env_steps and primitive not in READ_ONLY_PRIMITIVES:
            raise RuntimeError(f"Exceeded max_env_steps={self.max_env_steps}.")
        method = getattr(self.primitives, primitive)
        return _jsonable(method(*args, **kwargs))

    def _enforce_side_effect_client(self, primitive: str, client: dict[str, Any] | None) -> None:
        if primitive in READ_ONLY_PRIMITIVES:
            return
        self._assert_workspace_facade_unchanged()
        client = client or {}
        argv0 = str(client.get("argv0") or "")
        entrypoint = Path(argv0).name
        if entrypoint != "solve.py":
            raise RuntimeError(
                f"Side-effect primitive '{primitive}' must be called from solve.py, not {entrypoint or 'unknown entrypoint'}."
            )
        try:
            pid = int(client.get("pid"))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Side-effect primitive '{primitive}' requires a valid client pid.") from exc
        self._assert_live_solve_process(pid, primitive)
        if self.side_effect_client_pid is None:
            self.side_effect_client_pid = pid
            self.primitives.record_harness_event(
                "side_effect_client_locked",
                {
                    "source": "primitive_server",
                    "entrypoint": entrypoint,
                    "client_pid": pid,
                    "runtime_mode": self.runtime_mode,
                    "policy": "one persistent Python process owns the episode",
                },
                side_effect=False,
            )
            return
        if pid != self.side_effect_client_pid:
            raise RuntimeError(
                f"Side-effect primitive '{primitive}' must stay in the persistent Python process "
                f"(expected pid {self.side_effect_client_pid}, got {pid})."
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
            "scenario_name": self.task.scenario_name,
            "difficulty": self.task.difficulty,
            "seed": self.task.seed,
            "task_type": self.task.task_type,
            "goal_text": self.task.goal_text,
            "success": final.success,
            "score": final.score_normalized,
            "completed": final.completed,
            "stopped_reason": stopped_reason,
            "exception": execution_payload.get("exception"),
            "env_steps": self.primitives.metrics["env_steps"],
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
            "benchmark": "DiscoveryWorld",
            "track": "text_json",
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "suite_task_count": suite_task_count,
            "start_index": start_index,
            "end_index_exclusive": start_index + 1,
            "is_full_suite_run": False,
            "requested_tasks": 1,
            "completed_tasks": 1,
            "num_tasks": 1,
            "successes": int(final.success),
            "success_rate": 1.0 if final.success else 0.0,
            "success_by_task_type": _success_by_task_type(row["task_type"], row["success"]),
            "avg_steps": float(row["env_steps"]),
            "avg_score": float(final.score_normalized),
            "invalid_action_count": row["invalid_action_count"],
            "wrapper_no_match_count": row["wrapper_no_match_count"],
            "wrapper_ambiguity_count": row["wrapper_ambiguity_count"],
            "avg_code_attempts": float(code_attempts),
            "runtime_mode": self.runtime_mode,
            "code_exception_count": row["code_exception_count"],
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": 0,
            "timeout_count": int(timed_out),
            "blocked_count": 0,
            "exception_count": int(completed.returncode != 0),
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
            "boundary": (
                "OpenHands-style local workspace smoke. DiscoveryWorld API and private runtime objects are served "
                "through a primitive facade and are not copied into the agent workspace. This adapter is teleport-enabled."
            ),
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "scenario_name": row["scenario_name"],
                    "difficulty": row["difficulty"],
                    "seed": row["seed"],
                    "task_type": row["task_type"],
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
            client = dict(payload.get("client") or {})
            with self.server.call_lock:
                result = self.server.session.call(primitive, args, kwargs, client=client)
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
    def __init__(self, session: DiscoveryWorldOpenHandsSession):
        super().__init__(("127.0.0.1", 0), PrimitiveRequestHandler)
        self.session = session
        self.session_token = secrets.token_urlsafe(32)
        self.call_lock = threading.RLock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one DiscoveryWorld OpenHands-style instance.")
    parser.add_argument("--instance-id", default="discoveryworld_debug_0")
    parser.add_argument("--suite", default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--code-timeout-seconds", type=int, default=180)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--solution", type=Path, default=None, help="Optional initial cell.py to copy into the workspace.")
    parser.add_argument("--force-manifest", action="store_true")
    repl_runtime.add_runtime_args(parser)
    add_executor_args(parser)
    args = parser.parse_args()
    openhands_binary = resolve_or_exit_openhands(args, benchmark="DiscoveryWorld", track="text_json")

    instance = load_instance(args.instance_id)
    suite = args.suite or str(instance["suite"])
    start_index = args.start_index if args.start_index is not None else int(instance["start_index"])
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else int(instance["max_env_steps"])

    config = get_config()
    manifest = build_manifest(config, force=args.force_manifest)
    suite_task_count = len(manifest[suite])
    tasks = select_suite(manifest, suite=suite, start_index=start_index, num_tasks=1)
    if not tasks:
        raise RuntimeError(f"No DiscoveryWorld task selected for suite={suite} start_index={start_index}.")
    task = tasks[0]

    session = DiscoveryWorldOpenHandsSession(
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
        "SDL_VIDEODRIVER": "dummy",
        "PYGAME_HIDE_SUPPORT_PROMPT": "1",
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
            benchmark_label="DiscoveryWorld",
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
    session: DiscoveryWorldOpenHandsSession,
    workspace_dir: Path,
    turn_index: int,
    max_turns: int,
    completed: subprocess.CompletedProcess[str],
    timed_out: bool,
) -> dict[str, Any]:
    final = session.backend.check_success()
    verifier = _jsonable(final)
    try:
        observation = session.primitives.observe_world()
    except Exception as exc:
        observation = {"error": f"{type(exc).__name__}: {exc}"}
    current_actions: dict[str, Any] = {}
    for key, getter in {
        "known_actions": session.primitives.list_known_actions,
        "nearby_objects": session.primitives.list_nearby_objects,
        "accessible_objects": session.primitives.list_accessible_objects,
        "inventory": session.primitives.list_inventory,
        "teleport_locations": session.primitives.list_teleport_locations,
    }.items():
        try:
            current_actions[key] = getter()
        except Exception as exc:
            current_actions[f"{key}_error"] = f"{type(exc).__name__}: {exc}"
    payload = {
        "success": bool(getattr(final, "success", False)),
        "terminal": bool(getattr(final, "success", False) or getattr(final, "completed", False)),
        "score": getattr(final, "score_normalized", 0.0),
        "verifier": verifier,
        "current_observation": observation,
        "current_actions": current_actions,
        "env_steps": session.primitives.metrics.get("env_steps", 0),
        "remaining_env_steps": max(0, session.max_env_steps - int(session.primitives.metrics.get("env_steps", 0))),
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
            scenario_name=task.scenario_name,
            difficulty=task.difficulty,
            seed=task.seed,
            task_type=task.task_type,
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
    get_action_schema,
    get_task_context,
    list_accessible_objects,
    list_inventory,
    list_nearby_objects,
    list_teleport_locations,
    observe_world,
    read_evidence,
    validate_action_call,
    write_evidence,
)

ctx = get_task_context()
obs = observe_world()
objects = list_accessible_objects()
inventory = list_inventory()
schema = get_action_schema()
nearby = list_nearby_objects(max_distance=3)
move_validation = validate_action_call("move_direction", direction="north")
locations = list_teleport_locations()
write_evidence(
    "initial_probe",
    {
        "goal": ctx["goal_text"],
        "location": obs.get("agentLocation"),
        "accessible_objects": objects[:20],
        "inventory": inventory,
        "schema_primitives": sorted(schema.get("primitives", {}))[:20],
        "nearby_objects": nearby[:20],
        "move_validation": move_validation,
        "teleport_locations": list(locations)[:20],
    },
)
print("TASK:", ctx["task_id"])
print("GOAL:", ctx["goal_text"])
print("OBJECTS:", objects[:10])
print("INVENTORY:", inventory[:10])
print("LOCATION:", obs.get("agentLocation"))
print("LOCATIONS:", list(locations)[:10])
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
