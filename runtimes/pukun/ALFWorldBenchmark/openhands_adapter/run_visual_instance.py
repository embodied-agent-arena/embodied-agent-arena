#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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
ROB_ROOT = ROOT.parent
ALFRED_ROOT = ROB_ROOT / "ALFREDOfficialBenchmark"
sys.path.insert(0, str(ROB_ROOT / "scripts"))
sys.path.insert(0, str(ALFRED_ROOT / "src"))

import persistent_repl_runtime as repl_runtime
from openhands_bridge import add_executor_args, build_execution_log_paths, build_executor_command, persist_execution_logs, resolve_or_exit_openhands, run_executor
from alfred_official_adapter.backend import AlfredOfficialBackend
from alfred_official_adapter.primitive_cards import get_primitive_cards
from alfred_official_adapter.primitives import AlfredOfficialPrimitives
from alfred_official_adapter.trace import TraceWriter


DEFAULT_MANIFEST = ROOT / "outputs" / "manifests" / "robench_alfworld_visual_data_v1.json"

ACTION_PRIMITIVES = {
    "approach_object",
    "close_object",
    "explore_room",
    "locate_object",
    "look",
    "move_ahead",
    "open_located_object",
    "open_object",
    "pickup_located_object",
    "pickup_object",
    "place_held_object",
    "put_object",
    "rotate",
    "scan_scene",
    "search_scene",
    "slice_object",
    "toggle_located_object",
    "toggle_object",
}

ALLOWED_PRIMITIVES = {card["name"] for card in get_primitive_cards()}


class AlfredOpenHandsSession:
    def __init__(
        self,
        *,
        task: dict[str, Any],
        suite: str,
        instance_id: str,
        max_env_steps: int,
        max_primitive_calls: int,
        runtime_mode: str = repl_runtime.RUNTIME_MODE,
    ):
        self.task = dict(task)
        self.suite = suite
        self.instance_id = instance_id
        self.max_env_steps = max_env_steps
        self.max_primitive_calls = max_primitive_calls
        self.runtime_mode = runtime_mode
        self.primitive_call_count = 0
        self.primitive_budget_exceeded = False
        self.backend = AlfredOfficialBackend()
        self.trace = TraceWriter(ROOT / "outputs")
        self.backend.reset_task(self.task)
        self.primitives = AlfredOfficialPrimitives(self.backend, self.trace, self.task, suite)
        self.primitives.active_max_env_steps = max_env_steps

    def call(self, primitive: str, args: list[Any], kwargs: dict[str, Any]) -> Any:
        if primitive == "verify_predicate":
            primitive = "check_success"
        if primitive not in ALLOWED_PRIMITIVES:
            raise ValueError(f"Primitive '{primitive}' is not exposed in this OpenHands workspace.")
        self.primitive_call_count += 1
        if self.primitive_call_count > self.max_primitive_calls:
            self.primitive_budget_exceeded = True
            raise RuntimeError(f"Exceeded max_primitive_calls={self.max_primitive_calls}.")
        if primitive in ACTION_PRIMITIVES and self.primitives.metrics["env_steps"] >= self.max_env_steps:
            raise RuntimeError(f"Exceeded max_env_steps={self.max_env_steps}.")
        method = getattr(self.primitives, primitive)
        return _jsonable(method(*args, **kwargs))

    def record_agent_code(self, solve_path: Path) -> None:
        code = solve_path.read_text(encoding="utf-8")
        self.primitives.record_harness_event(
            "agent_code_generated",
            {
                "source": "openhands_workspace_code_cell",
                "code": code,
                "workspace_file": str(solve_path.name),
                "primitive_cards": get_primitive_cards(),
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
        timed_out: bool,
        *,
        source: str,
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
            "final_success": bool(final_verification.get("success")),
            "final_verification": final_verification,
        }
        self.primitives.record_harness_event("code_execution_finished", payload, side_effect=False)
        return payload

    def write_summary(
        self,
        *,
        completed: subprocess.CompletedProcess[str],
        timed_out: bool,
        execution_payload: dict[str, Any],
        workspace_dir: Path,
        suite_task_count: int,
        task_index: int,
        agent_label: str,
        executor_source: str,
        code_attempts: int = 1,
        event_log_paths: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        final = self.backend.check_success()
        stopped_reason = "success" if final.get("success") else "agent_exit_0"
        if timed_out:
            stopped_reason = "code_timeout"
        elif self.primitive_budget_exceeded:
            stopped_reason = "primitive_call_budget"
        elif completed.returncode != 0:
            stopped_reason = "agent_error"

        row = {
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "benchmark": self.task.get("benchmark", "ALFRED"),
            "track": self.task.get("track", "official_visual"),
            "instance_id": self.instance_id,
            "task_id": self.task.get("task_id"),
            "task_type": self.task.get("task_type"),
            "source_split": self.task.get("source_split"),
            "scene": self.task.get("floor_plan"),
            "goal_text": self.task.get("goal_text"),
            "success": bool(final.get("success")),
            "score": float(final.get("score") or 0.0),
            "completed": bool(final.get("completed")),
            "stopped_reason": stopped_reason,
            "exception": execution_payload.get("exception"),
            "env_steps": self.primitives.metrics.get("env_steps", 0),
            "invalid_action_count": self.primitives.metrics.get("invalid_action_count", 0),
            "wrapper_no_match_count": self.primitives.metrics.get("wrapper_no_match_count", 0),
            "wrapper_ambiguity_count": self.primitives.metrics.get("wrapper_ambiguity_count", 0),
            "code_attempts": code_attempts,
            "code_exception_count": int(completed.returncode != 0 and not timed_out and not self.primitive_budget_exceeded),
            "code_timeout_count": int(timed_out),
            "code_primitive_budget_count": int(self.primitive_budget_exceeded),
            "workspace_dir": str(workspace_dir),
            "event_log_paths": event_log_paths or {},
        }
        summary = {
            "run_id": self.trace.run_id,
            "benchmark": row["benchmark"],
            "track": row["track"],
            "agent": agent_label,
            "executor_source": executor_source,
            "suite": self.suite,
            "instance_id": self.instance_id,
            "suite_task_count": suite_task_count,
            "task_index": task_index,
            "start_index": task_index,
            "end_index_exclusive": task_index + 1,
            "is_full_suite_run": False,
            "requested_tasks": 1,
            "completed_tasks": 1,
            "num_tasks": 1,
            "successes": int(row["success"]),
            "success_rate": 1.0 if row["success"] else 0.0,
            "success_by_task_type": _success_by_task_type(row["task_type"], row["success"]),
            "avg_steps": float(row["env_steps"]),
            "avg_score": float(row["score"]),
            "invalid_action_count": row["invalid_action_count"],
            "wrapper_no_match_count": row["wrapper_no_match_count"],
            "wrapper_ambiguity_count": row["wrapper_ambiguity_count"],
            "avg_code_attempts": float(code_attempts),
            "runtime_mode": self.runtime_mode,
            "code_exception_count": row["code_exception_count"],
            "code_timeout_count": row["code_timeout_count"],
            "code_primitive_budget_count": row["code_primitive_budget_count"],
            "timeout_count": int(timed_out or self.primitive_budget_exceeded),
            "exception_count": int(bool(row["exception"]) and not timed_out and not self.primitive_budget_exceeded),
            "workspace_dir": str(workspace_dir),
            "trace_dir": str(self.trace.trace_dir),
            "event_log_paths": event_log_paths or {},
            "boundary": (
                "OpenHands-style local workspace smoke. Official ALFRED trajectory loading, ThorEnv, "
                "goal-condition checker, and private trajectory fields remain behind the primitive server."
            ),
            "tasks": [
                {
                    "task_id": row["task_id"],
                    "task_type": row["task_type"],
                    "source_split": row["source_split"],
                    "scene": row["scene"],
                    "goal_text": row["goal_text"],
                    "suite_index": task_index,
                    "success": row["success"],
                    "env_steps": row["env_steps"],
                    "score": row["score"],
                    "stopped_reason": row["stopped_reason"],
                    "exception": row["exception"],
                    "code_attempts": row["code_attempts"],
                }
            ],
        }
        self.trace.write_report(f"{self.trace.run_id}_summary.json", summary)
        write_task_csv(self.trace.report_dir / f"{self.trace.run_id}_tasks.csv", [row])
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
            with self.server.call_lock:
                result = self.server.session.call(primitive, args, kwargs)
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
    def __init__(self, session: AlfredOpenHandsSession):
        super().__init__(("127.0.0.1", 0), PrimitiveRequestHandler)
        self.session = session
        self.session_token = secrets.token_urlsafe(32)
        self.call_lock = threading.RLock()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ALFWorld Visual OpenHands-style instance.")
    parser.add_argument("--instance-id", default="alfworld_visual_debug_0")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", default=None)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--max-primitive-calls", type=int, default=None)
    parser.add_argument("--code-timeout-seconds", type=int, default=240)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--solution", type=Path, default=None, help="Optional initial cell.py to copy into the workspace.")
    repl_runtime.add_runtime_args(parser)
    add_executor_args(parser)
    args = parser.parse_args()

    instance = load_instance(args.instance_id)
    run_from_parent_args(args, instance)


def run_from_parent_args(args: argparse.Namespace, instance: dict[str, Any]) -> None:
    openhands_binary = resolve_or_exit_openhands(args, benchmark="ALFWorld", track="visual")
    suite = args.suite or str(instance["suite"])
    requested_index = args.task_index if getattr(args, "task_index", None) is not None else getattr(args, "start_index", None)
    task_index = requested_index if requested_index is not None else int(instance["task_index"])
    max_env_steps = args.max_env_steps if args.max_env_steps is not None else int(instance["max_env_steps"])
    max_primitive_calls = args.max_primitive_calls if args.max_primitive_calls is not None else max(80, max_env_steps * 8)
    manifest = args.manifest or DEFAULT_MANIFEST
    task, suite_task_count = select_task(manifest, suite, task_index)

    session = AlfredOpenHandsSession(
        task=task,
        suite=suite,
        instance_id=args.instance_id,
        max_env_steps=max_env_steps,
        max_primitive_calls=max_primitive_calls,
        runtime_mode=getattr(args, "runtime_mode", repl_runtime.RUNTIME_MODE),
    )
    workspace_dir = args.workspace_dir or ROOT / "outputs" / "openhands_workspaces" / session.trace.run_id / args.instance_id
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
    session.primitives.set_public_frame_dir(workspace_dir / "frames")
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
            outputs_dir=ROOT / "outputs",
            benchmark_label="ALFWorld Visual",
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
            timed_out,
            source=executor_source,
            event_log_paths=event_log_paths,
        )
        summary = session.write_summary(
            completed=completed,
            timed_out=timed_out,
            execution_payload=execution_payload,
            workspace_dir=workspace_dir,
            suite_task_count=suite_task_count,
            task_index=task_index,
            agent_label=agent_label,
            executor_source=executor_source,
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
    session: AlfredOpenHandsSession,
    workspace_dir: Path,
    turn_index: int,
    max_turns: int,
    completed: subprocess.CompletedProcess[str],
    timed_out: bool,
) -> dict[str, Any]:
    verifier = session.backend.check_success()
    visual_evidence: dict[str, Any] = {}
    try:
        visual_evidence["frame"] = session.primitives.get_frame(label=f"repl_turn_{turn_index}")
    except Exception as exc:
        visual_evidence["frame_error"] = f"{type(exc).__name__}: {exc}"
    try:
        visual_evidence["visible_objects"] = session.primitives.detect_objects()
    except Exception as exc:
        visual_evidence["visible_objects_error"] = f"{type(exc).__name__}: {exc}"
    payload = {
        "success": bool(verifier.get("success")),
        "terminal": bool(verifier.get("success")) or session.primitive_budget_exceeded,
        "score": float(verifier.get("score") or 0.0),
        "verifier": verifier,
        "env_steps": session.primitives.metrics.get("env_steps", 0),
        "remaining_env_steps": max(0, session.max_env_steps - int(session.primitives.metrics.get("env_steps", 0))),
        "metrics": session.primitives.metrics,
        "evidence": getattr(session.primitives, "evidence", {}),
        "visual_evidence": visual_evidence,
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


def select_task(manifest_path: Path, suite: str, task_index: int) -> tuple[dict[str, Any], int]:
    from task_pool import load_selected_pool
    manifest = load_selected_pool()
    if manifest is None:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    if suite not in manifest:
        raise KeyError(f"Suite not found: {suite}")
    tasks = manifest[suite]
    if task_index < 0 or task_index >= len(tasks):
        raise IndexError(f"task-index {task_index} out of range for suite {suite}")
    return remap_task_paths(dict(tasks[task_index])), len(tasks)


def remap_task_paths(task: dict[str, Any]) -> dict[str, Any]:
    """Map host ALFWorld data paths to the container data volume when requested."""
    data_root = os.environ.get("ALFWORLD_DATA")
    if not data_root:
        return task
    for key in ("traj_path", "gamefile"):
        value = task.get(key)
        if isinstance(value, str) and value:
            task[key] = remap_alfworld_data_path(value, Path(data_root))
    return task


def remap_alfworld_data_path(path_value: str, data_root: Path) -> str:
    path = Path(path_value)
    local_data_root = ROOT / "data"
    try:
        return str(data_root / path.relative_to(local_data_root))
    except ValueError:
        pass

    parts = path.parts
    for index, part in enumerate(parts[:-1]):
        if part == "data" and index > 0 and parts[index - 1] == "ALFWorldBenchmark":
            return str(data_root.joinpath(*parts[index + 1 :]))
    return path_value


def prepare_workspace(
    workspace_dir: Path,
    *,
    instance: dict[str, Any],
    task: dict[str, Any],
    suite: str,
    instance_id: str,
    max_env_steps: int,
    solution: Path | None,
    executor: str,
) -> None:
    repl_runtime.create_workspace(workspace_dir, allowed_root=ROOT / "outputs" / "openhands_workspaces")
    shutil.copy2(ADAPTER_DIR / "visual_primitive_api.py", workspace_dir / "primitive_api.py")
    cards = get_primitive_cards()
    (workspace_dir / "primitive_cards.json").write_text(json.dumps(cards, indent=2, ensure_ascii=False), encoding="utf-8")
    (workspace_dir / "primitive_cards.md").write_text(render_cards_markdown(cards), encoding="utf-8")
    template = (ADAPTER_DIR / "visual_task_template.md").read_text(encoding="utf-8")
    steps = task.get("step_by_step_instructions") or []
    step_text = "\n".join(f"- {step}" for step in steps) if steps else "(none)"
    (workspace_dir / "task.md").write_text(
        template.format(
            benchmark=instance["benchmark"],
            track=instance["track"],
            suite=suite,
            instance_id=instance_id,
            task_id=task.get("task_id"),
            task_type=task.get("task_type"),
            source_split=task.get("source_split"),
            scene=task.get("floor_plan"),
            goal_text=task.get("goal_text"),
            step_by_step_instructions=step_text,
            run_mode_guidance=run_mode_guidance(suite=suite, instance_id=instance_id),
            max_env_steps=max_env_steps,
            boundary=instance["boundary"],
        ),
        encoding="utf-8",
    )
    if solution is not None:
        shutil.copy2(solution, workspace_dir / "cell.py")
    elif executor == "openhands-headless":
        (workspace_dir / "cell.py").write_text(STARTER_SOLVE_PY, encoding="utf-8")
    else:
        (workspace_dir / "cell.py").write_text(PROBE_SOLVE_PY, encoding="utf-8")


def run_mode_guidance(*, suite: str, instance_id: str) -> str:
    if "debug" in suite.lower() or "debug" in instance_id.lower() or "smoke" in suite.lower():
        return (
            "This is a wiring/boundary smoke run. Keep `cell.py` as a concise probe: "
            "import `primitive_api.py`, call a small number of perception/evidence/verifier "
            "primitives, let the harness execute the cell, then finish. Do not attempt a full "
            "visual task solution, do not perform iterative exploration, and do not keep "
            "editing after `check_success()` returns false. A false success check is acceptable "
            "for this smoke run; runtime cleanliness and trace/evaluator plumbing are the target."
        )
    return (
        "This is an evaluation run. Write bounded code that attempts the task using only the "
        "public primitives and the stated environment-step budget. Avoid unbounded exploration "
        "or repeated self-debugging; each revision should be justified by primitive observations."
    )


def render_cards_markdown(cards: list[dict[str, Any]]) -> str:
    lines = [
        "# Primitive Cards",
        "",
        "These are the only primitives exposed for this OpenHands workspace.",
        "",
    ]
    for card in cards:
        arguments = json.dumps(card["arguments"], ensure_ascii=False)
        limitations = "; ".join(card["limitations"])
        lines.extend(
            [
                f"## `{card['name']}`",
                "",
                f"- `signature`: `{card['signature']}`",
                f"- `canonical_family`: `{card['canonical_family']}`",
                f"- `description`: {card['description']}",
                f"- `arguments`: `{arguments}`",
                f"- `returns`: `{card['returns']}`",
                f"- `side_effect`: {card['side_effect']}",
                f"- `leakage_level`: {card['leakage_level']}",
                f"- `limitations`: {limitations}",
                f"- `backend_source`: {card['backend_source']}",
                "",
            ]
        )
    return "\n".join(lines)


def write_task_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_csvable(row) for row in rows)


STARTER_SOLVE_PY = '''"""Write the next bounded code cell for the persistent RoBench session.

Use only primitive_api.py. The harness executes this cell and preserves its globals
for later cells. Do not execute it during authoring.
"""
'''


PROBE_SOLVE_PY = '''from primitive_api import (
    check_success,
    check_progress_public,
    detect_objects,
    get_task_context,
    get_frame,
    inspect_current_view,
    list_actions,
    observe,
    read_evidence,
    read_observed_spatial_map,
    read_search_memory,
    remember_visible_objects,
    write_evidence,
)

ctx = get_task_context()
obs = observe()
frame = get_frame(label="initial_probe")
view = inspect_current_view(label="initial_probe_view")
objects = detect_objects()
memory = remember_visible_objects(label="initial_probe")
search_memory = read_search_memory(limit=20)
spatial_map = read_observed_spatial_map(limit=20)
actions = list_actions()
progress = check_progress_public()
write_evidence(
    "initial_probe",
    {
        "goal": ctx["goal_text"],
        "visible_object_count": len(objects),
        "remembered_count": memory["remembered_count"],
        "inspect_remembered_count": view["remembered_count"],
        "search_memory_count": len(search_memory),
        "map_visited_cells": len(spatial_map.get("visited_cells", [])),
        "map_seen_objects": len(spatial_map.get("seen_objects", [])),
        "map_frontiers": len(spatial_map.get("frontiers", [])),
        "visible_object_types": sorted({obj.get("objectType") for obj in objects})[:20],
        "frame_shape": obs.get("frame_shape"),
        "frame": frame,
        "num_actions": len(actions),
        "action_sample": actions[:12],
        "progress": progress,
    },
)
print("TASK:", ctx["task_id"])
print("GOAL:", ctx["goal_text"])
print("VISIBLE_OBJECTS:", len(objects))
print("FRAME:", frame)
print("MEMORY:", len(search_memory))
print("ACTIONS:", actions[:10])
print("EVIDENCE:", read_evidence())
print("PROGRESS:", progress)
print("VERIFY:", check_success())
'''


def _csvable(row: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (dict, list, tuple)):
            output[key] = json.dumps(value, ensure_ascii=False)
        else:
            output[key] = value
    return output


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
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
