#!/usr/bin/env python3
"""Run one W5 affordance case through the existing model client and episode loop."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
PUKUN = Path(os.environ.get("PUKUN_ROOT", ROOT / "runtimes/pukun"))
sys.path[:0] = [str(ROOT / "src"), str(PUKUN / "scripts")]
import persistent_repl_runtime as runtime
from openhands_bridge import add_executor_args, resolve_or_exit_openhands
from embodied_harness.w5_runtime import BENCHMARKS, W5Session, load_catalog


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/call" or not secrets.compare_digest(
                self.headers.get("Authorization", ""), "Bearer " + self.server.token):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 1000000:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            with self.server.call_lock:
                result = self.server.session.call(request["primitive"], request.get("args", []),
                                                  request.get("kwargs", {}), request.get("client", {}))
            response = dict(ok=True, result=result)
        except Exception as exc:
            response = dict(ok=False, error=f"{type(exc).__name__}: {exc}")
        body = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env-file", type=Path, default=None)
    p.add_argument("--benchmark", choices=list(BENCHMARKS), required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--data-receipt-sha256")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id")
    p.add_argument("--agent-trace-root", type=Path)
    p.add_argument("--solution", type=Path)
    p.add_argument("--code-timeout-seconds", type=int, default=300)
    p.add_argument("--max-env-steps", type=int, default=8)
    p.add_argument("--max-agent-iterations", type=int)
    p.add_argument("--max-agent-attempts", type=int, default=1)
    p.add_argument("--llm-num-retries", type=int, default=0)
    p.add_argument("--max-cost-usd", type=float)
    p.add_argument("--model-provider", choices=["codex-exec", "cursor-exec", "openai-compatible"])
    runtime.add_runtime_args(p, default_max_code_turns=12)
    add_executor_args(p)
    p.add_argument("--allow-unsealed-runtime", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(trial_timeout_seconds=900, max_primitive_calls=160)
    a = p.parse_args()
    if a.max_agent_iterations is not None:
        a.max_code_turns = a.max_agent_iterations
    if a.model_provider:
        a.executor = a.model_provider
    if a.max_agent_attempts != 1 or a.llm_num_retries != 0:
        p.error("W5 uses one persistent episode without automatic model retries")
    if a.max_env_steps != 8:
        p.error("W5 shares the W2 native action budget; keep --max-env-steps 8")
    a.max_model_images = 32
    a.compact_observation_history = True
    resolve_or_exit_openhands(a, benchmark=BENCHMARKS[a.benchmark], track="affordance")
    if a.executor not in {"codex-exec", "cursor-exec", "openai-compatible", "probe"}:
        p.error("Use codex-exec, cursor-exec, openai-compatible, or probe")
    if a.data_receipt_sha256:
        receipt = a.data_root / "download_receipt.json"
        if hashlib.sha256(receipt.read_bytes()).hexdigest() != a.data_receipt_sha256:
            raise ValueError("Data revision receipt changed after sampling; regenerate the manifest")
    catalog = load_catalog(a.data_root)
    if a.sample_id not in catalog:
        raise KeyError(f"Unknown W5 sample: {a.sample_id}")
    case = catalog[a.sample_id]
    if case.get("benchmark") != a.benchmark:
        raise ValueError(f"Sample {a.sample_id} belongs to {case.get('benchmark')}, not {a.benchmark}")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = a.agent_trace_root or a.output.parent / ("w5_" + uuid.uuid4().hex)
    outputs.mkdir(parents=True, exist_ok=True)
    workspace = runtime.create_workspace(outputs / "workspace", allowed_root=outputs)
    transport_source = (PUKUN / "HumanCLAWBenchmark/openhands_adapter/primitive_api.py").read_text()
    transport_source = transport_source.split("\ndef get_task_context():", 1)[0]
    (workspace / "primitive_api.py").write_text(transport_source + '''
def observe(): return _call("observe")
def step(action, **arguments): return _call("step", action=action, arguments=arguments)
def submit(answer, evidence_refs=None, confidence=None):
    return _call("submit", answer=answer, evidence_refs=evidence_refs or [], confidence=confidence)
''')
    (workspace / "primitive_cards.md").write_text('''Use import primitive_api as p.
Only observe / step / submit exist. The only legal action is SUBMIT.
p.observe() returns the public task, attached images and remaining budget.
p.submit(answer, evidence_refs=[], confidence=None) commits the native-format answer once.
Public images are already attached. You may observe and submit in the same first cell.
Attached JPEGs may be downscaled; submit point_2d and bbox_2d in original pixel coordinates
from task_summary.images width/height, origin top-left.
Look at the image and submit a region. Do not write extra vision pipelines.
The native evaluator is private and runs after submission; no correctness feedback is exposed.
''')
    (workspace / "cell.py").write_text(a.solution.read_text() if a.solution else "import primitive_api as p\nprint(p.observe())\n")
    session = server = None
    report = dict(case_id=a.case_id or f"w5_{a.benchmark}:{a.sample_id}",
                  benchmark_id=f"w5_{a.benchmark}", official_score=None, official_score_eligible=True,
                  execution_success=False, task_success=None, verifier_status="not_run",
                  execution_mode="persistent_repl_code",
                  observation_profile=dict(version="w2_public_v2", deduplication="exact_rgb_pixels",
                                           history="latest_observation_with_code_and_notes"))
    try:
        session = W5Session(case, a.data_root, workspace, outputs, outputs.name, executor=a.executor)
        observation = session.feedback_observation()
        (workspace / "task.md").write_text(
            "Solve this spatial task using its native public evidence and answer format. "
            "Images are already attached. Observe if needed, then submit. Python state persists.\n"
            + json.dumps(observation["task_summary"], ensure_ascii=False)
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.session, server.token, server.call_lock = session, secrets.token_urlsafe(32), threading.RLock()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = dict(os.environ, ROBENCH_PRIMITIVE_SERVER_URL=f"http://127.0.0.1:{server.server_port}",
                   ROBENCH_PRIMITIVE_SERVER_TOKEN=server.token)

        def feedback(turn, max_turns, proc, timed_out):
            current = session.feedback_observation()
            remaining = current["remaining_budget"]["environment_actions"]
            return runtime.write_turn_feedback_files(
                session=session, workspace_dir=workspace, turn_index=turn, max_turns=max_turns,
                completed=proc, timed_out=timed_out,
                payload=dict(success=False, terminal=current["terminal"] or remaining <= 0,
                             env_steps=session.steps, remaining_env_steps=remaining,
                             stdout_tail=session.model_stdout(proc.stdout),
                             current_observation=current, verifier={"available_to_agent": False}))

        completed, timed_out, events, _, _, _ = runtime.run_persistent_repl_code(
            args=a, session=session, workspace_dir=workspace, cell_path=workspace / "cell.py",
            openhands_binary=None, env=env, outputs_dir=outputs,
            benchmark_label=BENCHMARKS[a.benchmark], feedback_writer=feedback)
        result = session.finish()
        stop = events["episode_loop"]["stop_reason"]
        report.update(event_log_paths=events, native_result=result, workspace_dir=str(workspace),
                      llm_usage=events.get("model", {}).get("usage", {}),
                      execution_success=completed.returncode == 0 and not timed_out,
                      outcome="timeout" if timed_out else "budget_exhausted" if stop == "budget_exhausted" else "agent_failure")
        if result is not None:
            success = result.get("passed")
            evaluated = result.get("task_outcome") in {"correct", "incorrect"}
            report.update(task_success=success, verifier_status="evaluated" if evaluated else "not_evaluated",
                          official_score=result.get("official_score"),
                          outcome="timeout" if timed_out else "success" if success else "task_failure" if evaluated else "verifier_failure")
        report["ok"] = result is not None and result.get("submission_valid") is True and not timed_out
    except Exception as exc:
        report.update(ok=False, outcome="runtime_failure", error=f"{type(exc).__name__}: {exc}")
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if session:
            session.close()
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("case_id", "outcome", "error", "task_success", "verifier_status")}))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
