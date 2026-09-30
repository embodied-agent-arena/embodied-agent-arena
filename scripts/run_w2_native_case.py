#!/usr/bin/env python3
"""Run one W2 source case through the existing model client, REPL and episode loop."""
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
from embodied_harness.w2_runtime import BENCHMARKS, W2Session


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
    p.add_argument("--perception", choices=["none", "yoloe"], default="none")
    p.add_argument("--yoloe-model", type=Path)
    p.add_argument("--yoloe-sha256")
    p.add_argument("--perception-python", type=Path, default=Path("/usr/local/bin/python"))
    p.add_argument("--perception-device", default="cuda:0")
    p.add_argument("--video-initial-frames", type=int, choices=range(1, 9), default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--case-id")
    p.add_argument("--agent-trace-root", type=Path)
    p.add_argument("--solution", type=Path)
    p.add_argument("--code-timeout-seconds", type=int, default=300)
    p.add_argument("--max-env-steps", type=int, default=8)
    p.add_argument("--max-agent-iterations", type=int)
    p.add_argument("--max-agent-attempts", type=int, default=1)
    p.add_argument("--llm-num-retries", type=int, default=0)
    p.add_argument("--model-provider", choices=["codex-exec", "cursor-exec", "openai-compatible"])
    runtime.add_runtime_args(p, default_max_code_turns=12)
    add_executor_args(p)
    p.add_argument("--allow-unsealed-runtime", action="store_true", help=argparse.SUPPRESS)
    p.set_defaults(trial_timeout_seconds=900, max_primitive_calls=160)
    a = p.parse_args()
    if a.perception == "yoloe" and not (a.yoloe_model and a.yoloe_sha256):
        p.error("YOLOE requires --yoloe-model and --yoloe-sha256")
    if a.max_agent_iterations is not None:
        a.max_code_turns = a.max_agent_iterations
    if a.model_provider:
        a.executor = a.model_provider
    if a.max_agent_attempts != 1 or a.llm_num_retries != 0:
        p.error("W2 uses one persistent episode without automatic model retries")
    a.max_model_images = 32
    a.compact_observation_history = True
    if a.max_env_steps != 8:
        p.error("W2 native action policy has 8 actions; keep --max-env-steps 8")
    resolve_or_exit_openhands(a, benchmark=BENCHMARKS[a.benchmark], track="spatial")
    if a.executor not in {"codex-exec", "cursor-exec", "openai-compatible", "probe"}:
        p.error("Use codex-exec, cursor-exec, openai-compatible, or probe")
    from w2_harness.offline_arena.adapters import create_adapter
    a.output.parent.mkdir(parents=True, exist_ok=True)
    outputs = a.agent_trace_root or a.output.parent / ("w2_" + uuid.uuid4().hex)
    outputs.mkdir(parents=True, exist_ok=True)
    workspace = runtime.create_workspace(outputs / "workspace", allowed_root=outputs)
    # Reuse the transport already used by Pukun; expose only W2 native actions.
    transport_source = (PUKUN / "HumanCLAWBenchmark/openhands_adapter/primitive_api.py").read_text()
    transport_source = transport_source.split("\ndef get_task_context():", 1)[0]
    (workspace / "primitive_api.py").write_text(transport_source + '''
def observe(): return _call("observe")
def step(action, **arguments): return _call("step", action=action, arguments=arguments)
def submit(answer, evidence_refs=None, confidence=None):
    return _call("submit", answer=answer, evidence_refs=evidence_refs or [], confidence=confidence)
''')
    (workspace / "primitive_cards.md").write_text('''Use import primitive_api as p.
p.observe() returns the public task, choices, assets, legal actions and remaining native budget.
p.step(action, **arguments) invokes an action from available_actions verbatim. Action names are uppercase.
OPEN_ASSET/GET_VIEW: asset_id. CROP_REGION: asset_id and region. GET_FRAME: asset_id, frame_index.
The initial public images are attached after your first cell; observing does not require GET_FRAME.
For the first cell, call p.observe() and yield. Do not OPEN_ASSET/GET_VIEW the initial images.
Read attached_images and transport_images first; reopening an attached view wastes a native action.
For video, GET_FRAME requires an explicit frame_index or timestamp_ms, even when using a frame asset as the anchor.
GET_FRAME_WINDOW accepts asset_id and frame_indices (at most eight). For example:
p.step("GET_FRAME_WINDOW", asset_id=obs["visible_assets"][0]["asset_id"], frame_indices=[10, 20, 30])
Choose real indices using public source_frame_index metadata or transport_images bindings.frame_index.
The initial video images sample the full clip uniformly. They may still miss relevant objects.
If a question's objects are absent, request unseen frames between the observed indices with GET_FRAME_WINDOW
and yield to inspect them before answering. An empty YOLOE detection list is not proof that an object is absent.
Do not treat a guessed answer as visually verified. At most 32 distinct images may be attached per turn.
Plan a final submission within the remaining model and native budgets. Each model request is charged
for the full current prompt and attached images, not just the generated code. Reserve one model turn
to inspect newly requested images and submit; do not spend the final turn requesting more images.
The native evidence limit is 32 records, including the initially exposed evidence. Once that limit
is reached, reason from consumed evidence and submit instead of requesting another frame window.
For videos, reserve roughly 60,000 tokens for the final image-bearing request. If the current model
budget has fewer than 120,000 tokens remaining, prefer a submission from the images already received
over another broad search. Target a small number of missing views when further inspection is needed.
This reserve is planning guidance, not an additional action limit. State uncertainty honestly when
the public evidence is incomplete; do not claim unseen objects or unsupported spatial relations.
Read action feedback and repair invalid arguments; never infer image contents from asset names.
p.submit(answer, evidence_refs=[], confidence=None) commits your native-format answer once.
MCQ: choice label. Numeric tasks: a number. RoboSpatial context: native normalized point list.
Use current valid_submission_evidence_refs only for evidence actually consumed.
The native evaluator is private and runs after submission; no correctness feedback is exposed.
''')
    (workspace / "cell.py").write_text(a.solution.read_text() if a.solution else "import primitive_api as p\nprint(p.observe())\n")
    adapter = session = server = perception = None
    report = dict(case_id=a.case_id or f"w2_{a.benchmark}:{a.sample_id}", benchmark_id=f"w2_{a.benchmark}",
                  official_score=None, official_score_eligible=False, execution_success=False,
                  task_success=None, verifier_status="not_run", execution_mode="persistent_repl_code",
                  observation_profile=dict(version="w2_public_v2", deduplication="exact_rgb_pixels",
                                           history="latest_observation_with_code_and_notes",
                                           video_initial_frames=a.video_initial_frames))
    try:
        if a.data_receipt_sha256 and hashlib.sha256((a.data_root / "download_receipt.json").read_bytes()).hexdigest() != a.data_receipt_sha256:
            raise ValueError("Data revision receipt changed after sampling; regenerate the manifest")
        adapter_options = {"sample_allowlist": [a.sample_id], "raw_sample_size": a.video_initial_frames} if a.benchmark == "vsi_bench" else {}
        adapter = create_adapter(BENCHMARKS[a.benchmark], data_root=a.data_root,
                                 cache_root=outputs / "adapter_cache", **adapter_options)
        if a.perception == "yoloe":
            from embodied_harness.w2_perception import YOLOEObservation
            worker = Path(__file__).resolve().with_name("yoloe_worker.py")
            perception = YOLOEObservation(python=a.perception_python, worker=worker,
                model=a.yoloe_model, sha256=a.yoloe_sha256, device=a.perception_device, outputs=outputs)
            with (workspace / "primitive_cards.md").open("a") as f:
                f.write("\nYOLOE automatically parses the currently attached public RGB images after every cell. "
                        "Read current_observation.perception (also p.observe()['perception'] after images arrive). "
                        "Detections reference attached image_index; bbox_xyxy uses that image's pixel coordinates. "
                        "These are fallible estimates, not ground truth or 3D geometry. Keep checking original images. "
                        "No additional primitive is required; native image actions refresh detections next turn.\n")
        session = W2Session(adapter, a.sample_id, workspace, outputs, outputs.name, executor=a.executor,
                            perception=perception)
        observation = session.observe()
        from w2_harness.offline_arena.contracts import action_argument_schema
        with (workspace / "primitive_cards.md").open("a") as f:
            f.write("\nNative action argument schemas:\n" + json.dumps({
                action: action_argument_schema(action) for action in observation["available_actions"]
            }, ensure_ascii=False))
        (workspace / "task.md").write_text("Solve this spatial task using its native public evidence and answer format. "
            "Observe first, then yield for images. Python state persists; use additional media actions if needed.\n"
            + json.dumps(observation["task_summary"], ensure_ascii=False))
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.session, server.token, server.call_lock = session, secrets.token_urlsafe(32), threading.RLock()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        env = dict(os.environ, ROBENCH_PRIMITIVE_SERVER_URL=f"http://127.0.0.1:{server.server_port}",
                   ROBENCH_PRIMITIVE_SERVER_TOKEN=server.token)

        def feedback(turn, max_turns, proc, timed_out):
            current = session.feedback_observation()
            remaining = current["remaining_budget"]["environment_actions"]
            return runtime.write_turn_feedback_files(session=session, workspace_dir=workspace,
                turn_index=turn, max_turns=max_turns, completed=proc, timed_out=timed_out,
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
        if adapter:
            adapter.close()
        if perception:
            report["perception"] = dict(mode=a.perception, **perception.stats)
            perception.close()
        else:
            report["perception"] = dict(mode=a.perception, initialized=False)
        a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report.get(k) for k in ("case_id", "outcome", "error", "task_success", "verifier_status")}))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
