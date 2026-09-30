#!/usr/bin/env python3
"""Thin Pukun adapter for run_native_full_evaluation.py; no second scheduler or agent loop."""
import argparse
import hashlib
import json
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path(os.environ.get('AGENTIC_EMBODIED_ARENA_PUKUN_ROOT', Path(__file__).resolve().parents[1] / 'runtimes/pukun'))


def run_adapter(command, root, timeout):
    # The adapter and its worker processes must not survive a timed-out batch task.
    from persistent_repl_runtime import _terminate_descendants
    process = subprocess.Popen(command, cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    previous = signal.getsignal(signal.SIGTERM)
    def interrupted(signum, frame):
        raise KeyboardInterrupt("task cancelled")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    except BaseException:
        _terminate_descendants(process.pid)
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--case-id', required=True)
    p.add_argument('--entry-id', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--agent-trace-root', type=Path, required=True)
    p.add_argument('--pukun-root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--benchmark-python')
    p.add_argument('--suite')
    index = p.add_mutually_exclusive_group()
    index.add_argument('--start-index', type=int)
    index.add_argument('--task-index', type=int)
    p.add_argument('--expected-task-id')
    p.add_argument('--pool-manifest', type=Path)
    p.add_argument('--pool-sha256')
    p.add_argument('--model-provider', choices=['codex-exec', 'cursor-exec', 'openai-compatible'], default='codex-exec')
    p.add_argument('--executor', choices=['codex-exec', 'cursor-exec', 'probe', 'openai-compatible'], default=None)
    p.add_argument('--env-file', type=Path, default=None)
    p.add_argument('--model')
    p.add_argument('--codex-executable', default='codex')
    p.add_argument('--codex-reasoning-effort', default='low')
    p.add_argument('--cursor-executable', default='agent')
    p.add_argument('--allow-unsealed-runtime', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--max-agent-attempts', type=int, default=1)
    p.add_argument('--max-agent-iterations', type=int, default=4)
    p.add_argument('--llm-num-retries', type=int, default=0)
    p.add_argument('--max-total-tokens', type=int, default=120000)
    p.add_argument('--max-cost-usd', type=float)
    p.add_argument('--max-env-steps', type=int, default=50)
    p.add_argument('--max-primitive-calls', type=int, default=220)
    p.add_argument('--trial-timeout-seconds', type=int, default=900)
    p.add_argument('--repl-sandbox', choices=['auto', 'required', 'off'], default='auto')
    a = p.parse_args()
    if a.benchmark_python is None:
        variable = 'EMBODIED_ARENA_NATIVE_PYTHON_' + a.entry_id.upper().replace('-', '_')
        a.benchmark_python = os.environ.get(variable, sys.executable)
    if a.executor is None:
        a.executor = a.model_provider
    elif a.executor != 'probe' and a.executor != a.model_provider:
        p.error('--executor conflicts with --model-provider')
    if a.executor in {'codex-exec', 'cursor-exec'} and not a.model:
        p.error('--model is required for CLI evaluation')
    if a.max_agent_attempts != 1 or a.llm_num_retries != 0 or a.max_cost_usd is not None:
        p.error('Pukun uses one persistent episode, no model retry, token budgets only')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.agent_trace_root.mkdir(parents=True, exist_ok=True)
    report = dict(case_id=a.case_id, benchmark_id=a.entry_id, execution_mode='code_replay' if a.executor == 'probe' else 'persistent_repl_code',
                  official_score_eligible=False)
    try:
        if a.pool_manifest and hashlib.sha256(a.pool_manifest.read_bytes()).hexdigest() != a.pool_sha256:
            raise ValueError('task pool changed after sampling; regenerate manifest')
        root = a.pukun_root.resolve()
        sys.path.insert(0, str(root/'scripts'))
        from run_openhands_adapter import extract_last_json_object
        command = [sys.executable, str(root/'scripts/run_openhands_adapter.py'), '--entry-id', a.entry_id,
            '--python', a.benchmark_python, '--executor', a.executor, '--runtime-mode', 'persistent_repl_code',
            '--max-code-turns', str(a.max_agent_iterations), '--max-total-tokens', str(a.max_total_tokens),
            '--max-env-steps', str(a.max_env_steps), '--max-primitive-calls', str(a.max_primitive_calls),
            '--trial-timeout-seconds', str(a.trial_timeout_seconds), '--repl-sandbox', a.repl_sandbox,
            '--no-benchmark-env', '--skip-eval']
        if a.executor == 'cursor-exec':
            command.extend(['--cursor-executable', a.cursor_executable])
        else:
            command.extend(['--codex-executable', a.codex_executable, '--codex-reasoning-effort', a.codex_reasoning_effort])
        if a.executor in {'codex-exec', 'cursor-exec', 'openai-compatible'}:
            command.append('--use-full')
        if a.env_file is not None:
            command.extend(['--env-file', str(a.env_file.resolve()), '--no-deepseek-llm-map'])
        for option in ('suite', 'start_index', 'task_index', 'model'):
            value = getattr(a, option)
            if value is not None:
                command.extend(['--'+option.replace('_', '-'), str(value)])
        if a.entry_id in {"alfworld_visual", "alfred_official_visual"} and not os.environ.get("DISPLAY"):
            xvfb = shutil.which("xvfb-run")
            if not xvfb:
                raise RuntimeError("visual tracks require DISPLAY or xvfb-run")
            command = [xvfb, "-a", "-s", "-screen 0 1024x768x24 -nolisten tcp", *command]
        completed = run_adapter(command, root, a.trial_timeout_seconds + 120)
        (a.agent_trace_root/'adapter.stdout').write_text(completed.stdout)
        (a.agent_trace_root/'adapter.stderr').write_text(completed.stderr)
        payload = extract_last_json_object(completed.stdout) or {}
        summary = payload.get('run_summary') or {}
        events = summary.get('event_log_paths') or {}
        loop = events.get('episode_loop') or {}
        model = events.get('model') or {}
        usage = model.get('usage') or {}
        actual = [t.get('task_id') for t in summary.get('tasks', [])]
        exhausted = list(dict.fromkeys([*loop.get('budget_exhausted', []),
                                       *usage.get('budget_exhausted', [])]))
        outcome = 'task_failure'
        if a.expected_task_id and actual and actual != [a.expected_task_id]:
            outcome = 'verifier_failure'
        elif loop.get('stop_reason') == 'timeout' or summary.get('timeout_count'):
            outcome = 'timeout'
        elif summary.get('successes') == 1 and summary.get('completed_tasks') == 1 and not completed.returncode:
            outcome = 'success'
        elif exhausted:
            outcome = 'budget_exhausted'
        elif any(t.get('author_error') for t in events.get('turns', [])):
            outcome = 'agent_failure'
        elif completed.returncode or not summary:
            outcome = 'runtime_failure'
        elif a.expected_task_id and actual != [a.expected_task_id]:
            outcome = 'verifier_failure'
        elif summary.get('exception_count') or summary.get('code_exception_count'):
            outcome = 'agent_failure'
        report.update(outcome=outcome, ok=outcome == 'success', llm_usage=usage,
            episode_loop=loop, budget_exhausted=exhausted,
            expected_task_id=a.expected_task_id, actual_task_ids=actual, pukun=payload,
            agent_attempts=[dict(iterations=summary.get('avg_code_attempts', 0))], model=model)
    except subprocess.TimeoutExpired:
        report.update(outcome='timeout', ok=False, error='Pukun adapter timeout')
    except Exception as exc:
        report.update(outcome='runtime_failure', ok=False, error=f'{type(exc).__name__}: {exc}')
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output), outcome=report['outcome']), ensure_ascii=False))
    return 0 if report.get('ok') else 1


if __name__ == '__main__':
    raise SystemExit(main())
