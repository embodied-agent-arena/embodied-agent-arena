"""Model-independent batch campaigns using the existing evaluation scheduler.

The campaign adds durable batch checkpoints and whole-case HTTP recovery; it
does not change the agent loop, episode budgets, native actions or scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import sys
import threading
import time
from urllib.parse import urlsplit

from .full_evaluation import (
    TERMINAL_STATUSES, TaskManifestError, _atomic_write_json,
    _terminate_active_processes, expand_manifest, manifest_hash, run_evaluation,
)
from .native_agent_loop import _load_env_file
from .campaign_history import discover, reusable_results
from .suite_readiness import validate_verified_suite


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--suite", choices=["embodied19-verified", "embodied19-1pct-min10", "embodied19-1pct"], default="embodied19-verified")
    p.add_argument("--manifest", type=Path, help="Custom full_evaluation manifest instead of the built-in suite")
    p.add_argument("--benchmark", action="append", default=[], help="Repeat or comma-separate benchmark IDs")
    p.add_argument("--list", action="store_true", help="List selected benchmarks without running or loading credentials")
    p.add_argument("--provider", choices=["codex-exec", "cursor-exec", "openai-compatible"], default="codex-exec")
    p.add_argument("--model", help="Exact model ID; never substituted automatically")
    p.add_argument("--reasoning-effort", help="Codex CLI reasoning effort (default: high); not sent to API models")
    p.add_argument("--codex-executable")
    p.add_argument("--cursor-executable", help="Cursor Agent CLI executable (default: agent on PATH)")
    p.add_argument("--base-url", help="OpenAI-compatible API base, for example https://api.gpt.ge/v1")
    p.add_argument("--env-file", type=Path, help="Credential/config dotenv; LLM_*, OPENAI_* and GPTGE_* keys accepted")
    p.add_argument("--max-output-tokens", type=int, default=1800)
    p.add_argument("--request-timeout", type=float, default=360, help="Request ceiling; per-case phase limits still apply")
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--workers", type=int, help="Concurrent cases in the shared GPU admission queue")
    p.add_argument("--batch-size", type=int, default=8, help="Cases per checkpointed batch")
    p.add_argument("--gpus", help="Comma-separated physical GPU IDs; empty string for CPU-only")
    p.add_argument("--gpu-memory-gb", type=float)
    p.add_argument("--gpu-reserve-gb", type=float)
    p.add_argument("--gpu-queue-lock", type=Path, help="Separate cooperative queue lock; requires explicit external GPU reservation")
    p.add_argument("--external-gpu-reserve-gb", type=float, default=0, help="Additional per-GPU reservation for independent campaigns")
    p.add_argument("--python", dest="python_executable", help="Case-runner Python (benchmark-specific interpreters remain in the manifest)")
    p.add_argument("--native-artifact-root", type=Path, help="Location of existing sealed native runtime receipts")
    p.add_argument("--runtime-env-file", type=Path, help="Override suite environment file")
    p.add_argument(
        "--allow-unsealed-runtime",
        action="store_true",
        help="Probe only: skip native launch-receipt preflight after declared harness edits",
    )
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--dry-run", action="store_true", help="Plan only: no models, simulators or GPU locks")
    p.add_argument("--resume", action="store_true", help="Require an existing checkpoint (same configuration resumes automatically)")
    p.add_argument("--reuse-from", type=Path, action="append", default=[], help="Additional prior campaign directory or verified history index; repeatable")
    p.add_argument("--no-reuse", action="store_true", help="Disable cross-campaign reuse; same-output checkpoint still resumes")
    p.add_argument("--retry-http", action="store_true", help="Put cases ending in HTTP errors into subsequent rounds")
    p.add_argument("--max-http-rounds", type=int, default=0, help="Recovery rounds after the initial round; 0 = unlimited when --retry-http")
    p.add_argument("--backoff-seconds", type=float, default=30)
    p.add_argument("--max-backoff-seconds", type=float, default=900)
    return p


def load_suite(root: Path, args):
    filename = {'embodied19-verified': 'embodied19_verified.json',
                'embodied19-1pct-min10': 'embodied19_1pct_min10.json',
                'embodied19-1pct': 'embodied19_1pct.json'}[args.suite]
    path = args.manifest or root / 'configs/suites' / filename
    document = json.loads(path.read_text())
    if document.get('readiness') or (not args.manifest and args.suite == 'embodied19-verified'):
        validate_verified_suite(document)
    tasks = [t.to_dict() for t in expand_manifest(document) if t.enabled]
    wanted = {name.strip() for value in args.benchmark for name in value.split(",") if name.strip()}
    known = {t["benchmark_id"] for t in tasks}
    if wanted - known:
        raise ValueError("Unknown benchmarks: " + ", ".join(sorted(wanted - known)))
    if wanted:
        tasks = [t for t in tasks if t["benchmark_id"] in wanted]
    if not tasks:
        raise ValueError("No enabled tasks selected")
    # Reject overrides hidden in task records rather than silently testing a
    # different model. Users of custom manifests must keep model selection here.
    reserved = {"--model", "--model-provider", "--executor", "--codex-executable",
                "--codex-reasoning-effort", "--cursor-executable", "--env-file", "--host-port"}
    for task in tasks:
        if any(a.split("=", 1)[0] in reserved for a in task["runner_args"]):
            raise ValueError("Task contains campaign-owned runner options: " + task["task_id"])
        for key in task["env"]:
            if key.startswith(("LLM_", "GPTGE_", "OPENAI_")) or any(s in key.upper() for s in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")):
                raise ValueError("Task environment contains model settings or credentials: " + key)
    result = copy.deepcopy(document)
    result["tasks"] = tasks
    result.pop("benchmarks", None)
    # Budgets have already been expanded into each task. Leaving inherited
    # runner arguments here would duplicate them when a batch is expanded again.
    for key in ("budgets", "runner_args", "env", "resources"):
        result.pop(key, None)
    result["unavailable_cases"] = {
        b: n for b, n in document.get("unavailable_cases", {}).items()
        if b in {t["benchmark_id"] for t in tasks}
    }
    harness = result.setdefault("harness", {})
    if "port_base" in harness:
        raise ValueError("port_base is not supported by the mixed native/Pukun dispatcher")
    for option, key in [(args.workers, "max_workers"), (args.gpu_memory_gb, "gpu_memory_gb"),
                        (args.gpu_reserve_gb, "gpu_memory_reserve_gb")]:
        if option is not None:
            harness[key] = option
    if args.gpus is not None:
        harness["gpus"] = [g.strip() for g in args.gpus.split(",") if g.strip()]
    harness.setdefault("max_workers", 2)
    harness.setdefault("gpus", ["0"])
    harness.setdefault("gpu_memory_gb", 140)
    harness.setdefault("gpu_memory_reserve_gb", 16)
    external = args.external_gpu_reserve_gb
    if not math.isfinite(external) or external < 0:
        raise ValueError('External GPU reservation must be finite and nonnegative')
    if args.gpu_queue_lock and external <= 0:
        raise ValueError('A separate GPU queue requires --external-gpu-reserve-gb')
    harness['gpu_memory_reserve_gb'] += external
    if args.gpu_queue_lock:
        result['scheduling'] = {'gpu_queue_lock': str(args.gpu_queue_lock.resolve()),
                                'external_gpu_reserve_gb': external}
    harness["respect_existing_gpu_usage"] = True
    return result


def _first(values, names, default=""):
    return next((values[name].strip() for name in names if values.get(name, "").strip()), default)


def model_settings(root: Path, args):
    if not args.model or not args.model.strip():
        raise ValueError("--model is required")
    if args.max_output_tokens < 1 or not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("Output token limit and request timeout must be positive")
    if not math.isfinite(args.temperature) or not 0 <= args.temperature <= 2:
        raise ValueError("Temperature must be between 0 and 2")
    profile = {"provider": args.provider, "model": args.model,
               "max_output_tokens": args.max_output_tokens, "request_timeout": args.request_timeout,
               "temperature": args.temperature}
    env = {"LLM_MAX_TOKENS": str(args.max_output_tokens), "LLM_REQUEST_TIMEOUT_SECONDS": str(args.request_timeout),
           "LLM_TEMPERATURE": str(args.temperature), "LLM_MODEL": args.model}
    common = ["--model-provider", args.provider, "--model", args.model, "--env-file", "/dev/null"]
    if args.provider == "codex-exec":
        if args.base_url or args.env_file:
            raise ValueError("--base-url and --env-file are API options; CLI uses its existing saved login")
        if args.max_output_tokens != 1800 or args.request_timeout != 360 or args.temperature != 0:
            raise ValueError("API generation overrides are unavailable for CLI; CLI uses the frozen runner/phase settings")
        if args.cursor_executable:
            raise ValueError("--cursor-executable is only valid with --provider cursor-exec")
        local = root.parent.parent / ".codex-tools/bin/codex"
        executable = args.codex_executable or shutil.which("codex") or (str(local) if local.exists() else "codex")
        if not args.dry_run and not shutil.which(executable):
            raise ValueError("Codex executable not found; specify --codex-executable")
        effort = args.reasoning_effort or "high"
        profile.update(codex_executable=executable, reasoning_effort=effort)
        common += ["--codex-executable", executable, "--codex-reasoning-effort", effort]
    elif args.provider == "cursor-exec":
        if args.base_url or args.env_file:
            raise ValueError("--base-url and --env-file are API options; CLI uses its existing saved login")
        if args.max_output_tokens != 1800 or args.request_timeout != 360 or args.temperature != 0:
            raise ValueError("API generation overrides are unavailable for CLI; CLI uses the frozen runner/phase settings")
        if args.reasoning_effort or args.codex_executable:
            raise ValueError("Codex reasoning effort/executable are invalid for Cursor CLI; encode effort in the model ID")
        executable = args.cursor_executable or shutil.which("agent") or "agent"
        if not args.dry_run and not shutil.which(executable):
            raise ValueError("Cursor CLI executable not found; specify --cursor-executable")
        profile.update(cursor_executable=executable)
        common += ["--cursor-executable", executable]
    else:
        if args.reasoning_effort or args.codex_executable or args.cursor_executable:
            raise ValueError("Reasoning effort/executable are CLI-only options")
        if args.env_file and not args.env_file.is_file():
            raise ValueError("API dotenv does not exist")
        values = _load_env_file(args.env_file) if args.env_file else {}
        environment = {k: v for k, v in os.environ.items() if k.startswith(("LLM_", "OPENAI_", "GPTGE_"))}
        base_names = ("LLM_BASE_URL", "OPENAI_BASE_URL", "GPTGE_BASE_URL")
        base = args.base_url or _first(values, base_names) or _first(environment, base_names)
        parsed = urlsplit(base)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path.rstrip("/").endswith("/panel"):
            raise ValueError("--base-url must be a credential-free API base URL, not a panel URL")
        base = base.rstrip("/")
        if not parsed.path.strip("/"):
            base += "/v1"
        key_names = ("LLM_API_KEY", "OPENAI_API_KEY", "GPTGE_API_KEY")
        key = _first(values, key_names)
        secret_file = values.get("GPTGE_SECRETS_FILE") or environment.get("GPTGE_SECRETS_FILE")
        if not key and secret_file and not args.dry_run:
            secret = Path(secret_file).expanduser()
            if not secret.is_absolute():
                secret = (args.env_file.parent if args.env_file else root) / secret
            key = _first(_load_env_file(secret), key_names)
        if not key and not secret_file:
            key = _first(environment, key_names)
        if not args.dry_run and not key:
            raise ValueError("Missing API key: configure LLM_API_KEY, OPENAI_API_KEY or GPTGE_API_KEY in --env-file/environment")
        env.update(LLM_API_KEY=key, LLM_BASE_URL=base)
        profile["base_url"] = base
    return profile, env, common


def configure_runtime(root: Path, manifest, args):
    runtime = manifest.setdefault("runtime", {})
    env_file = args.runtime_env_file or runtime.get("env_file")
    if env_file:
        path = Path(str(env_file).replace("{pk_root}", str(root.parent)).replace("{root}", str(root)))
        if not path.is_file():
            raise ValueError("Runtime environment file missing: " + str(path))
        updates = _load_env_file(path)
        # Runtime config is public provenance; never serialize credentials.
        for key in updates:
            if key.startswith(("LLM_", "OPENAI_", "GPTGE_")) or any(s in key.upper() for s in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")):
                raise ValueError("Runtime environment must not contain credentials")
        runtime["env_file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        os.environ.update(updates)
    updates = dict(runtime.get("environment", {}))
    for key in updates:
        if key.startswith(("LLM_", "OPENAI_", "GPTGE_")) or any(s in key.upper() for s in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")):
            raise ValueError("Runtime environment must not contain model settings or credentials")
    if args.native_artifact_root:
        updates["EMBODIED_ARENA_ARTIFACT_ROOT"] = str(args.native_artifact_root.resolve())
    updates["EMBODIED_ARENA_ROOT"] = str(root)
    runtime["environment"] = updates
    os.environ.update(updates)


def runtime_preflight(manifest, *, allow_unsealed_runtime=False):
    from .native_runtime_receipt import validate_native_runtime_launch_receipt
    checks = {}
    selected = {t["benchmark_id"] for t in manifest["tasks"]}
    for benchmark, name in manifest.get("runtime", {}).get("input_locks", {}).items():
        if benchmark not in selected:
            continue
        changed = []
        for filename, digest in json.loads(Path(name).read_text()).items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
                if not allow_unsealed_runtime:
                    raise ValueError("Runtime input lock changed: " + filename)
                changed.append(filename)
        if changed:
            checks[benchmark] = {"input_lock_skipped": True, "changed": changed}
        else:
            checks[benchmark] = {"input_lock_verified": name}
    native = [t for t in manifest["tasks"]
              if not t["benchmark_id"].startswith(("w1_", "w2_", "w5_")) and not any(a == "--entry-id" or a.startswith("--entry-id=") for a in t["runner_args"])]
    seen = set()
    for task in native:
        benchmark = task["benchmark_id"]
        updates = task.get("env", {})
        binding = {k: v for k, v in updates.items()
                   if not k.startswith(("EMBODIED_ARENA_POOL_", "ARENA_CASE_STUDY_"))}
        signature = (benchmark, json.dumps(binding, sort_keys=True))
        if signature in seen:
            continue
        seen.add(signature)
        name = updates.get("ARENA_REPORTING_BENCHMARK", benchmark)
        if allow_unsealed_runtime:
            checks[name] = {"launch_receipt_skipped": True}
            continue
        # The final W4 routes select interpreters and receipts in task.env.
        # Validate the same binding that the case subprocess will receive.
        before = {k: os.environ.get(k) for k in updates}
        try:
            os.environ.update(updates)
            checks[name] = validate_native_runtime_launch_receipt(benchmark)
        finally:
            for key, value in before.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return checks


def apply_case_study_model_gate(manifest, profile):
    """Force case-study recording off for models outside the configured allow-list.

    Recording is normally decided per task by ``ARENA_CASE_STUDY_HZ``, and the
    launchers already derive manifests with the correct flags. This is a second
    gate at the campaign level so a hand-run plan cannot record frames for a
    model that is not meant to have them.

    It deliberately lives here rather than in ``run_benchmark_case.py`` or
    ``case_study_recording.py``: those two files are part of the native runtime
    content seal, so editing them invalidates every benchmark's launch receipt.
    Unset ``ARENA_CASE_STUDY_RECORD_MODELS`` keeps the previous behaviour.
    """
    raw = os.environ.get("ARENA_CASE_STUDY_RECORD_MODELS", "")
    allowed = {item.strip() for item in raw.split(",") if item.strip()}
    if not allowed:
        return 0
    model = str((profile or {}).get("model") or "")
    if model in allowed:
        return 0
    changed = 0
    for task in manifest.get("tasks", []):
        env = task.get("env")
        if isinstance(env, dict) and env.get("ARENA_CASE_STUDY_HZ") != "0":
            env["ARENA_CASE_STUDY_HZ"] = "0"
            changed += 1
    if changed:
        print(
            f"case-study recording disabled for {model!r}: "
            f"{changed} task flag(s) forced to 0 (allowed: {', '.join(sorted(allowed))})"
        )
    return changed


def existing_fable_humanclaw(root, manifest, scheduled):
    """The legacy recovery process predates campaign output locks."""
    if manifest['model_profile']['model'] != 'claude-fable-5-1' or not any(
            t['benchmark_id'] == 'humanclaw' and t['task_id'] in scheduled for t in manifest['tasks']):
        return None
    path = root / 'artifacts/humanclaw-sol-fable-20260914/progress.json'
    try:
        pid = int(json.loads(path.read_text())['pid'])
        command = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0', ' ')
        if 'humanclaw-sol-fable-20260914/' in command:
            return pid
    except (OSError, ValueError, KeyError):
        pass
    return None


def ending_http_codes(status, report):
    """Classify final errors only; never search prompts or generated code."""
    if status.get("outcome") not in {"agent_failure", "runtime_failure", "timeout"}:
        return []
    attempts = report.get("agent_attempts") or []
    node = attempts[-1] if attempts else report
    exception = node.get("exception") or {}
    codes = set()
    code = exception.get("status_code")
    if exception.get("category") == "model_api" and isinstance(code, int) and 400 <= code <= 599:
        codes.add(code)
    episode = (
        node.get("episode_loop")
        or report.get("episode_loop")
        or (node.get("event_log_paths") or {}).get("episode_loop")
        or (report.get("event_log_paths") or {}).get("episode_loop")
        or {}
    )
    turns = episode.get("turns") or []
    messages = [exception.get("message", "")]
    if turns:
        last = turns[-1] or {}
        messages.append(last.get("error") or last.get("author_error") or "")
    for message in messages:
        codes.update(int(c) for c in re.findall(r"Model API HTTP ([45]\d\d)\b", str(message)))
    return sorted(codes)


def status_result(path: Path):
    status = json.loads(path.read_text())
    if status.get("outcome") not in TERMINAL_STATUSES:
        return None
    report_path = Path(status.get("artifacts", {}).get("runner_report", ""))
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    usage = status.get("budget", {}).get("usage", {})
    return {"task_id": status["task_id"], "benchmark": status["task"]["benchmark_id"],
            "outcome": status["outcome"], "http_codes": ending_http_codes(status, report),
            "report_available": report_path.is_file(), "runner_report": str(report_path),
            "status_file": str(path), "total_tokens": usage.get("total_tokens"),
            "tokens_available": usage.get("token_usage_available", False),
            "duration_seconds": status.get("duration_seconds")}


@contextmanager
def try_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("Another process owns this campaign output") from None
        yield


class Campaign:
    def __init__(self, manifest, output, *, runner_script, python, common_args,
                 policy, evaluate=run_evaluation, sleep=time.sleep):
        self.manifest, self.output = manifest, Path(output)
        self.runner_script, self.python = str(runner_script), python
        self.common_args, self.policy = list(common_args), policy
        self.evaluate, self.sleep = evaluate, sleep
        self.digest = manifest_hash(manifest)
        self.tasks = {t["task_id"]: t for t in manifest["tasks"]}
        self.mutex = threading.RLock()
        self.state = {}

    def initialize(self, resume=False, reused=None):
        path = self.output / "checkpoint.json"
        if path.exists():
            self.state = json.loads(path.read_text())
            if self.state["manifest_hash"] != self.digest:
                old = json.loads((self.output / 'effective-manifest.json').read_text())
                old_tasks = {t['task_id']: t for t in old['tasks']}
                compatible = (all(self.tasks.get(k) == t for k, t in old_tasks.items())
                              and {k: v for k, v in old.items() if k not in {'tasks', 'harness'}}
                              == {k: v for k, v in self.manifest.items() if k not in {'tasks', 'harness'}})
                if not compatible or self.state.get('active_batch'):
                    raise ValueError("Resume configuration differs: model, endpoint, suite, budget or recovery settings changed; active batches require their original configuration")
                added = [k for k in self.tasks if k not in old_tasks]
                self.state['pending'].extend(added)
                self.state.update(manifest_hash=self.digest, phase='ready')
                _atomic_write_json(self.output / 'effective-manifest.json', self.manifest)
        else:
            if resume:
                raise ValueError("No campaign checkpoint to resume")
            existing = [p for p in self.output.iterdir() if p.name not in {"campaign.lock", "dry-run", ".last-launch-unix"}] if self.output.exists() else []
            if existing:
                raise ValueError("Output directory is not empty; choose a new campaign directory")
            self.output.mkdir(parents=True, exist_ok=True)
            self.state = {"manifest_hash": self.digest, "round": 0, "batch_index": 0,
                          "pending": list(self.tasks), "next_http": [], "active_batch": None,
                          "latest": {}, "first": {}, "attempts": [], "phase": "ready", "failed_streak": 0}
            reused = dict(reused or {})
            self.state['latest'].update(reused)
            self.state['reused'] = reused
            # Explicit HTTP recovery may rerun imported errors. All other
            # terminal outcomes, including task failures and timeouts, stay put.
            self.state['pending'] = [k for k in self.tasks if k not in reused]
            self.state['next_http'] = [k for k, r in reused.items()
                                       if self.policy['retry_http'] and r['http_codes']]
            if not self.state['pending'] and not self.state['next_http']:
                self.state['phase'] = 'finished'
            _atomic_write_json(self.output / "effective-manifest.json", self.manifest)
        self.state["pid"] = os.getpid()
        self.checkpoint()

    def checkpoint(self):
        with self.mutex:
            _atomic_write_json(self.output / "checkpoint.json", self.state)

    def publish(self):
        with self.mutex:
            latest = dict(self.state["latest"])
            active = []
            batch = self.state.get("active_batch")
            if batch:
                for path in (self.output / batch["directory"] / "statuses").glob("*.json"):
                    try:
                        result = status_result(path)
                        if result:
                            latest[result["task_id"]] = result
                        else:
                            active.append(path.stem)
                    except (OSError, ValueError, KeyError):
                        pass
            counts = Counter(r["outcome"] for r in latest.values())
            http = [k for k, r in latest.items() if r["http_codes"]]
            non_http_errors = [k for k, r in latest.items() if not r["http_codes"] and r["outcome"] in {"agent_failure", "runtime_failure", "timeout", "verifier_failure"}]
            progress = {"phase": self.state["phase"], "pid": self.state["pid"],
                        "updated_at_unix": time.time(), "model": self.manifest["model_profile"],
                        "selected_cases": len(self.tasks), "ended_unique_cases": len(latest),
                        "not_yet_ended": len(self.tasks) - len(latest), "active_tasks": active,
                        "round": self.state["round"], "batch": self.state["batch_index"],
                        "committed_attempts": len(self.state["attempts"]),
                        "reused_cases": len(self.state.get('reused', {})),
                        "latest_http_error_cases": len(http), "non_http_error_cases": len(non_http_errors),
                        "outcomes": dict(counts), "unavailable_cases": self.manifest.get("unavailable_cases", {})}
            _atomic_write_json(self.output / "progress.json", progress)
            _atomic_write_json(self.output / "latest-results.json", latest)
            # Imported entries are baseline observations, not their original
            # first attempts. Preserve that provenance instead of relabelling.
            _atomic_write_json(self.output / "first-results.json", self.state["first"])
            _atomic_write_json(self.output / "reused-results.json", self.state.get('reused', {}))
            _atomic_write_json(self.output / "attempts.json", self.state["attempts"])
            rows = []
            for benchmark in sorted({t["benchmark_id"] for t in self.tasks.values()}):
                selected = sum(t["benchmark_id"] == benchmark for t in self.tasks.values())
                group = [r for r in latest.values() if r["benchmark"] == benchmark]
                c = Counter(r["outcome"] for r in group)
                rows.append({"benchmark": benchmark, "selected": selected, "ended": len(group),
                             "pending_assets": self.manifest.get("unavailable_cases", {}).get(benchmark, 0),
                             **{o: c[o] for o in sorted(TERMINAL_STATUSES)},
                             "http_errors": sum(bool(r["http_codes"]) for r in group)})
            tmp = self.output / ".benchmarks.tmp"
            with tmp.open("w") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
            tmp.replace(self.output / "benchmarks.csv")
            lines = ["# Benchmark campaign", "", f"Model: {self.manifest['model_profile']['model']}", "",
                     f"Phase: {progress['phase']}; ended {len(latest)}/{len(self.tasks)}; HTTP errors {len(http)}; non-HTTP errors {len(non_http_errors)}.", "",
                     "Ended includes failures, errors, timeouts and budget exhaustion. HTTP recovery re-executes the same episode coordinate with the original per-attempt budgets; aggregate attempts/tokens are separate.", "",
                     "| Benchmark | Selected | Ended | Success | Scored | Budget exhausted | Task failure | HTTP errors |",
                     "|---|---:|---:|---:|---:|---:|---:|---:|"]
            for r in rows:
                lines.append(f"| {r['benchmark']} | {r['selected']} | {r['ended']} | {r['success']} | {r['scored']} | {r['budget_exhausted']} | {r['task_failure']} | {r['http_errors']} |")
            tmp = self.output / ".RESULTS.tmp"
            tmp.write_text("\n".join(lines) + "\n"); tmp.replace(self.output / "RESULTS.md")

    def phase(self, name):
        self.state["phase"] = name
        self.checkpoint(); self.publish()

    def run(self):
        """A pinned active batch is reused after interruption; committed batches never rerun."""
        while True:
            if not self.state["pending"] and not self.state["active_batch"]:
                retries = self.state["next_http"]
                if not retries:
                    self.phase("finished")
                    return 0
                limit = self.policy["max_http_rounds"]
                if limit and self.state["round"] >= limit:
                    self.phase("http_retry_limit_reached")
                    return 2
                self.state.update(pending=retries, next_http=[], round=self.state["round"] + 1)
                self.checkpoint()
            if self.state.get("backoff_until", 0) > time.time():
                self.phase("backoff")
                while time.time() < self.state["backoff_until"]:
                    self.sleep(min(10, self.state["backoff_until"] - time.time()))
            if not self.state["active_batch"]:
                ids = self.state["pending"][:self.policy["batch_size"]]
                index = self.state["batch_index"] + 1
                directory = f"batches/{index:06d}"
                batch_manifest = copy.deepcopy(self.manifest)
                batch_manifest["tasks"] = [self.tasks[k] for k in ids]
                _atomic_write_json(self.output / directory / "manifest.json", batch_manifest)
                self.state.update(batch_index=index, active_batch={"ids": ids, "directory": directory})
                self.checkpoint()
            batch = self.state["active_batch"]
            folder = self.output / batch["directory"]
            wave = json.loads((folder / "manifest.json").read_text())
            expected = copy.deepcopy(self.manifest)
            expected["tasks"] = [self.tasks[k] for k in batch["ids"]]
            if wave != expected:
                raise ValueError("Pinned batch manifest changed; refusing to mix results")
            self.phase("running")
            self.evaluate(wave, output_dir=folder, resume=True, python_executable=self.python,
                          runner_script=self.runner_script, common_runner_args=self.common_args)
            results = [status_result(folder / "statuses" / f"{k}.json") for k in batch["ids"]]
            if any(r is None for r in results):
                raise RuntimeError("Batch returned without terminal statuses")
            if any(not r["report_available"] for r in results):
                raise RuntimeError("Runner failed before producing a report; inspect batch stderr before resuming")
            http = [r["task_id"] for r in results if r["http_codes"]]
            with self.mutex:
                for result in results:
                    key = result["task_id"]
                    self.state["latest"][key] = result
                    self.state["first"].setdefault(key, result)
                    self.state["attempts"].append({**result, "round": self.state["round"], "batch": self.state["batch_index"]})
                if self.policy["retry_http"]:
                    self.state["next_http"].extend(http)
                self.state["pending"] = self.state["pending"][len(batch["ids"]):]
                self.state["active_batch"] = None
                self.state["failed_streak"] = self.state["failed_streak"] + 1 if len(http) == len(results) else 0
                if http and self.policy["retry_http"]:
                    delay = min(self.policy["max_backoff_seconds"], self.policy["backoff_seconds"] * 2 ** min(self.state["failed_streak"], 10))
                    self.state["backoff_until"] = time.time() + delay
                self.checkpoint()
            self.publish()
            print(json.dumps({"batch": self.state["batch_index"], "round": self.state["round"],
                              "ended": len(results), "http_errors": len(http), "output": str(self.output)}), flush=True)


def main(root: Path, argv=None):
    args = parser().parse_args(argv)
    campaign = None
    initialized = False
    try:
        if args.batch_size < 1 or args.max_http_rounds < 0:
            raise ValueError("Batch size must be positive; HTTP round limit must be nonnegative")
        if any(not math.isfinite(v) or v < 0 for v in (args.backoff_seconds, args.max_backoff_seconds)):
            raise ValueError("Backoff durations must be finite and nonnegative")
        manifest = load_suite(root, args)
        if args.list:
            listing = {"benchmarks": dict(sorted(Counter(t['benchmark_id'] for t in manifest['tasks']).items())),
                       "selected_cases": len(manifest['tasks']), "unavailable_cases": manifest['unavailable_cases']}
            if manifest.get('readiness'):
                readiness = manifest['readiness']
                selected = {t['task_id'] for t in manifest['tasks']}
                listing['verified_supplements'] = len(selected & set(readiness['supplement_receipts']))
                listing['supplement_gaps'] = {b: n for b, n in readiness['gaps'].items() if b in listing['benchmarks']}
            print(json.dumps(listing, indent=2))
            return 0
        profile, model_env, common = model_settings(root, args)
        if args.allow_unsealed_runtime:
            common.append("--allow-unsealed-runtime")
        configure_runtime(root, manifest, args)
        # Seal-safe second gate: never record case-study frames for a model
        # outside ARENA_CASE_STUDY_RECORD_MODELS.
        apply_case_study_model_gate(manifest, profile)
        # The wrapper is a standalone process: never persist model_env or keys.
        os.environ.update(model_env)
        python = args.python_executable or sys.executable
        policy = {"batch_size": args.batch_size, "retry_http": args.retry_http, "max_http_rounds": args.max_http_rounds,
                  "backoff_seconds": args.backoff_seconds, "max_backoff_seconds": args.max_backoff_seconds}
        manifest.update(model_profile=profile, recovery_policy=policy,
                        execution={"python": python, "dispatcher": str(root / 'scripts/run_benchmark_case.py')})
        digest = manifest_hash(manifest)
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", args.model)[:70]
        output = (args.output_dir or root / "reports/campaigns" / f"{label}-{digest[:12]}").resolve()
        reused, diagnostics = ({}, []) if args.no_reuse else reusable_results(
            manifest, discover(root, output, args.reuse_from))
        reuse_plan = {'selected_cases': len(manifest['tasks']), 'reused_cases': len(reused),
                      'new_cases': len(manifest['tasks']) - len(reused),
                      'imported_http_retries': sum(bool(r['http_codes']) for r in reused.values()) if args.retry_http else 0,
                      'results': reused, 'diagnostics': diagnostics}
        scheduled = {t['task_id'] for t in manifest['tasks'] if t['task_id'] not in reused
                     or (args.retry_http and reused[t['task_id']]['http_codes'])}
        legacy_pid = existing_fable_humanclaw(root, manifest, scheduled)
        reuse_plan['existing_humanclaw_recovery_pid'] = legacy_pid
        if args.dry_run:
            # Keep plans separate from real outputs so a dry-run never corrupts a campaign.
            dry_output = output / "dry-run" / digest[:12]
            planned = copy.deepcopy(manifest)
            planned['tasks'] = [t for t in manifest['tasks'] if t['task_id'] not in reused
                                or (args.retry_http and reused[t['task_id']]['http_codes'])]
            summary = run_evaluation(planned, output_dir=dry_output, dry_run=True,
                                     python_executable=python, runner_script=str(root / 'scripts/run_benchmark_case.py'),
                                     common_runner_args=common) if planned['tasks'] else {'selected_task_count': 0}
            _atomic_write_json(dry_output / "effective-manifest.json", manifest)
            _atomic_write_json(dry_output / "dry-run.json", summary)
            _atomic_write_json(dry_output / "reuse-plan.json", reuse_plan)
            print(json.dumps({"dry_run": True, 'selected_cases': len(manifest['tasks']),
                              'reused_cases': len(reused), 'scheduled_cases': summary['selected_task_count'],
                              'existing_humanclaw_recovery_pid': legacy_pid, "output": str(dry_output)}))
            return 0
        if legacy_pid:
            raise ValueError(f'Fable HumanCLAW recovery is already running (PID {legacy_pid}); select other benchmarks with --benchmark to avoid duplicate evaluations')
        output.mkdir(parents=True, exist_ok=True)
        with try_lock(output / "campaign.lock"):
            campaign = Campaign(manifest, output, runner_script=root / "scripts/run_benchmark_case.py",
                                python=python, common_args=common, policy=policy)
            campaign.initialize(resume=args.resume, reused=reused)
            _atomic_write_json(output / 'reuse-plan.json', reuse_plan)
            initialized = True
            if campaign.state['phase'] == 'finished':
                campaign.publish(); print("Campaign already finished: " + str(output)); return 0
            stop = threading.Event()
            def monitor():
                while not stop.wait(10):
                    campaign.publish()
            def interrupted(signum, frame):
                raise KeyboardInterrupt()
            previous = signal.signal(signal.SIGTERM, interrupted)
            thread = threading.Thread(target=monitor, daemon=True)
            thread.start()
            try:
                # Cooperate with previous dedicated campaigns. No case budget is
                # started until the process obtains the GPU lock and calls the scheduler.
                needs_gpu = any(t['resources'].get('gpu_memory_gb') != 0 for t in manifest['tasks'])
                lock_path = args.gpu_queue_lock or root / ".cache/gptge-batch.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                with lock_path.open('a') as gpu:
                    if needs_gpu:
                        campaign.phase('waiting_for_gpu_queue')
                        while True:
                            try:
                                fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB); break
                            except BlockingIOError:
                                time.sleep(5)
                    campaign.phase('preflight')
                    # A campaign may have completed while this one waited for
                    # the shared queue. Refresh before launching a fresh batch.
                    if not args.no_reuse and not campaign.state['active_batch'] and campaign.state['round'] == 0:
                        refreshed, _ = reusable_results(manifest, discover(root, output, args.reuse_from))
                        imported = {k: r for k, r in refreshed.items() if k in campaign.state['pending']}
                        campaign.state['latest'].update(imported)
                        campaign.state.setdefault('reused', {}).update(imported)
                        campaign.state['pending'] = [k for k in campaign.state['pending'] if k not in imported]
                        if args.retry_http:
                            campaign.state['next_http'].extend(k for k, r in imported.items() if r['http_codes'])
                        campaign.checkpoint()
                    if not campaign.state['pending'] and not campaign.state['active_batch'] and not campaign.state['next_http']:
                        return campaign.run()
                    _atomic_write_json(
                        output / 'runtime-preflight.json',
                        runtime_preflight(manifest, allow_unsealed_runtime=args.allow_unsealed_runtime),
                    )
                    return campaign.run()
            finally:
                stop.set(); thread.join(timeout=15)
                _terminate_active_processes()
                signal.signal(signal.SIGTERM, previous)
    except KeyboardInterrupt:
        if campaign and initialized:
            campaign.phase('interrupted')
        print('Interrupted; rerun the same command with --resume.', file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, TaskManifestError) as exc:
        if campaign and initialized:
            campaign.phase('error')
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2
