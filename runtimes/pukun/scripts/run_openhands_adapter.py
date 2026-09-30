#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from audit_openhands_event_boundary import audit_event_boundary


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "openhands_adapter_registry.json"
DEEPSEEK_TO_LLM_ENV = {
    "DEEPSEEK_MODEL": "LLM_MODEL",
    "DEEPSEEK_API_KEY": "LLM_API_KEY",
    "DEEPSEEK_BASE_URL": "LLM_BASE_URL",
}
OPENAI_COMPATIBLE_CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
SECRET_VALUE_PATTERNS = [
    re.compile(r"(?<![A-Za-z])sk-[A-Za-z0-9_-]{20,}"),
]


def load_registry(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_instances(entry: dict[str, Any]) -> list[dict[str, Any]]:
    adapter_dir = ROOT / entry["root"] / entry.get("adapter_dir", "openhands_adapter")
    instances_path = adapter_dir / "instances.jsonl"
    rows: list[dict[str, Any]] = []
    with instances_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def find_entry(
    registry: dict[str, Any],
    entry_id: str | None,
    instance_id: str | None,
) -> dict[str, Any]:
    entries = registry.get("entries") or []
    if entry_id:
        for entry in entries:
            if entry.get("entry_id") == entry_id:
                return entry
        raise KeyError(f"Unknown entry_id: {entry_id}")
    if instance_id:
        for entry in entries:
            try:
                ids = {row.get("instance_id") for row in load_instances(entry)}
            except FileNotFoundError:
                continue
            if instance_id in ids:
                return entry
        raise KeyError(f"Could not infer entry for instance_id: {instance_id}")
    raise ValueError("Provide --entry-id or --instance-id.")


def resolve_executable(cwd: Path, value: str) -> str:
    if value.startswith("/") or "/" in value:
        # Do not follow venv interpreter symlinks. On macOS, resolving
        # `.venv/bin/python` to the framework Python bypasses the virtualenv
        # site-packages and makes installed benchmark dependencies disappear.
        candidate = (cwd / value) if not value.startswith("/") else Path(value)
        return str(candidate.absolute())
    return value


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if stripped:
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(obj, dict):
                return obj

    decoder = json.JSONDecoder()
    found: dict[str, Any] | None = None
    found_with_run_id: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found = obj
            if "run_id" in obj:
                found_with_run_id = obj
    return found_with_run_id or found


def command_supports(script_path: Path, argument_name: str) -> bool:
    try:
        text = script_path.read_text(encoding="utf-8")
    except OSError:
        return False
    if argument_name in text:
        return True
    if argument_name in {
        "--runtime-mode",
        "--max-primitive-calls",
        "--max-code-turns",
        "--trial-timeout-seconds",
        "--max-cell-bytes",
        "--max-cell-output-chars",
        "--repl-sandbox",
        "--author-sandbox",
        "--repl-memory-mb",
        "--repl-max-open-files",
        "--repl-max-file-mb",
    }:
        return "repl_runtime.add_runtime_args" in text
    if argument_name in {"--executor", "--openhands-command", "--openhands-extra-arg", "--model", "--codex-executable", "--codex-reasoning-effort", "--cursor-executable", "--max-total-tokens"}:
        return "add_executor_args" in text and "openhands_bridge" in text
    return False


def parse_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def is_secret_key_name(name: str) -> bool:
    upper = name.upper()
    return any(hint in upper for hint in SECRET_KEY_HINTS)


def normalize_deepseek_model_for_litellm(model: str) -> tuple[str, bool]:
    stripped = model.strip()
    if not stripped or "/" in stripped:
        return stripped, False
    return f"deepseek/{stripped}", True


def normalize_openai_compatible_base_url(url: str) -> tuple[str, bool]:
    stripped = url.strip().rstrip("/")
    if stripped.lower().endswith(OPENAI_COMPATIBLE_CHAT_COMPLETIONS_SUFFIX):
        return stripped[: -len(OPENAI_COMPATIBLE_CHAT_COMPLETIONS_SUFFIX)], True
    return stripped, False


def load_dotenv_into_env(env: dict[str, str], path: Path, secret_values: set[str]) -> dict[str, Any]:
    values = parse_dotenv(path)
    loaded: list[str] = []
    skipped_existing: list[str] = []
    for key, value in values.items():
        if key in env:
            if is_secret_key_name(key) and env.get(key):
                secret_values.add(env[key])
            skipped_existing.append(key)
            continue
        env[key] = value
        if is_secret_key_name(key) and value:
            secret_values.add(value)
        loaded.append(key)
    return {
        "path": str(path),
        "loaded_key_names": sorted(loaded),
        "skipped_existing_key_names": sorted(skipped_existing),
    }


def map_deepseek_to_openhands_env(env: dict[str, str], secret_values: set[str]) -> dict[str, str]:
    mapped: dict[str, str] = {}
    for source_key, target_key in DEEPSEEK_TO_LLM_ENV.items():
        if env.get(target_key):
            continue
        source_value = env.get(source_key)
        if source_value:
            target_value = source_value
            transform_notes: list[str] = []
            if source_key == "DEEPSEEK_MODEL":
                target_value, changed = normalize_deepseek_model_for_litellm(source_value)
                if changed:
                    transform_notes.append("provider-prefixed for LiteLLM")
            elif source_key == "DEEPSEEK_BASE_URL":
                target_value, changed = normalize_openai_compatible_base_url(source_value)
                if changed:
                    transform_notes.append("normalized from chat/completions endpoint")
            env[target_key] = target_value
            if is_secret_key_name(source_key) or is_secret_key_name(target_key):
                secret_values.add(source_value)
                secret_values.add(target_value)
            mapped[target_key] = (
                f"{source_key} ({', '.join(transform_notes)})"
                if transform_notes
                else source_key
            )
    return mapped


def collect_secret_values(env: dict[str, str]) -> set[str]:
    values: set[str] = set()
    for key, value in env.items():
        if value and is_secret_key_name(key):
            values.add(value)
    return values


def redact_text(text: str, secret_values: set[str]) -> str:
    redacted = text
    for value in sorted(secret_values, key=len, reverse=True):
        if len(value) < 8:
            continue
        redacted = redacted.replace(value, "[REDACTED_SECRET]")
    for pattern in SECRET_VALUE_PATTERNS:
        redacted = pattern.sub("[REDACTED_SECRET]", redacted)
    return redacted


def redact_jsonable(value: Any, secret_values: set[str]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secret_values)
    if isinstance(value, list):
        return [redact_jsonable(item, secret_values) for item in value]
    if isinstance(value, dict):
        return {key: redact_jsonable(item, secret_values) for key, item in value.items()}
    return value


def build_env(
    *,
    base_env: dict[str, str],
    cwd: Path,
    executor: str | None,
    env_files: list[Path],
    use_benchmark_env: bool,
    map_deepseek_env: bool,
) -> tuple[dict[str, str], dict[str, Any], set[str]]:
    env = dict(base_env)
    secret_values = collect_secret_values(env)
    loaded_files: list[dict[str, Any]] = []
    candidate_files: list[Path] = []
    if executor == "openhands-headless" and use_benchmark_env:
        candidate_files.append(cwd / ".env")
    candidate_files.extend(env_files)

    seen: set[Path] = set()
    missing_files: list[str] = []
    for path in candidate_files:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = (ROOT / resolved).resolve()
        else:
            resolved = resolved.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            missing_files.append(str(resolved))
            continue
        loaded_files.append(load_dotenv_into_env(env, resolved, secret_values))

    mapped = map_deepseek_to_openhands_env(env, secret_values) if map_deepseek_env else {}
    llm_present = {name: bool(env.get(name)) for name in ["LLM_MODEL", "LLM_API_KEY", "LLM_BASE_URL"]}
    return env, {
        "loaded_dotenv_files": loaded_files,
        "missing_dotenv_files": missing_files,
        "deepseek_to_llm_mapped": mapped,
        "llm_env_present": llm_present,
        "note": "Only variable names and mapping sources are reported. Values are never printed.",
    }, secret_values


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def audit_openhands_event_boundaries(
    *,
    workspace: str,
    event_paths: dict[str, Any],
) -> dict[str, Any] | None:
    turns = event_paths.get("turns")
    if isinstance(turns, list) and turns:
        audits: list[dict[str, Any]] = []
        violations: list[Any] = []
        warnings: list[Any] = []
        for turn in turns:
            if not isinstance(turn, dict) or not turn.get("author_stdout"):
                continue
            audit = audit_event_boundary(
                workspace_dir=Path(str(workspace)),
                stdout_path=Path(str(turn["author_stdout"])),
                stderr_path=Path(str(turn["author_stderr"])) if turn.get("author_stderr") else None,
            )
            audit["turn"] = turn.get("turn")
            audits.append(audit)
            violations.extend(audit.get("violations") or [])
            warnings.extend(audit.get("warnings") or [])
        if not audits:
            return None
        return {
            "status": "failed" if violations else "passed",
            "workspace_dir": workspace,
            "turn_count": len(audits),
            "audits": audits,
            "violation_count": len(violations),
            "violations": violations,
            "warning_count": len(warnings),
            "warnings": warnings,
            "policy_note": (
                "Aggregated OpenHands event boundary audit across all multi-turn author logs. "
                "It complements benchmark trace leakage audits; it does not prove OS/container isolation."
            ),
        }
    if event_paths.get("stdout"):
        return audit_event_boundary(
            workspace_dir=Path(str(workspace)),
            stdout_path=Path(str(event_paths["stdout"])),
            stderr_path=Path(str(event_paths["stderr"])) if event_paths.get("stderr") else None,
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one W3 OpenHands-compatible adapter from the root registry.")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--python", default=None, help="Explicit benchmark Python; overrides registry venv paths.")
    parser.add_argument("--entry-id", default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--use-full", action="store_true", help="Use the registry full_instance_id for this entry.")
    parser.add_argument("--suite", default=None, help="Override the adapter suite when supported.")
    parser.add_argument("--start-index", type=int, default=None, help="Override text/symbolic shard start index when supported.")
    parser.add_argument("--task-index", type=int, default=None, help="Override visual shard task index when supported.")
    parser.add_argument("--max-env-steps", type=int, default=None)
    parser.add_argument("--max-primitive-calls", type=int, default=None)
    parser.add_argument("--code-timeout-seconds", type=int, default=None)
    parser.add_argument(
        "--runtime-mode",
        choices=["persistent_repl_code"],
        default=None,
        help="One persistent coding-agent conversation, Python interpreter, and benchmark episode.",
    )
    parser.add_argument("--max-code-turns", type=int, default=None)
    parser.add_argument("--trial-timeout-seconds", type=int, default=None)
    parser.add_argument("--max-cell-bytes", type=int, default=None)
    parser.add_argument("--max-cell-output-chars", type=int, default=None)
    parser.add_argument("--repl-sandbox", choices=["auto", "required", "off"], default=None)
    parser.add_argument("--author-sandbox", choices=["auto", "required", "off"], default=None)
    parser.add_argument("--repl-memory-mb", type=int, default=None)
    parser.add_argument("--repl-max-open-files", type=int, default=None)
    parser.add_argument("--repl-max-file-mb", type=int, default=None)
    parser.add_argument("--solution", type=Path, default=None)
    parser.add_argument(
        "--executor",
        choices=["probe", "openhands-headless", "codex-exec", "cursor-exec", "openai-compatible"],
        default=None,
        help="Forward an adapter executor mode when the target adapter supports it.",
    )
    parser.add_argument("--openhands-command", default=None, help="Forward a custom OpenHands CLI path/name when supported.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--codex-executable", default=None)
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--cursor-executable", default=None)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument(
        "--openhands-extra-arg",
        action="append",
        default=[],
        help="Forward one extra argument to an OpenHands-backed adapter. Repeat for multiple args.",
    )
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument(
        "--strict-event-boundary",
        action="store_true",
        help="Exit non-zero when the OpenHands event log boundary audit reports violations.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        type=Path,
        default=[],
        help="Load a dotenv file into the adapter child process without printing values. Repeatable.",
    )
    parser.add_argument(
        "--no-benchmark-env",
        action="store_true",
        help="Do not auto-load <benchmark-root>/.env for --executor openhands-headless.",
    )
    parser.add_argument(
        "--no-deepseek-llm-map",
        action="store_true",
        help="Do not map DEEPSEEK_MODEL/API_KEY/BASE_URL to OpenHands LLM_MODEL/API_KEY/BASE_URL.",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    registry = load_registry(args.registry.resolve())
    if args.list:
        for entry in registry.get("entries") or []:
            print(
                f"{entry.get('entry_id')}\t{entry.get('status')}\t"
                f"{entry.get('benchmark')} / {entry.get('track')}\t"
                f"default={entry.get('default_instance_id')}"
            )
        return

    entry = find_entry(registry, args.entry_id, args.instance_id)
    if entry.get("status") != "runnable":
        print(
            json.dumps(
                {
                    "entry_id": entry.get("entry_id"),
                    "benchmark": entry.get("benchmark"),
                    "track": entry.get("track"),
                    "status": entry.get("status"),
                    "block_reason": entry.get("block_reason"),
                    "full_v1_status": entry.get("full_v1_status"),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        raise SystemExit(2)

    instance_id = args.instance_id
    if args.use_full:
        instance_id = entry.get("full_instance_id")
    if not instance_id:
        instance_id = entry.get("default_instance_id")
    if not instance_id:
        raise ValueError(f"{entry.get('entry_id')} has no instance_id to run.")

    run_cfg = entry["run"]
    cwd = (ROOT / run_cfg["cwd"]).resolve()
    run_python = resolve_executable(cwd, args.python or run_cfg["python"])
    run_script = Path(run_cfg["script"])
    run_script_abs = cwd / run_script

    command = [run_python, str(run_script), "--instance-id", str(instance_id)]
    if args.suite is not None:
        if not command_supports(run_script_abs, "--suite"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --suite.")
        command.extend(["--suite", str(args.suite)])
    if args.start_index is not None:
        if not command_supports(run_script_abs, "--start-index"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --start-index.")
        command.extend(["--start-index", str(args.start_index)])
    if args.task_index is not None:
        if not command_supports(run_script_abs, "--task-index"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --task-index.")
        command.extend(["--task-index", str(args.task_index)])
    if args.max_env_steps is not None and command_supports(run_script_abs, "--max-env-steps"):
        command.extend(["--max-env-steps", str(args.max_env_steps)])
    if args.max_primitive_calls is not None and command_supports(run_script_abs, "--max-primitive-calls"):
        command.extend(["--max-primitive-calls", str(args.max_primitive_calls)])
    if args.code_timeout_seconds is not None and command_supports(run_script_abs, "--code-timeout-seconds"):
        command.extend(["--code-timeout-seconds", str(args.code_timeout_seconds)])
    if args.runtime_mode is not None:
        if command_supports(run_script_abs, "--runtime-mode"):
            command.extend(["--runtime-mode", args.runtime_mode])
        elif args.runtime_mode != "persistent_repl_code":
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --runtime-mode {args.runtime_mode}.")
    if args.max_code_turns is not None and command_supports(run_script_abs, "--max-code-turns"):
        command.extend(["--max-code-turns", str(args.max_code_turns)])
    for option, value in [
        ("--trial-timeout-seconds", args.trial_timeout_seconds),
        ("--max-cell-bytes", args.max_cell_bytes),
        ("--max-cell-output-chars", args.max_cell_output_chars),
        ("--repl-sandbox", args.repl_sandbox),
        ("--author-sandbox", args.author_sandbox),
        ("--repl-memory-mb", args.repl_memory_mb),
        ("--repl-max-open-files", args.repl_max_open_files),
        ("--repl-max-file-mb", args.repl_max_file_mb),
    ]:
        if value is not None and command_supports(run_script_abs, option):
            command.extend([option, str(value)])
    if args.solution is not None:
        command.extend(["--solution", str(args.solution.resolve())])
    if args.executor is not None:
        if not command_supports(run_script_abs, "--executor"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --executor.")
        command.extend(["--executor", args.executor])
    if args.openhands_command is not None:
        if not command_supports(run_script_abs, "--openhands-command"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --openhands-command.")
        command.extend(["--openhands-command", args.openhands_command])
    for option, value in [
        ("--model", args.model), ("--codex-executable", args.codex_executable),
        ("--codex-reasoning-effort", args.codex_reasoning_effort),
        ("--cursor-executable", args.cursor_executable), ("--max-total-tokens", args.max_total_tokens),
    ]:
        if value is not None:
            if not command_supports(run_script_abs, option):
                raise ValueError(f"{entry.get('entry_id')} adapter does not support {option}")
            command.extend([option, str(value)])
    if args.openhands_extra_arg:
        if not command_supports(run_script_abs, "--openhands-extra-arg"):
            raise ValueError(f"{entry.get('entry_id')} adapter does not support --openhands-extra-arg.")
        for extra_arg in args.openhands_extra_arg:
            command.extend(["--openhands-extra-arg", extra_arg])

    env, env_setup, secret_values = build_env(
        base_env=dict(os.environ),
        cwd=cwd,
        executor=args.executor,
        env_files=args.env_file,
        use_benchmark_env=not args.no_benchmark_env,
        map_deepseek_env=not args.no_deepseek_llm_map,
    )
    run_completed = run_command(command, cwd=cwd, env=env)
    run_json = extract_last_json_object(redact_text(run_completed.stdout, secret_values))
    run_id = run_json.get("run_id") if run_json else None

    eval_completed: subprocess.CompletedProcess[str] | None = None
    eval_json: dict[str, Any] | None = None
    if not args.skip_eval:
        if run_id:
            eval_python = resolve_executable(cwd, args.python or run_cfg["evaluate_python"])
            eval_command = [
                eval_python,
                run_cfg["evaluate_script"],
                "--run-id",
                str(run_id),
            ]
            eval_completed = run_command(eval_command, cwd=cwd, env=env)
            eval_json = extract_last_json_object(redact_text(eval_completed.stdout, secret_values))
        elif run_completed.returncode == 0:
            print(
                json.dumps(
                    {
                        "entry_id": entry.get("entry_id"),
                        "instance_id": instance_id,
                        "status": "run_completed_but_run_id_missing",
                        "stdout_tail": redact_text(run_completed.stdout, secret_values)[-4000:],
                        "stderr_tail": redact_text(run_completed.stderr, secret_values)[-4000:],
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            raise SystemExit(1)

    event_boundary: dict[str, Any] | None = None
    if run_json and args.executor == "openhands-headless":
        workspace = run_json.get("workspace_dir")
        event_paths = run_json.get("event_log_paths") or {}
        if workspace and event_paths:
            event_boundary = audit_openhands_event_boundaries(
                workspace=str(workspace),
                event_paths=event_paths,
            )

    payload = {
        "entry_id": entry.get("entry_id"),
        "benchmark": entry.get("benchmark"),
        "track": entry.get("track"),
        "instance_id": instance_id,
        "runtime_mode": args.runtime_mode or "persistent_repl_code",
        "max_code_turns": args.max_code_turns,
        "run_command": command,
        "run_returncode": run_completed.returncode,
        "run_id": run_id,
        "run_summary": redact_jsonable(run_json, secret_values),
        "eval_returncode": eval_completed.returncode if eval_completed else None,
        "eval_summary": redact_jsonable(eval_json, secret_values),
        "openhands_event_boundary": redact_jsonable(event_boundary, secret_values),
        "env_setup": env_setup,
        "stderr_tail": {
            "run": redact_text(run_completed.stderr, secret_values)[-4000:],
            "eval": redact_text(eval_completed.stderr, secret_values)[-4000:] if eval_completed else "",
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if run_completed.returncode != 0:
        raise SystemExit(run_completed.returncode)
    if eval_completed and eval_completed.returncode != 0:
        raise SystemExit(eval_completed.returncode)
    if args.strict_event_boundary and args.executor == "openhands-headless" and not event_boundary:
        raise SystemExit(1)
    if args.strict_event_boundary and event_boundary and event_boundary.get("violation_count"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
