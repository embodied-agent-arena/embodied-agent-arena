#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shlex
from pathlib import Path
from typing import Any


SECRET_NAME_RE = re.compile(r"(?i)(api[_-]?key|secret|token|password|credential)")
PY_ENV_API_RE = re.compile(r"(?i)(os\.environ|os\.getenv|process\.env)")
ABS_PATH_RE = re.compile(r"(/Users/[^\s\"']+|/home/[^\s\"']+|/private/[^\s\"']+|/tmp/[^\s\"']+)")
FORBIDDEN_COMMAND_RE = re.compile(
    r"(?i)(\.env|DEEPSEEK_API_KEY|LLM_API_KEY|OPENAI_API_KEY|"
    r"ROBENCH_PRIMITIVE_SESSION_TOKEN|traj_data|low_actions|pddl)"
)
PYTHON_COMMANDS = {"python", "python3", "python3.10", "python3.11", "python3.12", "python3.13", "python3.14"}
SENSITIVE_OBSERVATION_RE = re.compile(
    r"(?i)((DEEPSEEK_API_KEY|LLM_API_KEY|OPENAI_API_KEY|ROBENCH_PRIMITIVE_SESSION_TOKEN)\s*=|"
    r"sk-[A-Za-z0-9_-]{20,})"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OpenHands event logs for workspace-boundary violations.")
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--stdout", type=Path, default=None)
    parser.add_argument("--stderr", type=Path, default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    result = audit_event_boundary(
        workspace_dir=args.workspace_dir,
        stdout_path=args.stdout,
        stderr_path=args.stderr,
        summary_path=args.summary,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["violation_count"]:
        raise SystemExit(1)


def audit_event_boundary(
    *,
    workspace_dir: Path,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    summary_path: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace_dir.expanduser().resolve()
    if summary_path and (stdout_path is None or stderr_path is None):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        paths = summary.get("event_log_paths") or {}
        stdout_path = Path(paths["stdout"]) if paths.get("stdout") else stdout_path
        stderr_path = Path(paths["stderr"]) if paths.get("stderr") else stderr_path

    violations: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    event_count = 0
    parsed_stdout_events = 0

    if stdout_path and stdout_path.exists():
        with stdout_path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                event_count += 1
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    scan_text(
                        text=line,
                        source="stdout_unparsed",
                        line_number=line_number,
                        workspace=workspace,
                        violations=violations,
                        warnings=warnings,
                    )
                    continue
                parsed_stdout_events += 1
                audit_event(
                    event=event,
                    source_path=stdout_path,
                    line_number=line_number,
                    workspace=workspace,
                    violations=violations,
                    warnings=warnings,
                )

    if stderr_path and stderr_path.exists():
        text = stderr_path.read_text(encoding="utf-8", errors="replace")
        if text.strip():
            scan_text(
                text=text,
                source="stderr",
                line_number=None,
                workspace=workspace,
                violations=violations,
                warnings=warnings,
            )

    return {
        "status": "passed" if not violations else "failed",
        "workspace_dir": str(workspace),
        "stdout_path": str(stdout_path) if stdout_path else None,
        "stderr_path": str(stderr_path) if stderr_path else None,
        "events": event_count,
        "parsed_stdout_events": parsed_stdout_events,
        "violation_count": len(violations),
        "violations": violations[:100],
        "warning_count": len(warnings),
        "warnings": warnings[:100],
        "policy_note": (
            "OpenHands event boundary audit checks whether agent-visible tool actions or observations "
            "attempt to read outside the prepared workspace, inspect process/env state, or expose "
            "secret-like variable names. It complements benchmark trace leakage audits; it does not "
            "prove OS/container isolation."
        ),
    }


def audit_event(
    *,
    event: dict[str, Any],
    source_path: Path,
    line_number: int,
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    tool = str(event.get("tool_name") or "")
    kind = str(event.get("kind") or "")
    action = event.get("action") if isinstance(event.get("action"), dict) else {}
    observation = event.get("observation") if isinstance(event.get("observation"), dict) else {}
    base = {
        "event_file": str(source_path),
        "line": line_number,
        "tool": tool,
        "event_kind": kind,
    }

    if action:
        audit_action(action, base, workspace, violations, warnings)
    if observation:
        audit_observation(observation, base, workspace, violations, warnings)


def audit_action(
    action: dict[str, Any],
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    kind = str(action.get("kind") or "")
    command = str(action.get("command") or "")
    path_text = str(action.get("path") or "")
    prompt = str(action.get("prompt") or "")

    if path_text:
        audit_path(path_text, {**base, "action_kind": kind, "field": "action.path"}, workspace, violations, warnings)
    if command:
        audit_command(command, {**base, "action_kind": kind}, workspace, violations, warnings)
    if prompt:
        audit_prompt(prompt, {**base, "action_kind": kind}, workspace, violations, warnings)


def audit_observation(
    observation: dict[str, Any],
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    path_text = str(observation.get("path") or "")
    if path_text:
        audit_path(path_text, {**base, "field": "observation.path"}, workspace, violations, warnings)

    metadata = observation.get("metadata")
    if isinstance(metadata, dict):
        working_dir = str(metadata.get("working_dir") or "")
        if working_dir:
            audit_path(working_dir, {**base, "field": "observation.metadata.working_dir"}, workspace, violations, warnings)

    text = json.dumps(observation, ensure_ascii=False)
    scan_text(
        text=text,
        source="observation",
        line_number=base.get("line"),
        workspace=workspace,
        violations=violations,
        warnings=warnings,
        base=base,
    )


def audit_path(
    path_text: str,
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        return
    try:
        resolved = path.resolve()
    except OSError:
        resolved = path.absolute()
    if is_within(resolved, workspace):
        return
    violations.append({**base, "kind": "outside_workspace_path", "value": str(resolved)})


def audit_command(
    command: str,
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    lower = command.lower()
    tokens = shell_tokens(command)
    if shell_invokes(tokens, {"env", "printenv", "set"}) and (
        "grep" in lower or SECRET_NAME_RE.search(command) or "robench" in lower
    ):
        violations.append({**base, "kind": "env_probe_command", "value": compact(command)})
    if PY_ENV_API_RE.search(command) and (
        SECRET_NAME_RE.search(command) or "robench" in lower or "deepseek" in lower or "llm_" in lower
    ):
        violations.append({**base, "kind": "env_probe_command", "value": compact(command)})
    if shell_invokes(tokens, {"ps", "pgrep", "pkill", "killall", "lsof"}):
        violations.append({**base, "kind": "process_probe_command", "value": compact(command)})
    if shell_invokes(tokens, PYTHON_COMMANDS) and ("-c" in tokens or "-" in tokens or "<<" in command):
        violations.append({**base, "kind": "ad_hoc_python_command", "value": compact(command)})
    if "../" in command or "/.." in command:
        violations.append({**base, "kind": "parent_path_command", "value": compact(command)})
    if FORBIDDEN_COMMAND_RE.search(command):
        violations.append({**base, "kind": "forbidden_token_command", "value": compact(command)})
    scan_paths_in_text(command, {**base, "field": "action.command"}, workspace, violations, warnings)


def shell_tokens(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return command.split()


def shell_invokes(tokens: list[str], forbidden: set[str]) -> bool:
    command_boundary = True
    assignment_re = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*")
    for token in tokens:
        bare = token.strip()
        if bare in {"&&", "||", ";", "|"}:
            command_boundary = True
            continue
        if not bare:
            continue
        if command_boundary and assignment_re.match(bare):
            continue
        command_name = Path(bare).name
        if command_boundary and command_name in forbidden:
            return True
        command_boundary = False
    return False


def audit_prompt(
    prompt: str,
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    if FORBIDDEN_COMMAND_RE.search(prompt):
        violations.append({**base, "kind": "forbidden_token_task_prompt", "value": compact(prompt)})
    scan_paths_in_text(prompt, {**base, "field": "action.prompt"}, workspace, violations, warnings)


def scan_text(
    *,
    text: str,
    source: str,
    line_number: int | None,
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
    base: dict[str, Any] | None = None,
) -> None:
    row = dict(base or {})
    row.setdefault("source", source)
    if line_number is not None:
        row.setdefault("line", line_number)
    if SENSITIVE_OBSERVATION_RE.search(text):
        violations.append({**row, "kind": "sensitive_value_or_assignment_observed", "value": compact(text)})
    scan_paths_in_text(text, {**row, "field": source}, workspace, violations, warnings)


def scan_paths_in_text(
    text: str,
    base: dict[str, Any],
    workspace: Path,
    violations: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    for match in ABS_PATH_RE.finditer(text):
        raw = match.group(1).rstrip(",:;)].")
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path.absolute()
        resolved_text = str(resolved)
        workspace_text = str(workspace)
        if is_within(resolved, workspace) or workspace_text.startswith(resolved_text) or resolved_text.startswith(workspace_text):
            continue
        # OpenHands SDK may report its own observation save directory inside
        # terminal observations. That automatic runtime leak is a warning.
        # If the agent actively puts such a path in a tool action, it is reading
        # outside the benchmark workspace and must be a violation.
        if is_openhands_runtime_path(resolved_text) and not is_agent_action_field(base):
            target = warnings
        else:
            target = violations
        target.append({**base, "kind": "outside_workspace_path_in_text", "value": str(resolved)})


def is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def is_openhands_runtime_path(path: str) -> bool:
    return (
        "/.openhands/" in path
        or "/.local/share/uv/tools/openhands/" in path
        or "/site-packages/openhands/" in path
    )


def is_agent_action_field(base: dict[str, Any]) -> bool:
    return str(base.get("field") or "").startswith("action.")


def compact(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."


if __name__ == "__main__":
    main()
