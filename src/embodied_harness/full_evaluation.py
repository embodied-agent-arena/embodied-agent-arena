"""Lightweight, concurrent, resumable orchestration for native evaluations.

The orchestrator uses only the Python standard library. It expands a manifest,
launches one isolated native-loop process per case, accounts budgets, and gates
same-GPU concurrency by declared memory reservations.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from .paths import resolve_project_root

JsonDict = dict[str, Any]
FULL_EVALUATION_CONTRACT = "agentic-embodied-arena-native-full-evaluation/v3"
TASK_STATUS_CONTRACT = "agentic-embodied-arena-native-task-status/v3"
TERMINAL_STATUSES = frozenset(
    {
        "success",
        "scored",
        "task_failure",
        "agent_failure",
        "runtime_failure",
        "timeout",
        "budget_exhausted",
        "skipped",
        "verifier_failure",
    }
)
FAILURE_STATUSES = TERMINAL_STATUSES - {"success", "scored", "skipped"}
_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9_.-]+")
_TASK_BUDGET_FIELDS = frozenset(
    {
        "timeout_seconds",
        "max_agent_attempts",
        "max_agent_iterations",
        "llm_num_retries",
        "max_total_tokens",
        "max_cost_usd",
    }
)
_RUNNER_BUDGET_ARGUMENTS = {
    "max_agent_attempts": "--max-agent-attempts",
    "max_agent_iterations": "--max-agent-iterations",
    "llm_num_retries": "--llm-num-retries",
    "max_total_tokens": "--max-total-tokens",
    "max_cost_usd": "--max-cost-usd",
}
_MANAGED_RUNNER_OPTIONS = frozenset({"--host-port", *_RUNNER_BUDGET_ARGUMENTS.values()})
_TASK_RESOURCE_FIELDS = frozenset({"gpu_memory_gb", "exclusive_gpu"})
_PROCESS_TERMINATION_GRACE_SECONDS = 5.0
_ACTIVE_PROCESS_LOCK = Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen[str]] = set()


class TaskManifestError(ValueError):
    """Raised when a full-evaluation task manifest is malformed."""


@dataclass(frozen=True)
class EvaluationTask:
    benchmark_id: str
    case_id: str
    split: str
    variation: str
    seed: int | str
    task_id: str
    runner_args: tuple[str, ...] = ()
    env: tuple[tuple[str, str], ...] = ()
    budgets: tuple[tuple[str, int | float], ...] = ()
    resources: tuple[tuple[str, int | float | bool], ...] = ()
    enabled: bool = True

    def identity(self) -> JsonDict:
        return {
            "benchmark_id": self.benchmark_id,
            "case_id": self.case_id,
            "split": self.split,
            "variation": self.variation,
            "seed": self.seed,
        }

    def to_dict(self) -> JsonDict:
        return {
            **self.identity(),
            "task_id": self.task_id,
            "runner_args": list(self.runner_args),
            "env": dict(self.env),
            "budgets": dict(self.budgets),
            "resources": dict(self.resources),
            "enabled": self.enabled,
        }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Return a content hash that is independent of JSON whitespace/key order."""

    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def deterministic_run_id(manifest_digest: str) -> str:
    return f"native-full-{manifest_digest[:20]}"


def load_manifest(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TaskManifestError(f"Invalid JSON task manifest {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TaskManifestError("Task manifest root must be a JSON object")
    return payload


def _values(
    item: Mapping[str, Any], plural: str, singular: str, default: Any
) -> list[Any]:
    value = item.get(plural, item.get(singular, default))
    if isinstance(value, list):
        if not value:
            raise TaskManifestError(f"{plural} must not be empty")
        return value
    return [value]


def _string(value: Any, field: str) -> str:
    if value is None or str(value).strip() == "":
        raise TaskManifestError(f"Task field {field!r} is required")
    return str(value)


def _runner_args(*levels: Mapping[str, Any]) -> tuple[str, ...]:
    result: list[str] = []
    for level in levels:
        raw = level.get("runner_args", [])
        if not isinstance(raw, list) or not all(
            isinstance(value, (str, int, float)) for value in raw
        ):
            raise TaskManifestError(
                "runner_args must be a JSON array of scalar command arguments"
            )
        result.extend(str(value) for value in raw)
    return tuple(result)


def _environment(*levels: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    merged: dict[str, str] = {}
    for level in levels:
        raw = level.get("env", {})
        if not isinstance(raw, dict):
            raise TaskManifestError("env must be a JSON object")
        merged.update({str(key): str(value) for key, value in raw.items()})
    return tuple(sorted(merged.items()))


def _positive_number(value: Any, field: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TaskManifestError(f"budget {field!r} must be a positive number")
    return value


def _budgets(*levels: Mapping[str, Any]) -> tuple[tuple[str, int | float], ...]:
    merged: dict[str, int | float] = {}
    for level in levels:
        raw = level.get("budgets", {})
        if not isinstance(raw, dict):
            raise TaskManifestError("budgets must be a JSON object")
        unknown = sorted(set(raw) - _TASK_BUDGET_FIELDS)
        if unknown:
            raise TaskManifestError(f"Unknown task budget fields: {unknown}")
        for key, value in raw.items():
            if key == "llm_num_retries":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 0 <= value <= 9
                ):
                    raise TaskManifestError(
                        "budget 'llm_num_retries' must be an integer between 0 and 9"
                    )
            elif key == "max_agent_iterations":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not 1 <= value <= 500
                ):
                    raise TaskManifestError(
                        "budget 'max_agent_iterations' must be an integer between 1 and 500"
                    )
            elif key in {"max_agent_attempts", "max_total_tokens"}:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise TaskManifestError(
                        f"budget {key!r} must be a positive integer"
                    )
            else:
                _positive_number(value, key)
            merged[key] = value
    return tuple(sorted(merged.items()))


def _resources(
    *levels: Mapping[str, Any],
) -> tuple[tuple[str, int | float | bool], ...]:
    merged: dict[str, int | float | bool] = {}
    for level in levels:
        raw = level.get("resources", {})
        if not isinstance(raw, dict):
            raise TaskManifestError("resources must be a JSON object")
        unknown = sorted(set(raw) - _TASK_RESOURCE_FIELDS)
        if unknown:
            raise TaskManifestError(f"Unknown task resource fields: {unknown}")
        for key, value in raw.items():
            if key == "exclusive_gpu":
                if not isinstance(value, bool):
                    raise TaskManifestError("resource 'exclusive_gpu' must be boolean")
            elif (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or value < 0
            ):
                raise TaskManifestError(
                    "resource 'gpu_memory_gb' must be a non-negative number"
                )
            merged[key] = value
    if merged.get("exclusive_gpu") is True and merged.get("gpu_memory_gb") == 0:
        raise TaskManifestError(
            "exclusive_gpu=true cannot be combined with gpu_memory_gb=0"
        )
    return tuple(sorted(merged.items()))


def _slug(value: Any) -> str:
    cleaned = _SAFE_COMPONENT.sub("-", str(value)).strip("-._")
    return cleaned[:48] or "default"


def _make_task(
    *,
    benchmark_id: Any,
    case_id: Any,
    split: Any,
    variation: Any,
    seed: Any,
    runner_args: tuple[str, ...],
    env: tuple[tuple[str, str], ...],
    budgets: tuple[tuple[str, int | float], ...],
    resources: tuple[tuple[str, int | float | bool], ...],
    enabled: bool,
) -> EvaluationTask:
    if seed is None:
        # Existing representative-case manifests use JSON null when the
        # benchmark owns reset seeding internally.  Keep that dimension
        # explicit and deterministic instead of rejecting the manifest.
        seed = "default"
    identity = {
        "benchmark_id": _string(benchmark_id, "benchmark_id"),
        "case_id": _string(case_id, "case_id"),
        "split": _string(split, "split"),
        "variation": _string(variation, "variation"),
        "seed": seed,
    }
    if isinstance(seed, bool) or not isinstance(seed, (int, str)):
        raise TaskManifestError("seed must be an integer, string, or null")
    suffix = hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()[:12]
    task_id = "--".join(
        (
            _slug(identity["benchmark_id"]),
            _slug(identity["case_id"]),
            _slug(identity["split"]),
            _slug(identity["variation"]),
            _slug(identity["seed"]),
            suffix,
        )
    )
    return EvaluationTask(
        **identity,
        task_id=task_id,
        runner_args=runner_args,
        env=env,
        budgets=budgets,
        resources=resources,
        enabled=enabled,
    )


def _expand_record(
    record: Mapping[str, Any], *parents: Mapping[str, Any]
) -> list[EvaluationTask]:
    levels = (*parents, record)
    benchmark_id = next(
        (
            level.get("benchmark_id")
            for level in reversed(levels)
            if level.get("benchmark_id") is not None
        ),
        None,
    )
    case_id = record.get("case_id", record.get("task_id"))
    splits = _values(
        record,
        "splits",
        "split",
        next(
            (
                p.get("splits", p.get("split"))
                for p in reversed(parents)
                if p.get("splits", p.get("split")) is not None
            ),
            "default",
        ),
    )
    variations = _values(
        record,
        "variations",
        "variation",
        next(
            (
                p.get("variations", p.get("variation"))
                for p in reversed(parents)
                if p.get("variations", p.get("variation")) is not None
            ),
            "default",
        ),
    )
    seeds = _values(
        record,
        "seeds",
        "seed",
        next(
            (
                p.get("seeds", p.get("seed"))
                for p in reversed(parents)
                if p.get("seeds", p.get("seed")) is not None
            ),
            0,
        ),
    )
    enabled = all(bool(level.get("enabled", True)) for level in levels)
    return [
        _make_task(
            benchmark_id=benchmark_id,
            case_id=case_id,
            split=split,
            variation=variation,
            seed=seed,
            runner_args=_runner_args(*levels),
            env=_environment(*levels),
            budgets=_budgets(*levels),
            resources=_resources(*levels),
            enabled=enabled,
        )
        for split in splits
        for variation in variations
        for seed in seeds
    ]


def expand_manifest(manifest: Mapping[str, Any]) -> list[EvaluationTask]:
    """Expand supported manifest shapes into a stable, de-duplicated task list."""

    expanded: list[EvaluationTask] = []
    top_tasks = manifest.get("tasks")
    if top_tasks is not None:
        if not isinstance(top_tasks, list):
            raise TaskManifestError("tasks must be a JSON array")
        for record in top_tasks:
            if not isinstance(record, dict):
                raise TaskManifestError("Each task must be a JSON object")
            expanded.extend(_expand_record(record, manifest))
    else:
        benchmarks = manifest.get("benchmarks")
        if not isinstance(benchmarks, list):
            raise TaskManifestError("Manifest must contain a tasks or benchmarks array")
        for benchmark in benchmarks:
            if not isinstance(benchmark, dict):
                raise TaskManifestError("Each benchmark must be a JSON object")
            cases = benchmark.get("cases", benchmark.get("representative_cases"))
            if not isinstance(cases, list):
                raise TaskManifestError(
                    "Each benchmark must contain cases or representative_cases"
                )
            for case in cases:
                if not isinstance(case, dict):
                    raise TaskManifestError("Each case must be a JSON object")
                expanded.extend(_expand_record(case, manifest, benchmark))
    by_identity: dict[str, EvaluationTask] = {}
    for task in expanded:
        identity = _canonical_json(task.identity())
        if identity in by_identity:
            raise TaskManifestError(
                f"Duplicate expanded task identity: {task.identity()}"
            )
        by_identity[identity] = task
    return sorted(
        by_identity.values(), key=lambda task: _canonical_json(task.identity())
    )


def select_tasks(
    tasks: Iterable[EvaluationTask],
    *,
    benchmark_ids: Sequence[str] = (),
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[EvaluationTask]:
    if shard_count < 1:
        raise TaskManifestError("shard_count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise TaskManifestError(
            "shard_index must satisfy 0 <= shard_index < shard_count"
        )
    wanted = set(benchmark_ids)
    available = {task.benchmark_id for task in tasks}
    missing = sorted(wanted - available)
    if missing:
        raise TaskManifestError(f"Unknown benchmark ids: {missing}")
    filtered = [task for task in tasks if not wanted or task.benchmark_id in wanted]
    return [
        task
        for index, task in enumerate(filtered)
        if index % shard_count == shard_index
    ]


def build_task_command(
    task: EvaluationTask,
    *,
    output_dir: Path,
    python_executable: str = sys.executable,
    runner_script: str = "scripts/run_native_universal_case.py",
    common_runner_args: Sequence[str] = (),
    host_port: int | None = None,
    budgets: Mapping[str, int | float] | None = None,
) -> list[str]:
    task_dir = output_dir / "tasks" / task.task_id
    forwarded = [*common_runner_args, *task.runner_args]
    managed = {
        option
        for argument in forwarded
        for option in _MANAGED_RUNNER_OPTIONS
        if argument == option or argument.startswith(f"{option}=")
    }
    requested_managed = {
        *({"--host-port"} if host_port is not None else set()),
        *{
            option
            for field, option in _RUNNER_BUDGET_ARGUMENTS.items()
            if budgets is not None and field in budgets
        },
    }
    conflicts = sorted(managed & requested_managed)
    if conflicts:
        raise TaskManifestError(
            "Structured scheduler/budget settings conflict with runner_args: "
            f"{conflicts}"
        )
    command = [
        python_executable,
        runner_script,
        "--case-id",
        task.case_id,
        "--output",
        str(task_dir / "runner_report.json"),
        "--agent-trace-root",
        str(task_dir / "agent_trace"),
        *forwarded,
    ]
    if host_port is not None:
        command.extend(["--host-port", str(host_port)])
    for field, option in _RUNNER_BUDGET_ARGUMENTS.items():
        if budgets is not None and field in budgets:
            command.extend([option, str(budgets[field])])
    return command


def _json_payload(text: str) -> JsonDict:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value or "")


def _run_process(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    capture_output: bool,
    text: bool,
    timeout: float | None,
    check: bool,
) -> subprocess.CompletedProcess[str]:
    """Run one case with a killable process group and a short cleanup window."""

    if not capture_output or not text:
        raise ValueError(
            "lightweight harness process runner requires capture_output=True and text=True"
        )
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(env),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=(os.name == "posix"),
    )
    with _ACTIVE_PROCESS_LOCK:
        _ACTIVE_PROCESSES.add(process)
    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = _terminate_process(process)
            raise subprocess.TimeoutExpired(
                list(command),
                timeout,
                output=stdout,
                stderr=stderr,
            )
        except BaseException:
            _terminate_process(process)
            raise
    finally:
        with _ACTIVE_PROCESS_LOCK:
            _ACTIVE_PROCESSES.discard(process)
    completed = subprocess.CompletedProcess(
        list(command),
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    if check:
        completed.check_returncode()
    return completed


def _terminate_process(process: subprocess.Popen[str]) -> tuple[str, str]:
    if process.poll() is None:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - the maintained runtime is Linux.
            process.terminate()
    try:
        return process.communicate(timeout=_PROCESS_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - the maintained runtime is Linux.
            process.kill()
        return process.communicate()


def _terminate_active_processes() -> None:
    with _ACTIVE_PROCESS_LOCK:
        active = list(_ACTIVE_PROCESSES)
    for process in active:
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:  # pragma: no cover - the maintained runtime is Linux.
                process.terminate()
    deadline = time.monotonic() + _PROCESS_TERMINATION_GRACE_SECONDS
    while (
        any(process.poll() is None for process in active)
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    for process in active:
        if process.poll() is not None:
            continue
        if os.name == "posix":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        else:  # pragma: no cover - the maintained runtime is Linux.
            process.kill()


def _gateway_budget_exhausted(payload: Mapping[str, Any]) -> list[str]:
    exhausted: list[str] = []
    raw_attempts = payload.get("agent_attempts")
    if not isinstance(raw_attempts, list):
        return exhausted
    for attempt in raw_attempts:
        if not isinstance(attempt, dict):
            continue
        raw_names = attempt.get("budget_exhausted")
        if not isinstance(raw_names, list):
            continue
        # A gateway marks its last permitted call as exhausted (used == limit).
        # Successful verification on that call is still within the budget.
        turns = attempt.get("turns") or []
        last_turn = turns[-1] if turns and isinstance(turns[-1], dict) else {}
        snapshot = last_turn.get("gateway_budget") or {}
        limits = snapshot.get("limits") or {}
        used = snapshot.get("used") or {}
        successful_final_call = (
            attempt.get("success") is True
            and not attempt.get("exception")
            and (attempt.get("episode_loop") or {}).get("stop_reason") == "success"
            and last_turn.get("execution_ok") is True
            and last_turn.get("harness_verifier_attempted") is True
            and last_turn.get("harness_verifier_ok") is True
        )
        for name in raw_names:
            normalized = str(name).strip()
            if (successful_final_call and normalized in limits
                    and used.get(normalized) == limits[normalized]):
                continue
            if normalized and normalized not in exhausted:
                exhausted.append(normalized)
    return exhausted


def classify_result(
    *, returncode: int, payload: Mapping[str, Any], timed_out: bool = False
) -> str:
    """Map runner/process outcomes onto the stable full-evaluation taxonomy."""

    if timed_out:
        return "timeout"
    explicit = payload.get("outcome", payload.get("classification"))
    if explicit in TERMINAL_STATUSES:
        return str(explicit)
    if _gateway_budget_exhausted(payload):
        return "budget_exhausted"
    stage = str(payload.get("stage", "")).lower()
    status = str(payload.get("status", "")).lower()
    blocker = payload.get("blocker") if isinstance(payload.get("blocker"), dict) else {}
    blocker_type = str(blocker.get("type", "")).lower()
    artifacts = (
        payload.get("agent_trace_artifacts")
        if isinstance(payload.get("agent_trace_artifacts"), dict)
        else {}
    )
    if (
        "verif" in stage
        or "verif" in blocker_type
        or artifacts.get("summary_success") is False
    ):
        return "verifier_failure"
    if (
        "agent" in stage
        or "agenttrace" in blocker_type
        or "traceartifacts" in blocker_type
    ):
        return "agent_failure"
    if status in {"incomplete", "failed", "failure", "task_failure"}:
        return "task_failure"
    if returncode != 0:
        return "runtime_failure"
    if payload.get("ok") is True:
        return "success"
    if status in {"blocked", "error", "exception"} or blocker:
        return "runtime_failure"
    return "runtime_failure"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_attempt_log(*, attempt_path: Path, latest_path: Path, content: str) -> None:
    """Write log bytes once and expose the latest attempt through a hard link."""

    attempt_path.write_text(content, encoding="utf-8")
    temporary = latest_path.with_name(
        f".{latest_path.name}.{os.getpid()}.{time.time_ns()}"
    )
    try:
        os.link(attempt_path, temporary)
    except OSError:
        temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, latest_path)


def status_path(output_dir: Path, task: EvaluationTask) -> Path:
    return output_dir / "statuses" / f"{task.task_id}.json"


def read_status(path: Path) -> JsonDict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True, slots=True)
class WorkerSlot:
    slot_id: int
    host_port: int
    gpu_id: str | None = None
    gpu_memory_requested_gb: float = 0.0
    gpu_memory_reserved_gb: float = 0.0
    exclusive_gpu: bool = False

    def to_dict(self) -> JsonDict:
        return {
            "slot_id": self.slot_id,
            "host_port": self.host_port,
            "gpu_id": self.gpu_id,
            "gpu_memory_requested_gb": self.gpu_memory_requested_gb,
            "gpu_memory_reserved_gb": self.gpu_memory_reserved_gb,
            "exclusive_gpu": self.exclusive_gpu,
        }

    def allocated(
        self,
        *,
        gpu_id: str | None,
        requested_gb: float,
        reserved_gb: float,
        exclusive: bool,
    ) -> "WorkerSlot":
        return WorkerSlot(
            slot_id=self.slot_id,
            host_port=self.host_port,
            gpu_id=gpu_id,
            gpu_memory_requested_gb=requested_gb,
            gpu_memory_reserved_gb=reserved_gb,
            exclusive_gpu=exclusive,
        )


def _task_resource_request(task: EvaluationTask) -> tuple[float | None, bool]:
    resources = dict(task.resources)
    raw_memory = resources.get("gpu_memory_gb")
    memory = float(raw_memory) if isinstance(raw_memory, (int, float)) else None
    return memory, resources.get("exclusive_gpu") is True


def _task_budgets(
    task: EvaluationTask,
    *,
    timeout_seconds: float | None,
    overrides: Mapping[str, int | float] | None,
) -> dict[str, int | float]:
    budgets = dict(task.budgets)
    if timeout_seconds is not None:
        budgets["timeout_seconds"] = _positive_number(
            timeout_seconds, "timeout_seconds"
        )
    if overrides:
        budgets.update(dict(_budgets({"budgets": dict(overrides)})))
    return budgets


def _worker_slots(
    *,
    max_workers: int,
    port_base: int,
    gpu_ids: Sequence[str | int],
) -> list[WorkerSlot]:
    if isinstance(max_workers, bool) or max_workers < 1:
        raise TaskManifestError("max_workers must be >= 1")
    if isinstance(port_base, bool) or not isinstance(port_base, int):
        raise TaskManifestError("port_base must be an integer")
    if not 1 <= port_base <= 65535 or port_base + max_workers - 1 > 65535:
        raise TaskManifestError("worker host-port range must stay between 1 and 65535")
    normalized_gpus: list[str] = []
    for value in gpu_ids:
        gpu = str(value).strip()
        if not gpu or any(character.isspace() for character in gpu) or "," in gpu:
            raise TaskManifestError(
                "each GPU id must be one non-empty value without whitespace or commas"
            )
        normalized_gpus.append(gpu)
    return [
        WorkerSlot(
            slot_id=index,
            host_port=port_base + index,
            gpu_id=(
                normalized_gpus[index % len(normalized_gpus)]
                if normalized_gpus
                else None
            ),
        )
        for index in range(max_workers)
    ]


def _gpu_memory_inventory() -> list[JsonDict]:
    """Return a one-shot physical GPU memory snapshot from nvidia-smi."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if completed.returncode:
        return []
    result: list[JsonDict] = []
    for raw_line in completed.stdout.splitlines():
        parts = [item.strip() for item in raw_line.split(",")]
        if len(parts) != 4:
            continue
        try:
            total_gib = int(parts[2]) / 1024.0
            free_gib = int(parts[3]) / 1024.0
        except ValueError:
            continue
        result.append(
            {
                "index": parts[0],
                "uuid": parts[1],
                "memory_total_gib": round(total_gib, 6),
                "memory_free_gib": round(free_gib, 6),
            }
        )
    return result


def _selected_gpu_memory(
    gpu_ids: Sequence[str | int], inventory: Sequence[Mapping[str, Any]]
) -> dict[str, JsonDict]:
    selected: dict[str, JsonDict] = {}
    for raw_gpu in gpu_ids:
        gpu = str(raw_gpu)
        match = next(
            (
                dict(item)
                for item in inventory
                if str(item.get("index")) == gpu or str(item.get("uuid")) == gpu
            ),
            None,
        )
        if match is not None:
            selected[gpu] = match
    return selected


def _usage_from_payload(
    payload: Mapping[str, Any], *, duration_seconds: float
) -> JsonDict:
    raw_usage = payload.get("llm_usage")
    llm_usage = raw_usage if isinstance(raw_usage, dict) else {}
    raw_attempts = payload.get("agent_attempts")
    usage_mode = str(llm_usage.get("usage_mode") or "none")
    code_replay = payload.get("execution_mode") == "code_replay"
    return {
        "wall_time_seconds": round(duration_seconds, 6),
        "agent_attempts": len(raw_attempts) if isinstance(raw_attempts, list) else 0,
        "model_responses": _as_int(llm_usage.get("response_count")),
        "prompt_tokens": _as_int(llm_usage.get("prompt_tokens")),
        "completion_tokens": _as_int(llm_usage.get("completion_tokens")),
        "total_tokens": _as_int(llm_usage.get("total_tokens")),
        "cost_usd": _as_float(llm_usage.get("accumulated_cost")),
        "usage_mode": usage_mode,
        "token_usage_available": (
            llm_usage.get("token_usage_available") is True
            or llm_usage.get("available") is True
            or usage_mode not in {"", "none"}
            or code_replay
        ),
        "provider_usage_available": (
            llm_usage.get("provider_usage_available") is True
            or llm_usage.get("available") is True
        ),
        "cost_available": (
            llm_usage.get("cost_available") is True
            or llm_usage.get("available") is True
            or code_replay
        ),
        "public_gateway_budget_exhausted": _gateway_budget_exhausted(payload),
    }


def _budget_report(
    limits: Mapping[str, int | float],
    usage: Mapping[str, Any],
) -> JsonDict:
    exhausted: list[str] = []
    unverifiable: list[str] = []
    max_tokens = limits.get("max_total_tokens")
    token_usage_available = usage.get("token_usage_available") is True
    if max_tokens is not None:
        if not token_usage_available:
            unverifiable.append("max_total_tokens")
        elif _as_int(usage.get("total_tokens")) > _as_int(max_tokens):
            exhausted.append("max_total_tokens")
    max_cost = limits.get("max_cost_usd")
    if max_cost is not None:
        if usage.get("cost_available") is not True:
            unverifiable.append("max_cost_usd")
        elif _as_float(usage.get("cost_usd")) > _as_float(max_cost):
            exhausted.append("max_cost_usd")
    public_gateway_exhausted = usage.get("public_gateway_budget_exhausted")
    if isinstance(public_gateway_exhausted, list):
        exhausted.extend(
            f"public_gateway.{name}"
            for name in public_gateway_exhausted
            if f"public_gateway.{name}" not in exhausted
        )
    return {
        "limits": dict(limits),
        "usage": dict(usage),
        "exhausted": exhausted,
        "unverifiable": unverifiable,
        "within_budget": None if unverifiable else not exhausted,
        "enforcement": {
            "timeout_seconds": "hard_process_timeout",
            "max_agent_attempts": "hard_runner_limit",
            "max_agent_iterations": "hard_runner_limit_per_attempt",
            "llm_num_retries": "hard_runner_limit_per_request",
            "max_total_tokens": "hard_pre_request_plus_post_response_gate",
            "max_cost_usd": "hard_pre_request_plus_post_response_gate_when_cost_available",
            "public_gateway": "hard_in_process_counter",
        },
    }


def _feedback(
    *,
    outcome: str,
    payload: Mapping[str, Any],
    exception: Mapping[str, Any] | None,
    budget: Mapping[str, Any],
) -> JsonDict:
    blocker = payload.get("blocker") if isinstance(payload.get("blocker"), dict) else {}
    attempts = payload.get("agent_attempts")
    last_attempt = (
        attempts[-1]
        if isinstance(attempts, list) and attempts and isinstance(attempts[-1], dict)
        else {}
    )
    turns = last_attempt.get("turns") if isinstance(last_attempt, dict) else []
    last_turn = (
        turns[-1]
        if isinstance(turns, list) and turns and isinstance(turns[-1], dict)
        else {}
    )
    trace = (
        payload.get("agent_trace_artifacts")
        if isinstance(payload.get("agent_trace_artifacts"), dict)
        else {}
    )
    verifier = (
        trace.get("official_verifier_result")
        if isinstance(trace.get("official_verifier_result"), dict)
        else {}
    )
    if not verifier and isinstance(last_attempt.get("verifier"), dict):
        verifier = last_attempt["verifier"]
    native_artifacts = (
        last_attempt.get("artifacts")
        if isinstance(last_attempt.get("artifacts"), dict)
        else {}
    )
    last_public_progress = (
        last_turn.get("public_progress")
        if isinstance(last_turn.get("public_progress"), dict)
        else None
    )
    summary = (
        blocker.get("message")
        or trace.get("summary_final_message")
        or verifier.get("message")
        or last_turn.get("execution_error")
        or (exception or {}).get("message")
        or (
            "task completed successfully"
            if outcome == "success"
            else outcome.replace("_", " ")
        )
    )
    next_steps = {
        "success": "none",
        "scored": "none",
        "task_failure": "inspect the public trace feedback, then retry the task with corrected grounded actions",
        "agent_failure": "inspect agent_trace and the last agent attempt before retrying",
        "runtime_failure": "inspect blocker plus stderr; repair the runtime before retrying",
        "timeout": "inspect the last trace event; raise the timeout only if the task was still making progress",
        "budget_exhausted": "increase the exhausted limit or reduce turns/retries before retrying",
        "verifier_failure": "inspect the harness-side verifier report and public final message",
        "skipped": "none",
    }
    return {
        "summary": str(summary),
        "stage": payload.get("stage"),
        "blocker": dict(blocker),
        "exception": dict(exception) if exception else None,
        "trace_complete": trace.get("complete"),
        "last_public_message": trace.get("summary_final_message"),
        "last_public_progress": last_public_progress,
        "harness_verifier": dict(verifier),
        "native_attempt_artifacts": dict(native_artifacts),
        "retryable": (
            bool(exception.get("retryable"))
            if exception is not None and "retryable" in exception
            else (
                bool(blocker.get("retryable"))
                if "retryable" in blocker
                else outcome in FAILURE_STATUSES
            )
        ),
        "suggested_next_step": next_steps.get(outcome, "inspect task artifacts"),
        "budget_exhausted": list(budget.get("exhausted") or []),
        "budget_unverifiable": list(budget.get("unverifiable") or []),
        "pool_binding": (
            dict(payload.get("pool_binding"))
            if isinstance(payload.get("pool_binding"), dict)
            else None
        ),
        "official_score_eligible": (
            bool(payload.get("pool_binding", {}).get("bound"))
            if isinstance(payload.get("pool_coordinate"), dict)
            and isinstance(payload.get("pool_binding"), dict)
            else not isinstance(payload.get("pool_coordinate"), dict)
        ),
    }


def execute_task(
    task: EvaluationTask,
    *,
    run_id: str,
    manifest_digest: str,
    output_dir: Path,
    command: Sequence[str],
    timeout_seconds: float | None,
    budgets: Mapping[str, int | float] | None = None,
    worker_slot: WorkerSlot | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = _run_process,
) -> JsonDict:
    started_at = _utc_now()
    started_monotonic = time.monotonic()
    task_dir = output_dir / "tasks" / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    previous = read_status(status_path(output_dir, task))
    previous_attempt = (
        _as_int(previous.get("attempt_number"))
        if previous
        and previous.get("contract") == TASK_STATUS_CONTRACT
        and previous.get("run_id") == run_id
        and previous.get("manifest_hash") == manifest_digest
        else 0
    )
    attempt_number = previous_attempt + 1
    attempt_dir = task_dir / "attempts" / f"{attempt_number:04d}"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    report_path = task_dir / "runner_report.json"
    stdout_path = task_dir / "stdout.log"
    stderr_path = task_dir / "stderr.log"
    # A retry must never classify a stale report left by an earlier attempt.
    report_path.unlink(missing_ok=True)
    limits = dict(budgets or {})
    allocation = worker_slot.to_dict() if worker_slot is not None else None
    running_status: JsonDict = {
        "contract": TASK_STATUS_CONTRACT,
        "run_id": run_id,
        "manifest_hash": manifest_digest,
        "task": task.to_dict(),
        "task_id": task.task_id,
        "status": "running",
        "attempt_number": attempt_number,
        "command": list(command),
        "started_at": started_at,
        "worker": allocation,
        "budget": {
            "limits": limits,
            "usage": {},
            "exhausted": [],
            "within_budget": None,
        },
        "artifacts": {
            "runner_report": str(report_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "attempt_dir": str(attempt_dir),
        },
    }
    _atomic_write_json(status_path(output_dir, task), running_status)
    env = os.environ.copy()
    env.update(dict(task.env))
    env.update(
        {
            "EMBODIED_ARENA_EVALUATION_RUN_ID": run_id,
            "EMBODIED_ARENA_EVALUATION_TASK_ID": task.task_id,
            "EMBODIED_ARENA_BENCHMARK_ID": task.benchmark_id,
            "EMBODIED_ARENA_SPLIT": task.split,
            "EMBODIED_ARENA_VARIATION": task.variation,
            "EMBODIED_ARENA_EVALUATION_SEED": str(task.seed),
            "EMBODIED_ARENA_EVALUATION_ATTEMPT": str(attempt_number),
        }
    )
    if limits:
        env["EMBODIED_ARENA_EVALUATION_BUDGETS"] = _canonical_json(limits)
    if worker_slot is not None:
        env["EMBODIED_ARENA_EVALUATION_WORKER_ID"] = str(worker_slot.slot_id)
        if worker_slot.gpu_id is not None:
            env["CUDA_VISIBLE_DEVICES"] = worker_slot.gpu_id
            env["NVIDIA_VISIBLE_DEVICES"] = worker_slot.gpu_id
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["NVIDIA_VISIBLE_DEVICES"] = "none"
    timed_out = False
    exception: JsonDict | None = None
    try:
        completed = process_runner(
            list(command),
            cwd=resolve_project_root(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        returncode = int(completed.returncode)
        stdout = _as_text(completed.stdout)
        stderr = _as_text(completed.stderr)
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _as_text(exc.stdout)
        stderr = _as_text(exc.stderr)
        exception = {
            "category": "timeout",
            "type": type(exc).__name__,
            "message": f"task exceeded {timeout_seconds} seconds",
            "retryable": True,
        }
    except OSError as exc:
        returncode = 127
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exception = {
            "category": "process_launch",
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": True,
        }
    except Exception as exc:  # noqa: BLE001 - one broken case must not abort concurrent peers.
        returncode = 70
        stdout = ""
        stderr = f"{type(exc).__name__}: {exc}"
        exception = {
            "category": "harness",
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": False,
        }
    _write_attempt_log(
        attempt_path=attempt_dir / "stdout.log",
        latest_path=stdout_path,
        content=stdout,
    )
    _write_attempt_log(
        attempt_path=attempt_dir / "stderr.log",
        latest_path=stderr_path,
        content=stderr,
    )
    payload = read_status(report_path) or _json_payload(stdout)
    outcome = classify_result(
        returncode=returncode, payload=payload, timed_out=timed_out
    )
    duration_seconds = time.monotonic() - started_monotonic
    usage = _usage_from_payload(payload, duration_seconds=duration_seconds)
    budget = _budget_report(limits, usage)
    if budget["exhausted"]:
        outcome = "budget_exhausted"
    if payload:
        _atomic_write_json(attempt_dir / "runner_report.json", payload)
    pool_coordinate = (
        payload.get("pool_coordinate")
        if isinstance(payload.get("pool_coordinate"), dict)
        else None
    )
    pool_binding = (
        payload.get("pool_binding")
        if isinstance(payload.get("pool_binding"), dict)
        else None
    )
    pool_marked = bool(
        pool_coordinate is not None
        or any(key.startswith("EMBODIED_ARENA_POOL_") for key, _ in task.env)
    )
    official_score_eligible = payload.get("official_score_eligible") is not False and (
        not pool_marked
        or bool(pool_binding and pool_binding.get("bound") is True)
    )
    pool_binding_status = (
        "not_pool"
        if not pool_marked
        else str((pool_binding or {}).get("mode") or "coordinate_only")
    )
    status: JsonDict = {
        "contract": TASK_STATUS_CONTRACT,
        "run_id": run_id,
        "manifest_hash": manifest_digest,
        "task": task.to_dict(),
        "task_id": task.task_id,
        "outcome": outcome,
        "status": outcome,
        "attempt_number": attempt_number,
        "previous_outcome": previous.get("outcome") if previous else None,
        "returncode": returncode,
        "timed_out": timed_out,
        "command": list(command),
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(duration_seconds, 6),
        "worker": allocation,
        "budget": budget,
        "feedback": _feedback(
            outcome=outcome,
            payload=payload,
            exception=exception,
            budget=budget,
        ),
        "exception": exception,
        "pool_coordinate": pool_coordinate,
        "pool_binding": pool_binding,
        "pool_binding_status": pool_binding_status,
        "official_score_eligible": official_score_eligible,
        "runner_report": payload,
        "artifacts": {
            "runner_report": str(report_path),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "attempt_dir": str(attempt_dir),
            "attempt_runner_report": (
                str(attempt_dir / "runner_report.json") if payload else None
            ),
        },
    }
    # The runner report is allowed to be empty on a process-launch failure;
    # keep feedback consistent with the manifest's POOL_* marker in that case.
    feedback = status.get("feedback")
    if isinstance(feedback, dict):
        feedback["pool_binding"] = pool_binding
        feedback["official_score_eligible"] = official_score_eligible
    _atomic_write_json(attempt_dir / "status.json", status)
    _atomic_write_json(status_path(output_dir, task), status)
    return status


def _status_usage(status: Mapping[str, Any]) -> Mapping[str, Any]:
    budget = status.get("budget")
    if not isinstance(budget, dict):
        return {}
    usage = budget.get("usage")
    return usage if isinstance(usage, dict) else {}


def aggregate_statuses(
    *,
    output_dir: Path,
    run_id: str,
    manifest_digest: str,
    selected_tasks: Sequence[EvaluationTask],
) -> JsonDict:
    """Rebuild an idempotent aggregate from canonical per-task status files."""

    statuses: list[JsonDict] = []
    tasks_by_id = {task.task_id: task for task in selected_tasks}
    for task in selected_tasks:
        status = read_status(status_path(output_dir, task))
        if not status:
            continue
        if status.get("contract") != TASK_STATUS_CONTRACT:
            continue
        if (
            status.get("run_id") != run_id
            or status.get("manifest_hash") != manifest_digest
        ):
            continue
        if (
            status.get("task_id") != task.task_id
            or status.get("outcome") not in TERMINAL_STATUSES
        ):
            continue
        statuses.append(status)
    counts = Counter(str(status["outcome"]) for status in statuses)
    benchmark_summaries: dict[str, JsonDict] = {}
    for task in selected_tasks:
        benchmark_summaries.setdefault(
            task.benchmark_id,
            {
                "selected_task_count": 0,
                "completed_task_count": 0,
                "official_score_eligible_task_count": 0,
                "pool_binding_status_counts": {},
                "outcome_counts": {status: 0 for status in sorted(TERMINAL_STATUSES)},
            },
        )["selected_task_count"] += 1
    for status in statuses:
        benchmark_id = tasks_by_id[str(status["task_id"])].benchmark_id
        benchmark = benchmark_summaries[benchmark_id]
        benchmark["completed_task_count"] += 1
        benchmark["outcome_counts"][str(status["outcome"])] += 1
        if status.get("official_score_eligible") is True:
            benchmark["official_score_eligible_task_count"] += 1
        binding_status = str(status.get("pool_binding_status") or "unknown")
        binding_counts = benchmark["pool_binding_status_counts"]
        binding_counts[binding_status] = binding_counts.get(binding_status, 0) + 1
    for benchmark in benchmark_summaries.values():
        benchmark["pending_task_count"] = (
            benchmark["selected_task_count"] - benchmark["completed_task_count"]
        )
    resource_usage = {
        "task_wall_time_seconds": round(
            sum(_as_float(status.get("duration_seconds")) for status in statuses),
            6,
        ),
        "model_responses": sum(
            _as_int(_status_usage(status).get("model_responses")) for status in statuses
        ),
        "prompt_tokens": sum(
            _as_int(_status_usage(status).get("prompt_tokens")) for status in statuses
        ),
        "completion_tokens": sum(
            _as_int(_status_usage(status).get("completion_tokens"))
            for status in statuses
        ),
        "total_tokens": sum(
            _as_int(_status_usage(status).get("total_tokens")) for status in statuses
        ),
        "cost_usd": round(
            sum(
                _as_float(_status_usage(status).get("cost_usd")) for status in statuses
            ),
            8,
        ),
    }
    return {
        "contract": FULL_EVALUATION_CONTRACT,
        "run_id": run_id,
        "manifest_hash": manifest_digest,
        "selected_task_count": len(selected_tasks),
        "completed_task_count": len(statuses),
        "pending_task_count": len(selected_tasks) - len(statuses),
        "outcome_counts": {
            status: counts.get(status, 0) for status in sorted(TERMINAL_STATUSES)
        },
        "successful_task_count": counts.get("success", 0),
        "failed_task_count": sum(counts.get(status, 0) for status in FAILURE_STATUSES),
        "official_score_eligible_task_count": sum(
            1 for status in statuses if status.get("official_score_eligible") is True
        ),
        "pool_binding_status_counts": dict(
            Counter(
                str(status.get("pool_binding_status") or "unknown")
                for status in statuses
            )
        ),
        "resource_usage": resource_usage,
        "benchmarks": benchmark_summaries,
        "task_statuses": [
            {
                "task_id": status["task_id"],
                "outcome": status["outcome"],
                "status": status["outcome"],
                "attempt_number": status.get("attempt_number"),
                "worker": status.get("worker"),
                "feedback": (
                    status["feedback"].get("summary")
                    if isinstance(status.get("feedback"), dict)
                    else None
                ),
                "budget_exhausted": (
                    status["budget"].get("exhausted", [])
                    if isinstance(status.get("budget"), dict)
                    else []
                ),
                "pool_binding_status": status.get("pool_binding_status"),
                "official_score_eligible": status.get("official_score_eligible"),
                "status_path": str(
                    output_dir / "statuses" / f"{status['task_id']}.json"
                ),
            }
            for status in statuses
        ],
    }


def run_evaluation(
    manifest: Mapping[str, Any],
    *,
    output_dir: Path,
    benchmark_ids: Sequence[str] = (),
    shard_index: int = 0,
    shard_count: int = 1,
    resume: bool = False,
    retry_failed: bool = False,
    dry_run: bool = False,
    python_executable: str = sys.executable,
    runner_script: str = "scripts/run_native_universal_case.py",
    common_runner_args: Sequence[str] = (),
    timeout_seconds: float | None = None,
    max_workers: int | None = None,
    gpu_ids: Sequence[str | int] | None = None,
    gpu_memory_gb: float | None = None,
    gpu_memory_reserve_gb: float | None = None,
    respect_existing_gpu_usage: bool | None = None,
    port_base: int | None = None,
    budget_overrides: Mapping[str, int | float] | None = None,
    process_runner: Callable[..., subprocess.CompletedProcess[str]] = _run_process,
) -> JsonDict:
    loop_started = time.monotonic()
    output_dir = output_dir.resolve()
    digest = manifest_hash(manifest)
    run_id = deterministic_run_id(digest)
    selected = select_tasks(
        expand_manifest(manifest),
        benchmark_ids=benchmark_ids,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    harness_config = manifest.get("harness", {})
    if not isinstance(harness_config, dict):
        raise TaskManifestError("harness must be a JSON object")
    unknown_harness_fields = sorted(
        set(harness_config)
        - {
            "max_workers",
            "port_base",
            "gpus",
            "gpu_memory_gb",
            "gpu_memory_reserve_gb",
            "respect_existing_gpu_usage",
        }
    )
    if unknown_harness_fields:
        raise TaskManifestError(
            f"Unknown harness configuration fields: {unknown_harness_fields}"
        )
    configured_gpus = gpu_ids
    if configured_gpus is None:
        raw_gpus = harness_config.get("gpus", [])
        if not isinstance(raw_gpus, list):
            raise TaskManifestError("harness.gpus must be a JSON array")
        configured_gpus = raw_gpus
    effective_workers = max_workers
    if effective_workers is None:
        raw_workers = harness_config.get("max_workers")
        if raw_workers is not None:
            effective_workers = raw_workers
        elif configured_gpus:
            effective_workers = len(configured_gpus)
        else:
            effective_workers = 1
    if isinstance(effective_workers, bool) or not isinstance(effective_workers, int):
        raise TaskManifestError("max_workers must be an integer")
    raw_gpu_memory = (
        gpu_memory_gb
        if gpu_memory_gb is not None
        else harness_config.get("gpu_memory_gb")
    )
    if raw_gpu_memory is not None and (
        isinstance(raw_gpu_memory, bool)
        or not isinstance(raw_gpu_memory, (int, float))
        or raw_gpu_memory <= 0
    ):
        raise TaskManifestError("harness.gpu_memory_gb must be a positive number")
    total_gpu_memory_gb = float(raw_gpu_memory) if raw_gpu_memory is not None else None
    raw_gpu_reserve = (
        gpu_memory_reserve_gb
        if gpu_memory_reserve_gb is not None
        else harness_config.get("gpu_memory_reserve_gb", 0)
    )
    if (
        isinstance(raw_gpu_reserve, bool)
        or not isinstance(raw_gpu_reserve, (int, float))
        or raw_gpu_reserve < 0
    ):
        raise TaskManifestError(
            "harness.gpu_memory_reserve_gb must be a non-negative number"
        )
    if total_gpu_memory_gb is None and raw_gpu_reserve:
        raise TaskManifestError("gpu_memory_reserve_gb requires harness.gpu_memory_gb")
    usable_gpu_memory_gb = (
        total_gpu_memory_gb - float(raw_gpu_reserve)
        if total_gpu_memory_gb is not None
        else None
    )
    if usable_gpu_memory_gb is not None and usable_gpu_memory_gb <= 0:
        raise TaskManifestError(
            "GPU reserve must leave a positive usable memory capacity"
        )
    raw_respect_existing = (
        respect_existing_gpu_usage
        if respect_existing_gpu_usage is not None
        else harness_config.get(
            "respect_existing_gpu_usage", total_gpu_memory_gb is not None
        )
    )
    if not isinstance(raw_respect_existing, bool):
        raise TaskManifestError("harness.respect_existing_gpu_usage must be boolean")
    if raw_respect_existing and configured_gpus and total_gpu_memory_gb is None:
        raise TaskManifestError(
            "respect_existing_gpu_usage requires harness.gpu_memory_gb"
        )
    effective_respect_existing = bool(
        raw_respect_existing and configured_gpus and usable_gpu_memory_gb is not None
    )
    configured_port_base = port_base
    if configured_port_base is None:
        raw_port_base = harness_config.get("port_base", 8011)
        if isinstance(raw_port_base, bool) or not isinstance(raw_port_base, int):
            raise TaskManifestError("harness.port_base must be an integer")
        configured_port_base = raw_port_base
    slots = _worker_slots(
        max_workers=effective_workers,
        port_base=configured_port_base,
        gpu_ids=configured_gpus,
    )
    selected_gpu_ids = list(
        dict.fromkeys(slot.gpu_id for slot in slots if slot.gpu_id is not None)
    )
    gpu_inventory = _gpu_memory_inventory() if effective_respect_existing else []
    selected_inventory = _selected_gpu_memory(selected_gpu_ids, gpu_inventory)
    if effective_respect_existing:
        missing_inventory = [
            gpu for gpu in selected_gpu_ids if gpu not in selected_inventory
        ]
        if missing_inventory:
            raise TaskManifestError(
                "cannot read current free memory for selected GPUs: "
                f"{missing_inventory}"
            )
    admission_capacity_by_gpu: dict[str, float] = {}
    for gpu in selected_gpu_ids:
        capacity = float(usable_gpu_memory_gb or 0.0)
        if effective_respect_existing:
            free_gib = float(selected_inventory[gpu]["memory_free_gib"])
            capacity = min(capacity, max(0.0, free_gib - float(raw_gpu_reserve)))
        admission_capacity_by_gpu[gpu] = round(capacity, 6)
    manages_host_ports = (
        port_base is not None
        or "port_base" in harness_config
        or "openhands" in Path(runner_script).name.lower()
    )
    plans: list[JsonDict] = []
    execution_counts: Counter[str] = Counter()
    invocation_statuses: list[JsonDict] = []
    runnable: list[tuple[EvaluationTask, dict[str, int | float]]] = []
    dry_run_runnable_count = 0

    def allocated_slot(task: EvaluationTask, slot: WorkerSlot) -> WorkerSlot:
        requested, exclusive = _task_resource_request(task)
        explicit_cpu_only = requested == 0
        gpu_id = None if explicit_cpu_only else slot.gpu_id
        requested_value = float(requested or 0.0)
        reserved = (
            float(usable_gpu_memory_gb or requested_value)
            if exclusive
            else requested_value
        )
        return slot.allocated(
            gpu_id=gpu_id,
            requested_gb=requested_value,
            reserved_gb=reserved,
            exclusive=exclusive,
        )

    for task_index, task in enumerate(selected):
        budgets = _task_budgets(
            task,
            timeout_seconds=timeout_seconds,
            overrides=budget_overrides,
        )
        preview_slot = allocated_slot(task, slots[task_index % len(slots)])
        command = build_task_command(
            task,
            output_dir=output_dir,
            python_executable=python_executable,
            runner_script=runner_script,
            common_runner_args=common_runner_args,
            host_port=preview_slot.host_port if manages_host_ports else None,
            budgets=budgets,
        )
        existing = read_status(status_path(output_dir, task))
        matching_existing = bool(
            existing
            and existing.get("contract") == TASK_STATUS_CONTRACT
            and existing.get("run_id") == run_id
            and existing.get("manifest_hash") == digest
            and existing.get("outcome") in TERMINAL_STATUSES
        )
        skip_reason: str | None = None
        if not task.enabled:
            skip_reason = "disabled"
        elif (
            matching_existing
            and retry_failed
            and existing.get("outcome") not in FAILURE_STATUSES
        ):
            skip_reason = "already_successful"
        elif matching_existing and resume and not retry_failed:
            skip_reason = "already_completed"
        requested_memory, exclusive_gpu = _task_resource_request(task)
        if (
            skip_reason is None
            and (
                exclusive_gpu or (requested_memory is not None and requested_memory > 0)
            )
            and not configured_gpus
        ):
            raise TaskManifestError(
                f"task {task.task_id} requests GPU resources but harness.gpus is empty"
            )
        if (
            skip_reason is None
            and requested_memory is not None
            and usable_gpu_memory_gb is not None
            and requested_memory > usable_gpu_memory_gb
        ):
            raise TaskManifestError(
                f"task {task.task_id} requests {requested_memory} GiB GPU memory, "
                f"above usable capacity {usable_gpu_memory_gb} GiB"
            )
        if (
            skip_reason is None
            and usable_gpu_memory_gb is not None
            and requested_memory is None
        ):
            raise TaskManifestError(
                f"task {task.task_id} must declare resources.gpu_memory_gb "
                "when GPU memory admission is enabled"
            )
        currently_admissible = bool(
            requested_memory == 0
            or requested_memory is None
            or any(
                requested_memory <= capacity + 1e-9
                for capacity in admission_capacity_by_gpu.values()
            )
        )
        if (
            skip_reason is None
            and not dry_run
            and effective_respect_existing
            and not currently_admissible
        ):
            raise TaskManifestError(
                f"task {task.task_id} requests {requested_memory} GiB, but current "
                "per-GPU admission capacity after existing usage and reserve is "
                f"{admission_capacity_by_gpu}; clear the external GPU workload, "
                "lower the task reservation, or explicitly disable existing-usage protection"
            )
        if dry_run:
            if skip_reason is None:
                dry_run_runnable_count += 1
            plans.append(
                {
                    "task": task.to_dict(),
                    "command": command,
                    "budget": {"limits": budgets},
                    "resources": dict(task.resources),
                    "worker": preview_slot.to_dict(),
                    "allocation_provisional": effective_workers > 1,
                    "currently_admissible": currently_admissible,
                    "would_skip": skip_reason,
                }
            )
            continue
        if skip_reason == "disabled" and not matching_existing:
            skipped_at = _utc_now()
            skipped_budget = _budget_report(
                budgets,
                _usage_from_payload({}, duration_seconds=0.0),
            )
            skipped_feedback = _feedback(
                outcome="skipped",
                payload={},
                exception=None,
                budget=skipped_budget,
            )
            skipped_feedback["summary"] = "task disabled by manifest"
            skipped_feedback["stage"] = "scheduling"
            skipped_status: JsonDict = {
                "contract": TASK_STATUS_CONTRACT,
                "run_id": run_id,
                "manifest_hash": digest,
                "task": task.to_dict(),
                "task_id": task.task_id,
                "outcome": "skipped",
                "status": "skipped",
                "reason": skip_reason,
                "command": command,
                "started_at": skipped_at,
                "finished_at": skipped_at,
                "duration_seconds": 0.0,
                "worker": None,
                "budget": skipped_budget,
                "feedback": skipped_feedback,
            }
            _atomic_write_json(status_path(output_dir, task), skipped_status)
            execution_counts["skipped"] += 1
            continue
        if skip_reason:
            execution_counts["skipped"] += 1
            continue
        runnable.append((task, budgets))

    def run_on_slot(
        item: tuple[EvaluationTask, dict[str, int | float]],
        slot: WorkerSlot,
    ) -> JsonDict:
        task, budgets = item
        try:
            command = build_task_command(
                task,
                output_dir=output_dir,
                python_executable=python_executable,
                runner_script=runner_script,
                common_runner_args=common_runner_args,
                host_port=slot.host_port if manages_host_ports else None,
                budgets=budgets,
            )
            task_timeout = budgets.get("timeout_seconds")
            return execute_task(
                task,
                run_id=run_id,
                manifest_digest=digest,
                output_dir=output_dir,
                command=command,
                timeout_seconds=(
                    float(task_timeout) if task_timeout is not None else None
                ),
                budgets=budgets,
                worker_slot=slot,
                process_runner=process_runner,
            )
        except Exception as exc:  # noqa: BLE001 - preserve peer task scheduling.
            terminal_at = _utc_now()
            exception = {
                "category": "scheduler",
                "type": type(exc).__name__,
                "message": str(exc),
                "retryable": False,
            }
            budget = _budget_report(
                budgets,
                _usage_from_payload({}, duration_seconds=0.0),
            )
            status: JsonDict = {
                "contract": TASK_STATUS_CONTRACT,
                "run_id": run_id,
                "manifest_hash": digest,
                "task": task.to_dict(),
                "task_id": task.task_id,
                "outcome": "runtime_failure",
                "status": "runtime_failure",
                "returncode": 70,
                "timed_out": False,
                "started_at": terminal_at,
                "finished_at": terminal_at,
                "duration_seconds": 0.0,
                "worker": slot.to_dict(),
                "budget": budget,
                "feedback": _feedback(
                    outcome="runtime_failure",
                    payload={},
                    exception=exception,
                    budget=budget,
                ),
                "exception": exception,
                "runner_report": {},
                "artifacts": {},
            }
            try:
                _atomic_write_json(status_path(output_dir, task), status)
            except OSError:
                status["status_persisted"] = False
            return status

    planned_peak_concurrency = min(
        dry_run_runnable_count if dry_run else len(runnable), len(slots)
    )
    peak_concurrency = 0
    reserved_by_gpu: dict[str, float] = {str(gpu): 0.0 for gpu in configured_gpus}
    peak_reserved_by_gpu = dict(reserved_by_gpu)
    exclusive_gpus: set[str] = set()

    def can_allocate(task: EvaluationTask, base_slot: WorkerSlot) -> bool:
        requested, exclusive = _task_resource_request(task)
        if requested == 0:
            return True
        gpu_id = base_slot.gpu_id
        if requested is not None and requested > 0 and gpu_id is None:
            return False
        if exclusive and gpu_id is None:
            return False
        if requested is None:
            return True
        assert gpu_id is not None
        if gpu_id in exclusive_gpus:
            return False
        used = reserved_by_gpu.get(gpu_id, 0.0)
        if exclusive:
            return used == 0
        if usable_gpu_memory_gb is None:
            return True
        capacity = admission_capacity_by_gpu.get(gpu_id, usable_gpu_memory_gb)
        return used + requested <= capacity + 1e-9

    def reserve(task: EvaluationTask, base_slot: WorkerSlot) -> WorkerSlot:
        allocation = allocated_slot(task, base_slot)
        gpu_id = allocation.gpu_id
        if gpu_id is not None:
            reserved_by_gpu[gpu_id] = (
                reserved_by_gpu.get(gpu_id, 0.0) + allocation.gpu_memory_reserved_gb
            )
            if allocation.exclusive_gpu:
                exclusive_gpus.add(gpu_id)
            peak_reserved_by_gpu[gpu_id] = max(
                peak_reserved_by_gpu.get(gpu_id, 0.0),
                reserved_by_gpu[gpu_id],
            )
        return allocation

    def release(allocation: WorkerSlot) -> None:
        gpu_id = allocation.gpu_id
        if gpu_id is None:
            return
        reserved_by_gpu[gpu_id] = max(
            0.0,
            reserved_by_gpu.get(gpu_id, 0.0) - allocation.gpu_memory_reserved_gb,
        )
        if allocation.exclusive_gpu:
            exclusive_gpus.discard(gpu_id)

    if not dry_run and runnable:
        if planned_peak_concurrency == 1:
            for item in runnable:
                allocation = reserve(item[0], slots[0])
                peak_concurrency = 1
                status = run_on_slot(item, allocation)
                release(allocation)
                invocation_statuses.append(status)
                execution_counts[str(status["outcome"])] += 1
        else:
            pending = list(runnable)
            available_slots = list(slots)
            in_flight: dict[Future[JsonDict], tuple[WorkerSlot, WorkerSlot]] = {}
            with ThreadPoolExecutor(
                max_workers=planned_peak_concurrency,
                thread_name_prefix="arena-bench",
            ) as executor:
                try:
                    while pending or in_flight:
                        launched = True
                        while available_slots and pending and launched:
                            launched = False
                            for base_slot in sorted(
                                list(available_slots), key=lambda item: item.slot_id
                            ):
                                pending_index = next(
                                    (
                                        index
                                        for index, item in enumerate(pending)
                                        if can_allocate(item[0], base_slot)
                                    ),
                                    None,
                                )
                                if pending_index is None:
                                    continue
                                item = pending.pop(pending_index)
                                allocation = reserve(item[0], base_slot)
                                available_slots.remove(base_slot)
                                future = executor.submit(run_on_slot, item, allocation)
                                in_flight[future] = (base_slot, allocation)
                                peak_concurrency = max(peak_concurrency, len(in_flight))
                                launched = True
                        if not in_flight and pending:
                            task = pending[0][0]
                            requested, exclusive = _task_resource_request(task)
                            raise TaskManifestError(
                                "resource admission deadlock for "
                                f"{task.task_id}: gpu_memory_gb={requested}, "
                                f"exclusive_gpu={exclusive}"
                            )
                        if not in_flight:
                            break
                        completed, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in sorted(
                            completed,
                            key=lambda item: in_flight[item][0].slot_id,
                        ):
                            base_slot, allocation = in_flight.pop(future)
                            release(allocation)
                            available_slots.append(base_slot)
                            status = future.result()
                            invocation_statuses.append(status)
                            execution_counts[str(status["outcome"])] += 1
                except KeyboardInterrupt:
                    _terminate_active_processes()
                    for future in in_flight:
                        future.cancel()
                    raise
    aggregate = aggregate_statuses(
        output_dir=output_dir,
        run_id=run_id,
        manifest_digest=digest,
        selected_tasks=selected,
    )
    aggregate.update(
        {
            "dry_run": dry_run,
            "shard": {"index": shard_index, "count": shard_count},
            "benchmark_ids": list(benchmark_ids),
            "execution_counts": dict(sorted(execution_counts.items())),
            "harness": {
                "batch_contract": FULL_EVALUATION_CONTRACT,
                "task_status_contract": TASK_STATUS_CONTRACT,
                "single_case_runner": runner_script,
                "single_case_process_isolation": True,
                "official_verifier_agent_callable": False,
                "agent_visible_interface_mode": "universal",
            },
            "scheduler": {
                "strategy": "next_available_memory_gated_worker",
                "configured_workers": len(slots),
                "peak_concurrency": 0 if dry_run else peak_concurrency,
                "planned_peak_concurrency": planned_peak_concurrency,
                "worker_slots": [slot.to_dict() for slot in slots],
                "gpu_oversubscribed": bool(
                    configured_gpus
                    and len(slots) > len(configured_gpus)
                    and usable_gpu_memory_gb is None
                ),
                "shared_gpu_workers": bool(
                    configured_gpus and len(slots) > len(configured_gpus)
                ),
                "gpu_memory_admission_enabled": usable_gpu_memory_gb is not None,
                "gpu_memory_gb_per_device": total_gpu_memory_gb,
                "gpu_memory_reserve_gb_per_device": float(raw_gpu_reserve),
                "gpu_memory_usable_gb_per_device": usable_gpu_memory_gb,
                "respect_existing_gpu_usage": effective_respect_existing,
                "gpu_memory_inventory_at_start": list(selected_inventory.values()),
                "gpu_memory_admission_capacity_gb": dict(
                    sorted(admission_capacity_by_gpu.items())
                ),
                "peak_reserved_gpu_memory_gb": dict(
                    sorted(peak_reserved_by_gpu.items())
                ),
                "host_ports_managed": manages_host_ports,
                "loop_wall_time_seconds": round(time.monotonic() - loop_started, 6),
            },
            "plans": plans,
        }
    )
    task_wall_time = sum(
        _as_float(status.get("duration_seconds")) for status in invocation_statuses
    )
    loop_wall_time = float(aggregate["scheduler"]["loop_wall_time_seconds"])
    aggregate["scheduler"]["invocation_task_wall_time_seconds"] = round(
        task_wall_time, 6
    )
    aggregate["scheduler"]["effective_parallelism"] = (
        round(task_wall_time / loop_wall_time, 4) if loop_wall_time > 0 else 0.0
    )
    if not dry_run:
        _atomic_write_json(output_dir / "summary.json", aggregate)
    return aggregate
