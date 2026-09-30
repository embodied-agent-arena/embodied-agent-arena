from __future__ import annotations

import ast
import contextlib
import io
import os
import signal
from dataclasses import dataclass
from typing import Any, Callable

from .primitives import ScienceWorldTextPrimitives


class CodeExecutionTimeout(TimeoutError):
    pass


class CodeStepBudgetExceeded(RuntimeError):
    pass


class CodePrimitiveCallBudgetExceeded(RuntimeError):
    pass


class UnsafeAgentCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodeExecutionResult:
    code: str
    attempts: int
    stdout: str
    stderr: str
    exception: str | None
    timed_out: bool
    primitive_budget_exceeded: bool
    trace_limit_exceeded: bool
    final_success: bool
    final_verification: dict[str, Any]

    @property
    def metrics(self) -> dict[str, Any]:
        budget_exceeded = bool(self.exception and self.exception.startswith("CodeStepBudgetExceeded"))
        primitive_budget_exceeded = bool(self.primitive_budget_exceeded)
        return {
            "code_attempts": self.attempts,
            "code_exception_count": int(
                self.exception is not None
                and not self.timed_out
                and not budget_exceeded
                and not primitive_budget_exceeded
                and not self.trace_limit_exceeded
            ),
            "code_timeout_count": int(self.timed_out),
            "code_primitive_budget_count": int(primitive_budget_exceeded),
            "code_trace_limit_count": int(self.trace_limit_exceeded),
        }


def execute_agent_code(
    primitives: ScienceWorldTextPrimitives,
    code: str,
    *,
    attempt_index: int = 1,
    timeout_seconds: int = 60,
    max_env_steps: int | None = None,
    max_primitive_calls: int | None = None,
) -> CodeExecutionResult:
    if max_primitive_calls is None:
        max_primitive_calls = default_primitive_call_budget(max_env_steps)
    primitives.record_harness_event(
        "code_execution_started",
        {
            "code_attempt_index": attempt_index,
            "timeout_seconds": timeout_seconds,
            "max_env_steps": max_env_steps,
            "max_primitive_calls": max_primitive_calls,
        },
        side_effect=False,
    )

    stdout = io.StringIO()
    stderr = io.StringIO()
    exception: str | None = None
    timed_out = False
    primitive_budget_exceeded = False
    trace_limit_exceeded = False
    namespace = _execution_namespace(
        primitives,
        max_env_steps=max_env_steps,
        max_primitive_calls=max_primitive_calls,
    )
    try:
        tree = ast.parse(code, filename="<scienceworld_agent_code>", mode="exec")
        _validate_agent_code(tree)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), _time_limit(timeout_seconds):
            exec(compile(tree, "<scienceworld_agent_code>", "exec"), namespace, namespace)
    except CodeExecutionTimeout as exc:
        timed_out = True
        exception = f"{type(exc).__name__}: {exc}"
    except CodePrimitiveCallBudgetExceeded as exc:
        primitive_budget_exceeded = True
        exception = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - exercised through scripts.
        trace_limit_exceeded = type(exc).__name__ == "TraceLimitExceeded"
        exception = f"{type(exc).__name__}: {exc}"

    try:
        final_verification = primitives.check_success()
    except Exception as exc:  # pragma: no cover - defensive after trace/runtime failure.
        if exception is None:
            exception = f"{type(exc).__name__}: {exc}"
            trace_limit_exceeded = type(exc).__name__ == "TraceLimitExceeded"
        final_verification = {
            "success": False,
            "done": False,
            "score": 0.0,
            "reward": 0.0,
            "source": "code_executor_final_verification_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    result = CodeExecutionResult(
        code=code,
        attempts=attempt_index,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        exception=exception,
        timed_out=timed_out,
        primitive_budget_exceeded=primitive_budget_exceeded,
        trace_limit_exceeded=trace_limit_exceeded,
        final_success=bool(final_verification["success"]),
        final_verification=final_verification,
    )
    primitives.record_harness_event(
        "code_execution_finished",
        {
            "code_attempt_index": attempt_index,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exception": result.exception,
            "timed_out": result.timed_out,
            "primitive_budget_exceeded": result.primitive_budget_exceeded,
            "trace_limit_exceeded": result.trace_limit_exceeded,
            "final_success": result.final_success,
            "final_verification": result.final_verification,
        },
        side_effect=False,
    )
    return result


def _execution_namespace(
    primitives: ScienceWorldTextPrimitives,
    *,
    max_env_steps: int | None,
    max_primitive_calls: int | None,
) -> dict[str, Any]:
    primitive_call_count = 0

    def count_call() -> None:
        nonlocal primitive_call_count
        primitive_call_count += 1
        if max_primitive_calls is not None and primitive_call_count > max_primitive_calls:
            raise CodePrimitiveCallBudgetExceeded(f"Exceeded max_primitive_calls={max_primitive_calls}.")

    def get_task_context() -> Any:
        count_call()
        return primitives.get_task_context()

    def observe_text_world() -> Any:
        count_call()
        return primitives.observe_text_world()

    def list_actions() -> Any:
        count_call()
        return primitives.list_actions()

    def inspect_current_state(query: Any = None, limit: int = 80) -> Any:
        count_call()
        return primitives.inspect_current_state(query=query, limit=limit)

    def filter_actions(
        include: Any = None,
        exclude: Any = None,
        startswith: str | None = None,
        limit: int = 80,
        exclude_failed: bool = True,
        incl: Any = None,
        contains: Any = None,
    ) -> Any:
        count_call()
        include = include if include is not None else (incl if incl is not None else contains)
        return primitives.filter_actions(
            include=include,
            exclude=exclude,
            startswith=startswith,
            limit=limit,
            exclude_failed=exclude_failed,
        )

    def list_recent_failures(limit: int = 20) -> Any:
        count_call()
        return primitives.list_recent_failures(limit=limit)

    def get_score_state() -> Any:
        count_call()
        return primitives.get_score_state()

    def step_text_action(action: str) -> Any:
        count_call()
        return primitives.step_text_action(action)

    def look() -> Any:
        count_call()
        return primitives.look()

    def inventory() -> Any:
        count_call()
        return primitives.inventory()

    def check_success() -> Any:
        count_call()
        return primitives.check_success()

    def write_evidence(key: str, value: Any) -> Any:
        count_call()
        return primitives.write_evidence(key, value)

    def read_evidence() -> Any:
        count_call()
        return primitives.read_evidence()

    return {
        "__builtins__": _safe_builtins(),
        "get_task_context": get_task_context,
        "observe_text_world": observe_text_world,
        "list_actions": list_actions,
        "inspect_current_state": inspect_current_state,
        "filter_actions": filter_actions,
        "list_recent_failures": list_recent_failures,
        "get_score_state": get_score_state,
        "step_text_action": _budgeted(step_text_action, primitives, max_env_steps),
        "look": look,
        "inventory": inventory,
        "check_success": check_success,
        "write_evidence": write_evidence,
        "read_evidence": read_evidence,
        "true": True,
        "false": False,
        "null": None,
    }


def _budgeted(
    primitive: Callable[..., Any],
    primitives: ScienceWorldTextPrimitives,
    max_env_steps: int | None,
) -> Callable[..., Any]:
    if max_env_steps is None:
        return primitive

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if primitives.metrics["env_steps"] >= max_env_steps:
            raise CodeStepBudgetExceeded(f"Exceeded max_env_steps={max_env_steps}.")
        return primitive(*args, **kwargs)

    return wrapped


def default_primitive_call_budget(max_env_steps: int | None) -> int:
    raw = os.environ.get("SCIENCEWORLD_MAX_PRIMITIVE_CALLS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    if max_env_steps is None:
        return 250
    return max(100, max_env_steps * 5)


def _safe_builtins() -> dict[str, Any]:
    return {
        "AssertionError": AssertionError,
        "Exception": Exception,
        "RuntimeError": RuntimeError,
        "ValueError": ValueError,
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "callable": callable,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "repr": repr,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
    }


def _validate_agent_code(tree: ast.AST) -> None:
    banned_names = {
        "__import__",
        "breakpoint",
        "classmethod",
        "compile",
        "delattr",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "memoryview",
        "object",
        "open",
        "property",
        "setattr",
        "staticmethod",
        "super",
        "type",
        "vars",
    }
    banned_nodes = (
        ast.AsyncFunctionDef,
        ast.ClassDef,
        ast.Delete,
        ast.Global,
        ast.Import,
        ast.ImportFrom,
        ast.Lambda,
        ast.Nonlocal,
        ast.Raise,
        ast.Try,
        ast.With,
    )
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            raise UnsafeAgentCodeError(f"Unsupported syntax in generated code: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id in banned_names:
            raise UnsafeAgentCodeError(f"Unsafe name in generated code: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise UnsafeAgentCodeError(f"Unsafe attribute access in generated code: {node.attr}")


@contextlib.contextmanager
def _time_limit(seconds: int):
    if seconds <= 0:
        yield
        return

    def handler(_signum, _frame):
        raise CodeExecutionTimeout(f"Agent code timed out after {seconds} seconds.")

    old_handler = signal.getsignal(signal.SIGALRM)
    old_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, old_timer[0], old_timer[1])
        signal.signal(signal.SIGALRM, old_handler)
