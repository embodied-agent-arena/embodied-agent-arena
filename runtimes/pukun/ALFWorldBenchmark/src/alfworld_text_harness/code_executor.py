from __future__ import annotations

import contextlib
import ast
import io
import signal
from dataclasses import dataclass
from typing import Any, Callable

from .primitives import ALFWorldTextPrimitives


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
    final_success: bool
    final_verification: dict[str, Any]

    @property
    def metrics(self) -> dict[str, Any]:
        return {
            "code_attempts": self.attempts,
            "code_exception_count": int(
                self.exception is not None and not self.timed_out and not self.primitive_budget_exceeded
            ),
            "code_timeout_count": int(self.timed_out),
            "code_primitive_budget_count": int(self.primitive_budget_exceeded),
        }


def execute_agent_code(
    primitives: ALFWorldTextPrimitives,
    code: str,
    *,
    attempt_index: int = 1,
    timeout_seconds: int = 30,
    max_env_steps: int | None = None,
    max_primitive_calls: int | None = None,
) -> CodeExecutionResult:
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
    namespace = _execution_namespace(
        primitives,
        max_env_steps=max_env_steps,
        max_primitive_calls=max_primitive_calls,
    )
    try:
        tree = ast.parse(code, filename="<alfworld_agent_code>", mode="exec")
        _validate_agent_code(tree)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), _time_limit(timeout_seconds):
            exec(compile(tree, "<alfworld_agent_code>", "exec"), namespace, namespace)
    except CodeExecutionTimeout as exc:
        timed_out = True
        exception = f"{type(exc).__name__}: {exc}"
    except CodePrimitiveCallBudgetExceeded as exc:
        primitive_budget_exceeded = True
        exception = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - exercised through task scripts.
        exception = f"{type(exc).__name__}: {exc}"

    final_verification = primitives.check_success()
    result = CodeExecutionResult(
        code=code,
        attempts=attempt_index,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        exception=exception,
        timed_out=timed_out,
        primitive_budget_exceeded=primitive_budget_exceeded,
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
            "final_success": result.final_success,
            "final_verification": result.final_verification,
        },
        side_effect=False,
    )
    return result


def _execution_namespace(
    primitives: ALFWorldTextPrimitives,
    *,
    max_env_steps: int | None,
    max_primitive_calls: int | None = None,
) -> dict[str, Any]:
    primitive_call_counter = {"count": 0}

    def get_task_context() -> Any:
        return primitives.get_task_context()

    def observe_text_state() -> Any:
        return primitives.observe_text_state()

    def list_actions() -> Any:
        return primitives.list_actions()

    def match_actions(
        intent: str | None = None,
        object_name: str | None = None,
        receptacle_name: str | None = None,
        include: Any = None,
        limit: int = 20,
    ) -> Any:
        return primitives.match_actions(
            intent=intent,
            object_name=object_name,
            receptacle_name=receptacle_name,
            include=include,
            limit=limit,
        )

    def examine_object(name: str) -> Any:
        return primitives.examine_object(name)

    def look() -> Any:
        return primitives.look()

    def inventory() -> Any:
        return primitives.inventory()

    def go_to(name: str) -> Any:
        return primitives.go_to(name)

    def open_object(name: str) -> Any:
        return primitives.open_object(name)

    def close_object(name: str) -> Any:
        return primitives.close_object(name)

    def pickup_object(name: str) -> Any:
        return primitives.pickup_object(name)

    def place_object(obj: str, receptacle: str) -> Any:
        return primitives.place_object(obj, receptacle)

    def toggle_object(name: str) -> Any:
        return primitives.toggle_object(name)

    def clean_object(obj: str) -> Any:
        return primitives.clean_object(obj)

    def heat_object(obj: str) -> Any:
        return primitives.heat_object(obj)

    def cool_object(obj: str) -> Any:
        return primitives.cool_object(obj)

    def check_success() -> Any:
        return primitives.check_success()

    def write_evidence(key: str, value: Any) -> Any:
        return primitives.write_evidence(key, value)

    def read_evidence() -> Any:
        return primitives.read_evidence()

    return {
        "__builtins__": _safe_builtins(),
        "get_task_context": _budgeted(
            get_task_context,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "observe_text_state": _budgeted(
            observe_text_state,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "list_actions": _budgeted(
            list_actions,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "match_actions": _budgeted(
            match_actions,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "examine_object": _budgeted(
            examine_object,
            primitives,
            max_env_steps,
            max_primitive_calls,
            primitive_call_counter,
        ),
        "look": _budgeted(look, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "inventory": _budgeted(inventory, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "go_to": _budgeted(go_to, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "open_object": _budgeted(open_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "close_object": _budgeted(close_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "pickup_object": _budgeted(pickup_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "place_object": _budgeted(place_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "toggle_object": _budgeted(toggle_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "clean_object": _budgeted(clean_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "heat_object": _budgeted(heat_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "cool_object": _budgeted(cool_object, primitives, max_env_steps, max_primitive_calls, primitive_call_counter),
        "check_success": _budgeted(
            check_success,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "write_evidence": _budgeted(
            write_evidence,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
        "read_evidence": _budgeted(
            read_evidence,
            primitives,
            max_env_steps=None,
            max_primitive_calls=max_primitive_calls,
            primitive_call_counter=primitive_call_counter,
        ),
    }


def _budgeted(
    primitive: Callable[..., Any],
    primitives: ALFWorldTextPrimitives,
    max_env_steps: int | None = None,
    max_primitive_calls: int | None = None,
    primitive_call_counter: dict[str, int] | None = None,
) -> Callable[..., Any]:
    if max_env_steps is None and max_primitive_calls is None:
        return primitive

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if max_primitive_calls is not None:
            if primitive_call_counter is None:
                raise RuntimeError("Primitive call counter is required when max_primitive_calls is set.")
            if primitive_call_counter["count"] >= max_primitive_calls:
                raise CodePrimitiveCallBudgetExceeded(f"Exceeded max_primitive_calls={max_primitive_calls}.")
            primitive_call_counter["count"] += 1
        if max_env_steps is not None and primitives.metrics["env_steps"] >= max_env_steps:
            raise CodeStepBudgetExceeded(f"Exceeded max_env_steps={max_env_steps}.")
        return primitive(*args, **kwargs)

    return wrapped


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
