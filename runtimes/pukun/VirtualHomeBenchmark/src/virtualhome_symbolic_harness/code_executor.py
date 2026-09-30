from __future__ import annotations

import ast
import contextlib
import io
import signal
from dataclasses import dataclass
from typing import Any

from .primitives import VirtualHomeSymbolicPrimitives


class CodeExecutionTimeout(TimeoutError):
    pass


class UnsafeAgentCodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class CodeExecutionResult:
    code: str
    stdout: str
    stderr: str
    exception: str | None
    timed_out: bool
    final_success: bool
    final_verification: dict[str, Any]

    @property
    def metrics(self) -> dict[str, int]:
        return {
            "code_exception_count": int(self.exception is not None and not self.timed_out),
            "code_timeout_count": int(self.timed_out),
        }


def execute_agent_code(
    primitives: VirtualHomeSymbolicPrimitives,
    code: str,
    *,
    timeout_seconds: int = 30,
) -> CodeExecutionResult:
    primitives.record_harness_event("code_execution_started", {"timeout_seconds": timeout_seconds})
    stdout = io.StringIO()
    stderr = io.StringIO()
    exception = None
    timed_out = False
    namespace = _execution_namespace(primitives)
    try:
        tree = ast.parse(code, filename="<virtualhome_agent_code>", mode="exec")
        _validate_agent_code(tree)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), _time_limit(timeout_seconds):
            exec(compile(tree, "<virtualhome_agent_code>", "exec"), namespace, namespace)
    except CodeExecutionTimeout as exc:
        timed_out = True
        exception = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    final_verification = primitives.check_activity_success()
    result = CodeExecutionResult(
        code=code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        exception=exception,
        timed_out=timed_out,
        final_success=bool(final_verification["success"]),
        final_verification=final_verification,
    )
    primitives.record_harness_event(
        "code_execution_finished",
        {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exception": result.exception,
            "timed_out": result.timed_out,
            "final_success": result.final_success,
            "final_verification": result.final_verification,
        },
    )
    return result


def _execution_namespace(primitives: VirtualHomeSymbolicPrimitives) -> dict[str, Any]:
    return {
        "__builtins__": _safe_builtins(),
        "get_task_context": primitives.get_task_context,
        "list_actions": primitives.list_actions,
        "list_executable_actions": primitives.list_executable_actions,
        "query_symbolic_state": primitives.query_symbolic_state,
        "validate_program_step": primitives.validate_program_step,
        "explain_action_preconditions": primitives.explain_action_preconditions,
        "execute_program_step": primitives.execute_program_step,
        "write_evidence": primitives.write_evidence,
        "read_evidence": primitives.read_evidence,
        "check_activity_success": primitives.check_activity_success,
    }


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
        "compile",
        "dir",
        "eval",
        "exec",
        "getattr",
        "globals",
        "input",
        "locals",
        "open",
        "setattr",
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
        if isinstance(node, ast.Name) and (node.id in banned_names or "__" in node.id):
            raise UnsafeAgentCodeError(f"Unsafe name in generated code: {node.id}")
        if isinstance(node, ast.Attribute) and "__" in node.attr:
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
