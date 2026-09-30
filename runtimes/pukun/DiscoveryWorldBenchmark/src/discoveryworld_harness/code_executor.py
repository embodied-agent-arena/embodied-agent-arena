from __future__ import annotations

import ast
import contextlib
import io
import os
import signal
from dataclasses import dataclass
from typing import Any, Callable

from .primitives import DiscoveryWorldPrimitives


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
    primitives: DiscoveryWorldPrimitives,
    code: str,
    *,
    attempt_index: int = 1,
    timeout_seconds: int = 60,
    max_env_steps: int | None = None,
    max_primitive_calls: int | None = None,
) -> CodeExecutionResult:
    if max_primitive_calls is None:
        max_primitive_calls = _default_primitive_call_budget(max_env_steps)
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
        tree = ast.parse(code, filename="<discoveryworld_agent_code>", mode="exec")
        _validate_agent_code(tree)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), _time_limit(timeout_seconds):
            exec(compile(tree, "<discoveryworld_agent_code>", "exec"), namespace, namespace)
    except CodeExecutionTimeout as exc:
        timed_out = True
        exception = f"{type(exc).__name__}: {exc}"
    except CodePrimitiveCallBudgetExceeded as exc:
        primitive_budget_exceeded = True
        exception = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
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
    primitives: DiscoveryWorldPrimitives,
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

    def observe_world() -> Any:
        count_call()
        return primitives.observe_world()

    def list_known_actions() -> Any:
        count_call()
        return primitives.list_known_actions()

    def get_action_schema() -> Any:
        count_call()
        return primitives.get_action_schema()

    def list_accessible_objects() -> Any:
        count_call()
        return primitives.list_accessible_objects()

    def list_nearby_objects(query: str | None = None, max_distance: int | float | None = None) -> Any:
        count_call()
        return primitives.list_nearby_objects(query=query, max_distance=max_distance)

    def list_inventory() -> Any:
        count_call()
        return primitives.list_inventory()

    def validate_action_call(
        primitive_name: str,
        arguments: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        count_call()
        return primitives.validate_action_call(primitive_name, arguments, **kwargs)

    def list_teleport_locations() -> Any:
        count_call()
        return primitives.list_teleport_locations()

    def move_direction(direction: str) -> Any:
        count_call()
        return primitives.move_direction(direction)

    def rotate_direction(direction: str) -> Any:
        count_call()
        return primitives.rotate_direction(direction)

    def teleport_to_location(location_name: str) -> Any:
        count_call()
        return primitives.teleport_to_location(location_name)

    def pickup_object(obj: Any) -> Any:
        count_call()
        return primitives.pickup_object(obj)

    def drop_object(obj: Any) -> Any:
        count_call()
        return primitives.drop_object(obj)

    def put_object(obj: Any, target: Any) -> Any:
        count_call()
        return primitives.put_object(obj, target)

    def open_object(obj: Any) -> Any:
        count_call()
        return primitives.open_object(obj)

    def close_object(obj: Any) -> Any:
        count_call()
        return primitives.close_object(obj)

    def activate_object(obj: Any) -> Any:
        count_call()
        return primitives.activate_object(obj)

    def deactivate_object(obj: Any) -> Any:
        count_call()
        return primitives.deactivate_object(obj)

    def use_object(obj: Any, target: Any) -> Any:
        count_call()
        return primitives.use_object(obj, target)

    def read_object(obj: Any) -> Any:
        count_call()
        return primitives.read_object(obj)

    def eat_object(obj: Any) -> Any:
        count_call()
        return primitives.eat_object(obj)

    def wait() -> Any:
        count_call()
        return primitives.wait()

    def talk_to(agent: Any) -> Any:
        count_call()
        return primitives.talk_to(agent)

    def choose_dialog_option(option_index: int) -> Any:
        count_call()
        return primitives.choose_dialog_option(option_index)

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
        "observe_world": observe_world,
        "list_known_actions": list_known_actions,
        "get_action_schema": get_action_schema,
        "list_accessible_objects": list_accessible_objects,
        "list_nearby_objects": list_nearby_objects,
        "list_inventory": list_inventory,
        "validate_action_call": validate_action_call,
        "list_teleport_locations": list_teleport_locations,
        "move_direction": _budgeted(move_direction, primitives, max_env_steps),
        "rotate_direction": _budgeted(rotate_direction, primitives, max_env_steps),
        "teleport_to_location": _budgeted(teleport_to_location, primitives, max_env_steps),
        "pickup_object": _budgeted(pickup_object, primitives, max_env_steps),
        "drop_object": _budgeted(drop_object, primitives, max_env_steps),
        "put_object": _budgeted(put_object, primitives, max_env_steps),
        "open_object": _budgeted(open_object, primitives, max_env_steps),
        "close_object": _budgeted(close_object, primitives, max_env_steps),
        "activate_object": _budgeted(activate_object, primitives, max_env_steps),
        "deactivate_object": _budgeted(deactivate_object, primitives, max_env_steps),
        "use_object": _budgeted(use_object, primitives, max_env_steps),
        "read_object": _budgeted(read_object, primitives, max_env_steps),
        "eat_object": _budgeted(eat_object, primitives, max_env_steps),
        "wait": _budgeted(wait, primitives, max_env_steps),
        "talk_to": _budgeted(talk_to, primitives, max_env_steps),
        "choose_dialog_option": _budgeted(choose_dialog_option, primitives, max_env_steps),
        "check_success": check_success,
        "write_evidence": write_evidence,
        "read_evidence": read_evidence,
    }


def _budgeted(
    primitive: Callable[..., Any],
    primitives: DiscoveryWorldPrimitives,
    max_env_steps: int | None,
) -> Callable[..., Any]:
    if max_env_steps is None:
        return primitive

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        if primitives.metrics["env_steps"] >= max_env_steps:
            raise CodeStepBudgetExceeded(f"Exceeded max_env_steps={max_env_steps}.")
        return primitive(*args, **kwargs)

    return wrapped


def _default_primitive_call_budget(max_env_steps: int | None) -> int:
    raw = os.environ.get("DISCOVERYWORLD_MAX_PRIMITIVE_CALLS", "").strip()
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
        "pow": pow,
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
        if isinstance(node, ast.Name):
            if node.id in banned_names or "__" in node.id:
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
