from __future__ import annotations

import ast
import contextlib
import io
import signal
from dataclasses import dataclass
from typing import Any, Callable

from .primitives import AlfredOfficialPrimitives


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
        step_budget_exceeded = bool(self.exception and self.exception.startswith("CodeStepBudgetExceeded"))
        return {
            "code_attempts": self.attempts,
            "code_exception_count": int(
                self.exception is not None
                and not self.timed_out
                and not self.primitive_budget_exceeded
                and not step_budget_exceeded
            ),
            "code_timeout_count": int(self.timed_out),
            "code_primitive_budget_count": int(self.primitive_budget_exceeded),
            "code_step_budget_count": int(step_budget_exceeded),
        }


def execute_agent_code(
    primitives: AlfredOfficialPrimitives,
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
    previous_step_budget = primitives.active_max_env_steps
    primitives.active_max_env_steps = max_env_steps
    try:
        tree = ast.parse(code, filename="<alfred_official_agent_code>", mode="exec")
        _validate_agent_code(tree)
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), _time_limit(timeout_seconds):
            exec(compile(tree, "<alfred_official_agent_code>", "exec"), namespace, namespace)
    except CodeExecutionTimeout as exc:
        timed_out = True
        exception = f"{type(exc).__name__}: {exc}"
    except CodePrimitiveCallBudgetExceeded as exc:
        primitive_budget_exceeded = True
        exception = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        exception = f"{type(exc).__name__}: {exc}"
    finally:
        primitives.active_max_env_steps = previous_step_budget

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
    primitives: AlfredOfficialPrimitives,
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

    def list_actions() -> Any:
        count_call()
        return primitives.list_actions()

    def observe() -> Any:
        count_call()
        return primitives.observe()

    def get_frame(label: str = "frame") -> Any:
        count_call()
        return primitives.get_frame(label)

    def inspect_current_view(label: str | None = None, remember: bool = True) -> Any:
        count_call()
        return primitives.inspect_current_view(label=label, remember=remember)

    def detect_objects(query: str | None = None, queries: Any = None) -> Any:
        count_call()
        if query is None and queries is not None:
            query_list = queries if isinstance(queries, (list, tuple, set)) else [queries]
            hits: list[dict[str, Any]] = []
            seen: set[str] = set()
            for item in query_list:
                for obj in primitives.detect_objects(str(item)):
                    object_id = str(obj.get("objectId") or id(obj))
                    if object_id not in seen:
                        seen.add(object_id)
                        hits.append(obj)
            return hits
        return primitives.detect_objects(query)

    def remember_visible_objects(label: str | None = None, query: str | None = None) -> Any:
        count_call()
        return primitives.remember_visible_objects(label=label, query=query)

    def recall_visible_objects(query: str | None = None, label: str | None = None, limit: int = 50) -> Any:
        count_call()
        return primitives.recall_visible_objects(query=query, label=label, limit=limit)

    def read_search_memory(query: str | None = None, label: str | None = None, limit: int = 50) -> Any:
        count_call()
        return primitives.read_search_memory(query=query, label=label, limit=limit)

    def read_observed_spatial_map(query: str | None = None, limit: int = 50) -> Any:
        count_call()
        return primitives.read_observed_spatial_map(query=query, limit=limit)

    def scan_scene(
        queries: Any = None,
        rotations: int = 4,
        include_tilts: bool = True,
        remember: bool = True,
    ) -> Any:
        count_call()
        return primitives.scan_scene(
            queries=queries,
            rotations=rotations,
            include_tilts=include_tilts,
            remember=remember,
        )

    def search_scene(
        queries: Any = None,
        rounds: int = 3,
        scan_rotations: int = 4,
        include_tilts: bool = True,
        remember: bool = True,
        rotations: int | None = None,
    ) -> Any:
        count_call()
        if rotations is not None:
            scan_rotations = rotations
        return primitives.search_scene(
            queries=queries,
            rounds=rounds,
            scan_rotations=scan_rotations,
            include_tilts=include_tilts,
            remember=remember,
        )

    def explore_room(
        queries: Any = None,
        step_budget: int = 32,
        include_tilts: bool = True,
        remember: bool = True,
    ) -> Any:
        count_call()
        return primitives.explore_room(
            queries=queries,
            step_budget=step_budget,
            include_tilts=include_tilts,
            remember=remember,
        )

    def locate_object(
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
        include_tilts: bool = True,
        open_containers: bool = False,
    ) -> Any:
        count_call()
        return primitives.locate_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            include_tilts=include_tilts,
            open_containers=open_containers,
        )

    def ground_object(query: str) -> Any:
        count_call()
        return primitives.ground_object(query)

    def query_object_state(obj: Any) -> Any:
        count_call()
        return primitives.query_object_state(obj)

    def query_inventory() -> Any:
        count_call()
        return primitives.query_inventory()

    def move_ahead() -> Any:
        count_call()
        return primitives.move_ahead()

    def rotate(direction: str) -> Any:
        count_call()
        return primitives.rotate(direction)

    def look(direction: str) -> Any:
        count_call()
        return primitives.look(direction)

    def approach_object(obj: Any, max_steps: int = 3, stop_distance: float = 1.25) -> Any:
        count_call()
        return primitives.approach_object(obj, max_steps=max_steps, stop_distance=stop_distance)

    def open_object(obj: Any) -> Any:
        count_call()
        return primitives.open_object(obj)

    def close_object(obj: Any) -> Any:
        count_call()
        return primitives.close_object(obj)

    def pickup_object(obj: Any) -> Any:
        count_call()
        return primitives.pickup_object(obj)

    def put_object(obj: Any, receptacle: Any) -> Any:
        count_call()
        return primitives.put_object(obj, receptacle)

    def toggle_object(obj: Any, on: bool = True) -> Any:
        count_call()
        return primitives.toggle_object(obj, on)

    def slice_object(obj: Any) -> Any:
        count_call()
        return primitives.slice_object(obj)

    def pickup_located_object(
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
        open_containers: bool = True,
    ) -> Any:
        count_call()
        return primitives.pickup_located_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
            open_containers=open_containers,
        )

    def place_held_object(
        receptacle_query: Any,
        obj: Any = None,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 36,
    ) -> Any:
        count_call()
        return primitives.place_held_object(
            receptacle_query,
            obj=obj,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
        )

    def toggle_located_object(
        query: Any,
        aliases: Any = None,
        on: bool = True,
        support_queries: Any = None,
        search_budget: int = 36,
    ) -> Any:
        count_call()
        return primitives.toggle_located_object(
            query,
            aliases=aliases,
            on=on,
            support_queries=support_queries,
            search_budget=search_budget,
        )

    def open_located_object(
        query: Any,
        aliases: Any = None,
        support_queries: Any = None,
        search_budget: int = 24,
    ) -> Any:
        count_call()
        return primitives.open_located_object(
            query,
            aliases=aliases,
            support_queries=support_queries,
            search_budget=search_budget,
        )

    def write_evidence(key: str, value: Any) -> Any:
        count_call()
        return primitives.write_evidence(key, value)

    def read_evidence() -> Any:
        count_call()
        return primitives.read_evidence()

    def check_success() -> Any:
        count_call()
        return primitives.check_success()

    def check_progress_public() -> Any:
        count_call()
        return primitives.check_progress_public()

    return {
        "__builtins__": _safe_builtins(),
        "get_task_context": get_task_context,
        "list_actions": list_actions,
        "observe": observe,
        "get_frame": get_frame,
        "inspect_current_view": inspect_current_view,
        "detect_objects": detect_objects,
        "remember_visible_objects": remember_visible_objects,
        "recall_visible_objects": recall_visible_objects,
        "read_search_memory": read_search_memory,
        "read_observed_spatial_map": read_observed_spatial_map,
        "scan_scene": _budgeted(scan_scene, primitives, max_env_steps),
        "search_scene": _budgeted(search_scene, primitives, max_env_steps),
        "explore_room": _budgeted(explore_room, primitives, max_env_steps),
        "locate_object": _budgeted(locate_object, primitives, max_env_steps),
        "ground_object": ground_object,
        "query_object_state": query_object_state,
        "query_inventory": query_inventory,
        "move_ahead": _budgeted(move_ahead, primitives, max_env_steps),
        "rotate": _budgeted(rotate, primitives, max_env_steps),
        "look": _budgeted(look, primitives, max_env_steps),
        "approach_object": _budgeted(approach_object, primitives, max_env_steps),
        "open_object": _budgeted(open_object, primitives, max_env_steps),
        "close_object": _budgeted(close_object, primitives, max_env_steps),
        "pickup_object": _budgeted(pickup_object, primitives, max_env_steps),
        "put_object": _budgeted(put_object, primitives, max_env_steps),
        "toggle_object": _budgeted(toggle_object, primitives, max_env_steps),
        "slice_object": _budgeted(slice_object, primitives, max_env_steps),
        "pickup_located_object": _budgeted(pickup_located_object, primitives, max_env_steps),
        "place_held_object": _budgeted(place_held_object, primitives, max_env_steps),
        "toggle_located_object": _budgeted(toggle_located_object, primitives, max_env_steps),
        "open_located_object": _budgeted(open_located_object, primitives, max_env_steps),
        "write_evidence": write_evidence,
        "read_evidence": read_evidence,
        "check_progress_public": check_progress_public,
        "check_success": check_success,
        "verify_predicate": check_success,
        "true": True,
        "false": False,
        "null": None,
    }


def _budgeted(
    primitive: Callable[..., Any],
    primitives: AlfredOfficialPrimitives,
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
    if max_env_steps is None:
        return 120
    return max(80, max_env_steps * 6)


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
