from __future__ import annotations

import ast
from contextlib import redirect_stdout
from dataclasses import dataclass
import io
import signal
from types import FrameType
from typing import Any

from .backend import EmbodiedBackend
from .schemas import PrimitiveResult


class UnsafeCodeError(ValueError):
    pass


@dataclass(slots=True)
class ExecutionResult:
    ok: bool
    stdout: str = ""
    error: str | None = None
    result: Any = None


class PrimitiveFacade:
    def __init__(self, backend: EmbodiedBackend) -> None:
        self._backend = backend

    def list(self, level: str | None = None) -> list[dict[str, Any]]:
        return [card.to_dict() for card in self._backend.list_primitives(level=level)]

    def call(self, name: str | None = None, **kwargs: Any) -> PrimitiveResult:
        primitive_name = name or kwargs.pop("interface", None) or kwargs.pop("name", None)
        if primitive_name is None:
            raise ValueError("PrimitiveFacade.call needs name=... or interface=...")
        return self._backend.call_primitive(str(primitive_name), **kwargs)

    def invoke(self, interface: str, **kwargs: Any) -> PrimitiveResult:
        return self.call(interface, **kwargs)

    def __getattr__(self, name: str) -> Any:
        def call_named(**kwargs: Any) -> PrimitiveResult:
            return self.call(name, **kwargs)

        return call_named


class StatefulCodeRunner:
    """Stateful Python runner for M0 smoke tests.

    This is a bounded execution wrapper, not a security sandbox. Production
    isolation should run code in a separate process/container.
    """

    DISALLOWED_NODES = (ast.Import, ast.ImportFrom)
    DISALLOWED_CALLS = {"__import__", "compile", "eval", "exec", "input", "open"}

    SAFE_BUILTINS = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "getattr": getattr,
        "hasattr": hasattr,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "print": print,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "Exception": Exception,
    }

    def __init__(self, backend: EmbodiedBackend, timeout_seconds: float = 2.0) -> None:
        self.backend = backend
        self.timeout_seconds = timeout_seconds
        facade = PrimitiveFacade(backend)
        self.globals: dict[str, Any] = {
            "__builtins__": self.SAFE_BUILTINS,
            "backend": backend,
            "primitives": facade,
            "call": facade.call,
            "memory": {},
        }

    def bind_task(self, task: Any) -> None:
        self.globals["task"] = task

    def execute(self, code: str) -> ExecutionResult:
        try:
            self._validate_ast(code)
        except UnsafeCodeError as exc:
            self.backend.record_event("code_rejected", {"code": code, "error": str(exc)})
            return ExecutionResult(ok=False, error=str(exc))

        self.backend.record_event("code_cell", {"code": code})
        stream = io.StringIO()
        try:
            with self._timeout(), redirect_stdout(stream):
                compiled = compile(
                    _rewrite_allowed_primitives_imports(ast.parse(code)),
                    "<embodied-agent-code>",
                    "exec",
                )
                exec(compiled, self.globals, self.globals)
        except Exception as exc:  # noqa: BLE001 - runner must capture agent failures.
            error = f"{type(exc).__name__}: {exc}"
            self.backend.record_event("code_error", {"error": error})
            return ExecutionResult(ok=False, stdout=stream.getvalue(), error=error)

        result = self.globals.get("result")
        self.backend.record_event("code_result", {"stdout": stream.getvalue(), "result": repr(result)})
        return ExecutionResult(ok=True, stdout=stream.getvalue(), result=result)

    def _validate_ast(self, code: str) -> None:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, self.DISALLOWED_NODES) and not _is_allowed_primitives_import(node):
                raise UnsafeCodeError(f"Disallowed syntax: {type(node).__name__}")
            if isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in self.DISALLOWED_CALLS:
                    raise UnsafeCodeError(f"Disallowed call: {call_name}")

    @staticmethod
    def _call_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return None

    def _timeout(self) -> "_AlarmTimeout":
        return _AlarmTimeout(self.timeout_seconds)


def _rewrite_allowed_primitives_imports(tree: ast.AST) -> ast.AST:
    """Keep the injected facade; do not execute a real import of `primitives`."""

    rewritten: list[ast.stmt] = []
    for statement in tree.body:
        if isinstance(statement, ast.Import) and _is_allowed_primitives_import(statement):
            continue
        if isinstance(statement, ast.ImportFrom) and _is_allowed_primitives_import(statement):
            for alias in statement.names:
                target = alias.asname or alias.name
                rewritten.append(
                    ast.Assign(
                        targets=[ast.Name(id=target, ctx=ast.Store())],
                        value=ast.Attribute(
                            value=ast.Name(id="primitives", ctx=ast.Load()),
                            attr=alias.name,
                            ctx=ast.Load(),
                        ),
                    )
                )
            continue
        rewritten.append(statement)
    tree.body = rewritten
    return ast.fix_missing_locations(tree)


def _is_allowed_primitives_import(node: ast.AST) -> bool:
    """Allow the two import forms models actually emit for the injected facade."""

    if isinstance(node, ast.Import):
        return all(alias.name == "primitives" and alias.asname in {None, "primitives"} for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return (
            node.module == "primitives"
            and (node.level or 0) == 0
            and bool(node.names)
            and all(alias.name in {"call", "invoke", "list"} for alias in node.names)
        )
    return False


class _AlarmTimeout:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self._old_handler: Any = None

    def __enter__(self) -> None:
        if self.seconds <= 0 or not hasattr(signal, "setitimer"):
            return
        self._old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self.seconds <= 0 or not hasattr(signal, "setitimer"):
            return
        signal.setitimer(signal.ITIMER_REAL, 0)
        if self._old_handler is not None:
            signal.signal(signal.SIGALRM, self._old_handler)

    @staticmethod
    def _handle_timeout(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("agent code timed out")
