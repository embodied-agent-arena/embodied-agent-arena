from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any


def _assert_solve_py_entrypoint() -> None:
    entrypoint = os.path.basename(sys.argv[0] if sys.argv else "")
    if entrypoint != "solve.py":
        raise RuntimeError(
            "VirtualHome primitive_api calls must be made by editing and running python solve.py. "
            "Do not use python -c, stdin Python, notebooks, heredocs, or temporary scripts; "
            "put read-only probes and execute_program_step(...) calls inside solve.py."
        )


def _server_url() -> str:
    value = os.environ.get("ROBENCH_PRIMITIVE_SERVER_URL")
    if not value:
        raise RuntimeError("ROBENCH_PRIMITIVE_SERVER_URL is not set. Run this inside run_instance.py/OpenHands.")
    return value.rstrip("/")


def _auth_headers() -> dict[str, str]:
    token = os.environ.get("ROBENCH_PRIMITIVE_SERVER_TOKEN")
    if not token:
        raise RuntimeError("ROBENCH_PRIMITIVE_SERVER_TOKEN is not set.")
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def _post(payload: bytes, timeout: int) -> dict[str, Any]:
    socket_path = os.environ.get("ROBENCH_PRIMITIVE_SERVER_SOCKET")
    if not socket_path:
        request = urllib.request.Request(f"{_server_url()}/call", data=payload, method="POST", headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=timeout) as response: return json.loads(response.read().decode("utf-8"))
    token = os.environ.get("ROBENCH_PRIMITIVE_SERVER_TOKEN") or ""; chunks: list[bytes] = []
    wire = (f"POST /call HTTP/1.0\r\nContent-Type: application/json\r\nAuthorization: Bearer {token}\r\nContent-Length: {len(payload)}\r\n\r\n").encode("ascii") + payload
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout); client.connect(socket_path); client.sendall(wire)
        while True:
            chunk = client.recv(65536)
            if not chunk: break
            chunks.append(chunk)
    head, separator, body = b"".join(chunks).partition(b"\r\n\r\n")
    if not separator: raise RuntimeError("Primitive server returned malformed HTTP.")
    status = int(head.split(b" ", 2)[1])
    if status >= 400: raise RuntimeError(f"Primitive server rejected request ({status}): {body.decode('utf-8', errors='replace')}")
    return json.loads(body.decode("utf-8"))


def _call(primitive: str, *args: Any, **kwargs: Any) -> Any:
    _assert_solve_py_entrypoint()
    payload = json.dumps(
        {
            "primitive": primitive,
            "args": args,
            "kwargs": kwargs,
            "client": {
                "pid": os.getpid(),
                "argv0": sys.argv[0] if sys.argv else "",
                "cwd": os.getcwd(),
                "api_path": __file__,
            },
        }
    ).encode("utf-8")
    try:
        result = _post(payload, 120)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Primitive server request failed for {primitive}: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Primitive {primitive} failed: {result.get('error')}")
    return result.get("result")


def get_task_context() -> dict[str, Any]:
    return _call("get_task_context")


def list_actions(query: str | None = None, limit: int = 80) -> list[str]:
    return _call("list_actions", query, limit)


def list_executable_actions(query: str | None = None, limit: int = 40) -> list[str]:
    return _call("list_executable_actions", query, limit)


def query_symbolic_state(scope: str | None = None) -> dict[str, Any]:
    return _call("query_symbolic_state", scope)


def validate_program_step(action_line: str) -> dict[str, Any]:
    return _call("validate_program_step", action_line)


def explain_action_preconditions(action_line: str) -> dict[str, Any]:
    return _call("explain_action_preconditions", action_line)


def execute_program_step(action_line: str) -> dict[str, Any]:
    return _call("execute_program_step", action_line)


def write_evidence(key: str, value: Any) -> dict[str, Any]:
    return _call("write_evidence", key, value)


def read_evidence() -> dict[str, Any]:
    return _call("read_evidence")


def check_activity_success() -> dict[str, Any]:
    return _call("check_activity_success")


__all__ = [
    "check_activity_success",
    "execute_program_step",
    "explain_action_preconditions",
    "get_task_context",
    "list_actions",
    "list_executable_actions",
    "query_symbolic_state",
    "read_evidence",
    "validate_program_step",
    "write_evidence",
]
