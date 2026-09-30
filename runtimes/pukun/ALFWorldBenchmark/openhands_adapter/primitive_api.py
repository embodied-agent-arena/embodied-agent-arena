from __future__ import annotations

import json
import os
import socket
import sys
import urllib.error
import urllib.request
from typing import Any


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
    return _post_primitive_payload(payload, timeout)


def _post_primitive_payload(payload: bytes, timeout: int) -> dict[str, Any]:
    socket_path = os.environ.get("ROBENCH_PRIMITIVE_SERVER_SOCKET")
    if not socket_path:
        request = urllib.request.Request(f"{_server_url()}/call", data=payload, method="POST", headers=_auth_headers())
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    token = os.environ.get("ROBENCH_PRIMITIVE_SERVER_TOKEN") or ""
    request_bytes = (
        f"POST /call HTTP/1.0\r\nContent-Type: application/json\r\nAuthorization: Bearer {token}\r\n"
        f"Content-Length: {len(payload)}\r\n\r\n"
    ).encode("ascii") + payload
    chunks: list[bytes] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(socket_path)
        client.sendall(request_bytes)
        while True:
            chunk = client.recv(65536)
            if not chunk:
                break
            chunks.append(chunk)
    head, separator, body = b"".join(chunks).partition(b"\r\n\r\n")
    if not separator:
        raise RuntimeError("Primitive server returned a malformed HTTP response.")
    status = int(head.split(b" ", 2)[1])
    if status >= 400:
        raise RuntimeError(f"Primitive server rejected request ({status}): {body.decode('utf-8', errors='replace')}")
    return json.loads(body.decode("utf-8"))


def _call(primitive: str, *args: Any, **kwargs: Any) -> Any:
    payload = json.dumps(
        {
            "primitive": primitive,
            "args": args,
            "kwargs": kwargs,
            "client": {
                "pid": os.getpid(),
                "argv0": sys.argv[0] if sys.argv else "",
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


def observe_text_state() -> str:
    return _call("observe_text_state")


def list_actions() -> list[str]:
    return _call("list_actions")


def match_actions(
    intent: str | None = None,
    object_name: str | None = None,
    receptacle_name: str | None = None,
    include: Any = None,
    limit: int = 20,
) -> dict[str, Any]:
    return _call(
        "match_actions",
        intent=intent,
        object_name=object_name,
        receptacle_name=receptacle_name,
        include=include,
        limit=limit,
    )


def examine_object(name: str) -> dict[str, Any]:
    return _call("examine_object", name)


def look() -> dict[str, Any]:
    return _call("look")


def inventory() -> dict[str, Any]:
    return _call("inventory")


def go_to(name: str) -> dict[str, Any]:
    return _call("go_to", name)


def open_object(name: str) -> dict[str, Any]:
    return _call("open_object", name)


def close_object(name: str) -> dict[str, Any]:
    return _call("close_object", name)


def pickup_object(name: str) -> dict[str, Any]:
    return _call("pickup_object", name)


def place_object(obj: str, receptacle: str) -> dict[str, Any]:
    return _call("place_object", obj, receptacle)


def toggle_object(name: str) -> dict[str, Any]:
    return _call("toggle_object", name)


def clean_object(obj: str) -> dict[str, Any]:
    return _call("clean_object", obj)


def heat_object(obj: str) -> dict[str, Any]:
    return _call("heat_object", obj)


def cool_object(obj: str) -> dict[str, Any]:
    return _call("cool_object", obj)


def write_evidence(key: str, value: Any) -> dict[str, Any]:
    return _call("write_evidence", key, value)


def read_evidence() -> dict[str, Any]:
    return _call("read_evidence")


def check_success() -> dict[str, Any]:
    return _call("check_success")


__all__ = [
    "check_success",
    "clean_object",
    "close_object",
    "cool_object",
    "examine_object",
    "get_task_context",
    "go_to",
    "heat_object",
    "inventory",
    "list_actions",
    "look",
    "match_actions",
    "observe_text_state",
    "open_object",
    "pickup_object",
    "place_object",
    "read_evidence",
    "toggle_object",
    "write_evidence",
]
