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
    payload = json.dumps(
        {
            "primitive": primitive,
            "args": args,
            "kwargs": kwargs,
            "client": {"pid": os.getpid(), "argv0": sys.argv[0] if sys.argv else ""},
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


def observe_world() -> dict[str, Any]:
    return _call("observe_world")


def list_known_actions() -> dict[str, Any]:
    return _call("list_known_actions")


def get_action_schema() -> dict[str, Any]:
    return _call("get_action_schema")


def list_accessible_objects() -> list[dict[str, Any]]:
    return _call("list_accessible_objects")


def list_nearby_objects(query: str | None = None, max_distance: int | float | None = None) -> list[dict[str, Any]]:
    return _call("list_nearby_objects", query=query, max_distance=max_distance)


def list_inventory() -> list[dict[str, Any]]:
    return _call("list_inventory")


def validate_action_call(
    primitive_name: str,
    arguments: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return _call("validate_action_call", primitive_name, arguments, **kwargs)


def list_teleport_locations() -> dict[str, Any]:
    return _call("list_teleport_locations")


def move_direction(direction: str) -> dict[str, Any]:
    return _call("move_direction", direction)


def rotate_direction(direction: str) -> dict[str, Any]:
    return _call("rotate_direction", direction)


def teleport_to_location(location_name: str) -> dict[str, Any]:
    return _call("teleport_to_location", location_name)


def pickup_object(obj: Any) -> dict[str, Any]:
    return _call("pickup_object", obj)


def drop_object(obj: Any) -> dict[str, Any]:
    return _call("drop_object", obj)


def put_object(obj: Any, target: Any) -> dict[str, Any]:
    return _call("put_object", obj, target)


def open_object(obj: Any) -> dict[str, Any]:
    return _call("open_object", obj)


def close_object(obj: Any) -> dict[str, Any]:
    return _call("close_object", obj)


def activate_object(obj: Any) -> dict[str, Any]:
    return _call("activate_object", obj)


def deactivate_object(obj: Any) -> dict[str, Any]:
    return _call("deactivate_object", obj)


def use_object(obj: Any, target: Any) -> dict[str, Any]:
    return _call("use_object", obj, target)


def read_object(obj: Any) -> dict[str, Any]:
    return _call("read_object", obj)


def eat_object(obj: Any) -> dict[str, Any]:
    return _call("eat_object", obj)


def wait() -> dict[str, Any]:
    return _call("wait")


def talk_to(agent: Any) -> dict[str, Any]:
    return _call("talk_to", agent)


def choose_dialog_option(option_index: int) -> dict[str, Any]:
    return _call("choose_dialog_option", option_index)


def write_evidence(key: str, value: Any) -> dict[str, Any]:
    return _call("write_evidence", key, value)


def read_evidence() -> dict[str, Any]:
    return _call("read_evidence")


def check_success() -> dict[str, Any]:
    return _call("check_success")


__all__ = [
    "activate_object",
    "check_success",
    "choose_dialog_option",
    "close_object",
    "deactivate_object",
    "drop_object",
    "eat_object",
    "get_action_schema",
    "get_task_context",
    "list_accessible_objects",
    "list_inventory",
    "list_known_actions",
    "list_nearby_objects",
    "list_teleport_locations",
    "move_direction",
    "observe_world",
    "open_object",
    "pickup_object",
    "put_object",
    "read_evidence",
    "read_object",
    "rotate_direction",
    "talk_to",
    "teleport_to_location",
    "use_object",
    "validate_action_call",
    "wait",
    "write_evidence",
]
