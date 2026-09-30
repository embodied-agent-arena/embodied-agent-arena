from __future__ import annotations

import json
import os
import socket
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
    request_bytes = (f"POST /call HTTP/1.0\r\nContent-Type: application/json\r\nAuthorization: Bearer {token}\r\nContent-Length: {len(payload)}\r\n\r\n").encode("ascii") + payload
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout); client.connect(socket_path); client.sendall(request_bytes)
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
    payload = json.dumps({"primitive": primitive, "args": args, "kwargs": kwargs}).encode("utf-8")
    try:
        result = _post(payload, 240)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Primitive server request failed for {primitive}: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Primitive {primitive} failed: {result.get('error')}")
    return result.get("result")


def get_task_context() -> dict[str, Any]:
    return _call("get_task_context")


def list_actions() -> list[str]:
    return _call("list_actions")


def observe() -> dict[str, Any]:
    return _call("observe")


def get_frame(label: str = "frame") -> dict[str, Any]:
    return _call("get_frame", label)


def inspect_current_view(label: str | None = None, remember: bool = True) -> dict[str, Any]:
    return _call("inspect_current_view", label, remember)


def detect_objects(query: str | None = None, queries: Any = None) -> list[dict[str, Any]]:
    if query is None and queries is not None:
        query_list = queries if isinstance(queries, (list, tuple, set)) else [queries]
        hits: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in query_list:
            for obj in _call("detect_objects", str(item)):
                object_id = str(obj.get("objectId") or id(obj))
                if object_id not in seen:
                    seen.add(object_id)
                    hits.append(obj)
        return hits
    return _call("detect_objects", query)


def remember_visible_objects(label: str | None = None, query: str | None = None) -> dict[str, Any]:
    return _call("remember_visible_objects", label, query)


def recall_visible_objects(query: str | None = None, label: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return _call("recall_visible_objects", query, label, limit)


def read_search_memory(query: str | None = None, label: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    return _call("read_search_memory", query, label, limit)


def read_observed_spatial_map(query: str | None = None, limit: int = 50) -> dict[str, Any]:
    return _call("read_observed_spatial_map", query, limit)


def scan_scene(queries: Any = None, rotations: int = 4, include_tilts: bool = True, remember: bool = True) -> dict[str, Any]:
    return _call("scan_scene", queries, rotations, include_tilts, remember)


def search_scene(queries: Any = None, rounds: int = 3, scan_rotations: int = 4, include_tilts: bool = True, remember: bool = True, rotations: int | None = None) -> dict[str, Any]:
    if rotations is not None:
        scan_rotations = rotations
    return _call("search_scene", queries, rounds, scan_rotations, include_tilts, remember)


def explore_room(queries: Any = None, step_budget: int = 32, include_tilts: bool = True, remember: bool = True) -> dict[str, Any]:
    return _call("explore_room", queries, step_budget, include_tilts, remember)


def locate_object(
    query: Any,
    aliases: Any = None,
    support_queries: Any = None,
    search_budget: int = 36,
    include_tilts: bool = True,
    open_containers: bool = False,
) -> dict[str, Any]:
    return _call("locate_object", query, aliases, support_queries, search_budget, include_tilts, open_containers)


def ground_object(query: str) -> dict[str, Any]:
    return _call("ground_object", query)


def query_object_state(obj: Any) -> dict[str, Any]:
    return _call("query_object_state", obj)


def query_inventory() -> list[dict[str, Any]]:
    return _call("query_inventory")


def move_ahead() -> dict[str, Any]:
    return _call("move_ahead")


def rotate(direction: str) -> dict[str, Any]:
    return _call("rotate", direction)


def look(direction: str) -> dict[str, Any]:
    return _call("look", direction)


def approach_object(obj: Any, max_steps: int = 3, stop_distance: float = 1.25) -> dict[str, Any]:
    return _call("approach_object", obj, max_steps, stop_distance)


def open_object(obj: Any) -> dict[str, Any]:
    return _call("open_object", obj)


def close_object(obj: Any) -> dict[str, Any]:
    return _call("close_object", obj)


def pickup_object(obj: Any) -> dict[str, Any]:
    return _call("pickup_object", obj)


def put_object(obj: Any, receptacle: Any) -> dict[str, Any]:
    return _call("put_object", obj, receptacle)


def toggle_object(obj: Any, on: bool = True) -> dict[str, Any]:
    return _call("toggle_object", obj, on)


def slice_object(obj: Any) -> dict[str, Any]:
    return _call("slice_object", obj)


def pickup_located_object(
    query: Any,
    aliases: Any = None,
    support_queries: Any = None,
    search_budget: int = 36,
    open_containers: bool = True,
) -> dict[str, Any]:
    return _call("pickup_located_object", query, aliases, support_queries, search_budget, open_containers)


def place_held_object(
    receptacle_query: Any,
    obj: Any = None,
    aliases: Any = None,
    support_queries: Any = None,
    search_budget: int = 36,
) -> dict[str, Any]:
    return _call("place_held_object", receptacle_query, obj, aliases, support_queries, search_budget)


def toggle_located_object(
    query: Any,
    aliases: Any = None,
    on: bool = True,
    support_queries: Any = None,
    search_budget: int = 36,
) -> dict[str, Any]:
    return _call("toggle_located_object", query, aliases, on, support_queries, search_budget)


def open_located_object(
    query: Any,
    aliases: Any = None,
    support_queries: Any = None,
    search_budget: int = 24,
) -> dict[str, Any]:
    return _call("open_located_object", query, aliases, support_queries, search_budget)


def write_evidence(key: str, value: Any) -> dict[str, Any]:
    return _call("write_evidence", key, value)


def read_evidence() -> dict[str, Any]:
    return _call("read_evidence")


def check_success() -> dict[str, Any]:
    return _call("check_success")


def check_progress_public() -> dict[str, Any]:
    return _call("check_progress_public")


def verify_predicate() -> dict[str, Any]:
    return _call("check_success")


__all__ = [
    "approach_object",
    "check_progress_public",
    "check_success",
    "close_object",
    "detect_objects",
    "explore_room",
    "get_frame",
    "get_task_context",
    "ground_object",
    "inspect_current_view",
    "list_actions",
    "locate_object",
    "look",
    "move_ahead",
    "observe",
    "open_located_object",
    "open_object",
    "pickup_located_object",
    "pickup_object",
    "place_held_object",
    "put_object",
    "query_inventory",
    "query_object_state",
    "read_evidence",
    "read_observed_spatial_map",
    "read_search_memory",
    "recall_visible_objects",
    "remember_visible_objects",
    "rotate",
    "scan_scene",
    "search_scene",
    "slice_object",
    "toggle_located_object",
    "toggle_object",
    "verify_predicate",
    "write_evidence",
]
