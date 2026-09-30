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
        result = _post(payload, 300)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Primitive server request failed for {primitive}: {exc}") from exc
    if not result.get("ok"):
        raise RuntimeError(f"Primitive {primitive} failed: {result.get('error')}")
    return result.get("result")


def get_task_context():
    return _call("get_task_context")

def observe():
    return _call("observe")

def walk_forward(speed="slow", visible_state=""):
    return _call("walk_forward", speed=speed, visible_state=visible_state)

def turn(direction="left", degrees=30, visible_state=""):
    return _call("turn", direction=direction, degrees=degrees, visible_state=visible_state)

def step_back(distance=.25, visible_state=""):
    return _call("step_back", distance=distance, visible_state=visible_state)

def side_step(direction="left", distance=.25, visible_state=""):
    return _call("side_step", direction=direction, distance=distance, visible_state=visible_state)

def sit(height=.5, visible_state=""):
    return _call("sit", height=height, visible_state=visible_state)

def climb_up(visible_state=""):
    return _call("climb_up", visible_state=visible_state)

def climb_down(visible_state=""):
    return _call("climb_down", visible_state=visible_state)

def stop(visible_state=""):
    return _call("stop", visible_state=visible_state)
