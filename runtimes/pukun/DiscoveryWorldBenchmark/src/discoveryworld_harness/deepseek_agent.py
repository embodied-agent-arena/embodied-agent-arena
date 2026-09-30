from __future__ import annotations

import json
import multiprocessing
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from .agents import AgentRunResult
from .code_executor import execute_agent_code
from .env_file import load_env_file
from .primitive_cards import get_primitive_cards, render_primitive_cards_for_prompt
from .primitives import DiscoveryWorldPrimitives

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekV3CodeAgent:
    def __init__(self, root_dir: Path, agent_name: str = "deepseek_v3_code"):
        load_env_file(root_dir / ".env")
        self.name = agent_name
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it in your shell.")

    def run(self, primitives: DiscoveryWorldPrimitives, max_steps: int) -> AgentRunResult:
        context = primitives.get_task_context()
        execution = None
        code_exception_count = 0
        code_timeout_count = 0
        code_primitive_budget_count = 0
        max_attempts = _max_code_attempts()
        feedback: dict[str, Any] | None = None
        stopped_reason = "not_success"
        for attempt_index in range(1, max_attempts + 1):
            observation = primitives.observe_world()
            code = self._generate_code(context, observation, max_steps, previous_feedback=feedback)
            primitives.record_harness_event(
                "agent_code_generated",
                {
                    "agent": self.name,
                    "model": self.model,
                    "code_attempt_index": attempt_index,
                    "executed_code": code,
                    "primitive_cards": get_primitive_cards(),
                    "repair_attempt": attempt_index > 1,
                },
                side_effect=False,
            )
            execution = execute_agent_code(
                primitives,
                code,
                attempt_index=attempt_index,
                timeout_seconds=_code_timeout_seconds(),
                max_env_steps=max_steps,
                max_primitive_calls=_primitive_call_budget(max_steps),
            )
            code_exception_count += int(
                execution.exception is not None and not execution.timed_out and not execution.primitive_budget_exceeded
            )
            code_timeout_count += int(execution.timed_out)
            code_primitive_budget_count += int(execution.primitive_budget_exceeded)
            if execution.final_success:
                stopped_reason = "success"
                break
            if execution.primitive_budget_exceeded:
                stopped_reason = "primitive_call_budget"
                break
            if execution.timed_out:
                stopped_reason = "code_timeout"
                break
            if execution.exception:
                stopped_reason = "code_exception"
            else:
                stopped_reason = "not_success"
            feedback = {
                "previous_code": code,
                "previous_stdout": execution.stdout[-2000:],
                "previous_stderr": execution.stderr[-2000:],
                "previous_exception": execution.exception,
                "previous_final_verification": execution.final_verification,
                "current_observation": primitives.observe_world(),
                "current_inventory": primitives.list_inventory(),
                "current_accessible_objects": primitives.list_accessible_objects(),
                "evidence": primitives.read_evidence(),
            }
        if execution is None:
            raise RuntimeError("DeepSeek code agent did not execute any code attempts.")
        return AgentRunResult(
            stopped_reason=stopped_reason,
            steps_attempted=primitives.metrics["env_steps"],
            metrics={
                "code_attempts": execution.attempts,
                "code_exception_count": code_exception_count,
                "code_timeout_count": code_timeout_count,
                "code_primitive_budget_count": code_primitive_budget_count,
            },
        )

    def _generate_code(
        self,
        context: dict[str, Any],
        observation: dict[str, Any],
        max_steps: int,
        previous_feedback: dict[str, Any] | None = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding agent controlling DiscoveryWorld through safe Python primitives. "
                    "Write one Python program that solves the task by calling only the provided primitives. "
                    "This is a task-first benchmark harness, not a repository exploration task: solve the current "
                    "DiscoveryWorld instance through primitive calls only. "
                    "Do not import modules, open files, call external services, inspect dunder attributes, "
                    "or use hidden/oracle state. "
                    "The executor rejects import/from, try/except, with, class, lambda, raise, open/eval/exec, "
                    "getattr/setattr/globals/locals, dunder names, and unsafe builtin names such as dir. "
                    "Use variable names like direction instead of dir. Keep the code plain Python loops and if statements. "
                    "Do not import math; for square roots use x ** 0.5 or pow(x, 0.5). "
                    "Each action primitive is thin: at most one native DiscoveryWorld action plus tick. "
                    "Objects can be referenced by uuid or by a unique current observation name; if names are ambiguous, use uuid. "
                    "Use observe_world(), list_accessible_objects(), list_inventory(), list_teleport_locations(), "
                    "and write_evidence() to adapt. "
                    "Follow task_recipe. Do not treat nearbyObjects as directly interactable; interaction primitives "
                    "normally require list_accessible_objects() or inventory. Once an object/person/receptacle is accessible, "
                    "act immediately rather than continuing to wander. "
                    "Important DiscoveryWorld rule: nearbyObjects tells you where objects are, but action primitives that take objects "
                    "usually require the object to be in list_accessible_objects() or inventory. If an object is visible in "
                    "nearbyObjects but not accessible, navigate or rotate until it becomes accessible. "
                    "For cardinal directions north/east/south/west, if the target is distance 1 in that direction, first call "
                    "rotate_direction(direction), then observe/list_accessible_objects again before pickup/open/talk/put/use. "
                    "For diagonal directions such as north-west, move one cardinal component at a time while legal, then rotate toward "
                    "the final cardinal direction when distance is 1. Do not stop just because direct west/north movement is blocked; "
                    "check whether the object is beside you and needs rotation instead. "
                    "For pick-and-place: locate the target object in nearbyObjects, move/rotate until accessible, pickup_object(uuid), "
                    "then locate the target receptacle, move/rotate until accessible, and put_object(held_uuid_or_name, target_uuid). "
                    "For dialog: move/rotate until the person is accessible, talk_to(uuid), then read dialog_box and call choose_dialog_option(index). "
                    "Dialog schema is exactly: obs['dialog_box']['is_in_dialog'], obs['dialog_box']['dialogIn'], "
                    "and obs['dialog_box']['dialogOptions'] as a dict mapping option numbers to text. "
                    "If dialogIn says select the option that says `lime`, find the dialogOptions entry whose value is lime "
                    "and call choose_dialog_option(int(option_number)). "
                    "Primitive action results are StepResult objects, not dicts: use result.valid_action, result.error, "
                    "result.success, result.completed, result.score, or result.verification. Do not call result.get(...). "
                    "check_success() only returns a safe verifier summary; call it near the end and after likely completion. "
                    'Return JSON only: {"code": "<python code>"}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "initial_observation": observation,
                        "max_env_steps": max_steps,
                        "primitive_cards": render_primitive_cards_for_prompt(),
                        "w3_prompt_contract": _w3_prompt_contract(),
                        "task_recipe": _discoveryworld_task_recipe(context),
                        "execution_rules": [
                            "Use uuid from list_accessible_objects/list_inventory whenever possible.",
                            "For adjacent objects, rotate toward the cardinal direction before interacting.",
                            "For diagonal objects, move one cardinal component at a time, then rotate and re-list accessible objects.",
                            "After invalid action feedback, change heading/object uuid rather than repeating the same call.",
                        ],
                        "previous_feedback": previous_feedback,
                        "example_style": "\n".join(
                            [
                                "ctx = get_task_context()",
                                "write_evidence('goal', ctx['goal_text'])",
                                "obs = observe_world()",
                                "objects = list_accessible_objects()",
                                "locations = list_teleport_locations()",
                                "# Prefer uuid from visible objects when object names repeat.",
                                "# If a target is in obs['nearbyObjects']['objects']['west'] with distance 1, call rotate_direction('west') before interacting.",
                                "# If a target is north-west, move north/west as legal, then rotate toward the side where the target is distance 1.",
                                "# Use only thin calls such as move_direction, pickup_object, put_object, talk_to.",
                                "# Dialog: options = obs['dialog_box']['dialogOptions']; choose the option whose text appears in obs['dialog_box']['dialogIn'].",
                                "check_success()",
                            ]
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        data = self._chat_completion(messages)
        content = data["choices"][0]["message"]["content"]
        return _parse_code(content)

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
            "response_format": {"type": "json_object"},
        }
        return _chat_completion_isolated(
            endpoint=_chat_endpoint(self.base_url),
            payload=payload,
            api_key=self.api_key,
            timeout_seconds=_api_timeout_seconds(),
        )


def _chat_completion_isolated(
    *,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    parent_conn, child_conn = multiprocessing.get_context("spawn").Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_chat_completion_child,
        args=(child_conn, endpoint, payload, api_key, timeout_seconds),
    )
    process.start()
    child_conn.close()
    try:
        if not parent_conn.poll(timeout_seconds + 5):
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            raise RuntimeError(f"DeepSeek API request timed out after {timeout_seconds} seconds.")
        status, value = parent_conn.recv()
        if status == "ok":
            return value
        raise RuntimeError(f"DeepSeek API request failed: {_redact(str(value))}")
    finally:
        parent_conn.close()
        if process.is_alive():
            process.terminate()
        process.join(timeout=2)


def _chat_completion_child(
    conn: Any,
    endpoint: str,
    payload: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
) -> None:
    socket.setdefaulttimeout(timeout_seconds)
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    try:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout_seconds, context=ssl_context) as response:
            conn.send(("ok", json.loads(response.read().decode("utf-8"))))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        conn.send(("error", f"DeepSeek API HTTP {exc.code}: {body}"))
    except (TimeoutError, urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        conn.send(("error", str(exc)))
    finally:
        conn.close()


def _chat_endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return base + "/chat/completions"


def _parse_code(content: str) -> str:
    text = content.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("code"), str):
            return data["code"]
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def _w3_prompt_contract() -> list[str]:
    return [
        "Write one top-level Python solution for the current DiscoveryWorld task.",
        "Use task_context, observation, primitive cards, accessible objects, inventory, and primitive feedback only.",
        "Make nearby objects accessible before interacting; do not guess uuid/name arguments.",
        "Use evidence to record target uuid/name choices and final verifier status.",
        "If feedback shows invalid/no_match, change navigation/rotation/object reference strategy.",
    ]


def _discoveryworld_task_recipe(context: dict[str, Any]) -> dict[str, Any]:
    task_type = str(context.get("task_type") or context.get("scenario_name") or "").lower()
    goal = str(context.get("goal_text") or "").lower()
    if "pick" in task_type or "place" in task_type or ("pick up" in goal and "place" in goal):
        steps = [
            "observe_world and identify target object/receptacle names from goal_text",
            "find target in nearbyObjects; move/rotate until it appears in list_accessible_objects",
            "pickup_object using accessible uuid/name",
            "find receptacle; move/rotate until accessible",
            "put_object using held object uuid/name and receptacle uuid/name",
            "check_success",
        ]
    elif "dialog" in task_type or "talk" in goal or "option" in goal:
        steps = [
            "find requested person/NPC in nearbyObjects",
            "move/rotate until person is accessible",
            "talk_to(person uuid)",
            "read obs['dialog_box']['dialogIn'] and dialogOptions",
            "choose the option index whose text matches the instruction",
            "check_success",
        ]
    elif any(word in task_type or word in goal for word in ("use", "activate", "read", "measure", "test")):
        steps = [
            "identify tool/device/target from goal_text",
            "make tool/device accessible",
            "pickup/use/open/activate with exact primitive arguments from accessible objects",
            "observe_world and write evidence",
            "check_success",
        ]
    else:
        steps = [
            "observe_world",
            "list accessible objects and inventory",
            "use task_context goal_text to choose a concrete object/person/device",
            "navigate/rotate until accessible",
            "perform one task-relevant side-effect primitive",
            "check_success",
        ]
    return {
        "benchmark": "DiscoveryWorld",
        "task_type": task_type,
        "steps": steps,
        "navigation_notes": [
            "nearbyObjects is evidence, not permission to interact.",
            "distance 1 in a cardinal direction usually needs rotate_direction(cardinal) before interaction.",
            "diagonal directions require one cardinal move at a time.",
            "teleport_to_location is only valid for the teleport-enabled profile.",
        ],
    }


def _code_timeout_seconds() -> int:
    raw = os.environ.get("DISCOVERYWORLD_CODE_TIMEOUT_SECONDS", "60")
    try:
        return int(raw)
    except ValueError:
        return 60


def _api_timeout_seconds() -> int:
    raw = os.environ.get("DISCOVERYWORLD_API_TIMEOUT_SECONDS", "90")
    try:
        return max(5, int(raw))
    except ValueError:
        return 90


def _max_code_attempts() -> int:
    raw = os.environ.get("DISCOVERYWORLD_MAX_CODE_ATTEMPTS", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _primitive_call_budget(max_steps: int) -> int:
    raw = os.environ.get("DISCOVERYWORLD_MAX_PRIMITIVE_CALLS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(100, max_steps * 5)


def _redact(value: str) -> str:
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", value)
    return value[:2000]
