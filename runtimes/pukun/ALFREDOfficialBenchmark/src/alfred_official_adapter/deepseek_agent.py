from __future__ import annotations

import json
import multiprocessing
import os
import re
import socket
import ssl
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import certifi

from .agents import AgentRunResult
from .code_executor import execute_agent_code
from .env_file import load_default_deepseek_env
from .primitive_cards import get_primitive_cards, render_primitive_cards_for_prompt
from .primitives import AlfredOfficialPrimitives


DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


class DeepSeekCodeAgent:
    def __init__(self, root_dir: Path, agent_name: str = "deepseek_v3_code"):
        load_default_deepseek_env(root_dir)
        self.name = agent_name
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it in your shell.")

    def run(self, primitives: AlfredOfficialPrimitives, max_steps: int) -> AgentRunResult:
        context = primitives.get_task_context()
        execution = None
        code_exception_count = 0
        code_timeout_count = 0
        code_primitive_budget_count = 0
        code_step_budget_count = 0
        max_attempts = _max_code_attempts()
        feedback: dict[str, Any] | None = None
        stopped_reason = "not_success"
        for attempt_index in range(1, max_attempts + 1):
            observation = primitives.observe()
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
                execution.exception is not None
                and not execution.timed_out
                and not execution.primitive_budget_exceeded
                and not _is_step_budget_exceeded(execution.exception)
            )
            code_timeout_count += int(execution.timed_out)
            code_primitive_budget_count += int(execution.primitive_budget_exceeded)
            code_step_budget_count += int(_is_step_budget_exceeded(execution.exception))
            if execution.final_success:
                stopped_reason = "success"
                break
            if _is_step_budget_exceeded(execution.exception):
                stopped_reason = "max_steps"
                break
            if execution.primitive_budget_exceeded:
                stopped_reason = "primitive_call_budget"
                break
            if execution.timed_out:
                stopped_reason = "code_timeout"
                break
            stopped_reason = "code_exception" if execution.exception else "not_success"
            feedback = {
                "previous_stdout": execution.stdout[-2000:],
                "previous_stderr": execution.stderr[-2000:],
                "previous_exception": execution.exception,
                "previous_final_verification": execution.final_verification,
                "current_observation": primitives.observe(),
                "current_inventory": primitives.query_inventory(),
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
                "code_step_budget_count": code_step_budget_count,
                "final_success": execution.final_success,
                "final_verification": execution.final_verification,
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
                    "You are a coding agent controlling official visual household tasks through safe Python primitives. "
                    "Write one Python program that attempts to solve the household task by calling only the provided primitives. "
                    "This is a task-first embodied benchmark harness, not a repository exploration task. Do not read files, "
                    "build a general planner, or search for hidden ALFRED/THOR state. "
                    "Do not import modules, open files, call external services, inspect dunder attributes, or use hidden state. "
                    "The executor rejects import/from, try/except, with, class, lambda, raise, open/eval/exec, "
                    "getattr/setattr/globals/locals, dunder names, and unsafe builtin names such as dir. "
                    "Use direct top-level Python only: no def, no class, no helper functions, and no nested generated API. "
                    "Use plain Python loops and if statements. "
                    "Each action primitive is thin: one THOR native action, no path finding or expert replay. "
                    "Object primitives accept an objectId, a unique object type/name, or an object dict from detect_objects(). "
                    "If multiple visible objects match, use an objectId from detect_objects(). "
                    "Use observe(), detect_objects(), scan_scene(), search_scene(), explore_room(), locate_object(), "
                    "pickup_located_object(), place_held_object(), toggle_located_object(), open_located_object(), "
                    "remember_visible_objects(), recall_visible_objects(), ground_object(), query_object_state(), "
                    "query_inventory(), write_evidence(), and check_success() to adapt. "
                    "Prefer the performance primitives for hard visual tasks: locate_object searches, approaches, and "
                    "reacquires a target using only observed visual memory; pickup_located_object, place_held_object, "
                    "toggle_located_object, and open_located_object perform one household subgoal after bounded locate. "
                    "Use manual scan_scene/search_scene/explore_room only as a fallback or when you need extra evidence. "
                    "For visual search fallback, call scan_scene(queries=[...], rotations=4, include_tilts=True, remember=True) "
                    "or a short search_scene(queries=[...], rounds=1, scan_rotations=4, include_tilts=True, remember=True) "
                    "before spending many manual rotate/look steps. If local search does not find the target or light source, "
                    "call explore_room(queries=[...], step_budget=32 to 48, include_tilts=True, remember=True) once to cover more poses. "
                    "After any explore_room call, immediately call read_observed_spatial_map(query=target_query_or_None, limit=20) "
                    "then choose an approach/interact/check action before any additional broad search. "
                    "Do not spend the whole budget on repeated observe/look/rotate/scan calls. Once a plausible target, "
                    "receptacle, lamp, or appliance is visible, switch from search to approach and interaction. "
                    "Avoid spending most of the budget on search_scene(rounds=3) before explore_room. "
                    "Use recall_visible_objects(query) to reuse objectIds "
                    "seen earlier, but memory can be stale. For any object from recall_visible_objects(), search_scene(), or explore_room(), "
                    "Use read_observed_spatial_map(query=None) after a scan/explore loop or repeated blocked movement to inspect visited cells, "
                    "blocked moves, frontier hints, and seen-object memory; it is observed-only and does not provide a global map or path. "
                    "call approach_object(memory_obj, max_steps=3 to 5) first, then call detect_objects(query) or ground_object(query) "
                    "from the current view before pickup/toggle/open. If ground_object or query_object_state returns a dict with an "
                    "'error' key, do not pass that dict to another primitive and do not index ['objectId'] from it. "
                    "If search finds a likely landmark or receptacle but not the target, use approach_object(landmark, max_steps=2 or 3) "
                    "then scan again. Move_ahead only when needed; check action result.valid_action or result.error before assuming movement "
                    "or interaction succeeded. "
                    "Use object_query_hints: combine target_object_queries, light_source_queries, support_landmark_queries, "
                    "and container_queries in the first search_scene call. If the target remains unseen, approach a likely support "
                    "landmark or visible openable container, open it when appropriate, and scan the target aliases again. "
                    "Use broad object aliases from task guidance; for lights, try FloorLamp, DeskLamp, Lamp, and any "
                    "visible toggleable object if the exact type is absent. Never call pickup_object, put_object, "
                    "toggle_object, or query_object_state with None. "
                    "For look-at-object-in-light tasks, start with pickup_located_object(target_aliases, support_queries=landmarks, "
                    "search_budget=36, open_containers=True), then toggle_located_object(['FloorLamp','DeskLamp','Lamp'], on=True, "
                    "support_queries=landmarks, search_budget=36). If a wrapper returns ok=false, inspect its locate/events and "
                    "fall back to manual locate_object/scan/open. "
                    "For placing, prefer pickup_located_object(target, ...) then place_held_object(receptacle, ...). "
                    "Primitive action results are StepResult objects: use result.valid_action, result.error, result.success, "
                    "result.completed, result.score, or result.verification. Do not call result.get(...). "
                    "If task_context.step_by_step_instructions is non-empty, treat it as public natural-language subgoal guidance "
                    "from the benchmark and use it to order your search, navigation, and interaction attempts. It is not expert "
                    "low-level action replay, and it does not grant hidden object ids, masks, maps, paths, or simulator state. "
                    "Follow task_recipe exactly enough to make progress: after a bounded search phase, perform the next household "
                    "interaction subgoal and call check_progress_public()/check_success() rather than continuing to scan. "
                    "On repair attempts, inspect previous_feedback.evidence and previous wrapper results; do not replace a failed "
                    "wrapper call with long manual movement. If place_held_object fails with a native placement error, try one "
                    "different visible receptacle alias from the task/public instructions or stop and report progress. "
                    "Manual move_ahead/rotate repair should be at most three actions before returning to locate/place/toggle wrappers. "
                    "The task context is public only; do not assume access to expert plans, low-level trajectories, masks, or hidden object ids. "
                    "Return JSON only: {\"code\": \"<python code>\"}"
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    _jsonable(
                        {
                        "task_context": context,
                        "initial_observation": observation,
                        "max_env_steps": max_steps,
                        "primitive_cards": render_primitive_cards_for_prompt(),
                        "task_guidance": _visual_task_guidance(context),
                        "object_query_hints": _object_query_hints(context),
                        "w3_prompt_contract": _w3_prompt_contract(),
                        "task_recipe": _visual_task_recipe(context),
                        "search_budget_rules": {
                            "max_initial_scan_scene_calls": 1,
                            "max_explore_room_calls": 1,
                            "max_consecutive_manual_look_rotate": 6,
                            "after_search_must_choose": ["locate_object", "pickup_located_object", "place_held_object", "toggle_located_object", "open_located_object", "approach_object", "pickup_object", "put_object", "toggle_object", "open_object", "slice_object", "check_progress_public", "check_success"],
                        },
                        "previous_feedback": previous_feedback,
                        "example_style": "\n".join(
                            [
                                "ctx = get_task_context()",
                                "obs = observe()",
                                "landmarks = ['Shelf', 'Drawer', 'Cabinet', 'CounterTop', 'Table', 'Desk']",
                                "pickup = pickup_located_object('Book', aliases=['Book'], support_queries=landmarks, search_budget=36, open_containers=True)",
                                "write_evidence('pickup', pickup)",
                                "if pickup.get('ok'):",
                                "    place = place_held_object('Shelf', aliases=['Shelf', 'Desk', 'Table'], support_queries=landmarks, search_budget=36)",
                                "    write_evidence('place', place)",
                                "else:",
                                "    located = locate_object('Book', aliases=['Book'], support_queries=landmarks, search_budget=24, open_containers=True)",
                                "    write_evidence('locate_fallback', located)",
                                "write_evidence('progress', check_progress_public())",
                                "check_success()",
                            ]
                        ),
                        }
                    ),
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
        "Write one top-level Python solution for the current visual household task.",
        "Use only task_context, observation, primitive cards, remembered visible objects, evidence, and verifier feedback.",
        "For hard visual tasks, prefer bounded performance primitives before manual scan loops.",
        "Bound visual search; once a candidate object/receptacle/light is visible, approach/reacquire and interact.",
        "Never use stale memory directly for high-risk interactions: approach first, then reacquire from current view.",
        "Use check_progress_public()/check_success() after each subgoal chain to avoid wasting the remaining budget.",
    ]


def _visual_task_recipe(context: dict[str, Any]) -> dict[str, Any]:
    task_type = str(context.get("task_type") or "").lower()
    goal = _context_language_text(context).lower()
    if "look_at_obj_in_light" in task_type or "lamp" in goal or "light" in goal:
        steps = [
            "build aliases for target object and light source",
            "call pickup_located_object(target aliases, support_queries=landmarks, open_containers=True)",
            "call toggle_located_object(light aliases, on=True, support_queries=landmarks)",
            "if a wrapper returns ok=false, inspect locate/events and run one targeted locate_object or open_located_object fallback",
            "check_progress_public and check_success",
        ]
    elif "pick_clean" in task_type or "clean" in goal or "rinse" in goal:
        steps = [
            "pickup_located_object(target aliases, support_queries=likely containers/supports)",
            "locate/open/toggle sink or faucet using locate_object/toggle_located_object as available",
            "place_held_object(final receptacle aliases) after cleaning interaction if progress allows",
            "check_success",
        ]
    elif "pick_heat" in task_type or "heat" in goal or "hot" in goal:
        steps = [
            "pickup_located_object(target aliases, support_queries=likely containers/supports)",
            "open_located_object(microwave aliases), then place_held_object(microwave aliases)",
            "toggle_located_object(microwave aliases, on=True) if needed by the task",
            "retrieve/place target in final receptacle if primitive progress indicates this subgoal remains",
            "check_success",
        ]
    elif "pick_cool" in task_type or "cool" in goal or "chilled" in goal:
        steps = [
            "pickup_located_object(target aliases, support_queries=likely containers/supports)",
            "open_located_object(fridge aliases), then place_held_object(fridge aliases)",
            "retrieve/place target in final receptacle if primitive progress indicates this subgoal remains",
            "check_success",
        ]
    elif "pick_two" in task_type or "two" in goal:
        steps = [
            "repeat for two matching target objects",
            "pickup_located_object target with observed search and open_containers=True",
            "place_held_object receptacle after each pickup",
            "verify progress before searching for second target",
        ]
    elif "pick" in task_type or "place" in task_type or "move" in goal:
        steps = [
            "pickup_located_object(target aliases, support_queries=likely supports/containers, open_containers=True)",
            "place_held_object(receptacle aliases, support_queries=likely room landmarks)",
            "if either wrapper fails, run locate_object fallback and interact with current_object only when visible_now is true",
            "check_success",
        ]
    else:
        steps = [
            "scan once for all object_query_hints",
            "approach the first plausible subgoal object",
            "perform the household interaction required by task_context",
            "write evidence and check_success",
        ]
    return {
        "benchmark": str(context.get("benchmark") or "ALFRED/ALFWorld Visual"),
        "task_type": task_type,
        "steps": steps,
        "anti_loop_rules": [
            "Do not call look/rotate more than six times in a row.",
            "Do not call explore_room more than once.",
            "Prefer pickup_located_object/place_held_object/toggle_located_object/open_located_object over hand-written long navigation loops.",
            "After a remembered object is selected, approach then detect/ground again before interaction.",
        ],
    }


def _visual_task_guidance(context: dict[str, Any]) -> str:
    task_type = str(context.get("task_type") or "").lower()
    goal = _context_language_text(context).lower()
    if task_type == "look_at_obj_in_light" or "light" in goal or "lamp" in goal:
        return (
            "For look-at-object-in-light tasks: identify the target object and a light source. First try "
            "pickup_located_object(target, aliases=target_aliases, support_queries=landmarks, search_budget=36, "
            "open_containers=True), then toggle_located_object('Lamp', aliases=['FloorLamp','DeskLamp','Lamp'], "
            "on=True, support_queries=landmarks, search_budget=36). These wrappers search, approach, and reacquire "
            "using only observed visual memory. If either wrapper returns ok=false, inspect locate/events, run one "
            "targeted locate_object/open_located_object fallback, then interact only with current-view objects. Do not "
            "spend the whole budget on scanning one exact object type or on search_scene(rounds=3). If a variable is "
            "None, do not call an object primitive with it."
        )
    steps = context.get("step_by_step_instructions") or []
    if isinstance(steps, list) and steps:
        return (
            "Use the public step-by-step language instructions as a subgoal order. They tell you likely landmarks, "
            "containers, receptacles, and interaction order, but they are not hidden simulator state. For hard visual "
            "subgoals prefer pickup_located_object, place_held_object, toggle_located_object, open_located_object, "
            "then fall back to bounded locate_object/manual scan only if a wrapper fails. Interact only with visible "
            "or inventory objects, and check success."
        )
    return "Use visible metadata only, scan locally with bounded rotate/look loops, interact only with visible or inventory objects, and check success."


def _object_query_hints(context: dict[str, Any]) -> dict[str, list[str]]:
    goal = _context_language_text(context).lower()
    hints: dict[str, list[str]] = {}
    support_landmarks = [
        "Shelf",
        "Drawer",
        "Cabinet",
        "Desk",
        "Table",
        "CoffeeTable",
        "SideTable",
        "Dresser",
        "CounterTop",
        "Sofa",
        "ArmChair",
        "Bed",
    ]
    container_queries = [
        "Drawer",
        "Cabinet",
        "Fridge",
        "Microwave",
        "Safe",
        "Box",
    ]
    if "credit card" in goal or "creditcard" in goal:
        hints["target_object_queries"] = ["CreditCard", "credit card", "card"]
        hints["support_landmark_queries"] = support_landmarks
        hints["container_queries"] = container_queries
    if "towel" in goal:
        hints["target_object_queries"] = ["Towel", "HandTowel", "PaperTowel", "towel"]
        hints["support_landmark_queries"] = ["TowelHolder", "SinkBasin", "CounterTop", "Shelf", "Drawer", "Cabinet"]
        hints["container_queries"] = ["Drawer", "Cabinet"]
    if "book" in goal:
        hints["target_object_queries"] = ["Book", "book"]
        hints["support_landmark_queries"] = ["Shelf", "Desk", "Table", "CoffeeTable", "Sofa", "Bed", "Dresser"]
        hints["container_queries"] = ["Drawer", "Cabinet"]
    if "key" in goal:
        hints["target_object_queries"] = ["KeyChain", "key", "Key"]
        hints["support_landmark_queries"] = support_landmarks
        hints["container_queries"] = container_queries
    if "remote" in goal:
        hints["target_object_queries"] = ["RemoteControl", "remote"]
        hints["support_landmark_queries"] = ["Sofa", "ArmChair", "Table", "CoffeeTable", "TVStand", "Shelf", "Drawer", "Cabinet"]
        hints["container_queries"] = ["Drawer", "Cabinet"]
    if "lamp" in goal or "light" in goal:
        hints["light_source_queries"] = ["FloorLamp", "DeskLamp", "Lamp", "light"]
        hints.setdefault("support_landmark_queries", support_landmarks)
    if "shelf" in goal:
        hints.setdefault("support_landmark_queries", support_landmarks)
        hints["explicit_receptacle_queries"] = ["Shelf"]
    if "drawer" in goal:
        hints.setdefault("container_queries", container_queries)
        hints["explicit_receptacle_queries"] = ["Drawer"]
    if "cabinet" in goal:
        hints.setdefault("container_queries", container_queries)
        hints["explicit_receptacle_queries"] = ["Cabinet"]
    if "table" in goal or "desk" in goal:
        hints.setdefault("support_landmark_queries", support_landmarks)
        hints["explicit_receptacle_queries"] = ["Desk", "Table", "CoffeeTable", "SideTable"]
    return hints


def _context_language_text(context: dict[str, Any]) -> str:
    parts = [str(context.get("goal_text") or "")]
    steps = context.get("step_by_step_instructions") or []
    if isinstance(steps, list):
        parts.extend(str(item) for item in steps)
    return " ".join(parts)


def _is_step_budget_exceeded(exception: str | None) -> bool:
    return bool(exception and exception.startswith("CodeStepBudgetExceeded"))


def _code_timeout_seconds() -> int:
    raw = os.environ.get("ALFRED_CODE_TIMEOUT_SECONDS", "60")
    try:
        return int(raw)
    except ValueError:
        return 60


def _api_timeout_seconds() -> int:
    raw = os.environ.get("ALFRED_API_TIMEOUT_SECONDS", "90")
    try:
        return max(5, int(raw))
    except ValueError:
        return 90


def _max_code_attempts() -> int:
    raw = os.environ.get("ALFRED_MAX_CODE_ATTEMPTS", "2")
    try:
        return max(1, int(raw))
    except ValueError:
        return 2


def _primitive_call_budget(max_steps: int) -> int:
    raw = os.environ.get("ALFRED_MAX_PRIMITIVE_CALLS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(80, max_steps * 6)


def _redact(value: str) -> str:
    value = re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", value)
    return value[:2000]


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)
