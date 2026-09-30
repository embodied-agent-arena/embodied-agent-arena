from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import certifi

from .agents import AgentRunResult
from .code_executor import default_primitive_call_budget, execute_agent_code
from .env_file import load_env_file
from .primitive_cards import get_primitive_cards, render_primitive_cards_for_prompt
from .primitives import ScienceWorldTextPrimitives

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekV4ActionAgent:
    name = "deepseek_v4"

    def __init__(self, root_dir: Path, model: str | None = None):
        load_env_file(root_dir / ".env")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        self.api_timeout_seconds = _api_timeout_seconds()
        self.ssl_context = _build_ssl_context()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it in your shell.")

    def run(self, primitives: ScienceWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        steps = 0
        history: list[dict[str, Any]] = []
        while steps < max_steps:
            status = primitives.check_success()
            if status["success"]:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if status["done"]:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)

            context = primitives.get_task_context()
            observation = primitives.observe_text_world()
            actions = primitives.list_actions()
            if not actions:
                return AgentRunResult(stopped_reason="no_actions", steps_attempted=steps)

            action = self._choose_action(context, observation, actions, history)
            if action not in actions:
                primitives.write_evidence("deepseek_invalid_action", {"action": action, "available": actions})
                return AgentRunResult(stopped_reason="model_invalid_action", steps_attempted=steps)

            result = primitives.step_text_action(action)
            history.append({"action": action, "observation_after": result.observation_after, "score": result.score})
            history = history[-10:]
            steps += 1
            if result.success:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if result.done:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)
        return AgentRunResult(stopped_reason="max_steps", steps_attempted=steps)

    def _chat_completion(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.api_timeout_seconds, context=self.ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {_redact(body)}") from exc

    def _choose_action(
        self,
        context: dict[str, Any],
        observation: str,
        actions: list[str],
        history: list[dict[str, Any]],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are controlling ScienceWorld. Choose exactly one grounded action copied verbatim "
                    "from valid_actions. Do not invent actions. Prefer actions that make progress toward "
                    "the task description. Return JSON only: {\"action\": \"<one valid action>\"}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "observation": observation,
                        "action_history": history,
                        "valid_actions": actions,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        data = self._chat_completion(messages)
        return _parse_action(data["choices"][0]["message"]["content"])


class DeepSeekV4CodeAgent(DeepSeekV4ActionAgent):
    name = "deepseek_v4_code"

    def run(self, primitives: ScienceWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        context = primitives.get_task_context()
        observation = primitives.observe_text_world()
        actions = primitives.list_actions()
        execution = None
        feedback: dict[str, Any] | None = None
        max_attempts = _max_code_attempts()
        for attempt_index in range(1, max_attempts + 1):
            code = self._generate_code(context, observation, actions, max_steps, feedback=feedback)
            primitives.record_harness_event(
                "agent_code_generated",
                {
                    "code_attempt_index": attempt_index,
                    "executed_code": code,
                    "primitive_cards": get_primitive_cards(),
                    "has_repair_feedback": feedback is not None,
                },
                side_effect=False,
            )
            execution = execute_agent_code(
                primitives,
                code,
                attempt_index=attempt_index,
                timeout_seconds=_code_timeout_seconds(),
                max_env_steps=max_steps,
                max_primitive_calls=default_primitive_call_budget(max_steps),
            )
            if execution.final_success:
                break
            if (
                execution.timed_out
                or execution.primitive_budget_exceeded
                or execution.trace_limit_exceeded
                or (execution.exception and execution.exception.startswith("CodeStepBudgetExceeded"))
            ):
                break
            feedback = {
                "previous_code": code,
                "previous_exception": execution.exception,
                "previous_stdout": execution.stdout[-2000:],
                "previous_stderr": execution.stderr[-2000:],
                "current_verification": execution.final_verification,
                "current_observation": primitives.observe_text_world(),
                "current_valid_actions": primitives.list_actions()[:120],
                "repair_instruction": (
                    "The previous code did not solve the task. Do not repeat the same navigation/action loop. "
                    "Use current_valid_actions exactly, prefer actions that change score/state, and stop invalid repeats."
                ),
            }
        if execution is None:
            raise RuntimeError("DeepSeek code agent did not execute any code attempts.")
        if execution.final_success:
            stopped_reason = "success"
        elif execution.timed_out:
            stopped_reason = "code_timeout"
        elif execution.primitive_budget_exceeded:
            stopped_reason = "primitive_call_budget"
        elif execution.trace_limit_exceeded:
            stopped_reason = "trace_limit"
        elif execution.exception and execution.exception.startswith("CodeStepBudgetExceeded"):
            stopped_reason = "max_steps"
        elif execution.exception:
            stopped_reason = "code_exception"
        else:
            stopped_reason = "not_success"
        return AgentRunResult(
            stopped_reason=stopped_reason,
            steps_attempted=primitives.metrics["env_steps"],
            metrics=execution.metrics,
        )

    def _generate_code(
        self,
        context: dict[str, Any],
        observation: str,
        actions: list[str],
        max_steps: int,
        feedback: dict[str, Any] | None = None,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an embodied coding agent for ScienceWorld. Write Python code that calls only "
                    "the provided primitives. Do not use import statements. Do not define functions/classes. "
                    "This is a task-first benchmark harness, not a repo exploration task: solve the current ScienceWorld "
                    "instance by choosing exact grounded environment actions through primitives only. "
                    "Do not open files, call external services, use try/except, with, raise, lambda, exit(), quit(), "
                    "SystemExit, or hidden/oracle state. "
                    "Write direct top-level Python code only. observe_text_world() and look() return strings, not dicts. "
                    "StepResult has fields .valid_action, .success, .done, .score, .observation_after, and .error. "
                    "list_actions() returns grounded valid actions. filter_actions(...) only filters that current list; "
                    "it does not act or plan. Every step_text_action(action) argument must be copied exactly from "
                    "list_actions() or filter_actions(...). filter_actions(...) returns a list: never pass the list "
                    "itself to step_text_action; select one string such as candidates[0]. Never make template strings "
                    "such as f'go to {room}'. "
                    "Do not return a read-only-only program: unless check_success() is already success/done or there "
                    "are no valid actions, make at least one relevant step_text_action attempt. "
                    "After every step_text_action call, inspect result.valid_action and result.error; if valid_action is "
                    "False, choose a different current action rather than repeating it. If result.done is True, stop "
                    "acting immediately and only call check_success(); some wrong but valid actions terminate the task. "
                    "Start by calling inspect_current_state(query=None). Use state['current_location'] to decide where "
                    "you are; never infer the current room merely because a door or exit name appears in observation text. "
                    "Before go/teleport actions, compare the target room with state['current_location']; do not repeat "
                    "a navigation action to the room you are already in. Use state['loop_warnings'], "
                    "state['substance_candidates'], state['action_groups'], and state['query_actions'] to choose exact "
                    "grounded actions before falling back to broad list_actions scans. When you need a generic focus "
                    "candidate, prefer state['action_groups'].get('focus_objects', []) over raw focus actions so you "
                    "do not focus on agent, air, inventory, doors, or room names. "
                    "ScienceWorld tasks often require focusing on objects, going to rooms, picking up objects, "
                    "moving objects to devices/containers, activating devices, and checking score. "
                    "Follow task_recipe. For experiment/state-change tasks, do not just navigate: focus on the substance, "
                    "move it into the relevant container/device, activate or use the device, then check score/success. "
                    "For boil/melt/freeze tasks, never default to going to the kitchen first. Water or other substances "
                    "often start in the bathroom, fountain, toilet, sink, foundry, or another current room. Inspect the "
                    "current state and focus an available substance candidate before any room change when possible. "
                    "For find-* tasks, first focus on a thing matching the requested category, then move that same "
                    "thing to the named box/location using an exact current action; do not move unrelated food, cups, "
                    "containers, tables, or rooms just because they mention the target color. After focusing, derive "
                    "the focused thing name from the chosen focus action and search exact move actions that contain "
                    "both that thing name and the target box/location. "
                    "Use write_evidence for observations and useful hypotheses. End by calling check_success(). "
                    "Return JSON only: {\"code\": \"<python code>\"}."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "initial_observation": observation,
                        "initial_valid_actions": actions,
                        "max_env_steps": max_steps,
                        "max_primitive_calls": default_primitive_call_budget(max_steps),
                        "primitive_cards": render_primitive_cards_for_prompt(),
                        "w3_prompt_contract": _w3_prompt_contract(),
                        "task_guidance": _task_guidance(context),
                        "task_recipe": _scienceworld_task_recipe(context),
                        "execution_rules": [
                            "Every step_text_action argument must be copied exactly from current list_actions/filter_actions.",
                            "Do not repeat an invalid/parser-no-match action.",
                            "After focus/move/use actions, inspect score/state before choosing the next action.",
                            "If repair_feedback is present, change strategy instead of replaying previous_code.",
                        ],
                        "initial_action_hints": _initial_action_hints(context, actions),
                        "repair_feedback": feedback,
                        "example_style": "\n".join(
                            [
                                "ctx = get_task_context()",
                                "write_evidence('task', ctx.get('goal_text'))",
                                "state = inspect_current_state(limit=40)",
                                "write_evidence('initial_state', state)",
                                "for step in range(20):",
                                "    status = check_success()",
                                "    if status['success'] or status['done']:",
                                "        break",
                                "    state = inspect_current_state(limit=40)",
                                "    if state.get('loop_warnings'):",
                                "        write_evidence('loop_warnings', state['loop_warnings'])",
                                "    actions = list_actions()",
                                "    chosen = None",
                                "    candidates = filter_actions(startswith='focus on ', limit=30)",
                                "    if not candidates:",
                                "        candidates = filter_actions(startswith='go to ', limit=30)",
                                "    if candidates:",
                                "        chosen = candidates[0]",
                                "    else:",
                                "        break",
                                "    result = step_text_action(chosen)",
                                "    if not result.valid_action:",
                                "        write_evidence('bad_action', {'action': chosen, 'error': result.error})",
                                "        break",
                                "    write_evidence('last_action', {'action': chosen, 'score': result.score})",
                                "    if result.done:",
                                "        break",
                                "check_success()",
                            ]
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        data = self._chat_completion(messages)
        return _parse_code(data["choices"][0]["message"]["content"])


def _parse_action(content: str) -> str:
    try:
        return str(json.loads(content)["action"])
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise RuntimeError("DeepSeek response did not contain JSON action.")
        return str(json.loads(match.group(0))["action"])


def _w3_prompt_contract() -> list[str]:
    return [
        "Write one top-level Python solution for the current ScienceWorld task.",
        "Use only task_context, observation, primitive cards, valid action lists, and primitive feedback.",
        "Every environment-changing action must be an exact string from the current valid action set.",
        "After invalid/no-match/done feedback, do not repeat that action; inspect state and choose a different exact action.",
        "Use evidence and score/verifier feedback to decide whether the current experiment is progressing.",
    ]


def _scienceworld_task_recipe(context: dict[str, Any]) -> dict[str, Any]:
    task_type = str(context.get("task_type") or context.get("task_name") or "").lower()
    goal = str(context.get("goal_text") or "").lower()
    if task_type.startswith("find-") or "find" in goal:
        steps = [
            "inspect_current_state(query for requested category and target box)",
            "choose one high-confidence target object from current focus actions",
            "step exact focus action",
            "keep the same object identity/name",
            "move or pick that object using exact current action",
            "go/teleport to target box/location only after object is selected",
            "move the same object to the target box/location",
            "check_success",
        ]
    elif any(word in task_type or word in goal for word in ("boil", "melt", "freeze", "change-the-state")):
        heat_or_cold = "cold" if ("freeze" in task_type or "freeze" in goal) else "heat"
        device_examples = ["freezer", "fridge", "ice", "cold room"] if heat_or_cold == "cold" else ["stove", "oven", "burner", "fire", "foundry", "furnace", "hot plate"]
        steps = [
            "identify the named substance and exact focus action",
            "focus the substance before leaving the room if available; do not default to kitchen",
            "find a relevant device/container: " + ", ".join(device_examples),
            "move/place substance into or near that device using exact current action",
            "activate/use/open device as current actions allow",
            "check score/verifier after each device action",
        ]
    elif any(word in task_type or word in goal for word in ("thermometer", "measure", "temperature")):
        steps = [
            "find/pick up measuring instrument if needed",
            "focus or place it with the target substance",
            "perform exact use/measure action",
            "move target/result to requested box/location",
            "check_success",
        ]
    elif any(word in task_type or word in goal for word in ("electric", "circuit", "conduct")):
        steps = [
            "inspect available electrical components",
            "focus/pick relevant object",
            "connect/move/use components with exact valid actions",
            "activate/test circuit if action exists",
            "check_success",
        ]
    else:
        steps = [
            "inspect_current_state(query=None)",
            "use task_guidance and action_groups to choose an exact grounded action",
            "execute one action that changes object/device state",
            "inspect score/state",
            "continue until success/done or budget",
        ]
    return {
        "benchmark": "ScienceWorld",
        "task_type": task_type,
        "steps": steps,
        "anti_loop_rules": [
            "Never repeat a parser-no-match action.",
            "Do not teleport/go to the current_location.",
            "Do not navigate to kitchen just because the task says boil; first inspect/focus current water/substance candidates.",
            "Do not pass a list returned by filter_actions directly to step_text_action.",
            "Prefer actions that mention the same target object/substance across focus, move, and device steps.",
        ],
    }


def _task_guidance(context: dict[str, Any]) -> str:
    task_type = str(context.get("task_type") or context.get("task_name") or "").lower()
    goal = str(context.get("goal_text") or "").lower()
    if task_type.startswith("find-living") or "living thing" in goal:
        return (
            "For this find-living-thing task: wrong focus actions can immediately end the episode with a large penalty, "
            "so do not focus on ambiguous food, fruit, tools, rooms, doors, containers, liquids, or furniture. In "
            "particular, orange, apple, banana, potato, cup, bowl, table, agent, inventory, doors, and colored boxes are "
            "not acceptable living-thing candidates. Prefer high-confidence animal/insect/fish/amphibian/reptile/bird "
            "names such as bee, blue jay, dove, butterfly, frog, lizard, fish, cat, dog, mouse, rabbit, rat, snake, or "
            "turtle; plant/tree names are acceptable only when the object name itself says plant/tree, such as adult pea "
            "plant, adult apple tree, or adult cherry tree. "
            "Start with inspect_current_state(query=['bee','blue jay','dove','butterfly','frog','lizard','fish','cat',"
            "'dog','mouse','rabbit','snake','turtle','animal','bird','plant','tree']). If no high-confidence target is visible in "
            "query_actions or visible_entries, explore greenhouse or outside first using an exact current go/teleport action, then "
            "inspect again. After focusing a high-confidence target, keep that same object name and move it to the "
            "target box/location with an exact current action. Stop immediately if any action returns done."
        )
    if task_type.startswith("find-"):
        return (
            "For this find-* task: do not go to the target box room first. First inspect the initial/current observation "
            "and current focus actions, then choose a category-matching thing before moving it. Explore rooms only if "
            "no plausible target is visible. For animal goals, plausible names can include bird-like or insect-like "
            "names such as blue jay, dove, butterfly, frog, lizard, fish, cat, dog, mouse, or rabbit. "
            "Execute one exact focus action. If score improves but success is false, keep the same focused object name. "
            "Then first move or pick up that same object to inventory when such an action exists, then go/teleport the "
            "agent to the target room, then execute an exact 'move <same object> to <target box/location>' action. "
            "Do not move orange, cup, bowl, table, inventory, or a colored liquid merely because the goal mentions a "
            "colored box. After every room change, call inspect_current_state() and use current_location; door names in "
            "the observation are exits, not proof that you are already in that room."
        )
    if task_type in {"boil", "melt", "freeze"} or any(word in goal for word in ("boil", "melt", "freeze")):
        if "freeze" in task_type or "freeze" in goal:
            return (
            "For freeze tasks: first locate and focus the named substance if an exact focus action exists. If the "
            "visible text says 'substance called water' but actions call it 'substance in <container>', use "
            "inspect_current_state(... )['substance_candidates'] and choose exact matching_actions such as "
            "'focus on substance in toilet' or 'focus on substance in fountain'; do not navigate away from a room "
            "that currently has a target substance candidate before trying the exact focus action. Prefer "
            "cold devices/locations only: freezer, fridge/refrigerator, cold room, outside, ice, or other explicitly "
                "cold containers. Avoid foundry, stove, oven, burner, fire pit, hot plate, bathtub, sink, toilet, and "
                "unrelated objects. Use inspect_current_state(query=[substance, 'freezer', 'fridge', 'cold', 'ice']) "
                "after every room change; if state['current_location'] already equals a target room, do not teleport "
                "there again. After each step_text_action, call check_success()/get_score_state and stop if done."
            )
        return (
            "For boil/melt matter-state tasks: first locate and focus the named substance if an exact focus action exists. "
            "If the visible text says 'substance called water' but actions call it 'substance in <container>', use "
            "inspect_current_state(... )['substance_candidates'] and choose exact matching_actions such as "
            "'focus on substance in toilet' or 'focus on substance in fountain'; do not navigate away from a room "
            "that currently has a target substance candidate before trying the exact focus action. "
            "Prefer heat devices/locations only: stove, oven, burner, fire pit, foundry/furnace, hot plate, or other "
            "explicitly hot containers. Avoid bathtub, sink, toilet, freezer, fridge, cold room, and unrelated objects. "
            "Use inspect_current_state(query=[substance, 'stove', 'oven', 'burner', 'fire', 'foundry', 'furnace', 'hot']) "
            "after every room change; if state['current_location'] already equals a target room, do not teleport there "
            "again. Do not spend steps picking unrelated objects such as soap, stopwatch, or table. If the substance appears "
            "inside a container but no direct focus action exists, inspect current action groups for container/device "
            "actions such as open, activate, move, dunk, pour, or put; choose only exact grounded actions. "
            "After each action, call get_score_state() or check_success(); a higher score is progress but not success "
            "unless success or done is true."
        )
    if "thermometer" in goal or task_type == "use-thermometer":
        return (
            "For thermometer tasks: write code without imports, try/except, raise, SystemExit, exit(), or quit(); "
            "use natural script completion to stop. Repeatedly inspect_current_state(query=['thermometer', 'unknown substance', "
            "'unknown substance b', 'red box', 'green box']) after room changes. Focus or pick up the thermometer "
            "using exact current actions, then focus unknown substance B. Focusing is not carrying: before leaving the "
            "living room, execute an exact 'pick up unknown substance' or 'move unknown substance to inventory' action "
            "if one is available, then move it to the correct box using exact current move actions in the bathroom. "
            "Do not assume the substance moved with you; verify inventory and current action groups before moving to "
            "the bathroom. If inspect_current_state()['current_location'] is "
            "already the target room, do not repeat the same go/teleport action. If a move-to-box action does not "
            "change score, do not repeat the same box move; inspect evidence and choose a different grounded action "
            "or stop when done."
        )
    if "paint" in task_type or "paint" in goal:
        return (
            "For paint-mixing tasks: only execute exact current dunk/mix/pour/focus actions returned by filter_actions; "
            "avoid unrelated move actions unless the object is clearly the requested paint or container."
        )
    return "Use exact current actions only, re-list or re-filter after every environment step, and stop if the verifier is done."


def _initial_action_hints(context: dict[str, Any], actions: list[str]) -> dict[str, Any]:
    goal = str(context.get("goal_text") or "")
    task_type = str(context.get("task_type") or context.get("task_name") or "").lower()
    hints: dict[str, Any] = {
        "focus_actions_sample": _prefix_sample(actions, "focus on ", 60),
        "go_or_teleport_actions_sample": _prefix_sample(actions, "go to ", 30)
        + _prefix_sample(actions, "teleport to ", 30),
    }
    if task_type.startswith("find-animal"):
        animal_terms = [
            "blue jay",
            "dove",
            "butterfly",
            "bird",
            "cat",
            "dog",
            "fish",
            "frog",
            "lizard",
            "mouse",
            "rabbit",
            "rat",
            "snake",
            "turtle",
        ]
        hints["category_focus_candidates_from_initial_actions"] = [
            action for action in actions if action.startswith("focus on ") and any(term in action.lower() for term in animal_terms)
        ][:30]
    box_terms = _target_box_terms(goal)
    if box_terms:
        hints["initial_move_actions_to_target_box_sample"] = [
            action
            for action in actions
            if action.startswith("move ") and all(term in action.lower() for term in box_terms)
        ][:50]
        hints["target_box_terms"] = box_terms
    return hints


def _prefix_sample(actions: list[str], prefix: str, limit: int) -> list[str]:
    return [action for action in actions if action.startswith(prefix)][:limit]


def _target_box_terms(goal: str) -> list[str]:
    lowered = goal.lower()
    colors = ["red", "green", "blue", "orange", "yellow", "black", "white", "purple"]
    for color in colors:
        if f"{color} box" in lowered:
            return [color, "box"]
    if "box" in lowered:
        return ["box"]
    return []


def _parse_code(content: str) -> str:
    try:
        parsed = json.loads(content)
        if "code" in parsed:
            return str(parsed["code"]).strip()
    except Exception:
        pass

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if "code" in parsed:
                return str(parsed["code"]).strip()
        except Exception:
            pass

    fenced = re.search(r"```(?:python)?\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()


def _redact(value: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", value)


def _build_ssl_context() -> ssl.SSLContext:
    ca_bundle = os.environ.get("DEEPSEEK_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    return ssl.create_default_context(cafile=certifi.where())


def _code_timeout_seconds() -> int:
    raw = os.environ.get("SCIENCEWORLD_CODE_TIMEOUT_SECONDS", "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def _api_timeout_seconds() -> int:
    raw = os.environ.get("DEEPSEEK_API_TIMEOUT_SECONDS", "90").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 90


def _max_code_attempts() -> int:
    raw = os.environ.get("SCIENCEWORLD_MAX_CODE_ATTEMPTS", "2").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 2
