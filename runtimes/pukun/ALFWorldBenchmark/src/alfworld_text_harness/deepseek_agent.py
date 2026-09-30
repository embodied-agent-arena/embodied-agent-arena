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
from .code_executor import execute_agent_code
from .env_file import load_env_file
from .primitive_cards import get_primitive_cards, render_primitive_cards_for_prompt
from .primitives import ALFWorldTextPrimitives

DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekV4ActionAgent:
    name = "deepseek_v4"

    def __init__(self, root_dir: Path, model: str | None = None):
        load_env_file(root_dir / ".env")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        self.ssl_context = _build_ssl_context()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it in your shell.")

    def run(self, primitives: ALFWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        steps = 0
        context = primitives.get_task_context()
        official_task = None
        history: list[dict[str, Any]] = []
        while steps < max_steps:
            status = primitives.check_success()
            if status["success"]:
                return AgentRunResult(stopped_reason="success", steps_attempted=steps)
            if status["done"]:
                return AgentRunResult(stopped_reason="done", steps_attempted=steps)

            observation = primitives.observe_text_state()
            official_task = _extract_official_task(observation) or official_task
            actions = primitives.list_actions()
            if not actions:
                return AgentRunResult(stopped_reason="no_actions", steps_attempted=steps)

            action = self._choose_action(context, official_task, observation, actions, history)
            if action not in actions:
                primitives.write_evidence("deepseek_invalid_action", {"action": action, "available": actions})
                return AgentRunResult(stopped_reason="model_invalid_action", steps_attempted=steps)

            result = primitives.step_text_action(action)
            history.append(
                {
                    "action": action,
                    "observation_after": result.observation_after,
                    "success": result.success,
                    "done": result.done,
                }
            )
            history = history[-8:]
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
        if self.model == "deepseek-v4-pro":
            payload["reasoning_effort"] = "medium"
            payload["thinking"] = {"type": "enabled"}

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
            with urllib.request.urlopen(request, timeout=90, context=self.ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {_redact(body)}") from exc

    def _choose_action(
        self,
        context: dict[str, Any],
        official_task: str | None,
        observation: str,
        actions: list[str],
        history: list[dict[str, Any]],
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are controlling an ALFWorld text environment. "
                    "The official task extracted from the environment observation is authoritative; "
                    "the human goal_text in task_context may be noisy. "
                    "Choose exactly one action copied verbatim from admissible_actions. "
                    "Use action_history to avoid undoing progress, repeating examine/look without benefit, "
                    "or moving an object away from its required receptacle. "
                    "If you are holding the target object and a target receptacle/location is reachable, go there or place/move the object there. "
                    'Return JSON only: {"action": "<one admissible action>"}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "official_task": official_task,
                        "observation": observation,
                        "action_history": history,
                        "admissible_actions": actions,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        data = self._chat_completion(messages)
        content = data["choices"][0]["message"]["content"]
        return _parse_action(content)


class DeepSeekV4CodeAgent(DeepSeekV4ActionAgent):
    name = "deepseek_v4_code"

    def run(self, primitives: ALFWorldTextPrimitives, max_steps: int) -> AgentRunResult:
        context = primitives.get_task_context()
        observation = primitives.observe_text_state()
        actions = primitives.list_actions()
        code = self._generate_code(context, observation, actions, max_steps)
        primitives.record_harness_event(
            "agent_code_generated",
            {
                "code_attempt_index": 1,
                "executed_code": code,
                "primitive_cards": get_primitive_cards(),
            },
            side_effect=False,
        )
        execution = execute_agent_code(
            primitives,
            code,
            attempt_index=1,
            timeout_seconds=_code_timeout_seconds(),
            max_env_steps=max_steps,
            max_primitive_calls=_code_max_primitive_calls(max_steps),
        )
        if execution.final_success:
            stopped_reason = "success"
        elif execution.primitive_budget_exceeded:
            stopped_reason = "primitive_call_budget"
        elif execution.timed_out:
            stopped_reason = "code_timeout"
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
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an embodied coding agent for ALFWorld Text-only. "
                    "Write Python code that calls only the provided primitives to solve the task. "
                    "This is a task-first benchmark harness, not a repository exploration task: do not read files, "
                    "create tools, or browse for extra information. Solve the current environment instance through "
                    "the primitive calls only. "
                    "Do not import modules, open files, call external services, or use hidden/oracle state. "
                    "The official task in task_context/observation is authoritative. "
                    "Use list_actions() and observe_text_state() inside the code to adapt after each step. "
                    "Follow the provided task_recipe: after each navigation or object action, refresh list_actions() "
                    "and either execute the next recipe action or call check_success(); do not spend the budget on "
                    "repeated look/examine calls once the needed object or receptacle is known. "
                    "Household wrappers are thin: each call executes at most one admissible native action. "
                    "For any StepResult, result.valid_action tells whether that primitive executed a native action; "
                    "result.success means the whole ALFWorld task is solved, not that an intermediate pickup/open succeeded. "
                    "After go_to(location), use result.observation_after and list_actions(); do not immediately call look(), "
                    "because look may replace the arrival observation and waste a step. "
                    "To find an object, check current admissible actions for a matching take command, then call pickup_object. "
                    "If the target object location is unknown, search visible locations by calling go_to(location), "
                    "observing, and then pickup_object(target) only when the object is visible or admissible. "
                    "Use goal_hints to hardcode the correct object, receptacle, count, and required condition in your code. "
                    "If a wrapper returns valid_action=False or no_match/ambiguous, immediately inspect list_actions() "
                    "for numbered variants such as '<object> 1' or '<receptacle> 2', then try a different location or alias. "
                    "For clean tasks, pick the object, navigate to a sink or sinkbasin, call clean_object, then place it. "
                    "For heat/hot tasks, pick the object, navigate to a microwave, call heat_object, then place it. "
                    "For cool tasks, pick the object, navigate to a fridge, call cool_object, then place it. "
                    "For light/examine tasks, find the object and desklamp, use/toggle the desklamp if admissible, then examine the object if admissible. "
                    "For two-object tasks, repeat pick/place for two matching objects if possible. "
                    "After holding the target object, navigate to a matching target receptacle and call place_object. "
                    "Do not require an admissible command starting with 'put' before placing; ALFWorld often uses "
                    "'move <object> to <receptacle>', and place_object handles put/move/place commands. "
                    "End by calling check_success(). "
                    'Return JSON only: {"code": "<python code>"}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "goal_hints": _goal_hints(str(context.get("goal_text", ""))),
                        "w3_prompt_contract": _w3_prompt_contract(),
                        "task_recipe": _alfworld_task_recipe(context, observation),
                        "execution_rules": [
                            "Act early: after finding a plausible target, pickup/condition/place instead of continuing to look.",
                            "Refresh list_actions() after every side-effect primitive.",
                            "Use numbered ALFWorld names from admissible actions when base names are ambiguous.",
                            "Call check_success() after each completed subgoal chain.",
                        ],
                        "initial_observation": observation,
                        "initial_admissible_actions": actions,
                        "max_env_steps": max_steps,
                        "primitive_cards": render_primitive_cards_for_prompt(),
                        "example_style": "\n".join(
                            [
                                "ctx = get_task_context()",
                                "goal = ctx.get('goal_text', '').lower()",
                                "write_evidence('goal', ctx.get('goal_text'))",
                                "# Replace these with concrete values from goal_hints/task_recipe.",
                                "target_object = '<object from goal_hints>'",
                                "target_receptacle = '<receptacle from goal_hints>'",
                                "holding_target = False",
                                "locations = [a.replace('go to ', '', 1) for a in list_actions() if a.startswith('go to ')]",
                                "for location in locations:",
                                "    if target_receptacle and target_receptacle in location:",
                                "        continue",
                                "    arrival = go_to(location)",
                                "    actions = list_actions()",
                                "    candidates = [a for a in actions if a.startswith('take ') and target_object in a]",
                                "    if candidates:",
                                "        obj_name = candidates[0].replace('take ', '', 1).split(' from ')[0]",
                                "        pickup = pickup_object(obj_name)",
                                "        if pickup.valid_action:",
                                "            holding_target = True",
                                "            break",
                                "if holding_target:",
                                "    target_location = target_receptacle",
                                "    for action in list_actions():",
                                "        if action.startswith('go to ') and target_receptacle in action:",
                                "            target_location = action.replace('go to ', '', 1)",
                                "            go_to(target_location)",
                                "            break",
                                "    place_object(target_object, target_location)",
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


def _parse_action(content: str) -> str:
    try:
        return str(json.loads(content)["action"])
    except Exception:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise RuntimeError("DeepSeek response did not contain JSON action.")
        return str(json.loads(match.group(0))["action"])


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
    cafile = ca_bundle or certifi.where()
    return ssl.create_default_context(cafile=cafile)


def _code_timeout_seconds() -> int:
    raw = os.environ.get("ALFWORLD_CODE_TIMEOUT_SECONDS", "30").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


def _code_max_primitive_calls(max_steps: int) -> int:
    raw = os.environ.get("ALFWORLD_CODE_MAX_PRIMITIVE_CALLS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(200, max_steps * 10)


def _w3_prompt_contract() -> list[str]:
    return [
        "Write one top-level Python solution for the current task; do not browse files or create a generic agent.",
        "Use task_context, current observation, admissible actions, primitive cards, and primitive results as the only public sources.",
        "Make progress with side-effect primitives; do not spend the whole budget on observe/look/examine.",
        "After a failed primitive call, change object alias, location, or subgoal strategy instead of repeating it.",
        "End with check_success() and write evidence for important choices.",
    ]


def _alfworld_task_recipe(context: dict[str, Any], observation: str) -> dict[str, Any]:
    goal = (_extract_official_task(observation) or str(context.get("goal_text") or "")).lower()
    hints = _goal_hints(goal)
    condition = hints.get("required_condition")
    task_type = str(context.get("task_type") or "").lower()
    if condition == "clean" or "clean" in task_type:
        steps = ["find target object", "pickup target", "go to sink/sinkbasin", "clean_object", "go to receptacle", "place_object", "check_success"]
    elif condition == "hot" or "heat" in task_type:
        steps = ["find target object", "pickup target", "go to microwave", "heat_object", "go to receptacle", "place_object", "check_success"]
    elif condition == "cool" or "cool" in task_type:
        steps = ["find target object", "pickup target", "go to fridge", "cool_object", "go to receptacle", "place_object", "check_success"]
    elif condition == "light" or "look_at" in task_type:
        steps = ["find target object", "pickup/examine target if admissible", "find lamp/desklamp", "toggle_object lamp on", "check_success"]
    elif hints.get("count") == "2" or "two" in goal:
        steps = ["repeat twice: find one matching target", "pickup target", "go to receptacle", "place_object", "check_success"]
    else:
        steps = ["find target object", "pickup target", "go to receptacle", "place_object", "check_success"]
    return {
        "benchmark": "ALFWorld Text",
        "goal_source": goal,
        "goal_hints": hints,
        "steps": steps,
        "grounding_notes": [
            "Use exact numbered object/receptacle names from list_actions when available.",
            "The wrappers resolve one admissible command; they do not solve navigation or search.",
            "If the target is not visible at the current location, try another go_to location from list_actions.",
        ],
    }


def _goal_hints(goal_text: str) -> dict[str, str | None]:
    lower = goal_text.lower()
    clean_match = re.search(r"clean some ([a-z0-9_ -]+?) and put it in ([a-z0-9_ -]+?)\.", lower)
    if clean_match:
        return {
            "target_object": clean_match.group(1).strip(),
            "target_receptacle": clean_match.group(2).strip(),
            "required_condition": "clean",
            "condition_location": "sink",
            "count": "1",
        }
    heat_match = re.search(r"(?:put a hot|heat some) ([a-z0-9_ -]+?) (?:in|and put it in) ([a-z0-9_ -]+?)\.", lower)
    if heat_match:
        return {
            "target_object": heat_match.group(1).strip(),
            "target_receptacle": heat_match.group(2).strip(),
            "required_condition": "hot",
            "condition_location": "microwave",
            "count": "1",
        }
    cool_match = re.search(r"(?:put a cool|cool some) ([a-z0-9_ -]+?) (?:in|and put it in) ([a-z0-9_ -]+?)\.", lower)
    if cool_match:
        return {
            "target_object": cool_match.group(1).strip(),
            "target_receptacle": cool_match.group(2).strip(),
            "required_condition": "cool",
            "condition_location": "fridge",
            "count": "1",
        }
    two_match = re.search(r"put two ([a-z0-9_ -]+?) in ([a-z0-9_ -]+?)\.", lower)
    if two_match:
        return {
            "target_object": two_match.group(1).strip(),
            "target_receptacle": two_match.group(2).strip(),
            "required_condition": None,
            "condition_location": None,
            "count": "2",
        }
    look_match = re.search(r"(?:examine the|look at) ([a-z0-9_ -]+?) (?:with|under) the ([a-z0-9_ -]+?)\.", lower)
    if look_match:
        return {
            "target_object": look_match.group(1).strip(),
            "target_receptacle": None,
            "required_condition": "light",
            "condition_location": look_match.group(2).strip(),
            "count": "1",
        }
    put_match = re.search(r"put an? ([a-z0-9_ -]+?) in ([a-z0-9_ -]+?)\.", lower)
    if put_match:
        return {
            "target_object": put_match.group(1).strip(),
            "target_receptacle": put_match.group(2).strip(),
            "required_condition": None,
            "condition_location": None,
            "count": "1",
        }
    return {
        "target_object": None,
        "target_receptacle": None,
        "required_condition": None,
        "condition_location": None,
        "count": None,
    }


def _extract_official_task(observation: str) -> str | None:
    match = re.search(r"Your task is to:\s*(.+?)(?:\n|$)", observation)
    if not match:
        return None
    return match.group(1).strip().rstrip(".") + "."
