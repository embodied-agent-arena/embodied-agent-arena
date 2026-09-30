from __future__ import annotations

import json
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .agents import AgentResult
from .code_executor import execute_agent_code
from .env_file import load_env_file
from .primitive_cards import primitive_cards
from .primitives import VirtualHomeSymbolicPrimitives


DEFAULT_MODEL = "deepseek-v4-flash"
DEFAULT_BASE_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekV4CodeAgent:
    name = "deepseek_v4_code"

    def __init__(self, model: str | None = None):
        root = Path(__file__).resolve().parents[2]
        load_env_file(root / ".env")
        load_env_file(root.parent / "ALFWorldBenchmark" / ".env")
        self.api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        self.model = model or os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
        self.base_url = os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
        self.ssl_context = _build_ssl_context()
        if not self.api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set. Add it to .env or export it in your shell.")

    def run(self, primitives: VirtualHomeSymbolicPrimitives, max_steps: int) -> AgentResult:
        context = primitives.get_task_context()
        actions = primitives.list_actions()
        state = primitives.query_symbolic_state()
        code = self._generate_code(context, actions, state, max_steps)
        primitives.record_harness_event(
            "agent_code_generated",
            {
                "agent": self.name,
                "code_attempt_index": 1,
                "executed_code": code,
                "primitive_cards": primitives.cards_for_prompt(),
            },
        )
        result = execute_agent_code(
            primitives,
            code,
            timeout_seconds=_code_timeout_seconds(),
        )
        metrics = result.metrics | {"code_attempts": 1}
        if result.final_success:
            stopped_reason = "success"
        elif result.timed_out:
            stopped_reason = "code_timeout"
        elif result.exception:
            stopped_reason = "code_exception"
        else:
            stopped_reason = "not_success"
        return AgentResult(stopped_reason, metrics)

    def _generate_code(
        self,
        context: dict[str, Any],
        actions: list[str],
        state: dict[str, Any],
        max_steps: int,
    ) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a coding agent controlling a VirtualHome symbolic Evolving Graph task. "
                    "Write Python code that calls only the provided primitive functions. "
                    "This is a task-first benchmark harness, not a repository exploration task: solve the current "
                    "symbolic household task by executing valid program steps from the provided action list. "
                    "Do not import modules, open files, use network, define classes, use try/except, "
                    "or access hidden/oracle state. Each execute_program_step(action_line) call executes "
                    "one VirtualHome program line and is limited by max_env_steps. "
                    "Use list_actions(), query_symbolic_state(), write_evidence(), execute_program_step(), "
                    "and check_activity_success(). Choose action strings copied exactly from list_actions(). "
                    "Follow task_recipe. VirtualHome actions have preconditions: if an interaction fails because the agent "
                    "is not close to or not facing an object, choose a matching walk/face/turn/open precondition action "
                    "from list_actions() before retrying the household action. Do not repeat the same failing action. "
                    "The task goal is authoritative. Stop after check_activity_success() reports success or "
                    "after the step budget is exhausted. "
                    'Return JSON only: {"code": "<python code>"}'
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task_context": context,
                        "initial_actions": actions,
                        "initial_symbolic_state": state,
                        "max_env_steps": max_steps,
                        "primitive_cards": primitive_cards(),
                        "w3_prompt_contract": _w3_prompt_contract(),
                        "task_recipe": _virtualhome_task_recipe(context),
                        "execution_rules": [
                            "Every execute_program_step argument must be copied exactly from list_actions().",
                            "Satisfy close/facing/open/holding preconditions before object interaction.",
                            "After a failed action, query state and choose a different precondition action.",
                            "Do not execute a whole gold program or invent actions not present in list_actions().",
                        ],
                        "example_style": "\n".join(
                            [
                                "ctx = get_task_context()",
                                "actions = list_actions()",
                                "state = query_symbolic_state()",
                                "write_evidence('goal', ctx['goal_text'])",
                                "for action in actions:",
                                "    if '[Open]' in action and '<fridge>' in action:",
                                "        execute_program_step(action)",
                                "        break",
                                "check_activity_success()",
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
        payload: dict[str, Any] = {
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
            with urllib.request.urlopen(request, timeout=_api_timeout_seconds(), context=self.ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek API HTTP {exc.code}: {_redact(body)}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DeepSeek API request failed: {_redact(str(exc))}") from exc
        except TimeoutError as exc:
            raise RuntimeError("DeepSeek API request timed out") from exc


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


def _w3_prompt_contract() -> list[str]:
    return [
        "Write one top-level Python solution for the current VirtualHome symbolic task.",
        "Use only task_context, list_actions, query_symbolic_state, primitive cards, and primitive feedback.",
        "Copy action lines exactly from list_actions; do not invent VirtualHome syntax.",
        "Use query_symbolic_state to check location, holding, open/closed, close/facing preconditions.",
        "After failure feedback, execute a precondition action instead of repeating the same household action.",
    ]


def _virtualhome_task_recipe(context: dict[str, Any]) -> dict[str, Any]:
    task_type = str(context.get("task_type") or "").lower()
    goal = str(context.get("goal_text") or context.get("instruction") or "").lower()
    if "open" in task_type or goal.startswith("open"):
        steps = ["find object in list_actions", "walk/face object if needed", "execute [Open] action", "check_activity_success"]
    elif "close" in task_type or goal.startswith("close"):
        steps = ["find object", "walk/face object if needed", "execute [Close] action", "check_activity_success"]
    elif any(word in task_type or word in goal for word in ("put", "place", "move")):
        steps = [
            "walk/face target object",
            "pickup/grab object if required",
            "walk/face receptacle",
            "open receptacle if required and action exists",
            "put/place object using exact action line",
            "check_activity_success",
        ]
    elif any(word in task_type or word in goal for word in ("switch", "turn on", "turn off")):
        steps = ["walk/face switchable object", "execute switch on/off action", "check_activity_success"]
    elif any(word in task_type or word in goal for word in ("sit", "lie")):
        steps = ["walk to target furniture", "face/align if needed", "execute sit/lie action", "check_activity_success"]
    else:
        steps = [
            "query symbolic state",
            "select the first goal-relevant executable action",
            "if it fails, satisfy close/facing/holding/open precondition",
            "retry a different exact action line",
            "check_activity_success",
        ]
    return {
        "benchmark": "VirtualHome Symbolic",
        "task_type": task_type,
        "steps": steps,
        "precondition_notes": [
            "not_close_to_object usually means walk/face/turn toward the object first.",
            "closed container/device usually needs an open action before put/place.",
            "holding requirements usually need grab/pickup before put/place/use.",
        ],
    }


def _build_ssl_context() -> ssl.SSLContext:
    ca_bundle = os.environ.get("DEEPSEEK_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _code_timeout_seconds() -> int:
    raw = os.environ.get("VIRTUALHOME_CODE_TIMEOUT_SECONDS", "30").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 30


def _api_timeout_seconds() -> int:
    raw = os.environ.get("DEEPSEEK_API_TIMEOUT_SECONDS", "90").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 90


def _redact(value: str) -> str:
    return re.sub(r"sk-[A-Za-z0-9_-]+", "sk-***", value)
