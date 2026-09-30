"""Small, dependency-free agent loop for the native embodied harness.

The controller speaks an OpenAI-compatible chat-completions protocol, executes
model-authored Python through the repository's bounded code runner, and keeps
the official verifier outside the agent-visible namespace.  Benchmark packages
remain isolated in their pinned Python environments through the JSONL bridge.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .native_case_registry import (
    NativeRuntimeBinding,
    NativeRuntimeUnavailable,
    load_native_case,
    make_native_backend,
)
from .pool_coordinate_bridge import (
    attach_pool_coordinate,
    bind_pool_coordinate,
    effective_seed,
    read_pool_coordinate,
    record_pool_binding,
)
from .native_prompting import (
    extract_python_code,
    public_universal_trace,
    system_prompt,
    task_prompt,
    trace_dict_to_jsonl,
)
from .native_runtime_receipt import (
    NativeRuntimeReceiptError,
    validate_native_runtime_launch_receipt,
)
from .episode_loop import EpisodeLoop, is_timeout
from .runner import ExecutionResult, StatefulCodeRunner
from .schemas import PrimitiveCard, PrimitiveResult, VerificationResult
from .universal_interface import UNIVERSAL_INTERFACE_NAMES, UniversalEmbodiedBackend

NATIVE_LOOP_CONTRACT = "agentic-embodied-arena/native-agent-loop/v1"
DEFAULT_MODEL = "qwen3.5-27b"
MODEL_PROVIDER_OPENAI_COMPATIBLE = "openai_compatible"
MODEL_PROVIDER_CODEX_EXEC = "codex_exec"
MODEL_PROVIDER_CURSOR_EXEC = "cursor_exec"
_API_KEY_NAMES = (
    "LLM_API_KEY",
    "OPENAI_API_KEY",
    "DASHSCOPE_API_KEY",
    "OPENROUTER_API_KEY",
    "API_KEY",
)
_BASE_URL_NAMES = (
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "DASHSCOPE_BASE_URL",
    "OPENROUTER_BASE_URL",
    "API_BASE_URL",
    "BASE_URL",
)
_MODEL_NAMES = (
    "LLM_MODEL",
    "OPENAI_MODEL",
    "DASHSCOPE_MODEL",
    "OPENROUTER_MODEL",
    "API_MODEL",
    "MODEL",
)


class ModelRequestError(RuntimeError):
    def __init__(
        self, message: str, *, retryable: bool, status_code: int | None = None,
        reported_usage: dict | None = None
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.reported_usage = reported_usage


class LoopBudgetExceeded(RuntimeError):
    def __init__(self, names: list[str]) -> None:
        self.names = list(dict.fromkeys(names))
        super().__init__("loop budget exhausted: " + ", ".join(self.names))


@dataclass(slots=True)
class NativeModelConfig:
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    model: str = DEFAULT_MODEL
    provider: str = MODEL_PROVIDER_OPENAI_COMPATIBLE
    temperature: float = 0.0
    max_tokens_per_response: int = 1800
    request_timeout_seconds: float = 120.0
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    codex_executable: str = "codex"
    codex_reasoning_effort: str | None = None
    cursor_executable: str = "agent"


@dataclass(slots=True)
class NativeLoopBudgets:
    max_agent_attempts: int = 1
    max_agent_iterations: int = 8
    llm_num_retries: int = 0
    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    max_primitive_calls: int | None = None
    max_verifier_calls: int | None = None


@dataclass(slots=True)
class ModelCompletion:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None
    provider_usage_available: bool
    usage_mode: str
    reasoning_tokens: int | None = None
    visible_tokens: int | None = None


class NativeModelClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
    ) -> ModelCompletion: ...


@dataclass(slots=True)
class UsageLedger:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    accumulated_cost: float = 0.0
    response_count: int = 0
    provider_usage_available: bool = True
    cost_available: bool = True
    usage_modes: set[str] = field(default_factory=set)
    known_reasoning_tokens: int = 0
    known_visible_tokens: int = 0
    reasoning_usage_responses: int = 0
    visible_usage_responses: int = 0

    model_request_attempts: int = 0
    request_retries: int = 0
    retry_wait_seconds: float = 0.0
    unknown_usage_attempts: int = 0

    def add(self, completion: ModelCompletion) -> None:
        self.prompt_tokens += completion.prompt_tokens
        self.completion_tokens += completion.completion_tokens
        self.total_tokens += completion.total_tokens
        self.response_count += 1
        if completion.reasoning_tokens is not None:
            self.known_reasoning_tokens += completion.reasoning_tokens
            self.reasoning_usage_responses += 1
        if completion.visible_tokens is not None:
            self.known_visible_tokens += completion.visible_tokens
            self.visible_usage_responses += 1
        self.provider_usage_available &= completion.provider_usage_available
        self.usage_modes.add(completion.usage_mode)
        if completion.cost_usd is None:
            self.cost_available = False
        else:
            self.accumulated_cost += completion.cost_usd

    def exhausted(self, budgets: NativeLoopBudgets) -> list[str]:
        """Return limits that prevent another model request.

        Reaching a limit exactly is allowed for the response that consumed the
        remaining budget, but prevents a subsequent request.
        """

        names: list[str] = []
        if (
            budgets.max_total_tokens is not None
            and self.total_tokens >= budgets.max_total_tokens
        ):
            names.append("max_total_tokens")
        if (
            budgets.max_cost_usd is not None
            and self.cost_available
            and self.accumulated_cost >= budgets.max_cost_usd
        ):
            names.append("max_cost_usd")
        return names

    def violated(self, budgets: NativeLoopBudgets) -> list[str]:
        """Return limits exceeded by the response that just completed."""

        names: list[str] = []
        if (
            budgets.max_total_tokens is not None
            and self.total_tokens > budgets.max_total_tokens
        ):
            names.append("max_total_tokens")
        if (
            budgets.max_cost_usd is not None
            and self.cost_available
            and self.accumulated_cost > budgets.max_cost_usd
        ):
            names.append("max_cost_usd")
        return names

    def to_dict(self, budgets: NativeLoopBudgets) -> dict[str, Any]:
        unverifiable = []
        if budgets.max_cost_usd is not None and not self.cost_available:
            unverifiable.append("max_cost_usd")
        return {
            "available": self.response_count > 0 and self.provider_usage_available,
            "token_usage_available": bool(self.usage_modes),
            "provider_usage_available": (
                self.response_count > 0 and self.provider_usage_available
            ),
            "usage_mode": "+".join(sorted(self.usage_modes)) or "none",
            "response_count": self.response_count,
            "model_request_attempts": self.model_request_attempts,
            "request_retries": self.request_retries,
            "retry_wait_seconds": round(self.retry_wait_seconds, 6),
            "unknown_usage_attempts": self.unknown_usage_attempts,
            "all_transport_usage_known": self.unknown_usage_attempts == 0,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.known_reasoning_tokens if self.response_count and self.reasoning_usage_responses == self.response_count else None,
            "visible_tokens": self.known_visible_tokens if self.response_count and self.visible_usage_responses == self.response_count else None,
            "known_reasoning_tokens": self.known_reasoning_tokens,
            "known_visible_tokens": self.known_visible_tokens,
            "reasoning_usage_responses": self.reasoning_usage_responses,
            "visible_usage_responses": self.visible_usage_responses,
            "budget_token_policy": "per_response_visible_text; case_total_includes_prompt_and_reasoning",
            "total_tokens": self.total_tokens,
            "accumulated_cost": round(self.accumulated_cost, 8),
            "cost_available": self.cost_available,
            "budget_exhausted": self.exhausted(budgets),
            "budget_unverifiable": unverifiable,
        }


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _load_env_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    result: dict[str, str] = {}
    pending: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip().removeprefix("export ").strip()] = _clean_env_value(
                value
            )
            pending = None
        elif line.endswith(":"):
            pending = line[:-1].strip()
        elif pending:
            result[pending] = _clean_env_value(line)
            pending = None
    return result


def _first(source: Mapping[str, str], names: tuple[str, ...]) -> str:
    return next((str(source[name]).strip() for name in names if source.get(name)), "")


def _optional_float(source: Mapping[str, str], name: str) -> float | None:
    value = str(source.get(name) or "").strip()
    return float(value) if value else None


def _normalize_base_url(value: str) -> str:
    cleaned = value.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return urlunparse(parsed._replace(path="/v1"))
    return cleaned


def load_native_model_config(
    env_file: str | Path = ".env",
    *,
    provider: str = MODEL_PROVIDER_OPENAI_COMPATIBLE,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens_per_response: int | None = None,
    request_timeout_seconds: float | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    codex_executable: str = "codex",
    codex_reasoning_effort: str | None = None,
    cursor_executable: str = "agent",
) -> NativeModelConfig:
    merged = _load_env_file(Path(env_file))
    merged.update({str(key): str(value) for key, value in os.environ.items()})
    normalized_provider = str(provider).strip().lower().replace("-", "_")
    if normalized_provider not in {
        MODEL_PROVIDER_OPENAI_COMPATIBLE,
        MODEL_PROVIDER_CODEX_EXEC,
        MODEL_PROVIDER_CURSOR_EXEC,
    }:
        raise ValueError(f"Unsupported native model provider: {provider!r}")
    api_key = _first(merged, _API_KEY_NAMES)
    base_url = _first(merged, _BASE_URL_NAMES)
    if normalized_provider == MODEL_PROVIDER_OPENAI_COMPATIBLE and not api_key:
        raise ValueError("Missing model API key (LLM_API_KEY or OPENAI_API_KEY)")
    if normalized_provider == MODEL_PROVIDER_OPENAI_COMPATIBLE and not base_url:
        raise ValueError("Missing model base URL (LLM_BASE_URL or OPENAI_BASE_URL)")
    configured_model = model or _first(merged, _MODEL_NAMES)
    if normalized_provider == MODEL_PROVIDER_CODEX_EXEC and not configured_model:
        raise ValueError("--model is required with --model-provider codex-exec")
    if normalized_provider == MODEL_PROVIDER_CURSOR_EXEC and not configured_model:
        raise ValueError("--model is required with --model-provider cursor-exec")
    executable = str(codex_executable or "codex").strip()
    cursor_bin = str(cursor_executable or "agent").strip()
    if (
        normalized_provider == MODEL_PROVIDER_CODEX_EXEC
        and shutil.which(executable) is None
    ):
        raise ValueError(f"Codex CLI executable is unavailable: {executable!r}")
    if (
        normalized_provider == MODEL_PROVIDER_CURSOR_EXEC
        and shutil.which(cursor_bin) is None
    ):
        raise ValueError(f"Cursor CLI executable is unavailable: {cursor_bin!r}")
    config = NativeModelConfig(
        api_key=api_key,
        base_url=_normalize_base_url(base_url) if base_url else "",
        model=configured_model or DEFAULT_MODEL,
        provider=normalized_provider,
        temperature=temperature if temperature is not None else float(merged.get("LLM_TEMPERATURE") or 0),
        max_tokens_per_response=max_tokens_per_response if max_tokens_per_response is not None else int(merged.get("LLM_MAX_TOKENS") or 1800),
        request_timeout_seconds=request_timeout_seconds if request_timeout_seconds is not None else float(merged.get("LLM_REQUEST_TIMEOUT_SECONDS") or 120),
        input_cost_per_million=(
            input_cost_per_million
            if input_cost_per_million is not None
            else _optional_float(merged, "LLM_INPUT_COST_PER_MILLION")
        ),
        output_cost_per_million=(
            output_cost_per_million
            if output_cost_per_million is not None
            else _optional_float(merged, "LLM_OUTPUT_COST_PER_MILLION")
        ),
        codex_executable=executable,
        codex_reasoning_effort=(
            str(codex_reasoning_effort).strip()
            if codex_reasoning_effort is not None
            else None
        ),
        cursor_executable=cursor_bin,
    )
    if (config.max_tokens_per_response <= 0 or not math.isfinite(config.request_timeout_seconds)
            or config.request_timeout_seconds <= 0 or not math.isfinite(config.temperature)
            or not 0 <= config.temperature <= 2):
        raise ValueError("Require positive model max tokens/timeout and temperature between 0 and 2")
    return config


def _estimated_tokens(text: str) -> int:
    # Deliberately conservative for mixed Chinese/code prompts. Provider usage,
    # when returned, replaces this estimate in the ledger.
    return max(1, math.ceil(len(text) / 2.0))


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(str(item.get("content") or "") for item in messages)


class OpenAICompatibleModelClient:
    def __init__(self, config: NativeModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        image_paths: list[Path] | None = None,
    ) -> ModelCompletion:
        # Only the harness supplies exposed public frames. Never mutate the
        # saved text transcript or place base64 image data in trace files.
        request_messages: list[dict[str, Any]] = [dict(message) for message in messages]
        if image_paths:
            if len(image_paths) > 32:
                raise ModelRequestError("At most 32 public frames per request", retryable=False)
            parts: list[dict[str, Any]] = []
            for path in image_paths:
                mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(path.suffix.lower())
                if mime is None or path.stat().st_size > 10 * 1024 * 1024:
                    raise ModelRequestError("Public frame must be PNG/JPEG and at most 10 MiB", retryable=False)
                data = base64.b64encode(path.read_bytes()).decode("ascii")
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}})
            if request_messages and request_messages[-1].get("role") == "user":
                parts.insert(0, {"type": "text", "text": str(request_messages[-1]["content"])})
                request_messages[-1]["content"] = parts
            else:
                request_messages.append({"role": "user", "content": parts})
        payload = json.dumps(
            {
                "model": self.config.model,
                "messages": request_messages,
                "temperature": self.config.temperature,
                "max_tokens": max_tokens,
            },
            ensure_ascii=False,
        ).encode("utf-8")
        if getattr(self, "request_archive_dir", None) is not None:
            from .w4_rgb import archive_request
            self.last_request_id = archive_request(self.request_archive_dir, payload, image_paths,
                                                  getattr(self, "request_observation", None))
        request = Request(
            f"{self.config.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        request_started = time.monotonic()
        try:
            with urlopen(
                request, timeout=self.config.request_timeout_seconds
            ) as response:
                document = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            # Gateways can echo Authorization headers in their error body.
            # Keep only status metadata in the evaluator's persisted failure.
            exc.close()
            retryable = exc.code in {408, 409, 425, 429} or exc.code >= 500
            raise ModelRequestError(
                f"Model API HTTP {exc.code}",
                retryable=retryable,
                status_code=exc.code,
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            cause = getattr(exc, "reason", None)
            detail = f"; cause_type={type(cause).__name__}" if cause is not None else ""
            raise ModelRequestError(
                f"Model API transport failure: {type(exc).__name__}{detail}",
                retryable=True,
            ) from exc
        except json.JSONDecodeError as exc:
            raise ModelRequestError(
                f"Model API returned invalid JSON: {exc}", retryable=True
            ) from exc
        try:
            content = str(document["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelRequestError(
                "Model API response is missing choices[0].message.content",
                retryable=False, reported_usage=document.get("usage"),
            ) from exc
        if getattr(self, "request_archive_dir", None) is not None:
            _atomic_json(Path(self.request_archive_dir) / (self.last_request_id + ".response.json"),
                         {"request_id": self.last_request_id, "content": content,
                          "wall_time": time.time(), "latency_seconds": time.monotonic() - request_started,
                          "usage": document.get("usage"), "finish_reason": document["choices"][0].get("finish_reason"),
                          "provider_response": document})
        usage = document.get("usage") if isinstance(document.get("usage"), dict) else {}
        provider_usage = all(
            isinstance(usage.get(name), (int, float))
            for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if provider_usage:
            prompt_tokens = int(usage["prompt_tokens"])
            completion_tokens = int(usage["completion_tokens"])
            total_tokens = int(usage["total_tokens"])
            usage_mode = "provider"
        else:
            prompt_tokens = _estimated_tokens(_messages_text(messages))
            completion_tokens = _estimated_tokens(content)
            # Without provider usage, include a conservative image allowance.
            prompt_tokens += 8192 * len(image_paths or [])
            total_tokens = prompt_tokens + completion_tokens
            usage_mode = "conservative_estimate"
        raw_cost = usage.get("cost", document.get("cost"))
        cost: float | None
        if isinstance(raw_cost, (int, float)) and not isinstance(raw_cost, bool):
            cost = float(raw_cost)
        elif (
            self.config.input_cost_per_million is not None
            and self.config.output_cost_per_million is not None
        ):
            cost = (
                prompt_tokens * self.config.input_cost_per_million
                + completion_tokens * self.config.output_cost_per_million
            ) / 1_000_000.0
        else:
            cost = None
        details = usage.get("completion_tokens_details") or usage.get("output_tokens_details") or {}
        def token_detail(name):
            value = details.get(name) if isinstance(details, dict) else None
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0 else None
        reasoning_tokens = token_detail("reasoning_tokens")
        visible_tokens = token_detail("text_tokens")
        if visible_tokens is None and reasoning_tokens is not None and provider_usage:
            visible_tokens = max(0, completion_tokens - reasoning_tokens)
        return ModelCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            provider_usage_available=provider_usage,
            usage_mode=usage_mode,
            reasoning_tokens=reasoning_tokens,
            visible_tokens=visible_tokens,
        )


_CODEX_SAFE_ITEM_TYPES = frozenset({"agent_message", "reasoning"})
_CODEX_SECRET_ENV_NAMES = frozenset({*_API_KEY_NAMES, "CODEX_API_KEY"})


def _codex_exec_prompt(messages: list[dict[str, str]], *, max_tokens: int) -> str:
    transcript = "\n\n".join(
        f"<{str(item.get('role') or 'user').upper()}>\n{str(item.get('content') or '')}"
        for item in messages
    )
    return (
        "You are the code-generation model inside a sealed embodied-benchmark harness. "
        "Do not call tools, shell commands, web search, MCP, subagents, or file APIs. "
        "Do not inspect the local machine. Use only the task/interface evidence in the "
        "transcript below. Return executable Python only, with no prose or Markdown fence. "
        f"Keep the response within {max_tokens} tokens.\n\n" + transcript
    )


def _codex_exec_environment() -> dict[str, str]:
    """Keep CLI login discovery while withholding model-provider secrets from tools."""

    environment = {str(key): str(value) for key, value in os.environ.items()}
    for name in list(environment):
        upper = name.upper()
        if name in _CODEX_SECRET_ENV_NAMES or any(
            marker in upper for marker in ("API_KEY", "PASSWORD", "SECRET", "TOKEN")
        ):
            environment.pop(name, None)
    environment["NO_COLOR"] = "1"
    return environment


class CodexExecModelClient:
    """Use saved Codex CLI/App authentication as a sealed completion process.

    The JSONL stream is also a policy boundary: a completion is rejected if
    Codex invokes any tool.  The benchmark agent may act only later through the
    native loop's bounded universal interfaces.
    """

    def __init__(self, config: NativeModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        image_paths: list[Path] | None = None,
    ) -> ModelCompletion:
        with tempfile.TemporaryDirectory(prefix="arena-codex-completion-") as raw_root:
            isolated_root = Path(raw_root)
            command = [
                self.config.codex_executable,
                "exec",
                "--ephemeral",
                "--json",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ignore-user-config",
                "--ignore-rules",
                "--color",
                "never",
                "--cd",
                str(isolated_root),
                "--model",
                self.config.model,
            ]
            if self.config.codex_reasoning_effort:
                command.extend(
                    [
                        "--config",
                        f'model_reasoning_effort="{self.config.codex_reasoning_effort}"',
                    ]
                )
            for index, source in enumerate(image_paths or []):
                target = isolated_root / f"frame_{index}{source.suffix}"
                shutil.copyfile(source, target)
                command.extend(["--image", str(target)])
            command.append("-")
            prompt = _codex_exec_prompt(messages, max_tokens=max_tokens)
            if getattr(self, "request_archive_dir", None) is not None:
                from .w4_rgb import archive_codex_request
                self.last_request_id = archive_codex_request(self.request_archive_dir, command, prompt,
                    image_paths, getattr(self, "request_observation", None))
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    cwd=isolated_root,
                    env=_codex_exec_environment(),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.request_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelRequestError(
                    "Codex CLI completion timed out",
                    retryable=True,
                ) from exc
            except OSError as exc:
                raise ModelRequestError(
                    f"Codex CLI launch failed: {type(exc).__name__}: {exc}",
                    retryable=False,
                ) from exc
        if getattr(self, "request_archive_dir", None) is not None:
            _atomic_json(Path(self.request_archive_dir) / (self.last_request_id + ".response.json"),
                         {"request_id": self.last_request_id, "stdout": completed.stdout,
                          "stderr": completed.stderr, "returncode": completed.returncode})
        if completed.returncode != 0:
            detail = completed.stderr[-2000:].strip()
            if not detail:
                for raw_line in reversed(completed.stdout.splitlines()):
                    try:
                        event = json.loads(raw_line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event_error = event.get("error")
                    message = event.get("message")
                    if isinstance(event_error, dict):
                        message = event_error.get("message") or message
                    if isinstance(message, str) and message.strip():
                        detail = message.strip()[-2000:]
                        break
            if not detail:
                detail = "Codex CLI returned no diagnostic message"
            raise ModelRequestError(
                f"Codex CLI exited with status {completed.returncode}: {detail}",
                retryable=(
                    completed.returncode in {75, 124}
                    or "selected model is at capacity" in detail.lower()
                ),
            )

        events: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ModelRequestError(
                    f"Codex CLI emitted invalid JSONL at line {line_number}",
                    retryable=False,
                ) from exc
            if not isinstance(event, dict):
                raise ModelRequestError(
                    f"Codex CLI emitted a non-object event at line {line_number}",
                    retryable=False,
                )
            events.append(event)

        forbidden_items: list[str] = []
        messages_out: list[str] = []
        usage: dict[str, Any] = {}
        for event in events:
            event_type = str(event.get("type") or "")
            item = event.get("item") if isinstance(event.get("item"), dict) else None
            if item is not None:
                item_type = str(item.get("type") or "unknown")
                if item_type not in _CODEX_SAFE_ITEM_TYPES:
                    forbidden_items.append(item_type)
                if event_type == "item.completed" and item_type == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str) and text.strip():
                        messages_out.append(text)
            if event_type == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = dict(event["usage"])
            if event_type in {"turn.failed", "error"}:
                raise ModelRequestError(
                    f"Codex CLI reported {event_type}",
                    retryable=event_type == "error",
                )
        if forbidden_items:
            raise ModelRequestError(
                "Codex completion crossed the no-tool evaluation boundary: "
                + ",".join(sorted(set(forbidden_items))),
                retryable=False,
            )
        if not messages_out:
            raise ModelRequestError(
                "Codex CLI JSONL stream has no final agent message",
                retryable=False,
            )

        content = messages_out[-1]
        provider_usage = all(
            isinstance(usage.get(name), (int, float))
            for name in ("input_tokens", "output_tokens")
        )
        if provider_usage:
            prompt_tokens = int(usage["input_tokens"])
            completion_tokens = int(usage["output_tokens"])
            total_tokens = prompt_tokens + completion_tokens
            usage_mode = "codex_exec_jsonl"
        else:
            prompt_tokens = _estimated_tokens(_messages_text(messages))
            completion_tokens = _estimated_tokens(content)
            total_tokens = prompt_tokens + completion_tokens
            usage_mode = "codex_exec_conservative_estimate"
        cost = None
        if (
            self.config.input_cost_per_million is not None
            and self.config.output_cost_per_million is not None
        ):
            cost = (
                prompt_tokens * self.config.input_cost_per_million
                + completion_tokens * self.config.output_cost_per_million
            ) / 1_000_000.0
        return ModelCompletion(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            provider_usage_available=provider_usage,
            usage_mode=usage_mode,
        )


_CURSOR_SAFE_EVENT_TYPES = frozenset({"system", "user", "thinking", "assistant", "result"})


def _cursor_event_is_tool(event: Mapping[str, Any]) -> str | None:
    event_type = str(event.get("type") or "")
    if event_type and event_type not in _CURSOR_SAFE_EVENT_TYPES:
        return event_type
    message = event.get("message") if isinstance(event.get("message"), dict) else None
    content = message.get("content") if message is not None else event.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = str(block.get("type") or "")
            if block_type and block_type not in {"text", "thinking"}:
                return block_type
    return None


class CursorExecModelClient:
    """Use saved Cursor CLI login as a sealed completion process.

    Mirrors CodexExecModelClient: isolated workspace, no-tool ask mode, and a
    stream-json audit. The benchmark agent may act only later through the
    native loop's bounded universal interfaces.
    """

    def __init__(self, config: NativeModelConfig) -> None:
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        image_paths: list[Path] | None = None,
    ) -> ModelCompletion:
        with tempfile.TemporaryDirectory(prefix="arena-cursor-completion-") as raw_root:
            isolated_root = Path(raw_root)
            prompt = _codex_exec_prompt(messages, max_tokens=max_tokens)
            # Cursor ask-mode still emits Read/image tool_call if frames are
            # dropped as workspace files. Keep the directory empty and inline
            # the same public frames Codex would pass via --image.
            if image_paths:
                prompt += (
                    "\n\nPublic images are already inlined below as data URLs. "
                    "The workspace is empty. Do not call tools or open files.\n"
                )
                for index, source in enumerate(image_paths):
                    mime = {
                        ".png": "image/png",
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                    }.get(source.suffix.lower())
                    if mime is None or source.stat().st_size > 10 * 1024 * 1024:
                        raise ModelRequestError(
                            "Public frame must be PNG/JPEG and at most 10 MiB",
                            retryable=False,
                        )
                    data = base64.b64encode(source.read_bytes()).decode("ascii")
                    prompt += f"\n![public_frame_{index}](data:{mime};base64,{data})\n"
            command = [
                self.config.cursor_executable,
                "-p",
                "--mode",
                "ask",
                "--trust",
                "--sandbox",
                "disabled",
                "--model",
                self.config.model,
                "--output-format",
                "stream-json",
                "--workspace",
                str(isolated_root),
                prompt,
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(isolated_root),
                    env=_codex_exec_environment(),
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.config.request_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise ModelRequestError(
                    "Cursor CLI completion timed out",
                    retryable=True,
                ) from exc
            except OSError as exc:
                raise ModelRequestError(
                    f"Cursor CLI launch failed: {type(exc).__name__}: {exc}",
                    retryable=False,
                ) from exc

        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            if not detail:
                detail = "Cursor CLI returned no diagnostic message"
            raise ModelRequestError(
                f"Cursor CLI exited with status {completed.returncode}: {detail}",
                retryable=completed.returncode in {75, 124},
            )

        events: list[dict[str, Any]] = []
        for line_number, raw_line in enumerate(completed.stdout.splitlines(), start=1):
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ModelRequestError(
                    f"Cursor CLI emitted invalid JSONL at line {line_number}",
                    retryable=False,
                ) from exc
            if not isinstance(event, dict):
                raise ModelRequestError(
                    f"Cursor CLI emitted a non-object event at line {line_number}",
                    retryable=False,
                )
            events.append(event)

        forbidden: list[str] = []
        result_text = ""
        usage: dict[str, Any] = {}
        for event in events:
            tool = _cursor_event_is_tool(event)
            if tool is not None:
                forbidden.append(tool)
            if str(event.get("type") or "") == "result":
                if event.get("is_error") or str(event.get("subtype") or "") not in {
                    "success",
                    "",
                }:
                    message = event.get("result") or event.get("error") or "Cursor CLI result error"
                    raise ModelRequestError(
                        f"Cursor CLI reported {event.get('subtype') or 'error'}: {message}",
                        retryable=False,
                    )
                text = event.get("result")
                if isinstance(text, str) and text.strip():
                    result_text = text
                raw_usage = event.get("usage")
                if isinstance(raw_usage, dict):
                    usage = raw_usage
        if forbidden:
            raise ModelRequestError(
                "Cursor completion crossed the no-tool evaluation boundary: "
                + ",".join(sorted(set(forbidden))),
                retryable=False,
            )
        if not result_text.strip():
            raise ModelRequestError(
                "Cursor CLI stream-json has no final assistant result",
                retryable=False,
            )

        input_tokens = usage.get("inputTokens", usage.get("input_tokens"))
        output_tokens = usage.get("outputTokens", usage.get("output_tokens"))
        provider_usage = isinstance(input_tokens, (int, float)) and isinstance(
            output_tokens, (int, float)
        )
        if provider_usage:
            prompt_tokens = int(input_tokens)
            completion_tokens = int(output_tokens)
            total_tokens = prompt_tokens + completion_tokens
            usage_mode = "cursor_exec_jsonl"
        else:
            prompt_tokens = _estimated_tokens(_messages_text(messages))
            completion_tokens = _estimated_tokens(result_text)
            total_tokens = prompt_tokens + completion_tokens
            usage_mode = "cursor_exec_conservative_estimate"
        cost = None
        if (
            self.config.input_cost_per_million is not None
            and self.config.output_cost_per_million is not None
        ):
            cost = (
                prompt_tokens * self.config.input_cost_per_million
                + completion_tokens * self.config.output_cost_per_million
            ) / 1_000_000.0
        return ModelCompletion(
            content=result_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            provider_usage_available=provider_usage,
            usage_mode=usage_mode,
        )


def _compact_prompt_value(value: Any, *, depth: int = 0, _budget: list[int] | None = None) -> Any:
    # Bound the whole text projection: per-axis truncation alone expands images
    # into tens of thousands of placeholders. The Python result stays intact.
    if _budget is None:
        _budget = [24000]
    if _budget[0] <= 0:
        return "<prompt-content-omitted>"
    _budget[0] -= 16
    if depth >= 7:
        return "<depth-truncated>"
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, str):
            limit = min(4000, max(0, _budget[0] // 6))
            if len(value) > limit:
                value = value[:limit] + "<truncated>"
        _budget[0] -= len(json.dumps(value, ensure_ascii=False, default=str))
        return value
    if isinstance(value, Path):
        return "<local-path>"
    if isinstance(value, dict):
        compact = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 96 or _budget[0] <= 0:
                compact["<omitted_key_count>"] = len(value) - index
                break
            key = str(key)[:256]
            _budget[0] -= len(json.dumps(key, ensure_ascii=False)) + 4
            compact[key] = _compact_prompt_value(item, depth=depth + 1, _budget=_budget)
        return compact
    if isinstance(value, (list, tuple)):
        if len(value) > 256 and all(
            item is None or isinstance(item, (bool, int, float)) for item in value[:64]
        ):
            return {"array_content_omitted": True, "element_count": len(value)}
        result = []
        for index, item in enumerate(value):
            if index >= 64 or _budget[0] <= 0:
                result.append({"truncated_item_count": len(value) - index})
                break
            result.append(_compact_prompt_value(item, depth=depth + 1, _budget=_budget))
        return result
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None and dtype is not None:
        return {"array_content_omitted": True, "shape": [int(item) for item in shape],
                "dtype": str(dtype)}
    if hasattr(value, "to_dict"):
        return _compact_prompt_value(value.to_dict(), depth=depth + 1, _budget=_budget)
    return _compact_prompt_value(repr(value)[:1000], depth=depth + 1, _budget=_budget)


def _interface_fingerprint(
    *,
    case_id: str,
    benchmark_id: str,
    cards: list[PrimitiveCard],
    unavailable_interfaces: list[str],
) -> str:
    payload = {
        "contract": NATIVE_LOOP_CONTRACT,
        "case_id": case_id,
        "benchmark_id": benchmark_id,
        "interfaces": [card.to_dict() for card in cards],
        "unavailable_interfaces": sorted(unavailable_interfaces),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _universal_code_contract_error(
    code: str, *, unavailable_interfaces: list[str] | tuple[str, ...] = (),
    native_names: set[str] | None = None,
) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f"agent_code_syntax_error: {exc}"
    forbidden_names = {"backend", "gateway", "verifier"}
    if any(
        isinstance(node, ast.Name) and node.id in forbidden_names
        for node in ast.walk(tree)
    ):
        return "agent_code_contract_error: backend/verifier is harness-only"
    forbidden_attributes = {
        "_backend",
        "call_primitive",
        "get_trace",
        "record_event",
        "verify",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and (
            node.attr.startswith("_") or node.attr in forbidden_attributes
        ):
            return "agent_code_contract_error: private backend/verifier access is forbidden"
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {"getattr", "hasattr"}:
            return "agent_code_contract_error: dynamic attribute access is forbidden"
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "primitives"
            and node.func.attr not in {"call", "invoke", "list"}
        ):
            return "agent_code_contract_error: use primitives.call/invoke with a universal interface"
        if not (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "primitives"
            and node.func.attr in {"call", "invoke"}
        ):
            continue
        if node.args and native_names is None:
            return (
                "agent_code_contract_error: universal calls require keyword arguments"
            )
        if any(item.arg is None for item in node.keywords):
            return "agent_code_contract_error: universal calls may not receive **kwargs"
        interface = next(
            (
                item.value.value
                for item in node.keywords
                if item.arg in {"interface", "name"}
                and isinstance(item.value, ast.Constant)
                and isinstance(item.value.value, str)
            ),
            None,
        )
        if interface is None and node.func.attr == "invoke":
            interface = next(
                (
                    item.value.value
                    for item in node.keywords
                    if item.arg == "interface"
                    and isinstance(item.value, ast.Constant)
                    and isinstance(item.value.value, str)
                ),
                None,
            )
        if native_names is not None and node.args:
            first = node.args[0]
            interface = first.value if len(node.args) == 1 and isinstance(first, ast.Constant) else None
        if interface not in (UNIVERSAL_INTERFACE_NAMES if native_names is None else native_names):
            return (
                f"agent_code_contract_error: unknown {'universal interface' if native_names is None else 'native primitive'} {interface!r}"
            )
        if interface in unavailable_interfaces:
            return (
                "agent_code_contract_error: universal interface "
                f"{interface!r} is unavailable for this benchmark"
            )
    return None


def _called_interfaces(trace: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    for event in trace.get("events", []):
        if (
            not isinstance(event, dict)
            or event.get("event_type") != "universal_gateway_call"
        ):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        name = str(payload.get("interface") or "")
        if name and name not in result:
            result.append(name)
    return result


def _native_public_contract(gateway) -> dict[str, Any]:
    return {"required_interfaces": [], "called_interfaces": list(gateway.called),
            "missing_interfaces": [], "satisfied": True,
            "interface_mode": "native-primitives"}


def _public_contract(called: list[str]) -> dict[str, Any]:
    required = ["task.context", "scene.observe", "progress.check"]
    missing = [name for name in required if name not in called]
    acted = any(name in called for name in ("action.execute", "policy.invoke"))
    return {
        "required_interfaces": required,
        "called_interfaces": called,
        "missing_interfaces": missing,
        "action_or_policy_executed": acted,
        "satisfied": not missing and acted,
    }


def _progress_payload(execution: ExecutionResult) -> dict[str, Any] | None:
    result = execution.result
    if isinstance(result, PrimitiveResult) and result.name == "progress.check":
        return _compact_prompt_value(result.to_dict())
    return None


def _gateway_budget(gateway: UniversalEmbodiedBackend) -> dict[str, Any]:
    snapshot = getattr(gateway, "_budget_snapshot", None)
    if callable(snapshot):
        return dict(snapshot())
    return {}


def _loop_budget_payload(
    usage: UsageLedger,
    budgets: NativeLoopBudgets,
    *,
    attempt_number: int,
    completed_turns: int,
) -> dict[str, Any]:
    token_remaining = (
        None
        if budgets.max_total_tokens is None
        else max(0, budgets.max_total_tokens - usage.total_tokens)
    )
    cost_remaining = (
        None
        if budgets.max_cost_usd is None or not usage.cost_available
        else max(0.0, budgets.max_cost_usd - usage.accumulated_cost)
    )
    return {
        "limits": asdict(budgets),
        "used": {
            "agent_attempts_started": attempt_number,
            "agent_iterations_in_attempt": completed_turns,
            "model_responses": usage.response_count,
            "total_tokens": usage.total_tokens,
            "cost_usd": round(usage.accumulated_cost, 8),
        },
        "remaining": {
            "agent_attempts_after_current": max(
                0, budgets.max_agent_attempts - attempt_number
            ),
            "agent_iterations_in_attempt": max(
                0, budgets.max_agent_iterations - completed_turns
            ),
            "total_tokens": token_remaining,
            "cost_usd": cost_remaining,
        },
        "cost_available": usage.cost_available,
        "exhausted": usage.exhausted(budgets),
    }


def _last_public_error(trace: Mapping[str, Any]) -> dict[str, Any] | None:
    for event in reversed(list(trace.get("events", []))):
        if not isinstance(event, dict):
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        result = (
            payload.get("result") if isinstance(payload.get("result"), dict) else {}
        )
        if result and result.get("ok") is False:
            return _compact_prompt_value(
                {
                    "interface": payload.get("interface"),
                    "error": result.get("error"),
                    "output": result.get("output"),
                }
            )
    return None


def _repair_message(
    *,
    turn: int,
    execution: ExecutionResult,
    contract: Mapping[str, Any],
    progress: Mapping[str, Any] | None,
    gateway_budget: Mapping[str, Any],
    loop_budget: Mapping[str, Any],
    last_error: Mapping[str, Any] | None,
    official_passed: bool,
    interface_mode: str = "universal",
) -> str:
    feedback = {
        "turn": turn,
        "execution_ok": execution.ok,
        "execution_error": execution.error,
        "execution_stdout_tail": execution.stdout[-4000:],
        "public_contract": contract,
        "public_progress": progress,
        "last_public_interface_error": last_error,
        "gateway_budget": gateway_budget,
        "agent_loop_budget": loop_budget,
        "harness_official_check_passed": official_passed,
    }
    if interface_mode == "native":
        return (
            "Continue the same episode using the listed benchmark-native primitives. "
            "Python variables and environment state persist. Check returned results and correct errors. "
            "Return executable Python only; official scoring is performed by the harness.\n"
            "Feedback JSON:\n" + json.dumps(feedback, ensure_ascii=False, sort_keys=True, default=str)
        )
    return (
        "The previous code cell did not finish the task. Return corrected executable Python only. "
        "Use only public universal interfaces; never call backend or a verifier. Reuse valid handles in memory, "
        "diagnose public errors, perform the remaining grounded action/policy, and end with progress.check.\n"
        "Public feedback JSON:\n"
        + json.dumps(feedback, ensure_ascii=False, sort_keys=True, default=str)
    )


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(
            payload, stream, ensure_ascii=False, indent=2, sort_keys=True, default=str
        )
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def _write_attempt_artifacts(
    root: Path,
    *,
    attempt_number: int,
    summary: Mapping[str, Any],
    public_trace: Mapping[str, Any],
) -> dict[str, str]:
    directory = root / f"attempt-{attempt_number:03d}"
    summary_path = directory / "summary.json"
    trace_path = directory / "trace.json"
    trace_jsonl_path = directory / "trace.jsonl"
    _atomic_json(summary_path, summary)
    _atomic_json(trace_path, public_trace)
    trace_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    trace_jsonl_path.write_text(
        trace_dict_to_jsonl(dict(public_trace)).rstrip() + "\n", encoding="utf-8"
    )
    return {
        "summary": str(summary_path),
        "trace": str(trace_path),
        "trace_jsonl": str(trace_jsonl_path),
    }


def _exception_payload(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, (NativeRuntimeUnavailable, NativeRuntimeReceiptError)):
        category, retryable = "environment", False
    elif isinstance(exc, ModelRequestError):
        # A transport timeout is still a model API error, even when its cause
        # is TimeoutError; reserve timeout for execution/deadline failures.
        category, retryable = "model_api", exc.retryable
    elif is_timeout(exc):
        category, retryable = "timeout", True
    elif isinstance(exc, LoopBudgetExceeded):
        category, retryable = "budget", False
    elif isinstance(exc, TimeoutError):
        category, retryable = "timeout", True
    elif isinstance(exc, (ValueError, KeyError)):
        category, retryable = "configuration", False
    else:
        category, retryable = "native_runtime", True
    result = {
        "category": category,
        "type": type(exc).__name__,
        "message": str(exc)[:4000],
        "retryable": retryable,
    }
    if isinstance(exc, ModelRequestError):
        result["status_code"] = exc.status_code
    return result


def _request_retry_event(client, **event):
    """Append sanitized transport evidence; never include headers or API keys."""
    directory = getattr(client, "request_archive_dir", None)
    if directory is None:
        return
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    event.update(wall_time=time.time(), request_id=getattr(client, "last_request_id", None))
    request = directory / (str(event["request_id"]) + ".json")
    if request.is_file():
        event["wire_sha256"] = json.loads(request.read_text())["wire_sha256"]
    with (directory / "retry-events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")


def _reported_error_usage(exc):
    raw = getattr(exc, "reported_usage", None)
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    if not isinstance(raw, dict) or not all(isinstance(raw.get(k), (int, float))
            and not isinstance(raw[k], bool) and math.isfinite(raw[k]) and raw[k] >= 0 for k in keys):
        return None
    return ModelCompletion(content="", **{k: int(raw[k]) for k in keys}, cost_usd=None,
                           provider_usage_available=True, usage_mode="provider_error_response")


def completion_with_budget(*, client, config, budgets, usage, messages,
                           loop: EpisodeLoop | None = None, client_kwargs=None):
    """Shared preflight, accounting and post-response gate before code execution."""
    phase_limit = config.request_timeout_seconds
    image_paths = (client_kwargs or {}).get("image_paths") or []
    observation = getattr(client, "request_observation", None) or {}
    signature = tuple((i.get("episode_id"), i.get("camera_id"), i.get("width"), i.get("height"))
                      for i in observation.get("images", [])) if image_paths else ()
    text_length = len(_messages_text(messages))
    try:
        exhausted = usage.exhausted(budgets)
        if exhausted:
            raise LoopBudgetExceeded(exhausted)
        max_tokens = config.max_tokens_per_response
        if budgets.max_total_tokens is not None:
            estimated_prompt = _estimated_tokens(_messages_text(messages))
            if isinstance(client, OpenAICompatibleModelClient):
                estimated_prompt += 8192 * len(image_paths)
                basis = getattr(client, "_w4_prompt_budget_basis", None)
                if (signature and basis and basis["signature"] == signature
                        and text_length >= basis["text_length"]):
                    # Same episode and camera geometry: provider usage is a
                    # better basis than reserving 8192 for every small image.
                    # Actual usage is still checked before any action executes.
                    estimated_prompt = (basis["prompt_tokens"]
                        + math.ceil((text_length - basis["text_length"]) / 2)
                        + max(1024, math.ceil(basis["prompt_tokens"] * .1)))
            remaining = budgets.max_total_tokens - usage.total_tokens
            if remaining <= estimated_prompt:
                raise LoopBudgetExceeded(["max_total_tokens"])
            max_tokens = min(max_tokens, remaining - estimated_prompt)
        error: ModelRequestError | None = None
        logical_id = hashlib.sha256(os.urandom(16)).hexdigest()
        for retry in range(budgets.llm_num_retries + 1):
            started = time.monotonic()
            try:
                if loop is not None:
                    config.request_timeout_seconds = loop.phase_timeout(phase_limit)
                exhausted = usage.exhausted(budgets)
                if exhausted:
                    raise LoopBudgetExceeded(exhausted)
                if hasattr(client, "last_request_id"):
                    client.last_request_id = None
                usage.model_request_attempts += 1
                usage.request_retries += int(retry > 0)
                _request_retry_event(client, event="attempt_started", logical_request_id=logical_id,
                                     attempt=retry + 1, turn=(getattr(client, "request_observation", None) or {}).get("turn"))
                completion = client.complete(messages, max_tokens=max_tokens, **(client_kwargs or {}))
                usage.add(completion)
                _request_retry_event(client, event="attempt_succeeded", logical_request_id=logical_id,
                    attempt=retry + 1, elapsed_seconds=time.monotonic() - started,
                    usage={k: getattr(completion, k, None) for k in ("prompt_tokens", "completion_tokens",
                        "total_tokens", "reasoning_tokens", "visible_tokens", "provider_usage_available", "usage_mode")})
                if signature and completion.provider_usage_available:
                    client._w4_prompt_budget_basis = dict(signature=signature,
                        text_length=text_length, prompt_tokens=completion.prompt_tokens)
                violated = usage.violated(budgets)
                if violated:
                    raise LoopBudgetExceeded(violated)
                # max_tokens requests a visible-text cap. Reasoning is retained
                # separately and charged to the case total above. Provider
                # completion_tokens may include thinking and special tokens.
                if loop is not None:
                    loop.phase_timeout(phase_limit)
                return completion
            except ModelRequestError as exc:
                error = exc
                reported = _reported_error_usage(exc)
                if reported is not None:
                    usage.add(reported)
                else:
                    usage.unknown_usage_attempts += 1
                retry_allowed = exc.retryable and retry < budgets.llm_num_retries
                _request_retry_event(client, event="attempt_failed", logical_request_id=logical_id,
                    attempt=retry + 1, elapsed_seconds=time.monotonic() - started,
                    error_type=type(exc).__name__, category="model_api", status_code=exc.status_code,
                    retry_allowed=retry_allowed,
                    usage=({k: getattr(reported, k) for k in ("prompt_tokens", "completion_tokens", "total_tokens")}
                           if reported is not None else None))
                if not retry_allowed:
                    raise
                exhausted = usage.exhausted(budgets)
                if exhausted:
                    raise LoopBudgetExceeded(exhausted) from exc
                delay = min(10.0 * (2 ** min(retry, 2)), 40.0)
                if loop is not None:
                    delay = loop.phase_timeout(delay)
                _request_retry_event(client, event="retry_wait_started", logical_request_id=logical_id,
                                     attempt=retry + 1, planned_wait_seconds=delay)
                wait_started = time.monotonic()
                try:
                    time.sleep(delay)
                finally:
                    waited = time.monotonic() - wait_started
                    usage.retry_wait_seconds += waited
                    _request_retry_event(client, event="retry_wait_finished", logical_request_id=logical_id,
                                         attempt=retry + 1, actual_wait_seconds=waited)
        assert error is not None
        raise error
    finally:
        config.request_timeout_seconds = phase_limit


def _native_call_may_step(name: str, kwargs: Mapping[str, Any]) -> bool:
    """Conservative action detection for opt-in W4 action-only verification.

    Unknown operations are treated as actions. Failed action calls can have
    partial physical effects, so their status is intentionally not consulted.
    """
    if name == "submit_capx_action":
        name = str(kwargs.get("action") or "")
    return not name.startswith(("observe_", "inspect_", "get_", "list_",
                               "enumerate_", "locate_", "ground_", "query_",
                               "compose_", "record_", "build_", "sample_"))


class NativePrimitiveBackend:
    """Budgeted pass-through to each benchmark's public native primitives."""

    def __init__(self, backend, *, public_rgb=None):
        self._backend = backend
        self.public_rgb = public_rgb
        self._limits = {}
        self._used = {"primitive_calls": 0, "verifier_calls": 0}
        self.called = []
        self.action_calls = 0

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def reset(self, task_id, *, seed=None, config=None, budgets=None):
        self._limits = dict(budgets or {})
        self._used = {"primitive_calls": 0, "verifier_calls": 0}
        self.called = []
        self.action_calls = 0
        task = self._backend.reset(task_id, seed=seed, config=config)
        task.budgets.update(self._limits)
        return task

    def unavailable_interfaces(self):
        return []

    def _budget_snapshot(self):
        return {
            "limits": dict(self._limits), "used": dict(self._used),
            "remaining": {k: max(0, v - self._used[k]) for k, v in self._limits.items()},
            "exhausted": [k for k, v in self._limits.items() if self._used[k] >= v],
        }

    def _consume(self, name):
        limit = self._limits.get(name)
        if limit is not None and self._used[name] >= limit:
            raise LoopBudgetExceeded([name])
        self._used[name] += 1

    def call_primitive(self, name, **kwargs):
        if name not in {card.name for card in self._backend.list_primitives()}:
            return PrimitiveResult(name=name, ok=False, output={}, error="unknown_native_primitive")
        self._consume("primitive_calls")
        if _native_call_may_step(name, kwargs):
            self.action_calls += 1
        if name not in self.called:
            self.called.append(name)
        result = self._backend.call_primitive(name, **kwargs)
        if self.public_rgb is not None:
            self.public_rgb.capture(name, result)
        trace_result = (self.public_rgb.trace_result(result)
                        if self.public_rgb is not None and name == "observe_cliport_rgbd" and result.ok
                        else result.to_dict())
        self._backend.record_event("native_gateway_call", {
            "name": name, "kwargs": kwargs, "result": trace_result,
        })
        return result

    def verify(self, scope="task", **kwargs):
        self._consume("verifier_calls")
        return self._backend.verify(scope=scope, **kwargs)


class NativeAgentLoop:
    def __init__(
        self,
        *,
        model_config: NativeModelConfig | None,
        budgets: NativeLoopBudgets,
        trace_root: str | Path,
        in_process: bool = False,
        code_timeout_seconds: float = 300.0,
        native_timeout_seconds: float | None = None,
        trial_timeout_seconds: float | None = None,
        replay_code: str | None = None,
        client: NativeModelClient | None = None,
        require_content_seal: bool = True,
        interface_mode: str = "universal",
        public_rgb_feedback: bool = False,
        direct_rgb_feedback: bool = False,
    ) -> None:
        if replay_code is None and model_config is None:
            raise ValueError("model_config is required unless replay_code is provided")
        if interface_mode not in {"native", "universal"}:
            raise ValueError("interface_mode must be native or universal")
        self.interface_mode = interface_mode
        self.public_rgb_feedback = public_rgb_feedback
        self.direct_rgb_feedback = direct_rgb_feedback
        if direct_rgb_feedback and interface_mode != "native":
            raise ValueError("Direct W4 RGB requires native interface mode")
        if direct_rgb_feedback and (budgets.max_verifier_calls is None
                                    or budgets.max_verifier_calls <= budgets.max_agent_iterations):
            raise ValueError("W4 direct RGB requires max_verifier_calls >= max_agent_iterations + 1")
        if direct_rgb_feedback and model_config is not None and model_config.provider == MODEL_PROVIDER_CURSOR_EXEC:
            raise ValueError("W4 direct RGB requires API image_url or Codex --image; Cursor image delivery is not validated")
        self.model_config = model_config
        self.budgets = budgets
        self.trace_root = Path(trace_root)
        self.in_process = in_process
        self.code_timeout_seconds = code_timeout_seconds
        self.native_timeout_seconds = native_timeout_seconds
        self.trial_timeout_seconds = trial_timeout_seconds
        self.replay_code = replay_code
        self.require_content_seal = require_content_seal
        if client is not None:
            self.client = client
        elif model_config is None:
            self.client = None
        elif model_config.provider == MODEL_PROVIDER_CODEX_EXEC:
            self.client = CodexExecModelClient(model_config)
        elif model_config.provider == MODEL_PROVIDER_CURSOR_EXEC:
            self.client = CursorExecModelClient(model_config)
        else:
            self.client = OpenAICompatibleModelClient(model_config)
        if self.direct_rgb_feedback and self.client is not None:
            self.client.request_archive_dir = self.trace_root / "model_requests"
        self.usage = UsageLedger()

    def _completion(self, messages: list[dict[str, str]], *, loop: EpisodeLoop | None = None, image_paths=None) -> ModelCompletion:
        if self.replay_code is not None:
            return ModelCompletion(
                content=self.replay_code,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                provider_usage_available=False,
                usage_mode="replay",
            )
        if self.client is None or self.model_config is None:
            raise RuntimeError("model client is unavailable")
        return completion_with_budget(client=self.client, config=self.model_config,
                                      budgets=self.budgets, usage=self.usage,
                                      messages=messages, loop=loop,
                                      client_kwargs={"image_paths": image_paths} if image_paths else None)

    def run_case(self, case_id: str, *, seed: int | None = None) -> dict[str, Any]:
        started_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        attempts: list[dict[str, Any]] = []
        top_exception: dict[str, Any] | None = None
        success = False
        for attempt_number in range(1, self.budgets.max_agent_attempts + 1):
            if self.usage.exhausted(self.budgets):
                break
            try:
                attempt = self._run_attempt(
                    case_id, attempt_number=attempt_number, seed=seed
                )
            except Exception as exc:  # noqa: BLE001 - report every case failure.
                top_exception = _exception_payload(exc)
                attempt = {
                    "attempt_number": attempt_number,
                    "success": False,
                    "exception": top_exception,
                    "budget_exhausted": (
                        exc.names if isinstance(exc, LoopBudgetExceeded) else []
                    ),
                    "turns": [],
                }
            attempts.append(attempt)
            success = bool(attempt.get("success"))
            attempt_exception = attempt.get("exception")
            top_exception = (
                dict(attempt_exception) if isinstance(attempt_exception, dict) else None
            )
            if success or _native_score_completed(attempt):
                top_exception = None
                break
            if self.replay_code is not None:
                break
            if top_exception and not top_exception.get("retryable"):
                break
        exhausted = list(self.usage.exhausted(self.budgets))
        for attempt in attempts:
            for name in attempt.get("budget_exhausted", []):
                if name not in exhausted:
                    exhausted.append(name)
        scored = bool(attempts and _native_score_completed(attempts[-1]) and top_exception is None and not exhausted)
        if success:
            outcome = "success"
        elif scored:
            outcome = "scored"
        elif top_exception and top_exception.get("category") == "timeout":
            outcome = "timeout"
        elif exhausted:
            outcome = "budget_exhausted"
        elif top_exception and top_exception.get("category") in {"model_api", "native_runtime", "environment"}:
            outcome = "runtime_failure"
        elif attempts and any(
            item.get("verifier", {}).get("attempted") for item in attempts
        ):
            outcome = "task_failure"
        elif top_exception and top_exception.get("category") in {
            "environment",
            "native_runtime",
            "timeout",
        }:
            outcome = "runtime_failure"
        else:
            outcome = "agent_failure"
        benchmark_id = next(
            (
                str(item.get("benchmark_id"))
                for item in attempts
                if item.get("benchmark_id")
            ),
            None,
        )
        report = {
            "schema_version": NATIVE_LOOP_CONTRACT,
            "stage": "native_agent_case_complete"
            if success or scored
            else "native_agent_case_failed",
            "status": outcome,
            "outcome": outcome,
            "ok": success or scored,
            "case_id": case_id,
            "benchmark_id": benchmark_id,
            "benchmark_group": os.environ.get("ARENA_REPORTING_BENCHMARK", benchmark_id),
            "model": self.model_config.model
            if self.model_config is not None
            else "code-replay",
            "model_provider": self.model_config.provider
            if self.model_config is not None
            else "code_replay",
            "execution_mode": "code_replay"
            if self.replay_code is not None
            else (
                "codex_exec_online"
                if self.model_config is not None
                and self.model_config.provider == MODEL_PROVIDER_CODEX_EXEC
                else (
                    "cursor_exec_online"
                    if self.model_config is not None
                    and self.model_config.provider == MODEL_PROVIDER_CURSOR_EXEC
                    else "model_api_online"
                )
            ),
            "interface_mode": "native-primitives" if self.interface_mode == "native" else "universal-13",
            "official_verifier_agent_callable": False,
            "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": round(time.monotonic() - started, 6),
            "agent_attempts": attempts,
            "llm_usage": self.usage.to_dict(self.budgets),
            "budgets": asdict(self.budgets),
            "budget_exhausted": exhausted,
            "exception": top_exception,
            # Keep the frozen pool coordinate at the case level as well as in
            # the attempt summary.  Batch aggregators generally consume only
            # runner_report.json and should not need to understand the nested
            # turn/attempt schema to join results back to the 1% pool.
            "pool_coordinate": next(
                (
                    item.get("pool_coordinate")
                    for item in attempts
                    if item.get("pool_coordinate") is not None
                ),
                None,
            ),
            "pool_binding": next(
                (
                    item.get("pool_binding")
                    for item in attempts
                    if item.get("pool_binding") is not None
                ),
                {"mode": "none", "bound": False},
            ),
        }
        if not success and not scored:
            report["blocker"] = {
                "type": (
                    top_exception.get("type")
                    if top_exception
                    else "NativeAgentTaskIncomplete"
                ),
                "category": top_exception.get("category") if top_exception else outcome,
                "message": (
                    top_exception.get("message")
                    if top_exception
                    else "Agent loop ended without harness-side task success"
                ),
                "retryable": (
                    bool(top_exception.get("retryable"))
                    if top_exception
                    else outcome == "task_failure"
                ),
            }
        return report

    def _run_attempt(
        self, case_id: str, *, attempt_number: int, seed: int | None
    ) -> dict[str, Any]:
        case = load_native_case(case_id)
        backend = None
        binding: NativeRuntimeBinding | None = None
        launch_receipt: dict[str, Any] | None = None
        gateway: UniversalEmbodiedBackend | None = None
        turns: list[dict[str, Any]] = []
        verifier = VerificationResult(
            ok=False, scope="task", message="not attempted", metadata={}
        )
        verifier_attempted = False
        budget_exhausted: list[str] = []
        attempt_exception: dict[str, Any] | None = None
        interface_fingerprint = ""
        public_trace: dict[str, Any] = {}
        pool_coordinate = None
        pool_binding: dict[str, Any] = {"mode": "none", "bound": False}
        try:
            if not self.in_process and self.require_content_seal:
                # Fail closed before paying simulator startup cost or exposing
                # a mutated runtime to a task process.
                launch_receipt = validate_native_runtime_launch_receipt(
                    case.benchmark_id
                )
            backend, binding = make_native_backend(
                case,
                in_process=self.in_process,
                timeout_seconds=self.native_timeout_seconds,
            )
            from .subprocess_backend_bridge import SubprocessBackendBridge
            if isinstance(backend, SubprocessBackendBridge):
                backend.strict_episode = True
            direct_rgb = None
            if self.direct_rgb_feedback:
                from .w4_rgb import W4RGBFeedback, SUPPORTED
                if case.benchmark_id not in SUPPORTED:
                    raise ValueError("Direct RGB is restricted to supported W4 backends")
                direct_rgb = W4RGBFeedback(backend, case.benchmark_id, self.trace_root / f"rgb-{attempt_number:03d}")
            public_rgb = None
            if not self.direct_rgb_feedback and self.public_rgb_feedback and self.interface_mode == "native" and case.benchmark_id == "cliport":
                from .native_visual_feedback import PublicRGBFeedback, CLIPORT_VISUAL_INSTRUCTIONS
                public_rgb = PublicRGBFeedback(self.trace_root / "public_rgb")
            gateway = (NativePrimitiveBackend(backend, public_rgb=public_rgb) if self.interface_mode == "native"
                       else UniversalEmbodiedBackend(backend))
            # The scheduler carries the frozen 1% coordinate in POOL_* env
            # variables.  Resolve it only after the runtime is constructed so
            # ordinary representative runs remain byte-for-byte compatible.
            # A selected pool task must bind to its upstream episode before
            # reset; metadata alone cannot turn a representative case into
            # a valid sampled evaluation case.
            pool_coordinate = read_pool_coordinate(
                fallback_seed=case.seed if seed is None else seed
            )
            pool_binding = bind_pool_coordinate(backend, pool_coordinate)
            if pool_coordinate is not None and pool_binding.get("bound") is not True:
                raise NativeRuntimeUnavailable(
                    "Selected evaluation pool task has no native episode binding: "
                    f"{pool_coordinate.task_id!r}; {pool_binding}. "
                    "Bind the actual upstream task before starting a 1% evaluation."
                )
            reset_seed = effective_seed(case.seed, seed, pool_coordinate)
            task = gateway.reset(
                case.task_id,
                seed=reset_seed,
                config=dict(case.reset_config),
                budgets={
                    name: limit for name, limit in {
                        "primitive_calls": self.budgets.max_primitive_calls,
                        "verifier_calls": self.budgets.max_verifier_calls,
                    }.items() if limit is not None
                },
            )
            attach_pool_coordinate(
                task,
                pool_coordinate,
                case_id=case.case_id,
                canonical_task_id=case.task_id,
                canonical_seed=case.seed,
                reset_config=case.reset_config,
                binding=pool_binding,
            )
            record_pool_binding(gateway, pool_coordinate, pool_binding)
            observation = gateway.observe()
            cards = gateway.list_primitives()
            unavailable_interfaces = gateway.unavailable_interfaces()
            interface_fingerprint = _interface_fingerprint(
                case_id=case.case_id,
                benchmark_id=case.benchmark_id,
                cards=cards,
                unavailable_interfaces=unavailable_interfaces,
            )
            prompt = task_prompt(
                task=_compact_prompt_value(task.to_dict()),
                observation=_compact_prompt_value(observation.to_dict()),
                primitives=cards,
                interface_mode=self.interface_mode,
                unavailable_interfaces=unavailable_interfaces,
            )
            prompt += (
                "\nAgent-loop budget JSON:\n"
                + json.dumps(
                    _loop_budget_payload(
                        self.usage,
                        self.budgets,
                        attempt_number=attempt_number,
                        completed_turns=0,
                    ),
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            messages = [
                {
                    "role": "system",
                    "content": system_prompt(interface_mode=self.interface_mode),
                },
                {"role": "user", "content": prompt},
            ]
            if direct_rgb is not None:
                from .w4_rgb import INSTRUCTIONS
                messages[0]["content"] += "\n" + INSTRUCTIONS
            if public_rgb is not None:
                messages[0]["content"] += "\n" + CLIPORT_VISUAL_INSTRUCTIONS
            runner = StatefulCodeRunner(
                gateway,
                timeout_seconds=(self.code_timeout_seconds if os.environ.get("ARENA_EXPLICIT_CODE_BUDGET") == "1" else case.code_timeout_seconds or self.code_timeout_seconds),
            )
            runner.bind_task(task)
            runner.globals.pop("backend", None)
            max_iterations = (
                1 if self.replay_code is not None else self.budgets.max_agent_iterations
            )
            code_phase_timeout = runner.timeout_seconds
            model_phase_timeout = self.model_config.request_timeout_seconds if self.model_config else 0
            loop = EpisodeLoop(max_iterations, self.trial_timeout_seconds or
                               max_iterations * (code_phase_timeout + model_phase_timeout) + 30,
                               replay=self.replay_code is not None)
            solved = False
            for turn_number in loop:
                sent_images = []
                if direct_rgb is not None and self.client is not None:
                    # A pre-request budget/vision failure must not inherit
                    # the previous turn's request ID.
                    self.client.last_request_id = None
                try:
                    active_rgb = direct_rgb or public_rgb
                    if direct_rgb is not None:
                        direct_rgb.refresh(turn_number)
                        messages[-1]["content"] += direct_rgb.describe()
                    sent_images = list(active_rgb.images) if active_rgb is not None else []
                    if direct_rgb is not None and self.client is not None:
                        self.client.request_observation = dict(turn=turn_number, attempt=attempt_number,
                                                               images=sent_images)
                    completion = (self._completion(messages, loop=loop, image_paths=active_rgb.paths)
                                  if active_rgb is not None else self._completion(messages, loop=loop))
                except (LoopBudgetExceeded, ModelRequestError, TimeoutError) as exc:
                    attempt_exception = _exception_payload(exc)
                    if isinstance(exc, LoopBudgetExceeded):
                        for name in exc.names:
                            if name not in budget_exhausted:
                                budget_exhausted.append(name)
                    trace_dict = gateway.get_trace().to_dict()
                    gateway_budget = _gateway_budget(gateway)
                    for name in gateway_budget.get("exhausted") or []:
                        if name not in budget_exhausted:
                            budget_exhausted.append(str(name))
                    turns.append(
                        {
                            "turn": turn_number,
                            "stage": "model_completion",
                            "model_images": sent_images,
                            "model_request_id": getattr(self.client, "last_request_id", None),
                            "code": "",
                            "execution_ok": False,
                            "execution_error": str(exc),
                            "stdout": "",
                            "public_contract": (_native_public_contract(gateway) if self.interface_mode == "native"
                                                else _public_contract(_called_interfaces(trace_dict))),
                            "public_progress": None,
                            "gateway_budget": gateway_budget,
                            "agent_loop_budget": _loop_budget_payload(
                                self.usage,
                                self.budgets,
                                attempt_number=attempt_number,
                                completed_turns=turn_number - 1,
                            ),
                            "harness_verifier_attempted": verifier_attempted,
                            "harness_verifier_ok": (
                                verifier.ok if verifier_attempted else False
                            ),
                            "exception": attempt_exception,
                        }
                    )
                    loop.finish_turn(stage="model_completion", error=str(exc),
                                     timed_out=is_timeout(exc), budget_exhausted=budget_exhausted)
                    break
                raw = completion.content
                code = extract_python_code(raw)
                contract_error = _universal_code_contract_error(
                    code, unavailable_interfaces=unavailable_interfaces,
                    native_names={card.name for card in cards} if self.interface_mode == "native" else None,
                )
                try:
                    runner.timeout_seconds = loop.phase_timeout(code_phase_timeout)
                except TimeoutError as exc:
                    loop.finish_turn(timed_out=True, error=str(exc))
                    attempt_exception = _exception_payload(exc)
                    break
                actions_before = getattr(gateway, "action_calls", 0)
                execution = (
                    ExecutionResult(ok=False, error=contract_error)
                    if contract_error
                    else runner.execute(code)
                )
                episode_lost = bool(getattr(backend, "episode_lost", False))
                if episode_lost:
                    execution.ok = False
                    execution.error = "native_episode_lost: backend process exited; case ended without resetting the environment"
                    attempt_exception = _exception_payload(RuntimeError(execution.error))
                trace_dict = gateway.get_trace().to_dict()
                called = _called_interfaces(trace_dict)
                public_contract = (_native_public_contract(gateway) if self.interface_mode == "native"
                                   else _public_contract(called))
                progress = (_compact_prompt_value(execution.result.to_dict())
                            if self.interface_mode == "native" and isinstance(execution.result, PrimitiveResult)
                            else _progress_payload(execution))
                action_only_verification = (
                    os.environ.get("ARENA_VERIFY_AFTER_ACTION_ONLY") == "1"
                    and self.interface_mode == "native"
                    and case.benchmark_id in {"robocasa", "robocasa365", "capx", "robowits"}
                )
                verification_due = (not action_only_verification
                                    or gateway.action_calls > actions_before)
                verified_this_turn = False
                # A completed physical action survives an ordinary Python error
                # later in the cell. Verify that state without resetting it.
                recoverable_python_error = (
                    self.interface_mode == "native"
                    and gateway.action_calls > actions_before
                    and not contract_error and not episode_lost
                    and (execution.error or "").split(":", 1)[0] in {
                        "ZeroDivisionError", "OverflowError", "TypeError", "ValueError",
                        "NameError", "UnboundLocalError", "KeyError", "IndexError",
                        "AttributeError", "AssertionError",
                    }
                )
                if (execution.ok or recoverable_python_error) and public_contract["satisfied"] and verification_due:
                    verifier = gateway.verify(scope="task")
                    verifier_attempted = True
                    verified_this_turn = True
                solved = public_contract["satisfied"] and verified_this_turn and verifier.ok
                gateway_budget = _gateway_budget(gateway)
                budget_exhausted = list(gateway_budget.get("exhausted") or [])
                turns.append(
                    {
                        "turn": turn_number,
                        "stage": "agent_execution",
                        "code": code,
                        "execution_ok": execution.ok,
                        "execution_error": execution.error,
                        "stdout": execution.stdout[-4000:],
                        "model_images": sent_images,
                        "model_request_id": getattr(self.client, "last_request_id", None),
                        "public_contract": public_contract,
                        "public_progress": progress,
                        "gateway_budget": gateway_budget,
                        "harness_verifier_attempted": verified_this_turn,
                        "verification_policy": "after_action" if action_only_verification else "after_execution",
                        "verification_skipped_read_only": action_only_verification and not verification_due,
                        "harness_verifier_ok": verifier.ok
                        if verified_this_turn
                        else False,
                    }
                )
                loop.finish_turn(execution_ok=execution.ok, success=solved,
                                 terminal=bool(verified_this_turn and verifier.metadata.get("native_episode_done", False)),
                                 timed_out=bool(execution.error and execution.error.startswith("TimeoutError:")),
                                 budget_exhausted=budget_exhausted, error=execution.error,
                                 stage="native_runtime" if episode_lost else "execution")
                if loop.stop_reason == "timeout":
                    solved = False
                    attempt_exception = _exception_payload(TimeoutError(execution.error or "Global trial deadline exhausted"))
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": _repair_message(
                            turn=turn_number,
                            interface_mode=self.interface_mode,
                            execution=execution,
                            contract=public_contract,
                            progress=progress,
                            gateway_budget=gateway_budget,
                            last_error=_last_public_error(trace_dict),
                            loop_budget=_loop_budget_payload(
                                self.usage,
                                self.budgets,
                                attempt_number=attempt_number,
                                completed_turns=turn_number,
                            ),
                            official_passed=verifier.ok
                            if verifier_attempted
                            else False,
                        ),
                    }
                )
                if public_rgb is not None:
                    image_info = [{"image_index": i, **{k: item[k] for k in ("camera_index", "width", "height")}}
                                  for i, item in enumerate(public_rgb.images)]
                    messages[-1]["content"] += "\nPublic RGB attached to this request, in order: " + json.dumps(image_info)
            budget_exhausted = list(dict.fromkeys([*budget_exhausted, *loop.budget_exhausted]))
            trace = gateway.get_trace().to_dict()
            if not solved:
                trace["final_status"] = "terminated" if loop.stop_reason == "terminal" else "failed"
            public_trace = (dict(trace, trace_visibility="evaluator_native")
                            if self.interface_mode == "native" else public_universal_trace(trace))
            summary = {
                "schema_version": NATIVE_LOOP_CONTRACT,
                "case_id": case.case_id,
                "benchmark_id": case.benchmark_id,
                "attempt_number": attempt_number,
                "success": solved,
                "pool_coordinate": (
                    pool_coordinate.to_dict() if pool_coordinate is not None else None
                ),
                "pool_binding": dict(pool_binding),
                "exception": attempt_exception,
                "interface_fingerprint": interface_fingerprint,
                "turns": turns,
                "episode_loop": loop.report(),
                "native_episode_lost": bool(getattr(backend, "episode_lost", False)),
                "verifier": {
                    "attempted": verifier_attempted,
                    "ok": verifier.ok if verifier_attempted else False,
                    "message": verifier.message
                    if verifier_attempted
                    else "not attempted",
                    "metrics": verifier.metrics if verifier_attempted else {},
                    "metadata": verifier.metadata if verifier_attempted else {},
                },
                "runtime": (
                    {
                        **binding.to_public_dict(),
                        "content_receipt": launch_receipt,
                    }
                    if binding is not None
                    else {"uses_subprocess": False, "content_receipt": None}
                ),
            }
            artifacts = _write_attempt_artifacts(
                self.trace_root,
                attempt_number=attempt_number,
                summary=summary,
                public_trace=public_trace,
            )
            return {
                **summary,
                "budget_exhausted": budget_exhausted,
                "trace_event_count": len(public_trace.get("events", [])),
                "artifact_count": len(public_trace.get("artifacts", {})),
                "artifacts": artifacts,
            }
        finally:
            close = getattr(backend, "close", None) if backend is not None else None
            if callable(close):
                close()


def _native_score_completed(attempt: dict[str, Any]) -> bool:
    verifier = attempt.get("verifier", {})
    return bool(attempt.get("episode_loop", {}).get("stop_reason") == "terminal"
                and verifier.get("attempted")
                and verifier.get("metadata", {}).get("native_success_defined") is False
                and "native_episode_return" in verifier.get("metrics", {}))
