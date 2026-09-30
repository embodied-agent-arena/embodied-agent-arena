from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Protocol
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen

from .runner import ExecutionResult, StatefulCodeRunner
from .schemas import PrimitiveCard, VerificationResult
from .universal_interface import (
    UNIVERSAL_CONTRACT_SCHEMA_VERSION,
    UNIVERSAL_INTERFACE_NAMES,
    UniversalEmbodiedBackend,
    _sanitize_agent_payload,
    universal_adapter_profiles,
    universal_contract_manifest,
)
from .w4_benchmark_backend import W4BenchmarkBackend


DEFAULT_MODEL = "qwen3.5-27b"


@dataclass(slots=True)
class APIModelConfig:
    api_key: str
    base_url: str
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_tokens: int = 1200


@dataclass(slots=True)
class AgentTurn:
    turn: int
    code: str
    execution_ok: bool
    verification_ok: bool
    error: str | None = None
    verifier_message: str = ""


@dataclass(slots=True)
class AgentTaskResult:
    task_id: str
    benchmark_id: str
    success: bool
    interface_mode: str = "native"
    turns: list[AgentTurn] = field(default_factory=list)
    final_message: str = ""
    trace_event_count: int = 0
    artifact_count: int = 0
    summary_path: str | None = None
    trace_path: str | None = None
    trace_jsonl_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentBenchmarkReport:
    model: str
    task_count: int
    success_count: int
    results: list[AgentTaskResult]

    @property
    def success_rate(self) -> float:
        return self.success_count / self.task_count if self.task_count else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "task_count": self.task_count,
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "results": [result.to_dict() for result in self.results],
        }


class ChatClient(Protocol):
    def complete(self, messages: list[dict[str, str]], *, model: str, temperature: float, max_tokens: int) -> str:
        ...


class OpenAICompatibleChatClient:
    def __init__(self, api_key: str, base_url: str) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any | None = None
        try:
            from openai import OpenAI
        except ModuleNotFoundError:
            self._client = None
        else:
            self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def complete(self, messages: list[dict[str, str]], *, model: str, temperature: float, max_tokens: int) -> str:
        if self._client is not None:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = response.choices[0].message.content
            return content or ""
        return self._complete_with_urllib(messages, model=model, temperature=temperature, max_tokens=max_tokens)

    def _complete_with_urllib(
        self,
        messages: list[dict[str, str]],
        *,
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenAI-compatible API request failed: HTTP {exc.code}: {detail}") from exc
        try:
            return str(data["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"OpenAI-compatible API response missing choices[0].message.content: {data}") from exc


def load_api_model_config(env_file: str | Path = ".env", model: str | None = None) -> APIModelConfig:
    data = _parse_env_file(Path(env_file))
    api_key = _first_present(data, "API_KEY", "OPENAI_API_KEY", "apikey", "api_key")
    base_url = _first_present(data, "API_BASE_URL", "OPENAI_BASE_URL", "BASE_URL", "base_url", "api_base_url")
    configured_model = model or _first_present(
        data,
        "MODEL",
        "API_MODEL",
        "OPENAI_MODEL",
        "DEFAULT_MODEL",
        "model",
        "model_name",
        required=False,
    )
    if not api_key:
        raise ValueError(f"Missing API key in {env_file}. Expected API_KEY/OPENAI_API_KEY or apikey.")
    if not base_url:
        raise ValueError(f"Missing API base URL in {env_file}. Expected API_BASE_URL/OPENAI_BASE_URL or base_url.")
    return APIModelConfig(api_key=api_key, base_url=_normalize_base_url(base_url), model=configured_model or DEFAULT_MODEL)


def _parse_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    data: dict[str, str] = {}
    pending_key: str | None = None
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            key = key.strip().removeprefix("export ").strip()
            data[key] = _clean_env_value(value)
            pending_key = None
            continue
        if line.endswith(":"):
            pending_key = line[:-1].strip()
            continue
        if pending_key:
            data[pending_key] = _clean_env_value(line)
            pending_key = None
    return data


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        cleaned = cleaned[1:-1]
    return cleaned.strip()


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return urlunparse(parsed._replace(path="/v1"))
    return cleaned


def _first_present(data: dict[str, str], *keys: str, required: bool = True) -> str:
    for key in keys:
        value = data.get(key)
        if value:
            return value
    if required:
        return ""
    return ""


def extract_python_code(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


class APICodingAgentRunner:
    def __init__(
        self,
        backend: W4BenchmarkBackend | None = None,
        client: ChatClient | None = None,
        config: APIModelConfig | None = None,
        max_turns: int = 4,
        code_timeout_seconds: float = 2.0,
        interface_mode: str = "native",
        trace_output_dir: str | Path | None = None,
    ) -> None:
        if interface_mode not in {"native", "universal"}:
            raise ValueError("interface_mode must be 'native' or 'universal'.")
        self.native_backend = backend or W4BenchmarkBackend()
        self.backend = UniversalEmbodiedBackend(self.native_backend) if interface_mode == "universal" else self.native_backend
        self.config = config or APIModelConfig(api_key="", base_url="", model=DEFAULT_MODEL)
        self.client = client
        self.max_turns = max_turns
        self.code_timeout_seconds = code_timeout_seconds
        self.interface_mode = interface_mode
        self.trace_output_dir = Path(trace_output_dir) if trace_output_dir is not None else None

    def run_tasks(self, task_ids: list[str]) -> AgentBenchmarkReport:
        results = [self.run_task(task_id) for task_id in task_ids]
        return AgentBenchmarkReport(
            model=self.config.model,
            task_count=len(results),
            success_count=sum(1 for result in results if result.success),
            results=results,
        )

    def run_task(self, task_id: str) -> AgentTaskResult:
        task = self.backend.reset(task_id)
        benchmark_id = task.metadata.get("benchmark_id", "unknown")
        runner = StatefulCodeRunner(self.backend, timeout_seconds=self.code_timeout_seconds)
        runner.bind_task(task)
        observation = self.backend.observe()
        primitive_cards = self.backend.list_primitives()
        messages = [
            {"role": "system", "content": _system_prompt(interface_mode=self.interface_mode)},
            {
                "role": "user",
                "content": _task_prompt(
                    task=task.to_dict(),
                    observation=observation.to_dict(),
                    primitives=primitive_cards,
                    interface_mode=self.interface_mode,
                ),
            },
        ]
        turns: list[AgentTurn] = []
        final_verification = VerificationResult(ok=False, scope="task", message="not attempted")
        solved = False
        for turn_index in range(1, self.max_turns + 1):
            raw = self._complete(messages)
            code = extract_python_code(raw)
            execution = runner.execute(code)
            final_verification = _verification_from_execution(execution)
            if final_verification is None:
                final_verification = self.backend.verify(scope="task")
            turns.append(
                AgentTurn(
                    turn=turn_index,
                    code=code,
                    execution_ok=execution.ok,
                    verification_ok=final_verification.ok,
                    error=execution.error,
                    verifier_message=final_verification.message,
                )
            )
            solved = execution.ok and final_verification.ok
            if solved:
                break
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": _repair_prompt(turn_index, execution, final_verification, interface_mode=self.interface_mode),
                }
            )
        trace = self.backend.get_trace()
        if not solved:
            trace.final_status = "failed"
        trace_paths: dict[str, Path] = {}
        if self.trace_output_dir is not None:
            trace_paths = write_agent_task_trace_artifacts(
                self.trace_output_dir,
                task=task.to_dict(),
                benchmark_id=str(benchmark_id),
                model=self.config.model,
                interface_mode=self.interface_mode,
                success=solved,
                turns=[asdict(turn) for turn in turns],
                final_message=final_verification.message,
                trace=trace.to_dict(),
                trace_jsonl=trace.to_jsonl(),
            )
        return AgentTaskResult(
            task_id=task_id,
            benchmark_id=str(benchmark_id),
            success=solved,
            interface_mode=self.interface_mode,
            turns=turns,
            final_message=final_verification.message,
            trace_event_count=len(trace.events),
            artifact_count=len(trace.artifacts),
            summary_path=str(trace_paths["summary"]) if trace_paths else None,
            trace_path=str(trace_paths["trace"]) if trace_paths else None,
            trace_jsonl_path=str(trace_paths["trace_jsonl"]) if trace_paths else None,
        )

    def _complete(self, messages: list[dict[str, str]]) -> str:
        if self.client is None:
            if not self.config.api_key or not self.config.base_url:
                raise ValueError("A real API run needs api_key/base_url or an injected test client.")
            self.client = OpenAICompatibleChatClient(api_key=self.config.api_key, base_url=self.config.base_url)
        return self.client.complete(
            messages,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )


def _verification_from_execution(execution: ExecutionResult) -> VerificationResult | None:
    result = execution.result
    if isinstance(result, VerificationResult):
        return result
    if hasattr(result, "ok") and hasattr(result, "message") and hasattr(result, "scope"):
        return result
    return None


def write_agent_task_trace_artifacts(
    output_dir: str | Path,
    *,
    task: dict[str, Any],
    benchmark_id: str,
    model: str,
    interface_mode: str,
    success: bool,
    turns: list[dict[str, Any]],
    final_message: str,
    trace: dict[str, Any],
    trace_jsonl: str,
) -> dict[str, Path]:
    task_id = str(task.get("task_id", trace.get("task_id", "unknown-task")))
    task_dir = Path(output_dir) / _safe_trace_path_component(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    summary_path = task_dir / "summary.json"
    trace_path = task_dir / "trace.json"
    trace_jsonl_path = task_dir / "trace.jsonl"
    public_trace = _public_universal_trace(trace) if interface_mode == "universal" else trace
    public_trace_jsonl = _trace_dict_to_jsonl(public_trace) if interface_mode == "universal" else trace_jsonl
    summary = {
        "schema_version": "agentic-embodied-arena-agent-case/v1",
        "task_id": task_id,
        "benchmark_id": benchmark_id,
        "model": model,
        "interface_mode": interface_mode,
        "success": success,
        "final_message": final_message,
        "turn_count": len(turns),
        "trace_event_count": len(public_trace.get("events", [])),
        "artifact_count": len(public_trace.get("artifacts", {})),
        "agent_turns": turns,
        "task": task,
        "arena_contract": {
            "agent_visible_interfaces": list(UNIVERSAL_INTERFACE_NAMES) if interface_mode == "universal" else None,
            "contract_schema_version": UNIVERSAL_CONTRACT_SCHEMA_VERSION if interface_mode == "universal" else None,
            "handle_prefixes": _universal_handle_prefixes() if interface_mode == "universal" else None,
            "evidence_required_interfaces": _universal_evidence_required_interfaces() if interface_mode == "universal" else None,
            "official_verifier_agent_callable": False,
            "requires_evidence_before_execute": interface_mode == "universal",
        },
        "artifacts": {
            "summary": summary_path.name,
            "trace": trace_path.name,
            "trace_jsonl": trace_jsonl_path.name,
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_path.write_text(json.dumps(public_trace, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    trace_jsonl_path.write_text(public_trace_jsonl.rstrip() + "\n", encoding="utf-8")
    return {"summary": summary_path, "trace": trace_path, "trace_jsonl": trace_jsonl_path}


def _safe_trace_path_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "task"


def _public_universal_trace(trace: dict[str, Any]) -> dict[str, Any]:
    allowed_event_prefixes = ("universal_gateway_", "code_")
    native_names = _universal_native_names()
    events = [
        _hide_native_trace_details(_sanitize_agent_payload(event), native_names)
        for event in trace.get("events", [])
        if str(event.get("event_type", "")).startswith(allowed_event_prefixes)
    ]
    artifacts = {
        key: _hide_native_trace_details(
            _sanitize_agent_payload({"visual_artifact": value})["visual_artifact"],
            native_names,
        )
        for key, value in dict(trace.get("artifacts", {})).items()
        if str(key).startswith(("obs:", "ev:", "act:"))
    }
    return {
        "task_id": trace.get("task_id"),
        "events": events,
        "artifacts": artifacts,
        "metrics": _hide_native_trace_details(trace.get("metrics", {}), native_names),
        "final_status": trace.get("final_status"),
        "trace_visibility": "agent_public_universal",
    }


def _hide_native_trace_details(value: Any, native_names: set[str]) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if _is_private_trace_key(key_text) or key_text in native_names:
                continue
            sanitized[key_text] = _hide_native_trace_details(item, native_names)
        return sanitized
    if isinstance(value, list):
        return [_hide_native_trace_details(item, native_names) for item in value if not _is_private_trace_string(item, native_names)]
    if isinstance(value, tuple):
        return [_hide_native_trace_details(item, native_names) for item in value if not _is_private_trace_string(item, native_names)]
    if isinstance(value, Path):
        return str(value)
    if all(hasattr(value, attribute) for attribute in ("shape", "dtype", "size")):
        try:
            element_count = int(value.size)
            if element_count > 4096:
                return {
                    "array_content_omitted": True,
                    "shape": [int(dimension) for dimension in value.shape],
                    "dtype": str(value.dtype),
                    "element_count": element_count,
                }
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _hide_native_trace_details(value.tolist(), native_names)
    if hasattr(value, "item"):
        try:
            return _hide_native_trace_details(value.item(), native_names)
        except (TypeError, ValueError):
            pass
    if _is_private_trace_string(value, native_names):
        return "[native_detail_hidden]"
    return value


def _is_private_trace_key(key: str) -> bool:
    lowered = key.lower()
    blocked = (
        "available_apis",
        "api_trace",
        "backend_trace",
        "config",
        "debug_command",
        "install_hint",
        "live_smoke_command",
        "native_steps",
        "primitive_cards",
        "raw_api",
    )
    return lowered in blocked or lowered.endswith("_api") or lowered.endswith("_apis")


def _is_private_trace_string(value: Any, native_names: set[str]) -> bool:
    if not isinstance(value, str):
        return False
    return any(name and name in value for name in native_names)


def _universal_native_names() -> set[str]:
    names: set[str] = set()
    for profile in universal_adapter_profiles(include_internal_names=True).values():
        for key, value in profile.items():
            if key.endswith("_primitives") and isinstance(value, list):
                names.update(str(item) for item in value)
    return names


def _trace_dict_to_jsonl(trace: dict[str, Any]) -> str:
    lines = [json.dumps(event, sort_keys=True) for event in trace.get("events", [])]
    footer = {
        "event_type": "trace_footer",
        "task_id": trace.get("task_id"),
        "final_status": trace.get("final_status"),
        "metrics": trace.get("metrics", {}),
        "artifacts": trace.get("artifacts", {}),
        "trace_visibility": trace.get("trace_visibility"),
    }
    lines.append(json.dumps(footer, sort_keys=True))
    return "\n".join(lines)


def _system_prompt(interface_mode: str = "native") -> str:
    if interface_mode == "universal":
        return (
            "You are a coding agent controlling an embodied benchmark through the Agentic Embodied Arena "
            "universal interface. Return only executable Python code. Do not include prose. Do not import "
            "modules, open files, or use unsafe calls. Use only primitives.call(interface='<name>', keyword=value, ...) "
            "where <name> is one of the 13 high-level interfaces. Never call backend.verify(); official scoring is "
            "harness-side only. Every interface call must use keyword arguments. Store useful handles from returned "
            "PrimitiveResult objects with result.output['key'] or result['key']. Record evidence before preparing or "
            "executing actions. Finish by assigning result = primitives.call(interface='progress.check')."
        )
    return (
        "You are a coding agent controlling an embodied benchmark through Python primitives. "
        "Return only executable Python code. Do not include prose. Do not import modules, open files, "
        "or use unsafe calls. Use only primitives.<name>(keyword=value, ...) and backend.verify(scope='task'). "
        "Every primitive call must go through the primitives object and must use keyword arguments; "
        "positional primitive arguments are not supported. "
        "Primitive calls return PrimitiveResult objects; read data with result.output['key'] or the shortcut result['key']. "
        "Store the final verifier result in a variable named result."
    )


def _task_prompt(
    task: dict[str, Any],
    observation: dict[str, Any],
    primitives: list[PrimitiveCard],
    interface_mode: str = "native",
) -> str:
    primitive_lines = "\n".join(_primitive_hint(card) for card in primitives)
    if interface_mode == "universal":
        return (
            "Solve this embodied benchmark task by writing one Python code cell.\n\n"
            f"Task spec JSON:\n{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Initial observation JSON:\n{json.dumps(observation, ensure_ascii=False, sort_keys=True)}\n\n"
            "Available high-level interfaces:\n"
            f"{primitive_lines}\n\n"
            "Universal interface contract:\n"
            f"{_universal_prompt_contract()}\n\n"
            "Rules:\n"
            "- Use only primitives.call(interface='<interface-name>', keyword=value, ...).\n"
            f"- Interface names are exactly: {', '.join(UNIVERSAL_INTERFACE_NAMES)}.\n"
            "- Do not call benchmark-native primitive names; they are hidden behind the adapter.\n"
            "- Do not call backend.verify(); final official scoring is hidden from you and run by the harness.\n"
            "- Start with task.context and scene.observe when useful, then enumerate/inspect/locate entities.\n"
            "- Record at least one evidence handle with evidence.record before action.prepare.\n"
            "- Execute only action handles returned by action.prepare.\n"
            "- Finish with: result = primitives.call(interface='progress.check').\n"
        )
    return (
        "Solve this embodied benchmark task by writing one Python code cell.\n\n"
        f"Task spec JSON:\n{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n\n"
        f"Initial observation JSON:\n{json.dumps(observation, ensure_ascii=False, sort_keys=True)}\n\n"
        "Available primitive signatures and usage hints:\n"
        f"{primitive_lines}\n\n"
        "Rules:\n"
        "- Every primitive call must be written as primitives.name(keyword=value, ...). Never call primitive names directly.\n"
        "- Positional primitive arguments are invalid; use keyword arguments even when there is only one argument.\n"
        "- Primitive calls return PrimitiveResult objects. Use res.output['key'] or res['key'] to read returned data.\n"
        "- First inspect/ground the task with observation/get/locate/inspect primitives when available.\n"
        "- Record at least one evidence artifact with primitives.record_w4_evidence(key=..., value=...).\n"
        "- Use the task goal and primitive outputs to choose action arguments.\n"
        "- Finish with: result = backend.verify(scope='task').\n"
    )


def _repair_prompt(
    turn_index: int,
    execution: ExecutionResult,
    verification: VerificationResult,
    interface_mode: str = "native",
) -> str:
    if interface_mode == "universal":
        return (
            f"Turn {turn_index} failed. Write a corrected Python code cell only.\n"
            f"Execution ok: {execution.ok}\n"
            f"Execution error: {execution.error or ''}\n"
            "Harness-side official scoring did not pass or execution failed. The scoring details remain hidden. "
            "Use public failure.diagnose, progress.check, scene.observe, entity.locate, and evidence.record to repair. "
            "Use only primitives.call(interface='<interface-name>', keyword=value, ...) and finish with "
            "result = primitives.call(interface='progress.check')."
        )
    return (
        f"Turn {turn_index} failed. Write a corrected Python code cell only.\n"
        f"Execution ok: {execution.ok}\n"
        f"Execution error: {execution.error or ''}\n"
        f"Verifier ok: {verification.ok}\n"
        f"Verifier message: {verification.message}\n"
        "Use primitives.<name>(keyword=value, ...) exactly; no direct primitive calls and no positional primitive args. "
        "Finish with result = backend.verify(scope='task')."
    )


def _universal_prompt_contract() -> str:
    manifest = universal_contract_manifest(include_internal_adapter_names=False)
    handle_prefixes = {
        name: contract["prefix"]
        for name, contract in manifest["handle_contract"].items()
    }
    lines = [
        f"- schema_version: {manifest['schema_version']}",
        f"- handle_prefixes: {json.dumps(handle_prefixes, sort_keys=True)}",
        "- evidence_required_interfaces: "
        + ", ".join(manifest["evidence_contract"]["evidence_required_interfaces"]),
        "- official verifier and low-level native parameters are harness-side only.",
        "- interface handle/error map:",
    ]
    for contract in manifest["interface_contracts"]:
        handle_inputs = ", ".join(contract["handle_inputs"]) or "none"
        handle_outputs = ", ".join(contract["handle_outputs"]) or "none"
        error_codes = ", ".join(contract["error_codes"]) or "none"
        requires_evidence = "yes" if contract["requires_evidence"] else "no"
        lines.append(
            f"  * {contract['name']}: inputs=[{handle_inputs}], outputs=[{handle_outputs}], "
            f"requires_evidence={requires_evidence}, errors=[{error_codes}]"
        )
    return "\n".join(lines)


def _universal_handle_prefixes() -> dict[str, str]:
    manifest = universal_contract_manifest(include_internal_adapter_names=False)
    return {
        name: contract["prefix"]
        for name, contract in manifest["handle_contract"].items()
    }


def _universal_evidence_required_interfaces() -> list[str]:
    manifest = universal_contract_manifest(include_internal_adapter_names=False)
    return list(manifest["evidence_contract"]["evidence_required_interfaces"])


def _primitive_hint(card: PrimitiveCard) -> str:
    """Render a card without inventing task arguments or an action strategy."""

    arguments = ", ".join(f"{name}=<{schema}>" for name, schema in card.input_schema.items())
    description = f" — {card.description}" if card.description else ""
    return f"- {card.name}({arguments}){description}"


def _resolve_task_ids(backend: W4BenchmarkBackend, task: str, benchmark: str | None) -> list[str]:
    task_ids = backend.list_task_ids()
    if task == "all":
        selected = task_ids
    else:
        selected = [task]
    if benchmark:
        selected = [
            task_id
            for task_id in selected
            if backend._tasks[task_id].benchmark_id == benchmark  # noqa: SLF001 - CLI selector over local smoke manifest.
        ]
    unknown = [task_id for task_id in selected if task_id not in task_ids]
    if unknown:
        raise KeyError(f"Unknown task ids: {unknown}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run API-model coding-agent smoke tests over W4 benchmark primitives.")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default=None, help="Override model from env file. Defaults to MODEL/API_MODEL/DEFAULT_MODEL or qwen3.5-27b.")
    parser.add_argument("--task", default="all", help="Task id or 'all'.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark id filter, e.g. maniskill or calvin.")
    parser.add_argument("--max-turns", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--interface-mode", choices=["native", "universal"], default="native")
    parser.add_argument("--trace-output-dir", default=None, help="Optional directory for per-task summary/trace artifacts.")
    parser.add_argument("--output", default=None, help="Optional JSON report path.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args(argv)

    config = load_api_model_config(args.env_file, model=args.model)
    config.temperature = args.temperature
    config.max_tokens = args.max_tokens
    backend = W4BenchmarkBackend()
    task_ids = _resolve_task_ids(backend, args.task, args.benchmark)
    report = APICodingAgentRunner(
        backend=backend,
        config=config,
        max_turns=args.max_turns,
        interface_mode=args.interface_mode,
        trace_output_dir=args.trace_output_dir,
    ).run_tasks(task_ids)
    payload = report.to_dict()
    text = json.dumps(payload, ensure_ascii=False, indent=args.indent)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.success_count == report.task_count else 1


if __name__ == "__main__":
    raise SystemExit(main())
