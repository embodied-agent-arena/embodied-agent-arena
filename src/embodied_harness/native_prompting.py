"""Dependency-light prompting and public trace helpers for the native loop.

This module intentionally imports only control-plane schema code.  In
particular, it must not import the legacy API runner or benchmark adapters:
those pull simulator dependencies into every campaign controller process.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from .schemas import PrimitiveCard
from .universal_interface import (
    UNIVERSAL_INTERFACE_NAMES,
    _sanitize_agent_payload,
    universal_adapter_profiles,
    universal_contract_manifest,
)


def extract_python_code(text: str) -> str:
    fenced = re.search(
        r"```(?:python|py)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def public_universal_trace(trace: dict[str, Any]) -> dict[str, Any]:
    """Return only the trace surface available to an evaluated agent."""

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
        "metrics": _hide_native_trace_details(
            trace.get("metrics", {}), native_names
        ),
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
    if isinstance(value, (list, tuple)):
        return [
            _hide_native_trace_details(item, native_names)
            for item in value
            if not _is_private_trace_string(item, native_names)
        ]
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
    return (
        lowered in blocked
        or lowered.endswith("_api")
        or lowered.endswith("_apis")
    )


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


def trace_dict_to_jsonl(trace: dict[str, Any]) -> str:
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


def system_prompt(*, interface_mode: str = "universal") -> str:
    if interface_mode == "native":
        return (
            "You are a coding agent controlling an embodied benchmark. Return executable Python only. "
            "Use primitives.call(name='<native primitive>', keyword=value, ...) with the listed native names. "
            "Check each PrimitiveResult.ok and read its output; print observations or assign result for feedback. "
            "Python variables and memory persist between code cells in the same episode. "
            "The harness executes the code and returns feedback after each turn. "
            "Keep native action arguments as documented. Do not import modules, open files or access private attributes. "
            "Dynamic attribute access (getattr/hasattr/setattr) is unavailable; use documented result.ok/result.output fields. "
            "Write primitive arguments explicitly; do not expand **kwargs. Only pass fields listed for that primitive. "
            "For exception handling use Exception, not unavailable RuntimeError/ValueError classes. "
            "Use a for loop instead of next(), which is also unavailable. "
            "PrimitiveResult is not JSON data: use result.ok/result.output rather than serializing the result object. "
            "Print compact decision-relevant fields instead of full arrays or repeated scene dictionaries. "
            "For continuous control, test a short bounded batch and check actual displacement before extending it. "
            "If repeated actions do not improve the observed state, change the documented control parameters or approach. "
            "Do not call backend or a verifier: the harness runs the benchmark's original verifier after execution."
        )
    if interface_mode != "universal":
        raise ValueError("the native harness exposes only the universal interface")
    return (
        "You are a coding agent controlling an embodied benchmark through the "
        "Agentic Embodied Arena universal interface. Return only executable "
        "Python code. Do not include prose. Do not import modules, open files, "
        "or use unsafe calls. Use only primitives.call(interface='<name>', "
        "keyword=value, ...) where <name> is one of the 13 high-level "
        "interfaces. Never call backend.verify(); official scoring is "
        "harness-side only. Every interface call must use keyword arguments. "
        "Check PrimitiveResult.ok before reading output fields, use output.get() "
        "for optional fields, and never assume a failed call returned a handle. "
        "Do not call an interface listed as unavailable for this benchmark. "
        "Store useful handles only from successful PrimitiveResult objects. Record evidence before "
        "preparing or executing actions. Finish by assigning result = "
        "primitives.call(interface='progress.check')."
    )


def task_prompt(
    task: dict[str, Any],
    observation: dict[str, Any],
    primitives: list[PrimitiveCard],
    *,
    interface_mode: str = "universal",
    unavailable_interfaces: list[str] | tuple[str, ...] = (),
) -> str:
    if interface_mode == "native":
        return (
            "Solve the task using the benchmark's native primitives.\n\n"
            f"Task spec JSON:\n{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n\n"
            f"Initial observation JSON:\n{json.dumps(observation, ensure_ascii=False, sort_keys=True)}\n\n"
            "Available native primitives (names and arguments are passed through unchanged):\n"
            + "\n".join(_primitive_hint(card) for card in primitives)
            + "\n\nUse primitives.call(name='<listed name>', argument=value, ...). "
            "Write arguments explicitly as keyword arguments. Check result.ok before using result.output. "
            "The initial observation and task are already supplied; no extra interface rituals are required."
        )
    if interface_mode != "universal":
        raise ValueError("unknown interface mode")
    primitive_lines = "\n".join(_primitive_hint(card) for card in primitives)
    unavailable = ", ".join(unavailable_interfaces) or "none"
    return (
        "Solve this embodied benchmark task by writing one Python code cell.\n\n"
        f"Task spec JSON:\n{json.dumps(task, ensure_ascii=False, sort_keys=True)}\n\n"
        "Initial observation JSON:\n"
        f"{json.dumps(observation, ensure_ascii=False, sort_keys=True)}\n\n"
        "Available high-level interfaces:\n"
        f"{primitive_lines}\n\n"
        f"Unavailable for this benchmark: {unavailable}.\n\n"
        "Universal interface contract:\n"
        f"{_universal_prompt_contract()}\n\n"
        "Rules:\n"
        "- Use only primitives.call(interface='<interface-name>', keyword=value, ...).\n"
        f"- Interface names are exactly: {', '.join(UNIVERSAL_INTERFACE_NAMES)}.\n"
        "- Never call an interface listed as unavailable for this benchmark.\n"
        "- After every call, check result.ok before reading output; use output.get() for optional fields.\n"
        "- Do not call benchmark-native primitive names; they are hidden behind the adapter.\n"
        "- Do not call backend.verify(); final official scoring is hidden from you and run by the harness.\n"
        "- Start with task.context and scene.observe when useful, then enumerate/inspect/locate entities.\n"
        "- Record at least one evidence handle with evidence.record before action.prepare.\n"
        "- Execute only action handles returned by action.prepare.\n"
        "- Finish with: result = primitives.call(interface='progress.check').\n"
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
        + ", ".join(
            manifest["evidence_contract"]["evidence_required_interfaces"]
        ),
        "- official verifier and low-level native parameters are harness-side only.",
        "- interface handle/error map:",
    ]
    for contract in manifest["interface_contracts"]:
        handle_inputs = ", ".join(contract["handle_inputs"]) or "none"
        handle_outputs = ", ".join(contract["handle_outputs"]) or "none"
        error_codes = ", ".join(contract["error_codes"]) or "none"
        requires_evidence = "yes" if contract["requires_evidence"] else "no"
        lines.append(
            f"  * {contract['name']}: inputs=[{handle_inputs}], "
            f"outputs=[{handle_outputs}], requires_evidence={requires_evidence}, "
            f"errors=[{error_codes}]"
        )
    return "\n".join(lines)


def _primitive_hint(card: PrimitiveCard) -> str:
    arguments = ", ".join(
        f"{name}=<{schema}>" for name, schema in card.input_schema.items()
    )
    description = f" — {card.description}" if card.description else ""
    return f"- {card.name}({arguments}){description}"
