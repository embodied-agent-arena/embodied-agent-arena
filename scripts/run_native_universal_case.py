#!/usr/bin/env python3
"""Run one benchmark case with the native lightweight harness and loop."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from _repo_bootstrap import bootstrap_repo_src
except ModuleNotFoundError:  # pragma: no cover
    from scripts._repo_bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from embodied_harness.native_agent_loop import (  # noqa: E402
    MODEL_PROVIDER_CODEX_EXEC,
    MODEL_PROVIDER_CURSOR_EXEC,
    MODEL_PROVIDER_OPENAI_COMPATIBLE,
    NativeAgentLoop,
    NativeLoopBudgets,
    _atomic_json,
    load_native_model_config,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--interface-mode", choices=("native", "universal"), default="universal")
    parser.add_argument("--public-rgb-feedback", action="store_true", help="Attach explicitly observed CLIPort RGB to the next model request")
    parser.add_argument("--direct-rgb-feedback", action="store_true", help="W4 current multiview RGB with exact input archive")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agent-trace-root", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--model-provider",
        choices=("openai-compatible", "codex-exec", "cursor-exec"),
        default="openai-compatible",
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--codex-executable", default="codex")
    parser.add_argument("--codex-reasoning-effort", default=None)
    parser.add_argument("--cursor-executable", default="agent")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--request-timeout-seconds", type=float, default=None)
    parser.add_argument("--code-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--native-timeout-seconds", type=float, default=None)
    parser.add_argument("--trial-timeout-seconds", type=float, default=None)
    parser.add_argument("--max-agent-attempts", type=int, default=1)
    parser.add_argument("--max-agent-iterations", type=int, default=8)
    parser.add_argument("--llm-num-retries", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=None)
    parser.add_argument("--max-primitive-calls", type=int, default=None)
    parser.add_argument("--max-verifier-calls", type=int, default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    parser.add_argument("--input-cost-per-million", type=float, default=None)
    parser.add_argument("--output-cost-per-million", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--code-file", type=Path, default=None)
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument(
        "--allow-unsealed-runtime",
        action="store_true",
        help="Development only: bypass the native content-receipt launch gate.",
    )
    # Accepted only so an older batch manifest remains runnable. The native
    # harness opens no Agent Server and does not consume this port.
    parser.add_argument("--host-port", type=int, default=None, help=argparse.SUPPRESS)
    return parser


def _validate_positive(
    parser: argparse.ArgumentParser, value: float | int | None, name: str
) -> None:
    if value is not None and value <= 0:
        parser.error(f"{name} must be positive")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.request_timeout_seconds is None and args.model_provider in {
        "codex-exec",
        "cursor-exec",
    }:
        # Long reasoning turns can exceed two minutes; the episode deadline
        # still caps this phase, and explicit per-run limits take precedence.
        args.request_timeout_seconds = 360.0
    for name in (
        "max_tokens",
        "request_timeout_seconds",
        "code_timeout_seconds",
        "max_agent_attempts",
        "max_agent_iterations",
        "max_total_tokens",
        "max_primitive_calls",
        "max_verifier_calls",
        "max_cost_usd",
    ):
        _validate_positive(parser, getattr(args, name), name)
    if not 0 <= args.llm_num_retries <= 9:
        parser.error("llm_num_retries must be between 0 and 9")
    replay_code = args.code_file.read_text(encoding="utf-8") if args.code_file else None
    model_config = None
    if replay_code is None:
        try:
            model_config = load_native_model_config(
                args.env_file,
                provider=(
                    MODEL_PROVIDER_CODEX_EXEC
                    if args.model_provider == "codex-exec"
                    else MODEL_PROVIDER_CURSOR_EXEC
                    if args.model_provider == "cursor-exec"
                    else MODEL_PROVIDER_OPENAI_COMPATIBLE
                ),
                model=args.model,
                temperature=args.temperature,
                max_tokens_per_response=args.max_tokens,
                request_timeout_seconds=args.request_timeout_seconds,
                input_cost_per_million=args.input_cost_per_million,
                output_cost_per_million=args.output_cost_per_million,
                codex_executable=args.codex_executable,
                codex_reasoning_effort=args.codex_reasoning_effort,
                cursor_executable=args.cursor_executable,
            )
        except Exception as exc:  # noqa: BLE001 - always persist a runner report.
            payload = {
                "schema_version": "agentic-embodied-arena/native-agent-loop/v1",
                "stage": "native_agent_configuration",
                "status": "runtime_failure",
                "outcome": "runtime_failure",
                "ok": False,
                "case_id": args.case_id,
                "model_provider": args.model_provider,
                "blocker": {
                    "type": type(exc).__name__,
                    "category": "configuration",
                    "message": str(exc),
                },
            }
            _atomic_json(args.output, payload)
            print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            return 2
    seed = args.seed
    if seed is None:
        raw_seed = str(os.environ.get("EMBODIED_ARENA_EVALUATION_SEED", "")).strip()
        if raw_seed and raw_seed != "default":
            try:
                seed = int(raw_seed)
            except ValueError:
                parser.error("EMBODIED_ARENA_EVALUATION_SEED must be an integer")
    loop = NativeAgentLoop(
        model_config=model_config,
        budgets=NativeLoopBudgets(
            max_agent_attempts=args.max_agent_attempts,
            max_agent_iterations=args.max_agent_iterations,
            llm_num_retries=args.llm_num_retries,
            max_total_tokens=args.max_total_tokens,
            max_primitive_calls=args.max_primitive_calls,
            max_verifier_calls=args.max_verifier_calls,
            max_cost_usd=args.max_cost_usd,
        ),
        trace_root=args.agent_trace_root,
        in_process=args.in_process,
        code_timeout_seconds=args.code_timeout_seconds,
        native_timeout_seconds=args.native_timeout_seconds,
        trial_timeout_seconds=args.trial_timeout_seconds,
        replay_code=replay_code,
        require_content_seal=not args.allow_unsealed_runtime,
        interface_mode=args.interface_mode,
        public_rgb_feedback=args.public_rgb_feedback,
        direct_rgb_feedback=args.direct_rgb_feedback,
    )
    report = loop.run_case(args.case_id, seed=seed)
    _atomic_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
