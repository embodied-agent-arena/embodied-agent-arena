"""Codex author for the existing persistent Pukun runtime.

Model transport and token accounting come from the sibling native harness.
The author only produces code; Pukun executes it in its existing episode/REPL.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


def _native_imports():
    default = Path(__file__).resolve().parents[3]
    root = Path(os.environ.get("EMBODIED_ARENA_ROOT", str(default))).resolve()
    if not (root / "src/embodied_harness/native_agent_loop.py").is_file():
        raise RuntimeError("Set EMBODIED_ARENA_ROOT to the existing native harness checkout")
    if str(root / "src") not in sys.path:
        sys.path.insert(0, str(root / "src"))
    from embodied_harness.native_agent_loop import (
        CodexExecModelClient, NativeLoopBudgets, NativeModelConfig, UsageLedger,
        extract_python_code,
    )
    return CodexExecModelClient, NativeLoopBudgets, NativeModelConfig, UsageLedger, extract_python_code


_EMPTY_CELL_REPAIR = (
    "Previous reply was empty or not an executable Python cell. "
    "Return only the next Python code cell. Do not use markdown fences. "
    "Do not return an empty reply."
)


def usable_python_cell(text: str, extract=None) -> str:
    """Keep closed-fence extraction; also accept an unclosed ```python opener.

    Shared extract_python_code is left unchanged so W4 native-loop scoring
    stays on the existing contract. Empty replies stay empty here.
    """
    raw = text if isinstance(text, str) else ""
    extracted = (extract or (lambda value: value))(raw).strip()
    if extracted and not extracted.lstrip().startswith("```"):
        return extracted
    stripped = raw.strip()
    if stripped.startswith("```"):
        first, _, rest = stripped.partition("\n")
        if first[3:].strip().lower() in {"", "python", "py"}:
            body = rest.rstrip()
            if body.endswith("```"):
                body = body[:-3].rstrip()
            return body.strip()
    return extracted


class CodexCellAuthor:
    def __init__(self, args: Any, workspace: Path, *, client: Any = None):
        Client, Budgets, Config, Ledger, self.extract_code = _native_imports()
        executor = getattr(args, "executor", "codex-exec")
        api_mode = executor == "openai-compatible"
        cursor_mode = executor == "cursor-exec"
        if not api_mode and not args.model:
            raise ValueError("--model is required with --executor " + executor)
        self.config = Config(
            model=args.model, provider="codex_exec",
            codex_executable=args.codex_executable,
            codex_reasoning_effort=args.codex_reasoning_effort,
        )
        if cursor_mode:
            from embodied_harness.native_agent_loop import CursorExecModelClient
            self.config = Config(
                model=args.model, provider="cursor_exec",
                cursor_executable=getattr(args, "cursor_executable", "agent"),
            )
            Client = CursorExecModelClient
        elif api_mode:
            from embodied_harness.native_agent_loop import load_native_model_config, OpenAICompatibleModelClient
            # The adapter loads the explicitly selected dotenv into its process
            # environment before spawning this author. Do not discover another file.
            self.config = load_native_model_config(Path("/dev/null"), model=args.model)
            Client = OpenAICompatibleModelClient
        self.client = client if client is not None else Client(self.config)
        self.request_timeout = self.config.request_timeout_seconds if api_mode else None
        from embodied_harness.native_agent_loop import completion_with_budget, LoopBudgetExceeded, _compact_prompt_value
        self.complete = completion_with_budget
        self.compact_feedback = _compact_prompt_value
        self.budget_error = LoopBudgetExceeded
        self.blocked_budgets = []
        self.budgets = Budgets(max_total_tokens=args.max_total_tokens,
                               max_agent_iterations=args.max_code_turns)
        self.usage = Ledger()
        self.workspace = workspace
        self.image_paths = []
        self.compact_observation_history = bool(getattr(args, "compact_observation_history", False))
        self.previous_feedback = None
        self.previous_feedback_summary = None
        self.max_images = int(getattr(args, "max_model_images", 4))
        if not 1 <= self.max_images <= 32:
            raise ValueError("max_model_images must be between 1 and 32")
        self.runtime_limits = {key: getattr(args, key, None) for key in
                               ("max_code_turns", "max_total_tokens", "code_timeout_seconds",
                                "trial_timeout_seconds", "max_env_steps", "max_primitive_calls") }
        self.messages = [
            {"role": "system", "content": (
                "Return only the next Python code cell, not a file-edit command. "
                "Use the documented primitive_api functions. The harness runs the cell once. "
                "The same Python globals, evidence store and benchmark episode persist across turns. "
                "Print concise observations needed to choose your next action. Do not replay earlier cells. "
                "Do not access files, processes, network, environment variables or evaluator internals. "
                "Never call Codex tools. The task's references to writing cell.py mean returning its contents. "
                "References to reading feedback/cells mean using this transcript; do not open those files. "
                "Import primitive_api normally inside the returned cell. Primitive calls execute there, "
                "not during model authoring. Public check_success/check_activity_success summaries are allowed; "
                "private evaluator objects remain forbidden. Start by observing the current state. "
                "For vision, inspect at most four frames in a cell, then yield: the harness attaches the "
                "latest four exposed frames to the NEXT model request. Do not infer image contents from filenames "
                "or submit an answer before receiving the needed images. caption_frame is an optional external "
                "VLM; inspect_frame/get_frame plus attached images requires no separate caption service."
            )},
            {"role": "user", "content": (
                (workspace / "task.md").read_text(encoding="utf-8")
                + "\n\nPublic primitive API:\n"
                + (workspace / "primitive_cards.md").read_text(encoding="utf-8")
            )},
        ]
        self.messages[0]["content"] += "\nRuntime limits: " + json.dumps(self.runtime_limits)
        if self.max_images != 4:
            self.messages[0]["content"] = self.messages[0]["content"].replace(
                "at most four frames", f"at most {self.max_images} frames"
            ).replace("latest four exposed frames", f"latest {self.max_images} exposed frames")

    def next_cell(self, *, timeout_seconds: float, loop=None) -> str:
        self.config.request_timeout_seconds = min(timeout_seconds, self.request_timeout) if self.request_timeout else timeout_seconds
        public_budget = dict(turn=len([m for m in self.messages if m["role"] == "assistant"]) + 1,
                             max_turns=self.budgets.max_agent_iterations,
                             used_tokens=self.usage.total_tokens,
                             remaining_tokens=(self.budgets.max_total_tokens - self.usage.total_tokens
                                               if self.budgets.max_total_tokens is not None else None),
                             phase_timeout_seconds=timeout_seconds)
        # Refresh the existing next-turn prompt without adding duplicate transcript turns.
        self.messages[-1]["content"] += "\nCurrent model budget: " + json.dumps(public_budget)
        kwargs = {"image_paths": self.image_paths} if self.image_paths else {}
        try:
            completion = self.complete(client=self.client, config=self.config,
                                       budgets=self.budgets, usage=self.usage,
                                       messages=self.messages, loop=loop, client_kwargs=kwargs)
        except self.budget_error as exc:
            self.blocked_budgets = exc.names
            raise
        self.messages.append({"role": "assistant", "content": completion.content})
        code = usable_python_cell(completion.content, self.extract_code)
        if not code:
            self.messages.append({"role": "user", "content": _EMPTY_CELL_REPAIR})
            try:
                completion = self.complete(
                    client=self.client, config=self.config,
                    budgets=self.budgets, usage=self.usage,
                    messages=self.messages, loop=loop, client_kwargs=kwargs)
            except self.budget_error as exc:
                self.blocked_budgets = exc.names
                raise
            self.messages.append({"role": "assistant", "content": completion.content})
            code = usable_python_cell(completion.content, self.extract_code)
        if not code:
            raise ValueError("Model returned an empty code cell")
        return code

    def feedback(self, value: dict[str, Any]) -> None:
        # Keep official verifier objects and scores out of the author transcript.
        keys = (
            "turn", "max_turns", "returncode", "timed_out", "stdout_tail", "stderr_tail",
            "current_observation", "current_actions", "visual_evidence", "evidence",
            "env_steps", "remaining_env_steps", "primitive_calls", "remaining_primitive_calls",
        )
        public = {key: self.compact_feedback(value[key]) for key in keys if key in value}
        # Keep native feedback/trace intact; only bound the model-visible copy.
        public["feedback_note"] = (
            "Large collections may be truncated. Use scoped/query/limit public primitives "
            "inside the next cell to inspect relevant objects; full Python state persists.")
        while len(json.dumps(public, ensure_ascii=False, default=str)) > 32000:
            key = max(public, key=lambda k: len(json.dumps(public[k], ensure_ascii=False, default=str)))
            text = json.dumps(public[key], ensure_ascii=False, default=str)
            public[key] = {"truncated": True, "preview": text[:len(text) // 3]}
        frames = self.workspace / "frames"
        if frames.is_dir() and frames.resolve().is_relative_to(self.workspace.resolve()):
            # Only frames exposed by the public primitive facade, never backend data.
            candidates = [p for p in frames.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                          and p.resolve().is_relative_to(frames.resolve()) and p.is_file()
                          and p.stat().st_size <= 10 * 1024 * 1024]
            self.image_paths = sorted(candidates, key=lambda p: (p.stat().st_ctime_ns, p.name))[-self.max_images:]
            public["attached_images"] = [str(p.relative_to(self.workspace)) for p in self.image_paths]
            public["image_order"] = "Images are attached in the same order as attached_images."
        if self.compact_observation_history and self.previous_feedback is not None:
            self.previous_feedback["content"] = (
                "Earlier turn notes (the latest observation supersedes its scene snapshot):\n"
                + json.dumps(self.previous_feedback_summary, ensure_ascii=False, default=str))
        message = {"role": "user", "content": (
            "Continue the same episode and interpreter. Public feedback:\n"
            + json.dumps(public, ensure_ascii=False, default=str)
        )}
        self.messages.append(message)
        if self.compact_observation_history:
            self.previous_feedback = message
            self.previous_feedback_summary = {key: public[key] for key in
                ("turn", "returncode", "timed_out", "stdout_tail", "stderr_tail", "env_steps") if key in public}

    def report(self) -> dict[str, Any]:
        usage = self.usage.to_dict(self.budgets)
        usage["budget_exhausted"] = list(dict.fromkeys([
            *usage["budget_exhausted"], *self.blocked_budgets]))
        return {
            "provider": self.config.provider.replace("_", "-"), "model": self.config.model,
            "reasoning_effort": self.config.codex_reasoning_effort,
            "usage": usage,
            "runtime_limits": self.runtime_limits,
            "conversation_state_owner": "harness", "same_episode": True,
            "image_transport": (
                "chat_completions_image_url" if self.config.provider == "openai_compatible"
                else "cursor_exec_image" if self.config.provider == "cursor_exec"
                else "codex_exec_image"
            ), "max_images_per_response": self.max_images,
        }
