from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from embodied_harness import native_agent_loop, native_case_registry
from embodied_harness.native_agent_loop import (
    CodexExecModelClient,
    CursorExecModelClient,
    ModelCompletion,
    ModelRequestError,
    NativeAgentLoop,
    NativeLoopBudgets,
    NativeModelConfig,
    UsageLedger,
    load_native_model_config,
    _universal_code_contract_error,
)
from embodied_harness.native_case_registry import _sanitized_parent_environment
from embodied_harness.w4_benchmark_backend import W4BenchmarkBackend

SOLVE_CODE = """
ctx = primitives.call(interface="task.context")
obs = primitives.call(interface="scene.observe")
entities = primitives.call(interface="entity.enumerate")
cube = primitives.call(interface="entity.locate", entity="cube")
goal = primitives.call(interface="entity.locate", entity="goal_pad")
ev = primitives.call(
    interface="evidence.record",
    key="plan",
    value={"object": "cube", "target": "goal_pad"},
    source_handles=[obs["evidence_handle"], cube["evidence_handle"], goal["evidence_handle"]],
)
prepared = primitives.call(
    interface="action.prepare",
    action={"type": "pick_place", "object": "cube", "target": "goal_pad"},
    evidence_handles=[ev["evidence_handle"], cube["evidence_handle"], goal["evidence_handle"]],
)
executed = primitives.call(interface="action.execute", action_handle=prepared["action_handle"])
result = primitives.call(interface="progress.check")
"""


class FakeClient:
    def complete(self, messages, *, max_tokens):  # type: ignore[no-untyped-def]
        assert max_tokens > 0
        assert "Universal interface contract" in messages[-1]["content"]
        return ModelCompletion(
            content=SOLVE_CODE,
            prompt_tokens=100,
            completion_tokens=80,
            total_tokens=180,
            cost_usd=0.01,
            provider_usage_available=True,
            usage_mode="provider",
        )


class BudgetOvershootClient:
    def __init__(self) -> None:
        self.response_count = 0

    def complete(self, messages, *, max_tokens):  # type: ignore[no-untyped-def]
        assert max_tokens > 0
        self.response_count += 1
        if self.response_count == 1:
            return ModelCompletion(
                content='ctx = primitives.call(interface="task.context")',
                prompt_tokens=50,
                completion_tokens=50,
                total_tokens=100,
                cost_usd=None,
                provider_usage_available=True,
                usage_mode="provider",
            )
        return ModelCompletion(
            content=SOLVE_CODE,
            prompt_tokens=39_000,
            completion_tokens=1_000,
            total_tokens=40_000,
            cost_usd=None,
            provider_usage_available=True,
            usage_mode="provider",
        )


class IncompleteClient:
    def complete(self, messages, *, max_tokens):  # type: ignore[no-untyped-def]
        assert max_tokens > 0
        return ModelCompletion(
            content=(
                "ctx = primitives.call(interface='task.context')\n"
                "obs = primitives.call(interface='scene.observe')\n"
                "result = primitives.call(interface='progress.check')"
            ),
            prompt_tokens=100,
            completion_tokens=40,
            total_tokens=140,
            cost_usd=None,
            provider_usage_available=True,
            usage_mode="provider",
        )


def test_native_control_plane_does_not_import_heavy_legacy_modules() -> None:
    project_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
import sys
import embodied_harness.native_agent_loop
from embodied_harness.native_case_registry import load_native_case
case = load_native_case('cliport_place_red_in_green')
print(json.dumps({
    'case_id': case.case_id,
    'numpy_loaded': 'numpy' in sys.modules,
    'legacy_api_runner_loaded': 'embodied_harness.api_agent_runner' in sys.modules,
    'legacy_case_monolith_loaded': 'embodied_harness.live_api_agent_smoke' in sys.modules,
}))
""",
        ],
        cwd=project_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)

    assert payload == {
        "case_id": "cliport_place_red_in_green",
        "numpy_loaded": False,
        "legacy_api_runner_loaded": False,
        "legacy_case_monolith_loaded": False,
    }


def test_native_loop_solves_with_hidden_harness_verifier(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    launch_order: list[str] = []
    case = SimpleNamespace(
        case_id="unit-native",
        benchmark_id="maniskill",
        task_id="w4_maniskill_pick_cube",
        seed=0,
        reset_config={},
        code_timeout_seconds=2.0,
    )
    monkeypatch.setattr(native_agent_loop, "load_native_case", lambda _: case)

    def validate_receipt(_benchmark_id: str) -> dict[str, str]:
        launch_order.append("receipt")
        return {"receipt_sha256": "sealed"}

    def make_backend(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        launch_order.append("backend")
        return W4BenchmarkBackend(), None

    monkeypatch.setattr(
        native_agent_loop,
        "validate_native_runtime_launch_receipt",
        validate_receipt,
    )
    monkeypatch.setattr(
        native_agent_loop,
        "make_native_backend",
        make_backend,
    )
    config = NativeModelConfig(
        api_key="must-not-appear",
        base_url="https://example.test/v1",
        model="unit-model",
    )
    report = NativeAgentLoop(
        model_config=config,
        budgets=NativeLoopBudgets(
            max_agent_attempts=1,
            max_agent_iterations=2,
            max_total_tokens=30000,
        ),
        trace_root=tmp_path / "trace",
        client=FakeClient(),  # type: ignore[arg-type]
    ).run_case("unit-native")

    assert report["ok"] is True
    assert report["outcome"] == "success"
    assert report["official_verifier_agent_callable"] is False
    assert report["llm_usage"]["total_tokens"] == 180
    attempt = report["agent_attempts"][0]
    assert attempt["verifier"]["ok"] is True
    assert attempt["turns"][0]["public_contract"]["satisfied"] is True
    assert "must-not-appear" not in str(report)
    assert launch_order == ["receipt", "backend"]


def test_native_loop_preserves_partial_attempt_when_model_overshoots_budget(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    case = SimpleNamespace(
        case_id="unit-native-budget",
        benchmark_id="maniskill",
        task_id="w4_maniskill_pick_cube",
        seed=0,
        reset_config={},
        code_timeout_seconds=2.0,
    )
    monkeypatch.setattr(native_agent_loop, "load_native_case", lambda _: case)
    monkeypatch.setattr(
        native_agent_loop,
        "validate_native_runtime_launch_receipt",
        lambda _benchmark_id: {"receipt_sha256": "sealed"},
    )
    monkeypatch.setattr(
        native_agent_loop,
        "make_native_backend",
        lambda *_args, **_kwargs: (W4BenchmarkBackend(), None),
    )

    report = NativeAgentLoop(
        model_config=NativeModelConfig(
            api_key="test", base_url="https://example.test/v1", model="unit-model"
        ),
        budgets=NativeLoopBudgets(
            max_agent_attempts=1,
            max_agent_iterations=3,
            max_total_tokens=30_000,
        ),
        trace_root=tmp_path / "trace",
        client=BudgetOvershootClient(),  # type: ignore[arg-type]
    ).run_case("unit-native-budget")

    assert report["ok"] is False
    assert report["outcome"] == "budget_exhausted"
    assert report["budget_exhausted"] == ["max_total_tokens"]
    assert report["exception"]["type"] == "LoopBudgetExceeded"
    assert report["llm_usage"]["response_count"] == 2
    assert report["llm_usage"]["total_tokens"] == 40_100
    attempt = report["agent_attempts"][0]
    assert len(attempt["turns"]) == 2
    assert attempt["turns"][0]["stage"] == "agent_execution"
    assert attempt["turns"][0]["execution_ok"] is True
    assert attempt["turns"][1]["stage"] == "model_completion"
    assert attempt["turns"][1]["exception"]["type"] == "LoopBudgetExceeded"
    assert attempt["exception"] == report["exception"]
    for path in attempt["artifacts"].values():
        assert Path(path).is_file()


def test_native_loop_rejects_agent_access_to_verifier() -> None:
    assert "harness-only" in str(
        _universal_code_contract_error("result = backend.verify(scope='task')")
    )
    assert "unknown universal interface" in str(
        _universal_code_contract_error(
            "result = primitives.call(interface='native.secret')"
        )
    )
    assert "forbidden" in str(
        _universal_code_contract_error(
            "result = primitives._backend.verify(scope='task')"
        )
    )
    assert "forbidden" in str(
        _universal_code_contract_error(
            "result = getattr(primitives, '_' + 'backend').verify(scope='task')"
        )
    )
    assert "unavailable for this benchmark" in str(
        _universal_code_contract_error(
            "result = primitives.call(interface='policy.invoke', policy='pick_place')",
            unavailable_interfaces=["policy.invoke"],
        )
    )


def test_native_loop_reports_iteration_limit_as_budget_exhaustion(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    case = SimpleNamespace(
        case_id="unit-native-iterations",
        benchmark_id="maniskill",
        task_id="w4_maniskill_pick_cube",
        seed=0,
        reset_config={},
        code_timeout_seconds=2.0,
    )
    monkeypatch.setattr(native_agent_loop, "load_native_case", lambda _: case)
    monkeypatch.setattr(
        native_agent_loop,
        "validate_native_runtime_launch_receipt",
        lambda _benchmark_id: {"receipt_sha256": "sealed"},
    )
    monkeypatch.setattr(
        native_agent_loop,
        "make_native_backend",
        lambda *_args, **_kwargs: (W4BenchmarkBackend(), None),
    )

    report = NativeAgentLoop(
        model_config=NativeModelConfig(
            api_key="test", base_url="https://example.test/v1", model="unit-model"
        ),
        budgets=NativeLoopBudgets(
            max_agent_attempts=1,
            max_agent_iterations=2,
            max_total_tokens=30_000,
        ),
        trace_root=tmp_path / "trace",
        client=IncompleteClient(),  # type: ignore[arg-type]
    ).run_case("unit-native-iterations")

    assert report["outcome"] == "budget_exhausted"
    assert report["budget_exhausted"] == ["max_agent_iterations"]
    assert report["agent_attempts"][0]["budget_exhausted"] == [
        "max_agent_iterations"
    ]
    assert len(report["agent_attempts"][0]["turns"]) == 2


def test_token_budget_allows_exact_limit_and_rejects_overshoot(tmp_path) -> None:  # type: ignore[no-untyped-def]
    exact = NativeAgentLoop(
        model_config=NativeModelConfig(
            api_key="test", base_url="https://example.test/v1"
        ),
        budgets=NativeLoopBudgets(max_total_tokens=180),
        trace_root=tmp_path / "exact",
        client=FakeClient(),  # type: ignore[arg-type]
    )
    completion = exact._completion(
        [{"role": "user", "content": "Universal interface contract: short"}]
    )
    assert completion.total_tokens == 180
    assert exact.usage.exhausted(exact.budgets) == ["max_total_tokens"]

    ledger = UsageLedger(total_tokens=181)
    assert ledger.violated(NativeLoopBudgets(max_total_tokens=180)) == [
        "max_total_tokens"
    ]


def test_native_runtime_environment_does_not_receive_model_secrets() -> None:
    environment = _sanitized_parent_environment(
        {
            "PATH": "/usr/bin",
            "CUDA_VISIBLE_DEVICES": "0",
            "LLM_API_KEY": "secret",
            "OPENAI_API_KEY": "secret-2",
            "HF_TOKEN": "secret-3",
        }
    )

    assert environment == {"PATH": "/usr/bin", "CUDA_VISIBLE_DEVICES": "0"}


def test_robotwin2_native_environment_uses_local_jit_cache_and_runtime_icds(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    artifact_root = project_root / "artifacts"
    runtime_root = external_root / "environments/robotwin2"
    sapien_root = (
        runtime_root / "lib/python3.10/site-packages/sapien/vulkan_library"
    )
    sapien_root.mkdir(parents=True)
    (sapien_root / "nvidia_icd.json").write_text("{}\n", encoding="utf-8")
    (sapien_root / "10_nvidia.json").write_text("{}\n", encoding="utf-8")
    cuda_root = tmp_path / "cuda"
    (cuda_root / "bin").mkdir(parents=True)
    (cuda_root / "include").mkdir()
    (cuda_root / "lib64").mkdir()
    (cuda_root / "bin/nvcc").write_text("", encoding="utf-8")
    (cuda_root / "include/cuda_runtime_api.h").write_text("", encoding="utf-8")
    paths = SimpleNamespace(
        project_root=project_root,
        external_root=external_root,
        artifact_root=artifact_root,
        external_assets=lambda benchmark_id: external_root / "assets" / benchmark_id,
        external_environment=lambda benchmark_id: (
            external_root / "environments" / benchmark_id
        ),
        external_upstream=lambda benchmark_id: external_root / "upstreams" / benchmark_id,
    )
    monkeypatch.setattr(native_case_registry, "get_project_paths", lambda: paths)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "node-local"))
    monkeypatch.setenv("ROBOTWIN2_CUDA_HOME", str(cuda_root))
    for name in (
        "XDG_CACHE_HOME",
        "WARP_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "__EGL_VENDOR_LIBRARY_FILENAMES",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
    ):
        monkeypatch.delenv(name, raising=False)

    environment = native_case_registry._runtime_environment(
        "robotwin2", source_paths=(), asset_paths=()
    )

    expected_cache = (
        tmp_path
        / "node-local/agentic-embodied-arena-runtime-cache/robotwin2"
    )
    assert environment["XDG_CACHE_HOME"] == str(expected_cache / "xdg")
    assert environment["WARP_CACHE_PATH"] == str(expected_cache / "warp")
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        expected_cache / "torch_extensions"
    )
    assert environment["VK_ICD_FILENAMES"] == str(
        sapien_root / "nvidia_icd.json"
    )
    assert environment["__EGL_VENDOR_LIBRARY_FILENAMES"] == str(
        sapien_root / "10_nvidia.json"
    )
    assert environment["CUDA_HOME"] == str(cuda_root)
    assert environment["CUDACXX"] == str(cuda_root / "bin/nvcc")
    assert environment["PATH"].split(os.pathsep)[0] == str(cuda_root / "bin")
    assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[0] == str(
        cuda_root / "lib64"
    )


def test_robodojo_native_environment_uses_local_caches_and_sealed_vulkan(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    project_root = tmp_path / "project"
    external_root = tmp_path / "external"
    paths = SimpleNamespace(
        project_root=project_root,
        external_root=external_root,
        artifact_root=project_root / "artifacts",
        external_assets=lambda benchmark_id: external_root / "assets" / benchmark_id,
        external_environment=lambda benchmark_id: (
            external_root / "environments" / benchmark_id
        ),
        external_upstream=lambda benchmark_id: external_root / "upstreams" / benchmark_id,
    )
    monkeypatch.setattr(native_case_registry, "get_project_paths", lambda: paths)
    monkeypatch.setattr(
        native_case_registry,
        "_runtime_authorization_environment",
        lambda *_args, **_kwargs: {"OMNI_KIT_ACCEPT_EULA": "YES"},
    )
    monkeypatch.setenv("TMPDIR", str(tmp_path / "node-local"))
    monkeypatch.setenv("ROBODOJO_DISABLE_CUSTOM_VULKAN_ICD", "0")
    for name in (
        "XDG_CACHE_HOME",
        "XDG_RUNTIME_DIR",
        "NUMBA_CACHE_DIR",
        "WARP_CACHE_PATH",
        "TORCH_EXTENSIONS_DIR",
        "VK_ICD_FILENAMES",
        "VK_DRIVER_FILES",
    ):
        monkeypatch.delenv(name, raising=False)

    environment = native_case_registry._runtime_environment(
        "robodojo", source_paths=(), asset_paths=()
    )

    expected_cache = (
        tmp_path / "node-local/agentic-embodied-arena-runtime-cache/robodojo"
    )
    assert environment["XDG_CACHE_HOME"] == str(expected_cache / "xdg")
    assert environment["NUMBA_CACHE_DIR"] == str(expected_cache / "numba")
    assert environment["WARP_CACHE_PATH"] == str(expected_cache / "warp")
    assert environment["TORCH_EXTENSIONS_DIR"] == str(
        expected_cache / "torch_extensions"
    )
    assert environment["XDG_RUNTIME_DIR"] == str(expected_cache / "xdg-runtime")
    assert Path(environment["XDG_RUNTIME_DIR"]).stat().st_mode & 0o777 == 0o700
    assert environment["VK_IMPLICIT_LAYER_PATH"] == str(
        expected_cache / "empty-vulkan-implicit-layer"
    )
    expected_manifest = external_root / "assets/robodojo/runtime/nvidia_icd.json"
    assert environment["VK_ICD_FILENAMES"] == str(expected_manifest)
    assert environment["VK_DRIVER_FILES"] == str(expected_manifest)
    assert environment["QT_QPA_PLATFORM"] == "offscreen"
    assert environment["MAX_JOBS"] == "4"


def test_codex_exec_client_uses_jsonl_usage_and_audits_no_tool_boundary(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "unit"}),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "reasoning", "text": "private"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": SOLVE_CODE},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "turn.completed",
                            "usage": {"input_tokens": 120, "output_tokens": 80},
                        }
                    ),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-codex-tools")
    client = CodexExecModelClient(
        NativeModelConfig(
            provider="codex_exec",
            model="unit-codex",
            codex_executable="codex-unit",
            codex_reasoning_effort="high",
        )
    )

    completion = client.complete(
        [{"role": "user", "content": "Universal interface contract"}],
        max_tokens=500,
    )

    assert completion.content == SOLVE_CODE
    assert completion.total_tokens == 200
    assert completion.usage_mode == "codex_exec_jsonl"
    assert captured["command"][:2] == ["codex-unit", "exec"]
    assert "--ephemeral" in captured["command"]
    assert "--json" in captured["command"]
    assert (
        captured["command"][captured["command"].index("--sandbox") + 1] == "read-only"
    )
    assert "Do not call tools" in captured["input"]
    assert "OPENAI_API_KEY" not in captured["env"]


def test_codex_exec_client_rejects_any_codex_tool_event(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "command_execution", "command": "pwd"},
                        }
                    ),
                    json.dumps(
                        {
                            "type": "item.completed",
                            "item": {"type": "agent_message", "text": SOLVE_CODE},
                        }
                    ),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CodexExecModelClient(
        NativeModelConfig(
            provider="codex_exec",
            model="unit-codex",
            codex_executable="codex-unit",
        )
    )

    try:
        client.complete([{"role": "user", "content": "task"}], max_tokens=100)
    except ModelRequestError as exc:
        assert exc.retryable is False
        assert "command_execution" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Codex tool use must fail closed")


def test_codex_exec_client_reports_jsonl_error_when_stderr_is_empty(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            command,
            1,
            stdout=json.dumps(
                {
                    "type": "turn.failed",
                    "error": {"message": "requested model is not supported"},
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CodexExecModelClient(
        NativeModelConfig(
            provider="codex_exec",
            model="unsupported-model",
            codex_executable="codex-unit",
        )
    )

    try:
        client.complete([{"role": "user", "content": "task"}], max_tokens=100)
    except ModelRequestError as exc:
        assert "requested model is not supported" in str(exc)
        assert exc.retryable is False
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Codex JSONL failure detail must be reported")


def test_load_codex_exec_config_needs_no_http_key_or_base_url(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(native_agent_loop.shutil, "which", lambda _: "/usr/bin/codex")
    for name in (*native_agent_loop._API_KEY_NAMES, *native_agent_loop._BASE_URL_NAMES):
        monkeypatch.delenv(name, raising=False)

    config = load_native_model_config(
        tmp_path / "absent.env",
        provider="codex-exec",
        model="unit-codex",
    )

    assert config.provider == "codex_exec"
    assert config.api_key == ""
    assert config.base_url == ""


def test_cursor_exec_client_uses_stream_json_usage_and_ask_mode(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    captured = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured.update(kwargs)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    json.dumps({"type": "system", "subtype": "init"}),
                    json.dumps({"type": "thinking", "subtype": "delta", "text": "plan"}),
                    json.dumps(
                        {
                            "type": "assistant",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": SOLVE_CODE}],
                            },
                        }
                    ),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": SOLVE_CODE,
                            "usage": {"inputTokens": 120, "outputTokens": 80},
                        }
                    ),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-cursor-tools")
    client = CursorExecModelClient(
        NativeModelConfig(
            provider="cursor_exec",
            model="cursor-grok-4.6-high",
            cursor_executable="agent-unit",
        )
    )

    completion = client.complete(
        [{"role": "user", "content": "Universal interface contract"}],
        max_tokens=500,
    )

    assert completion.content == SOLVE_CODE
    assert completion.total_tokens == 200
    assert completion.usage_mode == "cursor_exec_jsonl"
    assert captured["command"][0] == "agent-unit"
    assert captured["command"][1:3] == ["-p", "--mode"]
    assert captured["command"][captured["command"].index("--mode") + 1] == "ask"
    assert captured["command"][captured["command"].index("--output-format") + 1] == "stream-json"
    assert "Do not call tools" in captured["command"][-1]
    assert "OPENAI_API_KEY" not in captured["env"]


def test_cursor_exec_inlines_images_without_workspace_files(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    source = tmp_path / "frame.png"
    source.write_bytes(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d49484452000000010000000108020000009077"
            "53de0000000a49444154789c6360000002000100ffff03000006000557bf2c"
            "0000000049454e44ae426082"
        )
    )
    captured = {}

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        captured["command"] = command
        captured["cwd"] = Path(kwargs["cwd"])
        captured["workspace_files"] = [p.name for p in Path(kwargs["cwd"]).iterdir()]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": SOLVE_CODE,
                            "usage": {"inputTokens": 10, "outputTokens": 3},
                        }
                    )
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CursorExecModelClient(
        NativeModelConfig(
            provider="cursor_exec",
            model="cursor-grok-4.6-high-fast",
            cursor_executable="agent-unit",
        )
    )
    result = client.complete(
        [{"role": "user", "content": "Describe public frame"}],
        max_tokens=100,
        image_paths=[source],
    )
    assert result.content == SOLVE_CODE
    assert captured["workspace_files"] == []
    assert "data:image/png;base64," in captured["command"][-1]
    assert "frame.png" not in captured["command"][-1]
    assert "already placed in the workspace" not in captured["command"][-1]


def test_cursor_exec_client_rejects_tool_events(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    json.dumps({"type": "tool_call", "name": "Shell"}),
                    json.dumps(
                        {
                            "type": "result",
                            "subtype": "success",
                            "is_error": False,
                            "result": SOLVE_CODE,
                        }
                    ),
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CursorExecModelClient(
        NativeModelConfig(
            provider="cursor_exec",
            model="cursor-grok-4.6-high",
            cursor_executable="agent-unit",
        )
    )

    try:
        client.complete([{"role": "user", "content": "task"}], max_tokens=100)
    except ModelRequestError as exc:
        assert exc.retryable is False
        assert "tool_call" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("Cursor tool use must fail closed")


def test_load_cursor_exec_config_needs_no_http_key_or_base_url(
    tmp_path, monkeypatch
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(native_agent_loop.shutil, "which", lambda _: "/usr/bin/agent")
    for name in (*native_agent_loop._API_KEY_NAMES, *native_agent_loop._BASE_URL_NAMES):
        monkeypatch.delenv(name, raising=False)

    config = load_native_model_config(
        tmp_path / "absent.env",
        provider="cursor-exec",
        model="cursor-grok-4.6-high",
    )

    assert config.provider == "cursor_exec"
    assert config.cursor_executable == "agent"
    assert config.api_key == ""
    assert config.base_url == ""


def test_codex_images_are_copied_into_isolated_completion_dir(tmp_path, monkeypatch):
    source = tmp_path / "frame.png"
    source.write_bytes(b"public-frame")
    def fake_run(command, **kwargs):
        copied = Path(command[command.index("--image") + 1])
        assert copied != source
        assert copied.parent == Path(kwargs["cwd"])
        assert copied.read_bytes() == b"public-frame"
        return subprocess.CompletedProcess(command, 0, stdout="\n".join([
            json.dumps({"type":"item.completed", "item":{"type":"agent_message", "text":"print(1)"}}),
            json.dumps({"type":"turn.completed", "usage":{"input_tokens":10, "output_tokens":3}}),
        ]), stderr="")
    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CodexExecModelClient(NativeModelConfig(model="test", provider="codex_exec"))
    result = client.complete([{"role":"user", "content":"Describe public frame"}], max_tokens=100,
                             image_paths=[source])
    assert result.content == "print(1)"


def test_native_repair_keeps_state_and_original_verifier(tmp_path, monkeypatch):
    case = SimpleNamespace(case_id='unit-repair', benchmark_id='maniskill',
        task_id='w4_maniskill_pick_cube', seed=0, reset_config={}, code_timeout_seconds=2.0)
    backend = W4BenchmarkBackend()
    reset_calls = []
    original_reset = backend.reset
    def reset(*args, **kwargs):
        reset_calls.append(1)
        return original_reset(*args, **kwargs)
    monkeypatch.setattr(backend, 'reset', reset)
    monkeypatch.setattr(native_agent_loop, 'load_native_case', lambda _: case)
    monkeypatch.setattr(native_agent_loop, 'make_native_backend', lambda *a, **kw: (backend, None))
    class RepairClient:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                code = 'counter = 40\nraise Exception("repair me")'
            else:
                assert 'repair me' in messages[-1]['content']
                code = 'counter += 2\nassert counter == 42\n' + SOLVE_CODE
            return ModelCompletion(code, 10, 10, 20, None, True, 'test')
    client = RepairClient()
    report = NativeAgentLoop(model_config=NativeModelConfig(), budgets=NativeLoopBudgets(),
        trace_root=tmp_path, in_process=True, client=client).run_case('unit-repair')
    attempt = report['agent_attempts'][0]
    assert report['ok'], report
    assert len(reset_calls) == 1
    assert client.calls == 2
    assert attempt['verifier']['ok'] is True
    assert attempt['turns'][0]['harness_verifier_attempted'] is False
    assert attempt['episode_loop']['stop_reason'] == 'success'


def test_unbound_pool_task_stops_before_reset_and_model_call(tmp_path, monkeypatch):
    case = SimpleNamespace(case_id='unit-pool', benchmark_id='maniskill',
        task_id='w4_maniskill_pick_cube', seed=0, reset_config={}, code_timeout_seconds=2.0)
    backend = W4BenchmarkBackend()
    def unexpected(*args, **kwargs):
        raise AssertionError('An unbound pool task must not run the representative episode')
    monkeypatch.setattr(backend, 'reset', unexpected)
    monkeypatch.setattr(native_agent_loop, 'load_native_case', lambda _: case)
    monkeypatch.setattr(native_agent_loop, 'make_native_backend', lambda *a, **kw: (backend, None))
    monkeypatch.setenv('EMBODIED_ARENA_POOL_TASK_ID', 'selected-upstream-task')
    client = SimpleNamespace(complete=unexpected)
    report = NativeAgentLoop(model_config=NativeModelConfig(), budgets=NativeLoopBudgets(),
        trace_root=tmp_path, in_process=True, client=client).run_case('unit-pool')
    assert report['ok'] is False
    assert report['exception']['category'] == 'environment'
    assert 'no native episode binding' in report['exception']['message']
    assert report['llm_usage']['response_count'] == 0


def test_native_code_timeout_stops_episode(tmp_path, monkeypatch):
    case = SimpleNamespace(case_id='unit-timeout', benchmark_id='maniskill',
        task_id='w4_maniskill_pick_cube', seed=0, reset_config={}, code_timeout_seconds=0.02)
    monkeypatch.setattr(native_agent_loop, 'load_native_case', lambda _: case)
    monkeypatch.setattr(native_agent_loop, 'make_native_backend', lambda *a, **kw: (W4BenchmarkBackend(), None))
    class SlowClient:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            return ModelCompletion('while True: pass', 10, 10, 20, None, True, 'test')
    client = SlowClient()
    report = NativeAgentLoop(model_config=NativeModelConfig(), budgets=NativeLoopBudgets(),
        trace_root=tmp_path, in_process=True, client=client).run_case('unit-timeout')
    assert report['outcome'] == 'timeout', report
    assert client.calls == 1
    assert report['agent_attempts'][0]['episode_loop']['stop_reason'] == 'timeout'


def test_codex_capacity_error_is_retryable(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, stdout=json.dumps({
            "type": "turn.failed", "error": {"message":
            "Selected model is at capacity. Please try a different model."}}), stderr="")
    monkeypatch.setattr(native_agent_loop.subprocess, "run", fake_run)
    client = CodexExecModelClient(NativeModelConfig(provider="codex_exec"))
    try:
        client.complete([{"role": "user", "content": "task"}], max_tokens=100)
    except ModelRequestError as exc:
        assert exc.retryable is True
        assert "at capacity" in str(exc)
    else:
        raise AssertionError("Capacity failure must be reported")


def test_lost_native_episode_stops_without_spending_more_model_turns(tmp_path, monkeypatch):
    case = SimpleNamespace(case_id='unit-lost', benchmark_id='maniskill',
        task_id='w4_maniskill_pick_cube', seed=0, reset_config={}, code_timeout_seconds=2.0)
    backend = W4BenchmarkBackend()
    monkeypatch.setattr(native_agent_loop, 'load_native_case', lambda _: case)
    monkeypatch.setattr(native_agent_loop, 'make_native_backend', lambda *a, **kw: (backend, None))
    class LostClient:
        calls = 0
        def complete(self, messages, **kwargs):
            self.calls += 1
            backend.episode_lost = True
            return ModelCompletion('print("interrupted")', 10, 10, 20, None, True, 'test')
    client = LostClient()
    report = NativeAgentLoop(model_config=NativeModelConfig(), budgets=NativeLoopBudgets(),
        trace_root=tmp_path, in_process=True, client=client).run_case('unit-lost')
    assert report['status'] == 'runtime_failure', report
    assert client.calls == 1
    attempt = report['agent_attempts'][0]
    assert attempt['native_episode_lost'] is True
    assert attempt['verifier']['attempted'] is False
    assert 'native_episode_lost' in attempt['turns'][0]['execution_error']


def test_strict_native_bridge_never_respawns_a_lost_episode(tmp_path, monkeypatch):
    from embodied_harness import subprocess_backend_bridge as bridge_module
    bridge = bridge_module.SubprocessBackendBridge(case_id='unit', python_executable=sys.executable,
        worker_script=tmp_path/'worker.py', cwd=tmp_path, strict_episode=True)
    bridge._last_reset_args = {'task_id': 'already-started'}
    bridge._process = SimpleNamespace(poll=lambda: 1)
    def unexpected_spawn(*args, **kwargs):
        raise AssertionError('implicit reset must not spawn another worker')
    monkeypatch.setattr(bridge_module.subprocess, 'Popen', unexpected_spawn)
    for _ in range(2):
        try:
            bridge._ensure_process()
            raise AssertionError('lost episode should fail')
        except RuntimeError as exc:
            assert 'native_episode_lost' in str(exc)
    assert bridge.episode_lost
