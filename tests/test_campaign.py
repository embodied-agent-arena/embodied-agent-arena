import copy
import importlib.util
import json
from pathlib import Path

import pytest

from embodied_harness.campaign import (
    Campaign, ending_http_codes, load_suite, main, model_settings, parser,
)
from embodied_harness.full_evaluation import expand_manifest, _atomic_write_json

ROOT = Path(__file__).resolve().parents[1]


def manifest(names=("a", "b", "c")):
    data = {"harness": {"max_workers": 2, "gpus": [], "gpu_memory_gb": 140,
                         "gpu_memory_reserve_gb": 16, "respect_existing_gpu_usage": True},
            "model_profile": {"model": "test-model", "provider": "openai-compatible"},
            "tasks": [{"benchmark_id": "test", "case_id": n, "resources": {"gpu_memory_gb": 0},
                       "budgets": {"max_agent_iterations": 8, "max_total_tokens": 320000}}
                      for n in names]}
    data["tasks"] = [t.to_dict() for t in expand_manifest(data)]
    return data


def policy(**overrides):
    return {"batch_size": 2, "retry_http": True, "max_http_rounds": 0,
            "backoff_seconds": 0, "max_backoff_seconds": 0, **overrides}


class FakeEvaluation:
    def __init__(self, outcomes, interrupt_once=False):
        self.outcomes = copy.deepcopy(outcomes)
        self.batches = []
        self.executed = []
        self.interrupt_once = interrupt_once

    def __call__(self, wave, *, output_dir, resume, **kwargs):
        assert resume is True
        self.batches.append([t["case_id"] for t in wave["tasks"]])
        for task in wave["tasks"]:
            status_path = output_dir / "statuses" / (task["task_id"] + ".json")
            if status_path.exists():
                continue
            name = task["case_id"]
            value = self.outcomes[name].pop(0)
            error = isinstance(value, int)
            outcome = "agent_failure" if error else value
            report_path = output_dir / "tasks" / task["task_id"] / "runner_report.json"
            report = {"episode_loop": {"turns": [{"error": f"ModelRequestError: Model API HTTP {value}" if error else None}]}}
            _atomic_write_json(report_path, report)
            _atomic_write_json(status_path, {"task_id": task["task_id"], "task": task,
                                            "outcome": outcome, "artifacts": {"runner_report": str(report_path)},
                                            "budget": {"usage": {"total_tokens": 42, "token_usage_available": True}}})
            self.executed.append(name)
            if self.interrupt_once:
                self.interrupt_once = False
                raise KeyboardInterrupt()
        return {}


def make_campaign(tmp_path, fake, data=None, **settings):
    return Campaign(data or manifest(), tmp_path, runner_script="dispatch.py", python="python",
                    common_args=[], policy=policy(**settings), evaluate=fake)






def test_mixed_dispatcher_supports_both_option_forms():
    path = ROOT / "scripts/run_benchmark_case.py"
    spec = importlib.util.spec_from_file_location("mixed_dispatch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.runner_name(["--entry-id", "humanclaw"]) == "run_pukun_native_case.py"
    assert module.runner_name(["--entry-id=humanclaw"]) == "run_pukun_native_case.py"
    assert module.runner_name(["--case-id", "native-case"]) == "run_native_universal_case.py"
    args = ["--model-provider", "openai-compatible", "--request-timeout-seconds", "120"]
    assert module.bounded_arguments(args, {"LLM_REQUEST_TIMEOUT_SECONDS": "60"})[-1] == "60.0"
    assert module.bounded_arguments(args, {"LLM_REQUEST_TIMEOUT_SECONDS": "360"})[-1] == "120.0"
    pukun = args + ["--entry-id", "humanclaw"]
    assert module.bounded_arguments(pukun, {"LLM_REQUEST_TIMEOUT_SECONDS": "60"}) == pukun
    assert module.runner_name(["--sample-id", "0MJFFKTE-flip"]) == "run_w2_native_case.py"
    assert module.runner_name(["--benchmark", "multispa", "--sample-id", "s00460_frame_00009"]) == "run_w1_native_case.py"
    assert module.runner_name(["--benchmark", "influx", "--sample-id", "bike_shot11_000000"]) == "run_w1_native_case.py"
    assert module.runner_name(["--benchmark", "mapfree", "--sample-id", "s00460_frame_00009"]) == "run_w1_native_case.py"
    assert module.runner_name(["--benchmark", "reasonaff", "--sample-id", "cup_003491-cup-handle"]) == "run_w5_native_case.py"
    assert module.runner_name(["--benchmark", "ragnet", "--sample-id", "demo"]) == "run_w5_native_case.py"
    assert module.runner_name(["--benchmark", "umd", "--sample-id", "mug_19_grasp"]) == "run_w5_native_case.py"


def test_cursor_cli_profile_matches_codex_shape():
    args = parser().parse_args([
        "--provider", "cursor-exec", "--model", "cursor-grok-4.6-high-fast",
        "--cursor-executable", "agent", "--dry-run",
    ])
    profile, env, common = model_settings(ROOT, args)
    assert profile == {
        "provider": "cursor-exec",
        "model": "cursor-grok-4.6-high-fast",
        "max_output_tokens": 1800,
        "request_timeout": 360,
        "temperature": 0,
        "cursor_executable": "agent",
    }
    assert common[:6] == ["--model-provider", "cursor-exec", "--model", "cursor-grok-4.6-high-fast", "--env-file", "/dev/null"]
    assert common[6:8] == ["--cursor-executable", "agent"]
    assert "--codex-reasoning-effort" not in common
    assert "LLM_API_KEY" not in env


def test_http_queue_repeats_only_http_cases(tmp_path):
    fake = FakeEvaluation({"a": [529, 529, "success"], "b": ["success"], "c": [400, "task_failure"]})
    campaign = make_campaign(tmp_path, fake)
    campaign.initialize()
    assert campaign.run() == 0
    assert fake.batches == [["a", "b"], ["c"], ["a", "c"], ["a"]]
    assert fake.executed.count("b") == 1
    assert len(campaign.state["attempts"]) == 6
    assert sum(r["total_tokens"] for r in campaign.state["attempts"]) == 252
    assert sum(r["total_tokens"] for r in campaign.state["latest"].values()) == 126
    assert campaign.state["first"][campaign.manifest["tasks"][0]["task_id"]]["http_codes"] == [529]
    assert all(not r["http_codes"] for r in campaign.state["latest"].values())
    assert json.loads((tmp_path / "progress.json").read_text())["phase"] == "finished"


def test_non_http_error_does_not_get_automatic_retries(tmp_path):
    fake = FakeEvaluation({"a": ["runtime_failure"], "b": ["timeout"], "c": ["budget_exhausted"]})
    campaign = make_campaign(tmp_path, fake)
    campaign.initialize(); campaign.run()
    assert len(fake.executed) == 3
    p = json.loads((tmp_path / "progress.json").read_text())
    assert p["non_http_error_cases"] == 2 and p["latest_http_error_cases"] == 0


def test_retry_round_limit_reports_unresolved_http(tmp_path):
    fake = FakeEvaluation({"a": [529, 529]})
    campaign = make_campaign(tmp_path, fake, manifest(["a"]), max_http_rounds=1)
    campaign.initialize()
    assert campaign.run() == 2
    assert campaign.state["phase"] == "http_retry_limit_reached"
    assert len(fake.executed) == 2
    assert campaign.state["next_http"]


def test_interrupt_resume_uses_pinned_batch_and_does_not_repeat_finished_case(tmp_path):
    fake = FakeEvaluation({"a": ["success"], "b": [529, "success"], "c": ["task_failure"]}, interrupt_once=True)
    first = make_campaign(tmp_path, fake)
    first.initialize()
    with pytest.raises(KeyboardInterrupt):
        first.run()
    second = make_campaign(tmp_path, fake)
    second.initialize(resume=True)
    assert second.run() == 0
    assert fake.executed == ["a", "b", "c", "b"]
    assert len(second.state["attempts"]) == 4
    third = make_campaign(tmp_path, fake)
    third.initialize(resume=True)
    assert third.run() == 0
    assert len(fake.executed) == 4


def test_resume_rejects_changed_model_without_modifying_checkpoint(tmp_path):
    campaign = make_campaign(tmp_path, FakeEvaluation({}))
    campaign.initialize()
    before = (tmp_path / "checkpoint.json").read_bytes()
    changed = manifest(); changed["model_profile"]["model"] = "different-model"
    other = make_campaign(tmp_path, FakeEvaluation({}), changed)
    with pytest.raises(ValueError, match="configuration differs"):
        other.initialize(resume=True)
    assert (tmp_path / "checkpoint.json").read_bytes() == before


def test_http_classifier_ignores_old_errors_and_prompt_text():
    status = {"outcome": "agent_failure"}
    report = {"prompt": "Model API HTTP 529", "episode_loop": {"turns": [
        {"error": "Model API HTTP 529"}, {"error": "empty code cell"}]}}
    assert ending_http_codes(status, report) == []
    report["episode_loop"]["turns"][-1]["error"] = "ModelRequestError: Model API HTTP 400"
    assert ending_http_codes(status, report) == [400]
    assert ending_http_codes({"outcome": "success"}, report) == []
    report = {"agent_attempts": [{"exception": {"category": "model_api", "status_code": 503}}]}
    assert ending_http_codes(status, report) == [503]
    # W2 runner reports nest the episode under event_log_paths.
    nested = {"event_log_paths": {"episode_loop": {"turns": [
        {"error": "ModelRequestError: Model API HTTP 529"}]}}}
    assert ending_http_codes(status, nested) == [529]
    nested["event_log_paths"]["episode_loop"]["turns"].append({"error": "empty code cell"})
    assert ending_http_codes(status, nested) == []


def test_api_profile_keeps_secret_out_of_public_plan(tmp_path, monkeypatch):
    for key in list(__import__('os').environ):
        if key.startswith(("LLM_", "OPENAI_", "GPTGE_")):
            monkeypatch.delenv(key)
    secret = tmp_path / "gptge.env"
    secret.write_text('GPTGE_API_KEY="test-secret-123"\n')
    monkeypatch.setenv("LLM_API_KEY", "unrelated-provider-key")
    args = parser().parse_args(["--provider", "openai-compatible", "--model", "some-arbitrary-model",
                                "--env-file", str(secret), "--base-url", "https://api.gpt.ge/v1"])
    profile, env, common = model_settings(ROOT, args)
    assert env["LLM_API_KEY"] == "test-secret-123"
    assert "test-secret-123" not in json.dumps([profile, common])
    assert profile["model"] == "some-arbitrary-model"
    args.base_url = "https://user:password@example.com/v1"
    with pytest.raises(ValueError, match="credential-free"):
        model_settings(ROOT, args)


def test_real_dry_run_cpu_custom_manifest_needs_no_api_key_and_can_initialize(tmp_path, monkeypatch):
    for key in list(__import__('os').environ):
        if key.startswith(("LLM_", "OPENAI_", "GPTGE_")):
            monkeypatch.delenv(key)
    source = tmp_path / "suite.json"; source.write_text(json.dumps(manifest(["a"])))
    output = tmp_path / "output"
    assert main(ROOT, ["--manifest", str(source), "--provider", "openai-compatible", "--model", "local-test",
                       "--base-url", "http://localhost:8000/v1", "--output-dir", str(output), "--dry-run"]) == 0
    plan = next((output / "dry-run").glob('*/effective-manifest.json'))
    data = json.loads(plan.read_text())
    assert "LLM_API_KEY" not in plan.read_text()
    campaign = make_campaign(output, FakeEvaluation({}), data)
    campaign.initialize()
    assert (output / "checkpoint.json").exists()


def test_custom_manifest_cannot_override_campaign_model(tmp_path):
    data = manifest(["a"])
    data["tasks"][0]["runner_args"] = ["--model=hidden-model"]
    p = tmp_path / "suite.json"; p.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="campaign-owned"):
        load_suite(ROOT, parser().parse_args(["--manifest", str(p)]))


def test_automatic_restart_and_append_only_extension(tmp_path):
    fake = FakeEvaluation({'a': ['task_failure'], 'b': ['budget_exhausted'], 'c': ['success']})
    first = make_campaign(tmp_path, fake, manifest(['a', 'b']))
    first.initialize(); first.run()
    restarted = make_campaign(tmp_path, fake, manifest(['a', 'b']))
    restarted.initialize(); restarted.run()
    assert fake.executed == ['a', 'b']
    extended = make_campaign(tmp_path, fake, manifest(['a', 'b', 'c']))
    extended.initialize(); extended.run()
    assert fake.executed == ['a', 'b', 'c']
    assert len(extended.state['latest']) == 3


def test_cross_campaign_reuse_skips_terminal_failures_and_checks_identity(tmp_path):
    from embodied_harness.campaign_history import reusable_results
    old = tmp_path / 'old'
    fake = FakeEvaluation({'a': ['success'], 'b': ['budget_exhausted'], 'c': ['task_failure']})
    campaign = make_campaign(old, fake)
    campaign.initialize(); campaign.run()
    data = manifest(['a', 'b', 'c', 'd'])
    reused, errors = reusable_results(data, [old / 'checkpoint.json'])
    assert len(reused) == 3 and not errors
    next_fake = FakeEvaluation({'d': ['success']})
    next_campaign = make_campaign(tmp_path / 'new', next_fake, data)
    next_campaign.initialize(reused=reused); next_campaign.run()
    assert next_fake.executed == ['d']
    assert len(next_campaign.state['attempts']) == 1
    assert len(next_campaign.state['first']) == 1
    assert len(next_campaign.state['latest']) == 4
    for mutation in ['model', 'budget', 'coordinate', 'runtime', 'endpoint', 'effort']:
        changed = copy.deepcopy(data)
        if mutation == 'model': changed['model_profile']['model'] = 'another-model'
        if mutation == 'budget':
            for t in changed['tasks']: t['budgets']['max_total_tokens'] += 1
        if mutation == 'coordinate':
            for t in changed['tasks']: t['seed'] += 1
        if mutation == 'runtime': changed['runtime'] = {'env_file_sha256': 'changed'}
        if mutation == 'endpoint': changed['model_profile']['base_url'] = 'https://different.example/v1'
        if mutation == 'effort': changed['model_profile']['reasoning_effort'] = 'max'
        assert not reusable_results(changed, [old / 'checkpoint.json'])[0]


def test_imported_http_error_requeues_but_success_never_repeats(tmp_path):
    from embodied_harness.campaign_history import reusable_results
    data = manifest(['a', 'b'])
    first = make_campaign(tmp_path / 'first', FakeEvaluation({'a': [529], 'b': ['success']}), data, retry_http=False)
    first.initialize(); first.run()
    reused, _ = reusable_results(data, [tmp_path / 'first/checkpoint.json'])
    fake = FakeEvaluation({'a': ['success']})
    second = make_campaign(tmp_path / 'second', fake, data)
    second.initialize(reused=reused); second.run()
    assert fake.executed == ['a']
    assert len(second.state['latest']) == 2


def test_history_uses_latest_not_best_and_ignores_missing_report(tmp_path):
    from embodied_harness.campaign_history import reusable_results
    data = manifest(['a'])
    one = make_campaign(tmp_path / 'one', FakeEvaluation({'a': ['success']}), data)
    one.initialize(); one.run()
    two = make_campaign(tmp_path / 'two', FakeEvaluation({'a': ['task_failure']}), data)
    two.initialize(); two.run()
    paths = [tmp_path / 'one/checkpoint.json', tmp_path / 'two/checkpoint.json']
    reused, _ = reusable_results(data, paths)
    assert next(iter(reused.values()))['outcome'] == 'task_failure'
    Path(next(iter(two.state['latest'].values()))['runner_report']).unlink()
    assert not reusable_results(data, [paths[1]])[0]


def test_legacy_index_hash_validation(tmp_path):
    import hashlib
    from embodied_harness.campaign_history import identity, reusable_results
    data = manifest(['a'])
    c = make_campaign(tmp_path / 'old', FakeEvaluation({'a': ['success']}), data)
    c.initialize(); c.run()
    result = next(iter(c.state['latest'].values()))
    status = Path(result['status_file'])
    index = tmp_path / 'history.json'
    index.write_text(json.dumps({'schema': 'campaign-history/v1', 'records': [{
        'identity': identity(data, data['tasks'][0]), 'status_file': str(status),
        'status_sha256': hashlib.sha256(status.read_bytes()).hexdigest()}]}))
    assert len(reusable_results(data, [index])[0]) == 1
    status.write_text(status.read_text() + '\n')
    reused, diagnostics = reusable_results(data, [index])
    assert not reused and diagnostics[0]['reason'] == 'status_changed'






def test_independent_gpu_queue_requires_and_adds_external_reservation(tmp_path):
    lock = str(tmp_path/'separate.lock')
    source = tmp_path / 'suite.json'; source.write_text(json.dumps(manifest(['a'])))
    with pytest.raises(ValueError, match='separate GPU queue requires'):
        load_suite(ROOT, parser().parse_args(['--manifest', str(source), '--gpu-queue-lock', lock]))
    m = load_suite(ROOT, parser().parse_args(['--manifest', str(source), '--gpu-queue-lock', lock, '--external-gpu-reserve-gb', '40']))
    assert m['harness']['gpu_memory_reserve_gb'] == 56
    assert m['harness']['respect_existing_gpu_usage'] is True
