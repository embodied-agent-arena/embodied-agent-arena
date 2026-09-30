"""CPU checks of the final W4 bindings and native episode semantics."""
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS
import json
import sys

import pytest

from embodied_harness import release
from embodied_harness.native_episode_budget import apply_native_episode_budget


def test_finite_horizon_is_scaled_once_without_altering_control(monkeypatch):
    monkeypatch.setenv('ARENA_NATIVE_EPISODE_BUDGET_MULTIPLIER', '2')
    monkeypatch.setenv('ARENA_CASE_STUDY_BENCHMARK', 'robocasa')
    env = NS(horizon=100, ignore_done=True, control_freq=20)
    events = []
    backend = NS(_env=env, record_event=lambda *x: events.append(x))
    task = NS(metadata={})
    apply_native_episode_budget(backend, task)
    apply_native_episode_budget(backend, task)
    assert (env.horizon, env.ignore_done, env.control_freq) == (200, True, 20)
    assert len(events) == 1
    assert task.metadata['native_episode_budget']['no_finite_active_limit']


def test_unlimited_native_horizon_stays_unlimited(monkeypatch):
    monkeypatch.setenv('ARENA_NATIVE_EPISODE_BUDGET_MULTIPLIER', '2')
    monkeypatch.setenv('ARENA_CASE_STUDY_BENCHMARK', 'cliport')
    native = NS(max_steps=float('inf'))
    backend = NS(_task=native, record_event=lambda *args: None)
    apply_native_episode_budget(backend, NS(metadata={}))
    assert native.max_steps == float('inf')


def test_libero_reset_retains_official_initial_state_and_native_settling(monkeypatch):
    from embodied_harness.capx_libero_support import install_reset_fix
    calls = []
    class Env:
        def reset(self, *, seed=None, options=None):
            obs, info = self.handle.reset(seed=seed)
            if self.handle.init_states is not None:
                self.handle.env.set_init_state(self.handle.init_states[seed - 1])
                self.handle.env.reset()
            calls.append('settle-original')
            return obs, info
    native = NS(seed=lambda s: calls.append(('seed', s)),
                reset=lambda: calls.append('reset'),
                set_init_state=lambda s: calls.append(('state', s)) or {'value': s})
    handle = NS(env=native, init_states=[100, 200], reset=lambda seed=None: None)
    old_reset = handle.reset
    monkeypatch.setitem(sys.modules, 'capx.envs.simulators.libero', NS(FrankaLiberoEnv=Env))
    install_reset_fix()
    instance = Env(); instance.handle = handle
    obs, info = instance.reset(seed=2)
    assert calls == [('seed', 2), 'reset', ('state', 200), 'settle-original']
    assert obs == {'value': 200} and info['harness_initial_state_index'] == 1
    assert handle.reset is old_reset and handle.init_states == [100, 200]
    with pytest.raises(ValueError):
        instance.reset(seed=3)


def test_release_data_selects_final_w4_groups_and_budgets():
    try:
        data = release.dataset_root(None)
    except ValueError:
        pytest.skip('Task dataset not installed')
    tasks = release.read_cases(data)
    w4 = [t for t in tasks if t['wave'] == 'W4']
    assert len(tasks) == 1000 and len(w4) == 183
    counts = Counter(t['env']['ARENA_REPORTING_BENCHMARK'] for t in w4)
    assert len(counts) == 13
    assert {k:v for k,v in counts.items() if k.startswith('capx')} == {
        'capx_libero_pro':10, 'capx_robosuite':5, 'capx_behavior1k':5}
    for task in w4:
        assert '--direct-rgb-feedback' in task['runner_args']
        assert task['env']['ARENA_NATIVE_EPISODE_BUDGET_MULTIPLIER'] == '2'
        assert int(release.option(task['runner_args'], '--max-verifier-calls')) > task['budgets']['max_agent_iterations']
    for path in (Path(__file__).resolve().parents[1]/'configs/capx/env_configs/libero').glob('*.yaml'):
        cfg = json.loads(path.read_text())['env']['cfg']
        assert not cfg['privileged'] and not cfg['low_level']['privileged']
        assert cfg['apis'] == ['FrankaLiberoApi']
        assert '{libero_environment_goal}' not in cfg['prompt']
