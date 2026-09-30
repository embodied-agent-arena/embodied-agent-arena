import pytest

from embodied_harness.maniskill_agent_runtime import ManiSkillAgentRuntimeBackend, ManiSkillRuntimeConfig


class RewardEnvironment:
    def __init__(self, success_defined=False):
        self.steps = 0
        self.success_defined = success_defined

    def reset(self, seed=None):
        return {}, {}

    def step(self, action):
        self.steps += 1
        info = {"success": True} if self.success_defined else {"fail": False}
        return {}, 1.75, False, self.steps == 2, info


def backend(success_defined=False):
    env = RewardEnvironment(success_defined)
    result = ManiSkillAgentRuntimeBackend(
        ManiSkillRuntimeConfig(env_id="test", task_instruction="Test native reward"),
        env_factory=lambda *_: env,
    )
    result.reset("test")
    return result, env


def test_native_reward_is_accumulated_without_inventing_success():
    runtime, env = backend()
    runtime._step_action([0])
    runtime._step_action([0])
    result = runtime.verify()
    assert result.ok is False
    assert "success" not in result.metrics
    assert result.metrics["native_episode_return"] == 3.5
    assert result.metrics["native_steps"] == 2
    assert result.metadata["native_episode_done"] is True
    assert result.metadata["native_success_defined"] is False
    with pytest.raises(RuntimeError, match="episode has ended"):
        runtime._step_action([0])
    assert env.steps == 2


def test_original_binary_success_remains_authoritative():
    runtime, _ = backend(success_defined=True)
    runtime._step_action([0])
    result = runtime.verify()
    assert result.ok is True
    assert result.metrics["success"] == 1.0
    assert result.metrics["native_episode_return"] == 1.75
    assert result.metadata["native_success_defined"] is True
