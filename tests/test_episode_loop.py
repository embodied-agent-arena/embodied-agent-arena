import pytest

from embodied_harness.episode_loop import EpisodeLoop
from embodied_harness.native_agent_loop import (
    ModelCompletion, ModelRequestError, NativeModelConfig, NativeLoopBudgets,
    UsageLedger, completion_with_budget,
)


def test_repair_then_native_success_stops_without_another_turn():
    loop = EpisodeLoop(4, 30)
    assert next(loop) == 1
    loop.finish_turn(error="ValueError: repair this cell")
    assert next(loop) == 2
    loop.finish_turn(execution_ok=True, success=True)
    assert list(loop) == []
    assert loop.report()["stop_reason"] == "success"


@pytest.mark.parametrize("flags, reason, budgets", [
    ({}, "budget_exhausted", ["max_agent_iterations"]),
    ({"terminal": True}, "terminal", []),
    ({"success": True}, "success", []),
    ({"success": True, "timed_out": True}, "timeout", []),
])
def test_last_turn_native_signals_take_precedence(flags, reason, budgets):
    loop = EpisodeLoop(1, 30)
    next(loop)
    loop.finish_turn(**flags)
    assert list(loop) == []
    assert loop.stop_reason == reason
    assert loop.budget_exhausted == budgets


def test_retry_and_post_response_use_same_deadline_and_account_usage():
    now = [0.0]
    loop = EpisodeLoop(4, 10, clock=lambda: now[0])
    config = NativeModelConfig(request_timeout_seconds=8)
    observed = []
    class Client:
        def complete(self, messages, **kwargs):
            observed.append(config.request_timeout_seconds)
            if len(observed) == 1:
                now[0] = 7
                raise ModelRequestError("transient", retryable=True)
            now[0] = 11
            return ModelCompletion("must_not_execute = True", 5, 5, 10, None, True, "test")
    usage = UsageLedger()
    with pytest.raises(TimeoutError):
        completion_with_budget(client=Client(), config=config,
                               budgets=NativeLoopBudgets(llm_num_retries=1),
                               usage=usage, messages=[], loop=loop)
    assert observed == [8, 3]
    assert config.request_timeout_seconds == 8
    assert usage.total_tokens == 10
    with pytest.raises(TimeoutError):
        loop.phase_timeout(8)
