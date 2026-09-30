import pytest

from embodied_harness.universal_interface import UniversalEmbodiedBackend
from embodied_harness.w4_benchmark_backend import W4BenchmarkBackend


def test_override_is_visible_and_enforced_and_reset_clears_usage():
    gateway = UniversalEmbodiedBackend(W4BenchmarkBackend())
    task = gateway.reset("w4_maniskill_pick_cube", budgets={"primitive_calls": 2, "verifier_calls": 1})
    assert task.budgets["primitive_calls"] == 2
    context = gateway.call_primitive("task.context")
    assert context.ok
    assert context.output["budget"]["remaining"]["primitive_calls"] == 1
    assert gateway.call_primitive("progress.check").ok
    assert not gateway.call_primitive("task.context").ok
    gateway.verify()
    rejected = gateway.verify()
    assert not rejected.ok
    assert "budget_exhausted" in str(rejected.to_dict())
    gateway.reset("w4_maniskill_pick_cube", budgets={"primitive_calls": 3, "verifier_calls": 2})
    assert gateway.call_primitive("task.context").output["budget"]["remaining"]["primitive_calls"] == 2


@pytest.mark.parametrize("limits", [{"primitive_calls": 0}, {"verifier_calls": True}, {"primitive_calls": -1}, {"native_steps": 3}])
def test_bad_override_rejected_before_native_reset(limits):
    class NeverReset:
        def reset(self, **kwargs):
            raise AssertionError("invalid limits must be rejected before simulator startup")
    with pytest.raises(ValueError):
        UniversalEmbodiedBackend(NeverReset()).reset("unused", budgets=limits)
