"""Import-only torch stub for CLIPort headless env smoke.

The live benchmark case does not execute learned models. Any real tensor or
checkpoint use must install upstream torch instead of using this stub.
"""


def _missing(*args, **kwargs):  # type: ignore[no-untyped-def]
    raise RuntimeError("torch is stubbed for CLIPort env smoke; install torch for model execution")


Tensor = object
float32 = "float32"
device = lambda *args, **kwargs: "stub-device"  # noqa: E731
no_grad = _missing
load = _missing
from_numpy = _missing
tensor = _missing
as_tensor = _missing
zeros = _missing
ones = _missing
cat = _missing
stack = _missing
