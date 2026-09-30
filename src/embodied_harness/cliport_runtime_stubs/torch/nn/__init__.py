"""Import-only torch.nn stub for CLIPort headless env smoke."""


class Module:
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("torch.nn is stubbed for CLIPort env smoke; install torch for model execution")


def __getattr__(name: str):  # pragma: no cover - only used in live env fallback.
    raise RuntimeError(f"torch.nn.{name} is stubbed for CLIPort env smoke; install torch for model execution")
