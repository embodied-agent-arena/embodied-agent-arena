"""Import-only torch.nn.functional stub for CLIPort headless env smoke."""


def __getattr__(name: str):  # pragma: no cover - only used in live env fallback.
    raise RuntimeError(f"torch.nn.functional.{name} is stubbed for CLIPort env smoke; install torch for model execution")
