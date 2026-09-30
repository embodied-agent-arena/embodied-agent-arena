"""Import-only kornia stub for CLIPort headless env smoke."""


def __getattr__(name: str):  # pragma: no cover - only used in live env fallback.
    raise RuntimeError(f"kornia.{name} is stubbed for CLIPort env smoke; install kornia for model execution")
