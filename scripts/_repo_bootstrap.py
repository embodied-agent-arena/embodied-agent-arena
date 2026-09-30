"""Make the repository's ``src`` package importable by direct script entry points."""

from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_repo_src() -> Path:
    """Prepend this checkout's ``src`` directory and return its resolved path."""
    src_dir = Path(__file__).resolve().parents[1] / "src"
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src_dir
