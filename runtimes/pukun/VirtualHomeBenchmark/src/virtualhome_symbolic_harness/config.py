from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessConfig:
    root_dir: Path
    outputs_dir: Path
    virtualhome_repo: Path
    seed: int = 13
    max_steps: int = 8
    official_task_limit: int | None = 30


def get_config() -> HarnessConfig:
    root = Path(__file__).resolve().parents[2]
    return HarnessConfig(
        root_dir=root,
        outputs_dir=root / "outputs",
        virtualhome_repo=Path(os.environ.get("EMBODIED_ARENA_EXTERNAL_ROOT", root.parents[2] / "external")) / "upstreams/virtualhome",
        seed=int(os.environ.get("VIRTUALHOME_SEED", "13")),
        max_steps=int(os.environ.get("VIRTUALHOME_MAX_STEPS", "8")),
        official_task_limit=_official_task_limit_from_env(),
    )


def _official_task_limit_from_env() -> int | None:
    raw = os.environ.get("VIRTUALHOME_OFFICIAL_LIMIT", "30").strip().lower()
    if raw in {"", "none", "all", "full", "0", "-1"}:
        return None
    return int(raw)
