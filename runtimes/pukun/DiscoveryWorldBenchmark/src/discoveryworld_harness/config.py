from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HarnessConfig:
    root_dir: Path
    official_root: Path
    outputs_dir: Path
    manifest_dir: Path
    seed: int = 0
    max_steps: int = 50
    task_timeout_seconds: int = 600
    suppress_official_output: bool = True

    @classmethod
    def default(cls) -> "HarnessConfig":
        root = Path(__file__).resolve().parents[2]
        outputs = root / "outputs"
        return cls(
            root_dir=root,
            official_root=Path(os.environ.get("EMBODIED_ARENA_EXTERNAL_ROOT", root.parents[2] / "external")) / "upstreams/discoveryworld",
            outputs_dir=outputs,
            manifest_dir=outputs / "manifests",
            seed=int(os.environ.get("DISCOVERYWORLD_SEED", "0")),
            max_steps=int(os.environ.get("DISCOVERYWORLD_MAX_STEPS", "50")),
            task_timeout_seconds=int(os.environ.get("DISCOVERYWORLD_TASK_TIMEOUT_SECONDS", "600")),
            suppress_official_output=os.environ.get("DISCOVERYWORLD_SHOW_OFFICIAL_LOGS", "").lower()
            not in {"1", "true", "yes"},
        )

    def ensure_dirs(self) -> None:
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> HarnessConfig:
    config = HarnessConfig.default()
    config.ensure_dirs()
    return config
