from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


BENCHMARK_NAME = "ScienceWorld"
TRACK_NAME = "text"


@dataclass(frozen=True)
class HarnessConfig:
    root_dir: Path
    scienceworld_root: Path
    outputs_dir: Path
    manifest_path: Path
    max_steps: int = 100
    seed: int = 42
    simplification: str = "easy"

    @classmethod
    def default(cls) -> "HarnessConfig":
        root_dir = Path(__file__).resolve().parents[2]
        outputs_dir = root_dir / "outputs"
        return cls(
            root_dir=root_dir,
            scienceworld_root=Path(os.environ.get("EMBODIED_ARENA_EXTERNAL_ROOT", root_dir.parents[2] / "external")) / "upstreams/scienceworld",
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifests" / "robench_scienceworld_text_v1.json",
            max_steps=int(os.environ.get("SCIENCEWORLD_MAX_STEPS", "100")),
            seed=int(os.environ.get("SCIENCEWORLD_SEED", "42")),
            simplification=os.environ.get("SCIENCEWORLD_SIMPLIFICATION", "easy"),
        )

    def ensure_dirs(self) -> None:
        (self.outputs_dir / "manifests").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "traces").mkdir(parents=True, exist_ok=True)


def get_config() -> HarnessConfig:
    config = HarnessConfig.default()
    config.ensure_dirs()
    return config
