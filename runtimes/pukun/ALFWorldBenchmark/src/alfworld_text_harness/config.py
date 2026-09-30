from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}

TASK_TYPE_ORDER = list(TASK_TYPES.values())


@dataclass(frozen=True)
class HarnessConfig:
    root_dir: Path
    alfworld_root: Path
    data_dir: Path
    base_config_path: Path
    outputs_dir: Path
    manifest_path: Path
    max_steps: int = 50
    seed: int = 42

    @classmethod
    def default(cls) -> "HarnessConfig":
        root_dir = Path(__file__).resolve().parents[2]
        data_dir = Path(os.environ.get("ALFWORLD_DATA", root_dir / "data")).resolve()
        outputs_dir = root_dir / "outputs"
        return cls(
            root_dir=root_dir,
            alfworld_root=root_dir / "alfworld",
            data_dir=data_dir,
            base_config_path=root_dir / "alfworld" / "configs" / "base_config.yaml",
            outputs_dir=outputs_dir,
            manifest_path=outputs_dir / "manifests" / "robench_alfworld_text_v1.json",
        )

    def load_base_config(self) -> dict[str, Any]:
        with self.base_config_path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["env"]["type"] = "AlfredTWEnv"
        config["general"]["use_cuda"] = False
        config["general"]["training_method"] = "dagger"
        config["env"]["expert_type"] = "handcoded"
        config["dataset"]["num_train_games"] = -1
        config["dataset"]["num_eval_games"] = -1
        config["dataset"]["data_path"] = str(self.data_dir / "json_2.1.1" / "train")
        config["dataset"]["eval_id_data_path"] = str(self.data_dir / "json_2.1.1" / "valid_seen")
        config["dataset"]["eval_ood_data_path"] = str(self.data_dir / "json_2.1.1" / "valid_unseen")
        config["logic"]["domain"] = str(self.data_dir / "logic" / "alfred.pddl")
        config["logic"]["grammar"] = str(self.data_dir / "logic" / "alfred.twl2")
        return config

    def ensure_dirs(self) -> None:
        (self.outputs_dir / "traces").mkdir(parents=True, exist_ok=True)
        (self.outputs_dir / "reports").mkdir(parents=True, exist_ok=True)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)


def get_config() -> HarnessConfig:
    config = HarnessConfig.default()
    config.ensure_dirs()
    return config
