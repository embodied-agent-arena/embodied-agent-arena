from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord


class TraceWriter:
    def __init__(self, config: HarnessConfig, run_id: str | None = None):
        self.config = config
        self.run_id = run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_p{os.getpid()}"
        self.trace_dir = config.outputs_dir / "traces" / self.run_id
        self.report_dir = config.outputs_dir / "reports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._call_index: dict[str, int] = {}

    def record_event(self, task: TaskRecord, event: dict[str, Any]) -> None:
        call_index = self._call_index.get(task.task_id, 0)
        self._call_index[task.task_id] = call_index + 1
        row = {
            "benchmark": "DiscoveryWorld",
            "track": "text_json",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "scenario_name": task.scenario_name,
            "difficulty": task.difficulty,
            "seed": task.seed,
            "task_type": task.task_type,
            "call_index": call_index,
            **event,
        }
        with (self.trace_dir / f"{_safe_name(task.task_id)}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> None:
        with (self.report_dir / f"{self.run_id}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
        if task_rows:
            fieldnames = sorted({key for row in task_rows for key in row.keys()})
            with (self.report_dir / f"{self.run_id}_tasks.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(_jsonable(row) for row in task_rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value
