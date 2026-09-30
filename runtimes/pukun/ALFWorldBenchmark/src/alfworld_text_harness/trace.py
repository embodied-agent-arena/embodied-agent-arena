from __future__ import annotations

import csv
import json
import os
import re
import secrets
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord


class TraceWriter:
    def __init__(self, config: HarnessConfig, run_id: str | None = None):
        self.config = config
        self.run_id = run_id or _new_run_id()
        self.trace_dir = config.outputs_dir / "traces" / self.run_id
        self.report_dir = config.outputs_dir / "reports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._call_index: dict[str, int] = {}

    def record_event(self, task: TaskRecord, event: dict[str, Any]) -> None:
        task_key = _safe_name(task.task_id)
        call_index = self._call_index.get(task.task_id, 0)
        self._call_index[task.task_id] = call_index + 1
        row = {
            "benchmark": "ALFWorld",
            "track": "text",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "source_split": task.source_split,
            "gamefile": task.gamefile,
            "call_index": call_index,
            **event,
        }
        with (self.trace_dir / f"{task_key}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any], task_rows: list[dict[str, Any]]) -> None:
        summary_path = self.report_dir / f"{self.run_id}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)

        csv_path = self.report_dir / f"{self.run_id}_tasks.csv"
        if task_rows:
            fieldnames = sorted({key for row in task_rows for key in row.keys()})
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(_jsonable(row) for row in task_rows)


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _new_run_id() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{timestamp}_p{os.getpid()}_{secrets.token_hex(2)}"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
