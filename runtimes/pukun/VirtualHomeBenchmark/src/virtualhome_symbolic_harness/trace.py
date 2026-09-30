from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord


class TraceWriter:
    def __init__(self, config: HarnessConfig):
        self.config = config
        self.run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{os.getpid()}"
        self.trace_dir = config.outputs_dir / "traces" / self.run_id
        self.report_dir = config.outputs_dir / "reports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._call_indices: dict[str, int] = {}

    def record_event(self, task: TaskRecord, event: dict[str, Any]) -> None:
        call_index = self._call_indices.get(task.task_id, 0)
        self._call_indices[task.task_id] = call_index + 1
        row = {
            "benchmark": "VirtualHome",
            "track": "symbolic_evolving_graph",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "task_type": task.task_type,
            "call_index": call_index,
            **event,
        }
        path = self.trace_dir / f"{task.task_id}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_summary(self, summary: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        self._write_report(summary, tasks, summary_suffix="summary", tasks_suffix="tasks")

    def write_partial_summary(self, summary: dict[str, Any], tasks: list[dict[str, Any]]) -> None:
        partial_summary = {**summary, "partial": True}
        self._write_report(partial_summary, tasks, summary_suffix="partial_summary", tasks_suffix="partial_tasks")

    def _write_report(
        self,
        summary: dict[str, Any],
        tasks: list[dict[str, Any]],
        *,
        summary_suffix: str,
        tasks_suffix: str,
    ) -> None:
        summary_path = self.report_dir / f"{self.run_id}_{summary_suffix}.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({**summary, "tasks": tasks}, f, indent=2, ensure_ascii=False)
        if tasks:
            csv_path = self.report_dir / f"{self.run_id}_{tasks_suffix}.csv"
            fieldnames = sorted({key for row in tasks for key in row.keys()})
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(tasks)
