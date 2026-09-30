from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, outputs_dir: Path, run_id: str | None = None):
        self.outputs_dir = outputs_dir
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.trace_dir = outputs_dir / "traces" / self.run_id
        self.report_dir = outputs_dir / "reports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._call_index: dict[str, int] = {}

    def record_event(self, task: dict[str, Any], event: dict[str, Any]) -> None:
        task_id = str(task.get("task_id") or "unknown_task")
        call_index = self._call_index.get(task_id, 0)
        self._call_index[task_id] = call_index + 1
        row = {
            "benchmark": task.get("benchmark", "ALFRED"),
            "track": task.get("track", "official_visual"),
            "run_id": self.run_id,
            "task_id": task_id,
            "source_split": task.get("source_split"),
            "task_type": task.get("task_type"),
            "scene": task.get("floor_plan"),
            "call_index": call_index,
            **event,
        }
        path = self.trace_dir / f"{_safe_name(task_id)}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")

    def write_report(self, filename: str, report: dict[str, Any]) -> Path:
        path = self.report_dir / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(_jsonable(report), f, indent=2, ensure_ascii=False)
        return path


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "trace"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value
