from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import HarnessConfig
from .manifest import TaskRecord


class TraceLimitExceeded(RuntimeError):
    pass


class TraceWriter:
    def __init__(self, config: HarnessConfig, run_id: str | None = None):
        self.config = config
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.trace_dir = config.outputs_dir / "traces" / self.run_id
        self.report_dir = config.outputs_dir / "reports"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.max_trace_bytes = _env_int("SCIENCEWORLD_TRACE_MAX_BYTES", 50 * 1024 * 1024)
        self.max_trace_events = _env_int("SCIENCEWORLD_TRACE_MAX_EVENTS", 5000)
        self._event_counts: dict[Path, int] = {}
        self._closed_paths: set[Path] = set()

    def record_event(self, task: TaskRecord, event: dict[str, Any]) -> None:
        path = self.trace_dir / f"{_safe_name(task.task_id)}.jsonl"
        if path in self._closed_paths:
            return
        row = {
            "benchmark": "ScienceWorld",
            "track": "text",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "variation_idx": task.variation_idx,
            "simplification": task.simplification,
            **event,
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        current_size = path.stat().st_size if path.exists() else 0
        current_events = self._event_counts.get(path, 0)
        reason = None
        if self.max_trace_events > 0 and current_events >= self.max_trace_events:
            reason = f"event_count>{self.max_trace_events}"
        elif self.max_trace_bytes > 0 and current_size + len(line.encode("utf-8")) > self.max_trace_bytes:
            reason = f"bytes>{self.max_trace_bytes}"
        if reason is not None:
            self._write_trace_truncated_event(path, task, reason, current_size, current_events)
            self._closed_paths.add(path)
            raise TraceLimitExceeded(
                f"Trace limit exceeded for {task.task_id}: {reason}. "
                "This usually means generated code is looping without progress."
            )
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
        self._event_counts[path] = current_events + 1

    def _write_trace_truncated_event(
        self,
        path: Path,
        task: TaskRecord,
        reason: str,
        current_size: int,
        current_events: int,
    ) -> None:
        if self.max_trace_bytes > 0 and current_size >= self.max_trace_bytes:
            return
        row = {
            "benchmark": "ScienceWorld",
            "track": "text",
            "run_id": self.run_id,
            "task_id": task.task_id,
            "task_name": task.task_name,
            "task_type": task.task_type,
            "variation_idx": task.variation_idx,
            "simplification": task.simplification,
            "primitive": "trace_truncated",
            "canonical_family": "CTRL",
            "side_effect": False,
            "reason": reason,
            "trace_bytes_before": current_size,
            "trace_events_before": current_events,
            "max_trace_bytes": self.max_trace_bytes,
            "max_trace_events": self.max_trace_events,
        }
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        if self.max_trace_bytes > 0 and current_size + len(line.encode("utf-8")) > self.max_trace_bytes:
            return
        with path.open("a", encoding="utf-8") as f:
            f.write(line)

    def write_summary(self, summary: dict[str, Any], task_rows: list[dict[str, Any]], partial: bool = False) -> None:
        suffix = "_partial" if partial else ""
        summary_path = self.report_dir / f"{self.run_id}{suffix}_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        if task_rows:
            csv_path = self.report_dir / f"{self.run_id}{suffix}_tasks.csv"
            fieldnames = sorted({key for row in task_rows for key in row.keys()})
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(task_rows)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default
