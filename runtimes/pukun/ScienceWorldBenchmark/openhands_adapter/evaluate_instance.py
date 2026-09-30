#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_text_trace_leakage import audit_trace_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one ScienceWorld Text OpenHands-style run.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary_path = args.summary or summary_for_run(args.run_id)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    run_id = str(summary["run_id"])
    trace_dir = ROOT / "outputs" / "traces" / run_id
    leakage = audit_trace_dir(trace_dir)
    result = {
        "benchmark": "ScienceWorld",
        "track": "text",
        "adapter": "openhands",
        "run_id": run_id,
        "summary_path": str(summary_path),
        "trace_dir": str(trace_dir),
        "successes": summary.get("successes", 0),
        "num_tasks": summary.get("num_tasks", 0),
        "success_rate": summary.get("success_rate", 0.0),
        "exception_count": summary.get("exception_count", 0),
        "code_exception_count": summary.get("code_exception_count", 0),
        "leakage_status": leakage["status"],
        "leakage_violation_count": leakage["violation_count"],
        "passed_runtime_gate": (
            summary.get("num_tasks") == 1
            and summary.get("exception_count", 0) == 0
            and summary.get("code_exception_count", 0) == 0
            and leakage["violation_count"] == 0
        ),
        "boundary": summary.get("boundary"),
    }
    output_path = args.output or ROOT / "outputs" / "reports" / f"{run_id}_openhands_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if leakage["violation_count"]:
        raise SystemExit(1)


def summary_for_run(run_id: str | None) -> Path:
    report_dir = ROOT / "outputs" / "reports"
    if run_id:
        path = report_dir / f"{run_id}_summary.json"
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    reports = sorted(report_dir.glob("*_summary.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise FileNotFoundError(f"No summary reports found under {report_dir}")
    return reports[-1]


if __name__ == "__main__":
    main()
