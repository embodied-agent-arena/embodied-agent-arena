#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alfworld_visual_harness.manifest import MANIFEST_ID, build_manifest, summarize_suites


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RoBench ALFWorld visual official-data manifest.")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "json_2.1.1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_dir = ROOT / "outputs" / "manifests"
    report_dir = ROOT / "outputs" / "reports"
    suites = build_manifest(args.data_root, manifest_dir, force=args.force, seed=args.seed)
    summary = {
        "benchmark": "ALFWorld",
        "track": "visual_data_adapter",
        "manifest_id": MANIFEST_ID,
        "data_root": str(args.data_root),
        "manifest_path": str(manifest_dir / f"{MANIFEST_ID}.json"),
        "status": "data_manifest_ready",
        "note": "This validates official ALFWorld/ALFRED visual trajectory parsing only; simulator and generated-code smoke use the shared legacy THOR visual backend.",
        "suites": summarize_suites(suites),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"{MANIFEST_ID}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
