#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROB_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from alfred_official_adapter.manifest import MANIFEST_ID, build_manifest, summarize_suites


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROB_ROOT / "ALFWorldBenchmark" / "data" / "json_2.1.1",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_dir = ROOT / "outputs" / "manifests"
    report_dir = ROOT / "outputs" / "reports"
    suites = build_manifest(args.data_root, manifest_dir, force=args.force, seed=args.seed)
    summary = {
        "benchmark": "ALFRED",
        "track": "official_visual_data_adapter",
        "manifest_id": MANIFEST_ID,
        "data_root": str(args.data_root),
        "manifest_path": str(manifest_dir / f"{MANIFEST_ID}.json"),
        "status": "data_manifest_ready",
        "note": "This validates official trajectory data parsing only; it does not start THOR or run a coding-agent visual episode.",
        "suites": summarize_suites(suites),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / f"{MANIFEST_ID}_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
