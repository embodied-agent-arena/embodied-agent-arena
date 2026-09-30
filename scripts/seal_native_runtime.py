#!/usr/bin/env python3
"""Capture versioned receipts for benchmark-native Python runtimes and assets."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

try:
    from _repo_bootstrap import bootstrap_repo_src
except ModuleNotFoundError:  # pragma: no cover
    from scripts._repo_bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from embodied_harness.native_runtime_receipt import (  # noqa: E402
    capture_native_runtime_receipt,
)
from embodied_harness.paths import (  # noqa: E402
    ARTIFACT_ROOT_ENV,
    EXTERNAL_ROOT_ENV,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("benchmark_id", nargs="+")
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--task", help="Use a packaged task's native interpreter, assets, and receipt route")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--full-content",
        action="store_true",
        help="Hash runtime prefixes and every required asset byte (one-time, expensive).",
    )
    parser.add_argument("--hash-runtime", action="store_true")
    parser.add_argument("--hash-assets", action="store_true")
    parser.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    os.environ[EXTERNAL_ROOT_ENV] = str(args.external_root.expanduser().resolve())
    if args.artifact_root is not None:
        os.environ[ARTIFACT_ROOT_ENV] = str(args.artifact_root.expanduser().resolve())
    if args.task:
        from embodied_harness import release
        root = Path(__file__).resolve().parents[1]
        data = release.dataset_root(str(args.data_root) if args.data_root else None)
        task = next((t for t in release.read_cases(data) if t['task_id'] == args.task), None)
        if task is None or args.benchmark_id != [task['benchmark_id']] or task['wave'] != 'W4':
            raise SystemExit('--task must identify a W4 task matching the single benchmark argument')
        env = release.environment(root, data)
        env.update(release.expand_paths(task.get('env', {}), release.path_bindings(root, data)))
        os.environ.update(env)
        if args.output_root is None and env.get('EMBODIED_ARENA_NATIVE_RECEIPT_ROOT'):
            args.output_root = Path(env['EMBODIED_ARENA_NATIVE_RECEIPT_ROOT'])
    rows = []
    failures = []
    identity_cache = {}
    for benchmark_id in dict.fromkeys(args.benchmark_id):
        try:
            receipt = capture_native_runtime_receipt(
                benchmark_id,
                output_root=args.output_root,
                hash_runtime=args.full_content or args.hash_runtime,
                hash_assets=args.full_content or args.hash_assets,
                identity_cache=identity_cache,
            )
        except Exception as exc:  # noqa: BLE001 - continue sealing peer runtimes.
            failures.append(
                {
                    "benchmark_id": benchmark_id,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue
        rows.append(
            {
                "benchmark_id": benchmark_id,
                "receipt_sha256": receipt["receipt_sha256"],
                "completeness": receipt["completeness"],
            }
        )
    report = {
        "ok": not failures,
        "sealed_count": len(rows),
        "failed_count": len(failures),
        "receipts": rows,
        "failures": failures,
    }
    print(json.dumps(report, ensure_ascii=False, indent=args.indent, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
