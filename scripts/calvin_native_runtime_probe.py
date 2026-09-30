#!/usr/bin/env python3
"""Physical CALVIN action-boundary and hidden-verifier parity probe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

try:
    from _repo_bootstrap import bootstrap_repo_src
except ModuleNotFoundError:  # pragma: no cover
    from scripts._repo_bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from embodied_harness.calvin_agent_runtime import (  # noqa: E402
    CALVINAgentRuntimeBackend,
    CALVINRuntimeConfig,
)
from embodied_harness.paths import EXTERNAL_ROOT_ENV  # noqa: E402


SCHEMA = "agentic-embodied-arena/calvin-native-runtime-probe/v1"

# Harness-only parity trajectory.  It is never included in an evaluated
# agent's prompt or primitive output.
_PARITY_ACTIONS = (
    *((-0.7, 0.9, 0.0, 0.0, 0.0, 0.0, 1.0),) * 6,
    *((0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),) * 5,
    *((0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),) * 6,
    *((0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0),) * 6,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def run_probe(external_root: Path) -> dict[str, Any]:
    started = time.monotonic()
    calvin_root = external_root / "upstreams/calvin"
    dataset_root = external_root / "assets/calvin/dataset"
    os.environ["CALVIN_ASSET_DATA_ROOT"] = str(calvin_root / "calvin_env/data")
    backend = CALVINAgentRuntimeBackend(
        CALVINRuntimeConfig(
            sequence_id="native_turn_off_led",
            calvin_root=str(calvin_root),
            dataset_root=str(dataset_root),
            live=True,
            show_gui=False,
            use_egl=False,
            policy_backend="calvin_native_agent",
            native_task_key="turn_off_led",
        )
    )
    actions_run = 0
    public_output_leaks: list[str] = []
    try:
        task = backend.reset("calvin_native_turn_off_led_probe", seed=0)
        observation = backend.observe()
        language = backend.call_primitive("get_calvin_language_subgoal")
        context = backend.call_primitive("get_calvin_runtime_context")
        cameras = backend.call_primitive("observe_calvin_cameras")
        for action in _PARITY_ACTIONS:
            step = backend.call_primitive(
                "submit_calvin_action",
                action=list(action),
                agent_context={"source": "harness_only_parity_probe"},
            )
            actions_run += 1
            serialized = json.dumps(step.output, ensure_ascii=False).lower()
            public_output_leaks.extend(
                token
                for token in ("success", "reward", "checker", "oracle")
                if token in serialized and token not in public_output_leaks
            )
            if (
                backend._last_official_verification
                and backend._last_official_verification.get(  # noqa: SLF001
                    "success"
                )
            ):
                break
        verification = backend.verify("task")
        return {
            "schema_version": SCHEMA,
            "ok": bool(verification.ok) and not public_output_leaks,
            "stage": "done",
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "policy_checkpoint_required": False,
            "task_id": task.task_id,
            "language_available": language.ok,
            "language": language.output.get("subgoal"),
            "observation_keys": sorted(observation.data),
            "camera_views": sorted(cameras.output.get("cameras", {})),
            "action_schema": context.output.get("available_skill_schema", {}).get(
                "action"
            ),
            "actions_run": actions_run,
            "agent_visible_private_field_leaks": public_output_leaks,
            "verification": verification.to_dict(),
        }
    finally:
        backend.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args(argv)
    external_root = Path(args.external_root).expanduser().resolve()
    os.environ[EXTERNAL_ROOT_ENV] = str(external_root)
    report = run_probe(external_root)
    output = Path(args.output).expanduser().resolve()
    _atomic_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if args.require_success and not report["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
