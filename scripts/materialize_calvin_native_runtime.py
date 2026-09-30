#!/usr/bin/env python3
"""Materialize the small checkpoint-free CALVIN runtime used by the native harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

try:
    from _repo_bootstrap import bootstrap_repo_src
except ModuleNotFoundError:  # pragma: no cover
    from scripts._repo_bootstrap import bootstrap_repo_src

bootstrap_repo_src()

from embodied_harness.paths import EXTERNAL_ROOT_ENV, get_project_paths  # noqa: E402


SCHEMA = "agentic-embodied-arena/calvin-native-materialization/v1"
PYTHON_VERSION = "3.10"
PINNED_PACKAGES = (
    "antlr4-python3-runtime==4.9.3",
    "numpy==1.26.4",
    "pybullet==3.2.7",
    "gym==0.26.2",
    "gym-notices==0.1.0",
    "hydra-core==1.3.2",
    "hydra-colorlog==1.2.0",
    "omegaconf==2.3.0",
    "cloudpickle==3.1.1",
    "colorlog==6.12.0",
    "contourpy==1.3.2",
    "cycler==0.12.1",
    "fonttools==4.64.0",
    "GitPython==3.1.45",
    "gitdb==4.0.12",
    "smmap==5.0.3",
    "opencv-python==4.11.0.86",
    "packaging==26.3",
    "pillow==12.3.0",
    "pyparsing==3.3.2",
    "python-dateutil==2.9.0.post0",
    "pytz==2026.3.post1",
    "PyYAML==6.0.3",
    "six==1.17.0",
    "tzdata==2026.3",
    "scipy==1.15.3",
    "rich==14.1.0",
    "markdown-it-py==4.2.0",
    "mdurl==0.1.2",
    "Pygments==2.21.0",
    "numpy-quaternion==2024.0.12",
    "pandas==2.3.2",
    "matplotlib==3.10.6",
    "kiwisolver==1.5.1",
    "numba==0.61.2",
    "llvmlite==0.44.0",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(argv: list[str], *, env: dict[str, str] | None = None) -> None:
    subprocess.run(argv, check=True, env=env)


def _git_revision(root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _compose_worker(calvin_root: Path, output: Path) -> None:
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(calvin_root / "calvin_env/conf"), version_base=None
    ):
        config = compose(
            config_name="config_data_collection",
            overrides=[
                "use_vr=false",
                "record=false",
                "cameras=static_and_gripper",
                "scene=calvin_scene_D_eval",
                "robot=panda_longer_finger",
                "data_path=${oc.env:CALVIN_ASSET_DATA_ROOT}",
            ],
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output, resolve=False)


def materialize(
    external_root: Path,
    *,
    install: bool,
    run_probe: bool,
    probe_output: Path,
) -> dict[str, Any]:
    project_root = get_project_paths().project_root
    calvin_root = external_root / "upstreams/calvin"
    environment_root = external_root / "environments/calvin"
    python = environment_root / "bin/python"
    config_path = external_root / "assets/calvin/dataset/.hydra/merged_config.yaml"
    expected = (
        calvin_root / "calvin_env/conf/config_data_collection.yaml",
        calvin_root / "calvin_env/data/calvin_table_D/urdf/calvin_table_D.urdf",
        calvin_root / "calvin_env/calvin_env/envs/play_table_env.py",
        calvin_root / "calvin_env/conf/tasks/new_playtable_tasks.yaml",
    )
    missing = [str(path) for path in expected if not path.is_file()]
    if missing:
        raise RuntimeError(
            "Pinned CALVIN checkout is incomplete: " + ", ".join(missing)
        )

    uv = shutil.which("uv")
    if install:
        if uv is None:
            raise RuntimeError(
                "uv is required to materialize the pinned CALVIN environment"
            )
        if not python.is_file():
            _run([uv, "venv", "--python", PYTHON_VERSION, str(environment_root)])
        _run([uv, "pip", "install", "--python", str(python), *PINNED_PACKAGES])
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--no-deps",
                "-e",
                str(calvin_root / "calvin_env"),
            ]
        )
        _run([uv, "pip", "check", "--python", str(python)])
    elif not python.is_file():
        raise RuntimeError(
            f"CALVIN Python is missing: {python}; rerun without --skip-install"
        )

    compose_env = dict(os.environ)
    compose_env[EXTERNAL_ROOT_ENV] = str(external_root)
    compose_env["CALVIN_ASSET_DATA_ROOT"] = str(calvin_root / "calvin_env/data")
    _run(
        [
            str(python),
            str(Path(__file__).resolve()),
            "--compose-worker",
            "--external-root",
            str(external_root),
        ],
        env=compose_env,
    )

    lock_path = environment_root / "requirements-native.lock"
    lock_path.write_text("\n".join(PINNED_PACKAGES) + "\n", encoding="utf-8")
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "ready": True,
        "policy_checkpoint_required": False,
        "python": str(python),
        "python_version": PYTHON_VERSION,
        "calvin_revision": _git_revision(calvin_root),
        "runtime_config": str(config_path),
        "runtime_config_sha256": _sha256(config_path),
        "pinned_packages": list(PINNED_PACKAGES),
        "official_assets_reused_from_checkout": True,
    }
    if run_probe:
        probe_env = dict(compose_env)
        probe_env["PYTHONPATH"] = str(project_root / "src")
        _run(
            [
                str(python),
                str(project_root / "scripts/calvin_native_runtime_probe.py"),
                "--external-root",
                str(external_root),
                "--output",
                str(probe_output),
                "--require-success",
            ],
            env=probe_env,
        )
        report["probe"] = str(probe_output)
    manifest = external_root / "assets/calvin/native-runtime-manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report["manifest"] = str(manifest)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--external-root", default=str(get_project_paths().external_root)
    )
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--probe-output", default="reports/calvin_native_runtime_probe.json"
    )
    parser.add_argument("--compose-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    external_root = Path(args.external_root).expanduser().resolve()
    if args.compose_worker:
        _compose_worker(
            external_root / "upstreams/calvin",
            external_root / "assets/calvin/dataset/.hydra/merged_config.yaml",
        )
        return 0
    os.environ[EXTERNAL_ROOT_ENV] = str(external_root)
    report = materialize(
        external_root,
        install=not args.skip_install,
        run_probe=args.probe,
        probe_output=Path(args.probe_output).expanduser().resolve(),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
