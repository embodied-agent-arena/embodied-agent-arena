from __future__ import annotations

import ast
import contextlib
import http.client
import json
import math
import os
import re
import signal
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .schemas import PrimitiveCard, PrimitiveResult, VerificationResult


DEFAULT_CONFIG = "env_configs/cube_stack/franka_robosuite_cube_stack.yaml"

API_ALIASES = {
    "observe_capx_scene": {
        "upstream": "env.get_observation()",
        "returns": "public structured observation plus any raw visual frames genuinely present upstream; evaluator fields are excluded",
    },
    "enumerate_capx_objects": {
        "upstream": "env.get_observation() -> recursively enumerate caller-selectable pose records",
        "returns": "generic object candidates with observation paths and public pose geometry",
    },
    "compose_capx_geometry": {
        "upstream": "caller-selected anchor + direction * (distance + extent terms)",
        "returns": "one caller-parameterized target pose; does not select objects, arms, or actions",
    },
    "submit_capx_action": {
        "upstream": "caller-selected public action on caller-selected arm",
        "returns": "the public API call result without task-success fields",
    },
    "capx_get_object_pose": {
        "upstream": "get_object_pose(object_name, return_bbox_extent=False)",
        "returns": "position XYZ meters, quaternion_wxyz, optional bbox_extent",
    },
    "capx_sample_grasp_pose": {
        "upstream": "sample_grasp_pose(object_name)",
        "returns": "grasp position XYZ meters and quaternion_wxyz",
    },
    "capx_goto_pose": {
        "upstream": "goto_pose(position, quaternion_wxyz, z_approach=0.0)",
        "returns": "None; movement happens in CaP-X simulator/control backend",
    },
    "capx_open_gripper": {
        "upstream": "open_gripper()",
        "returns": "None; gripper command executes in backend",
    },
    "capx_close_gripper": {
        "upstream": "close_gripper()",
        "returns": "None; gripper command executes in backend",
    },
}

FORBIDDEN_TRACE_TERMS = ("oracle", "checker", "success_function", "reward_function")
API_METHOD_BY_PRIMITIVE = {
    "capx_get_object_pose": "get_object_pose",
    "capx_sample_grasp_pose": "sample_grasp_pose",
    "capx_goto_pose": "goto_pose",
    "capx_open_gripper": "open_gripper",
    "capx_close_gripper": "close_gripper",
}
API_SERVER_NAMES = {
    "capx.serving.launch_sam3_server.main": "sam3",
    "capx.serving.launch_contact_graspnet_server.main": "contact_graspnet",
    "capx.serving.launch_pyroki_server.main": "pyroki",
    "capx.serving.launch_sam2_server.main": "sam2",
    "capx.serving.launch_owlvit_server.main": "owlvit",
    "capx.serving.launch_curobo_server.main": "curobo",
}
CAPX_PYROKI_PORT_ENV = "AGENTIC_EMBODIED_ARENA_CAPX_PYROKI_PORT"
CAPX_PYROKI_STARTUP_TIMEOUT_ENV = (
    "AGENTIC_EMBODIED_ARENA_CAPX_PYROKI_STARTUP_TIMEOUT_SECONDS"
)
CAPX_IMPORT_PREFLIGHT_TIMEOUT_ENV = (
    "AGENTIC_EMBODIED_ARENA_CAPX_IMPORT_PREFLIGHT_TIMEOUT_SECONDS"
)
CAPX_PYROKI_PANDA_ASSET_ENV = (
    "AGENTIC_EMBODIED_ARENA_CAPX_ASSET_PYROKI_PANDA_DESCRIPTION"
)
CAPX_PYROKI_HOST = "127.0.0.1"
CAPX_SAM3_PORT_ENV = "AGENTIC_EMBODIED_ARENA_CAPX_SAM3_PORT"
CAPX_CONTACT_GRASPNET_PORT_ENV = (
    "AGENTIC_EMBODIED_ARENA_CAPX_CONTACT_GRASPNET_PORT"
)
CAPX_PERCEPTION_STARTUP_TIMEOUT_ENV = (
    "AGENTIC_EMBODIED_ARENA_CAPX_PERCEPTION_STARTUP_TIMEOUT_SECONDS"
)
CAPX_PERCEPTION_HOST = "127.0.0.1"


def _capx_pyroki_port() -> int:
    raw = os.getenv(CAPX_PYROKI_PORT_ENV, "8116")
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{CAPX_PYROKI_PORT_ENV} must be an integer") from exc
    if not 1024 <= port <= 65535:
        raise ValueError(f"{CAPX_PYROKI_PORT_ENV} must be between 1024 and 65535")
    return port


def _capx_pyroki_startup_timeout_seconds() -> float:
    raw = os.getenv(CAPX_PYROKI_STARTUP_TIMEOUT_ENV, "180")
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CAPX_PYROKI_STARTUP_TIMEOUT_ENV} must be numeric"
        ) from exc
    if not 1.0 <= timeout_seconds <= 900.0:
        raise ValueError(
            f"{CAPX_PYROKI_STARTUP_TIMEOUT_ENV} must be between 1 and 900 seconds"
        )
    return timeout_seconds


def _capx_import_preflight_timeout_seconds() -> float:
    raw = os.getenv(CAPX_IMPORT_PREFLIGHT_TIMEOUT_ENV, "180")
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CAPX_IMPORT_PREFLIGHT_TIMEOUT_ENV} must be numeric"
        ) from exc
    if not 10.0 <= timeout_seconds <= 900.0:
        raise ValueError(
            f"{CAPX_IMPORT_PREFLIGHT_TIMEOUT_ENV} must be between 10 and 900 seconds"
        )
    return timeout_seconds


def _capx_service_port(environment_name: str, default: int) -> int:
    raw = os.getenv(environment_name, str(default))
    try:
        port = int(raw)
    except ValueError as exc:
        raise ValueError(f"{environment_name} must be an integer") from exc
    if not 1024 <= port <= 65535:
        raise ValueError(f"{environment_name} must be between 1024 and 65535")
    return port


def _capx_perception_startup_timeout_seconds() -> float:
    raw = os.getenv(CAPX_PERCEPTION_STARTUP_TIMEOUT_ENV, "240")
    try:
        timeout_seconds = float(raw)
    except ValueError as exc:
        raise ValueError(
            f"{CAPX_PERCEPTION_STARTUP_TIMEOUT_ENV} must be numeric"
        ) from exc
    if not 10.0 <= timeout_seconds <= 900.0:
        raise ValueError(
            f"{CAPX_PERCEPTION_STARTUP_TIMEOUT_ENV} must be between 10 and 900 seconds"
        )
    return timeout_seconds


def _assert_capx_pyroki_port_available(port: int) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
        try:
            listener.bind((CAPX_PYROKI_HOST, port))
        except OSError as exc:
            raise RuntimeError(
                f"evaluator-owned CaP-X PyRoKi port is unavailable: {CAPX_PYROKI_HOST}:{port}"
            ) from exc


def _stop_capx_pyroki_process(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)


def _capx_pyroki_panda_asset_root(capx_root: Path) -> Path:
    configured = os.getenv(CAPX_PYROKI_PANDA_ASSET_ENV)
    if configured:
        asset_root = Path(configured).expanduser().resolve()
    else:
        try:
            repository_root = capx_root.resolve().parents[2]
        except IndexError as exc:
            raise FileNotFoundError(
                f"{CAPX_PYROKI_PANDA_ASSET_ENV} is required outside the repository layout"
            ) from exc
        asset_root = (
            repository_root
            / "external"
            / "assets"
            / "capx"
            / "pyroki_panda_description"
        ).resolve()
    required = (
        asset_root / "LICENSE",
        asset_root / "urdf" / "panda.urdf",
        asset_root / "meshes" / "collision" / "hand.stl",
        asset_root / "meshes" / "visual" / "hand.dae",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "sealed CaP-X Panda description is incomplete: " + ", ".join(missing)
        )
    return asset_root


def _terminate_capx_pyroki_with_parent() -> None:
    """Make the Linux sidecar die even if the native worker is force-killed."""

    if sys.platform != "linux":
        return
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno))
    if os.getppid() == 1:
        os.kill(os.getpid(), signal.SIGTERM)


def _launch_capx_pyroki_server(
    capx_root: Path, *, port: int, panda_asset_root: Path
) -> subprocess.Popen[Any]:
    _assert_capx_pyroki_port_available(port)
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ""
    child_env["JAX_PLATFORM_NAME"] = "cpu"
    child_env["JAX_PLATFORMS"] = "cpu"
    child_env["GIT_TERMINAL_PROMPT"] = "0"
    child_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(capx_root), child_env.get("PYTHONPATH", ""))
        if value
    )
    program = (
        "import pathlib, sys, types; "
        "asset_root=pathlib.Path(sys.argv[2]).resolve(); "
        "import robot_descriptions; "
        "description=types.ModuleType('robot_descriptions.panda_description'); "
        "description.PACKAGE_PATH=str(asset_root); "
        "description.URDF_PATH=str(asset_root/'urdf'/'panda.urdf'); "
        "sys.modules['robot_descriptions.panda_description']=description; "
        "setattr(robot_descriptions, 'panda_description', description); "
        "from capx.serving.launch_pyroki_server import main; "
        "main(robot='panda_description', target_link='panda_hand', "
        "port=int(sys.argv[1]), host='127.0.0.1')"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", program, str(port), str(panda_asset_root)],
        cwd=capx_root,
        env=child_env,
        stdout=sys.stderr,
        stderr=sys.stderr,
        preexec_fn=_terminate_capx_pyroki_with_parent,
    )
    deadline = time.monotonic() + _capx_pyroki_startup_timeout_seconds()
    last_error = "server did not answer"
    try:
        while time.monotonic() < deadline:
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"evaluator-owned CaP-X PyRoKi exited during startup with code {return_code}"
                )
            connection = http.client.HTTPConnection(
                CAPX_PYROKI_HOST, port, timeout=1.0
            )
            try:
                connection.request("GET", "/docs")
                response = connection.getresponse()
                response.read()
                if response.status == 200 and process.poll() is None:
                    return process
                last_error = f"unexpected HTTP status {response.status}"
            except OSError as exc:
                last_error = str(exc)
            finally:
                connection.close()
            time.sleep(0.25)
        raise RuntimeError(
            "evaluator-owned CaP-X PyRoKi did not become ready on "
            f"{CAPX_PYROKI_HOST}:{port}: {last_error}"
        )
    except BaseException:
        _stop_capx_pyroki_process(process)
        raise


def _capx_external_root(capx_root: Path) -> Path:
    try:
        external_root = capx_root.resolve().parents[1]
    except IndexError as exc:
        raise FileNotFoundError(
            "CaP-X checkout must be below <external-root>/upstreams/capx"
        ) from exc
    if external_root.name != "external":
        raise FileNotFoundError(
            "CaP-X checkout must be below <external-root>/upstreams/capx"
        )
    return external_root


def release_capx_sidecar_cache_after_requests(module: Any) -> None:
    """Release unused CUDA allocator blocks inside the service's existing lock.

    The original inference function, inputs and weights are retained. The
    ContactGraspNet server needs eval/no-grad in the actual inference thread;
    its upstream startup leaves training mode and gradient recording enabled.
    This lets the simulator, SAM3 and ContactGraspNet share one GPU between
    sequential requests instead of retaining each service's peak allocation.
    """
    import gc
    import functools
    import torch
    original = module._run_on_gpu

    @functools.wraps(original)
    async def run(fn, *args, **kwargs):
        def infer_and_release():
            try:
                estimator = getattr(module, "_GRASP_ESTIMATOR", None)
                if estimator is not None:
                    estimator.model.eval()
                    with torch.no_grad():
                        return fn(*args, **kwargs)
                # SAM3 already selects bfloat16 in its own inference context.
                # Avoid retaining cast copies of all weights across image and
                # text inference; preserve the original operations and dtype.
                # Autocast cache state is thread-local, so restore it in this
                # actual inference worker even if the request raises.
                cache_enabled = torch.is_autocast_cache_enabled()
                torch.set_autocast_cache_enabled(False)
                try:
                    return fn(*args, **kwargs)
                finally:
                    torch.set_autocast_cache_enabled(cache_enabled)
            finally:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return await original(infer_and_release)
    module._run_on_gpu = run


def _launch_capx_perception_process(
    capx_root: Path,
    *,
    service: str,
    port: int,
) -> subprocess.Popen[Any]:
    _assert_capx_pyroki_port_available(port)
    external_root = _capx_external_root(capx_root)
    assets_root = external_root / "assets" / "capx"
    child_env = dict(os.environ)
    child_env["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(capx_root), child_env.get("PYTHONPATH", ""))
        if value
    )
    child_env["PYTHONNOUSERSITE"] = "1"
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    if service == "sam3":
        hf_home = Path(
            child_env.get("HF_HOME") or assets_root / "sam3-hf"
        ).resolve()
        snapshots = hf_home / "hub" / "models--facebook--sam3" / "snapshots"
        if not any(snapshots.glob("*/sam3.pt")):
            raise FileNotFoundError(f"pinned SAM3 checkpoint is missing below {hf_home}")
        child_env["HF_HOME"] = str(hf_home)
        child_env["HF_HUB_OFFLINE"] = "1"
        child_env["HF_HUB_DISABLE_XET"] = "1"
        child_env["TRANSFORMERS_OFFLINE"] = "1"
        program = (
            "import capx.serving.launch_sam3_server as service; "
            "from embodied_harness.capx_comparator_runtime import release_capx_sidecar_cache_after_requests; "
            "release_capx_sidecar_cache_after_requests(service); "
            "service.main(device='cuda', port=int(__import__('sys').argv[1]), "
            "host='127.0.0.1')"
        )
    elif service == "contact_graspnet":
        checkpoint = (
            capx_root
            / "capx/third_party/contact_graspnet_pytorch/checkpoints/"
            "contact_graspnet/checkpoints/model.pt"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"pinned ContactGraspNet checkpoint is missing: {checkpoint}"
            )
        child_env.setdefault(
            "TORCH_EXTENSIONS_DIR",
            str(external_root / "environments/capx/runtime-cache/torch_extensions"),
        )
        program = (
            "import capx.serving.launch_contact_graspnet_server as service; "
            "from embodied_harness.capx_comparator_runtime import release_capx_sidecar_cache_after_requests; "
            "release_capx_sidecar_cache_after_requests(service); "
            "service.main(device='cuda', port=int(__import__('sys').argv[1]), "
            "host='127.0.0.1')"
        )
    else:
        raise ValueError(f"unsupported CaP-X perception sidecar: {service}")
    return subprocess.Popen(
        [os.getenv("CAPX_SIDECAR_PYTHON", sys.executable), "-c", program, str(port)],
        cwd=capx_root,
        env=child_env,
        stdout=sys.stderr,
        stderr=sys.stderr,
        preexec_fn=_terminate_capx_pyroki_with_parent,
    )


def _wait_capx_perception_processes(
    processes: dict[str, tuple[subprocess.Popen[Any], int]],
) -> None:
    pending = dict(processes)
    deadline = time.monotonic() + _capx_perception_startup_timeout_seconds()
    last_errors: dict[str, str] = {}
    while pending and time.monotonic() < deadline:
        for name, (process, port) in list(pending.items()):
            return_code = process.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"evaluator-owned CaP-X {name} exited during startup with "
                    f"code {return_code}"
                )
            connection = http.client.HTTPConnection(
                CAPX_PERCEPTION_HOST,
                port,
                timeout=1.0,
            )
            try:
                connection.request("GET", "/docs")
                response = connection.getresponse()
                response.read()
                if response.status == 200 and process.poll() is None:
                    pending.pop(name)
                else:
                    last_errors[name] = f"unexpected HTTP status {response.status}"
            except OSError as exc:
                last_errors[name] = str(exc)
            finally:
                connection.close()
        if pending:
            time.sleep(0.25)
    if pending:
        details = ", ".join(
            f"{name}:{last_errors.get(name, 'server did not answer')}"
            for name in sorted(pending)
        )
        raise RuntimeError(
            "evaluator-owned CaP-X perception services did not become ready: "
            + details
        )


def _configure_capx_perception_clients(*, sam3_port: int, graspnet_port: int) -> None:
    import capx.integrations.vision.graspnet as graspnet_client
    import capx.integrations.vision.sam3 as sam3_client

    sam3_client.SERVICE_URL = f"http://{CAPX_PERCEPTION_HOST}:{sam3_port}"
    graspnet_client.SERVICE_URL = (
        f"http://{CAPX_PERCEPTION_HOST}:{graspnet_port}"
    )


def _capx_git_head(capx_root: Path) -> str | None:
    git_executable = (
        os.getenv("CAPX_GIT_EXECUTABLE")
        or os.getenv("GIT_PYTHON_GIT_EXECUTABLE")
        or shutil.which("git")
    )
    if not git_executable:
        return None
    try:
        head = subprocess.run(
            [git_executable, "rev-parse", "--short", "HEAD"],
            cwd=capx_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if head.returncode != 0:
        return None
    return head.stdout.strip() or None


@dataclass(frozen=True)
class CapXRuntimeContext:
    mode: str
    capx_root: str | None
    config_path: str
    route: str
    env_target: str | None
    low_level: Any
    configured_apis: list[str]
    prompt_source: str
    prompt: str | None
    api_servers: list[dict[str, Any]]
    agent_context: dict[str, Any]
    trace_evidence: dict[str, Any]
    route_capabilities: dict[str, Any]
    action_api_schema: dict[str, Any]
    live_result: dict[str, Any] | None = None


@dataclass
class CapXLiveSession:
    """One upstream CaP-X environment graph kept alive for an entire episode."""

    environment: Any
    low_level_environment: Any
    live_api: Any
    reset_observation: dict[str, Any]
    reset_info: dict[str, Any]
    pyroki_process: subprocess.Popen[Any] | None = None
    pyroki_port: int | None = None
    runtime_workdir: Path | None = None
    capx_root: Path | None = None
    sam3_process: subprocess.Popen[Any] | None = None
    contact_graspnet_process: subprocess.Popen[Any] | None = None
    sam3_port: int | None = None
    contact_graspnet_port: int | None = None
    serial_grasp_services: bool = False

    @contextlib.contextmanager
    def activate_runtime_workdir(self):
        """Run upstream API calls in the episode-owned writable directory."""

        if self.runtime_workdir is None:
            yield
            return
        previous_cwd = Path.cwd()
        os.chdir(self.runtime_workdir)
        try:
            yield
        finally:
            os.chdir(previous_cwd)

    def close(self) -> None:
        try:
            close = getattr(self.environment, "close", None)
            if callable(close):
                close()
            native = getattr(self.low_level_environment, "robosuite_env", None)
            close = getattr(native, "close", None)
            if callable(close) and native is not self.environment:
                close()
        finally:
            for attribute in ("contact_graspnet_process", "sam3_process"):
                process = getattr(self, attribute)
                setattr(self, attribute, None)
                _stop_capx_pyroki_process(process)
            process = self.pyroki_process
            self.pyroki_process = None
            _stop_capx_pyroki_process(process)
            runtime_workdir = self.runtime_workdir
            self.runtime_workdir = None
            if runtime_workdir is not None:
                shutil.rmtree(runtime_workdir, ignore_errors=True)
            from .behavior1k_agent_runtime import _shutdown_loaded_omnigibson
            _shutdown_loaded_omnigibson()

    def ensure_perception_sidecars(
        self,
        *,
        require_contact_graspnet: bool,
        require_sam3: bool = True,
    ) -> dict[str, Any]:
        """Lazily start only the model services required by a public API call."""

        if self.capx_root is None:
            raise RuntimeError("CaP-X session has no pinned source root")
        sam3_port = self.sam3_port or _capx_service_port(CAPX_SAM3_PORT_ENV, 8114)
        graspnet_port = self.contact_graspnet_port or _capx_service_port(
            CAPX_CONTACT_GRASPNET_PORT_ENV,
            8115,
        )
        launched: dict[str, tuple[subprocess.Popen[Any], int]] = {}
        memory_release = None
        if require_sam3 and self.sam3_process is None:
            # Scene creation can leave large unused PyTorch allocator blocks.
            # Releasing only that cache preserves tensors, physics and images.
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                reserved_before = torch_module.cuda.memory_reserved()
                allocated = torch_module.cuda.memory_allocated()
                torch_module.cuda.empty_cache()
                free_bytes, _ = torch_module.cuda.mem_get_info()
                memory_release = {"reserved_before": reserved_before,
                                  "allocated": allocated,
                                  "reserved_after": torch_module.cuda.memory_reserved(),
                                  "free_after": free_bytes}
                print("CaPX perception CUDA admission: " + json.dumps(memory_release),
                      file=sys.stderr, flush=True)
                if free_bytes < 5.5 * 1024 ** 3:
                    raise RuntimeError(
                        "CaPX perception admission deferred: need 5.5 GiB free after "
                        "releasing unused CUDA cache; memory=" + json.dumps(memory_release)
                    )
        try:
            if require_sam3 and self.sam3_process is None:
                self.sam3_process = _launch_capx_perception_process(
                    self.capx_root,
                    service="sam3",
                    port=sam3_port,
                )
                launched["sam3"] = (self.sam3_process, sam3_port)
            elif require_sam3 and self.sam3_process.poll() is not None:
                raise RuntimeError("evaluator-owned CaP-X SAM3 sidecar exited")
            if require_contact_graspnet and self.contact_graspnet_process is None:
                self.contact_graspnet_process = _launch_capx_perception_process(
                    self.capx_root,
                    service="contact_graspnet",
                    port=graspnet_port,
                )
                launched["contact_graspnet"] = (
                    self.contact_graspnet_process,
                    graspnet_port,
                )
            elif (
                require_contact_graspnet
                and self.contact_graspnet_process is not None
                and self.contact_graspnet_process.poll() is not None
            ):
                raise RuntimeError(
                    "evaluator-owned CaP-X ContactGraspNet sidecar exited"
                )
            _wait_capx_perception_processes(launched)
        except BaseException:
            for name, (process, _port) in launched.items():
                _stop_capx_pyroki_process(process)
                if name == "sam3":
                    self.sam3_process = None
                else:
                    self.contact_graspnet_process = None
            raise
        self.sam3_port = sam3_port
        self.contact_graspnet_port = graspnet_port
        _configure_capx_perception_clients(
            sam3_port=sam3_port,
            graspnet_port=graspnet_port,
        )
        return {
            "sam3": self.sam3_process is not None
            and self.sam3_process.poll() is None,
            "contact_graspnet": self.contact_graspnet_process is not None
            and self.contact_graspnet_process.poll() is None,
            "launched_now": sorted(launched),
            **({"cuda_cache_release": memory_release} if memory_release is not None else {}),
        }


def open_capx_live_session(
    capx_root: Path,
    config_path: Path,
    *,
    seed: int | None,
    api_name: str = "FrankaControlApi",
) -> CapXLiveSession:
    """Instantiate and reset the official upstream graph exactly once.

    The returned API and low-level evaluator object are children of the same
    high-level environment.  Callers must retain this session instead of
    reconstructing either object from a serialized observation.
    """

    capx_root = capx_root.resolve()
    config_path = config_path.resolve()
    if not (capx_root / "capx" / "envs" / "configs" / "instantiate.py").is_file():
        raise FileNotFoundError(f"CaP-X runtime package is unavailable: {capx_root}")
    if not config_path.is_file():
        raise FileNotFoundError(f"CaP-X runtime config is unavailable: {config_path}")
    root_text = str(capx_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    previous_cwd = Path.cwd()
    environment: Any | None = None
    pyroki_port = _capx_pyroki_port()
    sam3_port = _capx_service_port(CAPX_SAM3_PORT_ENV, 8114)
    graspnet_port = _capx_service_port(CAPX_CONTACT_GRASPNET_PORT_ENV, 8115)
    pyroki_process = None
    if api_name in {"FrankaControlApi", "FrankaControlSpillWipeApi", "FrankaLiberoApi"}:
        panda_asset_root = _capx_pyroki_panda_asset_root(capx_root)
        pyroki_process = _launch_capx_pyroki_server(
            capx_root, port=pyroki_port, panda_asset_root=panda_asset_root,
        )
    runtime_workdir = Path(tempfile.mkdtemp(prefix="aea-capx-episode-"))
    try:
        os.chdir(capx_root)
        if api_name == "R1ProControlApi":
            _prepare_capx_behavior_headless()
        import capx.integrations  # noqa: F401 - registers upstream APIs and environments
        if api_name == 'FrankaLiberoApi':
            from .capx_libero_support import install_reset_fix
            install_reset_fix()
        if api_name == "R1ProControlApi":
            _finish_capx_behavior_headless()
        _configure_capx_perception_clients(
            sam3_port=sam3_port,
            graspnet_port=graspnet_port,
        )
        from capx.envs.configs.instantiate import instantiate
        from capx.envs.configs.loader import DictLoader

        config = DictLoader.load(str(config_path))
        if not isinstance(config, dict) or not isinstance(config.get("env"), dict):
            raise ValueError("CaP-X runtime config must contain an env mapping")
        if api_name == "R1ProControlApi":
            config = prepare_capx_r1pro_scene_config(config, capx_root, runtime_workdir)
            # CuRobo's +/-5 m bounds are relative to the fixed robot root,
            # not the sampled task spawn. With root x=5 and spawn x~0.58,
            # valid household goals at world x<0 are outside those bounds.
            # Enlarge only the virtual-base planning workspace; collisions,
            # robot/object state and physical action execution are unchanged.
            from omnigibson.action_primitives import curobo as capx_curobo
            capx_curobo.m.HOLONOMIC_BASE_PRISMATIC_JOINT_LIMIT = 20.0
        if api_name in {"FrankaControlApi", "FrankaControlSpillWipeApi"} and seed is not None:
            # Upstream wrappers seed their own unused _rng, but omit the seed
            # when constructing Robosuite. Inject only the constructor seed,
            # before placements / robot initialization create their RNGs.
            from robosuite.environments.base import MujocoEnv
            original_init = MujocoEnv.__init__
            def seeded_init(instance, *args, **kwargs):
                kwargs['seed'] = seed
                return original_init(instance, *args, **kwargs)
            MujocoEnv.__init__ = seeded_init
            try:
                environment = instantiate(config["env"])
            finally:
                MujocoEnv.__init__ = original_init
        else:
            environment = instantiate(config["env"])
        reset_observation, reset_info = environment.reset(seed=seed)
        if api_name in {"FrankaControlApi", "FrankaControlSpillWipeApi"}:
            reset_info['harness_robosuite_constructor_seed'] = seed
        if api_name == "R1ProControlApi" and isinstance(reset_info, dict):
            reset_info["harness_planner_configuration"] = {
                "holonomic_base_prismatic_limit_m": 20.0,
                "reference_frame": "fixed_robot_root",
                "collision_checks_disabled": False,
                "task_state_modified": False,
            }
    except BaseException:
        if environment is not None:
            with contextlib.suppress(Exception):
                CapXLiveSession(
                    environment,
                    getattr(environment, "low_level_env", None),
                    None,
                    {},
                    {},
                    pyroki_process,
                    pyroki_port,
                    runtime_workdir,
                ).close()
                pyroki_process = None
                runtime_workdir = None
        else:
            _stop_capx_pyroki_process(pyroki_process)
            pyroki_process = None
        if runtime_workdir is not None:
            shutil.rmtree(runtime_workdir, ignore_errors=True)
        raise
    finally:
        os.chdir(previous_cwd)

    low_level = getattr(environment, "low_level_env", None)
    apis = getattr(environment, "_apis", None)
    live_api = apis.get(api_name) if isinstance(apis, dict) else None
    if low_level is None or live_api is None:
        with contextlib.suppress(Exception):
            CapXLiveSession(
                environment,
                low_level,
                live_api,
                {},
                {},
                pyroki_process,
                pyroki_port,
                runtime_workdir,
            ).close()
            pyroki_process = None
            runtime_workdir = None
        raise RuntimeError(
            f"CaP-X live environment did not expose low_level_env and _apis[{api_name!r}]"
        )
    if getattr(live_api, "_env", None) is not low_level:
        with contextlib.suppress(Exception):
            CapXLiveSession(
                environment,
                low_level,
                live_api,
                {},
                {},
                pyroki_process,
                pyroki_port,
                runtime_workdir,
            ).close()
            pyroki_process = None
            runtime_workdir = None
        raise RuntimeError(
            "CaP-X public API is not bound to the live episode low-level environment"
        )
    if not isinstance(reset_observation, dict) or not isinstance(reset_info, dict):
        with contextlib.suppress(Exception):
            CapXLiveSession(
                environment,
                low_level,
                live_api,
                {},
                {},
                pyroki_process,
                pyroki_port,
                runtime_workdir,
            ).close()
            pyroki_process = None
            runtime_workdir = None
        raise TypeError("CaP-X reset must return observation and info mappings")
    try:
        if api_name in {"FrankaControlApi", "FrankaControlSpillWipeApi", "FrankaLiberoApi"}:
            from capx.integrations.motion.pyroki import init_pyroki
            live_api.ik_solve_fn = init_pyroki(
                server_url=f"http://{CAPX_PYROKI_HOST}:{pyroki_port}"
            )
    except BaseException:
        with contextlib.suppress(Exception):
            CapXLiveSession(
                environment,
                low_level,
                live_api,
                {},
                {},
                pyroki_process,
                pyroki_port,
                runtime_workdir,
            ).close()
            runtime_workdir = None
        raise
    session = CapXLiveSession(
        environment=environment,
        low_level_environment=low_level,
        live_api=live_api,
        reset_observation=reset_observation,
        reset_info=reset_info,
        pyroki_process=pyroki_process,
        pyroki_port=pyroki_port,
        runtime_workdir=runtime_workdir,
        capx_root=capx_root,
        sam3_port=sam3_port,
        contact_graspnet_port=graspnet_port,
    )
    if api_name == "R1ProControlApi":
        return _start_r1pro_perception_sidecars(session)
    return session


def warm_capx_camera_intrinsics(live_api: Any, render: Any) -> dict[str, Any]:
    """Refresh renderer camera data after instance restore, without a task action."""
    import numpy as np
    for count in range(1, 4):
        render()
        matrix = np.asarray(live_api.get_camera_intrinsics())
        if (matrix.shape == (3, 3) and np.isfinite(matrix).all()
                and matrix[0, 0] > 0 and matrix[1, 1] > 0):
            return {"render_calls": count, "fx": float(matrix[0, 0]),
                    "fy": float(matrix[1, 1]), "task_actions_submitted": 0}
    raise RuntimeError("CaPX camera intrinsics unavailable after three render refreshes")


def move_capx_absolute_joint_targets(env, target_joint_positions, max_steps=20,
                                settle_steps=10, *, to_world):
    import torch
    if max_steps < 1 or settle_steps < 1:
        raise ValueError("max_steps and settle_steps must be positive")
    for idx in range(max_steps):
        # q_to_action reads the current base pose. A position-controlled
        # HolonomicBaseJointController consumes a fresh relative command.
        action = env.robot.q_to_action(to_world(env.robot, target_joint_positions))
        env.step(action)
        if idx % int(settle_steps) == 0:
            env._settle_robot()
        if torch.allclose(env.robot.get_joint_positions(), target_joint_positions, atol=0.005):
            return True
    return False




def install_capx_absolute_joint_target_control(session: CapXLiveSession) -> bool:
    """Preserve absolute joint targets on R1Pro's relative-position base.

    Recompute only the command conversion per step. Keep physical stepping,
    native settling, tolerances and caller budgets; never set robot state.
    """
    from types import MethodType
    native = getattr(session.live_api, "_env", None)
    robot = getattr(native, "robot", None)
    controllers = getattr(robot, "controllers", {})
    matches = any(
        any(cls.__name__ == "HolonomicBaseJointController" for cls in type(c).__mro__)
        and getattr(c, "motor_type", None) == "position"
        for c in controllers.values()
    )
    if not matches or not callable(getattr(native, "_move_to_joint_positions", None)):
        return False
    from omnigibson.action_primitives.curobo import holonomic_base_command_in_world_frame
    def move(self, target_joint_positions, max_steps=20, settle_steps=10):
        return move_capx_absolute_joint_targets(self, target_joint_positions,
            max_steps=max_steps, settle_steps=settle_steps,
            to_world=holonomic_base_command_in_world_frame)
    native._move_to_joint_positions = MethodType(move, native)
    session.reset_info["harness_absolute_joint_target_control"] = {
        "relative_base_command_recomputed_each_step": True,
        "physical_steps_and_settling_preserved": True,
        "state_setters_used": False,
    }
    return True


def release_capx_simulator_cache_before_segmentation(session: CapXLiveSession) -> None:
    """Release unused simulator allocator blocks before each SAM3 request.

    Navigation and manipulation can grow the simulator's PyTorch cache after
    sidecar startup. SAM3's transient inference allocation needs that memory
    even when ContactGraspNet has already been unloaded. Live tensors, service
    inputs and model computation are unchanged.
    """
    import functools

    def wrap(original):
        @functools.wraps(original)
        def segment(*args, **kwargs):
            # Covers nested public API calls as well as direct agent actions.
            # A prior grasp plan may have released SAM3 for GPU headroom.
            session.ensure_perception_sidecars(
                require_contact_graspnet=False, require_sam3=True)
            torch_module = sys.modules.get("torch")
            if torch_module is not None and torch_module.cuda.is_available():
                cuda = torch_module.cuda
                before = cuda.memory_reserved()
                allocated = cuda.memory_allocated()
                cuda.empty_cache()
                free, _ = cuda.mem_get_info()
                print("CaPX per-request simulator CUDA cache release: " + json.dumps({
                    "reserved_before": before, "allocated": allocated,
                    "reserved_after": cuda.memory_reserved(), "free_after": free,
                }), file=sys.stderr, flush=True)
            return original(*args, **kwargs)
        return segment

    for name in ("sam3_seg_fn", "sam3_point_prompt_fn"):
        original = getattr(session.live_api, name, None)
        if callable(original):
            setattr(session.live_api, name, wrap(original))


def serialize_capx_grasp_service_memory(session: CapXLiveSession) -> None:
    """Keep inference inputs/weights fixed; serialize owned GPU model residency.

    The current single episode calls these services sequentially. SAM3's mask
    is already on the CPU when grasp_net_plan_fn runs. The next public action
    restarts SAM3 through the existing ensure_perception_sidecars admission.
    ContactGraspNet is loaded only after the mask exists, then released even
    when inference raises; no idle model is left beside the next SAM3 request.
    """
    import functools
    original = session.live_api.grasp_net_plan_fn

    @functools.wraps(original)
    def plan(*args, **kwargs):
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        process = session.sam3_process
        if process is not None:
            _stop_capx_pyroki_process(process)
            session.sam3_process = None
            print("CaPX memory admission: released SAM3 before ContactGraspNet request",
                  file=sys.stderr, flush=True)
        try:
            session.ensure_perception_sidecars(
                require_contact_graspnet=True, require_sam3=False)
            return original(*args, **kwargs)
        finally:
            process = session.contact_graspnet_process
            session.contact_graspnet_process = None
            if process is not None:
                _stop_capx_pyroki_process(process)

    session.live_api.grasp_net_plan_fn = plan
    session.serial_grasp_services = True


def _start_r1pro_perception_sidecars(session: CapXLiveSession) -> CapXLiveSession:
    """Start official SAM3 + ContactGraspNet before any agent turn.

    R1Pro public grasp/perception APIs require these evaluator-owned sidecars.
    Missing weights, a CUDA crash, or a port that never listens must abort the
    episode immediately instead of burning agent turns on Connection refused.
    """

    try:
        import omnigibson as og
        session.reset_info["harness_camera_admission"] = warm_capx_camera_intrinsics(
            session.live_api, og.sim.render)
        # Upstream reset obtains its observation before load_task_instance.
        # Return the refreshed current instance instead of the earlier frame.
        session.reset_observation = session.low_level_environment.get_observation()
        install_capx_absolute_joint_target_control(session)
        # Current W4 protocol uses native navigation; do not register teleport_base.
        session.reset_info["harness_base_assist"] = {
            "enabled": False, "protocol": "native_navigation_only",
        }
        from .capx_checked_manipulation import install_checked_manipulation
        install_checked_manipulation(session)
        from .capx_planner_collision_policy import install_planner_collision_policy
        install_planner_collision_policy(session)
        from .capx_navigation_guard import install_navigation_guard
        install_navigation_guard(session)
        from .capx_carry_navigation import install_carry_navigation
        install_carry_navigation(session)
        release_capx_simulator_cache_before_segmentation(session)
        serialize_capx_grasp_service_memory(session)
        session.reset_info["harness_perception_residency"] = "on_demand_sam3_contact_graspnet_gpu"
        # Validate both services before agent turns, but never retain both.
        contact_preflight = session.ensure_perception_sidecars(
            require_contact_graspnet=True, require_sam3=False)
        session.reset_info["harness_contact_graspnet_startup_validated"] = bool(
            contact_preflight["contact_graspnet"])
        process = session.contact_graspnet_process
        session.contact_graspnet_process = None
        _stop_capx_pyroki_process(process)
        session.ensure_perception_sidecars(require_contact_graspnet=False)
        session.reset_info["harness_sam3_startup_validated"] = True
        # The startup check has passed; no perception model needs to occupy
        # the shared GPU until its first public API request.
        process = session.sam3_process
        session.sam3_process = None
        _stop_capx_pyroki_process(process)
    except Exception as exc:
        # OmniGibson shutdown may terminate the worker before the bridge can
        # serialize this exception. Retain the actual preflight cause first.
        print(f"CaPX perception preflight failed: {type(exc).__name__}: {exc}",
              file=sys.stderr, flush=True)
        with contextlib.suppress(Exception):
            session.close()
        raise RuntimeError(
            "CaP-X R1Pro perception sidecars (SAM3, ContactGraspNet) failed to start; "
            f"episode aborted before agent turns: {type(exc).__name__}: {exc}"
        ) from exc
    return session


def run_reset_smoke(
    capx_root: Path,
    config_path: Path,
    *,
    seed: int | None = 0,
) -> dict[str, Any]:
    """Reset one official CaP-X episode without invoking task evaluation.

    This is intentionally narrower than a rollout: it proves that the pinned
    Robosuite graph, public Franka API, and evaluator-owned PyRoKi sidecar can
    share one live episode. SAM3 and ContactGraspNet are not started because a
    reset does not call perception or grasp generation.
    """

    started_at = time.monotonic()
    session = open_capx_live_session(
        capx_root,
        config_path,
        seed=seed,
    )
    pyroki_process = session.pyroki_process
    runtime_workdir = session.runtime_workdir
    try:
        observation = session.reset_observation
        visual_observations: dict[str, dict[str, Any]] = {}
        for key, value in sorted(observation.items()):
            shape = _capx_array_shape(value)
            if not _is_capx_visual_frame([str(key)], shape):
                continue
            visual_observations[str(key)] = {
                "shape": shape,
                "dtype": str(getattr(value, "dtype", type(value).__name__)),
            }
        apis = getattr(session.environment, "_apis", None)
        result = {
            "schema_version": "agentic-embodied-arena/capx-reset-smoke/v1",
            "ok": True,
            "scope": "official_robosuite_reset_plus_pyroki_health",
            "seed": seed,
            "capx_root": str(capx_root.resolve()),
            "config_path": str(config_path.resolve()),
            "environment_type": type(session.environment).__name__,
            "low_level_type": type(session.low_level_environment).__name__,
            "api_type": type(session.live_api).__name__,
            "configured_api_names": sorted(apis) if isinstance(apis, dict) else [],
            "api_bound_to_same_low_level_state": (
                getattr(session.live_api, "_env", None)
                is session.low_level_environment
            ),
            "observation_keys": sorted(str(key) for key in observation),
            "visual_observations": visual_observations,
            "reset_info_keys": sorted(str(key) for key in session.reset_info),
            "pyroki": {
                "port": session.pyroki_port,
                "running_during_reset": (
                    pyroki_process is not None and pyroki_process.poll() is None
                ),
            },
            "sam3_started": False,
            "contact_graspnet_started": False,
            "official_verifier_invoked": False,
            "elapsed_seconds_before_teardown": round(
                time.monotonic() - started_at, 3
            ),
        }
    finally:
        session.close()
    result["teardown"] = {
        "pyroki_stopped": (
            pyroki_process is None or pyroki_process.poll() is not None
        ),
        "runtime_workdir_removed": (
            runtime_workdir is None or not runtime_workdir.exists()
        ),
    }
    result["elapsed_seconds"] = round(time.monotonic() - started_at, 3)
    return result


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - exercised by benchmark runtimes without PyYAML
        raise ModuleNotFoundError(
            "PyYAML is required only when loading CaP-X YAML configs; install pyyaml in the CaP-X runtime "
            "or run non-CaP-X benchmark adapters without calling CaP-X config primitives."
        ) from exc
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"CaP-X config must be a YAML mapping: {path}")
    return loaded


def resolve_config_path(capx_root: Path | None, config_path: str) -> Path:
    candidate = Path(config_path)
    if candidate.is_absolute():
        return candidate
    if capx_root is not None:
        rooted = capx_root / candidate
        if rooted.exists():
            return rooted
    return candidate


def infer_route(config_path: Path, config: dict[str, Any]) -> str:
    text = f"{config_path.as_posix()} {json.dumps(config, sort_keys=True)}".lower()
    if "libero" in text:
        return "libero"
    if "r1pro" in text or "b1k" in text or "behavior" in text:
        return "behavior"
    if "robosuite" in text or "franka" in text:
        return "robosuite"
    return "unknown"


def _safe_signature(node: ast.FunctionDef) -> str:
    args = [arg.arg for arg in node.args.args if arg.arg != "self"]
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    args.extend(arg.arg for arg in node.args.kwonlyargs)
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")
    return f"{node.name}({', '.join(args)})"


def _extract_class_schema(source_path: Path, class_name: str) -> dict[str, Any]:
    try:
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError) as exc:
        return {"class": class_name, "source": str(source_path), "error": str(exc), "functions": {}}

    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            functions = {}
            for item in node.body:
                if not isinstance(item, ast.FunctionDef):
                    continue
                if item.name.startswith("_") or item.name in {"functions"}:
                    continue
                functions[item.name] = {
                    "signature": _safe_signature(item),
                    "doc": ast.get_docstring(item) or "",
                }
            return {
                "class": class_name,
                "source": str(source_path),
                "class_doc": ast.get_docstring(node) or "",
                "functions": functions,
            }
    return {"class": class_name, "source": str(source_path), "error": "class_not_found", "functions": {}}


def extract_action_api_schema(capx_root: Path | None, configured_apis: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {"comparator_aliases": API_ALIASES, "configured_api_classes": {}}
    if capx_root is None:
        return schema
    integrations = capx_root / "capx" / "integrations"
    if not integrations.exists():
        schema["upstream_schema_error"] = f"missing integrations directory: {integrations}"
        return schema

    for api_class in configured_apis:
        matches: list[Path] = []
        for path in integrations.rglob("*.py"):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):
                continue
            if any(isinstance(node, ast.ClassDef) and node.name == api_class for node in tree.body):
                matches.append(path)
        if matches:
            schema["configured_api_classes"][api_class] = _extract_class_schema(matches[0], api_class)
        else:
            schema["configured_api_classes"][api_class] = {"class": api_class, "error": "class_not_found", "functions": {}}
    return schema


def run_import_preflight(capx_root: Path, config_path: Path) -> dict[str, Any]:
    timeout_seconds = _capx_import_preflight_timeout_seconds()
    script = (
        "import json\n"
        "from pathlib import Path\n"
        "from capx.utils.launch_utils import _load_config\n"
        "from capx.envs.launch import LaunchArgs\n"
        f"args = LaunchArgs(config_path={str(config_path)!r}, total_trials=1, num_workers=1, record_video=False)\n"
        "env_factory, config, api_servers = _load_config(args)\n"
        "preflight = {\n"
        "    'config_keys': sorted(config.keys()),\n"
        "    'api_servers': len(api_servers or []),\n"
        "    'env_factory_type': type(env_factory).__name__,\n"
        "    'env_factory_mapping': isinstance(env_factory, dict),\n"
        "    'env_factory_target': env_factory.get('_target_') if isinstance(env_factory, dict) else None,\n"
        "}\n"
        "print('CAPX_PREFLIGHT_JSON=' + json.dumps(preflight, sort_keys=True))\n"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{capx_root}:{env.get('PYTHONPATH', '')}"
    command = [sys.executable, "-c", "from capx.utils.launch_utils import _load_config; ..."]
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=capx_root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "blocker": "timeout",
            "timeout_seconds": exc.timeout,
            "stdout_tail": output[-4000:],
        }
    parsed = _parse_preflight_json(completed.stdout)
    ok = (
        completed.returncode == 0
        and isinstance(parsed, dict)
        and parsed.get("env_factory_mapping") is True
        and isinstance(parsed.get("env_factory_target"), str)
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "ok": ok,
        "parsed": parsed,
        "stdout_tail": completed.stdout[-4000:],
    }


def _parse_preflight_json(output: str) -> dict[str, Any] | None:
    return _parse_marker_json(output, "CAPX_PREFLIGHT_JSON=")


def parse_api_registration_preflight(output: str) -> dict[str, Any] | None:
    return _parse_marker_json(output, "CAPX_API_REGISTRATION_PREFLIGHT=")


def _parse_marker_json(output: str, marker: str) -> dict[str, Any] | None:
    for line in reversed(output.splitlines()):
        if line.startswith(marker):
            loaded = json.loads(line[len(marker) :])
            return loaded if isinstance(loaded, dict) else None
    return None


def refresh_capx_live_service_readiness(context: CapXRuntimeContext, session: CapXLiveSession) -> dict[str, Any]:
    """Replace pre-launch transport diagnostics with the live episode's ports."""
    ports = {"sam3": session.sam3_port, "contact_graspnet": session.contact_graspnet_port}
    effective_servers = []
    for server in context.api_servers:
        effective = dict(server)
        service = API_SERVER_NAMES.get(str(server.get("_target_")))
        if ports.get(service):
            effective.update(host=CAPX_PYROKI_HOST, port=ports[service])
        effective_servers.append(effective)
    deferred_contact = bool(
        getattr(session, "serial_grasp_services", False)
        and getattr(session, "contact_graspnet_process", None) is None
        and getattr(session, "reset_info", {}).get("harness_contact_graspnet_startup_validated"))
    deferred_sam3 = bool(
        getattr(session, "serial_grasp_services", False)
        and getattr(session, "sam3_process", None) is None
        and getattr(session, "reset_info", {}).get("harness_sam3_startup_validated"))
    deferred = {
        name for name, enabled in (("sam3", deferred_sam3),
                                   ("contact_graspnet", deferred_contact))
        if enabled
    }
    resident_servers = [server for server in effective_servers
                        if API_SERVER_NAMES.get(str(server.get("_target_"))) not in deferred]
    readiness = inspect_capx_api_server_readiness(resident_servers)
    if deferred:
        readiness["sidecar_count"] = len(effective_servers)
        readiness["residency_mode"] = "sequential_on_demand"
        readiness["deferred_services"] = sorted(deferred)
        for name in sorted(deferred):
            readiness.setdefault("checks", []).append({
                "name": name, "host": CAPX_PYROKI_HOST,
                "port": ports[name],
                "reachable": False, "check_attempted": False,
                "startup_validated": True, "state": "validated_then_unloaded",
                "required_on_demand": True,
            })
    readiness["checked_phase"] = "after_live_session_start"
    effective_ports = [server.get("port") for server in effective_servers]
    context.route_capabilities.update(api_server_readiness=readiness,
                                     api_server_blockers=readiness["blockers"],
                                     api_server_ports=effective_ports)
    context.trace_evidence.update(api_server_readiness=readiness,
                                  api_server_ports=effective_ports)
    return readiness


def inspect_capx_api_server_readiness(api_servers: list[Any], *, timeout_s: float = 0.2) -> dict[str, Any]:
    """Probe only localhost transport availability for configured CaP-X sidecars."""

    checks: list[dict[str, Any]] = []
    blockers: list[str] = []
    for index, server in enumerate(api_servers):
        if not isinstance(server, dict):
            blockers.append(f"capx_api_server_config_invalid:{index}")
            continue
        target = str(server.get("_target_") or "")
        service = API_SERVER_NAMES.get(target, "unknown")
        host = str(server.get("host") or "127.0.0.1")
        raw_port = server.get("port")
        try:
            port = int(raw_port)
        except (TypeError, ValueError):
            port = None
        check: dict[str, Any] = {
            "index": index,
            "name": service,
            "target": target,
            "host": host,
            "port": port,
            "device": server.get("device"),
            "timeout_s": timeout_s,
            "check_attempted": False,
            "reachable": False,
            "transport_only_check": True,
            "http_probe_performed": False,
            "model_or_segmentation_call_performed": False,
            "error_type": None,
            "error_errno": None,
            "error": None,
            "blocker": None,
        }
        if port is None:
            check["blocker"] = f"capx_api_server_port_missing_or_invalid:{service}:{index}"
            blockers.append(str(check["blocker"]))
            checks.append(check)
            continue
        check["check_attempted"] = True
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                pass
        except OSError as exc:
            check["error_type"] = type(exc).__name__
            check["error_errno"] = getattr(exc, "errno", None)
            check["error"] = str(exc)
            errno_slug = f"errno_{exc.errno}" if getattr(exc, "errno", None) is not None else type(exc).__name__
            check["blocker"] = f"capx_api_server_unreachable:{service}:{host}:{port}:{errno_slug}"
            blockers.append(str(check["blocker"]))
            checks.append(check)
            continue
        check["reachable"] = True
        checks.append(check)
    return {
        "ready": not blockers,
        "sidecar_count": len(checks),
        "checks": checks,
        "blockers": blockers,
    }


def build_context(
    capx_root: Path | None,
    config_path: Path,
    mode: str,
    *,
    prompt: str | None = None,
    query: str | None = None,
    agent_context: dict[str, Any] | None = None,
    include_preflight: bool = False,
) -> CapXRuntimeContext:
    config = load_config(config_path)
    env = config.get("env") or {}
    cfg = env.get("cfg") or {}
    if not isinstance(env, dict) or not isinstance(cfg, dict):
        raise ValueError("Expected CaP-X config keys env and env.cfg to be mappings")

    apis = cfg.get("apis") or []
    if not isinstance(apis, list) or not all(isinstance(item, str) for item in apis):
        raise ValueError("Expected env.cfg.apis to be a list of API names")

    config_prompt = cfg.get("prompt")
    resolved_prompt = prompt if prompt is not None else config_prompt
    prompt_source = "request.prompt" if prompt is not None else ("env.cfg.prompt" if config_prompt else "upstream task class default")
    api_servers = config.get("api_servers") or []
    if not isinstance(api_servers, list):
        raise ValueError("Expected api_servers to be a list when present")
    api_server_readiness = inspect_capx_api_server_readiness(api_servers)

    route = infer_route(config_path, config)
    action_api_schema = extract_action_api_schema(capx_root, apis)
    callable_aliases = dict(API_ALIASES)
    route_capabilities = {
        "route": route,
        "low_level": cfg.get("low_level"),
        "configured_api_classes": apis,
        "api_server_targets": [server.get("_target_") for server in api_servers if isinstance(server, dict)],
        "api_server_ports": [server.get("port") for server in api_servers if isinstance(server, dict)],
        "api_server_readiness": api_server_readiness,
        "api_server_blockers": api_server_readiness["blockers"],
        "covered_route_only": route in {"robosuite", "libero", "behavior"},
    }
    public_agent_context = {
        "system_prompt": "You are a coding agent controlling CaP-X by composing documented robot APIs.",
        "task_prompt": resolved_prompt if isinstance(resolved_prompt, str) else None,
        "query": query,
        "agent_context": agent_context or {},
        "task_prompt_note": "When prompt is null, CaP-X constructs it from the task class default plus API docstrings at runtime.",
        "configured_apis": apis,
        "callable_comparator_apis": callable_aliases,
        "direct_visual_primitive_surface": False,
        "modality_boundary": {
            "mode": "structured_perception_only_until_live_observation_is_inspected",
            "raw_visual_frames_available": False,
            "images_fabricated": False,
        },
        "visual_grounding_boundary": (
            "The CaP-X config/API trace does not directly expose RGB/depth/segmentation frames and is structured "
            "perception only. observe_capx_scene forwards raw visual "
            "frames generically when the injected upstream live observation actually contains them; otherwise "
            "the modality remains explicitly structured-perception-only."
        ),
        "forbidden_shortcut": "Do not expose a harness-level execute_policy_code completion primitive.",
        "evaluation_boundary": "Task evaluation remains outside the coding-agent primitive surface.",
        "private_evaluator_exposed": False,
    }
    display_config_path = str(config_path)
    if capx_root is not None:
        try:
            display_config_path = str(config_path.relative_to(capx_root))
        except ValueError:
            pass

    trace_evidence: dict[str, Any] = {
        "config_exists": config_path.exists(),
        "config_path": str(config_path),
        "env_target": env.get("_target_"),
        "low_level": cfg.get("low_level"),
        "configured_api_count": len(apis),
        "api_server_ports": route_capabilities["api_server_ports"],
        "api_server_readiness": api_server_readiness,
        "upstream_entrypoint": "capx/envs/launch.py",
        "live_command_template": (
            "uv run --no-sync --active capx/envs/launch.py --config-path "
            f"{display_config_path}"
        ),
    }
    if capx_root is not None:
        trace_evidence["capx_root"] = str(capx_root)
        trace_evidence["launch_entrypoint_exists"] = (capx_root / "capx" / "envs" / "launch.py").exists()
        trace_evidence["upstream_git_head"] = _capx_git_head(capx_root)
    if include_preflight and capx_root is not None:
        trace_evidence["import_preflight"] = run_import_preflight(capx_root, config_path)

    return CapXRuntimeContext(
        mode=mode,
        capx_root=str(capx_root) if capx_root else None,
        config_path=str(config_path),
        route=route,
        env_target=env.get("_target_"),
        low_level=cfg.get("low_level"),
        configured_apis=apis,
        prompt_source=prompt_source,
        prompt=resolved_prompt if isinstance(resolved_prompt, str) else None,
        api_servers=[server for server in api_servers if isinstance(server, dict)],
        agent_context=public_agent_context,
        trace_evidence=trace_evidence,
        route_capabilities=route_capabilities,
        action_api_schema=action_api_schema,
    )


def build_api_trace_primitive(
    *,
    capx_root: Path | None,
    config_path: Path,
    prompt: str | None = None,
    query: str | None = None,
    agent_context: dict[str, Any] | None = None,
    include_preflight: bool = False,
) -> dict[str, Any]:
    context = build_context(
        capx_root,
        config_path,
        "primitive",
        prompt=prompt,
        query=query,
        agent_context=agent_context,
        include_preflight=include_preflight,
    )
    payload = asdict(context)
    public_text = json.dumps(payload, sort_keys=True).lower()
    payload["private_evaluator_guard"] = {
        "blocked_private_terms_present": [term for term in FORBIDDEN_TRACE_TERMS if term in public_text],
        "private_evaluator_exposed": False,
    }
    return payload


def list_capx_comparator_primitives(context: CapXRuntimeContext | dict[str, Any]) -> list[PrimitiveCard]:
    """Return the public CaP-X API primitives visible to a coding agent.

    The comparator can execute these only when the caller injects a live CaP-X
    API object from inside an upstream runtime session. Otherwise calls fail at
    the real boundary with `requires_live_api_session`.
    """

    configured = _context_field(context, "configured_apis", [])
    route = _context_field(context, "route", "unknown")
    common = {
        "configured_apis": configured,
        "route": route,
        "requires_live_api_session": "bool",
        "agent_context": "dict|None",
    }
    cards = [
        _capx_card(
            "get_capx_api_trace",
            "L1",
            {"query": "str|None", "agent_context": "dict|None"},
            {"trace_evidence": "dict", "route_capabilities": "dict", "action_api_schema": "dict", **common},
            "Return the CaP-X API/config trace visible to a coding agent without running live trials.",
        ),
        _capx_card(
            "get_capx_available_apis",
            "L1",
            {"query": "str|None", "agent_context": "dict|None"},
            {"configured_apis": "list[str]", "callable_apis": "dict", "action_api_schema": "dict", **common},
            "Expose CaP-X public API docs and route capabilities without evaluator internals.",
        ),
        _capx_card(
            "get_capx_prompt",
            "L1",
            {"agent_context": "dict|None"},
            {"prompt": "str|None", "prompt_source": "str", **common},
            "Expose the task prompt source that a coding agent should condition on.",
        ),
        _capx_card(
            "build_capx_live_command_spec",
            "L1",
            {"model": "str|None", "server_url": "str|None", "total_trials": "int", "num_workers": "int", "agent_context": "dict|None"},
            {"command_spec": "dict", "execution_boundary": "dict", **common},
            "Generate the upstream CaP-X runner command without executing it or exposing secrets.",
        ),
        _capx_card(
            "observe_capx_scene",
            "L1",
            {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"},
            {
                "observation": "dict",
                "observation_keys": "list[str]",
                "modality_boundary": "dict",
                "visual_frames": "list[dict]",
                "visual_evidence_refs": "list[dict]",
                **common,
            },
            "Read the current public CaP-X observation and forward only genuinely present raw visual frames without exposing evaluator state.",
        ),
        _capx_card(
            "enumerate_capx_objects",
            "L2",
            {"agent_context": "dict|None"},
            {"objects": "list[dict]", **common},
            "Enumerate pose-shaped records from the public observation for caller-side selection.",
        ),
        _capx_card(
            "compose_capx_geometry",
            "L2",
            {
                "anchor_position": "list[float]",
                "direction": "list[float]",
                "distance": "float",
                "source_extent": "list[float]|None",
                "target_extent": "list[float]|None",
                "source_extent_scale": "float",
                "target_extent_scale": "float",
                "quaternion_wxyz": "list[float]",
                "agent_context": "dict|None",
            },
            {"geometry": "dict", **common},
            "Compose one pose from caller-selected geometry; no waypoint, task recipe, or action is inferred.",
        ),
        _capx_card(
            "submit_capx_action",
            "L3",
            {"arm": "str", "action": "str", "parameters": "dict", "visual_evidence_refs": "list[dict]|None", "agent_context": "dict|None"},
            {"executed": "bool", "selected_arm": "str", "selected_action": "str", "provenance": "dict", **common},
            "Dispatch exactly one caller-selected public arm action without planning a sequence.",
        ),
        _capx_card(
            "capx_get_object_pose",
            "L2",
            {"object_name": "str", "return_bbox_extent": "bool", "query": "str|None", "agent_context": "dict|None"},
            {"pose": "dict|list", **common},
            "Call public CaP-X get_object_pose on a live API object, or fail with a live-session boundary.",
        ),
        _capx_card(
            "capx_sample_grasp_pose",
            "L2",
            {"object_name": "str", "query": "str|None", "agent_context": "dict|None"},
            {"grasp_pose": "dict|list", **common},
            "Call public CaP-X sample_grasp_pose for agent-side grasp planning.",
        ),
        _capx_card(
            "capx_goto_pose",
            "L3",
            {"position": "list[float]", "quaternion_wxyz": "list[float]", "z_approach": "float", "visual_evidence_refs": "list[dict]|None", "agent_context": "dict|None"},
            {"executed": "bool", "provenance": "dict", **common},
            "Execute public CaP-X goto_pose on a live robot/control API object.",
        ),
        _capx_card(
            "capx_open_gripper",
            "L3",
            {"agent_context": "dict|None"},
            {"executed": "bool", **common},
            "Execute public CaP-X open_gripper on a live robot/control API object.",
        ),
        _capx_card(
            "capx_close_gripper",
            "L3",
            {"agent_context": "dict|None"},
            {"executed": "bool", **common},
            "Execute public CaP-X close_gripper on a live robot/control API object.",
        ),
        _capx_card(
            "record_capx_evidence",
            "L1",
            {"key": "str", "value": "any", "agent_context": "dict|None"},
            {"artifact_id": "str", **common},
            "Record agent-selected CaP-X API/pose/action evidence for the comparator trace.",
        ),
    ]
    return cards


def call_capx_comparator_primitive(
    context: CapXRuntimeContext | dict[str, Any],
    name: str,
    *,
    live_api: Any | None = None,
    live_env: Any | None = None,
    **kwargs: Any,
) -> PrimitiveResult:
    allowed = {card.name for card in list_capx_comparator_primitives(context)}
    if name not in allowed:
        return PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed by the CaP-X comparator")
    if name == "get_capx_api_trace":
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                **_capx_public_context(context, kwargs.get("agent_context"), kwargs.get("query")),
                "prompt": _context_field(context, "prompt"),
                "prompt_source": _context_field(context, "prompt_source", "unknown"),
                "trace_evidence": _context_field(context, "trace_evidence", {}),
            },
        )
    if name == "get_capx_available_apis":
        return PrimitiveResult(name=name, ok=True, output=_capx_public_context(context, kwargs.get("agent_context"), kwargs.get("query")))
    if name == "get_capx_prompt":
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                **_capx_public_context(context, kwargs.get("agent_context")),
                "prompt": _context_field(context, "prompt"),
                "prompt_source": _context_field(context, "prompt_source", "unknown"),
            },
        )
    if name == "build_capx_live_command_spec":
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                **_capx_public_context(context, kwargs.get("agent_context")),
                "command_spec": _build_capx_command_spec(
                    context,
                    model=kwargs.get("model"),
                    server_url=kwargs.get("server_url"),
                    total_trials=int(kwargs.get("total_trials", 1)),
                    num_workers=int(kwargs.get("num_workers", 1)),
                ),
                "execution_boundary": _capx_execution_boundary(context),
            },
        )
    if name == "record_capx_evidence":
        key = str(kwargs.get("key", "unnamed"))
        return PrimitiveResult(
            name=name,
            ok=True,
            output={
                **_capx_public_context(context, kwargs.get("agent_context")),
                "artifact_id": f"capx:evidence:{key}",
                "key": key,
                "value": _jsonable(kwargs.get("value")),
            },
            artifacts=[f"capx:evidence:{key}"],
        )
    if name == "compose_capx_geometry":
        try:
            geometry = _compose_capx_geometry(kwargs)
        except (KeyError, TypeError, ValueError) as exc:
            return PrimitiveResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        return PrimitiveResult(
            name=name,
            ok=True,
            output={**_capx_public_context(context, kwargs.get("agent_context")), "geometry": geometry},
        )
    observation_env = live_env or getattr(live_api, "_env", None)
    if name in {"observe_capx_scene", "enumerate_capx_objects"}:
        if observation_env is None:
            return _capx_requires_live_session(name, context, kwargs)
        try:
            observation = observation_env.get_observation()
            result = (
                _public_capx_observation(observation)
                if name == "observe_capx_scene"
                else _enumerate_capx_objects(observation)
            )
        except Exception as exc:
            return PrimitiveResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}")
        output_key = "observation" if name == "observe_capx_scene" else "objects"
        output = {
            **_capx_public_context(context, kwargs.get("agent_context"), kwargs.get("query")),
            output_key: result,
        }
        if name == "observe_capx_scene":
            modality = _capx_observation_modalities(observation)
            output["observation_keys"] = sorted(str(key) for key in observation)
            output["prompt"] = kwargs.get("prompt") if kwargs.get("prompt") is not None else _context_field(context, "prompt")
            output["prompt_source"] = "request.prompt" if kwargs.get("prompt") is not None else _context_field(context, "prompt_source", "unknown")
            output["modality_boundary"] = modality["modality_boundary"]
            output["visual_frames"] = modality["visual_frames"]
            output["visual_evidence_refs"] = modality["visual_evidence_refs"]
            output["direct_visual_primitive_surface"] = bool(modality["visual_frames"])
        return PrimitiveResult(name=name, ok=True, output=output)
    if live_api is None:
        return _capx_requires_live_session(name, context, kwargs)
    try:
        action_provenance = (
            _capx_visual_action_provenance(kwargs.get("visual_evidence_refs"), observation_env)
            if name in {"submit_capx_action", "capx_goto_pose"}
            else None
        )
        if name == "submit_capx_action":
            result = _submit_capx_action(live_api, kwargs)
        else:
            result = _call_capx_direct_api(live_api, name, kwargs)
    except Exception as exc:
        return PrimitiveResult(
            name=name,
            ok=False,
            output=_capx_public_context(context, kwargs.get("agent_context")),
            error=f"{type(exc).__name__}: {exc}",
        )
    return PrimitiveResult(
        name=name,
        ok=True,
        output={
            **_capx_public_context(context, kwargs.get("agent_context")),
            "requires_live_api_session": False,
            "executed": name.startswith("capx_") and name not in {"capx_get_object_pose", "capx_sample_grasp_pose"},
            "result": _jsonable(result),
            **(
                {"selected_arm": str(kwargs.get("arm")), "selected_action": str(kwargs.get("action"))}
                if name == "submit_capx_action"
                else {}
            ),
            **({"provenance": action_provenance} if action_provenance is not None else {}),
        },
    )


def verify_capx_live_result(live_result: dict[str, Any] | None) -> VerificationResult:
    if not live_result:
        return VerificationResult(ok=False, scope="capx_live", message="No CaP-X live_result is available.")
    summary = live_result.get("summary") or {}
    ok = bool(live_result.get("returncode") == 0 and summary.get("summary_found"))
    return VerificationResult(
        ok=ok,
        scope="capx_live",
        message="CaP-X live run completed and emitted an aggregate summary" if ok else "CaP-X live run did not complete cleanly",
        metrics={
            "returncode": live_result.get("returncode"),
            "summary_found": float(bool(summary.get("summary_found"))),
            "success_rate": summary.get("success_rate"),
            "avg_reward": summary.get("avg_reward"),
            "completed": summary.get("completed"),
        },
        metadata={"command": live_result.get("command"), "stdout_tail": live_result.get("stdout_tail", "")[-2000:]},
    )


def _capx_card(name: str, level: str, input_schema: dict[str, Any], output_schema: dict[str, Any], description: str) -> PrimitiveCard:
    return PrimitiveCard(
        name=name,
        capability_tags=["capx", "comparator", "agent_safe", "public_api"],
        input_schema=input_schema,
        output_schema=output_schema,
        preconditions=["CaP-X context built", "live API object injected for execution primitives"],
        side_effects=["robot/control state may change when live_api is provided"] if level == "L3" else [],
        cost={"sim_steps": "upstream-defined"},
        failure_modes=["requires_live_api_session", "missing_public_api", "upstream_runtime_error"],
        abstraction_level=level,
        leakage_risk="low",
        description=description,
    )


def _context_field(context: CapXRuntimeContext | dict[str, Any], field: str, default: Any = None) -> Any:
    if isinstance(context, dict):
        return context.get(field, default)
    return getattr(context, field, default)


def _capx_public_context(
    context: CapXRuntimeContext | dict[str, Any],
    agent_context: dict[str, Any] | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    if isinstance(context, CapXRuntimeContext):
        action_api_schema = context.action_api_schema
        route_capabilities = context.route_capabilities
        callable_apis = context.agent_context.get("callable_comparator_apis", API_ALIASES)
    else:
        action_api_schema = context.get("action_api_schema", {})
        route_capabilities = context.get("route_capabilities", {})
        callable_apis = (context.get("agent_context") or {}).get("callable_comparator_apis", API_ALIASES)
    return {
        "route": _context_field(context, "route", "unknown"),
        "configured_apis": _context_field(context, "configured_apis", []),
        "callable_apis": callable_apis,
        "action_api_schema": action_api_schema,
        "route_capabilities": route_capabilities,
        "direct_visual_primitive_surface": False,
        "modality_boundary": {
            "mode": "structured_perception_only",
            "raw_visual_frames_available": False,
            "images_fabricated": False,
        },
        "visual_grounding_boundary": (
            "No direct RGB/depth/segmentation frame has been observed at this boundary. CaP-X API/config context is structured "
            "perception only; observe_capx_scene exposes frames only when an injected live environment supplies them."
        ),
        "query": query,
        "agent_context": agent_context or {},
        "private_evaluator_exposed": False,
    }


def _capx_requires_live_session(name: str, context: CapXRuntimeContext | dict[str, Any], kwargs: dict[str, Any]) -> PrimitiveResult:
    return PrimitiveResult(
        name=name,
        ok=False,
        output={
            **_capx_public_context(context, kwargs.get("agent_context"), kwargs.get("query")),
            "requires_live_api_session": True,
            "requested_arguments": _jsonable(kwargs),
            "blocker": "requires_live_api_session",
            "execution_boundary": _capx_execution_boundary(context),
            "command_spec": _build_capx_command_spec(context),
            "live_session_note": "Inject the public CaP-X API object from inside an upstream runtime session to execute this primitive.",
        },
        error="requires_live_api_session",
    )


def _capx_execution_boundary(context: CapXRuntimeContext | dict[str, Any]) -> dict[str, Any]:
    route_capabilities = _context_field(context, "route_capabilities", {}) or {}
    api_server_readiness = route_capabilities.get("api_server_readiness") or {}
    api_server_blockers = list(api_server_readiness.get("blockers") or route_capabilities.get("api_server_blockers") or [])
    return {
        "actual_execution_requires": "upstream_capx_runner_or_injected_live_api",
        "blocker": "requires_upstream_runner",
        "api_server_blockers": api_server_blockers,
        "api_server_readiness": api_server_readiness,
        "sidecar_services_required": bool(api_server_readiness.get("sidecar_count", 0)),
        "can_execute_inside_facade_without_live_api": False,
        "evaluation_observation": "available_after_rollout_to_the_outer_harness_only",
        "direct_visual_primitive_surface": False,
        "runner_command_spec_available": True,
        "config_path": _context_field(context, "config_path"),
        "capx_root": _context_field(context, "capx_root"),
    }


def _build_capx_command_spec(
    context: CapXRuntimeContext | dict[str, Any],
    *,
    model: str | None = None,
    server_url: str | None = None,
    total_trials: int = 1,
    num_workers: int = 1,
) -> dict[str, Any]:
    capx_root = _context_field(context, "capx_root")
    config_path = _context_field(context, "config_path")
    config_arg = config_path
    if capx_root and config_path:
        try:
            config_arg = str(Path(config_path).resolve().relative_to(Path(capx_root).resolve()))
        except (OSError, ValueError):
            config_arg = config_path
    command = [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        str(config_arg),
        "--total-trials",
        str(total_trials),
        "--num-workers",
        str(num_workers),
    ]
    if model:
        command.extend(["--model", model])
    if server_url:
        command.extend(["--server-url", server_url])
    return {
        "command": command,
        "cwd": capx_root,
        "env": {"PYTHONPATH_prefix": capx_root},
        "prints_secrets": False,
        "executes_now": False,
        "requires": ["upstream CaP-X checkout", "configured simulator assets/services", "LLM endpoint only for generated-code live mode"],
        "summary_parser": "parse_capx_summary(stdout)",
    }


def _build_capx_pick_place_plan(kwargs: dict[str, Any]) -> dict[str, Any]:
    source = str(kwargs["source_object"])
    target_object = kwargs.get("target_object")
    target_position = kwargs.get("target_position")
    target_quaternion = kwargs.get("target_quaternion_wxyz") or [1.0, 0.0, 0.0, 0.0]
    clearance = float(kwargs.get("clearance", 0.12))
    required_inputs = []
    if target_position is None and not target_object:
        required_inputs.append("target_object or target_position")

    place_expr = "target_pose['position']" if target_object and target_position is None else repr(_jsonable(target_position))
    quat_expr = "target_pose.get('quaternion_wxyz', [1.0, 0.0, 0.0, 0.0])" if target_object and target_position is None else repr(_jsonable(target_quaternion))
    lines = [
        f"source_pose = get_object_pose({source!r}, return_bbox_extent=True)",
        f"grasp_pose = sample_grasp_pose({source!r})",
        "goto_pose(grasp_pose['position'], grasp_pose['quaternion_wxyz'], z_approach=0.0)",
        "close_gripper()",
        f"lift_position = [grasp_pose['position'][0], grasp_pose['position'][1], grasp_pose['position'][2] + {clearance!r}]",
        "goto_pose(lift_position, grasp_pose['quaternion_wxyz'], z_approach=0.0)",
    ]
    if target_object and target_position is None:
        lines.append(f"target_pose = get_object_pose({str(target_object)!r}, return_bbox_extent=True)")
    lines.extend(
        [
            f"place_position = {place_expr}",
            f"place_quaternion = {quat_expr}",
            f"approach_position = [place_position[0], place_position[1], place_position[2] + {clearance!r}]",
            "goto_pose(approach_position, place_quaternion, z_approach=0.0)",
            "goto_pose(place_position, place_quaternion, z_approach=0.0)",
            "open_gripper()",
        ]
    )
    action_plan = [
        {"step": "query_source_pose", "api": "get_object_pose", "object_name": source},
        {"step": "sample_grasp", "api": "sample_grasp_pose", "object_name": source},
        {"step": "move_to_grasp", "api": "goto_pose", "frame": "world"},
        {"step": "close_gripper", "api": "close_gripper"},
        {"step": "world_z_lift", "api": "goto_pose", "clearance": clearance},
        {"step": "resolve_place_pose", "api": "get_object_pose" if target_object and target_position is None else "agent_supplied_pose", "target_object": target_object},
        {"step": "world_z_approach_place", "api": "goto_pose", "clearance": clearance},
        {"step": "place", "api": "goto_pose"},
        {"step": "open_gripper", "api": "open_gripper"},
    ]
    return {
        "action_plan": action_plan,
        "python_code": "\n".join(lines),
        "required_inputs_remaining": required_inputs,
        "provenance": {
            "constructed_from": "public CaP-X action API aliases",
            "direct_visual_primitive_surface": False,
            "not_task_success_claim": True,
        },
    }


def _capx_object_query(name: str) -> str:
    return name.replace("_", " ")


def _numeric_vector(value: Any, length: int, field: str) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} numeric values")
    if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
        raise ValueError(f"{field} must contain exactly {length} numeric values")
    return [float(item) for item in value]


def _public_capx_observation(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise TypeError("public CaP-X observation must be a mapping")
    excluded_fragments = ("reward", "success", "completed", "terminated", "checker", "oracle")
    public: dict[str, Any] = {}
    for key, value in observation.items():
        key_text = str(key)
        if any(fragment in key_text.lower() for fragment in excluded_fragments):
            continue
        if isinstance(value, dict):
            public[key_text] = _public_capx_observation(value)
            continue
        shape = getattr(value, "shape", None)
        if shape is not None and len(shape) >= 2 and math.prod(int(item) for item in shape) > 64:
            public[key_text] = {"type": type(value).__name__, "shape": [int(item) for item in shape]}
            continue
        public[key_text] = _jsonable(value)
    return public


def _capx_observation_modalities(observation: Any) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise TypeError("public CaP-X observation must be a mapping")
    visual_frames: list[dict[str, Any]] = []
    evidence_refs: list[dict[str, Any]] = []
    excluded_fragments = ("reward", "success", "completed", "terminated", "checker", "oracle")

    def visit(value: Any, path: list[str]) -> None:
        if any(fragment in part.lower() for part in path for fragment in excluded_fragments):
            return
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, [*path, str(key)])
            return
        shape = _capx_array_shape(value)
        if not _is_capx_visual_frame(path, shape):
            return
        source_path = ".".join(path)
        modality = _capx_visual_modality(path)
        frame = {
            "source_path": source_path,
            "modality": modality,
            "shape": shape,
            "dtype": str(getattr(value, "dtype", type(value).__name__)),
            "data": _jsonable(value),
        }
        visual_frames.append(frame)
        evidence_refs.append(
            {
                "kind": "capx_live_visual_frame",
                "source_path": source_path,
                "modality": modality,
                "shape": shape,
            }
        )

    visit(observation, [])
    has_frames = bool(visual_frames)
    return {
        "modality_boundary": {
            "mode": "raw_visual_frames_and_structured_perception" if has_frames else "structured_perception_only",
            "raw_visual_frames_available": has_frames,
            "visual_frame_count": len(visual_frames),
            "images_fabricated": False,
            "source": "injected_upstream_live_observation",
        },
        "visual_frames": visual_frames,
        "visual_evidence_refs": evidence_refs,
    }


def _capx_visual_action_provenance(evidence_refs: Any, observation_env: Any | None) -> dict[str, Any]:
    refs = _normalize_capx_visual_evidence_refs(evidence_refs)
    available_refs: list[dict[str, Any]] = []
    validation = "no_visual_evidence_refs"
    if refs and observation_env is not None and callable(getattr(observation_env, "get_observation", None)):
        try:
            available_refs = _capx_observation_modalities(observation_env.get_observation())["visual_evidence_refs"]
            validation = "matched_live_observation"
        except Exception as exc:
            validation = f"live_observation_validation_failed:{type(exc).__name__}"
    elif refs:
        validation = "unverified_without_live_observation"
    available_keys = {
        (ref.get("source_path"), ref.get("modality"), tuple(ref.get("shape") or [])) for ref in available_refs
    }
    matched = [
        (ref.get("source_path"), ref.get("modality"), tuple(ref.get("shape") or [])) in available_keys
        for ref in refs
    ]
    return {
        "visual_evidence_refs": refs,
        "visual_evidence_ref_count": len(refs),
        "visual_evidence_validation": validation,
        "real_live_frame_refs_preserved": bool(refs) and bool(matched) and all(matched),
        "images_fabricated": False,
        "task_recipe_embedded": False,
    }


def _normalize_capx_visual_evidence_refs(evidence_refs: Any) -> list[dict[str, Any]]:
    if evidence_refs is None:
        return []
    if not isinstance(evidence_refs, list):
        raise TypeError("visual_evidence_refs must be a list of mappings")
    normalized: list[dict[str, Any]] = []
    for index, ref in enumerate(evidence_refs):
        if not isinstance(ref, dict) or not ref.get("source_path"):
            raise ValueError(f"visual_evidence_refs[{index}] must reference a real source_path")
        normalized.append({str(key): _jsonable(value) for key, value in ref.items()})
    return normalized


def _capx_array_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return [int(item) for item in shape]
        except (TypeError, ValueError):
            return None
    dimensions: list[int] = []
    current = value
    while isinstance(current, (list, tuple)):
        dimensions.append(len(current))
        if not current:
            break
        current = current[0]
    return dimensions or None


def _is_capx_visual_frame(path: list[str], shape: list[int] | None) -> bool:
    if shape is None or len(shape) < 2 or any(dimension <= 0 for dimension in shape):
        return False
    path_text = ".".join(path).lower()
    visual_name = any(token in path_text for token in ("rgb", "color", "image", "camera", "frame", "depth", "segment", "mask"))
    channel_image = len(shape) == 3 and (shape[-1] in {1, 3, 4} or shape[0] in {1, 3, 4})
    return visual_name or channel_image


def _capx_visual_modality(path: list[str]) -> str:
    path_text = ".".join(path).lower()
    if "depth" in path_text:
        return "depth"
    if "segment" in path_text or "mask" in path_text:
        return "segmentation"
    if "rgb" in path_text or "color" in path_text or "image" in path_text:
        return "rgb"
    return "visual_frame"


def _enumerate_capx_objects(observation: Any) -> list[dict[str, Any]]:
    public = _public_capx_observation(observation)
    candidates: list[dict[str, Any]] = []

    def numeric_list(value: Any, length: int) -> list[float] | None:
        if not isinstance(value, list) or len(value) != length:
            return None
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return None
        return [float(item) for item in value]

    def append_candidate(
        path: list[str],
        value: Any,
        *,
        entity_id: str | None = None,
        label: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        position = numeric_list(value, 3)
        pose = numeric_list(value, 7) or numeric_list(value, 10)
        if position is None and pose is None:
            return
        inferred_id = entity_id or ".".join(path)
        candidate: dict[str, Any] = {
            "entity_id": inferred_id,
            "label": label or inferred_id,
            "object_name": inferred_id,
            "observation_path": path,
            "caller_selection": ".".join(path),
            "geometry_tags": ["block_like"],
            "affordance_hints": ["grasp_candidate", "place_support_candidate"],
        }
        if position is not None:
            candidate["position"] = position
        else:
            candidate["position"] = pose[:3]
            candidate["pose"] = pose
            candidate["quaternion_wxyz"] = pose[3:7]
            if len(pose) == 10:
                candidate["extent"] = pose[7:10]
        if extra:
            candidate.update(extra)
        candidates.append(candidate)

    def visit(value: Any, path: list[str]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                if key_text.endswith("_pos"):
                    base = key_text[: -len("_pos")]
                    position = numeric_list(child, 3)
                    if position is None:
                        continue
                    extra: dict[str, Any] = {"position": position}
                    quat = numeric_list(value.get(f"{base}_quat"), 4)
                    if quat is not None:
                        extra["quaternion_wxyz"] = quat
                        extra["pose"] = [*position, *quat]
                    append_candidate([*path, key_text], position, entity_id=base, label=base, extra=extra)
            for key, child in value.items():
                visit(child, [*path, str(key)])
            return
        if not isinstance(value, list) or len(value) not in {3, 7, 10}:
            return
        if not all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            return
        lowered_path = ".".join(path).lower()
        if not any(token in lowered_path for token in ("pose", "object", "cube", "body")):
            return
        append_candidate(path, value, entity_id=path[-1] if path else None)

    visit(public, [])
    return candidates


def _compose_capx_geometry(kwargs: dict[str, Any]) -> dict[str, Any]:
    anchor = _numeric_vector(kwargs["anchor_position"], 3, "anchor_position")
    direction = _numeric_vector(kwargs["direction"], 3, "direction")
    norm = sum(component * component for component in direction) ** 0.5
    if norm <= 1e-12:
        raise ValueError("direction must be non-zero")
    unit = [component / norm for component in direction]
    distance = float(kwargs.get("distance", 0.0))
    extent_terms: dict[str, Any] = {}
    for prefix in ("source", "target"):
        extent = kwargs.get(f"{prefix}_extent")
        scale = float(kwargs.get(f"{prefix}_extent_scale", 0.0))
        if extent is None:
            if scale != 0.0:
                raise ValueError(f"{prefix}_extent is required when {prefix}_extent_scale is non-zero")
            continue
        vector = _numeric_vector(extent, 3, f"{prefix}_extent")
        projected = sum(abs(unit[index]) * vector[index] for index in range(3))
        contribution = scale * projected
        distance += contribution
        extent_terms[prefix] = {"extent": vector, "scale": scale, "contribution": contribution}
    quaternion = _numeric_vector(kwargs["quaternion_wxyz"], 4, "quaternion_wxyz")
    return {
        "position": [anchor[index] + unit[index] * distance for index in range(3)],
        "quaternion_wxyz": quaternion,
        "direction": unit,
        "signed_distance": distance,
        "extent_terms": extent_terms,
        "caller_selected": True,
    }


def build_capx_action_program(spec: dict[str, Any]) -> str:
    """Compile caller-owned object, geometry, arm, and action selections.

    The emitted executor implements only generic dataflow operations. It does
    not add object names, geometry constants, intermediate poses, action ordering, or
    evaluator reads beyond those present in ``spec``.
    """

    if not isinstance(spec, dict) or not isinstance(spec.get("actions"), list):
        raise ValueError("spec must be a mapping with an actions list")
    allowed_operations = {
        "observe",
        "observation_pose",
        "get_object_pose",
        "sample_grasp_pose",
        "geometry",
        "action",
    }
    for index, item in enumerate(spec["actions"]):
        if not isinstance(item, dict) or item.get("operation") not in allowed_operations:
            raise ValueError(f"actions[{index}] has an unsupported operation")
    serialized = json.dumps(spec, sort_keys=True)
    return f'''import json
import math

SPEC = json.loads({serialized!r})
VALUES = {{}}


def numeric_vector(value, length, field):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{{field}} must have length {{length}}")
    return [float(item) for item in value]


def split_pose(value):
    if isinstance(value, dict):
        return value["position"], value["quaternion_wxyz"], value.get("extent")
    if isinstance(value, (list, tuple)) and len(value) in (2, 3):
        return value[0], value[1], value[2] if len(value) == 3 else None
    if isinstance(value, (list, tuple)) and len(value) >= 7:
        return value[:3], value[3:7], value[7:10] if len(value) >= 10 else None
    raise ValueError("pose value has no public position/quaternion representation")


def rotate_wxyz(quaternion, vector):
    w, x, y, z = numeric_vector(quaternion, 4, "quaternion")
    vx, vy, vz = numeric_vector(vector, 3, "vector")
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)]


def resolve_method(action, arm):
    candidates = [action] if arm in ("default", "single") else [f"{{action}}_{{arm}}", f"{{action}}_arm{{arm}}"]
    for name in candidates:
        method = globals().get(name)
        if callable(method):
            return method
    raise AttributeError(f"caller-selected action/arm unavailable: {{action}}/{{arm}}")


def pose_records(value, path=()):
    records = []
    if isinstance(value, dict):
        for key, child in value.items():
            records.extend(pose_records(child, path + (str(key),)))
    elif hasattr(value, "shape") and tuple(value.shape) in ((3,), (7,), (10,)):
        records.append({{"path": list(path), "shape": list(value.shape)}})
    elif isinstance(value, (list, tuple)) and len(value) in (3, 7, 10):
        records.append({{"path": list(path), "shape": [len(value)]}})
    return records


def observation_value(path):
    value = obs
    for key in path:
        value = value[key]
    return value


for item in SPEC["actions"]:
    operation = item["operation"]
    if operation == "observe":
        print("CAPX_OBJECT_ENUMERATION=" + json.dumps(pose_records(obs), sort_keys=True))
    elif operation == "observation_pose":
        raw = observation_value(item["path"])
        raw = raw.tolist() if hasattr(raw, "tolist") else raw
        if not isinstance(raw, (list, tuple)) or len(raw) not in (3, 7):
            raise ValueError("caller-selected observation path is not a 3D or 7D pose")
        quaternion = raw[3:7] if len(raw) == 7 and "quaternion_wxyz" not in item else item["quaternion_wxyz"]
        VALUES[item["output"]] = {{"position": numeric_vector(raw[:3], 3, "position"), "quaternion_wxyz": numeric_vector(quaternion, 4, "quaternion"), "extent": numeric_vector(item["extent"], 3, "extent") if item.get("extent") is not None else None}}
    elif operation == "get_object_pose":
        VALUES[item["output"]] = get_object_pose(item["object_query"], return_bbox_extent=bool(item.get("return_bbox_extent", True)))
    elif operation == "sample_grasp_pose":
        VALUES[item["output"]] = sample_grasp_pose(item["object_query"])
    elif operation == "geometry":
        anchor_pos, anchor_quat, anchor_extent = split_pose(VALUES[item["anchor"]])
        direction = numeric_vector(item["direction"], 3, "direction")
        if item.get("direction_frame", "world") == "anchor":
            direction = rotate_wxyz(anchor_quat, direction)
        norm = math.sqrt(sum(component * component for component in direction))
        if norm <= 1e-12:
            raise ValueError("direction must be non-zero")
        direction = [component / norm for component in direction]
        distance = float(item.get("distance", 0.0))
        for term in item.get("extent_terms", []):
            _, _, extent = split_pose(VALUES[term["source"]])
            if extent is None:
                raise ValueError("extent term references a pose without extent")
            extent = numeric_vector(extent, 3, "extent")
            distance += float(term["scale"]) * sum(abs(direction[i]) * extent[i] for i in range(3))
        _, output_quaternion, _ = split_pose(VALUES[item["quaternion_from"]])
        VALUES[item["output"]] = {{"position": [float(anchor_pos[i]) + direction[i] * distance for i in range(3)], "quaternion_wxyz": numeric_vector(output_quaternion, 4, "quaternion")}}
    else:
        method = resolve_method(item["action"], item["arm"])
        repeat = int(item.get("repeat", 1))
        if repeat < 1:
            raise ValueError("repeat must be positive")
        for _ in range(repeat):
            if item["action"] == "goto_pose":
                position, quaternion, _ = split_pose(VALUES[item["pose"]])
                method(position, quaternion, z_approach=float(item.get("z_approach", 0.0)))
            else:
                method()
'''


def _submit_capx_action(live_api: Any, kwargs: dict[str, Any]) -> Any:
    arm = str(kwargs.get("arm") or "")
    action = str(kwargs.get("action") or "")
    parameters = kwargs.get("parameters") or {}
    if not arm:
        raise ValueError("arm is required and must be caller-selected")
    if action not in {"goto_pose", "open_gripper", "close_gripper"}:
        raise ValueError("action must be one of goto_pose, open_gripper, close_gripper")
    if not isinstance(parameters, dict):
        raise TypeError("parameters must be a mapping")
    method_names = [action] if arm in {"default", "single"} else [f"{action}_{arm}", f"{action}_arm{arm}"]
    method_name = next((name for name in method_names if callable(getattr(live_api, name, None))), None)
    if method_name is None:
        raise AttributeError(f"live_api does not expose caller-selected arm action; tried {method_names!r}")
    method = getattr(live_api, method_name)
    if action == "goto_pose":
        position = _numeric_vector(parameters["position"], 3, "parameters.position")
        quaternion = _numeric_vector(parameters["quaternion_wxyz"], 4, "parameters.quaternion_wxyz")
        return method(position, quaternion, z_approach=float(parameters.get("z_approach", 0.0)))
    if parameters:
        raise ValueError(f"{action} does not accept parameters")
    return method()


def _call_capx_direct_api(live_api: Any, primitive_name: str, kwargs: dict[str, Any]) -> Any:
    method_name = API_METHOD_BY_PRIMITIVE[primitive_name]
    method = getattr(live_api, method_name, None)
    if method is None:
        raise AttributeError(f"live_api does not expose public method {method_name!r}")
    if primitive_name == "capx_get_object_pose":
        return method(kwargs["object_name"], return_bbox_extent=bool(kwargs.get("return_bbox_extent", False)))
    if primitive_name == "capx_sample_grasp_pose":
        return method(kwargs["object_name"])
    if primitive_name == "capx_goto_pose":
        return method(kwargs["position"], kwargs["quaternion_wxyz"], z_approach=float(kwargs.get("z_approach", 0.0)))
    return method()


def _call_capx_grasp_object(live_api: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sample_grasp_pose = getattr(live_api, "sample_grasp_pose", None)
    goto_pose = getattr(live_api, "goto_pose", None)
    close_gripper = getattr(live_api, "close_gripper", None)
    missing = [name for name, method in (("sample_grasp_pose", sample_grasp_pose), ("goto_pose", goto_pose), ("close_gripper", close_gripper)) if method is None]
    if missing:
        raise AttributeError(f"live_api does not expose public method(s): {', '.join(missing)}")
    grasp_pose = sample_grasp_pose(kwargs["object_name"])
    position, quaternion = _split_capx_pose(grasp_pose)
    move_result = goto_pose(position, quaternion, z_approach=0.0)
    close_result = close_gripper()
    return {
        "sampled_grasp": _jsonable(grasp_pose),
        "position": _jsonable(position),
        "quaternion_wxyz": _jsonable(quaternion),
        "goto_result": _jsonable(move_result),
        "close_result": _jsonable(close_result),
    }


def _split_capx_pose(pose: Any) -> tuple[Any, Any]:
    if isinstance(pose, dict):
        position = pose.get("position") or pose.get("pos") or pose.get("xyz")
        quaternion = pose.get("quaternion_wxyz") or pose.get("quat") or pose.get("orientation")
        if position is not None and quaternion is not None:
            return position, quaternion
    if isinstance(pose, (list, tuple)) and len(pose) == 2:
        return pose[0], pose[1]
    if isinstance(pose, (list, tuple)) and len(pose) >= 7:
        return list(pose[:3]), list(pose[3:7])
    raise ValueError("Cannot split CaP-X grasp pose into position and quaternion_wxyz")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def parse_capx_summary(output: str) -> dict[str, Any]:
    matches = re.findall(r"(\d+\.\d+)/(\d+\.\d+)/(\d+)", output)
    if not matches:
        return {"summary_found": False}
    success_rate, avg_reward, completed = matches[-1]
    return {
        "summary_found": True,
        "success_rate": float(success_rate),
        "avg_reward": float(avg_reward),
        "completed": int(completed),
    }


def run_code_file_replay(
    capx_root: Path,
    config_path: Path,
    code_file: Path,
    *,
    seed: int | None = None,
    stdout_limit: int = 8000,
) -> dict[str, Any]:
    """Replay generated code through upstream CaP-X ``env.step(code)``."""
    if not code_file.exists():
        raise FileNotFoundError(f"Replay code file not found: {code_file}")
    if not (capx_root / "capx" / "envs").exists():
        raise FileNotFoundError(f"CaP-X env package not found under --capx-root: {capx_root}")

    script = f"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any


def jsonable(value: Any, max_list_items: int = 8) -> Any:
    if isinstance(value, dict):
        return {{str(key): jsonable(item, max_list_items=max_list_items) for key, item in value.items()}}
    if isinstance(value, (list, tuple)):
        if len(value) > max_list_items:
            return {{
                "type": type(value).__name__,
                "length": len(value),
                "sample": [jsonable(item, max_list_items=max_list_items) for item in value[:3]],
            }}
        return [jsonable(item, max_list_items=max_list_items) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        shape = getattr(value, "shape", None)
        if shape is not None:
            size = getattr(value, "size", None)
            if size is not None and int(size) > max_list_items:
                return {{"type": type(value).__name__, "shape": list(shape)}}
        return jsonable(value.tolist(), max_list_items=max_list_items)
    if hasattr(value, "item"):
        with contextlib.suppress(Exception):
            return jsonable(value.item())
    return repr(value)


capx_root = Path({str(capx_root)!r})
config_path = Path({str(config_path)!r})
code_file = Path({str(code_file)!r})
seed = {seed!r}
stdout_limit = {stdout_limit!r}
code = code_file.read_text(encoding="utf-8")
git_head = None
git_executable = os.getenv("CAPX_GIT_EXECUTABLE") or os.getenv("GIT_PYTHON_GIT_EXECUTABLE") or shutil.which("git")
if git_executable:
    try:
        git_process = subprocess.run(
            [git_executable, "rev-parse", "--short", "HEAD"],
            cwd=capx_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=10,
        )
        git_head = git_process.stdout.strip() if git_process.returncode == 0 else None
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        git_head = None
result: dict[str, Any] = {{
    "route": "code_file_replay",
    "capx_root": str(capx_root),
    "capx_git_head": git_head,
    "config_path": str(config_path),
    "code_file": str(code_file),
    "seed": seed,
    "code_bytes": len(code.encode("utf-8")),
    "ok": False,
    "live_execution_boundary": "upstream CaP-X CodeExecutionEnvBase.step(code)",
}}
env = None
try:
    if str(capx_root) not in sys.path:
        sys.path.insert(0, str(capx_root))
    from capx.envs.configs.instantiate import instantiate
    from capx.envs.configs.loader import DictLoader

    import capx.integrations  # noqa: F401 - API registry side effects

    cfg = DictLoader.load(str(config_path))
    if not isinstance(cfg, dict) or not isinstance(cfg.get("env"), dict):
        raise ValueError(f"Expected CaP-X config with dict key 'env': {{config_path}}")
    env = instantiate(cfg["env"])
    reset_obs, reset_info = env.reset(seed=seed)
    low_level = getattr(env, "low_level_env", None)
    step_obs, reward, terminated, truncated, info = env.step(code)
    result.update(
        {{
            "ok": True,
            "reset_info": jsonable(reset_info),
            "reset_observation": jsonable(reset_obs),
            "sandbox_rc": info.get("sandbox_rc") if isinstance(info, dict) else None,
            "stdout_tail": str(info.get("stdout", ""))[-stdout_limit:] if isinstance(info, dict) else "",
            "stderr_tail": str(info.get("stderr", ""))[-stdout_limit:] if isinstance(info, dict) else "",
            "reward": float(reward),
            "task_completed": bool(info.get("task_completed")) if isinstance(info, dict) else None,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "step_observation": jsonable(step_obs),
            "sim_step_count": getattr(low_level, "_sim_step_count", None),
        }}
    )
except BaseException as exc:
    result.update({{"error": repr(exc), "traceback_tail": traceback.format_exc()[-stdout_limit:]}})
finally:
    if env is not None:
        close = getattr(env, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()
print("CAPX_REPLAY_JSON=" + json.dumps(result, sort_keys=True))
"""
    use_uv = (capx_root / ".venv").exists() and shutil.which("uv") is not None
    command = ["uv", "run", "--no-sync", "--active", "python", "-c", script] if use_uv else [sys.executable, "-c", script]
    env_vars = os.environ.copy()
    env_vars["PYTHONPATH"] = f"{capx_root}:{env_vars.get('PYTHONPATH', '')}"
    env_vars.setdefault("UV_CACHE_DIR", "/tmp/capx-uv-cache")
    completed = subprocess.run(
        command,
        cwd=capx_root,
        env=env_vars,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    parsed = _parse_marker_json(completed.stdout, "CAPX_REPLAY_JSON=") or {}
    parsed.setdefault("ok", False)
    parsed["command"] = command[:4] + ["..."] if use_uv else command[:2] + ["..."]
    parsed["returncode"] = completed.returncode
    parsed["runner"] = "uv" if use_uv else "python"
    parsed["process_stdout_tail"] = completed.stdout[-stdout_limit:]
    return parsed


def run_live(
    capx_root: Path,
    config_path: Path,
    *,
    model: str | None,
    server_url: str | None,
    total_trials: int,
    num_workers: int,
) -> dict[str, Any]:
    if not (capx_root / "capx" / "envs" / "launch.py").exists():
        raise FileNotFoundError(f"CaP-X launch.py not found under --capx-root: {capx_root}")

    config_arg = str(config_path)
    try:
        config_arg = str(config_path.relative_to(capx_root))
    except ValueError:
        pass

    command = [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        config_arg,
        "--total-trials",
        str(total_trials),
        "--num-workers",
        str(num_workers),
    ]
    if model:
        command.extend(["--model", model])
    if server_url:
        command.extend(["--server-url", server_url])

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{capx_root}:{env.get('PYTHONPATH', '')}"
    completed = subprocess.run(
        command,
        cwd=capx_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "summary": parse_capx_summary(completed.stdout),
        "stdout_tail": completed.stdout[-8000:],
    }


def capx_native_api_schema(live_api: Any) -> dict[str, Any]:
    """Inspect only methods published by the original CaPX API registry."""
    import inspect
    schema = {name: {"signature": str(inspect.signature(fn)), "doc": inspect.getdoc(fn) or ""}
              for name, fn in live_api.functions().items() if callable(fn)}
    for name in ("find_object_base_rotate", "find_object_torso_rotate"):
        if name in schema:
            schema[name]["harness_execution_guidance"] = (
                "This original function performs multiple planning, motion and perception steps "
                "and may take many minutes. The code-cell timeout is shared by all calls in a cell. "
                "Run one long search or motion call per code cell, print its concise result, "
                "and continue in the next turn so feedback and native verification can run. "
                "Python variables and the episode persist between turns."
            )
    return schema


def call_capx_native_api(live_api: Any, action: str, parameters: dict[str, Any], values: dict[str, Any]) -> PrimitiveResult:
    """Keep original call semantics and non-JSON return objects in this episode."""
    import inspect
    import uuid
    import numpy as np
    functions = live_api.functions()
    if action not in functions or not callable(functions[action]):
        return PrimitiveResult(name="submit_capx_action", ok=False, error="unknown_native_public_api:" + action)
    if not isinstance(parameters, dict):
        return PrimitiveResult(name="submit_capx_action", ok=False, error="parameters_must_be_mapping")
    def decode(value):
        if isinstance(value, dict) and "native_value_handle" in value:
            handle = value["native_value_handle"]
            if handle not in values:
                raise ValueError("Unknown or expired native value handle")
            return values[handle]
        if isinstance(value, dict):
            return {key: decode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [decode(item) for item in value]
        return value
    def encode(value):
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        if isinstance(value, dict):
            return {str(key): encode(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [encode(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        handle = "capx-value-" + uuid.uuid4().hex
        values[handle] = value
        result = {"native_value_handle": handle, "type": type(value).__name__}
        array = value if isinstance(value, np.ndarray) else None
        if array is None and hasattr(value, "detach") and hasattr(value, "cpu"):
            array = value.detach().cpu().numpy()
        if array is not None:
            result["shape"] = list(array.shape)
            if array.size <= 128:
                result["values"] = array.tolist()
        return result
    try:
        if action in {"prepare_carry", "navigate_checked"}:
            live_api._arena_carry_navigation.feedback = {}
        if action == "navigate_to_pose" and hasattr(live_api, "_arena_navigation_feedback"):
            del live_api._arena_navigation_feedback
        method = functions[action]
        signature = inspect.signature(method)
        arguments = {key: decode(value) for key, value in parameters.items()}
        for key, value in list(arguments.items()):
            parameter = signature.parameters.get(key)
            annotation = str(parameter.annotation) if parameter is not None else ""
            # The upstream R1Pro joint method has no type annotation, but
            # accepts ndarray/tensor and calls .clone() on anything else.
            # JSON numeric vectors therefore need the same array decoding.
            joint_vector = action == "move_to_joint_positions" and key == "target_joint_positions"
            if isinstance(value, list) and joint_vector:
                # R1Pro's native joint state and controller updates are float32.
                # float64 would reach torch.from_numpy then fail index_put.
                array = np.asarray(value, dtype=np.float32)
                if array.ndim != 1 or not np.isfinite(array).all():
                    raise ValueError("target_joint_positions must be a finite numeric vector")
                arguments[key] = array
            elif (action == "grasp_object" and key in {"pregrasp_pose", "grasp_pose"}
                  and isinstance(value, (list, tuple))):
                # Both JSON-authored coordinates and sampled native handles
                # represent the public (xyz, quaternion) pair, despite the
                # upstream's ndarray annotation on this particular function.
                if len(value) != 2:
                    raise ValueError(key + " must be an (xyz[3], quaternion[4]) pair")
                pair = tuple(np.asarray(x, dtype=float) for x in value)
                if (pair[0].shape != (3,) or pair[1].shape != (4,)
                        or not all(np.isfinite(x).all() for x in pair)):
                    raise ValueError(key + " must contain finite xyz[3] and quaternion[4]")
                arguments[key] = pair
            elif isinstance(value, list) and "ndarray" in annotation:
                native_pose_pair = (len(value) == 2
                    and all(isinstance(x, np.ndarray) for x in value)
                    and value[0].shape == (3,) and value[1].shape == (4,))
                # R1Pro grasp_object annotates pose pairs as ndarray although
                # sample_grasp_pose returns (xyz[3], quaternion[4]). Preserve
                # those decoded native arrays instead of constructing a ragged
                # array and failing before the upstream method is invoked.
                arguments[key] = (tuple(np.asarray(x) for x in value)
                    if "tuple" in annotation.lower() or native_pose_pair else np.asarray(value))
        signature.bind(**arguments)
        result = method(**arguments)
        if action == "get_env_observation" and isinstance(result, dict):
            result = _public_capx_observation(result)
        output = {"native_api_result": encode(result), "selected_action": action, "executed": True}
        if (getattr(live_api, '_arena_checked_manipulation', False)
                and action in {"move_hand_checked", "grasp_object", "place_object_checked"}
                and isinstance(result, dict)):
            # These wrappers promise measured postconditions, not merely that
            # a Python call returned. Preserve the distinction at the boundary.
            if action == "move_hand_checked":
                ok = bool(result.get("arrived"))
            elif action == "grasp_object":
                ok = bool(result.get("grasped") and result.get("lifted")
                          and result.get("ready_to_move", True))
            else:
                ok = bool(result.get("released") and result.get("retreated"))
            output["executed"] = bool(result.get("executed", True))
            return PrimitiveResult(name="submit_capx_action", ok=ok, output=output,
                                   error=None if ok else result.get("reason") or "checked_manipulation_failed")
        if action in {"prepare_carry", "navigate_checked"}:
            ok = bool(result["ready_to_move"] if action == "prepare_carry" else result["ok"])
            output["executed"] = bool(result.get("executed"))
            return PrimitiveResult(name="submit_capx_action", ok=ok, output=output,
                                   error=None if ok else result.get("reason") or "checked_motion_failed")
        if action == "navigate_to_pose" and hasattr(live_api, "_arena_navigation_feedback"):
            output["navigation_feedback"] = dict(live_api._arena_navigation_feedback)
            return PrimitiveResult(name="submit_capx_action", ok=bool(result), output=output,
                                   error=None if result else "navigation_goal_not_reached")
        return PrimitiveResult(name="submit_capx_action", ok=True, output=output)
    except Exception as exc:
        if action in {"prepare_carry", "navigate_checked"}:
            feedback = getattr(getattr(live_api, "_arena_carry_navigation", None), "feedback", {})
            return PrimitiveResult(name="submit_capx_action", ok=False,
                                   output={"selected_action": action, "motion_feedback": encode(feedback)},
                                   error=f"{type(exc).__name__}: {exc}")
        if action == "navigate_to_pose" and hasattr(live_api, "_arena_navigation_feedback"):
            return PrimitiveResult(name="submit_capx_action", ok=False,
                                   output={"selected_action": action,
                                           "navigation_feedback": dict(live_api._arena_navigation_feedback)},
                                   error=f"{type(exc).__name__}: {exc}")
        return PrimitiveResult(name="submit_capx_action", ok=False, error=f"{type(exc).__name__}: {exc}")


def prepare_capx_r1pro_scene_config(config: dict[str, Any], capx_root: Path, runtime_workdir: Path) -> dict[str, Any]:
    """Complete the original task config's scene from its native activity template."""
    from copy import deepcopy
    import yaml
    config = deepcopy(config)
    low = config["env"]["cfg"]["low_level"]
    activity = low["activity_name"]
    data = Path(os.environ["OMNIGIBSON_DATA_PATH"])
    templates = list((data / "2025-challenge-task-instances/scenes").glob(f"*/json/*_task_{activity}_0_0_template.json"))
    if len(templates) != 1:
        raise ValueError(f"Expected one native scene template for CaPX activity {activity}; found {len(templates)}")
    template = templates[0]
    original = capx_root / "capx/third_party/b1k/OmniGibson/omnigibson/configs" / low["controller_cfg"]
    controller = yaml.safe_load(original.read_text())
    controller["scene"].update(scene_model=template.parent.parent.name, scene_instance=template.stem, load_room_types=None)
    controller["task"]["activity_name"] = activity
    target = runtime_workdir / "native_r1pro_controller.yaml"
    target.write_text(yaml.safe_dump(controller, sort_keys=False))
    low["controller_cfg"] = str(target)
    return config


def _prepare_capx_behavior_headless() -> None:
    # Reuse existing batch-node bootstrap; imports resolve to CaPX's pinned B1K
    # fork in its isolated Python prefix. Do not replace the native robot/task.
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")
    from .behavior1k_agent_runtime import (
        Behavior1KRuntimeConfig, _configure_isaacsim_noninteractive_eula,
        _configure_omnigibson_headless_viewer,
        _maybe_patch_isaacsim_headless_viewport_wait,
        _maybe_patch_omnigibson_fast_mesh_triangulation,
    )
    config = Behavior1KRuntimeConfig(headless=True)
    _configure_isaacsim_noninteractive_eula(config)


def _finish_capx_behavior_headless() -> None:
    # R1Pro's native import sets macros. Configure the batch viewer afterwards
    # so that bootstrap does not freeze HEADLESS before that original setup.
    from .behavior1k_agent_runtime import (
        Behavior1KRuntimeConfig, _configure_omnigibson_headless_viewer,
        _maybe_patch_isaacsim_headless_viewport_wait,
        _maybe_patch_omnigibson_fast_mesh_triangulation,
        _maybe_apply_minimal_kit_no_flowusd_override,
    )
    config = Behavior1KRuntimeConfig(headless=True)
    import omnigibson as og
    _configure_omnigibson_headless_viewer(config, og)
    _maybe_patch_isaacsim_headless_viewport_wait(config)
    _maybe_patch_omnigibson_fast_mesh_triangulation()
    _maybe_apply_minimal_kit_no_flowusd_override()
    if os.environ.get("EMBODIED_ARENA_CUROBO_SHARED_BUFFERS") == "0":
        # Defer the CuRobo import until after the native scene initializes USD.
        from functools import wraps
        from omnigibson.action_primitives.curobo import CuRoboMotionGenerator
        from .behavior1k_agent_runtime import _configure_native_curobo_shared_buffers
        original = CuRoboMotionGenerator.__init__
        if not getattr(original, "_arena_native_no_shared_buffers", False):
            @wraps(original)
            def initialize(self, *args, **kwargs):
                _configure_native_curobo_shared_buffers()
                original(self, *args, **kwargs)
            initialize._arena_native_no_shared_buffers = True
            CuRoboMotionGenerator.__init__ = initialize
