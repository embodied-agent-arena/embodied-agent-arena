from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .capx_comparator_runtime import API_ALIASES, build_api_trace_primitive
from .backend import EmbodiedBackend
from .behavior1k_agent_runtime import Behavior1KAgentRuntimeBackend, Behavior1KRuntimeConfig
from .robocasa365_agent_runtime import RoboCasa365AgentRuntimeBackend, RoboCasa365RuntimeConfig
from .robodojo_agent_runtime import RoboDojoAgentRuntimeBackend, RoboDojoRuntimeConfig
from .robotwin2_agent_runtime import RoboTwin2AgentRuntimeBackend, RoboTwin2RuntimeConfig
from .robowits_agent_runtime import RoboWitsAgentRuntimeBackend, RoboWitsRuntimeConfig
from .schemas import EpisodeTrace, Observation, PrimitiveCard, PrimitiveResult, TaskSpec, VerificationResult


@dataclass(frozen=True, slots=True)
class W4EnvironmentSpec:
    benchmark_id: str
    directory: str
    simulator: str
    python: str
    package_manager: str
    requires_gpu: bool
    headless: str
    assets: list[str] = field(default_factory=list)
    install_hint: str = ""
    smoke_task: str = ""
    live_smoke_command: str = ""
    smoke_mode: str = "offline_symbolic"
    offline_adapter: str = "partial"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "directory": self.directory,
            "simulator": self.simulator,
            "python": self.python,
            "package_manager": self.package_manager,
            "requires_gpu": self.requires_gpu,
            "headless": self.headless,
            "assets": list(self.assets),
            "install_hint": self.install_hint,
            "smoke_task": self.smoke_task,
            "live_smoke_command": self.live_smoke_command,
            "smoke_mode": self.smoke_mode,
            "offline_adapter": self.offline_adapter,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class W4SmokeTask:
    task_id: str
    benchmark_id: str
    instruction: str
    goal: dict[str, Any]
    initial_state: dict[str, Any]
    primitive_names: list[str]


def w4_environment_specs() -> dict[str, W4EnvironmentSpec]:
    specs = [
        W4EnvironmentSpec(
            benchmark_id="maniskill",
            directory="benchmarks/operation/maniskill",
            simulator="SAPIEN / Gymnasium",
            python="3.10+ recommended",
            package_manager="pip/conda",
            requires_gpu=True,
            headless="Native RGB/depth/segmentation observations require a working Vulkan rendering device.",
            assets=[],
            install_hint="conda env create -f benchmarks/operation/maniskill/environment.yml; configure Vulkan/SAPIEN before visual smoke.",
            smoke_task="Caller-selected official environment with native visual observations and generic controls",
            live_smoke_command="python -m embodied_harness.maniskill_runtime_smoke --env-id <official-environment-id> --obs-mode rgb+depth+segmentation --control-mode pd_ee_delta_pos --sim-backend physx_cpu --render-backend gpu",
            smoke_mode="prior_free_visual_observe_and_caller_authored_control",
            offline_adapter="partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="vimabench",
            directory="benchmarks/operation/vimabench",
            simulator="PyBullet / Gym-style tabletop",
            python="3.9",
            package_manager="conda + pip",
            requires_gpu=False,
            headless="Official PyBullet DIRECT/headless mode for runtime smoke; GUI recorder/debug paths require a display.",
            assets=["tabletop assets"],
            install_hint="conda env create -f benchmarks/operation/vimabench/environment.yml",
            smoke_task="visual_manipulation",
            live_smoke_command='python -m embodied_harness.vimabench_runtime_smoke --task-name visual_manipulation --query "target object described by the prompt" --output reports/live_vimabench_smoke.json',
            smoke_mode="agent_native_prompt_rgbd_grounding_then_live_pybullet_step",
            offline_adapter="partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="cliport",
            directory="benchmarks/operation/cliport",
            simulator="PyBullet / Ravens tabletop",
            python="3.8",
            package_manager="conda prefix env plus upstream pip editable install",
            requires_gpu=False,
            headless="Use PyBullet headless/disp=False; learned models expect a single NVIDIA GPU.",
            assets=["Ravens/CLIPort assets"],
            install_hint="Use external/environments/cliport; install the checkout from external/upstreams/cliport in editable mode.",
            smoke_task="stack-block-pyramid-seq-seen-colors",
            live_smoke_command="CLIPORT_ROOT=external/upstreams/cliport PYTHONPATH=src/embodied_harness/cliport_runtime_stubs:src python -m embodied_harness.cliport_runtime_smoke --task-name stack-block-pyramid-seq-seen-colors --mode test --seed 10001 --query 'the object mentioned by the language goal' --context '{\"plan_step\":\"ground pick target\"}' --minimal-step --output reports/cliport_runtime_smoke.json",
            smoke_mode="live_headless_pybullet_minimal_step",
            offline_adapter="partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="vlabench",
            directory="benchmarks/operation/vlabench",
            simulator="MuJoCo / dm_control",
            python="3.10 recommended",
            package_manager="conda + pip, optional upstream Docker/OpenPI path",
            requires_gpu=False,
            headless="Preferred EGL/osmesa; this worktree observed EGL display init failure and osmesa missing GL symbols, with glfw requiring DISPLAY for rendering.",
            assets=["mesh assets", "task assets"],
            install_hint="conda env create -f benchmarks/operation/vlabench/environment.yml; clone https://github.com/OpenMOSS/VLABench; pip install -r requirements.txt; pip install -e .; python scripts/download_assets.py; export VLABENCH_ROOT=/path/to/VLABench/VLABench.",
            smoke_task="select_toy live reset and instruction/context-conditioned observation summary",
            live_smoke_command="MUJOCO_GL=glfw benchmarks/operation/vlabench/run_live_smoke.sh",
            smoke_mode="live_import_configured_asset_blocked",
            offline_adapter="partial",
            notes=[
                "MuJoCo/dm_control simulator label is confirmed by upstream requirements and LM4ManipDMEnv.",
                "Upstream env.get_observation returns rgb/depth/segmentation/robot state/end-effector state/task observables.",
                "Upstream evaluation attaches env.task.get_instruction() to each observation before policy prediction.",
                "Harness primitives intentionally exclude checker/oracle/success/progress helpers; noop_step returns only simulator step evidence and action schema.",
            ],
        ),
        W4EnvironmentSpec(
            benchmark_id="robocasa",
            directory="benchmarks/operation/robocasa",
            simulator="RoboCasa365 / robosuite / MuJoCo kitchen",
            python="3.11 recommended by current upstream README",
            package_manager="conda + pip git installs",
            requires_gpu=False,
            headless="Use MUJOCO_GL=egl and PYOPENGL_PLATFORM=egl where offscreen GPU rendering is required.",
            assets=["kitchen assets (~10GB)", "RoboCasa macros_private.py generated by setup_macros", "RoboCasa object and fixture assets"],
            install_hint="conda env create -f benchmarks/operation/robocasa/environment.yml; python -m robocasa.scripts.setup_macros; python -m robocasa.scripts.download_kitchen_assets.",
            smoke_task="robocasa/CoffeeServeMug",
            live_smoke_command='python -m embodied_harness.robocasa_runtime_smoke --env-id robocasa/CoffeeServeMug --split pretrain --direct-robosuite-state-only --state-delta-motion --motion-skill grasp_place --motion-horizon 80 --grasp-hold-steps 8 --place-relation auto --place-horizon 120 --agent-context \'{"caller":"coding-agent","policy":"public-observation-primitives-only"}\' --rollout-steps 1 --require-verification-success --output reports/robocasa_transport_state_smoke.json',
            smoke_mode="visual_preflight_plus_direct_robosuite_state_delta_controller_with_dry_contract_fallback",
            offline_adapter="partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="capx",
            directory="benchmarks/operation/capx",
            simulator="CaP-Gym over robosuite / covered routes / BEHAVIOR",
            python="3.10",
            package_manager="conda bootstrap + uv upstream",
            requires_gpu=True,
            headless="Depends on selected CaP-X route; service ports and CUDA notes required.",
            assets=["selected CaP-X env configs", "optional perception/motion service weights"],
            install_hint="conda env create -f environment.yml, then ./install_capx.sh --route robosuite|libero|behavior.",
            smoke_task="primitive/preflight inspect of env_configs/cube_stack/franka_robosuite_cube_stack.yaml, then upstream quick regression when CaP-X is installed",
            live_smoke_command="python capx_runtime_smoke.py --capx-root /path/to/cap-x --mode preflight && python capx_runtime_smoke.py --capx-root /path/to/cap-x --mode live --total-trials 1 --num-workers 1",
            smoke_mode="comparator_trace",
            offline_adapter="partial",
            notes=[
                "CaP-X is a comparator harness, not a single benchmark adapter; covered routes stay inside this comparator.",
                "The comparator CLI emits prompt/API/context trace evidence and delegates live execution to upstream capx/envs/launch.py.",
            ],
        ),
        W4EnvironmentSpec(
            benchmark_id="rlbench",
            directory="benchmarks/operation/rlbench",
            simulator="CoppeliaSim v4.1.0 / PyRep",
            python="PyRep-compatible Python; usually 3.8-3.10",
            package_manager="conda/pip",
            requires_gpu=False,
            headless="Prefer headless=True with CoppeliaSim root/library path; use DISPLAY=:99 or xvfb-run -a when no display server is present. EGL is not assumed for PyRep. On the current host the successful route uses Mesa swrast plus LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7.",
            assets=["CoppeliaSim scenes", "task assets"],
            install_hint="bash benchmarks/operation/rlbench/install_rlbench_runtime.sh; export CoppeliaSim paths before live smoke.",
            smoke_task="ReachTarget import/reset/observe/ground/minimal-step plus qwen pose-action motion",
            live_smoke_command="python -m embodied_harness.rlbench_runtime_smoke --task-name ReachTarget --query 'the target I should reach' --output reports/live_rlbench_env_case.json",
            smoke_mode="contract_ci_then_required_live_rlbench_pyrep_step",
            offline_adapter="partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="calvin",
            directory="benchmarks/operation/calvin",
            simulator="PyBullet / CALVIN",
            python="3.10",
            package_manager="conda/pip",
            requires_gpu=False,
            headless="PyBullet EGL path is supported by the wrapper; validate EGL before full challenge rollout.",
            assets=[
                "CALVIN debug dataset (~1.3GB)",
                "official task_D_D dataset for MCIL baseline evaluation",
                "official D_D_static_rgb_baseline/mcil_baseline.ckpt policy checkpoint",
            ],
            install_hint="Materialize the pinned CALVIN Python 3.10 environment, then run scripts/slurm_calvin_operation_environment_seal.sbatch to seal the official MCIL model-step and verifier evidence.",
            smoke_task="debug language sequence state/camera/subgoal smoke",
            live_smoke_command='python -m embodied_harness.calvin_runtime_smoke --calvin-root "$CALVIN_ROOT" --dataset-root "$CALVIN_ROOT/dataset/task_D_D/training" --sequence-id debug_language_sequence --query "what language skill should run next?" --output reports/live_calvin_runtime_smoke.json',
            smoke_mode="live_debug_dataset_env_reset_observe_step",
            offline_adapter="contract-only",
        ),
        W4EnvironmentSpec(
            benchmark_id="behavior1k",
            directory="benchmarks/operation/behavior1k",
            simulator="BEHAVIOR-1K / OmniGibson / Isaac Sim",
            python="Official BEHAVIOR-1K environment",
            package_manager="Official BEHAVIOR-1K setup.sh",
            requires_gpu=True,
            headless="OmniGibson headless mode still requires the supported Isaac Sim GPU runtime.",
            assets=["BEHAVIOR dataset assets", "BDDL task definitions", "Isaac Sim runtime"],
            install_hint="Use the official BEHAVIOR-1K setup.sh and dataset installation flow.",
            smoke_task="turning_on_radio",
            live_smoke_command="python -m embodied_harness.behavior1k_preflight --create-env --probe-tags --output reports/behavior1k_preflight_live.json",
            smoke_mode="existing_agent_runtime_dry_by_default_live_on_request",
            offline_adapter="yes/partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="robocasa365",
            directory="benchmarks/operation/robocasa365",
            simulator="RoboCasa365 / robosuite / MuJoCo",
            python="3.11 recommended by RoboCasa",
            package_manager="conda + pip git installs",
            requires_gpu=False,
            headless="Use the existing RoboCasa offscreen EGL runtime configuration.",
            assets=["RoboCasa kitchen assets", "RoboCasa365 dataset metadata"],
            install_hint="Install RoboCasa and configure the RoboCasa365 dataset root before live reset.",
            smoke_task="robocasa365/CoffeeSetupMug",
            live_smoke_command="python -m embodied_harness.robocasa365_preflight --help",
            smoke_mode="existing_agent_runtime_dry_by_default_live_on_request",
            offline_adapter="yes/partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="robowits",
            directory="benchmarks/operation/robowits",
            simulator="Genesis World",
            python="Repo-local RoboWits Python environment",
            package_manager="upstream environment and asset setup",
            requires_gpu=True,
            headless="Genesis backend and device are selected by the runtime config.",
            assets=["RoboWits evaluation JSON", "HF assets", "optional BlenderKit assets"],
            install_hint="Install the upstream RoboWits environment and assets, then configure repo_path.",
            smoke_task="RoboWits task 01 episode 0",
            live_smoke_command="python -m embodied_harness.robowits_preflight --help",
            smoke_mode="existing_agent_runtime_dry_by_default_live_on_request",
            offline_adapter="yes/partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="robotwin2",
            directory="benchmarks/operation/robotwin2",
            simulator="SAPIEN 3 / RoboTwin 2.0",
            python="3.10",
            package_manager="upstream conda/pip environment",
            requires_gpu=True,
            headless="Live visual reset requires a supported Vulkan-capable GPU runtime.",
            assets=["RoboTwin2 objects", "embodiments", "optional background textures"],
            install_hint="Install the upstream RoboTwin2 environment and configure repo_path and assets.",
            smoke_task="place_empty_cup / demo_clean",
            live_smoke_command="python scripts/robotwin2_preflight.py --help",
            smoke_mode="existing_agent_runtime_dry_by_default_live_on_request",
            offline_adapter="yes/partial",
        ),
        W4EnvironmentSpec(
            benchmark_id="robodojo",
            directory="benchmarks/operation/robodojo",
            simulator="Isaac Sim / Isaac Lab",
            python=">=3.11",
            package_manager="upstream RoboDojo environment",
            requires_gpu=True,
            headless="Live evaluation requires Isaac Sim, Isaac Lab, assets, and the configured policy service.",
            assets=["RoboDojo task assets", "robot assets", "environment layouts"],
            install_hint="Install the upstream RoboDojo runtime and initialize its assets before live evaluation.",
            smoke_task="caller-selected RoboDojo task",
            live_smoke_command="python -m embodied_harness.robodojo_agent_runtime --help",
            smoke_mode="existing_agent_runtime_dry_by_default_live_on_request",
            offline_adapter="yes/partial",
        ),
    ]
    return {spec.benchmark_id: spec for spec in specs}


class W4BenchmarkBackend(EmbodiedBackend):
    """Benchmark-first W4 smoke backend.

    The original W4 families use symbolic benchmark-specific smoke primitives.
    Families with an existing agent runtime delegate to that runtime so the
    public action boundary and harness-only verifier remain authoritative.
    """

    def __init__(self, tasks: dict[str, W4SmokeTask] | None = None) -> None:
        self.environment_specs = w4_environment_specs()
        self._tasks = tasks or self._default_tasks()
        self._runtime_backend: EmbodiedBackend | None = None
        self._task: W4SmokeTask | None = None
        self._task_spec: TaskSpec | None = None
        self._state: dict[str, Any] = {}
        self._trace: EpisodeTrace | None = None
        self._capx_official_environment: Any | None = None

    @staticmethod
    def _default_tasks() -> dict[str, W4SmokeTask]:
        tasks = [
            W4SmokeTask(
                task_id="w4_maniskill_pick_cube",
                benchmark_id="maniskill",
                instruction="Use ManiSkill action primitives to grasp the cube and place it on the goal pad.",
                goal={"object": "cube", "target": "goal_pad", "required_state": "placed"},
                initial_state={
                    "object": "cube",
                    "target": "goal_pad",
                    "holding": None,
                    "placed": False,
                    "actor_poses": {"cube": [0, 0, 0, 1, 0, 0, 0], "goal_pad": [1, 0, 0, 1, 0, 0, 0]},
                },
                primitive_names=[
                    "observe_maniskill_state",
                    "locate_maniskill_actor",
                    "grasp_maniskill_actor",
                    "place_maniskill_actor_on",
                ],
            ),
            W4SmokeTask(
                task_id="w4_vima_prompt_pick_place",
                benchmark_id="vimabench",
                instruction="Read the VIMA prompt and solve it by composing tabletop pick/place actions.",
                goal={"object": "red_block", "target": "green_container", "relation": "in"},
                initial_state={
                    "prompt": "put the red block into the green container",
                    "objects": ["red_block", "green_container"],
                    "holding": None,
                    "placements": {},
                },
                primitive_names=[
                    "observe_vima_prompt",
                    "observe_vima_scene",
                    "pick_vima_object",
                    "place_vima_object",
                ],
            ),
            W4SmokeTask(
                task_id="w4_cliport_pick_place",
                benchmark_id="cliport",
                instruction="Use CLIPort/Ravens action primitives to put the yellow bowl on the blue mat.",
                goal={"object": "yellow_bowl", "target": "blue_mat", "relation": "on"},
                initial_state={
                    "language_goal": "put the yellow bowl on the blue mat",
                    "objects": ["yellow_bowl", "blue_mat"],
                    "holding": None,
                    "placements": {},
                    "reward": 0.0,
                    "done": False,
                },
                primitive_names=[
                    "observe_cliport_rgbd",
                    "get_cliport_task_language_goal",
                    "pick_cliport_object",
                    "place_cliport_object",
                ],
            ),
            W4SmokeTask(
                task_id="w4_vlabench_policy_skill",
                benchmark_id="vlabench",
                instruction="Use VLABench action-skill primitives to close the drawer.",
                goal={"skill": "close_drawer", "target": "drawer", "required_state": "drawer_closed"},
                initial_state={"instruction": "close the drawer", "drawer_closed": False, "skill_calls": []},
                primitive_names=[
                    "observe_vlabench_scene",
                    "get_vlabench_instruction",
                    "execute_vlabench_skill",
                ],
            ),
            W4SmokeTask(
                task_id="w4_robocasa_kitchen_atomic",
                benchmark_id="robocasa",
                instruction="Use RoboCasa kitchen action primitives to close the cabinet.",
                goal={"object": "cabinet", "state": "closed"},
                initial_state={"objects": {"cabinet": {"state": "open"}}, "last_action": None},
                primitive_names=[
                    "observe_robocasa_kitchen_state",
                    "inspect_robocasa_object",
                    "close_robocasa_fixture",
                    "open_robocasa_fixture",
                ],
            ),
            W4SmokeTask(
                task_id="w4_capx_comparator_trace",
                benchmark_id="capx",
                instruction="Use CaP-X public perception and motion APIs to manipulate the cube without verifier shortcuts.",
                goal={"object": "cube", "required_pose_visit": True},
                initial_state={
                    "available_apis": list(API_ALIASES),
                    "object_poses": {
                        "cube": [0.4, 0.1, 0.2, 1, 0, 0, 0],
                        "red_cube": [0.4, 0.1, 0.2, 1, 0, 0, 0],
                        "green_cube": [0.45, 0.1, 0.18, 1, 0, 0, 0],
                    },
                    "sampled_grasps": {},
                    "visited_pose": None,
                    "gripper": "open",
                    "grasped_object": None,
                },
                primitive_names=[
                    "get_capx_api_trace",
                    "get_capx_prompt",
                    "get_capx_available_apis",
                    *API_ALIASES.keys(),
                ],
            ),
            W4SmokeTask(
                task_id="w4_rlbench_waypoint",
                benchmark_id="rlbench",
                instruction="Use RLBench arm and gripper action primitives to reach the target handle and close the gripper.",
                goal={"target": "target_handle", "gripper": "closed"},
                initial_state={"arm_at": None, "gripper": "open", "scene_targets": ["target_handle"]},
                primitive_names=[
                    "observe_rlbench_scene",
                    "ground_rlbench_target",
                    "step_rlbench_action",
                    "move_rlbench_arm_to",
                    "open_rlbench_gripper",
                    "close_rlbench_gripper",
                ],
            ),
            W4SmokeTask(
                task_id="w4_calvin_subgoal_chain",
                benchmark_id="calvin",
                instruction="Use CALVIN action primitives to turn on the light.",
                goal={"subgoal": "turn_on_light", "light": "on"},
                initial_state={"subgoal": "turn_on_light", "light": "off", "completed_subgoals": []},
                primitive_names=[
                    "observe_calvin_state",
                    "get_calvin_language_subgoal",
                    "toggle_calvin_light",
                    "execute_calvin_language_skill",
                ],
            ),
            W4SmokeTask(
                task_id="w4_behavior1k_runtime",
                benchmark_id="behavior1k",
                instruction="Use the existing BEHAVIOR-1K agent runtime without evaluator or prior leakage.",
                goal={"success_source": "harness_only_official_evaluator"},
                initial_state={},
                primitive_names=[],
            ),
            W4SmokeTask(
                task_id="w4_robocasa365_runtime",
                benchmark_id="robocasa365",
                instruction="Use the existing RoboCasa365 agent runtime and suite-specific public primitives.",
                goal={"success_source": "harness_only_official_evaluator"},
                initial_state={},
                primitive_names=[],
            ),
            W4SmokeTask(
                task_id="w4_robowits_runtime",
                benchmark_id="robowits",
                instruction="Use the existing RoboWits observation, geometry, control, and evidence runtime.",
                goal={"success_source": "harness_only_official_evaluator"},
                initial_state={},
                primitive_names=[],
            ),
            W4SmokeTask(
                task_id="w4_robotwin2_runtime",
                benchmark_id="robotwin2",
                instruction="Use the existing RoboTwin2 visual and generic dual-arm control runtime.",
                goal={"success_source": "harness_only_private_env_check"},
                initial_state={},
                primitive_names=[],
            ),
            W4SmokeTask(
                task_id="w4_robodojo_runtime",
                benchmark_id="robodojo",
                instruction="Use the existing RoboDojo observation and public action-schema runtime.",
                goal={"success_source": "harness_only_official_evaluator"},
                initial_state={},
                primitive_names=[],
            ),
        ]
        return {task.task_id: task for task in tasks}

    def list_task_ids(self) -> list[str]:
        return sorted(self._tasks)

    def environment_summary(self) -> dict[str, dict[str, Any]]:
        return {benchmark_id: spec.to_dict() for benchmark_id, spec in self.environment_specs.items()}

    def reset(self, task_id: str, seed: int | None = None, config: dict[str, Any] | None = None) -> TaskSpec:
        if task_id not in self._tasks:
            raise KeyError(f"Unknown W4 task: {task_id}")
        self._capx_official_environment = None
        self._task = self._tasks[task_id]
        self._runtime_backend = self._construct_runtime_backend(self._task.benchmark_id)
        if self._runtime_backend is not None:
            task_spec = self._runtime_backend.reset(task_id, seed=seed, config=config)
            task_spec.metadata.setdefault("environment", self.environment_specs[self._task.benchmark_id].to_dict())
            return task_spec
        self._state = deepcopy(self._task.initial_state)
        self._state["seed"] = seed
        self._state["config"] = config or {}
        spec = self.environment_specs[self._task.benchmark_id]
        self._task_spec = TaskSpec(
            task_id=self._task.task_id,
            source=f"w4:{self._task.benchmark_id}",
            instruction=self._task.instruction,
            goal=deepcopy(self._task.goal),
            initial_state=deepcopy(self._task.initial_state),
            budgets={"primitive_calls": 12, "verifier_calls": 4},
            tags=["w4", "benchmark_first", self._task.benchmark_id],
            allowed_primitive_levels=["L1", "L2", "L3", "L4"],
            metadata={
                "benchmark_id": self._task.benchmark_id,
                "benchmark_directory": spec.directory,
                "environment": spec.to_dict(),
                "oracle_leakage_level": "none_action_primitives_only",
            },
        )
        self._trace = EpisodeTrace(task_id=task_id)
        self.record_event(
            "reset",
            {
                "seed": seed,
                "config": config or {},
                "task": self._task_spec.to_dict(),
                "benchmark_id": self._task.benchmark_id,
            },
        )
        return self._task_spec

    def observe(self) -> Observation:
        if self._runtime_backend is not None:
            return self._runtime_backend.observe()
        self._require_task()
        obs = Observation(
            step=len(self.get_trace().events),
            data={
                "instruction": self._task_spec.instruction if self._task_spec else "",
                "benchmark_id": self._task.benchmark_id if self._task else "",
                "state": deepcopy(self._state),
            },
            metadata={"benchmark_first": True},
        )
        self.record_event("observe", obs.to_dict())
        return obs

    def list_primitives(self, level: str | None = None) -> list[PrimitiveCard]:
        if self._runtime_backend is not None:
            return self._runtime_backend.list_primitives(level=level)
        self._require_task()
        cards = [self._primitive_card(name) for name in self._task.primitive_names]
        cards.append(self._primitive_card("record_w4_evidence"))
        if level is not None:
            cards = [card for card in cards if card.abstraction_level == level]
        self.record_event("list_primitives", {"level": level, "count": len(cards), "benchmark_id": self._task.benchmark_id})
        return cards

    def call_primitive(self, name: str, **kwargs: Any) -> PrimitiveResult:
        if self._runtime_backend is not None:
            return self._runtime_backend.call_primitive(name, **kwargs)
        self._require_task()
        allowed = set(self._task.primitive_names) | {"record_w4_evidence"}
        if name not in allowed:
            result = PrimitiveResult(name=name, ok=False, error=f"Primitive {name!r} is not exposed for {self._task.task_id}")
        else:
            handler = getattr(self, f"_primitive_{name}", None)
            if handler is None:
                result = PrimitiveResult(name=name, ok=False, error=f"Missing smoke handler for primitive: {name}")
            else:
                result = handler(**kwargs)
        self.record_event(
            "primitive_call",
            {
                "benchmark_id": self._task.benchmark_id,
                "name": name,
                "kwargs": kwargs,
                "result": result.to_dict(),
            },
        )
        return result

    def verify(self, scope: str = "task", **kwargs: Any) -> VerificationResult:
        if self._runtime_backend is not None:
            return self._runtime_backend.verify(scope=scope, **kwargs)
        self._require_task()
        checker = getattr(self, f"_verify_{self._task.benchmark_id}", None)
        if scope != "task":
            result = VerificationResult(ok=False, scope=scope, message=f"Unsupported W4 verification scope: {scope}")
        elif checker is None:
            result = VerificationResult(ok=False, scope=scope, message=f"Missing verifier for {self._task.benchmark_id}")
        else:
            result = checker()
        if result.ok:
            self.get_trace().final_status = "success"
        self.record_event("verifier_call", {"benchmark_id": self._task.benchmark_id, **result.to_dict()})
        return result

    def get_trace(self) -> EpisodeTrace:
        if self._runtime_backend is not None:
            return self._runtime_backend.get_trace()
        if self._trace is None:
            raise RuntimeError("Backend has not been reset.")
        return self._trace

    @staticmethod
    def _construct_runtime_backend(benchmark_id: str) -> EmbodiedBackend | None:
        constructors = {
            "behavior1k": lambda: Behavior1KAgentRuntimeBackend(Behavior1KRuntimeConfig(live=False)),
            "robocasa365": lambda: RoboCasa365AgentRuntimeBackend(RoboCasa365RuntimeConfig(live=False)),
            "robowits": lambda: RoboWitsAgentRuntimeBackend(RoboWitsRuntimeConfig(live=False)),
            "robotwin2": lambda: RoboTwin2AgentRuntimeBackend(RoboTwin2RuntimeConfig(live=False)),
            "robodojo": lambda: RoboDojoAgentRuntimeBackend(RoboDojoRuntimeConfig(live=False)),
        }
        constructor = constructors.get(benchmark_id)
        return constructor() if constructor is not None else None

    def _primitive_card(self, name: str) -> PrimitiveCard:
        tags = ["w4", self._task.benchmark_id if self._task else "unknown"]
        level = "L2"
        input_schema: dict[str, Any] = {"kwargs": "benchmark-specific"}
        output_schema: dict[str, Any] = {"result": "dict"}
        if name.startswith(
            (
                "step_",
                "execute_",
                "move_",
                "grasp_",
                "place_",
                "pick_",
                "close_",
                "open_",
                "toggle_",
                "capx_goto",
                "capx_grasp",
            )
        ):
            level = "L3"
        if name.startswith(("observe_", "get_", "locate_", "inspect_", "capx_get")):
            level = "L1" if "observe" in name else "L2"
        if name == "record_w4_evidence":
            input_schema = {"key": "str", "value": "any", "agent_context": "dict|None"}
            output_schema = {"artifact_id": "str", "agent_context": "dict"}
            level = "L1"
        elif name == "get_capx_prompt":
            input_schema = {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}
            output_schema = {"prompt": "str", "query": "str|None", "agent_context": "dict"}
        elif name == "get_capx_available_apis":
            input_schema = {"prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}
            output_schema = {"apis": "list[str]", "query": "str|None", "agent_context": "dict"}
        elif name == "capx_get_object_pose":
            input_schema = {"object_name": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}
            output_schema = {"object_name": "str", "pose": "list[float]", "query": "str|None", "agent_context": "dict"}
        elif name == "capx_sample_grasp_pose":
            input_schema = {"object_name": "str", "prompt": "str|None", "query": "str|None", "agent_context": "dict|None"}
            output_schema = {"object_name": "str", "position": "list[float]", "quaternion_wxyz": "list[float]", "query": "str|None", "agent_context": "dict"}
        elif name == "capx_goto_pose":
            input_schema = {
                "position": "list[float]",
                "quaternion_wxyz": "list[float]",
                "z_approach": "float",
                "prompt": "str|None",
                "query": "str|None",
                "agent_context": "dict|None",
            }
            output_schema = {"success": "bool", "visited_pose": "list[float]", "position": "list[float]", "quaternion_wxyz": "list[float]"}
        elif name in {"capx_open_gripper", "capx_close_gripper"}:
            input_schema = {"agent_context": "dict|None"}
            output_schema = {"success": "bool", "gripper": "open|closed", "agent_context": "dict"}
        return PrimitiveCard(
            name=name,
            capability_tags=tags,
            input_schema=input_schema,
            output_schema=output_schema,
            cost={"primitive_calls": 1},
            failure_modes=["wrong_arguments", "wrong_order", "backend_not_configured"],
            abstraction_level=level,
            leakage_risk="L2_task_state" if name.startswith(("observe_", "get_", "locate_", "inspect_")) else "none",
            description=f"Benchmark-specific W4 action primitive: {name}.",
        )

    def _primitive_record_w4_evidence(
        self,
        key: str,
        value: Any,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        artifact_id = f"w4:{self._task.benchmark_id}:{key}"
        payload = {"benchmark_id": self._task.benchmark_id, "key": key, "value": value, "agent_context": agent_context or {}}
        self._state.setdefault("evidence", {})[key] = value
        self.get_trace().add_artifact(artifact_id, payload)
        return PrimitiveResult(name="record_w4_evidence", ok=True, output={"artifact_id": artifact_id, "agent_context": agent_context or {}}, artifacts=[artifact_id])

    def _primitive_observe_maniskill_state(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_maniskill_state", ok=True, output={"state": deepcopy(self._state)})

    def _primitive_locate_maniskill_actor(self, actor_name: str) -> PrimitiveResult:
        pose = self._state.get("actor_poses", {}).get(actor_name)
        if pose is None:
            return PrimitiveResult(name="locate_maniskill_actor", ok=False, error="actor_not_found")
        return PrimitiveResult(name="locate_maniskill_actor", ok=True, output={"actor_name": actor_name, "pose": list(pose)})

    def _primitive_grasp_maniskill_actor(self, actor_name: str) -> PrimitiveResult:
        if actor_name != self._state.get("object"):
            return PrimitiveResult(name="grasp_maniskill_actor", ok=False, output={"success": False}, error="cannot_grasp_actor")
        self._state["holding"] = actor_name
        return PrimitiveResult(name="grasp_maniskill_actor", ok=True, output={"success": True, "holding": self._state["holding"]})

    def _primitive_place_maniskill_actor_on(self, actor_name: str, target_name: str) -> PrimitiveResult:
        if self._state.get("holding") != actor_name:
            return PrimitiveResult(name="place_maniskill_actor_on", ok=False, error="actor_not_held")
        self._state["holding"] = None
        self._state["placed"] = actor_name == self._task.goal["object"] and target_name == self._task.goal["target"]
        return PrimitiveResult(
            name="place_maniskill_actor_on",
            ok=bool(self._state["placed"]),
            output={"success": bool(self._state["placed"]), "placed": self._state["placed"], "actor_name": actor_name, "target_name": target_name},
            error=None if self._state["placed"] else "wrong_place_target",
        )

    def _verify_maniskill(self) -> VerificationResult:
        ok = bool(self._state.get("placed"))
        return VerificationResult(ok=ok, scope="task", message="PickCube goal satisfied" if ok else "cube is not placed", metrics={"success": float(ok)})

    def _primitive_observe_vima_prompt(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_vima_prompt", ok=True, output={"prompt": self._state["prompt"]})

    def _primitive_observe_vima_scene(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_vima_scene", ok=True, output={"objects": list(self._state["objects"]), "holding": self._state.get("holding")})

    def _primitive_pick_vima_object(self, object_name: str) -> PrimitiveResult:
        if object_name not in self._state["objects"]:
            return PrimitiveResult(name="pick_vima_object", ok=False, output={"success": False}, error="object_not_found")
        self._state["holding"] = object_name
        return PrimitiveResult(name="pick_vima_object", ok=True, output={"success": True, "holding": object_name})

    def _primitive_place_vima_object(self, object_name: str, target_name: str) -> PrimitiveResult:
        if self._state.get("holding") != object_name:
            return PrimitiveResult(name="place_vima_object", ok=False, output={"success": False}, error="object_not_held")
        if target_name not in self._state["objects"]:
            return PrimitiveResult(name="place_vima_object", ok=False, output={"success": False}, error="target_not_found")
        self._state["holding"] = None
        self._state["placements"][object_name] = target_name
        ok = object_name == self._task.goal["object"] and target_name == self._task.goal["target"]
        return PrimitiveResult(name="place_vima_object", ok=ok, output={"success": ok, "placements": deepcopy(self._state["placements"])}, error=None if ok else "wrong_target")

    def _verify_vimabench(self) -> VerificationResult:
        ok = self._state.get("placements", {}).get(self._task.goal["object"]) == self._task.goal["target"]
        return VerificationResult(ok=ok, scope="task", message="VIMA tabletop action succeeded" if ok else "VIMA action missing or wrong", metrics={"success": float(ok)})

    def _primitive_observe_cliport_rgbd(self) -> PrimitiveResult:
        artifact_id = "cliport_rgbd:smoke"
        return PrimitiveResult(name="observe_cliport_rgbd", ok=True, output={"rgbd_artifact": artifact_id, "artifact_id": artifact_id})

    def _primitive_get_cliport_task_language_goal(self) -> PrimitiveResult:
        return PrimitiveResult(name="get_cliport_task_language_goal", ok=True, output={"language_goal": self._state["language_goal"]})

    def _primitive_pick_cliport_object(self, object_name: str) -> PrimitiveResult:
        if object_name not in self._state["objects"]:
            return PrimitiveResult(name="pick_cliport_object", ok=False, output={"success": False}, error="object_not_found")
        self._state["holding"] = object_name
        return PrimitiveResult(name="pick_cliport_object", ok=True, output={"success": True, "holding": object_name})

    def _primitive_place_cliport_object(self, object_name: str, target_name: str) -> PrimitiveResult:
        if self._state.get("holding") != object_name:
            return PrimitiveResult(name="place_cliport_object", ok=False, output={"success": False}, error="object_not_held")
        if target_name not in self._state["objects"]:
            return PrimitiveResult(name="place_cliport_object", ok=False, output={"success": False}, error="target_not_found")
        self._state["holding"] = None
        self._state["placements"][object_name] = target_name
        ok = object_name == self._task.goal["object"] and target_name == self._task.goal["target"]
        self._state["done"] = ok
        self._state["reward"] = 1.0 if ok else 0.0
        return PrimitiveResult(name="place_cliport_object", ok=ok, output={"success": ok, "reward": self._state["reward"], "done": self._state["done"]}, error=None if ok else "wrong_target")

    def _verify_cliport(self) -> VerificationResult:
        ok = bool(self._state.get("done"))
        return VerificationResult(ok=ok, scope="task", message="CLIPort reward done" if ok else "CLIPort task not done", metrics={"reward": self._state.get("reward", 0.0)})

    def _primitive_observe_vlabench_scene(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_vlabench_scene", ok=True, output={"scene": "drawer_scene", "drawer_closed": self._state["drawer_closed"]})

    def _primitive_get_vlabench_instruction(self) -> PrimitiveResult:
        return PrimitiveResult(name="get_vlabench_instruction", ok=True, output={"instruction": self._state["instruction"]})

    def _primitive_execute_vlabench_skill(self, skill_name: str, target_name: str, horizon: int = 10) -> PrimitiveResult:
        call = {"skill_name": skill_name, "target_name": target_name, "horizon": horizon}
        self._state["skill_calls"].append(call)
        normalized = skill_name.lower().replace(" ", "_")
        if normalized in {"close_drawer", "close"} and target_name == self._task.goal["target"] and horizon > 0:
            self._state["drawer_closed"] = True
        return PrimitiveResult(name="execute_vlabench_skill", ok=True, output={"success": bool(self._state["drawer_closed"]), "skill_call": call, "drawer_closed": self._state["drawer_closed"]})

    def _verify_vlabench(self) -> VerificationResult:
        ok = bool(self._state["drawer_closed"])
        return VerificationResult(ok=ok, scope="task", message="VLABench policy task succeeded" if ok else "drawer still open", metrics={"success": float(ok), "skill_calls": len(self._state["skill_calls"])})

    def _primitive_observe_robocasa_kitchen_state(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_robocasa_kitchen_state", ok=True, output={"objects": deepcopy(self._state["objects"])})

    def _primitive_inspect_robocasa_object(self, object_name: str) -> PrimitiveResult:
        obj = self._state["objects"].get(object_name)
        if obj is None:
            return PrimitiveResult(name="inspect_robocasa_object", ok=False, error="object_not_found")
        return PrimitiveResult(name="inspect_robocasa_object", ok=True, output={"object_name": object_name, **deepcopy(obj)})

    def _primitive_close_robocasa_fixture(self, fixture_name: str) -> PrimitiveResult:
        if fixture_name not in self._state["objects"]:
            return PrimitiveResult(name="close_robocasa_fixture", ok=False, output={"success": False}, error="fixture_not_found")
        self._state["objects"][fixture_name]["state"] = "closed"
        self._state["last_action"] = {"op": "close", "object": fixture_name}
        return PrimitiveResult(name="close_robocasa_fixture", ok=True, output={"success": True, "object_name": fixture_name, "state": "closed"})

    def _primitive_open_robocasa_fixture(self, fixture_name: str) -> PrimitiveResult:
        if fixture_name not in self._state["objects"]:
            return PrimitiveResult(name="open_robocasa_fixture", ok=False, output={"success": False}, error="fixture_not_found")
        self._state["objects"][fixture_name]["state"] = "open"
        self._state["last_action"] = {"op": "open", "object": fixture_name}
        return PrimitiveResult(name="open_robocasa_fixture", ok=True, output={"success": True, "object_name": fixture_name, "state": "open"})

    def _verify_robocasa(self) -> VerificationResult:
        ok = self._state["objects"]["cabinet"]["state"] == "closed"
        return VerificationResult(ok=ok, scope="task", message="RoboCasa atomic kitchen task satisfied" if ok else "cabinet still open", metrics={"success": float(ok)})

    def _primitive_get_capx_prompt(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        trace = self._build_capx_trace(prompt=prompt or self._task.instruction, query=query, agent_context=agent_context)
        if trace is not None:
            return PrimitiveResult(name="get_capx_prompt", ok=True, output={"prompt": trace.get("prompt"), "prompt_source": trace.get("prompt_source"), "trace_evidence": trace.get("trace_evidence")})
        return PrimitiveResult(name="get_capx_prompt", ok=True, output={"prompt": self._task.instruction, "query": query, "agent_context": agent_context or {}})

    def _primitive_get_capx_available_apis(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        trace = self._build_capx_trace(prompt=prompt, query=query, agent_context=agent_context)
        if trace is not None:
            return PrimitiveResult(name="get_capx_available_apis", ok=True, output={"apis": trace.get("configured_apis", []), "route_capabilities": trace.get("route_capabilities"), "action_api_schema": trace.get("action_api_schema")})
        return PrimitiveResult(name="get_capx_available_apis", ok=True, output={"apis": list(self._state["available_apis"]), "query": query, "agent_context": agent_context or {}})

    def _primitive_get_capx_api_trace(
        self,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        trace = self._build_capx_trace(prompt=prompt, query=query, agent_context=agent_context)
        if trace is None:
            return PrimitiveResult(
                name="get_capx_api_trace",
                ok=False,
                error="capx_runtime_config_required",
                output={"required_reset_config": {"capx_root": "/path/to/cap-x", "config_path": "env_configs/...yaml"}},
            )
        return PrimitiveResult(name="get_capx_api_trace", ok=True, output=trace)

    def _build_capx_trace(
        self,
        *,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        runtime = self._state.get("config", {}).get("capx_runtime") or self._state.get("capx_runtime")
        if not isinstance(runtime, dict):
            return None
        config_path = runtime.get("config_path")
        if not config_path:
            return None
        capx_root = Path(runtime["capx_root"]).resolve() if runtime.get("capx_root") else None
        resolved_config = Path(config_path)
        if not resolved_config.is_absolute() and capx_root is not None:
            resolved_config = capx_root / resolved_config
        return build_api_trace_primitive(
            capx_root=capx_root,
            config_path=resolved_config.resolve(),
            prompt=prompt,
            query=query,
            agent_context=agent_context,
            include_preflight=bool(runtime.get("include_preflight", False)),
        )

    def _primitive_capx_get_object_pose(
        self,
        object_name: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        pose = self._state["object_poses"].get(object_name)
        if pose is None:
            return PrimitiveResult(name="capx_get_object_pose", ok=False, error="object_not_found")
        return PrimitiveResult(name="capx_get_object_pose", ok=True, output={"object_name": object_name, "pose": list(pose), "query": query, "agent_context": agent_context or {}})

    def _primitive_capx_sample_grasp_pose(
        self,
        object_name: str,
        prompt: str | None = None,
        query: str | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        pose = self._state["object_poses"].get(object_name)
        if pose is None:
            return PrimitiveResult(name="capx_sample_grasp_pose", ok=False, error="object_not_found")
        grasp = {"position": list(pose[:3]), "quaternion_wxyz": list(pose[3:7])}
        self._state["sampled_grasps"][object_name] = grasp
        return PrimitiveResult(name="capx_sample_grasp_pose", ok=True, output={"object_name": object_name, **grasp, "query": query, "agent_context": agent_context or {}})

    def _primitive_capx_goto_pose(
        self,
        position: list[float] | None = None,
        quaternion_wxyz: list[float] | None = None,
        z_approach: float = 0.0,
        pose: list[float] | None = None,
        prompt: str | None = None,
        query: str | None = None,
        visual_evidence_refs: list[dict[str, Any]] | None = None,
        agent_context: dict[str, Any] | None = None,
    ) -> PrimitiveResult:
        if pose is not None:
            target_pose = list(pose)
            position = list(pose[:3])
            quaternion_wxyz = list(pose[3:7])
        elif position is not None and quaternion_wxyz is not None:
            target_pose = list(position) + list(quaternion_wxyz)
        else:
            return PrimitiveResult(name="capx_goto_pose", ok=False, error="position_and_quaternion_required")
        self._state["visited_pose"] = target_pose
        return PrimitiveResult(
            name="capx_goto_pose",
            ok=True,
            output={
                "success": True,
                "visited_pose": target_pose,
                "position": list(position),
                "quaternion_wxyz": list(quaternion_wxyz),
                "z_approach": z_approach,
                "visual_evidence_ref_count": len(visual_evidence_refs or []),
                "query": query,
                "agent_context": agent_context or {},
            },
        )

    def _primitive_capx_open_gripper(self, agent_context: dict[str, Any] | None = None) -> PrimitiveResult:
        self._state["gripper"] = "open"
        return PrimitiveResult(name="capx_open_gripper", ok=True, output={"success": True, "gripper": "open", "agent_context": agent_context or {}})

    def _primitive_capx_close_gripper(self, agent_context: dict[str, Any] | None = None) -> PrimitiveResult:
        self._state["gripper"] = "closed"
        return PrimitiveResult(name="capx_close_gripper", ok=True, output={"success": True, "gripper": "closed", "agent_context": agent_context or {}})

    def bind_capx_official_environment(self, environment: Any) -> None:
        """Bind the live CaP-X simulator whose private checker owns task success.

        This is a harness-only binding: it is deliberately absent from the
        agent primitive surface.  The environment must be the same live
        instance used for reset and stepping so verifier parity is evaluated
        against the exact state the agent changed.
        """

        self._require_task()
        if self._task.benchmark_id != "capx":
            raise RuntimeError("CaP-X official environment can only be bound to a CaP-X task.")
        if not callable(getattr(environment, "task_completed", None)):
            raise TypeError("CaP-X official environment must expose callable task_completed().")
        self._capx_official_environment = environment

    def _verify_capx(self) -> VerificationResult:
        if self._capx_official_environment is not None:
            ok = bool(self._capx_official_environment.task_completed())
            return VerificationResult(
                ok=ok,
                scope="task",
                message=(
                    "CaP-X official task checker succeeded"
                    if ok
                    else "CaP-X official task checker did not report success"
                ),
                metrics={"success": float(ok), "official_verifier_bound": 1.0},
                metadata={
                    "official_symbol": "FrankaRobosuiteCubesLowLevel.task_completed"
                },
            )
        ok = self._state.get("visited_pose") == self._state["object_poses"].get(self._task.goal["object"])
        return VerificationResult(
            ok=ok,
            scope="task",
            message=(
                "CaP-X action API reached target pose"
                if ok
                else "CaP-X target pose not reached"
            ),
            metrics={"success": float(ok), "official_verifier_bound": 0.0},
        )

    def _primitive_observe_rlbench_scene(self) -> PrimitiveResult:
        return PrimitiveResult(
            name="observe_rlbench_scene",
            ok=True,
            output={
                "scene": "rlbench_smoke_scene",
                "targets": list(self._state["scene_targets"]),
                "gripper": self._state["gripper"],
                "action_schema": {"minimal_step_action": {"shape": [8]}},
            },
        )

    def _primitive_ground_rlbench_target(self, query: str | None = None, target_name: str | None = None) -> PrimitiveResult:
        selected = target_name or self._task.goal["target"]
        if selected not in self._state["scene_targets"]:
            return PrimitiveResult(name="ground_rlbench_target", ok=False, output={"query": query, "selected": None}, error="target_not_found")
        return PrimitiveResult(
            name="ground_rlbench_target",
            ok=True,
            output={
                "query": query,
                "selected": {"label": selected, "score": 1.0, "pose_world": None},
                "candidates": [{"label": selected, "score": 1.0}],
                "mask_evidence": {"source": "symbolic_contract_only"},
                "pose_evidence": {"source": "symbolic_contract_only", "pose_world": None},
            },
        )

    def _primitive_step_rlbench_action(self, action: list[float] | None = None) -> PrimitiveResult:
        return PrimitiveResult(
            name="step_rlbench_action",
            ok=True,
            output={"success": True, "action": action or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], "reward": 0.0, "terminate": False},
        )

    def _primitive_move_rlbench_arm_to(self, target_name: str) -> PrimitiveResult:
        if target_name not in self._state["scene_targets"]:
            return PrimitiveResult(name="move_rlbench_arm_to", ok=False, output={"success": False}, error="target_not_found")
        self._state["arm_at"] = target_name
        return PrimitiveResult(name="move_rlbench_arm_to", ok=True, output={"success": True, "arm_at": target_name, "position": target_name})

    def _primitive_open_rlbench_gripper(self) -> PrimitiveResult:
        self._state["gripper"] = "open"
        return PrimitiveResult(name="open_rlbench_gripper", ok=True, output={"success": True, "gripper": "open", "state": "open"})

    def _primitive_close_rlbench_gripper(self) -> PrimitiveResult:
        self._state["gripper"] = "closed"
        return PrimitiveResult(name="close_rlbench_gripper", ok=True, output={"success": True, "gripper": "closed", "state": "closed"})

    def _verify_rlbench(self) -> VerificationResult:
        ok = self._state.get("arm_at") == self._task.goal["target"] and self._state.get("gripper") == self._task.goal["gripper"]
        return VerificationResult(ok=ok, scope="task", message="RLBench waypoint reached" if ok else "required waypoint not reached", metrics={"success": float(ok)})

    def _primitive_observe_calvin_state(self) -> PrimitiveResult:
        return PrimitiveResult(name="observe_calvin_state", ok=True, output={"state": deepcopy(self._state)})

    def _primitive_get_calvin_language_subgoal(self) -> PrimitiveResult:
        return PrimitiveResult(name="get_calvin_language_subgoal", ok=True, output={"subgoal": self._state["subgoal"]})

    def _primitive_toggle_calvin_light(self, state: str) -> PrimitiveResult:
        normalized = state.lower()
        if normalized not in {"on", "off"}:
            return PrimitiveResult(name="toggle_calvin_light", ok=False, output={"success": False}, error="invalid_light_state")
        self._state["light"] = normalized
        if normalized == "on" and "turn_on_light" not in self._state["completed_subgoals"]:
            self._state["completed_subgoals"].append("turn_on_light")
        return PrimitiveResult(name="toggle_calvin_light", ok=True, output={"success": True, "light": self._state["light"], "state": self._state["light"]})

    def _primitive_execute_calvin_language_skill(self, subgoal: str, horizon: int = 10) -> PrimitiveResult:
        if subgoal == "turn_on_light" and horizon > 0:
            self._state["light"] = "on"
            if subgoal not in self._state["completed_subgoals"]:
                self._state["completed_subgoals"].append(subgoal)
        success = subgoal in self._state["completed_subgoals"]
        return PrimitiveResult(name="execute_calvin_language_skill", ok=True, output={"success": success, "light": self._state["light"], "state": self._state["light"], "completed_subgoals": list(self._state["completed_subgoals"])})

    def _verify_calvin(self) -> VerificationResult:
        ok = self._state.get("light") == self._task.goal["light"]
        return VerificationResult(ok=ok, scope="task", message="CALVIN subgoal chain satisfied" if ok else "CALVIN subgoal missing", metrics={"success": float(ok)})

    def _require_task(self) -> None:
        if self._task is None:
            raise RuntimeError("Call reset() before using the backend.")
