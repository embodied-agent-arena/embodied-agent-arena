# Installing the interactive benchmarks

Install only the benchmarks you intend to run. The repository contains the W3/W4
adapters, agent loop, task interfaces, scoring logic, source pins, and configuration
patches. Simulator installations, upstream source checkouts, scene assets, model
weights, and Python environments are downloaded separately. The dataset supplies
the selected episodes and initialization metadata, not a simulator installation.

## Shared layout

Use Linux. Install the main harness as described in the [README](../README.md).
Keep the harness environment separate from native simulator environments: some
simulators require NumPy 2, while the offline harness uses NumPy 1.

Run the following from the code checkout:

```bash
export ARENA_CODE="$PWD"
export EMBODIED_ARENA_EXTERNAL_ROOT="$ARENA_CODE/external"
mkdir -p "$EMBODIED_ARENA_EXTERNAL_ROOT"/{upstreams,environments,assets}
```

The installation layout is:

```text
code/
  external/
    upstreams/<benchmark>/       # pinned upstream source and submodules
    environments/<benchmark>/    # native Python environment
    assets/<benchmark>/          # separately downloaded assets
data/
  cases.jsonl
  episodes/                     # selected task pools, graphs, seeds, trajectories
```

These three external directories are ignored by Git. Set
`EMBODIED_ARENA_EXTERNAL_ROOT` before every command if they live elsewhere.
Do not resample task pools when installing an environment; `cases.jsonl` and
`episodes/` already specify the evaluation cases.

W3 source commits are recorded in [w3-sources.lock.json](../external/w3-sources.lock.json).
W4 repositories, commits, and nested dependencies are recorded in
[sources.lock.yaml](../external/sources.lock.yaml). The latter is JSON-compatible
YAML. Check out the recorded commits before following each upstream's installation
instructions; the latest default branch may expose a different API.
Additional CaP-X LIBERO source pins are in
[w4-release-sources.json](../configs/w4-release-sources.json), which also binds
the W4 harness files to the control versions used by the final execution plans.

For example, obtain the ScienceWorld source:

```bash
git clone https://github.com/allenai/ScienceWorld.git \
  "$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/scienceworld"
git -C "$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/scienceworld" \
  checkout e8216d6044e8e39be9fcb185e3b2dfb602584b52
```

For a W4 benchmark, print all required repositories before cloning them:

```bash
python - rlbench <<'PY'
import json, os, sys
from pathlib import Path
lock = json.loads(Path('external/sources.lock.yaml').read_text())
row = next(r for r in lock['sources'] if r['benchmark_id'] == sys.argv[1])
root = Path(os.environ['EMBODIED_ARENA_EXTERNAL_ROOT']) / row['checkout']
for repo in row['repositories']:
    print(repo['url'], repo['revision'], root / repo['checkout'])
PY
```

Clone each repository to its printed path, check out its printed commit, then
initialize its submodules with `git submodule update --init --recursive`.
An explicitly listed nested repository must also match its own recorded commit.

## W3: six benchmarks, 157 cases

The Python package versions used by the adapters are recorded in
[pukun.json](../configs/environments/pukun.json) and
[humanclaw.json](../configs/environments/humanclaw.json). These are package
inventories, not portable virtual environments or universal installation scripts.

| Benchmark / selector | Cases | Native requirements | Official installation entry |
|---|---:|---|---|
| ALFRED / `alfred_official_visual` | 20 | Python 3.10, ALFWorld source, AI2-THOR 2.1.0, Linux THOR binary, OpenGL/X display | [ALFWorld at the pinned revision](https://github.com/alfworld/alfworld/tree/aaba6870f86c5be6a08a491f32a50b906227bc3e) |
| ALFWorld / `alfworld_visual` | 20 | Same environment as ALFRED | [ALFWorld installation](https://github.com/alfworld/alfworld/blob/aaba6870f86c5be6a08a491f32a50b906227bc3e/README.md) |
| ScienceWorld / `scienceworld_text` | 73 | Python 3.10, `scienceworld==1.3.0`, Java runtime and ScienceWorld JAR | [ScienceWorld installation](https://github.com/allenai/ScienceWorld/blob/e8216d6044e8e39be9fcb185e3b2dfb602584b52/README.md) |
| DiscoveryWorld / `discoveryworld` | 10 | Python 3.10, upstream package, Pygame and upstream resources | [DiscoveryWorld installation](https://github.com/allenai/discoveryworld/tree/fd591323920be0d3786ef350955de1945aa571e5) |
| VirtualHome / `virtualhome_symbolic` | 21 | Python 3.10, Evolving Graph source and its Python dependencies | [VirtualHome installation](https://github.com/xavierpuigf/virtualhome/tree/58970fd80951c2eaa1af713e0917d1a105353ad8) |
| HumanCLAW / `humanclaw` | 13 | Python 3.11, patched Habitat-Sim with Bullet, HSSD scenes, motion-controller assets, NVIDIA EGL | [HumanCLAW installation](https://github.com/Human-CLAW/HumanCLAW/tree/13a8b6d0bf9cc1e458aff0e7442b526288325acb) |

### ScienceWorld, DiscoveryWorld, and VirtualHome

These three adapters execute on CPU. Create a Python 3.10 environment, for example
at `external/environments/pukun`, and install the selected upstream's requirements
there. ScienceWorld can be installed with:

```bash
python3.10 -m venv "$EMBODIED_ARENA_EXTERNAL_ROOT/environments/pukun"
"$EMBODIED_ARENA_EXTERNAL_ROOT/environments/pukun/bin/python" \
  -m pip install scienceworld==1.3.0
java -version
export EMBODIED_ARENA_NATIVE_PYTHON_SCIENCEWORLD_TEXT=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/environments/pukun/bin/python"
```

Install DiscoveryWorld's pinned source and requirements in the same environment
or a separate one. Its resource files must remain with the upstream checkout at
`external/upstreams/discoveryworld`; use `SDL_VIDEODRIVER=dummy` on a headless host.
Select its interpreter with `EMBODIED_ARENA_NATIVE_PYTHON_DISCOVERYWORLD`.

For VirtualHome, place the pinned checkout at `external/upstreams/virtualhome`
and install the dependencies of its `virtualhome/simulation/evolving_graph`
modules. The selected symbolic tasks use the graphs in the dataset and do not
require the Unity executable. Select the interpreter with
`EMBODIED_ARENA_NATIVE_PYTHON_VIRTUALHOME_SYMBOLIC`.

### ALFRED and ALFWorld visual

Both adapters load `alfworld.env.thor_env` from `external/upstreams/alfworld`.
Install the pinned ALFWorld package's visual dependencies in a Python 3.10
environment, retaining `ai2thor==2.1.0` and `numpy==1.26.4`. Obtain the matching
official Linux THOR executable using the upstream installation procedure.
The selected `traj_data.json` files are supplied in the task dataset.

```bash
export ALFWORLD_ROOT="$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/alfworld"
export EMBODIED_ARENA_NATIVE_PYTHON_ALFRED_OFFICIAL_VISUAL=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/environments/pukun/bin/python"
export EMBODIED_ARENA_NATIVE_PYTHON_ALFWORLD_VISUAL=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/environments/pukun/bin/python"
# Set this to the actual executable obtained from the THOR 2.1 release:
export ALFWORLD_THOR_EXECUTABLE=/absolute/path/to/thor-2.1-Linux64
```

On headless machines install Xvfb and its system X/OpenGL dependencies. Run the
evaluation under a working X display, for example `xvfb-run -a arena run ...`.
Installing the Python package alone does not supply a working display or GPU driver.

### HumanCLAW

Use the pinned HumanCLAW source under `external/upstreams/humanclaw`. Follow its
HalfPhysics setup: Habitat-Sim commit
`acbe6f4922e68145e401e55c30f9dfea460a3f24`, its specified Bullet submodule, and
the upstream patch/build flags. A generic Habitat-Sim wheel is not an equivalent
installation. Obtain the official HSSD validation scenes, HumanCLAW scene
supplements, and `paper_fullval_v1` motion assets through the upstream instructions.

The selected tasks explicitly use `external/environments/humanclaw/bin/python`.
Install the native environment at that path, or link that directory to your
existing installation. The explicit task interpreter takes precedence over the
generic interpreter environment variable.

```bash
export HUMANCLAW_ROOT="$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/humanclaw"
export HUMANCLAW_HSSD_SCENE_DATASET_CONFIG=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/assets/humanclaw/prepared/hssd-hab.scene_dataset_config.json"
mkdir -p "$EMBODIED_ARENA_EXTERNAL_ROOT/deployment/humanclaw"
cp configs/nvidia-egl-vendor.json \
  "$EMBODIED_ARENA_EXTERNAL_ROOT/deployment/humanclaw/10_nvidia.json"
```

The EGL manifest refers to the host's `libEGL_nvidia.so.0`; install a compatible
driver on the host. Keep scene paths inside the HSSD configuration consistent
with the separately downloaded assets. Task pools are supplied under
`data/episodes/humanclaw`.

## W4: thirteen reporting groups, 183 cases

Use a separate native environment per benchmark. RoboCasa and RoboCasa365 may
share their interpreter. Package inventories are in
[configs/environments](../configs/environments); native asset paths and source
patches are declared in
[operation_environment_lock.json](../configs/operation_environment_lock.json).
The native adapter refines these declarations in
[native_case_registry.py](../src/embodied_harness/native_case_registry.py).

| Benchmark / selector | Cases | Runtime reference | Required upstream assets and setup |
|---|---:|---|---|
| CALVIN / `calvin` | 10 | Python 3.10, PyBullet 3.2.7, NumPy 1.26.4 | [Pinned source](https://github.com/mees/calvin/tree/fa03f01f19c65920e18cf37398a9ce859274af76); `calvin_env` submodule and scene assets; helper below creates the simulator config |
| CaP-X–Robosuite / `capx` | 5 | CaP-X dependencies, CUDA/PyTorch, MuJoCo 3.5.0 | [Pinned installation](https://github.com/capgym/cap-x/tree/53e9966d7a8e2fa7494676772bccc35280f5c0ed); robosuite, Panda description, SAM 3 and Contact-GraspNet assets |
| CaP-X–LIBERO-PRO / `capx` | 10 | Isolated Python 3.10, MuJoCo 2.3.7, robosuite 1.4, shared CaP-X services | [LIBERO-PRO source](https://github.com/uynitsuj/LIBERO-PRO/tree/5368540790fde9a18584d64105ec5ac16d8926bd); setup below |
| CaP-X–BEHAVIOR-1K / `capx` | 5 | Isaac Sim 4.5.0, CaP-X's pinned OmniGibson/BDDL, CUDA/PyTorch | Same CaP-X installation, its nested B1K checkout, official scene/robot/task assets under `assets/behavior1k/datasets` |
| CLIPort / `cliport` | 18 | Python 3.10, PyBullet 3.2.7, NumPy 1.26.4 | [Pinned source](https://github.com/cliport/cliport/tree/2be5c47b5bb9bb7040ad90693288b87b1e18e7ad); Ravens assets at `assets/cliport` |
| ManiSkill / `maniskill` | 10 | Python 3.11, SAPIEN 3.0.3, NumPy 2.2.5 | [Pinned installation](https://github.com/haosulab/ManiSkill/tree/42b68244c1497cef889b04c4f4a78aa01c927f4e); download assets for the selected task IDs |
| RLBench / `rlbench` | 10 | Python 3.10, pinned PyRep, CoppeliaSim 4.1.0 | [Pinned installation](https://github.com/stepjam/RLBench/tree/02720bba4c73fe02eb75df946b8791b806028a9d); CoppeliaSim at `assets/rlbench/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04` |
| RoboCasa / `robocasa` | 10 | Python 3.11, MuJoCo 3.3.1, robosuite 1.5.2 | [Pinned installation](https://github.com/robocasa/robocasa/tree/b4684e6ee37d377cc392e98302a6b916d588b415); kitchen assets at `assets/robocasa` |
| RoboCasa365 / `robocasa365` | 10 | Same interpreter and assets as RoboCasa | Same pinned source, with a checkout at `upstreams/robocasa365` |
| RoboTwin 2.0 / `robotwin2` | 50 | Python 3.10, SAPIEN 3.0.0b1, CUDA/PyTorch, pinned cuRobo | [Pinned installation](https://github.com/RoboTwin-Platform/RoboTwin/tree/c3ddfa8b97d5519efa828b075999bd0006778e5e); `assets/robotwin2/objects` and `embodiments` for all selected tasks |
| RoboWits / `robowits` | 10 | Python 3.11, MuJoCo 3.13.0, CUDA/PyTorch | [Pinned installation](https://github.com/UMass-Embodied-AGI/RoboWits/tree/9cc30aedbbebea86b975c0d66ea9ac227ca38535); official HF assets at `assets/robowits/hf_assets` |
| VIMA-Bench / `vimabench` | 10 | Python 3.10, PyBullet 3.2.7, NumPy 1.26.4 | [Pinned installation](https://github.com/vimalabs/VIMABench/tree/97e0af11e126fd477c9d81eeabfb6c739021c680); assets inside `upstreams/vimabench/vima_bench/tasks/assets` |
| VLABench / `vlabench` | 25 | Python 3.10, MuJoCo 3.2.2, NumPy 1.25, pinned RRT dependency | [Pinned installation](https://github.com/OpenMOSS/VLABench/tree/cf588fe60c0c7282174fe979f5913170cfe69017); object assets at `assets/vlabench` |

These groups use eleven adapter IDs. The three CaP-X groups share `capx` and are
distinguished by each task's `ARENA_REPORTING_BENCHMARK`. `--benchmark capx` selects
all twenty tasks; use exact task IDs to select a subset. The source lock also
retains standalone BEHAVIOR-1K and RoboDojo adapter declarations; neither has
standalone cases in this release.

### Bind a native interpreter

After following the selected upstream installation, choose its interpreter:

```bash
export EMBODIED_ARENA_NATIVE_PYTHON_RLBENCH=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/environments/rlbench/bin/python"
```

Replace `RLBENCH` with the upper-case benchmark ID. Without an override, the
launcher searches `external/environments/<benchmark>/bin/python` and the recorded
environment layout. Do not install the entire arena package into every native
environment: the launcher supplies its adapter source paths, while the simulator
keeps its own dependency versions.

RLBench additionally requires the official CoppeliaSim library paths before
building/installing the pinned PyRep dependency:

```bash
export COPPELIASIM_ROOT=\
"$EMBODIED_ARENA_EXTERNAL_ROOT/assets/rlbench/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
```

MuJoCo-based environments generally use `MUJOCO_GL=egl` for headless rendering;
THOR and CoppeliaSim require their upstream-supported X/display configuration.
Use a driver compatible with each simulator and its installed CUDA/PyTorch build.

### CALVIN helper

After checking out CALVIN and the pinned `calvin_env` dependency, the included
helper installs its small native environment and composes the evaluation scene
configuration. It requires `uv` on `PATH`.

```bash
python scripts/materialize_calvin_native_runtime.py \
  --external-root "$EMBODIED_ARENA_EXTERNAL_ROOT"
```

For an already compatible environment, add `--skip-install`. Its output includes
`assets/calvin/dataset/.hydra/merged_config.yaml` and
`environments/calvin/requirements-native.lock`. This agent-controlled route does
not require the MCIL policy checkpoint or CALVIN training trajectories.

### BEHAVIOR-1K and CaP-X configuration

Apply the supplied patches to the exact checkouts before recording a runtime:

```bash
git -C "$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/behavior1k" apply \
  "$ARENA_CODE/configs/patches/behavior1k/archived-asset-compatibility.patch"
git -C "$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/capx" apply \
  "$ARENA_CODE/configs/patches/capx/base-inplace-rotation.patch"
git -C "$EMBODIED_ARENA_EXTERNAL_ROOT/upstreams/capx/capx/third_party/b1k" apply \
  "$ARENA_CODE/configs/patches/capx/object-level-joints.patch"
```

Apply only the rows for the benchmark being installed. CaP-X's nested B1K checkout
must be at `272ec5ca9936453c4a8fd335c4dfba61245e33ca`. The expected revisions and
patch hashes are also recorded in `operation_environment_lock.json`.

BEHAVIOR's native launcher retains the upstream terms/key requirements. After
obtaining the official authorization and assets, the included local helper can
record the installation's confirmation:

```bash
python scripts/confirm_runtime_terms.py --scope behavior1k
python scripts/materialize_behavior1k_assets.py --help
```

The asset helper downloads the official packages; it does not obtain or accept a
license on the user's behalf. A single-case asset subset is insufficient for all
five selected CaP-X BEHAVIOR tasks. Keep the official key local at
`assets/behavior1k/datasets/omnigibson.key`.

BEHAVIOR also expects an NVIDIA Vulkan ICD manifest at
`assets/behavior1k/runtime/nvidia_icd.json` and a local OmniGibson 4.5 Kit config at
`assets/behavior1k/runtime/omnigibson_4_5_0_no_flowusd_no_xr.kit`. Prepare the Kit
config from the installed OmniGibson template, retaining `isaacsim.exp.base`
and omitting the FlowUSD/XR extensions for the native headless route. Use the
host's actual NVIDIA driver library in the ICD manifest.

For CaP-X, copy the supplied configuration after creating its environment:

```bash
cp configs/capx_runtime_config.yaml \
  "$EMBODIED_ARENA_EXTERNAL_ROOT/environments/capx/runtime_config.yaml"
```

[capx_minimal_closure.json](../configs/capx_minimal_closure.json) records the
robosuite/Panda resource locations. The native asset declarations additionally
specify SAM 3 revision `3c879f39826c281e95690f02c7821c4de09afae7` under
`assets/capx/sam3-hf`, and the Contact-GraspNet checkpoint under the CaP-X source
tree. Obtain both from their official providers. CaP-X's own perception services
are distinct from W2's optional YOLOE backend; W2 still defaults to perception off.

### CaP-X LIBERO-PRO isolation

Keep LIBERO's older simulator dependencies separate from the Robosuite route.
Clone LIBERO-PRO at `5368540790fde9a18584d64105ec5ac16d8926bd` to
`upstreams/libero-pro`, and [its robosuite fork](https://github.com/Max-Fu/robosuite/tree/a498b087d4bc5a3981e3d27030d09bc537a537f3)
at `a498b087d4bc5a3981e3d27030d09bc537a537f3` to `upstreams/capx-libero-robosuite`.
Install their upstream requirements in `environments/capx_libero_pro`, using
Python 3.10 and the version constraints in
[capx_libero_pro.txt](../configs/environments/capx_libero_pro.txt).
Install the two pinned source checkouts and CaP-X's shared Python dependencies
there; keep SAM 3, Contact-GraspNet, and IK service execution in `environments/capx`.
The task configuration selects both interpreters explicitly.

Obtain the official LIBERO scene assets and initial-state files. Preserve the
`libero/bddl_files`, `libero/init_files`, and `libero/assets` layout inside the
pinned LIBERO-PRO checkout. Create its local path configuration:

```bash
python - <<'PY'
import json, os
from pathlib import Path
external = Path(os.environ['EMBODIED_ARENA_EXTERNAL_ROOT'])
libero = external / 'upstreams/libero-pro/libero'
config = external / 'assets/capx/libero-config'
config.mkdir(parents=True, exist_ok=True)
(config / 'config.yaml').write_text(json.dumps({
    'benchmark_root': str(libero), 'bddl_files': str(libero / 'bddl_files'),
    'init_states': str(libero / 'init_files'), 'assets': str(libero / 'assets'),
    'datasets': str(external / 'assets/capx/libero-demonstrations'),
}, indent=2))
PY
```

Demonstration trajectories are not consumed by this agent-controlled route.
The ten task configurations are included under `configs/capx/env_configs/libero`.
They use the official initial state at index 0; the CaP-X trial seed is 1 because
that wrapper indexes initial states from 1. The retained reset fix prevents a
second reset from discarding the selected state and preserves native settling.

### Record and check the installed W4 runtime

Keep the native installer's dependency locks at the paths returned by:

```bash
python - rlbench <<'PY'
import sys
from embodied_harness.native_case_registry import native_runtime_lock_declarations
for path in native_runtime_lock_declarations(sys.argv[1]):
    print(path)
PY
```

For Conda-based installations these comprise an explicit Conda package export
(`requirements.lock`) and the pip overlay inventory (`pip-overlay.lock.json`).
Record the installed environment, for example:

```bash
export ARENA_NATIVE_PREFIX="$EMBODIED_ARENA_EXTERNAL_ROOT/environments/rlbench"
conda list --prefix "$ARENA_NATIVE_PREFIX" --explicit \
  > "$ARENA_NATIVE_PREFIX/requirements.lock"
"$ARENA_NATIVE_PREFIX/bin/python" -m pip inspect \
  > "$ARENA_NATIVE_PREFIX/pip-overlay.lock.json"
```

The pip inventory records installed package versions and direct-source metadata;
it is not itself a `pip install -r` file. For the pip-based CLIPort/RoboTwin routes,
retain the dependency installer's resolved, hashed requirements as
`environments/<benchmark>/requirements.lock`. CALVIN's helper writes its own
`requirements-native.lock`. The reference inventories in this repository describe
versions; they do not substitute for the locks of a newly installed environment.

Create a receipt on the machine where the simulator will run:

```bash
python scripts/seal_native_runtime.py rlbench \
  --external-root "$EMBODIED_ARENA_EXTERNAL_ROOT" --full-content
```

CaP-X requires one receipt for each installed route. Add `--task` to bind the same
interpreter, configuration roots, and receipt directory as an actual dataset case:

```bash
python scripts/seal_native_runtime.py capx --data-root ../data \
  --task capx-robosuite--cube_lifting--seed-50 \
  --external-root "$EMBODIED_ARENA_EXTERNAL_ROOT" --full-content
python scripts/seal_native_runtime.py capx --data-root ../data \
  --task capx-libero-pro--libero_10--task-1--init-0 \
  --external-root "$EMBODIED_ARENA_EXTERNAL_ROOT" --full-content
```

Repeat with a `capx_behavior1k` task ID from `arena list` after installing that
route. Route receipts live under `artifacts/native-runtime-receipts/<group>/capx/`.
For RoboWits, use `--task` as well to bind the supplied episode corrections.

Use the selected benchmark ID, and inspect the printed `completeness` fields.
The launch checks require matching source revisions/declared patches, required
assets, dependency-lock records, and full runtime/asset content hashes. A receipt
without `--full-content` does not satisfy the content checks. Receipts belong to
the local installation and are not included in the release. Recreate them after
changing the installation. Receipt creation and a scheduling dry run do not
replace an actual environment reset/step check.

## Run through the same harness and loop

First validate the task data and inspect a plan; these commands do not start a
simulator or contact a model:

```bash
arena validate --data-root ../data --hashes
arena run --data-root ../data --benchmark scienceworld_text -- \
  --provider openai-compatible --model MODEL_ID \
  --base-url https://api.example.com/v1 --dry-run --output-dir outputs/plan-w3
```

Once that benchmark's native environment is installed, remove `--dry-run`, supply
your local credentials, and start with one worker:

```bash
arena run --data-root ../data --benchmark scienceworld_text -- \
  --provider openai-compatible --model MODEL_ID --env-file .env \
  --workers 1 --retry-http --max-http-rounds 3 --output-dir outputs/scienceworld
```

For GPU tasks, also specify the actual GPU IDs and memory available to the
scheduler, such as `--gpus 0 --gpu-memory-gb 48` on a 48 GiB device. Native GPU
requirements differ by benchmark; increasing API concurrency does not increase
simulator capacity. Select one task with `--case TASK_ID`, or a whole installed
track with `--wave W3` / `--wave W4`. Credentials, outputs, caches, downloaded
assets, and installed environments remain local.
For multiple models and repetitions, use `arena batch` and the configuration
example in the [README](../README.md#multi-model-batches). GPU memory reservations,
exclusive-device requirements, and VLABench's host-memory admission remain active
across the batch; API worker limits do not bypass these resource checks.
