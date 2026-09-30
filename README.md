# <img src="public/assets/logo-readme.png" width="32" height="32" align="absmiddle" alt=""> Embodied Agent Arena: Are Frontier VLM Agents Ready to Be Robot Generalists?

<div align="center">

<a href="https://embodied-agent-arena.github.io/embodied-agent-arena/"><img src="public/assets/icon-project.svg" width="19" height="19" align="absmiddle" alt=""> Project Page</a> &nbsp; | &nbsp; <a href="https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf"><img src="public/assets/icon-paper.svg" width="19" height="19" align="absmiddle" alt=""> Paper</a>

<p>
Haojian Huang<sup>1,3</sup> · Pukun Zhao<sup>3</sup> · Zexi Li<sup>2</sup> · Yehang Zhang<sup>1,3</sup> · Yangkai Wei<sup>3</sup><br>
Wenqian Li<sup>2</sup> · Han Yang<sup>3</sup> · Kaiwen Zhou<sup>3</sup> · Ying-Cong Chen<sup>1</sup> · Yinchuan Li<sup>3</sup>
</p>

<p><sup>1</sup> HKUST (Guangzhou) &nbsp; <sup>2</sup> The Chinese University of Hong Kong &nbsp; <sup>3</sup> Knowin AI</p>

</div>

---

This is the official repository for [**Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena**](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf).

Embodied Agent Arena evaluates seven frontier vision-language agents on 1,000 cases across Geometry, Spatial Reasoning, Affordance, Task Planning, and Manipulation. It combines 32 established sources with **GeoProbe**, a new 168-case geometric-estimation benchmark. A unified agent harness connects model-generated programs to source environments, and task-level analyses examine how perception, reasoning, and intermediate actions translate into complete robotic tasks.

![Embodied Agent Arena: representative tasks across five robotic capabilities.](public/assets/overview.webp)

## 📦 Benchmark

**GeoProbe** evaluates camera motion, object displacement, depth, and scale. Controlled Blender scenes isolate these geometric factors; real images extend the evaluation to natural scenes.

| Capability | What the agent must do |
| :--- | :--- |
| **Geometry** | Estimate camera parameters, depth, motion, and scale; trace spatial paths. |
| **Spatial Reasoning** | Relate objects and viewpoints, compare distances, and reason about scene layout. |
| **Affordance** | Locate usable contacts and functional regions for an intended action. |
| **Task Planning** | Coordinate actions and feedback to satisfy household and scientific goals. |
| **Manipulation** | Execute placement, insertion, articulation, and sequential control tasks. |

See the [paper](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) for benchmark construction and evaluation protocols.

## 📊 Results

- **Estimation, contact grounding, and planning.** Astra has the lowest error on all five controlled Blender estimation targets, leads valid-contact prediction on UMD and ReasonAff, and leads or ties the best model across all six planning sources.
- **Spatial reference frames.** Astra excels at connecting views, while inferring camera movement and object-facing directions exposes different weaknesses.
- **Progress and task completion.** Accurate traces and longer action sequences can still miss required endpoints or final goals. Household manipulation additionally demands coordinated navigation, object handling, and environment-state changes.

The [project page](https://embodied-agent-arena.github.io/embodied-agent-arena/#findings) provides the seven-agent comparison, task-level results, and recorded case studies.

## Evaluation code

Task definitions, benchmark adapters, persistent agent execution, and scoring for
1,000 cases across W1–W5. The code repository and dataset are distributed separately.

| Track | Cases | Execution |
|---|---:|---|
| W1 Geometry | 370 | Offline RGB inputs |
| W2 Spatial reasoning | 220 | Offline images or videos |
| W3 Task planning | 157 | Interactive environments |
| W4 Manipulation | 183 | Interactive environments |
| W5 Affordance | 70 | Offline RGB inputs |

## Installation

Use Linux and Python 3.10 or newer. Install from this source checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[offline,test]'
```

The offline installation does not install YOLOE or simulator dependencies.
W2 uses `--perception none` by default.

## Dataset

Place the prepared dataset next to the code checkout, or specify its location:

```bash
arena list --data-root ../data
arena validate --data-root ../data --hashes
```

After the Hugging Face repository is published, download an explicit dataset revision:

```bash
python -m pip install -e '.[hub]'
python scripts/fetch_data.py --repo-id OWNER/DATASET --revision COMMIT_SHA --output ../data
```

`cases.jsonl` is the canonical task index. Each task retains its ID, seed, variation,
budget, input references, and evaluation adapter. Reference annotations remain on
the evaluator side of the agent interface. TraceSpatial includes the same 56 scenes
in both 2D and 3D, with RGB as the model input. GT depth is used only by its scorer.

## Run

Inspect a plan without a model request:

```bash
arena run --data-root ../data --benchmark w1_vlm_depth -- \
  --provider openai-compatible --model MODEL_ID \
  --base-url https://api.example.com/v1 --dry-run --output-dir outputs/plan
```

Set credentials in the shell or copy `.env.example` to `.env` and fill it locally.
For an actual API run:

```bash
arena run --data-root ../data --benchmark w1_vlm_depth -- \
  --provider openai-compatible --model MODEL_ID --env-file .env \
  --workers 1 --retry-http --max-http-rounds 3 --output-dir outputs/evaluation
```

Use `--wave W1`, `--benchmark w2_mindcube`, or `--case TASK_ID` to select cases.
API model IDs are supplied by the user. CLI access is available through
`--provider codex-exec --model MODEL_ID`; CLI credentials use the existing login.
Rerun an identical command to resume its checkpoint. HTTP retries preserve the
selected task and per-attempt limits.

## Multi-model batches

Edit [configs/batch.example.json](configs/batch.example.json) with your model IDs,
endpoints, credential-file paths, selected tasks, and available resources:

```bash
# Inspect the plan without starting an evaluation.
arena batch --config configs/batch.example.json --output-dir outputs/batch
# Start, or resume the saved configuration after interruption.
arena batch --config configs/batch.example.json --output-dir outputs/batch --execute
arena batch --output-dir outputs/batch --execute --resume
arena batch --output-dir outputs/batch --status
```

| Configuration | Meaning |
|---|---|
| `models` | Any number of API or CLI model profiles; each has a unique `name` and exact `model` ID |
| `max_workers` | Global concurrent-case limit |
| `workers_per_model`, model `workers` | Default and per-model concurrency ceilings; available capacity is shared dynamically |
| `rounds`, model `rounds` | Independent repetitions per model and task |
| `selection.waves/benchmarks/cases` | Track, adapter ID, or exact task-ID filters |
| `max_attempts`, `retry_on` | Attempt ceiling including the first attempt; default retries transient network failures only |
| `batch_size` | CPU cases per child campaign; default 1 enables immediate reassignment |
| `resources` | GPU IDs, per-device memory capacity, and reserved memory; task exclusivity is retained |
| `benchmark_limits` | Optional concurrency ceilings by adapter or W4 reporting group |
| `budgets` | Optional per-case turn, total-token, timeout, or request-retry overrides |

Per-model overrides also support request timeout and maximum response tokens.
CLI profiles omit API-only endpoint, credential-file, and generation settings;
use `null` for inherited endpoint/credential-file defaults when mixing providers.
Omitted case budgets come from the dataset. W4 defaults to 3,600 response tokens
and a 720-second request ceiling; the other tracks use 1,800 and 360 seconds.
These request ceilings remain subject to each case's phase and overall limits.
Paths in the JSON configuration are relative to that configuration file.

The supervisor invokes the existing campaign, harness, and agent loop. It retains
every attempt, resumes live child processes, and never retries a valid wrong
answer merely because it is wrong. `timeout`, `invalid`, and `runtime` retries
must be enabled explicitly. Authentication/configuration failures pause the
affected model; after fixing its credentials, resume with `--retry-paused`.
The saved plan binds code, data, model settings, and repetition counts. Use a new
output directory to change them. `status.json` records execution progress;
`completed` means a terminal evaluation, not a correct answer.

GPU admission applies across all models, and CaP-X workers receive separate service
ports. Increase `max_workers` and per-model ceilings together to permit more
concurrency; GPU limits may still reduce the number of active cases. The supervisor
runs in the foreground and can be hosted in your usual `tmux` or service session.

## Score saved offline answers

Input JSONL records have the form `{"task_id": "...", "answer": ...}`.
The answer follows that task's submission schema.

```bash
arena score --data-root ../data --benchmark w5_umd \
  --predictions predictions.jsonl --output outputs/scores.jsonl
```

Missing predictions remain explicit unsuccessful submissions. W1 emits the task's
geometric errors, W2 its native answer metrics, and W5 its region/point metrics. W3 and W4
are evaluated inside their native environments, not from arbitrary saved answers.

TraceSpatial scoring runs in a separate CPU process. Create its environment with
`python3 -m venv runtimes/tracespatial/scorer-env` and install
`runtimes/tracespatial/requirements.lock.txt` there. Alternatively set
`EMBODIED_ARENA_TRACESPATIAL_PYTHON` to a compatible interpreter.

## Interactive environments

W3/W4 adapter code and task definitions are included. Simulator checkouts, licensed
assets, native binaries, and environment-specific Python installations are separate.
Follow [the interactive installation guide](docs/INSTALL.md) for all six W3 benchmarks
and thirteen W4 reporting groups, including source revisions, dependencies, asset locations,
interpreter selection, and runtime checks. Install only the environments you use.
`external/w3-sources.lock.json` pins W3 sources;
`external/sources.lock.yaml` pins W4 sources; the CaP-X LIBERO additions and final
harness file hashes are in `configs/w4-release-sources.json`. `configs/environments/` records runtime
package versions. Set `EMBODIED_ARENA_EXTERNAL_ROOT` for another installation root.
Native interpreters can be selected using `EMBODIED_ARENA_NATIVE_PYTHON_<BENCHMARK>`.
For W3, `<BENCHMARK>` is the upper-case entry ID, such as `SCIENCEWORLD_TEXT`;
an explicit `--benchmark-python` in a task takes precedence. ALFRED/ALFWorld share
the upstream ALFWorld checkout at `external/upstreams/alfworld` (`ALFWORLD_ROOT`
overrides it).

GPU tasks require explicit resource settings, for example on a 48 GiB GPU:

```bash
arena run --data-root ../data --wave W4 -- \
  --provider openai-compatible --model MODEL_ID --env-file .env \
  --gpus 0 --gpu-memory-gb 48 --workers 1 --output-dir outputs/w4
```

Use the actual device IDs and capacities of your machine. `--dry-run` checks the
task plan without starting an environment or contacting a model.

Native runtime integrity checks are retained. After installing the corresponding
W4 environment, use `scripts/seal_native_runtime.py --full-content` to create local
runtime receipts, as described in the installation guide.
`--allow-unsealed-runtime` is an explicit development option, not the release default.
A dataset validation or dry run does not certify that a simulator is installed.
The W4 task index uses the final execution configurations, including direct RGB
feedback and native episode budgets. CaP-X–Robosuite (5), CaP-X–LIBERO-PRO (10), and
CaP-X–BEHAVIOR-1K (5) share the `capx` adapter; select individual tasks with `--case`.

## Development

```bash
python -m pytest -q
```

Core modules live in `src/embodied_harness/`, W2 adapters in `src/w2_harness/`,
and shared persistent execution and environment adapters in `runtimes/`.
Task prompts and primitive cards are runtime inputs, including files ending in `.md`.
Project-owned code is MIT licensed. The bundled RoboTracer scorer retains its
Apache-2.0 license. Dataset and simulator terms are separate; see [NOTICE](NOTICE).

## 🛠️ Project Page

The static website is in [`public/`](public/). Edit `index.html`, `styles.css`, and `app.js` for layout, styling, and interactive results; figures are in [`public/assets/`](public/assets/).

From the repository root, start a local preview:

```sh
python3 -m http.server 8000 --directory public
```

Open `http://localhost:8000`. Changes to `public/` on `main` are published automatically to [GitHub Pages](https://embodied-agent-arena.github.io/embodied-agent-arena/) by the repository’s Pages workflow.

## 📖 Citation

If this work is useful for your research, please cite:

```bibtex
@misc{huang2026embodiedagentarena,
  title  = {Are Frontier VLM Agents Ready to Be Robot Generalists?
            An Empirical Study with the Embodied Agent Arena},
  author = {Huang, Haojian and Zhao, Pukun and Li, Zexi and Zhang, Yehang
            and Wei, Yangkai and Li, Wenqian and Yang, Han and Zhou, Kaiwen
            and Chen, Ying-Cong and Li, Yinchuan},
  year   = {2026},
  url    = {https://github.com/embodied-agent-arena/embodied-agent-arena}
}
```
