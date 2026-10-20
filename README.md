<h1 align="center">
  <img src="public/assets/logo-transparent.png" width="64" height="64" align="absmiddle" alt="">&nbsp; Embodied Agent Arena
</h1>

<h3 align="center">
  Are Frontier VLM Agents Ready to Be Robot Generalists?<br>
  <em>An Empirical Study with the Embodied Agent Arena</em>
</h3>

<p align="center">
  <a href="https://embodied-agent-arena.github.io/embodied-agent-arena/"><img src="https://img.shields.io/badge/Project_Page-Visit-4F7C8A?style=flat-square&amp;logo=googlechrome&amp;logoColor=white" height="24" align="absmiddle" alt="Project Page"></a>
  &nbsp;
  <a href="https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf"><img src="https://img.shields.io/badge/Paper-PDF-B45C7E?style=flat-square&amp;logo=readthedocs&amp;logoColor=white" height="24" align="absmiddle" alt="Paper PDF"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/uuu-Quant/Embodied-Agent-Arena"><img src="https://img.shields.io/badge/Dataset-Hugging_Face-8174A8?style=flat-square&amp;logo=huggingface&amp;logoColor=FFD21E" height="24" align="absmiddle" alt="Dataset on Hugging Face"></a>
</p>

<div align="center">

<p>
Haojian Huang<sup>1,3</sup> · Pukun Zhao<sup>3</sup> · Zexi Li<sup>2,3</sup> · Yehang Zhang<sup>1,3</sup> · Yangkai Wei<sup>3</sup><br>
Wenqian Li<sup>3</sup> · Han Yang<sup>3</sup> · Kaiwen Zhou<sup>3</sup> · Ying-Cong Chen<sup>1,3,†</sup> · Yinchuan Li<sup>3,†</sup>
</p>

<p><sup>1</sup> HKUST (GZ) &nbsp; <sup>2</sup> The Chinese University of Hong Kong &nbsp; <sup>3</sup> Knowin AI<br>
<sup>†</sup> Corresponding authors</p>

</div>

---

## Overview

Embodied Agent Arena compares **GPT-6 Astra, GPT-6 Sol, Claude Fable 5.1, Claude Opus 5.5, DeepSeek 4.1 Flash-1017, Gemini 3.8 Flash, Qwen 3.8 Max, Qwen 3.5 397B-A17B, and Qwen 3.5 27B** on **1,000 cases across five robotic capabilities**: Geometry, Spatial Reasoning, Affordance, Task Planning, and Manipulation. It combines 32 established sources with **GeoProbe**, our 168-case geometric-estimation benchmark. A minimal harness retains source-native observations and helpers while leaving task interpretation, reasoning, and action selection to the model.

### Cross-Model Results

<!-- BEGIN RESULT_TABLE -->
| Model | Rot.&nbsp;↓ | Trans.&nbsp;↓ | Spatial&nbsp;↑ | AbsRel&nbsp;↓ | Box&nbsp;↑ | Polygon&nbsp;↑ | Plan&nbsp;↑ | Manip.&nbsp;↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT&#8209;6&nbsp;Astra | **17.2**&nbsp;±&nbsp;6.7 | **106.9**&nbsp;±&nbsp;36.0 | **69.4**&nbsp;±&nbsp;1.0 | 0.214&nbsp;±&nbsp;0.016 | 0.286&nbsp;±&nbsp;0.010 | 0.379 | **80.3** | 41.8 |
| GPT&#8209;6&nbsp;Sol | 27.0&nbsp;±&nbsp;1.5 | 151.9&nbsp;±&nbsp;6.2 | 60.7&nbsp;±&nbsp;2.5 | 0.273&nbsp;±&nbsp;0.016 | 0.264&nbsp;±&nbsp;0.010 | 0.325 | 44.6 | 19.8 |
| Claude&nbsp;Fable&nbsp;5.1 | 24.2&nbsp;±&nbsp;8.3 | 116.6&nbsp;±&nbsp;42.6 | 62.0&nbsp;±&nbsp;4.6 | 0.185&nbsp;±&nbsp;0.011 | 0.284&nbsp;±&nbsp;0.013 | 0.328 | 63.7 | 24.7 |
| Qwen&nbsp;3.8&nbsp;Max | 20.5&nbsp;±&nbsp;3.5 | 138.1&nbsp;±&nbsp;10.2 | 68.9&nbsp;±&nbsp;1.1 | 0.206&nbsp;±&nbsp;0.033 | 0.272&nbsp;±&nbsp;0.017 | 0.307 | 35.7 | 9.3 |
| Gemini&nbsp;3.8&nbsp;Flash | 23.7&nbsp;±&nbsp;5.6 | 138.0&nbsp;±&nbsp;41.7 | 65.4&nbsp;±&nbsp;1.3 | **0.161**&nbsp;±&nbsp;0.028 | 0.251&nbsp;±&nbsp;0.081 | 0.291 | 48.4 | 18.7 |
| Qwen&nbsp;3.5&nbsp;397B&#8209;A17B | 38.4&nbsp;±&nbsp;1.1 | 249.8&nbsp;±&nbsp;10.2 | 50.6&nbsp;±&nbsp;1.8 | 0.251&nbsp;±&nbsp;0.027 | 0.116&nbsp;±&nbsp;0.014 | 0.092 | 17.8 | 9.9 |
| Qwen&nbsp;3.5&nbsp;27B | 38.4&nbsp;±&nbsp;3.6 | 213.7&nbsp;±&nbsp;12.6 | 48.0&nbsp;±&nbsp;0.9 | 0.288&nbsp;±&nbsp;0.023 | 0.013&nbsp;±&nbsp;0.006 | 0.051 | 19.1 | 11.0 |
| Claude&nbsp;Opus&nbsp;5.5 | 23.4&nbsp;±&nbsp;3.7 | 135.8&nbsp;±&nbsp;12.7 | 67.5&nbsp;±&nbsp;5.1 | 0.184&nbsp;±&nbsp;0.040 | **0.311**&nbsp;±&nbsp;0.020 | **0.397** | 48.4 | **47.8** |
| DeepSeek&nbsp;4.1&nbsp;Flash&#8209;1017 | 30.2&nbsp;±&nbsp;3.9 | 182.0&nbsp;±&nbsp;29.0 | 51.6&nbsp;±&nbsp;1.5 | 0.309&nbsp;±&nbsp;0.035 | 0.217&nbsp;±&nbsp;0.011 | 0.249 | 31.2 | 20.9 |
<!-- END RESULT_TABLE -->

Mean ± run SD: five runs (Fable: three). Polygon, Planning, and Manipulation use one run.

Mean scores from [Table 2](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf#page=7); **bold** marks the best mean. Affordance reports box–mask and direct-polygon IoU. [Model IDs and evaluation settings](https://embodied-agent-arena.github.io/embodied-agent-arena/model-guide.html#protocol).

[Full Results & Evaluation Settings](https://embodied-agent-arena.github.io/embodied-agent-arena/#results) · [Results CSV](public/data/model-comparison.csv)

![Task Planning and Manipulation recordings from nine models](public/assets/reference-arena.gif)

![Benchmark construction: 1,000 cases, five capability domains, GeoProbe, and the unified evaluation harness](public/assets/construction.webp)

## Execution Analysis

Across the same 157 Planning tasks, **19 defeat all nine models**. Combining their successes covers 138 tasks, only 12 more than Astra alone. Unsuccessful episodes consume **55.5–86.9% of recorded tokens** per model. Explore the [interactive comparisons](https://embodied-agent-arena.github.io/embodied-agent-arena/#technical-report) and [videos with action traces](https://embodied-agent-arena.github.io/embodied-agent-arena/#execution-analysis), or read the [complete analysis and measurement definitions](https://embodied-agent-arena.github.io/embodied-agent-arena/technical-report.html).

## News

- **2026.10.20** — Compare [nine-model polygon predictions](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=W5-w5_ragnet-02#latest-cases) on identical images: region shape and target selection reveal different failures. Updated manuscript figures show measured run SDs and distinguish observation settings from uncertainty.
- **2026.10.20** — Explore [Inside the Execution](https://embodied-agent-arena.github.io/embodied-agent-arena/#execution-analysis): 46 task-matched comparisons and 143 episodes across nine models. Compare videos, goal progress, key decisions and resource use; expand a decision to inspect its actions and feedback. The [technical analysis](https://embodied-agent-arena.github.io/embodied-agent-arena/technical-report.html) examines shared failures, token allocation and recovery.
- **2026.10.19** — Added DeepSeek 4.1 Flash-1017, direct-polygon Affordance results, and completed Opus 5.5 Planning results. The [project page](https://embodied-agent-arena.github.io/embodied-agent-arena/#results), [CSV](https://embodied-agent-arena.github.io/embodied-agent-arena/data/model-comparison.csv), and [author manuscript](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) now cover nine models.

- **2026.10.15** — Expanded the [interactive gallery](https://embodied-agent-arena.github.io/embodied-agent-arena/#latest-cases) with 152 model recordings across task planning and manipulation. Browse eight-model showcases or inspect matching tasks, with instructions and run details.

- **2026.10.12** — Added Claude Opus 5.5 to the quantitative comparison, with repetition statistics and explicit evaluation coverage.

- **2026.10.12** — Explore the [129-case atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) and five new qualitative case figures in the [updated manuscript](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf#page=53).
- **2026.10.05** — Published the [seven-model comparison](#cross-model-results), downloadable results, and a [model selection and evaluation guide](https://embodied-agent-arena.github.io/embodied-agent-arena/model-guide.html).

- **2026.10.05** — The [updated manuscript](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) is available; [arXiv v2](https://arxiv.org/abs/2610.00854v2) includes the revised discussion and model settings.

- **2026.10.03** — Added a visual [overview](#overview) of task planning and manipulation.
- **2026.10.01** — The [paper](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) and [project page](https://embodied-agent-arena.github.io/embodied-agent-arena/) are available. The manuscript has been submitted to arXiv.
- **2026.10.01** — Released the [evaluation harness](#quick-start), multi-model batch runner, and [1,000-case dataset](https://huggingface.co/datasets/uuu-Quant/Embodied-Agent-Arena).

## Milestones

- [x] **Paper and project page** — Study, results, and case studies.
- [x] **Evaluation code** — Agent harness, benchmark adapters, scoring, and installation guides.
- [x] **Benchmark data** — 1,000-case dataset with a pinned download revision.
- [ ] **Reproduction reports** — Per-model evaluation outputs and consolidated analysis reports.
- [ ] **Analysis tools** — Scripts for aggregating results and reproducing the paper's figures.

## 📦 Benchmark

**GeoProbe** evaluates camera motion, object displacement, depth, and scale. Controlled Blender scenes isolate these geometric factors; real images extend the evaluation to natural scenes.

| Capability | Cases | What the agent must do |
| :--- | ---: | :--- |
| **Geometry** | 370 | Estimate camera parameters, depth, motion, and scale; trace spatial paths. |
| **Spatial Reasoning** | 220 | Relate objects and viewpoints, compare distances, and reason about scene layout. |
| **Task Planning** | 157 | Coordinate actions and feedback to satisfy household and scientific goals. |
| **Manipulation** | 183 | Execute placement, insertion, articulation, and sequential control tasks. |
| **Affordance** | 70 | Locate usable contacts and functional regions for an intended action. |

## Quick Start

### 1. Install

Use Linux and Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[offline,test,hub]'
```

For Task Planning and Manipulation, also follow the [interactive environment installation guide](docs/INSTALL.md) for the benchmarks you want to run.

### 2. Download the dataset

Download the [Hugging Face dataset](https://huggingface.co/datasets/uuu-Quant/Embodied-Agent-Arena) at its fixed release revision and validate it:

```bash
python scripts/fetch_data.py \
  --repo-id uuu-Quant/Embodied-Agent-Arena \
  --revision 5f729375eb35163a024625dd8ea72ac57a427a24 \
  --output ../data
arena validate --data-root ../data --hashes
arena list --data-root ../data
```

### 3. Run an evaluation

Copy [.env.example](.env.example) to `.env` and fill in your credentials. Replace `MODEL_ID` with your API model ID:

```bash
arena run --data-root ../data --benchmark w1_vlm_depth -- \
  --provider openai-compatible --model MODEL_ID --env-file .env \
  --workers 1 --retry-http --max-http-rounds 3 --output-dir outputs/evaluation
```

Select tasks with `--wave`, `--benchmark`, or `--case`. Add `--dry-run` to inspect the plan; repeat the command to resume its checkpoint. GPU environments require device and memory settings described in the [installation guide](docs/INSTALL.md).

For **multiple models**, edit [configs/batch.example.json](configs/batch.example.json), then run:

```bash
arena batch --config configs/batch.example.json --output-dir outputs/batch --execute
```

To **score saved offline predictions**, provide JSONL records with `task_id` and `answer` fields:

```bash
arena score --data-root ../data --benchmark w5_umd \
  --predictions predictions.jsonl --output outputs/scores.jsonl
```

## 📖 Citation

If this work is useful for your research, please cite:

<!-- BEGIN CITATION -->
```bibtex
@article{huang2026frontier,
  title={Are Frontier VLM Agents Ready to Be Robot Generalists? An Empirical Study with the Embodied Agent Arena},
  author={Huang, Haojian and Zhao, Pukun and Li, Zexi and Zhang, Yehang and Wei, Yangkai and Li, Wenqian and Yang, Han and Zhou, Kaiwen and Chen, Ying-Cong and Li, Yinchuan},
  journal={arXiv preprint arXiv:2610.00854},
  year={2026}
}
```
<!-- END CITATION -->
