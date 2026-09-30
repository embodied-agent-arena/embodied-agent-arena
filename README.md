# 🤖 Embodied Agent Arena: Are Frontier VLM Agents Ready to Be Robot Generalists?

<div align="center">

[🌐 Project Page](https://embodied-agent-arena.github.io/embodied-agent-arena/) | [📄 Paper](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) | [📝 OpenReview](https://openreview.net/forum?id=T0b4fgyHFg)

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

The [project page](https://embodied-agent-arena.github.io/embodied-agent-arena/#results) provides the seven-agent comparison, task-level results, and recorded case studies.

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
