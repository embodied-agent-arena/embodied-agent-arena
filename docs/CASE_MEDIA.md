# 🎬 Case media coverage

The [Case Atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) contains **181 cases**, each with **ten model results**: 72 fixed-input cases and 109 execution tasks. Every execution task has a visual preview or action log. Text-only tasks use their own recorded outcomes and decisions.

| Execution evidence | Model–task records |
| :--- | ---: |
| Evaluation scene previews: video, observations or state replay | 346 |
| Animated action logs | 25 |
| Independent same-task demo, excluded from scores | 3 |
| Result and log; labeled scene context where available | 716 |

**Scene only** identifies task context, not a recording of that model's behavior. The image source is attached to the task and identified in its detail view. Evaluation results, report hashes, and model-specific video fields remain separate. The three Qwen records on the featured HumanCLAW task stopped at the initial API request and produced no actions.

On 2026-10-22, **30 overlooked original recordings** were recovered from the supplied archives by matching the task ID, model, and report SHA-256. Their scores were unchanged.

- [Full model–task coverage CSV](../public/data/case-recording-coverage.csv)
- [Recovered recording provenance](../public/data/case-recovery-audit.json)
- [Ten-model case index](../public/data/case-comparisons.json)
