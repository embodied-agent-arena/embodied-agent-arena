# 🎬 Case media coverage

The [Case Atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) contains **181 cases**, each with **ten model records**: 72 fixed-input cases and 109 execution tasks. Visual tasks use matching recordings or a labeled shared scene. Text-only tasks display recorded outcomes and decisions.

| Execution evidence | Model–task records |
| :--- | ---: |
| Evaluation scene previews: video, observations or state replay | 362 |
| Decision-log animations, archived but not used as scene previews | 25 |
| Independent same-task demo, excluded from scores | 3 |
| Result and log; labeled scene context where available | 700 |

**Scene only** identifies task context, not a recording of that model's behavior. The image source is attached to the exact task and identified in its detail view. This also applies to DeepSeek-1017 on CALVIN sequence 0557: its outcome remains **Budget reached**, while its preview uses the shared scene instead of a log animation.

**Rerun required** marks an API-interrupted attempt. An audit of all 1,090 execution records identified **21 terminal API failures**: 9 before actions and 12 after execution began. Some were stored as `runtime_failure`; the display now separates them from errors in generated code or the environment. Recovered API errors followed by continued execution are not marked. [Exact case IDs, model IDs and recorded budgets](../public/data/report/case-reruns.csv) · [Audit metadata](../public/data/report/case-execution-health.json).

On 2026-10-23, **39 delivered Qwen HumanCLAW attempts** replaced the earlier records, with **38 matching observation-frame replays**. The six HumanCLAW tasks already in the atlas now use the corresponding 18 replacement records. All three Qwen agents on the homepage chair task have their own replay. The overview has one pending API-interrupted attempt: Qwen Max on the RoboCasa365 cutting-board task.

The [complete HumanCLAW refresh index](https://embodied-agent-arena.github.io/embodied-agent-arena/humanclaw-refresh.html) also includes tasks outside those six atlas comparisons. It contains **two additional provider-interrupted Qwen 3.8 Max attempts**: `102816756:20:couch` (`qwen-02`) and `106878960_174887073:26:toilet` (`qwen-08`), both at seed 42. Both have real observation replays and remain **Rerun required**. These are separate from the 21 atlas interruptions. Qwen Max on `105515211_173104185:14:toilet` (`qwen-07`) reached its budget without saving observation frames and therefore uses a labeled **Scene only** reference. [Replacement results and provenance](../public/data/report/humanclaw-qwen-refresh.json).

On 2026-10-22, **30 overlooked original recordings** were recovered from the supplied archives by matching the task ID, model, and report SHA-256. Their scores were unchanged.

- [Full model–task coverage CSV](../public/data/case-recording-coverage.csv)
- [Recovered recording provenance](../public/data/case-recovery-audit.json)
- [Ten-model case index](../public/data/case-comparisons.json)
