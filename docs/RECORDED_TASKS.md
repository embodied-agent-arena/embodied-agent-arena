# Tasks in the overview

The [interactive overview](https://embodied-agent-arena.github.io/embodied-agent-arena/#cases) compares **10 models on the same 11 fixed tasks**. Each column uses one native task identity across every model row. There are **110 results and 104 matching evaluation previews**, plus **3 independently recorded same-task demos** (excluded from scores). The remaining **3 records failed before actions began**.

## Fixed tasks

| Benchmark | Task | Compare |
| :--- | :--- | :--- |
| ALFRED | Clean the soap bar and place it in the bathtub. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-alfred_official_visual#latest-cases) |
| ALFWorld | Put a cleaned bar of soap on a counter. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-alfworld_visual#latest-cases) |
| DiscoveryWorld | Excavate three sites, date the artifacts, and place the red flag at the oldest site. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-discoveryworld#latest-cases) |
| HumanCLAW | Find the chair and move until the body is touching the target, then stop. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-humanclaw#latest-cases) |
| CALVIN | Lift the blue block from the sliding cabinet, put it in the drawer, retrieve it, stack it, and switch off the light. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-calvin#latest-cases) |
| RLBench | Close Laptop Lid | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=task-video-1fd3d0c0ff#latest-cases) |
| ManiSkill | Insert the charger into the socket. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-maniskill#latest-cases) |
| CLIPort | packing boxes pairs unseen colors | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=task-video-cefac4d024#latest-cases) |
| RoboCasa | Open the left drawer. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-robocasa#latest-cases) |
| RoboCasa365 | Remove the broccoli and steak from the cutting board to prepare it for cleaning. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-robocasa365#latest-cases) |
| BEHAVIOR-1K | Outfit a basic toolbox with the required tools. | [Ten models](https://embodied-agent-arena.github.io/embodied-agent-arena/?case=showcase-capx_behavior1k#latest-cases) |

## Model rows

1. **GPT-6 Astra** — 8 scene previews; 11 task results.
2. **GPT-6 Sol** — 11 scene previews; 11 task results.
3. **Claude Fable 5.1** — 11 scene previews; 11 task results.
4. **Gemini 3.8 Flash** — 11 scene previews; 11 task results.
5. **Qwen 3.8 Max** — 10 scene previews; 11 task results.
6. **Qwen 3.5 397B-A17B** — 10 scene previews; 11 task results.
7. **Qwen 3.5 27B** — 10 scene previews; 11 task results.
8. **Claude Opus 5.5** — 11 scene previews; 11 task results.
9. **DeepSeek 4.1 Flash-1017** — 11 scene previews; 11 task results.
10. **Gemini 4 Argon** — 11 scene previews; 11 task results.

The [task and media manifest](../public/data/report/overview-provenance.json) records each task ID, native identity hash, model, report digest and matching media digest. The [case index](../public/data/case-comparisons.json) supplies all ten-model comparisons.
