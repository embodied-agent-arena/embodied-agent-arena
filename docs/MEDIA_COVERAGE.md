# Media coverage

## Homepage overview

**11 fixed tasks × 10 models = 110 evaluated results.** Media is matched by model, native task ID, record ID and report SHA-256.

- **107 original-evaluation previews**: footage, saved observations, verified action replay or saved-state render. The type is shown in the case details.
- **3 same-task demos**: Astra on CALVIN, RLBench and CLIPort. Each has a separate report digest, is visibly labeled and is excluded from evaluation.
- All three Qwen agents on HumanCLAW’s “Reach the chair” task now have their own **Observed frames** replay from the delivered replacement attempt. These supersede the earlier API-interrupted records.

The overview’s one **API error**, Qwen Max on the RoboCasa365 cutting-board task, remains a final unsuccessful attempt with its available recording. API errors are distinct from runtime errors and budget stops. Across the full atlas and HumanCLAW refresh index, **23 terminal API interruptions count as final failures**: 21 in the atlas and two additional Qwen Max cases. [Error inventory](../public/data/report/case-api-errors.csv).

The Qwen 3.5 27B RoboCasa365 preview is a slideshow of **24 original input RGB observations**. Its case detail lets readers select a round; intervening actions are not depicted. HumanCLAW observation replays likewise show saved observations rather than continuous action footage.

[Overview manifest](../public/data/report/overview-provenance.json) · [HumanCLAW replacement provenance](../public/data/report/humanclaw-qwen-refresh.json) · [Earlier supplement audit](../public/data/homepage-supplement-audit.json)

## Execution-library coverage

The 46-task, 460-record execution library contains **226 original-evaluation scene previews**. The other **234 records lack original scene footage**:

| Available evidence | Records |
| :--- | ---: |
| Result and log, without original scene footage | 228 |
| Separate same-task demo only | 3 |
| API error before actions, without original scene footage | 3 |

These categories describe media availability; they do not change an episode’s recorded outcome. Results and logs remain accessible. Where used, **Scene only** identifies an explicitly labeled image from the same task, not footage of the missing model run. No reruns are scheduled.

The [media-availability inventory](../public/data/execution/missing-original-media.csv) gives the model, benchmark, native task ID, record and report digest for every record without original footage.
