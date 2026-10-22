# Manipulation execution settings

All ten models use the same CALVIN sequence IDs and native completion criteria. The table records the actual ceilings of the adopted runs; these are not observed consumption.

| Model | Rounds | Total tokens | Minutes | Automatic RGB flag | Completed / 10 |
| --- | ---: | ---: | ---: | --- | ---: |
| Astra | 20 | 640,000 | 60 | absent | 3 |
| Sol | 20 | 640,000 | 60 | absent | 0 |
| Fable 5.1 | 16 | 640,000 | 60 | absent | 0 |
| Qwen Max | 20 | 640,000 | 60 | absent | 0 |
| Gemini 3.8 Flash | 20 | 640,000 | 60 | absent | 0 |
| Qwen 397B | 20 | 640,000 | 60 | absent | 0 |
| Qwen 27B | 20 | 640,000 | 60 | absent | 0 |
| Opus 5.5 | 40 | 1,280,000 | 120 | present | 1 |
| DeepSeek-1017 | 40 | 1,280,000 | 120 | present | 1 |
| Gemini 4 Argon | 40 | 1,280,000 | 120 | present | 4 |

The first seven models retain the source-native observation schedule. Opus 5.5, DeepSeek-1017 and Gemini 4 Argon receive automatic RGB feedback each decision round. An absent automatic-RGB flag does not imply that native observation tools are unavailable.

All 100 report hashes were checked before deriving the [CALVIN progress and recovery analysis](SPATIAL_RECOVERY.md). Its mean prefixes and full-sequence counts reproduce the adopted CALVIN results. [Event summaries, budgets and hashes](../public/data/spatial-recovery/calvin-episodes.json) support the per-model counts.

The all-model log archive also contains later Manipulation runs for several original models. A `dataset=main` label alone does not identify the adopted aggregate population. Selected case videos remain linked to their own execution records; they do not replace the adopted CALVIN episodes used in Figure 8.

[Earlier four-model configuration audit](../public/data/w4-protocol-audit.json) retains archive member paths for the original settings check.
