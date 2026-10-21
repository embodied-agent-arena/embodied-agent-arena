# Manipulation protocol provenance check

The paper uses a common task inventory, native tools and task-success criteria. That does not establish identical observation schedules or execution ceilings across evaluation batches.

## Verified CALVIN records

All ten CALVIN task IDs are shared across the four cohorts below. Every report was checked against its supplied SHA-256. Values are configured ceilings, not observed consumption.

| Model | Rounds | Total tokens | Trial minutes | Direct RGB flag | Completed / 10 |
| --- | ---: | ---: | ---: | --- | ---: |
| astra | 20 | 640,000 | 60 | absent | 3 |
| opus55 | 40 | 1,280,000 | 120 | present | 1 |
| deepseek | 40 | 1,280,000 | 120 | present | 1 |
| gemini4argon | 40 | 1,280,000 | 120 | present | 4 |

Astra's ten records reproduce the retained CALVIN Baseline result (3/10). The absence of the automatic RGB flag does not mean the model has no access to RGB; native observation tools remain a separate interface. Opus 5.5 records also carry the direct-RGB flag, so a blanket “eight native-schedule models versus two per-round RGB models” description is unsupported.

## Aggregate scope

The all-model log archive includes later Manipulation runs for several original models. Its `dataset=main` label alone does not identify the paper's retained Baseline population. Those runs must not replace retained aggregate values without matching the adopted manifest. This check changes no scores.

The Baseline versus RGB-expanded comparison is a comparison of recorded execution protocols, with both feedback and execution-resource differences. It should not be described as an isolated observation-frequency intervention under identical budgets. The documented expanded ceilings also describe the supplied DeepSeek-1017 and Gemini 4 Argon CALVIN reports.

[Per-task evidence and report hashes](../public/data/w4-protocol-audit.json) retain archive member paths. A fully identical-settings claim requires the corresponding matched run manifests and reports; it cannot be established from a shared interface or shared task denominator alone.
