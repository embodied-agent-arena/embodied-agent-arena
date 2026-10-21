# Reproducing the execution analysis

**Task completion now compares ten models**, including Gemini&nbsp;4&nbsp;Argon: 1,570 outcomes on the same 157 Planning tasks. Their combined successes cover 142 tasks; 15 are unsuccessful for all ten models.

**Time, token use and recovery** now cover all ten models: 1,570 adopted Planning episodes. The explorer has 46 shared tasks × 10 models (460 records), including 189 detailed action/feedback traces and 271 round summaries. All 109 Argon gallery executions are matched to the supplied logs by report SHA-256; prior retries are excluded.

## Published evidence

- [Planning outcomes CSV](../public/data/report/planning-outcomes.csv) and [JSON](../public/data/report/planning-outcomes.json): ten-model outcomes joined by exact task ID, with report digests.
- [Planning episode metrics](../public/data/report/planning-episodes.csv): one row per episode across ten models, including outcome, duration, usage and recovery counts.
- [Analysis data](../public/data/report/metrics.json): ten-model capability profiles and execution-log summaries.
- [Complete execution index](../public/data/execution/complete-index.json) and [detailed trace index](../public/data/execution/index.json): individual report links, video links, settings, round timestamps, executed code, arguments, feedback and verifier outcomes.
- [Editable figures](https://embodied-agent-arena.github.io/embodied-agent-arena/assets/analysis/editable-figures.zip): current manuscript figures and execution-analysis figures, with each experiment's scope in the included manifest.
- [Overview provenance](../public/data/report/overview-provenance.json): all 110 cells in the 11-task × 10-model overview, including native identity and matching media hashes.

## Build the report

Python 3.10+; standard library only. The original author-supplied archives are retained separately from the public repository. Use the published CSV/JSON downloads above to inspect the evidence without them.

```bash
# Verify the nine-model archive and derive episode/resource statistics.
python scripts/analyze_execution_report.py \
  --archive /path/to/all-model-logs-20261019.zip \
  --output /path/to/analysis
python scripts/build_technical_report.py --analysis /path/to/analysis

# Verify and join the 157 Argon Planning outcomes (unpacked results archive).
python scripts/build_planning_outcomes.py \
  --input /path/to/gemini4argon-results-20261021

# Add adopted Argon logs to the existing published records and metrics.
python scripts/import_argon_logs.py \
  --archive /path/to/gemini4argon-logs-20261021.zip \
  --output /path/to/argon-analysis
python scripts/refresh_execution_views.py
python scripts/build_technical_report.py --analysis /path/to/argon-analysis
```

The Argon importer checks every adopted Planning report against the existing outcome digest, and all 109 gallery reports against their execution records. It preserves the other nine models' measurements. The [import audit](../public/data/argon-logs-audit.json) records the archive digest, selected attempts and code-cell alignment. Run the original nine-model publication first when rebuilding from scratch; the Argon import extends those published records.

For the public case index and task map, no external archive is needed:

```bash
python scripts/build_case_atlas.py
python scripts/build_overview_matrix.py
```

## Episode traces

```bash
python scripts/build_execution_analysis.py --source /path/to/extracted-execution-materials
```

The trace-builder input contains `cases.json`, `records.json`, `reports/<case-id>.json` and `evidence/<case-id>.json`. The builder aligns code cells to rounds by timestamps and execution status; unmatched cells remain separately available. This is a lower-level import step for prepared trace materials. The published trace JSON files already contain the evidence displayed in the explorer.

## Measurement definitions

- **Token use:** recorded input plus output. Missing usage is excluded. Failure-token share divides usage in unsuccessful episodes by usage in all episodes with recorded usage.
- **Recovery:** native final success after an earlier execution error. Budget guards and timed-out rounds are excluded from execution errors.
- **Elapsed time:** reported episode duration, including setup where recorded; campaign elapsed time is used if supplied, otherwise the last recorded loop duration. This is not model inference latency.
- **Video time:** the recording's own playback clock, separate from execution wall time. Per-episode settings and budgets accompany each record.

Manipulation traces include later observation configurations; the main results retain the adopted Baseline population. Case examples explain individual behavior; benchmark success rates use the complete evaluation population.

## Preview

```bash
python -m http.server 8785 --directory public
```

Open <http://127.0.0.1:8785/#technical-report> or <http://127.0.0.1:8785/#execution-analysis>.
