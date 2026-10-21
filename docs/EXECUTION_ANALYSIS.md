# Reproducing the execution analysis

**Task completion now compares ten models**, including Gemini 4 Argon: 1,570 outcomes on the same 157 Planning tasks. Their combined successes cover 142 tasks; 15 are unsuccessful for all ten models.

**Time, token use and recovery** come from the nine-model detailed-log cohort: 1,413 Planning episodes. The selected execution explorer contains 46 task groups and 414 records, of which 143 have detailed action traces. Argon's results and recordings appear in the [ten-model Case Atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html); its detailed action/feedback traces are still pending.

## Published evidence

- [Planning outcomes CSV](../public/data/report/planning-outcomes.csv) and [JSON](../public/data/report/planning-outcomes.json): ten-model outcomes joined by exact task ID, with report digests.
- [Planning episode metrics](../public/data/report/planning-episodes.csv): one row per episode in the nine-model log cohort, including outcome, duration, usage and recovery counts.
- [Analysis data](../public/data/report/metrics.json): ten-model capability profiles, plus the scoped nine-model log summaries.
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

# Verify and join the 157 Argon Planning outcomes (unpacked results archive).
python scripts/build_planning_outcomes.py \
  --input /path/to/gemini4argon-results-20261021

# Publish the ten-model outcome report and scoped execution analyses.
python scripts/build_technical_report.py --analysis /path/to/analysis
```

The first command verifies source-report digests and reproduces the nine-model Planning counts. The second verifies the Argon report digests, exact task IDs and 110/157 completion count. The final command writes static HTML, CSV/JSON downloads and the interactive report section. All three support `--repo` to select a checkout.

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
- **Elapsed time:** reported episode duration, including setup where recorded; the last recorded loop duration is used when no episode total exists. This is not model inference latency.
- **Video time:** the recording's own playback clock, separate from execution wall time. Per-episode settings and budgets accompany each record.

Manipulation traces include later observation configurations; the main results retain the adopted Baseline population. Case examples explain individual behavior; benchmark success rates use the complete evaluation population.

## Preview

```bash
python -m http.server 8785 --directory public
```

Open <http://127.0.0.1:8785/#technical-report> or <http://127.0.0.1:8785/#execution-analysis>.
