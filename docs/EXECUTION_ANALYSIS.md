# Reproducing the execution analysis

**Task completion compares ten models**, including Gemini&nbsp;4&nbsp;Argon: 1,570 outcomes on the same 157 Planning tasks. Their combined successes cover 142 tasks; 15 are unsuccessful for all ten models.

**Time, token use and recovery** cover all ten models: 1,570 adopted Planning episodes. The explorer has 46 shared tasks × 10 models (460 records), including 189 detailed action/feedback traces and 271 round summaries. All 109 Gemini 4 Argon gallery executions are matched to the supplied logs by report SHA-256; prior retries are excluded.

**Failure cost:** Astra’s 31 unsuccessful episodes consume 18,326,033 tokens, versus 14,670,756 across 126 successes. The mean unsuccessful episode uses **5.1×** as many tokens (591,162 vs. 116,435). These totals use all 157 recorded episodes.

## Published evidence

- [Planning outcomes CSV](../public/data/report/planning-outcomes.csv) and [JSON](../public/data/report/planning-outcomes.json): ten-model outcomes joined by exact task ID, with report digests.
- [Planning episode metrics](../public/data/report/planning-episodes.csv): one row per episode across ten models, including outcome, duration, usage and recovery counts.
- [Analysis data](../public/data/report/metrics.json): ten-model capability profiles and execution-log summaries.
- [Complete execution index](../public/data/execution/complete-index.json) and [detailed trace index](../public/data/execution/index.json): individual report links, video links, settings, round timestamps, executed code, arguments, feedback and verifier outcomes.
- [Editable figures](https://embodied-agent-arena.github.io/embodied-agent-arena/assets/analysis/editable-figures.zip): current manuscript figures and execution-analysis figures, with each experiment's scope in the included manifest.
- [Overview provenance](../public/data/report/overview-provenance.json): all 110 cells in the 11-task × 10-model overview, including native identity and matching media hashes.

## Build the report

Python 3.10+; standard library only. The public CSV/JSON files contain the adopted ten-model evaluation, including the 39 Qwen HumanCLAW replacement records. Original author-supplied archives are retained separately.

After an audit has adopted the current data, refresh the report tables with the corresponding analysis directory:

```bash
python scripts/build_technical_report.py --analysis /path/to/current-adopted-analysis
```

The directory must contain `report-metrics.json` and `planning-episodes.json` matching the current published records. The renderer checks all 1,570 episode identities, outcomes, usage and durations, plus the ten-model aggregates, before writing. It rejects an older nine-model analysis, an analysis preceding the HumanCLAW update, or mismatched report hashes. It updates only the three table bodies in `public/technical-report.html`, preserving the current prose. It does not import data or rewrite the homepage or claim ledger; the checked-in template is used only if the report page is absent.

The [Qwen HumanCLAW update](HUMANCLAW_QWEN_REFRESH.md) documents the current replacement import, native identities, observation replays and final error policy. Audit and adopt any further data changes before using the renderer.

### Historical import stages

These commands document earlier stages of the evaluation. They are not a current publication sequence: run them only in an isolated checkout when reproducing those historical snapshots.

```bash
# Historical nine-model archive analysis.
python scripts/analyze_execution_report.py \
  --archive /path/to/all-model-logs-20261019.zip \
  --output /path/to/nine-model-analysis

# Historical Gemini 4 Argon outcome and log import.
python scripts/build_planning_outcomes.py \
  --input /path/to/gemini4argon-results-20261021
python scripts/import_argon_logs.py \
  --archive /path/to/gemini4argon-logs-20261021.zip \
  --output /path/to/argon-analysis
```

The Gemini 4 Argon importer checked each adopted Planning report against its outcome digest and all 109 gallery reports against their execution records. The [import audit](../public/data/argon-logs-audit.json) preserves that stage's archive digest, selected attempts and code-cell alignment. The subsequent Qwen HumanCLAW import replaced 39 records and recomputed their aggregate results.

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

Episodes terminated by API or execution errors count as unsuccessful final outcomes. The error category remains available in each record; no reruns are scheduled. The [Qwen HumanCLAW update](HUMANCLAW_QWEN_REFRESH.md) gives the final per-task records and observation replays.

- **Token use:** recorded input plus output. Missing usage is excluded, including 536 transport attempts across 34 Qwen HumanCLAW records. Failure-token share divides usage in unsuccessful episodes by usage in all episodes with recorded usage.
- **Recovery:** native final success after an earlier execution error. Budget guards and timed-out rounds are excluded from execution errors.
- **Elapsed time:** reported episode duration, including setup where recorded; campaign elapsed time is used if supplied, otherwise the last recorded loop duration. Qwen HumanCLAW durations use the worker scheduling interval, including setup, teardown and retry waits.
- **Video time:** the recording's own playback clock, separate from execution wall time. Per-episode settings and budgets accompany each record.

Manipulation traces retain their recorded observation configurations; the main results use the adopted evaluation assignments. The [protocol audit](W4_PROTOCOL_AUDIT.md) details their schedules and execution ceilings. Case examples explain individual behavior; benchmark success rates use the complete evaluation population.

## Preview

```bash
python -m http.server 8785 --directory public
```

Open <http://127.0.0.1:8785/#technical-report> or <http://127.0.0.1:8785/#execution-analysis>.
