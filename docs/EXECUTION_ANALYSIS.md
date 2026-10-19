# Reproducing the execution analysis

The report joins measured Planning outcomes, token usage, elapsed time and native task feedback. The full Planning cohort contains 1,413 reports (nine models × 157 assignments). The video explorer contains 143 episodes grouped into 46 identical-task comparisons.

## Published evidence

- `public/data/report/planning-episodes.csv`: one row per Planning episode, with source ID, SHA-256, outcome, duration, token usage and recovery counts.
- `public/data/report/metrics.json`: model/source summaries, pairwise overlap and the 157-task outcome matrix.
- `public/data/execution/index.json` and individual JSON files: video links, source IDs, settings, round timestamps, executed code, primitive arguments, feedback and verifier outcomes.
- `public/assets/analysis/`: editable draw.io figures and SVG/PDF/PNG exports.
- `public/data/report/overview-provenance.json`: provenance for the two added overview rows.

## Build the aggregate report

Python 3.9+; standard library only. Supply the original author-provided log archive, retained separately from the website:

```sh
python scripts/analyze_execution_report.py --archive /path/to/all-model-logs-20261019.zip --output /path/to/analysis
python scripts/build_technical_report.py --analysis /path/to/analysis
```

The first command verifies all Planning source-report digests, checks the common task IDs and reproduces the main-table completion counts. The second emits static HTML, CSV/JSON downloads and the interactive report section. Both support `--repo` to select a checkout.

## Build the episode explorer

```sh
python scripts/build_execution_analysis.py --source /path/to/extracted-execution-materials
```

The input directory contains `cases.json`, `records.json`, `reports/<case-id>.json` and `evidence/<case-id>.json`. These are the extracted case/report matches from the supplied archives. `public/data/case-media.json` supplies the published media mapping. The builder aligns code cells to rounds by timestamp and execution status, preserves unmatched cells separately, and writes an alignment audit beside the input directory. The published trace JSON files are sufficient to inspect and reuse the displayed evidence without those private input paths.

## Measurement definitions

Token use is recorded input plus output, excluding missing usage. Failed-token share sums usage in unsuccessful episodes and divides by total recorded usage. Recovery requires native final success after an earlier execution error, excluding budget guards and timed-out rounds. Total episode time can include recorded setup; round timestamps start at the agent loop. Where an import-status duration differs from the matched report's episode duration, the latter is used in case displays and the former is retained separately in the media manifest. The full Planning timing summary retains the durations specified in its CSV and their source fields.

The overview is illustrative. Videos and saved-observation sequences keep their encoded playback clocks; they are not stretched to execution wall time. Manipulation episode settings remain attached to each recording and do not replace the original Baseline aggregate. Missing round values remain unknown in the plots.

## Preview

```sh
python -m http.server 8785 --directory public
```

Open `http://127.0.0.1:8785/#technical-report` or `http://127.0.0.1:8785/?analysis=sequence#execution-analysis`.
