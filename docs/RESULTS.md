# Results and reproduction

The **21 October 2026 author manuscript** and the public results compare **ten models**, including **Gemini 4 Argon (`gemini-4-argon`)**. The published arXiv version has its own version history; the current author manuscript is linked below.

## Results and evidence

| Resource | Coverage | Download / view |
| :--- | :--- | :--- |
| Main comparison | 10 models × 8 metrics, matching manuscript Table 2 | [CSV](../public/data/model-comparison.csv) · [Model IDs and settings](https://embodied-agent-arena.github.io/embodied-agent-arena/model-guide.html#protocol) |
| Source-level results | Benchmark-specific means, run SDs and run counts, including Argon | [CSV](../public/data/source-results-mean-sd.csv) |
| Repeated-run details | Run means and valid sample counts for the repeated metrics | [CSV](../public/data/complementary-results-mean-sd.csv) |
| Planning outcomes | 10 models × the same 157 tasks; 1,570 recorded outcomes | [CSV](../public/data/report/planning-outcomes.csv) · [JSON](../public/data/report/planning-outcomes.json) · [Analysis](https://embodied-agent-arena.github.io/embodied-agent-arena/technical-report.html) |
| Case atlas | 181 comparisons: 72 perception/affordance cases and 109 execution tasks; ten-model results in each | [Browse](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) · [Static cases](../public/data/case-studies.json) · [Execution comparisons](../public/data/case-comparisons.json) |
| Overview task map | 11 fixed tasks × 10 models; 90 matching scene previews | [Task descriptions](RECORDED_TASKS.md) · [Identity and media manifest](../public/data/report/overview-provenance.json) |
| Execution analysis | Ten-model logs: 1,570 Planning episodes; 46 selected tasks with 460 records, including 189 detailed action/feedback traces | [Planning episode CSV](../public/data/report/planning-episodes.csv) · [Selected episode CSV](../public/data/execution/complete-episodes.csv) · [Definitions and rebuild instructions](EXECUTION_ANALYSIS.md) |
| Manuscript and figures | Latest author PDF/source; native draw.io figures with vector and raster exports | [PDF](https://embodied-agent-arena.github.io/embodied-agent-arena/paper.pdf) · [LaTeX source](https://embodied-agent-arena.github.io/embodied-agent-arena/author-source.zip) · [Figure ZIP](https://embodied-agent-arena.github.io/embodied-agent-arena/assets/analysis/editable-figures.zip) · [Figure manifest](../public/assets/analysis/figure-manifest.json) |

**Argon coverage:** all five main-result domains, repeated metrics, polygon predictions, Planning outcomes and the case gallery are included. Detailed logs are joined to all 109 Argon gallery records by report hash. Time, token use and recovery now compare all ten models; video playback and episode timestamps retain their separate clocks.

**Statistics:** repeated metrics use mean ± run SD (five runs; Fable: three). Polygon IoU, Planning and Manipulation use one run. Qwen-Max's 11 missing polygon predictions count as errors in the 70-case denominator. Per-case examples and aggregate scores serve different purposes; each execution record retains its own settings and budget.

## Video supplement

The [21 October media supplement](../public/data/video-supplement-audit.json) matches all 70 assignments in the first seven overview columns. **65 have media**: 27 original video excerpts, 27 saved-observation sequences, 4 verified action replays and 7 saved-state renders. The full 11-column overview now has **90 scene previews across 110 results**. Media types and provenance are available in each updated case record.

The five remaining gaps in the supplied seven-task set are Astra/CALVIN, Astra/RLBench, and the three Qwen models/HumanCLAW. The first two lack the selected run's original recording; the three Qwen executions failed before actions began. The other four overview columns are outside this supplement's scope.

```bash
python scripts/import_video_supplement.py --archive /path/to/ten-model-video-supplement-20261021.zip
python scripts/build_overview_matrix.py
python scripts/build_case_atlas.py
```

## Rebuild the public outputs

Run from the repository root with Python 3.10+. These builders use the standard library and the checked-in data:

```bash
# Main HTML/README tables, model guide, CSV, and citation metadata.
python scripts/build_public_results.py
# Exact-task overview and its task documentation.
python scripts/build_overview_matrix.py
# Searchable case atlas from the same ten-model case manifests.
python scripts/build_case_atlas.py
```

To regenerate the execution report from the original supplied logs and add the Argon outcome join, follow [Execution Analysis](EXECUTION_ANALYSIS.md). The published CSV/JSON records can be inspected without those archives. The original full log archives are not included in the repository.

To refresh the figure bundle from exported manuscript figures:

```bash
python scripts/package_figure_sources.py --figures /path/to/manuscript-figures
```

The figure bundle contains editable **draw.io** files and matching **SVG/PDF/PNG** exports. Figures 3–8 include Argon where the experiment includes it; the original observation-setting ablation keeps its original model cohort. The token-allocation figure covers ten models; the progress figure compares Opus 5.5, Argon and the matching Gemini 3.8 or DeepSeek-1017 execution. The bundle provides editable figure sources; automated regeneration of every paper figure from raw logs is not yet packaged.

## Check consistency

```bash
python -m unittest discover -s tests -p 'test_public*.py'
node tests/check_case_rendering.cjs
node tests/check_execution_rendering.cjs
python -m http.server 8785 --directory public
```

Open <http://127.0.0.1:8785/>. The checks compare model scores, repeated-run statistics, task identities and media/report matches against the published manifests.
