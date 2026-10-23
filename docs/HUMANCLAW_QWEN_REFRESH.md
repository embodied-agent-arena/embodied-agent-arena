# Qwen HumanCLAW refresh — 23 October 2026

All 39 latest delivered HumanCLAW attempts replace the previous adopted records for the same three models, 13 native tasks and seed 42. Selection follows the latest recorded attempt, not the best outcome. These replacements are not additional repeated runs.

[View all 39 runs](https://embodied-agent-arena.github.io/embodied-agent-arena/humanclaw-refresh.html) · [Replacement ledger](../public/data/humanclaw-refresh/replacement-manifest.json) · [Independent audit](../public/data/report/planning-comparison-audit.json)

## Results

| Model | HumanCLAW | Planning before | Planning now | Recorded tokens in unsuccessful Planning episodes |
| :--- | ---: | ---: | ---: | ---: |
| Qwen 3.8 Max | 2/13 | 56/157 · 35.7% | 58/157 · 36.9% | 81.2% |
| Qwen 3.5 397B-A17B | 2/13 | 28/157 · 17.8% | 30/157 · 19.1% | 91.9% |
| Qwen 3.5 27B | 0/13 | 30/157 · 19.1% | 30/157 · 19.1% | 94.3% |

The four new successes are navigation tasks. None of the three models completes a sitting task. Two Qwen Max assignments remain provider-interrupted; they retain their assigned denominator and are labeled **Rerun required**, rather than interpreted as evidence of physical task failure. The ten-model success union remains **142/157**, with **15** tasks unsolved by all models and **8** solved by all ten.

Table 2, Figures 3 and 7, the Planning execution profile, the README comparison, interactive charts and affected appendix tables use these replacement results. Other capability scores are unchanged. The current author PDF includes the earlier expanded experimental analysis and this refresh; the previously submitted arXiv archive remains unchanged.

## Footage and remaining gaps

The 38 supplied MP4 files replay the run's actual saved head-camera observations. Each is labeled **Observed frames**. They are indexed at two observations per playback second and encoded at 24 fps; they are not continuous action recordings, and playback duration is not task elapsed time. All 38 files decoded successfully. The homepage chair task now has genuine observed-frame replays for all three Qwen models.

| Model | Native case ID | Seed | Current evidence | Follow-up |
| :--- | :--- | ---: | :--- | :--- |
| Qwen 3.8 Max | `102816756:20:couch` | 42 | Provider interruption; 26 environment steps, 27 observations | Rerun the evaluation |
| Qwen 3.8 Max | `106878960_174887073:26:toilet` | 42 | Provider interruption; 63 environment steps, 64 observations | Rerun the evaluation |
| Qwen 3.8 Max | `105515211_173104185:14:toilet` | 42 | Response budget reached before the first observation | Valid budget stop; capture a replay only if needed for presentation |

[Machine-readable gap list](../public/data/humanclaw-refresh/remaining-gaps.csv). The last row uses a labeled **Scene only** reference from the same task; it does not represent this model's execution. The two provider interruptions above are outside the six HumanCLAW comparisons in the wider case atlas and are visible in the complete refresh page.

## Measurement and provenance

- The result archive's 2,546 files and the video archive's 81 files passed their supplied checksum checks. Model, native task, seed, raw report hash, video hash and ordered observation hashes were matched before adoption.
- The refreshed attempts retain the recorded 101-code-round, 100-environment-step, 400-primitive-call, two-million-token and 10,800-second ceilings. The supplied rerun package also revises shared API/input/transport handling; these are replacement attempts under that supplied harness, not a controlled model-only intervention.
- Recorded token totals include reported input and output usage. Usage is unknown for 536 transport attempts across 34 replacement records; it is not imputed. Qwen Max has usable episode totals for 145/157 Planning assignments; the two smaller Qwen models have 157/157.
- Refreshed durations are worker scheduling intervals and include setup, teardown and retry waits. Other records retain their original duration sources. These quantities are not pure model inference latency.
- Native success flags determine completion. Geometric proximity, a navigation success flag and an actively issued stop are separate conditions; reaching a target alone does not imply a successful episode.
- The independent audit freshly verifies all 39 replacements. It proves the remaining 1,531 rows unchanged against the pre-refresh data and preserves their historical raw-archive verification rather than claiming to have re-read those archives.

## Rebuild media and audit

Extract the two supplied archives, then run from the repository root:

```bash
python scripts/import_humanclaw_qwen_refresh.py \
  --results /path/to/humanclaw-qwen-results-20261023 \
  --videos /path/to/humanclaw-qwen-videos-20261023 \
  --ffmpeg /path/to/ffmpeg
python scripts/audit_case_execution_health.py
python scripts/refresh_execution_views.py
python scripts/build_case_atlas.py
python scripts/build_overview_matrix.py
python scripts/build_public_results.py
```

The importer updates case evidence; aggregate replacement data are checked in separately. `scripts/audit_planning_comparison.py` verifies those aggregates using the original log archives plus `--humanclaw-results`, or an audited `--baseline-public-data` ZIP plus the replacement package. The replacement ledger contains all old and new report hashes. Native draw.io figures and their SVG/PDF/PNG exports are included in the [figure bundle](../public/assets/analysis/editable-figures.zip).
