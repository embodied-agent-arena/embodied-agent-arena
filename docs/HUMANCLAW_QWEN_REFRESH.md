# Qwen HumanCLAW refresh — 23 October 2026

HumanCLAW evaluates three Qwen models on the same 13 tasks with seed 42. Results use the latest attempt per task and retain one episode per model–task pair.

[View all 39 runs](https://embodied-agent-arena.github.io/embodied-agent-arena/humanclaw-refresh.html) · [Replacement ledger](../public/data/humanclaw-refresh/replacement-manifest.json) · [Independent audit](../public/data/report/planning-comparison-audit.json)

## Results

| Model | HumanCLAW | Planning before | Planning now | Recorded tokens in unsuccessful Planning episodes |
| :--- | ---: | ---: | ---: | ---: |
| Qwen 3.8 Max | 2/13 | 56/157 · 35.7% | 58/157 · 36.9% | 81.2% |
| Qwen 3.5 397B-A17B | 2/13 | 28/157 · 17.8% | 30/157 · 19.1% | 91.9% |
| Qwen 3.5 27B | 0/13 | 30/157 · 19.1% | 30/157 · 19.1% | 94.3% |

The four new successes are navigation tasks. None of the three models completes a sitting task. Two Qwen Max attempts end in provider interruptions. Both count as unsuccessful outcomes and are labeled **API error**, with their original error reasons and partial observation replays retained. No reruns are scheduled. The ten-model success union remains **142/157**, with **15** tasks unsolved by all models and **8** solved by all ten.

Table 2, Figures 3 and 7, the Planning execution profile, the README comparison, interactive charts and affected appendix tables use these replacement results. Other capability scores are unchanged. The current 73-page author PDF includes the expanded experimental analysis and this refresh. This version was submitted as a replacement of arXiv:2610.00854 on 23 October 2026 and is awaiting announcement.

## Footage and final error outcomes

The 38 supplied MP4 files replay the run's actual saved head-camera observations. Each is labeled **Observed frames**. They are indexed at two observations per playback second and encoded at 24 fps; they are not continuous action recordings, and playback duration is not task elapsed time. All 38 files decoded successfully. The homepage chair task now has genuine observed-frame replays for all three Qwen models.

| Model | Native case ID | Seed | Current evidence | Final treatment |
| :--- | :--- | ---: | :--- | :--- |
| Qwen 3.8 Max | `102816756:20:couch` | 42 | Provider interruption; 26 environment steps, 27 observations | Unsuccessful · API error |
| Qwen 3.8 Max | `106878960_174887073:26:toilet` | 42 | Provider interruption; 63 environment steps, 64 observations | Unsuccessful · API error |
| Qwen 3.8 Max | `105515211_173104185:14:toilet` | 42 | Response budget reached before the first observation | Unsuccessful · Budget reached; Scene only |

[Error and media inventory](../public/data/humanclaw-refresh/remaining-gaps.csv). The last row uses a labeled **Scene only** reference from the same task; it does not represent this model's execution. The two provider interruptions above are outside the six HumanCLAW comparisons in the wider case atlas and are visible in the complete refresh page.

## Measurement and provenance

- The result archive's 2,546 files and the video archive's 81 files passed their supplied checksum checks. Model, native task, seed, raw report hash, video hash and ordered observation hashes were matched before adoption.
- The refreshed attempts retain the recorded 101-code-round, 100-environment-step, 400-primitive-call, two-million-token and 10,800-second ceilings. The supplied package also revises shared API, input, and transport handling; those settings are retained in the episode metadata.
- Recorded token totals include reported input and output usage. Usage is unknown for 536 transport attempts across 34 replacement records; it is not imputed. Qwen Max has usable episode totals for 145/157 Planning assignments; the two smaller Qwen models have 157/157.
- Refreshed durations are worker scheduling intervals and include setup, teardown and retry waits. Other records retain their original duration sources. These quantities are not pure model inference latency.
- Native success flags determine completion. Geometric proximity, a navigation success flag and an actively issued stop are separate conditions; reaching a target alone does not imply a successful episode.
- The audit records the 39 replacement reports and confirms that the other 1,531 Planning rows are unchanged. Their original archive checks remain in the provenance record.

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
