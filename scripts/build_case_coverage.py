#!/usr/bin/env python3
"""Publish the recording coverage separately from evaluation outcomes."""
import argparse,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'public'
parser=argparse.ArgumentParser()
parser.add_argument('--docs-only',action='store_true')
args=parser.parse_args()
data=json.loads((PUB/'data/case-comparisons.json').read_text())
health=json.loads((PUB/'data/report/case-execution-health.json').read_text())
overview=json.loads((PUB/'data/report/overview-provenance.json').read_text())
rows=[]
for g in data['groups']:
    for rid in g['runIds']:
        r=data['runs'][rid]
        kind='action_log_animation' if r.get('mediaKind')=='Animated execution log' else 'evaluation_scene_preview' if r.get('video') or r.get('animation') else 'independent_demo' if r.get('demoMedia') else 'scene_only'
        rows.append(dict(case_id=g['id'],benchmark=g['benchmark'],task_id=g['taskId'],model=r['model'],
                         record_id=rid,report_sha256=r['sourceReportSha256'],outcome=r['outcome'],
                         stored_media=kind,
                         display=('scene_only' if g['sceneContext']['kind']=='scene' else 'text_result') if kind in ('action_log_animation','scene_only') else kind,
                         recording=r.get('video') or r.get('animation') or '',
                         scene_reference=g['sceneContext']['sourceRunId'],details=r['detailUrl']))
if not args.docs_only:
    with (PUB/'data/case-recording-coverage.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
counts={kind:sum(r['stored_media']==kind for r in rows) for kind in ['evaluation_scene_preview','action_log_animation','independent_demo','scene_only']}
(ROOT/'docs/CASE_MEDIA.md').write_text(f'''# 🎬 Case media coverage

The [Case Atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) contains **181 cases**, each with **ten model records**: 72 fixed-input cases and 109 execution tasks. Visual tasks use matching recordings or a labeled shared scene. Text-only tasks display recorded outcomes and decisions.

| Execution evidence | Model–task records |
| :--- | ---: |
| Evaluation scene previews: video, observations or state replay | {counts['evaluation_scene_preview']} |
| Decision-log animations, archived but not used as scene previews | {counts['action_log_animation']} |
| Independent same-task demo, excluded from scores | {counts['independent_demo']} |
| Result and log; labeled scene context where available | {counts['scene_only']} |

**Scene only** identifies task context, not a recording of that model's behavior. The image source is attached to the exact task and identified in its detail view. This also applies to DeepSeek-1017 on CALVIN sequence 0557: its outcome remains **Budget reached**, while its preview uses the shared scene instead of a log animation.

**API error** marks a terminal API interruption and counts as an unsuccessful attempt. The atlas contains **{health['errorCount']} API errors**: {health['byPhase']['before_actions']} before actions and {health['byPhase']['after_actions']} after execution began. Including the additional HumanCLAW records below gives **{health['totalErrorCount']} final API errors** across the two collections. These remain distinct from **Runtime error** and **Budget reached**. Recovered API errors followed by continued execution are not classified as terminal interruptions. Original outcomes, native success flags and scores are preserved; no reruns are scheduled. [Exact case IDs, model IDs and recorded budgets](../public/data/report/case-api-errors.csv) · [Audit metadata](../public/data/report/case-execution-health.json).

On 2026-10-23, **39 delivered Qwen HumanCLAW attempts** replaced the earlier records, with **38 matching observation-frame replays**. The six HumanCLAW tasks already in the atlas now use the corresponding 18 replacement records. All three Qwen agents on the homepage chair task have their own replay. The overview contains **{overview['apiErrorCount']} API error**: Qwen Max on the RoboCasa365 cutting-board task, counted as unsuccessful.

The [complete HumanCLAW refresh index](https://embodied-agent-arena.github.io/embodied-agent-arena/humanclaw-refresh.html) also includes tasks outside those six atlas comparisons. It contains **two additional provider-interrupted Qwen 3.8 Max attempts**: `102816756:20:couch` (`qwen-02`) and `106878960_174887073:26:toilet` (`qwen-08`), both at seed 42. Both retain their recorded observation replays and are final **API errors**, counted as unsuccessful. These are separate from the 21 atlas interruptions. Qwen Max on `105515211_173104185:14:toilet` (`qwen-07`) reached its budget without saving observation frames and therefore uses a labeled **Scene only** reference. [Replacement results and provenance](../public/data/report/humanclaw-qwen-refresh.json).

On 2026-10-22, **30 overlooked original recordings** were recovered from the supplied archives by matching the task ID, model, and report SHA-256. Their scores were unchanged.

- [Full model–task coverage CSV](../public/data/case-recording-coverage.csv)
- [Recovered recording provenance](../public/data/case-recovery-audit.json)
- [Ten-model case index](../public/data/case-comparisons.json)
''')
print(counts)
