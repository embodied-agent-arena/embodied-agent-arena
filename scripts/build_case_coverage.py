#!/usr/bin/env python3
"""Publish the recording coverage separately from evaluation outcomes."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];PUB=ROOT/'public'
data=json.loads((PUB/'data/case-comparisons.json').read_text())
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

**Rerun required** marks an API-interrupted attempt. An audit of all 1,090 execution records identified **39 terminal API failures**: 25 before actions and 14 after execution began. Some were stored as `runtime_failure`; the display now separates them from errors in generated code or the environment. Recovered API errors followed by continued execution are not marked. [Exact case IDs, model IDs and recorded budgets](../public/data/report/case-reruns.csv) · [Audit metadata](../public/data/report/case-execution-health.json).

The original outcomes, report hashes and aggregate scores remain unchanged until replacement runs are supplied. The overview contains four pending attempts: the three Qwen agents on the HumanCLAW chair task and Qwen Max on the RoboCasa365 cutting-board task.

On 2026-10-22, **30 overlooked original recordings** were recovered from the supplied archives by matching the task ID, model, and report SHA-256. Their scores were unchanged.

- [Full model–task coverage CSV](../public/data/case-recording-coverage.csv)
- [Recovered recording provenance](../public/data/case-recovery-audit.json)
- [Ten-model case index](../public/data/case-comparisons.json)
''')
print(counts)
