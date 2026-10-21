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
                         display=kind,recording=r.get('video') or r.get('animation') or '',
                         scene_reference=g['sceneContext']['sourceRunId'],details=r['detailUrl']))
with (PUB/'data/case-recording-coverage.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
counts={kind:sum(r['display']==kind for r in rows) for kind in ['evaluation_scene_preview','action_log_animation','independent_demo','scene_only']}
(ROOT/'docs/CASE_MEDIA.md').write_text(f'''# 🎬 Case media coverage

The [Case Atlas](https://embodied-agent-arena.github.io/embodied-agent-arena/case-atlas.html) contains **181 cases**, each with **ten model results**: 72 fixed-input cases and 109 execution tasks. Every execution task has a visual preview or action log. Text-only tasks use their own recorded outcomes and decisions.

| Execution evidence | Model–task records |
| :--- | ---: |
| Evaluation scene previews: video, observations or state replay | {counts['evaluation_scene_preview']} |
| Animated action logs | {counts['action_log_animation']} |
| Independent same-task demo, excluded from scores | {counts['independent_demo']} |
| Result and log; labeled scene context where available | {counts['scene_only']} |

**Scene only** identifies task context, not a recording of that model's behavior. The image source is attached to the task and identified in its detail view. Evaluation results, report hashes, and model-specific video fields remain separate. The three Qwen records on the featured HumanCLAW task stopped at the initial API request and produced no actions.

On 2026-10-22, **30 overlooked original recordings** were recovered from the supplied archives by matching the task ID, model, and report SHA-256. Their scores were unchanged.

- [Full model–task coverage CSV](../public/data/case-recording-coverage.csv)
- [Recovered recording provenance](../public/data/case-recovery-audit.json)
- [Ten-model case index](../public/data/case-comparisons.json)
''')
print(counts)
