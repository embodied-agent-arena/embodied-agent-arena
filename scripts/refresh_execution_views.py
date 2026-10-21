#!/usr/bin/env python3
"""Refresh public execution copy, tables and downloads from the checked indices."""
import csv,html,json,re
from pathlib import Path
from public_json import dumps
R=Path(__file__).resolve().parents[1];U=R/'public';D=U/'data/execution'
content=json.loads((R/'scripts/execution-story-content.json').read_text())
indices={}
for name in ['index.json','complete-index.json']:
    d=json.loads((D/name).read_text())
    for s in d['stories']:
        s.update(content.get(s['id'],{}))
        if not s.get('featured'):s['takeaway']=f"{sum(r['success'] for r in s['runs'])} of {len(s['runs'])} recorded models completed this task. Compare actions, feedback and resource use."
    (D/name).write_text(dumps(d)+'\n');indices[name]=d
for filename,key in [('episodes.csv','index.json'),('complete-episodes.csv','complete-index.json')]:
    rows=[]
    for s in indices[key]['stories']:
        for r in s['runs']:
            v=r['stats'];rows.append(dict(case_group=s['id'],benchmark=s['benchmark'],model=r['model'],task_id=r['taskId'],record_id=r['recordId'],report_sha256=r['reportSha256'],outcome=r['outcome'],protocol=r['protocol'],elapsed_seconds=v['elapsedSeconds'],model_calls=v['modelCalls'],rounds=v['rounds'],total_tokens=v['tokens'],completed_subgoals=v.get('completedSubgoals'),required_subgoals=v.get('requiredSubgoals'),environment_steps=v.get('environmentSteps'),native_steps=v.get('nativeSteps'),tool_calls=v.get('toolCalls'),stop_limits=';'.join(r['stopLimits']),evidence_url=r['detailUrl']))
    with (D/filename).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
# Crawlable companion: the same ten choices and evidence links as the explorer.
p=U/'execution-guide.html';old=p.read_text();head=old[:old.index('<body>')]
h=html.escape;index=indices['complete-index.json'];sections=[]
fmt=lambda n:'—' if n is None else f'{n:,.1f}' if isinstance(n,float) else f'{n:,}'
for s in index['stories']:
    rows=[]
    for r in s['runs']:
        v=r['stats'];outcome=f"{v['completedSubgoals']}/{v['requiredSubgoals']} goals" if v.get('completedSubgoals') is not None else 'Completed' if r['success'] else 'Not completed'
        rows.append(f'<tr><th scope="row">{h(r["model"])}</th><td>{outcome}</td><td>{fmt(v["modelCalls"])}</td><td>{fmt(v["elapsedSeconds"])}</td><td>{fmt(v["tokens"])}</td><td><a href="{h(r["detailUrl"])}">Trace JSON</a></td></tr>')
    sections.append(f'<section id="{s["id"]}"><p class="eyebrow">{h(s["benchmark"])} · {h(s["domain"])}</p><h2>{h(s["title"])}</h2><p><strong>Task:</strong> {h(s["instruction"])}</p><p>{h(s["takeaway"])}</p><p>{h(s["answer"])}</p><div class="guide-table"><table><thead><tr><th>Model</th><th>Outcome</th><th>Model calls</th><th>Time (s)</th><th>Tokens</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p><a href="index.html?analysis={s["id"]}#execution-analysis">Watch and inspect this comparison →</a></p></section>')
head=head.replace('compare nine VLM agents','compare ten VLM agents')
p.write_text(head+f'<body><main class="wrap"><a href="index.html#execution-analysis">← Interactive Execution Explorer</a><h1>Inside the Execution</h1><p>{len(index["stories"])} tasks · {index["runCount"]} recorded episodes · 10 models. Each comparison uses the same task; settings and budgets remain attached to its execution record.</p>'+''.join(sections)+'<section><h2>Evidence and Timing</h2><p>Source hashes tie the recorded actions and feedback to each result. Video playback and episode wall time use separate clocks. Native calls, model responses and environment steps are distinct counters.</p><p><a href="data/execution/complete-episodes.csv">Download episode metrics</a> · <a href="data/argon-logs-audit.json">Gemini 4 Argon log verification</a></p></section></main></body></html>')
print(f'Refreshed {len(index["stories"])} exact-task groups and {index["runCount"]} episode records.')
