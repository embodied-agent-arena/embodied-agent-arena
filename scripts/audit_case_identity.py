#!/usr/bin/env python3
"""Verify task/media joins and regroup gallery recordings by native task.

Run after build_execution_analysis.py. The input directory contains the
original case index, reports and evidence; no private paths are published.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--source', type=Path, required=True)
args = ap.parse_args()
pub = Path(__file__).resolve().parents[1] / 'public'
media_path = pub / 'data/case-media.json'
media = json.loads(media_path.read_text())
cases = json.loads((args.source / 'cases.json').read_text())
index = json.loads((pub / 'data/execution/index.json').read_text())
legacy = json.loads((pub / 'data/case-studies.json').read_text())
audit = {}

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def identity(case_id):
    c = cases[case_id]
    m = media['runs'][case_id]
    raw = json.loads((args.source / 'reports' / (case_id + '.json')).read_text())
    d = raw.get('runner_report', raw)
    binding = d.get('pool_binding') or raw.get('pool_binding') or {}
    coordinate = d.get('pool_coordinate') or raw.get('pool_coordinate') or {}
    assert c['task_id'] == m['taskId'], case_id
    detail=json.loads((pub/'data/execution'/f'{case_id}.json').read_text())
    matched=c['source_sha256'] == m['sourceReportSha256']
    if not matched:
        assert not detail['video'] and not detail['poster'], (case_id, 'cross-episode media join')
    actual = d.get('actual_task_ids')
    if actual:
        assert actual == [d['expected_task_id']], (case_id, actual)
        native = {'task': actual[0]}
        summary=d.get('pukun',{}).get('run_summary',{})
        # HumanCLAW deliberately hides evaluator success in agent-facing
        # feedback. Its completed single-task evaluation is authoritative.
        if c['benchmark']=='humanclaw' and matched and summary.get('requested_tasks')==1:
            if summary.get('successes')==1 and summary.get('avg_score')==1:
                m.update(outcome='success',success=True,note='')
                m['outcomeEvidence']={'source':'pukun.run_summary',
                    'requestedTasks':1,'successes':1,'score':1,
                    'reportSha256':c['source_sha256']}
    else:
        assert binding.get('bound'), (case_id, 'unbound native episode')
        # Config hashes, seeds, sequence/episode IDs and revision all count.
        native = {'binding': {k: v for k, v in binding.items()
                              if not k.endswith(('file', 'path'))},
                  'entry': coordinate.get('entry_id'),
                  'revision': coordinate.get('source_revision')}
    source_video = (c.get('media') or {}).get('path', '')
    if source_video and '/tasks/' in source_video:
        # CaP-X uses a canonical task alias; its original wrapper ID is retained
        # in the raw report and video directory.
        assert any('/tasks/' + task + '/' in source_video
                   for task in [c['task_id'], raw.get('task_id', '')] if task), case_id
    for field in ('video', 'poster'):
        asset = m.get(field)
        if asset:
            assert f'/{case_id}/' in asset, (case_id, field)
            assert (pub / asset).is_file(), (case_id, asset)
    return {'taskId': c['task_id'], 'nativeIdentity': native,
            'identitySha256': digest(native), 'reportSha256': c['source_sha256'],
            'sourceVideoSha256': (c.get('media') or {}).get('source_sha256'),
            'video': detail.get('video'), 'mediaKind': detail['mediaKind'],
            'mediaMatchesReport':matched}

for story in index['stories']:
    checks = {r['id']: identity(r['id']) for r in story['runs']}
    assert len({r['taskId'] for r in checks.values()}) == 1, story['id']
    assert len({r['identitySha256'] for r in checks.values()}) == 1, story['id']
    audit[story['id']] = checks

# One common instruction per task; model-specific progress is separate.
by_task = collections.defaultdict(list)
for run_id, run in media['runs'].items():
    if not run.get('demoOnly'):
        by_task[(run['benchmark'], run['taskId'])].append(run_id)
stories = {s['runs'][0]['taskId']: s for s in index['stories']}
legacy_tasks = {c.get('sample'): c for c in legacy if not c.get('predictions')}
old_groups = media['groups']
benchmark_roots = {}
for g in old_groups:
    if g['id'].startswith('showcase-'):
        benchmark_roots[g['benchmark']] = g['id']
primary = {}
for benchmark in benchmark_roots:
    options = [(t, ids) for (b, t), ids in by_task.items() if b == benchmark]
    primary[benchmark] = max(options, key=lambda x: (stories.get(x[0], {}).get('featured', False), len(x[1])))[0]
groups = []
for (benchmark, task_id), ids in by_task.items():
    first = media['runs'][ids[0]]
    s = stories.get(task_id)
    source = legacy_tasks.get(task_id, {})
    title = s.get('taskLabel') if s else None
    title = title or source.get('title') or first['taskLabel']
    instruction = (s or {}).get('instruction') or source.get('question') or first['instruction']
    is_primary = primary.get(benchmark) == task_id
    group_id = benchmark_roots[benchmark] if is_primary else 'task-video-' + digest(task_id)[:10]
    domain_key = 'planning' if ids[0].startswith('W3') else 'manipulation'
    groups.append({'id': group_id, 'domainKey': domain_key,
        'domain': 'Task Planning' if domain_key == 'planning' else 'Manipulation',
        'benchmark': benchmark, 'title': title, 'shortTask': instruction,
        'question': instruction, 'taskId': task_id, 'sample': task_id,
        'protocolNote': 'Model settings and episode budgets are available in each recording’s details.',
        'predictions': [], 'images': [], 'runIds': ids, 'galleryMode': 'same-task',
        'poster': first['poster'], 'executionId': (s or {}).get('id')})
    for rid in ids:
        media['runs'][rid]['comparisonInstruction'] = instruction
        media['runs'][rid]['comparisonTaskLabel'] = title
    assert len({media['runs'][i]['taskId'] for i in ids}) == 1
media['groups'] = groups
media['identityAudit'] = 'data/execution/identity-audit.json'
media_path.write_text(json.dumps(media, ensure_ascii=False, indent=2) + '\n')
(pub / 'data/execution/identity-audit.json').write_text(json.dumps({
    'date': '2026-10-20', 'comparisonCount': len(audit),
    'recordCount': sum(map(len, audit.values())),
    'checks': ['canonical task', 'actual native task or bound episode', 'seed/config/revision where recorded',
               'source report hash', 'source-video task directory', 'published asset ownership'],
    'comparisons': audit}, ensure_ascii=False, separators=(',', ':')) + '\n')
print(f'Verified {len(audit)} execution groups; gallery now has {len(groups)} task-specific groups, '
      f'{sum(len(g["runIds"]) > 1 for g in groups)} with multiple models.')
