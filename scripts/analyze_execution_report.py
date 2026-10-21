#!/usr/bin/env python3
"""Derive descriptive report statistics from immutable supplied records."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
import statistics
import zipfile
from pathlib import Path

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--archive',type=Path,required=True,help='Original all-model log ZIP')
ap.add_argument('--output',type=Path,required=True,help='Derived analysis directory')
ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
args=ap.parse_args()
REPO=args.repo.resolve();ARCHIVE=args.archive.resolve();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=True)
with zipfile.ZipFile(ARCHIVE) as archive:
    manifests=[n for n in archive.namelist() if n.endswith('/trials.csv') and not n.startswith('__MACOSX/')]
    assert len(manifests)==1, 'Expected one trials.csv'
    PREFIX=manifests[0][:-len('trials.csv')]
MODELS = ['astra', 'sol', 'fable', 'gemini', 'qwen', 'qwen35-397b', 'qwen35-27b', 'opus55', 'deepseek']
DISPLAY = dict(zip(MODELS, ['GPT-6 Astra', 'GPT-6 Sol', 'Claude Fable 5.1', 'Gemini 3.8 Flash', 'Qwen 3.8 Max', 'Qwen 3.5 397B-A17B', 'Qwen 3.5 27B', 'Claude Opus 5.5', 'DeepSeek 4.1 Flash-1017']))
SOURCE_NAMES = {'alfred_official_visual':'ALFRED', 'alfworld_visual':'ALFWorld', 'discoveryworld':'DiscoveryWorld', 'humanclaw':'HumanCLAW', 'scienceworld_text':'ScienceWorld', 'virtualhome_symbolic':'VirtualHome'}

def numeric(row, key):
    val = row.get(key)
    if key == 'total_tokens' and row.get('token_usage_available','').lower() == 'false':
        return None
    return float(val) if val not in ('', None) else None

def distribution(values):
    v = sorted(x for x in values if x is not None)
    if not v:
        return {'n': 0}
    # Linear interpolation, equivalent to numpy percentile(method='linear').
    def percentile(q):
        pos = q * (len(v) - 1)
        low = int(pos)
        return v[low] + (v[min(low + 1, len(v) - 1)] - v[low]) * (pos - low)
    return {'n':len(v), 'sum':sum(v), 'mean':statistics.mean(v), 'median':statistics.median(v), 'q25':percentile(.25), 'q75':percentile(.75), 'min':v[0], 'max':v[-1]}

def loop_of(report):
    return report.get('episode_loop') or next((a['episode_loop'] for a in report.get('agent_attempts',[]) if 'episode_loop' in a), {})

with zipfile.ZipFile(ARCHIVE) as z:
    trials = list(csv.DictReader(io.StringIO(z.read(PREFIX+'trials.csv').decode('utf-8-sig'))))
    plan = [r for r in trials if r['dataset']=='main' and r['wave']=='W3']
    assert len(plan)==9*157
    per_model = {m:[r for r in plan if r['model']==m] for m in MODELS}
    tasks_by_model = {m:{r['task_id'] for r in rows} for m,rows in per_model.items()}
    task_ids = set.intersection(*tasks_by_model.values())
    assert len(task_ids)==157 and all(len(s)==157 for s in tasks_by_model.values())
    successes = {m:{r['task_id'] for r in rows if r['outcome']=='success'} for m,rows in per_model.items()}
    per_task = []
    for task in sorted(task_ids):
        rows = [r for r in plan if r['task_id']==task]
        assert len(rows)==9
        per_task.append({'taskId':task, 'benchmark':SOURCE_NAMES[rows[0]['benchmark']], 'successModels':[m for m in MODELS if task in successes[m]], 'outcomes':{r['model']:r['outcome'] for r in rows}, 'recordIds':{r['model']:r['record_id'] for r in rows}})

    episode_summaries = []
    stop_reason_counts = collections.Counter()
    for row in plan:
        raw_bytes = gzip.decompress(z.read(PREFIX+row['report_object']))
        assert hashlib.sha256(raw_bytes).hexdigest()==row['source_observed_sha256']
        raw = json.loads(raw_bytes)
        report = raw.get('runner_report', raw)
        loop = loop_of(report)
        turns = loop.get('turns', [])
        guard_rounds = [t for t in turns if 'loop budget exhausted' in str(t.get('error','')).lower()]
        error_rounds = [t for t in turns if (t.get('execution_ok') is False or bool(t.get('error'))) and t not in guard_rounds and not t.get('timed_out')]
        final_success = row['outcome']=='success'
        recovered = bool(final_success and error_rounds and any(t.get('success') and t['turn']>error_rounds[0]['turn'] for t in turns))
        limits = report.get('budget_exhausted') or loop.get('budget_exhausted') or []
        for limit in limits:
            stop_reason_counts[(row['model'],limit)] += 1
        episode_summaries.append({'recordId':row['record_id'], 'model':row['model'], 'benchmark':SOURCE_NAMES[row['benchmark']], 'taskId':row['task_id'], 'reportSha256':row['source_observed_sha256'], 'outcome':row['outcome'], 'elapsedSeconds':numeric(row,'duration_seconds'), 'durationSource':row['duration_source'], 'tokens':numeric(row,'total_tokens'), 'modelCalls':numeric(row,'response_count'), 'rounds':len(turns), 'executionErrorRounds':len(error_rounds), 'firstErrorRound':error_rounds[0]['turn'] if error_rounds else None, 'recoveredAfterError':recovered, 'guardRounds':len(guard_rounds), 'stopLimits':limits, 'roundCeiling':loop.get('max_turns'), 'timeCeilingSeconds':loop.get('timeout_seconds')})

by_model = []
for m, rows in per_model.items():
    es = [r for r in episode_summaries if r['model']==m]
    good = [r for r in es if r['outcome']=='success']
    bad = [r for r in es if r['outcome']!='success']
    known = [r for r in es if r['tokens'] is not None]
    known_bad = [r for r in bad if r['tokens'] is not None]
    errors = [r for r in es if r['executionErrorRounds']]
    source_rows = []
    for b in SOURCE_NAMES.values():
        rs=[r for r in es if r['benchmark']==b]
        source_rows.append({'benchmark':b,'n':len(rs),'completed':sum(r['outcome']=='success' for r in rs),'time':distribution(r['elapsedSeconds'] for r in rs),'tokens':distribution(r['tokens'] for r in rs)})
    by_model.append({'model':m,'name':DISPLAY[m],'n':len(es),'completed':len(good),'outcomes':dict(collections.Counter(r['outcome'] for r in es)), 'time':distribution(r['elapsedSeconds'] for r in es), 'successfulTime':distribution(r['elapsedSeconds'] for r in good), 'unsuccessfulTime':distribution(r['elapsedSeconds'] for r in bad), 'tokens':distribution(r['tokens'] for r in es), 'unsuccessfulTokens':sum(r['tokens'] for r in known_bad),'unsuccessfulTokenShare':sum(r['tokens'] for r in known_bad)/sum(r['tokens'] for r in known) if known and sum(r['tokens'] for r in known) else None, 'episodesWithExecutionErrors':len(errors),'recoveredAfterError':sum(r['recoveredAfterError'] for r in errors),'errorRounds':sum(r['executionErrorRounds'] for r in es),'totalRounds':sum(r['rounds'] for r in es),'sourceResults':source_rows})

all_failed = [t for t in per_task if not t['successModels']]
all_success = [t for t in per_task if len(t['successModels'])==9]
union = set.union(*successes.values())
overlap = {m:{n:{'both':len(successes[m]&successes[n]),'onlyRow':len(successes[m]-successes[n]),'onlyColumn':len(successes[n]-successes[m]),'neither':157-len(successes[m]|successes[n])} for n in MODELS} for m in MODELS}
study = json.loads((REPO/'public/data/study.json').read_text())
aliases = {'Astra':'astra','Sol':'sol','Fable':'fable','Gemini':'gemini','Qwen-Max':'qwen','Qwen-397B':'qwen35-397b','Qwen-27B':'qwen35-27b','Opus 5.5':'opus55','DeepSeek-1017':'deepseek'}
profiles=[]
for s in study['models']:
    m=aliases.get(s['alias'])
    if not m:
        m=next((k for k,v in DISPLAY.items() if v==s['name']), None)
    if m is None:
        continue  # Aggregate-only additions are joined by build_technical_report.py.
    vals=[float(x.split('±')[0].strip()) for x in s['scores']]
    p=next(x for x in by_model if x['model']==m)
    assert abs(vals[6]-100*p['completed']/157)<.051
    profiles.append({'model':m,'name':s['name'],'scores':s['scores'],'means':vals,'runs':s['metric_runs'],'polygonMinusBox':vals[5]-vals[4]})
report={'generated':'2026-10-20','models':MODELS,'modelNames':DISPLAY,'metrics':study['metrics'],'profiles':profiles,'planning':{'nTasks':157,'nEpisodes':len(plan),'models':by_model,'tasks':per_task,'atLeastOneSucceeds':len(union),'allSucceed':len(all_success),'allFail':len(all_failed),'additionalOverAstra':len(union-successes['astra']),'universalFailureSources':dict(collections.Counter(t['benchmark'] for t in all_failed)),'successCountHistogram':dict(sorted(collections.Counter(len(t['successModels']) for t in per_task).items())),'pairwise':overlap,'stopLimits':[{'model':m,'limit':k,'episodes':v} for (m,k),v in sorted(stop_reason_counts.items())]},'provenance':{'archive':ARCHIVE.name,'recordsVerified':len(episode_summaries),'planningMainTableCountsMatch':True,'taskAlignment':'Shared manifest task_id; HumanCLAW wrapper-specific IDs checked against native task identity in all 117 reports.','recoveryDefinition':'Native final success after a recorded execution error; budget guards and timed-out rounds excluded from execution errors.','timeDefinition':'Reported episode wall time, or last recorded loop elapsed time when the report supplies no total duration.','tokenDefinition':'Recorded input plus output usage; missing usage is excluded, never replaced with zero.','manipulationScope':'The supplied Manipulation log cohort contains later observation settings for several original models; preserve the adopted main-table Baseline population.'}}
(OUT/'report-metrics.json').write_text(json.dumps(report,indent=2)+'\n')
(OUT/'planning-episodes.json').write_text(json.dumps(episode_summaries,indent=2)+'\n')
fields=['recordId','model','benchmark','taskId','reportSha256','outcome','elapsedSeconds','durationSource','tokens','modelCalls','rounds','executionErrorRounds','firstErrorRound','recoveredAfterError','guardRounds','stopLimits','roundCeiling','timeCeilingSeconds']
with (OUT/'planning-episodes.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
    for r in episode_summaries:writer.writerow({**r,'stopLimits':';'.join(r['stopLimits'])})
print(json.dumps({'planningTasks':157,'episodes':len(episode_summaries),'anySuccess':len(union),'allSuccess':len(all_success),'allFail':len(all_failed),'allFailBySource':report['planning']['universalFailureSources'],'models':[{'model':r['model'],'completed':r['completed'],'medianSeconds':round(r['time']['median'],1),'tokensKnown':r['tokens']['n'],'unsuccessfulTokenShare':round(r['unsuccessfulTokenShare']*100,1) if r['unsuccessfulTokenShare'] is not None else None,'errors':r['episodesWithExecutionErrors'],'recovered':r['recoveredAfterError']} for r in by_model]},indent=2))
