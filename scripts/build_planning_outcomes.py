#!/usr/bin/env python3
"""Join Argon's outcome reports to the existing 157-task Planning comparison.

This does not infer fine-grained traces from videos, nor alter the nine-model
log-derived resource/recovery cohort. Supply the unpacked Argon results archive.
"""
from pathlib import Path
import argparse,copy,csv,gzip,hashlib,json,collections
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--input',type=Path,required=True)
ap.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
a=ap.parse_args();u=a.repo/'public';old=json.loads((u/'data/report/metrics.json').read_text())
raw=[r for r in json.loads((a.input/'selected-results.json').read_text()) if r['scope']=='w3']
by_id={r['task_id']:r for r in raw};tasks=copy.deepcopy(old['planning']['tasks'])
assert len(raw)==len(by_id)==len(tasks)==157
assert set(by_id)=={t['taskId'] for t in tasks},'Task populations must match exactly'
models=old['models']+['gemini4argon'];assert len(set(models))==10
names={**old['shortNames'],'gemini':'Gemini 3.8','gemini4argon':'Gemini 4 Argon'}
gallery=json.loads((u/'data/case-comparisons.json').read_text());case_ids={g['taskId']:g['id'] for g in gallery['groups']}
for t in tasks:
 r=by_id[t['taskId']];content=gzip.decompress((a.input/r['report_archive']).read_bytes());sha=hashlib.sha256(content).hexdigest()
 assert sha==r['source_sha256'];report=json.loads(content);report=report.get('runner_report',report)
 assert report['outcome']==r['outcome']
 t['outcomes']['gemini4argon']=r['outcome']
 if r['outcome']=='success':t['successModels'].append('gemini4argon')
 t['argonReportSha256']=sha;t['caseId']=case_ids.get(t['taskId'])
 t['recordIds']['gemini4argon']='argon-'+sha[:16]
completed=sum(r['outcome']=='success' for r in raw);assert completed==110
study=json.loads((u/'data/study.json').read_text());argon=next(m for m in study['models'] if m['identifier']=='gemini-4-argon')
assert round(completed/157*100,1)==float(argon['scores'][6])
sources=list(dict.fromkeys(t['benchmark'] for t in tasks))
summary=[dict(model=m,completed=sum(m in t['successModels'] for t in tasks),sourceResults=[dict(benchmark=b,n=sum(t['benchmark']==b for t in tasks),completed=sum(t['benchmark']==b and m in t['successModels'] for t in tasks)) for b in sources]) for m in models]
result=dict(models=models,shortNames=names,tasks=tasks,sourceResults=summary,nTasks=157,nReports=1570,atLeastOneSucceeds=sum(bool(t['successModels']) for t in tasks),allFail=sum(not t['successModels'] for t in tasks),allSucceed=sum(len(t['successModels'])==10 for t in tasks),provenance={'argonArchive':'gemini4argon-results-20261021.zip','verifiedArgonReports':157,'identity':'Exact manifest task_id match for every task','baseCohort':'data/report/metrics.json: nine original models','scope':'Final task outcomes. Step-by-step traces remain the nine-model cohort.'})
(u/'data/report/planning-outcomes.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
with (u/'data/report/planning-outcomes.csv').open('w', newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['task_id','benchmark','model','outcome','record_id','argon_report_sha256'],lineterminator='\n')
 writer.writeheader()
 for t in tasks:
  for m in models:
   writer.writerow(dict(task_id=t['taskId'],benchmark=t['benchmark'],model=m,outcome=t['outcomes'][m],record_id=t['recordIds'][m],argon_report_sha256=t['argonReportSha256'] if m=='gemini4argon' else ''))
print({k:result[k] for k in ['nReports','atLeastOneSucceeds','allFail','allSucceed']})
