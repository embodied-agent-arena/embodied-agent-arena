#!/usr/bin/env python3
"""Check every public Planning outcome against the two supplied log archives."""
import argparse,collections,csv,gzip,hashlib,io,json,zipfile
from pathlib import Path

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--logs',type=Path,required=True);ap.add_argument('--argon-logs',type=Path,required=True)
 ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'public/data/report/planning-comparison-audit.json')
 a=ap.parse_args();u=Path(__file__).resolve().parents[1]/'public'
 data=json.loads((u/'data/report/planning-outcomes.json').read_text());eps=json.loads((u/'data/report/planning-episodes.json').read_text())
 tasks={t['taskId']:t for t in data['tasks']};idx={(e['model'],e['taskId']):e for e in eps}
 assert len(tasks)==157 and len(idx)==len(eps)==1570
 verified=set()
 def check(model,task,outcome,record,blob):
  t=tasks[task];e=idx[model,task];digest=hashlib.sha256(blob).hexdigest();raw=json.loads(blob);report=raw.get('runner_report',raw)
  assert e['recordId']==t['recordIds'][model]==record
  assert digest==e['reportSha256']
  assert report['outcome']==outcome==e['outcome']==t['outcomes'][model]
  assert (outcome=='success')==(model in t['successModels'])
  assert (model,task) not in verified;verified.add((model,task))
 with zipfile.ZipFile(a.logs) as z:
  prefix='all-model-logs-20261019/'
  rows=csv.DictReader(io.StringIO(z.read(prefix+'trials.csv').decode('utf-8-sig')))
  for r in rows:
   if r['dataset']=='main' and r['wave']=='W3':
    blob=gzip.decompress(z.read(prefix+r['report_object']))
    assert hashlib.sha256(blob).hexdigest()==r['source_observed_sha256']
    check(r['model'],r['task_id'],r['outcome'],r['record_id'],blob)
 with zipfile.ZipFile(a.argon_logs) as z:
  prefix=next(n[:-9] for n in z.namelist() if n.endswith('/slots.csv') and not n.startswith('__MACOSX/'))
  rows=csv.DictReader(io.StringIO(z.read(prefix+'slots.csv').decode('utf-8-sig')))
  for r in rows:
   if r['scope']=='w3':
    assert r['adopted']=='True';t=tasks[r['task_id']];blob=gzip.decompress(z.read(prefix+r['report_object']))
    assert hashlib.sha256(blob).hexdigest()==t['argonReportSha256']
    check('gemini4argon',r['task_id'],r['outcome'],t['recordIds']['gemini4argon'],blob)
 assert verified==set(idx)
 def counts(ts):
  c=collections.Counter(('astra' in t['successModels'],'opus55' in t['successModels']) for t in ts)
  return dict(tasks=len(ts),both=c[True,True],astraCompletedOpusIncomplete=c[True,False],opusCompletedAstraIncomplete=c[False,True],neither=c[False,False],astraAloneAmongTen=sum(t['successModels']==['astra'] for t in ts),noneOfTen=sum(not t['successModels'] for t in ts),allTen=sum(len(t['successModels'])==10 for t in ts))
 result=dict(scope='All 157 Planning tasks, one episode per model; not all 1,000 benchmark cases.',models=data['models'],verifiedEpisodes=len(verified),sourceArchives=[a.logs.name,a.argon_logs.name],verification='Every adopted report SHA-256, task ID, record ID and final outcome matches the public episode and task tables.',pair=['astra','opus55'],counts=counts(data['tasks']),bySource={s:counts([t for t in tasks.values() if t['benchmark']==s]) for s in sorted({t['benchmark'] for t in tasks.values()})})
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts']));print('Verified',len(verified),'source reports')
if __name__=='__main__':main()
