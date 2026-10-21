#!/usr/bin/env python3
"""Join adopted Gemini 4 Argon logs to existing, hash-verified task comparisons."""
import argparse,collections,csv,gzip,hashlib,io,json,re,statistics,zipfile
from pathlib import Path
from execution_trace import build,loop_of
from public_json import dumps as trace_json

ROOT=Path(__file__).resolve().parents[1]
MODEL='gemini4argon'; NAME='Gemini 4 Argon'
SOURCES={'alfred_official_visual':'ALFRED','alfworld_visual':'ALFWorld','discoveryworld':'DiscoveryWorld','humanclaw':'HumanCLAW','scienceworld_text':'ScienceWorld','virtualhome_symbolic':'VirtualHome'}

def read(p):return json.loads(p.read_text())
def write(p,d):
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(trace_json(d)+'\n')
def csv_write(p,rows,fields):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:w.writerow({k:';'.join(r[k]) if isinstance(r.get(k),list) else r.get(k) for k in fields})
def distribution(values):
    v=sorted(x for x in values if x is not None)
    if not v:return {'n':0}
    def quantile(q):
        pos=q*(len(v)-1);i=int(pos);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(pos-i)
    return dict(n=len(v),sum=sum(v),mean=statistics.mean(v),median=statistics.median(v),q25=quantile(.25),q75=quantile(.75),min=v[0],max=v[-1])
def text_content(content):
    return content if isinstance(content,str) else '\n'.join(x.get('text','') for x in content if isinstance(x,dict))
def transcript_feedback(messages):
    result={};code=''
    for m in messages:
        text=text_content(m.get('content',''))
        if m.get('role')=='assistant':
            code=re.sub(r'^```(?:python)?\s*|\s*```$','',text.strip())
        elif m.get('role')=='user' and 'Public feedback:' in text:
            value=json.JSONDecoder().raw_decode(text.split('Public feedback:',1)[1].lstrip())[0]
            if 'turn' in value:result[value['turn']]={'feedback':value,'executed_code':code}
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--repo',type=Path,default=ROOT)
    args=ap.parse_args();u=args.repo/'public';w=args.output;w.mkdir(parents=True,exist_ok=True)
    comparisons=read(u/'data/case-comparisons.json');runs={r['id']:r for r in comparisons['runs'].values() if r['modelKey']==MODEL}
    outcomes=read(u/'data/report/planning-outcomes.json');planning={t['taskId']:t for t in outcomes['tasks']}
    archive_hash=hashlib.file_digest(args.archive.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(args.archive.read_bytes()).hexdigest()
    records={};cases={};media={};reports={};slots_by_task={};evidence_audit=[];planning_rows=[]
    with zipfile.ZipFile(args.archive) as z:
        p=next(n[:-len('slots.csv')] for n in z.namelist() if n.endswith('/slots.csv') and not n.startswith('__MACOSX/'))
        slots=list(csv.DictReader(io.StringIO(z.read(p+'slots.csv').decode('utf-8-sig'))))
        slots_by_task={s['task_id']:s for s in slots if s['scope'] in ['w3','w4']}
        assert len(slots_by_task)==340 and all(s['adopted']=='True' for s in slots_by_task.values())
        target_ids={slots_by_task[r['taskId']]['attempt_id'] for r in runs.values()}
        files=collections.defaultdict(list)
        for f in csv.DictReader(io.TextIOWrapper(z.open(p+'files.csv'),encoding='utf-8-sig')):
            if f['attempt_id'] in target_ids:files[f['attempt_id']].append(f)
        def unpack(path):return gzip.decompress(z.read(p+path))
        def loadfile(f):
            blob=unpack(f['bundle_path']);assert hashlib.sha256(blob).hexdigest()==f['unpacked_sha256'];return blob
        # Verify all adopted Planning reports against the already published outcome join.
        for task,t in planning.items():
            s=slots_by_task[task];blob=unpack(s['report_object']);digest=hashlib.sha256(blob).hexdigest()
            assert digest==t['argonReportSha256'];assert s['outcome']==t['outcomes'][MODEL]
            raw=json.loads(blob);d=raw.get('runner_report',raw);loop=loop_of(d);turns=loop.get('turns',[]);usage=json.loads(s['usage'])
            guards=[t for t in turns if 'loop budget exhausted' in str(t.get('error','')).lower()]
            errors=[t for t in turns if (t.get('execution_ok') is False or bool(t.get('error'))) and t not in guards and not t.get('timed_out')]
            recovered=bool(s['outcome']=='success' and errors and any(t.get('success') and t['turn']>errors[0]['turn'] for t in turns))
            duration=raw.get('duration_seconds') or d.get('duration_seconds')
            duration_source='report'
            if duration is None:
                duration=float(s['elapsed_seconds']) if s['elapsed_seconds'] else (turns[-1].get('elapsed_seconds') if turns else None)
                duration_source='campaign' if s['elapsed_seconds'] else 'loop'
            planning_rows.append(dict(recordId=t['recordIds'][MODEL],model=MODEL,benchmark=t['benchmark'],taskId=task,reportSha256=digest,outcome=s['outcome'],elapsedSeconds=duration,durationSource=duration_source,tokens=float(s['total_tokens']) if usage.get('token_usage_available') and s['total_tokens'] else None,modelCalls=float(s['response_count']) if s['response_count'] else None,rounds=len(turns),executionErrorRounds=len(errors),firstErrorRound=errors[0]['turn'] if errors else None,recoveredAfterError=recovered,guardRounds=len(guards),stopLimits=d.get('budget_exhausted') or loop.get('budget_exhausted') or [],roundCeiling=loop.get('max_turns'),timeCeilingSeconds=loop.get('timeout_seconds')))
        for rid,r in runs.items():
            s=slots_by_task[r['taskId']];blob=unpack(s['report_object']);digest=hashlib.sha256(blob).hexdigest();assert digest==r['sourceReportSha256'];assert s['outcome']==r['outcome']
            raw=json.loads(blob);reports[rid]=raw
            case=dict(source_sha256=digest,model=MODEL,task_id=r['taskId'],outcome=r['outcome'],success=r['success'],benchmark=s['benchmark'],wave=s['scope'].upper(),media_kind='native_video')
            cases[rid]=case;records[digest]=dict(source_sha256=digest,source_observed_sha256=digest,record_id=r['recordId']);media[rid]=r
            evidence=dict(report_sha256=digest,traces=[],turn_feedback=[],responses=[]);feedback={};codes={};verified=[];transcript=None;seen=set()
            for f in files[s['attempt_id']]:
                name=Path(f['source']).name
                relevant=name=='trace.json' or re.fullmatch(r'turn_\d+_feedback\.json',name) or re.fullmatch(r'turn_\d+\.py',name) or name=='model_transcript.json' or (f['role']=='execution_log' and '/api_requests/' in f['source'] and re.fullmatch(r'[0-9a-f]{32}\.json',name))
                if not relevant or (name,f['unpacked_sha256']) in seen:continue
                seen.add((name,f['unpacked_sha256']));b=loadfile(f);verified.append({'name':name,'sha256':f['unpacked_sha256']})
                if name=='trace.json':evidence['traces'].append({'trace':json.loads(b)})
                elif name=='model_transcript.json':transcript=json.loads(b)
                elif re.fullmatch(r'turn_\d+_feedback\.json',name):
                    fb=json.loads(b);n=fb['turn'];assert n not in feedback or feedback[n]['feedback']==fb;feedback[n]={'feedback':fb}
                elif re.fullmatch(r'turn_\d+\.py',name):codes[int(re.search(r'\d+',name)[0])]=b.decode()
                else:
                    meta=json.loads(b)
                    if meta.get('latency_seconds') is not None:evidence['responses'].append({k:meta[k] for k in ['request_id','latency_seconds'] if k in meta})
            if transcript:
                for n,fb in transcript_feedback(transcript).items():
                    if n not in feedback:feedback[n]=fb
                    elif 'executed_code' not in feedback[n]:feedback[n]['executed_code']=fb['executed_code']
            for n,fb in sorted(feedback.items()):
                if n in codes:fb['executed_code']=codes[n]
                evidence['turn_feedback'].append(fb)
            for folder,value in [('reports',raw),('evidence',evidence)]:write(w/folder/(rid+'.json'),value)
            evidence_audit.append(dict(runId=rid,taskId=r['taskId'],recordId=r['recordId'],reportSha256=digest,adoptedAttemptId=s['attempt_id'],slotId=s['slot_id'],historicalAttempts=int(s['attempt_count'])-1,sourceFiles=verified,feedbackRounds=len(feedback),nativeTraceFiles=len(evidence['traces'])))
    write(w/'cases.json',cases);write(w/'records.json',list(records.values()))
    details={};align=[];staged=w/'traces';staged.mkdir(exist_ok=True)
    for rid,r in runs.items():
        _,audit=build(rid,{},cases=cases,records=records,media=media,work=w,out=staged)
        d=read(staged/(rid+'.json'));old=read(u/r['detailUrl'])
        d['detailUrl']=r['detailUrl'];d['stats']['elapsedSeconds']=r['elapsedSeconds']
        d.update({k:r[k] for k in ['video','animation','poster','mediaKind','mediaNote','mediaProvenance','mediaDurationSeconds','mediaReportSha256'] if k in r})
        d['sourceReportSha256']=r['sourceReportSha256'];d['detailedTraceAvailable']=True
        if not any(s.get('code') or s.get('calls') or s.get('feedback') for s in d['steps']):d['evidenceLevel']='round_summary'
        # Preserve published identity, outcome, result, budget and media fields exactly.
        old.update(d);old['note']='';old['budgets']=dict(r['budgets']);old['elapsedSeconds']=r['elapsedSeconds'];old['modelRounds']=r['modelRounds'];old['tokens']=r['tokens']
        loop=loop_of(reports[rid].get('runner_report',reports[rid]))
        old['loopBudgets']={k:loop[k] for k in ['max_turns','timeout_seconds'] if k in loop}
        write(u/r['detailUrl'],old);details[rid]=old;align.append(audit)
        r['detailedTraceAvailable']=True;r['note']='';r['evidenceLevel']=old['evidenceLevel'];r['progress']=old['progress']
    for rid,r in runs.items():comparisons['runs'][rid]=r
    write(u/'data/case-comparisons.json',comparisons)
    # Extend every existing exact-task group; keep all original model traces intact.
    by_task={d['taskId']:d for d in details.values()};cohort_indices={}
    for filename in ['index.json','complete-index.json']:
        index=read(u/'data/execution'/filename)
        for story in index['stories']:
            task=story['runs'][0]['taskId'];d=by_task[task]
            entry={k:v for k,v in d.items() if k not in ['steps','callCounts','verifier','unassignedCells','milestones']}
            story['runs']=[r for r in story['runs'] if r['modelKey']!=MODEL]+[entry]
            assert len({r['taskId'] for r in story['runs']})==1
        allruns=[r for s in index['stories'] for r in s['runs']]
        index.update(updated='2026-10-21',modelCount=10,runCount=len(allruns),roundCount=sum(len(read(u/r['detailUrl'])['steps']) for r in allruns),toolCallCount=sum((r['stats'].get('toolCalls') or 0) for r in allruns),scope='Ten models on the same tasks; recorded actions and feedback, with round summaries where native call events are unavailable.')
        write(u/'data/execution'/filename,index);cohort_indices[filename]=index
    identity=read(u/'data/execution/identity-audit.json')
    legacy=read(u/'data/case-media.json')
    for group in legacy['groups']:
        if group['taskId'] in by_task:
            r=runs[by_task[group['taskId']]['id']]
            legacy['runs'][r['id']]=r
            if r['id'] not in group['runIds']:group['runIds'].append(r['id'])
    for story in cohort_indices['index.json']['stories']:
        d=story['runs'][-1];r=runs[d['id']]
        identity['comparisons'][story['id']][d['id']]={k:v for k,v in dict(taskId=d['taskId'],nativeIdentity=r['nativeIdentity'],identitySha256=r['identitySha256'],reportSha256=d['reportSha256'],video=d['video'],mediaKind=d['mediaKind'],mediaMatchesReport=True).items()}
    identity['date']='2026-10-21';identity['recordCount']=sum(len(rs) for rs in identity['comparisons'].values())
    write(u/'data/execution/identity-audit.json',identity);(u/'data/case-media.json').write_text(json.dumps(legacy,indent=2,ensure_ascii=False)+'\n')
    episode_base=[r for r in read(u/'data/report/planning-episodes.json') if r['model']!=MODEL];episodes=episode_base+planning_rows
    assert len(episodes)==1570 and sum(r['outcome']=='success' for r in planning_rows)==110
    metrics=read(u/'data/report/metrics.json');metrics['models']=[m for m in metrics['models'] if m!=MODEL]+[MODEL];metrics['modelNames'][MODEL]=NAME
    good=[r for r in planning_rows if r['outcome']=='success'];bad=[r for r in planning_rows if r['outcome']!='success'];known=[r for r in planning_rows if r['tokens'] is not None];errors=[r for r in planning_rows if r['executionErrorRounds']]
    source_results=[]
    for b in SOURCES.values():
        rs=[r for r in planning_rows if r['benchmark']==b];source_results.append(dict(benchmark=b,n=len(rs),completed=sum(r['outcome']=='success' for r in rs),time=distribution(r['elapsedSeconds'] for r in rs),tokens=distribution(r['tokens'] for r in rs)))
    model=dict(model=MODEL,name=NAME,n=157,completed=len(good),outcomes=dict(collections.Counter(r['outcome'] for r in planning_rows)),time=distribution(r['elapsedSeconds'] for r in planning_rows),successfulTime=distribution(r['elapsedSeconds'] for r in good),unsuccessfulTime=distribution(r['elapsedSeconds'] for r in bad),tokens=distribution(r['tokens'] for r in planning_rows),unsuccessfulTokens=sum(r['tokens'] for r in bad if r['tokens'] is not None),unsuccessfulTokenShare=sum(r['tokens'] for r in bad if r['tokens'] is not None)/sum(r['tokens'] for r in known),episodesWithExecutionErrors=len(errors),recoveredAfterError=sum(r['recoveredAfterError'] for r in errors),errorRounds=sum(r['executionErrorRounds'] for r in planning_rows),totalRounds=sum(r['rounds'] for r in planning_rows),sourceResults=source_results)
    plan=metrics['planning'];plan['models']=[m for m in plan['models'] if m['model']!=MODEL]+[model];plan['nEpisodes']=1570;plan['tasks']=outcomes['tasks']
    for k in ['atLeastOneSucceeds','allFail','allSucceed']:plan[k]=outcomes[k]
    success={m:{t['taskId'] for t in outcomes['tasks'] if m in t['successModels']} for m in metrics['models']}
    plan['pairwise']={m:{n:dict(both=len(success[m]&success[n]),onlyRow=len(success[m]-success[n]),onlyColumn=len(success[n]-success[m]),neither=157-len(success[m]|success[n])) for n in metrics['models']} for m in metrics['models']}
    plan['universalFailureSources']=dict(collections.Counter(t['benchmark'] for t in outcomes['tasks'] if not t['successModels']))
    plan['additionalOverAstra']=plan['atLeastOneSucceeds']-len(success['astra'])
    plan['successCountHistogram']=dict(sorted(collections.Counter(len(t['successModels']) for t in outcomes['tasks']).items()))
    plan['stopLimits']=[x for x in plan['stopLimits'] if x['model']!=MODEL]+[dict(model=MODEL,limit=k,episodes=n) for k,n in collections.Counter(x for r in planning_rows for x in r['stopLimits']).items()]
    metrics['provenance']['timeDefinition']='Reported episode wall time; campaign elapsed time if supplied, otherwise the last recorded loop time.'
    metrics['generated']='2026-10-21';metrics['provenance'].update(argonArchive=args.archive.name,argonArchiveSha256=archive_hash,recordsVerified=1570,adoptedOnly=True)
    write(w/'report-metrics.json',metrics);write(w/'planning-episodes.json',episodes);csv_write(w/'planning-episodes.csv',episodes,list(episode_base[0]))
    write(w/'alignment-audit.json',align)
    audit=dict(archive=args.archive.name,archiveSha256=archive_hash,adoptedPlanningReports=157,planningSuccesses=110,matchedGalleryReports=len(details),selectedExplorerTasks=len(cohort_indices['complete-index.json']['stories']),historicalAttemptsExcluded=True,caseEvidence=evidence_audit,alignment=align,planningMetrics=model)
    write(u/'data/argon-logs-audit.json',audit)
    print(json.dumps({'matchedGalleryReports':len(details),'planning':{k:v for k,v in model.items() if k not in ['sourceResults','time','tokens','successfulTime','unsuccessfulTime']},'explorerRecords':cohort_indices['complete-index.json']['runCount'],'matchedCodeCells':sum(a['matchedCodeCells'] for a in align),'unassignedCells':sum(a['unassignedCodeCells'] for a in align)},indent=2))

if __name__=='__main__':main()
