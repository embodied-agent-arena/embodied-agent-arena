#!/usr/bin/env python3
"""Build complete, exact-task gallery rows from the original log archive.

The local evidence cache contains verified main reports and records extracted
by task ID from all-model-logs-20261019.zip. Media is joined by BOTH task ID and
report digest, never by benchmark, list position, or a display title.
"""
import argparse, collections, csv, gzip, hashlib, html, io, json, re, zipfile
from pathlib import Path

MODELS=['astra','sol','fable','gemini','qwen','qwen35-397b','qwen35-27b','opus55','deepseek']
NAMES=['GPT-6 Astra','GPT-6 Sol','Claude Fable 5.1','Gemini 3.8 Flash','Qwen 3.8 Max','Qwen 3.5 397B-A17B','Qwen 3.5 27B','Claude Opus 5.5','DeepSeek 4.1 Flash-1017']

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if not any(s in k.lower() for s in ['path','credential','api_key','authorization','image','artifact','context'])}
    if isinstance(x,list):return [clean(v) for v in x[:30]]
    if isinstance(x,str):return re.sub(r'/(?:home|Users|opt|mnt|tmp)/[^\s\"\'<>]+','[local file]',x)[:2500]
    return x

def loop_of(d):return d.get('episode_loop') or next((a['episode_loop'] for a in d.get('agent_attempts',[]) if 'episode_loop' in a),{})
def number(r,k):return float(r[k]) if r.get(k) not in ('',None) else None

def build(source,pub):
    reports=json.loads((source/'reports.json').read_text());records=json.loads((source/'records.json').read_text())
    media=json.loads((pub/'data/case-media.json').read_text());legacy=json.loads((pub/'data/case-studies.json').read_text())
    existing=json.loads((pub/'data/execution/index.json').read_text())
    indexed={(v['row']['task_id'],v['row']['model']):v for v in reports.values()}
    groups={g['taskId']:dict(g) for g in media['groups']}
    aliases={}
    for c in legacy:
        if c['predictions']:continue
        task=c['sample']
        if task not in groups:
            groups[task]={k:c[k] for k in ['id','domain','domainKey','benchmark','title','shortTask','question','sample']}
            groups[task].update(taskId=task,predictions=[],images=[],galleryMode='same-task')
        aliases[c['id']]=groups[task]['id']
    # Candidate videos are exact executions. An older run is never substituted
    # for the main report, even if its canonical task happens to be identical.
    videos=collections.defaultdict(list)
    for r in media['runs'].values():
        if not r.get('demoOnly'):videos[(r['taskId'],r['modelKey'],r['sourceReportSha256'])].append(r)
    for c in legacy:
        if c['predictions']:continue
        r={'video':c.get('video'),'animation':c.get('animation'),'poster':c.get('poster'),'mediaKind':c.get('mediaKind'),'frames':[{'image':x} for x in c['images']], 'sourceReportSha256':c['sourceReportSha256']}
        videos[(c['sample'],'opus55',c['sourceReportSha256'])].append(r)
    detail_by_report={}
    for st in existing['stories']:
        for r in st['runs']:
            detail_by_report[(r['taskId'],r['modelKey'],r['reportSha256'])]=json.loads((pub/r['detailUrl']).read_text())
    out=pub/'data/case-records';out.mkdir(exist_ok=True)
    runs={};checks={};excluded=[];unstarted=[]
    for task,g in groups.items():
        entries=[indexed[task,m] for m in MODELS]
        known=[x['identity'] for x in entries if x['identity'].get('task') or x['identity'].get('binding',{}).get('bound')]
        assert known and len({digest(x) for x in known})==1,(task,known)
        identity=known[0];g.update(runIds=[],nativeIdentity=identity,identitySha256=digest(identity))
        g['taskLabel']=identity.get('task') or identity.get('binding',{}).get('sequence_id') or identity.get('entry') or task
        g['taskLabel']=g['taskLabel'].split('::')[0]
        checks[g['id']]={}
        for model,name,x in zip(MODELS,NAMES,entries):
            row=x['row'];raw=x['raw'];d=raw.get('runner_report',raw);rec=records[row['record_id']];loop=loop_of(d);turns=loop.get('turns',[])
            rid='record-'+row['record_id'];sha=row['source_observed_sha256'];native=x['identity'];verified=native==identity
            if not verified:
                assert row['outcome']!='success' and not turns,(task,model)
                unstarted.append(rid)
            candidates=videos.get((task,model,sha),[]) if verified else []
            film=next((v for v in candidates if v.get('video') or v.get('animation')), {})
            for candidate in media['runs'].values():
                if candidate['taskId']==task and candidate['modelKey']==model and candidate['sourceReportSha256']!=sha:excluded.append(candidate['id'])
            budgets={**raw.get('task',{}).get('budgets',{}),**d.get('budgets',{}),**raw.get('budget',{}).get('limits',{})}
            budgets.update({k:v for k,v in {'max_rounds':loop.get('max_turns'),'max_code_turns':loop.get('max_turns'),'timeout_seconds':loop.get('timeout_seconds')}.items() if v is not None})
            h=raw.get('feedback',{}).get('harness_verifier',{});hm=h.get('metrics',{});hh=h.get('metadata',{});usage=d.get('llm_usage') or rec.get('usage') or {}
            protocol='Baseline' if row['wave']=='W3' or model=='opus55' else 'Per-round RGB' if 'rgb' in rec.get('source','').lower() or model=='deepseek' else 'Recorded native interface'
            r=dict(id=rid,model=name,modelKey=model,benchmark=g['benchmark'],taskId=task,taskLabel=g['title'],instruction=g['question'],comparisonInstruction=g['question'],comparisonTaskLabel=g['title'],outcome=row['outcome'],success=row['outcome']=='success',protocol=protocol,video=film.get('video'),animation=film.get('animation'),poster=film.get('poster'),frames=film.get('frames',[]),mediaKind=film.get('mediaKind') or 'Execution log',elapsedSeconds=number(row,'duration_seconds'),modelRounds=len(turns),tokens=number(row,'total_tokens') if row.get('token_usage_available')!='False' else None,budgets=budgets,sourceReportSha256=sha,mediaReportSha256=sha if film else None,recordId=row['record_id'],recorded=True,nativeIdentity=native,identitySha256=digest(native) if verified else None,executionStarted=verified,detailUrl=f'data/case-records/{rid}.json',note='')
            if not verified:r['note']='The assigned evaluation failed before the native task started.'
            detail=detail_by_report.get((task,model,sha))
            if detail:
                detail=dict(detail);detail.update(id=rid,detailUrl=r['detailUrl'],video=r['video'],poster=r['poster'],mediaKind=r['mediaKind'],mediaMatchesReport=bool(film))
            else:
                steps=[]
                for i,t in enumerate(turns):
                    err=clean(t.get('error'));success=bool(t.get('success'));at=t.get('elapsed_seconds');prev=turns[i-1].get('elapsed_seconds',0) if i else 0
                    steps.append(dict(round=t['turn'],at=at,duration=at-prev if at is not None and prev is not None else None,title='Task complete' if success else 'Execution error' if err else 'Decision round',error=err,budgetStop=bool(err and 'budget' in str(err).lower()),success=success,score=None,goalsMet=None,goalsTotal=None,environmentSteps=None,currentSubgoal=None,toolCalls=None,cumulativeToolCalls=None,code='',stdout='',feedback=clean({k:v for k,v in t.items() if k not in ['elapsed_seconds','turn']}),calls=[]))
                progress={'value':hm['completed_count'],'total':hh.get('required_count',5),'label':'Goals completed'} if hm.get('completed_count') is not None else {'value':int(r['success']),'total':1,'label':'Task completion'}
                stats=dict(elapsedSeconds=r['elapsedSeconds'],modelCalls=number(row,'response_count'),rounds=len(turns),tokens=r['tokens'],promptTokens=number(row,'prompt_tokens'),completionTokens=number(row,'completion_tokens'),codeErrors=sum(bool(t.get('error')) for t in turns),toolCalls=None,environmentSteps=None,apiSeconds=None,apiMeanSeconds=None,completedSubgoals=hm.get('completed_count'),requiredSubgoals=hh.get('required_count'),nativeSteps=hm.get('native_steps'),finalScore=None,actionSubmissions=None,finalGoalDistanceCm=None,finalGoalAngleDegrees=None)
                markers=sorted({0,len(steps)-1}|{i for i,s in enumerate(steps) if s['error'] or s['success']}) if steps else []
                milestones=[{'index':i,'at':steps[i]['at'],'round':steps[i]['round'],'label':steps[i]['title'],'kind':'success' if steps[i]['success'] else 'error' if steps[i]['error'] else 'action','detail':''} for i in markers]
                detail=dict(id=rid,model=name,modelKey=model,benchmark=g['benchmark'],taskId=task,instruction=g['question'],outcome=r['outcome'],success=r['success'],protocol=protocol,stats=stats,budgets=budgets,stopLimits=d.get('budget_exhausted') or loop.get('budget_exhausted') or [],verifier=clean(h),callCounts={},video=r['video'],poster=r['poster'],mediaKind=r['mediaKind'],videoNote='Recorded excerpt; video time differs from episode time.',evidenceLevel='round_summary',timingNote='Decision timestamps report elapsed episode time.',recordId=r['recordId'],reportSha256=sha,steps=steps,unassignedCells=[],detailUrl=r['detailUrl'],mediaMatchesReport=bool(film),progress=progress,milestones=milestones)
            detail.update(nativeIdentity=native,identitySha256=r['identitySha256'],executionStarted=verified)
            r['progress']=detail['progress']
            (pub/r['detailUrl']).write_text(json.dumps(detail,ensure_ascii=False,separators=(',',':'))+'\n')
            runs[rid]=r;g['runIds'].append(rid)
            checks[g['id']][model]={'recordId':r['recordId'],'taskId':task,'reportSha256':sha,'identitySha256':r['identitySha256'],'executionStarted':verified,'mediaReportSha256':r['mediaReportSha256']}
        assert len(g['runIds'])==9
        g['poster']=next((runs[i]['poster'] for i in g['runIds'] if runs[i]['poster']),None)
        g['videoCount']=sum(bool(runs[i]['video'] or runs[i]['animation']) for i in g['runIds'])
    data={'updated':'2026-10-20','models':NAMES,'groups':list(groups.values()),'runs':runs,'aliases':aliases,'audit':'data/case-comparison-audit.json'}
    (pub/'data/case-comparisons.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    audit={'updated':data['updated'],'sourceArchive':'all-model-logs-20261019.zip','taskCount':len(groups),'recordCount':len(runs),'videoCount':sum(bool(r['video'] or r['animation']) for r in runs.values()),'excludedDifferentExecutionMedia':sorted(set(excluded)),'failedBeforeNativeExecution':unstarted,'comparisons':checks}
    (pub/'data/case-comparison-audit.json').write_text(json.dumps(audit,ensure_ascii=False,separators=(',',':'))+'\n')
    # Preserve the detailed, audited stories. Add omitted models using exact-task
    # records, retaining the existing execution and media for each named example.
    complete=json.loads(json.dumps(existing))
    for st in complete['stories']:
        task=st['runs'][0]['taskId'];g=groups[task];by_model={r['modelKey']:r for r in st['runs']}
        for rid in g['runIds']:
            r=runs[rid]
            if r['modelKey'] not in by_model:
                detail=json.loads((pub/r['detailUrl']).read_text());by_model[r['modelKey']]={k:v for k,v in detail.items() if k not in ['steps','unassignedCells','verifier','callCounts','milestones']}
        st['runs']=[by_model[k] for k in MODELS];st['comparisonId']=g['id']
    # Keep statements tied to the named comparison, not the old subset size.
        if st['id']=='time':st['takeaway']='Astra, Sol and DeepSeek-1017 reach the same chair. Compare elapsed time alongside decision count.'
        if st['id']=='horizon':st['takeaway']='All nine agents stop without completing the toolbox. Inspect which budget ends each attempt.'
    complete['runCount']=sum(len(st['runs']) for st in complete['stories']);complete['scope']='Nine models per exact task; full action traces or recorded round summaries.'
    (pub/'data/execution/complete-index.json').write_text(json.dumps(complete,ensure_ascii=False,separators=(',',':'))+'\n')
    # The static evidence page exposes the same nine selectable records.
    guide=pub/'execution-guide.html'
    old=guide.read_text();head=old[:old.index('<body>')]
    h=html.escape
    def fmt(v):return '—' if v is None else f'{v:,.1f}'.removesuffix('.0')
    sections=[];csv_rows=[]
    for st in complete['stories']:
        cells=[]
        for r in st['runs']:
            stats=r['stats'];progress=r['progress']
            outcome=('Completed' if r['success'] else 'Not completed') if progress['label']=='Task completion' else f"{fmt(progress['value'])}/{fmt(progress['total'])} {progress['label'].lower()}"
            cells.append('<tr><th scope="row">'+h(r['model'])+'</th><td>'+h(outcome)+'</td><td>'+fmt(stats['modelCalls'])+'</td><td>'+fmt(stats['elapsedSeconds'])+'</td><td>'+fmt(stats['tokens'])+'</td><td><a href="'+h(r['detailUrl'])+'">Execution record ↗</a></td></tr>')
            csv_rows.append([st['id'],r['taskId'],r['model'],r['recordId'],r['reportSha256'],r['outcome'],stats['modelCalls'],stats['elapsedSeconds'],stats['tokens'],r['detailUrl']])
        sections.append('<section id="'+h(st['id'])+'"><p>'+h(st['benchmark'])+'</p><h2>'+h(st['title'])+'</h2><p><strong>Task:</strong> '+h(st['instruction'])+'</p><p>'+h(st['takeaway'])+'</p><div class="guide-table"><table><thead><tr><th>Model</th><th>Outcome</th><th>Model calls</th><th>Time (s)</th><th>Tokens</th><th>Evidence</th></tr></thead><tbody>'+''.join(cells)+'</tbody></table></div><p><a href="index.html?analysis='+h(st['id'])+'#execution-analysis">Compare models interactively ↗</a></p></section>')
    guide.write_text(head+'<body><main class="wrap"><a href="index.html#execution-analysis">← Interactive Execution Explorer</a><h1>Inside the Execution</h1><p>'+str(len(complete['stories']))+' tasks. Nine models per task. '+str(complete['runCount'])+' execution records.</p><p><a href="data/execution/complete-episodes.csv" download>Download episode metrics ↓</a></p>'+''.join(sections)+'<section><h2>Evidence and Timing</h2><p>Each result links to its original report digest and native task identity. Videos are linked to the same execution report; records without matching video expose the available execution log. Decision timestamps use episode time, separately from video playback. Full action traces and per-round summaries retain their recorded level of detail.</p></section></main></body></html>')
    with (pub/'data/execution/complete-episodes.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['case_group','task_id','model','record_id','report_sha256','outcome','model_calls','elapsed_seconds','tokens','evidence_url']);writer.writerows(csv_rows)
    index=pub/'index.html';text=index.read_text();text=re.sub(r'46 Tasks · \d+ Episodes · 9 Models',f"46 Tasks · {complete['runCount']} Episodes · 9 Models",text);index.write_text(text)
    print(json.dumps({k:v for k,v in audit.items() if k!='comparisons'},indent=2))

def prepare_cache(archive,source,pub):
    """Extract only the exact showcased tasks; verify original report digests."""
    source.mkdir(parents=True,exist_ok=True)
    media=json.loads((pub/'data/case-media.json').read_text())
    legacy=json.loads((pub/'data/case-studies.json').read_text())
    tasks={g['taskId'] for g in media['groups']}|{c['sample'] for c in legacy if not c['predictions']}
    with zipfile.ZipFile(archive) as z:
        manifest=next(n for n in z.namelist() if n.endswith('/trials.csv') and not n.startswith('__MACOSX/'))
        prefix=manifest[:-len('trials.csv')]
        rows=[r for r in csv.DictReader(io.StringIO(z.read(manifest).decode('utf-8-sig'))) if r['dataset']=='main' and r['wave'] in ['W3','W4'] and r['task_id'] in tasks]
        reports={}
        for r in rows:
            content=gzip.decompress(z.read(prefix+r['report_object']))
            assert hashlib.sha256(content).hexdigest()==r['source_observed_sha256']
            raw=json.loads(content);d=raw.get('runner_report',raw);actual=d.get('actual_task_ids')
            if actual:
                assert actual==[d['expected_task_id']]
                identity={'task':actual[0]}
            else:
                binding=d.get('pool_binding') or raw.get('pool_binding') or {};coordinate=d.get('pool_coordinate') or raw.get('pool_coordinate') or {}
                identity={'binding':{k:v for k,v in binding.items() if not k.endswith(('file','path'))},'entry':coordinate.get('entry_id'),'revision':coordinate.get('source_revision')}
            reports[r['record_id']]={'row':r,'raw':raw,'identity':identity}
        records={}
        with z.open(prefix+'records.jsonl') as f:
            for line in f:
                r=json.loads(line)
                if r['record_id'] in reports:records[r['record_id']]=r
    assert set(records)==set(reports)
    (source/'reports.json').write_text(json.dumps(reports))
    (source/'records.json').write_text(json.dumps(records))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True,help='Local evidence cache; never published');ap.add_argument('--archive',type=Path,help='Original all-model-logs archive; prepare the evidence cache before building');args=ap.parse_args()
    pub=Path(__file__).resolve().parents[1]/'public'
    if args.archive:prepare_cache(args.archive,args.source,pub)
    build(args.source,pub)
