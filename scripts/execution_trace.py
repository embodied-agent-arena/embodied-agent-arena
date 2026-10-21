"""Build public execution traces from verified reports and recorded events."""
import ast,collections,datetime,hashlib,json,re
from public_json import dumps as trace_json

def clean(x,depth=0):
    if depth>6:return '[nested data omitted]'
    if isinstance(x,dict):return {k:clean(v,depth+1) for k,v in x.items() if not any(t in k.lower() for t in ['path','base64','credential','api_key','authorization','agent_context','camera','image','artifact','provenance','native_value_handle'])}
    if isinstance(x,list):return [clean(v,depth+1) for v in x[:24]]+(['…'] if len(x)>24 else [])
    if isinstance(x,str):
        x=re.sub(r'/(?:home|Users|opt|mnt|tmp)/[^\s\"\'<>]+','[local file]',x)
        return x[:2500]+('…' if len(x)>2500 else '')
    return x

def loop_of(d):
    return d.get('episode_loop') or next((a['episode_loop'] for a in d.get('agent_attempts',[]) if 'episode_loop' in a),{})
def clock(s):
    try:return datetime.datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()
    except (ValueError,TypeError,AttributeError):return None

def method_names(code):
    try:
        tree=ast.parse(code)
        return list(dict.fromkeys(n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id in ['p','api','primitive_api','primitives']))
    except SyntaxError:return []

def enrich(data):
    """Expose observed task progress; never substitute action count for progress."""
    steps=data['steps'];stats=data['stats'];final=steps[-1] if steps else {}
    if stats['completedSubgoals'] is not None:
        progress={'value':stats['completedSubgoals'],'total':stats['requiredSubgoals'],'label':'Goals completed'}
    elif final.get('goalsMet') is not None:
        progress={'value':final['goalsMet'],'total':final['goalsTotal'],'label':'Goal conditions'}
    elif stats.get('finalScore') is not None:
        progress={'value':stats['finalScore'],'total':100 if data['benchmark']=='ScienceWorld' else 1,'label':'Task score'}
    else:progress={'value':int(data['success']),'total':1,'label':'Task completion'}
    data['progress']=progress
    info=data['verifier'].get('metadata',{}).get('info_summary',{})
    dist=info.get('obj_to_goal_dist');angle=info.get('obj_to_goal_angle')
    stats['finalGoalDistanceCm']=round(dist[0]*100,2) if isinstance(dist,list) and dist else None
    stats['finalGoalAngleDegrees']=round(angle[0]*180/3.141592653589793,2) if isinstance(angle,list) and angle else None
    stats['actionSubmissions']=sum(n for name,n in data['callCounts'].items() if name in ['submit_calvin_action','apply_maniskill_action','submit_capx_action']) or None
    last_goal=None;milestones=[]
    for s in steps:
        kind='action';label='Action';detail=''
        if s['budgetStop']:kind='stop';label='Budget reached'
        elif s['error']:kind='error';label='Execution error'
        elif s['success']:kind='success';label='Task complete'
        elif s['score'] is not None:label=f"Score {s['score']:.2g}";detail='Task feedback'
        elif s.get('currentSubgoal'):
            detail=s['currentSubgoal'];goal=detail.lower()
            label=('Lift block' if 'sliding cabinet' in goal else 'Store block' if 'store' in goal else
                   'Retrieve block' if 'from the drawer' in goal else 'Stack blocks' if 'stack' in goal else
                   'Light off' if 'light bulb' in goal else 'Current goal')
        else:
            names=[c['name'] for c in s['calls']]
            joined=' '.join(names)
            if 'grasp' in joined:label='Grasp attempt'
            elif 'move_' in joined or 'apply_' in joined or 'submit_' in joined:label='Control action'
            elif 'ground_' in joined or 'locate_' in joined:label='Locate target'
            elif 'inspect_' in joined or 'observe_' in joined:label='Observe state'
            elif s['environmentSteps'] is not None:label='Environment action'
        if s.get('goalsMet') is not None:detail=f"{s['goalsMet']} / {s['goalsTotal']} goal conditions"
        s['event']={'kind':kind,'label':label,'detail':detail}
        goal=s.get('currentSubgoal')
        changed=goal and goal!=last_goal
        if changed:last_goal=goal
        score_change=s.get('score') is not None and (not milestones or s['score']!=milestones[-1].get('score'))
        if not milestones or changed or score_change or kind in ['error','success','stop']:
            milestones.append({'index':steps.index(s),'round':s['round'],'at':s['at'],'label':label,'detail':detail,'kind':kind,'score':s.get('score')})
    if steps and (not milestones or milestones[-1]['index']!=len(steps)-1):
        milestones.append({'index':len(steps)-1,'round':final['round'],'at':final['at'],'label':'Final check','detail':'Completed' if data['success'] else 'Not completed','kind':'success' if data['success'] else 'stop'})
    data['milestones']=milestones
    return data

def build(id,story,*,cases,records,media,work,out):
    c=cases[id];r=records[c['source_sha256']];m=media[id];e=json.loads((work/'evidence'/f'{id}.json').read_text());raw=json.loads((work/'reports'/f'{id}.json').read_text());d=raw.get('runner_report',raw);loop=loop_of(d);turns=loop.get('turns',[]);usage=d.get('llm_usage',{});start=clock(d.get('started_at'))
    assert c['source_sha256']==r['source_observed_sha256']
    summary_only='report_sha256' not in e
    if summary_only:e={'report_sha256':c['source_sha256'],'traces':[],'turn_feedback':[],'responses':[]}
    assert c['source_sha256']==e['report_sha256']
    assert r['record_id'] and c['task_id']==m['taskId']
    events=[]
    for tr in e.get('traces',[]): events.extend(tr['trace'].get('events',[]))
    start=next((v['timestamp'] for v in events if v['event_type']=='pool_coordinate_binding'),start)
    # Consecutive code-cell blocks preserve every recorded primitive call in order.
    blocks=[];block=None
    for v in events:
        p=v.get('payload',{});typ=v['event_type']
        if typ=='code_rejected':
            block={'code':p.get('code',''),'calls':[],'timestamp':v.get('timestamp'),'end':v.get('timestamp'),'stdout':'','error':clean(p.get('error',p.get('reason',p)))};blocks.append(block);block=None
        elif typ=='code_cell':
            block={'code':p.get('code',''),'calls':[],'timestamp':v.get('timestamp'),'stdout':'','error':None};blocks.append(block)
        elif block is not None:
            if typ=='native_gateway_call':
                z=p.get('result',{});block['calls'].append({'name':p.get('name'),'arguments':clean(p.get('kwargs',{})),'ok':z.get('ok'),'error':clean(z.get('error')),'output':clean(z.get('output')),'at':round(v['timestamp']-start,3) if start and v.get('timestamp') else None})
            elif typ=='code_result':block['stdout']=clean(p.get('stdout',''));block['end']=v.get('timestamp');block=None
            elif typ=='code_error':block['error']=clean(p.get('error',p.get('reason',p)));block['end']=v.get('timestamp');block=None
    # Join each executed cell to the recorded model round using its timestamp.
    used=set();match={};alignment=[]
    for b in blocks:
        if start and b['timestamp'] and turns:
            elapsed=b.get('end',b['timestamp'])-start
            candidates=[t for t in turns if t['turn'] not in used and t.get('elapsed_seconds') is not None and abs(t['elapsed_seconds']-elapsed)<=5 and bool(t.get('error'))==bool(b.get('error'))]
            if candidates:
                t=min(candidates,key=lambda t:abs(t['elapsed_seconds']-elapsed));used.add(t['turn']);match[t['turn']]=b;alignment.append({'turn':t['turn'],'cellEnd':round(elapsed,3),'roundEnd':t['elapsed_seconds']})
    unassigned=[b for b in blocks if not any(b is v for v in match.values())]
    fb={x['feedback']['turn']:x for x in e.get('turn_feedback',[]) if 'turn' in x.get('feedback',{})}
    steps=[];prev=0;call_total=0;subgoal=None
    for t in turns:
        n=t['turn'];b=match.get(n,{});f=fb.get(n,{});v=f.get('feedback',{});co=v.get('current_observation',{});code=b.get('code') or f.get('executed_code','');calls=b.get('calls',[]);call_total+=len(calls)
        for call in calls:
            if call['name']=='get_calvin_language_subgoal' and isinstance(call.get('output'),dict):subgoal=call['output'].get('subgoal')
        score=v.get('score');ver=v.get('verifier',{});goal_done=ver.get('goal_conditions_met');goal_total=ver.get('goal_conditions_total')
        score=score if score is not None else ver.get('score_normalized',ver.get('scoreNormalized'))
        at=t.get('elapsed_seconds');error=clean(t.get('error') or b.get('error') or (v.get('stderr_tail') if v.get('returncode',0)!=0 else None));names=list(dict.fromkeys(x['name'] for x in calls)) or method_names(code)
        budget_stop=bool(error and ('budget' in str(error).lower() or 'limit' in str(error).lower()) and 'loop budget' in str(error).lower())
        title=story.get('notes',{}).get(c['model'],{}).get(n)
        if not title:
            if budget_stop:title='The episode stops at its budget limit before another action is executed.'
            elif error:title='Execution error; inspect the returned feedback.'
            elif t.get('success') or v.get('success'):title='The episode reaches its success condition.'
            elif names:title=' · '.join(re.sub(r'^(submit_|observe_|get_|inspect_)','',s).replace('_',' ') for s in names[:3])
            else:title='Inspect the observation and execute the next code cell.'
        env_steps=v.get('env_steps',(co.get('env_steps',co.get('world_steps')) if isinstance(co,dict) else None))
        steps.append({'round':n,'at':at,'duration':round(at-prev,3) if at is not None else None,'title':title,'error':error,'budgetStop':budget_stop,'success':bool(t.get('success') or v.get('success')),'score':score,'goalsMet':goal_done,'goalsTotal':goal_total,'environmentSteps':env_steps,'currentSubgoal':subgoal,'toolCalls':len(calls),'cumulativeToolCalls':call_total,'code':clean(code),'codeExcerpt':len(code)>2500,'stdout':clean(b.get('stdout') or v.get('stdout_tail','')),'feedback':clean({k:v[k] for k in ['success','terminal','score','env_steps','remaining_env_steps','verifier'] if k in v}),'calls':calls})
        if at is not None:prev=at
    h=raw.get('feedback',{}).get('harness_verifier',{});stop=d.get('budget_exhausted') or loop.get('budget_exhausted') or [];budgets=dict(m.get('budgets',{}));budgets.update(d.get('budgets',{}));budgets['max_rounds']=loop.get('max_turns');budgets['timeout_seconds']=loop.get('timeout_seconds',budgets.get('timeout_seconds'))
    duration=d.get('duration_seconds') or (turns[-1].get('elapsed_seconds') if turns else None)
    responses=list({v.get('request_id',v.get('sha256')):v for v in e.get('responses',[])}.values());lat=[v['latency_seconds'] for v in responses if isinstance(v.get('latency_seconds'),(float,int))];complete=len(lat)==usage.get('response_count',-1)
    stats={'elapsedSeconds':duration,'modelCalls':usage.get('response_count'),'rounds':len(turns),'tokens':usage.get('total_tokens'),'promptTokens':usage.get('prompt_tokens'),'completionTokens':usage.get('completion_tokens'),'codeErrors':sum(bool(t.get('error')) or not t.get('execution_ok',True) for t in turns),'toolCalls':sum(len(b['calls']) for b in blocks) if events else None,'environmentSteps':next((s['environmentSteps'] for s in reversed(steps) if s['environmentSteps'] is not None),None),'apiSeconds':round(sum(lat),3) if complete else None,'apiMeanSeconds':round(sum(lat)/len(lat),3) if complete and lat else None,'completedSubgoals':h.get('metrics',{}).get('completed_count'),'requiredSubgoals':h.get('metadata',{}).get('required_count'),'nativeSteps':h.get('metrics',{}).get('native_steps'),'finalScore':next((s['score'] for s in reversed(steps) if s['score'] is not None),None)}
    counts=collections.Counter(x['name'] for b in blocks for x in b['calls'])
    data={'id':id,'model':m['model'],'modelKey':c['model'],'benchmark':m['benchmark'],'taskId':c['task_id'],'instruction':m['instruction'],'outcome':c['outcome'],'success':c.get('success',c.get('outcome')=='success'),'protocol':m['protocol'],'stats':stats,'budgets':{k:v for k,v in budgets.items() if v is not None},'stopLimits':stop,'verifier':clean(h),'callCounts':dict(counts),'video':m['video'],'poster':m['poster'],'mediaKind':m['mediaKind'],'videoNote':'Saved observations, displayed at uniform intervals.' if c['media_kind']=='saved_observations' else 'Recorded video excerpt; the trace covers the full episode.','evidenceLevel':'round_summary' if summary_only else 'action_trace','timingNote': 'Video playback and wall-clock trace time are separate.','recordId':r['record_id'],'reportSha256':c['source_sha256'],'steps':steps,'unassignedCells':[{**b,'code':clean(b['code'])} for b in unassigned],'detailUrl':f'data/execution/{id}.json'}
    data['mediaMatchesReport']=m.get('sourceReportSha256')==c['source_sha256']
    if not data['mediaMatchesReport']:
        # An earlier RGB recording must never illustrate a later/main log.
        data.update(video=None,poster=None,mediaKind='Action trace only',
                    videoNote='No matching video for this execution.',
                    protocol='Main evaluation; per-round image delivery not recorded')
        data['budgets']={k:v for k,v in {**d.get('budgets',{}),
            'max_rounds':loop.get('max_turns'),
            'timeout_seconds':loop.get('timeout_seconds')}.items() if v is not None}
    data=enrich(data)
    (out/f'{id}.json').write_text(trace_json(data)+'\n')
    audit={'id':id,'recordId':r['record_id'],'reportSha256':c['source_sha256'],'evidenceSha256':hashlib.sha256((work/'evidence'/f'{id}.json').read_bytes()).hexdigest(),'rounds':len(turns),'codeCells':len(blocks),'matchedCodeCells':len(match),'toolCalls':sum(len(b['calls']) for b in blocks),'unassignedCodeCells':len(unassigned),'alignment':alignment}
    return {k:v for k,v in data.items() if k not in ['steps','callCounts','verifier','unassignedCells','milestones']},audit

