#!/usr/bin/env python3
"""Build the execution explorer from linked, locally audited case/log extracts.

Usage: python scripts/build_execution_analysis.py --source PATH
The source directory contains cases.json, records.json, evidence/ and reports/.
No raw prompts, credentials, absolute paths, or invented video-time alignment are published.
"""
import argparse, ast, collections, datetime, hashlib, json, re
from pathlib import Path
from public_json import dumps as trace_json

ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);args=ap.parse_args()
root=Path(__file__).resolve().parents[1];pub=root/'public';work=args.source
cases=json.loads((work/'cases.json').read_text());records={r['source_sha256']:r for r in json.loads((work/'records.json').read_text())};media=json.loads((pub/'data/case-media.json').read_text())['runs']
out=pub/'data/execution';out.mkdir(parents=True,exist_ok=True)
CONTENT=json.loads((root/'scripts/execution-story-content.json').read_text())
STORIES=[
 dict(id='recovery',label='Recovering From Errors',domain='Planning',benchmark='DiscoveryWorld',match='discoveryworld',task='Archaeology_Dating',title='Can an Agent Recover and Still Finish?',instruction='Excavate three sites, date the artifacts, and place the red flag at the oldest site.',takeaway='An early error need not end an episode. Opus 5.5 recovers from two initial code errors and finishes in seven rounds; DeepSeek-1017 reaches a partial score of 0.5, then uses its remaining rounds without completing the task.',question='What matters after a mistake?',answer='Whether subsequent actions change the task state. Watch the native score rise after Opus collects the tools, dates the artifacts, and places the flag; compare the stalled final rounds of DeepSeek-1017.',default=['opus55','deepseek'],notes={'opus55':{1:'Code fails before any environment action.',2:'A second code error; task score remains zero.',3:'Collects the shovel, flag and radiocarbon meter. Score reaches 0.3.',4:'Excavates artifacts; score reaches 0.7. Reading the meter directly is rejected.',5:'Uses the meter on an artifact and obtains its age.',6:'Searches for the remaining buried artifact. Score stays at 0.7.',7:'Compares 890,000, 150,000 and 725,000 years; places the flag at site 1. Native score reaches 1.0.'},'deepseek':{5:'Dates an artifact at 890,000 years; native score reaches 0.5.',6:'Code error; score remains 0.5.',7:'Another code error; no additional environment steps.',8:'Final round ends with an error; native task remains incomplete.'}}),
 dict(id='sequence',label='Finishing the Sequence',domain='Manipulation',benchmark='CALVIN',match='calvin',task='calvin_sequence_0526',title='Does More Motion Mean More Progress?',instruction='Lift the blue block from the sliding cabinet, put it in the drawer, retrieve it, stack it, and switch off the light.',takeaway='On sequence 0526, Opus 5.5 completes all five subgoals with 338 action submissions. Gemini makes hundreds of submissions but completes none before its token budget ends.',question='Which progress signal should we trust?',answer='The native sequence verifier. Motion, a valid tool call and a completed subgoal are different events. Open a round to inspect the requested action, current subgoal and returned state.',default=['opus55','gemini']),
 dict(id='drawer',label='Calls vs. Completion',domain='Manipulation',benchmark='RoboCasa',match='robocasa',task='target-OpenDrawer',title='A Successful Call Is Not a Successful Task.',instruction='Open the left drawer.',takeaway='Fable 5.1 reaches the native drawer-success condition in 20 rounds. Sol uses all 24 rounds without completing it. The trace separates successful tool execution from the environment’s task check.',question='Where does execution stop translating into progress?',answer='Inspect observations, target grounding and end-effector moves alongside their returned feedback. Tool “ok” reports that a call ran; the final native verifier decides whether the drawer task is complete.',default=['fable','sol']),
 dict(id='goal',label='Meeting Every Condition',domain='Planning',benchmark='ALFRED',match='alfred_official_visual',task='trial_T20190908_064808_086925',title='One Condition Is Not the Whole Goal.',instruction='Clean the soap bar and place it in the bathtub.',takeaway='Gemini completes this task within eight rounds. DeepSeek-1017 also uses eight rounds but finishes with only one of three goal conditions met. Finding and moving an object is only part of the instruction.',question='Why can plausible actions still fail?',answer='ALFRED checks the requested final state, including the cleaning condition. Inspect the per-round goal count and the returned object state, rather than judging success from movement alone.',default=['gemini','deepseek']),
 dict(id='time',label='The Cost of a Decision',domain='Planning',benchmark='HumanCLAW',match='humanclaw',task='104862417_172226382-10-chair',title='Fewer Decisions Do Not Always Mean Less Time.',instruction='Find the chair and move until the body is touching the target, then stop.',takeaway='All three recorded agents complete this navigation task. Astra uses 18 rounds in about 471 seconds, Sol 17 in 669 seconds, and DeepSeek-1017 65 in 406 seconds. Round count and elapsed time capture different costs.',question='How should execution efficiency be compared?',answer='Read task completion, environment steps, model calls, tokens and elapsed time together. This is a navigation episode; it does not change the separate sitting-task result.',default=['astra','deepseek']),
 dict(id='horizon',label='When Budgets Run Out',domain='Manipulation',benchmark='BEHAVIOR-1K',match='capx_behavior1k',task='outfit_a_basic_toolbox',title='Which Resource Runs Out First?',instruction='Outfit a basic toolbox with the required tools.',takeaway='None of these eight recorded episodes passes the native completion check. Some stop at the token limit and others at the round limit, with recorded durations ranging from about 12 to 133 minutes.',question='Would a longer time limit alone fix the failure?',answer='These episodes already have a 180-minute time ceiling. Their recorded stopping conditions are token or round limits. More allowed time alone does not remove those limits; whether a larger overall budget helps requires another experiment.',default=['astra','opus55']),
 dict(id='precision',label='Contact and Precision',domain='Manipulation',benchmark='ManiSkill',match='maniskill',task='PlugCharger',title='The Last Contact Can Remain Unsolved.',instruction='Insert the charger into the socket.',takeaway='Qwen Max and Qwen 397B each reach 400 native steps without task success. DeepSeek-1017 stops after 32 model rounds and 385 native steps. Motion must satisfy the insertion criterion to count.',question='What does the final state establish?',answer='The native verifier reports no successful insertion in these three episodes. Use the action parameters and feedback to inspect the attempts; the recordings do not isolate a single causal explanation.',default=['qwen','deepseek'])
]

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
        if not milestones or changed or score_change or kind in ['success','stop']:
            milestones.append({'index':steps.index(s),'round':s['round'],'at':s['at'],'label':label,'detail':detail,'kind':kind,'score':s.get('score')})
    if steps and (not milestones or milestones[-1]['index']!=len(steps)-1):
        milestones.append({'index':len(steps)-1,'round':final['round'],'at':final['at'],'label':'Final check','detail':'Completed' if data['success'] else 'Not completed','kind':'success' if data['success'] else 'stop'})
    data['milestones']=milestones
    return data

def build(id,story):
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

# Cover all same-task video comparisons already present in the gallery.
# The seven question-led comparisons remain the entry points; other tasks form
# a searchable library with descriptive, strictly native outcomes.
by_task=collections.defaultdict(list)
for case_id,c in cases.items():
    if case_id in media and c.get('source_sha256') in records and (work/'reports'/f'{case_id}.json').exists():
        by_task[(c['benchmark'],c['task_id'])].append(case_id)
claimed=set()
published=[];audit=[]
for st in STORIES:
    st.update(CONTENT.get(st['id'],{}))
    ids=[case_id for case_id,c in cases.items() if c['benchmark']==st['match'] and st['task'] in c['task_id']]
    runs=[]
    for case_id in ids:
        run,a=build(case_id,st);runs.append(run);audit.append(a)
        claimed.add((cases[case_id]['benchmark'],cases[case_id]['task_id']))
    assert len({r['taskId'] for r in runs})==1,'A comparison must use the same task ID.'
    item={k:v for k,v in st.items() if k not in ['match','task','notes']};item['runs']=runs;item['featured']=True;published.append(item)
for (benchmark,task_id),ids in sorted(by_task.items()):
    if len(ids)<2 or (benchmark,task_id) in claimed:continue
    # AnymalC-Spin uses continuous reward, not binary task success.
    if any(cases[i].get('outcome')=='scored' for i in ids):continue
    first=media[ids[0]]
    title=first.get('instruction') or first['benchmark']
    story_id='task-'+hashlib.sha256(task_id.encode()).hexdigest()[:10]
    ok=[i for i in ids if cases[i].get('success',cases[i].get('outcome')=='success')];bad=[i for i in ids if not cases[i].get('success',cases[i].get('outcome')=='success')]
    chosen=[ok[0] if ok else ids[0],bad[0] if bad else ids[1]]
    st={'id':story_id,'label':first['benchmark'],'domain':'Planning' if cases[ids[0]]['wave']=='W3' else 'Manipulation','benchmark':first['benchmark'],'title':title,'instruction':title,'default':[cases[i]['model'] for i in chosen],'featured':False,'question':'Which Actions Changed the Outcome?','answer':'Select a decision round to inspect the executed actions, returned feedback and native completion check.'}
    st.update(CONTENT.get(story_id,{}))
    if benchmark=='calvin' and 'instruction' not in CONTENT.get(story_id,{}):
        st['instruction']=max((media[i]['instruction'] for i in ids),key=len)
    if story_id=='task-cc4c299566':
        st['instruction']='Task 10 · screw, dustpan and container (episode 000).'
    runs=[]
    for case_id in ids:
        run,a=build(case_id,st);runs.append(run);audit.append(a)
    completed=sum(r['success'] for r in runs)
    st['takeaway']=f"{completed} of {len(runs)} recorded models completed this task. Compare their execution traces and resource use."
    st['runs']=runs;published.append(st)
index={'updated':'2026-10-20','stories':published,'modelCount':len({r['modelKey'] for st in published for r in st['runs']}),'runCount':len(audit),'roundCount':sum(a['rounds'] for a in audit),'toolCallCount':sum(a['toolCalls'] for a in audit),'scope':'Selected episodes from the existing gallery. Same task within each comparison; per-record observation settings and budgets are available in Settings. These examples do not replace the aggregate benchmark results.'}
(out/'index.json').write_text(json.dumps(index,ensure_ascii=False,separators=(',',':'))+'\n');(work/'alignment-audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in index.items() if k!='stories'},indent=2))

# Static, crawlable companion uses exactly the same evidence and claim text.
import csv, html
h=html.escape
parts=[]
for st in published:
    rows=[]
    for r in st['runs']:
        stats=r['stats'];end=(f"{stats['completedSubgoals']}/{stats['requiredSubgoals']} goals" if stats['completedSubgoals'] is not None else 'Completed' if r['success'] else 'Not completed')
        rows.append(f'<tr><th scope="row">{h(r["model"])}</th><td>{end}</td><td>{stats["modelCalls"]}</td><td>{stats["elapsedSeconds"]:.1f}</td><td>{stats["tokens"]:,}</td><td><a href="{r["detailUrl"]}">Trace JSON</a></td></tr>')
    parts.append(f'<section id="{st["id"]}"><p class="eyebrow">{h(st["benchmark"])} · {h(st["domain"])}</p><h2>{h(st["title"])}</h2><p><strong>Task:</strong> {h(st["instruction"])}</p><p>{h(st["takeaway"])}</p><p>{h(st["answer"])}</p><div class="guide-table"><table><thead><tr><th>Model</th><th>Outcome</th><th>Model calls</th><th>Time (s)</th><th>Tokens</th><th>Evidence</th></tr></thead><tbody>{"".join(rows)}</tbody></table></div><p><a href="index.html?analysis={st["id"]}#execution-analysis">Watch and inspect this comparison →</a></p></section>')
guide='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Inside the Execution — Embodied Agent Arena</title><meta name="description" content="Detailed same-task case studies compare nine VLM agents through videos, executed actions, task feedback, time and token use."><link rel="canonical" href="https://embodied-agent-arena.github.io/embodied-agent-arena/execution-guide.html"><link rel="stylesheet" href="styles.css"><style>main{padding-block:64px}h1{font-size:44px;line-height:1.2;margin:24px 0}h2{font-size:30px;line-height:1.3}p{margin:20px 0;max-width:900px}section{padding:48px 0;border-bottom:1px solid #e0e5e0}.guide-table{overflow:auto}table{border-collapse:collapse;font-size:15px;min-width:780px;width:100%}th,td{text-align:left;padding:12px;border-bottom:1px solid #e3e7e3}th{font-weight:600}a{text-decoration:underline;text-underline-offset:4px}.guide-nav{display:flex;gap:20px;flex-wrap:wrap;font-size:16px}@media(max-width:640px){h1{font-size:32px}h2{font-size:25px}}</style></head><body><main class="wrap"><a href="index.html#execution-analysis">← Interactive Execution Explorer</a><h1>Inside the Execution</h1><p>7 tasks. 33 recorded episodes. 9 models. These comparisons follow the same task within each group, with per-record observations and budgets retained in the linked evidence.</p><nav class="guide-nav" aria-label="Execution studies">'''+''.join(f'<a href="#{s["id"]}">{h(s["benchmark"])}</a>' for s in published)+'''</nav>'''+''.join(parts)+'''<section><h2>Evidence and Timing</h2><p>Each trace identifies its source record and report SHA-256. The 711 recorded rounds retain their elapsed time, executed code excerpts, feedback and 6,972 retained primitive calls. Planning also exposes native score feedback and environment-step counts. Tool calls, model responses and environment steps are distinct units.</p><p>Video excerpts and saved observations have their own playback clock. They are not linearly stretched to episode wall time. Total duration includes setup where the report records it; round timestamps start at the agent loop. Input and output tokens are included, and missing billing data is not reported as zero cost.</p><p>These selected episodes explain behavior, rather than estimate benchmark-wide rates. See the <a href="index.html#results">full benchmark comparison</a> for aggregate results, and <a href="data/execution/episodes.csv">download the episode metrics</a>.</p></section></main></body></html>'''
guide=guide.replace('7 tasks. 33 recorded episodes. 9 models.',f"{len(published)} tasks. {len(audit)} recorded episodes. {index['modelCount']} models.").replace('711 recorded rounds',f"{index['roundCount']:,} recorded rounds").replace('6,972 retained primitive calls',f"{index['toolCallCount']:,} retained primitive calls")
(pub/'execution-guide.html').write_text(guide)
with (out/'episodes.csv').open('w',newline='') as f:
    writer=csv.writer(f,lineterminator='\n');writer.writerow(['case_group','benchmark','model','task_id','record_id','report_sha256','outcome','protocol','elapsed_seconds','model_calls','rounds','total_tokens','completed_subgoals','required_subgoals','environment_steps','native_steps','tool_calls','stop_limits'])
    for st in published:
        for r in st['runs']:
            v=r['stats'];writer.writerow([st['id'],r['benchmark'],r['model'],r['taskId'],r['recordId'],r['reportSha256'],r['outcome'],r['protocol'],v['elapsedSeconds'],v['modelCalls'],v['rounds'],v['tokens'],v['completedSubgoals'],v['requiredSubgoals'],v['environmentSteps'],v['nativeSteps'],v['toolCalls'],';'.join(r['stopLimits'])])
