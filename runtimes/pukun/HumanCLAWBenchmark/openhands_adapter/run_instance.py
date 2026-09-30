#!/usr/bin/env python3
"""HumanCLAW in Pukun's existing REPL/loop; native simulator and scoring stay private."""
import argparse,hashlib,json,os,secrets,shutil,sys,threading,uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];ADAPTER=Path(__file__).resolve().parent
EXTERNAL=Path(os.environ.get('EMBODIED_ARENA_EXTERNAL_ROOT',ROOT.parents[2]/'external'))
sys.path.insert(0,str(ROOT.parent/'scripts'))
import persistent_repl_runtime as repl_runtime
from openhands_bridge import add_executor_args,resolve_or_exit_openhands

class Session:
    runtime_mode=repl_runtime.RUNTIME_MODE
    def __init__(self,args,task,workspace,outputs,run_id):
        self.trace=SimpleNamespace(run_id=run_id);self.workspace=workspace;self.task=task
        self.worker=ThreadPoolExecutor(max_workers=1,thread_name_prefix='humanclaw-native')
        self.code_turn=0;self.acted=False;self.owner_pid=None;self.events=outputs/'events.jsonl'
        self.args=args;self.episode=None
    def initialize(self):
        from backend import NativeEpisode
        from humanclaw_bench.config import load_config
        config_path=self.workspace.parent/'unused-native-model.json'
        config_path.write_text(json.dumps({'backend':'filesystem_queue','model':'pukun-policy-owns-decisions','max_tokens':1,'temperature':0.0,'response_format':'json_object','queue_dir':str(self.workspace.parent/'unused-native-queue')}))
        # The original evaluator constructs its objects; our subclass takes control
        # at run_episode_rollouts. Its VLM planner/verifier never makes a model call.
        self.episode=NativeEpisode(dict(profile=load_config('paper_fullval_v1'),scene_id=self.task['scene_id'],episode_id=self.task['episode_id'],object_category=self.task['object_category'],scene_dataset_config=os.environ.get('HUMANCLAW_HSSD_SCENE_DATASET_CONFIG',str(EXTERNAL/'assets/humanclaw/prepared/hssd-hab.scene_dataset_config.json')),model_config_path=str(config_path),compute_metrics=True,save_video=False,max_steps=self.args.max_env_steps,output_root=str(self.workspace.parent/'native'),device='cuda',n_rollouts=1))
        try:self.episode.evaluate_main()
        except BaseException:
            if self.episode.env is not None:self.episode.env.close()
            raise
    def native(self,method,*args):return self.worker.submit(getattr(self.episode,method),*args).result()
    def log(self,event,**data):
        with self.events.open('a') as f:f.write(json.dumps({'event':event,**data},default=str)+'\n')
    def record_agent_code(self,path):
        self.code_turn+=1;self.acted=False
        self.log('code',turn=self.code_turn,path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def record_execution_started(self,**kwargs):self.log('execution_started',**kwargs)
    def call(self,name,args,kwargs,client):
        if args:raise ValueError('Use explicit named arguments')
        if name=='get_task_context':return self.native('context')
        if name=='observe':return self.native('observe',self.workspace)
        from backend import ACTIONS
        if name not in ACTIONS:raise ValueError('Primitive not exposed')
        if self.code_turn<2:raise ValueError('First observe and yield to receive the ego image')
        if self.acted:raise ValueError('One native action per cell; yield for the next ego image')
        pid=int(client.get('pid',0))
        if pid<=0 or Path(client.get('argv0','')).name!='solve.py':raise ValueError('Actions require the persistent Python worker')
        if self.owner_pid is None:self.owner_pid=pid
        if pid!=self.owner_pid:raise ValueError('Episode belongs to another interpreter')
        visible=kwargs.pop('visible_state','')
        allowed={'walk_forward':{'speed'},'turn':{'direction','degrees'},'step_back':{'distance'},'side_step':{'direction','distance'},'sit':{'height'},'stop':set(),'climb_up':set(),'climb_down':set()}
        if set(kwargs)-allowed[name]:raise ValueError('Unexpected action arguments')
        result=self.native('act',name,kwargs,visible);self.acted=True
        self.log('native_action',turn=self.code_turn,primitive=name,arguments=kwargs,visible_state=visible,result=result)
        return result

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path!='/call' or not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+self.server.token):self.send_error(403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=1000000:raise ValueError('Invalid request length')
            p=json.loads(self.rfile.read(length))
            with self.server.call_lock:r=self.server.session.call(p['primitive'],p.get('args',[]),p.get('kwargs',{}),p.get('client',{}))
            payload={'ok':True,'result':r}
        except Exception as e:payload={'ok':False,'error':f'{type(e).__name__}: {e}'}
        body=json.dumps(payload).encode();self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def log_message(self,*args):pass

def main():
    p=argparse.ArgumentParser();p.add_argument('--instance-id',default='humanclaw_full_0');p.add_argument('--suite',choices=['fullval'],default='fullval');p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--max-env-steps',type=int,default=100);p.add_argument('--code-timeout-seconds',type=int,default=300)
    p.add_argument('--solution',type=Path);p.add_argument('--workspace-dir',type=Path)
    repl_runtime.add_runtime_args(p,default_max_code_turns=101);add_executor_args(p);a=p.parse_args()
    resolve_or_exit_openhands(a,benchmark='HumanCLAW',track='egocentric')
    if a.executor=='openhands-headless':p.error('This adapter currently validates codex-exec and probe only')
    if not 1<=a.max_env_steps<=100:p.error('max-env-steps must preserve the native maximum100')
    pool=json.loads(Path(os.environ.get('EMBODIED_ARENA_W3_TASK_POOL',ROOT/'pool.json')).read_text());task=pool['tasks'][a.start_index]
    upstream=Path(os.environ.get('HUMANCLAW_ROOT',str(EXTERNAL/'upstreams/humanclaw')))
    source_lock=json.loads((ROOT/'source_lock.json').read_text())
    for path,digest in source_lock.items():
        if hashlib.sha256((upstream/path).read_bytes()).hexdigest()!=digest:raise ValueError('Native source changed after integration validation')
    for path,digest in pool['shards'].items():
        if hashlib.sha256((upstream/path).read_bytes()).hexdigest()!=digest:raise ValueError('Native task shard changed after sampling')
    run_id='humanclaw_'+uuid.uuid4().hex;outputs=Path(os.environ.get('HUMANCLAW_OUTPUT_ROOT', str(ROOT/'outputs')))/run_id;outputs.mkdir(parents=True,exist_ok=False)
    workspace=a.workspace_dir or outputs/'workspace';repl_runtime.create_workspace(workspace,allowed_root=outputs)
    for name in ['primitive_api.py','primitive_cards.md']:shutil.copy2(ADAPTER/name,workspace/name)
    (workspace/'cell.py').write_text(a.solution.read_text() if a.solution else 'import primitive_api as p\nprint(p.observe())\n')
    session=Session(a,task,workspace,outputs,run_id);server=None;metrics=None
    try:
        session.worker.submit(session.initialize).result()
        context=session.native('context')
        (workspace/'task.md').write_text(context['instruction']+'\n\nYou control a physical humanoid through egocentric RGB. First observe and yield; then one native action per code cell. Do not assume commanded movement succeeded. Stop commits the current pose.\n')
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.session=session;server.token=secrets.token_urlsafe(32);server.call_lock=threading.RLock()
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        env={**os.environ,'ROBENCH_PRIMITIVE_SERVER_URL':f'http://127.0.0.1:{server.server_port}','ROBENCH_PRIMITIVE_SERVER_TOKEN':server.token}
        def feedback(turn,max_turns,proc,timed_out):
            observation=session.native('observe',workspace)
            return repl_runtime.write_turn_feedback_files(session=session,workspace_dir=workspace,turn_index=turn,max_turns=max_turns,completed=proc,timed_out=timed_out,payload={'success':False,'terminal':observation['terminal'],'env_steps':observation['env_steps'],'remaining_env_steps':a.max_env_steps-observation['env_steps'],'current_observation':observation,'verifier':{'available_to_agent':False,'note':'Native metrics collected privately; no hidden-state feedback'}})
        completed,timed_out,events,label,source,attempts=repl_runtime.run_persistent_repl_code(args=a,session=session,workspace_dir=workspace,cell_path=workspace/'cell.py',openhands_binary=None,env=env,outputs_dir=outputs,benchmark_label='HumanCLAW',feedback_writer=feedback)
        steps=session.native('context')['env_steps'];metrics=session.native('finish')
        success_metrics=(metrics or {}).get('success',{});metric='interact_sr' if success_metrics.get('is_interact_episode') else 'nav_sr_20cm';success=bool(success_metrics.get(metric,False))
        code_errors=sum(t.get('cell_returncode') not in (None,0) for t in events.get('turns',[]))
        summary=dict(run_id=run_id,benchmark='HumanCLAW',runtime_mode=a.runtime_mode,executor_source=source,instance_id=a.instance_id,requested_tasks=1,completed_tasks=int(metrics is not None),successes=int(success),avg_score=float(success),avg_code_attempts=attempts,timeout_count=int(timed_out),exception_count=int(metrics is None),code_exception_count=0,event_log_paths=events,native_metrics=metrics,native_metrics_path=str(session.episode.native_output/'metrics.json'),summary_success_metric=metric,official_score_eligible=False,policy_boundary='Pukun coding policy replaces native planner/action-selection verifier; original motion/physics and scoring metrics retained',tasks=[dict(task_id=task['task_id'],score=float(success),success=success,env_steps=steps,stopped_reason=events.get('episode_loop',{}).get('stop_reason'))])
        summary['code_exception_count']=code_errors
        (outputs/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
        return 0 if metrics is not None and not timed_out else 1
    finally:
        if server is not None:server.shutdown();server.server_close()
        if session.episode is not None and metrics is None and session.episode.env is not None:
            session.worker.submit(session.episode.env.close).result()
        session.worker.shutdown(wait=True)
if __name__=='__main__':raise SystemExit(main())
