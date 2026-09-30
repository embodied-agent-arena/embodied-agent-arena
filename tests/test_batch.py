"""Batch orchestration: limits, independent trials, recovery, and real child adoption."""
from pathlib import Path
import copy
import fcntl
import json
import os
import subprocess
import sys
import time

import pytest

from embodied_harness import batch, release

ROOT = Path(__file__).resolve().parents[1]


def task(name, *, gpu=0, group='offline', wave='W1'):
    # Distinct tasks can share case_id; the supervisor must use task_id.
    return dict(task_id=name, case_id='shared-case', benchmark_id='capx' if group.startswith('capx_') else group,
                wave=wave, runner_args=[], env={'ARENA_REPORTING_BENCHMARK': group},
                resources={'gpu_memory_gb': gpu}, budgets={}, enabled=True, seed=0, split='test', variation=name)


def plan(tmp_path, *, models=None, tasks=None, workers=3, **overrides):
    value = dict(project_root=str(ROOT), data_root=str(tmp_path),
                 models=models or {'a':dict(model='custom-a', provider='openai-compatible', workers=3, rounds=1),
                                   'b':dict(model='custom-b', provider='openai-compatible', workers=2, rounds=1)},
                 tasks=tasks or [task('one'), task('two'), task('three')],
                 max_workers=workers, batch_size=1, max_attempts=2, retry_on=['network'],
                 backoff_seconds=0, max_backoff_seconds=0, poll_seconds=.02,
                 resources=dict(gpus=['0','1'], gpu_memory_gb=24, gpu_reserve_gb=2),
                 benchmark_limits={}, port_base=29000)
    value.update(overrides)
    value['plan_sha256'] = batch.digest(value)
    return value


def simulated_supervisor(tmp_path, p):
    supervisor = batch.Supervisor(tmp_path/'out', p)
    supervisor.spawn = lambda job: job.update(state='running')
    supervisor.obtain_gpu_lease = lambda: True
    return supervisor


def test_dynamic_allocation_and_arbitrary_model_count(tmp_path):
    p = plan(tmp_path, tasks=[task(str(i)) for i in range(9)], workers=5,
             models={n:dict(model='api-'+n, provider='openai-compatible', workers=4, rounds=1) for n in ['x','y','z']})
    s = simulated_supervisor(tmp_path, p)
    s.schedule()
    assert len(s.state['jobs']) == 5
    assert {j['alias'] for j in s.state['jobs']} == {'x','y','z'}
    for job in s.state['jobs']:
        job['state'] = 'finished'
        for key in job['keys']:
            s.state['slots'][key]['state'] = 'completed'
    s.state['paused_models'] = {'x':'credentials', 'y':'credentials'}
    s.schedule()
    active = [j for j in s.state['jobs'] if j['state'] in batch.ACTIVE]
    assert len(active) == 4 and all(j['alias'] == 'z' for j in active)


def test_gpu_memory_and_benchmark_admission(tmp_path):
    tasks = [task(str(i), gpu=12, group='capx_libero_pro' if i%2 else 'capx_robosuite', wave='W4') for i in range(5)]
    s = simulated_supervisor(tmp_path, plan(tmp_path, tasks=tasks, workers=8,
        models={'a':dict(model='m', provider='openai-compatible', workers=8, rounds=1)}))
    s.schedule()
    active = [j for j in s.state['jobs'] if j['state'] in batch.ACTIVE]
    assert len(active) == 2
    assert {j['allocation']['gpu'] for j in active} == {'0','1'}
    assert len({j['port_base'] for j in active}) == 2
    s = simulated_supervisor(tmp_path/'limited', plan(tmp_path, tasks=tasks, workers=8, benchmark_limits={'capx':1}))
    s.schedule()
    assert len(s.state['jobs']) == 1


def test_exclusive_gpu_jobs_do_not_share_a_device(tmp_path):
    tasks = [task(str(i), gpu=4, wave='W4') for i in range(4)]
    tasks[0]['resources']['exclusive_gpu'] = True
    s = simulated_supervisor(tmp_path, plan(tmp_path, tasks=tasks, workers=8,
        models={'a':dict(model='m', provider='openai-compatible', workers=8, rounds=1)}))
    s.schedule()
    active = [j for j in s.state['jobs'] if j['state'] in batch.ACTIVE]
    assert len(active) == 4  # One exclusive job, three shared jobs on the other GPU.
    assert sum(j['allocation']['exclusive'] for j in active) == 1
    exclusive_device = next(j['allocation']['gpu'] for j in active if j['allocation']['exclusive'])
    assert sum(j['allocation']['gpu'] == exclusive_device for j in active) == 1


def test_gpu_launch_intent_waits_for_cooperative_lease(tmp_path):
    p = plan(tmp_path, tasks=[task('gpu', gpu=4)], models={'a':dict(model='m', workers=1, rounds=1)})
    s = simulated_supervisor(tmp_path, p)
    s.schedule()
    job = s.state['jobs'][0]
    job['state'] = 'starting'
    s.spawn = batch.Supervisor.spawn.__get__(s)
    s.obtain_gpu_lease = lambda: False
    s.refresh()
    assert not s.children
    assert job['state'] == 'starting'


def test_valid_wrong_answers_and_terminal_failures_are_not_network_retried():
    assert batch.classify({'outcome':'task_failure'}, {'native_result':{'submission_valid':True, 'passed':False}}) == 'completed'
    assert batch.classify({'outcome':'budget_exhausted'}, {'outcome':'budget_exhausted'}) == 'completed'
    assert batch.classify({'outcome':'runtime_failure'}, {}) == 'runtime'
    for code, kind in [(429,'network'),(503,'network'),(401,'credentials'),(400,'configuration')]:
        assert batch.classify({'outcome':'agent_failure'}, {'exception':{'category':'model_api','status_code':code}}) == kind


FAKE_RUNNER = '''
import json, pathlib, sys, time
folder=pathlib.Path(sys.argv[1]); mode=sys.argv[2]; delay=float(sys.argv[3])
doc=json.loads((folder/'manifest.json').read_text());time.sleep(delay)
for t in doc['tasks']:
    target=folder/'campaign/batches/000001';report=target/'tasks'/t['task_id']/'runner_report.json'
    report.parent.mkdir(parents=True,exist_ok=True)
    if mode=='network':
        payload={'exception':{'category':'model_api','status_code':503},'outcome':'agent_failure'}
    elif mode=='auth':
        payload={'exception':{'category':'model_api','status_code':401},'outcome':'agent_failure'}
    else:
        payload={'native_result':{'submission_valid':True,'passed':False},'outcome':'task_failure'}
    report.write_text(json.dumps(payload))
    status=target/'statuses'/(t['task_id']+'.json');status.parent.mkdir(exist_ok=True)
    status.write_text(json.dumps({'task_id':t['task_id'],'task':t,'outcome':payload['outcome'],'artifacts':{'runner_report':str(report)}}))
'''


def fake_factory(supervisor, file, behavior, delay=.01):
    file.write_text(FAKE_RUNNER)
    def command(job):
        slots = [supervisor.state['slots'][k] for k in job['keys']]
        return [sys.executable, str(file), str(supervisor.out/'jobs'/job['id']), behavior(slots), str(delay)]
    supervisor.command_factory = command


def test_real_processes_retry_once_and_preserve_rounds_and_task_ids(tmp_path):
    p = plan(tmp_path, models={'custom':dict(model='anything', provider='openai-compatible', workers=3, rounds=2)},
             tasks=[task('one'), task('two')])
    s = batch.Supervisor(tmp_path/'out', p)
    fake_factory(s, tmp_path/'fake.py', lambda ss:'network' if ss[0]['task_id']=='one' and ss[0]['attempts']==1 else 'wrong')
    assert s.run() == 0
    assert len(s.state['slots']) == 4
    assert sorted(slot['attempts'] for slot in s.state['slots'].values()) == [1,1,2,2]
    assert all(slot['state']=='completed' for slot in s.state['slots'].values())
    assert all(slot['result']['outcome']=='task_failure' for slot in s.state['slots'].values())
    assert len(s.state['jobs']) == 6
    resumed = batch.Supervisor(s.out, p, resume=True)
    assert resumed.run() == 0
    assert len(resumed.state['jobs']) == 6


def test_restart_adopts_live_wrapper_without_duplicate_attempt(tmp_path):
    p = plan(tmp_path, models={'a':dict(model='m', provider='openai-compatible', workers=1, rounds=1)}, tasks=[task('only')])
    first = batch.Supervisor(tmp_path/'out', p)
    fake_factory(first, tmp_path/'fake.py', lambda ss:'wrong', delay=.4)
    first.schedule()
    resumed = batch.Supervisor(first.out, p, resume=True)
    fake_factory(resumed, tmp_path/'fake2.py', lambda ss:'wrong')
    assert resumed.run() == 0
    assert len(resumed.state['jobs']) == 1
    assert next(iter(resumed.state['slots'].values()))['attempts'] == 1
    for child in first.children.values():
        child.wait(timeout=5)


def test_campaign_process_inherits_gpu_reservation_without_using_gpu(tmp_path):
    p = plan(tmp_path, models={'a':dict(model='m', provider='openai-compatible', workers=1, rounds=1)},
             tasks=[task('only', gpu=1, wave='W4')])
    s = batch.Supervisor(tmp_path/'out', p)
    s.gpu_lease = (tmp_path/'reservation.lock').open('a')
    fcntl.flock(s.gpu_lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    script = tmp_path/'fake.py'
    fake_factory(s, script, lambda ss:'wrong')
    script.write_text("import os\nos.fstat(int(os.environ['ARENA_BATCH_GPU_LEASE_FD']))\n" + FAKE_RUNNER)
    assert s.run() == 0


def test_retry_bound_and_auth_failure_do_not_spin(tmp_path):
    p = plan(tmp_path, models={'a':dict(model='m', provider='openai-compatible', workers=1, rounds=1)}, tasks=[task('only')])
    s = batch.Supervisor(tmp_path/'network', p)
    fake_factory(s, tmp_path/'network.py', lambda ss:'network')
    assert s.run() == 2
    assert next(iter(s.state['slots'].values()))['attempts'] == 2
    assert next(iter(s.state['slots'].values()))['state'] == 'failed'
    s = batch.Supervisor(tmp_path/'auth', p)
    fake_factory(s, tmp_path/'auth.py', lambda ss:'auth')
    assert s.run() == 2
    assert s.state['paused_models'] == {'a':'credentials'}
    assert next(iter(s.state['slots'].values()))['attempts'] == 1
    resumed = batch.Supervisor(s.out, p, resume=True, retry_paused=True)
    fake_factory(resumed, tmp_path/'fixed.py', lambda ss:'wrong')
    assert resumed.run() == 0


def test_resume_rejects_changed_plan(tmp_path):
    p = plan(tmp_path)
    s = batch.Supervisor(tmp_path/'out', p);s.save()
    changed=copy.deepcopy(p);changed['models']['a']['model']='changed';changed['plan_sha256']=batch.digest(changed)
    with pytest.raises(ValueError,match='configuration differs'):
        batch.Supervisor(s.out,changed,resume=True)


def test_w4_command_keeps_native_limits_and_uses_generic_model_profile(tmp_path):
    p = plan(tmp_path,tasks=[task('one',gpu=10,group='capx_robosuite',wave='W4')])
    s=simulated_supervisor(tmp_path,p);s.schedule();job=s.state['jobs'][0];cmd=s.command(job)
    assert cmd[cmd.index('--max-output-tokens')+1]=='3600'
    assert cmd[cmd.index('--request-timeout')+1]=='720'
    assert '--retry-http' not in cmd  # Whole-case retries have one owner.
    doc=batch.read(s.out/'jobs'/job['id']/'manifest.json')
    assert doc['tasks'][0]['env']['AGENTIC_EMBODIED_ARENA_CAPX_SAM3_PORT']==str(job['port_base'])


def test_api_overrides_reach_native_runner_and_cli_keeps_native_flags(tmp_path):
    selected = task('one', gpu=10, wave='W4')
    selected['runner_args'] = ['--max-tokens', '3600', '--request-timeout-seconds=720', '--trial-timeout-seconds', '10800']
    p = plan(tmp_path, tasks=[selected], models={
        'api':dict(provider='openai-compatible', model='m', workers=1, rounds=1, max_output_tokens=5000, request_timeout=900),
        'cli':dict(provider='codex-exec', model='m', workers=1, rounds=1)})
    s = simulated_supervisor(tmp_path, p); s.schedule()
    jobs = {j['alias']:j for j in s.state['jobs']}
    api = batch.read(s.out/'jobs'/jobs['api']['id']/'manifest.json')['tasks'][0]
    assert release.option(api['runner_args'], '--max-tokens') == '5000'
    assert release.option(api['runner_args'], '--request-timeout-seconds') == '900'
    assert release.option(api['runner_args'], '--trial-timeout-seconds') == '10800'
    cmd = s.command(jobs['cli'])
    assert '--max-output-tokens' not in cmd and '--request-timeout' not in cmd
    cli = batch.read(s.out/'jobs'/jobs['cli']['id']/'manifest.json')['tasks'][0]
    assert cli['runner_args'] == selected['runner_args']


def test_example_configuration_is_plan_only_and_honors_custom_counts(tmp_path):
    try:
        data=release.dataset_root(None)
    except ValueError:
        pytest.skip('Task dataset not installed')
    cfg=batch.read(ROOT/'configs/batch.example.json');cfg['data_root']=str(data)
    cfg.update(rounds=3,max_workers=11,workers_per_model=6)
    cfg['models']=[{'name':'a','model':'not-a-fixed-model'},{'name':'b','model':'another-model','rounds':2,'workers':9}]
    file=tmp_path/'batch.json';batch.atomic(file,cfg)
    p=batch.make_plan(file,ROOT)
    assert p['max_workers']==11
    assert p['models']['a']['rounds']==3 and p['models']['b']['rounds']==2
    assert p['models']['b']['workers']==9
    assert len(p['tasks'])==660
    assert batch.main(['--config',str(file),'--output-dir',str(tmp_path/'out')])==0
    assert not (tmp_path/'out').exists()
    cfg['models'][0]['api_key']='do-not-save-this'
    batch.atomic(file,cfg)
    with pytest.raises(ValueError,match='Invalid model fields'):
        batch.make_plan(file,ROOT)


def test_preflight_uses_each_task_route_and_restores_parent(monkeypatch):
    from embodied_harness import campaign, native_runtime_receipt
    called=[]
    monkeypatch.setenv('ARENA_REPORTING_BENCHMARK','parent')
    def check(benchmark):
        called.append((benchmark,os.environ['ARENA_REPORTING_BENCHMARK'],os.environ['EMBODIED_ARENA_NATIVE_RECEIPT_ROOT']))
        return {'ok':True}
    monkeypatch.setattr(native_runtime_receipt,'validate_native_runtime_launch_receipt',check)
    tasks=[task('a',group='capx_robosuite',wave='W4'),task('b',group='capx_libero_pro',wave='W4')]
    for t in tasks:t['env']['EMBODIED_ARENA_NATIVE_RECEIPT_ROOT']='/receipts/'+batch.group(t)
    result=campaign.runtime_preflight({'tasks':tasks})
    assert len(called)==2 and len(result)==2
    assert os.environ['ARENA_REPORTING_BENCHMARK']=='parent'
    assert 'EMBODIED_ARENA_NATIVE_RECEIPT_ROOT' not in os.environ
