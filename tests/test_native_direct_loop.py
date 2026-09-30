from types import SimpleNamespace
import json
import pytest
from embodied_harness import native_agent_loop as m
from embodied_harness.backend import EmbodiedBackend
from embodied_harness.schemas import TaskSpec,Observation,PrimitiveCard,PrimitiveResult,VerificationResult,EpisodeTrace

class EpisodeBackend(EmbodiedBackend):
    def __init__(self):
        self.resets=0;self.value=0;self.verifications=[];self.actions=[]
    def reset(self,task_id,seed=None,config=None):
        self.resets+=1;self.trace=EpisodeTrace(task_id=task_id)
        return TaskSpec(task_id=task_id,source='test',instruction='Increment twice',budgets={'primitive_calls':64,'verifier_calls':4})
    def observe(self):return Observation(step=0,data={'value':self.value})
    def list_primitives(self,level=None):return [PrimitiveCard(name='native_increment',capability_tags=['action'],input_schema={'amount':'integer'})]
    def call_primitive(self,name,**kwargs):
        self.actions.append((name,kwargs));self.value+=kwargs['amount']
        r=PrimitiveResult(name=name,ok=True,output={'value':self.value});self.record_event('primitive_call',{'name':name,'result':r.to_dict()});return r
    def verify(self,scope='task',**kwargs):
        self.verifications.append((id(self),self.value));return VerificationResult(ok=self.value==2,scope=scope,message='native check',metrics={'value':self.value})
    def get_trace(self):return self.trace

class Client:
    def __init__(self):self.turn=0;self.messages=[]
    def complete(self,messages,*,max_tokens):
        self.messages.append(messages.copy());self.turn+=1
        assert 'universal interface'not in messages[0]['content'].lower()
        if self.turn==1:
            assert 'native_increment'in messages[1]['content']
            task_json=messages[1]['content'].split('Task spec JSON:\n')[1].split('\n\n')[0]
            assert json.loads(task_json)['budgets']['verifier_calls']==3
            code="saved_amount=1\nresult=primitives.call(name='native_increment',amount=saved_amount)"
        else:
            feedback=json.loads(messages[-1]['content'].split('Feedback JSON:\n')[1])
            assert feedback['public_progress']['output']['value']==1
            code="result=primitives.call(name='native_increment',amount=saved_amount)"
        return m.ModelCompletion(content=code,prompt_tokens=100,completion_tokens=50,total_tokens=150,cost_usd=None,provider_usage_available=True,usage_mode='provider')

def run(tmp_path,monkeypatch,limit=8):
    backend=EpisodeBackend();client=Client()
    case=SimpleNamespace(case_id='native-test',benchmark_id='maniskill',task_id='native-test',seed=0,reset_config={},code_timeout_seconds=2)
    monkeypatch.setattr(m,'load_native_case',lambda _:case)
    monkeypatch.setattr(m,'make_native_backend',lambda *a,**kw:(backend,None))
    report=m.NativeAgentLoop(model_config=m.NativeModelConfig(),budgets=m.NativeLoopBudgets(max_agent_iterations=3,max_primitive_calls=limit,max_verifier_calls=3),trace_root=tmp_path,client=client,in_process=True,interface_mode='native').run_case(case.case_id)
    return backend,client,report

def test_native_direct_preserves_episode_python_feedback_actions_and_verifier(tmp_path,monkeypatch):
    backend,client,report=run(tmp_path,monkeypatch)
    assert report['ok'],report
    assert report['interface_mode']=='native-primitives'
    assert backend.resets==1 and client.turn==2
    assert backend.actions==[('native_increment',{'amount':1})]*2
    assert backend.verifications==[(id(backend),1),(id(backend),2)]
    assert report['llm_usage']['total_tokens']==300
    trace=json.loads(open(report['agent_attempts'][0]['artifacts']['trace']).read())
    assert sum(e['event_type']=='primitive_call'for e in trace['events'])==2
    calls=[e['payload']for e in trace['events']if e['event_type']=='native_gateway_call']
    assert [c['kwargs']for c in calls]==[{'amount':1}]*2
    assert [c['result']['output']['value']for c in calls]==[1,2]

def test_native_direct_budget_prevents_extra_action(tmp_path,monkeypatch):
    backend,client,report=run(tmp_path,monkeypatch,limit=1)
    assert report['outcome']=='budget_exhausted',report
    assert backend.value==1 and len(backend.actions)==1
    assert backend.verifications==[(id(backend),1)]

@pytest.mark.parametrize('code',["result=backend.verify()","result=primitives._backend.verify()","result=primitives.call(name='unlisted')"])
def test_native_direct_rejects_hidden_verifier_and_unlisted_names(code):
    assert m._universal_code_contract_error(code,native_names={'native_increment'})


def test_nested_image_feedback_has_global_text_bound_without_changing_python_data():
    image = [[[17, 23, 42] for _ in range(480)] for _ in range(480)]
    raw = {"action_schema": {"position": [0.1, 0.2, 0.3]}, "rgb": image}
    projected = m._compact_prompt_value(raw)
    assert len(json.dumps(projected)) < 26000
    assert projected["action_schema"] == raw["action_schema"]
    assert len(raw["rgb"]) == 480 and len(raw["rgb"][479]) == 480
    assert raw["rgb"][479][479] == [17, 23, 42]
    assert "omitted" in json.dumps(projected) or "truncated" in json.dumps(projected)


def test_text_projection_preserves_small_native_feedback_exactly():
    raw = {"ok": True, "output": {"position": [0.1, -0.2, 0.3], "quaternion": [1, 0, 0, 0], "objects": ["cap", "sink"]}}
    assert m._compact_prompt_value(raw) == raw


def test_model_outage_after_native_action_is_runtime_failure(tmp_path, monkeypatch):
    original = Client.complete
    def interrupted(self, messages, *, max_tokens):
        if self.turn:
            raise m.ModelRequestError("Selected model is at capacity", retryable=True)
        return original(self, messages, max_tokens=max_tokens)
    monkeypatch.setattr(Client, "complete", interrupted)
    backend, client, report = run(tmp_path, monkeypatch)
    assert report["outcome"] == "runtime_failure", report
    assert report["exception"]["category"] == "model_api"
    assert backend.resets == 1
    assert backend.verifications == [(id(backend), 1)]
    assert backend.value == 1
