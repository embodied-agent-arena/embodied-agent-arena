import hashlib,json
from pathlib import Path
import pytest
from embodied_harness.robotwin2_agent_runtime import RoboTwin2AgentRuntimeBackend,RoboTwin2RuntimeConfig

@pytest.fixture
def bound_input(tmp_path,monkeypatch):
    repo=tmp_path/'repo';(repo/'envs').mkdir(parents=True);(repo/'script').mkdir();(repo/'envs/beat_block_hammer.py').write_text('class beat_block_hammer: pass\n');source=repo/'script/eval_policy.py';source.write_text('native evaluation source\n')
    index=tmp_path/'index.json';data={'task':'beat_block_hammer','task_config':'demo_clean','user_seed':0,'native_test_num':100,'generator_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'complete_object_assets':True,'episodes':[{'episode_index':0,'seed':100007,'episode_info':{'info':{'{A}':'hammer/base0'}}}]};index.write_text(json.dumps(data));monkeypatch.setenv('ROBOTWIN2_NATIVE_SEED_INDEX',str(index));monkeypatch.setenv('ROBOTWIN2_NATIVE_SEED_INDEX_SHA256',hashlib.sha256(index.read_bytes()).hexdigest())
    backend=RoboTwin2AgentRuntimeBackend(RoboTwin2RuntimeConfig(repo_path=str(repo)));coord={'task_id':'beat_block_hammer','variation':'demo_clean::beat_block_hammer::accepted_episode_000','seed':0};return backend,coord,index,data

def test_ordinal_resolves_actual_accepted_seed_not_arithmetic_seed(bound_input):
    backend,coord,_,_=bound_input;r=backend.bind_pool_coordinate(coord);assert r['bound'] and r['native_seed']==100007 and r['user_seed']==0;assert backend._selected_native_episode['native_seed']==100007;assert 'episode_info' not in r

def test_unmaterialized_episode_is_rejected(bound_input):
    backend,coord,_,_=bound_input;coord['variation']='demo_clean::beat_block_hammer::accepted_episode_001'
    with pytest.raises(ValueError,match='not been materialized'):backend.bind_pool_coordinate(coord)

def test_changed_index_is_rejected(bound_input):
    backend,coord,index,data=bound_input;data['episodes'][0]['seed']=100008;index.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='digest changed'):backend.bind_pool_coordinate(coord)

def test_incomplete_assets_index_is_rejected(bound_input,monkeypatch):
    backend,coord,index,data=bound_input;data['complete_object_assets']=False;index.write_text(json.dumps(data));monkeypatch.delenv('ROBOTWIN2_NATIVE_SEED_INDEX_SHA256')
    with pytest.raises(ValueError,match='complete assets'):backend.bind_pool_coordinate(coord)


def test_native_verifier_context_restores_expert_metadata_without_actions():
    from types import SimpleNamespace
    from embodied_harness.robotwin2_agent_runtime import _restore_native_verifier_context
    module = SimpleNamespace(ArmTag=lambda arm: ("native_arm", arm))
    for task in ("open_laptop", "place_object_scale", "put_object_cabinet"):
        env = SimpleNamespace(object=SimpleNamespace(get_pose=lambda: SimpleNamespace(p=[.1, .2, .73])))
        _restore_native_verifier_context(env, module, task, {"info": {"{a}": "right"}})
        assert env.arm_tag == ("native_arm", "right")
        assert getattr(env, "origin_z", None) == (.73 if task == "put_object_cabinet" else None)
    env = SimpleNamespace()
    _restore_native_verifier_context(env, module, "beat_block_hammer", {})
    assert vars(env) == {}

def test_native_verifier_context_rejects_missing_accepted_arm():
    from types import SimpleNamespace
    import pytest
    from embodied_harness.robotwin2_agent_runtime import _restore_native_verifier_context
    with pytest.raises(ValueError, match="arm metadata"):
        _restore_native_verifier_context(SimpleNamespace(), SimpleNamespace(), "open_laptop", {"info": {}})
