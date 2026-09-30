import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from embodied_harness.w1_runtime import W1Session
from embodied_harness.w1_scoring import score_case

K = dict(fx=1000., fy=1000., cx=500., cy=375.)

def case(k):
    return dict(benchmark='influx', sample_id='test', gt={'intrinsics':k},
                images=[dict(width=1000,height=750,path='a.png',role='frame')])

@pytest.mark.parametrize('value', [float('nan'),float('inf'),None,False])
@pytest.mark.parametrize('field', list(K))
def test_invalid_truth_is_unscored_and_json_safe(value, field):
    result=score_case(K,case({**K,field:value}))
    assert result['evaluation_valid'] is False
    assert result['passed'] is None and result['official_score'] is None
    json.dumps(result,allow_nan=False)

def test_session_preserves_answer_without_counting_invalid_gt_as_failure(tmp_path):
    s=W1Session(case({**K,'fx':float('nan')}),tmp_path,tmp_path,tmp_path,'run',executor='probe')
    s.submit({'answer':K})
    result=s.finish()
    assert result['submission_valid'] is True
    assert result['prediction']==K
    assert result['task_outcome']=='invalid_ground_truth'
    assert result['passed'] is None and result['official_score_eligible'] is False
    assert 'gt' not in s.observe()['task_summary']
    json.dumps(result,allow_nan=False)

@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1.,0.])
def test_invalid_prediction_does_not_pass(value):
    r=score_case({**K,'fx':value},case(K))
    assert r['passed'] is False and r['invalid_reason']=='invalid_prediction'
    json.dumps(r,allow_nan=False)

def assets_module():
    path=Path(__file__).parents[1]/'scripts/deploy_w1_assets.py'
    spec=importlib.util.spec_from_file_location('w1_assets_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def test_extrapolation_is_explicit_and_zero_principal_point_is_valid():
    a=assets_module()
    record={'intrinsics_gt':{k:float('nan') for k in K},'intrinsics_gt_extrapolated':K}
    assert a.select_influx_intrinsics(record)==(None,None)
    assert a.select_influx_intrinsics(record,allow_extrapolated=True)==(K,'intrinsics_gt_extrapolated')
    native={**K,'cx':0.,'cy':0.}
    assert a.select_influx_intrinsics({'intrinsics_gt':native},allow_extrapolated=True)==(native,'intrinsics_gt')

def test_sampler_fills_requested_count_using_valid_frames(tmp_path,monkeypatch):
    a=assets_module();d=tmp_path/'dataset/influx';d.mkdir(parents=True)
    bad={'intrinsics_gt':{k:float('nan') for k in K},'intrinsics_gt_extrapolated':K}
    good={'intrinsics_gt':K}
    (d/'gt_validation_dict_v1.json').write_text(json.dumps({'bad':{'0':bad,'1':bad},'good':{'0':bad,'1':good,'2':good}}))
    (d/'video_frame_count_and_split_v1.json').write_text(json.dumps({n:{'split':'val'} for n in ['bad','good']}))
    monkeypatch.setattr(a,'ranked',lambda items,**kw:sorted(items))
    def download(url,dest):dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b'fixture')
    def extract(video,dest,index):dest.parent.mkdir(parents=True,exist_ok=True);Image.new('RGB',(8,8)).save(dest);return True
    monkeypatch.setattr(a,'download',download);monkeypatch.setattr(a,'extract_frame',extract)
    result=a.materialize_influx(tmp_path,videos=1,frames_per_video=2)
    assert [r['sample_id'] for r in result]==['good_000001','good_000002']
    assert all(r['gt_provenance']['field']=='intrinsics_gt' for r in result)
    json.dumps(result,allow_nan=False)
