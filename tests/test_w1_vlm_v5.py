"""Integration contracts for the local vlm_test W1 extension."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest
from embodied_harness.w1_runtime import W1Session, load_catalog
from embodied_harness.w1_scoring import score_case
from embodied_harness.vlm_protocols import intrinsics_protocol, rt_protocol, depth_protocol
ROOT = Path(__file__).resolve().parents[1]
from embodied_harness.release import dataset_root
try: DATA = dataset_root(None) / 'assets/w1_geoprobe'
except ValueError: pytest.skip('Download the optional dataset integration fixtures first', allow_module_level=True)
CASES = list(load_catalog(DATA).values())

@pytest.mark.parametrize('case', CASES, ids=lambda c:c['sample_id'])
def test_gold_native_score(case):
    result = score_case(case['gt'], case)
    assert result['submission_valid'] is True
    assert result['passed'] in (None, True)
    assert result.get(result['primary_metric'], 0) == pytest.approx(0, abs=2e-6) or result.get('exact_match') is True
    assert result['official_score'] is None

@pytest.mark.parametrize('benchmark', sorted({c['benchmark'] for c in CASES}))
@pytest.mark.parametrize('prediction', [{}, None, {'x':float('nan')}, {'x':True}])
def test_invalid_predictions(benchmark, prediction):
    case = next(c for c in CASES if c['benchmark']==benchmark)
    assert score_case(prediction, case)['submission_valid'] is False

@pytest.mark.parametrize('benchmark', sorted({c['benchmark'] for c in CASES}))
def test_private_gt_and_single_submission(tmp_path, benchmark):
    case = copy.deepcopy(next(c for c in CASES if c['benchmark']==benchmark))
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    session = W1Session(case, DATA, workspace, tmp_path, 'test', executor='openai-compatible')
    public = session.feedback_observation()
    def verify(value):
        text = json.dumps(value)
        assert '"gt"' not in text and '"provenance"' not in text and '"evaluation"' not in text
        assert str(DATA) not in text and 'catalog.jsonl' not in text
    verify(public)
    with pytest.raises(ValueError, match='observe and yield'):
        session.submit({'answer':case['gt']})
    cell=workspace/'cell.py';cell.write_text('print(1)')
    session.record_agent_code(cell)
    session.feedback_observation()
    session.record_agent_code(cell)
    assert len(public['transport_images']) == len(case['images'])
    verify(session.submit({'answer':case['gt']}))
    assert session.observe()['remaining_budget']['submission_remaining'] == 0
    with pytest.raises(ValueError, match='already submitted'):
        session.submit({'answer':case['gt']})
    result = session.finish()
    assert result['submission_valid'] and result['official_score_eligible'] is False
    if result['passed'] is None: assert result['task_outcome']=='scored'

def test_image_integrity(tmp_path):
    case=copy.deepcopy(CASES[0]);case['images'][0]['sha256']='bad'
    workspace=tmp_path/'ws';workspace.mkdir()
    session=W1Session(case,DATA,workspace,tmp_path,'test',executor='probe')
    with pytest.raises(ValueError,match='hash mismatch'):session.feedback_observation()

def test_motion_zero_denominators():
    stationary=next(c for c in CASES if c['benchmark']=='vlm_motion_fixed' and not c['gt']['moved'])
    result=score_case({'moved':True,'direction':'right','distance_cm':5},stationary)
    assert result['stationary_false_positive'] and result['distance_ape_pct'] is None
    real=next(c for c in CASES if c['benchmark']=='vlm_motion_real')
    result=score_case({'moved':False,'direction':'none','distance_cm':0},real)
    assert result['absolute_error_cm']==2 and result['distance_ape_pct']==100

def test_native_error_parity_and_depth_missingness():
    for c in CASES:
        b=c['benchmark'];p=copy.deepcopy(c['gt'])
        if b=='vlm_intrinsics':
            p['fx']*=1.2
            expected=intrinsics_protocol.errors(p,c['gt'],c['images'][0]['width'],c['images'][0]['height'])
        elif b=='vlm_pose':
            p['translation_cm']=[-v for v in p['translation_cm']]
            expected=rt_protocol.errors(p,c['gt'])
        elif b=='vlm_depth':
            p['depth_m']=[v*2 if i%2 else None for i,v in enumerate(p['depth_m'])]
            expected=depth_protocol.depth_errors(p['depth_m'],c['gt']['depth_m'])
        else:continue
        actual=score_case(p,c)
        for k,v in expected.items():assert actual[k]==v
    c=next(c for c in CASES if c['benchmark']=='vlm_depth')
    r=score_case({'depth_m':[None]*24},c)
    assert r['coverage']==0 and r['abs_rel'] is None

def test_dispatch_all_new_and_legacy():
    spec=importlib.util.spec_from_file_location('dispatcher',ROOT/'scripts/run_benchmark_case.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for b in {c['benchmark'] for c in CASES}|{'multispa','influx','mapfree'}:
        assert m.runner_name(['--benchmark='+b,'--sample-id=x'])=='run_w1_native_case.py'
    assert m.runner_name(['--benchmark','reasonaff','--sample-id','x'])=='run_w5_native_case.py'
    assert m.runner_name(['--benchmark','mindcube','--sample-id','x'])=='run_w2_native_case.py'
