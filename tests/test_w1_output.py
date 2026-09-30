import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from embodied_harness.w1_output import extract_cell, public_question, author_output

CODE='import primitive_api as p\np.submit({"f": 800})'


@pytest.mark.parametrize('text,selection', [
    (CODE,'whole_python'),
    ('Explanation\n```python\n'+CODE+'\n```','python_fence'),
    ('```py\n'+CODE+'\n```','python_fence'),
    ('```\n'+CODE+'\n```','python_fence'),
    ('```json\n{"f":800}\n```\n\n'+CODE,'python_suffix'),
    ('Here is the explanation:\n'+CODE,'python_suffix'),
])
def test_accept_only_existing_python(text,selection):
    r=extract_cell(text)
    assert r['code']==CODE and r['selection']==selection

@pytest.mark.parametrize('text', [
    '', '{"f":800}', '```json\n{"f":800}\n```',
    '```json\n'+CODE+'\n```',
    '```python\n'+CODE+'\n```\n```python\n'+CODE+'\n```',
    '```python\n'+CODE+'\n```\n'+CODE,
    '```python\nx =\n```\n'+CODE,
    'Explanation\n'+CODE+'\nAnother suggestion\n'+CODE,
    '```python\n{"f":800}\n```',
])
def test_no_automatic_answer_wrapping_or_ambiguous_selection(text):
    result=extract_cell(text)
    assert result['synthetic_feedback'] and result['selection']=='format_error'
    with pytest.raises(ValueError,match='W1 output format'):
        exec(result['code'],{})

def test_does_not_rewrite_values_or_json_booleans():
    code='import primitive_api as p\np.submit({"moved": false, "direction":"none", "distance_cm":0})'
    assert extract_cell(code)['code']==code

def test_scoped_hook_restores_shared_author_on_exception(tmp_path,monkeypatch):
    sentinel=lambda text,extract=None: 'legacy'
    module=SimpleNamespace(usable_python_cell=sentinel)
    monkeypatch.setitem(sys.modules,'codex_cell_author',module)
    path=tmp_path/'extract.jsonl'
    with pytest.raises(RuntimeError):
        with author_output(path):
            assert module.usable_python_cell('```json\n{}\n```\n'+CODE)==CODE
            raise RuntimeError('test')
    assert module.usable_python_cell is sentinel
    row=json.loads(path.read_text())
    assert row['code']==CODE and row['selection']=='python_suffix' and 'raw_reply' in row

def test_public_format_preserves_coordinate_constraints_and_catalog():
    from embodied_harness.release import dataset_root
    try: root=dataset_root(None)
    except ValueError: pytest.skip('Download the optional dataset integration fixtures first')
    cases=[json.loads(l) for l in (root/'assets/w1_geoprobe/catalog.jsonl').read_text().splitlines()]
    for c in cases:
        original=c['question'];formatted=public_question(original)
        assert c['question']==original
        assert '只输出' not in formatted and 'p.submit' in formatted
        assert 'True, False and None' in formatted
        assert 'task_summary, transport_images and remaining_budget' in formatted
        if c['benchmark']=='vlm_pose':assert 'Rz(rz) @ Ry(ry) @ Rx(rx)' in formatted
        if c['benchmark']=='vlm_depth':assert '24' in formatted and 'None' in formatted


