from pathlib import Path

from PIL import Image
import numpy as np

from embodied_harness.w5_runtime import W5Session, load_catalog
from embodied_harness.w5_scoring import parse_submission, score_case


def _case(tmp_path=None, **overrides):
    case = {
        "sample_id": "demo",
        "benchmark": "reasonaff",
        "task_family": "cup-handle",
        "answer_type": "region",
        "question": "grasp the cup",
        "images": [{"path": "images/a.png", "role": "frame", "width": 8, "height": 8}],
        "gt": {"bbox_2d": [2, 2, 6, 6], "point_2d": [4, 4], "mask_path": "masks/a.png"},
    }
    case.update(overrides)
    return case


def test_bbox_threshold_and_point_hit(tmp_path):
    data = tmp_path / "data"
    (data / "masks").mkdir(parents=True)
    mask = np.zeros((8, 8), dtype="uint8")
    mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(data / "masks" / "a.png")
    case = _case()
    good = score_case(parse_submission({"bbox_2d": [2, 2, 6, 6], "point_2d": [4, 4]}), case, data)
    bad = score_case(parse_submission({"bbox_2d": [0, 0, 1, 1], "point_2d": [0, 0]}), case, data)
    assert good["passed"] is True
    assert bad["passed"] is False


def test_session_observe_submit_matches_w2_loop(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    (data / "masks").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(data / "images" / "a.png")
    mask = np.zeros((8, 8), dtype="uint8")
    mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(data / "masks" / "a.png")
    catalog = data / "catalog.jsonl"
    catalog.write_text(
        '{"sample_id":"demo","benchmark":"reasonaff","task_family":"cup-handle",'
        '"answer_type":"region","question":"grasp","images":[{"path":"images/a.png","role":"frame",'
        '"width":8,"height":8}],"gt":{"bbox_2d":[2,2,6,6],"point_2d":[4,4],"mask_path":"masks/a.png"}}\n'
    )
    assert "demo" in load_catalog(data)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = W5Session(_case(), data, workspace, tmp_path, "run", executor="probe")
    public = session.observe()
    assert public["observation_format"] == "w2_public_v2"
    assert public["available_actions"] == ["SUBMIT"]
    assert session.max_env_steps == 8
    session.feedback_observation()
    assert (workspace / "frames" / "000.jpg").is_file()
    session.submit({"answer": {"bbox_2d": [2, 2, 6, 6], "point_2d": [4, 4]}})
    result = session.finish()
    assert result["passed"] is True
    assert result["submission_valid"] is True
