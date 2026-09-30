from pathlib import Path

from PIL import Image

from embodied_harness.w1_runtime import W1Session, load_catalog
from embodied_harness.w1_scoring import parse_submission, score_case


def _case(**overrides):
    case = {
        "sample_id": "demo",
        "benchmark": "multispa",
        "task_family": "camera_translation_distance",
        "answer_type": "scalar",
        "units": "mm",
        "question": "How far did the camera move?",
        "images": [{"path": "images/a.png", "role": "reference", "width": 8, "height": 8}],
        "gt": {"value": 100.0},
    }
    case.update(overrides)
    return case


def test_multispa_relative_l2_threshold():
    case = _case()
    passed = score_case(parse_submission("95"), case)
    failed = score_case(parse_submission("50"), case)
    assert passed["passed"] is True
    assert failed["passed"] is False


def test_influx_k_thresholds():
    case = _case(
        benchmark="influx",
        answer_type="intrinsics",
        images=[{"path": "images/a.png", "role": "frame", "width": 1000, "height": 750}],
        gt={"intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 375.0}},
    )
    good = score_case({"fx": 1040, "fy": 960, "cx": 510, "cy": 370}, case)
    bad = score_case({"fx": 700, "fy": 700, "cx": 100, "cy": 100}, case)
    assert good["passed"] is True
    assert bad["passed"] is False


def test_mapfree_pose_joint_threshold():
    case = _case(
        benchmark="mapfree",
        answer_type="pose_wxyz_t",
        gt={"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "tx": 0.10, "ty": 0.0, "tz": 0.0},
    )
    good = score_case({"qw": 1, "qx": 0, "qy": 0, "qz": 0, "tx": 0.11, "ty": 0, "tz": 0}, case)
    bad = score_case({"qw": 1, "qx": 0, "qy": 0, "qz": 0, "tx": 1.0, "ty": 0, "tz": 0}, case)
    assert good["passed"] is True
    assert abs(good["translation_error_cm"] - 1.0) < 1e-6
    assert bad["passed"] is False


def test_session_observe_submit_matches_w2_loop(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(data / "images" / "a.png")
    catalog = data / "catalog.jsonl"
    catalog.write_text(
        '{"sample_id":"demo","benchmark":"multispa","task_family":"camera_translation_distance",'
        '"answer_type":"scalar","question":"distance","images":[{"path":"images/a.png","role":"reference",'
        '"width":8,"height":8}],"gt":{"value":100}}\n'
    )
    assert "demo" in load_catalog(data)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = W1Session(_case(), data, workspace, tmp_path, "run", executor="probe")
    public = session.observe()
    assert public["observation_format"] == "w2_public_v2"
    assert public["available_actions"] == ["SUBMIT"]
    assert session.max_env_steps == 8
    session.feedback_observation()
    assert (workspace / "frames" / "000.jpg").is_file()
    session.submit({"answer": 100})
    result = session.finish()
    assert result["passed"] is True
    assert result["submission_valid"] is True


def test_transport_jpeg_keeps_native_coordinates(tmp_path):
    from embodied_harness.open_loop_media import TRANSPORT_MAX_EDGE, write_transport_frame

    source = tmp_path / "big.png"
    Image.new("RGB", (2560, 1440), "blue").save(source)
    dest, native, transport = write_transport_frame(source, tmp_path / "frames", 0)
    assert dest.suffix == ".jpg"
    assert dest.stat().st_size < 10 * 1024 * 1024
    assert native == (2560, 1440)
    assert max(transport) <= TRANSPORT_MAX_EDGE


def test_first_turn_submit_after_attach(tmp_path):
    data = tmp_path / "data"
    (data / "images").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(data / "images" / "a.png")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    session = W1Session(_case(), data, workspace, tmp_path, "run", executor="openai-compatible")
    try:
        session.submit({"answer": 100})
        raise AssertionError("submit before attach must fail")
    except ValueError:
        pass
    session.feedback_observation()
    session.submit({"answer": 100})
    assert session.finish()["passed"] is True
