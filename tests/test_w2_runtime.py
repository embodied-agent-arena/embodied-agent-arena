"""Check W2 privacy and actual-media delivery before evidence commitment."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from embodied_harness.w2_runtime import W2Session


@pytest.fixture
def session(tmp_path):
    import base64
    import csv
    import io

    from PIL import Image
    from w2_harness.offline_arena.adapters.multiview import MMSIBenchAdapter

    images = []
    for color in ("red", "blue"):
        payload = io.BytesIO()
        Image.new("RGB", (32, 32), color).save(payload, format="PNG")
        images.append(base64.b64encode(payload.getvalue()).decode("ascii"))
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    with (dataset / "MMSI_bench.tsv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", "image", "question", "answer", "category"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"index": "fixture", "image": repr(images), "question": "Which view is red? Options: A: First; B: Second", "answer": "A", "category": "relative_position"})
    adapter = MMSIBenchAdapter(tmp_path, cache_root=tmp_path / "cache")
    work = tmp_path / "workspace"
    work.mkdir()
    value = W2Session(adapter, "fixture", work, tmp_path, "test-w2", executor="codex-exec")
    try:
        yield value
    finally:
        value.close()
        adapter.close()


def test_media_committed_only_after_next_model_response(session, tmp_path):
    code = tmp_path / "cell.py"
    code.write_text("print('public')")
    session.record_agent_code(code)
    assert not session.observe()["valid_submission_evidence_refs"]
    public = session.feedback_observation()
    assert public["answer_bearing_media_visible"]
    assert list((session.workspace / "frames").glob("*.png"))
    # Prepared images are still provisional until the next response arrives.
    assert not session.observe()["valid_submission_evidence_refs"]
    session.record_agent_code(code)
    assert session.observe()["valid_submission_evidence_refs"]
    assert "data_url" not in json.dumps(public)


def test_evaluator_is_private_and_first_turn_cannot_submit(session, tmp_path):
    client = dict(pid=1234, argv0="solve.py")
    with pytest.raises(ValueError, match="evaluate is private"):
        session.call("evaluate", [], {}, client)
    with pytest.raises(ValueError, match="First observe"):
        session.call("submit", [], dict(answer="A"), client)
    assert session.finish() is None


def test_extra_interpreter_cannot_take_over_episode(session):
    session.call("observe", [], {}, dict(pid=1234, argv0="solve.py"))
    with pytest.raises(ValueError, match="another interpreter"):
        session.call("observe", [], {}, dict(pid=5678, argv0="solve.py"))


def test_perception_augments_feedback_without_committing_evidence_or_spending_actions(session):
    from types import SimpleNamespace
    seen = []
    def observe(descriptors, frames, visible_assets):
        seen.extend(descriptors)
        assert all("data_url" not in d for d in descriptors)
        assert (frames / "000.png").is_file()
        return {"model": "fake", "detections": [{"label": "chair"}]}
    session.perception = SimpleNamespace(observe=observe)
    before = session.observe()["remaining_budget"]
    feedback = session.feedback_observation()
    assert seen and feedback["perception"]["detections"][0]["label"] == "chair"
    assert session.observe()["perception"] == feedback["perception"]
    assert not session.observe()["valid_submission_evidence_refs"]
    assert session.observe()["remaining_budget"] == before


def test_duplicate_payloads_attach_once_and_preserve_native_bindings(session, tmp_path):
    first = session.feedback_observation()
    session.max_images = len(first["transport_images"])
    client = dict(pid=1234, argv0="solve.py")
    asset = first["transport_images"][0]["bindings"][0]["asset_id"]
    before = session.observe()["remaining_budget"]["environment_actions"]
    action = "GET_VIEW" if "GET_VIEW" in first["available_actions"] else "OPEN_ASSET"
    result = session.call("step", [], dict(action=action, arguments={"asset_id": asset}), client)
    assert result["remaining_budget"]["environment_actions"] == before - 1
    second = session.feedback_observation()
    assert any(len(image["bindings"]) > 1 for image in second["transport_images"])
    assert len(second["transport_images"]) == len(first["transport_images"])
    assert len(list((session.workspace / "frames").glob("*.png"))) == len(first["transport_images"])
    refs = {b["evidence_ref"] for i in second["transport_images"] for b in i["bindings"]}
    assert refs <= set(second["valid_submission_evidence_refs"])
    assert not session.observe()["valid_submission_evidence_refs"]
    code = tmp_path / "cell.py"
    code.write_text("print('read images')")
    session.record_agent_code(code)
    assert refs <= set(session.observe()["valid_submission_evidence_refs"])


def test_compact_feedback_preserves_task_and_removes_audit_fields(session):
    public = session.feedback_observation()
    native = session.environment.observe().to_dict()
    assert public["task_summary"]["question"] == native["task_summary"]["question"]
    assert public["remaining_budget"] == native["remaining_budget"]
    assert "current_evidence_index" not in public
    assert "source_media_hash" not in json.dumps(public)
    assert "public_observation_full" in (session.outputs / "events.jsonl").read_text()
    stdout = "42\n" + str(public) + "\ncomputed direction: left"
    filtered = session.model_stdout(stdout)
    assert "42" in filtered and "computed direction: left" in filtered
    assert "w2_public_v2" not in filtered
    assert "included in current_observation" in session.model_stdout(json.dumps(public, indent=2))
