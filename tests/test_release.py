"""Release boundaries: portable tasks, selected initial states, and offline scoring."""
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from embodied_harness import release

ROOT = Path(__file__).resolve().parents[1]
PUKUN = ROOT / "runtimes/pukun"


@pytest.fixture
def dataset():
    try:
        return release.dataset_root(None)
    except ValueError:
        pytest.skip("Download the task dataset to run data integration checks")


def test_paths_cannot_escape_dataset(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "linked").symlink_to(outside, target_is_directory=True)
    for path in ("../outside", str(outside), "linked/input.png"):
        with pytest.raises(ValueError, match="escapes"):
            release.confined(data, path)


def test_unknown_selection_is_not_silently_ignored():
    cases = [{"task_id": "a", "wave": "W1", "benchmark_id": "geometry"}]
    with pytest.raises(ValueError, match="Unknown"):
        release.select_cases(cases, SimpleNamespace(wave=[], benchmark=[], case=["b"]))


def test_unresolved_paths_fail_before_launch():
    with pytest.raises(ValueError, match="Unresolved"):
        release.expand_paths({"args": ["${MISSING}/input.png"]}, {})


def test_missing_prediction_is_unsuccessful(dataset, tmp_path):
    task = next(t for t in release.read_cases(dataset) if t["wave"] == "W1")
    predictions = tmp_path / "empty.jsonl"
    predictions.write_text("")
    metrics = release.score_predictions(dataset, [task], predictions)[0]["metrics"]
    assert metrics["passed"] is False
    assert metrics["submission_valid"] is False
    assert metrics["reason"] == "missing_prediction"


def test_duplicate_prediction_is_rejected(dataset, tmp_path):
    task = release.read_cases(dataset)[0]
    predictions = tmp_path / "duplicates.jsonl"
    row = json.dumps({"task_id": task["task_id"], "answer": {}}) + "\n"
    predictions.write_text(row * 2)
    with pytest.raises(ValueError, match="Duplicate prediction"):
        release.score_predictions(dataset, [task], predictions)


def test_task_index_and_default_perception(dataset):
    result = release.check_dataset(dataset)
    assert result["ok"], result["errors"]
    assert result["waves"] == {"W1": 370, "W2": 220, "W3": 157, "W4": 183, "W5": 70}
    cases = release.read_cases(dataset)
    for task in cases:
        if task["wave"] == "W2":
            assert release.option(task["runner_args"], "--perception") == "none"
            assert task["resources"]["gpu_memory_gb"] == 0
            assert not any("yoloe" in arg.lower() for arg in task["runner_args"])


@pytest.mark.parametrize("benchmark,family", [
    ("alfred_official_visual", "ALFREDOfficialBenchmark"),
    ("alfworld_visual", "ALFWorldBenchmark"),
    ("discoveryworld", "DiscoveryWorldBenchmark"),
    ("scienceworld_text", "ScienceWorldBenchmark"),
    ("virtualhome_symbolic", "VirtualHomeBenchmark"),
    ("humanclaw", "HumanCLAWBenchmark"),
])
def test_selected_w3_pools_resolve_original_tasks(dataset, monkeypatch, benchmark, family):
    tasks = [t for t in release.read_cases(dataset) if t["benchmark_id"] == benchmark]
    pool = dataset / "episodes" / benchmark / "pool.json"
    monkeypatch.setenv("EMBODIED_ARENA_W3_TASK_POOL", str(pool))
    monkeypatch.setenv("EMBODIED_ARENA_DATA_ROOT", str(dataset))
    monkeypatch.setenv("EMBODIED_ARENA_EXTERNAL_ROOT", str(ROOT / "external"))
    monkeypatch.syspath_prepend(str(PUKUN / "scripts"))
    monkeypatch.syspath_prepend(str(PUKUN / family / "src"))
    if benchmark in {"alfred_official_visual", "alfworld_visual"}:
        name = "run_visual_instance.py" if benchmark == "alfworld_visual" else "run_instance.py"
        spec = importlib.util.spec_from_file_location("release_" + benchmark, PUKUN / family / "openhands_adapter" / name)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    elif benchmark != "humanclaw":
        package = {"discoveryworld": "discoveryworld_harness", "scienceworld_text": "scienceworld_text_harness", "virtualhome_symbolic": "virtualhome_symbolic_harness"}[benchmark]
        module = importlib.import_module(package + ".manifest")
        manifest = module.build_manifest(SimpleNamespace())
    else:
        manifest = json.loads(pool.read_text())
    for task in tasks:
        args = task["runner_args"]
        index = int(release.option(args, "--task-index") or release.option(args, "--start-index"))
        suite = release.option(args, "--suite")
        expected = release.option(args, "--expected-task-id")
        assert release.option(args, "--pool-sha256") == release.file_sha256(pool)
        if benchmark in {"alfred_official_visual", "alfworld_visual"}:
            # A missing fallback proves the selected data bundle is actually used.
            row, count = module.select_task(Path("absent-fallback.json"), suite, index)
            assert row["task_id"] == expected
            assert Path(row["traj_path"]).is_file()
            assert count == len(tasks)
        elif benchmark == "humanclaw":
            assert manifest["tasks"][index]["task_id"] == expected
        else:
            selected = module.select_suite(manifest, suite=suite, start_index=index, num_tasks=1)
            assert len(selected) == 1 and selected[0].task_id == expected
