import copy
import json
import subprocess
import sys

import pytest

from scripts import run_ai_single_reference_batch_development as driver


@pytest.fixture
def cfg():
    return json.loads((driver.ROOT / driver.CONFIG).read_text(encoding="utf-8"))


def idle_snapshot():
    return {"available": True, "free_mib": 11000, "used_mib": 1000,
            "utilization_percent": 0, "known_compute_pids": []}


@pytest.mark.parametrize("field,value", [("free_mib", 9000), ("used_mib", 2600),
                                         ("utilization_percent", 98), ("known_compute_pids", [777])])
def test_busy_resource_each_independent_gate(cfg, field, value):
    snapshot = idle_snapshot()
    snapshot[field] = value
    assert driver.resource_admission(snapshot, cfg["budget"]) == "DEFERRED_GPU_BUSY"


def test_unknown_resources_and_own_pid(cfg):
    assert driver.resource_admission({"available": False}, cfg["budget"]) == "DEFERRED_RESOURCE_STATE_UNKNOWN"
    snapshot = idle_snapshot()
    snapshot["known_compute_pids"] = [777]
    assert driver.resource_admission(snapshot, cfg["budget"], own_pid=777) == "READY_FOR_EXPLICIT_LOCAL_INFERENCE"


def test_wddm_desktop_observations_do_not_permanently_defer(cfg, monkeypatch):
    outputs = iter(["0, GPU-test, Test GPU, 12288, 1000, 11288, 0\n",
                    "GPU-test, 111, C:\\Apps\\chrome.exe, [N/A]\n"])
    monkeypatch.setattr(driver.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, next(outputs), ""))
    snapshot = driver.gpu_snapshot()
    assert snapshot["process_observations"][0]["pid"] == 111
    assert snapshot["known_compute_pids"] == []
    assert driver.resource_admission(snapshot, cfg["budget"]) == "READY_FOR_EXPLICIT_LOCAL_INFERENCE"


@pytest.mark.parametrize("kind", ["seed", "sources", "training", "budget"])
def test_frozen_contract_drift_rejected(cfg, kind):
    parent = {key: cfg["inference"][key] for key in ("seed", "steps", "size")}
    if kind == "seed":
        cfg["inference"]["seed"] = 49
    elif kind == "sources":
        cfg["sources"].reverse()
    elif kind == "training":
        cfg["training"] = True
    else:
        cfg["budget"]["maximum_new_forwards"] = 9
    with pytest.raises(ValueError):
        driver.validate_contract(cfg, parent)


def test_zero_forward_resource_defer_retries_without_overwrite(tmp_path):
    base = tmp_path / "round"
    first = driver.create_attempt(base, "config-hash")
    report = {"status": "DEFERRED_GPU_BUSY", "new_forwards": 0}
    (first / "report.json").write_text(json.dumps(report))
    second = driver.create_attempt(base, "config-hash")
    assert first != second
    assert json.loads((first / "report.json").read_text()) == report


@pytest.mark.parametrize("reason", ["partial", "event", "unfinished", "owner"])
def test_previous_forward_or_uncertain_attempt_never_repeats(tmp_path, reason):
    base = tmp_path / "round"
    first = driver.create_attempt(base, "config-hash")
    if reason != "unfinished":
        report = {"status": "DEFERRED_GPU_BUSY", "new_forwards": 0}
        if reason == "partial":
            report.update(status="FAILED_PARTIAL_OUTPUTS_PRESERVED", new_forwards=1)
        (first / "report.json").write_text(json.dumps(report))
    if reason == "event":
        driver.record_event(first, {"stage": "FORWARD_START"})
    with pytest.raises((ValueError, FileNotFoundError)):
        driver.create_attempt(base, "other" if reason == "owner" else "config-hash")
    assert len(list(base.glob("attempt-*"))) == 1


@pytest.mark.parametrize("forwards,seconds,peak,before,error", [
    (8, 20, 0, True, RuntimeError), (9, 20, 0, False, RuntimeError),
    (0, 600, 0, False, TimeoutError), (1, 20, 9126805505, False, MemoryError),
])
def test_execution_budget_stops_at_boundaries(cfg, forwards, seconds, peak, before, error):
    with pytest.raises(error):
        driver.check_execution_budget(forwards, seconds, peak, cfg["budget"], before_new_forward=before)
    driver.check_execution_budget(8, 599, 9126805504, cfg["budget"])


def test_worker_resource_defer_never_initializes_models(tmp_path, monkeypatch, cfg):
    base = tmp_path / "round"
    out = base / "attempt-0001"
    out.mkdir(parents=True)
    cfg = copy.deepcopy(cfg)
    cfg["output"] = str(base)
    lock = {"config_sha256": "hash", "implementation_sha256": {}}
    lock_path = out / "execution_lock.json"
    lock_path.write_text(json.dumps(lock))
    bundle = {**lock, "config": cfg, "parent": {}, "sources": []}
    monkeypatch.setattr(driver, "verify_assets", lambda: bundle)
    monkeypatch.setattr(driver, "gpu_snapshot", lambda: {"available": False})
    from scripts import run_ai_vcg_reference as parent
    monkeypatch.setattr(parent, "initialize", lambda *a: pytest.fail("must not load models"))
    assert driver.run_worker(lock_path) == 0
    report = json.loads((out / "report.json").read_text())
    assert report["status"] == "DEFERRED_RESOURCE_STATE_UNKNOWN"
    assert report["new_forwards"] == 0
    assert not (out / "events.jsonl").exists()


def test_supervisor_terminates_only_its_owned_child(tmp_path):
    result = driver.supervise_worker([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, 0.1)
    assert result["status"] == "OWNED_WORKER_TERMINATED_AT_WALL_BUDGET"
    assert result["foreign_processes_modified"] is False
    assert result["partial_outputs_preserved"] is True
