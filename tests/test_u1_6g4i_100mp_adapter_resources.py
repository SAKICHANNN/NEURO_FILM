from __future__ import annotations

import json
from pathlib import Path

import psutil

import scripts.audit_u1_6g4i_100mp_adapter_resources as audit


def _small_config(tmp_path: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/u1_6g4i_100mp_adapter_resource_v1.json").read_text(
            encoding="utf-8"
        )
    )
    height, width = 73, 101
    pixel_count = height * width
    config.update(
        {
            "shape": [height, width, 3],
            "pixel_count": pixel_count,
            "input_generation_row_chunk": 7,
            "source_row_chunk": 11,
            "coarse_row_chunk": 3,
            "tile_size": 7,
            "composite_row_chunk": 9,
            "rss_sample_interval_seconds": 0.005,
            "worker_timeout_seconds": 30,
        }
    )
    config["preflight"] = {
        "available_physical_memory_bytes_min": 1,
        "competing_process_rss_bytes_max": 2**63 - 1,
        "minimum_free_disk_bytes": 1,
    }
    floor = pixel_count * 4 * (3 + 4 + 3)
    config["gates"].update(
        {
            "known_live_array_floor_bytes": floor,
            "peak_process_tree_rss_bytes_max": 1024**3,
            "worker_total_seconds_max": 30,
            "adapter_seconds_max": 25,
            "source_generation_seconds_max": 5,
        }
    )
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_small_parent_repeats_hashes_and_cleans_failure(tmp_path: Path) -> None:
    report = audit.run_parent(_small_config(tmp_path), tmp_path / "workers")
    assert report["preflight"]["passed"]
    assert report["failure_probe"]["passed"]
    assert report["repeat_hashes_pass"]
    assert report["gate_result"]["automatic_pass"]
    assert not report["gate_result"]["renderer_integration_allowed"]
    for run in report["runs"]:
        assert run["passed"]
        assert len(run["monitor"]["observed_process_ids"]) >= 1
        assert run["worker"]["metadata"]["public_ndarray_count"] == 1
    assert not list(tmp_path.rglob("*.tmp"))
    assert not (tmp_path / "workers/failure_probe.json").exists()


def test_preflight_failure_launches_no_worker(tmp_path: Path, monkeypatch) -> None:
    config_path = _small_config(tmp_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["preflight"]["available_physical_memory_bytes_min"] = 2**63 - 1
    config_path.write_text(json.dumps(config), encoding="utf-8")

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight failure must not launch a worker")

    monkeypatch.setattr(audit, "_launch_worker", forbidden)
    report = audit.run_parent(config_path, tmp_path / "workers")
    assert not report["preflight"]["passed"]
    assert report["runs"] == []
    assert report["gate_result"]["run_deferred_by_preflight"]
    assert not report["gate_result"]["automatic_pass"]


def test_worker_rejects_shape_pixel_count_mismatch(tmp_path: Path) -> None:
    config = json.loads(_small_config(tmp_path).read_text(encoding="utf-8"))
    config["pixel_count"] += 1
    try:
        audit.worker(config)
    except ValueError as error:
        assert "pixel_count" in str(error)
    else:
        raise AssertionError("mismatched frozen pixel count must fail")


def test_failure_injection_precedes_input_construction(tmp_path: Path, monkeypatch) -> None:
    config = json.loads(_small_config(tmp_path).read_text(encoding="utf-8"))

    def forbidden(*args, **kwargs):
        raise AssertionError("input must not be allocated")

    monkeypatch.setattr(audit, "analytic_highlight_field", forbidden)
    try:
        audit.worker(config, inject_failure=True)
    except RuntimeError as error:
        assert "before_input_allocation" in str(error)
    else:
        raise AssertionError("injected failure must propagate")


def test_live_frozen_preflight_is_observable(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs/u1_6g4i_100mp_adapter_resource_v1.json").read_text(
            encoding="utf-8"
        )
    )
    result = audit.preflight(config, tmp_path)
    assert isinstance(result["passed"], bool)
    assert result["available_physical_memory_bytes"] <= psutil.virtual_memory().total
    assert result["free_disk_bytes"] > 0
