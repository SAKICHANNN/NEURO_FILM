from __future__ import annotations

from scripts.run_u6_p4l_exact_area_live_performance import (
    CONFIG_SHA256,
    ROOT,
)
from src.eval.physical_exact_area_live_performance import (
    evaluate_live_performance_records,
    load_contract,
)


CONFIG = ROOT / "configs/u6_p4l_exact_area_live_performance_v1.json"


def test_contract_is_exact_and_local_reference_only() -> None:
    contract, p4k = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    assert contract["worker"]["network_allowed"] is False
    assert contract["worker"]["gpu_allowed"] is False
    assert contract["photograph_access_allowed"] is False
    assert p4k["node"] == "U6.P4K"


def test_record_gate_accepts_exact_bounded_fresh_runs() -> None:
    contract, _ = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    expected = contract["worker"]["expected_stream_sha256"]
    runs = [
        {
            "stream_sha256": expected,
            "process_tree_peak_rss_bytes": 600_000_000,
            "wall_seconds": 60.0,
            "liveness_samples": 3000,
            "exit_code": 0,
            "stderr_bytes": 0,
            "owned_temp_residue_count": 0,
        },
        {
            "stream_sha256": expected,
            "process_tree_peak_rss_bytes": 610_000_000,
            "wall_seconds": 61.0,
            "liveness_samples": 3050,
            "exit_code": 0,
            "stderr_bytes": 0,
            "owned_temp_residue_count": 0,
        },
    ]
    report = evaluate_live_performance_records(contract, runs)
    assert report["automatic_pass"] is True
    assert all(report["checks"].values())


def test_record_gate_rejects_nonempty_stderr_and_wrong_hash() -> None:
    contract, _ = load_contract(ROOT, CONFIG, CONFIG_SHA256)
    expected = contract["worker"]["expected_stream_sha256"]
    runs = [
        {
            "stream_sha256": expected,
            "process_tree_peak_rss_bytes": 600_000_000,
            "wall_seconds": 60.0,
            "liveness_samples": 3000,
            "exit_code": 0,
            "stderr_bytes": 0,
            "owned_temp_residue_count": 0,
        },
        {
            "stream_sha256": "0" * 64,
            "process_tree_peak_rss_bytes": 610_000_000,
            "wall_seconds": 61.0,
            "liveness_samples": 3050,
            "exit_code": 0,
            "stderr_bytes": 1,
            "owned_temp_residue_count": 0,
        },
    ]
    report = evaluate_live_performance_records(contract, runs)
    assert report["automatic_pass"] is False
    assert report["checks"]["stream_hash_exact"] is False
    assert report["checks"]["repeat_hash_exact"] is False
    assert report["checks"]["stderr_empty"] is False
