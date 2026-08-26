from __future__ import annotations

from scripts.audit_u7_6a_three_stock_shared_context import evaluate_rows


def _row(mode: str, wall: float, peak: int) -> dict:
    return {
        "mode": mode,
        "wall_seconds": wall,
        "peak_process_tree_rss_bytes": peak,
        "output_sha256": ["a", "b", "c"],
        "finite_bounded": [True, True, True],
        "stderr": "",
    }


def test_frozen_gate_evaluation_passes_exact_faster_candidate() -> None:
    rows = [
        _row("baseline", 10.0, 100),
        _row("candidate", 9.0, 102),
        _row("candidate", 9.1, 101),
        _row("baseline", 10.1, 100),
    ]
    result = evaluate_rows(
        rows,
        {
            "median_wall_ratio_max": 0.97,
            "candidate_peak_process_tree_rss_ratio_max": 1.05,
            "candidate_repeat_wall_ratio_max": 1.15,
            "candidate_repeat_peak_rss_ratio_max": 1.15,
        },
    )
    assert result["decision"] == "PASS"
    assert all(result["gate_results"].values())


def test_pixel_drift_fails_closed() -> None:
    rows = [
        _row("baseline", 10.0, 100),
        _row("candidate", 9.0, 100),
        _row("candidate", 9.0, 100),
        _row("baseline", 10.0, 100),
    ]
    rows[1]["output_sha256"][1] = "drift"
    result = evaluate_rows(
        rows,
        {
            "median_wall_ratio_max": 0.97,
            "candidate_peak_process_tree_rss_ratio_max": 1.05,
            "candidate_repeat_wall_ratio_max": 1.15,
            "candidate_repeat_peak_rss_ratio_max": 1.15,
        },
    )
    assert result["decision"] == "FAIL_CLOSED"
    assert result["gate_results"]["all_three_output_arrays_byte_exact"] is False
