from __future__ import annotations

import numpy as np

from src.eval.canoncgt_full_chain_atlas_response import _stats, summarize_rows


def _config() -> dict:
    return {
        "prescore_gates": {
            "required_reference_count": 2,
            "required_atlas_rows_per_reference": 1,
            "all_values_finite": True,
            "minimum_response_value": 0.0,
            "maximum_response_value": 1.0,
            "maximum_response_out_of_range_fraction": 0.0,
            "maximum_canonical_out_of_range_fraction": 0.1,
            "minimum_reference_response_pairwise_rmse_median": 0.001,
        }
    }


def _row(reference_id: str, response: np.ndarray, distance: float) -> dict:
    return {
        "reference_id": reference_id,
        "atlas_id": "rgb",
        "canonical": _stats(np.full((2, 2, 3), 0.5, dtype=np.float32)),
        "response": _stats(response),
        "reference_pairwise_response_rmse_median": distance,
    }


def test_stats_preserves_raw_out_of_range_fact() -> None:
    values = np.array([[[-0.1, 0.5, 1.1]]], dtype=np.float32)
    result = _stats(values)
    assert result["finite"] is True
    assert result["out_of_range_fraction"] == 2.0 / 3.0
    assert result["minimum"] < 0.0
    assert result["maximum"] > 1.0


def test_summary_passes_complete_bounded_sensitive_rows() -> None:
    rows = [
        _row("a", np.full((2, 2, 3), 0.4, dtype=np.float32), 0.02),
        _row("b", np.full((2, 2, 3), 0.6, dtype=np.float32), 0.02),
    ]
    summary = summarize_rows(rows, _config())
    assert summary["decision"] == "PASS_FULL_CHAIN_ATLAS_OBSERVATION_PRESCORE"
    assert all(summary["gates"].values())


def test_summary_fails_raw_response_escape_without_clipping() -> None:
    rows = [
        _row("a", np.full((2, 2, 3), -0.01, dtype=np.float32), 0.02),
        _row("b", np.full((2, 2, 3), 0.6, dtype=np.float32), 0.02),
    ]
    summary = summarize_rows(rows, _config())
    assert summary["decision"] == "FAIL_CLOSED_FULL_CHAIN_ATLAS_OBSERVATION_PRESCORE"
    assert summary["gates"]["response_intrinsically_cube_bounded"] is False
