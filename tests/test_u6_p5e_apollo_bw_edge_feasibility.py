from __future__ import annotations

import numpy as np

from src.eval.apollo_bw_edge_feasibility import analyze_edge_roi


ANALYSIS = {
    "plateau_rows": 40,
    "minimum_column_span_code": 30000,
    "robust_fit_iterations": 5,
    "minimum_residual_pixels": 3.0,
    "mad_multiplier": 5.0,
    "profile_column_stride": 2,
    "profile_radius_pixels": 60,
    "oversampling": 4,
    "phase_bin_count": 8,
    "negative_step_tolerance": 0.002,
}


def _edge(width_pixels: float) -> np.ndarray:
    height, width = 240, 256
    y, x = np.indices((height, width), dtype=np.float64)
    center = 105.0 + 0.08 * x
    normalized = 1.0 / (1.0 + np.exp(-(y - center) / width_pixels))
    return np.round(normalized * 62000.0).astype(np.uint16)[..., None]


def test_sharp_slanted_edge_has_phase_coverage_and_small_width() -> None:
    metrics = analyze_edge_roi(_edge(1.5), ANALYSIS)
    assert metrics["valid_column_fraction"] == 1.0
    assert metrics["line_inlier_fraction"] == 1.0
    assert metrics["edge_shift_pixels"] > 16.0
    assert min(metrics["phase_bin_counts"]) > 10
    assert metrics["width_10_90_pixels"] < 10.0


def test_broad_transition_remains_measurably_broad() -> None:
    metrics = analyze_edge_roi(_edge(24.0), ANALYSIS)
    assert metrics["width_10_90_pixels"] > 80.0
    assert metrics["negative_step_fraction"] == 0.0
