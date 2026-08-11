from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.luminance_eigen_affine_transport import (
    select_luminance_eigen_affine_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb48_origin_anchored_luminance_linear_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb48_origin_anchored_luminance_linear_development_decision_v1.json"


def test_cb48_contract_forbids_all_offsets() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB48"
    assert "origin-anchored" in contract["operator"]["luminance_fit"]
    assert "any coordinate intercept" in contract["operator"]["forbidden"]


def test_cb48_origin_anchored_fit_maps_black_to_black() -> None:
    y, x = np.mgrid[0:16, 0:32]
    source = np.stack(
        [0.02 * x, 0.015 * x + 0.002 * y, 0.01 * x], axis=-1
    ).astype(np.float32)
    target = np.asarray(source * [0.9, 0.8, 1.05], dtype=np.float32)
    candidate, scale, luma_error, facts = select_luminance_eigen_affine_candidate(
        source,
        target,
        curve=object(),
        strength=0.2,
        boundary_epsilon=1.0 / 65535.0,
        dose_grid=[1.0, 0.5, 0.0],
        maximum_gradient_ratio=2.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        minimum_luminance_slope=0.25,
        maximum_luminance_slope=2.0,
        origin_anchored=True,
    )
    assert facts["fitted_luminance_intercept"] == 0.0
    assert np.array_equal(candidate[0, 0], np.zeros(3, dtype=np.float32))
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert np.max(np.abs(luma_error)) <= 1e-6
    assert np.all(scale == facts["global_dose"])


def test_cb48_decision_closes_unconstrained_matrix_for_nonnegative_fit() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_sha256_exact"] is True
    assert decision["metrics"]["zero_dose_source_count"] == 7
    assert decision["metrics"]["maximum_lstar_inversion_fraction"] == 0.0
    assert decision["decision"] == (
        "close_origin_anchored_unconstrained_linear_open_nonnegative_y_eigen_matrix"
    )
