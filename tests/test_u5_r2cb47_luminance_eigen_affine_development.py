from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.luminance_eigen_affine_transport import (
    select_luminance_eigen_affine_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb47_luminance_eigen_affine_development_v1.json"
DECISION = ROOT / "configs/u5_r2cb47_luminance_eigen_affine_development_decision_v1.json"


def test_cb47_contract_freezes_positive_luminance_eigen_affine() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["experiment_id"] == "U5.R2CB47"
    assert contract["operator"]["minimum_luminance_slope"] == 0.25
    assert "positive affine map" in contract["operator"]["execution"]


def test_cb47_global_affine_preserves_order_and_fits_chroma() -> None:
    y, x = np.mgrid[0:16, 0:32]
    source = np.stack(
        [0.05 + 0.02 * x, 0.04 + 0.015 * x + 0.002 * y, 0.03 + 0.01 * x],
        axis=-1,
    ).astype(np.float32)
    target = source.copy()
    target[..., 0] = 0.9 * source[..., 0] + 0.03 * source[..., 1]
    target[..., 1] = 0.85 * source[..., 1]
    target[..., 2] = 1.05 * source[..., 2]
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
    )
    assert candidate.dtype == np.float32
    assert np.min(candidate) >= 0.0 and np.max(candidate) <= 1.0
    assert facts["fitted_luminance_slope"] > 0.0
    assert facts["selected_lstar_inversion_fraction"] == 0.0
    assert np.max(np.abs(luma_error)) <= 1e-6
    assert np.all(scale == facts["global_dose"])


def test_cb47_decision_closes_offsets_for_origin_anchored_linear() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_sha256_exact"] is True
    assert decision["metrics"]["zero_dose_source_count"] == 9
    assert decision["metrics"]["negative_fitted_luminance_intercept_count"] == 9
    assert decision["metrics"]["maximum_lstar_inversion_fraction"] == 0.0
    assert decision["decision"] == (
        "close_luminance_eigen_affine_open_origin_anchored_linear"
    )
