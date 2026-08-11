from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.nonnegative_y_eigen_matrix_transport import (
    _fit_matrix,
    select_nonnegative_y_eigen_candidate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2cb49_nonnegative_y_eigen_matrix_development_v1.json"
DECISION = (
    ROOT / "configs/u5_r2cb49_nonnegative_y_eigen_matrix_development_decision_v1.json"
)


def test_cb49_contract_freezes_nonnegative_cube_safe_matrix() -> None:
    c = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert c["experiment_id"] == "U5.R2CB49"
    assert "M>=0" in c["operator"]["constraints"]


def test_cb49_fit_satisfies_all_matrix_constraints() -> None:
    rng = np.random.default_rng(49)
    source = rng.uniform(0, 0.8, (8, 16, 3)).astype(np.float32)
    target = np.asarray(
        source @ [[0.8, 0.1, 0.0], [0.05, 0.8, 0.05], [0.0, 0.1, 0.75]],
        dtype=np.float32,
    )
    matrix, slope, _ = _fit_matrix(
        source, target, minimum_luminance_slope=0.25, maximum_luminance_slope=1.0
    )
    assert np.min(matrix) >= -1e-10
    assert np.max(matrix.sum(axis=1)) <= 1 + 1e-10
    assert (
        np.max(np.abs(LEGACY_LAB_Y_WEIGHTS @ matrix - slope * LEGACY_LAB_Y_WEIGHTS))
        <= 1e-9
    )


def test_cb49_candidate_is_safe_and_order_preserving() -> None:
    rng = np.random.default_rng(50)
    source = rng.uniform(0.01, 0.8, (8, 16, 3)).astype(np.float32)
    target = np.asarray(source * 0.8, dtype=np.float32)
    candidate, scale, error, facts = select_nonnegative_y_eigen_candidate(
        source,
        target,
        curve=object(),
        strength=0.2,
        boundary_epsilon=1 / 65535,
        dose_grid=[1.0, 0.0],
        maximum_gradient_ratio=2.0,
        maximum_lstar_inversion_fraction=0.0,
        lstar_order_epsilon=0.0001,
        minimum_luminance_slope=0.25,
        maximum_luminance_slope=1.0,
    )
    assert np.min(candidate) >= 0 and np.max(candidate) <= 1
    assert facts["selected_lstar_inversion_fraction"] == 0
    assert np.max(np.abs(error)) <= 1e-6
    assert np.all(scale == facts["global_dose"])


def test_cb49_decision_closes_global_linear_family() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["repeat_report_sha256_exact"] is True
    assert decision["metrics"]["maximum_lstar_inversion_fraction"] == 0.0
    assert decision["metrics"]["population_median_style_delta_e76"] < 5.0
    assert decision["decision"] == (
        "close_global_linear_safety_family_open_analytic_y_chromaticity_factorization"
    )
