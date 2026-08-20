from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.canoncgt_bounded_basis_selector import (
    BoundedBasisOperator,
    basis_features,
    fit_candidates,
    jacobian_diagnostics,
)


def test_extended_basis_features_are_finite_without_clipping() -> None:
    values = np.asarray(((-0.04, 0.5, 1.04), (0.2, 0.4, 0.8)))
    for representation, width in (("affine", 4), ("gaussian", 68), ("lattice", 68)):
        features = basis_features(
            values,
            representation=representation,
            asinh_scale=2.0,
            coordinate_input_minimum=-0.05,
            coordinate_input_maximum=1.05,
            basis_side=4,
            gaussian_sigma=0.23,
        )
        assert features.shape == (2, width)
        assert np.all(np.isfinite(features))


def test_three_way_fit_recovers_smooth_bounded_operator() -> None:
    rng = np.random.default_rng(101)
    source = rng.uniform(-0.02, 1.02, size=(96, 96, 3)).astype(np.float32)
    coefficients = np.zeros((68, 3), dtype=np.float64)
    coefficients[1] = (2.3, 0.1, 0.0)
    coefficients[2] = (0.0, 2.2, 0.1)
    coefficients[3] = (0.1, 0.0, 2.4)
    truth = BoundedBasisOperator(coefficients, "gaussian", 2.0, -0.05, 1.05, 4, 0.23)
    target = truth.apply(source).reshape(source.shape)
    operators, metrics = fit_candidates(
        source,
        target,
        maximum_rows=9000,
        ridge_alpha=1.0e-4,
        asinh_scale=2.0,
        coordinate_input_minimum=-0.05,
        coordinate_input_maximum=1.05,
        basis_side=4,
        gaussian_sigma=0.23,
        target_logit_epsilon=1.0e-5,
    )
    assert metrics["partition_counts"] == {
        "fit": 3000,
        "selection": 3000,
        "audit": 3000,
    }
    assert metrics["gaussian"]["audit"]["p95"] < 1.0e-4
    assert np.all(np.isfinite(operators["gaussian"].coefficients))


def test_identity_like_affine_operator_has_positive_jacobian() -> None:
    coefficients = np.zeros((4, 3), dtype=np.float64)
    coefficients[1] = (2.5, 0.0, 0.0)
    coefficients[2] = (0.0, 2.5, 0.0)
    coefficients[3] = (0.0, 0.0, 2.5)
    operator = BoundedBasisOperator(coefficients, "affine", 2.0, -0.05, 1.05, 4, 0.23)
    diagnostics = jacobian_diagnostics(
        operator, grid_size=5, input_minimum=-0.05, input_maximum=1.05
    )
    assert diagnostics["minimum_determinant"] > 0.0
    assert diagnostics["minimum_singular_value"] > 0.0


def test_runner_has_no_drive_literal_and_requires_output() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "scripts/run_u5_r2repid10_canoncgt_bounded_basis_selector.py"
    ).read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source
