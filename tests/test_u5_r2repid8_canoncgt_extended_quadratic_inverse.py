from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.eval.canoncgt_extended_quadratic_inverse import (
    ExtendedQuadraticOperator,
    fit_extended_quadratic,
    jacobian_diagnostics,
    quadratic_features,
    range_escape_magnitude,
)


def test_quadratic_features_accept_extended_finite_rgb() -> None:
    values = np.asarray(((-0.03, 0.5, 1.04), (0.2, 0.4, 0.8)))
    features = quadratic_features(values, 2.0)
    assert features.shape == (2, 10)
    assert np.all(np.isfinite(features))
    assert np.all(features[:, 0] == 1.0)


def test_extended_quadratic_fit_recovers_synthetic_operator() -> None:
    rng = np.random.default_rng(81)
    source = rng.uniform(-0.02, 1.02, size=(96, 96, 3)).astype(np.float32)
    coefficients = np.zeros((10, 3), dtype=np.float64)
    coefficients[0] = (0.1, -0.1, 0.05)
    coefficients[1] = (2.4, 0.1, 0.0)
    coefficients[2] = (0.0, 2.2, 0.1)
    coefficients[3] = (0.1, 0.0, 2.3)
    coefficients[4:7] = 0.03
    truth = ExtendedQuadraticOperator(coefficients=coefficients, asinh_scale=2.0)
    target = truth.apply(source).reshape(source.shape)
    fitted, metrics = fit_extended_quadratic(
        source,
        target,
        maximum_rows=8000,
        ridge_alpha=1.0e-4,
        asinh_scale=2.0,
        target_logit_epsilon=1.0e-5,
    )
    assert metrics["holdout_error_p95"] < 1.0e-5
    assert np.max(np.abs(fitted.coefficients - coefficients)) < 1.0e-4


def test_identity_like_operator_has_positive_jacobian() -> None:
    coefficients = np.zeros((10, 3), dtype=np.float64)
    coefficients[1] = (2.5, 0.0, 0.0)
    coefficients[2] = (0.0, 2.5, 0.0)
    coefficients[3] = (0.0, 0.0, 2.5)
    operator = ExtendedQuadraticOperator(coefficients=coefficients, asinh_scale=2.0)
    diagnostics = jacobian_diagnostics(operator, grid_size=5, input_minimum=-0.05, input_maximum=1.05)
    assert diagnostics["minimum_determinant"] > 0.0
    assert diagnostics["minimum_singular_value"] > 0.0


def test_range_escape_magnitude_is_not_fraction_or_projection() -> None:
    values = np.asarray(((-0.02, 0.5, 1.07), (0.0, 1.0, 0.5)))
    assert range_escape_magnitude(values) == pytest.approx(0.07)


def test_runner_has_no_drive_literal_and_requires_output() -> None:
    source = (Path(__file__).resolve().parents[1] / "scripts/run_u5_r2repid8_canoncgt_extended_quadratic_inverse.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--output", type=Path, required=True)' in source
    assert "D:\\" not in source
    assert "P:\\" not in source
