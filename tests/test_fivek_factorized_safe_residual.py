from __future__ import annotations

import numpy as np

from src.eval.fivek_factorized_safe_residual import (
    analytical_safe_residual,
    fit_factorized_parameter_model,
    predict_factorized_parameters,
)


def test_factorized_model_recovers_direction_and_magnitude() -> None:
    rng = np.random.default_rng(19)
    x = rng.normal(size=(160, 6))
    raw = x @ rng.normal(scale=0.1, size=(6, 14))
    raw += np.linspace(0.05, 0.2, 14)
    model = fit_factorized_parameter_model(x, raw, 1.0e-8)
    predicted, magnitude, clipped = predict_factorized_parameters(model, x)
    assert predicted.shape == raw.shape
    assert magnitude.shape == (len(x),)
    assert clipped.shape == (len(x),)
    assert np.all(np.isfinite(predicted))
    error = float(np.mean(np.linalg.norm(predicted - raw, axis=1)))
    constant_error = float(
        np.mean(np.linalg.norm(raw - np.mean(raw, axis=0), axis=1))
    )
    assert error < 0.25
    assert error < 0.3 * constant_error


def test_analytical_safe_residual_stays_strictly_interior_without_clipping() -> None:
    epsilon = 1.0 / 65535.0
    source = np.asarray(
        [[0.25, 0.5, 0.75], [0.01, 0.99, 0.5], [0.5, 0.5, 0.5]]
    )
    candidate = np.asarray(
        [[0.0, 1.0, 0.8], [0.0, 1.0, 0.5], [0.6, 0.4, 0.5]]
    )
    output, scale = analytical_safe_residual(
        source, candidate, epsilon=epsilon, margin_multiplier=2.0
    )
    assert np.all(output > epsilon)
    assert np.all(output < 1.0 - epsilon)
    assert np.all((scale >= 0.0) & (scale <= 1.0))
    assert np.any(scale < 1.0)
    assert np.allclose(output[2], candidate[2])


def test_source_boundary_pixels_are_not_reclassified_or_hard_clipped() -> None:
    epsilon = 1.0 / 65535.0
    source = np.asarray([[0.0, 0.5, 1.0]])
    candidate = np.asarray([[0.0, 0.75, 1.0]])
    output, scale = analytical_safe_residual(
        source, candidate, epsilon=epsilon, margin_multiplier=2.0
    )
    assert np.array_equal(output, candidate)
    assert np.array_equal(scale, np.ones(1))
