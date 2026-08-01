from __future__ import annotations

import numpy as np

from src.eval.fivek_conditional_explicit_operator import (
    _bounded_transports,
    effective_case_parameters,
    fit_parameter_ridge,
    predict_parameters,
)


OPERATOR = {
    "lower_bounds": [-0.7, -1.0, -0.7, -0.7, -1.0, -1.0, -0.7, -0.7, -0.7, -0.7, -1.0, -1.0, -1.0, -1.0],
    "upper_bounds": [0.7, 1.0, 0.7, 0.7, 1.0, 1.0, 0.7, 0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0],
    "safe_dose_grid_size": 5,
    "safe_dose_finite_difference": 1.0e-5,
    "minimum_jacobian_determinant": 0.02,
    "maximum_jacobian_condition": 30.0,
    "safe_dose_bisection_iterations": 16,
}


def test_effective_case_parameters_bind_dose() -> None:
    parameters = np.linspace(-0.2, 0.2, 14)
    report = {
        "case_bank": [
            {"parameters": parameters.tolist(), "dose": 0.5},
            {"parameters": (-parameters).tolist(), "dose": 0.25},
        ]
    }
    result = effective_case_parameters(report)
    assert np.allclose(result[0], parameters * 0.5)
    assert np.allclose(result[1], -parameters * 0.25)


def test_multioutput_ridge_recovers_linear_parameter_map() -> None:
    rng = np.random.default_rng(13)
    x = rng.normal(size=(100, 8))
    coefficients = rng.normal(scale=0.05, size=(8, 14))
    y = x @ coefficients + rng.normal(scale=0.01, size=(1, 14))
    model = fit_parameter_ridge(x, y, 1.0e-8)
    predicted = predict_parameters(model, x)
    assert predicted.shape == (100, 14)
    assert float(np.max(np.abs(predicted - y))) < 1.0e-7


def test_predicted_parameters_are_training_bounded_and_cube_safe() -> None:
    training = np.stack(
        (np.full(14, -0.1), np.zeros(14), np.full(14, 0.1))
    )
    raw = np.stack((np.full(14, -1.0), np.full(14, 1.0)))
    transports, rows, clipped = _bounded_transports(raw, training, OPERATOR)
    assert float(np.mean(clipped)) == 1.0
    grid = np.linspace(0.0, 1.0, 27).reshape(3, 3, 3)
    rgb = np.stack((grid, grid, grid), axis=-1).reshape(-1, 3)
    for transport, row in zip(transports, rows, strict=True):
        output = transport.apply(rgb)
        assert np.all(np.isfinite(output))
        assert np.all((output >= 0.0) & (output <= 1.0))
        assert row["dose"] >= 0.0
