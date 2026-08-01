from __future__ import annotations

import numpy as np

from src.roll2film.luma_conditioned_positive_matrix import (
    LumaConditionedPositiveMatrix,
    fit_luma_conditioned_positive_matrix,
)


def test_luma_conditioned_matrix_is_cube_safe_and_deterministic() -> None:
    rng = np.random.default_rng(3)
    rgb = rng.random((1024, 3))
    parameters = np.linspace(-6.0, 2.0, 18)
    operator = LumaConditionedPositiveMatrix(parameters, 0.35)
    first = operator.apply(rgb)
    second = operator.apply(rgb)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0.0)
    assert np.all(first <= 1.0)
    np.testing.assert_allclose(np.sum(operator.matrices, axis=2), 1.0, atol=1e-15)
    assert np.all(operator.matrices >= 0.0)


def test_fit_recovers_known_luma_conditioned_mixing() -> None:
    rng = np.random.default_rng(5)
    source = rng.uniform(0.02, 0.98, (3000, 3))
    truth_parameters = np.asarray([-2.0, -5.0, -4.0, -2.5, -3.0, -5.0] * 3)
    truth = LumaConditionedPositiveMatrix(truth_parameters, 0.35)
    target = truth.apply(source)
    fitted, converged = fit_luma_conditioned_positive_matrix(
        source,
        target,
        maximum_off_diagonal=0.35,
        logit_bounds=(-8.0, 4.0),
        initial_logit=-4.0,
        identity_shrinkage=1e-6,
        maximum_fit_samples=3000,
        maximum_evaluations=160,
        loss="linear",
        loss_scale=0.02,
    )
    assert converged
    identity_rmse = float(np.sqrt(np.mean(np.square(source - target))))
    fitted_rmse = float(np.sqrt(np.mean(np.square(fitted.apply(source) - target))))
    assert fitted_rmse < identity_rmse * 0.1
