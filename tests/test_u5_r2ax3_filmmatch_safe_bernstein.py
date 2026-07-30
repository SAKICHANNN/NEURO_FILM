import json

import numpy as np

from src.roll2film.safe_bernstein_lut import (
    SafeBernsteinLUTOperator,
    fit_safe_bernstein_lut,
)


def test_safe_bernstein_lut_is_bounded_positive_and_replayable() -> None:
    source = np.random.default_rng(61).uniform(size=(512, 3))
    target = np.column_stack(
        (
            source[:, 0] ** 0.8,
            np.clip(source[:, 1] + 0.1 * source[:, 0] * source[:, 2], 0, 1),
            source[:, 2] ** 1.15,
        )
    )
    fit = fit_safe_bernstein_lut(
        source,
        target,
        degree=3,
        identity_ridge=0.01,
        jacobian_floor=1e-4,
        safety_grid_size=9,
        strength_steps=101,
        maximum_iterations=300,
    )
    axis = np.linspace(0.0, 1.0, 11)
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    output = fit.operator.apply(cube)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(fit.operator.jacobian_determinants(cube)) > 0.0
    assert fit.development_rgb_rmse < 0.04
    replay = SafeBernsteinLUTOperator.from_dict(
        json.loads(json.dumps(fit.operator.to_dict()))
    )
    assert np.array_equal(replay.apply(cube), output)


def test_safe_bernstein_identity_control_points_are_identity() -> None:
    axis = np.linspace(0.0, 1.0, 4)
    controls = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    operator = SafeBernsteinLUTOperator(
        degree=3,
        fitted_control_points=controls,
        strength=1.0,
        jacobian_floor=1e-4,
    )
    source = np.random.default_rng(62).uniform(size=(128, 3))
    assert np.allclose(operator.apply(source), source, atol=5e-16)
    assert np.allclose(operator.jacobian_determinants(source), 1.0, atol=2e-15)
