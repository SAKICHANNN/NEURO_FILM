import json

import numpy as np

from src.roll2film.factorized_monotone_bernstein import (
    FactorizedMonotoneBernsteinOperator,
    fit_factorized_monotone_bernstein,
)


def test_factorized_monotone_bernstein_is_safe_and_replayable() -> None:
    source = np.random.default_rng(71).uniform(size=(512, 3))
    target = np.column_stack(
        (
            source[:, 0] ** 0.8,
            np.clip(source[:, 1] ** 1.1 + 0.04 * source[:, 0], 0, 1),
            source[:, 2] ** 1.2,
        )
    )
    fit = fit_factorized_monotone_bernstein(
        source,
        target,
        segment_count=5,
        curve_learned_mixture=0.75,
        matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
        restart_count=1,
        maximum_function_evaluations=500,
        function_tolerance=1e-9,
        parameter_tolerance=1e-9,
        gradient_tolerance=1e-9,
        loss="soft_l1",
        loss_scale=0.02,
        seed=1,
        residual_degree=3,
        residual_identity_ridge=0.01,
        jacobian_floor=1e-4,
        safety_grid_size=9,
        strength_steps=101,
        maximum_residual_iterations=300,
    )
    axis = np.linspace(0.01, 0.99, 9)
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    output = fit.operator.apply(cube)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(fit.operator.jacobian_determinants(cube)) > 0.0
    replay = FactorizedMonotoneBernsteinOperator.from_dict(
        json.loads(json.dumps(fit.operator.to_dict()))
    )
    assert np.array_equal(replay.apply(cube), output)
