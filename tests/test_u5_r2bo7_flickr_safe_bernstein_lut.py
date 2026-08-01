from __future__ import annotations

import numpy as np

from src.roll2film.factorized_monotone_bernstein import fit_factorized_monotone_bernstein


def test_factorized_safe_lut_recovers_smooth_bounded_witness() -> None:
    rng = np.random.default_rng(11)
    source = rng.uniform(0.02, 0.98, (1800, 3))
    target = source.copy()
    target[:, 0] += 0.05 * source[:, 1] * (1.0 - source[:, 0])
    target[:, 2] -= 0.04 * source[:, 0] * source[:, 2]
    target = np.clip(target, 0.0, 1.0)
    fit = fit_factorized_monotone_bernstein(
        source,
        target,
        segment_count=3,
        curve_learned_mixture=0.75,
        matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
        restart_count=1,
        maximum_function_evaluations=300,
        function_tolerance=1e-9,
        parameter_tolerance=1e-9,
        gradient_tolerance=1e-9,
        loss="soft_l1",
        loss_scale=0.02,
        seed=11,
        residual_degree=2,
        residual_identity_ridge=0.01,
        jacobian_floor=0.001,
        safety_grid_size=9,
        strength_steps=51,
        maximum_residual_iterations=300,
    )
    output = fit.operator.apply(source)
    assert fit.converged
    assert fit.operator.residual.strength > 0.0
    assert np.all(output >= 0.0)
    assert np.all(output <= 1.0)
    assert np.sqrt(np.mean(np.square(output - target))) < np.sqrt(np.mean(np.square(source - target)))
