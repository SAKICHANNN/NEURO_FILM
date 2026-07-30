import numpy as np

from src.eval.filmmatch_domain_balanced_capacity import _domain_weights
from src.roll2film.generalized_monotone_curve_matrix import (
    fit_generalized_monotone_curve_matrix,
)


def test_domain_weights_assign_exact_group_mass() -> None:
    weights = _domain_weights(100, 10, 0.75)
    assert np.isclose(np.mean(weights), 1.0)
    assert np.isclose(np.sum(weights[:100]) / np.sum(weights), 0.25)
    assert np.isclose(np.sum(weights[100:]) / np.sum(weights), 0.75)


def test_generalized_monotone_accepts_positive_sample_weights() -> None:
    source = np.random.default_rng(81).uniform(size=(128, 3))
    target = source**0.9
    fit = fit_generalized_monotone_curve_matrix(
        source,
        target,
        segment_count=5,
        curve_learned_mixture=0.75,
        matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
        restart_count=1,
        maximum_function_evaluations=100,
        function_tolerance=1e-9,
        parameter_tolerance=1e-9,
        gradient_tolerance=1e-9,
        loss="soft_l1",
        loss_scale=0.02,
        seed=1,
        sample_weights=np.linspace(0.5, 1.5, len(source)),
    )
    assert fit.converged
    assert fit.development_rgb_rmse < 0.03
