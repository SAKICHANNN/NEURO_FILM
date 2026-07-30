import json

import numpy as np

from src.roll2film.triangular_conditional_monotone import (
    TriangularConditionalMonotoneOperator,
    fit_triangular_conditional_monotone,
)


def test_triangular_conditional_operator_is_safe_positive_and_replayable() -> None:
    source = np.random.default_rng(51).uniform(size=(256, 3))
    target = source.copy()
    target[:, 1] = np.clip(
        source[:, 1] ** (0.75 + 0.35 * source[:, 0]), 0.0, 1.0
    )
    target[:, 2] = np.clip(
        source[:, 2] ** (0.7 + 0.2 * source[:, 0] + 0.2 * source[:, 1]),
        0.0,
        1.0,
    )
    fit = fit_triangular_conditional_monotone(
        source,
        target,
        channel_order=(0, 1, 2),
        segment_count=5,
        learned_mixture=0.95,
        free_logit_bounds=(-4.0, 4.0),
        conditioner_l2=0.0,
        restart_count=1,
        maximum_function_evaluations=500,
        function_tolerance=1e-9,
        parameter_tolerance=1e-9,
        gradient_tolerance=1e-9,
        loss="soft_l1",
        loss_scale=0.02,
        seed=1,
    )
    output = fit.operator.apply(source)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(fit.operator.jacobian_determinants(source)) > 0.0
    assert fit.development_rgb_rmse < 0.025
    replay = TriangularConditionalMonotoneOperator.from_dict(
        json.loads(json.dumps(fit.operator.to_dict()))
    )
    assert np.array_equal(replay.apply(source), output)


def test_triangular_conditional_identity_is_exact() -> None:
    operator = TriangularConditionalMonotoneOperator(
        channel_order=(2, 0, 1),
        segment_count=5,
        learned_mixture=0.75,
        stage_parameters=(
            np.zeros((1, 4)),
            np.zeros((2, 4)),
            np.zeros((3, 4)),
        ),
    )
    source = np.random.default_rng(52).uniform(size=(128, 3))
    assert np.allclose(operator.apply(source), source, atol=2e-16)
    assert np.allclose(operator.jacobian_determinants(source), 1.0, atol=2e-16)
