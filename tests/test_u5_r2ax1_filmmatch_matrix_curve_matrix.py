import json

import numpy as np
import pytest

from src.roll2film.generalized_monotone_curve_matrix import (
    PositiveMatrixCurveMatrixOperator,
    fit_positive_matrix_curve_matrix,
)


def test_matrix_curve_matrix_is_safe_and_replayable() -> None:
    source = np.random.default_rng(41).uniform(size=(256, 3))
    target = np.column_stack(
        (source[:, 0] ** 0.8, source[:, 1] ** 1.1, source[:, 2] ** 1.2)
    )
    fit = fit_positive_matrix_curve_matrix(
        source,
        target,
        segment_count=5,
        curve_learned_mixture=0.95,
        input_matrix_identity_mixture=0.25,
        output_matrix_identity_mixture=0.25,
        free_logit_bounds=(-8.0, 4.0),
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
    replay = PositiveMatrixCurveMatrixOperator.from_dict(
        json.loads(json.dumps(fit.operator.to_dict()))
    )
    assert np.array_equal(replay.apply(source), output)


def test_matrix_curve_matrix_rejects_invalid_fit_contract() -> None:
    source = np.full((16, 3), 0.5)
    with pytest.raises(ValueError, match="invalid matrix-curve-matrix fit data"):
        fit_positive_matrix_curve_matrix(
            source,
            source,
            segment_count=5,
            curve_learned_mixture=0.95,
            input_matrix_identity_mixture=1.1,
            output_matrix_identity_mixture=0.25,
            free_logit_bounds=(-8.0, 4.0),
            restart_count=1,
            maximum_function_evaluations=1,
            function_tolerance=1e-9,
            parameter_tolerance=1e-9,
            gradient_tolerance=1e-9,
            loss="soft_l1",
            loss_scale=0.02,
            seed=1,
        )
