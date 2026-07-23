from __future__ import annotations

import numpy as np

from src.eval.constrained_explicit_distillation import (
    BoundedCurveMatrixOperator,
    fit_operator,
    operator_diagnostics,
    synthetic_grid,
)


def _config() -> dict:
    return {
        "fit": {
            "synthetic_grid_size": 5,
            "curve_knot_count": 5,
            "minimum_curve_interval": 0.0001,
            "maximum_matrix_mix": 0.45,
            "seed": 20260723,
            "restarts": 1,
            "steps": 80,
            "learning_rate": 0.03,
            "identity_regularization": 0.0001,
            "dtype": "float64",
            "device": "cpu",
        },
        "structure_gates": {
            "matrix_minimum_determinant": 0.05,
            "matrix_minimum_entry": -1e-12,
            "matrix_maximum_row_sum_error": 1e-12,
            "minimum_curve_interval": 0.0001,
            "node_minimum": 0.0,
            "node_maximum": 1.0,
            "minimum_corresponding_channel_grid_step": 1e-7,
            "minimum_tetrahedron_jacobian_determinant": 1e-8,
            "maximum_replay_absolute_error": 1e-12,
        },
    }


def test_identity_operator_roundtrip() -> None:
    curves = np.tile(np.linspace(0.0, 1.0, 5), (3, 1))
    operator = BoundedCurveMatrixOperator(curves, np.eye(3))
    points = synthetic_grid(5)
    assert np.array_equal(operator.apply(points), points)
    replay = BoundedCurveMatrixOperator.from_dict(operator.to_dict())
    assert np.array_equal(replay.apply(points), points)


def test_fit_is_deterministic_and_bounded() -> None:
    config = _config()
    inputs = synthetic_grid(5)
    target = np.clip(inputs ** np.asarray([0.8, 1.1, 0.9]), 0.0, 1.0)
    first, first_audit = fit_operator(inputs, target, config["fit"])
    second, second_audit = fit_operator(inputs, target, config["fit"])
    assert first.to_dict() == second.to_dict()
    assert first_audit == second_audit
    output = first.apply(inputs)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert first_audit["target_rmse"] < 0.03


def test_fitted_operator_passes_structure_audit() -> None:
    config = _config()
    inputs = synthetic_grid(5)
    target = np.clip(inputs ** np.asarray([0.8, 1.1, 0.9]), 0.0, 1.0)
    operator, _ = fit_operator(inputs, target, config["fit"])
    report = operator_diagnostics(operator, config)
    assert report["structure_safe"]
    assert report["minimum_tetrahedron_jacobian_determinant"] > 0.0
