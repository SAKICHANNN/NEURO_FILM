from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.interval_mobius_coupling import (
    IntervalMobiusCouplingOperator,
    finite_difference_jacobians,
    fit_interval_mobius_coupling,
)


STAGES = (0, 1, 2, 1, 2, 0)
LIMITS = {
    "maximum_lower_lift": 0.18,
    "maximum_upper_compression": 0.18,
    "maximum_log_odds_shift": 0.8,
}


def test_identity_replay_inverse_and_partition_parity() -> None:
    operator = IntervalMobiusCouplingOperator.identity(
        stage_channels=STAGES,
        **LIMITS,
    )
    points = np.random.default_rng(31).uniform(size=(37, 3))
    output = operator.apply(points)
    assert np.array_equal(output, points)
    assert np.array_equal(operator.inverse(output), points)
    replay = IntervalMobiusCouplingOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(replay.apply(points), output)
    assert np.array_equal(
        np.concatenate((operator.apply(points[:17]), operator.apply(points[17:]))),
        output,
    )


def test_nonzero_operator_is_bounded_invertible_and_orientation_preserving() -> None:
    coefficients = np.zeros((len(STAGES), 3, 6), dtype=np.float64)
    coefficients[:, 0, 0] = np.linspace(0.02, 0.08, len(STAGES))
    coefficients[:, 1, 0] = np.linspace(0.07, 0.01, len(STAGES))
    coefficients[:, 2, :3] = np.array([0.1, 0.04, -0.03])
    operator = IntervalMobiusCouplingOperator(
        stage_channels=STAGES,
        coefficients=coefficients,
        **LIMITS,
    )
    points = np.random.default_rng(9).uniform(0.1, 0.9, size=(41, 3))
    output = operator.apply(points)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.max(np.abs(operator.inverse(output) - points)) < 1e-12
    jacobians = finite_difference_jacobians(operator, points, step=1e-6)
    assert np.min(np.linalg.det(jacobians)) > 0.0
    assert operator.minimum_stage_derivative(points) > 0.0


def test_deterministic_cpu_fit_improves_a_coupled_target() -> None:
    axis = (np.arange(5, dtype=np.float64) + 0.5) / 5.0
    source = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    truth_coefficients = np.zeros((len(STAGES), 3, 6), dtype=np.float64)
    truth_coefficients[:, 0, 0] = [0.04, 0.02, 0.06, 0.03, 0.05, 0.01]
    truth_coefficients[:, 1, 0] = [0.02, 0.05, 0.01, 0.04, 0.03, 0.06]
    truth_coefficients[:, 2, :3] = np.array([0.12, 0.03, -0.02])
    truth = IntervalMobiusCouplingOperator(
        stage_channels=STAGES,
        coefficients=truth_coefficients,
        **LIMITS,
    )
    target = truth.apply(source)
    kwargs = dict(
        stage_channels=STAGES,
        maximum_absolute_coefficient=1.0,
        endpoint_head_bias_initialization=1e-4,
        log_odds_head_initialization=0.0,
        seed=17,
        steps=240,
        learning_rate=0.03,
        coefficient_l2=1e-5,
        gradient_clip_norm=10.0,
        thread_count=1,
        **LIMITS,
    )
    first = fit_interval_mobius_coupling(source, target, **kwargs)
    second = fit_interval_mobius_coupling(source, target, **kwargs)
    assert np.array_equal(first.coefficients, second.coefficients)
    fitted = np.sqrt(np.mean((first.apply(source) - target) ** 2))
    identity = np.sqrt(np.mean((source - target) ** 2))
    assert fitted < 0.2 * identity


def test_inverse_rejects_points_outside_operator_image() -> None:
    coefficients = np.zeros((len(STAGES), 3, 6), dtype=np.float64)
    coefficients[:, 0, 0] = 1.0
    operator = IntervalMobiusCouplingOperator(
        stage_channels=STAGES,
        coefficients=coefficients,
        **LIMITS,
    )
    with pytest.raises(ValueError, match="outside"):
        operator.inverse(np.zeros((1, 3), dtype=np.float64))


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        IntervalMobiusCouplingOperator.identity(
            stage_channels=(3,),
            **LIMITS,
        )
    with pytest.raises(ValueError):
        IntervalMobiusCouplingOperator(
            stage_channels=STAGES,
            coefficients=np.zeros((len(STAGES), 3, 5)),
            **LIMITS,
        )
    operator = IntervalMobiusCouplingOperator.identity(
        stage_channels=STAGES,
        **LIMITS,
    )
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        finite_difference_jacobians(
            operator, np.array([[0.0, 0.5, 0.5]]), step=1e-6
        )
