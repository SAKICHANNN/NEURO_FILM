from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.monotone_coupling import (
    TriangularMonotoneCouplingOperator,
    finite_difference_jacobians,
    fit_triangular_monotone_coupling,
    regular_conditioner_centers,
)


STAGES = (0, 1, 2, 1, 2, 0)


def test_identity_replay_inverse_and_partition_parity() -> None:
    operator = TriangularMonotoneCouplingOperator.identity(
        stage_channels=STAGES,
        axis_size=2,
        sigma=0.3,
    )
    points = np.random.default_rng(31).uniform(size=(37, 3))
    output = operator.apply(points)
    assert np.array_equal(output, points)
    assert np.array_equal(operator.inverse(output), points)
    replay = TriangularMonotoneCouplingOperator.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(replay.apply(points), output)
    assert np.array_equal(
        np.concatenate((operator.apply(points[:17]), operator.apply(points[17:]))),
        output,
    )


def test_nonzero_operator_is_bounded_invertible_and_orientation_preserving() -> None:
    centers = regular_conditioner_centers(2)
    coefficients = np.zeros((len(STAGES), 3 + len(centers)))
    coefficients[:, 0] = np.linspace(-0.2, 0.2, len(STAGES))
    coefficients[:, 3:] = 0.05
    operator = TriangularMonotoneCouplingOperator(
        stage_channels=STAGES,
        centers=centers,
        sigma=0.3,
        coefficients=coefficients,
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
    truth_centers = regular_conditioner_centers(2)
    truth_coefficients = np.zeros((len(STAGES), 3 + len(truth_centers)))
    truth_coefficients[:, 0] = [0.2, -0.15, 0.1, 0.12, -0.08, 0.06]
    truth_coefficients[:, 1:3] = 0.04
    truth = TriangularMonotoneCouplingOperator(
        stage_channels=STAGES,
        centers=truth_centers,
        sigma=0.3,
        coefficients=truth_coefficients,
    )
    target = truth.apply(source)
    kwargs = dict(
        stage_channels=STAGES,
        axis_size=2,
        sigma=0.3,
        epsilon=1e-12,
        maximum_absolute_coefficient=1.0,
        seed=17,
        steps=200,
        learning_rate=0.03,
        coefficient_l2=1e-5,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    first = fit_triangular_monotone_coupling(source, target, **kwargs)
    second = fit_triangular_monotone_coupling(source, target, **kwargs)
    assert np.array_equal(first.coefficients, second.coefficients)
    fitted = np.sqrt(np.mean((first.apply(source) - target) ** 2))
    identity = np.sqrt(np.mean((source - target) ** 2))
    assert fitted < 0.15 * identity


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        regular_conditioner_centers(1)
    with pytest.raises(ValueError):
        TriangularMonotoneCouplingOperator.identity(
            stage_channels=(3,), axis_size=2, sigma=0.3
        )
    operator = TriangularMonotoneCouplingOperator.identity(
        stage_channels=STAGES, axis_size=2, sigma=0.3
    )
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        finite_difference_jacobians(
            operator, np.array([[0.0, 0.5, 0.5]]), step=1e-6
        )
