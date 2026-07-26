from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
    finite_difference_jacobians,
    fit_cube_diffeomorphic_colour_flow,
)


def _grid(axis_size: int) -> np.ndarray:
    axis = (np.arange(axis_size, dtype=np.float64) + 0.5) / axis_size
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def test_identity_is_exact_replayable_and_partition_stable() -> None:
    operator = CubeDiffeomorphicColourFlow.identity(axis_size=3, integration_steps=8)
    points = np.random.default_rng(71).uniform(size=(37, 3))
    output = operator.apply(points)
    assert np.array_equal(output, points)
    assert np.array_equal(operator.inverse(output), points)
    replay = CubeDiffeomorphicColourFlow.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(replay.apply(points), output)
    assert np.array_equal(
        np.concatenate((operator.apply(points[:13]), operator.apply(points[13:]))),
        output,
    )


def test_nonzero_flow_is_bounded_invertible_and_orientation_preserving() -> None:
    rng = np.random.default_rng(72)
    velocity = rng.normal(scale=0.18, size=(3, 3, 3, 3))
    operator = CubeDiffeomorphicColourFlow(velocity, integration_steps=16)
    points = rng.uniform(0.05, 0.95, size=(31, 3))
    output = operator.apply(points)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.max(np.abs(operator.inverse(output) - points)) < 1e-9
    jacobians = finite_difference_jacobians(operator, points, step=1e-6)
    assert np.min(np.linalg.det(jacobians)) > 0.0


def test_cube_corners_are_fixed() -> None:
    velocity = np.ones((2, 2, 2, 3), dtype=np.float64)
    operator = CubeDiffeomorphicColourFlow(velocity, integration_steps=8)
    corners = np.stack(
        np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0], indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    assert np.array_equal(operator.apply(corners), corners)


def test_deterministic_fit_recovers_a_known_flow() -> None:
    source = _grid(4)
    truth_grid = np.zeros((2, 2, 2, 3), dtype=np.float64)
    truth_grid[..., 0] = 0.35
    truth_grid[..., 1] = -0.22
    truth_grid[..., 2] = 0.12
    truth = CubeDiffeomorphicColourFlow(truth_grid, integration_steps=8)
    target = truth.apply(source)
    kwargs = dict(
        axis_size=2,
        integration_steps=8,
        maximum_absolute_coefficient=1.0,
        seed=73,
        steps=180,
        learning_rate=0.04,
        coefficient_l2=1e-6,
        velocity_smoothness_l2=1e-6,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    first = fit_cube_diffeomorphic_colour_flow(source, target, **kwargs)
    second = fit_cube_diffeomorphic_colour_flow(source, target, **kwargs)
    assert np.array_equal(first.velocity_grid, second.velocity_grid)
    fitted = np.sqrt(np.mean((first.apply(source) - target) ** 2))
    identity = np.sqrt(np.mean((source - target) ** 2))
    assert fitted < 0.05 * identity


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        CubeDiffeomorphicColourFlow.identity(axis_size=1)
    operator = CubeDiffeomorphicColourFlow.identity(axis_size=2)
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        finite_difference_jacobians(
            operator, np.array([[0.0, 0.5, 0.5]]), step=1e-6
        )
