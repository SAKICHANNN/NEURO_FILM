from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.bounded_gaussian_residual import (
    BoundedGaussianResidualOperator,
    fit_bounded_gaussian_residual,
    normalized_gaussian_weights,
    published_formula_initialization_witness,
    regular_rgb_centers,
    signed_headroom_map,
)


def test_regular_centers_and_weights_are_deterministic() -> None:
    centers = regular_rgb_centers(2)
    assert centers.shape == (8, 3)
    points = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    weights = normalized_gaussian_weights(
        points, centers, sigma=0.3, epsilon=1e-12
    )
    assert weights.shape == (2, 8)
    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-10)
    assert np.array_equal(
        weights,
        normalized_gaussian_weights(
            points, centers, sigma=0.3, epsilon=1e-12
        ),
    )


def test_signed_headroom_is_bounded_and_identity_at_zero() -> None:
    rgb = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.8]])
    residual = np.array([[50.0, -50.0, -50.0], [-2.0, 0.0, 2.0]])
    output = signed_headroom_map(rgb, residual)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.array_equal(signed_headroom_map(rgb, np.zeros_like(rgb)), rgb)


def test_identity_operator_replay_and_partition_parity() -> None:
    operator = BoundedGaussianResidualOperator.identity(
        axis_size=2, sigma=0.32
    )
    points = np.random.default_rng(7).uniform(size=(31, 3))
    output = operator.apply(points)
    assert np.array_equal(output, points)
    replay = BoundedGaussianResidualOperator.from_dict(
        json.loads(json.dumps(operator.to_dict()))
    )
    assert np.array_equal(replay.apply(points), output)
    assert np.array_equal(
        np.concatenate((operator.apply(points[:13]), operator.apply(points[13:]))),
        output,
    )


def test_deterministic_fit_improves_over_identity() -> None:
    axis = (np.arange(7, dtype=np.float64) + 0.5) / 7.0
    source = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    target = signed_headroom_map(
        source,
        np.column_stack(
            (
                0.35 * (source[:, 2] - 0.5),
                0.2 * (source[:, 0] - source[:, 2]),
                -0.3 * (source[:, 0] - 0.5),
            )
        ),
    )
    first = fit_bounded_gaussian_residual(
        source,
        target,
        axis_size=2,
        sigma=0.32,
        epsilon=1e-12,
        ridge=1e-4,
    )
    second = fit_bounded_gaussian_residual(
        source,
        target,
        axis_size=2,
        sigma=0.32,
        epsilon=1e-12,
        ridge=1e-4,
    )
    assert np.array_equal(first.coefficients, second.coefficients)
    fitted_error = np.sqrt(np.mean((first.apply(source) - target) ** 2))
    identity_error = np.sqrt(np.mean((source - target) ** 2))
    assert fitted_error < 0.1 * identity_error


def test_published_formula_exposes_global_initialization_ambiguity() -> None:
    points = np.array([[0.25, 0.5, 0.75], [1.0, 1.0, 1.0]])
    zero_global = published_formula_initialization_witness(
        points,
        axis_size=3,
        sigma=0.15,
        epsilon=1e-6,
        identity_global=False,
    )
    identity_global = published_formula_initialization_witness(
        points,
        axis_size=3,
        sigma=0.15,
        epsilon=1e-6,
        identity_global=True,
    )
    assert np.max(np.abs(zero_global - points)) < 1e-5
    assert np.max(identity_global) > 1.5


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        regular_rgb_centers(1)
    with pytest.raises(ValueError):
        BoundedGaussianResidualOperator.identity(axis_size=2, sigma=0.0)
    operator = BoundedGaussianResidualOperator.identity(axis_size=2, sigma=0.32)
    with pytest.raises(ValueError):
        operator.apply(np.array([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        operator.apply(np.array([[0.1, 0.2, 0.3]]), strength=1.1)
