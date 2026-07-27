from __future__ import annotations

import json

import numpy as np
import pytest

from src.roll2film.bounded_convex_gradient import (
    BoundedConvexGradientMap,
    fit_bounded_convex_gradient_map,
)


def _anchors() -> np.ndarray:
    return np.asarray(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9],
            [0.1, 0.9, 0.1],
            [0.1, 0.9, 0.9],
            [0.9, 0.1, 0.1],
            [0.9, 0.1, 0.9],
            [0.9, 0.9, 0.1],
            [0.9, 0.9, 0.9],
        ],
        dtype=np.float64,
    )


def test_identity_is_exact_replayable_and_partition_stable() -> None:
    operator = BoundedConvexGradientMap.identity(anchors=_anchors())
    points = np.random.default_rng(29042).uniform(size=(37, 3))
    output = operator.apply(points)
    assert np.array_equal(output, points)
    assert np.array_equal(operator.inverse(output), points)
    assert np.array_equal(
        operator.jacobians(points),
        np.broadcast_to(np.eye(3), (len(points), 3, 3)),
    )
    replay = BoundedConvexGradientMap.from_dict(
        json.loads(json.dumps(operator.to_dict(), sort_keys=True))
    )
    assert np.array_equal(replay.apply(points), output)
    assert np.array_equal(
        np.concatenate((operator.apply(points[:13]), operator.apply(points[13:]))),
        output,
    )


def test_nonzero_map_is_bounded_spd_and_invertible_on_its_image() -> None:
    biases = np.linspace(-0.4, 0.4, 8, dtype=np.float64)
    biases -= np.mean(biases)
    operator = BoundedConvexGradientMap(
        anchors=_anchors(),
        centered_biases=biases,
        strength=0.55,
    )
    points = np.random.default_rng(29043).uniform(size=(31, 3))
    output = operator.apply(points)
    eigenvalues = np.linalg.eigvalsh(operator.jacobians(points))
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(eigenvalues) >= 1.0 - operator.strength - 1e-12
    restored = operator.inverse(output)
    assert np.max(np.abs(restored - points)) < 1e-10


def test_analytic_jacobian_matches_finite_difference() -> None:
    biases = np.asarray([-0.3, -0.2, -0.1, 0.0, 0.0, 0.1, 0.2, 0.3])
    operator = BoundedConvexGradientMap(
        anchors=_anchors(),
        centered_biases=biases,
        strength=0.4,
    )
    points = np.random.default_rng(29044).uniform(0.1, 0.9, size=(5, 3))
    step = 1e-6
    columns = []
    for channel in range(3):
        offset = np.zeros(3)
        offset[channel] = step
        columns.append(
            (operator.apply(points + offset) - operator.apply(points - offset))
            / (2.0 * step)
        )
    finite = np.stack(columns, axis=-1)
    assert np.max(np.abs(finite - operator.jacobians(points))) < 1e-9


def test_deterministic_fit_improves_on_known_map() -> None:
    source = np.random.default_rng(29045).uniform(size=(96, 3))
    biases = np.linspace(-0.25, 0.25, 8)
    truth = BoundedConvexGradientMap(
        anchors=_anchors(),
        centered_biases=biases,
        strength=0.45,
    )
    target = truth.apply(source)
    kwargs = dict(
        initial_anchors=_anchors(),
        temperature=0.15,
        maximum_strength=0.65,
        maximum_absolute_centered_bias=1.0,
        seed=29046,
        steps=180,
        learning_rate=0.03,
        bias_l2=1e-5,
        anchor_l2_to_initial=1e-6,
        gradient_clip_norm=10.0,
        thread_count=1,
    )
    first = fit_bounded_convex_gradient_map(source, target, **kwargs)
    second = fit_bounded_convex_gradient_map(source, target, **kwargs)
    assert np.array_equal(first.anchors, second.anchors)
    assert np.array_equal(first.centered_biases, second.centered_biases)
    assert first.strength == second.strength
    fitted = np.sqrt(np.mean((first.apply(source) - target) ** 2))
    identity = np.sqrt(np.mean((source - target) ** 2))
    assert fitted < 0.1 * identity


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        BoundedConvexGradientMap(
            anchors=np.zeros((1, 3)),
            centered_biases=np.zeros(1),
            strength=0.0,
        )
    operator = BoundedConvexGradientMap.identity(anchors=_anchors())
    with pytest.raises(ValueError):
        operator.apply(np.asarray([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        BoundedConvexGradientMap(
            anchors=_anchors(),
            centered_biases=np.ones(8),
            strength=0.2,
        )
