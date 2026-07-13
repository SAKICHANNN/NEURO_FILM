from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.operators import AffineColorOperator
from src.roll2film.simulator import default_truth_operator


def test_affine_operator_round_trip_and_serialization() -> None:
    operator = default_truth_operator()
    rgb = np.random.default_rng(7).uniform(0.1, 0.9, size=(64, 3))
    reconstructed = operator.inverse(operator.apply(rgb))
    restored = AffineColorOperator.from_dict(operator.to_dict())

    assert np.max(np.abs(reconstructed - rgb)) < 1e-12
    assert np.allclose(restored.matrix, operator.matrix)
    assert np.allclose(restored.bias, operator.bias)
    assert operator.determinant > 0


def test_identity_operator_and_simulator_are_deterministic() -> None:
    rgb = np.random.default_rng(9).uniform(0.0, 1.0, size=(32, 3))
    identity = AffineColorOperator.identity()
    assert np.array_equal(identity.apply(rgb), rgb)

    from src.roll2film.simulator import PseudoRollConfig, simulate_pseudo_roll

    config = PseudoRollConfig(frames=3, pixels_per_frame=32, seed=99)
    first = simulate_pseudo_roll(config)
    second = simulate_pseudo_roll(config)
    assert all(np.array_equal(a, b) for a, b in zip(first.target_frames, second.target_frames))


def test_affine_operator_rejects_singular_or_wrong_shape() -> None:
    with pytest.raises(ValueError, match="shape"):
        AffineColorOperator(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="orientation-preserving"):
        AffineColorOperator(np.zeros((3, 3)), np.zeros(3))
