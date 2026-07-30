from __future__ import annotations

import numpy as np
import pytest

from src.roll2film.density_residual_guard import (
    apply_target_density_residual_guard,
)


KWARGS = {
    "neutral_strength": 0.15,
    "opponent_strength": 0.35,
    "neutral_weights": np.asarray([0.2126, 0.7152, 0.0722]),
    "density_floor": 2.0**-16,
    "hard_boundary_epsilon_encoded_srgb": 0.5 / 255.0,
    "guard_boundary_epsilon_encoded_srgb": 1.0 / 255.0,
}


def test_identity_is_exact_at_cube_endpoints() -> None:
    source = np.asarray(
        [[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.2, 0.4, 0.8]]]
    )
    result = apply_target_density_residual_guard(source, source, **KWARGS)
    assert np.array_equal(result.output, source)
    assert np.all(result.neutral_scale == 1.0)
    assert np.all(result.opponent_scale == 1.0)


def test_neutral_density_delta_remains_neutral() -> None:
    source = np.full((2, 3, 3), 0.4)
    target = np.full_like(source, 0.2)
    result = apply_target_density_residual_guard(source, target, **KWARGS)
    assert np.max(np.ptp(result.output, axis=-1)) <= 1e-15
    assert np.all(result.opponent_scale == 1.0)


def test_shared_scale_stays_inside_source_inclusive_rails() -> None:
    source = np.asarray(
        [[[0.0, 0.5, 1.0], [0.0001, 0.9999, 0.25]]],
        dtype=np.float64,
    )
    target = 1.0 - source
    result = apply_target_density_residual_guard(
        source,
        target,
        **{**KWARGS, "neutral_strength": 1.0, "opponent_strength": 2.0},
    )
    assert np.all(np.isfinite(result.output))
    assert np.all(result.output >= -1e-14)
    assert np.all(result.output <= 1.0 + 1e-14)
    assert np.all((result.neutral_scale >= 0.0) & (result.neutral_scale <= 1.0))
    assert np.all((result.opponent_scale >= 0.0) & (result.opponent_scale <= 1.0))


def test_invalid_inputs_fail_closed() -> None:
    source = np.full((1, 1, 3), 0.5)
    with pytest.raises(ValueError):
        apply_target_density_residual_guard(
            source,
            source,
            **{**KWARGS, "density_floor": 0.0},
        )
    with pytest.raises(ValueError):
        apply_target_density_residual_guard(
            source,
            source,
            **{
                **KWARGS,
                "neutral_weights": np.asarray([1.0, 0.0, 0.0]),
            },
        )
