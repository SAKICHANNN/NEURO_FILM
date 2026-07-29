from __future__ import annotations

import numpy as np
import pytest

from src.film_physics.native_standard_strength import (
    apply_native_standard_display_strength,
)


def test_native_standard_strength_is_exact_bounded_convex_path() -> None:
    source = np.ascontiguousarray(
        np.array([[[0.1, 0.4, 0.8]]], dtype=np.float32)
    )
    styled = np.ascontiguousarray(
        np.array([[[0.9, 0.2, 0.6]]], dtype=np.float32)
    )
    assert np.array_equal(
        apply_native_standard_display_strength(
            source, styled, strength=0.0
        ),
        source,
    )
    assert np.array_equal(
        apply_native_standard_display_strength(
            source, styled, strength=1.0
        ),
        styled,
    )
    middle = apply_native_standard_display_strength(
        source, styled, strength=0.5
    )
    np.testing.assert_allclose(
        middle,
        np.array([[[0.5, 0.3, 0.7]]], dtype=np.float32),
        rtol=0.0,
        atol=1e-7,
    )


@pytest.mark.parametrize("strength", [-0.01, 1.01, float("nan")])
def test_native_standard_strength_rejects_invalid_strength(
    strength: float,
) -> None:
    source = np.zeros((1, 1, 3), dtype=np.float32)
    styled = np.ones((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        apply_native_standard_display_strength(
            source, styled, strength=strength
        )
