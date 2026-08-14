from __future__ import annotations

import numpy as np
import pytest

from src.color_engine.oklab_analytical_interior import (
    analytical_oklab_interior_rec2020,
)
from src.color_engine.oklch_local_minde import linear_rec2020_to_oklab


def test_in_gamut_pixels_are_bit_exact() -> None:
    source = np.asarray(
        [[[0.0, 0.5, 1.0], [0.2, 0.3, 0.4]]], dtype=np.float32
    )
    output, ratio = analytical_oklab_interior_rec2020(source)
    assert np.array_equal(output, source)
    assert np.array_equal(ratio, np.ones((1, 2), dtype=np.float32))


def test_extended_pixels_reach_interior_without_changing_hue_direction() -> None:
    source = np.asarray(
        [[[-0.02, 0.35, 0.8], [1.08, 0.4, 0.1], [-0.01, -0.005, 0.02]]],
        dtype=np.float32,
    )
    output, ratio = analytical_oklab_interior_rec2020(source)
    margin = 2.0 / 65535.0
    assert np.all(output >= margin)
    assert np.all(output <= 1.0 - margin)
    assert np.all((ratio >= 0.0) & (ratio <= 1.0))

    before = linear_rec2020_to_oklab(source)
    after = linear_rec2020_to_oklab(output)
    before_hue = np.arctan2(before[..., 2], before[..., 1])
    after_hue = np.arctan2(after[..., 2], after[..., 1])
    delta = np.abs((after_hue - before_hue + np.pi) % (2.0 * np.pi) - np.pi)
    valid = (np.hypot(before[..., 1], before[..., 2]) > 1e-4) & (
        np.hypot(after[..., 1], after[..., 2]) > 4e-6
    )
    assert float(np.max(np.degrees(delta[valid]))) < 0.01


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"softness": 0.0}, "softness"),
        ({"margin": 0.5}, "margin"),
        ({"iterations": 0}, "iterations"),
    ],
)
def test_invalid_parameters_fail_closed(kwargs: dict[str, float], message: str) -> None:
    source = np.zeros((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        analytical_oklab_interior_rec2020(source, **kwargs)


def test_invalid_source_contract_fails_closed() -> None:
    with pytest.raises(TypeError, match="float32"):
        analytical_oklab_interior_rec2020(np.zeros((1, 1, 3), dtype=np.float64))
    bad = np.zeros((1, 1, 3), dtype=np.float32)
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        analytical_oklab_interior_rec2020(bad)
