from __future__ import annotations

import numpy as np
import pytest

from src.color_engine.oklch_local_minde import (
    linear_rec2020_to_oklab,
    local_minde_rec2020,
    oklab_to_linear_rec2020,
)


def test_oklab_roundtrip_extended_rec2020() -> None:
    rng = np.random.default_rng(8142026)
    source = rng.uniform(-0.2, 1.2, size=(19, 23, 3)).astype(np.float32)
    restored = oklab_to_linear_rec2020(linear_rec2020_to_oklab(source))
    assert np.max(np.abs(restored - source.astype(np.float64))) < 2e-7


def test_local_minde_preserves_in_gamut_pixels_exactly() -> None:
    source = np.asarray(
        [[[0.0, 0.25, 1.0], [0.2, 0.3, 0.4], [1.0, 1.0, 1.0]]],
        dtype=np.float32,
    )
    mapped, ratio = local_minde_rec2020(source)
    assert np.array_equal(mapped, source)
    assert np.array_equal(ratio, np.ones((1, 3), dtype=np.float32))


def test_local_minde_maps_extended_values_without_mutation() -> None:
    source = np.asarray(
        [[[-0.15, 0.45, 1.2], [1.3, 0.1, 0.2], [0.2, 1.15, -0.1]]],
        dtype=np.float32,
    )
    before = source.copy()
    mapped_a, ratio_a = local_minde_rec2020(source)
    mapped_b, ratio_b = local_minde_rec2020(source)
    assert np.array_equal(source, before)
    assert np.array_equal(mapped_a, mapped_b)
    assert np.array_equal(ratio_a, ratio_b)
    assert np.isfinite(mapped_a).all()
    assert np.min(mapped_a) >= 0.0
    assert np.max(mapped_a) <= 1.0
    assert np.all((ratio_a >= 0.0) & (ratio_a <= 1.0))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"jnd": 0.0}, "jnd"), ({"epsilon": float("nan")}, "epsilon")],
)
def test_local_minde_rejects_invalid_parameters(
    kwargs: dict[str, float], message: str
) -> None:
    source = np.zeros((1, 1, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=message):
        local_minde_rec2020(source, **kwargs)


def test_local_minde_rejects_nonfinite_or_wrong_dtype() -> None:
    with pytest.raises(TypeError, match="float32"):
        local_minde_rec2020(np.zeros((1, 1, 3), dtype=np.float64))
    source = np.zeros((1, 1, 3), dtype=np.float32)
    source[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        local_minde_rec2020(source)
