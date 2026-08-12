from __future__ import annotations

import numpy as np
import pytest

from src.film_physics.independent_density_nps import (
    _quantized_histogram_midranks,
)


def test_histogram_midranks_use_tied_bin_midpoints() -> None:
    values = np.asarray([[0.0, 0.0, 0.5, 1.0]], dtype=np.float64)
    result = _quantized_histogram_midranks(values, rank_bins=3)
    assert np.array_equal(result, np.asarray([[0.25, 0.25, 0.625, 0.875]]))


def test_histogram_midranks_are_monotone_finite_and_repeat_exact() -> None:
    values = np.random.default_rng(20260813).normal(size=(97, 101))
    first = _quantized_histogram_midranks(values, rank_bins=65536)
    second = _quantized_histogram_midranks(values, rank_bins=65536)
    order = np.argsort(values.reshape(-1), kind="stable")
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))
    assert np.all(np.diff(first.reshape(-1)[order]) >= 0.0)
    assert np.all((first > 0.0) & (first < 1.0))


@pytest.mark.parametrize("rank_bins", [True, 1, 65537])
def test_histogram_midranks_reject_invalid_bins(rank_bins: object) -> None:
    with pytest.raises(ValueError):
        _quantized_histogram_midranks(np.ones((2, 2)), rank_bins=rank_bins)  # type: ignore[arg-type]
