from __future__ import annotations

import numpy as np

from src.film_physics.bounded_histogram_copula import (
    _midpoint_table,
    _quantize,
)


def test_quantizer_and_midpoints_match_fixed_histogram_definition() -> None:
    values = np.asarray([[0.0, 0.0, 0.5, 1.0]], dtype=np.float64)
    quantized = _quantize(values, 0.0, 1.0, 3)
    counts = np.bincount(quantized.reshape(-1), minlength=3).astype(np.uint64)
    table = _midpoint_table(counts, values.size)
    assert np.array_equal(quantized, np.asarray([[0, 0, 1, 2]], dtype=np.uint16))
    assert np.array_equal(table[quantized], np.asarray([[0.25, 0.25, 0.625, 0.875]]))


def test_constant_quantizer_is_zero_bin() -> None:
    values = np.full((7, 9), 3.5, dtype=np.float64)
    assert np.array_equal(_quantize(values, 3.5, 3.5, 65536), np.zeros_like(values, dtype=np.uint16))
