from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_p247_acescg_openexr_24mp_resources import _fill_probe


def test_fill_probe_is_block_size_invariant() -> None:
    first = np.empty((19, 31, 3), dtype=np.float32)
    second = np.empty_like(first)
    _fill_probe(first, row_block=4, period=17)
    _fill_probe(second, row_block=11, period=17)
    assert np.array_equal(first, second)
    assert first.min() < 0.0
    assert first.max() > 1.0


def test_fill_probe_rejects_wrong_layout() -> None:
    with pytest.raises(ValueError, match="HxWx3 float32"):
        _fill_probe(np.empty((2, 3, 4), dtype=np.float32), row_block=1, period=8)
    with pytest.raises(ValueError, match="HxWx3 float32"):
        _fill_probe(np.empty((2, 3, 3), dtype=np.float64), row_block=1, period=8)
