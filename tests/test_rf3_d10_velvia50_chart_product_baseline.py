from __future__ import annotations

import numpy as np

from src.eval.velvia50_chart_product_baseline import (
    _fold_rmse,
    _mosaic,
    _sample_mosaic,
    _stable_id,
)


def test_mosaic_roundtrip_preserves_patch_order() -> None:
    patches = np.arange(24 * 3, dtype=np.float32).reshape(24, 3) / 100.0
    mosaic = _mosaic(patches, 6, 4, 3)
    assert mosaic.shape == (18, 12, 3)
    assert np.allclose(_sample_mosaic(mosaic, 6, 4, 3), patches)


def test_fold_rmse_uses_chart_rows() -> None:
    target = np.zeros((24, 3), dtype=np.float64)
    candidate = np.zeros_like(target)
    candidate[4:8] = 1.0
    assert _fold_rmse(candidate, target, 6) == [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]


def test_stable_id_is_key_order_independent() -> None:
    assert _stable_id({"a": 1, "b": 2}) == _stable_id({"b": 2, "a": 1})
