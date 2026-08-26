from __future__ import annotations

import numpy as np

from src.eval.portra400_chart_product_baseline import _fold_rmse, _stable_id


def test_fold_rmse_uses_disjoint_block_assignments() -> None:
    target = np.zeros((8, 3), dtype=np.float64)
    candidate = np.zeros_like(target)
    candidate[1::2] = 1.0
    blocks = np.arange(8, dtype=np.int64)
    values = _fold_rmse(candidate, target, blocks, 2)
    assert values == [0.0, 1.0]


def test_stable_id_is_key_order_independent() -> None:
    assert _stable_id({"a": 1, "b": 2}) == _stable_id({"b": 2, "a": 1})
