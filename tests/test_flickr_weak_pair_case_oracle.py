from __future__ import annotations

import numpy as np

from src.eval.flickr_weak_pair_case_oracle import _rmse


def test_rmse_and_case_oracle_order_are_exact() -> None:
    target = np.asarray([[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]])
    candidates = {
        "bad": target + 0.1,
        "best": target + 0.01,
        "middle": target + 0.04,
    }
    errors = {key: _rmse(value, target) for key, value in candidates.items()}
    ordered = sorted(errors, key=lambda key: (errors[key], key))
    assert ordered == ["best", "middle", "bad"]
    assert np.isclose(errors["best"], 0.01)
