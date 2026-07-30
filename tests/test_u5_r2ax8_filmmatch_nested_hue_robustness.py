import numpy as np

from src.eval.filmmatch_domain_balanced_capacity import _domain_weights


def test_nested_weights_remain_finite_for_small_inner_population() -> None:
    weights = _domain_weights(100, 4, 0.75)
    assert weights.shape == (104,)
    assert np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    assert np.isclose(np.mean(weights), 1.0)
