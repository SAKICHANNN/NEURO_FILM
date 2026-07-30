import numpy as np

from src.eval.filmmatch_luma_proxy_router import fit_conservative_threshold


def test_threshold_fit_prefers_precise_separation() -> None:
    feature = np.asarray([0.1, 0.2, 0.3, 0.8, 0.9])
    label = np.asarray([False, False, False, True, True])
    threshold, metrics = fit_conservative_threshold(feature, label)
    assert 0.3 < threshold < 0.8
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
