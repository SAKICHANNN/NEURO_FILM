from __future__ import annotations

import numpy as np

from src.eval.flickr_bw_density_acutance_challenger import render_acutance_candidate


def test_density_acutance_is_bounded_mean_preserving_and_distinct() -> None:
    y, x = np.mgrid[:120, :160]
    basic = 0.25 + 0.5 * (x >= 80) + 0.02 * np.sin(y / 8.0)
    valid = np.ones_like(basic, dtype=bool)
    density, density_scale, density_drift = render_acutance_candidate(
        basic, valid, 0.9, 2.4, 0.35, 3 / 255, 252 / 255, 1.0, 0.08, True
    )
    linear, linear_scale, linear_drift = render_acutance_candidate(
        basic, valid, 0.9, 2.4, 0.35, 3 / 255, 252 / 255, 1.0, 0.08, False
    )
    assert 0.0 < density_scale <= 1.0
    assert 0.0 < linear_scale <= 1.0
    assert abs(density_drift) < 1e-12
    assert abs(linear_drift) < 1e-12
    assert float(np.max(np.abs(density - basic))) <= 0.08 + 1e-12
    assert float(np.min(density)) >= 3 / 255
    assert float(np.max(density)) <= 252 / 255
    assert not np.array_equal(density, linear)
