from __future__ import annotations

import numpy as np

from src.eval.flickr_bw_density_structure_challenger import (
    safe_density_candidate,
    synthesize_spectral_field,
)


EDGES = np.asarray([0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.000001])


def test_spectral_field_is_exact_and_profile_sensitive() -> None:
    low = synthesize_spectral_field((96, 128), EDGES, np.asarray([8, 4, 2, 1, 0.5, 0.25]), 29)
    repeat = synthesize_spectral_field((96, 128), EDGES, np.asarray([8, 4, 2, 1, 0.5, 0.25]), 29)
    high = synthesize_spectral_field((96, 128), EDGES, np.asarray([0.25, 0.5, 1, 2, 4, 8]), 29)
    np.testing.assert_array_equal(low, repeat)
    assert not np.array_equal(low, high)
    assert abs(float(np.mean(low))) < 1e-12
    assert abs(float(np.std(low)) - 1.0) < 1e-12


def test_safe_density_candidate_preserves_mean_and_bounds() -> None:
    y, x = np.mgrid[:72, :96]
    basic = 0.2 + 0.6 * x / 95.0
    valid = np.ones_like(basic, dtype=bool)
    field = 0.25 * np.sin(x / 3.0) * np.cos(y / 5.0)
    candidate, scale, drift = safe_density_candidate(basic, valid, field, 3 / 255, 252 / 255, 1.0)
    assert 0.0 < scale <= 1.0
    assert abs(drift) < 1e-12
    assert float(np.min(candidate)) >= 3 / 255
    assert float(np.max(candidate)) <= 252 / 255
    assert not np.array_equal(candidate, basic)
