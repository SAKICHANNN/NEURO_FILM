import numpy as np
import pytest

from src.eval.photo_texture_reference import (
    describe_patch,
    plane_residual,
    select_patches,
)


def test_plane_removed_and_input_owned():
    y, x = np.mgrid[:64, :64]
    a = np.repeat((0.3 + 0.001 * x + 0.002 * y)[..., None], 3, -1)
    before = a.copy()
    assert abs(plane_residual(a)).max() < 1e-12
    np.testing.assert_array_equal(a, before)


def test_white_noise_statistics():
    a = 0.5 + np.random.default_rng(3).normal(0, 0.01, (64, 64, 3))
    desc, residual, spectrum = describe_patch(a)
    np.testing.assert_allclose(desc["std_rgb"], 0.01, rtol=0.08)
    assert abs(desc["luma_lag1"]) < 0.06
    assert 0.75 < desc["jpeg8_boundary_ratio"] < 1.25
    assert np.isfinite(spectrum).all() and spectrum.min() >= 0
    assert abs(residual.mean()) < 1e-12


def test_selection_is_disjoint_deterministic_and_rejects_extremes():
    cfg = {
        "patch": 64,
        "stride": 64,
        "max_patches": 8,
        "max_extreme_fraction": 0.01,
        "max_smoothed_gradient_p90": 0.01,
    }
    a = np.full((128, 128, 3), 0.5)
    first = select_patches(a, cfg)
    assert first == select_patches(a, cfg)
    assert len(first) == 4
    assert len({(row[1], row[2]) for row in first}) == 4
    assert not select_patches(np.zeros_like(a), cfg)


def test_invalid_rejected():
    with pytest.raises(ValueError):
        plane_residual(np.full((64, 64, 3), np.nan))
    with pytest.raises(ValueError):
        plane_residual(np.ones((64, 64)))
