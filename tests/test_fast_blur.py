from __future__ import annotations

import time

import numpy as np

from src.filmfx.fast_blur import gaussian_filter_safe


def test_large_sigma_small_image_returns_quickly_and_preserves_shape() -> None:
    image = np.zeros((48, 64), dtype=np.float32)
    image[24, 32] = 1.0

    start = time.perf_counter()
    blurred = gaussian_filter_safe(image, sigma=120.0)
    elapsed = time.perf_counter() - start

    assert blurred.shape == image.shape
    assert blurred.dtype == np.float32
    assert np.isfinite(blurred).all()
    assert elapsed < 2.0


def test_large_sigma_rgb_preserves_channels() -> None:
    image = np.zeros((60, 80, 3), dtype=np.float32)
    image[30, 40, 0] = 1.0
    image[30, 40, 1] = 0.5

    blurred = gaussian_filter_safe(image, sigma=(80.0, 80.0, 0.0))

    assert blurred.shape == image.shape
    assert float(blurred[..., 0].max()) > float(blurred[..., 1].max())
    assert float(blurred[..., 2].max()) == 0.0
