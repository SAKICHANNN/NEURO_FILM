"""Development-only flat-patch residuals; residual is not identified film noise."""

import numpy as np
from scipy.ndimage import gaussian_filter


def plane_residual(patch):
    a = np.asarray(patch, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 8:
        raise ValueError("Expected HWC RGB patch, at least 8x8")
    if not np.isfinite(a).all() or a.min() < 0 or a.max() > 1:
        raise ValueError("Expected finite unit-range RGB")
    yy, xx = np.mgrid[: a.shape[0], : a.shape[1]]
    design = np.stack([np.ones_like(xx), xx, yy], -1).reshape(-1, 3)
    coefficients = np.linalg.lstsq(design, a.reshape(-1, 3), rcond=None)[0]
    return a - (design @ coefficients).reshape(a.shape)


def select_patches(image, cfg):
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 3 or a.shape[2] != 3 or not np.isfinite(a).all():
        raise ValueError("Expected finite HWC RGB")
    if a.min() < 0 or a.max() > 1:
        raise ValueError("Expected unit-range RGB")
    size, stride = cfg["patch"], cfg["stride"]
    if size < 8 or stride < size or cfg["max_patches"] < 1:
        raise ValueError("Invalid disjoint patch configuration")
    smooth = gaussian_filter(a @ np.array([0.2126, 0.7152, 0.0722]), 2.0)
    dy, dx = np.gradient(smooth)
    gradient = np.hypot(dx, dy)
    candidates = []
    for y in range(0, a.shape[0] - size + 1, stride):
        for x in range(0, a.shape[1] - size + 1, stride):
            p = a[y : y + size, x : x + size]
            extreme = float(np.mean(np.any((p < 0.03) | (p > 0.97), axis=-1)))
            g = float(np.quantile(gradient[y : y + size, x : x + size], 0.9))
            if (
                extreme < cfg["max_extreme_fraction"]
                and g <= cfg["max_smoothed_gradient_p90"]
            ):
                candidates.append((g, y, x, extreme))
    return sorted(candidates)[: cfg["max_patches"]]


def describe_patch(patch):
    residual = plane_residual(patch)
    centered = residual.reshape(-1, 3)
    cov = centered.T @ centered / len(centered)
    luma = residual @ np.array([0.2126, 0.7152, 0.0722])
    energy = float(np.mean(luma**2))
    lag = float(
        (np.mean(luma[1:] * luma[:-1]) + np.mean(luma[:, 1:] * luma[:, :-1])) / 2
    )
    window = np.outer(np.hanning(luma.shape[0]), np.hanning(luma.shape[1]))
    spectrum = abs(np.fft.fft2(luma * window)) ** 2 / np.sum(window**2)
    dy, dx = np.diff(residual, axis=0), np.diff(residual, axis=1)
    # Coordinates are multiples of64 in the stored raster (no EXIF rotation).
    by = (np.arange(1, residual.shape[0]) % 8) == 0
    bx = (np.arange(1, residual.shape[1]) % 8) == 0
    boundary = (np.mean(dy[by] ** 2) + np.mean(dx[:, bx] ** 2)) / 2
    interior = (np.mean(dy[~by] ** 2) + np.mean(dx[:, ~bx] ** 2)) / 2
    return (
        {
            "mean_rgb": np.mean(patch, (0, 1)).tolist(),
            "std_rgb": np.sqrt(np.diag(cov)).tolist(),
            "cov_rgb": cov.tolist(),
            "luma_std": float(np.sqrt(energy)),
            "luma_lag1": lag / (energy + 1e-15),
            "jpeg8_boundary_ratio": float(boundary / (interior + 1e-15)),
        },
        residual,
        spectrum,
    )
