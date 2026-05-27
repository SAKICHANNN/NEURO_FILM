"""Deterministic film-effect layer generators."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .layers import FilmLayer


def luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def grain_residual_layer(
    base_rgb: np.ndarray,
    *,
    strength: float = 0.018,
    seed: int = 7,
    color: bool = True,
    name: str = "grain",
) -> FilmLayer:
    rng = np.random.default_rng(seed)
    lum = luminance(base_rgb)[..., None]
    channels = 3 if color else 1
    noise = rng.normal(0.0, 1.0, size=base_rgb.shape[:2] + (channels,)).astype(np.float32)
    if not color:
        noise = np.repeat(noise, 3, axis=2)
    noise = noise - gaussian_filter(noise, sigma=(1.2, 1.2, 0.0))
    noise = noise - noise.mean(axis=(0, 1), keepdims=True)
    noise = noise / max(float(noise.std()), 1e-6)
    envelope = 0.45 + 0.75 * (1.0 - lum)
    residual = noise * envelope * strength
    residual = residual - residual.mean(axis=(0, 1), keepdims=True)
    return FilmLayer(name=name, mode="residual", residual=residual.astype(np.float32))


def halation_layer(
    base_rgb: np.ndarray,
    *,
    strength: float = 0.16,
    threshold: float = 0.78,
    edge_threshold: float = 0.08,
    min_radius: float = 1.1,
    max_radius: float = 10.0,
    radius_gamma: float = 1.35,
    scale_count: int = 6,
    name: str = "halation",
) -> FilmLayer:
    lum = luminance(base_rgb)
    gy, gx = np.gradient(lum)
    edge = np.hypot(gx, gy)
    highlight = _smoothstep(threshold, 1.0, lum)
    edge_mask = _smoothstep(0.0, edge_threshold, edge)
    support = highlight * edge_mask

    # Approximate a spatially varying scattering radius. Each source pixel gets a
    # continuous target radius from its exposure, then is softly assigned across
    # Gaussian scale-space before blurring.
    scale_count = max(3, int(scale_count))
    min_radius = max(0.4, float(min_radius))
    max_radius = max(min_radius + 0.1, float(max_radius))
    sigmas = np.geomspace(min_radius, max_radius, scale_count).astype(np.float32)
    radius = min_radius + (max_radius - min_radius) * np.power(highlight, radius_gamma)
    log_radius = np.log(np.maximum(radius, 1e-4))[..., None]
    log_sigmas = np.log(sigmas).reshape(1, 1, scale_count)
    bandwidth = max(float(np.log(max_radius / min_radius) / max(scale_count - 1, 1)) * 0.9, 1e-3)
    scale_weights = np.exp(-0.5 * ((log_sigmas - log_radius) / bandwidth) ** 2).astype(np.float32)
    scale_weights /= np.maximum(scale_weights.sum(axis=2, keepdims=True), 1e-6)

    halo = np.zeros_like(lum, dtype=np.float32)
    for index, sigma in enumerate(sigmas):
        source = support * scale_weights[..., index]
        halo += gaussian_filter(source, sigma=float(sigma))

    alpha = np.clip(halo * strength, 0.0, min(0.22, strength))
    rgb = np.zeros_like(base_rgb, dtype=np.float32)
    rgb[..., 0] = 1.0
    rgb[..., 1] = 0.34
    rgb[..., 2] = 0.12
    return FilmLayer(name=name, mode="screen", rgb=rgb, alpha=alpha[..., None].astype(np.float32))


def dust_scratch_layer(
    shape: tuple[int, int, int],
    *,
    strength: float = 0.08,
    seed: int = 7,
    name: str = "dust_scratch",
) -> FilmLayer:
    height, width, _ = shape
    rng = np.random.default_rng(seed)
    alpha = np.zeros((height, width, 1), dtype=np.float32)
    speck_count = max(1, int(height * width * 0.00018 * strength * 10.0))
    for _ in range(speck_count):
        y = rng.integers(0, height)
        x = rng.integers(0, width)
        radius = rng.integers(1, 3)
        y0, y1 = max(0, y - radius), min(height, y + radius + 1)
        x0, x1 = max(0, x - radius), min(width, x + radius + 1)
        alpha[y0:y1, x0:x1, 0] = np.maximum(alpha[y0:y1, x0:x1, 0], rng.uniform(0.08, 0.20) * strength)
    scratch_count = max(0, int(width * 0.012 * strength))
    for _ in range(scratch_count):
        x = rng.integers(0, width)
        y0 = rng.integers(0, max(1, height // 3))
        length = rng.integers(max(4, height // 6), max(5, height))
        y1 = min(height, y0 + length)
        alpha[y0:y1, max(0, x - 1) : min(width, x + 1), 0] = np.maximum(
            alpha[y0:y1, max(0, x - 1) : min(width, x + 1), 0],
            rng.uniform(0.03, 0.10) * strength,
        )
    rgb = np.ones(shape, dtype=np.float32)
    return FilmLayer(name=name, mode="alpha", rgb=rgb, alpha=np.clip(alpha, 0.0, 0.25))
