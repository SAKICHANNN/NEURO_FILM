"""Chroma-only residual rendering helpers."""

from __future__ import annotations

import numpy as np
from skimage.color import lab2rgb, rgb2lab


def bounded_chroma_residual(
    rgb: np.ndarray,
    delta_ab: np.ndarray,
    *,
    max_delta: float = 12.0,
    output_margin: int = 4,
) -> np.ndarray:
    """Apply a bounded Lab a/b residual while preserving source Lab L."""
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32) / 255.0
    lab = rgb2lab(np.clip(rgb, 0.0, 1.0))
    delta = np.asarray(delta_ab, dtype=np.float32)
    if delta.shape != lab[..., 1:3].shape:
        raise ValueError(f"delta_ab shape {delta.shape} does not match image chroma shape {lab[..., 1:3].shape}")
    delta = np.clip(delta, -max_delta, max_delta)
    out_lab = lab.copy()
    out_lab[..., 1:3] = lab[..., 1:3] + delta
    out_rgb = np.clip(lab2rgb(out_lab), 0.0, 1.0)
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out_rgb = np.clip(out_rgb, low, high)
    return out_rgb.astype(np.float32)


def constant_residual(shape: tuple[int, int], delta_a: float, delta_b: float) -> np.ndarray:
    """Create a constant ``[H, W, 2]`` chroma residual map."""
    residual = np.zeros((shape[0], shape[1], 2), dtype=np.float32)
    residual[..., 0] = delta_a
    residual[..., 1] = delta_b
    return residual
