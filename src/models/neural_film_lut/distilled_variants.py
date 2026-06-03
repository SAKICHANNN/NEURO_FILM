"""Numpy distilled Neural LUT V2 variants.

These runtimes are deliberately small and deterministic. They let us compare
architecture ideas before spending more time on heavier torch training.
"""

from __future__ import annotations

import numpy as np

from src.models.neural_film_lut.distilled_seplut import apply_3d_residual_np, image_features


CONTEXT_NAMES = ["neutral", "skin_warm", "sky_cyan", "foliage", "highlight", "shadow"]


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    t = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def nilut_basis(rgb: np.ndarray) -> np.ndarray:
    """Return polynomial NILUT basis for RGB pixels in the final dimension."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    chroma = rgb.max(axis=-1) - rgb.min(axis=-1)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.stack(
        [
            np.ones_like(r),
            r,
            g,
            b,
            r * r,
            g * g,
            b * b,
            r * g,
            r * b,
            g * b,
            luma,
            chroma,
            luma * chroma,
        ],
        axis=-1,
    ).astype(np.float32)


def context_weights(rgb: np.ndarray) -> np.ndarray:
    """Soft deterministic context weights for a context-aware 4D LUT proxy."""
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    chroma = maxc - minc
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    neutral = 1.0 - _smoothstep(0.025, 0.075, chroma)
    warm = _smoothstep(0.03, 0.16, chroma) * _smoothstep(0.00, 0.16, r - b) * _smoothstep(0.00, 0.10, r - g)
    sky = _smoothstep(0.02, 0.14, chroma) * _smoothstep(-0.03, 0.12, b - r) * _smoothstep(-0.04, 0.10, g - r)
    foliage = _smoothstep(0.02, 0.16, chroma) * _smoothstep(-0.02, 0.13, g - r) * _smoothstep(-0.02, 0.12, g - b)
    highlight = _smoothstep(0.70, 0.92, luma)
    shadow = 1.0 - _smoothstep(0.12, 0.34, luma)
    weights = np.stack([neutral, warm, sky, foliage, highlight, shadow], axis=-1)
    weights = np.maximum(weights, 1e-4)
    return (weights / weights.sum(axis=-1, keepdims=True)).astype(np.float32)


def apply_distilled_nilut(
    rgb: np.ndarray,
    *,
    style_index: int,
    coeffs: np.ndarray,
    gate_weights: np.ndarray,
    strength: float,
    residual_limit: float = 0.22,
    output_margin: int = 4,
) -> tuple[np.ndarray, float]:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    residual = nilut_basis(rgb) @ coeffs[style_index]
    residual = np.clip(residual, -residual_limit, residual_limit)
    base = np.clip(rgb + residual, 0.0, 1.0)
    gate = float(np.clip(image_features(rgb) @ gate_weights[style_index], 0.45, 1.65))
    out = rgb + (base - rgb) * float(strength) * gate
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out = np.clip(out, low, high)
    return out.astype(np.float32), gate


def apply_distilled_context4d(
    rgb: np.ndarray,
    *,
    style_index: int,
    residual4d: np.ndarray,
    gate_weights: np.ndarray,
    strength: float,
    output_margin: int = 4,
) -> tuple[np.ndarray, float]:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    weights = context_weights(rgb)
    residual = np.zeros_like(rgb, dtype=np.float32)
    for context_index in range(residual4d.shape[1]):
        context_residual = apply_3d_residual_np(rgb, residual4d[style_index, context_index])
        residual += context_residual * weights[..., context_index : context_index + 1]
    base = np.clip(rgb + residual, 0.0, 1.0)
    gate = float(np.clip(image_features(rgb) @ gate_weights[style_index], 0.45, 1.65))
    out = rgb + (base - rgb) * float(strength) * gate
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out = np.clip(out, low, high)
    return out.astype(np.float32), gate
