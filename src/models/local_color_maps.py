"""Heuristic local bounded color maps for color-only film rendering research."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import compress_to_srgb_gamut, lab_to_rgb_no_clip


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _skin_mask(lab: np.ndarray) -> np.ndarray:
    return (
        (lab[..., 0] > 20.0)
        & (lab[..., 0] < 92.0)
        & (lab[..., 1] > 4.0)
        & (lab[..., 1] < 28.0)
        & (lab[..., 2] > 4.0)
        & (lab[..., 2] < 46.0)
    ).astype(np.float32)


def build_local_masks(source_rgb: np.ndarray) -> dict[str, np.ndarray]:
    source_lab = rgb2lab(np.clip(source_rgb, 0.0, 1.0))
    chroma = np.linalg.norm(source_lab[..., 1:3], axis=2)
    sky = (
        _smoothstep(35.0, 65.0, source_lab[..., 0])
        * _smoothstep(-2.0, -16.0, source_lab[..., 2])
        * (1.0 - _smoothstep(20.0, 45.0, chroma))
    )
    foliage = (
        _smoothstep(-2.0, -18.0, source_lab[..., 1])
        * _smoothstep(8.0, 32.0, source_lab[..., 2])
        * _smoothstep(18.0, 80.0, source_lab[..., 0])
    )
    warm_highlight = _smoothstep(70.0, 94.0, source_lab[..., 0]) * _smoothstep(3.0, 20.0, source_lab[..., 2])
    neutral = (1.0 - _smoothstep(5.0, 14.0, chroma)).astype(np.float32)
    skin = _skin_mask(source_lab)
    protect = np.clip(np.maximum(neutral * 0.85, skin * 0.95), 0.0, 1.0)

    def clean(mask: np.ndarray) -> np.ndarray:
        mask = np.clip(mask * (1.0 - protect), 0.0, 1.0)
        return gaussian_filter(mask.astype(np.float32), sigma=1.0)

    return {
        "sky": clean(sky),
        "foliage": clean(foliage),
        "warm_highlight": clean(warm_highlight),
        "protect": protect.astype(np.float32),
    }


def apply_local_color_maps(
    source_rgb: np.ndarray,
    base_rgb: np.ndarray,
    *,
    style: str,
    strength: float = 1.0,
    output_margin: int = 4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply bounded local color maps on top of a safe base render."""
    source_rgb = source_rgb.astype(np.float32)
    base_rgb = base_rgb.astype(np.float32)
    source_lab = rgb2lab(np.clip(source_rgb, 0.0, 1.0))
    base_lab = rgb2lab(np.clip(base_rgb, 0.0, 1.0))
    masks = build_local_masks(source_rgb)
    out_lab = base_lab.copy()
    style_scale = {
        "velvia_50": 1.15,
        "ektar_100": 1.10,
        "portra_400": 1.10,
        "portra_800": 1.10,
        "vision3_250d": 1.05,
        "vision3_500t": 1.05,
    }.get(style, 0.55)
    amount = strength * style_scale

    sky = masks["sky"][..., None]
    foliage = masks["foliage"][..., None]
    warm = masks["warm_highlight"][..., None]
    protect = masks["protect"][..., None]

    out_lab[..., 1:3] += sky * np.asarray([-0.8, -2.2], dtype=np.float32) * amount
    out_lab[..., 1:3] += foliage * np.asarray([-1.8, 1.4], dtype=np.float32) * amount
    out_lab[..., 1:3] += warm * np.asarray([0.7, 1.5], dtype=np.float32) * amount
    base_chroma = np.linalg.norm(base_lab[..., 1:3], axis=2, keepdims=True)
    rich_mask = _smoothstep(10.0, 24.0, base_chroma) * (1.0 - protect)
    semantic_mask = np.clip(np.maximum.reduce([sky, foliage, warm, rich_mask * 0.65]), 0.0, 1.0)
    chroma_gain = 1.0 + semantic_mask * (0.075 * amount)
    out_lab[..., 1:3] *= chroma_gain
    out_lab[..., 1:3] = out_lab[..., 1:3] * (1.0 - protect * 0.12) + base_lab[..., 1:3] * (protect * 0.12)

    compressed = compress_to_srgb_gamut(source_lab, out_lab)
    result = lab_to_rgb_no_clip(compressed)
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        result = np.clip(result, low, high)
    metrics = {f"{name}_mean": float(value.mean()) for name, value in masks.items()}
    return result.astype(np.float32), metrics
