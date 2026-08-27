"""Non-promotional artifact diagnostics for bounded FilmCase color renders."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label, sobel
from skimage.color import rgb2lab

from src.color_engine.srgb_transfer import encoded_srgb_to_linear


def _encoded_rgb(value: np.ndarray, label_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3 or array.size == 0:
        raise ValueError(f"{label_name} must be a non-empty HxWx3 array")
    if array.dtype == np.uint8:
        encoded = array.astype(np.float32) / np.float32(255.0)
    elif array.dtype == np.uint16:
        encoded = array.astype(np.float32) / np.float32(65535.0)
    elif np.issubdtype(array.dtype, np.floating):
        encoded = array.astype(np.float32, copy=False)
    else:
        raise ValueError(f"{label_name} must use uint8, uint16 or floating RGB")
    if not np.isfinite(encoded).all() or np.any((encoded < 0.0) | (encoded > 1.0)):
        raise ValueError(f"{label_name} must contain finite encoded RGB in [0, 1]")
    return np.ascontiguousarray(encoded)


def _linear_luminance(encoded: np.ndarray) -> np.ndarray:
    linear = encoded_srgb_to_linear(encoded)
    return np.asarray(
        linear @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        dtype=np.float32,
    )


def structural_render_diagnostics(
    before_rgb: np.ndarray,
    after_rgb: np.ndarray,
    *,
    source_edge_percentile: float = 75.0,
    texture_sigma: float = 1.0,
    texture_source_percentile: float = 75.0,
    flat_source_percentile: float = 25.0,
    minimum_denominator: float = 1e-6,
) -> dict[str, Any]:
    """Report colour-induced edge and texture changes without deciding safety."""

    before = _encoded_rgb(before_rgb, "before_rgb")
    after = _encoded_rgb(after_rgb, "after_rgb")
    if before.shape != after.shape:
        raise ValueError("before_rgb and after_rgb must have matching shapes")
    percentiles = (
        source_edge_percentile,
        texture_source_percentile,
        flat_source_percentile,
    )
    if any(not 0.0 <= float(value) <= 100.0 for value in percentiles):
        raise ValueError("diagnostic percentiles must be in [0, 100]")
    if not np.isfinite(texture_sigma) or texture_sigma <= 0.0:
        raise ValueError("texture_sigma must be finite and positive")
    if not np.isfinite(minimum_denominator) or minimum_denominator <= 0.0:
        raise ValueError("minimum_denominator must be finite and positive")

    before_luma = _linear_luminance(before)
    after_luma = _linear_luminance(after)
    before_gx = sobel(before_luma, axis=1, mode="reflect")
    before_gy = sobel(before_luma, axis=0, mode="reflect")
    after_gx = sobel(after_luma, axis=1, mode="reflect")
    after_gy = sobel(after_luma, axis=0, mode="reflect")
    before_gradient = np.hypot(before_gx, before_gy)
    after_gradient = np.hypot(after_gx, after_gy)
    edge_floor = max(
        float(np.percentile(before_gradient, source_edge_percentile)),
        float(minimum_denominator),
    )
    edge_mask = before_gradient >= edge_floor
    if not np.any(edge_mask):
        raise ValueError("source has no supported edge pixels")
    edge_gain = after_gradient[edge_mask] / np.maximum(
        before_gradient[edge_mask], minimum_denominator
    )
    edge_cosine = (
        before_gx[edge_mask] * after_gx[edge_mask]
        + before_gy[edge_mask] * after_gy[edge_mask]
    ) / np.maximum(
        before_gradient[edge_mask] * after_gradient[edge_mask],
        minimum_denominator,
    )
    edge_cosine = np.clip(edge_cosine, -1.0, 1.0)

    before_high = np.abs(
        before_luma - gaussian_filter(before_luma, sigma=texture_sigma, mode="reflect")
    )
    after_high = np.abs(
        after_luma - gaussian_filter(after_luma, sigma=texture_sigma, mode="reflect")
    )
    texture_floor = max(
        float(np.percentile(before_high, texture_source_percentile)),
        float(minimum_denominator),
    )
    texture_mask = before_high >= texture_floor
    if not np.any(texture_mask):
        raise ValueError("source has no supported texture pixels")
    texture_gain = after_high[texture_mask] / np.maximum(
        before_high[texture_mask], minimum_denominator
    )
    flat_ceiling = float(np.percentile(before_high, flat_source_percentile))
    flat_mask = before_high <= flat_ceiling
    if not np.any(flat_mask):
        raise ValueError("source has no supported flat-region pixels")
    flat_added = np.maximum(after_high[flat_mask] - before_high[flat_mask], 0.0)

    return {
        "diagnostic_only": True,
        "source_edge_pixels": int(np.count_nonzero(edge_mask)),
        "source_edge_fraction": float(np.mean(edge_mask)),
        "edge_direction_cosine_p05": float(np.percentile(edge_cosine, 5.0)),
        "edge_direction_cosine_median": float(np.median(edge_cosine)),
        "edge_gain_p05": float(np.percentile(edge_gain, 5.0)),
        "edge_gain_median": float(np.median(edge_gain)),
        "edge_gain_p95": float(np.percentile(edge_gain, 95.0)),
        "source_texture_pixels": int(np.count_nonzero(texture_mask)),
        "texture_gain_median": float(np.median(texture_gain)),
        "texture_gain_p95": float(np.percentile(texture_gain, 95.0)),
        "source_flat_pixels": int(np.count_nonzero(flat_mask)),
        "flat_new_high_frequency_p95": float(np.percentile(flat_added, 95.0)),
        "flat_new_high_frequency_p99": float(np.percentile(flat_added, 99.0)),
        "flat_new_high_frequency_max": float(np.max(flat_added)),
        "review_instruction": "Use independent metric queues to select original-resolution review targets; these values never authorize or veto a render automatically.",
    }


def chroma_speckle_diagnostics(
    before_rgb: np.ndarray, after_rgb: np.ndarray
) -> dict[str, Any]:
    """Measure newly amplified, high-frequency chroma islands.

    This deliberately returns diagnostics only. Thresholds flag review targets;
    they do not decide a severe-artifact veto without full-resolution visual
    adjudication.
    """

    if (
        before_rgb.shape != after_rgb.shape
        or before_rgb.ndim != 3
        or before_rgb.shape[2] != 3
    ):
        raise ValueError("before_rgb and after_rgb must have matching HxWx3 shapes")
    before = rgb2lab(before_rgb.astype(np.float32) / 255.0)
    after = rgb2lab(after_rgb.astype(np.float32) / 255.0)
    before_chroma = np.linalg.norm(before[..., 1:3], axis=2)
    after_chroma = np.linalg.norm(after[..., 1:3], axis=2)
    before_high = before_chroma - gaussian_filter(before_chroma, sigma=1.0)
    after_high = after_chroma - gaussian_filter(after_chroma, sigma=1.0)
    amplified = np.abs(after_high) - np.abs(before_high)
    saturated_source = before_chroma >= max(
        12.0, float(np.percentile(before_chroma, 75))
    )
    threshold = max(2.0, float(np.percentile(amplified, 99.5)))
    mask = saturated_source & (amplified >= threshold)
    labels, count = label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    areas = np.bincount(labels.ravel())[1:] if count else np.asarray([], dtype=np.int64)
    # Isolated one-pixel chroma impulses are also relevant speckle evidence;
    # they are not discarded merely because they lack a neighboring pixel.
    island_count = int(np.count_nonzero((areas >= 1) & (areas <= 400)))
    return {
        "diagnostic_only": True,
        "before_chroma_mean": float(before_chroma.mean()),
        "after_chroma_mean": float(after_chroma.mean()),
        "amplified_high_frequency_chroma_threshold": threshold,
        "saturated_source_percent": float(saturated_source.mean() * 100.0),
        "speckle_candidate_percent": float(mask.mean() * 100.0),
        "speckle_candidate_pixel_count": int(mask.sum()),
        "small_island_count": island_count,
        "largest_island_pixels": int(areas.max()) if areas.size else 0,
        "review_instruction": "Inspect flagged high-chroma regions at original resolution; do not treat this metric as a pass/fail verdict.",
    }
