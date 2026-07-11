"""Non-promotional artifact diagnostics for bounded FilmCase color renders."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, label
from skimage.color import rgb2lab


def chroma_speckle_diagnostics(before_rgb: np.ndarray, after_rgb: np.ndarray) -> dict[str, Any]:
    """Measure newly amplified, high-frequency chroma islands.

    This deliberately returns diagnostics only. Thresholds flag review targets;
    they do not decide a severe-artifact veto without full-resolution visual
    adjudication.
    """

    if before_rgb.shape != after_rgb.shape or before_rgb.ndim != 3 or before_rgb.shape[2] != 3:
        raise ValueError("before_rgb and after_rgb must have matching HxWx3 shapes")
    before = rgb2lab(before_rgb.astype(np.float32) / 255.0)
    after = rgb2lab(after_rgb.astype(np.float32) / 255.0)
    before_chroma = np.linalg.norm(before[..., 1:3], axis=2)
    after_chroma = np.linalg.norm(after[..., 1:3], axis=2)
    before_high = before_chroma - gaussian_filter(before_chroma, sigma=1.0)
    after_high = after_chroma - gaussian_filter(after_chroma, sigma=1.0)
    amplified = np.abs(after_high) - np.abs(before_high)
    saturated_source = before_chroma >= max(12.0, float(np.percentile(before_chroma, 75)))
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
