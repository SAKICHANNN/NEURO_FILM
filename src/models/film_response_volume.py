"""Stock-specific deterministic film response volume research layer."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import compress_to_srgb_gamut, lab_to_rgb_no_clip


def _smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _angular_mask(hue_degrees: np.ndarray, center: float, width: float) -> np.ndarray:
    delta = np.abs((hue_degrees - center + 180.0) % 360.0 - 180.0)
    return np.exp(-0.5 * (delta / max(width, 1e-3)) ** 2).astype(np.float32)


def _skin_mask(lab: np.ndarray) -> np.ndarray:
    return (
        (lab[..., 0] > 20.0)
        & (lab[..., 0] < 92.0)
        & (lab[..., 1] > 4.0)
        & (lab[..., 1] < 28.0)
        & (lab[..., 2] > 4.0)
        & (lab[..., 2] < 46.0)
    ).astype(np.float32)


STYLE_PARAMS: dict[str, dict] = {
    "ektar_100": {
        "contrast": 1.12,
        "toe": 0.030,
        "shoulder": 0.030,
        "lift": -0.006,
        "global_ab": (0.15, -0.35),
        "shadow_ab": (-1.1, -0.9),
        "mid_ab": (0.25, 0.10),
        "highlight_ab": (0.95, 0.85),
        "sat": 1.10,
        "sat_shadow": 0.95,
        "sat_highlight": 1.02,
        "hue_pushes": [
            (28.0, 32.0, (1.25, 0.45)),
            (205.0, 42.0, (-0.75, -1.30)),
            (115.0, 38.0, (-0.45, 0.55)),
        ],
        "neutral_protect": 0.48,
        "skin_protect": 0.42,
    },
    "portra_400": {
        "contrast": 1.04,
        "toe": -0.006,
        "shoulder": 0.055,
        "lift": 0.012,
        "global_ab": (0.28, 0.50),
        "shadow_ab": (-0.60, -0.45),
        "mid_ab": (0.40, 0.80),
        "highlight_ab": (0.55, 1.25),
        "sat": 1.00,
        "sat_shadow": 0.92,
        "sat_highlight": 0.88,
        "hue_pushes": [
            (35.0, 34.0, (0.35, 0.85)),
            (115.0, 42.0, (-0.70, 0.20)),
            (215.0, 45.0, (-0.25, -0.55)),
        ],
        "neutral_protect": 0.62,
        "skin_protect": 0.55,
    },
    "portra_800": {
        "contrast": 1.07,
        "toe": 0.005,
        "shoulder": 0.050,
        "lift": 0.006,
        "global_ab": (0.15, 0.80),
        "shadow_ab": (-0.75, -0.20),
        "mid_ab": (0.35, 0.95),
        "highlight_ab": (0.70, 1.35),
        "sat": 1.04,
        "sat_shadow": 0.94,
        "sat_highlight": 0.90,
        "hue_pushes": [
            (35.0, 36.0, (0.50, 1.00)),
            (115.0, 42.0, (-0.55, 0.30)),
            (230.0, 48.0, (-0.30, -0.40)),
        ],
        "neutral_protect": 0.58,
        "skin_protect": 0.52,
    },
    "velvia_50": {
        "contrast": 1.18,
        "toe": 0.045,
        "shoulder": 0.020,
        "lift": -0.012,
        "global_ab": (0.25, -0.45),
        "shadow_ab": (-0.85, -1.25),
        "mid_ab": (0.25, -0.20),
        "highlight_ab": (0.35, 0.45),
        "sat": 1.20,
        "sat_shadow": 1.03,
        "sat_highlight": 1.05,
        "hue_pushes": [
            (115.0, 38.0, (-1.40, 1.10)),
            (215.0, 40.0, (-1.10, -1.90)),
            (315.0, 42.0, (0.90, -1.10)),
        ],
        "neutral_protect": 0.42,
        "skin_protect": 0.38,
    },
    "vision3_250d": {
        "contrast": 1.06,
        "toe": 0.010,
        "shoulder": 0.070,
        "lift": 0.006,
        "global_ab": (0.05, 0.65),
        "shadow_ab": (-0.80, -0.55),
        "mid_ab": (0.20, 0.75),
        "highlight_ab": (0.50, 1.20),
        "sat": 1.03,
        "sat_shadow": 0.94,
        "sat_highlight": 0.92,
        "hue_pushes": [
            (40.0, 38.0, (0.30, 1.00)),
            (205.0, 45.0, (-0.60, -0.70)),
            (115.0, 42.0, (-0.35, 0.35)),
        ],
        "neutral_protect": 0.56,
        "skin_protect": 0.48,
    },
    "vision3_500t": {
        "contrast": 1.08,
        "toe": 0.018,
        "shoulder": 0.075,
        "lift": 0.004,
        "global_ab": (-0.15, 0.20),
        "shadow_ab": (-1.25, -1.15),
        "mid_ab": (-0.10, 0.15),
        "highlight_ab": (0.70, 1.05),
        "sat": 1.02,
        "sat_shadow": 0.95,
        "sat_highlight": 0.90,
        "hue_pushes": [
            (205.0, 45.0, (-1.10, -1.30)),
            (35.0, 42.0, (0.50, 0.95)),
            (110.0, 45.0, (-0.45, 0.15)),
        ],
        "neutral_protect": 0.54,
        "skin_protect": 0.46,
    },
}


def _apply_tone(lab: np.ndarray, params: dict, strength: float) -> np.ndarray:
    out = lab.copy()
    y = np.clip(out[..., 0] / 100.0, 0.0, 1.0)
    pivot = 0.48
    contrast = 1.0 + (float(params["contrast"]) - 1.0) * strength
    toned = pivot + (y - pivot) * contrast
    shadow = 1.0 - _smoothstep(0.10, 0.42, y)
    highlight = _smoothstep(0.60, 0.96, y)
    toned -= float(params["toe"]) * strength * shadow * (0.55 - y)
    toned -= float(params["shoulder"]) * strength * highlight * (y - 0.55)
    toned += float(params["lift"]) * strength * (1.0 - y) ** 2
    toned = np.clip(toned, 0.0, 1.0)

    # Preserve fine luminance texture while allowing a real stock-like curve.
    source_detail = lab[..., 0] - gaussian_filter(lab[..., 0], sigma=1.2)
    tone_base = gaussian_filter(toned * 100.0, sigma=1.2)
    tone_detail = toned * 100.0 - tone_base
    detail_mix = np.clip(0.58 - 0.10 * strength, 0.35, 0.58)
    out[..., 0] = np.clip(tone_base + tone_detail * (1.0 - detail_mix) + source_detail * detail_mix, 0.0, 100.0)
    return out


def apply_film_response_volume(
    source_rgb: np.ndarray,
    base_rgb: np.ndarray,
    *,
    style: str,
    strength: float = 1.0,
    output_margin: int = 4,
) -> tuple[np.ndarray, dict[str, float]]:
    """Apply a deterministic stock-specific tone/color response on safe-rich."""
    params = STYLE_PARAMS.get(style)
    if not params:
        raise ValueError(f"Unsupported film response style: {style}")

    strength = float(np.clip(strength, 0.0, 2.5))
    source_lab = rgb2lab(np.clip(source_rgb.astype(np.float32), 0.0, 1.0))
    base_lab = rgb2lab(np.clip(base_rgb.astype(np.float32), 0.0, 1.0))
    out_lab = _apply_tone(base_lab, params, strength)

    y = np.clip(base_lab[..., 0] / 100.0, 0.0, 1.0)[..., None]
    source_chroma = np.linalg.norm(source_lab[..., 1:3], axis=2, keepdims=True)
    base_chroma = np.linalg.norm(base_lab[..., 1:3], axis=2, keepdims=True)
    neutral = 1.0 - _smoothstep(4.0, 14.0, source_chroma)
    skin = _skin_mask(source_lab)[..., None]
    protect = np.clip(neutral * float(params["neutral_protect"]) + skin * float(params["skin_protect"]), 0.0, 0.92)

    shadow = (1.0 - _smoothstep(0.16, 0.48, y)) * (1.0 - protect * 0.65)
    mid = _smoothstep(0.20, 0.48, y) * (1.0 - _smoothstep(0.62, 0.90, y)) * (1.0 - protect * 0.55)
    highlight = _smoothstep(0.58, 0.94, y) * (1.0 - protect * 0.50)
    active = _smoothstep(6.0, 22.0, source_chroma) * (1.0 - protect * 0.70)

    sat_curve = (
        float(params["sat"])
        + (float(params["sat_shadow"]) - 1.0) * shadow
        + (float(params["sat_highlight"]) - 1.0) * highlight
    )
    sat_gain = 1.0 + (sat_curve - 1.0) * strength * active
    out_lab[..., 1:3] *= sat_gain

    out_lab[..., 1:3] += np.asarray(params["global_ab"], dtype=np.float32) * strength * (1.0 - protect)
    out_lab[..., 1:3] += np.asarray(params["shadow_ab"], dtype=np.float32) * strength * shadow
    out_lab[..., 1:3] += np.asarray(params["mid_ab"], dtype=np.float32) * strength * mid
    out_lab[..., 1:3] += np.asarray(params["highlight_ab"], dtype=np.float32) * strength * highlight

    hue = (np.degrees(np.arctan2(base_lab[..., 2], base_lab[..., 1])) + 360.0) % 360.0
    hue_active = _smoothstep(8.0, 24.0, base_chroma)[..., 0] * (1.0 - protect[..., 0] * 0.75)
    for center, width, vector in params["hue_pushes"]:
        mask = (_angular_mask(hue, center, width) * hue_active)[..., None]
        out_lab[..., 1:3] += np.asarray(vector, dtype=np.float32) * strength * mask

    out_lab[..., 1:3] = out_lab[..., 1:3] * (1.0 - protect * 0.28) + base_lab[..., 1:3] * (protect * 0.28)
    compressed = compress_to_srgb_gamut(source_lab, out_lab)
    result = lab_to_rgb_no_clip(compressed)
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        result = np.clip(result, low, high)
    metrics = {
        "neutral_protect_mean": float(neutral.mean()),
        "skin_protect_mean": float(skin.mean()),
        "active_chroma_mean": float(active.mean()),
        "shadow_mask_mean": float(shadow.mean()),
        "highlight_mask_mean": float(highlight.mean()),
    }
    return result.astype(np.float32), metrics
