"""Numpy distilled SepLUT runtime for Neural Film LUT V2."""

from __future__ import annotations

import numpy as np


DISTILLED_STYLE_NAMES = ["ektar_100", "portra_400", "portra_800", "velvia_50", "vision3_250d", "vision3_500t"]


def image_features(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    luma = rgb @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    return np.asarray([1.0, float(luma.mean()), float(luma.std()), float(chroma.mean())], dtype=np.float32)


def apply_1d_lut_np(rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    size = lut.shape[-1]
    coords = np.clip(rgb, 0.0, 1.0) * (size - 1)
    lower = np.floor(coords).astype(np.int32)
    upper = np.clip(lower + 1, 0, size - 1)
    frac = coords - lower
    out = np.empty_like(rgb, dtype=np.float32)
    for channel in range(3):
        lo = lut[channel, lower[..., channel]]
        hi = lut[channel, upper[..., channel]]
        out[..., channel] = lo * (1.0 - frac[..., channel]) + hi * frac[..., channel]
    return out


def apply_3d_residual_np(rgb: np.ndarray, residual: np.ndarray) -> np.ndarray:
    size = residual.shape[0]
    original_shape = rgb.shape
    flat_rgb = np.clip(rgb.reshape(-1, 3), 0.0, 1.0)
    coords = flat_rgb * (size - 1)
    lower = np.floor(coords).astype(np.int32)
    upper = np.clip(lower + 1, 0, size - 1)
    frac = coords - lower
    r0, g0, b0 = lower[:, 0], lower[:, 1], lower[:, 2]
    r1, g1, b1 = upper[:, 0], upper[:, 1], upper[:, 2]
    wr, wg, wb = frac[:, 0:1], frac[:, 1:2], frac[:, 2:3]
    flat = residual.reshape(size * size * size, 3)

    def gather(ri: np.ndarray, gi: np.ndarray, bi: np.ndarray) -> np.ndarray:
        return flat[ri * size * size + gi * size + bi]

    c000 = gather(r0, g0, b0)
    c001 = gather(r0, g0, b1)
    c010 = gather(r0, g1, b0)
    c011 = gather(r0, g1, b1)
    c100 = gather(r1, g0, b0)
    c101 = gather(r1, g0, b1)
    c110 = gather(r1, g1, b0)
    c111 = gather(r1, g1, b1)
    c00 = c000 * (1.0 - wb) + c001 * wb
    c01 = c010 * (1.0 - wb) + c011 * wb
    c10 = c100 * (1.0 - wb) + c101 * wb
    c11 = c110 * (1.0 - wb) + c111 * wb
    c0 = c00 * (1.0 - wg) + c01 * wg
    c1 = c10 * (1.0 - wg) + c11 * wg
    return (c0 * (1.0 - wr) + c1 * wr).reshape(original_shape)


def apply_distilled_seplut(
    rgb: np.ndarray,
    *,
    style_index: int,
    lut1d: np.ndarray,
    residual3d: np.ndarray,
    gate_weights: np.ndarray,
    strength: float,
    output_margin: int = 4,
) -> tuple[np.ndarray, float]:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    after_1d = apply_1d_lut_np(rgb, lut1d[style_index])
    base = np.clip(after_1d + apply_3d_residual_np(after_1d, residual3d[style_index]), 0.0, 1.0)
    gate = float(np.clip(image_features(rgb) @ gate_weights[style_index], 0.45, 1.65))
    out = rgb + (base - rgb) * float(strength) * gate
    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out = np.clip(out, low, high)
    return out.astype(np.float32), gate
