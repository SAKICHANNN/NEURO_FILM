"""Deterministic compositor for film-effect layers."""

from __future__ import annotations

import numpy as np

from .layers import FilmLayer, validate_layer


def _alpha(layer: FilmLayer) -> np.ndarray:
    alpha = np.asarray(layer.alpha, dtype=np.float32)
    if alpha.ndim == 2:
        alpha = alpha[..., None]
    return np.clip(alpha, 0.0, 1.0)


def _soft_light(base: np.ndarray, blend: np.ndarray) -> np.ndarray:
    low = 2.0 * base * blend + base * base * (1.0 - 2.0 * blend)
    high = 2.0 * base * (1.0 - blend) + np.sqrt(np.clip(base, 0.0, 1.0)) * (2.0 * blend - 1.0)
    return np.where(blend <= 0.5, low, high)


def composite_layers(base_rgb: np.ndarray, layers: list[FilmLayer], *, output_margin: int = 0) -> np.ndarray:
    """Composite film-effect layers over ``base_rgb`` in ``[0, 1]``."""
    out = np.asarray(base_rgb, dtype=np.float32).copy()
    if out.ndim != 3 or out.shape[2] != 3:
        raise ValueError(f"base_rgb must be [H, W, 3], got {out.shape}")
    out = np.clip(out, 0.0, 1.0)

    for layer in layers:
        if not layer.enabled:
            continue
        validate_layer(layer, out.shape)
        if layer.mode == "alpha":
            alpha = _alpha(layer)
            out = out * (1.0 - alpha) + np.clip(layer.rgb, 0.0, 1.0) * alpha
        elif layer.mode == "screen":
            alpha = _alpha(layer)
            screened = 1.0 - (1.0 - out) * (1.0 - np.clip(layer.rgb, 0.0, 1.0))
            out = out * (1.0 - alpha) + screened * alpha
        elif layer.mode == "additive":
            alpha = _alpha(layer)
            out = out + np.clip(layer.rgb, 0.0, 1.0) * alpha
        elif layer.mode == "soft_light":
            alpha = _alpha(layer)
            blended = _soft_light(out, np.clip(layer.rgb, 0.0, 1.0))
            out = out * (1.0 - alpha) + blended * alpha
        elif layer.mode == "residual":
            out = out + layer.residual
        out = np.clip(out, 0.0, 1.0)

    if output_margin > 0:
        low = output_margin / 255.0
        high = 1.0 - low
        out = np.clip(out, low, high)
    return out.astype(np.float32)
