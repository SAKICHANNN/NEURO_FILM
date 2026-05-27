"""Layer schema and safety metrics for film effects."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class FilmLayer:
    name: str
    mode: str
    rgb: np.ndarray | None = None
    alpha: np.ndarray | None = None
    residual: np.ndarray | None = None
    enabled: bool = True


def validate_layer(layer: FilmLayer, shape: tuple[int, int, int]) -> None:
    height, width, channels = shape
    if channels != 3:
        raise ValueError(f"base image must have 3 channels, got {shape}")
    if layer.mode in {"alpha", "screen", "additive", "soft_light"}:
        if layer.rgb is None or layer.alpha is None:
            raise ValueError(f"{layer.mode} layer requires rgb and alpha")
        if layer.rgb.shape != shape:
            raise ValueError(f"layer rgb shape {layer.rgb.shape} does not match {shape}")
        if layer.alpha.shape not in {(height, width), (height, width, 1)}:
            raise ValueError(f"layer alpha shape {layer.alpha.shape} does not match {(height, width)}")
    elif layer.mode == "residual":
        if layer.residual is None:
            raise ValueError("residual layer requires residual")
        if layer.residual.shape != shape:
            raise ValueError(f"layer residual shape {layer.residual.shape} does not match {shape}")
    else:
        raise ValueError(f"Unsupported layer mode: {layer.mode}")


def layer_metrics(layer: FilmLayer) -> dict:
    metrics = {"name": layer.name, "mode": layer.mode, "enabled": layer.enabled}
    if layer.alpha is not None:
        alpha = np.asarray(layer.alpha, dtype=np.float32)
        metrics.update(
            {
                "alpha_min": float(alpha.min()),
                "alpha_max": float(alpha.max()),
                "alpha_mean": float(alpha.mean()),
                "affected_percent": float((alpha > 1e-4).mean() * 100.0),
            }
        )
    if layer.residual is not None:
        residual = np.asarray(layer.residual, dtype=np.float32)
        metrics.update(
            {
                "residual_abs_max": float(np.abs(residual).max()),
                "residual_abs_mean": float(np.abs(residual).mean()),
            }
        )
    return metrics
