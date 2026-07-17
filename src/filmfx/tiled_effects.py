"""Experimental tiled adapters for explicitly finite-support film effects."""

from __future__ import annotations

import math

import numpy as np

from src.inference.tiled_render import TiledExecutionMetadata, execute_tiled_local_operator

from .compositor import composite_layers
from .effects import halation_layer


def _finite_number(value: object, label: str, low: float, high: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not low <= result <= high:
        raise ValueError(f"{label} must be in [{low}, {high}]")
    return result


def _integer(value: object, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in [{low}, {high}]")
    return value


def simple_halation_required_halo(*, min_radius: float = 1.1, max_radius: float = 10.0) -> int:
    """Return gradient support plus the current direct Gaussian kernel radius."""

    minimum = _finite_number(min_radius, "min_radius", 0.4, 32.0)
    maximum = _finite_number(max_radius, "max_radius", 0.0, 32.0)
    resolved_minimum = max(0.4, minimum)
    resolved_maximum = max(resolved_minimum + 0.1, maximum)
    gaussian_radius = int(round(3.0 * resolved_maximum))
    if gaussian_radius > 32:
        raise ValueError("simple halation radius enters the unsupported downsample branch")
    return 1 + gaussian_radius


def composite_simple_halation_tiled(
    base_rgb: np.ndarray,
    *,
    tile_size: int,
    strength: float = 0.16,
    threshold: float = 0.78,
    edge_threshold: float = 0.08,
    min_radius: float = 1.1,
    max_radius: float = 10.0,
    radius_gamma: float = 1.35,
    scale_count: int = 6,
    output_margin: int = 0,
) -> tuple[np.ndarray, TiledExecutionMetadata]:
    """Composite the existing simple-halation layer through the U1.6A tiler."""

    resolved_strength = _finite_number(strength, "strength", 0.0, 1.0)
    resolved_threshold = _finite_number(threshold, "threshold", 0.0, 1.0)
    resolved_edge_threshold = _finite_number(edge_threshold, "edge_threshold", 1e-6, 1.0)
    resolved_min_radius = _finite_number(min_radius, "min_radius", 0.4, 32.0)
    resolved_max_radius = _finite_number(max_radius, "max_radius", 0.0, 32.0)
    resolved_radius_gamma = _finite_number(radius_gamma, "radius_gamma", 0.05, 8.0)
    resolved_scale_count = _integer(scale_count, "scale_count", 3, 32)
    resolved_output_margin = _integer(output_margin, "output_margin", 0, 32)
    halo = simple_halation_required_halo(
        min_radius=resolved_min_radius,
        max_radius=resolved_max_radius,
    )

    def render_tile(tile: np.ndarray, _) -> np.ndarray:
        layer = halation_layer(
            tile,
            strength=resolved_strength,
            threshold=resolved_threshold,
            edge_threshold=resolved_edge_threshold,
            min_radius=resolved_min_radius,
            max_radius=resolved_max_radius,
            radius_gamma=resolved_radius_gamma,
            scale_count=resolved_scale_count,
        )
        return composite_layers(tile, [layer], output_margin=resolved_output_margin)

    return execute_tiled_local_operator(
        base_rgb,
        render_tile,
        tile_size=tile_size,
        halo=halo,
    )
