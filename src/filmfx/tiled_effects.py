"""Experimental tiled adapters for explicitly finite-support film effects."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.inference.tiled_render import TileWindow, TiledExecutionMetadata, execute_tiled_local_operator

from .compositor import composite_layers
from .effects import halation_layer
from .layers import FilmLayer


@dataclass(frozen=True)
class DustScratchContext:
    """Compact deterministic geometry for the legacy dust/scratch effect."""

    source_shape: tuple[int, int, int]
    strength: float
    seed: int
    speck_rects: np.ndarray
    speck_alpha: np.ndarray
    scratch_rects: np.ndarray
    scratch_alpha: np.ndarray

    @property
    def event_bytes(self) -> int:
        return int(
            self.speck_rects.nbytes
            + self.speck_alpha.nbytes
            + self.scratch_rects.nbytes
            + self.scratch_alpha.nbytes
        )


@dataclass(frozen=True)
class DustScratchExecutionMetadata:
    """Tiled topology plus sparse-context bounds for dust/scratch execution."""

    tiled: TiledExecutionMetadata
    speck_count: int
    scratch_count: int
    context_bytes: int


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


def _shape3(value: object, label: str = "shape") -> tuple[int, int, int]:
    if not isinstance(value, tuple) or len(value) != 3:
        raise ValueError(f"{label} must be an (H, W, 3) tuple")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"{label} dimensions must be positive integers")
    if value[2] != 3:
        raise ValueError(f"{label} must have three channels")
    return value


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result


def build_dust_scratch_context(
    shape: tuple[int, int, int],
    *,
    strength: float = 0.08,
    seed: int = 7,
) -> DustScratchContext:
    """Replay the legacy PCG64 event sequence without allocating a dense layer."""

    source_shape = _shape3(shape)
    resolved_strength = _finite_number(strength, "strength", 0.0, 1.0)
    resolved_seed = _integer(seed, "seed", 0, 2**63 - 1)
    height, width, _ = source_shape
    rng = np.random.default_rng(resolved_seed)

    speck_count = max(1, int(height * width * 0.00018 * resolved_strength * 10.0))
    speck_rects = np.empty((speck_count, 4), dtype=np.int32)
    speck_alpha = np.empty((speck_count,), dtype=np.float32)
    for index in range(speck_count):
        y = int(rng.integers(0, height))
        x = int(rng.integers(0, width))
        radius = int(rng.integers(1, 3))
        speck_rects[index] = (
            max(0, y - radius),
            min(height, y + radius + 1),
            max(0, x - radius),
            min(width, x + radius + 1),
        )
        speck_alpha[index] = np.float32(rng.uniform(0.08, 0.20) * resolved_strength)

    scratch_count = max(0, int(width * 0.012 * resolved_strength))
    scratch_rects = np.empty((scratch_count, 4), dtype=np.int32)
    scratch_alpha = np.empty((scratch_count,), dtype=np.float32)
    for index in range(scratch_count):
        x = int(rng.integers(0, width))
        y0 = int(rng.integers(0, max(1, height // 3)))
        length = int(rng.integers(max(4, height // 6), max(5, height)))
        scratch_rects[index] = (
            y0,
            min(height, y0 + length),
            max(0, x - 1),
            min(width, x + 1),
        )
        scratch_alpha[index] = np.float32(rng.uniform(0.03, 0.10) * resolved_strength)

    return DustScratchContext(
        source_shape=source_shape,
        strength=resolved_strength,
        seed=resolved_seed,
        speck_rects=_readonly(speck_rects),
        speck_alpha=_readonly(speck_alpha),
        scratch_rects=_readonly(scratch_rects),
        scratch_alpha=_readonly(scratch_alpha),
    )


def _validate_dust_scratch_context(context: DustScratchContext) -> None:
    if not isinstance(context, DustScratchContext):
        raise ValueError("context must be DustScratchContext")
    _shape3(context.source_shape, "context.source_shape")
    _finite_number(context.strength, "context.strength", 0.0, 1.0)
    _integer(context.seed, "context.seed", 0, 2**63 - 1)
    pairs = (
        (context.speck_rects, context.speck_alpha, "speck"),
        (context.scratch_rects, context.scratch_alpha, "scratch"),
    )
    height, width, _ = context.source_shape
    for rects, alpha, label in pairs:
        if (
            not isinstance(rects, np.ndarray)
            or rects.dtype != np.int32
            or rects.ndim != 2
            or rects.shape[1:] != (4,)
        ):
            raise ValueError(f"context.{label}_rects must be int32 [N, 4]")
        if (
            not isinstance(alpha, np.ndarray)
            or alpha.dtype != np.float32
            or alpha.shape != (rects.shape[0],)
            or not np.isfinite(alpha).all()
        ):
            raise ValueError(f"context.{label}_alpha must be finite float32 [N]")
        if rects.size:
            if (
                (rects[:, 0] < 0).any()
                or (rects[:, 1] > height).any()
                or (rects[:, 2] < 0).any()
                or (rects[:, 3] > width).any()
                or (rects[:, 0] >= rects[:, 1]).any()
                or (rects[:, 2] >= rects[:, 3]).any()
            ):
                raise ValueError(f"context.{label}_rects contain invalid bounds")
        if (alpha < 0.0).any() or (alpha > 0.25).any():
            raise ValueError(f"context.{label}_alpha is outside [0, 0.25]")
        if rects.flags.writeable or alpha.flags.writeable:
            raise ValueError(f"context.{label} arrays must be read-only")


def dust_scratch_alpha_window(
    context: DustScratchContext,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Render one exact global-coordinate alpha window from sparse events."""

    _validate_dust_scratch_context(context)
    return _dust_scratch_alpha_window_unchecked(context, y0=y0, y1=y1, x0=x0, x1=x1)


def _dust_scratch_alpha_window_unchecked(
    context: DustScratchContext,
    *,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
) -> np.ndarray:
    """Render a window after the caller has validated the immutable context."""

    height, width, _ = context.source_shape
    y0 = _integer(y0, "y0", 0, height - 1)
    y1 = _integer(y1, "y1", 1, height)
    x0 = _integer(x0, "x0", 0, width - 1)
    x1 = _integer(x1, "x1", 1, width)
    if y0 >= y1 or x0 >= x1:
        raise ValueError("window bounds must be ordered")

    alpha = np.zeros((y1 - y0, x1 - x0, 1), dtype=np.float32)
    for rects, values in (
        (context.speck_rects, context.speck_alpha),
        (context.scratch_rects, context.scratch_alpha),
    ):
        intersects = (
            (rects[:, 0] < y1)
            & (rects[:, 1] > y0)
            & (rects[:, 2] < x1)
            & (rects[:, 3] > x0)
        )
        for rect, value in zip(rects[intersects], values[intersects], strict=True):
            local_y0 = max(int(rect[0]), y0) - y0
            local_y1 = min(int(rect[1]), y1) - y0
            local_x0 = max(int(rect[2]), x0) - x0
            local_x1 = min(int(rect[3]), x1) - x0
            region = alpha[local_y0:local_y1, local_x0:local_x1, 0]
            alpha[local_y0:local_y1, local_x0:local_x1, 0] = np.maximum(region, value)
    return np.clip(alpha, 0.0, 0.25).astype(np.float32, copy=False)


def composite_dust_scratch_tiled(
    base_rgb: np.ndarray,
    *,
    tile_size: int,
    strength: float = 0.08,
    seed: int = 7,
    output_margin: int = 0,
    context: DustScratchContext | None = None,
) -> tuple[np.ndarray, DustScratchExecutionMetadata]:
    """Composite the legacy dust/scratch field from an exact sparse context."""

    if not isinstance(base_rgb, np.ndarray) or base_rgb.ndim != 3:
        raise ValueError("base_rgb must be an HWC numpy array")
    source_shape = _shape3(tuple(int(item) for item in base_rgb.shape), "base_rgb.shape")
    if base_rgb.dtype != np.float32 or not np.isfinite(base_rgb).all():
        raise ValueError("base_rgb must be a finite float32 array")
    resolved_tile_size = _integer(tile_size, "tile_size", 1, 2**31 - 1)
    resolved_strength = _finite_number(strength, "strength", 0.0, 1.0)
    resolved_seed = _integer(seed, "seed", 0, 2**63 - 1)
    resolved_output_margin = _integer(output_margin, "output_margin", 0, 32)
    if context is None:
        context = build_dust_scratch_context(
            source_shape,
            strength=resolved_strength,
            seed=resolved_seed,
        )
    _validate_dust_scratch_context(context)
    if (
        context.source_shape != source_shape
        or context.strength != resolved_strength
        or context.seed != resolved_seed
    ):
        raise ValueError("context does not match base shape, strength and seed")

    def render_tile(tile: np.ndarray, window: TileWindow) -> np.ndarray:
        alpha = _dust_scratch_alpha_window_unchecked(
            context,
            y0=window.expanded_y0,
            y1=window.expanded_y1,
            x0=window.expanded_x0,
            x1=window.expanded_x1,
        )
        layer = FilmLayer(
            name="dust_scratch",
            mode="alpha",
            rgb=np.ones(tile.shape, dtype=np.float32),
            alpha=alpha,
        )
        return composite_layers(tile, [layer], output_margin=resolved_output_margin)

    rendered, tiled = execute_tiled_local_operator(
        base_rgb,
        render_tile,
        tile_size=resolved_tile_size,
        halo=0,
    )
    return rendered, DustScratchExecutionMetadata(
        tiled=tiled,
        speck_count=int(context.speck_rects.shape[0]),
        scratch_count=int(context.scratch_rects.shape[0]),
        context_bytes=context.event_bytes,
    )


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
