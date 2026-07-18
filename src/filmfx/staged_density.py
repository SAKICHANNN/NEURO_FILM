"""Research-only staged executor for the fixed default density-halation graph."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .effects import _sigmoid, _smoothstep, _softplus, _srgb_to_linear, luminance
from .fast_blur import gaussian_filter_direct
from .global_resample import (
    ChunkInvariantRowStageMetadata,
    StagedGlobalField,
    build_chunk_invariant_global_stage,
    build_chunk_invariant_global_stage_from_rows,
    plan_chunk_invariant_global_resample,
    reconstruct_chunk_invariant_full,
    reconstruct_chunk_invariant_window,
)
from .gradient_window import coordinate_gradient_window
from .halation_dag import (
    DENSITY_FAMILY,
    available_halation_integration_capabilities,
    build_halation_resource_plan,
)
from .layers import FilmLayer
from .streaming_percentile import StreamingPercentileResult, exact_streaming_percentiles


STAGED_DENSITY_VERSION = "staged-density-halation-v1-defaults"
_SOURCE_PERCENTILE = 99.7
_SOURCE_ROW_HALO = 1
_LOCAL_ABS_SIGMA = 9.8
_NEAR_SIGMA = 2.4
_MID_SIGMA = 10.0
_LOCAL_MEAN_SIGMA = 28.0
_TAIL_SIGMA = 26.0
_GLARE_SIGMA = 70.0
_TILE_HALO = max(int(round(3.0 * _LOCAL_ABS_SIGMA)), int(round(3.0 * _MID_SIGMA)))
_DECLARED_WORKSPACE_SCALARS_PER_EXPANDED_PIXEL = 64


@dataclass(frozen=True)
class DensityGlobalContextMetadata:
    name: str
    sigma: float
    coarse_shape: tuple[int, ...]
    coarse_bytes: int
    row_stage: ChunkInvariantRowStageMetadata


@dataclass(frozen=True)
class StagedDensityExecutionMetadata:
    version: str
    source_shape: tuple[int, int, int]
    tile_size: int
    source_row_chunk: int
    coarse_row_chunk: int
    tile_count: int
    percentile: StreamingPercentileResult
    contexts: tuple[DensityGlobalContextMetadata, ...]
    input_bytes: int
    output_bytes: int
    global_context_bytes: int
    percentile_histogram_bytes: int
    source_read_calls: int
    total_source_read_bytes: int
    max_source_window_shape: tuple[int, int, int]
    max_source_window_bytes: int
    declared_max_tile_workspace_bytes: int
    persistent_derived_full_scalar_bytes: int
    scratch_disk_bytes: int
    static_plan_fingerprint: str


@dataclass
class _SourceReadAudit:
    source_shape: tuple[int, int, int]
    calls: int = 0
    total_bytes: int = 0
    max_bytes: int = 0
    max_shape: tuple[int, int, int] = (0, 0, 3)

    def record(self, shape: tuple[int, int, int]) -> None:
        if shape[0] >= self.source_shape[0]:
            raise ValueError("derived source requests must remain below full source height")
        value_bytes = math.prod(shape) * np.dtype(np.float32).itemsize
        self.calls += 1
        self.total_bytes += value_bytes
        if value_bytes > self.max_bytes:
            self.max_bytes = value_bytes
            self.max_shape = shape


def _validate_input(base_rgb, *, tile_size, source_row_chunk, coarse_row_chunk):
    if not isinstance(base_rgb, np.ndarray) or base_rgb.ndim != 3 or base_rgb.shape[2] != 3:
        raise ValueError("base_rgb must be an HxWx3 numpy array")
    if base_rgb.dtype != np.float32 or not np.isfinite(base_rgb).all():
        raise ValueError("base_rgb must contain finite float32 values")
    if base_rgb.shape[0] < 2 or base_rgb.shape[1] < 2:
        raise ValueError("base_rgb spatial dimensions must be at least 2")
    if float(base_rgb.min()) < 0.0 or float(base_rgb.max()) > 1.0:
        raise ValueError("base_rgb must be in [0, 1]")
    resolved = []
    for value, label in (
        (tile_size, "tile_size"),
        (source_row_chunk, "source_row_chunk"),
        (coarse_row_chunk, "coarse_row_chunk"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} must be a positive integer")
        resolved.append(value)
    tile, source_chunk, coarse_chunk = resolved
    if source_chunk >= base_rgb.shape[0]:
        raise ValueError("source_row_chunk must remain below source height")
    if tile + 2 * (_TILE_HALO + _SOURCE_ROW_HALO) >= base_rgb.shape[0]:
        raise ValueError("tile_size is too large to preserve bounded source-height requests")
    return base_rgb, tile, source_chunk, coarse_chunk


def _linear_luma(base: np.ndarray) -> np.ndarray:
    return np.maximum(luminance(_srgb_to_linear(base)), np.float32(1e-6))


def _source_raw_from_luma(y: np.ndarray) -> np.ndarray:
    log_e = np.log2(y / np.float32(0.18) + np.float32(1e-6))
    raw = _softplus((log_e - np.float32(2.2)) / np.float32(0.46))
    return np.power(raw, np.float32(1.35)).astype(np.float32, copy=False)


def _base_window(base, bounds, audit):
    y0, y1, x0, x1 = bounds
    if not (0 <= y0 < y1 <= base.shape[0] and 0 <= x0 < x1 <= base.shape[1]):
        raise ValueError("source window is outside base_rgb")
    if audit is not None:
        audit.record((y1 - y0, x1 - x0, 3))
    return base[y0:y1, x0:x1]


def _luma_window(base, bounds, audit):
    return _linear_luma(_base_window(base, bounds, audit))


def _source_raw_window(base, bounds, audit):
    return _source_raw_from_luma(_luma_window(base, bounds, audit))


def _percentile(base, row_chunk, audit):
    height, width = base.shape[:2]

    def factory():
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            yield _source_raw_window(base, (y0, y1, 0, width), audit)

    return exact_streaming_percentiles(
        factory, count=height * width, percentiles=(_SOURCE_PERCENTILE,)
    )


def _weighted_source_window(base, bounds, source_reference, audit):
    y0, y1, x0, x1 = bounds
    base_window = _base_window(base, bounds, audit)
    y = _linear_luma(base_window)

    def y_reader(read_y0, read_y1, read_x0, read_x1):
        return _luma_window(base, (read_y0, read_y1, read_x0, read_x1), audit)

    gradient_y, gradient_x, _ = coordinate_gradient_window(
        y_reader, base.shape[:2], y0=y0, y1=y1, x0=x0, x1=x1
    )
    edge = np.hypot(gradient_x, gradient_y)
    source = _source_raw_from_luma(y)
    source = np.clip(source / np.float32(max(source_reference, 1e-6)), 0.0, 2.5)
    maxc = base_window.max(axis=2)
    minc = base_window.min(axis=2)
    chroma = maxc - minc
    white_hot = _smoothstep(0.60, 0.94, maxc) * (
        1.0 - _smoothstep(0.20, 0.60, chroma)
    )
    color_hot = _smoothstep(0.64, 0.96, maxc) * _smoothstep(0.08, 0.55, chroma)
    specular = np.clip(0.50 + 0.65 * white_hot + 0.25 * color_hot, 0.0, 1.20)
    edge_confidence = 0.30 + 0.70 * _smoothstep(0.002, 0.045, edge)
    return (source * specular * edge_confidence).astype(np.float32), edge.astype(np.float32)


def _background_window(base, bounds, audit):
    return np.minimum(_luma_window(base, bounds, audit), np.float32(0.36))


def _global_stage_from_rows(base, *, name, sigma, coarse_row_chunk, reader):
    plan = plan_chunk_invariant_global_resample(base.shape[:2], sigma)
    stage, row_metadata = build_chunk_invariant_global_stage_from_rows(
        base.shape[:2], plan, coarse_row_chunk=coarse_row_chunk, reader=reader
    )
    return stage, DensityGlobalContextMetadata(
        name=name,
        sigma=sigma,
        coarse_shape=stage.coarse.shape,
        coarse_bytes=stage.coarse_bytes,
        row_stage=row_metadata,
    )


def _crop(array, expanded, core):
    ey0, _, ex0, _ = expanded
    y0, y1, x0, x1 = core
    return array[y0 - ey0 : y1 - ey0, x0 - ex0 : x1 - ex0]


def _fixed_layer(base, alpha, *, name):
    rgb = np.empty_like(base, dtype=np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 1.0, 0.96, 0.86
    return FilmLayer(name=name, mode="screen", rgb=rgb, alpha=alpha[..., None])


def execute_staged_density_halation_default(
    base_rgb: np.ndarray,
    *,
    tile_size: int = 256,
    source_row_chunk: int = 64,
    coarse_row_chunk: int = 7,
    name: str = "staged_density_halation",
) -> tuple[FilmLayer, StagedDensityExecutionMetadata]:
    """Execute the frozen default density graph without full derived scalar fields."""

    base, tile, source_chunk, coarse_chunk = _validate_input(
        base_rgb,
        tile_size=tile_size,
        source_row_chunk=source_row_chunk,
        coarse_row_chunk=coarse_row_chunk,
    )
    height, width, _ = base.shape
    audit = _SourceReadAudit(base.shape)
    percentile = _percentile(base, source_chunk, audit)
    source_reference = percentile.values[0]
    local_mean, local_metadata = _global_stage_from_rows(
        base,
        name="local_mean",
        sigma=_LOCAL_MEAN_SIGMA,
        coarse_row_chunk=coarse_chunk,
        reader=lambda y0, y1: _background_window(base, (y0, y1, 0, width), audit),
    )
    tail, tail_metadata = _global_stage_from_rows(
        base,
        name="tail",
        sigma=_TAIL_SIGMA,
        coarse_row_chunk=coarse_chunk,
        reader=lambda y0, y1: _weighted_source_window(
            base, (y0, y1, 0, width), source_reference, audit
        )[0],
    )
    glare, glare_metadata = _global_stage_from_rows(
        base,
        name="glare",
        sigma=_GLARE_SIGMA,
        coarse_row_chunk=coarse_chunk,
        reader=lambda y0, y1: _weighted_source_window(
            base, (y0, y1, 0, width), source_reference, audit
        )[0],
    )
    alpha = np.empty((height, width), dtype=np.float32)
    tile_count = 0
    max_expanded_pixels = 0
    for y0 in range(0, height, tile):
        y1 = min(height, y0 + tile)
        for x0 in range(0, width, tile):
            x1 = min(width, x0 + tile)
            core = (y0, y1, x0, x1)
            expanded = (
                max(0, y0 - _TILE_HALO),
                min(height, y1 + _TILE_HALO),
                max(0, x0 - _TILE_HALO),
                min(width, x1 + _TILE_HALO),
            )
            max_expanded_pixels = max(
                max_expanded_pixels,
                (expanded[1] - expanded[0]) * (expanded[3] - expanded[2]),
            )
            base_core = _base_window(base, core, audit)
            background_expanded = _background_window(base, expanded, audit)
            local_mean_expanded = reconstruct_chunk_invariant_window(
                local_mean,
                y0=expanded[0],
                y1=expanded[1],
                x0=expanded[2],
                x1=expanded[3],
            )
            local_abs_expanded = gaussian_filter_direct(
                np.abs(background_expanded - local_mean_expanded), _LOCAL_ABS_SIGMA
            )
            weighted_expanded, edge_expanded = _weighted_source_window(
                base, expanded, source_reference, audit
            )
            near_expanded = gaussian_filter_direct(weighted_expanded, _NEAR_SIGMA)
            mid_expanded = gaussian_filter_direct(weighted_expanded, _MID_SIGMA)
            local_mean_core = _crop(local_mean_expanded, expanded, core)
            local_abs_core = _crop(local_abs_expanded, expanded, core)
            near_core = _crop(near_expanded, expanded, core)
            mid_core = _crop(mid_expanded, expanded, core)
            edge_core = _crop(edge_expanded, expanded, core)
            tail_core = reconstruct_chunk_invariant_window(
                tail, y0=y0, y1=y1, x0=x0, x1=x1
            )
            glare_core = reconstruct_chunk_invariant_window(
                glare, y0=y0, y1=y1, x0=x0, x1=x1
            )
            dark = _sigmoid((0.20 - local_mean_core) * 1.25 * 8.0)
            contrast = _smoothstep(0.006, 0.14, local_abs_core + edge_core * 1.8)
            maxc = base_core.max(axis=2)
            minc = base_core.min(axis=2)
            chroma = maxc - minc
            r, g, b = base_core[..., 0], base_core[..., 1], base_core[..., 2]
            skin = (
                _smoothstep(0.16, 0.42, r)
                * _smoothstep(0.09, 0.34, g)
                * (1.0 - _smoothstep(0.02, 0.30, b - g))
                * _smoothstep(0.02, 0.20, r - b)
                * (1.0 - _smoothstep(0.42, 0.72, chroma))
            )
            visibility = dark * (0.40 + 0.60 * contrast) * (
                1.0 - np.clip(skin * 0.35, 0.0, 0.75)
            )
            density = visibility * (
                0.42 * near_core
                + 0.33 * mid_core
                + 0.25 * tail_core
                + glare_core * 0.20
            )
            alpha[y0:y1, x0:x1] = np.clip(
                (1.0 - np.exp(-density * 0.34)) * 0.85, 0.0, 0.26
            )
            tile_count += 1
    layer = _fixed_layer(base, alpha, name=name)
    contexts = (local_metadata, tail_metadata, glare_metadata)
    static_plan = build_halation_resource_plan(
        DENSITY_FAMILY,
        base.shape[:2],
        tile_size=tile,
        available_capabilities=available_halation_integration_capabilities(),
    )
    metadata = StagedDensityExecutionMetadata(
        version=STAGED_DENSITY_VERSION,
        source_shape=base.shape,
        tile_size=tile,
        source_row_chunk=source_chunk,
        coarse_row_chunk=coarse_chunk,
        tile_count=tile_count,
        percentile=percentile,
        contexts=contexts,
        input_bytes=int(base.nbytes),
        output_bytes=int(layer.rgb.nbytes + layer.alpha.nbytes),
        global_context_bytes=sum(context.coarse_bytes for context in contexts),
        percentile_histogram_bytes=percentile.histogram_bytes,
        source_read_calls=audit.calls,
        total_source_read_bytes=audit.total_bytes,
        max_source_window_shape=audit.max_shape,
        max_source_window_bytes=audit.max_bytes,
        declared_max_tile_workspace_bytes=(
            max_expanded_pixels
            * _DECLARED_WORKSPACE_SCALARS_PER_EXPANDED_PIXEL
            * np.dtype(np.float32).itemsize
        ),
        persistent_derived_full_scalar_bytes=0,
        scratch_disk_bytes=0,
        static_plan_fingerprint=static_plan.fingerprint,
    )
    return layer, metadata


def _v2_blur_materialized(field: np.ndarray, sigma: float) -> np.ndarray:
    plan = plan_chunk_invariant_global_resample(field.shape, sigma)
    if plan.coarse_shape == field.shape:
        return gaussian_filter_direct(field, sigma)
    return reconstruct_chunk_invariant_full(build_chunk_invariant_global_stage(field, plan))


def materialized_density_halation_v2_reference(
    base_rgb: np.ndarray,
    *,
    source_row_chunk: int = 64,
    name: str = "materialized_density_halation_v2_reference",
) -> FilmLayer:
    """Full-derived-array numerical reference for the staged default executor."""

    base, _, source_chunk, _ = _validate_input(
        base_rgb, tile_size=1, source_row_chunk=source_row_chunk, coarse_row_chunk=1
    )
    y = _linear_luma(base)
    source_raw = _source_raw_from_luma(y)

    def factory():
        for y0 in range(0, base.shape[0], source_chunk):
            yield source_raw[y0 : min(base.shape[0], y0 + source_chunk)]

    reference = exact_streaming_percentiles(
        factory,
        count=base.shape[0] * base.shape[1],
        percentiles=(_SOURCE_PERCENTILE,),
    ).values[0]
    source = np.clip(source_raw / np.float32(max(reference, 1e-6)), 0.0, 2.5)
    maxc = base.max(axis=2)
    minc = base.min(axis=2)
    chroma = maxc - minc
    white_hot = _smoothstep(0.60, 0.94, maxc) * (
        1.0 - _smoothstep(0.20, 0.60, chroma)
    )
    color_hot = _smoothstep(0.64, 0.96, maxc) * _smoothstep(0.08, 0.55, chroma)
    specular = np.clip(0.50 + 0.65 * white_hot + 0.25 * color_hot, 0.0, 1.20)
    gradient_y, gradient_x = np.gradient(y)
    edge = np.hypot(gradient_x, gradient_y)
    weighted = source * specular * (0.30 + 0.70 * _smoothstep(0.002, 0.045, edge))
    background = np.minimum(y, 0.36)
    local_mean = _v2_blur_materialized(background, _LOCAL_MEAN_SIGMA)
    local_abs = gaussian_filter_direct(np.abs(background - local_mean), _LOCAL_ABS_SIGMA)
    dark = _sigmoid((0.20 - local_mean) * 1.25 * 8.0)
    contrast = _smoothstep(0.006, 0.14, local_abs + edge * 1.8)
    r, g, b = base[..., 0], base[..., 1], base[..., 2]
    skin = (
        _smoothstep(0.16, 0.42, r)
        * _smoothstep(0.09, 0.34, g)
        * (1.0 - _smoothstep(0.02, 0.30, b - g))
        * _smoothstep(0.02, 0.20, r - b)
        * (1.0 - _smoothstep(0.42, 0.72, chroma))
    )
    visibility = dark * (0.40 + 0.60 * contrast) * (
        1.0 - np.clip(skin * 0.35, 0.0, 0.75)
    )
    near = gaussian_filter_direct(weighted, _NEAR_SIGMA)
    mid = gaussian_filter_direct(weighted, _MID_SIGMA)
    tail = _v2_blur_materialized(weighted, _TAIL_SIGMA)
    glare = _v2_blur_materialized(weighted, _GLARE_SIGMA)
    density = visibility * (0.42 * near + 0.33 * mid + 0.25 * tail + glare * 0.20)
    alpha = np.clip((1.0 - np.exp(-density * 0.34)) * 0.85, 0.0, 0.26)
    return _fixed_layer(base, alpha.astype(np.float32), name=name)
