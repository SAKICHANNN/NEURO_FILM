"""Public full-frame RGB adapter for an explicit safe-Lab source context.

The historical CLI module is hash-pinned by frozen AO6 contracts. This adapter
keeps those bytes immutable while exposing its already-tested two-pass kernel
to newer explicit composition code.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    _style_transfer_rgb_with_context,
    _validate_style_rgb,
    build_safe_lab_source_context,
)
from src.color_engine.safe_lab import (
    SafeLabSourceContext,
    safe_lab_context_from_lab,
    validate_safe_lab_source_context,
)
from src.inference.tiled_render import (
    TiledExecutionMetadata,
    TileWindow,
    execute_tiled_local_operator,
)


def style_transfer_rgb_with_source_context(
    rgb: np.ndarray,
    stats: dict,
    style: str,
    strength: float,
    luma_strength: float,
    grain: float,
    seed: int,
    gamut_safe: bool,
    gamut_mode: str | None = None,
    tone_rolloff: float = 0.0,
    shadow_floor_l: float = 1.0,
    highlight_ceiling_l: float = 99.0,
    preserve_luma_detail_strength: float = 0.0,
    chroma_curve_strength: float = 0.0,
    output_margin: int = 0,
    guardrails: dict | None = None,
    neutral_protect: float | None = None,
    skin_protect: float | None = None,
    max_chroma_gain: float | None = None,
    max_chroma_boost: float | None = None,
    max_chroma_absolute: float | None = None,
    dither: float | None = None,
    *,
    source_context: SafeLabSourceContext,
    gamut_workers: int = 1,
) -> np.ndarray:
    """Apply safe-Lab using a designated same-frame reduction."""

    value = _validate_style_rgb(rgb)
    validate_safe_lab_source_context(source_context)
    if tuple(int(size) for size in value.shape) != source_context.source_shape:
        raise ValueError("source_context must describe the same full-frame shape")
    lab = rgb2lab(value)
    return _style_transfer_rgb_with_context(
        value,
        stats,
        style,
        strength,
        luma_strength,
        grain,
        seed,
        gamut_safe,
        gamut_mode=gamut_mode,
        tone_rolloff=tone_rolloff,
        shadow_floor_l=shadow_floor_l,
        highlight_ceiling_l=highlight_ceiling_l,
        preserve_luma_detail_strength=preserve_luma_detail_strength,
        chroma_curve_strength=chroma_curve_strength,
        output_margin=output_margin,
        guardrails=guardrails,
        neutral_protect=neutral_protect,
        skin_protect=skin_protect,
        max_chroma_gain=max_chroma_gain,
        max_chroma_boost=max_chroma_boost,
        max_chroma_absolute=max_chroma_absolute,
        dither=dither,
        source_context=source_context,
        precomputed_lab=lab,
        gamut_workers=gamut_workers,
    )


def build_safe_lab_source_context_bounded(
    rgb: np.ndarray,
    *,
    scratch_directory: Path,
    row_chunk: int = 128,
) -> SafeLabSourceContext:
    """Build the exact legacy context with bounded RGB-to-Lab temporaries.

    The complete Lab array is file-backed so the unchanged legacy NumPy
    reduction retains its dtype and reduction order without keeping the
    conversion result resident in process memory.
    """

    if isinstance(row_chunk, bool) or not isinstance(row_chunk, int) or row_chunk < 1:
        raise ValueError("row_chunk must be a positive integer")
    directory = Path(scratch_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ValueError("scratch_directory must be a directory")

    value = _validate_style_rgb(rgb)
    descriptor, raw_path = tempfile.mkstemp(
        prefix="safe_lab_context_",
        suffix=".f32",
        dir=directory,
    )
    os.close(descriptor)
    path = Path(raw_path)
    mapped: np.memmap | None = None
    try:
        mapped = np.memmap(path, dtype=np.float32, mode="w+", shape=value.shape)
        for y0 in range(0, value.shape[0], row_chunk):
            y1 = min(y0 + row_chunk, value.shape[0])
            mapped[y0:y1] = rgb2lab(value[y0:y1])
        mapped.flush()
        return safe_lab_context_from_lab(
            mapped,
            tuple(int(size) for size in value.shape),
        )
    finally:
        if mapped is not None:
            mapped.flush()
            del mapped
        path.unlink(missing_ok=True)


def style_transfer_rgb_tiled_with_source_context(
    rgb: np.ndarray,
    stats: dict,
    style: str,
    strength: float,
    luma_strength: float,
    grain: float,
    seed: int,
    gamut_safe: bool,
    gamut_mode: str | None = None,
    tone_rolloff: float = 0.0,
    shadow_floor_l: float = 1.0,
    highlight_ceiling_l: float = 99.0,
    preserve_luma_detail_strength: float = 0.0,
    chroma_curve_strength: float = 0.0,
    output_margin: int = 0,
    guardrails: dict | None = None,
    neutral_protect: float | None = None,
    skin_protect: float | None = None,
    max_chroma_gain: float | None = None,
    max_chroma_boost: float | None = None,
    max_chroma_absolute: float | None = None,
    dither: float | None = None,
    *,
    source_context: SafeLabSourceContext,
    tile_size: int,
    workers: int = 1,
) -> tuple[np.ndarray, TiledExecutionMetadata]:
    """Apply exact legacy tiling while reusing a same-frame reduction."""

    if grain > 0:
        raise ValueError("tiled safe-Lab does not support legacy colour-core grain")
    value = _validate_style_rgb(rgb)
    validate_safe_lab_source_context(source_context)
    if tuple(int(size) for size in value.shape) != source_context.source_shape:
        raise ValueError("source_context must describe the same full-frame shape")
    halo = 5 if preserve_luma_detail_strength > 0 else 0

    def render_tile(tile: np.ndarray, window: TileWindow) -> np.ndarray:
        return _style_transfer_rgb_with_context(
            tile,
            stats,
            style,
            strength,
            luma_strength,
            grain,
            seed,
            gamut_safe,
            gamut_mode=gamut_mode,
            tone_rolloff=tone_rolloff,
            shadow_floor_l=shadow_floor_l,
            highlight_ceiling_l=highlight_ceiling_l,
            preserve_luma_detail_strength=preserve_luma_detail_strength,
            chroma_curve_strength=chroma_curve_strength,
            output_margin=output_margin,
            guardrails=guardrails,
            neutral_protect=neutral_protect,
            skin_protect=skin_protect,
            max_chroma_gain=max_chroma_gain,
            max_chroma_boost=max_chroma_boost,
            max_chroma_absolute=max_chroma_absolute,
            dither=dither,
            source_context=source_context,
            dither_window=window,
        )

    return execute_tiled_local_operator(
        value,
        render_tile,
        tile_size=tile_size,
        halo=halo,
        workers=workers,
    )


__all__ = [
    "build_safe_lab_source_context",
    "build_safe_lab_source_context_bounded",
    "style_transfer_rgb_tiled_with_source_context",
    "style_transfer_rgb_with_source_context",
]
