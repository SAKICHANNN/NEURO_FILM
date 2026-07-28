"""Public full-frame RGB adapter for an explicit safe-Lab source context.

The historical CLI module is hash-pinned by frozen AO6 contracts. This adapter
keeps those bytes immutable while exposing its already-tested two-pass kernel
to newer explicit composition code.
"""

from __future__ import annotations

import numpy as np
from skimage.color import rgb2lab

from scripts.pipeline_color_baseline import (
    _style_transfer_rgb_with_context,
    _validate_style_rgb,
)
from src.color_engine.safe_lab import (
    SafeLabSourceContext,
    validate_safe_lab_source_context,
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
    )


__all__ = ["style_transfer_rgb_with_source_context"]
