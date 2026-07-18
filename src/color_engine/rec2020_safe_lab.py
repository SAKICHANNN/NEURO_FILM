"""Isolated colour-only safe-Lab adapter for display-linear Rec.2020."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from src.preprocess.types import DecodeWarning, WorkingImage

from .gamut import compress_chroma_to_working_gamut, compress_source_to_working_gamut
from .lab import lab_to_linear_rgb, linear_rgb_to_lab
from .safe_lab import apply_safe_lab_transform, safe_lab_context_from_lab


REC2020_SAFE_LAB_STYLES = frozenset(
    {
        "ektar_100",
        "portra_400",
        "portra_800",
        "velvia_50",
        "vision3_250d",
        "vision3_500t",
    }
)
REC2020_GAMUT_TOLERANCE = 2e-6


def apply_rec2020_safe_lab(
    working: WorkingImage,
    *,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    style: str,
    strength: float,
    luma_strength: float,
    gamut_mode: str,
    tone_rolloff: float = 0.0,
    shadow_floor_l: float = 1.0,
    highlight_ceiling_l: float = 99.0,
    preserve_luma_detail_strength: float = 0.0,
    chroma_curve_strength: float = 0.0,
    neutral_protect: float = 0.0,
    skin_protect: float = 0.0,
    max_chroma_gain: float | None = None,
    max_chroma_boost: float | None = None,
    max_chroma_absolute: float | None = None,
) -> WorkingImage:
    """Apply the research-only safe-Lab look without leaving Rec.2020."""

    if not isinstance(working, WorkingImage):
        raise TypeError("working must be a WorkingImage")
    if working.working_space != "linear_rec2020":
        raise ValueError("Rec.2020 safe-Lab requires linear_rec2020 working_space")
    if working.transfer_state != "display_linear":
        raise ValueError("Rec.2020 safe-Lab requires display_linear transfer_state")
    if working.pixels.shape[0] == 0 or working.pixels.shape[1] == 0:
        raise ValueError("Rec.2020 safe-Lab requires non-empty pixels")
    if (
        np.any(working.pixels < -REC2020_GAMUT_TOLERANCE)
        or np.any(working.pixels > 1.0 + REC2020_GAMUT_TOLERANCE)
    ):
        raise ValueError("Rec.2020 safe-Lab input is outside the working gamut")
    if style not in REC2020_SAFE_LAB_STYLES:
        raise ValueError("style is not enabled for the Rec.2020 colour-only adapter")
    if gamut_mode not in {"source", "chroma"}:
        raise ValueError("gamut_mode must be source or chroma")

    source_lab = linear_rgb_to_lab(working.pixels, working_space="linear_rec2020")
    context = safe_lab_context_from_lab(source_lab)
    styled_lab = apply_safe_lab_transform(
        source_lab,
        source_context=context,
        destination_mean=destination_mean,
        destination_std=destination_std,
        style=style,
        strength=strength,
        luma_strength=luma_strength,
        tone_rolloff=tone_rolloff,
        shadow_floor_l=shadow_floor_l,
        highlight_ceiling_l=highlight_ceiling_l,
        preserve_luma_detail_strength=preserve_luma_detail_strength,
        chroma_curve_strength=chroma_curve_strength,
        neutral_protect=neutral_protect,
        skin_protect=skin_protect,
        max_chroma_gain=max_chroma_gain,
        max_chroma_boost=max_chroma_boost,
        max_chroma_absolute=max_chroma_absolute,
    )
    if gamut_mode == "source":
        output_lab = compress_source_to_working_gamut(
            source_lab,
            styled_lab,
            working_space="linear_rec2020",
        )
    else:
        output_lab = compress_chroma_to_working_gamut(
            styled_lab,
            working_space="linear_rec2020",
        )
    pixels = lab_to_linear_rgb(output_lab, working_space="linear_rec2020")
    if (
        np.any(pixels < -REC2020_GAMUT_TOLERANCE)
        or np.any(pixels > 1.0 + REC2020_GAMUT_TOLERANCE)
        or not np.isfinite(pixels).all()
    ):
        raise ValueError("Rec.2020 safe-Lab output violates the working gamut")

    warnings = list(working.warnings)
    warnings.append(
        DecodeWarning(
            "rec2020_safe_lab_research",
            "Applied isolated colour-only safe-Lab research adapter; no effects, HDR, calibration, or production integration.",
        )
    )
    return WorkingImage(
        pixels=np.asarray(np.clip(pixels, 0.0, 1.0), dtype=np.float32),
        working_space=working.working_space,
        transfer_state=working.transfer_state,
        source_transfer_state=working.source_transfer_state,
        source_profile=working.source_profile,
        hdr_metadata=deepcopy(working.hdr_metadata),
        orientation_applied=working.orientation_applied,
        alpha_policy=working.alpha_policy,
        bit_depth_in=working.bit_depth_in,
        source_path=working.source_path,
        warnings=warnings,
    )


__all__ = [
    "REC2020_GAMUT_TOLERANCE",
    "REC2020_SAFE_LAB_STYLES",
    "apply_rec2020_safe_lab",
]
