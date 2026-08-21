"""Pure deterministic safe-Lab rendering entry points."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from scripts.pipeline_color_baseline import style_transfer_rgb
from src.preprocess import WorkingImage, working_image_to_srgb_float

from .render_contract import COLOR_PARAMETER_KEYS, validate_render_profile


class StyleSafeEngineError(ValueError):
    """Raised when a resolved style or engine result violates the v1 contract."""


def _validated_parameters(parameters: Mapping[str, Any]) -> dict[str, Any]:
    if set(parameters) != COLOR_PARAMETER_KEYS:
        raise StyleSafeEngineError("safe-Lab style parameter keys drifted")
    numeric = COLOR_PARAMETER_KEYS - {
        "gamut_safe",
        "gamut_mode",
        "use_guardrails",
    }
    if any(
        isinstance(parameters[key], bool)
        or not isinstance(parameters[key], (int, float))
        or not math.isfinite(float(parameters[key]))
        for key in numeric
    ):
        raise StyleSafeEngineError("safe-Lab style parameters must be finite")
    if not isinstance(parameters["gamut_safe"], bool) or not isinstance(
        parameters["use_guardrails"], bool
    ):
        raise StyleSafeEngineError("safe-Lab boolean parameters drifted")
    if parameters["gamut_mode"] not in {"off", "source", "chroma"}:
        raise StyleSafeEngineError("safe-Lab gamut mode drifted")
    return dict(parameters)


def render_resolved_safe_lab_rgb(
    encoded_srgb: np.ndarray,
    *,
    style: str,
    style_statistics: Mapping[str, Any],
    style_parameters: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Render one already-resolved style without file or CLI state."""

    source = np.asarray(encoded_srgb)
    if (
        source.dtype != np.float32
        or source.ndim != 3
        or source.shape[2] != 3
        or source.size == 0
        or not np.isfinite(source).all()
        or np.any((source < 0.0) | (source > 1.0))
    ):
        raise StyleSafeEngineError("encoded_srgb must be finite bounded HxWx3 float32")
    if not isinstance(style, str) or not style:
        raise StyleSafeEngineError("style must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise StyleSafeEngineError("seed must be an integer")
    parameters = _validated_parameters(style_parameters)
    output = np.ascontiguousarray(
        style_transfer_rgb(
            source,
            dict(style_statistics),
            style,
            strength=float(parameters["strength"]),
            luma_strength=float(parameters["luma_strength"]),
            grain=float(parameters["grain"]),
            seed=seed,
            gamut_safe=parameters["gamut_safe"],
            gamut_mode=parameters["gamut_mode"],
            tone_rolloff=float(parameters["tone_rolloff"]),
            shadow_floor_l=float(parameters["shadow_floor_l"]),
            highlight_ceiling_l=float(parameters["highlight_ceiling_l"]),
            preserve_luma_detail_strength=float(
                parameters["preserve_luma_detail"]
            ),
            chroma_curve_strength=float(parameters["chroma_curve_strength"]),
            output_margin=int(parameters["output_margin"]),
            guardrails=dict(guardrails) if parameters["use_guardrails"] else None,
            dither=float(parameters["dither"]),
        ),
        dtype=np.float32,
    )
    if not np.isfinite(output).all() or np.any((output < 0.0) | (output > 1.0)):
        raise StyleSafeEngineError("safe-Lab output is non-finite or unbounded")
    return output


def render_style_safe_working_image(
    working: WorkingImage,
    *,
    profile: Mapping[str, Any],
    style: str,
    style_statistics: Mapping[str, Any],
    guardrails: Mapping[str, Any],
    seed: int,
) -> np.ndarray:
    """Render a validated v1 profile from one WorkingImage to float32 sRGB."""

    validate_render_profile(profile)
    styles = profile["style_parameters"]
    if style not in styles:
        raise StyleSafeEngineError(f"style is absent from profile: {style}")
    source = working_image_to_srgb_float(working)
    return render_resolved_safe_lab_rgb(
        source,
        style=style,
        style_statistics=style_statistics,
        style_parameters=styles[style],
        guardrails=guardrails,
        seed=seed,
    )
