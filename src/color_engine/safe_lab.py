"""Pure CIELAB-domain implementation of the deterministic safe-Lab look."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter


B_AND_W_STYLES = frozenset({"hp5", "tri_x_400"})


@dataclass(frozen=True)
class SafeLabSourceContext:
    """Immutable full-image reduction needed by the safe-Lab operator."""

    source_shape: tuple[int, int, int]
    pixel_count: int
    lab_mean: tuple[float, float, float]
    lab_std: tuple[float, float, float]


def validate_safe_lab_source_context(context: SafeLabSourceContext) -> None:
    """Fail closed when a context cannot describe one finite Lab image."""

    if not isinstance(context, SafeLabSourceContext):
        raise ValueError("source_context must be SafeLabSourceContext")
    height, width, channels = context.source_shape
    if channels != 3 or height <= 0 or width <= 0 or context.pixel_count != height * width:
        raise ValueError("source_context shape/pixel_count is invalid")
    mean = np.asarray(context.lab_mean, dtype=np.float32)
    std = np.asarray(context.lab_std, dtype=np.float32)
    if mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("source_context Lab statistics must be finite three-channel values")
    if np.any(std < 1e-3):
        raise ValueError("source_context Lab std violates the legacy floor")


def _validate_lab_pair(source_lab: np.ndarray, target_lab: np.ndarray | None = None) -> None:
    if not isinstance(source_lab, np.ndarray):
        raise TypeError("source_lab must be a numpy ndarray")
    if source_lab.ndim != 3 or source_lab.shape[2] != 3 or not np.isfinite(source_lab).all():
        raise ValueError("source_lab must be finite HxWx3 Lab")
    if source_lab.shape[0] == 0 or source_lab.shape[1] == 0:
        raise ValueError("source_lab must have non-empty spatial dimensions")
    if target_lab is not None:
        if not isinstance(target_lab, np.ndarray):
            raise TypeError("target_lab must be a numpy ndarray")
        if target_lab.shape != source_lab.shape or not np.isfinite(target_lab).all():
            raise ValueError("target_lab must be finite and match source_lab")


def safe_lab_context_from_lab(
    lab: np.ndarray,
    source_shape: tuple[int, int, int] | None = None,
) -> SafeLabSourceContext:
    """Compute the exact legacy mean/std reduction from a complete Lab image."""

    _validate_lab_pair(lab)
    shape = tuple(int(size) for size in (source_shape or lab.shape))
    if shape != tuple(int(size) for size in lab.shape):
        raise ValueError("source_shape must match the complete Lab image")
    flattened = lab.reshape(-1, 3)
    mean = flattened.mean(axis=0)
    std = np.maximum(flattened.std(axis=0), 1e-3)
    return SafeLabSourceContext(
        source_shape=shape,
        pixel_count=int(flattened.shape[0]),
        lab_mean=tuple(float(value) for value in mean),
        lab_std=tuple(float(value) for value in std),
    )


def smoothstep(edge0: float, edge1: float, value: np.ndarray) -> np.ndarray:
    x = np.clip((value - edge0) / max(edge1 - edge0, 1e-6), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def skin_like_mask(lab: np.ndarray) -> np.ndarray:
    return (
        (lab[..., 0] > 20.0)
        & (lab[..., 0] < 92.0)
        & (lab[..., 1] > 4.0)
        & (lab[..., 1] < 28.0)
        & (lab[..., 2] > 4.0)
        & (lab[..., 2] < 46.0)
    )


def apply_tone_rolloff(
    lab: np.ndarray,
    strength: float,
    shadow_floor_l: float,
    highlight_ceiling_l: float,
) -> np.ndarray:
    if strength <= 0:
        return lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = lab.copy()
    luminance = np.clip(out[..., 0] / 100.0, 0.0, 1.0)
    smooth = luminance * luminance * (3.0 - 2.0 * luminance)
    low = np.clip(shadow_floor_l / 100.0, 0.0, 0.25)
    high = np.clip(highlight_ceiling_l / 100.0, 0.75, 1.0)
    rolled = low + smooth * (high - low)
    out[..., 0] = 100.0 * ((1.0 - strength) * luminance + strength * rolled)
    return out


def apply_color_guardrails(
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    neutral_protect: float,
    skin_protect: float,
    max_chroma_gain: float | None,
    max_chroma_boost: float | None,
    max_chroma_absolute: float | None,
) -> np.ndarray:
    out = target_lab.copy()
    source_ab = source_lab[..., 1:3]
    out_ab = out[..., 1:3]
    source_chroma = np.linalg.norm(source_ab, axis=2, keepdims=True)

    if neutral_protect > 0:
        neutral_weight = 1.0 - smoothstep(4.0, 14.0, source_chroma)
        blend = np.clip(neutral_weight * neutral_protect, 0.0, 1.0)
        out_ab = out_ab * (1.0 - blend) + source_ab * blend

    if skin_protect > 0:
        skin_weight = skin_like_mask(source_lab)[..., None].astype(np.float32) * np.clip(skin_protect, 0.0, 1.0)
        out_ab = out_ab * (1.0 - skin_weight) + source_ab * skin_weight

    if max_chroma_gain is not None or max_chroma_boost is not None or max_chroma_absolute is not None:
        gain = float(max_chroma_gain if max_chroma_gain is not None else 999.0)
        boost = float(max_chroma_boost if max_chroma_boost is not None else 999.0)
        cap = np.maximum(source_chroma * gain, source_chroma + boost)
        if max_chroma_absolute is not None:
            cap = np.minimum(cap, float(max_chroma_absolute))
        out_chroma = np.maximum(np.linalg.norm(out_ab, axis=2, keepdims=True), 1e-6)
        scale = np.minimum(1.0, cap / out_chroma)
        knee = smoothstep(0.82, 1.0, out_chroma / np.maximum(cap, 1e-6))
        out_ab = out_ab * ((1.0 - knee) + knee * scale)

    out[..., 1:3] = out_ab
    return out


def apply_chroma_curve(source_lab: np.ndarray, target_lab: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return target_lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = target_lab.copy()
    source_chroma = np.linalg.norm(source_lab[..., 1:3], axis=2, keepdims=True)
    saturated_weight = smoothstep(24.0, 60.0, source_chroma)
    scale = 1.0 - strength * saturated_weight
    out[..., 1:3] = source_lab[..., 1:3] + (out[..., 1:3] - source_lab[..., 1:3]) * scale
    return out


def preserve_luma_detail(source_lab: np.ndarray, target_lab: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return target_lab
    strength = float(np.clip(strength, 0.0, 1.0))
    out = target_lab.copy()
    source_l = source_lab[..., 0]
    target_l = target_lab[..., 0]
    source_detail = source_l - gaussian_filter(source_l, sigma=1.1)
    target_base = gaussian_filter(target_l, sigma=1.1)
    target_detail = target_l - target_base
    out[..., 0] = np.clip(target_base + target_detail * (1.0 - strength) + source_detail * strength, 0.0, 100.0)
    return out


def apply_safe_lab_transform(
    source_lab: np.ndarray,
    *,
    source_context: SafeLabSourceContext,
    destination_mean: np.ndarray,
    destination_std: np.ndarray,
    style: str,
    strength: float,
    luma_strength: float,
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
) -> np.ndarray:
    """Apply the portable safe-Lab look without RGB or gamut assumptions."""

    _validate_lab_pair(source_lab)
    validate_safe_lab_source_context(source_context)
    destination_mean = np.asarray(destination_mean, dtype=np.float32)
    destination_std = np.asarray(destination_std, dtype=np.float32)
    if (
        destination_mean.shape != (3,)
        or destination_std.shape != (3,)
        or not np.isfinite(destination_mean).all()
        or not np.isfinite(destination_std).all()
        or np.any(destination_std <= 0)
    ):
        raise ValueError("destination Lab statistics must be finite positive three-channel values")

    source_mean = np.asarray(source_context.lab_mean, dtype=np.float32)
    source_std = np.asarray(source_context.lab_std, dtype=np.float32)
    transferred = (source_lab - source_mean) / source_std * destination_std + destination_mean
    out = source_lab.copy()
    out[..., 0] = source_lab[..., 0] + luma_strength * strength * (transferred[..., 0] - source_lab[..., 0])
    out[..., 1:] = source_lab[..., 1:] + strength * (transferred[..., 1:] - source_lab[..., 1:])

    if style in B_AND_W_STYLES:
        gray_strength = min(1.0, strength * 1.35)
        out[..., 1:] *= 1.0 - gray_strength
        contrast = 1.0 + 0.30 * strength
        out[..., 0] = np.clip((out[..., 0] - 50.0) * contrast + 50.0, 0.0, 100.0)

    out = apply_chroma_curve(source_lab, out, chroma_curve_strength)
    out = preserve_luma_detail(source_lab, out, preserve_luma_detail_strength)
    out = apply_color_guardrails(
        source_lab,
        out,
        neutral_protect=float(neutral_protect),
        skin_protect=float(skin_protect),
        max_chroma_gain=max_chroma_gain,
        max_chroma_boost=max_chroma_boost,
        max_chroma_absolute=max_chroma_absolute,
    )
    return apply_tone_rolloff(out, tone_rolloff, shadow_floor_l, highlight_ceiling_l)


__all__ = [
    "B_AND_W_STYLES",
    "SafeLabSourceContext",
    "apply_chroma_curve",
    "apply_color_guardrails",
    "apply_safe_lab_transform",
    "apply_tone_rolloff",
    "preserve_luma_detail",
    "safe_lab_context_from_lab",
    "skin_like_mask",
    "smoothstep",
    "validate_safe_lab_source_context",
]
