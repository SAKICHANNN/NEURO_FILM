"""Destination-working-space gamut policies for Lab-domain colour operators."""

from __future__ import annotations

import numpy as np

from .lab import lab_to_linear_rgb


def _validate_lab(value: np.ndarray, label: str) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{label} must be a numpy ndarray")
    if value.dtype != np.float32:
        raise TypeError(f"{label} must be float32")
    if value.ndim != 3 or value.shape[2] != 3 or not np.isfinite(value).all():
        raise ValueError(f"{label} must be finite HxWx3 Lab")
    if value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{label} must have non-empty spatial dimensions")


def in_working_gamut(
    lab: np.ndarray,
    *,
    working_space: str,
    tolerance: float = 0.0,
) -> np.ndarray:
    """Return a per-pixel mask for an explicit linear destination gamut."""

    _validate_lab(lab, "lab")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    linear = lab_to_linear_rgb(lab, working_space=working_space)
    return np.all((linear >= -tolerance) & (linear <= 1.0 + tolerance), axis=-1)


def compress_source_to_working_gamut(
    source_lab: np.ndarray,
    target_lab: np.ndarray,
    *,
    working_space: str,
    iterations: int = 24,
    tolerance: float = 2e-6,
) -> np.ndarray:
    """Compress along the source-to-target Lab segment in the destination gamut."""

    _validate_lab(source_lab, "source_lab")
    _validate_lab(target_lab, "target_lab")
    if target_lab.shape != source_lab.shape:
        raise ValueError("target_lab must match source_lab")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not in_working_gamut(source_lab, working_space=working_space, tolerance=tolerance).all():
        raise ValueError("source Lab endpoint is outside destination working gamut")

    low = np.zeros(source_lab.shape[:2] + (1,), dtype=np.float32)
    high = np.ones_like(low)
    delta = target_lab - source_lab
    for _ in range(iterations):
        mid = (low + high) * 0.5
        candidate = source_lab + delta * mid
        valid = in_working_gamut(candidate, working_space=working_space)[..., None]
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    result = np.asarray(source_lab + delta * low, dtype=np.float32)
    if not in_working_gamut(result, working_space=working_space, tolerance=tolerance).all():
        raise ValueError("source compression did not produce destination-gamut Lab")
    return result


def compress_chroma_to_working_gamut(
    target_lab: np.ndarray,
    *,
    working_space: str,
    iterations: int = 24,
    tolerance: float = 2e-6,
) -> np.ndarray:
    """Reduce chroma at fixed L and hue until the destination gamut contains it."""

    _validate_lab(target_lab, "target_lab")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    neutral = target_lab.copy()
    neutral[..., 1:] = 0.0
    if not in_working_gamut(neutral, working_space=working_space, tolerance=tolerance).all():
        raise ValueError("same-L neutral endpoint is outside destination working gamut")

    low = np.zeros(target_lab.shape[:2] + (1,), dtype=np.float32)
    high = np.ones_like(low)
    delta = target_lab - neutral
    for _ in range(iterations):
        mid = (low + high) * 0.5
        candidate = neutral + delta * mid
        valid = in_working_gamut(candidate, working_space=working_space)[..., None]
        low = np.where(valid, mid, low)
        high = np.where(valid, high, mid)
    result = np.asarray(neutral + delta * low, dtype=np.float32)
    if not in_working_gamut(result, working_space=working_space, tolerance=tolerance).all():
        raise ValueError("chroma compression did not produce destination-gamut Lab")
    return result


__all__ = [
    "compress_chroma_to_working_gamut",
    "compress_source_to_working_gamut",
    "in_working_gamut",
]
