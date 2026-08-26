"""Deterministic capture-metadata AWB ridge primitives for P241."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np


class CaptureMetadataAwbError(ValueError):
    """Raised when metadata or a frozen ridge payload is invalid."""


FULL_FEATURE_NAMES = (
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "log2_exposure_seconds",
    "log2_iso_speed",
    "log2_f_number",
)
TIME_FEATURE_INDICES = (0, 1, 2, 3)
EXPOSURE_FEATURE_INDICES = (4, 5, 6)


@dataclass(frozen=True)
class FrozenRidge:
    """A standardized deterministic multi-output ridge model."""

    alpha: float
    feature_indices: tuple[int, ...]
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[tuple[float, float], ...]
    intercept: tuple[float, float]


def capture_features(facts: dict[str, Any]) -> np.ndarray:
    """Return the exact seven source-only capture features frozen by P241."""

    try:
        timestamp = datetime.fromisoformat(str(facts["date_time_original"]))
        exposure = float(facts["exposure_time_seconds"])
        iso = float(facts["iso_speed"])
        f_number = float(facts["f_number"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureMetadataAwbError("capture facts are incomplete") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in (exposure, iso, f_number)):
        raise CaptureMetadataAwbError("capture facts must be positive and finite")

    seconds = timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second
    hour_phase = 2.0 * math.pi * seconds / 86400.0
    day_phase = 2.0 * math.pi * (timestamp.timetuple().tm_yday - 1) / 365.2425
    result = np.asarray(
        [
            math.sin(hour_phase),
            math.cos(hour_phase),
            math.sin(day_phase),
            math.cos(day_phase),
            math.log2(exposure),
            math.log2(iso),
            math.log2(f_number),
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(result)):
        raise CaptureMetadataAwbError("capture features must be finite")
    return result


def neutral_log_target(facts: dict[str, Any]) -> np.ndarray:
    """Map a green-normalized AsShotNeutral triple to two log ratios."""

    try:
        neutral = np.asarray(
            facts["as_shot_neutral_green_normalized"], dtype=np.float64
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CaptureMetadataAwbError("AsShotNeutral target is missing") from exc
    if neutral.shape != (3,) or not np.all(np.isfinite(neutral)) or np.any(neutral <= 0.0):
        raise CaptureMetadataAwbError("AsShotNeutral target is invalid")
    if not math.isclose(float(neutral[1]), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise CaptureMetadataAwbError("AsShotNeutral target is not green-normalized")
    return np.log(neutral[[0, 2]])


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    feature_indices: Sequence[int],
    alpha: float,
) -> FrozenRidge:
    """Fit one fixed standardized ridge with an unpenalized target mean."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    indices = tuple(int(index) for index in feature_indices)
    if x.ndim != 2 or x.shape[1] != len(FULL_FEATURE_NAMES):
        raise CaptureMetadataAwbError("feature matrix shape is invalid")
    if y.shape != (x.shape[0], 2) or x.shape[0] < 3:
        raise CaptureMetadataAwbError("target matrix shape is invalid")
    if not indices or len(set(indices)) != len(indices):
        raise CaptureMetadataAwbError("feature indices are invalid")
    if any(index < 0 or index >= x.shape[1] for index in indices):
        raise CaptureMetadataAwbError("feature index is out of range")
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise CaptureMetadataAwbError("ridge alpha must be positive and finite")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise CaptureMetadataAwbError("ridge inputs must be finite")

    selected = x[:, indices]
    mean = np.mean(selected, axis=0)
    scale = np.std(selected, axis=0, ddof=0)
    if np.any(scale <= 1e-12):
        raise CaptureMetadataAwbError("ridge feature scale is degenerate")
    standardized = (selected - mean) / scale
    intercept = np.mean(y, axis=0)
    centered = y - intercept
    gram = standardized.T @ standardized
    coefficients = np.linalg.solve(
        gram + alpha * np.eye(len(indices), dtype=np.float64),
        standardized.T @ centered,
    )
    return FrozenRidge(
        alpha=float(alpha),
        feature_indices=indices,
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        coefficients=tuple(
            (float(row[0]), float(row[1])) for row in coefficients
        ),
        intercept=(float(intercept[0]), float(intercept[1])),
    )


def apply_ridge(model: FrozenRidge, features: np.ndarray) -> np.ndarray:
    """Apply a validated frozen ridge and return log-neutral predictions."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FULL_FEATURE_NAMES):
        raise CaptureMetadataAwbError("feature matrix shape is invalid")
    indices = model.feature_indices
    mean = np.asarray(model.mean, dtype=np.float64)
    scale = np.asarray(model.scale, dtype=np.float64)
    coefficients = np.asarray(model.coefficients, dtype=np.float64)
    intercept = np.asarray(model.intercept, dtype=np.float64)
    if (
        mean.shape != (len(indices),)
        or scale.shape != mean.shape
        or coefficients.shape != (len(indices), 2)
        or intercept.shape != (2,)
        or np.any(scale <= 1e-12)
    ):
        raise CaptureMetadataAwbError("frozen ridge payload is invalid")
    result = ((x[:, indices] - mean) / scale) @ coefficients + intercept
    if not np.all(np.isfinite(result)):
        raise CaptureMetadataAwbError("ridge output must be finite")
    return result


def log_target_to_neutral(log_target: np.ndarray) -> np.ndarray:
    """Return green-normalized neutral triples from two log ratios."""

    values = np.asarray(log_target, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise CaptureMetadataAwbError("log-neutral predictions are invalid")
    result = np.column_stack((np.exp(values[:, 0]), np.ones(len(values)), np.exp(values[:, 1])))
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise CaptureMetadataAwbError("neutral predictions are invalid")
    return result


def angular_errors_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Measure per-row chromaticity-vector angle in degrees."""

    predicted_array = np.asarray(predicted, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if predicted_array.shape != target_array.shape or predicted_array.ndim != 2 or predicted_array.shape[1] != 3:
        raise CaptureMetadataAwbError("neutral arrays must be matching Nx3 matrices")
    if (
        not np.all(np.isfinite(predicted_array))
        or not np.all(np.isfinite(target_array))
        or np.any(predicted_array <= 0.0)
        or np.any(target_array <= 0.0)
    ):
        raise CaptureMetadataAwbError("neutral arrays must be positive and finite")
    dots = np.sum(predicted_array * target_array, axis=1)
    norms = np.linalg.norm(predicted_array, axis=1) * np.linalg.norm(target_array, axis=1)
    cosines = np.clip(dots / norms, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


__all__ = [
    "EXPOSURE_FEATURE_INDICES",
    "FULL_FEATURE_NAMES",
    "TIME_FEATURE_INDICES",
    "CaptureMetadataAwbError",
    "FrozenRidge",
    "angular_errors_degrees",
    "apply_ridge",
    "capture_features",
    "fit_ridge",
    "log_target_to_neutral",
    "neutral_log_target",
]
