"""Deterministic helpers for the RF1.2 BlueNeg nested LOO diagnosis."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from skimage.color import deltaE_ciede2000, rgb2lab


class BlueNegNestedLOOError(ValueError):
    """Raised when an RF1.2 fold violates the frozen contract."""


def stable_select(
    values: Sequence[str],
    *,
    count: int,
    seed: int,
    namespace: str,
    exclude: str | None = None,
) -> tuple[str, ...]:
    """Hash-select a fixed number of unique values, optionally excluding one."""
    candidates = sorted({str(value) for value in values if str(value) != exclude})
    if count <= 0 or len(candidates) < count:
        raise BlueNegNestedLOOError("support pool cannot satisfy the frozen budget")
    ordered = sorted(
        candidates,
        key=lambda value: hashlib.sha256(
            f"{seed}:{namespace}:{value}".encode()
        ).hexdigest(),
    )
    return tuple(ordered[:count])


def count_preserving_label_permutation(
    frame_to_roll: Mapping[str, str], *, seed: int
) -> dict[str, str]:
    """Permute physical-roll labels while preserving their exact counts."""
    frames = sorted(str(frame) for frame in frame_to_roll)
    labels = np.asarray([str(frame_to_roll[frame]) for frame in frames], dtype=object)
    rng = np.random.default_rng(seed)
    permuted = labels[rng.permutation(len(labels))]
    result = dict(zip(frames, (str(value) for value in permuted), strict=True))
    before = {label: int(np.sum(labels == label)) for label in np.unique(labels)}
    after_values = np.asarray(list(result.values()), dtype=object)
    after = {
        label: int(np.sum(after_values == label)) for label in np.unique(after_values)
    }
    if before != after:
        raise BlueNegNestedLOOError("shuffled roll labels changed group counts")
    return result


def source_descriptor(linear_rgb: np.ndarray) -> np.ndarray:
    """Compute the frozen target-free global descriptor for one source frame."""
    pixels = np.asarray(linear_rgb, dtype=np.float64).reshape(-1, 3)
    if len(pixels) < 16 or not np.all(np.isfinite(pixels)):
        raise BlueNegNestedLOOError("descriptor requires finite RGB pixels")
    lab = _to_lab(pixels)
    chroma = np.linalg.norm(lab[:, 1:3], axis=1)
    return np.concatenate(
        (
            pixels.mean(axis=0),
            pixels.std(axis=0),
            np.percentile(lab[:, 0], [10.0, 50.0, 90.0]),
            [float(chroma.mean()), float(chroma.std())],
        )
    )


def robust_standardize(descriptors: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Median/IQR-standardize descriptors with a finite zero-spread fallback."""
    names = sorted(descriptors)
    if not names:
        raise BlueNegNestedLOOError("descriptor table is empty")
    matrix = np.stack([np.asarray(descriptors[name], dtype=np.float64) for name in names])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise BlueNegNestedLOOError("descriptor table must be finite and rectangular")
    median = np.median(matrix, axis=0)
    iqr = np.percentile(matrix, 75.0, axis=0) - np.percentile(matrix, 25.0, axis=0)
    scale = np.where(iqr > 1e-8, iqr, 1.0)
    return {name: (matrix[index] - median) / scale for index, name in enumerate(names)}


def nearest_other_rolls(
    query: str,
    descriptors: Mapping[str, np.ndarray],
    frame_to_roll: Mapping[str, str],
    *,
    count: int,
) -> tuple[str, ...]:
    """Retrieve the closest source descriptors while excluding the query roll."""
    if query not in descriptors or query not in frame_to_roll:
        raise BlueNegNestedLOOError("query is absent from descriptor metadata")
    query_roll = str(frame_to_roll[query])
    candidates = [
        name
        for name in descriptors
        if name != query and str(frame_to_roll[name]) != query_roll
    ]
    if len(candidates) < count:
        raise BlueNegNestedLOOError("retrieval pool cannot satisfy support budget")
    query_feature = np.asarray(descriptors[query], dtype=np.float64)
    ordered = sorted(
        candidates,
        key=lambda name: (
            float(np.linalg.norm(np.asarray(descriptors[name]) - query_feature)),
            name,
        ),
    )
    return tuple(ordered[:count])


def style_match_output(
    source: np.ndarray,
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    alpha_max: float = 1.5,
) -> tuple[np.ndarray, float, float]:
    """Match candidate style to reference using source pixels only."""
    source_pixels = np.asarray(source, dtype=np.float64)
    candidate_pixels = np.asarray(candidate, dtype=np.float64)
    reference_pixels = np.asarray(reference, dtype=np.float64)
    if (
        source_pixels.shape != candidate_pixels.shape
        or source_pixels.shape != reference_pixels.shape
        or source_pixels.ndim != 2
        or source_pixels.shape[1] != 3
    ):
        raise BlueNegNestedLOOError("style matching requires equal Nx3 arrays")
    if alpha_max <= 0.0:
        raise BlueNegNestedLOOError("style-match alpha range must be positive")
    source_lab = _to_lab(source_pixels)
    target_style = float(
        np.mean(deltaE_ciede2000(source_lab, _to_lab(reference_pixels)))
    )
    direction = candidate_pixels - source_pixels

    def objective(alpha: float) -> float:
        matched = source_pixels + float(alpha) * direction
        style = float(np.mean(deltaE_ciede2000(source_lab, _to_lab(matched))))
        return (style - target_style) ** 2

    result = minimize_scalar(
        objective,
        bounds=(0.0, float(alpha_max)),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 40},
    )
    alpha = float(result.x)
    matched = source_pixels + alpha * direction
    achieved = float(np.mean(deltaE_ciede2000(source_lab, _to_lab(matched))))
    return matched, alpha, achieved


def _to_lab(linear_rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(linear_rgb, dtype=np.float64)
    encoded = np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(np.maximum(values, 0.0), 1.0 / 2.4) - 0.055,
    )
    shape = encoded.shape
    return rgb2lab(encoded.reshape(-1, 1, 3)).reshape(shape)
