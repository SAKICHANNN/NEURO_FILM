"""Scanner-convolved NPS/ACF feasibility metrics for uniform film scans."""

from __future__ import annotations

from itertools import combinations
import math
from typing import Any

import numpy as np


class UniformGrainNpsError(ValueError):
    """Raised when an NPS feasibility input violates the frozen contract."""


def _quadratic_detrend(values: np.ndarray) -> np.ndarray:
    height, width = values.shape
    y, x = np.mgrid[-1.0:1.0:complex(height), -1.0:1.0:complex(width)]
    design = np.stack(
        (
            np.ones_like(x),
            x,
            y,
            x * x,
            x * y,
            y * y,
        ),
        axis=-1,
    ).reshape(-1, 6)
    coefficients, *_ = np.linalg.lstsq(
        design,
        values.reshape(-1),
        rcond=None,
    )
    trend = (design @ coefficients).reshape(values.shape)
    return values - trend


def radial_nps_signature(
    crop: np.ndarray,
    *,
    band_edges_cycles_per_pixel: list[float],
) -> np.ndarray:
    """Return a normalized log-radial NPS shape in scanner-code space."""
    values = np.asarray(crop, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) < 64
        or not np.all(np.isfinite(values))
    ):
        raise UniformGrainNpsError("NPS crop must be finite 2D and at least 64px")
    mean = float(np.mean(values))
    if mean <= 0.0:
        raise UniformGrainNpsError("NPS crop mean must be positive")
    edges = np.asarray(band_edges_cycles_per_pixel, dtype=np.float64)
    if (
        edges.ndim != 1
        or len(edges) < 3
        or np.any(~np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
        or edges[0] <= 0.0
        or edges[-1] > math.sqrt(0.5)
    ):
        raise UniformGrainNpsError("invalid radial NPS band edges")
    relative = _quadratic_detrend(values / mean)
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    transformed = np.fft.rfft2(relative * window)
    power = np.square(np.abs(transformed))
    fy = np.fft.fftfreq(values.shape[0])[:, None]
    fx = np.fft.rfftfreq(values.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    bands = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        selected = power[(radius >= lower) & (radius < upper)]
        if selected.size == 0:
            raise UniformGrainNpsError("empty radial NPS band")
        bands.append(float(np.mean(selected)))
    profile = np.asarray(bands, dtype=np.float64)
    total = float(np.sum(profile))
    if total <= 0.0 or not np.all(np.isfinite(profile)):
        raise UniformGrainNpsError("degenerate radial NPS")
    normalized = profile / total
    signature = np.log(np.maximum(normalized, np.finfo(np.float64).tiny))
    signature -= float(np.mean(signature))
    norm = float(np.linalg.norm(signature))
    if norm <= 0.0:
        raise UniformGrainNpsError("constant radial NPS signature")
    output = signature / norm
    output.setflags(write=False)
    return output


def acf_lag_signature(
    crop: np.ndarray,
    *,
    lags_yx: list[list[int]],
) -> np.ndarray:
    """Return normalized detrended scanner-code autocorrelation at fixed lags."""
    values = np.asarray(crop, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) < 64
        or not np.all(np.isfinite(values))
    ):
        raise UniformGrainNpsError("ACF crop must be finite 2D and at least 64px")
    mean = float(np.mean(values))
    if mean <= 0.0:
        raise UniformGrainNpsError("ACF crop mean must be positive")
    residual = _quadratic_detrend(values / mean)
    variance = float(np.mean(np.square(residual)))
    if variance <= 0.0:
        raise UniformGrainNpsError("degenerate ACF crop")
    correlations: list[float] = []
    for lag in lags_yx:
        if (
            len(lag) != 2
            or any(not isinstance(value, int) for value in lag)
            or any(value < 0 for value in lag)
            or lag == [0, 0]
        ):
            raise UniformGrainNpsError("invalid nonzero ACF lag")
        dy, dx = lag
        if dy >= values.shape[0] or dx >= values.shape[1]:
            raise UniformGrainNpsError("ACF lag leaves crop")
        left = residual[
            : values.shape[0] - dy if dy else None,
            : values.shape[1] - dx if dx else None,
        ]
        right = residual[
            dy:,
            dx:,
        ]
        correlations.append(float(np.mean(left * right) / variance))
    output = np.asarray(correlations, dtype=np.float64)
    if not np.all(np.isfinite(output)):
        raise UniformGrainNpsError("non-finite ACF signature")
    output.setflags(write=False)
    return output


def fixed_fractional_crops(
    image: np.ndarray,
    *,
    crop_size: int,
    centers_yx: list[list[float]],
) -> list[np.ndarray]:
    """Extract exact center-position crops without content-based selection."""
    values = np.asarray(image)
    if values.ndim != 2 or crop_size < 64:
        raise UniformGrainNpsError("crop source must be 2D")
    crops: list[np.ndarray] = []
    for center in centers_yx:
        if len(center) != 2 or any(not 0.0 < float(v) < 1.0 for v in center):
            raise UniformGrainNpsError("invalid fractional crop center")
        center_y = int(round(float(center[0]) * (values.shape[0] - 1)))
        center_x = int(round(float(center[1]) * (values.shape[1] - 1)))
        y0 = center_y - crop_size // 2
        x0 = center_x - crop_size // 2
        y1 = y0 + crop_size
        x1 = x0 + crop_size
        if y0 < 0 or x0 < 0 or y1 > values.shape[0] or x1 > values.shape[1]:
            raise UniformGrainNpsError("fractional crop leaves image")
        crops.append(values[y0:y1, x0:x1])
    return crops


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.shape != rhs.shape or lhs.ndim != 1:
        raise UniformGrainNpsError("signature shape mismatch")
    denominator = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if denominator <= 0.0:
        raise UniformGrainNpsError("zero-length signature")
    return float(np.dot(lhs, rhs) / denominator)


def _nearest_centroid_accuracy(
    signatures: np.ndarray,
    labels: np.ndarray,
) -> float:
    predictions: list[int] = []
    for index in range(len(labels)):
        train = np.arange(len(labels)) != index
        centroids = []
        for label in (0, 1):
            selected = signatures[train & (labels == label)]
            if len(selected) == 0:
                raise UniformGrainNpsError("leave-one-out class is empty")
            centroid = np.mean(selected, axis=0)
            centroid /= np.linalg.norm(centroid)
            centroids.append(centroid)
        similarities = [
            cosine_similarity(signatures[index], centroid)
            for centroid in centroids
        ]
        predictions.append(int(similarities[1] > similarities[0]))
    recalls = [
        float(np.mean(np.asarray(predictions)[labels == label] == label))
        for label in (0, 1)
    ]
    return float(np.mean(recalls))


def exact_balanced_label_permutation(
    signatures: np.ndarray,
    labels: list[str],
) -> dict[str, Any]:
    """Evaluate group-level LOOCV and its exact balanced-label null."""
    values = np.asarray(signatures, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(labels):
        raise UniformGrainNpsError("signature matrix mismatch")
    unique = sorted(set(labels))
    if len(unique) != 2:
        raise UniformGrainNpsError("exact pilot requires two stock labels")
    binary = np.asarray([unique.index(value) for value in labels], dtype=np.int8)
    count_one = int(np.sum(binary))
    if count_one != len(binary) // 2:
        raise UniformGrainNpsError("exact pilot requires balanced labels")
    observed = _nearest_centroid_accuracy(values, binary)
    null: list[float] = []
    for positive in combinations(range(len(binary)), count_one):
        permuted = np.zeros(len(binary), dtype=np.int8)
        permuted[list(positive)] = 1
        null.append(_nearest_centroid_accuracy(values, permuted))
    # The enumeration is exhaustive and already contains the observed
    # assignment (and its label-swapped complement), so no Monte-Carlo
    # add-one correction applies.
    p_value = float(sum(value >= observed for value in null) / len(null))
    return {
        "balanced_accuracy": observed,
        "exact_permutation_p_value": p_value,
        "balanced_partitions": len(null),
        "null_accuracy_values": sorted(set(null)),
    }


def evaluate_signatures(
    *,
    scan_ids: list[str],
    stock_ids: list[str],
    crop_signatures: list[np.ndarray],
    gates: dict[str, float],
) -> dict[str, Any]:
    """Evaluate scan-group repeatability before any model-parameter fitting."""
    if not (
        len(scan_ids) == len(stock_ids) == len(crop_signatures)
        and len(scan_ids) >= 4
        and len(set(scan_ids)) == len(scan_ids)
    ):
        raise UniformGrainNpsError("scan groups are invalid")
    per_scan = []
    spatial_correlations: list[float] = []
    for rows in crop_signatures:
        values = np.asarray(rows, dtype=np.float64)
        if values.ndim != 2 or len(values) < 2:
            raise UniformGrainNpsError("each scan requires multiple crops")
        signature = np.mean(values, axis=0)
        signature /= np.linalg.norm(signature)
        per_scan.append(signature)
        spatial_correlations.extend(
            cosine_similarity(row, signature) for row in values
        )
    matrix = np.asarray(per_scan)
    within: list[float] = []
    cross: list[float] = []
    for left, right in combinations(range(len(scan_ids)), 2):
        similarity = cosine_similarity(matrix[left], matrix[right])
        target = within if stock_ids[left] == stock_ids[right] else cross
        target.append(similarity)
    if not within or not cross:
        raise UniformGrainNpsError("within and cross comparisons are required")
    classification = exact_balanced_label_permutation(matrix, stock_ids)
    within_median = float(np.median(within))
    cross_median = float(np.median(cross))
    crop_median = float(np.median(spatial_correlations))
    repeat_pass = bool(
        crop_median >= float(gates["minimum_median_crop_to_scan_similarity"])
        and within_median
        >= float(gates["minimum_median_within_stock_similarity"])
    )
    association_pass = bool(
        classification["balanced_accuracy"]
        >= float(gates["minimum_leave_one_scan_out_balanced_accuracy"])
        and classification["exact_permutation_p_value"]
        <= float(gates["maximum_exact_permutation_p_value"])
        and within_median - cross_median
        >= float(gates["minimum_within_minus_cross_similarity"])
    )
    if not repeat_pass:
        branch = "close_observable_nps_source_as_unstable"
    elif association_pass:
        branch = "retain_stock_associated_scanner_convolved_signature_only"
    else:
        branch = "retain_generic_scanner_convolved_signature_only"
    return {
        "scan_count": len(scan_ids),
        "median_crop_to_scan_similarity": crop_median,
        "median_within_stock_similarity": within_median,
        "median_cross_stock_similarity": cross_median,
        "within_minus_cross_similarity": within_median - cross_median,
        "classification": classification,
        "repeatability_pass": repeat_pass,
        "stock_association_pass": repeat_pass and association_pass,
        "branch": branch,
    }


__all__ = [
    "UniformGrainNpsError",
    "acf_lag_signature",
    "cosine_similarity",
    "evaluate_signatures",
    "exact_balanced_label_permutation",
    "fixed_fractional_crops",
    "radial_nps_signature",
]
