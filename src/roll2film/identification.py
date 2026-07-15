"""Unpaired group-level explicit-operator identification baselines."""

from __future__ import annotations

import numpy as np

from .operators import AffineColorOperator
from .splines import AffineMonotoneSplineOperator, RationalQuadraticSpline


def estimate_gaussian_transport_operator(
    neutral_prior: np.ndarray,
    target_frames: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    regularization: float = 1e-6,
    normalize_frame_exposure: bool = False,
) -> AffineColorOperator:
    """Estimate the unique SPD Gaussian OT map from an unpaired target set.

    The estimator receives no source/target correspondence. The SPD restriction
    resolves the covariance-map rotational ambiguity for this falsification
    baseline; it is a scientific assumption, not a general film model.
    """
    source = _pixels(neutral_prior)
    frames = tuple(_pixels(frame) for frame in target_frames)
    if not frames:
        raise ValueError("at least one target frame is required")
    if normalize_frame_exposure:
        frames = _normalize_exposure(frames)
    target = np.concatenate(frames, axis=0)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_cov = np.cov(source, rowvar=False) + regularization * np.eye(3)
    target_cov = np.cov(target, rowvar=False) + regularization * np.eye(3)
    source_sqrt = _symmetric_matrix_power(source_cov, 0.5)
    source_inv_sqrt = _symmetric_matrix_power(source_cov, -0.5)
    middle = source_sqrt @ target_cov @ source_sqrt
    matrix = source_inv_sqrt @ _symmetric_matrix_power(middle, 0.5) @ source_inv_sqrt
    matrix = 0.5 * (matrix + matrix.T)
    bias = target_mean - matrix @ source_mean
    return AffineColorOperator(matrix=matrix, bias=bias)


def estimate_affine_spline_transport_operator(
    neutral_prior: np.ndarray,
    target_frames: tuple[np.ndarray, ...] | list[np.ndarray],
    *,
    knot_quantiles: tuple[float, ...] = (0.001, 0.03, 0.12, 0.35, 0.65, 0.88, 0.97, 0.999),
    iterations: int = 8,
    regularization: float = 1e-6,
    normalize_frame_photometric: bool = False,
) -> AffineMonotoneSplineOperator:
    """Fit an L2 affine-plus-monotone map from unmatched source/target pixels.

    Alternating Gaussian transport and marginal quantile splines is a simple,
    deterministic baseline. It deliberately exposes the source-prior and gauge
    assumptions instead of hiding them in a learned renderer.
    """
    source = _pixels(neutral_prior)
    frames = tuple(_pixels(frame) for frame in target_frames)
    if not frames:
        raise ValueError("at least one target frame is required")
    if normalize_frame_photometric:
        frames = _normalize_frame_rgb_means(frames)
    target = np.concatenate(frames, axis=0)
    quantiles = np.asarray(knot_quantiles, dtype=np.float64)
    if (
        quantiles.ndim != 1
        or len(quantiles) < 4
        or np.any(np.diff(quantiles) <= 0.0)
        or quantiles[0] <= 0.0
        or quantiles[-1] >= 1.0
    ):
        raise ValueError("knot_quantiles must be strictly increasing inside (0, 1)")
    if iterations < 1:
        raise ValueError("iterations must be positive")

    affine = estimate_gaussian_transport_operator(
        source,
        frames,
        regularization=regularization,
    )
    splines: tuple[RationalQuadraticSpline, ...] = tuple(
        RationalQuadraticSpline.identity() for _ in range(3)
    )
    for _ in range(iterations):
        intermediate = affine.apply(source)
        fitted: list[RationalQuadraticSpline] = []
        for channel in range(3):
            x_knots = _strictly_increasing(np.quantile(intermediate[:, channel], quantiles))
            y_knots = _strictly_increasing(np.quantile(target[:, channel], quantiles))
            fitted.append(RationalQuadraticSpline.from_knots(x_knots, y_knots))
        splines = tuple(fitted)
        linearized_target = np.stack(
            [splines[channel].inverse(target[:, channel]) for channel in range(3)],
            axis=-1,
        )
        affine = estimate_gaussian_transport_operator(
            source,
            [linearized_target],
            regularization=regularization,
        )
    return AffineMonotoneSplineOperator(affine, splines)  # type: ignore[arg-type]


def _normalize_frame_rgb_means(frames: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    means = np.asarray([frame.mean(axis=0) for frame in frames], dtype=np.float64)
    reference = np.median(means, axis=0)
    return tuple(
        frame * (reference / np.maximum(mean, 1e-8))
        for frame, mean in zip(frames, means)
    )


def _normalize_exposure(frames: tuple[np.ndarray, ...]) -> tuple[np.ndarray, ...]:
    luma = np.array([np.mean(frame @ np.array([0.2126, 0.7152, 0.0722])) for frame in frames])
    target_luma = float(np.median(luma))
    return tuple(frame * (target_luma / max(float(value), 1e-8)) for frame, value in zip(frames, luma))


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if float(eigenvalues.min()) <= 0:
        raise ValueError("covariance must be positive definite")
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def _pixels(values: np.ndarray) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 3:
        raise ValueError("pixels must have shape (N, 3)")
    if len(pixels) < 4 or not np.all(np.isfinite(pixels)):
        raise ValueError("pixels must contain at least four finite RGB samples")
    return pixels


def _strictly_increasing(values: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    for index in range(1, len(result)):
        result[index] = max(result[index], result[index - 1] + epsilon)
    return result
