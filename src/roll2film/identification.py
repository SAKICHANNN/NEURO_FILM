"""Unpaired group-level explicit-operator identification baselines."""

from __future__ import annotations

import numpy as np

from .operators import AffineColorOperator


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
