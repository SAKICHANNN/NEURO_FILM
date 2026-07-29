"""Deterministic two-stage sparse-outlier rejection for paired film fitting."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .positive_film_fitting import (
    PositiveFilmFitResult,
    fit_positive_film_response_operator,
)


@dataclass(frozen=True)
class TwoStagePositiveFilmFitResult:
    """Initial robust fit, label-blind rejected rows, and retained-row refit."""

    stage_one: PositiveFilmFitResult
    stage_two: PositiveFilmFitResult
    residual_scores: np.ndarray
    rejected_row_indices: np.ndarray
    retained_row_mask: np.ndarray


def rank_sparse_residual_rejections(
    residual_scores: np.ndarray,
    *,
    rejected_row_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select a fixed residual fraction with a stable original-index tie break."""

    scores = np.asarray(residual_scores, dtype=np.float64)
    if (
        scores.ndim != 1
        or scores.size < 12
        or not np.all(np.isfinite(scores))
        or np.any(scores < 0.0)
        or not np.isfinite(rejected_row_fraction)
        or rejected_row_fraction <= 0.0
        or rejected_row_fraction >= 0.5
    ):
        raise ValueError("residual scores and rejected-row fraction are invalid")
    rejected_count = int(math.floor(scores.size * rejected_row_fraction))
    if rejected_count < 1 or scores.size - rejected_count < 12:
        raise ValueError("rejected-row fraction leaves invalid retained support")
    original_indices = np.arange(scores.size, dtype=np.int64)
    ranked = np.lexsort((original_indices, -scores))
    rejected = np.sort(ranked[:rejected_count].astype(np.int64, copy=False))
    retained = np.ones(scores.size, dtype=bool)
    retained[rejected] = False
    rejected.setflags(write=False)
    retained.setflags(write=False)
    return rejected, retained


def fit_two_stage_sparse_rejection_operator(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    *,
    rejected_row_fraction: float,
    identity_mixture: float = 0.25,
    restart_count: int = 3,
    maximum_function_evaluations: int = 2500,
    function_tolerance: float = 1e-11,
    parameter_tolerance: float = 1e-11,
    gradient_tolerance: float = 1e-11,
    stage_one_loss_scale: float = 0.005,
    stage_two_loss_scale: float = 0.005,
    seed: int = 20250728,
) -> TwoStagePositiveFilmFitResult:
    """Fit soft-L1, reject fixed high-residual rows, then refit with linear loss."""

    source = np.asarray(source_rgb, dtype=np.float64)
    target = np.asarray(target_rgb, dtype=np.float64)
    source_before = source.copy()
    target_before = target.copy()
    common = {
        "model": "two_matrix",
        "identity_mixture": identity_mixture,
        "restart_count": restart_count,
        "maximum_function_evaluations": maximum_function_evaluations,
        "function_tolerance": function_tolerance,
        "parameter_tolerance": parameter_tolerance,
        "gradient_tolerance": gradient_tolerance,
        "seed": seed,
    }
    stage_one = fit_positive_film_response_operator(
        source,
        target,
        loss="soft_l1",
        loss_scale=stage_one_loss_scale,
        **common,
    )
    residual = stage_one.operator.apply(source) - target
    scores = np.sqrt(np.sum(np.square(residual), axis=1))
    rejected, retained = rank_sparse_residual_rejections(
        scores, rejected_row_fraction=rejected_row_fraction
    )
    stage_two = fit_positive_film_response_operator(
        source[retained],
        target[retained],
        loss="linear",
        loss_scale=stage_two_loss_scale,
        **common,
    )
    if not np.array_equal(source, source_before) or not np.array_equal(
        target, target_before
    ):
        raise RuntimeError("two-stage paired fitting mutated its input arrays")
    scores.setflags(write=False)
    rejected.setflags(write=False)
    retained.setflags(write=False)
    return TwoStagePositiveFilmFitResult(
        stage_one=stage_one,
        stage_two=stage_two,
        residual_scores=scores,
        rejected_row_indices=rejected,
        retained_row_mask=retained,
    )


__all__ = [
    "TwoStagePositiveFilmFitResult",
    "fit_two_stage_sparse_rejection_operator",
    "rank_sparse_residual_rejections",
]
