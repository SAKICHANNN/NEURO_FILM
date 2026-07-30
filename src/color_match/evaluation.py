"""Known-operator evaluation for reference-match algorithm promotion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.color_engine import linear_rgb_to_lab
from src.preprocess import WorkingImage

from .contracts import ReferenceMatchContractError

_FLOAT32_GAMUT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class KnownOperatorSampleMetrics:
    """One source/candidate comparison against a same-content known target."""

    sample_id: str
    source_target_delta_e76_median: float
    source_target_delta_e76_p95: float
    candidate_target_delta_e76_median: float
    candidate_target_delta_e76_p95: float
    median_improvement_fraction: float
    candidate_new_boundary_fraction: float


@dataclass(frozen=True)
class KnownOperatorBatchMetrics:
    """Aggregate evidence without converting metric success into an aesthetic claim."""

    samples: tuple[KnownOperatorSampleMetrics, ...]
    improved_sample_count: int
    regressed_sample_count: int
    improvement_rate: float
    median_improvement_fraction: float
    worst_improvement_fraction: float
    maximum_new_boundary_fraction: float


def aggregate_known_operator_samples(
    samples: Iterable[KnownOperatorSampleMetrics],
) -> KnownOperatorBatchMetrics:
    """Aggregate already evaluated rows without retaining image tensors."""

    if isinstance(samples, (str, bytes, KnownOperatorSampleMetrics)):
        raise ReferenceMatchContractError(
            "samples must be an iterable of KnownOperatorSampleMetrics"
        )
    try:
        rows = tuple(samples)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "samples must be an iterable of KnownOperatorSampleMetrics"
        ) from exc
    if not rows or any(
        not isinstance(row, KnownOperatorSampleMetrics) for row in rows
    ):
        raise ReferenceMatchContractError(
            "samples must contain at least one KnownOperatorSampleMetrics"
        )
    ids = tuple(row.sample_id for row in rows)
    if any(
        not isinstance(sample_id, str) or not sample_id.strip()
        for sample_id in ids
    ):
        raise ReferenceMatchContractError("sample IDs must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise ReferenceMatchContractError("sample IDs must be unique")
    improvements = np.asarray(
        [row.median_improvement_fraction for row in rows],
        dtype=np.float64,
    )
    boundaries = np.asarray(
        [row.candidate_new_boundary_fraction for row in rows],
        dtype=np.float64,
    )
    if (
        not np.isfinite(improvements).all()
        or not np.isfinite(boundaries).all()
        or np.any((boundaries < 0.0) | (boundaries > 1.0))
    ):
        raise ReferenceMatchContractError(
            "known-operator samples must contain finite valid metrics"
        )
    improved = int(np.count_nonzero(improvements > 0.0))
    regressed = int(np.count_nonzero(improvements < 0.0))
    return KnownOperatorBatchMetrics(
        samples=rows,
        improved_sample_count=improved,
        regressed_sample_count=regressed,
        improvement_rate=float(improved / len(rows)),
        median_improvement_fraction=float(np.median(improvements)),
        worst_improvement_fraction=float(np.min(improvements)),
        maximum_new_boundary_fraction=float(np.max(boundaries)),
    )


def _images(
    values: Iterable[WorkingImage],
    label: str,
) -> tuple[WorkingImage, ...]:
    if isinstance(values, (str, bytes, WorkingImage)):
        raise ReferenceMatchContractError(
            f"{label} must be an iterable of WorkingImage values"
        )
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            f"{label} must be an iterable of WorkingImage values"
        ) from exc
    if not result or any(not isinstance(value, WorkingImage) for value in result):
        raise ReferenceMatchContractError(
            f"{label} must contain at least one WorkingImage"
        )
    return result


def _validate_triplet(
    source: WorkingImage,
    target: WorkingImage,
    candidate: WorkingImage,
    *,
    sample_id: str,
) -> None:
    images = (source, target, candidate)
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ReferenceMatchContractError("sample IDs must be non-empty strings")
    if any(image.transfer_state != "display_linear" for image in images):
        raise ReferenceMatchContractError(
            f"known-operator sample {sample_id} must be display-linear"
        )
    if any(image.working_space != "linear_srgb" for image in images):
        raise ReferenceMatchContractError(
            f"known-operator sample {sample_id} must use linear_srgb"
        )
    if target.pixels.shape != source.pixels.shape or candidate.pixels.shape != source.pixels.shape:
        raise ReferenceMatchContractError(
            f"known-operator sample {sample_id} shapes must match"
        )
    if any(
        not np.isfinite(image.pixels).all()
        or float(np.min(image.pixels)) < -_FLOAT32_GAMUT_TOLERANCE
        or float(np.max(image.pixels))
        > 1.0 + _FLOAT32_GAMUT_TOLERANCE
        for image in images
    ):
        raise ReferenceMatchContractError(
            f"known-operator sample {sample_id} must be finite and within "
            "float32 gamut tolerance"
        )


def _delta_e76(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(first - second, axis=-1)


def evaluate_known_operator_batch(
    sources: Iterable[WorkingImage],
    targets: Iterable[WorkingImage],
    candidates: Iterable[WorkingImage],
    *,
    sample_ids: Iterable[str],
    boundary_epsilon: float = 1.0 / 65535.0,
) -> KnownOperatorBatchMetrics:
    """Compare a matcher with neutral input against known same-look targets.

    Targets are evaluation-only. This function does not fit or render a recipe.
    """

    source_items = _images(sources, "sources")
    target_items = _images(targets, "targets")
    candidate_items = _images(candidates, "candidates")
    try:
        ids = tuple(sample_ids)
    except TypeError as exc:
        raise ReferenceMatchContractError(
            "sample_ids must be an iterable of strings"
        ) from exc
    count = len(source_items)
    if len(target_items) != count or len(candidate_items) != count or len(ids) != count:
        raise ReferenceMatchContractError(
            "sources, targets, candidates and sample_ids must have equal length"
        )
    if len(set(ids)) != len(ids):
        raise ReferenceMatchContractError("sample_ids must be unique")
    if (
        isinstance(boundary_epsilon, bool)
        or not isinstance(boundary_epsilon, (int, float))
        or not np.isfinite(float(boundary_epsilon))
        or float(boundary_epsilon) < 0.0
        or float(boundary_epsilon) >= 0.5
    ):
        raise ReferenceMatchContractError(
            "boundary_epsilon must be finite and within [0, 0.5)"
        )
    epsilon = float(boundary_epsilon)

    rows: list[KnownOperatorSampleMetrics] = []
    for sample_id, source, target, candidate in zip(
        ids,
        source_items,
        target_items,
        candidate_items,
    ):
        _validate_triplet(source, target, candidate, sample_id=sample_id)
        source_lab = linear_rgb_to_lab(
            source.pixels,
            working_space="linear_srgb",
        )
        target_lab = linear_rgb_to_lab(
            target.pixels,
            working_space="linear_srgb",
        )
        candidate_lab = linear_rgb_to_lab(
            candidate.pixels,
            working_space="linear_srgb",
        )
        source_delta = _delta_e76(source_lab, target_lab)
        candidate_delta = _delta_e76(candidate_lab, target_lab)
        source_median = float(np.median(source_delta))
        candidate_median = float(np.median(candidate_delta))
        improvement = (
            0.0
            if source_median <= 1e-12
            else (source_median - candidate_median) / source_median
        )
        source_boundary = np.any(
            (source.pixels <= epsilon) | (source.pixels >= 1.0 - epsilon),
            axis=-1,
        )
        candidate_boundary = np.any(
            (candidate.pixels <= epsilon)
            | (candidate.pixels >= 1.0 - epsilon),
            axis=-1,
        )
        new_boundary = candidate_boundary & ~source_boundary
        rows.append(
            KnownOperatorSampleMetrics(
                sample_id=sample_id,
                source_target_delta_e76_median=source_median,
                source_target_delta_e76_p95=float(
                    np.percentile(source_delta, 95)
                ),
                candidate_target_delta_e76_median=candidate_median,
                candidate_target_delta_e76_p95=float(
                    np.percentile(candidate_delta, 95)
                ),
                median_improvement_fraction=float(improvement),
                candidate_new_boundary_fraction=float(
                    np.mean(new_boundary, dtype=np.float64)
                ),
            )
        )
    return aggregate_known_operator_samples(rows)


__all__ = [
    "KnownOperatorBatchMetrics",
    "KnownOperatorSampleMetrics",
    "aggregate_known_operator_samples",
    "evaluate_known_operator_batch",
]
