"""Exact parallel target construction with bounded shared-slice worker output."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.eval.circular_hue_fraction_transport import _hue_fraction
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.nonexpansive_fraction_transport import NonexpansiveFractionTransportError
from src.eval.nonexpansive_fraction_transport_external_sort import (
    _external_fraction_map,
)


def _bounds(height: int, row_chunk: int) -> list[tuple[int, int]]:
    return [
        (y0, min(height, y0 + row_chunk))
        for y0 in range(0, height, row_chunk)
    ]


def _bounded_parallel_map(
    executor: ThreadPoolExecutor,
    function: object,
    bounds: list[tuple[int, int]],
    *,
    workers: int,
) -> list[object]:
    results: list[object] = []
    for start in range(0, len(bounds), workers):
        batch = bounds[start : start + workers]
        futures = [executor.submit(function, item) for item in batch]
        results.extend(future.result() for future in futures)
    return results


def nonexpansive_fraction_transport_target_shared_parallel(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_valid_fraction: float = 1e-4,
    fraction_knots: int = 257,
    maximum_fraction_slope: float = 1.0,
    row_chunk: int = 128,
    scratch_root: Path | None = None,
    workers: int = 4,
) -> np.ndarray:
    """Build the exact target while workers write only disjoint shared slices."""
    del boundary_epsilon
    base = np.asarray(safe_base_linear)
    ao6 = np.asarray(ao6_linear)
    w = np.asarray(weights, dtype=np.float64)
    if (
        base.dtype != np.float32
        or ao6.dtype != np.float32
        or base.shape != ao6.shape
        or base.ndim != 3
        or base.shape[-1] != 3
        or w.shape != (3,)
        or not np.isfinite(base).all()
        or not np.isfinite(ao6).all()
        or abs(float(np.sum(w)) - 1.0) > 1e-12
        or isinstance(row_chunk, bool)
        or not isinstance(row_chunk, int)
        or row_chunk <= 0
        or isinstance(workers, bool)
        or not isinstance(workers, int)
        or workers <= 0
        or (scratch_root is not None and not Path(scratch_root).is_dir())
    ):
        raise NonexpansiveFractionTransportError("CB72 input drift")

    height, width, _ = base.shape
    basis = _plane_basis(w)
    base_luma = np.empty((height, width), dtype=np.float64)
    base_fraction = np.empty((height, width), dtype=np.float64)
    ao6_fraction = np.empty((height, width), dtype=np.float64)
    base_valid = np.empty((height, width), dtype=bool)
    ao6_valid = np.empty((height, width), dtype=bool)
    dot_terms = np.empty(height * width, dtype=np.float64)
    cross_terms = np.empty(height * width, dtype=np.float64)
    all_bounds = _bounds(height, row_chunk)

    def analyse(
        bounds: tuple[int, int],
        base_fraction_out: np.ndarray = base_fraction,
        ao6_fraction_out: np.ndarray = ao6_fraction,
        base_valid_out: np.ndarray = base_valid,
        ao6_valid_out: np.ndarray = ao6_valid,
    ) -> tuple[np.ndarray, np.ndarray]:
        y0, y1 = bounds
        base_rows = base[y0:y1].astype(np.float64)
        ao6_rows = ao6[y0:y1].astype(np.float64)
        b_luma, b_unit, b_fraction, b_valid = _hue_fraction(base_rows, w, basis)
        _, a_unit, a_fraction, a_valid = _hue_fraction(ao6_rows, w, basis)
        base_luma[y0:y1] = b_luma
        base_fraction_out[y0:y1] = b_fraction
        ao6_fraction_out[y0:y1] = a_fraction
        base_valid_out[y0:y1] = b_valid
        ao6_valid_out[y0:y1] = a_valid
        fit = (
            b_valid
            & a_valid
            & (b_fraction >= minimum_valid_fraction)
            & (a_fraction >= minimum_valid_fraction)
        )
        b_fit = b_unit[fit]
        a_fit = a_unit[fit]
        return (
            b_fit[:, 0] * a_fit[:, 0] + b_fit[:, 1] * a_fit[:, 1],
            b_fit[:, 0] * a_fit[:, 1] - b_fit[:, 1] * a_fit[:, 0],
        )

    offset = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for dots, crosses in _bounded_parallel_map(
            executor, analyse, all_bounds, workers=workers
        ):
            count = int(dots.size)
            dot_terms[offset : offset + count] = dots
            cross_terms[offset : offset + count] = crosses
            offset += count
    if offset < 2:
        raise NonexpansiveFractionTransportError("CB72 hue population failed")
    angle = float(
        np.arctan2(
            float(np.sum(cross_terms[:offset])), float(np.sum(dot_terms[:offset]))
        )
    )
    del dot_terms, cross_terms

    mapped_fraction, _ = _external_fraction_map(
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
        fraction_knots=fraction_knots,
        maximum_fraction_slope=maximum_fraction_slope,
        row_chunk=row_chunk,
        scratch_root=scratch_root,
    )
    del base_fraction, ao6_fraction, base_valid, ao6_valid

    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    target = np.empty(base.shape, dtype=np.float32)

    def synthesize(bounds: tuple[int, int]) -> None:
        y0, y1 = bounds
        base_rows = base[y0:y1].astype(np.float64)
        _, unit, _, _ = _hue_fraction(base_rows, w, basis)
        rotated_unit = np.empty_like(unit)
        rotated_unit[..., 0] = cosine * unit[..., 0] - sine * unit[..., 1]
        rotated_unit[..., 1] = sine * unit[..., 0] + cosine * unit[..., 1]
        rotated_rgb = rotated_unit @ basis.T
        luma_rows = base_luma[y0:y1]
        maximum = _maximum_chroma_magnitude(luma_rows, rotated_rgb)
        fraction_rows = mapped_fraction[y0:y1]
        target_chroma = np.zeros_like(rotated_rgb)
        valid = fraction_rows > 0.0
        target_chroma[valid] = (
            fraction_rows[valid, None] * maximum[valid, None] * rotated_rgb[valid]
        )
        target[y0:y1] = np.asarray(
            luma_rows[..., None] + target_chroma, dtype=np.float32
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        _bounded_parallel_map(executor, synthesize, all_bounds, workers=workers)
    if (
        not np.isfinite(target).all()
        or np.min(target) < -1e-7
        or np.max(target) > 1.0 + 1e-7
    ):
        raise NonexpansiveFractionTransportError("CB72 target invariant failed")
    return target


__all__ = ["nonexpansive_fraction_transport_target_shared_parallel"]
