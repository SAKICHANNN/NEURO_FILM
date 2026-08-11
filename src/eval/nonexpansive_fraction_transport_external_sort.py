"""Output-exact external-sort target construction for CB64."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from src.eval.circular_hue_fraction_transport import _hue_fraction
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.nonexpansive_fraction_transport import NonexpansiveFractionTransportError


def _count_sorted_groups(values: np.memmap, *, scan_chunk: int) -> int:
    groups = 1
    previous = float(values[0])
    for start in range(1, values.size, scan_chunk):
        chunk = np.asarray(values[start : start + scan_chunk])
        groups += int(chunk[0] != previous)
        groups += int(np.count_nonzero(chunk[1:] != chunk[:-1]))
        previous = float(chunk[-1])
    return groups


def _selected_group_bounds(
    values: np.memmap,
    selected: np.ndarray,
    *,
    group_count: int,
    scan_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    requested = np.unique(
        np.concatenate((selected, np.minimum(selected + 1, group_count)))
    )
    positions: dict[int, int] = {0: 0, group_count: int(values.size)}
    groups_before = 1
    previous = float(values[0])
    for start in range(1, values.size, scan_chunk):
        chunk = np.asarray(values[start : start + scan_chunk])
        changes = np.flatnonzero(
            np.concatenate(
                ((np.asarray([chunk[0] != previous], dtype=bool)), chunk[1:] != chunk[:-1])
            )
        )
        if changes.size:
            group_ids = groups_before + np.arange(changes.size, dtype=np.int64)
            left = int(np.searchsorted(requested, group_ids[0]))
            right = int(np.searchsorted(requested, group_ids[-1], side="right"))
            for requested_id in requested[left:right]:
                local = int(requested_id - group_ids[0])
                positions[int(requested_id)] = start + int(changes[local])
            groups_before += int(changes.size)
        previous = float(chunk[-1])
    starts = np.asarray([positions[int(index)] for index in selected], dtype=np.int64)
    ends = np.asarray([positions[int(index + 1)] for index in selected], dtype=np.int64)
    unique_values = np.asarray(values[starts], dtype=np.float64)
    return unique_values, starts, ends


def _sorted_quantiles(
    sorted_values: np.memmap, quantiles: np.ndarray
) -> np.ndarray:
    position = quantiles * float(sorted_values.size - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, sorted_values.size - 1)
    alpha = position - lower
    return (1.0 - alpha) * sorted_values[lower] + alpha * sorted_values[upper]


def _external_fraction_map(
    base_fraction: np.ndarray,
    ao6_fraction: np.ndarray,
    base_valid: np.ndarray,
    ao6_valid: np.ndarray,
    *,
    minimum_valid_fraction: float,
    fraction_knots: int,
    maximum_fraction_slope: float,
    row_chunk: int,
    scratch_root: Path | None,
) -> tuple[np.ndarray, dict[str, float | int]]:
    height = base_fraction.shape[0]
    base_count = 0
    ao6_count = 0
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        base_count += int(
            np.count_nonzero(
                base_valid[y0:y1]
                & (base_fraction[y0:y1] >= minimum_valid_fraction)
            )
        )
        ao6_count += int(
            np.count_nonzero(
                ao6_valid[y0:y1]
                & (ao6_fraction[y0:y1] >= minimum_valid_fraction)
            )
        )
    if base_count < 2 or ao6_count < 2 or fraction_knots < 3:
        raise NonexpansiveFractionTransportError("CB64 fraction population failed")

    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="cb64-") as temporary:
        temporary_path = Path(temporary)
        base_values = np.memmap(
            temporary_path / "base-fractions.f64",
            mode="w+",
            dtype=np.float64,
            shape=(base_count,),
        )
        ao6_values = np.memmap(
            temporary_path / "ao6-fractions.f64",
            mode="w+",
            dtype=np.float64,
            shape=(ao6_count,),
        )
        base_offset = 0
        ao6_offset = 0
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            base_mask = base_valid[y0:y1] & (
                base_fraction[y0:y1] >= minimum_valid_fraction
            )
            ao6_mask = ao6_valid[y0:y1] & (
                ao6_fraction[y0:y1] >= minimum_valid_fraction
            )
            base_rows = base_fraction[y0:y1][base_mask]
            ao6_rows = ao6_fraction[y0:y1][ao6_mask]
            base_values[base_offset : base_offset + base_rows.size] = base_rows
            ao6_values[ao6_offset : ao6_offset + ao6_rows.size] = ao6_rows
            base_offset += int(base_rows.size)
            ao6_offset += int(ao6_rows.size)
        base_values.sort(kind="quicksort")
        ao6_values.sort(kind="quicksort")

        group_count = _count_sorted_groups(base_values, scan_chunk=1_048_576)
        selected_count = min(group_count, fraction_knots - 1)
        selected = np.rint(
            np.linspace(0, group_count - 1, selected_count, dtype=np.float64)
        ).astype(np.int64)
        selected = np.unique(selected)
        unique_values, starts, ends = _selected_group_bounds(
            base_values,
            selected,
            group_count=group_count,
            scan_chunk=1_048_576,
        )
        group_quantiles = (starts.astype(np.float64) + ends.astype(np.float64)) / (
            2.0 * float(base_count)
        )
        knot_x = np.concatenate((np.asarray([0.0]), unique_values))
        raw_y = np.concatenate(
            (np.asarray([0.0]), _sorted_quantiles(ao6_values, group_quantiles))
        )
        delta_x = np.diff(knot_x)
        if np.any(delta_x <= 0.0):
            raise NonexpansiveFractionTransportError("CB64 knot order failed")
        delta_y = np.maximum(np.diff(raw_y), 0.0)
        limited_delta = np.minimum(delta_y, maximum_fraction_slope * delta_x)
        knot_y = np.concatenate((np.asarray([0.0]), np.cumsum(limited_delta)))
        slopes = limited_delta / delta_x

        mapped = np.zeros_like(base_fraction)
        for y0 in range(0, height, row_chunk):
            y1 = min(height, y0 + row_chunk)
            mask = base_valid[y0:y1] & (
                base_fraction[y0:y1] >= minimum_valid_fraction
            )
            mapped_rows = mapped[y0:y1]
            mapped_rows[mask] = np.interp(base_fraction[y0:y1][mask], knot_x, knot_y)
        facts: dict[str, float | int] = {
            "knot_count": int(knot_x.size),
            "maximum_observed_fraction_slope": float(np.max(slopes)),
            "maximum_mapped_fraction": float(np.max(mapped)),
        }
        del base_values, ao6_values
    if (
        not np.isfinite(mapped).all()
        or np.min(mapped) < 0.0
        or np.max(mapped) > 1.0 + 1e-12
        or facts["maximum_observed_fraction_slope"]
        > maximum_fraction_slope + 1e-12
    ):
        raise NonexpansiveFractionTransportError("CB64 nonexpansive invariant failed")
    return mapped, facts


def nonexpansive_fraction_transport_target_external_sorted(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_valid_fraction: float = 1e-4,
    fraction_knots: int = 257,
    maximum_fraction_slope: float = 1.0,
    row_chunk: int = 64,
    scratch_root: Path | None = None,
) -> np.ndarray:
    """Build the frozen target with external fraction sorting and bounded rows."""
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
        or (scratch_root is not None and not Path(scratch_root).is_dir())
    ):
        raise NonexpansiveFractionTransportError("CB64 input drift")

    height, width, _ = base.shape
    basis = _plane_basis(w)
    base_luma = np.empty((height, width), dtype=np.float64)
    base_fraction = np.empty((height, width), dtype=np.float64)
    ao6_fraction = np.empty((height, width), dtype=np.float64)
    base_valid = np.empty((height, width), dtype=bool)
    ao6_valid = np.empty((height, width), dtype=bool)
    dot_terms = np.empty(height * width, dtype=np.float64)
    cross_terms = np.empty(height * width, dtype=np.float64)
    offset = 0
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        base_rows = base[y0:y1].astype(np.float64)
        ao6_rows = ao6[y0:y1].astype(np.float64)
        b_luma, b_unit, b_fraction, b_valid = _hue_fraction(base_rows, w, basis)
        _, a_unit, a_fraction, a_valid = _hue_fraction(ao6_rows, w, basis)
        base_luma[y0:y1] = b_luma
        base_fraction[y0:y1] = b_fraction
        ao6_fraction[y0:y1] = a_fraction
        base_valid[y0:y1] = b_valid
        ao6_valid[y0:y1] = a_valid
        fit = (
            b_valid
            & a_valid
            & (b_fraction >= minimum_valid_fraction)
            & (a_fraction >= minimum_valid_fraction)
        )
        b_fit = b_unit[fit]
        a_fit = a_unit[fit]
        count = b_fit.shape[0]
        dot_terms[offset : offset + count] = (
            b_fit[:, 0] * a_fit[:, 0] + b_fit[:, 1] * a_fit[:, 1]
        )
        cross_terms[offset : offset + count] = (
            b_fit[:, 0] * a_fit[:, 1] - b_fit[:, 1] * a_fit[:, 0]
        )
        offset += count
    if offset < 2:
        raise NonexpansiveFractionTransportError("CB64 hue population failed")
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
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
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
    if (
        not np.isfinite(target).all()
        or np.min(target) < -1e-7
        or np.max(target) > 1.0 + 1e-7
    ):
        raise NonexpansiveFractionTransportError("CB64 target invariant failed")
    return target


__all__ = ["nonexpansive_fraction_transport_target_external_sorted"]
