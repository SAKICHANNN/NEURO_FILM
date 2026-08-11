"""Output-exact, lower-residency materialization of the CB32 target."""

from __future__ import annotations

import numpy as np

from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.monotone_fraction_quantile_transport import _circular_rotation_angle
from src.eval.nonexpansive_fraction_transport import (
    NonexpansiveFractionTransportError,
    _nonexpansive_fraction_map,
)


def _hue_fraction_row_materialized(
    image: np.ndarray,
    weights: np.ndarray,
    basis: np.ndarray,
    *,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Match legacy float64 facts without full-frame masked temporaries."""
    luma = np.sum(image * weights, axis=-1)
    chroma = image - luma[..., None]
    xy = chroma @ basis
    del chroma
    magnitude = np.linalg.norm(xy, axis=-1)
    valid = magnitude > 1e-12
    unit_xy = np.zeros_like(xy)
    for y0 in range(0, image.shape[0], row_chunk):
        y1 = min(image.shape[0], y0 + row_chunk)
        valid_rows = valid[y0:y1]
        unit_xy[y0:y1][valid_rows] = (
            xy[y0:y1][valid_rows] / magnitude[y0:y1][valid_rows, None]
        )
    del xy
    maximum = np.empty_like(luma)
    for y0 in range(0, image.shape[0], row_chunk):
        y1 = min(image.shape[0], y0 + row_chunk)
        unit_rgb = unit_xy[y0:y1] @ basis.T
        maximum[y0:y1] = _maximum_chroma_magnitude(luma[y0:y1], unit_rgb)
    fraction = np.zeros_like(luma)
    for y0 in range(0, image.shape[0], row_chunk):
        y1 = min(image.shape[0], y0 + row_chunk)
        valid_rows = valid[y0:y1]
        fraction[y0:y1][valid_rows] = (
            magnitude[y0:y1][valid_rows] / maximum[y0:y1][valid_rows]
        )
    return luma, unit_xy, fraction, valid


def nonexpansive_fraction_transport_target_row_materialized(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_valid_fraction: float = 1e-4,
    fraction_knots: int = 257,
    maximum_fraction_slope: float = 1.0,
    row_chunk: int = 64,
) -> np.ndarray:
    """Preserve the float64 global fit while materializing final rows in bounds."""
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
    ):
        raise NonexpansiveFractionTransportError("CB60 input drift")

    basis = _plane_basis(w)
    base64 = base.astype(np.float64)
    base_luma, base_unit, base_fraction, base_valid = _hue_fraction_row_materialized(
        base64, w, basis, row_chunk=row_chunk
    )
    del base64
    ao664 = ao6.astype(np.float64)
    _, ao6_unit, ao6_fraction, ao6_valid = _hue_fraction_row_materialized(
        ao664, w, basis, row_chunk=row_chunk
    )
    del ao664

    angle = _circular_rotation_angle(
        base_unit,
        ao6_unit,
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    mapped_fraction, _ = _nonexpansive_fraction_map(
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
        fraction_knots=fraction_knots,
        maximum_fraction_slope=maximum_fraction_slope,
    )
    del ao6_unit, ao6_fraction, ao6_valid, base_fraction, base_valid

    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    target = np.empty(base.shape, dtype=np.float32)
    for y0 in range(0, base.shape[0], row_chunk):
        y1 = min(base.shape[0], y0 + row_chunk)
        unit_rows = base_unit[y0:y1]
        rotated_unit = np.empty_like(unit_rows)
        rotated_unit[..., 0] = cosine * unit_rows[..., 0] - sine * unit_rows[..., 1]
        rotated_unit[..., 1] = sine * unit_rows[..., 0] + cosine * unit_rows[..., 1]
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
        raise NonexpansiveFractionTransportError("CB60 target invariant failed")
    return target


__all__ = ["nonexpansive_fraction_transport_target_row_materialized"]
