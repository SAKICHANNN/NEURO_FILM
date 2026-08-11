"""Exact analytic selector without retaining the unused full luma-error plane."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.analytic_y_chromaticity_streaming import (
    _gradient_ratio_streamed,
    _inversion_fraction_streamed,
)
from src.eval.analytic_y_chromaticity_transport import (
    _bounded_same_y_reconstruction,
    _normalized_zero_y_chromaticity,
)
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.direct_lstar_monotone_tone_transport import _lstar_to_neutral_linear
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_minmax import _anchored_curve


def _candidate_for_dose_compact(
    source: np.ndarray,
    target: np.ndarray,
    *,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose: float,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    candidate = np.empty_like(source)
    gamut_scale = np.empty(source.shape[:2], dtype=np.float32)
    maximum_luminance_error = 0.0
    lower = float(
        np.float32(boundary_epsilon) + 4 * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper = float(upper_edge - 4 * np.spacing(upper_edge))
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        source_rows = source[y0:y1]
        target_rows = target[y0:y1]
        source_lstar = linear_rgb_to_lab(
            source_rows, working_space="linear_srgb"
        )[..., 0]
        mapped_lstar = 100.0 * _anchored_curve(
            source_lstar.astype(np.float64) / 100.0,
            curve,
            strength=strength,
            epsilon=boundary_epsilon,
        )
        desired_y = np.clip(_lstar_to_neutral_linear(mapped_lstar), lower, upper)
        source_chroma = _normalized_zero_y_chromaticity(
            source_rows, boundary_epsilon
        )
        target_chroma = _normalized_zero_y_chromaticity(
            target_rows, boundary_epsilon
        )
        chroma = source_chroma + dose * (target_chroma - source_chroma)
        chroma -= (chroma @ LEGACY_LAB_Y_WEIGHTS)[..., None]
        candidate_rows, scale_rows, error_rows = _bounded_same_y_reconstruction(
            desired_y,
            chroma,
            boundary_epsilon=boundary_epsilon,
        )
        candidate[y0:y1] = candidate_rows
        gamut_scale[y0:y1] = scale_rows
        maximum_luminance_error = max(
            maximum_luminance_error, float(np.max(np.abs(error_rows)))
        )
    return candidate, gamut_scale, maximum_luminance_error


def select_analytic_y_chromaticity_candidate_compact(
    source_linear: np.ndarray,
    full_target_linear: np.ndarray,
    *,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose_grid: list[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
    row_chunk: int,
    scratch_root: Path | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Execute frozen selection while retaining no full float64 error plane."""
    source = np.asarray(source_linear)
    target = np.asarray(full_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or source.ndim != 3
        or source.shape[2] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or isinstance(row_chunk, bool)
        or not isinstance(row_chunk, int)
        or row_chunk <= 0
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise CharacteristicLstarTransportError("CB71 selector input drift")
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="cb71-") as temporary:
        scratch_dir = Path(temporary)
        for dose in doses:
            candidate, gamut_scale, maximum_luminance_error = (
                _candidate_for_dose_compact(
                    source,
                    target,
                    curve=curve,
                    strength=strength,
                    boundary_epsilon=boundary_epsilon,
                    dose=float(dose),
                    row_chunk=row_chunk,
                )
            )
            boundary_count = 0
            for y0 in range(0, source.shape[0], row_chunk):
                y1 = min(source.shape[0], y0 + row_chunk)
                boundary = (
                    (candidate[y0:y1] <= boundary_epsilon)
                    & (source[y0:y1] > boundary_epsilon)
                ) | (
                    (candidate[y0:y1] >= 1.0 - boundary_epsilon)
                    & (source[y0:y1] < 1.0 - boundary_epsilon)
                )
                boundary_count += int(np.count_nonzero(np.any(boundary, axis=-1)))
            boundary_fraction = boundary_count / (source.shape[0] * source.shape[1])
            gradient_ratio = _gradient_ratio_streamed(
                source,
                candidate,
                row_chunk=row_chunk,
                scratch_dir=scratch_dir,
            )
            inversion_fraction = _inversion_fraction_streamed(
                source,
                candidate,
                epsilon=lstar_order_epsilon,
                row_chunk=row_chunk,
            )
            if (
                boundary_fraction == 0.0
                and gradient_ratio <= maximum_gradient_ratio
                and inversion_fraction <= maximum_lstar_inversion_fraction
            ):
                return candidate, {
                    "characteristic_strength": float(strength),
                    "global_dose": float(dose),
                    "selected_gradient_ratio": float(gradient_ratio),
                    "selected_lstar_inversion_fraction": float(inversion_fraction),
                    "selected_new_boundary_fraction": float(boundary_fraction),
                    "median_gamut_scale": float(np.median(gamut_scale)),
                    "fraction_gamut_scale_below_0p8": float(
                        np.mean(gamut_scale < 0.8)
                    ),
                    "maximum_luminance_error": maximum_luminance_error,
                }
    raise CharacteristicLstarTransportError(
        "CB71 dose grid has no safe analytic Y/chromaticity candidate"
    )


__all__ = ["select_analytic_y_chromaticity_candidate_compact"]
