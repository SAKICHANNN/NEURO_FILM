"""Output-exact row-streamed execution for the CB50 analytic transport."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.analytic_y_chromaticity_transport import (
    _bounded_same_y_reconstruction,
    _normalized_zero_y_chromaticity,
)
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.direct_lstar_monotone_tone_transport import _lstar_to_neutral_linear
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_minmax import _anchored_curve


def _gradient_magnitudes(image: np.ndarray, output: np.memmap, *, row_chunk: int) -> None:
    height, width, _ = image.shape
    offset = 0
    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        values = np.linalg.norm(np.diff(image[y0:y1], axis=1), axis=-1).reshape(-1)
        output[offset : offset + values.size] = values
        offset += values.size
    for y0 in range(0, height - 1, row_chunk):
        y1 = min(height - 1, y0 + row_chunk)
        values = np.linalg.norm(image[y0 + 1 : y1 + 1] - image[y0:y1], axis=-1).reshape(-1)
        output[offset : offset + values.size] = values
        offset += values.size
    if offset != output.size or width < 2:
        raise CharacteristicLstarTransportError("CB54 gradient scratch size drift")
    output.flush()


def _gradient_ratio_streamed(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    row_chunk: int,
    scratch_dir: Path,
) -> float:
    height, width, _ = source.shape
    count = height * (width - 1) + (height - 1) * width
    source_path = scratch_dir / "source_gradients.f32"
    candidate_path = scratch_dir / "candidate_gradients.f32"
    source_values = np.memmap(source_path, mode="w+", dtype=np.float32, shape=(count,))
    candidate_values = np.memmap(candidate_path, mode="w+", dtype=np.float32, shape=(count,))
    try:
        _gradient_magnitudes(source, source_values, row_chunk=row_chunk)
        _gradient_magnitudes(candidate, candidate_values, row_chunk=row_chunk)
        source_q = float(np.quantile(source_values, 0.999))
        candidate_q = float(np.quantile(candidate_values, 0.999))
        return candidate_q / max(source_q, 1e-12)
    finally:
        del source_values
        del candidate_values


def _inversion_fraction_streamed(
    source: np.ndarray,
    candidate: np.ndarray,
    *,
    epsilon: float,
    row_chunk: int,
) -> float:
    inversions = 0
    comparisons = 0
    previous_source: np.ndarray | None = None
    previous_candidate: np.ndarray | None = None
    for y0 in range(0, source.shape[0], row_chunk):
        y1 = min(source.shape[0], y0 + row_chunk)
        source_lstar = linear_rgb_to_lab(source[y0:y1], working_space="linear_srgb")[..., 0]
        candidate_lstar = linear_rgb_to_lab(candidate[y0:y1], working_space="linear_srgb")[..., 0]
        source_horizontal = np.diff(source_lstar, axis=1)
        candidate_horizontal = np.diff(candidate_lstar, axis=1)
        material = np.abs(source_horizontal) > epsilon
        comparisons += int(np.count_nonzero(material))
        inversions += int(
            np.count_nonzero(
                material & (source_horizontal * candidate_horizontal < -(epsilon**2))
            )
        )
        source_vertical = np.diff(source_lstar, axis=0)
        candidate_vertical = np.diff(candidate_lstar, axis=0)
        material = np.abs(source_vertical) > epsilon
        comparisons += int(np.count_nonzero(material))
        inversions += int(
            np.count_nonzero(
                material & (source_vertical * candidate_vertical < -(epsilon**2))
            )
        )
        if previous_source is not None and previous_candidate is not None:
            source_boundary = source_lstar[0] - previous_source
            candidate_boundary = candidate_lstar[0] - previous_candidate
            material = np.abs(source_boundary) > epsilon
            comparisons += int(np.count_nonzero(material))
            inversions += int(
                np.count_nonzero(
                    material & (source_boundary * candidate_boundary < -(epsilon**2))
                )
            )
        previous_source = source_lstar[-1].copy()
        previous_candidate = candidate_lstar[-1].copy()
    return 0.0 if comparisons == 0 else inversions / comparisons


def _candidate_for_dose(
    source: np.ndarray,
    target: np.ndarray,
    *,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose: float,
    row_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    candidate = np.empty_like(source)
    gamut_scale = np.empty(source.shape[:2], dtype=np.float32)
    luma_error = np.empty(source.shape[:2], dtype=np.float64)
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
        luma_error[y0:y1] = error_rows
    return candidate, gamut_scale, luma_error


def select_analytic_y_chromaticity_candidate_streamed(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    """Execute the frozen selector with bounded row temporaries and disk quantiles."""
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
        raise CharacteristicLstarTransportError("CB54 selector input drift")
    with tempfile.TemporaryDirectory(dir=scratch_root, prefix="cb54-") as temporary:
        scratch_dir = Path(temporary)
        for dose in doses:
            candidate, gamut_scale, luma_error = _candidate_for_dose(
                source,
                target,
                curve=curve,
                strength=strength,
                boundary_epsilon=boundary_epsilon,
                dose=float(dose),
                row_chunk=row_chunk,
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
                effective_scale = np.asarray(float(dose) * gamut_scale, dtype=np.float32)
                return (
                    candidate,
                    effective_scale,
                    luma_error,
                    {
                        "characteristic_strength": float(strength),
                        "global_dose": float(dose),
                        "selected_gradient_ratio": float(gradient_ratio),
                        "selected_lstar_inversion_fraction": float(
                            inversion_fraction
                        ),
                        "selected_new_boundary_fraction": float(boundary_fraction),
                        "median_gamut_scale": float(np.median(gamut_scale)),
                        "fraction_gamut_scale_below_0p8": float(
                            np.mean(gamut_scale < 0.8)
                        ),
                        "maximum_luminance_error": float(
                            np.max(np.abs(luma_error))
                        ),
                    },
                )
    raise CharacteristicLstarTransportError(
        "CB54 dose grid has no safe analytic Y/chromaticity candidate"
    )


__all__ = ["select_analytic_y_chromaticity_candidate_streamed"]
