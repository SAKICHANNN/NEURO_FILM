"""CB50 monotone analytic luminance plus bounded normalized chromaticity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.characteristic_lstar_transport import evaluate as evaluate_characteristic
from src.eval.direct_lstar_monotone_tone_transport import _lstar_to_neutral_linear
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _new_boundary_fraction,
)

SCHEMA = "neuro_film.u5_r2cb50_analytic_y_chromaticity_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb50_analytic_y_chromaticity_development_report.v1"
EXPERIMENT_ID = "U5.R2CB50"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB50 contract structure drift")
    return payload


def _normalized_zero_y_chromaticity(rgb: np.ndarray, epsilon: float) -> np.ndarray:
    rgb64 = np.asarray(rgb, dtype=np.float64)
    y = rgb64 @ LEGACY_LAB_Y_WEIGHTS
    residual = rgb64 - y[..., None]
    normalized = np.divide(
        residual,
        y[..., None],
        out=np.zeros_like(residual),
        where=y[..., None] > epsilon,
    )
    # Remove the final floating-point projection residue explicitly.
    normalized -= (normalized @ LEGACY_LAB_Y_WEIGHTS)[..., None]
    return normalized


def _bounded_same_y_reconstruction(
    desired_y: np.ndarray,
    normalized_chromaticity: np.ndarray,
    *,
    boundary_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(desired_y, dtype=np.float64)
    chroma = np.asarray(normalized_chromaticity, dtype=np.float64)
    if y.shape != chroma.shape[:-1] or chroma.shape[-1] != 3:
        raise CharacteristicLstarTransportError("CB50 reconstruction shape drift")
    neutral = np.repeat(y[..., None], 3, axis=-1)
    residual = y[..., None] * chroma
    residual -= (residual @ LEGACY_LAB_Y_WEIGHTS)[..., None]
    lower = float(
        np.float32(boundary_epsilon) + 4 * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper = float(upper_edge - 4 * np.spacing(upper_edge))
    scale = np.ones(y.shape, dtype=np.float64)
    for channel in range(3):
        delta = residual[..., channel]
        positive = delta > 0.0
        negative = delta < 0.0
        scale = np.minimum(
            scale,
            np.where(
                positive,
                np.divide(
                    upper - neutral[..., channel],
                    delta,
                    out=np.full_like(delta, np.inf),
                    where=positive,
                ),
                np.inf,
            ),
        )
        scale = np.minimum(
            scale,
            np.where(
                negative,
                np.divide(
                    neutral[..., channel] - lower,
                    -delta,
                    out=np.full_like(delta, np.inf),
                    where=negative,
                ),
                np.inf,
            ),
        )
    scale = np.clip(scale, 0.0, 1.0)
    limited = scale < 1.0
    scale[limited] = np.nextafter(scale[limited], 0.0)
    candidate = np.asarray(neutral + scale[..., None] * residual, dtype=np.float32)
    error = candidate.astype(np.float64) @ LEGACY_LAB_Y_WEIGHTS - y
    return candidate, scale.astype(np.float32), error


def select_analytic_y_chromaticity_candidate(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_linear)
    target = np.asarray(full_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise CharacteristicLstarTransportError("CB50 selector input drift")
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    mapped_lstar = 100.0 * _anchored_curve(
        source_lstar.astype(np.float64) / 100.0,
        curve,
        strength=strength,
        epsilon=boundary_epsilon,
    )
    desired_y = _lstar_to_neutral_linear(mapped_lstar)
    lower = float(
        np.float32(boundary_epsilon) + 4 * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper = float(upper_edge - 4 * np.spacing(upper_edge))
    desired_y = np.clip(desired_y, lower, upper)
    source_chroma = _normalized_zero_y_chromaticity(source, boundary_epsilon)
    target_chroma = _normalized_zero_y_chromaticity(target, boundary_epsilon)
    selected: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    selected_boundary = float("inf")
    for dose in doses:
        chroma = source_chroma + float(dose) * (target_chroma - source_chroma)
        chroma -= (chroma @ LEGACY_LAB_Y_WEIGHTS)[..., None]
        candidate, gamut_scale, luma_error = _bounded_same_y_reconstruction(
            desired_y, chroma, boundary_epsilon=boundary_epsilon
        )
        boundary = _new_boundary_fraction(source, candidate, boundary_epsilon)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(candidate, working_space="linear_srgb")[
            ..., 0
        ]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        if (
            boundary == 0.0
            and gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
        ):
            selected = candidate, gamut_scale, luma_error
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            selected_boundary = float(boundary)
            break
    if selected is None:
        raise CharacteristicLstarTransportError(
            "CB50 dose grid has no safe analytic Y/chromaticity candidate"
        )
    candidate, gamut_scale, luma_error = selected
    effective_scale = np.asarray(selected_dose * gamut_scale, dtype=np.float32)
    return (
        candidate,
        effective_scale,
        luma_error,
        {
            "characteristic_strength": float(strength),
            "global_dose": selected_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
            "selected_new_boundary_fraction": selected_boundary,
            "median_gamut_scale": float(np.median(gamut_scale)),
            "fraction_gamut_scale_below_0p8": float(np.mean(gamut_scale < 0.8)),
            "maximum_luminance_error": float(np.max(np.abs(luma_error))),
        },
    )


def evaluate_with_context(
    config: dict[str, Any],
    root: Path,
    output_dir: Path,
    *,
    report_schema: str,
    experiment_id: str,
    contract_filename: str,
    prerequisite_path_key: str,
    prerequisite_sha_key: str,
    prerequisite_required_key: str,
    diagnostic_decision: str,
    pass_decision: str,
    close_decision: str,
) -> dict[str, Any]:
    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=select_analytic_y_chromaticity_candidate,
        report_schema=report_schema,
        experiment_id=experiment_id,
        contract_filename=contract_filename,
        prerequisite_path_key=prerequisite_path_key,
        prerequisite_sha_key=prerequisite_sha_key,
        prerequisite_required_key=prerequisite_required_key,
        diagnostic_decision=diagnostic_decision,
        pass_decision=pass_decision,
        close_decision=close_decision,
    )
    if report.get("rows"):
        rows = report["rows"]
        report["metrics"]["population_median_gamut_scale"] = float(
            np.median([row["median_gamut_scale"] for row in rows])
        )
        report["metrics"]["population_median_fraction_gamut_scale_below_0p8"] = float(
            np.median([row["fraction_gamut_scale_below_0p8"] for row in rows])
        )
        gates = config["automatic_gates"]
        report["checks"]["gamut_retention"] = (
            report["metrics"]["population_median_gamut_scale"]
            >= gates["minimum_population_median_gamut_scale"]
            and report["metrics"]["population_median_fraction_gamut_scale_below_0p8"]
            <= gates["maximum_population_median_fraction_gamut_scale_below_0p8"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = close_decision
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    return evaluate_with_context(
        config,
        root,
        output_dir,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb50_analytic_y_chromaticity_development_v1.json",
        prerequisite_path_key="cb49_decision_path",
        prerequisite_sha_key="cb49_decision_sha256",
        prerequisite_required_key="cb49_required_decision",
        diagnostic_decision="close_analytic_y_chromaticity_before_complete_render",
        pass_decision="open_analytic_y_chromaticity_severe_review_then_blind_development",
        close_decision="close_analytic_y_chromaticity_without_rescue",
    )


__all__ = [
    "_bounded_same_y_reconstruction",
    "_normalized_zero_y_chromaticity",
    "evaluate",
    "evaluate_with_context",
    "load_contract",
    "select_analytic_y_chromaticity_candidate",
]
