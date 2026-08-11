"""CB42 positive source-RGB scalar solve for the frozen L-star curve."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_lstar_transport import (
    CharacteristicLstarTransportError,
)
from src.eval.characteristic_lstar_transport import (
    evaluate as evaluate_characteristic,
)
from src.eval.direct_lstar_monotone_tone_transport import _lstar_to_neutral_linear
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.safe_base_ao6_chroma_direction import apply_safe_base_direction_target

SCHEMA = "neuro_film.u5_r2cb42_chromaticity_scalar_lstar_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb42_chromaticity_scalar_lstar_development_report.v1"
EXPERIMENT_ID = "U5.R2CB42"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB42 contract structure drift")
    return payload


def select_chromaticity_scalar_candidate(
    source_linear: np.ndarray,
    chroma_target_linear: np.ndarray,
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
    target = np.asarray(chroma_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise CharacteristicLstarTransportError("CB42 selector input drift")

    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    mapped_lstar = 100.0 * _anchored_curve(
        source_lstar.astype(np.float64) / 100.0,
        curve,
        strength=strength,
        epsilon=boundary_epsilon,
    )
    desired_y = _lstar_to_neutral_linear(mapped_lstar)
    source64 = source.astype(np.float64)
    source_y = np.sum(source64 * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    scalar = np.divide(
        desired_y,
        source_y,
        out=np.ones_like(desired_y),
        where=source_y > 0.0,
    )
    interior_upper = float(
        np.float32(1.0 - boundary_epsilon)
        - np.float32(4.0) * np.spacing(np.float32(1.0 - boundary_epsilon))
    )
    interior_lower = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    maximum_scalar = np.min(
        np.divide(
            interior_upper,
            source64,
            out=np.full_like(source64, np.inf),
            where=source64 > 0.0,
        ),
        axis=-1,
    )
    minimum_scalar = np.max(
        np.divide(
            interior_lower,
            source64,
            out=np.zeros_like(source64),
            where=source64 > boundary_epsilon,
        ),
        axis=-1,
    )
    limited = (scalar > maximum_scalar) | (scalar < minimum_scalar)
    scalar = np.clip(scalar, minimum_scalar, maximum_scalar)
    tone_base = np.asarray(source64 * scalar[..., None], dtype=np.float32)
    tone_lstar = linear_rgb_to_lab(tone_base, working_space="linear_srgb")[..., 0]
    tone_y = np.sum(tone_base.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)

    target64 = target.astype(np.float64)
    target_y = np.sum(target64 * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    desired = np.asarray(
        tone_y[..., None] + (target64 - target_y[..., None]), dtype=np.float32
    )
    full_candidate, full_scale, _ = apply_safe_base_direction_target(
        source,
        tone_base,
        desired,
        weights=LEGACY_LAB_Y_WEIGHTS,
        boundary_epsilon=boundary_epsilon,
    )
    residual = full_candidate.astype(np.float64) - tone_base.astype(np.float64)
    diagnostics: list[dict[str, float]] = []
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    for dose in doses:
        candidate = np.asarray(
            tone_base.astype(np.float64) + float(dose) * residual, dtype=np.float32
        )
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(candidate, working_space="linear_srgb")[
            ..., 0
        ]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        diagnostics.append(
            {
                "dose": float(dose),
                "gradient_ratio": float(gradient),
                "lstar_inversion_fraction": float(inversion),
            }
        )
        if (
            gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
        ):
            selected = candidate
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            break
    if selected is None:
        best_gradient = min(diagnostics, key=lambda row: row["gradient_ratio"])
        best_order = min(diagnostics, key=lambda row: row["lstar_inversion_fraction"])
        raise CharacteristicLstarTransportError(
            "CB42 dose grid has no safe chromaticity-scalar candidate; "
            f"minimum_gradient={best_gradient['gradient_ratio']:.17g}; "
            f"minimum_inversion={best_order['lstar_inversion_fraction']:.17g}"
        )
    selected_y = np.sum(selected.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    facts = {
        "characteristic_strength": float(strength),
        "global_dose": selected_dose,
        "selected_gradient_ratio": selected_gradient,
        "selected_lstar_inversion_fraction": selected_inversion,
        "maximum_tone_lstar_error": float(np.max(np.abs(tone_lstar - mapped_lstar))),
        "scalar_gamut_limited_fraction": float(np.mean(limited)),
    }
    return (
        selected,
        np.asarray(full_scale.astype(np.float64) * selected_dose, dtype=np.float32),
        selected_y - tone_y,
        facts,
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=select_chromaticity_scalar_candidate,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb42_chromaticity_scalar_lstar_development_v1.json",
        prerequisite_path_key="cb41_decision_path",
        prerequisite_sha_key="cb41_decision_sha256",
        prerequisite_required_key="cb41_required_decision",
        diagnostic_decision="close_chromaticity_scalar_before_complete_render",
        pass_decision="open_chromaticity_scalar_severe_review_then_blind_development",
        close_decision="close_chromaticity_scalar_without_rescue",
    )
    if report.get("rows"):
        gates = config["automatic_gates"]
        report["metrics"]["maximum_tone_lstar_error"] = max(
            row["maximum_tone_lstar_error"] for row in report["rows"]
        )
        report["metrics"]["maximum_scalar_gamut_limited_fraction"] = max(
            row["scalar_gamut_limited_fraction"] for row in report["rows"]
        )
        report["checks"]["tone_scalar_solve"] = (
            report["metrics"]["maximum_tone_lstar_error"]
            <= gates["maximum_tone_lstar_error"]
            and report["metrics"]["maximum_scalar_gamut_limited_fraction"]
            <= gates["maximum_scalar_gamut_limited_fraction"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_chromaticity_scalar_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract", "select_chromaticity_scalar_candidate"]
