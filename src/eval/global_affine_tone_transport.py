"""CB43 global positive affine tone projection plus fixed CB32 chroma."""

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
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _new_boundary_fraction,
)
from src.eval.safe_base_ao6_chroma_direction import apply_safe_base_direction_target

SCHEMA = "neuro_film.u5_r2cb43_global_affine_tone_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb43_global_affine_tone_development_report.v1"
EXPERIMENT_ID = "U5.R2CB43"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB43 contract structure drift")
    return payload


def select_global_affine_tone_candidate(
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
    tone_dose_grid: list[float],
    minimum_affine_slope: float,
    maximum_affine_slope: float,
    minimum_affine_intercept: float = -float("inf"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    del curve, strength
    source = np.asarray(source_linear)
    target = np.asarray(chroma_target_linear)
    chroma_doses = np.asarray(dose_grid, dtype=np.float64)
    tone_doses = np.asarray(tone_dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or chroma_doses[0] != 1.0
        or chroma_doses[-1] != 0.0
        or tone_doses[0] != 1.0
        or tone_doses[-1] != 0.0
        or np.any(np.diff(chroma_doses) >= 0.0)
        or np.any(np.diff(tone_doses) >= 0.0)
    ):
        raise CharacteristicLstarTransportError("CB43 selector input drift")
    source64 = source.astype(np.float64)
    target64 = target.astype(np.float64)
    source_y = np.sum(source64 * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    target_y = np.sum(target64 * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    x = source_y.reshape(-1)
    y = target_y.reshape(-1)
    centered = x - float(np.mean(x))
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        raise CharacteristicLstarTransportError("CB43 source luminance is degenerate")
    fitted_slope = float(np.dot(centered, y - float(np.mean(y))) / variance)
    fitted_slope = float(
        np.clip(fitted_slope, minimum_affine_slope, maximum_affine_slope)
    )
    fitted_intercept = max(
        float(np.mean(y) - fitted_slope * np.mean(x)), minimum_affine_intercept
    )
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    target_chroma = target64 - target_y[..., None]
    selected: np.ndarray | None = None
    selected_scale: np.ndarray | None = None
    selected_tone_y: np.ndarray | None = None
    selected_tone_dose = -1.0
    selected_chroma_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    for tone_dose in tone_doses:
        slope = 1.0 + float(tone_dose) * (fitted_slope - 1.0)
        intercept = float(tone_dose) * fitted_intercept
        tone = np.asarray(slope * source64 + intercept, dtype=np.float32)
        if (
            np.min(tone) < 0.0
            or np.max(tone) > 1.0
            or _new_boundary_fraction(source, tone, boundary_epsilon) != 0.0
        ):
            continue
        tone_y = np.sum(tone.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)
        desired = np.asarray(tone_y[..., None] + target_chroma, dtype=np.float32)
        full, full_scale, _ = apply_safe_base_direction_target(
            source,
            tone,
            desired,
            weights=LEGACY_LAB_Y_WEIGHTS,
            boundary_epsilon=boundary_epsilon,
        )
        residual = full.astype(np.float64) - tone.astype(np.float64)
        for chroma_dose in chroma_doses:
            candidate = np.asarray(
                tone.astype(np.float64) + float(chroma_dose) * residual,
                dtype=np.float32,
            )
            gradient = _gradient_p999_ratio(source, candidate)
            candidate_lstar = linear_rgb_to_lab(candidate, working_space="linear_srgb")[
                ..., 0
            ]
            inversion = _gradient_inversion_fraction(
                source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
            )
            if (
                gradient <= maximum_gradient_ratio
                and inversion <= maximum_lstar_inversion_fraction
            ):
                selected = candidate
                selected_scale = np.asarray(
                    full_scale.astype(np.float64) * float(chroma_dose),
                    dtype=np.float32,
                )
                selected_tone_y = tone_y
                selected_tone_dose = float(tone_dose)
                selected_chroma_dose = float(chroma_dose)
                selected_gradient = float(gradient)
                selected_inversion = float(inversion)
                break
        if selected is not None:
            break
    if selected is None or selected_scale is None or selected_tone_y is None:
        raise CharacteristicLstarTransportError(
            "CB43 tone/chroma grids have no safe global-affine candidate"
        )
    selected_y = np.sum(selected.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    fit_rmse = float(
        np.sqrt(np.mean((fitted_slope * source_y + fitted_intercept - target_y) ** 2))
    )
    return (
        selected,
        selected_scale,
        selected_y - selected_tone_y,
        {
            "fitted_affine_slope": fitted_slope,
            "fitted_affine_intercept": fitted_intercept,
            "affine_fit_rmse": fit_rmse,
            "tone_dose": selected_tone_dose,
            "global_dose": selected_chroma_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
        },
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_global_affine_tone_candidate(
            *args,
            **kwargs,
            tone_dose_grid=op["tone_dose_grid"],
            minimum_affine_slope=float(op["minimum_affine_slope"]),
            maximum_affine_slope=float(op["maximum_affine_slope"]),
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb43_global_affine_tone_development_v1.json",
        prerequisite_path_key="cb42_decision_path",
        prerequisite_sha_key="cb42_decision_sha256",
        prerequisite_required_key="cb42_required_decision",
        diagnostic_decision="close_global_affine_tone_before_complete_render",
        pass_decision="open_global_affine_tone_severe_review_then_blind_development",
        close_decision="close_global_affine_tone_without_rescue",
    )
    if report.get("rows"):
        tone_doses = np.asarray(
            [row["tone_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_tone_dose"] = float(np.median(tone_doses))
        report["metrics"]["fraction_tone_dose_below_0p25"] = float(
            np.mean(tone_doses < 0.25)
        )
        gates = config["automatic_gates"]
        report["checks"]["tone_dose"] = (
            report["metrics"]["population_median_tone_dose"]
            >= gates["minimum_population_median_tone_dose"]
            and report["metrics"]["fraction_tone_dose_below_0p25"]
            <= gates["maximum_fraction_tone_dose_below_0p25"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_global_affine_tone_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract", "select_global_affine_tone_candidate"]
