"""CB47 global affine colour transform with analytic positive luminance row."""

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
from src.eval.characteristic_lstar_transport import evaluate as evaluate_characteristic
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _new_boundary_fraction,
)

SCHEMA = "neuro_film.u5_r2cb47_luminance_eigen_affine_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb47_luminance_eigen_affine_development_report.v1"
EXPERIMENT_ID = "U5.R2CB47"

_BASIS = np.asarray(
    [
        LEGACY_LAB_Y_WEIGHTS,
        [1.0, -1.0, 0.0],
        [0.0, 1.0, -1.0],
    ],
    dtype=np.float64,
)
_INVERSE_BASIS = np.linalg.inv(_BASIS)


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB47 contract structure drift")
    return payload


def select_luminance_eigen_affine_candidate(
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
    minimum_luminance_slope: float,
    maximum_luminance_slope: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    del curve, strength
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
        or not 0.0 < minimum_luminance_slope <= maximum_luminance_slope
    ):
        raise CharacteristicLstarTransportError("CB47 selector input drift")
    source64 = source.astype(np.float64)
    target64 = target.astype(np.float64)
    source_q = source64 @ _BASIS.T
    target_q = target64 @ _BASIS.T
    source_y = source_q[..., 0].reshape(-1)
    target_y = target_q[..., 0].reshape(-1)
    centered = source_y - float(np.mean(source_y))
    variance = float(np.dot(centered, centered))
    if variance <= 0.0:
        raise CharacteristicLstarTransportError("CB47 source luminance is degenerate")
    slope = float(
        np.clip(
            np.dot(centered, target_y - float(np.mean(target_y))) / variance,
            minimum_luminance_slope,
            maximum_luminance_slope,
        )
    )
    intercept = float(np.mean(target_y) - slope * np.mean(source_y))
    design = np.concatenate(
        [source_q.reshape(-1, 3), np.ones((source_y.size, 1), dtype=np.float64)],
        axis=1,
    )
    chroma_coefficients = np.linalg.lstsq(
        design, target_q.reshape(-1, 3)[:, 1:], rcond=None
    )[0]
    fitted_q = np.empty_like(source_q)
    fitted_q[..., 0] = slope * source_q[..., 0] + intercept
    fitted_q[..., 1:] = (
        design @ chroma_coefficients
    ).reshape(source_q.shape[:-1] + (2,))
    fitted = fitted_q @ _INVERSE_BASIS.T
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    selected_boundary = float("inf")
    selected_luma_error: np.ndarray | None = None
    for dose in doses:
        candidate = np.asarray(
            source64 + float(dose) * (fitted - source64), dtype=np.float32
        )
        if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
            continue
        boundary = _new_boundary_fraction(source, candidate, boundary_epsilon)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(
            candidate, working_space="linear_srgb"
        )[..., 0]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        if (
            boundary == 0.0
            and gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
        ):
            expected_y = (
                (1.0 - float(dose)) * source_q[..., 0]
                + float(dose) * fitted_q[..., 0]
            )
            actual_y = candidate.astype(np.float64) @ LEGACY_LAB_Y_WEIGHTS
            selected = candidate
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            selected_boundary = float(boundary)
            selected_luma_error = actual_y - expected_y
            break
    if selected is None or selected_luma_error is None:
        raise CharacteristicLstarTransportError(
            "CB47 dose grid has no safe luminance-eigen affine candidate"
        )
    fit_rmse = float(np.sqrt(np.mean((fitted - target64) ** 2)))
    return (
        selected,
        np.full(source.shape[:-1], selected_dose, dtype=np.float32),
        selected_luma_error,
        {
            "fitted_luminance_slope": slope,
            "fitted_luminance_intercept": intercept,
            "affine_fit_rmse": fit_rmse,
            "global_dose": selected_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
            "selected_new_boundary_fraction": selected_boundary,
        },
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_luminance_eigen_affine_candidate(
            *args,
            **kwargs,
            minimum_luminance_slope=float(op["minimum_luminance_slope"]),
            maximum_luminance_slope=float(op["maximum_luminance_slope"]),
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb47_luminance_eigen_affine_development_v1.json",
        prerequisite_path_key="cb46_decision_path",
        prerequisite_sha_key="cb46_decision_sha256",
        prerequisite_required_key="cb46_required_decision",
        diagnostic_decision="close_luminance_eigen_affine_before_complete_render",
        pass_decision="open_luminance_eigen_affine_severe_review_then_blind_development",
        close_decision="close_luminance_eigen_affine_without_rescue",
    )
    if report.get("rows"):
        doses = np.asarray(
            [row["global_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_global_dose"] = float(np.median(doses))
        report["metrics"]["fraction_global_dose_below_0p25"] = float(
            np.mean(doses < 0.25)
        )
        gates = config["automatic_gates"]
        report["checks"]["global_dose"] = (
            report["metrics"]["population_median_global_dose"]
            >= gates["minimum_population_median_global_dose"]
            and report["metrics"]["fraction_global_dose_below_0p25"]
            <= gates["maximum_fraction_global_dose_below_0p25"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_luminance_eigen_affine_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract", "select_luminance_eigen_affine_candidate"]
