"""CB45 whole-row authorization for the exact strong CB33 candidate."""

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
from src.eval.fujifilm_e6_dye_operator_photographic import _new_boundary_fraction
from src.eval.gradient_budgeted_fraction_transport import (
    select_gradient_budgeted_candidate,
)

SCHEMA = "neuro_film.u5_r2cb45_row_authorized_style_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb45_row_authorized_style_development_report.v1"
EXPERIMENT_ID = "U5.R2CB45"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB45 contract structure drift")
    return payload


def select_row_authorized_style_candidate(
    source_linear: np.ndarray,
    full_target_linear: np.ndarray,
    *,
    safe_base_linear: np.ndarray,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose_grid: list[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    del curve, strength
    source = np.asarray(source_linear)
    candidate, scale, luma_error, facts = select_gradient_budgeted_candidate(
        source,
        safe_base_linear,
        full_target_linear,
        weights=LEGACY_LAB_Y_WEIGHTS,
        boundary_epsilon=boundary_epsilon,
        dose_grid=dose_grid,
        maximum_gradient_ratio=maximum_gradient_ratio,
    )
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    candidate_lstar = linear_rgb_to_lab(
        candidate, working_space="linear_srgb"
    )[..., 0]
    inversion = _gradient_inversion_fraction(
        source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
    )
    boundary = _new_boundary_fraction(source, candidate, boundary_epsilon)
    veto_reasons: list[str] = []
    if facts["selected_gradient_ratio"] > maximum_gradient_ratio:
        veto_reasons.append("gradient")
    if inversion > maximum_lstar_inversion_fraction:
        veto_reasons.append("lstar_order")
    if boundary > 0.0:
        veto_reasons.append("new_boundary")
    authorized = not veto_reasons
    row_facts: dict[str, Any] = {
        **facts,
        "preauthorization_lstar_inversion_fraction": float(inversion),
        "preauthorization_new_boundary_fraction": float(boundary),
        "row_authorized": authorized,
        "row_veto_reasons": veto_reasons,
    }
    if authorized:
        return candidate, scale, luma_error, row_facts
    return (
        source.copy(),
        np.zeros(source.shape[:-1], dtype=np.float32),
        np.zeros(source.shape[:-1], dtype=np.float64),
        {**row_facts, "global_dose": 0.0},
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=select_row_authorized_style_candidate,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb45_row_authorized_style_development_v1.json",
        prerequisite_path_key="cb44_decision_path",
        prerequisite_sha_key="cb44_decision_sha256",
        prerequisite_required_key="cb44_required_decision",
        diagnostic_decision="close_row_authorized_style_before_complete_render",
        pass_decision="open_row_authorized_style_severe_review_then_blind_development",
        close_decision="close_row_authorized_style_without_rescue",
        selector_receives_safe_base=True,
    )
    if report.get("rows"):
        authorized = np.asarray(
            [row["row_authorized"] for row in report["rows"]], dtype=np.bool_
        )
        report["metrics"]["authorized_source_count"] = int(np.sum(authorized))
        report["metrics"]["authorized_source_fraction"] = float(np.mean(authorized))
        doses = np.asarray(
            [row["global_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_global_dose"] = float(np.median(doses))
        report["metrics"]["fraction_global_dose_below_0p25"] = float(
            np.mean(doses < 0.25)
        )
        gates = config["automatic_gates"]
        report["checks"]["row_authorization_coverage"] = (
            report["metrics"]["authorized_source_fraction"]
            >= gates["minimum_authorized_source_fraction"]
            and report["metrics"]["authorized_source_count"]
            >= gates["minimum_authorized_source_count"]
        )
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
            report["decision"] = "close_row_authorized_style_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = [
    "evaluate",
    "load_contract",
    "select_row_authorized_style_candidate",
]
