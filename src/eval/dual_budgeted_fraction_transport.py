"""CB35 global gradient- and L-star-order-budgeted CB32 execution."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import (
    apply_safe_base_direction_target,
    evaluate_direction_candidate,
)

SCHEMA = "neuro_film.u5_r2cb35_dual_budgeted_fraction_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb35_dual_budgeted_fraction_development_report.v1"
EXPERIMENT_ID = "U5.R2CB35"


class DualBudgetedFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise DualBudgetedFractionTransportError("CB35 contract structure drift")
    return payload


def select_dual_budgeted_candidate(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    full_target_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
    dose_grid: Sequence[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_linear)
    base = np.asarray(safe_base_linear)
    target = np.asarray(full_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or base.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != base.shape
        or source.shape != target.shape
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise DualBudgetedFractionTransportError("CB35 selector input drift")
    full_candidate, full_scale, _ = apply_safe_base_direction_target(
        source,
        base,
        target,
        weights=weights,
        boundary_epsilon=boundary_epsilon,
    )
    base64 = base.astype(np.float64)
    residual = full_candidate.astype(np.float64) - base64
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    for dose in doses:
        candidate = np.asarray(base64 + float(dose) * residual, dtype=np.float32)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(
            candidate, working_space="linear_srgb"
        )[..., 0]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
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
        raise DualBudgetedFractionTransportError("CB35 dose grid has no dual-safe dose")
    w = np.asarray(weights, dtype=np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    luma_error = np.sum(selected.astype(np.float64) * w, axis=-1) - base_luma
    effective_scale = np.asarray(
        full_scale.astype(np.float64) * selected_dose, dtype=np.float32
    )
    return selected, effective_scale, luma_error, {
        "global_dose": selected_dose,
        "selected_gradient_ratio": selected_gradient,
        "selected_lstar_inversion_fraction": selected_inversion,
    }


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb34_decision_path"],
        config["parents"]["cb34_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb34_required_decision"]:
        raise DualBudgetedFractionTransportError("CB34 decision drift")
    cb33 = _load_exact_json(
        root,
        config["parents"]["cb33_contract_path"],
        config["parents"]["cb33_contract_sha256"],
    )
    _load_exact_json(
        root,
        cb33["parents"]["cb32_contract_path"],
        cb33["parents"]["cb32_contract_sha256"],
    )
    op = config["operator"]
    facts: list[dict[str, float]] = []

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return nonexpansive_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=float(op["minimum_valid_fraction"]),
            fraction_knots=int(op["fraction_knots"]),
            maximum_fraction_slope=float(op["maximum_fraction_slope"]),
        )

    def candidate_builder(
        source_linear: np.ndarray,
        safe_base_linear: np.ndarray,
        full_target_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        candidate, scale, luma_error, row_facts = select_dual_budgeted_candidate(
            source_linear,
            safe_base_linear,
            full_target_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            dose_grid=op["dose_grid"],
            maximum_gradient_ratio=float(
                config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
            ),
            maximum_lstar_inversion_fraction=float(
                config["automatic_gates"][
                    "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
                ]
            ),
            lstar_order_epsilon=0.0001,
        )
        facts.append(row_facts)
        return candidate, scale, luma_error

    report = evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        candidate_builder=candidate_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb35_dual_budgeted_fraction_development_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
    )
    if len(facts) != len(report["rows"]):
        raise DualBudgetedFractionTransportError("CB35 diagnostic count drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([fact["global_dose"] for fact in facts], dtype=np.float64)
    metrics = report["metrics"]
    metrics["population_median_global_dose"] = float(np.median(doses))
    metrics["fraction_global_dose_below_0p25"] = float(np.mean(doses < 0.25))
    gates = config["automatic_gates"]
    report["checks"]["global_dose"] = (
        metrics["population_median_global_dose"]
        >= gates["minimum_population_median_global_dose"]
    )
    report["checks"]["global_dose_tail"] = (
        metrics["fraction_global_dose_below_0p25"]
        <= gates["maximum_fraction_global_dose_below_0p25"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    if not report["automatic_pass"]:
        report["blind_sheets"] = []
        report["sealed_mappings"] = {}
    report["decision"] = (
        "open_severe_review_then_blind_development_adjudication"
        if report["automatic_pass"]
        else "close_dual_budgeted_transport_without_rescue"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "DualBudgetedFractionTransportError",
    "evaluate",
    "load_contract",
    "select_dual_budgeted_candidate",
]
