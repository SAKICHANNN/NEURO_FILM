"""CB34 source-disjoint confirmation of the unchanged CB33 mechanism."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.gradient_budgeted_fraction_transport import (
    select_gradient_budgeted_candidate,
)
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb34_gradient_budgeted_fraction_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb34_gradient_budgeted_fraction_confirmation_report.v1"
EXPERIMENT_ID = "U5.R2CB34"


class GradientBudgetedFractionConfirmationError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GradientBudgetedFractionConfirmationError("CB34 contract structure drift")
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb33_decision_path"],
        config["parents"]["cb33_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb33_required_decision"]:
        raise GradientBudgetedFractionConfirmationError("CB33 decision drift")
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
        candidate, scale, luma_error, row_facts = select_gradient_budgeted_candidate(
            source_linear,
            safe_base_linear,
            full_target_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            dose_grid=op["dose_grid"],
            maximum_gradient_ratio=float(
                config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
            ),
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
        contract_filename="u5_r2cb34_gradient_budgeted_fraction_confirmation_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
    )
    if len(facts) != len(report["rows"]):
        raise GradientBudgetedFractionConfirmationError("CB34 diagnostic count drift")
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
        "open_severe_review_then_blind_confirmation_adjudication"
        if report["automatic_pass"]
        else "close_gradient_budgeted_confirmation_without_rescue"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "GradientBudgetedFractionConfirmationError",
    "evaluate",
    "load_contract",
]
