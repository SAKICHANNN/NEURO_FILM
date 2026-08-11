"""CB33 image-global gradient-budgeted execution for CB32."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.monotone_fraction_gold_stress import evaluate as evaluate_gold_stress
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import apply_safe_base_direction_target

SCHEMA = "neuro_film.u5_r2cb33_gradient_budgeted_fraction_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb33_gradient_budgeted_fraction_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB33"


class GradientBudgetedFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GradientBudgetedFractionTransportError("CB33 contract structure drift")
    return payload


def select_gradient_budgeted_candidate(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    full_target_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
    dose_grid: Sequence[float],
    maximum_gradient_ratio: float,
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
        or np.min(doses) < 0.0
        or np.max(doses) > 1.0
    ):
        raise GradientBudgetedFractionTransportError("CB33 selector input drift")
    full_candidate, full_scale, _ = apply_safe_base_direction_target(
        source,
        base,
        target,
        weights=weights,
        boundary_epsilon=boundary_epsilon,
    )
    base64 = base.astype(np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if (
        np.max(
            np.abs(
                np.sum(target.astype(np.float64) * w, axis=-1)
                - np.sum(base64 * w, axis=-1)
            )
        )
        > 1e-6
    ):
        raise GradientBudgetedFractionTransportError("CB33 target changed luminance")
    residual = full_candidate.astype(np.float64) - base64
    full_ratio = _gradient_p999_ratio(source, full_candidate)
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_ratio = float("inf")
    for dose in doses:
        candidate = np.asarray(base64 + float(dose) * residual, dtype=np.float32)
        ratio = _gradient_p999_ratio(source, candidate)
        if ratio <= maximum_gradient_ratio:
            selected = candidate
            selected_dose = float(dose)
            selected_ratio = float(ratio)
            break
    if selected is None:
        raise GradientBudgetedFractionTransportError("CB33 dose grid has no safe base")
    base_luma = np.sum(base64 * w, axis=-1)
    luma_error = np.sum(selected.astype(np.float64) * w, axis=-1) - base_luma
    effective_scale = np.asarray(
        full_scale.astype(np.float64) * selected_dose, dtype=np.float32
    )
    facts = {
        "global_dose": selected_dose,
        "full_candidate_gradient_ratio": float(full_ratio),
        "selected_gradient_ratio": selected_ratio,
    }
    return selected, effective_scale, luma_error, facts


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb32_decision_path"],
        config["parents"]["cb32_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb32_required_decision"]:
        raise GradientBudgetedFractionTransportError("CB32 decision drift")
    cb32 = _load_exact_json(
        root,
        config["parents"]["cb32_contract_path"],
        config["parents"]["cb32_contract_sha256"],
    )
    cb31 = _load_exact_json(
        root,
        cb32["parents"]["cb31_contract_path"],
        cb32["parents"]["cb31_contract_sha256"],
    )
    cb30 = _load_exact_json(
        root,
        cb31["parents"]["cb30_contract_path"],
        cb31["parents"]["cb30_contract_sha256"],
    )
    derived = dict(cb30)
    derived["claim_ceiling"] = config["claim_ceiling"]
    derived["automatic_gates"] = config["automatic_gates"]
    cb32_operator = cb32["operator"]
    facts: list[dict[str, float]] = []

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
        minimum_valid_fraction: float,
    ) -> np.ndarray:
        if minimum_valid_fraction != float(cb32_operator["minimum_valid_fraction"]):
            raise GradientBudgetedFractionTransportError(
                "CB33 fraction threshold drift"
            )
        return nonexpansive_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=minimum_valid_fraction,
            fraction_knots=int(cb32_operator["fraction_knots"]),
            maximum_fraction_slope=float(cb32_operator["maximum_fraction_slope"]),
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
            dose_grid=config["operator"]["dose_grid"],
            maximum_gradient_ratio=float(
                config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
            ),
        )
        facts.append(row_facts)
        return candidate, scale, luma_error

    report = evaluate_gold_stress(
        derived,
        root,
        output_dir,
        target_builder=target_builder,
        candidate_builder=candidate_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb33_gradient_budgeted_fraction_gold_stress_v1.json",
    )
    if len(facts) != len(report["rows"]):
        raise GradientBudgetedFractionTransportError("CB33 diagnostic count drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([row["global_dose"] for row in facts], dtype=np.float64)
    report["metrics"]["population_median_global_dose"] = float(np.median(doses))
    report["metrics"]["fraction_global_dose_below_0p25"] = float(np.mean(doses < 0.25))
    gates = config["automatic_gates"]
    report["checks"]["gold_style_salience"] = (
        report["metrics"]["gold_median_style_delta_e76"]
        >= gates["minimum_gold_median_style_delta_e76"]
    )
    report["checks"]["stress_style_salience"] = (
        report["metrics"]["stress_median_style_delta_e76"]
        >= gates["minimum_stress_median_style_delta_e76"]
    )
    report["checks"]["global_dose"] = (
        report["metrics"]["population_median_global_dose"]
        >= gates["minimum_population_median_global_dose"]
    )
    report["checks"]["global_dose_tail"] = (
        report["metrics"]["fraction_global_dose_below_0p25"]
        <= gates["maximum_fraction_global_dose_below_0p25"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_gradient_budgeted_development_severe_review"
        if report["automatic_pass"]
        else "close_gradient_budgeted_transport_before_visual_review"
    )
    value = dict(report)
    value.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return report


__all__ = [
    "GradientBudgetedFractionTransportError",
    "evaluate",
    "load_contract",
    "select_gradient_budgeted_candidate",
]
