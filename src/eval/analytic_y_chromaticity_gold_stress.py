"""CB52 unchanged CB50/CB51 mechanism on the frozen U4.1 gold/stress set."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.analytic_y_chromaticity_transport import (
    select_analytic_y_chromaticity_candidate,
)
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.monotone_fraction_gold_stress import evaluate as evaluate_gold_stress
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)

SCHEMA = "neuro_film.u5_r2cb52_analytic_y_chromaticity_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb52_analytic_y_chromaticity_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB52"


class AnalyticYChromaticityGoldStressError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise AnalyticYChromaticityGoldStressError("CB52 contract drift")
    return payload


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    cb51 = _load_exact_json(
        root,
        config["parents"]["cb51_decision_path"],
        config["parents"]["cb51_decision_sha256"],
    )
    if cb51.get("decision") != config["parents"]["cb51_required_decision"]:
        raise AnalyticYChromaticityGoldStressError("CB51 decision drift")
    cb11 = _load_exact_json(
        root,
        config["parents"]["cb11_contract_path"],
        config["parents"]["cb11_contract_sha256"],
    )
    curve = _compiled_curve(load_cb6(root / cb11["parents"]["cb6_contract_path"]))
    strength = float(cb11["operator"]["nominal_strength"])
    op = config["operator"]
    facts: list[dict[str, float]] = []

    def target_builder(
        safe_base: np.ndarray,
        ao6: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
        minimum_valid_fraction: float,
    ) -> np.ndarray:
        return nonexpansive_fraction_transport_target(
            safe_base,
            ao6,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=minimum_valid_fraction,
            fraction_knots=int(op["fraction_knots"]),
            maximum_fraction_slope=float(op["maximum_fraction_slope"]),
        )

    def candidate_builder(
        source: np.ndarray,
        safe_base: np.ndarray,
        target: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del safe_base, weights
        candidate, scale, error, row_facts = select_analytic_y_chromaticity_candidate(
            source,
            target,
            curve=curve,
            strength=strength,
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
            lstar_order_epsilon=float(op["lstar_order_epsilon"]),
        )
        facts.append(row_facts)
        return candidate, scale, error

    report = evaluate_gold_stress(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        candidate_builder=candidate_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb52_analytic_y_chromaticity_gold_stress_v1.json",
    )
    if len(facts) != len(report["rows"]):
        raise AnalyticYChromaticityGoldStressError("CB52 row diagnostics drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    metrics = report["metrics"]
    metrics["population_median_global_dose"] = float(
        np.median([row["global_dose"] for row in report["rows"]])
    )
    metrics["fraction_global_dose_below_0p25"] = float(
        np.mean([row["global_dose"] < 0.25 for row in report["rows"]])
    )
    metrics["population_median_gamut_scale"] = float(
        np.median([row["median_gamut_scale"] for row in report["rows"]])
    )
    metrics["population_median_fraction_gamut_scale_below_0p8"] = float(
        np.median([row["fraction_gamut_scale_below_0p8"] for row in report["rows"]])
    )
    metrics["gold_median_delta_e76_vs_cb11"] = float(
        np.median(
            [
                row["median_delta_e76_vs_cb11"]
                for row in report["rows"]
                if row["split"] == "gold"
            ]
        )
    )
    metrics["stress_median_delta_e76_vs_cb11"] = float(
        np.median(
            [
                row["median_delta_e76_vs_cb11"]
                for row in report["rows"]
                if row["split"] == "stress"
            ]
        )
    )
    gates = config["automatic_gates"]
    report["checks"].update(
        {
            "global_dose": metrics["population_median_global_dose"]
            >= gates["minimum_population_median_global_dose"]
            and metrics["fraction_global_dose_below_0p25"]
            <= gates["maximum_fraction_global_dose_below_0p25"],
            "gamut_retention": metrics["population_median_gamut_scale"]
            >= gates["minimum_population_median_gamut_scale"]
            and metrics["population_median_fraction_gamut_scale_below_0p8"]
            <= gates["maximum_population_median_fraction_gamut_scale_below_0p8"],
            "gold_visible_style": metrics["gold_median_style_delta_e76"]
            >= gates["minimum_gold_median_style_delta_e76"],
            "stress_visible_style": metrics["stress_median_style_delta_e76"]
            >= gates["minimum_stress_median_style_delta_e76"],
            "material_vs_cb11": min(
                metrics["gold_median_delta_e76_vs_cb11"],
                metrics["stress_median_delta_e76_vs_cb11"],
            )
            >= gates["minimum_split_median_delta_e76_vs_cb11"],
        }
    )
    report["automatic_pass"] = all(report["checks"].values())
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_cb52_partial_gold_stress_severe_review"
        if report["automatic_pass"]
        else "close_cb52_without_rescue"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = ["evaluate", "load_contract"]
