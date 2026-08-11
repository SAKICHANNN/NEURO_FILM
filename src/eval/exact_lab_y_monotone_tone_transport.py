"""CB38 exact legacy-CIELAB-Y monotone tone plus fixed CB32 chroma."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.monotone_source_tone_fraction_transport import (
    MonotoneSourceToneFractionTransportError,
    select_monotone_source_tone_candidate,
)
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb38_exact_lab_y_monotone_tone_development_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb38_exact_lab_y_monotone_tone_development_report.v1"
)
EXPERIMENT_ID = "U5.R2CB38"
LEGACY_LAB_Y_WEIGHTS = np.asarray([0.212671, 0.715160, 0.072169], dtype=np.float64)


class ExactLabYMonotoneToneTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise ExactLabYMonotoneToneTransportError("CB38 contract structure drift")
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb37_decision_path"],
        config["parents"]["cb37_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb37_required_decision"]:
        raise ExactLabYMonotoneToneTransportError("CB37 decision drift")
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
    facts: list[dict[str, float | int]] = []
    source_rows = _load_exact_json(
        root,
        config["population"]["manifest_path"],
        config["population"]["manifest_sha256"],
    )
    source_ids = [row["id"] for row in source_rows]
    failure: dict[str, Any] | None = None

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
        nonlocal failure
        del weights
        try:
            candidate, scale, luma_error, row_facts = (
                select_monotone_source_tone_candidate(
                    source_linear,
                    safe_base_linear,
                    full_target_linear,
                    weights=LEGACY_LAB_Y_WEIGHTS,
                    boundary_epsilon=boundary_epsilon,
                    tone_knot_count=int(op["tone_knot_count"]),
                    dose_grid=op["dose_grid"],
                    maximum_gradient_ratio=float(
                        config["automatic_gates"][
                            "maximum_p999_gradient_ratio_vs_source"
                        ]
                    ),
                    maximum_lstar_inversion_fraction=float(
                        config["automatic_gates"][
                            "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
                        ]
                    ),
                    lstar_order_epsilon=float(op["lstar_order_epsilon"]),
                )
            )
        except MonotoneSourceToneFractionTransportError as exc:
            failure = {
                "source_id": source_ids[len(facts)],
                "completed_source_count": len(facts),
                "reason": str(exc),
            }
            raise ExactLabYMonotoneToneTransportError(str(exc)) from exc
        facts.append(row_facts)
        return candidate, scale, luma_error

    try:
        report = evaluate_direction_candidate(
            config,
            root,
            output_dir,
            target_builder=target_builder,
            candidate_builder=candidate_builder,
            report_schema=REPORT_SCHEMA,
            experiment_id=EXPERIMENT_ID,
            contract_filename="u5_r2cb38_exact_lab_y_monotone_tone_development_v1.json",
            blind_seed=int(config["blind_protocol"]["seed"]),
        )
    except ExactLabYMonotoneToneTransportError:
        if failure is None:
            raise
        shutil.rmtree(output_dir)
        diagnostic: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": hashlib.sha256(
                (
                    root
                    / "configs/u5_r2cb38_exact_lab_y_monotone_tone_development_v1.json"
                ).read_bytes()
            ).hexdigest(),
            "automatic_pass": False,
            "checks": {"gradient_and_order_safe_candidate": False},
            "failure": failure,
            "partial_artifacts_removed": True,
            "visual_review_status": "forbidden",
            "decision": "close_exact_lab_y_monotone_tone_before_complete_render",
            "claim_ceiling": config["claim_ceiling"],
        }
        diagnostic["stable_evidence_id"] = hashlib.sha256(
            canonical_json(diagnostic)
        ).hexdigest()
        return diagnostic
    if len(facts) != len(report["rows"]):
        raise ExactLabYMonotoneToneTransportError("CB38 diagnostic count drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([float(fact["global_dose"]) for fact in facts])
    report["metrics"]["population_median_global_dose"] = float(np.median(doses))
    report["metrics"]["fraction_global_dose_below_0p25"] = float(
        np.mean(doses < 0.25)
    )
    report["metrics"]["minimum_tone_knot_count"] = min(
        int(fact["tone_knot_count"]) for fact in facts
    )
    gates = config["automatic_gates"]
    report["checks"]["global_dose"] = (
        report["metrics"]["population_median_global_dose"]
        >= gates["minimum_population_median_global_dose"]
    )
    report["checks"]["global_dose_tail"] = (
        report["metrics"]["fraction_global_dose_below_0p25"]
        <= gates["maximum_fraction_global_dose_below_0p25"]
    )
    report["checks"]["tone_support"] = (
        report["metrics"]["minimum_tone_knot_count"]
        >= gates["minimum_tone_knot_count"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    if not report["automatic_pass"]:
        report["blind_sheets"] = []
        report["sealed_mappings"] = {}
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_exact_lab_y_monotone_tone_severe_review_then_blind_development"
        if report["automatic_pass"]
        else "close_exact_lab_y_monotone_tone_without_rescue"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = ["ExactLabYMonotoneToneTransportError", "evaluate", "load_contract"]
