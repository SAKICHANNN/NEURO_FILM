"""CB24 source-disjoint confirmation of the unchanged CB23 operator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_ao6_factorized import _inputs
from src.eval.feasibility_bounded_gaussian_transport import (
    _maximum_feasible_step,
    feasibility_bounded_gaussian_transport_target,
)
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.global_chroma_gaussian_transport import (
    global_chroma_gaussian_transport_target,
)
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb24_feasibility_bounded_transport_confirmation_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb24_feasibility_bounded_transport_confirmation_report.v1"
)
EXPERIMENT_ID = "U5.R2CB24"


class FeasibilityBoundedTransportConfirmationError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FeasibilityBoundedTransportConfirmationError(
            "CB24 contract structure drift"
        )
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb23_decision_path"],
        config["parents"]["cb23_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb23_required_status"]:
        raise FeasibilityBoundedTransportConfirmationError("CB23 decision drift")
    operator = config["operator"]

    def target_builder(
        source_linear: np.ndarray,
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return feasibility_bounded_gaussian_transport_target(
            source_linear,
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_raw_covariance_eigenvalue=float(
                operator["minimum_raw_covariance_eigenvalue"]
            ),
            minimum_transport_eigenvalue=float(
                operator["minimum_transport_eigenvalue"]
            ),
            maximum_transport_eigenvalue=float(
                operator["maximum_transport_eigenvalue"]
            ),
            feasibility_quantile=float(operator["feasibility_quantile"]),
            minimum_retained_step_at_quantile=float(
                operator["minimum_retained_step_at_quantile"]
            ),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb24_feasibility_bounded_transport_confirmation_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
        source_aware_target_builder=True,
    )


def evaluate_preflight(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb23_decision_path"],
        config["parents"]["cb23_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb23_required_status"]:
        raise FeasibilityBoundedTransportConfirmationError("CB23 decision drift")
    cb11, ao6_config, artifact, source_rows, curve = _inputs(config, root)
    operator = config["operator"]
    cb11_operator = cb11["operator"]
    weights = np.asarray(cb11_operator["luminance_weights"], dtype=np.float64)
    epsilon = float(cb11_operator["boundary_epsilon"])
    quantile = float(operator["feasibility_quantile"])
    retained_step = float(operator["minimum_retained_step_at_quantile"])
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source = _load_rgb(
            root / source_row["decoded_path"],
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        safe_base, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=float(cb11_operator["nominal_strength"]),
            boundary_epsilon=epsilon,
        )
        gaussian_target = global_chroma_gaussian_transport_target(
            safe_base,
            ao6,
            weights=weights,
            boundary_epsilon=epsilon,
            minimum_raw_covariance_eigenvalue=float(
                operator["minimum_raw_covariance_eigenvalue"]
            ),
            minimum_transport_eigenvalue=float(
                operator["minimum_transport_eigenvalue"]
            ),
            maximum_transport_eigenvalue=float(
                operator["maximum_transport_eigenvalue"]
            ),
        )
        feasible = _maximum_feasible_step(
            source,
            safe_base,
            gaussian_target,
            boundary_epsilon=epsilon,
        )
        quantile_step = float(np.quantile(feasible, quantile, method="lower"))
        global_dose = min(1.0, quantile_step / retained_step)
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "feasibility_quantile_step": quantile_step,
                "global_dose": global_dose,
            }
        )
        if not np.isfinite(global_dose) or global_dose <= 0.0:
            break
    failed = bool(rows and rows[-1]["global_dose"] <= 0.0)
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2cb24_feasibility_bounded_transport_preflight_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "rows": rows,
        "source_count_read": len(rows),
        "source_count_unread": int(config["population"]["source_count_exact"])
        - len(rows),
        "failed_source_id": rows[-1]["id"] if failed else None,
        "passed": not failed
        and len(rows) == config["population"]["source_count_exact"],
        "decision": (
            "close_independent_confirmation_on_zero_global_dose"
            if failed
            else "preflight_global_dose_envelope_pass"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "FeasibilityBoundedTransportConfirmationError",
    "evaluate",
    "evaluate_preflight",
    "load_contract",
]
