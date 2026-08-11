"""CB24 source-disjoint confirmation of the unchanged CB23 operator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.feasibility_bounded_gaussian_transport import (
    feasibility_bounded_gaussian_transport_target,
)
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
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


__all__ = [
    "FeasibilityBoundedTransportConfirmationError",
    "evaluate",
    "load_contract",
]
