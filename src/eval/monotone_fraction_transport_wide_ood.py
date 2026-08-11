"""CB29 modern-camera OOD stress for the unchanged CB27 operator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.monotone_fraction_quantile_transport import (
    monotone_fraction_quantile_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb29_monotone_fraction_transport_wide_ood_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb29_monotone_fraction_transport_wide_ood_report.v1"
EXPERIMENT_ID = "U5.R2CB29"


class MonotoneFractionTransportWideOodError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise MonotoneFractionTransportWideOodError("CB29 contract structure drift")
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb28_decision_path"],
        config["parents"]["cb28_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb28_required_status"]:
        raise MonotoneFractionTransportWideOodError("CB28 decision drift")
    population = config["population"]
    source_review = _load_exact_json(
        root,
        population["visual_review_path"],
        population["visual_review_sha256"],
    )
    if (
        source_review.get("decision")
        != "pass_source_visual_gate_with_two_fixed_failures_and_no_replacement"
        or source_review.get("eligible_ids") != population["included_source_ids"]
        or source_review.get("confirmed_severe_source_artifact_count") != 0
    ):
        raise MonotoneFractionTransportWideOodError("CB29 source review drift")
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return monotone_fraction_quantile_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=float(operator["minimum_valid_fraction"]),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb29_monotone_fraction_transport_wide_ood_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
    )


__all__ = ["MonotoneFractionTransportWideOodError", "evaluate", "load_contract"]
