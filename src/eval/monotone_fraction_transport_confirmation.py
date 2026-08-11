"""CB28 source-disjoint confirmation of the unchanged CB27 operator."""

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

SCHEMA = "neuro_film.u5_r2cb28_monotone_fraction_transport_confirmation_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb28_monotone_fraction_transport_confirmation_report.v1"
)
EXPERIMENT_ID = "U5.R2CB28"


class MonotoneFractionTransportConfirmationError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise MonotoneFractionTransportConfirmationError(
            "CB28 contract structure drift"
        )
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb27_decision_path"],
        config["parents"]["cb27_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb27_required_status"]:
        raise MonotoneFractionTransportConfirmationError("CB27 decision drift")
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
        contract_filename="u5_r2cb28_monotone_fraction_transport_confirmation_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
    )


__all__ = [
    "MonotoneFractionTransportConfirmationError",
    "evaluate",
    "load_contract",
]
