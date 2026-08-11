"""CB21 source-disjoint confirmation of the unchanged CB20 operator."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.global_chroma_procrustes import global_chroma_procrustes_target
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = (
    "neuro_film.u5_r2cb21_global_chroma_procrustes_confirmation_contract.v1"
)
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb21_global_chroma_procrustes_confirmation_report.v1"
)
EXPERIMENT_ID = "U5.R2CB21"


class GlobalChromaProcrustesConfirmationError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GlobalChromaProcrustesConfirmationError("CB21 contract structure drift")
    return payload


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb20_decision_path"],
        config["parents"]["cb20_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb20_required_status"]:
        raise GlobalChromaProcrustesConfirmationError("CB20 decision drift")
    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=global_chroma_procrustes_target,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb21_global_chroma_procrustes_confirmation_v1.json",
        blind_seed=int(config["blind_protocol"]["seed"]),
    )


__all__ = [
    "GlobalChromaProcrustesConfirmationError",
    "evaluate",
    "load_contract",
]
