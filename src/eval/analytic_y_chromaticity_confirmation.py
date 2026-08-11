"""CB51 source-disjoint confirmation of the exact CB50 operator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.eval.analytic_y_chromaticity_transport import evaluate_with_context
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError

SCHEMA = "neuro_film.u5_r2cb51_analytic_y_chromaticity_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb51_analytic_y_chromaticity_confirmation_report.v1"
EXPERIMENT_ID = "U5.R2CB51"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB51 contract drift")
    return payload


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    return evaluate_with_context(
        config,
        root,
        output_dir,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb51_analytic_y_chromaticity_confirmation_v1.json",
        prerequisite_path_key="cb50_decision_path",
        prerequisite_sha_key="cb50_decision_sha256",
        prerequisite_required_key="cb50_required_decision",
        diagnostic_decision="close_cb51_before_complete_render",
        pass_decision="open_cb51_severe_review_then_blind_confirmation",
        close_decision="close_cb51_without_rescue",
    )


__all__ = ["evaluate", "load_contract"]
