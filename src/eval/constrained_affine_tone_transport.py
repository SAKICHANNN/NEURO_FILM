"""CB44 nonnegative-intercept global affine tone projection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_lstar_transport import (
    CharacteristicLstarTransportError,
)
from src.eval.characteristic_lstar_transport import (
    evaluate as evaluate_characteristic,
)
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.global_affine_tone_transport import select_global_affine_tone_candidate

SCHEMA = "neuro_film.u5_r2cb44_constrained_affine_tone_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb44_constrained_affine_tone_development_report.v1"
EXPERIMENT_ID = "U5.R2CB44"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB44 contract structure drift")
    return payload


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_global_affine_tone_candidate(
            *args,
            **kwargs,
            tone_dose_grid=op["tone_dose_grid"],
            minimum_affine_slope=float(op["minimum_affine_slope"]),
            maximum_affine_slope=float(op["maximum_affine_slope"]),
            minimum_affine_intercept=float(op["minimum_affine_intercept"]),
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb44_constrained_affine_tone_development_v1.json",
        prerequisite_path_key="cb43_decision_path",
        prerequisite_sha_key="cb43_decision_sha256",
        prerequisite_required_key="cb43_required_decision",
        diagnostic_decision="close_constrained_affine_tone_before_complete_render",
        pass_decision="open_constrained_affine_tone_severe_review_then_blind_development",
        close_decision="close_constrained_affine_tone_without_rescue",
    )
    if report.get("rows"):
        tone_doses = np.asarray(
            [row["tone_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_tone_dose"] = float(np.median(tone_doses))
        report["metrics"]["fraction_tone_dose_below_0p25"] = float(
            np.mean(tone_doses < 0.25)
        )
        gates = config["automatic_gates"]
        report["checks"]["tone_dose"] = (
            report["metrics"]["population_median_tone_dose"]
            >= gates["minimum_population_median_tone_dose"]
            and report["metrics"]["fraction_tone_dose_below_0p25"]
            <= gates["maximum_fraction_tone_dose_below_0p25"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_constrained_affine_tone_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract"]
