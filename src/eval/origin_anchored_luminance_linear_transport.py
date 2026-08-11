"""CB48 origin-anchored luminance-eigen global linear transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.characteristic_lstar_transport import evaluate as evaluate_characteristic
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.luminance_eigen_affine_transport import (
    select_luminance_eigen_affine_candidate,
)

SCHEMA = "neuro_film.u5_r2cb48_origin_anchored_luminance_linear_development_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb48_origin_anchored_luminance_linear_development_report.v1"
)
EXPERIMENT_ID = "U5.R2CB48"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB48 contract structure drift")
    return payload


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_luminance_eigen_affine_candidate(
            *args,
            **kwargs,
            minimum_luminance_slope=float(op["minimum_luminance_slope"]),
            maximum_luminance_slope=float(op["maximum_luminance_slope"]),
            origin_anchored=True,
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb48_origin_anchored_luminance_linear_development_v1.json",
        prerequisite_path_key="cb47_decision_path",
        prerequisite_sha_key="cb47_decision_sha256",
        prerequisite_required_key="cb47_required_decision",
        diagnostic_decision="close_origin_anchored_linear_before_complete_render",
        pass_decision="open_origin_anchored_linear_severe_review_then_blind_development",
        close_decision="close_origin_anchored_linear_without_rescue",
    )
    if report.get("rows"):
        doses = np.asarray(
            [row["global_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_global_dose"] = float(np.median(doses))
        report["metrics"]["fraction_global_dose_below_0p25"] = float(
            np.mean(doses < 0.25)
        )
        intercepts = np.asarray(
            [row["fitted_luminance_intercept"] for row in report["rows"]],
            dtype=np.float64,
        )
        report["metrics"]["maximum_absolute_luminance_intercept"] = float(
            np.max(np.abs(intercepts))
        )
        gates = config["automatic_gates"]
        report["checks"]["origin_anchor"] = (
            report["metrics"]["maximum_absolute_luminance_intercept"] == 0.0
        )
        report["checks"]["global_dose"] = (
            report["metrics"]["population_median_global_dose"]
            >= gates["minimum_population_median_global_dose"]
            and report["metrics"]["fraction_global_dose_below_0p25"]
            <= gates["maximum_fraction_global_dose_below_0p25"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_origin_anchored_linear_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract"]
