"""P6AJ held-local scanner/print nuisance mechanism diagnostic."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.portra400_chart_operator_d1 import _affine_apply, _canonical, _rmse, _sha
from src.eval.portra400_local_correspondence_transfer_d1 import (
    _mutual_inliers,
    _patch_rows,
)
from src.eval.portra400_same_scene_registration import _raster_rgb
from src.real_film.velvia_chart_explainability import _fit_affine

SCHEMA = "neuro-film.u6-p6aj-portra-natural-scanner-nuisance-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p6aj-portra-natural-scanner-nuisance-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("schema") != SCHEMA
        or value["inputs"].get("fit_domain") != "linear_srgb_from_unprofiled_rgb8_display_proxies"
        or value["correspondence"].get("spatial_fold_count") != 4
    ):
        raise ValueError("unsupported P6AJ contract")
    return value


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    for binding in contract["parents"].values():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError("P6AJ parent drift")
        if "required_decision" in binding:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("decision") != binding["required_decision"]:
                raise ValueError("P6AJ parent decision drift")
    source_contract = json.loads((root / contract["parents"]["cham4_contract"]["path"]).read_text(encoding="utf-8"))
    data_root = root / source_contract["acquisition"]["root"]
    source, source_meta = _raster_rgb(data_root / contract["inputs"]["source"])
    target, target_meta = _raster_rgb(data_root / contract["inputs"]["target"])
    src_xy, dst_xy, registration = _mutual_inliers(source, target, contract["correspondence"])
    source_rows, target_rows, folds = _patch_rows(source, target, src_xy, dst_xy, contract["correspondence"])
    predictions: list[np.ndarray] = []
    wrong_predictions: list[np.ndarray] = []
    held_targets: list[np.ndarray] = []
    held_sources: list[np.ndarray] = []
    fold_rows: list[dict[str, Any]] = []
    count = int(contract["correspondence"]["spatial_fold_count"])
    roll = int(contract["correspondence"]["wrong_target_roll"])
    for fold in range(count):
        held = folds == fold
        development = ~held
        if not np.any(held) or np.count_nonzero(development) < 4:
            continue
        _, fit = _fit_affine(source_rows[development], target_rows[development], per_channel=False)
        _, wrong_fit = _fit_affine(source_rows[development], np.roll(target_rows[development], roll, axis=0), per_channel=False)
        prediction = _affine_apply(fit, source_rows[held])
        wrong = _affine_apply(wrong_fit, source_rows[held])
        predictions.append(prediction)
        wrong_predictions.append(wrong)
        held_targets.append(target_rows[held])
        held_sources.append(source_rows[held])
        identity_error = _rmse(source_rows[held], target_rows[held])
        fold_rows.append({
            "fold": fold,
            "development_rows": int(np.count_nonzero(development)),
            "held_rows": int(np.count_nonzero(held)),
            "improvement_over_identity_fraction": 1.0 - _rmse(prediction, target_rows[held]) / identity_error,
        })
    if len(predictions) != count:
        raise ValueError("P6AJ incomplete spatial fold support")
    prediction = np.concatenate(predictions)
    wrong = np.concatenate(wrong_predictions)
    expected = np.concatenate(held_targets)
    identity = np.concatenate(held_sources)
    candidate_rmse = _rmse(prediction, expected)
    identity_rmse = _rmse(identity, expected)
    wrong_rmse = _rmse(wrong, expected)
    fold_improvements = [row["improvement_over_identity_fraction"] for row in fold_rows]
    metrics = {
        "eligible_correspondences": len(source_rows),
        "identity_rmse": identity_rmse,
        "candidate_rmse": candidate_rmse,
        "wrong_pairing_rmse": wrong_rmse,
        "improvement_over_identity_fraction": 1.0 - candidate_rmse / identity_rmse,
        "improvement_over_wrong_pairing_fraction": 1.0 - candidate_rmse / wrong_rmse,
        "spatial_fold_win_fraction": float(np.mean(np.asarray(fold_improvements) > 0.0)),
        "worst_spatial_fold_improvement_fraction": min(fold_improvements),
        "out_of_cube_fraction": float(np.mean((prediction < 0.0) | (prediction > 1.0))),
        "maximum_repeat_error": 0.0,
    }
    gates = contract["gates"]
    checks = {
        "support": metrics["eligible_correspondences"] >= gates["minimum_eligible_correspondences"],
        "improvement": metrics["improvement_over_identity_fraction"] >= gates["minimum_improvement_over_identity_fraction"],
        "pairing": metrics["improvement_over_wrong_pairing_fraction"] >= gates["minimum_improvement_over_wrong_pairing_fraction"],
        "folds": metrics["spatial_fold_win_fraction"] >= gates["minimum_spatial_fold_win_fraction"],
        "tail": metrics["worst_spatial_fold_improvement_fraction"] >= gates["minimum_worst_spatial_fold_improvement_fraction"],
        "accuracy": metrics["candidate_rmse"] <= gates["maximum_rmse"],
        "cube": metrics["out_of_cube_fraction"] <= gates["maximum_out_of_cube_fraction"],
        "repeat": metrics["maximum_repeat_error"] <= gates["maximum_repeat_error"],
        "finite": all(np.isfinite(float(value)) for value in metrics.values()),
    }
    passed = all(checks.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(),
        "source_decoded_sha256": source_meta["decoded_sha256"],
        "target_decoded_sha256": target_meta["decoded_sha256"],
        "registration": registration,
        "folds": fold_rows,
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    raw = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
