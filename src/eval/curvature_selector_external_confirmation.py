"""External-family confirmation of the frozen GLUT2 representation selector."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.c2lut_chromaticity_identifiability import _truth_lut
from src.eval.curvature_representation_selector_d0 import _curvature
from src.eval.gaussian_lut_representation_d0 import (
    _canonical,
    _cube,
    _fit,
    _gaussian_features,
    _lattice_features,
)

SCHEMA = "neuro-film.u5-r2glut3-curvature-selector-external-confirmation-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2glut3-curvature-selector-external-confirmation-result.v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported GLUT3 contract")
    return payload


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    source_path = root / str(contract["source_family"]["path"])
    if _sha(parent_path) != contract["parent"]["sha256"] or _sha(source_path) != contract["source_family"]["sha256"]:
        raise ValueError("GLUT3 parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("GLUT3 parent decision drift")
    source = json.loads(source_path.read_text(encoding="utf-8"))["fixture"]
    selector = parent["frozen_selector"]
    side, stride = 17, 2
    cube = _cube(side)
    index = np.rint(cube * (side - 1)).astype(int)
    mask = np.all(index % stride == 0, axis=1)
    train, test = cube[mask], cube[~mask]
    gf0 = _gaussian_features(train, 4, 0.23)
    gf1 = _gaussian_features(test, 4, 0.23)
    lf0 = _lattice_features(train, 4)
    lf1 = _lattice_features(test, 4)
    rows = []
    for r, g in contract["source_family"]["descriptors"]:
        for hidden in contract["source_family"]["hidden_coordinates"]:
            truth = _truth_lut(side, r, g, hidden, visible_scale=float(source["visible_residual_scale"]), hidden_scale=float(source["hidden_spectral_residual_scale"])).reshape(-1, 3)
            ge = float(np.sqrt(np.mean((gf1 @ _fit(gf0, truth[mask], 1e-4) - truth[~mask]) ** 2)))
            le = float(np.sqrt(np.mean((lf1 @ _fit(lf0, truth[mask], 1e-4) - truth[~mask]) ** 2)))
            curvature = _curvature(truth[mask], 9)
            selected = selector["high_curvature_representation"] if curvature >= selector["curvature_threshold"] else selector["low_curvature_representation"]
            rows.append({"descriptor": [r, g], "hidden": hidden, "curvature": curvature, "gaussian_rmse": ge, "lattice_rmse": le, "winner": "gaussian" if ge < le else "lattice", "selected": selected})
    selected = np.asarray([row[f"{row['selected']}_rmse"] for row in rows])
    best = np.asarray([min(row["gaussian_rmse"], row["lattice_rmse"]) for row in rows])
    fixed_g = np.asarray([row["gaussian_rmse"] for row in rows])
    fixed_l = np.asarray([row["lattice_rmse"] for row in rows])
    regret = float(np.sum(selected - best))
    worse_regret = max(float(np.sum(fixed_g - best)), float(np.sum(fixed_l - best)))
    metrics = {"row_count": len(rows), "selector_accuracy": float(np.mean([row["selected"] == row["winner"] for row in rows])), "regret_reduction_vs_worse_fixed_representation": 1.0 - regret / max(worse_regret, 1e-15), "worst_selected_vs_best_rmse_ratio": float(np.max(selected / np.maximum(best, 1e-15)))}
    gates = contract["gates"]
    checks = {"accuracy": metrics["selector_accuracy"] >= gates["minimum_selector_accuracy"], "regret": metrics["regret_reduction_vs_worse_fixed_representation"] >= gates["minimum_regret_reduction_vs_worse_fixed_representation"], "tail": metrics["worst_selected_vs_best_rmse_ratio"] <= gates["maximum_worst_selected_vs_best_rmse_ratio"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "parent_stable_evidence_id": parent["stable_evidence_id"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
