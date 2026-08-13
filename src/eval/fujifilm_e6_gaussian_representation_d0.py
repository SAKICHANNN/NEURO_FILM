"""Gaussian-basis representation of frozen Fujifilm E-6 technical operators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_e6_bounded_dye_operator import (
    STOCKS,
    _load_source_curves,
    compile_operator,
    load_contract as load_cb2,
)
from src.eval.gaussian_lut_representation_d0 import (
    _canonical,
    _cube,
    _fit,
    _gaussian_features,
    _jacobian_min,
    _lattice_features,
)
from src.film_physics.spectral_scanner import synthetic_profile_from_contract

SCHEMA = "neuro-film.u5-r2glut1-fujifilm-e6-gaussian-representation-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2glut1-fujifilm-e6-gaussian-representation-d0-result.v1"


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported E-6 Gaussian representation contract")
    for parent in payload["parents"].values():
        target = root / parent["path"]
        if _hash(target) != parent["sha256"]:
            raise ValueError(f"parent hash drift: {target}")
        if "required_decision" in parent:
            observed = json.loads(target.read_text(encoding="utf-8"))
            if observed.get("decision") != parent["required_decision"]:
                raise ValueError("parent decision drift")
    return payload


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    cb2 = load_cb2(root / contract["parents"]["fujifilm_operator_contract"]["path"])
    wavelength, curves = _load_source_curves(cb2, root)
    scanners = [synthetic_profile_from_contract(wavelength, row) for row in cb2["scanner_profiles"]]
    p = contract["representation"]
    side = int(cb2["compiler"]["lut_size"])
    cube = _cube(side)
    indices = np.rint(cube * (side - 1)).astype(int)
    mask = np.all(indices % int(p["fit_stride"]) == 0, axis=1)
    train, test = cube[mask], cube[~mask]
    gf_train = _gaussian_features(train, int(p["gaussian_side"]), float(p["gaussian_sigma"]))
    gf_test = _gaussian_features(test, int(p["gaussian_side"]), float(p["gaussian_sigma"]))
    lf_train = _lattice_features(train, int(p["lattice_side"]))
    lf_test = _lattice_features(test, int(p["lattice_side"]))
    rows = []
    for scanner in scanners:
        for stock in STOCKS:
            truth, _ = compile_operator(cube, curves[stock], scanner, cb2)
            train_truth, test_truth = truth[mask].astype(np.float64), truth[~mask].astype(np.float64)
            gw = _fit(gf_train, train_truth, float(p["ridge"]))
            lw = _fit(lf_train, train_truth, float(p["ridge"]))
            gp, lp = gf_test @ gw, lf_test @ lw
            grmse = float(np.sqrt(np.mean((gp - test_truth) ** 2)))
            lrmse = float(np.sqrt(np.mean((lp - test_truth) ** 2)))
            eps = 1.0 / 65535.0
            source_edge = (test <= eps) | (test >= 1.0 - eps)
            output_edge = (gp <= eps) | (gp >= 1.0 - eps)
            interior = cube[(cube > 0.02).all(axis=1) & (cube < 0.98).all(axis=1)]
            rows.append({"scanner": scanner.profile_id, "stock": stock, "gaussian_rmse": grmse, "lattice_rmse": lrmse, "improvement_percent": 100.0 * (lrmse - grmse) / max(lrmse, 1e-15), "new_boundary_fraction": float(np.mean(output_edge & ~source_edge)), "minimum_jacobian_determinant": _jacobian_min(interior, lambda x, w=gw: _gaussian_features(x, int(p["gaussian_side"]), float(p["gaussian_sigma"])) @ w)})
    metrics = {"output_count": len(rows), "outputs_beating_lattice": sum(r["improvement_percent"] > 0 for r in rows), "median_rmse_improvement_percent": float(np.median([r["improvement_percent"] for r in rows])), "worst_rmse_improvement_percent": min(r["improvement_percent"] for r in rows), "maximum_gaussian_rmse": max(r["gaussian_rmse"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows), "minimum_sampled_jacobian_determinant": min(r["minimum_jacobian_determinant"] for r in rows)}
    g = contract["gates"]
    checks = {"wins": metrics["outputs_beating_lattice"] >= g["minimum_outputs_beating_lattice"], "median": metrics["median_rmse_improvement_percent"] >= g["minimum_median_rmse_improvement_percent"], "tail": metrics["worst_rmse_improvement_percent"] >= g["minimum_worst_rmse_improvement_percent"], "accuracy": metrics["maximum_gaussian_rmse"] <= g["maximum_gaussian_rmse"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"], "jacobian": metrics["minimum_sampled_jacobian_determinant"] >= g["minimum_sampled_jacobian_determinant"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
