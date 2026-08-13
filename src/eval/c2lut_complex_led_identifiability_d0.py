"""Observer-metameric complex-LED D0 for illuminant-conditioned color operators."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.c2lut_spectral_metamer_identifiability_d0 import _apply, _canonical, _fit, _gaussian, _reflectances, _sensors

SCHEMA = "neuro-film.u5-r2bx2-c2lut-complex-led-identifiability-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2bx2-c2lut-complex-led-identifiability-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported BX2 contract")
    return payload


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("BX2 parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("BX2 parent decision drift")
    p = contract["population"]
    w = np.arange(400.0, 701.0, 10.0)
    observer, camera = _sensors(w)
    rng = np.random.default_rng(int(p["seed"]))
    reflectances = _reflectances(rng, w, int(p["reflectances"]))
    dev = int(p["development_reflectances"])
    rows = []
    for pair in range(int(p["illuminant_pairs"])):
        centers = np.asarray([420, 455, 490, 525, 560, 605, 655], dtype=np.float64) + (pair % 3 - 1) * 3.0
        peaks = np.stack([_gaussian(w, center, 8 + (pair + i) % 4 * 2) for i, center in enumerate(centers)], axis=1)
        weights = rng.uniform(0.65, 1.15, size=len(centers))
        base = 0.08 + peaks @ weights
        # Find the observer-null direction inside the peak coefficient space that maximizes camera response.
        obs_map = peaks.T @ observer
        _, _, vt = np.linalg.svd(obs_map.T, full_matrices=True)
        null_basis = vt[3:].T
        camera_map = peaks.T @ camera
        u, _, _ = np.linalg.svd(null_basis.T @ camera_map, full_matrices=False)
        coefficients = null_basis @ u[:, 0]
        direction = peaks @ coefficients
        direction /= max(float(np.max(np.abs(direction))), 1e-12)
        amplitude = 0.65 * float(np.min(base))
        illuminants = (base, base + amplitude * direction)
        white_xyz = [ill @ observer for ill in illuminants]
        white_rgb = [ill @ camera for ill in illuminants]
        xyz_error = float(np.max(np.abs(white_xyz[0] - white_xyz[1])))
        rgb_separation = float(np.linalg.norm(white_rgb[0] / np.sum(white_rgb[0]) - white_rgb[1] / np.sum(white_rgb[1])))
        rgb_sets, xyz_sets = [], []
        for illuminant in illuminants:
            rgb_sets.append(((reflectances * illuminant) @ camera) / np.sum(white_rgb[0]))
            xyz_sets.append(((reflectances * illuminant) @ observer) / np.sum(white_xyz[0]))
        shared = _fit(np.concatenate([x[:dev] for x in rgb_sets]), np.concatenate([x[:dev] for x in xyz_sets]))
        for hidden, (rgb, xyz) in enumerate(zip(rgb_sets, xyz_sets, strict=True)):
            oracle = _fit(rgb[:dev], xyz[:dev])
            se = float(np.sqrt(np.mean((_apply(rgb[dev:], shared) - xyz[dev:]) ** 2)))
            oe = float(np.sqrt(np.mean((_apply(rgb[dev:], oracle) - xyz[dev:]) ** 2)))
            rows.append({"pair": pair, "hidden_spd": hidden, "same_chromaticity_xyz_error": xyz_error, "camera_white_rgb_separation": rgb_separation, "shared_xyz_rmse": se, "oracle_xyz_rmse": oe, "oracle_improvement": 1.0 - oe / se})
    metrics = {"row_count": len(rows), "maximum_same_chromaticity_xyz_error": max(r["same_chromaticity_xyz_error"] for r in rows), "minimum_camera_white_rgb_separation": min(r["camera_white_rgb_separation"] for r in rows), "oracle_win_rate": float(np.mean([r["oracle_xyz_rmse"] < r["shared_xyz_rmse"] for r in rows])), "median_oracle_improvement_over_shared": float(np.median([r["oracle_improvement"] for r in rows])), "worst_oracle_improvement_over_shared": min(r["oracle_improvement"] for r in rows), "maximum_oracle_xyz_rmse": max(r["oracle_xyz_rmse"] for r in rows)}
    g = contract["gates"]
    checks = {"chromaticity": metrics["maximum_same_chromaticity_xyz_error"] <= g["maximum_same_chromaticity_xyz_error"], "camera_observation": metrics["minimum_camera_white_rgb_separation"] >= g["minimum_camera_white_rgb_separation"], "wins": metrics["oracle_win_rate"] >= g["minimum_oracle_win_rate"], "median": metrics["median_oracle_improvement_over_shared"] >= g["minimum_median_oracle_improvement_over_shared"], "tail": metrics["worst_oracle_improvement_over_shared"] >= g["minimum_worst_oracle_improvement_over_shared"], "accuracy": metrics["maximum_oracle_xyz_rmse"] <= g["maximum_oracle_xyz_rmse"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "parent_stable_evidence_id": parent["stable_evidence_id"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
