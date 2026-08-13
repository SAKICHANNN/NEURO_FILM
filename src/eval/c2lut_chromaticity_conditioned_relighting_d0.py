"""Clean-room Eq.1-5 C2LUT mechanism test on held-out analytic illuminants."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.c2lut_spectral_metamer_identifiability_d0 import _canonical, _gaussian, _reflectances, _sensors

SCHEMA = "neuro-film.u5-r2bx3-c2lut-chromaticity-conditioned-relighting-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2bx3-c2lut-chromaticity-conditioned-relighting-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported BX3 contract")
    return payload


def _features(rgb: np.ndarray, quadratic: bool = True) -> np.ndarray:
    r, g, b = rgb.T
    if not quadratic:
        return np.column_stack((np.ones(len(rgb)), r, g, b))
    return np.column_stack((np.ones(len(rgb)), r, g, b, r * r, g * g, b * b, r * g, r * b, g * b))


def _fit(rgb: np.ndarray, xyz: np.ndarray, ridge: float, quadratic: bool = True) -> np.ndarray:
    x = _features(rgb, quadratic)
    reg = np.eye(x.shape[1]) * ridge
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ xyz)


def _rmse(rgb: np.ndarray, xyz: np.ndarray, weights: np.ndarray, quadratic: bool = True) -> float:
    return float(np.sqrt(np.mean((_features(rgb, quadratic) @ weights - xyz) ** 2)))


def _illuminants(rng: np.random.Generator, w: np.ndarray, count: int) -> list[np.ndarray]:
    centers = np.asarray([420, 455, 490, 525, 560, 605, 655], dtype=np.float64)
    rows = []
    for index in range(count):
        peaks = np.stack([_gaussian(w, center + (index % 5 - 2) * 2.0, 9 + (index + j) % 5 * 2) for j, center in enumerate(centers)], axis=1)
        weights = rng.lognormal(0.0, 0.55, size=len(centers))
        rows.append(0.04 + peaks @ weights)
    return rows


def evaluate(contract: Mapping[str, Any]) -> dict[str, Any]:
    p = contract["population"]
    rng = np.random.default_rng(int(p["seed"]))
    w = np.arange(400.0, 701.0, 10.0)
    observer, camera = _sensors(w)
    d65 = 0.25 + 0.75 * _gaussian(w, 560, 95)
    reflectances = _reflectances(rng, w, int(p["reflectances"]))
    target = (reflectances * d65) @ observer
    target /= np.sum(d65 @ observer)
    dev = int(p["development_reflectances"])
    illuminants = _illuminants(rng, w, int(p["train_illuminants"]) + int(p["test_illuminants"]))
    records = []
    for index, illuminant in enumerate(illuminants):
        white = illuminant @ camera
        raw = (reflectances * illuminant) @ camera
        balanced = raw / white[None, :]
        rg = np.asarray([white[0] / np.sum(white), white[1] / np.sum(white)])
        records.append({"index": index, "rgb": balanced, "rg": rg})
    train = records[: int(p["train_illuminants"])]
    test = records[int(p["train_illuminants"]) :]
    ridge = float(contract["candidate"]["ridge"])
    shared_ccm = _fit(np.concatenate([r["rgb"][:dev] for r in train]), np.tile(target[:dev], (len(train), 1)), ridge, False)
    shared_quad = _fit(np.concatenate([r["rgb"][:dev] for r in train]), np.tile(target[:dev], (len(train), 1)), ridge, True)
    train_weights = [_fit(r["rgb"][:dev], target[:dev], ridge, True) for r in train]
    train_rg = np.stack([r["rg"] for r in train])
    rows = []
    k = int(contract["candidate"]["nearest_rg_train_illuminants"])
    for row in test:
        distances = np.linalg.norm(train_rg - row["rg"][None, :], axis=1)
        nearest = np.argsort(distances)[:k]
        inverse = 1.0 / np.maximum(distances[nearest], 1e-9)
        coefficients = inverse / np.sum(inverse)
        candidate = np.tensordot(coefficients, np.stack([train_weights[i] for i in nearest]), axes=1)
        oracle = _fit(row["rgb"][:dev], target[:dev], ridge, True)
        ccm_error = _rmse(row["rgb"][dev:], target[dev:], shared_ccm, False)
        shared_error = _rmse(row["rgb"][dev:], target[dev:], shared_quad, True)
        candidate_error = _rmse(row["rgb"][dev:], target[dev:], candidate, True)
        oracle_error = _rmse(row["rgb"][dev:], target[dev:], oracle, True)
        rows.append({"index": row["index"], "rg": row["rg"].tolist(), "shared_ccm_rmse": ccm_error, "shared_quadratic_rmse": shared_error, "conditioned_rmse": candidate_error, "oracle_rmse": oracle_error, "improvement_vs_shared_quadratic": 1.0 - candidate_error / shared_error, "candidate_to_oracle_rmse_ratio": candidate_error / max(oracle_error, 1e-15)})
    metrics = {"row_count": len(rows), "candidate_win_rate_vs_shared_quadratic": float(np.mean([r["conditioned_rmse"] < r["shared_quadratic_rmse"] for r in rows])), "median_improvement_vs_shared_quadratic": float(np.median([r["improvement_vs_shared_quadratic"] for r in rows])), "worst_improvement_vs_shared_quadratic": min(r["improvement_vs_shared_quadratic"] for r in rows), "maximum_candidate_to_oracle_rmse_ratio": max(r["candidate_to_oracle_rmse_ratio"] for r in rows), "maximum_candidate_xyz_rmse": max(r["conditioned_rmse"] for r in rows), "median_improvement_shared_quadratic_vs_ccm": float(np.median([1.0 - r["shared_quadratic_rmse"] / r["shared_ccm_rmse"] for r in rows]))}
    g = contract["gates"]
    checks = {"wins": metrics["candidate_win_rate_vs_shared_quadratic"] >= g["minimum_candidate_win_rate_vs_shared_quadratic"], "median": metrics["median_improvement_vs_shared_quadratic"] >= g["minimum_median_improvement_vs_shared_quadratic"], "tail": metrics["worst_improvement_vs_shared_quadratic"] >= g["minimum_worst_improvement_vs_shared_quadratic"], "oracle_gap": metrics["maximum_candidate_to_oracle_rmse_ratio"] <= g["maximum_candidate_to_oracle_rmse_ratio"], "accuracy": metrics["maximum_candidate_xyz_rmse"] <= g["maximum_candidate_xyz_rmse"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "paper_binding": contract["paper_binding"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
