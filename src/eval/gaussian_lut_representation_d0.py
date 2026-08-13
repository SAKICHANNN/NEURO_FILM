"""GLUT-inspired continuous explicit colour representation D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.u5-r2glut0-gaussian-lut-representation-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2glut0-gaussian-lut-representation-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported GLUT D0 contract")
    return payload


def _cube(side: int) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, side, dtype=np.float64)
    return np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)


def _target(points: np.ndarray, index: int) -> np.ndarray:
    phase = 2.0 * np.pi * index / 12.0
    direction = np.array([np.cos(phase), np.cos(phase + 2.1), np.cos(phase - 2.0)])
    center = np.array([0.28 + 0.32 * (index % 3) / 2.0, 0.35 + 0.28 * ((index // 3) % 2), 0.30 + 0.34 * ((index // 6) % 2)])
    width = 0.12 + 0.035 * (index % 4)
    radial = np.exp(-np.sum((points - center) ** 2, axis=1) / (2.0 * width**2))
    cross = np.stack((points[:, 1] - points[:, 2], points[:, 2] - points[:, 0], points[:, 0] - points[:, 1]), axis=1)
    residual = 0.028 * radial[:, None] * direction[None, :] + 0.010 * cross * np.sin(phase + points * np.pi)
    raw = points + residual
    # Smooth monotone analytic interior projection, never hard clip.  The
    # normalization keeps identity endpoints fixed at 0.01/0.99.
    low = 1.0 / (1.0 + np.exp(3.5))
    high = 1.0 / (1.0 + np.exp(-3.5))
    logistic = 1.0 / (1.0 + np.exp(-7.0 * (raw - 0.5)))
    return 0.01 + 0.98 * (logistic - low) / (high - low)


def _gaussian_features(points: np.ndarray, side: int, sigma: float) -> np.ndarray:
    centers = _cube(side)
    squared = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
    weights = np.exp(-squared / (2.0 * sigma**2))
    weights /= np.maximum(np.sum(weights, axis=1, keepdims=True), 1e-15)
    return np.concatenate((np.ones((points.shape[0], 1)), points, weights), axis=1)


def _lattice_features(points: np.ndarray, side: int) -> np.ndarray:
    centers = _cube(side)
    distance = np.abs(points[:, None, :] - centers[None, :, :])
    weights = np.prod(np.maximum(1.0 - distance * (side - 1), 0.0), axis=2)
    return np.concatenate((np.ones((points.shape[0], 1)), points, weights), axis=1)


def _fit(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    gram = features.T @ features
    gram.flat[:: gram.shape[0] + 1] += ridge
    return np.linalg.solve(gram, features.T @ target)


def _jacobian_min(points: np.ndarray, fn) -> float:
    eps = 1e-4
    base = fn(points)
    columns = []
    for channel in range(3):
        shifted = points.copy()
        shifted[:, channel] += eps
        columns.append((fn(shifted) - base) / eps)
    jacobian = np.stack(columns, axis=2)
    return float(np.min(np.linalg.det(jacobian)))


def evaluate(contract: Mapping[str, Any]) -> dict[str, Any]:
    p = contract["population"]
    points = _cube(p["grid_side"])
    lattice_side = p["gaussian_side"]
    train_mask = np.all((np.rint(points * (p["grid_side"] - 1)).astype(int) % p["train_stride"]) == 0, axis=1)
    train, test = points[train_mask], points[~train_mask]
    gf_train = _gaussian_features(train, p["gaussian_side"], p["gaussian_sigma"])
    gf_test = _gaussian_features(test, p["gaussian_side"], p["gaussian_sigma"])
    lf_train = _lattice_features(train, lattice_side)
    lf_test = _lattice_features(test, lattice_side)
    rows = []
    for index in range(p["target_count"]):
        train_target, test_target = _target(train, index), _target(test, index)
        gw = _fit(gf_train, train_target, p["ridge"])
        lw = _fit(lf_train, train_target, p["ridge"])
        gp, lp = gf_test @ gw, lf_test @ lw
        grmse = float(np.sqrt(np.mean((gp - test_target) ** 2)))
        lrmse = float(np.sqrt(np.mean((lp - test_target) ** 2)))
        improvement = 100.0 * (lrmse - grmse) / max(lrmse, 1e-15)
        eps = 1.0 / 65535.0
        source_boundary = (test <= eps) | (test >= 1.0 - eps)
        output_boundary = (gp <= eps) | (gp >= 1.0 - eps)
        interior = points[(points > 0.02).all(axis=1) & (points < 0.98).all(axis=1)]
        rows.append({"target": index, "gaussian_rmse": grmse, "lattice_rmse": lrmse, "improvement_percent": improvement, "new_boundary_fraction": float(np.mean(output_boundary & ~source_boundary)), "minimum_jacobian_determinant": _jacobian_min(interior, lambda x, w=gw: _gaussian_features(x, p["gaussian_side"], p["gaussian_sigma"]) @ w)})
    metrics = {"target_count": len(rows), "targets_beating_lattice": sum(r["improvement_percent"] > 0 for r in rows), "median_rmse_improvement_percent": float(np.median([r["improvement_percent"] for r in rows])), "worst_rmse_improvement_percent": min(r["improvement_percent"] for r in rows), "maximum_gaussian_rmse": max(r["gaussian_rmse"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows), "minimum_jacobian_determinant": min(r["minimum_jacobian_determinant"] for r in rows)}
    g = contract["gates"]
    checks = {"wins": metrics["targets_beating_lattice"] >= g["minimum_targets_beating_lattice"], "median": metrics["median_rmse_improvement_percent"] >= g["minimum_median_rmse_improvement_percent"], "tail": metrics["worst_rmse_improvement_percent"] >= g["minimum_worst_rmse_improvement_percent"], "accuracy": metrics["maximum_gaussian_rmse"] <= g["maximum_gaussian_rmse"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"], "jacobian": metrics["minimum_jacobian_determinant"] >= g["minimum_jacobian_determinant"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
