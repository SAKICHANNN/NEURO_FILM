"""Clean-room procedural D0 for region correspondence under geometric mismatch."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.u5-r2cham0-region-correspondence-lut-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham0-region-correspondence-lut-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported CHAM0 contract")
    return payload


def _features(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb.T
    return np.column_stack((np.ones(len(rgb)), r, g, b, r * g, r * b, g * b, r * r, g * g, b * b))


def _fit(source: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    x = _features(source)
    reg = np.eye(x.shape[1]) * ridge
    reg[0, 0] = 0.0
    return np.linalg.solve(x.T @ x + reg, x.T @ target)


def _truth(rgb: np.ndarray, scene: int) -> np.ndarray:
    phase = (scene + 1) * 0.37
    r, g, b = rgb.T
    out = np.column_stack((0.025 + 0.91 * r + 0.055 * g * b, 0.018 + 0.92 * g + 0.045 * r * b, 0.02 + 0.9 * b + 0.06 * r * g))
    out += 0.008 * np.sin(phase) * np.column_stack((g * (1 - r), b * (1 - g), r * (1 - b)))
    return np.clip(out, 0.0, 1.0)


def _bounded_residual(base: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    residual = candidate - base
    scale = np.ones(len(base), dtype=np.float64)
    for channel in range(3):
        positive = residual[:, channel] > 0.0
        negative = residual[:, channel] < 0.0
        scale[positive] = np.minimum(
            scale[positive],
            (1.0 - base[positive, channel]) / residual[positive, channel],
        )
        scale[negative] = np.minimum(
            scale[negative],
            -base[negative, channel] / residual[negative, channel],
        )
    return base + np.maximum(0.0, np.minimum(1.0, scale))[:, None] * residual


def evaluate(contract: Mapping[str, Any]) -> dict[str, Any]:
    p = contract["population"]
    rng = np.random.default_rng(int(p["seed"]))
    rows = []
    for scene in range(int(p["scenes"])):
        centers = rng.uniform(0.08, 0.92, size=(int(p["regions_per_scene"]), 3))
        source = np.clip(centers[:, None, :] + rng.normal(0.0, 0.025, size=(len(centers), int(p["pixels_per_region"]), 3)), 0.0, 1.0)
        target = _truth(source.reshape(-1, 3), scene).reshape(source.shape)
        # Frozen independent within-region permutations model local FoV/PoV mismatch.
        shuffled = np.empty_like(target)
        for region in range(len(centers)):
            shuffled[region] = target[region, rng.permutation(target.shape[1])]
        index_w = _fit(source.reshape(-1, 3), shuffled.reshape(-1, 3), float(contract["operator"]["ridge"]))
        region_w = _fit(np.mean(source, axis=1), np.mean(shuffled, axis=1), float(contract["operator"]["ridge"]))
        probe = rng.uniform(0.0, 1.0, size=(4096, 3))
        truth = _truth(probe, scene)
        index_out = _bounded_residual(probe, _features(probe) @ index_w)
        region_out = _bounded_residual(probe, _features(probe) @ region_w)
        index_rmse = float(np.sqrt(np.mean((index_out - truth) ** 2)))
        region_rmse = float(np.sqrt(np.mean((region_out - truth) ** 2)))
        boundary = float(np.mean(np.any((region_out < 0.0) | (region_out > 1.0), axis=1)))
        rows.append({"scene": scene, "index_pixel_rmse": index_rmse, "region_correspondence_rmse": region_rmse, "improvement": 1.0 - region_rmse / index_rmse, "new_boundary_fraction": boundary})
    metrics = {"row_count": len(rows), "region_win_rate": float(np.mean([r["region_correspondence_rmse"] < r["index_pixel_rmse"] for r in rows])), "median_improvement_over_index_pixels": float(np.median([r["improvement"] for r in rows])), "maximum_region_operator_rmse": max(r["region_correspondence_rmse"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows)}
    g = contract["gates"]
    checks = {"wins": metrics["region_win_rate"] >= g["minimum_region_win_rate"], "improvement": metrics["median_improvement_over_index_pixels"] >= g["minimum_median_improvement_over_index_pixels"], "accuracy": metrics["maximum_region_operator_rmse"] <= g["maximum_region_operator_rmse"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
