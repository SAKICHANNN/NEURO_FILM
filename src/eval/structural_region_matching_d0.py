"""Candidate-independent structural region matching before explicit color fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.eval.region_correspondence_lut_d0 import _bounded_residual, _canonical, _features, _fit, _truth

SCHEMA = "neuro-film.u5-r2cham1-structural-region-matching-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham1-structural-region-matching-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported CHAM1 contract")
    return payload


def _signature(region: np.ndarray) -> np.ndarray:
    luminance = region @ np.asarray([0.2126, 0.7152, 0.0722])
    centered = luminance - np.mean(luminance)
    scale = max(float(np.linalg.norm(centered)), 1e-12)
    return centered / scale


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("CHAM1 parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("CHAM1 parent decision drift")
    p = contract["population"]
    rng = np.random.default_rng(int(p["seed"]))
    rows = []
    for scene in range(int(p["scenes"])):
        regions = int(p["regions_per_scene"])
        samples = int(p["samples_per_region"])
        centers = rng.uniform(0.12, 0.88, size=(regions, 3))
        x = np.linspace(0.0, 1.0, samples, dtype=np.float64)
        source = np.empty((regions, samples, 3), dtype=np.float64)
        for region in range(regions):
            texture = np.sin((region + 2) * np.pi * x + 0.17 * scene) + 0.35 * np.cos((region + 5) * np.pi * x)
            direction = np.asarray([0.7, 1.0, 0.55 + 0.02 * region])
            source[region] = np.clip(centers[region] + 0.025 * texture[:, None] * direction, 0.0, 1.0)
        target = _truth(source.reshape(-1, 3), scene).reshape(source.shape)
        permutation = rng.permutation(regions)
        observed = target[permutation]
        source_sig = np.stack([_signature(row) for row in source])
        target_sig = np.stack([_signature(row) for row in observed])
        cost = 1.0 - source_sig @ target_sig.T
        src_index, observed_index = linear_sum_assignment(cost)
        inverse = np.empty(regions, dtype=np.int64)
        inverse[src_index] = observed_index
        matched = observed[inverse]
        expected = np.argsort(permutation)
        match_rate = float(np.mean(inverse == expected))
        unmatched_w = _fit(np.mean(source, axis=1), np.mean(observed, axis=1), 1e-4)
        matched_w = _fit(np.mean(source, axis=1), np.mean(matched, axis=1), 1e-4)
        probe = rng.uniform(0.0, 1.0, size=(4096, 3))
        truth = _truth(probe, scene)
        unmatched = _bounded_residual(probe, _features(probe) @ unmatched_w)
        candidate = _bounded_residual(probe, _features(probe) @ matched_w)
        ue = float(np.sqrt(np.mean((unmatched - truth) ** 2)))
        ce = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        boundary = float(np.mean(np.any((candidate < 0.0) | (candidate > 1.0), axis=1)))
        rows.append({"scene": scene, "exact_region_match_rate": match_rate, "unmatched_rmse": ue, "matched_rmse": ce, "improvement": 1.0 - ce / ue, "new_boundary_fraction": boundary})
    metrics = {"row_count": len(rows), "exact_region_match_rate": float(np.mean([r["exact_region_match_rate"] for r in rows])), "operator_win_rate": float(np.mean([r["matched_rmse"] < r["unmatched_rmse"] for r in rows])), "median_improvement_over_unmatched": float(np.median([r["improvement"] for r in rows])), "maximum_matched_operator_rmse": max(r["matched_rmse"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows)}
    g = contract["gates"]
    checks = {"matching": metrics["exact_region_match_rate"] >= g["minimum_exact_region_match_rate"], "wins": metrics["operator_win_rate"] >= g["minimum_operator_win_rate"], "improvement": metrics["median_improvement_over_unmatched"] >= g["minimum_median_improvement_over_unmatched"], "accuracy": metrics["maximum_matched_operator_rmse"] <= g["maximum_matched_operator_rmse"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "parent_stable_evidence_id": parent["stable_evidence_id"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
