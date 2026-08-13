"""Natural-image D1 for structural patch matching and bounded explicit color fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage import data, transform

from src.eval.region_correspondence_lut_d0 import _bounded_residual, _canonical, _features, _fit, _truth

SCHEMA = "neuro-film.u5-r2cham2-natural-patch-correspondence-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2cham2-natural-patch-correspondence-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported CHAM2 contract")
    return payload


def _signature(patch: np.ndarray) -> np.ndarray:
    gray = patch @ np.asarray([0.2126, 0.7152, 0.0722])
    gy, gx = np.gradient(gray)
    magnitude = np.hypot(gx, gy)
    small = transform.resize(magnitude, (8, 8), order=1, anti_aliasing=True, preserve_range=True)
    vector = small.ravel() - np.mean(small)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _patches(image: np.ndarray, side: int, count: int) -> np.ndarray:
    h, w = image.shape[:2]
    rows = int(np.sqrt(count))
    ys = np.linspace(0, h - side, rows, dtype=np.int64)
    xs = np.linspace(0, w - side, rows, dtype=np.int64)
    return np.stack([image[y : y + side, x : x + side] for y in ys for x in xs])


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("CHAM2 parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("CHAM2 parent decision drift")
    rng = np.random.default_rng(int(contract["evaluation"]["seed"]))
    rows = []
    for scene, binding in enumerate(contract["source"]["rows"]):
        raw = np.ascontiguousarray(getattr(data, binding["id"])())
        if hashlib.sha256(raw.tobytes()).hexdigest() != binding["sha256"]:
            raise ValueError("CHAM2 source identity drift")
        image = raw.astype(np.float64) / 255.0
        patches = _patches(image, int(contract["evaluation"]["patch_side"]), int(contract["evaluation"]["patches_per_image"]))
        target = _truth(patches.reshape(-1, 3), scene).reshape(patches.shape)
        permutation = rng.permutation(len(patches))
        observed = target[permutation]
        left = np.stack([_signature(patch) for patch in patches])
        right = np.stack([_signature(patch) for patch in observed])
        li, ri = linear_sum_assignment(1.0 - left @ right.T)
        inverse = np.empty(len(patches), dtype=np.int64)
        inverse[li] = ri
        expected = np.argsort(permutation)
        matched = observed[inverse]
        source_means = np.mean(patches, axis=(1, 2))
        unmatched_w = _fit(source_means, np.mean(observed, axis=(1, 2)), 1e-4)
        matched_w = _fit(source_means, np.mean(matched, axis=(1, 2)), 1e-4)
        flat = image.reshape(-1, 3)
        probe = flat[rng.choice(len(flat), size=min(int(contract["evaluation"]["probe_pixels_per_image"]), len(flat)), replace=False)]
        truth = _truth(probe, scene)
        unmatched = _bounded_residual(probe, _features(probe) @ unmatched_w)
        candidate = _bounded_residual(probe, _features(probe) @ matched_w)
        ue = float(np.sqrt(np.mean((unmatched - truth) ** 2)))
        ce = float(np.sqrt(np.mean((candidate - truth) ** 2)))
        rows.append({"id": binding["id"], "exact_patch_match_rate": float(np.mean(inverse == expected)), "unmatched_rmse": ue, "matched_rmse": ce, "improvement": 1.0 - ce / ue, "new_boundary_fraction": float(np.mean(np.any((candidate < 0.0) | (candidate > 1.0), axis=1)))})
    metrics = {"row_count": len(rows), "exact_patch_match_rate": float(np.mean([r["exact_patch_match_rate"] for r in rows])), "operator_win_rate": float(np.mean([r["matched_rmse"] < r["unmatched_rmse"] for r in rows])), "median_improvement_over_unmatched": float(np.median([r["improvement"] for r in rows])), "maximum_matched_operator_rmse": max(r["matched_rmse"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows)}
    g = contract["gates"]
    checks = {"matching": metrics["exact_patch_match_rate"] >= g["minimum_exact_patch_match_rate"], "wins": metrics["operator_win_rate"] >= g["minimum_operator_win_rate"], "improvement": metrics["median_improvement_over_unmatched"] >= g["minimum_median_improvement_over_unmatched"], "accuracy": metrics["maximum_matched_operator_rmse"] <= g["maximum_matched_operator_rmse"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "parent_stable_evidence_id": parent["stable_evidence_id"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
