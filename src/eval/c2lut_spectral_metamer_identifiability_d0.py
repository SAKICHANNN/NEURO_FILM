"""Physical spectral-integration D0 for same-chromaticity illuminant ambiguity."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro-film.u5-r2bx1-c2lut-spectral-metamer-identifiability-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2bx1-c2lut-spectral-metamer-identifiability-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported BX1 contract")
    return payload


def _gaussian(w: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - center) / width) ** 2)


def _sensors(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    observer = np.stack((_gaussian(w, 600, 45) + 0.35 * _gaussian(w, 445, 25), _gaussian(w, 550, 38), 1.25 * _gaussian(w, 445, 28)), axis=1)
    camera = np.stack((_gaussian(w, 610, 38) + 0.1 * _gaussian(w, 475, 20), _gaussian(w, 535, 34) + 0.12 * _gaussian(w, 610, 28), _gaussian(w, 455, 31) + 0.08 * _gaussian(w, 560, 25)), axis=1)
    return observer, camera


def _reflectances(rng: np.random.Generator, w: np.ndarray, count: int) -> np.ndarray:
    basis = np.stack([np.ones(len(w)), *(np.cos(k * np.pi * (w - w[0]) / (w[-1] - w[0])) for k in range(1, 9))], axis=1)
    coefficients = rng.normal(0.0, 0.55, size=(count, basis.shape[1]))
    coefficients[:, 0] = rng.uniform(-0.8, 0.8, size=count)
    logits = coefficients @ basis.T
    return 1.0 / (1.0 + np.exp(-logits))


def _fit(rgb: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    x = np.column_stack((rgb, np.ones(len(rgb))))
    return np.linalg.lstsq(x, xyz, rcond=None)[0]


def _apply(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return np.column_stack((rgb, np.ones(len(rgb)))) @ matrix


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("BX1 parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("BX1 parent decision drift")
    p = contract["population"]
    w = np.arange(p["wavelength_start_nm"], p["wavelength_end_nm"] + 1, p["wavelength_step_nm"], dtype=np.float64)
    observer, camera = _sensors(w)
    rng = np.random.default_rng(int(p["seed"]))
    reflectances = _reflectances(rng, w, int(p["reflectances"]))
    dev = int(p["development_reflectances"])
    rows = []
    for pair in range(int(p["illuminant_pairs"])):
        base = 0.35 + 0.65 * _gaussian(w, 470 + 28 * pair, 75 + 3 * pair)
        # Project a smooth random perturbation into the observer-white nullspace.
        raw = rng.normal(size=len(w))
        raw = np.convolve(raw, np.ones(5) / 5.0, mode="same")
        projection = observer @ np.linalg.lstsq(observer, raw, rcond=None)[0]
        null = raw - projection
        null /= max(float(np.max(np.abs(null))), 1e-12)
        amplitude = 0.28 * float(np.min(base))
        illuminants = (base, base + amplitude * null)
        white_xyz = [ill @ observer for ill in illuminants]
        xyz_error = float(np.max(np.abs(white_xyz[0] - white_xyz[1])))
        white_rgb = [ill @ camera for ill in illuminants]
        rgb_separation = float(np.linalg.norm(white_rgb[0] / np.sum(white_rgb[0]) - white_rgb[1] / np.sum(white_rgb[1])))
        rgb_sets, xyz_sets = [], []
        for illuminant in illuminants:
            rgb = (reflectances * illuminant) @ camera
            xyz = (reflectances * illuminant) @ observer
            rgb_sets.append(rgb / np.maximum(np.sum(white_rgb[0]), 1e-12))
            xyz_sets.append(xyz / np.maximum(np.sum(white_xyz[0]), 1e-12))
        shared = _fit(np.concatenate([x[:dev] for x in rgb_sets]), np.concatenate([x[:dev] for x in xyz_sets]))
        for hidden, (rgb, xyz) in enumerate(zip(rgb_sets, xyz_sets, strict=True)):
            oracle = _fit(rgb[:dev], xyz[:dev])
            shared_error = float(np.sqrt(np.mean((_apply(rgb[dev:], shared) - xyz[dev:]) ** 2)))
            oracle_error = float(np.sqrt(np.mean((_apply(rgb[dev:], oracle) - xyz[dev:]) ** 2)))
            rows.append({"pair": pair, "hidden_spd": hidden, "same_chromaticity_xyz_error": xyz_error, "camera_white_rgb_separation": rgb_separation, "shared_xyz_rmse": shared_error, "oracle_xyz_rmse": oracle_error, "oracle_improvement": 1.0 - oracle_error / shared_error})
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
