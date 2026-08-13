"""Candidate-independent curvature selector for Gaussian versus lattice LUTs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_e6_bounded_dye_operator import STOCKS, _load_source_curves, compile_operator, load_contract as load_cb2
from src.eval.gaussian_lut_representation_d0 import _canonical, _cube, _fit, _gaussian_features, _lattice_features, _target
from src.film_physics.spectral_scanner import synthetic_profile_from_contract

SCHEMA = "neuro-film.u5-r2glut2-curvature-representation-selector-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u5-r2glut2-curvature-representation-selector-d0-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported GLUT2 contract")
    return payload


def _curvature(values: np.ndarray, side: int) -> float:
    grid = values.reshape(side, side, side, 3)
    second = []
    for axis in range(3):
        second.append(np.diff(grid, n=2, axis=axis).ravel())
    return float(np.sqrt(np.mean(np.concatenate(second) ** 2)))


def _observations(contract: Mapping[str, Any], root: Path) -> list[dict[str, float | str]]:
    p = contract["population"]
    side = int(p["grid_side"])
    cube = _cube(side)
    index = np.rint(cube * (side - 1)).astype(int)
    mask = np.all(index % int(p["fit_stride"]) == 0, axis=1)
    train, test = cube[mask], cube[~mask]
    gf_train = _gaussian_features(train, int(p["gaussian_side"]), float(p["gaussian_sigma"]))
    gf_test = _gaussian_features(test, int(p["gaussian_side"]), float(p["gaussian_sigma"]))
    lf_train = _lattice_features(train, int(p["lattice_side"]))
    lf_test = _lattice_features(test, int(p["lattice_side"]))

    truths: list[tuple[str, np.ndarray]] = [
        (f"procedural-{i:02d}", _target(cube, i))
        for i in range(int(p["procedural_operator_count"]))
    ]
    cb2 = load_cb2(root / "configs/u5_r2cb2_fujifilm_e6_bounded_dye_operator_v1.json")
    wavelength, curves = _load_source_curves(cb2, root)
    scanners = [synthetic_profile_from_contract(wavelength, row) for row in cb2["scanner_profiles"]]
    for scanner in scanners:
        for stock in STOCKS:
            truth, _ = compile_operator(cube, curves[stock], scanner, cb2)
            truths.append((f"e6-{scanner.profile_id}-{stock}", truth.astype(np.float64)))
    rows = []
    for name, truth in truths:
        gw = _fit(gf_train, truth[mask], float(p["ridge"]))
        lw = _fit(lf_train, truth[mask], float(p["ridge"]))
        ge = float(np.sqrt(np.mean((gf_test @ gw - truth[~mask]) ** 2)))
        le = float(np.sqrt(np.mean((lf_test @ lw - truth[~mask]) ** 2)))
        rows.append(
            {
                "id": name,
                "family": name.split("-", 1)[0],
                "curvature": _curvature(truth[mask], (side + 1) // int(p["fit_stride"])),
                "gaussian_rmse": ge,
                "lattice_rmse": le,
                "winner": "gaussian" if ge < le else "lattice",
            }
        )
    return rows


def _fit_selector(rows: list[dict[str, float | str]]) -> dict[str, float | str]:
    candidates = sorted(float(row["curvature"]) for row in rows)
    thresholds = [0.0] + [
        (a + b) / 2.0 for a, b in zip(candidates[:-1], candidates[1:], strict=True)
    ] + [float("inf")]
    best: tuple[int, float, str, float] | None = None
    for threshold in thresholds:
        for high in ("gaussian", "lattice"):
            low = "lattice" if high == "gaussian" else "gaussian"
            correct = sum(
                (high if float(row["curvature"]) >= threshold else low) == row["winner"]
                for row in rows
            )
            candidate = (
                correct,
                -threshold if np.isfinite(threshold) else -1e300,
                high,
                threshold,
            )
            if best is None or candidate > best:
                best = candidate
    assert best is not None
    correct, _, high, threshold = best
    return {
        "training_accuracy": correct / len(rows),
        "curvature_threshold": threshold,
        "high_curvature_representation": high,
        "low_curvature_representation": "lattice" if high == "gaussian" else "gaussian",
    }


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    rows = _observations(contract, root)
    predictions = []
    folds = int(contract["population"]["cross_validation_folds"])
    for fold in range(folds):
        test = [row for i, row in enumerate(rows) if i % folds == fold]
        train = [row for i, row in enumerate(rows) if i % folds != fold]
        selector = _fit_selector(train)
        high = str(selector["high_curvature_representation"])
        low = str(selector["low_curvature_representation"])
        threshold = float(selector["curvature_threshold"])
        for row in test:
            selected = high if row["curvature"] >= threshold else low
            predictions.append({**row, "selected": selected})
    accuracy = float(np.mean([row["selected"] == row["winner"] for row in predictions]))
    selected = np.array([row[f"{row['selected']}_rmse"] for row in predictions])
    best = np.array([min(row["gaussian_rmse"], row["lattice_rmse"]) for row in predictions])
    lattice = np.array([row["lattice_rmse"] for row in predictions])
    gaussian = np.array([row["gaussian_rmse"] for row in predictions])
    regret = selected - best
    lattice_regret = lattice - best
    gaussian_regret = gaussian - best
    metrics = {"row_count": len(predictions), "selector_accuracy": accuracy, "oracle_regret_reduction_vs_always_lattice": 1.0 - float(np.sum(regret)) / max(float(np.sum(lattice_regret)), 1e-15), "oracle_regret_reduction_vs_always_gaussian": 1.0 - float(np.sum(regret)) / max(float(np.sum(gaussian_regret)), 1e-15), "worst_selected_vs_best_rmse_ratio": float(np.max(selected / np.maximum(best, 1e-15)))}
    g = contract["gates"]
    checks = {"accuracy": accuracy >= g["minimum_selector_accuracy"], "lattice_regret": metrics["oracle_regret_reduction_vs_always_lattice"] >= g["minimum_oracle_regret_reduction_vs_always_lattice"], "gaussian_regret": metrics["oracle_regret_reduction_vs_always_gaussian"] >= g["minimum_oracle_regret_reduction_vs_always_gaussian"], "tail": metrics["worst_selected_vs_best_rmse_ratio"] <= g["maximum_worst_selected_vs_best_rmse_ratio"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "frozen_selector": _fit_selector(rows), "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
