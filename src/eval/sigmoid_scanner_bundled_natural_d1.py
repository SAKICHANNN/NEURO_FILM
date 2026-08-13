"""Natural-photo D1 for the fixed sigmoid characteristic and scanner chain."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from skimage import data

from src.color_engine.srgb_transfer import encoded_srgb_to_linear
from src.eval.characteristic_scanner_spatial_smoke import _structured_transmittance
from src.eval.layer_gamma_photographic_development import _high_frequency_chroma_p999
from src.eval.sigmoid_characteristic_spatial_d0 import _canonical, _runtime
from src.film_physics.sigmoid_characteristic import render_sigmoid_scanner_positive
from src.film_physics.spatial_response import SpatialResponseProfile, apply_scanner_mtf

SCHEMA = "neuro-film.u6-p4il-sigmoid-scanner-bundled-natural-d1-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4il-sigmoid-scanner-bundled-natural-d1-result.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4IL contract")
    return payload


def evaluate(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parent_path = root / str(contract["parent"]["path"])
    if hashlib.sha256(parent_path.read_bytes()).hexdigest() != contract["parent"]["sha256"]:
        raise ValueError("P4IL parent identity drift")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != contract["parent"]["required_decision"]:
        raise ValueError("P4IL parent decision drift")
    curves, fit_rmse, compiler = _runtime(root, json.loads((root / "configs/u6_p4ij_sigmoid_characteristic_spatial_d0_v1.json").read_text()))
    mechanism = contract["mechanism"]
    profile = SpatialResponseProfile(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), tuple(mechanism["scanner_mtf_sigma_pixels_rgb"]), float(mechanism["gaussian_truncate"]))
    rows = []
    for index, source_id in enumerate(contract["source"]["ids"]):
        raw = np.ascontiguousarray(getattr(data, source_id)())
        encoded = raw.astype(np.float64) / 255.0
        source = np.ascontiguousarray(encoded_srgb_to_linear(encoded), dtype=np.float32)
        density = np.stack([curve.apply_normalized(source[..., channel]) for channel, curve in enumerate(curves)], axis=-1)
        base_t = np.ascontiguousarray(np.power(10.0, -density), dtype=np.float32)
        baseline, _ = render_sigmoid_scanner_positive(source, base_t, curves=curves, compiler=compiler)
        structured = _structured_transmittance(base_t, index, float(mechanism["structure_amplitude"]))
        candidate, receipt = render_sigmoid_scanner_positive(source, structured, curves=curves, compiler=compiler)
        replay, _ = render_sigmoid_scanner_positive(source, structured, curves=curves, compiler=compiler)
        baseline = apply_scanner_mtf(baseline, profile).astype(np.float32)
        candidate = apply_scanner_mtf(candidate, profile).astype(np.float32)
        replay = apply_scanner_mtf(replay, profile).astype(np.float32)
        absolute = np.abs(candidate.astype(np.float64) - baseline)
        epsilon = 1.0 / 65535.0
        before = (baseline <= epsilon) | (baseline >= 1.0 - epsilon)
        after = (candidate <= epsilon) | (candidate >= 1.0 - epsilon)
        rows.append({"id": source_id, "source_sha256": hashlib.sha256(raw.tobytes()).hexdigest(), "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(), "p95_abs_difference": float(np.percentile(absolute, 95)), "p99_abs_difference": float(np.percentile(absolute, 99)), "high_frequency_chroma_p999": _high_frequency_chroma_p999(candidate.astype(np.float64) - baseline.astype(np.float64)), "new_boundary_fraction": float(np.mean(after & ~before)), "repeat_error": float(np.max(np.abs(candidate - replay))), "minimum_shared_scale": receipt["minimum_shared_scale"]})
    metrics = {"row_count": len(rows), "curve_fit_rmse": list(fit_rmse), "population_p95_abs_difference": float(np.percentile([r["p95_abs_difference"] for r in rows], 95)), "population_p99_abs_difference": float(np.percentile([r["p99_abs_difference"] for r in rows], 99)), "maximum_high_frequency_chroma_p999": max(r["high_frequency_chroma_p999"] for r in rows), "maximum_new_boundary_fraction": max(r["new_boundary_fraction"] for r in rows), "maximum_repeat_error": max(r["repeat_error"] for r in rows), "minimum_shared_scale": min(r["minimum_shared_scale"] for r in rows)}
    g = contract["gates"]
    checks = {"material": metrics["population_p95_abs_difference"] >= g["minimum_population_p95_abs_difference"], "tail": metrics["population_p99_abs_difference"] <= g["maximum_population_p99_abs_difference"], "chroma": metrics["maximum_high_frequency_chroma_p999"] <= g["maximum_high_frequency_chroma_p999"], "boundary": metrics["maximum_new_boundary_fraction"] <= g["maximum_new_boundary_fraction"], "repeat": metrics["maximum_repeat_error"] <= g["maximum_repeat_error"], "retained_direction": metrics["minimum_shared_scale"] >= g["minimum_shared_scale"]}
    passed = all(checks.values())
    core = {"schema": REPORT_SCHEMA, "experiment_id": contract["experiment_id"], "config_sha256": hashlib.sha256(_canonical(contract)).hexdigest(), "parent_stable_evidence_id": parent["formal_runs"]["stable_evidence_id"], "rows": rows, "metrics": metrics, "checks": checks, "automatic_pass": passed, "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"], "claim_ceiling": contract["claim_ceiling"]}
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()
