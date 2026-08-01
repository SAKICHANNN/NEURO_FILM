"""Synthetic U6.P3P positive-spread backing-return evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.sensitometry_primitive import build_operator
from src.film_physics.backing_return import backing_return_profile_from_contract
from src.film_physics.contracts import PhysicalDomain, PhysicalDomainArray, PhysicalScale, PhysicalUnit
from src.film_physics.positive_spread_return import apply_positive_spread_backing_return
from src.film_physics.response_bounded_exposure import apply_response_bounded_exposure_residual, scan_transmittance


SCHEMA = "neuro_film.u6_p3p_positive_spread_backing_return.v1"


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _load_exact(root: Path, path: str, sha256: str) -> dict[str, Any]:
    raw = (root / path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"parent evidence drift: {path}")
    return json.loads(raw)


def _array(values: np.ndarray, pixel_pitch_um: float) -> PhysicalDomainArray:
    return PhysicalDomainArray(
        np.asarray(values, dtype=np.float64),
        PhysicalDomain.LAYER_EXPOSURE,
        PhysicalUnit.RELATIVE_LAYER_EXPOSURE,
        ("red", "green", "blue"),
        PhysicalScale(pixel_pitch_um),
    )


def evaluate_positive_spread(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    if (
        config.get("schema") != SCHEMA
        or config.get("training_allowed")
        or config.get("production_integration_allowed")
        or config["candidate"]["fitting_allowed"]
        or config["candidate"]["per_image_parameter_selection_allowed"]
        or config["candidate"]["hard_clipping_allowed"]
    ):
        raise ValueError("invalid U6.P3P contract")
    parents = config["parents"]
    p3d = _load_exact(root, parents["backing_return_contract"], parents["backing_return_contract_sha256"])
    sensitometry_contract = _load_exact(root, parents["sensitometry_contract"], parents["sensitometry_contract_sha256"])
    p3n = _load_exact(root, parents["p3n_decision"], parents["p3n_decision_sha256"])
    p3o = _load_exact(root, parents["p3o_decision"], parents["p3o_decision_sha256"])
    if p3n.get("decision") != "retain_development_challenger_require_fresh_value_confirmation" or p3o.get("decision") != "close_product_value_retain_research_primitive":
        raise ValueError("P3N/P3O parent decision drift")
    profile = backing_return_profile_from_contract(p3d)
    operator = build_operator(sensitometry_contract)
    shape = tuple(map(int, config["witnesses"]["shape"]))
    background = float(config["witnesses"]["background_exposure"])
    highlight = float(config["witnesses"]["highlight_exposure"])
    constant = np.full(shape, float(config["witnesses"]["constant_exposure"]), dtype=np.float64)
    impulse = np.full(shape, background, dtype=np.float64)
    center = (shape[0] // 2, shape[1] // 2)
    impulse[center[0], center[1], :] = highlight
    edge = np.full(shape, background, dtype=np.float64)
    edge[:, shape[1] // 2 :, :] = highlight

    constant_result = apply_positive_spread_backing_return(_array(constant, profile.pixel_pitch_um), profile)
    impulse_result = apply_positive_spread_backing_return(_array(impulse, profile.pixel_pitch_um), profile)
    edge_result = apply_positive_spread_backing_return(_array(edge, profile.pixel_pitch_um), profile)
    bounded = apply_response_bounded_exposure_residual(
        impulse,
        impulse_result.exposure.values,
        operator,
        maximum_transmittance_delta=float(config["candidate"]["maximum_transmittance_delta"]),
    )
    bounded_delta = np.abs(scan_transmittance(bounded.exposure, operator) - scan_transmittance(impulse, operator))
    radius = int(config["witnesses"]["far_radius_pixels"])
    far = impulse_result.residual[center[0], center[1] + radius]
    dark_side = edge_result.residual[:, : shape[1] // 2]
    bright_side = edge_result.residual[:, shape[1] // 2 :]
    metrics = {
        "constant_maximum_residual": float(np.max(np.abs(constant_result.residual))),
        "impulse_center_maximum_increment": float(np.max(impulse_result.residual[center])),
        "impulse_halo_energy": float(np.sum(impulse_result.residual)),
        "far_red_over_blue_ratio": float(far[0] / max(far[2], 1e-30)),
        "edge_dark_side_maximum_increment": float(np.max(dark_side)),
        "edge_bright_side_maximum_increment": float(np.max(bright_side)),
        "bounded_maximum_transmittance_delta": float(np.max(bounded_delta)),
        "bounded_p99_transmittance_delta": float(np.quantile(bounded_delta, 0.99)),
        "minimum_output_exposure": float(np.min(bounded.exposure)),
        "out_of_domain_fraction": float(np.mean(~np.isfinite(bounded.exposure) | (bounded.exposure < 0.0))),
    }
    gates = config["automatic_gates"]
    checks = {
        "constant_identity": metrics["constant_maximum_residual"] <= float(gates["maximum_constant_residual"]),
        "impulse_center": metrics["impulse_center_maximum_increment"] <= float(gates["maximum_impulse_center_increment"]),
        "impulse_halo": metrics["impulse_halo_energy"] >= float(gates["minimum_impulse_halo_energy"]),
        "spectral_tail": metrics["far_red_over_blue_ratio"] >= float(gates["minimum_far_red_over_blue_ratio"]),
        "edge_dark_side": metrics["edge_dark_side_maximum_increment"] >= float(gates["minimum_edge_dark_side_increment"]),
        "edge_bright_side": metrics["edge_bright_side_maximum_increment"] <= float(gates["maximum_edge_bright_side_increment"]),
        "response_maximum": metrics["bounded_maximum_transmittance_delta"] <= float(gates["maximum_bounded_transmittance_delta"]),
        "response_visible": metrics["bounded_maximum_transmittance_delta"] >= float(gates["minimum_bounded_transmittance_delta"]),
        "domain": metrics["out_of_domain_fraction"] <= float(gates["maximum_out_of_domain_fraction"]),
    }
    passed = bool(all(checks.values()))
    stable = {
        "schema": "neuro_film.u6_p3p_positive_spread_backing_return_report.v1",
        "node": config["node"],
        "metrics": metrics,
        "checks": checks,
        "automatic_pass": passed,
        "branch": config["branches"]["pass" if passed else "fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(_canonical_bytes(stable)).hexdigest()}


__all__ = ["SCHEMA", "evaluate_positive_spread"]
