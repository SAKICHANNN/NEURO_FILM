"""Frozen U6.P6ZA multiscale scanner-glare capacity audit."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    apply_scanner_glare,
    compile_scanner_glare_kernel,
    scanner_glare_second_moment,
)


class MultiscaleScannerGlareAuditError(RuntimeError):
    """Raised when the P6ZA contract or bound source identities drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise MultiscaleScannerGlareAuditError("contract must be an object")
    return payload


def _profile(payload: dict[str, Any]) -> MultiscaleScannerGlareProfile:
    return MultiscaleScannerGlareProfile(
        components=tuple(
            ScannerGlareComponent(
                weight=float(component["weight"]),
                sigma_pixels=float(component["sigma_pixels"]),
            )
            for component in payload["components"]
        ),
        flare_fraction=float(payload["flare_fraction"]),
        truncate_sigma=float(payload["truncate_sigma"]),
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p6za_multiscale_scanner_glare_contract.v1"
    ):
        raise MultiscaleScannerGlareAuditError("unsupported P6ZA contract")
    parents = contract["parents"]
    parent_hashes_exact = (
        _sha(root / parents["p6z_decision"]) == parents["p6z_decision_sha256"]
        and _sha(root / parents["scanner_primitive"])
        == parents["scanner_primitive_sha256"]
    )
    if not parent_hashes_exact:
        raise MultiscaleScannerGlareAuditError("parent hash mismatch")
    profile = _profile(contract["profile"])
    experiment = contract["experiment"]
    kernel_size = int(experiment["kernel_size"])
    candidate = compile_scanner_glare_kernel(profile, kernel_size=kernel_size)
    control_profile = MultiscaleScannerGlareProfile(
        components=(
            ScannerGlareComponent(
                weight=1.0,
                sigma_pixels=profile.second_moment_sigma_pixels,
            ),
        ),
        flare_fraction=profile.flare_fraction,
        truncate_sigma=profile.truncate_sigma,
    )
    control = compile_scanner_glare_kernel(control_profile, kernel_size=kernel_size)
    candidate_moment = scanner_glare_second_moment(candidate)
    control_moment = scanner_glare_second_moment(control)
    moment_relative_error = abs(candidate_moment - control_moment) / control_moment
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radial = np.hypot(xx, yy)
    tail_mask = radial >= float(experiment["tail_radius_pixels"])
    candidate_tail = float(np.sum(candidate[tail_mask]))
    control_tail = float(np.sum(control[tail_mask]))
    tail_ratio = candidate_tail / control_tail

    size = int(experiment["chart_size"])
    center = size // 2
    dark = float(experiment["dark_transmittance"])
    bright = float(experiment["bright_transmittance"])
    dark_half = int(experiment["dark_patch_half_width"])
    center_lifts = []
    pointwise_centers = []
    for bright_half in experiment["bright_context_half_widths"]:
        chart = np.full((size, size), dark, dtype=np.float64)
        extent = int(bright_half)
        chart[
            center - extent : center + extent + 1,
            center - extent : center + extent + 1,
        ] = bright
        chart[
            center - dark_half : center + dark_half + 1,
            center - dark_half : center + dark_half + 1,
        ] = dark
        rendered = apply_scanner_glare(
            chart, candidate, flare_fraction=profile.flare_fraction
        )
        center_lifts.append(float(rendered[center, center] - dark))
        pointwise_centers.append(
            float((1.0 - profile.flare_fraction) * dark + profile.flare_fraction * dark)
        )

    constant = np.full((size, size), 0.37, dtype=np.float64)
    constant_rendered = apply_scanner_glare(
        constant, candidate, flare_fraction=profile.flare_fraction
    )
    constant_error = float(np.max(np.abs(constant_rendered - constant)))
    impulse = np.zeros((size, size), dtype=np.float64)
    impulse[center, center] = 1.0
    impulse_rendered = apply_scanner_glare(
        impulse, candidate, flare_fraction=profile.flare_fraction
    )
    impulse_energy_error = abs(float(np.sum(impulse_rendered)) - 1.0)
    finite = bool(
        np.all(np.isfinite(candidate))
        and np.all(np.isfinite(control))
        and np.all(np.isfinite(center_lifts))
    )
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "kernel_minimum": float(np.min(candidate)),
        "kernel_sum_error": abs(float(np.sum(candidate)) - 1.0),
        "candidate_second_moment_sigma_pixels": candidate_moment,
        "control_second_moment_sigma_pixels": control_moment,
        "second_moment_relative_error": moment_relative_error,
        "candidate_center_peak": float(candidate[radius, radius]),
        "control_center_peak": float(control[radius, radius]),
        "candidate_tail_energy": candidate_tail,
        "control_tail_energy": control_tail,
        "multiscale_to_single_tail_energy_ratio": tail_ratio,
        "context_center_lifts": center_lifts,
        "pointwise_control_context_span": float(np.ptp(pointwise_centers)),
        "constant_field_identity_error": constant_error,
        "impulse_energy_error": impulse_energy_error,
        "all_values_finite": finite,
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "kernel_nonnegative": (measurements["kernel_minimum"] >= 0.0)
        is gates["kernel_nonnegative"],
        "kernel_sum_error_maximum": measurements["kernel_sum_error"]
        <= gates["kernel_sum_error_maximum"],
        "second_moment_relative_error_maximum": moment_relative_error
        <= gates["second_moment_relative_error_maximum"],
        "multiscale_center_peak_above_single_gaussian": (
            measurements["candidate_center_peak"] > measurements["control_center_peak"]
        )
        is gates["multiscale_center_peak_above_single_gaussian"],
        "minimum_multiscale_to_single_tail_energy_ratio": tail_ratio
        >= gates["minimum_multiscale_to_single_tail_energy_ratio"],
        "context_center_lift_strictly_increasing": all(
            right > left for left, right in pairwise(center_lifts)
        )
        is gates["context_center_lift_strictly_increasing"],
        "minimum_largest_context_center_lift": center_lifts[-1]
        >= gates["minimum_largest_context_center_lift"],
        "pointwise_control_context_span_maximum": measurements[
            "pointwise_control_context_span"
        ]
        <= gates["pointwise_control_context_span_maximum"],
        "constant_field_identity_error_maximum": constant_error
        <= gates["constant_field_identity_error_maximum"],
        "impulse_energy_error_maximum": impulse_energy_error
        <= gates["impulse_energy_error_maximum"],
        "all_values_finite": finite is gates["all_values_finite"],
        "display_rgb_effect_count_zero": (measurements["display_rgb_effect_count"] == 0)
        is gates["display_rgb_effect_count_zero"],
    }
    if set(gate_results) != set(gates):
        raise MultiscaleScannerGlareAuditError("P6ZA gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6za_multiscale_scanner_glare_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p6za_multiscale_scanner_glare_v1.json"
        ),
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(
        stable, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode()).hexdigest()
