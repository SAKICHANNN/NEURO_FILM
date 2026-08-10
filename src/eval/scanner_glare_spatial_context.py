"""Frozen U6.P6ZB scanner-glare spatial-context audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.scanner_glare import (
    MultiscaleScannerGlareProfile,
    ScannerGlareComponent,
    compile_scanner_glare_kernel,
)


class ScannerGlareSpatialContextError(RuntimeError):
    """Raised when the P6ZB contract or parent identities drift."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ScannerGlareSpatialContextError("contract must be an object")
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


def _band_mask(
    radial: np.ndarray, lower: float, upper: float, count: int
) -> np.ndarray:
    eligible = np.flatnonzero(((radial >= lower) & (radial < upper)).ravel())
    if eligible.size < count:
        raise ScannerGlareSpatialContextError("context band has insufficient pixels")
    radii = radial.ravel()[eligible]
    order = np.lexsort((eligible, radii))
    selected = eligible[order[:count]]
    mask = np.zeros(radial.shape, dtype=np.bool_)
    mask.ravel()[selected] = True
    return mask


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if (
        contract.get("schema")
        != "neuro_film.u6_p6zb_scanner_glare_spatial_context_contract.v1"
    ):
        raise ScannerGlareSpatialContextError("unsupported P6ZB contract")
    parents = contract["parents"]
    bindings = (
        ("p6za_contract", "p6za_contract_sha256"),
        ("p6za_evidence", "p6za_evidence_sha256"),
        ("scanner_glare_primitive", "scanner_glare_primitive_sha256"),
    )
    parent_hashes_exact = all(
        _sha(root / parents[path_key]) == parents[sha_key]
        for path_key, sha_key in bindings
    )
    if not parent_hashes_exact:
        raise ScannerGlareSpatialContextError("parent hash mismatch")
    evidence = load_contract(root / parents["p6za_evidence"])
    if evidence.get("automatic_pass") is not True:
        raise ScannerGlareSpatialContextError("P6ZA does not admit P6ZB")
    p6za = load_contract(root / parents["p6za_contract"])
    profile = _profile(p6za["profile"])
    kernel_size = int(p6za["experiment"]["kernel_size"])
    candidate = compile_scanner_glare_kernel(profile, kernel_size=kernel_size)
    single = compile_scanner_glare_kernel(
        MultiscaleScannerGlareProfile(
            components=(
                ScannerGlareComponent(1.0, profile.second_moment_sigma_pixels),
            ),
            flare_fraction=profile.flare_fraction,
            truncate_sigma=profile.truncate_sigma,
        ),
        kernel_size=kernel_size,
    )
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(coordinates, coordinates, indexing="ij")
    radial = np.hypot(xx, yy)
    experiment = contract["experiment"]
    count = int(experiment["selected_bright_pixel_count"])
    masks = {
        name: _band_mask(radial, float(bounds[0]), float(bounds[1]), count)
        for name, bounds in experiment["radial_context_bands_pixels"].items()
    }
    dark = float(experiment["dark_transmittance"])
    contrast = float(experiment["bright_transmittance"]) - dark
    candidate_lifts = {
        name: float(profile.flare_fraction * contrast * np.sum(candidate[mask]))
        for name, mask in masks.items()
    }
    single_lifts = {
        name: float(profile.flare_fraction * contrast * np.sum(single[mask]))
        for name, mask in masks.items()
    }
    lift_ratios = {name: candidate_lifts[name] / single_lifts[name] for name in masks}
    context_counts = {name: int(np.sum(mask)) for name, mask in masks.items()}
    global_mean = dark + contrast * count / float(kernel_size * kernel_size)
    global_centers = {
        name: (1.0 - profile.flare_fraction) * dark
        + profile.flare_fraction * global_mean
        for name in masks
    }
    pointwise_centers = {name: dark for name in masks}
    rotation_error = max(
        abs(float(np.sum(candidate[mask])) - float(np.sum(candidate[np.rot90(mask)])))
        for mask in masks.values()
    )
    finite = bool(
        np.all(np.isfinite(list(candidate_lifts.values())))
        and np.all(np.isfinite(list(single_lifts.values())))
        and np.all(np.isfinite(list(lift_ratios.values())))
    )
    measurements = {
        "parent_hashes_exact": parent_hashes_exact,
        "context_bright_pixel_counts": context_counts,
        "candidate_center_lifts": candidate_lifts,
        "single_gaussian_center_lifts": single_lifts,
        "candidate_to_single_gaussian_lift_ratios": lift_ratios,
        "global_control_context_span": float(np.ptp(list(global_centers.values()))),
        "pointwise_control_context_span": float(
            np.ptp(list(pointwise_centers.values()))
        ),
        "rotation_invariance_error": rotation_error,
        "all_values_finite": finite,
        "display_rgb_effect_count": 0,
    }
    gates = contract["automatic_gates"]
    gate_results = {
        "parent_hashes_exact": parent_hashes_exact is gates["parent_hashes_exact"],
        "all_contexts_equal_bright_pixel_count": (
            set(context_counts.values()) == {count}
        )
        is gates["all_contexts_equal_bright_pixel_count"],
        "candidate_near_lift_above_intermediate": (
            candidate_lifts["near"] > candidate_lifts["intermediate"]
        )
        is gates["candidate_near_lift_above_intermediate"],
        "candidate_intermediate_lift_above_far": (
            candidate_lifts["intermediate"] > candidate_lifts["far"]
        )
        is gates["candidate_intermediate_lift_above_far"],
        "candidate_far_lift_positive": (candidate_lifts["far"] > 0.0)
        is gates["candidate_far_lift_positive"],
        "minimum_candidate_far_to_single_gaussian_lift_ratio": lift_ratios["far"]
        >= gates["minimum_candidate_far_to_single_gaussian_lift_ratio"],
        "maximum_candidate_near_to_single_gaussian_lift_ratio": lift_ratios["near"]
        <= gates["maximum_candidate_near_to_single_gaussian_lift_ratio"],
        "minimum_candidate_intermediate_to_single_gaussian_lift_ratio": lift_ratios[
            "intermediate"
        ]
        >= gates["minimum_candidate_intermediate_to_single_gaussian_lift_ratio"],
        "global_control_context_span_maximum": measurements[
            "global_control_context_span"
        ]
        <= gates["global_control_context_span_maximum"],
        "pointwise_control_context_span_maximum": measurements[
            "pointwise_control_context_span"
        ]
        <= gates["pointwise_control_context_span_maximum"],
        "rotation_invariance_error_maximum": rotation_error
        <= gates["rotation_invariance_error_maximum"],
        "all_values_finite": finite is gates["all_values_finite"],
        "display_rgb_effect_count_zero": (measurements["display_rgb_effect_count"] == 0)
        is gates["display_rgb_effect_count_zero"],
    }
    if set(gate_results) != set(gates):
        raise ScannerGlareSpatialContextError("P6ZB gate vocabulary drift")
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p6zb_scanner_glare_spatial_context_report.v1",
        "contract_sha256": _sha(
            root / "configs/u6_p6zb_scanner_glare_spatial_context_v1.json"
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
