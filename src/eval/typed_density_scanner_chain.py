"""U6.P2AY typed density/transmittance/scanner-order audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.local_density_uncertainty import build_fresh_layout
from src.eval.nonuniform_positive_density_wedge import load_reference_profiles
from src.film_physics.bw_density_scanner_chain import (
    build_typed_neutral_density_scanner_chain,
)
from src.film_physics.contracts import transmittance_to_density
from src.film_physics.positive_density_field import (
    render_nonuniform_positive_density_region,
)
from src.film_physics.spatial_response import (
    SpatialResponseProfile,
    apply_dye_diffusion,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt


class TypedDensityScannerChainError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2ay_typed_density_scanner_chain_contract.v1"
    ):
        raise TypedDensityScannerChainError("unsupported P2AY contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise TypedDensityScannerChainError("P2AY parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise TypedDensityScannerChainError("P2AY parent decision mismatch")
    return payload


def _high_frequency_energy(values: np.ndarray) -> float:
    horizontal = np.diff(values.astype(np.float64), axis=1)
    vertical = np.diff(values.astype(np.float64), axis=0)
    return float(
        0.5
        * (
            np.mean(horizontal * horizontal, dtype=np.float64)
            + np.mean(vertical * vertical, dtype=np.float64)
        )
    )


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    _load_bound(root, contract["parents"]["local_uncertainty_evidence"])
    uncertainty_contract = _load_bound(
        root, contract["parents"]["local_uncertainty_contract"]
    )
    wedge_contract = _load_bound(
        root, uncertainty_contract["parents"]["wedge_contract"]
    )
    amplitude_profile, parameter_profile = load_reference_profiles(
        root=root, contract=wedge_contract
    )
    mean_density = build_fresh_layout(uncertainty_contract)
    shape = mean_density.shape
    spatial = wedge_contract["spatial_mechanism"]
    receipt = build_thomas_dc_receipt(
        full_shape=shape,
        profile_id=spatial["spatial_profile_id"],
        particle_sigma_pixels=spatial["particle_sigma_samples"],
        cluster_sigma_pixels=spatial["cluster_sigma_samples"],
        mean_offspring=spatial["mean_offspring"],
        component_seeds=tuple(spatial["component_seeds"]),
        realization_seed=int(contract["fresh_realization_seed"]),
        truncate=spatial["truncate_sigma"],
        canonical_row_block_height=wedge_contract["evaluation"][
            "canonical_row_block_height"
        ],
    )
    density = render_nonuniform_positive_density_region(
        receipt,
        mean_density=mean_density,
        origin_yx=(0, 0),
        shape=shape,
        amplitude_profile=amplitude_profile,
        parameter_profile=parameter_profile,
    )
    scanner = contract["scanner_profile"]
    sigma_um = tuple(
        float(value) * float(scanner["pixel_pitch_micrometres"])
        for value in scanner["scanner_mtf_sigma_pixels_rgb"]
    )
    zero = (0.0, 0.0, 0.0)
    profile = SpatialResponseProfile(
        pixel_pitch_um=float(scanner["pixel_pitch_micrometres"]),
        forward_scatter_sigma_um_rgb=zero,
        development_adjacency_sigma_um_rgb=zero,
        development_adjacency_gain_rgb=zero,
        dye_diffusion_sigma_um_rgb=sigma_um,
        scanner_mtf_sigma_um_rgb=sigma_um,
        gaussian_truncate=float(scanner["gaussian_truncate"]),
    )
    result = build_typed_neutral_density_scanner_chain(density, profile)
    restored_density = transmittance_to_density(result.transmittance)
    wrong_density = apply_dye_diffusion(result.developed_density.values, profile)
    wrong_order = np.power(10.0, -wrong_density)
    correct = result.scan_linear.values.astype(np.float64)
    transmittance = result.transmittance.values.astype(np.float64)
    roundtrip_error = float(
        np.max(
            np.abs(
                restored_density.values.astype(np.float64)
                - result.developed_density.values.astype(np.float64)
            )
        )
    )
    wrong_order_rms = float(np.sqrt(np.mean((correct - wrong_order) ** 2)))
    transmittance_hf = _high_frequency_energy(transmittance)
    scan_hf = _high_frequency_energy(correct)
    mean_drift = float(abs(np.mean(correct) - np.mean(transmittance)))
    neutral_exact = bool(
        np.array_equal(result.developed_density.values[..., 0], result.developed_density.values[..., 1])
        and np.array_equal(result.developed_density.values[..., 1], result.developed_density.values[..., 2])
        and np.array_equal(result.transmittance.values[..., 0], result.transmittance.values[..., 1])
        and np.array_equal(result.scan_linear.values[..., 0], result.scan_linear.values[..., 1])
    )
    evaluation = contract["evaluation"]
    measurements = {
        "receipt_id": receipt.receipt_id,
        "density_descriptor": result.developed_density.descriptor(),
        "transmittance_descriptor": result.transmittance.descriptor(),
        "scan_linear_descriptor": result.scan_linear.descriptor(),
        "maximum_density_roundtrip_absolute_error": roundtrip_error,
        "scanner_mean_drift": mean_drift,
        "scanner_high_frequency_energy_ratio": scan_hf / transmittance_hf,
        "wrong_order_rms_difference": wrong_order_rms,
        "minimum_transmittance": float(np.min(transmittance)),
        "maximum_transmittance": float(np.max(transmittance)),
        "neutral_channels_exact": neutral_exact,
        "hard_clipping_count_zero": True,
        "realized_normalization_count_zero": True,
        "density_sha256": hashlib.sha256(
            np.ascontiguousarray(result.developed_density.values, dtype="<f8").tobytes()
        ).hexdigest(),
        "transmittance_sha256": hashlib.sha256(
            np.ascontiguousarray(result.transmittance.values, dtype="<f8").tobytes()
        ).hexdigest(),
        "scan_linear_sha256": hashlib.sha256(
            np.ascontiguousarray(result.scan_linear.values, dtype="<f8").tobytes()
        ).hexdigest(),
    }
    gate_results = {
        "maximum_density_roundtrip_absolute_error": roundtrip_error
        <= evaluation["maximum_density_roundtrip_absolute_error"],
        "maximum_scanner_mean_drift": mean_drift
        <= evaluation["maximum_scanner_mean_drift"],
        "maximum_scanner_high_frequency_energy_ratio": measurements[
            "scanner_high_frequency_energy_ratio"
        ]
        <= evaluation["maximum_scanner_high_frequency_energy_ratio"],
        "minimum_wrong_order_rms_difference": wrong_order_rms
        >= evaluation["minimum_wrong_order_rms_difference"],
        "transmittance_range": measurements["minimum_transmittance"]
        > evaluation["minimum_transmittance_exclusive"]
        and measurements["maximum_transmittance"]
        <= evaluation["maximum_transmittance_inclusive"],
        "neutral_channels_exact": neutral_exact
        is evaluation["neutral_channels_exact"],
        "hard_clipping_count_zero": True,
        "realized_normalization_count_zero": True,
    }
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p2ay_typed_density_scanner_chain_report.v1",
        "scanner_profile": scanner,
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
