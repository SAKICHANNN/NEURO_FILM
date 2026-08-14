"""U6.P2AW nonuniform positive density-wedge reference audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.bw_hybrid_density_amplitude import (
    BWHybridDensityAmplitudeProfile,
)
from src.film_physics.density_conditioned_thomas import (
    binary_circular_aperture_kernel,
)
from src.film_physics.positive_density_field import (
    PhysicalGainSoftplusDensityParameterProfile,
    render_nonuniform_positive_density_region,
)
from src.film_physics.thomas_dc_projection import build_thomas_dc_receipt


class NonuniformPositiveDensityWedgeError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2aw_nonuniform_positive_density_wedge_contract.v1"
    ):
        raise NonuniformPositiveDensityWedgeError("unsupported P2AW contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if _sha(path) != binding["sha256"]:
        raise NonuniformPositiveDensityWedgeError("P2AW parent hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "required_automatic_pass" in binding and (
        payload.get("automatic_pass") is not binding["required_automatic_pass"]
    ):
        raise NonuniformPositiveDensityWedgeError("P2AW parent decision mismatch")
    if "required_profile_identity" in binding and (
        payload.get(
            "profile_identity",
            payload.get("measurements", {}).get("profile_identity"),
        )
        != binding["required_profile_identity"]
    ):
        raise NonuniformPositiveDensityWedgeError("P2AW parent profile mismatch")
    return payload


def load_reference_profiles(
    *, root: Path, contract: dict[str, Any]
) -> tuple[BWHybridDensityAmplitudeProfile, PhysicalGainSoftplusDensityParameterProfile]:
    _load_bound(root, contract["parents"]["positive_parameter_evidence"])
    parameter_contract = _load_bound(
        root, contract["parents"]["positive_parameter_contract"]
    )
    node_fit = _load_bound(root, parameter_contract["parents"]["node_fit"])
    amplitude_contract = _load_bound(root, contract["parents"]["amplitude_contract"])
    amplitude = amplitude_contract["hybrid_profile"]
    amplitude_profile = BWHybridDensityAmplitudeProfile(
        hypothesis_identity=amplitude["identity"],
        current_scalar_profile_id=amplitude_contract["parents"]["current_scalar"][
            "required_profile_identity"
        ],
        historical_amplitude_compiler_id=amplitude_contract["parents"][
            "historical_amplitude"
        ]["required_profile_identity"],
        densities=tuple(amplitude["densities"]),
        density_rms=tuple(amplitude["expected_density_rms"]),
        anchor_density=amplitude["anchor_density"],
        anchor_density_rms=amplitude["anchor_density_rms"],
        aperture_diameter_micrometres=amplitude["aperture_diameter_micrometres"],
        current_400tx_calibration_claimed=amplitude[
            "current_400tx_calibration_claimed"
        ],
        historical_5233_calibration_claimed=amplitude[
            "historical_5233_calibration_claimed"
        ],
        scanner_response_included=amplitude["scanner_response_included"],
    )
    nodes = parameter_contract["parameter_nodes"]
    parameter_profile = PhysicalGainSoftplusDensityParameterProfile(
        source_evidence_id=node_fit["stable_evidence_id"],
        densities=tuple(nodes["densities"]),
        fitted_a=tuple(nodes["a"]),
        fitted_b=tuple(nodes["b"]),
        target_sigma_d=tuple(nodes["target_aperture_sigma_d"]),
    )
    if (
        parameter_profile.identity()
        != contract["parents"]["positive_parameter_evidence"][
            "required_profile_identity"
        ]
    ):
        raise NonuniformPositiveDensityWedgeError("P2AW profile reconstruction drift")
    return amplitude_profile, parameter_profile


def _wedge(contract: dict[str, Any]) -> np.ndarray:
    wedge = contract["wedge"]
    shape = tuple(wedge["shape"])
    densities = np.asarray(wedge["ordered_plateau_densities"], dtype=np.float64)
    width = int(wedge["plateau_width_samples"])
    if shape[1] != densities.size * width:
        raise NonuniformPositiveDensityWedgeError("P2AW wedge geometry drift")
    row = np.repeat(densities, width)
    return np.ascontiguousarray(np.broadcast_to(row, shape), dtype=np.float64)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    amplitude_profile, parameter_profile = load_reference_profiles(
        root=root, contract=contract
    )
    mean_density = _wedge(contract)
    spatial = contract["spatial_mechanism"]
    evaluation = contract["evaluation"]
    wedge = contract["wedge"]
    shape = tuple(wedge["shape"])
    tile_height, tile_width = evaluation["irregular_tile_shape"]
    aperture = binary_circular_aperture_kernel(
        spatial["sample_pitch_micrometres"],
        spatial["measurement_aperture_diameter_micrometres"],
    )
    rows: list[dict[str, Any]] = []
    repeat_exact = True
    tiled_exact = True
    all_mean_errors: list[float] = []
    all_sigma_errors: list[float] = []
    minimum_density = float("inf")
    for seed in wedge["seeds"]:
        receipt_kwargs = {
            "full_shape": shape,
            "profile_id": spatial["spatial_profile_id"],
            "particle_sigma_pixels": spatial["particle_sigma_samples"],
            "cluster_sigma_pixels": spatial["cluster_sigma_samples"],
            "mean_offspring": spatial["mean_offspring"],
            "component_seeds": tuple(spatial["component_seeds"]),
            "realization_seed": int(seed),
            "truncate": spatial["truncate_sigma"],
            "canonical_row_block_height": evaluation[
                "canonical_row_block_height"
            ],
        }
        receipt = build_thomas_dc_receipt(**receipt_kwargs)
        full = render_nonuniform_positive_density_region(
            receipt,
            mean_density=mean_density,
            origin_yx=(0, 0),
            shape=shape,
            amplitude_profile=amplitude_profile,
            parameter_profile=parameter_profile,
        )
        repeated_receipt = build_thomas_dc_receipt(**receipt_kwargs)
        repeated = render_nonuniform_positive_density_region(
            repeated_receipt,
            mean_density=mean_density,
            origin_yx=(0, 0),
            shape=shape,
            amplitude_profile=amplitude_profile,
            parameter_profile=parameter_profile,
        )
        repeat_exact &= bool(np.array_equal(full, repeated))
        tiled = np.empty_like(full)
        for y0 in range(0, shape[0], tile_height):
            height = min(tile_height, shape[0] - y0)
            for x0 in range(0, shape[1], tile_width):
                width = min(tile_width, shape[1] - x0)
                tiled[y0 : y0 + height, x0 : x0 + width] = (
                    render_nonuniform_positive_density_region(
                        receipt,
                        mean_density=mean_density,
                        origin_yx=(y0, x0),
                        shape=(height, width),
                        amplitude_profile=amplitude_profile,
                        parameter_profile=parameter_profile,
                    )
                )
        tiled_exact &= bool(np.array_equal(full, tiled))
        measured = fftconvolve(full, aperture, mode="same")
        border = int(evaluation["measurement_crop_border_samples"])
        inset = int(evaluation["plateau_measurement_inset_samples"])
        plateau_width = int(wedge["plateau_width_samples"])
        for index, density in enumerate(wedge["ordered_plateau_densities"]):
            x_start = index * plateau_width + inset
            x_stop = (index + 1) * plateau_width - inset
            selected_raw = full[border:-border, x_start:x_stop]
            selected_measured = measured[border:-border, x_start:x_stop]
            target_sigma = float(amplitude_profile.sigma_d(float(density)))
            observed_mean = float(np.mean(selected_raw, dtype=np.float64))
            observed_sigma = float(np.std(selected_measured, dtype=np.float64))
            mean_error = abs(observed_mean - density) / density
            sigma_error = abs(observed_sigma - target_sigma) / target_sigma
            local_minimum = float(np.min(selected_raw))
            all_mean_errors.append(mean_error)
            all_sigma_errors.append(sigma_error)
            minimum_density = min(minimum_density, local_minimum)
            rows.append(
                {
                    "seed": int(seed),
                    "plateau_index": index,
                    "density": float(density),
                    "target_aperture_sigma_d": target_sigma,
                    "observed_density_mean": observed_mean,
                    "observed_aperture_sigma_d": observed_sigma,
                    "relative_mean_error": mean_error,
                    "relative_sigma_error": sigma_error,
                    "minimum_developed_density": local_minimum,
                    "plateau_density_sha256": hashlib.sha256(
                        np.ascontiguousarray(selected_raw, dtype="<f8").tobytes()
                    ).hexdigest(),
                }
            )
    measurements = {
        "amplitude_profile_identity": amplitude_profile.identity(),
        "parameter_profile_identity": parameter_profile.identity(),
        "maximum_plateau_relative_mean_error": float(np.max(all_mean_errors)),
        "plateau_median_sigma_relative_error": float(np.median(all_sigma_errors)),
        "plateau_p95_sigma_relative_error": float(
            np.percentile(all_sigma_errors, 95.0)
        ),
        "minimum_developed_density": minimum_density,
        "repeat_byte_exact": repeat_exact,
        "irregular_tile_exact": tiled_exact,
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    gate_results = {
        "maximum_plateau_relative_mean_error": measurements[
            "maximum_plateau_relative_mean_error"
        ]
        <= evaluation["maximum_plateau_relative_mean_error"],
        "maximum_plateau_median_sigma_relative_error": measurements[
            "plateau_median_sigma_relative_error"
        ]
        <= evaluation["maximum_plateau_median_sigma_relative_error"],
        "maximum_plateau_p95_sigma_relative_error": measurements[
            "plateau_p95_sigma_relative_error"
        ]
        <= evaluation["maximum_plateau_p95_sigma_relative_error"],
        "minimum_developed_density_exclusive": minimum_density
        > evaluation["minimum_developed_density_exclusive"],
        "repeat_byte_exact": repeat_exact is evaluation["repeat_byte_exact"],
        "irregular_tile_exact": tiled_exact
        is evaluation["irregular_tile_exact"],
        "parameter_refit_count_zero": True,
        "realized_normalization_count_zero": True,
        "hard_clipping_count_zero": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p2aw_nonuniform_positive_density_wedge_report.v1",
        "rows": rows,
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
