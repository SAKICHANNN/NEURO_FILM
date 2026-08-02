"""U6.P4AZ exact density-to-transmittance granularity evaluator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import ManufacturerCharacteristicPrior
from src.film_physics.transmittance_granularity import (
    TransmittanceGranularityProfile,
    density_delta_method_transmittance_rms,
    transmittance_moments_to_density_gaussian,
)

SCHEMA = "neuro_film.u6_p4az_density_to_transmittance_granularity_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4az_density_to_transmittance_granularity_report.v1"


class DensityTransmittanceGranularityError(RuntimeError):
    """Raised when frozen P4AZ evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise DensityTransmittanceGranularityError("P4AZ paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or compiler.get("density_distribution_family")
        != "gaussian_small_fluctuation_compatibility"
        or compiler.get("physical_transform") != "transmittance = pow(10, -density)"
        or compiler.get("moment_family") != "exact_log_normal"
        or compiler.get("channel_order") != ["red", "green", "blue"]
        or compiler.get("spatial_spectrum_selection_allowed")
        or compiler.get("clipping_or_display_rgb_noise_allowed")
        or compiler.get("parameter_refit_allowed")
        or evaluation.get("relative_exposure_domain_fractions")
        != [0.15, 0.3, 0.5, 0.7, 0.85]
        or evaluation.get("gauss_hermite_order") != 96
        or evaluation.get("maximum_relative_mean_quadrature_error") != 1e-12
        or evaluation.get("maximum_relative_rms_quadrature_error") != 1e-10
        or evaluation.get("maximum_density_inverse_error") != 1e-12
        or evaluation.get("maximum_rms_inverse_error") != 1e-12
        or evaluation.get("maximum_delta_method_relative_rms_error") != 0.001
        or evaluation.get("minimum_transmittance_rms_dynamic_range_ratio") != 100.0
        or not evaluation.get("require_positive_finite_moments")
        or not evaluation.get("require_two_byte_identical_reports")
    ):
        raise DensityTransmittanceGranularityError("P4AZ frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise DensityTransmittanceGranularityError(
            f"P4AZ parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _quadrature_moments(
    density_mean: float, density_rms: float, order: int
) -> tuple[float, float]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    density = density_mean + np.sqrt(2.0) * density_rms * nodes
    transmittance = np.power(10.0, -density)
    normalized = weights / np.sqrt(np.pi)
    mean = float(np.sum(normalized * transmittance, dtype=np.float64))
    variance = float(
        np.sum(normalized * np.square(transmittance - mean), dtype=np.float64)
    )
    return mean, float(np.sqrt(variance))


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4ax_decision")
    density_payload = _load_parent(parents, root, "p4ax_bundle")
    prior_payload = _load_parent(parents, root, "p2q_bundle")
    if (
        decision.get("decision") != "retain_amplitude_only_profile"
        or decision.get("profile_id") != parents["p4ax_profile_id"]
        or decision.get("automatic_pass") is not True
    ):
        raise DensityTransmittanceGranularityError("P4AZ parent decision mismatch")
    density_payload = dict(density_payload)
    profile_id = density_payload.pop("profile_id")
    density_profile = GranularityAmplitudeProfile.from_dict(density_payload)
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if (
        profile_id != parents["p4ax_profile_id"]
        or density_profile.identity() != profile_id
        or prior.identity() != parents["p2q_prior_identity"]
    ):
        raise DensityTransmittanceGranularityError("P4AZ profile identity mismatch")
    profile = TransmittanceGranularityProfile(
        density_profile, str(decision["stable_evidence_id"])
    )
    bundle = profile.to_dict()
    bundle["profile_id"] = profile.identity()

    fractions = contract["evaluation"]["relative_exposure_domain_fractions"]
    rows: list[dict[str, Any]] = []
    mean_errors: list[float] = []
    rms_errors: list[float] = []
    inverse_density_errors: list[float] = []
    inverse_rms_errors: list[float] = []
    delta_errors: list[float] = []
    transmittance_rms_values: list[float] = []
    for channel_index, curve in enumerate(prior.curves):
        for fraction in fractions:
            exposure = curve.domain[0] + float(fraction) * (
                curve.domain[1] - curve.domain[0]
            )
            input_value = np.full((1, 3), exposure, dtype=np.float64)
            moments = profile.evaluate(prior, input_value)
            density_mean = float(moments.density_mean[0, channel_index])
            density_rms = float(moments.density_rms[0, channel_index])
            transmittance_mean = float(moments.transmittance_mean[0, channel_index])
            transmittance_rms = float(moments.transmittance_rms[0, channel_index])
            quadrature_mean, quadrature_rms = _quadrature_moments(
                density_mean,
                density_rms,
                int(contract["evaluation"]["gauss_hermite_order"]),
            )
            recovered_mean, recovered_rms = transmittance_moments_to_density_gaussian(
                np.asarray([transmittance_mean]), np.asarray([transmittance_rms])
            )
            delta_rms = float(
                density_delta_method_transmittance_rms(
                    np.asarray([density_mean]), np.asarray([density_rms])
                )[0]
            )
            mean_error = abs(quadrature_mean / transmittance_mean - 1.0)
            rms_error = abs(quadrature_rms / transmittance_rms - 1.0)
            inverse_density_error = abs(float(recovered_mean[0]) - density_mean)
            inverse_rms_error = abs(float(recovered_rms[0]) - density_rms)
            delta_error = abs(delta_rms / transmittance_rms - 1.0)
            mean_errors.append(mean_error)
            rms_errors.append(rms_error)
            inverse_density_errors.append(inverse_density_error)
            inverse_rms_errors.append(inverse_rms_error)
            delta_errors.append(delta_error)
            transmittance_rms_values.append(transmittance_rms)
            rows.append(
                {
                    "channel": curve.layer,
                    "domain_fraction": float(fraction),
                    "log_exposure": exposure,
                    "density_mean": density_mean,
                    "density_rms": density_rms,
                    "transmittance_mean": transmittance_mean,
                    "transmittance_rms": transmittance_rms,
                    "quadrature_mean": quadrature_mean,
                    "quadrature_rms": quadrature_rms,
                    "delta_method_rms": delta_rms,
                }
            )
    dynamic_range = max(transmittance_rms_values) / min(transmittance_rms_values)
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "positive_finite_moments": all(
            np.isfinite(value) and value > 0.0
            for row in rows
            for value in (
                row["density_rms"],
                row["transmittance_mean"],
                row["transmittance_rms"],
            )
        ),
        "mean_quadrature": max(mean_errors)
        <= float(gates["maximum_relative_mean_quadrature_error"]),
        "rms_quadrature": max(rms_errors)
        <= float(gates["maximum_relative_rms_quadrature_error"]),
        "density_inverse": max(inverse_density_errors)
        <= float(gates["maximum_density_inverse_error"]),
        "rms_inverse": max(inverse_rms_errors)
        <= float(gates["maximum_rms_inverse_error"]),
        "delta_method_diagnostic": max(delta_errors)
        <= float(gates["maximum_delta_method_relative_rms_error"]),
        "nonconstant_transmittance_amplitude": dynamic_range
        >= float(gates["minimum_transmittance_rms_dynamic_range_ratio"]),
        "spatial_structure_unidentified": bundle["spatial_structure_status"]
        == "unidentified",
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_sha256": hashlib.sha256(_canonical_json(bundle)).hexdigest(),
        "probe_count": len(rows),
        "maximum_relative_mean_quadrature_error": max(mean_errors),
        "maximum_relative_rms_quadrature_error": max(rms_errors),
        "maximum_density_inverse_error": max(inverse_density_errors),
        "maximum_rms_inverse_error": max(inverse_rms_errors),
        "maximum_delta_method_relative_rms_error": max(delta_errors),
        "transmittance_rms_minimum": min(transmittance_rms_values),
        "transmittance_rms_maximum": max(transmittance_rms_values),
        "transmittance_rms_dynamic_range_ratio": dynamic_range,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_density_to_transmittance_amplitude_compiler"
            if automatic_pass
            else "close_density_to_transmittance_amplitude_compiler_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return bundle, report


def write_json(value: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "DensityTransmittanceGranularityError",
    "compile_and_evaluate",
    "load_contract",
    "write_json",
]
