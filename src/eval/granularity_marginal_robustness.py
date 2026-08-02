"""U6.P4BA density-marginal sensitivity evaluator for grain amplitude."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from src.film_physics.transmittance_granularity import (
    MarginalRobustTransmittanceProfile,
    TransmittanceGranularityProfile,
    density_gamma_to_transmittance_moments,
    density_gaussian_to_transmittance_moments,
    density_uniform_to_transmittance_moments,
)

SCHEMA = "neuro_film.u6_p4ba_granularity_marginal_robustness_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ba_granularity_marginal_robustness_report.v1"


class GranularityMarginalRobustnessError(RuntimeError):
    """Raised when frozen P4BA evidence or semantics drift."""


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
        raise GranularityMarginalRobustnessError("P4BA paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    shared = payload.get("shared_constraints", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or payload.get("candidate_marginals")
        != ["gaussian", "positive_gamma", "bounded_uniform"]
        or not shared.get("same_density_mean_and_variance")
        or shared.get("same_physical_transform") != "transmittance = pow(10, -density)"
        or not shared.get("same_15_frozen_p4az_probes")
        or shared.get("spatial_structure_selection_allowed")
        or shared.get("parameter_fit_allowed")
        or evaluation.get("gauss_hermite_order") != 96
        or evaluation.get("gauss_legendre_order") != 96
        or evaluation.get("maximum_gamma_mean_quadrature_relative_error") != 1e-11
        or evaluation.get("maximum_gamma_rms_quadrature_relative_error") != 1e-9
        or evaluation.get("maximum_uniform_mean_quadrature_relative_error") != 1e-11
        or evaluation.get("maximum_uniform_rms_quadrature_relative_error") != 1e-9
        or evaluation.get("maximum_cross_marginal_mean_relative_span") != 1e-6
        or evaluation.get("maximum_cross_marginal_rms_relative_span") != 0.0005
        or evaluation.get("minimum_probe_count") != 15
        or not evaluation.get("require_two_byte_identical_reports")
    ):
        raise GranularityMarginalRobustnessError("P4BA frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise GranularityMarginalRobustnessError(
            f"P4BA parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _gamma_quadrature(density_mean: float, density_rms: float) -> tuple[float, float]:
    """High-precision independent integral in standardized-density coordinates."""

    with mp.workdps(60):
        mean = mp.mpf(density_mean)
        sigma = mp.mpf(density_rms)
        shape = (mean / sigma) ** 2
        scale = sigma**2 / mean
        log_ten = mp.log(10)
        lower = -mean / sigma

        def integrand(value: mp.mpf, power: int) -> mp.mpf:
            density = mean + sigma * value
            log_pdf = (
                (shape - 1) * mp.log(density)
                - density / scale
                - mp.loggamma(shape)
                - shape * mp.log(scale)
                + mp.log(sigma)
            )
            return mp.exp(log_pdf - power * log_ten * density)

        intervals = [
            lower + mp.mpf("1e-40"),
            mp.mpf(-8),
            mp.mpf(-4),
            mp.mpf(0),
            mp.mpf(4),
            mp.mpf(8),
            mp.mpf(12),
        ]
        first = mp.quad(lambda value: integrand(value, 1), intervals)
        second = mp.quad(lambda value: integrand(value, 2), intervals)
        variance = second - first**2
        return float(first), float(mp.sqrt(variance))


def _uniform_quadrature(
    density_mean: float, density_rms: float, order: int
) -> tuple[float, float]:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half_width = np.sqrt(3.0) * density_rms
    density = density_mean + half_width * nodes
    transmittance = np.power(10.0, -density)
    normalized = weights / 2.0
    mean = float(np.sum(normalized * transmittance, dtype=np.float64))
    variance = float(
        np.sum(normalized * np.square(transmittance - mean), dtype=np.float64)
    )
    return mean, float(np.sqrt(variance))


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4az_decision")
    base_bundle = _load_parent(parents, root, "p4az_bundle")
    parent_report = _load_parent(parents, root, "p4az_report")
    if (
        decision.get("decision") != "retain_density_to_transmittance_amplitude_compiler"
        or decision.get("automatic_pass") is not True
        or parent_report.get("stable_evidence_id") != parents["p4az_stable_evidence_id"]
        or parent_report.get("probe_count") != 15
    ):
        raise GranularityMarginalRobustnessError("P4BA parent decision mismatch")
    base_bundle = dict(base_bundle)
    base_profile_id = base_bundle.pop("profile_id")
    base_profile = TransmittanceGranularityProfile.from_dict(base_bundle)
    if base_profile.identity() != base_profile_id:
        raise GranularityMarginalRobustnessError("P4BA base profile mismatch")
    profile = MarginalRobustTransmittanceProfile(base_profile)
    bundle = profile.to_dict()
    bundle["profile_id"] = profile.identity()

    rows: list[dict[str, Any]] = []
    gamma_mean_errors: list[float] = []
    gamma_rms_errors: list[float] = []
    uniform_mean_errors: list[float] = []
    uniform_rms_errors: list[float] = []
    mean_spans: list[float] = []
    rms_spans: list[float] = []
    for parent in parent_report["rows"]:
        density_mean = float(parent["density_mean"])
        density_rms = float(parent["density_rms"])
        gaussian = density_gaussian_to_transmittance_moments(
            np.asarray([density_mean]), np.asarray([density_rms])
        )
        gamma = density_gamma_to_transmittance_moments(
            np.asarray([density_mean]), np.asarray([density_rms])
        )
        uniform = density_uniform_to_transmittance_moments(
            np.asarray([density_mean]), np.asarray([density_rms])
        )
        gamma_q_mean, gamma_q_rms = _gamma_quadrature(density_mean, density_rms)
        uniform_q_mean, uniform_q_rms = _uniform_quadrature(
            density_mean,
            density_rms,
            int(contract["evaluation"]["gauss_legendre_order"]),
        )
        means = [
            float(gaussian.transmittance_mean[0]),
            float(gamma.transmittance_mean[0]),
            float(uniform.transmittance_mean[0]),
        ]
        rms_values = [
            float(gaussian.transmittance_rms[0]),
            float(gamma.transmittance_rms[0]),
            float(uniform.transmittance_rms[0]),
        ]
        gamma_mean_errors.append(abs(gamma_q_mean / means[1] - 1.0))
        gamma_rms_errors.append(abs(gamma_q_rms / rms_values[1] - 1.0))
        uniform_mean_errors.append(abs(uniform_q_mean / means[2] - 1.0))
        uniform_rms_errors.append(abs(uniform_q_rms / rms_values[2] - 1.0))
        mean_span = (max(means) - min(means)) / means[0]
        rms_span = (max(rms_values) - min(rms_values)) / rms_values[0]
        mean_spans.append(mean_span)
        rms_spans.append(rms_span)
        rows.append(
            {
                "channel": parent["channel"],
                "domain_fraction": parent["domain_fraction"],
                "density_mean": density_mean,
                "density_rms": density_rms,
                "gaussian_transmittance_mean_rms": [means[0], rms_values[0]],
                "gamma_transmittance_mean_rms": [means[1], rms_values[1]],
                "uniform_transmittance_mean_rms": [means[2], rms_values[2]],
                "cross_marginal_mean_relative_span": mean_span,
                "cross_marginal_rms_relative_span": rms_span,
            }
        )
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "probe_count": len(rows) >= int(gates["minimum_probe_count"]),
        "gamma_mean_quadrature": max(gamma_mean_errors)
        <= float(gates["maximum_gamma_mean_quadrature_relative_error"]),
        "gamma_rms_quadrature": max(gamma_rms_errors)
        <= float(gates["maximum_gamma_rms_quadrature_relative_error"]),
        "uniform_mean_quadrature": max(uniform_mean_errors)
        <= float(gates["maximum_uniform_mean_quadrature_relative_error"]),
        "uniform_rms_quadrature": max(uniform_rms_errors)
        <= float(gates["maximum_uniform_rms_quadrature_relative_error"]),
        "cross_marginal_mean_span": max(mean_spans)
        <= float(gates["maximum_cross_marginal_mean_relative_span"]),
        "cross_marginal_rms_span": max(rms_spans)
        <= float(gates["maximum_cross_marginal_rms_relative_span"]),
        "spatial_structure_unidentified": bundle["spatial_structure_status"]
        == "unidentified",
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_sha256": hashlib.sha256(_canonical_json(bundle)).hexdigest(),
        "probe_count": len(rows),
        "maximum_gamma_mean_quadrature_relative_error": max(gamma_mean_errors),
        "maximum_gamma_rms_quadrature_relative_error": max(gamma_rms_errors),
        "maximum_uniform_mean_quadrature_relative_error": max(uniform_mean_errors),
        "maximum_uniform_rms_quadrature_relative_error": max(uniform_rms_errors),
        "maximum_cross_marginal_mean_relative_span": max(mean_spans),
        "maximum_cross_marginal_rms_relative_span": max(rms_spans),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_marginal_robust_transmittance_interval"
            if automatic_pass
            else "retain_explicit_gaussian_assumption_and_close_interval"
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
    "GranularityMarginalRobustnessError",
    "compile_and_evaluate",
    "load_contract",
    "write_json",
]
