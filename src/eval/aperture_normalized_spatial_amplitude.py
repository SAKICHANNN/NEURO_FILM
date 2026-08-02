"""U6.P4AY aperture-normalized generic spatial amplitude experiment."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from src.film_physics.aperture_normalized_structure import (
    ApertureNormalizedSpatialHypothesis,
)
from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)
from src.film_physics.structure_compiler import counter_normal_region

SCHEMA = "neuro_film.u6_p4ay_aperture_normalized_spatial_amplitude_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u6_p4ay_aperture_normalized_spatial_amplitude_report.v1"
)


class ApertureNormalizationError(RuntimeError):
    """Raised when frozen P4AY inputs or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ApertureNormalizationError("P4AY paths must be relative")
    return root / relative


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compiler = payload.get("compiler", {})
    simulation = payload.get("simulation", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or compiler.get("sample_pitch_micrometres") != 1.0
        or compiler.get("aperture_diameter_micrometres") != 48.0
        or compiler.get("gaussian_truncate_sigma") != 4.0
        or compiler.get("profile_probe_domain_fractions") != [0.2, 0.5, 0.8]
        or compiler.get("parameter_fit_allowed")
        or compiler.get("hypothesis_selection_allowed")
        or compiler.get("spatial_hypotheses")
        != [
            {"id": "white", "family": "delta", "gaussian_sigma_micrometres": 0.0},
            {"id": "fine", "family": "gaussian", "gaussian_sigma_micrometres": 2.0},
            {"id": "coarse", "family": "gaussian", "gaussian_sigma_micrometres": 6.0},
        ]
        or simulation.get("field_shape") != [1024, 1024]
        or simulation.get("reference_sigma_d") != 0.01
        or simulation.get("seeds")
        != [
            0,
            1099511627776,
            2199023255552,
            3298534883328,
            5497558138880,
            7696581394432,
            12094627905536,
            14293651161088,
        ]
        or simulation.get("report_runs") != 2
        or evaluation.get("spectrum_probe_cycles_per_micrometre") != 0.1
        or evaluation.get("maximum_analytic_sigma_replay_error") != 1e-15
        or evaluation.get("maximum_monte_carlo_median_absolute_relative_error")
        != 0.015
        or evaluation.get("maximum_monte_carlo_p95_absolute_relative_error") != 0.05
        or evaluation.get("minimum_pairwise_power_transfer_separation") != 0.15
        or evaluation.get("minimum_profile_probe_sigma_d") != 0.001
        or evaluation.get("maximum_profile_probe_sigma_d") != 0.05
        or not evaluation.get("require_exact_repeat")
        or not evaluation.get("require_all_hypotheses")
    ):
        raise ApertureNormalizationError("P4AY frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ApertureNormalizationError(f"P4AY parent integrity mismatch: {stem}")
    return json.loads(path.read_text(encoding="utf-8"))


def _hypotheses(contract: Mapping[str, Any]) -> list[ApertureNormalizedSpatialHypothesis]:
    compiler = contract["compiler"]
    return [
        ApertureNormalizedSpatialHypothesis(
            hypothesis_id=str(row["id"]),
            family=str(row["family"]),
            gaussian_sigma_micrometres=float(row["gaussian_sigma_micrometres"]),
            sample_pitch_micrometres=float(compiler["sample_pitch_micrometres"]),
            aperture_diameter_micrometres=float(
                compiler["aperture_diameter_micrometres"]
            ),
            gaussian_truncate_sigma=float(compiler["gaussian_truncate_sigma"]),
        )
        for row in compiler["spatial_hypotheses"]
    ]


def _simulate_ratio(
    hypothesis: ApertureNormalizedSpatialHypothesis,
    *,
    full_shape: tuple[int, int],
    seed: int,
    target_sigma: float,
) -> float:
    white = counter_normal_region(
        full_shape, origin_yx=(0, 0), shape=full_shape, seed=seed
    )
    field = fftconvolve(white, hypothesis.spatial_kernel(), mode="same")
    field *= hypothesis.innovation_sigma(target_sigma)
    measured = fftconvolve(field, hypothesis.aperture_kernel(), mode="same")
    margin = (
        hypothesis.spatial_radius_samples
        + hypothesis.aperture_radius_samples
        + 2
    )
    interior = measured[margin:-margin, margin:-margin]
    return float(np.std(interior, dtype=np.float64) / target_sigma)


def evaluate_normalizer(contract: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4ax_decision")
    bundle = _load_parent(parents, root, "p4ax_bundle")
    report = _load_parent(parents, root, "p4ax_report")
    prior_payload = _load_parent(parents, root, "p2q_bundle")
    if (
        decision.get("decision") != "retain_amplitude_only_profile"
        or decision.get("profile_id") != parents["p4ax_profile_id"]
        or report.get("stable_evidence_id") != parents["p4ax_stable_evidence_id"]
        or report.get("automatic_pass") is not True
    ):
        raise ApertureNormalizationError("P4AY parent decision mismatch")
    profile = GranularityAmplitudeProfile.from_dict(bundle)
    if profile.identity() != parents["p4ax_profile_id"]:
        raise ApertureNormalizationError("P4AY profile identity mismatch")
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_prior_identity"]:
        raise ApertureNormalizationError("P4AY prior identity mismatch")

    fractions = contract["compiler"]["profile_probe_domain_fractions"]
    profile_probes: list[dict[str, Any]] = []
    for curve in prior.curves:
        lower, upper = curve.domain
        exposures = np.asarray(
            [lower + float(fraction) * (upper - lower) for fraction in fractions],
            dtype=np.float64,
        )
        sigma = profile.evaluate_channel(prior, curve.layer, exposures)
        profile_probes.extend(
            {
                "channel": curve.layer,
                "domain_fraction": float(fraction),
                "log_exposure": float(exposure),
                "sigma_d": float(value),
            }
            for fraction, exposure, value in zip(
                fractions, exposures, sigma, strict=True
            )
        )

    target = float(contract["simulation"]["reference_sigma_d"])
    full_shape = tuple(int(value) for value in contract["simulation"]["field_shape"])
    frequency = float(contract["evaluation"]["spectrum_probe_cycles_per_micrometre"])
    hypothesis_metrics: dict[str, Any] = {}
    analytic_errors: list[float] = []
    monte_carlo_errors: list[float] = []
    power: dict[str, float] = {}
    for hypothesis in _hypotheses(contract):
        innovation = hypothesis.innovation_sigma(target)
        analytic = hypothesis.analytic_aperture_sigma(innovation)
        analytic_error = abs(analytic - target)
        ratios = [
            _simulate_ratio(
                hypothesis,
                full_shape=full_shape,
                seed=int(seed),
                target_sigma=target,
            )
            for seed in contract["simulation"]["seeds"]
        ]
        absolute_error = np.abs(np.asarray(ratios, dtype=np.float64) - 1.0)
        transfer = hypothesis.power_transfer(frequency)
        analytic_errors.append(analytic_error)
        monte_carlo_errors.extend(absolute_error.tolist())
        power[hypothesis.hypothesis_id] = transfer
        hypothesis_metrics[hypothesis.hypothesis_id] = {
            "family": hypothesis.family,
            "gaussian_sigma_micrometres": hypothesis.gaussian_sigma_micrometres,
            "measurement_energy": hypothesis.measurement_energy(),
            "innovation_sigma_for_reference": innovation,
            "analytic_aperture_sigma": analytic,
            "analytic_sigma_error": analytic_error,
            "monte_carlo_aperture_sigma_ratios": ratios,
            "monte_carlo_median_absolute_relative_error": float(
                np.median(absolute_error)
            ),
            "monte_carlo_p95_absolute_relative_error": float(
                np.percentile(absolute_error, 95.0)
            ),
            "power_transfer_at_probe": transfer,
        }
    pairwise_power_separation = {
        f"{left}__{right}": abs(power[left] - power[right])
        for index, left in enumerate(power)
        for right in list(power)[index + 1 :]
    }
    all_probe_sigma = [float(row["sigma_d"]) for row in profile_probes]
    gates = contract["evaluation"]
    aggregate_median = float(np.median(monte_carlo_errors))
    aggregate_p95 = float(np.percentile(monte_carlo_errors, 95.0))
    gate_results = {
        "parent_identity": True,
        "all_hypotheses": set(hypothesis_metrics) == {"white", "fine", "coarse"},
        "analytic_sigma_replay": max(analytic_errors)
        <= float(gates["maximum_analytic_sigma_replay_error"]),
        "monte_carlo_median": aggregate_median
        <= float(gates["maximum_monte_carlo_median_absolute_relative_error"]),
        "monte_carlo_p95": aggregate_p95
        <= float(gates["maximum_monte_carlo_p95_absolute_relative_error"]),
        "spatial_hypotheses_remain_distinct": min(pairwise_power_separation.values())
        >= float(gates["minimum_pairwise_power_transfer_separation"]),
        "profile_probe_range": min(all_probe_sigma)
        >= float(gates["minimum_profile_probe_sigma_d"])
        and max(all_probe_sigma) <= float(gates["maximum_profile_probe_sigma_d"]),
        "no_fit_or_selection": True,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_probes": profile_probes,
        "hypotheses": hypothesis_metrics,
        "pairwise_power_transfer_separation": pairwise_power_separation,
        "aggregate_monte_carlo_median_absolute_relative_error": aggregate_median,
        "aggregate_monte_carlo_p95_absolute_relative_error": aggregate_p95,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_generic_aperture_normalization_primitive"
            if automatic_pass
            else "close_generic_aperture_normalization_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "ApertureNormalizationError",
    "evaluate_normalizer",
    "load_contract",
    "write_report",
]
