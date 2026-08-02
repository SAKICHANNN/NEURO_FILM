"""U6.P4BB exact compound-Poisson measurement-cell evaluator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from src.film_physics.transmittance_granularity import (
    ApertureCellCompoundPoissonProfile,
    MarginalRobustTransmittanceProfile,
    density_compound_poisson_to_transmittance_moments,
)

SCHEMA = "neuro_film.u6_p4bb_aperture_cell_compound_poisson_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bb_aperture_cell_compound_poisson_report.v1"


class ApertureCellCompoundPoissonError(RuntimeError):
    """Raised when frozen P4BB evidence or semantics drift."""


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
        raise ApertureCellCompoundPoissonError("P4BB paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("mechanism")
        != "equal_mark_compound_poisson_density_per_measurement_cell"
        or candidate.get("measurement_cell_micrometres") != 48.0
        or candidate.get("poisson_rate")
        != "density_mean_squared / density_variance"
        or candidate.get("density_mark") != "density_variance / density_mean"
        or candidate.get("transmittance")
        != "pow(10, -compound_poisson_density)"
        or candidate.get("spatial_sampling_allowed") is not False
        or candidate.get("microscopic_particle_interpretation_allowed") is not False
        or evaluation.get("minimum_probe_count") != 15
        or evaluation.get("maximum_density_moment_replay_relative_error") != 1e-14
        or evaluation.get("maximum_high_precision_transmittance_moment_relative_error")
        != 1e-12
        or evaluation.get("maximum_parent_interval_excursion_relative") != 1e-12
        or evaluation.get("minimum_poisson_rate") != 1.0
        or evaluation.get("maximum_poisson_rate") != 1_000_000.0
        or evaluation.get("maximum_density_mark") != 0.001
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ApertureCellCompoundPoissonError("P4BB frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ApertureCellCompoundPoissonError(
            f"P4BB parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _high_precision_moments(rate: float, mark: float) -> tuple[float, float]:
    with mp.workdps(80):
        lam = mp.mpf(rate)
        q = mp.mpf(mark)
        log_ten = mp.log(10)
        jump = mp.expm1(-log_ten * q)
        log_first = lam * jump
        first = mp.exp(log_first)
        rms = first * mp.sqrt(mp.expm1(lam * jump**2))
        return float(first), float(rms)


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0)


def _interval_excursion(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return (lower - value) / lower
    if value > upper:
        return (value - upper) / upper
    return 0.0


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4ba_decision")
    parent_bundle = _load_parent(parents, root, "p4ba_bundle")
    parent_report = _load_parent(parents, root, "p4ba_report")
    if (
        decision.get("decision") != "retain_marginal_robust_transmittance_interval"
        or decision.get("automatic_pass") is not True
        or parent_report.get("stable_evidence_id") != parents["p4ba_stable_evidence_id"]
        or parent_report.get("probe_count") != 15
    ):
        raise ApertureCellCompoundPoissonError("P4BB parent decision mismatch")

    serialized_parent = dict(parent_bundle)
    parent_profile_id = serialized_parent.pop("profile_id")
    parent_profile = MarginalRobustTransmittanceProfile.from_dict(serialized_parent)
    if parent_profile.identity() != parent_profile_id:
        raise ApertureCellCompoundPoissonError("P4BB parent profile mismatch")
    profile = ApertureCellCompoundPoissonProfile(
        parent_profile, str(parents["p4ba_stable_evidence_id"])
    )
    bundle = profile.to_dict()
    bundle["profile_id"] = profile.identity()

    rows: list[dict[str, Any]] = []
    density_replay_errors: list[float] = []
    high_precision_errors: list[float] = []
    interval_excursions: list[float] = []
    rates: list[float] = []
    marks: list[float] = []
    for parent in parent_report["rows"]:
        density_mean = float(parent["density_mean"])
        density_rms = float(parent["density_rms"])
        parameters, moments = density_compound_poisson_to_transmittance_moments(
            np.asarray([density_mean]), np.asarray([density_rms])
        )
        rate = float(parameters.poisson_rate[0])
        mark = float(parameters.density_mark[0])
        trans_mean = float(moments.transmittance_mean[0])
        trans_rms = float(moments.transmittance_rms[0])
        replay_mean = rate * mark
        replay_rms = np.sqrt(rate) * mark
        hp_mean, hp_rms = _high_precision_moments(rate, mark)
        family_pairs = [
            parent["gaussian_transmittance_mean_rms"],
            parent["gamma_transmittance_mean_rms"],
            parent["uniform_transmittance_mean_rms"],
        ]
        mean_lower = min(float(pair[0]) for pair in family_pairs)
        mean_upper = max(float(pair[0]) for pair in family_pairs)
        rms_lower = min(float(pair[1]) for pair in family_pairs)
        rms_upper = max(float(pair[1]) for pair in family_pairs)
        replay_error = max(
            _relative_error(replay_mean, density_mean),
            _relative_error(float(replay_rms), density_rms),
        )
        high_precision_error = max(
            _relative_error(trans_mean, hp_mean),
            _relative_error(trans_rms, hp_rms),
        )
        interval_excursion = max(
            _interval_excursion(trans_mean, mean_lower, mean_upper),
            _interval_excursion(trans_rms, rms_lower, rms_upper),
        )
        density_replay_errors.append(replay_error)
        high_precision_errors.append(high_precision_error)
        interval_excursions.append(interval_excursion)
        rates.append(rate)
        marks.append(mark)
        rows.append(
            {
                "channel": parent["channel"],
                "domain_fraction": parent["domain_fraction"],
                "density_mean": density_mean,
                "density_rms": density_rms,
                "poisson_rate_per_48um_cell": rate,
                "density_mark": mark,
                "transmittance_mean": trans_mean,
                "transmittance_rms": trans_rms,
                "density_moment_replay_relative_error": replay_error,
                "high_precision_transmittance_moment_relative_error": high_precision_error,
                "parent_interval_excursion_relative": interval_excursion,
            }
        )

    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "probe_count": len(rows) >= int(gates["minimum_probe_count"]),
        "density_moment_replay": max(density_replay_errors)
        <= float(gates["maximum_density_moment_replay_relative_error"]),
        "high_precision_transmittance_moments": max(high_precision_errors)
        <= float(gates["maximum_high_precision_transmittance_moment_relative_error"]),
        "inside_parent_marginal_interval": max(interval_excursions)
        <= float(gates["maximum_parent_interval_excursion_relative"]),
        "poisson_rate_bounds": min(rates) >= float(gates["minimum_poisson_rate"])
        and max(rates) <= float(gates["maximum_poisson_rate"]),
        "density_mark_bound": max(marks) <= float(gates["maximum_density_mark"]),
        "microstructure_unidentified": bundle["microstructure_status"]
        == "unidentified",
        "spatial_nps_unidentified": bundle["spatial_nps_status"] == "unidentified",
        "render_forbidden": bundle["render_allowed"] is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_sha256": hashlib.sha256(_canonical_json(bundle)).hexdigest(),
        "probe_count": len(rows),
        "minimum_poisson_rate": min(rates),
        "maximum_poisson_rate": max(rates),
        "minimum_density_mark": min(marks),
        "maximum_density_mark": max(marks),
        "maximum_density_moment_replay_relative_error": max(density_replay_errors),
        "maximum_high_precision_transmittance_moment_relative_error": max(
            high_precision_errors
        ),
        "maximum_parent_interval_excursion_relative": max(interval_excursions),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_aperture_cell_compound_poisson_amplitude_compiler"
            if automatic_pass
            else "close_aperture_cell_compound_poisson_without_rescue"
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
    "ApertureCellCompoundPoissonError",
    "compile_and_evaluate",
    "load_contract",
    "write_json",
]
