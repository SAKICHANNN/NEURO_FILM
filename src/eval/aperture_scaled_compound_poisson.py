"""U6.P4BG typed multi-aperture amplitude compiler evaluator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np

from src.film_physics.aperture_scaled_granularity import (
    ApertureScaledCompoundPoissonProfile,
    scale_density_rms_between_apertures,
)
from src.film_physics.transmittance_granularity import (
    ApertureCellCompoundPoissonProfile,
)

SCHEMA = "neuro_film.u6_p4bg_aperture_scaled_compound_poisson_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bg_aperture_scaled_compound_poisson_report.v1"


class ApertureScaledCompoundPoissonError(RuntimeError):
    """Raised when frozen P4BG inputs or semantics drift."""


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
        raise ApertureScaledCompoundPoissonError("P4BG paths must be relative")
    return root / path


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidate = payload.get("candidate", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or candidate.get("schema")
        != "neuro_film.aperture_scaled_compound_poisson_profile.v1"
        or candidate.get("reference_aperture_micrometres") != 48.0
        or candidate.get("minimum_aperture_micrometres") != 7.25
        or candidate.get("maximum_aperture_micrometres") != 384.0
        or candidate.get("density_mean_scaling") != "identity"
        or candidate.get("density_rms_scaling")
        != "sigma_to = sigma_from * aperture_from / aperture_to"
        or candidate.get("poisson_rate_scaling")
        != "rate_to = rate_from * (aperture_to / aperture_from)^2"
        or candidate.get("density_mark_scaling")
        != "mark_to = mark_from * (aperture_from / aperture_to)^2"
        or candidate.get("spatial_nps_selected") is not False
        or candidate.get("render_allowed") is not False
        or evaluation.get("apertures_micrometres")
        != [7.25, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0]
        or evaluation.get("required_probe_count") != 15
        or evaluation.get("required_scaled_row_count") != 105
        or evaluation.get("maximum_density_roundtrip_relative_error") != 1e-14
        or evaluation.get("maximum_scale_composition_relative_error") != 1e-14
        or evaluation.get("maximum_parameter_scaling_relative_error") != 1e-14
        or evaluation.get("maximum_high_precision_transmittance_relative_error")
        != 1e-12
        or evaluation.get("require_reference_aperture_exact_parent") is not True
        or evaluation.get("require_profile_roundtrip_identity") is not True
        or evaluation.get("require_two_byte_identical_bundles_and_reports") is not True
    ):
        raise ApertureScaledCompoundPoissonError("P4BG frozen contract drift")
    return payload


def _load_parent(parents: Mapping[str, Any], root: Path, stem: str) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise ApertureScaledCompoundPoissonError(
            f"P4BG parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual / expected - 1.0)


def _high_precision_moments(rate: float, mark: float) -> tuple[float, float]:
    with mp.workdps(80):
        lam = mp.mpf(rate)
        q = mp.mpf(mark)
        jump = mp.expm1(-mp.log(10) * q)
        mean = mp.exp(lam * jump)
        rms = mean * mp.sqrt(mp.expm1(lam * jump**2))
        return float(mean), float(rms)


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    p4bb_decision = _load_parent(parents, root, "p4bb_decision")
    p4bb_bundle = _load_parent(parents, root, "p4bb_bundle")
    p4bb_report = _load_parent(parents, root, "p4bb_report")
    p4bf_decision = _load_parent(parents, root, "p4bf_decision")
    p4bf_report = _load_parent(parents, root, "p4bf_report")
    if (
        p4bb_decision.get("automatic_pass") is not True
        or p4bb_report.get("stable_evidence_id")
        != parents["p4bb_stable_evidence_id"]
        or p4bf_decision.get("automatic_pass") is not True
        or p4bf_report.get("stable_evidence_id")
        != parents["p4bf_stable_evidence_id"]
        or p4bf_report.get("automatic_pass") is not True
    ):
        raise ApertureScaledCompoundPoissonError("P4BG parent decision mismatch")

    parent_payload = dict(p4bb_bundle)
    parent_id = parent_payload.pop("profile_id")
    parent_profile = ApertureCellCompoundPoissonProfile.from_dict(parent_payload)
    if parent_profile.identity() != parent_id:
        raise ApertureScaledCompoundPoissonError("P4BG parent profile mismatch")
    profile = ApertureScaledCompoundPoissonProfile(
        parent_profile, str(parents["p4bf_stable_evidence_id"])
    )
    bundle = profile.to_dict()
    bundle["profile_id"] = profile.identity()
    roundtripped = dict(bundle)
    roundtrip_id = roundtripped.pop("profile_id")
    restored = ApertureScaledCompoundPoissonProfile.from_dict(roundtripped)
    profile_roundtrip = restored.identity() == roundtrip_id

    apertures = [float(value) for value in contract["evaluation"]["apertures_micrometres"]]
    rows: list[dict[str, Any]] = []
    density_roundtrip_errors: list[float] = []
    composition_errors: list[float] = []
    parameter_errors: list[float] = []
    high_precision_errors: list[float] = []
    reference_exact: list[bool] = []
    for parent in p4bb_report["rows"]:
        density_mean = float(parent["density_mean"])
        base_rms = float(parent["density_rms"])
        base_rate = float(parent["poisson_rate_per_48um_cell"])
        base_mark = float(parent["density_mark"])
        for aperture in apertures:
            parameters, moments = profile.scale_density_moments(
                np.asarray([density_mean]), np.asarray([base_rms]), aperture
            )
            scaled_rms = float(moments.density_rms[0])
            rate = float(parameters.poisson_rate[0])
            mark = float(parameters.density_mark[0])
            recovered = float(
                scale_density_rms_between_apertures(
                    np.asarray([scaled_rms]), aperture, 48.0
                )[0]
            )
            via_24 = float(
                scale_density_rms_between_apertures(
                    scale_density_rms_between_apertures(
                        np.asarray([base_rms]), 48.0, 24.0
                    ),
                    24.0,
                    aperture,
                )[0]
            )
            expected_rate = base_rate * (aperture / 48.0) ** 2
            expected_mark = base_mark * (48.0 / aperture) ** 2
            hp_mean, hp_rms = _high_precision_moments(rate, mark)
            density_roundtrip_error = _relative_error(recovered, base_rms)
            composition_error = _relative_error(via_24, scaled_rms)
            parameter_error = max(
                _relative_error(rate, expected_rate),
                _relative_error(mark, expected_mark),
            )
            high_precision_error = max(
                _relative_error(float(moments.transmittance_mean[0]), hp_mean),
                _relative_error(float(moments.transmittance_rms[0]), hp_rms),
            )
            is_reference_exact = aperture != 48.0 or (
                density_mean == float(moments.density_mean[0])
                and base_rms == scaled_rms
                and base_rate == rate
                and base_mark == mark
                and float(parent["transmittance_mean"])
                == float(moments.transmittance_mean[0])
                and float(parent["transmittance_rms"])
                == float(moments.transmittance_rms[0])
            )
            density_roundtrip_errors.append(density_roundtrip_error)
            composition_errors.append(composition_error)
            parameter_errors.append(parameter_error)
            high_precision_errors.append(high_precision_error)
            reference_exact.append(is_reference_exact)
            rows.append(
                {
                    "channel": parent["channel"],
                    "domain_fraction": parent["domain_fraction"],
                    "aperture_micrometres": aperture,
                    "density_mean": density_mean,
                    "density_rms": scaled_rms,
                    "poisson_rate": rate,
                    "density_mark": mark,
                    "transmittance_mean": float(moments.transmittance_mean[0]),
                    "transmittance_rms": float(moments.transmittance_rms[0]),
                    "density_roundtrip_relative_error": density_roundtrip_error,
                    "scale_composition_relative_error": composition_error,
                    "parameter_scaling_relative_error": parameter_error,
                    "high_precision_transmittance_relative_error": high_precision_error,
                    "reference_aperture_exact_parent": is_reference_exact,
                }
            )

    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "probe_count": len(p4bb_report["rows"]) == int(gates["required_probe_count"]),
        "scaled_row_count": len(rows) == int(gates["required_scaled_row_count"]),
        "density_roundtrip": max(density_roundtrip_errors)
        <= float(gates["maximum_density_roundtrip_relative_error"]),
        "scale_composition": max(composition_errors)
        <= float(gates["maximum_scale_composition_relative_error"]),
        "parameter_scaling": max(parameter_errors)
        <= float(gates["maximum_parameter_scaling_relative_error"]),
        "high_precision_transmittance": max(high_precision_errors)
        <= float(gates["maximum_high_precision_transmittance_relative_error"]),
        "reference_aperture_exact_parent": all(reference_exact),
        "profile_roundtrip_identity": profile_roundtrip,
        "spatial_nps_unidentified": bundle["spatial_nps_status"] == "unidentified",
        "render_forbidden": bundle["render_allowed"] is False,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_sha256": hashlib.sha256(_canonical_json(bundle)).hexdigest(),
        "probe_count": len(p4bb_report["rows"]),
        "scaled_row_count": len(rows),
        "maximum_density_roundtrip_relative_error": max(density_roundtrip_errors),
        "maximum_scale_composition_relative_error": max(composition_errors),
        "maximum_parameter_scaling_relative_error": max(parameter_errors),
        "maximum_high_precision_transmittance_relative_error": max(
            high_precision_errors
        ),
        "minimum_poisson_rate": min(float(row["poisson_rate"]) for row in rows),
        "maximum_poisson_rate": max(float(row["poisson_rate"]) for row in rows),
        "minimum_density_mark": min(float(row["density_mark"]) for row in rows),
        "maximum_density_mark": max(float(row["density_mark"]) for row in rows),
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    report = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "rows": rows,
        "decision": (
            "retain_bounded_aperture_amplitude_compiler"
            if automatic_pass
            else "retain_48um_profile_only_without_rescue"
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
    "ApertureScaledCompoundPoissonError",
    "compile_and_evaluate",
    "load_contract",
    "write_json",
]
