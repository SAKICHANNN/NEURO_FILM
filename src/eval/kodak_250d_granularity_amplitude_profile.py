"""U6.P4AX compiler for the supported P4AW amplitude-only result."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.granularity_amplitude import GranularityAmplitudeProfile
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro_film.u6_p4ax_kodak_250d_granularity_amplitude_profile_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u6_p4ax_kodak_250d_granularity_amplitude_profile_report.v1"
)


class GranularityAmplitudeCompilerError(RuntimeError):
    """Raised when frozen P4AX inputs or semantics drift."""


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
        raise GranularityAmplitudeCompilerError("P4AX paths must be relative")
    return root / relative


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    compiler = payload.get("compiler", {})
    evaluation = payload.get("evaluation", {})
    if (
        payload.get("schema") != SCHEMA
        or compiler.get("source_parameter_set")
        != "characteristic_slope_without_p5j"
        or compiler.get("profile_schema")
        != "neuro_film.granularity_amplitude_profile.v1"
        or compiler.get("channel_order") != ["red", "green", "blue"]
        or compiler.get("aperture_diameter_micrometres") != 48.0
        or compiler.get("spatial_psf_or_nps_fields_allowed")
        or compiler.get("parameter_refit_allowed")
        or not evaluation.get("require_exact_parent_parameters")
        or not evaluation.get("require_exact_prior_identity")
        or not evaluation.get("require_two_byte_identical_bundles_and_reports")
        or not evaluation.get("require_all_p4aw_confirmation_predictions_exact")
        or evaluation.get("maximum_absolute_sigma_replay_error") != 1e-15
        or evaluation.get("minimum_sigma_d") != 0.001
        or evaluation.get("maximum_sigma_d") != 0.05
        or not evaluation.get("domain_mismatch_must_reject")
        or not evaluation.get("prior_identity_mismatch_must_reject")
        or evaluation.get("serialized_forbidden_tokens")
        != ["psf", "mtf", "nps", "scanner", "spatial_kernel"]
    ):
        raise GranularityAmplitudeCompilerError("P4AX frozen contract drift")
    return payload


def _load_parent(
    parents: Mapping[str, Any], root: Path, stem: str
) -> dict[str, Any]:
    path = _relative(root, str(parents[f"{stem}_path"]))
    if not path.is_file() or _hash_file(path) != parents[f"{stem}_sha256"]:
        raise GranularityAmplitudeCompilerError(
            f"P4AX parent integrity mismatch: {stem}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def compile_and_evaluate(
    contract: Mapping[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    parents = contract["parents"]
    decision = _load_parent(parents, root, "p4aw_decision")
    report = _load_parent(parents, root, "p4aw_report")
    _load_parent(parents, root, "p4aw_contract")
    prior_payload = _load_parent(parents, root, "p2q_bundle")
    if (
        decision.get("decision")
        != "close_effective_same_sheet_joint_family_without_rescue"
        or decision.get("failed_gates") != ["improvement_vs_wrong_channel_p5j"]
        or report.get("stable_evidence_id") != parents["p4aw_stable_evidence_id"]
        or report.get("automatic_pass") is not False
        or report.get("gate_results", {}).get(
            "improvement_vs_characteristic_slope_without_p5j"
        )
        is not True
    ):
        raise GranularityAmplitudeCompilerError("P4AX parent decision mismatch")
    prior = ManufacturerCharacteristicPrior.from_dict(prior_payload["prior"])
    if prior.identity() != parents["p2q_prior_identity"]:
        raise GranularityAmplitudeCompilerError("P4AX prior identity mismatch")

    fitted = report["parameters"]["characteristic_slope_without_p5j"]
    floors = fitted["channel_floor_variance"]
    profile = GranularityAmplitudeProfile(
        characteristic_prior_identity=prior.identity(),
        source_evidence_id=str(report["stable_evidence_id"]),
        channel_floor_variance={
            "red": float(floors["red"]),
            "green": float(floors["green"]),
            "blue": float(floors["blue"]),
        },
        shared_amplitude=float(fitted["shared_amplitude"]),
    )
    bundle = profile.to_dict()
    bundle["profile_id"] = profile.identity()

    confirmation = report["confirmation_rows"]
    expected = np.asarray(
        report["scores"]["characteristic_slope_without_p5j"]["predicted_sigma_d"],
        dtype=np.float64,
    )
    replay = np.asarray(
        [
            profile.evaluate_channel(
                prior,
                str(row["channel"]),
                np.asarray([float(row["log_exposure"])], dtype=np.float64),
            )[0]
            for row in confirmation
        ],
        dtype=np.float64,
    )
    replay_error = float(np.max(np.abs(replay - expected)))
    all_exposure = np.concatenate(
        [curve.log_exposure_knots for curve in prior.curves]
    )
    sampled_sigma = np.concatenate(
        [
            profile.evaluate_channel(prior, curve.layer, curve.log_exposure_knots)
            for curve in prior.curves
        ]
    )
    domain_reject = False
    try:
        profile.evaluate_channel(
            prior,
            "red",
            np.asarray([prior.curves[0].domain[0] - 1e-6], dtype=np.float64),
        )
    except ValueError:
        domain_reject = True
    wrong_prior_payload = prior.to_dict()
    wrong_prior_payload["source_evidence_id"] = "0" * 64
    wrong_prior = ManufacturerCharacteristicPrior.from_dict(wrong_prior_payload)
    prior_reject = False
    try:
        profile.evaluate_channel(
            wrong_prior, "red", np.asarray([prior.curves[0].domain[0]])
        )
    except ValueError:
        prior_reject = True
    encoded_bundle = json.dumps(bundle, sort_keys=True).lower()
    forbidden_present = [
        token
        for token in contract["evaluation"]["serialized_forbidden_tokens"]
        if token in encoded_bundle
    ]
    gates = contract["evaluation"]
    gate_results = {
        "parent_identity": True,
        "exact_parameter_copy": bundle["channel_floor_variance"]
        == {"red": floors["red"], "green": floors["green"], "blue": floors["blue"]}
        and bundle["shared_amplitude"] == fitted["shared_amplitude"],
        "confirmation_replay": replay_error
        <= float(gates["maximum_absolute_sigma_replay_error"]),
        "physical_range": float(np.min(sampled_sigma))
        >= float(gates["minimum_sigma_d"])
        and float(np.max(sampled_sigma)) <= float(gates["maximum_sigma_d"]),
        "domain_reject": domain_reject,
        "prior_identity_reject": prior_reject,
        "amplitude_only_serialization": not forbidden_present,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "experiment_id": contract["experiment_id"],
        "profile_id": profile.identity(),
        "profile_sha256": hashlib.sha256(_canonical_json(bundle)).hexdigest(),
        "parent_report_sha256": parents["p4aw_report_sha256"],
        "characteristic_prior_identity": prior.identity(),
        "confirmation_row_count": len(confirmation),
        "maximum_absolute_sigma_replay_error": replay_error,
        "sampled_log_exposure_minimum": float(np.min(all_exposure)),
        "sampled_log_exposure_maximum": float(np.max(all_exposure)),
        "sampled_sigma_d_minimum": float(np.min(sampled_sigma)),
        "sampled_sigma_d_maximum": float(np.max(sampled_sigma)),
        "forbidden_serialized_tokens_present": forbidden_present,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
    }
    result = {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": hashlib.sha256(_canonical_json(stable)).hexdigest(),
        "decision": (
            "retain_amplitude_only_profile"
            if automatic_pass
            else "close_amplitude_profile_compiler_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return bundle, result


def write_json(value: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "GranularityAmplitudeCompilerError",
    "compile_and_evaluate",
    "load_contract",
    "write_json",
]
