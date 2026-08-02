"""U6.P4BT canonical periodic field evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.clustered_nps_field import (
    discrete_periodogram,
    synthesize_clustered_nps_field,
)

SCHEMA = "neuro_film.u6_p4bt_clustered_nps_canonical_field_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bt_clustered_nps_canonical_field_report.v1"


class ClusteredNPSCanonicalFieldError(RuntimeError):
    """Raised when a P4BT parent, profile or exact invariant drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise ClusteredNPSCanonicalFieldError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ClusteredNPSCanonicalFieldError("parent must be a JSON object")
    return payload


def _validate_contract(contract: Mapping[str, Any]) -> None:
    synthesis = contract.get("synthesis", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or synthesis.get("field_shape") != [512, 512]
        or synthesis.get("seeds") != [2608022901, 2608022902, 2608022903]
        or synthesis.get("realized_amplitude_renormalization_allowed") is not False
        or synthesis.get("tile_or_crop_parity_claimed") is not False
        or synthesis.get("photographic_render_allowed") is not False
        or evaluation.get("required_seed_count") != 3
        or evaluation.get("required_row_count") != 3
        or evaluation.get("maximum_periodogram_relative_error") != 1e-10
        or evaluation.get("maximum_covariance_absolute_error") != 1e-12
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ClusteredNPSCanonicalFieldError("P4BT frozen contract drift")


def evaluate_clustered_nps_canonical_field(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    parents = contract["parents"]
    p4bs_contract = _load_bound(root, parents["p4bs_contract"])
    decision = _load_bound(root, parents["p4bs_decision"])
    report = _load_bound(root, parents["p4bs_report"])
    profile = contract["profile"]
    fits = report["fits"]["thomas"]
    if (
        decision.get("decision") != parents["p4bs_decision"]["required_decision"]
        or report.get("stable_evidence_id")
        != parents["p4bs_report"]["required_stable_evidence_id"]
        or p4bs_contract["models"]["thomas_candidate"] is None
        or profile["particle_sigma_pixels"] != fits["particle_sigma_pixels"]
        or profile["cluster_sigma_pixels"] != fits["cluster_sigma_pixels"]
        or profile["mean_offspring"] != fits["mean_offspring"]
        or profile.get("density_conditioning") is not None
        or profile.get("stock_id") is not None
    ):
        raise ClusteredNPSCanonicalFieldError("P4BT parent/profile drift")

    shape = tuple(int(value) for value in contract["synthesis"]["field_shape"])
    evaluation = contract["evaluation"]
    lags = evaluation["covariance_lags_yx"]
    rows: list[dict[str, Any]] = []
    periodogram_errors: list[float] = []
    covariance_errors: list[float] = []
    variance_errors: list[float] = []
    means: list[float] = []
    imaginary: list[float] = []
    hashes: list[str] = []
    repeats: list[bool] = []
    for seed in contract["synthesis"]["seeds"]:
        field = synthesize_clustered_nps_field(
            shape=shape,
            particle_sigma_pixels=float(profile["particle_sigma_pixels"]),
            cluster_sigma_pixels=float(profile["cluster_sigma_pixels"]),
            mean_offspring=float(profile["mean_offspring"]),
            seed=int(seed),
        )
        repeated = synthesize_clustered_nps_field(
            shape=shape,
            particle_sigma_pixels=float(profile["particle_sigma_pixels"]),
            cluster_sigma_pixels=float(profile["cluster_sigma_pixels"]),
            mean_offspring=float(profile["mean_offspring"]),
            seed=int(seed),
        )
        actual_periodogram = discrete_periodogram(field.values)
        selected = field.target_spectrum > 0.0
        periodogram_error = float(
            np.max(
                np.abs(
                    actual_periodogram[selected] / field.target_spectrum[selected] - 1.0
                )
            )
        )
        expected_covariance = np.fft.ifft2(field.target_spectrum).real
        actual_covariance = np.fft.ifft2(actual_periodogram).real
        covariance_error = max(
            abs(float(actual_covariance[dy, dx] - expected_covariance[dy, dx]))
            for dy, dx in lags
        )
        variance = float(np.mean(np.square(field.values), dtype=np.float64))
        mean = abs(float(np.mean(field.values, dtype=np.float64)))
        field_hash = field.field_sha256()
        repeat = field_hash == repeated.field_sha256() and np.array_equal(
            field.values, repeated.values
        )
        periodogram_errors.append(periodogram_error)
        covariance_errors.append(covariance_error)
        variance_errors.append(abs(variance - 1.0))
        means.append(mean)
        imaginary.append(field.ifft_imaginary_residual)
        hashes.append(field_hash)
        repeats.append(repeat)
        rows.append(
            {
                "seed": int(seed),
                "field_sha256": field_hash,
                "periodogram_relative_error": periodogram_error,
                "covariance_absolute_error": covariance_error,
                "variance": variance,
                "variance_absolute_error": abs(variance - 1.0),
                "absolute_sample_mean": mean,
                "ifft_imaginary_residual": field.ifft_imaginary_residual,
                "repeat_exact": repeat,
            }
        )
    checks = {
        "seed_count": len(rows) == evaluation["required_seed_count"],
        "row_count": len(rows) == evaluation["required_row_count"],
        "periodogram": max(periodogram_errors)
        <= evaluation["maximum_periodogram_relative_error"],
        "covariance": max(covariance_errors)
        <= evaluation["maximum_covariance_absolute_error"],
        "variance": max(variance_errors)
        <= evaluation["maximum_variance_absolute_error"],
        "sample_mean": max(means) <= evaluation["maximum_absolute_sample_mean"],
        "ifft_imaginary": max(imaginary)
        <= evaluation["maximum_ifft_imaginary_residual"],
        "repeat_exact": all(repeats),
        "seed_distinct_fields": len(set(hashes)) == len(hashes),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bt_clustered_nps_canonical_field_v1.json",
            "sha256",
        ),
        "profile": profile,
        "field_shape": list(shape),
        "row_count": len(rows),
        "maximum_periodogram_relative_error": max(periodogram_errors),
        "maximum_covariance_absolute_error": max(covariance_errors),
        "maximum_variance_absolute_error": max(variance_errors),
        "maximum_absolute_sample_mean": max(means),
        "maximum_ifft_imaginary_residual": max(imaginary),
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_clustered_nps_canonical_periodic_field"
            if automatic_pass
            else "close_clustered_nps_canonical_field"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ClusteredNPSCanonicalFieldError",
    "evaluate_clustered_nps_canonical_field",
    "write_report",
]
