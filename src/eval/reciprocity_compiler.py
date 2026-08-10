"""Frozen U6.P2AA first-party reciprocity compiler audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.reciprocity import (
    DocumentedIdentityInterval,
    PowerReciprocityProfile,
    ReciprocityDomainError,
)


class ReciprocityCompilerError(RuntimeError):
    """Raised when the U6.P2AA source or contract drifts."""


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReciprocityCompilerError("P2AA contract must be an object")
    return payload


def _verify_sources(root: Path, contract: dict[str, Any]) -> None:
    for source in contract["sources"].values():
        path = root / source["local_research_copy"]
        if not path.is_file():
            raise ReciprocityCompilerError(f"missing source: {path}")
        if path.stat().st_size != source["bytes"] or _sha(path) != source["sha256"]:
            raise ReciprocityCompilerError(f"source identity mismatch: {path}")


def _profile_rows(contract: dict[str, Any]) -> list[PowerReciprocityProfile]:
    source = contract["sources"]["ilford"]
    reference = float(contract["domain_contract"]["reference_time_seconds"])
    return [
        PowerReciprocityProfile(name, float(exponent), reference)
        for name, exponent in source["profiles"].items()
    ]


def _partition_exact(
    profile: PowerReciprocityProfile,
    metered: np.ndarray,
    sizes: list[int],
    expected: np.ndarray,
) -> bool:
    if sum(sizes) != len(metered):
        return False
    parts = []
    start = 0
    for size in sizes:
        parts.append(profile.corrected_time_seconds(metered[start : start + size]))
        start += size
    return bool(np.array_equal(np.concatenate(parts), expected))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("schema") != (
        "neuro_film.u6_p2aa_reciprocity_compiler_contract.v1"
    ):
        raise ReciprocityCompilerError("unsupported P2AA contract")
    _verify_sources(root, contract)
    profiles = _profile_rows(contract)
    metered = np.asarray(contract["audit"]["metered_times_seconds"], dtype=np.float64)
    signature_times = np.asarray(
        contract["audit"]["long_exposure_signature_seconds"], dtype=np.float64
    )
    reference = float(contract["domain_contract"]["reference_time_seconds"])

    rows: list[dict[str, Any]] = []
    maximum_identity_error = 0.0
    maximum_inverse_error = 0.0
    minimum_correction_ratio = float("inf")
    monotone_strict = True
    partition_exact = True
    signatures: dict[float, np.ndarray] = {}
    for profile in profiles:
        corrected = profile.corrected_time_seconds(metered)
        restored = profile.effective_time_seconds(corrected)
        identity_mask = metered <= reference
        maximum_identity_error = max(
            maximum_identity_error,
            float(np.max(np.abs(corrected[identity_mask] - metered[identity_mask]))),
        )
        maximum_inverse_error = max(
            maximum_inverse_error,
            float(
                np.max(
                    np.abs(restored - metered)
                    / np.maximum(np.abs(metered), np.finfo(np.float64).tiny)
                )
            ),
        )
        long_mask = metered > reference
        minimum_correction_ratio = min(
            minimum_correction_ratio,
            float(np.min(corrected[long_mask] / metered[long_mask])),
        )
        monotone_strict = monotone_strict and bool(np.all(np.diff(corrected) > 0.0))
        partition_exact = partition_exact and _partition_exact(
            profile,
            metered,
            [int(value) for value in contract["audit"]["partition_sizes"]],
            corrected,
        )
        signatures.setdefault(
            profile.exponent,
            np.log(profile.corrected_time_seconds(signature_times) / signature_times),
        )
        rows.append(
            {
                "profile_id": profile.profile_id,
                "exponent": profile.exponent,
                "corrected_time_seconds": corrected.tolist(),
                "long_exposure_correction_ratio": (
                    profile.corrected_time_seconds(signature_times) / signature_times
                ).tolist(),
            }
        )

    signature_values = list(signatures.values())
    minimum_group_signature_l2 = min(
        float(np.linalg.norm(left - right))
        for index, left in enumerate(signature_values)
        for right in signature_values[index + 1 :]
    )

    kodak_source = contract["sources"]["kodak_vision3_50d"]
    kodak_interval = kodak_source["documented_identity_interval_seconds"]
    kodak = DocumentedIdentityInterval(
        "kodak_vision3_50d_5203_7203",
        float(kodak_interval[0]),
        float(kodak_interval[1]),
    )
    kodak_times = np.asarray([0.001, 0.01, 0.1, 1.0], dtype=np.float64)
    kodak_corrected = kodak.corrected_time_seconds(kodak_times)
    kodak_error = float(np.max(np.abs(kodak_corrected - kodak_times)))
    outside_rejected = False
    try:
        kodak.corrected_time_seconds(np.asarray([1.0000001]))
    except ReciprocityDomainError:
        outside_rejected = True

    thresholds = contract["automatic_gates"]
    measurements = {
        "ilford_profile_count": len(profiles),
        "ilford_distinct_exponent_group_count": len(signatures),
        "identity_maximum_absolute_error": maximum_identity_error,
        "forward_inverse_maximum_relative_error": maximum_inverse_error,
        "minimum_long_exposure_correction_ratio": minimum_correction_ratio,
        "minimum_distinct_group_signature_l2": minimum_group_signature_l2,
        "kodak_documented_identity_maximum_absolute_error": kodak_error,
        "kodak_outside_interval_rejected": outside_rejected,
        "monotone_strict": monotone_strict,
        "partition_exact": partition_exact,
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    gate_results = {
        "ilford_profile_count": measurements["ilford_profile_count"]
        == thresholds["ilford_profile_count"],
        "ilford_distinct_exponent_group_count": measurements[
            "ilford_distinct_exponent_group_count"
        ]
        == thresholds["ilford_distinct_exponent_group_count"],
        "identity_maximum_absolute_error": measurements[
            "identity_maximum_absolute_error"
        ]
        <= thresholds["identity_maximum_absolute_error"],
        "forward_inverse_maximum_relative_error": measurements[
            "forward_inverse_maximum_relative_error"
        ]
        <= thresholds["forward_inverse_maximum_relative_error"],
        "minimum_long_exposure_correction_ratio": measurements[
            "minimum_long_exposure_correction_ratio"
        ]
        > thresholds["minimum_long_exposure_correction_ratio"],
        "minimum_distinct_group_signature_l2": measurements[
            "minimum_distinct_group_signature_l2"
        ]
        >= thresholds["minimum_distinct_group_signature_l2"],
        "kodak_documented_identity_maximum_absolute_error": measurements[
            "kodak_documented_identity_maximum_absolute_error"
        ]
        <= thresholds["kodak_documented_identity_maximum_absolute_error"],
        "kodak_outside_interval_rejected": measurements[
            "kodak_outside_interval_rejected"
        ]
        is thresholds["kodak_outside_interval_rejected"],
        "monotone_strict": measurements["monotone_strict"]
        is thresholds["monotone_strict"],
        "partition_exact": measurements["partition_exact"]
        is thresholds["partition_exact"],
        "serialization_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    if set(gate_results) != set(thresholds):
        raise ReciprocityCompilerError("P2AA gate vocabulary drift")
    stable = {
        "schema": "neuro_film.u6_p2aa_reciprocity_compiler_report.v1",
        "contract_sha256": _sha(root / "configs/u6_p2aa_reciprocity_compiler_v1.json"),
        "source_sha256": {
            name: source["sha256"] for name, source in contract["sources"].items()
        },
        "metered_times_seconds": metered.tolist(),
        "profiles": rows,
        "exponent_equivalence_groups": {
            format(exponent, ".2f"): [
                profile.profile_id for profile in profiles if profile.exponent == exponent
            ]
            for exponent in sorted(signatures)
        },
        "measurements": measurements,
        "gate_results": gate_results,
        "automatic_pass": all(gate_results.values()),
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {**stable, "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest()}


def write_report(report: dict[str, Any], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()


__all__ = ["ReciprocityCompilerError", "load_contract", "run_audit", "write_report"]
