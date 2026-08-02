"""U6.P4BV deterministic DC-receipt and nonzero-spectrum evaluation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.finite_support_thomas import (
    render_finite_support_thomas_region,
)
from src.film_physics.thomas_dc_projection import (
    ThomasDcProjectionError,
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)

SCHEMA = "neuro_film.u6_p4bv_thomas_dc_projection_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bv_thomas_dc_projection_report.v1"


class ThomasDcProjectionEvaluationError(RuntimeError):
    """Raised when the frozen P4BV evidence or semantics drift."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _validate_contract(contract: Mapping[str, Any]) -> None:
    profile = contract.get("profile", {})
    receipt = contract.get("receipt", {})
    evaluation = contract.get("evaluation", {})
    if (
        contract.get("schema") != SCHEMA
        or profile.get("profile_id")
        != "generic-scanner-convolved-thomas-p4bs-v1"
        or profile.get("component_seeds")
        != [2611923443488327891, 11400714819323198485]
        or profile.get("truncate_sigma") != 4.0
        or profile.get("maximum_halo_pixels") != 7
        or receipt.get("canonical_row_block_height") != 31
        or receipt.get("reducer") != "float64-neumaier-v1"
        or receipt.get("allow_variance_rescaling") is not False
        or evaluation.get("field_shape") != [257, 263]
        or evaluation.get("row_partitions") != [1, 7, 31, 64, 127]
        or evaluation.get("maximum_projected_absolute_mean") != 1e-15
        or evaluation.get("maximum_nonzero_spectrum_absolute_drift") != 1e-10
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise ThomasDcProjectionEvaluationError("P4BV frozen contract drift")


def _receipt_kwargs(contract: Mapping[str, Any], seed: int) -> dict[str, Any]:
    profile = contract["profile"]
    return {
        "full_shape": tuple(int(value) for value in contract["evaluation"]["field_shape"]),
        "profile_id": str(profile["profile_id"]),
        "particle_sigma_pixels": float(profile["particle_sigma_pixels"]),
        "cluster_sigma_pixels": float(profile["cluster_sigma_pixels"]),
        "mean_offspring": float(profile["mean_offspring"]),
        "component_seeds": tuple(int(value) for value in profile["component_seeds"]),
        "realization_seed": int(seed),
        "truncate": float(profile["truncate_sigma"]),
        "canonical_row_block_height": int(
            contract["receipt"]["canonical_row_block_height"]
        ),
    }


def evaluate_thomas_dc_projection(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    parent_binding = contract["parents"]["p4bu_decision"]
    parent_path = root / str(parent_binding["path"])
    if (
        not parent_path.is_file()
        or hash_file(parent_path, "sha256") != parent_binding["sha256"]
    ):
        raise ThomasDcProjectionEvaluationError("P4BU parent hash mismatch")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != parent_binding["required_decision"]:
        raise ThomasDcProjectionEvaluationError("P4BU parent decision mismatch")

    evaluation = contract["evaluation"]
    full_shape = tuple(int(value) for value in evaluation["field_shape"])
    border = int(contract["profile"]["maximum_halo_pixels"]) * 2
    rows: list[dict[str, Any]] = []
    projected_hashes: list[str] = []
    for seed in evaluation["seeds"]:
        kwargs = _receipt_kwargs(contract, int(seed))
        receipt = build_thomas_dc_receipt(**kwargs)
        repeated_receipt = build_thomas_dc_receipt(**kwargs)
        raw = render_finite_support_thomas_region(
            full_shape,
            origin_yx=(0, 0),
            shape=full_shape,
            particle_sigma_pixels=kwargs["particle_sigma_pixels"],
            cluster_sigma_pixels=kwargs["cluster_sigma_pixels"],
            mean_offspring=kwargs["mean_offspring"],
            component_seeds=kwargs["component_seeds"],
            realization_seed=kwargs["realization_seed"],
            truncate=kwargs["truncate"],
        )
        projected = render_dc_projected_thomas_region(
            receipt, origin_yx=(0, 0), shape=full_shape
        )
        assembled = np.empty_like(projected)
        row_exact = True
        for row_height in evaluation["row_partitions"]:
            for y0 in range(0, full_shape[0], int(row_height)):
                height = min(int(row_height), full_shape[0] - y0)
                assembled[y0 : y0 + height] = render_dc_projected_thomas_region(
                    receipt, origin_yx=(y0, 0), shape=(height, full_shape[1])
                )
            row_exact &= bool(np.array_equal(projected, assembled))
        raw_fft = np.fft.fft2(raw)
        projected_fft = np.fft.fft2(projected)
        nonzero = np.ones(full_shape, dtype=bool)
        nonzero[0, 0] = False
        spectrum_drift = float(
            np.max(np.abs(projected_fft[nonzero] - raw_fft[nonzero]))
        )
        raw_centered = raw - float(np.mean(raw, dtype=np.float64))
        projected_centered = projected - float(
            np.mean(projected, dtype=np.float64)
        )
        raw_covariance = np.fft.ifft2(
            np.square(np.abs(np.fft.fft2(raw_centered))) / raw.size
        ).real
        projected_covariance = np.fft.ifft2(
            np.square(np.abs(np.fft.fft2(projected_centered))) / projected.size
        ).real
        covariance_drift = float(
            np.max(np.abs(projected_covariance - raw_covariance))
        )
        variance_drift = abs(
            float(np.var(projected, dtype=np.float64))
            - float(np.var(raw, dtype=np.float64))
        )
        projected_mean = abs(float(np.mean(projected, dtype=np.float64)))
        interior_variance_error = abs(
            float(np.var(projected[border:-border, border:-border], dtype=np.float64))
            - 1.0
        )
        tampered_rejected = False
        try:
            render_dc_projected_thomas_region(
                replace(receipt, raw_mean=receipt.raw_mean + 0.125),
                origin_yx=(0, 0),
                shape=(1, 1),
            )
        except ThomasDcProjectionError:
            tampered_rejected = True
        digest = hashlib.sha256(
            np.ascontiguousarray(projected, dtype="<f8").tobytes()
        ).hexdigest()
        projected_hashes.append(digest)
        rows.append(
            {
                "seed": int(seed),
                "receipt_id": receipt.receipt_id,
                "raw_field_sha256": receipt.raw_field_sha256,
                "projected_field_sha256": digest,
                "raw_mean": receipt.raw_mean,
                "projected_absolute_mean": projected_mean,
                "maximum_nonzero_spectrum_absolute_drift": spectrum_drift,
                "maximum_centered_covariance_absolute_drift": covariance_drift,
                "variance_absolute_drift": variance_drift,
                "interior_variance_absolute_error": interior_variance_error,
                "row_partition_exact": row_exact,
                "repeat_receipt_exact": receipt == repeated_receipt,
                "tampered_receipt_rejected": tampered_rejected,
            }
        )
    checks = {
        "projected_mean": max(row["projected_absolute_mean"] for row in rows)
        <= evaluation["maximum_projected_absolute_mean"],
        "nonzero_spectrum": max(
            row["maximum_nonzero_spectrum_absolute_drift"] for row in rows
        )
        <= evaluation["maximum_nonzero_spectrum_absolute_drift"],
        "centered_covariance": max(
            row["maximum_centered_covariance_absolute_drift"] for row in rows
        )
        <= evaluation["maximum_centered_covariance_absolute_drift"],
        "variance": max(row["variance_absolute_drift"] for row in rows)
        <= evaluation["maximum_variance_absolute_drift"],
        "interior_variance": max(
            row["interior_variance_absolute_error"] for row in rows
        )
        <= evaluation["maximum_interior_variance_absolute_error"],
        "row_partition_exact": all(row["row_partition_exact"] for row in rows),
        "repeat_receipt_exact": all(
            row["repeat_receipt_exact"] for row in rows
        ),
        "tampered_receipt_rejected": all(
            row["tampered_receipt_rejected"] for row in rows
        ),
        "seed_distinct_fields": len(set(projected_hashes)) == len(projected_hashes),
        "finite": all(
            math.isfinite(float(value))
            for row in rows
            for key, value in row.items()
            if key.endswith(("mean", "drift", "error"))
        ),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4bv_thomas_dc_projection_v1.json", "sha256"
        ),
        "profile": contract["profile"],
        "receipt_contract": contract["receipt"],
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_two_pass_dc_projected_thomas_unit_field"
            if automatic_pass
            else "close_two_pass_dc_projected_thomas_unit_field"
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
    "ThomasDcProjectionEvaluationError",
    "_validate_contract",
    "evaluate_thomas_dc_projection",
    "write_report",
]
