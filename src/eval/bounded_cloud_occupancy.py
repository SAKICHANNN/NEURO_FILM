"""U6.P4CA bounded-cloud second-order and tail evaluation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.bounded_cloud_occupancy import (
    BoundedCloudOccupancyError,
    build_bounded_cloud_dc_receipt,
    render_dc_projected_bounded_cloud_region,
)
from src.film_physics.thomas_dc_projection import (
    build_thomas_dc_receipt,
    render_dc_projected_thomas_region,
)

SCHEMA = "neuro_film.u6_p4ca_bounded_cloud_occupancy_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4ca_bounded_cloud_occupancy_report.v1"


class BoundedCloudOccupancyEvaluationError(RuntimeError):
    """Raised when the P4CA contract or parent evidence drifts."""


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
        or profile.get("occupancy_trials_per_cell") != 16
        or profile.get("occupancy_probability") != 0.5
        or profile.get("component_seeds") != [2611923443488327891, 11400714819323198485]
        or profile.get("realized_variance_normalization_allowed") is not False
        or profile.get("pointwise_clipping_allowed") is not False
        or receipt.get("canonical_row_block_height") != 31
        or receipt.get("remove_only_full_field_dc") is not True
        or evaluation.get("field_shape") != [257, 263]
        or evaluation.get("row_partitions") != [1, 7, 31, 64, 127]
        or evaluation.get("extreme_absolute_field_threshold") != 4.0
        or evaluation.get("maximum_extreme_count_ratio_vs_gaussian") != 0.75
        or evaluation.get("require_two_byte_identical_reports") is not True
    ):
        raise BoundedCloudOccupancyEvaluationError("P4CA frozen contract drift")


def _load_parent(contract: Mapping[str, Any], root: Path, key: str) -> dict[str, Any]:
    binding = contract["parents"][key]
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise BoundedCloudOccupancyEvaluationError(f"P4CA parent mismatch: {key}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("decision") != binding["required_decision"]:
        raise BoundedCloudOccupancyEvaluationError(f"P4CA parent decision drift: {key}")
    return payload


def _radial_signature(power: np.ndarray, edges: np.ndarray) -> np.ndarray:
    fy = np.fft.fftfreq(power.shape[0])[:, None]
    fx = np.fft.fftfreq(power.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    bands = []
    for lower, upper in pairwise(edges):
        selected = power[(radius >= lower) & (radius < upper)]
        if selected.size == 0:
            raise BoundedCloudOccupancyEvaluationError("empty radial band")
        bands.append(float(np.mean(selected, dtype=np.float64)))
    values = np.asarray(bands, dtype=np.float64)
    values /= float(np.sum(values, dtype=np.float64))
    values = np.log(values)
    values -= float(np.mean(values, dtype=np.float64))
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        raise BoundedCloudOccupancyEvaluationError("degenerate radial signature")
    return values / norm


def _covariance(field: np.ndarray, lags: list[list[int]]) -> np.ndarray:
    centered = field - float(np.mean(field, dtype=np.float64))
    values = []
    for dy, dx in lags:
        left = centered[: centered.shape[0] - dy, : centered.shape[1] - dx]
        right = centered[dy:, dx:]
        values.append(float(np.mean(left * right, dtype=np.float64)))
    return np.asarray(values, dtype=np.float64)


def evaluate_bounded_cloud_occupancy(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    _validate_contract(contract)
    for key in ("p4bs_decision", "p4bu_decision", "p4bz_decision"):
        _load_parent(contract, root, key)
    profile = contract["profile"]
    receipt_spec = contract["receipt"]
    evaluation = contract["evaluation"]
    full_shape = tuple(int(value) for value in evaluation["field_shape"])
    border = int(evaluation["interior_border_pixels"])
    common = {
        "profile_id": str(profile["profile_id"]),
        "particle_sigma_pixels": float(profile["particle_sigma_pixels"]),
        "cluster_sigma_pixels": float(profile["cluster_sigma_pixels"]),
        "mean_offspring": float(profile["mean_offspring"]),
        "component_seeds": tuple(int(value) for value in profile["component_seeds"]),
        "truncate": float(profile["truncate_sigma"]),
        "canonical_row_block_height": int(receipt_spec["canonical_row_block_height"]),
    }
    candidate_interiors: list[np.ndarray] = []
    gaussian_interiors: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    receipt_ids: list[str] = []
    partition_exact: list[bool] = []
    repeat_exact: list[bool] = []
    projected_means: list[float] = []
    variances: list[float] = []
    for seed in evaluation["realization_seeds"]:
        candidate_receipt = build_bounded_cloud_dc_receipt(
            full_shape,
            realization_seed=int(seed),
            trials=int(profile["occupancy_trials_per_cell"]),
            probability=float(profile["occupancy_probability"]),
            **common,
        )
        repeated_receipt = build_bounded_cloud_dc_receipt(
            full_shape,
            realization_seed=int(seed),
            trials=int(profile["occupancy_trials_per_cell"]),
            probability=float(profile["occupancy_probability"]),
            **common,
        )
        candidate = render_dc_projected_bounded_cloud_region(
            candidate_receipt, origin_yx=(0, 0), shape=full_shape
        )
        assembled = np.empty_like(candidate)
        seed_partition_exact = True
        for row_height in evaluation["row_partitions"]:
            trial = np.empty_like(candidate)
            for y0 in range(0, full_shape[0], int(row_height)):
                height = min(int(row_height), full_shape[0] - y0)
                trial[y0 : y0 + height] = render_dc_projected_bounded_cloud_region(
                    candidate_receipt,
                    origin_yx=(y0, 0),
                    shape=(height, full_shape[1]),
                )
            seed_partition_exact &= bool(np.array_equal(candidate, trial))
            assembled = trial
        gaussian_receipt = build_thomas_dc_receipt(
            full_shape,
            profile_id="generic-scanner-convolved-thomas-p4bs-v1",
            realization_seed=int(seed),
            **{key: value for key, value in common.items() if key != "profile_id"},
        )
        gaussian = render_dc_projected_thomas_region(
            gaussian_receipt, origin_yx=(0, 0), shape=full_shape
        )
        interior = candidate[border:-border, border:-border]
        gaussian_interior = gaussian[border:-border, border:-border]
        candidate_interiors.append(interior)
        gaussian_interiors.append(gaussian_interior)
        projected_mean = abs(float(np.mean(candidate, dtype=np.float64)))
        variance = float(np.var(interior, dtype=np.float64))
        receipt_ids.append(candidate_receipt.receipt_id)
        partition_exact.append(
            seed_partition_exact and np.array_equal(candidate, assembled)
        )
        repeat_exact.append(candidate_receipt == repeated_receipt)
        projected_means.append(projected_mean)
        variances.append(variance)
        rows.append(
            {
                "realization_seed": int(seed),
                "receipt_id": candidate_receipt.receipt_id,
                "field_sha256": hashlib.sha256(
                    np.ascontiguousarray(candidate, dtype="<f8").tobytes()
                ).hexdigest(),
                "projected_absolute_mean": projected_mean,
                "interior_variance": variance,
                "interior_maximum_absolute_field": float(
                    np.max(np.abs(interior), initial=0.0)
                ),
                "row_partition_exact": partition_exact[-1],
                "repeat_receipt_exact": repeat_exact[-1],
            }
        )
    candidate_values = np.concatenate(
        [np.abs(v).reshape(-1) for v in candidate_interiors]
    )
    gaussian_values = np.concatenate(
        [np.abs(v).reshape(-1) for v in gaussian_interiors]
    )
    threshold = float(evaluation["extreme_absolute_field_threshold"])
    candidate_extreme_count = int(np.count_nonzero(candidate_values >= threshold))
    gaussian_extreme_count = int(np.count_nonzero(gaussian_values >= threshold))
    extreme_ratio = candidate_extreme_count / max(1, gaussian_extreme_count)

    candidate_power = np.mean(
        [np.square(np.abs(np.fft.fft2(v))) for v in candidate_interiors], axis=0
    )
    gaussian_power = np.mean(
        [np.square(np.abs(np.fft.fft2(v))) for v in gaussian_interiors], axis=0
    )
    edges = np.asarray(
        evaluation["radial_band_edges_cycles_per_pixel"], dtype=np.float64
    )
    candidate_signature = _radial_signature(candidate_power, edges)
    gaussian_signature = _radial_signature(gaussian_power, edges)
    radial_error = float(
        1.0
        - np.dot(candidate_signature, gaussian_signature)
        / (np.linalg.norm(candidate_signature) * np.linalg.norm(gaussian_signature))
    )
    candidate_covariance = np.mean(
        [_covariance(v, evaluation["covariance_lags_yx"]) for v in candidate_interiors],
        axis=0,
    )
    gaussian_covariance = np.mean(
        [_covariance(v, evaluation["covariance_lags_yx"]) for v in gaussian_interiors],
        axis=0,
    )
    covariance_error = float(
        np.max(np.abs(candidate_covariance - gaussian_covariance), initial=0.0)
    )
    candidate_maximum = float(np.max(candidate_values, initial=0.0))
    candidate_q9999 = float(np.quantile(candidate_values, 0.9999, method="linear"))
    tampered_rejected = False
    try:
        render_dc_projected_bounded_cloud_region(
            replace(
                build_bounded_cloud_dc_receipt(
                    full_shape,
                    realization_seed=int(evaluation["realization_seeds"][0]),
                    trials=int(profile["occupancy_trials_per_cell"]),
                    probability=float(profile["occupancy_probability"]),
                    **common,
                ),
                raw_mean=1.0,
            ),
            origin_yx=(0, 0),
            shape=(1, 1),
        )
    except BoundedCloudOccupancyError:
        tampered_rejected = True
    checks = {
        "extreme_tail_reduction": extreme_ratio
        <= float(evaluation["maximum_extreme_count_ratio_vs_gaussian"]),
        "maximum_absolute_field": candidate_maximum
        <= float(evaluation["maximum_candidate_absolute_field"]),
        "quantile_9999": candidate_q9999
        <= float(evaluation["maximum_candidate_absolute_quantile_9999"]),
        "radial_signature": radial_error
        <= float(evaluation["maximum_radial_signature_cosine_error_vs_gaussian"]),
        "covariance": covariance_error
        <= float(evaluation["maximum_covariance_absolute_error_vs_gaussian"]),
        "variance": max(abs(value - 1.0) for value in variances)
        <= float(evaluation["maximum_variance_absolute_error"]),
        "projected_mean": max(projected_means)
        <= float(evaluation["maximum_projected_absolute_mean"]),
        "row_partition_exact": all(partition_exact),
        "repeat_receipt_exact": all(repeat_exact),
        "tampered_receipt_rejected": tampered_rejected,
        "seed_distinct_fields": len(set(receipt_ids)) == len(receipt_ids),
        "finite": bool(
            np.all(np.isfinite(candidate_values))
            and np.all(np.isfinite(candidate_covariance))
        ),
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hash_file(
            root / "configs/u6_p4ca_bounded_cloud_occupancy_v1.json", "sha256"
        ),
        "profile": profile,
        "candidate_extreme_count": candidate_extreme_count,
        "gaussian_extreme_count": gaussian_extreme_count,
        "extreme_count_ratio_vs_gaussian": extreme_ratio,
        "candidate_maximum_absolute_field": candidate_maximum,
        "candidate_absolute_quantile_9999": candidate_q9999,
        "radial_signature_cosine_error_vs_gaussian": radial_error,
        "maximum_covariance_absolute_error_vs_gaussian": covariance_error,
        "maximum_variance_absolute_error": max(abs(value - 1.0) for value in variances),
        "maximum_projected_absolute_mean": max(projected_means),
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_bounded_cloud_occupancy_unit_field"
            if automatic_pass
            else "close_bounded_cloud_occupancy_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(
            stable, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "BoundedCloudOccupancyEvaluationError",
    "evaluate_bounded_cloud_occupancy",
    "write_report",
]
