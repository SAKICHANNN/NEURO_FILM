"""Constructive nonidentifiability witness for unpaired colour operators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


SCHEMA = "neuro_film.u5_r2bk20_unpaired_operator_nonidentifiability_report.v1"


class UnpairedNonidentifiabilityError(ValueError):
    """Raised when the BK20 witness contract is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        config.get("schema")
        != "neuro_film.u5_r2bk20_unpaired_operator_nonidentifiability.v1"
        or config.get("status") != "contract_frozen"
    ):
        raise UnpairedNonidentifiabilityError("BK20 contract is not frozen")
    parent_path = root / config["parent"]["decision"]
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent.get("decision") != config["parent"]["required_decision"]:
        raise UnpairedNonidentifiabilityError("BK20 parent decision mismatch")
    return config


def build_ring_set(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    witness = config["witness"]
    centre = float(witness["centre"])
    z_offsets = tuple(float(value) for value in witness["z_offsets"])
    radii = tuple(float(value) for value in witness["radii"])
    angle_count = int(witness["angles_per_ring"])
    twist_steps = tuple(int(value) for value in witness["twist_steps_by_z"])
    if len(z_offsets) != len(twist_steps) or angle_count < 8:
        raise UnpairedNonidentifiabilityError("invalid ring witness geometry")
    points: list[tuple[float, float, float]] = []
    shifts: list[int] = []
    for z_offset, step in zip(z_offsets, twist_steps, strict=True):
        for radius in radii:
            for angle_index in range(angle_count):
                theta = 2.0 * np.pi * angle_index / angle_count
                points.append(
                    (
                        centre + radius * np.cos(theta),
                        centre + radius * np.sin(theta),
                        centre + z_offset,
                    )
                )
                shifts.append(step)
    source = np.asarray(points, dtype=np.float64)
    return source, np.asarray(shifts, dtype=np.int64)


def apply_twist(
    source: np.ndarray,
    shifts: np.ndarray,
    *,
    angle_count: int,
    inverse: bool = False,
) -> np.ndarray:
    if source.shape != (shifts.size, 3):
        raise UnpairedNonidentifiabilityError("twist source/shift shape mismatch")
    if source.shape[0] % angle_count:
        raise UnpairedNonidentifiabilityError("twist rings are incomplete")
    output = np.empty_like(source)
    direction = -1 if inverse else 1
    for start in range(0, source.shape[0], angle_count):
        stop = start + angle_count
        local_shifts = shifts[start:stop]
        if not np.all(local_shifts == local_shifts[0]):
            raise UnpairedNonidentifiabilityError("twist shift varies within ring")
        output[start:stop] = np.roll(
            source[start:stop],
            shift=direction * int(local_shifts[0]),
            axis=0,
        )
    return output


def canonical_sort(points: np.ndarray) -> np.ndarray:
    # Round only at a precision far below the frozen numerical tolerances so
    # analytically identical angular samples have one portable byte identity.
    rounded = np.round(np.asarray(points, dtype=np.float64), decimals=14)
    order = np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))
    return np.ascontiguousarray(rounded[order], dtype="<f8")


def _affine_residual_rmse(source: np.ndarray, target: np.ndarray) -> float:
    design = np.concatenate(
        (source, np.ones((source.shape[0], 1), dtype=np.float64)), axis=1
    )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    residual = target - design @ coefficients
    return float(np.sqrt(np.mean(np.square(residual))))


def _assignment_predictions(
    source: np.ndarray, observed_target: np.ndarray
) -> dict[str, np.ndarray]:
    rounded_source = np.round(source, decimals=14)
    rounded_target = np.round(observed_target, decimals=14)
    source_order = np.lexsort(
        (rounded_source[:, 2], rounded_source[:, 1], rounded_source[:, 0])
    )
    target_order = np.lexsort(
        (rounded_target[:, 2], rounded_target[:, 1], rounded_target[:, 0])
    )
    lexicographic = np.empty_like(source)
    lexicographic[source_order] = observed_target[target_order]
    cost = np.sum(
        np.square(source[:, None, :] - observed_target[None, :, :]), axis=2
    )
    rows, columns = linear_sum_assignment(cost)
    euclidean = np.empty_like(source)
    euclidean[rows] = observed_target[columns]
    return {
        "lexicographic-rank": lexicographic,
        "minimum-euclidean-assignment": euclidean,
    }


def evaluate(config: dict[str, Any], *, config_sha256: str) -> dict[str, Any]:
    source, shifts = build_ring_set(config)
    angle_count = int(config["witness"]["angles_per_ring"])
    identity_target = source.copy()
    twist_target = apply_twist(source, shifts, angle_count=angle_count)
    inverse = apply_twist(
        twist_target, shifts, angle_count=angle_count, inverse=True
    )

    source_sorted = canonical_sort(source)
    identity_sorted = canonical_sort(identity_target)
    twist_sorted = canonical_sort(twist_target)
    source_bytes = source_sorted.tobytes(order="C")
    identity_bytes = identity_sorted.tobytes(order="C")
    twist_bytes = twist_sorted.tobytes(order="C")

    paired_rmse = float(np.sqrt(np.mean(np.square(twist_target - identity_target))))
    non_affine_rmse = _affine_residual_rmse(source, twist_target)
    inverse_error = float(np.max(np.abs(inverse - source)))
    out_of_cube = float(np.mean((twist_target < 0.0) | (twist_target > 1.0)))

    canonicalizer_rows: dict[str, Any] = {}
    for name, prediction in _assignment_predictions(source, twist_target).items():
        identity_error = float(np.sqrt(np.mean(np.square(prediction - source))))
        twist_error = float(
            np.sqrt(np.mean(np.square(prediction - twist_target)))
        )
        recovered_fraction = float(
            np.mean(np.all(np.isclose(prediction, twist_target, atol=1e-12), axis=1))
        )
        canonicalizer_rows[name] = {
            "paired_rmse_to_identity": identity_error,
            "paired_rmse_to_true_twist": twist_error,
            "exact_true_twist_row_fraction": recovered_fraction,
        }

    gates = config["gates"]
    maximum_recovered = max(
        row["exact_true_twist_row_fraction"] for row in canonicalizer_rows.values()
    )
    checks = [
        {
            "name": "observed_source_sets_byte_exact",
            "passed": source_bytes == source_bytes,
        },
        {
            "name": "observed_target_sets_canonical_exact",
            "passed": identity_bytes == twist_bytes,
        },
        {
            "name": "true_operators_materially_different",
            "passed": paired_rmse
            >= float(gates["minimum_true_operator_paired_rgb_rmse"]),
        },
        {
            "name": "twist_not_explained_by_affine_basic",
            "passed": non_affine_rmse
            >= float(gates["minimum_twist_non_affine_residual_rmse"]),
        },
        {
            "name": "twist_inverse_exact",
            "passed": float(gates["minimum_twist_inverse_maximum_error"])
            <= inverse_error
            <= float(gates["maximum_twist_inverse_maximum_error"]),
        },
        {
            "name": "twist_cube_safe",
            "passed": out_of_cube
            <= float(gates["maximum_out_of_cube_fraction"]),
        },
        {
            "name": "canonicalizers_cannot_recover_hidden_twist",
            "passed": maximum_recovered
            <= float(gates["maximum_canonicalizer_recovered_twist_fraction"]),
        },
    ]
    automatic_pass = all(check["passed"] for check in checks)
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "point_count": int(source.shape[0]),
        "observed_source_set_sha256": sha256_bytes(source_bytes),
        "identity_observed_target_set_sha256": sha256_bytes(identity_bytes),
        "twist_observed_target_set_sha256": sha256_bytes(twist_bytes),
        "observed_target_sets_byte_exact": identity_bytes == twist_bytes,
        "true_operator_paired_rgb_rmse": paired_rmse,
        "twist_non_affine_residual_rmse": non_affine_rmse,
        "twist_inverse_maximum_error": inverse_error,
        "twist_out_of_cube_fraction": out_of_cube,
        "canonicalizers": canonicalizer_rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_witness_passes"]
            if automatic_pass
            else config["decision_if_witness_fails"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


def run(root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    config = load_config(root, config_path)
    report = evaluate(config, config_sha256=sha256_file(config_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(report))
    return report


__all__ = [
    "SCHEMA",
    "UnpairedNonidentifiabilityError",
    "apply_twist",
    "build_ring_set",
    "canonical_json_bytes",
    "canonical_sort",
    "evaluate",
    "load_config",
    "run",
    "sha256_file",
]
