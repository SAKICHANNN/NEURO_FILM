"""Synthetic known-truth audit for hard latent operator modes.

This module deliberately does not consume photographic pixels.  It tests a
methodological prerequisite: residual directions that differ should survive
basic affine nuisance removal, while one direction at several strengths should
not be promoted into several modes.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "neuro_film.u5_r2bk19_synthetic_hard_mode_identifiability_report.v1"


class SyntheticHardModeError(ValueError):
    """Raised when the frozen synthetic contract is invalid."""


@dataclass(frozen=True)
class CaseSignature:
    group_id: int
    label: str
    strength: float
    direction: np.ndarray
    raw_residual: np.ndarray
    out_of_cube_fraction: float


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


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
        != "neuro_film.u5_r2bk19_synthetic_hard_mode_identifiability.v1"
        or config.get("status") != "contract_frozen"
    ):
        raise SyntheticHardModeError("BK19 contract is not frozen")
    parent = root / config["parent"]["decision"]
    decision = json.loads(parent.read_text(encoding="utf-8"))
    if decision.get("decision") != config["parent"]["required_decision"]:
        raise SyntheticHardModeError("BK19 parent decision mismatch")
    return config


def _grid(size: int, margin: float) -> np.ndarray:
    axis = np.linspace(margin, 1.0 - margin, size, dtype=np.float64)
    rr, gg, bb = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack((rr, gg, bb), axis=-1).reshape(-1, 3)


def _mode_residual(source: np.ndarray, label: str) -> np.ndarray:
    r, g, b = source.T
    envelope = 4.0 * source * (1.0 - source)
    if label == "Mode A":
        # Warm highlights and cyan-shadow suppression with non-affine hue/luma
        # coupling.  The envelope preserves cube endpoints.
        scalar = 0.018 * (0.35 + 0.65 * (r + g + b) / 3.0)
        field = np.stack(
            (
                scalar * (0.55 + 0.45 * g),
                scalar * (0.12 - 0.35 * b),
                -scalar * (0.65 + 0.25 * r),
            ),
            axis=1,
        )
    elif label == "Mode B":
        # A genuinely different direction: cool shadows, restrained green,
        # and warm upper reds.  It cannot be obtained by scalar strength alone.
        luma = (r + g + b) / 3.0
        scalar = 0.017 * (0.45 + 0.55 * (1.0 - luma))
        field = np.stack(
            (
                scalar * (0.35 * luma - 0.55 * b),
                -scalar * (0.25 + 0.20 * r),
                scalar * (0.65 + 0.25 * g),
            ),
            axis=1,
        )
    else:
        raise SyntheticHardModeError(f"unknown mode label: {label}")
    return envelope * field


def _basic_nuisance(source: np.ndarray, group_id: int) -> np.ndarray:
    phase = float(group_id + 1)
    matrix = np.asarray(
        [
            [1.0 + 0.006 * np.sin(phase), 0.002 * np.cos(phase), 0.0],
            [0.0015 * np.cos(0.7 * phase), 1.0 - 0.005 * np.sin(phase), 0.001],
            [0.0, -0.002 * np.sin(0.5 * phase), 1.0 + 0.004 * np.cos(phase)],
        ],
        dtype=np.float64,
    )
    offset = np.asarray(
        [
            0.0015 * np.sin(0.4 * phase),
            -0.0010 * np.cos(0.3 * phase),
            0.0012 * np.sin(0.6 * phase),
        ],
        dtype=np.float64,
    )
    return source @ matrix.T + offset


def _remove_basic_affine(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    design = np.concatenate(
        (source, np.ones((source.shape[0], 1), dtype=np.float64)), axis=1
    )
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    return target - design @ coefficients


def _smooth_residual_signature(source: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Project pixel residuals onto a fixed non-affine operator basis."""
    r, g, b = source.T
    design = np.stack(
        (
            r * r,
            g * g,
            b * b,
            r * g,
            r * b,
            g * b,
            r * g * b,
            r * r * g,
            g * g * b,
            b * b * r,
        ),
        axis=1,
    )
    coefficients, *_ = np.linalg.lstsq(design, residual, rcond=None)
    return coefficients.reshape(-1)


def _make_case(
    source: np.ndarray,
    *,
    group_id: int,
    label: str,
    strength: float,
    noise_sigma: float,
    seed: int,
) -> CaseSignature:
    target = _basic_nuisance(source, group_id) + strength * _mode_residual(
        source, label
    )
    rng = np.random.default_rng(seed + group_id * 101 + int(strength * 1000))
    noise = rng.normal(0.0, noise_sigma, size=target.shape)
    # Project noise away from the endpoints by the same smooth envelope.
    target = target + noise * (4.0 * source * (1.0 - source))
    out_of_cube = float(np.mean((target < 0.0) | (target > 1.0)))
    residual_image = _remove_basic_affine(source, target)
    residual = residual_image.reshape(-1)
    signature = _smooth_residual_signature(source, residual_image)
    norm = float(np.linalg.norm(signature))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise SyntheticHardModeError("degenerate residual signature")
    return CaseSignature(
        group_id=group_id,
        label=label,
        strength=float(strength),
        direction=signature / norm,
        raw_residual=residual,
        out_of_cube_fraction=out_of_cube,
    )


def cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(max(0.0, 1.0 - np.clip(np.dot(left, right), -1.0, 1.0)))


def _medoid(directions: list[np.ndarray]) -> np.ndarray:
    if not directions:
        raise SyntheticHardModeError("empty medoid input")
    matrix = np.stack(directions, axis=0)
    distance = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)
    return matrix[int(np.argmin(np.sum(distance, axis=1)))]


def _fit_two_medoids(directions: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if len(directions) < 2:
        raise SyntheticHardModeError("two medoids require at least two cases")
    matrix = np.stack(directions, axis=0)
    distance = 1.0 - np.clip(matrix @ matrix.T, -1.0, 1.0)
    first, second = np.unravel_index(np.argmax(distance), distance.shape)
    assignments = np.zeros(len(directions), dtype=np.int64)
    for _ in range(32):
        d0 = 1.0 - np.clip(matrix @ matrix[first], -1.0, 1.0)
        d1 = 1.0 - np.clip(matrix @ matrix[second], -1.0, 1.0)
        updated = (d1 < d0).astype(np.int64)
        if np.array_equal(updated, assignments) and _ > 0:
            break
        assignments = updated
        next_indices: list[int] = []
        for cluster in (0, 1):
            members = np.flatnonzero(assignments == cluster)
            if members.size == 0:
                raise SyntheticHardModeError("empty deterministic cluster")
            local = distance[np.ix_(members, members)]
            next_indices.append(int(members[int(np.argmin(np.sum(local, axis=1)))]))
        first, second = next_indices
    return matrix[first], matrix[second]


def _leave_one_group_out_accuracy(cases: list[CaseSignature]) -> float:
    correct = 0
    count = 0
    for held_group in sorted({case.group_id for case in cases}):
        train = [case for case in cases if case.group_id != held_group]
        held = [case for case in cases if case.group_id == held_group]
        medoids = _fit_two_medoids([case.direction for case in train])
        # K-medoids cluster indices are unordered.  Align them using only the
        # development groups' known synthetic truth before scoring held groups.
        train_assignments = [
            int(
                np.argmin(
                    [
                        cosine_distance(case.direction, medoid)
                        for medoid in medoids
                    ]
                )
            )
            for case in train
        ]
        mapping: dict[int, str] = {}
        for cluster in (0, 1):
            labels = [
                case.label
                for case, assignment in zip(train, train_assignments, strict=True)
                if assignment == cluster
            ]
            if not labels:
                raise SyntheticHardModeError("empty cluster during label alignment")
            mapping[cluster] = max(sorted(set(labels)), key=labels.count)
        for case in held:
            distances = [cosine_distance(case.direction, medoid) for medoid in medoids]
            predicted = mapping[int(np.argmin(distances))]
            correct += int(predicted == case.label)
            count += 1
    return float(correct / count)


def _directional_errors(
    cases: list[CaseSignature], medoids: tuple[np.ndarray, ...]
) -> np.ndarray:
    return np.asarray(
        [
            min(cosine_distance(case.direction, medoid) for medoid in medoids)
            for case in cases
        ],
        dtype=np.float64,
    )


def _evaluate_bank(cases: list[CaseSignature]) -> dict[str, Any]:
    one = (_medoid([case.direction for case in cases]),)
    two = _fit_two_medoids([case.direction for case in cases])
    error_k1 = _directional_errors(cases, one)
    error_k2 = _directional_errors(cases, two)
    denominator = max(float(np.mean(error_k1)), 1e-15)
    return {
        "mean_directional_error_k1": float(np.mean(error_k1)),
        "mean_directional_error_k2": float(np.mean(error_k2)),
        "k2_relative_directional_error_improvement": float(
            1.0 - float(np.mean(error_k2)) / denominator
        ),
        "k2_absolute_directional_error_improvement": float(
            np.mean(error_k1) - np.mean(error_k2)
        ),
        "medoid_cross_direction_cosine": float(np.dot(two[0], two[1])),
    }


def evaluate(config: dict[str, Any], *, config_sha256: str) -> dict[str, Any]:
    design = config["synthetic_design"]
    source = _grid(int(design["grid_size"]), float(design["grid_margin"]))
    groups = int(design["independent_groups"])
    noise = float(design["noise_sigma"])
    seed = int(design["seed"])

    true_modes = [
        _make_case(
            source,
            group_id=group_id,
            label=label,
            strength=1.0,
            noise_sigma=noise,
            seed=seed,
        )
        for group_id in range(groups)
        for label in ("Mode A", "Mode B")
    ]
    strengths = [
        _make_case(
            source,
            group_id=group_id,
            label="Mode A",
            strength=float(strength),
            noise_sigma=noise,
            seed=seed,
        )
        for group_id in range(groups)
        for strength in design["strength_path_negative_control"]["strengths"]
    ]

    true_metrics = _evaluate_bank(true_modes)
    true_metrics["leave_one_group_out_accuracy"] = _leave_one_group_out_accuracy(
        true_modes
    )
    true_metrics["independent_groups_per_mode"] = {
        label: len({case.group_id for case in true_modes if case.label == label})
        for label in ("Mode A", "Mode B")
    }
    true_metrics["maximum_single_group_share"] = float(1.0 / groups)

    strength_metrics = _evaluate_bank(strengths)
    same_group_cosines = []
    for group_id in range(groups):
        local = [case for case in strengths if case.group_id == group_id]
        for left_index, left in enumerate(local):
            for right in local[left_index + 1 :]:
                same_group_cosines.append(float(np.dot(left.direction, right.direction)))
    strength_metrics["minimum_same_group_cross_strength_cosine"] = float(
        min(same_group_cosines)
    )

    all_cases = true_modes + strengths
    maximum_oob = max(case.out_of_cube_fraction for case in all_cases)
    gates = config["gates"]
    checks = [
        {
            "name": "true_mode_leave_one_group_out_accuracy",
            "passed": true_metrics["leave_one_group_out_accuracy"]
            >= float(gates["minimum_true_mode_leave_one_group_out_accuracy"]),
        },
        {
            "name": "true_mode_hard_bank_gain",
            "passed": true_metrics["k2_absolute_directional_error_improvement"]
            >= float(
                gates[
                    "minimum_true_mode_k2_absolute_directional_error_improvement"
                ]
            ),
        },
        {
            "name": "true_mode_group_support",
            "passed": min(true_metrics["independent_groups_per_mode"].values())
            >= int(gates["minimum_independent_groups_per_true_mode"]),
        },
        {
            "name": "true_mode_group_concentration",
            "passed": true_metrics["maximum_single_group_share"]
            <= float(gates["maximum_single_group_share_per_true_mode"]),
        },
        {
            "name": "true_mode_direction_separation",
            "passed": true_metrics["medoid_cross_direction_cosine"]
            <= float(gates["maximum_true_mode_cross_direction_cosine"]),
        },
        {
            "name": "strength_path_direction_consistency",
            "passed": strength_metrics["minimum_same_group_cross_strength_cosine"]
            >= float(gates["minimum_same_direction_strength_cosine"]),
        },
        {
            "name": "strength_path_rejects_false_k2_value",
            "passed": strength_metrics["k2_absolute_directional_error_improvement"]
            <= float(
                gates[
                    "maximum_strength_path_k2_absolute_directional_error_improvement"
                ]
            ),
        },
        {
            "name": "synthetic_output_in_cube",
            "passed": maximum_oob
            <= float(gates["maximum_output_out_of_cube_fraction"]),
        },
    ]
    automatic_pass = all(check["passed"] for check in checks)
    return {
        "schema": SCHEMA,
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_sha256": config_sha256,
        "sample_counts": {
            "grid_points": int(source.shape[0]),
            "true_mode_cases": len(true_modes),
            "strength_path_cases": len(strengths),
            "independent_groups": groups,
        },
        "true_mode": true_metrics,
        "strength_path_negative_control": strength_metrics,
        "maximum_output_out_of_cube_fraction": maximum_oob,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            config["decision_if_pass"]
            if automatic_pass
            else config["decision_if_fail"]
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
    "CaseSignature",
    "SCHEMA",
    "SyntheticHardModeError",
    "canonical_json_bytes",
    "cosine_distance",
    "evaluate",
    "load_config",
    "run",
]
