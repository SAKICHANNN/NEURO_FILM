"""U6.P2V clean-room saturating interimage donor evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.spektrafilm_spatial_dir import sha256_file
from src.film_physics.interimage_development import InterimageDevelopmentOperator
from src.film_physics.langmuir_interimage import (
    LangmuirDonorProfile,
    apply_langmuir_interimage_development,
    langmuir_interimage_jacobian,
)


SCHEMA = "neuro_film.u6_p2v_langmuir_interimage_primitive_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P2V contract")
    return value


def _operator(payload: dict[str, Any]) -> InterimageDevelopmentOperator:
    return InterimageDevelopmentOperator(
        tuple(payload["density_min"]),
        tuple(payload["density_max"]),
        tuple(payload["slope"]),
        tuple(payload["midpoint"]),
        tuple(tuple(row) for row in payload["coupling"]),
    )


def _profile(config: dict[str, Any], *, linear: bool = False) -> LangmuirDonorProfile:
    donor = config["donor"]
    knee = (float("inf"),) * 3 if linear else tuple(donor["knee_fraction_of_capacity"])
    return LangmuirDonorProfile(
        tuple(donor["density_capacity"]),
        knee,
        tuple(donor["match_fraction_of_capacity"]),
    )


def _scalar_oracle(
    value: float, capacity: float, knee_fraction: float, match: float
) -> float:
    if np.isposinf(knee_fraction):
        return value
    knee = knee_fraction * capacity
    reference = match * capacity
    return value * (knee + reference) / (knee + value)


def _numeric_jacobian(
    operator: InterimageDevelopmentOperator,
    exposure: np.ndarray,
    donor: LangmuirDonorProfile,
    step: float,
) -> np.ndarray:
    rows = []
    for source in range(3):
        plus = exposure.copy()
        minus = exposure.copy()
        plus[:, source] += step
        minus[:, source] -= step
        rows.append(
            (
                apply_langmuir_interimage_development(operator, plus, donor)
                - apply_langmuir_interimage_development(operator, minus, donor)
            )
            / (2.0 * step)
        )
    return np.stack(rows, axis=-1)


def _validate_parents(root: Path, config: dict[str, Any]) -> None:
    parents = config["parents"]
    for key in ("u5_r2bs0_evidence", "u6_p2m_contract", "u6_p2m_decision"):
        if sha256_file(root / parents[key]) != parents[f"{key}_sha256"]:
            raise ValueError(f"U6.P2V parent drift: {key}")


def evaluate_langmuir_interimage(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    _validate_parents(root, config)
    profile = _profile(config)
    linear_profile = _profile(config, linear=True)
    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float64)
    density_grid = np.repeat(grid[:, None], 3, axis=1)
    donor_values = profile.apply(density_grid)
    oracle = np.empty_like(donor_values)
    for row in range(len(grid)):
        for channel in range(3):
            oracle[row, channel] = _scalar_oracle(
                density_grid[row, channel],
                profile.density_capacity[channel],
                profile.knee_fraction[channel],
                profile.match_fraction[channel],
            )
    scalar_oracle_error = float(np.max(np.abs(donor_values - oracle)))
    reference_error = float(
        np.max(np.abs(profile.apply(profile.match_density) - profile.match_density))
    )
    linear_limit_error = float(
        np.max(np.abs(linear_profile.apply(density_grid) - density_grid))
    )
    minimum_donor_derivative = float(np.min(profile.derivative(density_grid)))

    rng = np.random.default_rng(int(config["sampling"]["seed"]))
    low, high = map(float, config["sampling"]["log2_exposure_range"])
    count = int(config["sampling"]["random_rows_per_witness"])
    step = float(config["sampling"]["finite_difference_step"])
    witness_rows = []
    all_differences = []
    maximum_domain_violation = 0.0
    minimum_own = float("inf")
    maximum_cross = -float("inf")
    maximum_jacobian_error = 0.0
    maximum_partition_error = 0.0
    symmetric_neutral_spread = 0.0
    for payload in config["witnesses"]:
        operator = _operator(payload)
        exposure = rng.uniform(low, high, size=(count, 3))
        candidate = apply_langmuir_interimage_development(operator, exposure, profile)
        linear = operator.apply_log2_exposure(exposure)
        difference = np.linalg.norm(candidate - linear, axis=-1)
        all_differences.extend(difference.tolist())
        minimum = np.asarray(operator.density_min)
        maximum = np.asarray(operator.density_max)
        maximum_domain_violation = max(
            maximum_domain_violation,
            float(np.max(np.maximum(minimum - candidate, candidate - maximum))),
            0.0,
        )
        jacobian = langmuir_interimage_jacobian(operator, exposure, profile)
        minimum_own = min(
            minimum_own,
            min(float(np.min(jacobian[:, channel, channel])) for channel in range(3)),
        )
        maximum_cross = max(
            maximum_cross,
            max(
                float(np.max(jacobian[:, output, source]))
                for output in range(3)
                for source in range(3)
                if output != source
            ),
        )
        numeric = _numeric_jacobian(operator, exposure[:64], profile, step)
        maximum_jacobian_error = max(
            maximum_jacobian_error,
            float(np.max(np.abs(numeric - jacobian[:64]))),
        )
        for rows in config["sampling"]["row_partitions"]:
            tiled = np.concatenate(
                [
                    apply_langmuir_interimage_development(
                        operator, exposure[start : start + rows], profile
                    )
                    for start in range(0, count, rows)
                ]
            )
            maximum_partition_error = max(
                maximum_partition_error,
                float(np.max(np.abs(tiled - candidate))),
            )
        neutral_spread = None
        if payload["id"] == "symmetric-moderate":
            neutral = np.linspace(
                low,
                high,
                int(config["sampling"]["neutral_axis_rows"]),
            )
            neutral = np.repeat(neutral[:, None], 3, axis=1)
            neutral_output = apply_langmuir_interimage_development(
                operator, neutral, profile
            )
            symmetric_neutral_spread = float(np.max(np.ptp(neutral_output, axis=-1)))
            neutral_spread = symmetric_neutral_spread
        witness_rows.append(
            {
                "id": payload["id"],
                "median_density_difference_vs_linear": float(np.median(difference)),
                "maximum_density_difference_vs_linear": float(np.max(difference)),
                "symmetric_neutral_channel_spread": neutral_spread,
            }
        )

    summary = {
        "maximum_scalar_oracle_error": scalar_oracle_error,
        "maximum_reference_point_error": reference_error,
        "maximum_linear_limit_error": linear_limit_error,
        "minimum_donor_derivative": minimum_donor_derivative,
        "maximum_density_domain_violation": maximum_domain_violation,
        "minimum_own_channel_jacobian": minimum_own,
        "maximum_cross_channel_jacobian": maximum_cross,
        "maximum_finite_difference_jacobian_error": maximum_jacobian_error,
        "maximum_symmetric_neutral_channel_spread": symmetric_neutral_spread,
        "median_density_difference_vs_linear": float(np.median(all_differences)),
        "maximum_partition_error": maximum_partition_error,
    }
    gates = config["automatic_gates"]
    checks = {
        "scalar_oracle": summary["maximum_scalar_oracle_error"]
        <= gates["maximum_scalar_oracle_error"],
        "reference_point": summary["maximum_reference_point_error"]
        <= gates["maximum_reference_point_error"],
        "linear_limit": summary["maximum_linear_limit_error"]
        <= gates["maximum_linear_limit_error"],
        "donor_monotone": summary["minimum_donor_derivative"]
        >= gates["minimum_donor_derivative"],
        "density_domain": summary["maximum_density_domain_violation"]
        <= gates["maximum_density_domain_violation"],
        "own_jacobian": summary["minimum_own_channel_jacobian"]
        >= gates["minimum_own_channel_jacobian"],
        "cross_jacobian": summary["maximum_cross_channel_jacobian"]
        <= gates["maximum_cross_channel_jacobian"],
        "numeric_jacobian": summary["maximum_finite_difference_jacobian_error"]
        <= gates["maximum_finite_difference_jacobian_error"],
        "neutral_axis": summary["maximum_symmetric_neutral_channel_spread"]
        <= gates["maximum_symmetric_neutral_channel_spread"],
        "material_difference": summary["median_density_difference_vs_linear"]
        >= gates["minimum_median_density_difference_vs_linear"],
        "partition": summary["maximum_partition_error"]
        <= gates["maximum_partition_error"],
    }
    decision = (
        "pass-clean-room-primitive" if all(checks.values()) else "close-primitive"
    )
    stable_payload = {
        "summary": summary,
        "gate_checks": checks,
        "decision": decision,
        "witnesses": witness_rows,
    }
    stable = json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        **stable_payload,
        "stable_evidence_id": hashlib.sha256(stable).hexdigest(),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = ["SCHEMA", "evaluate_langmuir_interimage", "load_contract"]
