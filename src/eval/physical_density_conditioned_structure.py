"""Frozen U6.P4D density-conditioned material-structure evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.film_physics.density_conditioned_structure import (
    DensityConditionedLayerProfile,
    render_density_conditioned_structure,
    render_density_conditioned_structure_region,
)


SCHEMA = "neuro_film.u6_p4d_density_conditioned_structure_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4d_density_conditioned_structure_report.v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("U6.P4D contract hash mismatch")
    contract = json.loads(path.read_text(encoding="utf-8"))
    model = contract["model"]
    if (
        contract.get("schema") != SCHEMA
        or contract["training_allowed"]
        or contract["photograph_access_allowed"]
        or contract["production_integration_allowed"]
        or model["display_rgb_noise_allowed"]
        or model["signed_density_allowed"]
        or model["per_image_normalization_allowed"]
        or model["hard_clipping_allowed"]
    ):
        raise ValueError("unsupported U6.P4D contract")
    return contract


def profiles_from_contract(
    contract: dict[str, Any], *, seed_offset: int = 0
) -> tuple[DensityConditionedLayerProfile, ...]:
    maximum = float(
        contract["synthetic_evaluation"]["maximum_input_density"]
    )
    return tuple(
        DensityConditionedLayerProfile(
            layer_id=str(row["layer_id"]),
            grain_optical_density=float(row["grain_optical_density"]),
            correlation_sigma_pixels=float(
                row["correlation_sigma_pixels"]
            ),
            seed=int(row["seed"]) + int(seed_offset),
            maximum_target_density=maximum,
        )
        for row in contract["profiles"]
    )


def _flat_metrics(
    contract: dict[str, Any],
    profiles: tuple[DensityConditionedLayerProfile, ...],
) -> list[dict[str, Any]]:
    spec = contract["synthetic_evaluation"]
    height, width = (int(value) for value in spec["shape"])
    border = int(spec["interior_border_pixels"])
    rows: list[dict[str, Any]] = []
    for level in spec["flat_density_levels"]:
        target = np.full(
            (height, width, len(profiles)), float(level), dtype=np.float64
        )
        result = render_density_conditioned_structure(target, profiles)
        interior = result.density[border:-border, border:-border]
        rows.append(
            {
                "target_density": float(level),
                "channel_mean_density": [
                    float(np.mean(interior[..., channel], dtype=np.float64))
                    for channel in range(len(profiles))
                ],
                "channel_variance_density": [
                    float(np.var(interior[..., channel], dtype=np.float64))
                    for channel in range(len(profiles))
                ],
                "minimum_density": float(np.min(result.density)),
                "maximum_transmittance": float(
                    np.max(result.transmittance)
                ),
                "density_sha256": hashlib.sha256(
                    result.density.tobytes(order="C")
                ).hexdigest(),
                "transmittance_sha256": hashlib.sha256(
                    result.transmittance.tobytes(order="C")
                ).hexdigest(),
            }
        )
    return rows


def _ramp_metrics(
    contract: dict[str, Any],
    profiles: tuple[DensityConditionedLayerProfile, ...],
) -> tuple[dict[str, Any], bool, bool]:
    spec = contract["synthetic_evaluation"]
    height, width = (int(value) for value in spec["shape"])
    maximum = float(spec["maximum_input_density"])
    ramp = np.linspace(0.0, maximum, width, dtype=np.float64)
    target = np.broadcast_to(
        ramp[None, :, None], (height, width, len(profiles))
    ).copy()
    full = render_density_conditioned_structure(target, profiles)
    repeat = render_density_conditioned_structure(target, profiles)
    repeat_exact = bool(
        np.array_equal(full.density, repeat.density)
        and np.array_equal(full.transmittance, repeat.transmittance)
    )
    assembled_density = np.empty_like(full.density)
    assembled_transmittance = np.empty_like(full.transmittance)
    for rows in contract["synthetic_evaluation"]["row_partitions"]:
        assembled_density.fill(np.nan)
        assembled_transmittance.fill(np.nan)
        for y0 in range(0, height, int(rows)):
            y1 = min(height, y0 + int(rows))
            region = render_density_conditioned_structure_region(
                target,
                profiles,
                origin_yx=(y0, 0),
                shape=(y1 - y0, width),
            )
            assembled_density[y0:y1] = region.density
            assembled_transmittance[y0:y1] = region.transmittance
        if not (
            np.array_equal(full.density, assembled_density)
            and np.array_equal(full.transmittance, assembled_transmittance)
        ):
            partition_exact = False
            break
    else:
        partition_exact = True
    border = int(spec["interior_border_pixels"])
    interior = full.density[border:-border, :, :]
    column_means = np.mean(interior, axis=0, dtype=np.float64)
    correlations = [
        float(np.corrcoef(ramp, column_means[:, channel])[0, 1])
        for channel in range(len(profiles))
    ]
    mean_absolute_errors = [
        float(np.mean(np.abs(column_means[:, channel] - ramp)))
        for channel in range(len(profiles))
    ]
    return (
        {
            "minimum_target_output_correlation": min(correlations),
            "maximum_column_mean_absolute_error": max(
                mean_absolute_errors
            ),
            "channel_target_output_correlations": correlations,
            "channel_column_mean_absolute_errors": mean_absolute_errors,
            "density_sha256": hashlib.sha256(
                full.density.tobytes(order="C")
            ).hexdigest(),
            "transmittance_sha256": hashlib.sha256(
                full.transmittance.tobytes(order="C")
            ).hexdigest(),
        },
        repeat_exact,
        partition_exact,
    )


def evaluate_density_conditioned_structure(
    contract: dict[str, Any],
) -> dict[str, Any]:
    split = contract["synthetic_evaluation"]
    development_profiles = profiles_from_contract(
        contract, seed_offset=int(split["development_seed_offset"])
    )
    confirmation_profiles = profiles_from_contract(
        contract, seed_offset=int(split["confirmation_seed_offset"])
    )
    development_flat = _flat_metrics(contract, development_profiles)
    confirmation_flat = _flat_metrics(contract, confirmation_profiles)
    development_ramp, development_repeat, development_partition = (
        _ramp_metrics(contract, development_profiles)
    )
    confirmation_ramp, confirmation_repeat, confirmation_partition = (
        _ramp_metrics(contract, confirmation_profiles)
    )
    gates = contract["automatic_gates"]
    levels = [float(value) for value in split["flat_density_levels"]]

    def flat_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
        mean_errors = [
            abs(mean - float(row["target_density"]))
            for row in rows
            for mean in row["channel_mean_density"]
        ]
        minimum_step = min(
            rows[index + 1]["channel_mean_density"][channel]
            - rows[index]["channel_mean_density"][channel]
            for index in range(len(rows) - 1)
            for channel in range(len(development_profiles))
        )
        low_index = levels.index(0.3)
        high_index = levels.index(1.2)
        variance_ratios = [
            rows[high_index]["channel_variance_density"][channel]
            / rows[low_index]["channel_variance_density"][channel]
            for channel in range(len(development_profiles))
        ]
        return {
            "maximum_mean_absolute_error": max(mean_errors),
            "minimum_mean_step": float(minimum_step),
            "minimum_high_to_low_variance_ratio": min(variance_ratios),
            "minimum_density": min(row["minimum_density"] for row in rows),
            "maximum_transmittance": max(
                row["maximum_transmittance"] for row in rows
            ),
        }

    development_summary = flat_summary(development_flat)
    confirmation_summary = flat_summary(confirmation_flat)
    checks = {
        "flat_mean": max(
            development_summary["maximum_mean_absolute_error"],
            confirmation_summary["maximum_mean_absolute_error"],
        )
        <= float(gates["maximum_flat_mean_absolute_error"]),
        "flat_monotonic": min(
            development_summary["minimum_mean_step"],
            confirmation_summary["minimum_mean_step"],
        )
        >= float(gates["minimum_flat_mean_step"]),
        "density_conditioned_variance": min(
            development_summary["minimum_high_to_low_variance_ratio"],
            confirmation_summary["minimum_high_to_low_variance_ratio"],
        )
        >= float(gates["minimum_high_to_low_variance_ratio"]),
        "ramp_correlation": min(
            development_ramp["minimum_target_output_correlation"],
            confirmation_ramp["minimum_target_output_correlation"],
        )
        >= float(gates["minimum_ramp_target_output_correlation"]),
        "ramp_mean": max(
            development_ramp["maximum_column_mean_absolute_error"],
            confirmation_ramp["maximum_column_mean_absolute_error"],
        )
        <= float(gates["maximum_ramp_mean_absolute_error"]),
        "physical_domain": min(
            development_summary["minimum_density"],
            confirmation_summary["minimum_density"],
        )
        >= float(gates["minimum_density"])
        and max(
            development_summary["maximum_transmittance"],
            confirmation_summary["maximum_transmittance"],
        )
        <= float(gates["maximum_transmittance"]),
        "zero_identity": all(
            all(mean == 0.0 for mean in rows[0]["channel_mean_density"])
            and rows[0]["maximum_transmittance"] == 1.0
            for rows in (development_flat, confirmation_flat)
        ),
        "repeat_exact": development_repeat and confirmation_repeat,
        "row_partition_exact": (
            development_partition and confirmation_partition
        ),
    }
    return {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "profiles": [profile.__dict__ for profile in development_profiles],
        "development": {
            "flat": development_flat,
            "flat_summary": development_summary,
            "ramp": development_ramp,
        },
        "confirmation": {
            "flat": confirmation_flat,
            "flat_summary": confirmation_summary,
            "ramp": confirmation_ramp,
        },
        "checks": checks,
        "automatic_pass": all(checks.values()),
        "decision": (
            contract["branch_rule"]["pass"]
            if all(checks.values())
            else contract["branch_rule"]["fail"]
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_density_conditioned_structure",
    "load_contract",
    "profiles_from_contract",
]
