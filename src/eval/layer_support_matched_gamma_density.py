"""P4GZ synthetic validation of independent layer Gamma densities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.neutral_base_photographic_ablation import sha256_file
from src.film_physics.density_conditioned_thomas import (
    DensityConditionedThomasProfile,
)
from src.film_physics.independent_density_nps import (
    apply_layer_support_matched_gamma_density,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro-film.u6-p4gz-layer-support-matched-gamma-density-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4GZ contract")
    return payload


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    gy_path = root / parents["p4gy_result"]["path"]
    profile_path = root / parents["p4bw_bundle"]["path"]
    prior_path = root / parents["p2q_bundle"]["path"]
    if any(
        sha256_file(path) != binding["sha256"]
        for path, binding in (
            (gy_path, parents["p4gy_result"]),
            (profile_path, parents["p4bw_bundle"]),
            (prior_path, parents["p2q_bundle"]),
        )
    ):
        raise ValueError("P4GZ parent drift")
    gy = json.loads(gy_path.read_text(encoding="utf-8"))
    profile = DensityConditionedThomasProfile.from_dict(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        json.loads(prior_path.read_text(encoding="utf-8"))["prior"]
    )
    if (
        gy["decision"] != parents["p4gy_result"]["required_decision"]
        or profile.identity() != parents["p4bw_bundle"]["required_profile_id"]
        or prior.identity() != parents["p2q_bundle"]["required_prior_id"]
    ):
        raise ValueError("P4GZ parent identity drift")

    candidate = contract["candidate"]
    gates = contract["automatic_gates"]
    shape = tuple(int(value) for value in candidate["field_shape"])
    seeds = tuple(int(value) for value in candidate["layer_field_seeds"])
    rows: list[dict[str, Any]] = []
    active_moments: list[dict[str, float]] = []
    cross_correlations: list[float] = []
    for level in candidate["channel_levels"]:
        for ratio_values in candidate["rgb_ratios"]:
            ratio = np.asarray(ratio_values, dtype=np.float64)
            base_pixel = float(level) * ratio
            base = np.broadcast_to(base_pixel, (*shape, 3)).copy()
            output, diagnostics = apply_layer_support_matched_gamma_density(
                base, profile=profile, prior=prior, layer_seeds=seeds
            )
            repeated, repeated_diagnostics = apply_layer_support_matched_gamma_density(
                base, profile=profile, prior=prior, layer_seeds=seeds
            )
            channel_rows: list[dict[str, Any]] = []
            deltas: list[np.ndarray | None] = []
            for index, channel in enumerate(("red", "green", "blue")):
                value = float(base_pixel[index])
                lower, upper = prior.curves[index].domain
                exposure = lower + value * (upper - lower)
                sigma_d = float(
                    profile.amplitude_profile.evaluate_channel(
                        prior, channel, np.asarray([exposure])
                    )[0]
                )
                target_sigma = sigma_d * 4.0 * value * (1.0 - value)
                available = -np.log10(value) if value > 0.0 else np.inf
                active = bool(
                    available > np.finfo(np.float64).eps
                    and target_sigma > np.finfo(np.float64).tiny
                )
                if active:
                    delta = -np.log10(
                        output[..., index].astype(np.float64) / value
                    )
                    mean_in_sigma = abs(float(np.mean(delta))) / target_sigma
                    variance_error = abs(
                        float(np.var(delta)) / (target_sigma * target_sigma) - 1.0
                    )
                    p999_in_sigma = (
                        float(np.quantile(np.abs(delta), 0.999)) / target_sigma
                    )
                    minimum_density = float(np.min(available + delta))
                    active_moments.append(
                        {
                            "mean": mean_in_sigma,
                            "variance": variance_error,
                            "tail": p999_in_sigma,
                            "minimum_density": minimum_density,
                        }
                    )
                    deltas.append(delta)
                else:
                    mean_in_sigma = variance_error = p999_in_sigma = 0.0
                    minimum_density = max(0.0, available)
                    deltas.append(None)
                channel_rows.append(
                    {
                        "channel": channel,
                        "active": active,
                        "target_sigma_d": target_sigma,
                        "available_density": available,
                        "mean_absolute_in_target_sigma": mean_in_sigma,
                        "variance_relative_error": variance_error,
                        "p999_absolute_in_target_sigma": p999_in_sigma,
                        "minimum_developed_density": minimum_density,
                    }
                )
            pair_correlations: list[float] = []
            for left in range(3):
                for right in range(left + 1, 3):
                    if deltas[left] is not None and deltas[right] is not None:
                        correlation = abs(
                            float(
                                np.corrcoef(
                                    deltas[left].reshape(-1),
                                    deltas[right].reshape(-1),
                                )[0, 1]
                            )
                        )
                        pair_correlations.append(correlation)
                        cross_correlations.append(correlation)
            rows.append(
                {
                    "level": level,
                    "rgb_ratio": ratio_values,
                    "channels": channel_rows,
                    "maximum_cross_layer_density_correlation_absolute": max(
                        pair_correlations, default=0.0
                    ),
                    "minimum_output": float(np.min(output)),
                    "maximum_output": float(np.max(output)),
                    "support_degenerate_residual_absolute": diagnostics[
                        "support_degenerate_residual_absolute"
                    ],
                    "repeat_exact": bool(np.array_equal(output, repeated)),
                    "diagnostics_exact": diagnostics == repeated_diagnostics,
                    "finite": bool(np.all(np.isfinite(output))),
                    "hard_clipping_used": diagnostics["hard_clipping_used"] != 0.0,
                    "limited_fraction": diagnostics["limited_fraction"],
                    "output_sha256": hashlib.sha256(
                        memoryview(output).cast("B")
                    ).hexdigest(),
                }
            )
    checks = {
        "mean": max(item["mean"] for item in active_moments)
        <= gates["maximum_non_degenerate_mean_absolute_in_target_sigma"],
        "variance": max(item["variance"] for item in active_moments)
        <= gates["maximum_non_degenerate_variance_relative_error"],
        "finite_tail": max(item["tail"] for item in active_moments)
        <= gates["maximum_non_degenerate_p999_absolute_in_target_sigma"],
        "cross_layer_independence": max(cross_correlations)
        <= gates["maximum_cross_layer_density_correlation_absolute"],
        "nonnegative_density": min(
            item["minimum_density"] for item in active_moments
        )
        >= gates["minimum_developed_density"],
        "unit_transmittance": min(row["minimum_output"] for row in rows)
        >= gates["minimum_transmittance"]
        and max(row["maximum_output"] for row in rows)
        <= gates["maximum_transmittance"],
        "degenerate_identity": max(
            row["support_degenerate_residual_absolute"] for row in rows
        )
        <= gates["maximum_support_degenerate_residual_absolute"],
        "repeat_exact": all(
            row["repeat_exact"] and row["diagnostics_exact"] for row in rows
        ),
        "finite": all(row["finite"] for row in rows),
        "no_clipping_or_limiting": not any(row["hard_clipping_used"] for row in rows)
        and not any(row["limited_fraction"] != 0.0 for row in rows),
    }
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "rows": rows,
        "maximum_mean_absolute_in_target_sigma": max(
            item["mean"] for item in active_moments
        ),
        "maximum_variance_relative_error": max(
            item["variance"] for item in active_moments
        ),
        "maximum_p999_absolute_in_target_sigma": max(
            item["tail"] for item in active_moments
        ),
        "maximum_cross_layer_density_correlation_absolute": max(cross_correlations),
        "minimum_developed_density": min(
            item["minimum_density"] for item in active_moments
        ),
        "maximum_transmittance": max(row["maximum_output"] for row in rows),
        "gates": checks,
        "automatic_pass": all(checks.values()),
        "decision": contract["decision_if_pass"]
        if all(checks.values())
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": contract["schema"].replace("contract", "result"),
        "stable": stable,
        "stable_evidence_id": hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode(
                "ascii"
            )
        ).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
