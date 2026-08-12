"""P4GY synthetic support and moment validation for Gamma density."""

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
    apply_support_matched_gamma_density,
    compile_conservative_shared_sigma_d,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro-film.u6-p4gy-support-matched-gamma-density-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4GY contract")
    return payload


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    gx_path = root / parents["p4gx_result"]["path"]
    profile_path = root / parents["p4bw_bundle"]["path"]
    prior_path = root / parents["p2q_bundle"]["path"]
    if any(
        sha256_file(path) != binding["sha256"]
        for path, binding in (
            (gx_path, parents["p4gx_result"]),
            (profile_path, parents["p4bw_bundle"]),
            (prior_path, parents["p2q_bundle"]),
        )
    ):
        raise ValueError("P4GY parent drift")
    gx = json.loads(gx_path.read_text(encoding="utf-8"))
    profile = DensityConditionedThomasProfile.from_dict(
        json.loads(profile_path.read_text(encoding="utf-8"))
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        json.loads(prior_path.read_text(encoding="utf-8"))["prior"]
    )
    if (
        gx["decision"] != parents["p4gx_result"]["required_decision"]
        or profile.identity() != parents["p4bw_bundle"]["required_profile_id"]
        or prior.identity() != parents["p2q_bundle"]["required_prior_id"]
    ):
        raise ValueError("P4GY parent identity drift")

    candidate = contract["candidate"]
    gates = contract["automatic_gates"]
    if (
        candidate["amplitude_multiplier"] != 1.0
        or candidate["realized_variance_normalization_allowed"] is not False
        or candidate["clipping_allowed"] is not False
    ):
        raise ValueError("P4GY candidate policy drift")
    shape = tuple(int(value) for value in candidate["field_shape"])
    rows: list[dict[str, Any]] = []
    for level in candidate["luminance_levels"]:
        for ratio_values in candidate["rgb_ratios"]:
            ratio = np.asarray(ratio_values, dtype=np.float64)
            base_pixel = float(level) * ratio
            base = np.broadcast_to(base_pixel, (*shape, 3)).copy()
            output, diagnostics = apply_support_matched_gamma_density(
                base,
                profile=profile,
                prior=prior,
                seed=int(candidate["field_seed"]),
            )
            repeated, repeated_diagnostics = apply_support_matched_gamma_density(
                base,
                profile=profile,
                prior=prior,
                seed=int(candidate["field_seed"]),
            )
            sigma_d, _ = compile_conservative_shared_sigma_d(
                base, profile=profile, prior=prior
            )
            luminance = float(
                0.2126 * base_pixel[0]
                + 0.7152 * base_pixel[1]
                + 0.0722 * base_pixel[2]
            )
            target_sigma = float(sigma_d[0, 0] * 4.0 * luminance * (1.0 - luminance))
            maximum_channel = float(np.max(base_pixel))
            available_density = (
                -np.log10(maximum_channel) if maximum_channel > 0.0 else np.inf
            )
            nondegenerate = (
                available_density > np.finfo(np.float64).eps
                and target_sigma > np.finfo(np.float64).tiny
            )
            if nondegenerate:
                delta = -np.log10(
                    output[..., 0].astype(np.float64) / base_pixel[0]
                )
                mean_in_sigma = abs(float(np.mean(delta))) / target_sigma
                variance_relative_error = abs(
                    float(np.var(delta)) / (target_sigma * target_sigma) - 1.0
                )
                p999_in_sigma = (
                    float(np.quantile(np.abs(delta), 0.999)) / target_sigma
                )
                minimum_developed_density = float(np.min(available_density + delta))
            else:
                mean_in_sigma = 0.0
                variance_relative_error = 0.0
                p999_in_sigma = 0.0
                minimum_developed_density = max(0.0, available_density)
            rows.append(
                {
                    "level": level,
                    "rgb_ratio": ratio_values,
                    "nondegenerate": nondegenerate,
                    "target_sigma_d": target_sigma,
                    "available_density": available_density,
                    "mean_absolute_in_target_sigma": mean_in_sigma,
                    "variance_relative_error": variance_relative_error,
                    "p999_absolute_in_target_sigma": p999_in_sigma,
                    "minimum_developed_density": minimum_developed_density,
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
    active = [row for row in rows if row["nondegenerate"]]
    degenerate = [row for row in rows if not row["nondegenerate"]]
    checks = {
        "mean": max(row["mean_absolute_in_target_sigma"] for row in active)
        <= gates["maximum_non_degenerate_mean_absolute_in_target_sigma"],
        "variance": max(row["variance_relative_error"] for row in active)
        <= gates["maximum_non_degenerate_variance_relative_error"],
        "finite_tail": max(row["p999_absolute_in_target_sigma"] for row in active)
        <= gates["maximum_non_degenerate_p999_absolute_in_target_sigma"],
        "nonnegative_density": min(row["minimum_developed_density"] for row in rows)
        >= gates["maximum_negative_developed_density"],
        "unit_transmittance": min(row["minimum_output"] for row in rows)
        >= gates["minimum_transmittance"]
        and max(row["maximum_output"] for row in rows)
        <= gates["maximum_transmittance"],
        "degenerate_identity": max(
            row["support_degenerate_residual_absolute"] for row in degenerate
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
            row["mean_absolute_in_target_sigma"] for row in active
        ),
        "maximum_variance_relative_error": max(
            row["variance_relative_error"] for row in active
        ),
        "maximum_p999_absolute_in_target_sigma": max(
            row["p999_absolute_in_target_sigma"] for row in active
        ),
        "minimum_developed_density": min(
            row["minimum_developed_density"] for row in rows
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
