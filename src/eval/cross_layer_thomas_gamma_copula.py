"""P4HC synthetic validation of a fixed cross-layer Thomas/Gamma copula."""

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
    apply_cross_layer_thomas_gamma_copula,
)
from src.film_physics.manufacturer_characteristic import (
    ManufacturerCharacteristicPrior,
)

SCHEMA = "neuro-film.u6-p4hc-cross-layer-thomas-gamma-copula-contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4HC contract")
    return payload


def _load_parent(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"P4HC parent drift: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    hb = _load_parent(root, parents["p4hb_result"])
    df = _load_parent(root, parents["p4df_result"])
    profile = DensityConditionedThomasProfile.from_dict(
        _load_parent(root, parents["p4bw_bundle"])
    )
    prior = ManufacturerCharacteristicPrior.from_dict(
        _load_parent(root, parents["p2q_bundle"])["prior"]
    )
    candidate = contract["candidate"]
    if (
        hb.get("decision") != parents["p4hb_result"]["required_decision"]
        or df.get("decision") != parents["p4df_result"]["required_decision"]
        or profile.identity() != parents["p4bw_bundle"]["required_profile_id"]
        or prior.identity() != parents["p2q_bundle"]["required_prior_id"]
        or candidate["amplitude_multiplier"] != 1.0
        or candidate["fitting_allowed"] is not False
    ):
        raise ValueError("P4HC frozen identity drift")

    shape = tuple(int(value) for value in candidate["field_shape"])
    seeds = tuple(int(value) for value in candidate["layer_field_seeds"])
    correlation = np.asarray(candidate["correlation_matrix"], dtype=np.float64)
    block_height = int(candidate["canonical_receipt_row_block_height"])
    rows: list[dict[str, Any]] = []
    moments: list[dict[str, float]] = []
    correlation_errors: list[float] = []
    neighbor_drifts: list[float] = []
    for level in candidate["channel_levels"]:
        for ratio_values in candidate["rgb_ratios"]:
            pixel = float(level) * np.asarray(ratio_values, dtype=np.float64)
            base = np.broadcast_to(pixel, (*shape, 3)).copy()
            output, diagnostics = apply_cross_layer_thomas_gamma_copula(
                base,
                profile=profile,
                prior=prior,
                layer_seeds=seeds,
                correlation_matrix=correlation,
                canonical_receipt_row_block_height=block_height,
            )
            repeated, repeated_diagnostics = apply_cross_layer_thomas_gamma_copula(
                base,
                profile=profile,
                prior=prior,
                layer_seeds=seeds,
                correlation_matrix=correlation,
                canonical_receipt_row_block_height=block_height,
            )
            empirical = np.asarray(
                diagnostics["empirical_copula_correlation"], dtype=np.float64
            )
            correlation_errors.append(float(np.max(np.abs(empirical - correlation))))
            independent_neighbors = np.asarray(
                diagnostics["independent_neighbor_correlation"], dtype=np.float64
            )
            correlated_neighbors = np.asarray(
                diagnostics["correlated_neighbor_correlation"], dtype=np.float64
            )
            neighbor_drifts.append(
                float(np.max(np.abs(independent_neighbors - correlated_neighbors)))
            )
            for index, channel in enumerate(("red", "green", "blue")):
                value = float(pixel[index])
                lower, upper = prior.curves[index].domain
                exposure = lower + value * (upper - lower)
                sigma_d = float(
                    profile.amplitude_profile.evaluate_channel(
                        prior, channel, np.asarray([exposure])
                    )[0]
                )
                sigma = sigma_d * 4.0 * value * (1.0 - value)
                available = -np.log10(value) if value > 0.0 else np.inf
                if available > np.finfo(np.float64).eps and sigma > 0.0:
                    delta = -np.log10(output[..., index].astype(np.float64) / value)
                    moments.append(
                        {
                            "mean": abs(float(np.mean(delta))) / sigma,
                            "variance": abs(
                                float(np.var(delta)) / (sigma * sigma) - 1.0
                            ),
                            "tail": float(np.quantile(np.abs(delta), 0.999)) / sigma,
                            "minimum_density": float(np.min(available + delta)),
                        }
                    )
            rows.append(
                {
                    "level": level,
                    "rgb_ratio": ratio_values,
                    "output_sha256": hashlib.sha256(
                        memoryview(output).cast("B")
                    ).hexdigest(),
                    "repeat_exact": bool(np.array_equal(output, repeated)),
                    "diagnostics_exact": diagnostics == repeated_diagnostics,
                    "copula_correlation_absolute_error": correlation_errors[-1],
                    "neighbor_correlation_absolute_drift": neighbor_drifts[-1],
                    "minimum_output": float(np.min(output)),
                    "maximum_output": float(np.max(output)),
                    "minimum_developed_density": diagnostics[
                        "minimum_developed_density"
                    ],
                    "support_degenerate_residual_absolute": diagnostics[
                        "support_degenerate_residual_absolute"
                    ],
                    "finite": bool(np.all(np.isfinite(output))),
                    "hard_clipping_used": diagnostics["hard_clipping_used"] != 0.0,
                    "limited_fraction": diagnostics["limited_fraction"],
                }
            )

    gates = contract["automatic_gates"]
    checks = {
        "mean": max(item["mean"] for item in moments)
        <= gates["maximum_non_degenerate_mean_absolute_in_target_sigma"],
        "variance": max(item["variance"] for item in moments)
        <= gates["maximum_non_degenerate_variance_relative_error"],
        "tail": max(item["tail"] for item in moments)
        <= gates["maximum_non_degenerate_p999_absolute_in_target_sigma"],
        "copula_correlation": max(correlation_errors)
        <= gates["maximum_copula_correlation_absolute_error"],
        "spatial_topology": max(neighbor_drifts)
        <= gates["maximum_marginal_neighbor_correlation_absolute_drift"],
        "density_support": min(item["minimum_density"] for item in moments)
        >= gates["minimum_developed_density"],
        "unit_transmittance": min(row["minimum_output"] for row in rows)
        >= gates["minimum_transmittance"]
        and max(row["maximum_output"] for row in rows)
        <= gates["maximum_transmittance"],
        "degenerate_identity": max(
            row["support_degenerate_residual_absolute"] for row in rows
        )
        <= gates["maximum_support_degenerate_residual_absolute"],
        "repeat_identity": all(
            row["repeat_exact"] and row["diagnostics_exact"] for row in rows
        ),
        "finite_without_clipping_or_limiting": all(row["finite"] for row in rows)
        and not any(row["hard_clipping_used"] for row in rows)
        and not any(row["limited_fraction"] != 0.0 for row in rows),
    }
    stable = {
        "contract_sha256": hashlib.sha256(
            json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "profile_id": profile.identity(),
        "prior_id": prior.identity(),
        "correlation_profile_id": candidate["correlation_profile_id"],
        "rows": rows,
        "maximum_mean_absolute_in_target_sigma": max(item["mean"] for item in moments),
        "maximum_variance_relative_error": max(item["variance"] for item in moments),
        "maximum_p999_absolute_in_target_sigma": max(item["tail"] for item in moments),
        "maximum_copula_correlation_absolute_error": max(correlation_errors),
        "maximum_neighbor_correlation_absolute_drift": max(neighbor_drifts),
        "minimum_developed_density": min(item["minimum_density"] for item in moments),
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
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


__all__ = ["evaluate", "load_contract"]
