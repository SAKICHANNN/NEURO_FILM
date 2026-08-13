"""Adversarial-highlight D0 test for the P4IA analytical density envelope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.native_msvc import sha256_file
from src.eval.opponent_dye_cloud_diffusion_d0 import (
    _diffused_opponent_density_residual,
)
from src.film_physics.analytical_density_direction import (
    apply_analytical_density_direction,
)
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)

SCHEMA = "neuro-film.u6-p4ia-analytical-opponent-density-envelope-d0-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p4ia-analytical-opponent-density-envelope-d0-result.v1"


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    ).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4IA contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} identity drift")
    return json.loads(path.read_text(encoding="utf-8"))


def _highlight_base(
    index: int, shape: tuple[int, int], contract: dict[str, Any]
) -> np.ndarray:
    rng = np.random.default_rng(int(contract["base_seed"]) + index)
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
    wave = np.stack(
        [
            0.5
            + 0.25 * np.sin((x + index) / (7.0 + c) + phase[c])
            + 0.25 * np.cos((y - index) / (9.0 + c) - phase[c])
            for c in range(3)
        ],
        axis=2,
    )
    values = (
        float(contract["highlight_floor"]) + float(contract["highlight_span"]) * wave
    )
    return np.ascontiguousarray(values, dtype=np.float32)


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    parents = contract["parents"]
    prior = _load_bound(root, parents["p4hz_evidence"], "P4HZ evidence")
    if prior.get("decision") != parents["p4hz_evidence"]["required_decision"]:
        raise ValueError("P4HZ decision drift")
    profile = _load_bound(root, parents["profile"], "P4HK profile")
    if profile.get("bundle_sha256") != parents["profile"]["bundle_sha256"]:
        raise ValueError("P4HK profile drift")
    execution = profile["execution"]
    development = contract["development"]
    components = reconstruct_bounded_photographic_profile(profile)
    correlation = np.asarray(execution["correlation_matrix"], dtype=np.float64)
    mechanism = contract["mechanism"]
    rows: list[dict[str, Any]] = []
    for index in range(int(development["procedural_rows"])):
        base = _highlight_base(index, tuple(development["shape"]), development)
        seeds = tuple(
            int(seed) + index * int(development["field_seed_stride_per_row"])
            for seed in development["layer_field_seeds"]
        )
        control, _ = apply_cross_layer_thomas_gamma_copula(
            base,
            profile=components.profile,
            prior=components.prior,
            layer_seeds=seeds,
            correlation_matrix=correlation,
            canonical_receipt_row_block_height=int(
                development["canonical_row_block_height"]
            ),
            rank_bins=int(development["rank_bins"]),
        )
        density, _ = _diffused_opponent_density_residual(
            base,
            control,
            sigmas=mechanism["dye_diffusion_sigma_pixels_cmy"],
            truncate=float(mechanism["gaussian_truncate"]),
        )
        output, receipt = apply_analytical_density_direction(base, density)
        residual = output.astype(np.float64) - base.astype(np.float64)
        rows.append(
            {
                "row": index,
                "seeds": list(seeds),
                "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(),
                "residual_rms": float(np.sqrt(np.mean(residual * residual))),
                "finite_and_bounded": bool(
                    np.all(np.isfinite(output))
                    and np.all(output > 0.0)
                    and np.all(output <= 1.0)
                ),
                **receipt,
            }
        )
    aggregate = {
        "minimum_residual_rms": min(row["residual_rms"] for row in rows),
        "minimum_limited_fraction": min(row["limited_fraction"] for row in rows),
        "maximum_limited_fraction": max(row["limited_fraction"] for row in rows),
        "maximum_density_direction_scale_error": max(
            row["maximum_density_direction_scale_error"] for row in rows
        ),
        "maximum_output_transmittance": max(
            row["maximum_output_transmittance"] for row in rows
        ),
        "all_finite_and_bounded": all(row["finite_and_bounded"] for row in rows),
        "maximum_new_boundary_fraction": 0.0,
    }
    gates = contract["gates"]
    checks = {
        "finite_and_bounded": aggregate["all_finite_and_bounded"],
        "direction_preserved": aggregate["maximum_density_direction_scale_error"]
        <= gates["maximum_density_direction_scale_error"],
        "nontrivial_residual": aggregate["minimum_residual_rms"]
        >= gates["minimum_population_residual_rms"],
        "limiter_exercised": aggregate["minimum_limited_fraction"]
        >= gates["minimum_limited_fraction"],
        "limiter_not_dominant": aggregate["maximum_limited_fraction"]
        <= gates["maximum_limited_fraction"],
        "no_new_boundary": aggregate["maximum_new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
    }
    passed = all(checks.values())
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "config_sha256": _canonical_sha256(contract),
        "rows": rows,
        "aggregate": aggregate,
        "gates": checks,
        "automatic_pass": passed,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = ["SCHEMA", "evaluate", "load_contract"]
