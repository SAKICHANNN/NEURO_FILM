"""Procedural D0 test of zero-sum opponent dye-cloud diffusion."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.layer_gamma_photographic_development import _high_frequency_chroma_p999
from src.eval.native_msvc import sha256_file
from src.eval.neutral_base_photographic_ablation import _new_boundary_fraction
from src.eval.shared_dye_cloud_occupancy_d0 import _hf_luma_rms, _procedural_base
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)
from src.filmfx.fast_blur import gaussian_filter_direct

SCHEMA = "neuro-film.u6-p4hx-opponent-dye-cloud-diffusion-d0-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p4hx-opponent-dye-cloud-diffusion-d0-result.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4HX contract")
    return payload


def _load_bound(root: Path, binding: dict[str, Any], label: str) -> dict[str, Any]:
    path = root / binding["path"]
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"{label} identity drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must contain an object")
    return payload


def _validate(contract: dict[str, Any]) -> None:
    expected_development = {
        "procedural_rows": 12,
        "shape": [129, 131],
        "base_seed": 94117,
        "layer_field_seeds": [320260812, 320261821, 320262830],
        "field_seed_stride_per_row": 1009,
        "rank_bins": 65536,
        "canonical_row_block_height": 128,
        "photographic_pixels_allowed": False,
        "cohort_fitting_allowed": False,
    }
    expected_mechanism = {
        "id": "developed-density-common-plus-diffused-zero-sum-opponent-v1",
        "dye_diffusion_sigma_pixels_cmy": [0.72, 1.08, 1.52],
        "gaussian_truncate": 4.0,
        "common_density_definition": "per-pixel arithmetic mean of three developed-density residual layers",
        "opponent_definition": "developed-density residual minus common density",
        "post_diffusion_projection": "subtract per-pixel mean from the three independently diffused opponent layers",
        "hard_clipping_allowed": False,
        "posthoc_limiting_allowed": False,
    }
    expected_gates = {
        "maximum_candidate_to_control_high_frequency_chroma_p999_ratio": 0.8,
        "minimum_candidate_to_control_luma_high_frequency_rms_ratio": 0.9,
        "maximum_candidate_to_control_luma_high_frequency_rms_ratio": 1.1,
        "minimum_candidate_to_control_total_residual_rms_ratio": 0.75,
        "maximum_candidate_to_control_total_residual_rms_ratio": 1.05,
        "maximum_common_density_residual_absolute_error": 1e-12,
        "maximum_new_boundary_fraction": 0.0,
        "require_all_rows_finite_and_bounded": True,
        "require_two_exact_process_replays": True,
    }
    if (
        contract.get("development") != expected_development
        or contract.get("mechanism") != expected_mechanism
        or contract.get("gates") != expected_gates
    ):
        raise ValueError("U6.P4HX frozen mechanism drift")


def _diffuse_opponent_density(
    base: np.ndarray,
    control: np.ndarray,
    *,
    sigmas: list[float],
    truncate: float,
) -> tuple[np.ndarray, float]:
    base64 = np.asarray(base, dtype=np.float64)
    control64 = np.asarray(control, dtype=np.float64)
    if np.any(base64 <= 0.0) or np.any(control64 <= 0.0):
        raise RuntimeError("P4HX requires strictly positive transmittance")
    density_residual = -np.log10(control64 / base64)
    common = np.mean(density_residual, axis=2, keepdims=True)
    opponent = density_residual - common
    diffused = np.stack(
        [
            gaussian_filter_direct(
                opponent[..., channel], float(sigmas[channel]), truncate=truncate
            )
            for channel in range(3)
        ],
        axis=2,
    ).astype(np.float64)
    projected = diffused - np.mean(diffused, axis=2, keepdims=True)
    candidate_density_residual = common + projected
    common_error = float(
        np.max(
            np.abs(np.mean(candidate_density_residual, axis=2, keepdims=True) - common)
        )
    )
    candidate64 = base64 * np.power(10.0, -candidate_density_residual)
    if (
        not np.all(np.isfinite(candidate64))
        or np.any(candidate64 < 0.0)
        or np.any(candidate64 > 1.0)
    ):
        raise RuntimeError("P4HX candidate escaped the unit cube")
    return np.ascontiguousarray(candidate64, dtype=np.float32), common_error


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    _validate(contract)
    parents = contract["parents"]
    p4hw = _load_bound(root, parents["p4hw_evidence"], "P4HW evidence")
    if p4hw.get("decision") != parents["p4hw_evidence"]["required_decision"]:
        raise ValueError("P4HW decision drift")
    p4as = _load_bound(root, parents["p4as_decision"], "P4AS decision")
    if (
        p4as.get("decision") != parents["p4as_decision"]["required_decision"]
        or p4as.get("bundle_id") != parents["p4as_decision"]["required_bundle_id"]
    ):
        raise ValueError("P4AS bundle drift")
    profile_payload = _load_bound(root, parents["profile"], "P4HK profile")
    if profile_payload.get("bundle_sha256") != parents["profile"]["bundle_sha256"]:
        raise ValueError("P4HK profile bundle drift")
    execution = profile_payload.get("execution", {})
    development = contract["development"]
    if any(
        execution.get(key) != development[contract_key]
        for key, contract_key in (
            ("layer_field_seeds", "layer_field_seeds"),
            ("field_seed_stride_per_source", "field_seed_stride_per_row"),
            ("rank_bins", "rank_bins"),
            ("canonical_row_block_height", "canonical_row_block_height"),
        )
    ):
        raise ValueError("P4HK execution identity drift")
    components = reconstruct_bounded_photographic_profile(profile_payload)
    correlation = np.asarray(execution["correlation_matrix"], dtype=np.float64)
    shape = tuple(int(value) for value in development["shape"])
    mechanism = contract["mechanism"]
    rows: list[dict[str, Any]] = []
    for index in range(int(development["procedural_rows"])):
        base = _procedural_base(index, shape, int(development["base_seed"]))
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
        candidate, common_error = _diffuse_opponent_density(
            base,
            control,
            sigmas=mechanism["dye_diffusion_sigma_pixels_cmy"],
            truncate=float(mechanism["gaussian_truncate"]),
        )
        control_residual = control.astype(np.float64) - base
        candidate_residual = candidate.astype(np.float64) - base
        rows.append(
            {
                "row": index,
                "seeds": list(seeds),
                "control_high_frequency_chroma_p999": _high_frequency_chroma_p999(
                    control_residual
                ),
                "candidate_high_frequency_chroma_p999": _high_frequency_chroma_p999(
                    candidate_residual
                ),
                "control_luma_high_frequency_rms": _hf_luma_rms(control_residual),
                "candidate_luma_high_frequency_rms": _hf_luma_rms(candidate_residual),
                "control_total_residual_rms": float(
                    np.sqrt(np.mean(control_residual * control_residual))
                ),
                "candidate_total_residual_rms": float(
                    np.sqrt(np.mean(candidate_residual * candidate_residual))
                ),
                "common_density_residual_absolute_error": common_error,
                "new_boundary_fraction": _new_boundary_fraction(control, candidate),
                "finite_and_bounded": bool(
                    np.all(np.isfinite(candidate))
                    and np.all(candidate >= 0.0)
                    and np.all(candidate <= 1.0)
                ),
                "control_sha256": hashlib.sha256(control.tobytes()).hexdigest(),
                "candidate_sha256": hashlib.sha256(candidate.tobytes()).hexdigest(),
            }
        )
    aggregate = {
        "maximum_chroma_ratio": max(
            row["candidate_high_frequency_chroma_p999"]
            / row["control_high_frequency_chroma_p999"]
            for row in rows
        ),
        "minimum_luma_hf_ratio": min(
            row["candidate_luma_high_frequency_rms"]
            / row["control_luma_high_frequency_rms"]
            for row in rows
        ),
        "maximum_luma_hf_ratio": max(
            row["candidate_luma_high_frequency_rms"]
            / row["control_luma_high_frequency_rms"]
            for row in rows
        ),
        "minimum_total_rms_ratio": min(
            row["candidate_total_residual_rms"] / row["control_total_residual_rms"]
            for row in rows
        ),
        "maximum_total_rms_ratio": max(
            row["candidate_total_residual_rms"] / row["control_total_residual_rms"]
            for row in rows
        ),
        "maximum_common_density_residual_absolute_error": max(
            row["common_density_residual_absolute_error"] for row in rows
        ),
        "maximum_new_boundary_fraction": max(
            row["new_boundary_fraction"] for row in rows
        ),
        "all_finite_and_bounded": all(row["finite_and_bounded"] for row in rows),
    }
    gates = contract["gates"]
    checks = {
        "chroma_reduction": aggregate["maximum_chroma_ratio"]
        <= gates["maximum_candidate_to_control_high_frequency_chroma_p999_ratio"],
        "luma_floor": aggregate["minimum_luma_hf_ratio"]
        >= gates["minimum_candidate_to_control_luma_high_frequency_rms_ratio"],
        "luma_ceiling": aggregate["maximum_luma_hf_ratio"]
        <= gates["maximum_candidate_to_control_luma_high_frequency_rms_ratio"],
        "total_rms_floor": aggregate["minimum_total_rms_ratio"]
        >= gates["minimum_candidate_to_control_total_residual_rms_ratio"],
        "total_rms_ceiling": aggregate["maximum_total_rms_ratio"]
        <= gates["maximum_candidate_to_control_total_residual_rms_ratio"],
        "common_density_preserved": aggregate[
            "maximum_common_density_residual_absolute_error"
        ]
        <= gates["maximum_common_density_residual_absolute_error"],
        "new_boundaries": aggregate["maximum_new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "finite_and_bounded": aggregate["all_finite_and_bounded"],
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _canonical_sha256(contract),
        "p4as_bundle_id": p4as["bundle_id"],
        "profile_bundle_sha256": profile_payload["bundle_sha256"],
        "rows": rows,
        "aggregate": aggregate,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": contract[
            "decision_if_pass" if automatic_pass else "decision_if_fail"
        ],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": _canonical_sha256(core)}


__all__ = ["SCHEMA", "evaluate", "load_contract"]
