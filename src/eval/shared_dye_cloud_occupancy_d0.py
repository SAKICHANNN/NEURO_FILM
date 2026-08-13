"""Procedural D0 test of a shared dye-cloud occupancy constraint."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.layer_gamma_photographic_development import (
    _high_frequency_chroma_p999,
)
from src.eval.native_msvc import sha256_file
from src.eval.neutral_base_photographic_ablation import _new_boundary_fraction
from src.film_physics.bounded_photographic_profile import (
    reconstruct_bounded_photographic_profile,
)
from src.film_physics.independent_density_nps import (
    apply_cross_layer_thomas_gamma_copula,
)

SCHEMA = "neuro-film.u6-p4hw-shared-dye-cloud-occupancy-d0-contract.v1"
RESULT_SCHEMA = "neuro-film.u6-p4hw-shared-dye-cloud-occupancy-d0-result.v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4HW contract")
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
    development = contract["development"]
    control = np.asarray(
        contract["mechanisms"]["control"]["correlation_matrix"], dtype=np.float64
    )
    candidate = np.asarray(
        contract["mechanisms"]["candidate"]["correlation_matrix"], dtype=np.float64
    )
    gates = contract["gates"]
    expected_gates = {
        "maximum_candidate_to_control_high_frequency_chroma_p999_ratio": 0.8,
        "minimum_candidate_to_control_luma_high_frequency_rms_ratio": 0.75,
        "maximum_candidate_to_control_luma_high_frequency_rms_ratio": 1.25,
        "minimum_candidate_to_control_total_residual_rms_ratio": 0.9,
        "maximum_candidate_to_control_total_residual_rms_ratio": 1.1,
        "maximum_per_channel_residual_rms_relative_error": 0.03,
        "minimum_empirical_cross_layer_correlation_gain": 0.3,
        "maximum_new_boundary_fraction": 0.0,
        "require_all_rows_finite_and_bounded": True,
        "require_two_exact_process_replays": True,
    }
    if (
        development != {
            "procedural_rows": 12,
            "shape": [129, 131],
            "base_seed": 74021,
            "layer_field_seeds": [320260812, 320261821, 320262830],
            "field_seed_stride_per_row": 1009,
            "rank_bins": 65536,
            "canonical_row_block_height": 128,
            "photographic_pixels_allowed": False,
            "cohort_fitting_allowed": False,
        }
        or control.shape != (3, 3)
        or candidate.shape != (3, 3)
        or not np.array_equal(candidate, candidate.T)
        or not np.array_equal(np.diag(candidate), np.ones(3))
        or np.min(np.linalg.eigvalsh(candidate)) <= 0.0
        or float(contract["mechanisms"]["candidate"]["shared_factor_fraction"])
        != 0.85
        or gates != expected_gates
    ):
        raise ValueError("U6.P4HW frozen mechanism drift")


def _procedural_base(index: int, shape: tuple[int, int], seed: int) -> np.ndarray:
    y, x = np.mgrid[: shape[0], : shape[1]]
    xn = x / max(shape[1] - 1, 1)
    yn = y / max(shape[0] - 1, 1)
    phase = np.random.default_rng(seed + index).uniform(0.0, 2.0 * np.pi, size=6)
    channels = []
    for channel in range(3):
        value = (
            0.46
            + 0.18 * np.sin((1.0 + index % 4) * np.pi * xn + phase[channel])
            + 0.14 * np.cos((1.0 + (index + channel) % 3) * np.pi * yn + phase[channel + 3])
            + 0.08 * np.sin(2.0 * np.pi * (xn + yn) + 0.7 * channel)
        )
        channels.append(value)
    return np.ascontiguousarray(np.clip(np.stack(channels, axis=-1), 0.08, 0.92))


def _hf_luma_rms(residual: np.ndarray) -> float:
    from src.filmfx.fast_blur import gaussian_filter_safe

    luma = np.mean(np.asarray(residual, dtype=np.float64), axis=2)
    high = luma - gaussian_filter_safe(luma, sigma=(1.2, 1.2))
    return float(np.sqrt(np.mean(high * high)))


def _channel_rms(residual: np.ndarray) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float64)
    return np.sqrt(np.mean(values * values, axis=(0, 1)))


def evaluate(contract: dict[str, Any], *, root: Path) -> dict[str, Any]:
    _validate(contract)
    parents = contract["parents"]
    p7h = _load_bound(root, parents["p7h_evidence"], "P7H evidence")
    if p7h.get("decision") != parents["p7h_evidence"]["required_decision"]:
        raise ValueError("P7H decision drift")
    profile_payload = _load_bound(root, parents["profile"], "P4HK profile")
    if profile_payload.get("bundle_sha256") != parents["profile"]["bundle_sha256"]:
        raise ValueError("P4HK profile bundle drift")
    execution = profile_payload.get("execution")
    if not isinstance(execution, dict) or any(
        execution.get(key) != contract["development"][contract_key]
        for key, contract_key in (
            ("layer_field_seeds", "layer_field_seeds"),
            ("field_seed_stride_per_source", "field_seed_stride_per_row"),
            ("rank_bins", "rank_bins"),
            ("canonical_receipt_row_block_height", "canonical_row_block_height"),
        )
    ):
        raise ValueError("P4HK execution identity drift")
    if execution.get("correlation_matrix") != contract["mechanisms"]["control"][
        "correlation_matrix"
    ]:
        raise ValueError("P4HK control correlation drift")
    components = reconstruct_bounded_photographic_profile(profile_payload)
    development = contract["development"]
    shape = tuple(int(value) for value in development["shape"])
    matrices = {
        name: np.asarray(payload["correlation_matrix"], dtype=np.float64)
        for name, payload in contract["mechanisms"].items()
    }
    rows: list[dict[str, Any]] = []
    for index in range(int(development["procedural_rows"])):
        base = _procedural_base(index, shape, int(development["base_seed"]))
        seeds = tuple(
            int(seed) + index * int(development["field_seed_stride_per_row"])
            for seed in development["layer_field_seeds"]
        )
        rendered: dict[str, tuple[np.ndarray, dict[str, Any]]] = {}
        for name, correlation in matrices.items():
            rendered[name] = apply_cross_layer_thomas_gamma_copula(
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
        control, control_receipt = rendered["control"]
        candidate, candidate_receipt = rendered["candidate"]
        control_residual = control.astype(np.float64) - base
        candidate_residual = candidate.astype(np.float64) - base
        control_rms = _channel_rms(control_residual)
        candidate_rms = _channel_rms(candidate_residual)
        control_corr = np.asarray(
            control_receipt["empirical_copula_correlation"], dtype=np.float64
        )
        candidate_corr = np.asarray(
            candidate_receipt["empirical_copula_correlation"], dtype=np.float64
        )
        off_diagonal = ~np.eye(3, dtype=bool)
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
                "maximum_channel_rms_relative_error": float(
                    np.max(np.abs(candidate_rms - control_rms) / control_rms)
                ),
                "empirical_cross_layer_correlation_gain": float(
                    np.mean(candidate_corr[off_diagonal])
                    - np.mean(control_corr[off_diagonal])
                ),
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
        "maximum_channel_rms_relative_error": max(
            row["maximum_channel_rms_relative_error"] for row in rows
        ),
        "minimum_cross_layer_correlation_gain": min(
            row["empirical_cross_layer_correlation_gain"] for row in rows
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
        "marginal_energy": aggregate["maximum_channel_rms_relative_error"]
        <= gates["maximum_per_channel_residual_rms_relative_error"],
        "correlation_gain": aggregate["minimum_cross_layer_correlation_gain"]
        >= gates["minimum_empirical_cross_layer_correlation_gain"],
        "new_boundaries": aggregate["maximum_new_boundary_fraction"]
        <= gates["maximum_new_boundary_fraction"],
        "finite_and_bounded": aggregate["all_finite_and_bounded"],
    }
    automatic_pass = all(checks.values())
    core = {
        "schema": RESULT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": _canonical_sha256(contract),
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
