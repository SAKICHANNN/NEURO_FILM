"""Development-only moment-corrected physical LOD evaluation."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.eval.physical_structure_lod import _area_mean, _metrics, _profile
from src.film_physics import (
    CompoundPoissonProfile,
    render_compound_poisson,
    render_compound_poisson_region,
    rescale_compound_poisson_profile,
)


SCHEMA = "neuro_film.u6_p4c2_moment_corrected_lod_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4C2 contract")
    return payload


def _shot_field(profile: CompoundPoissonProfile, shape: tuple[int, int]) -> np.ndarray:
    probe = replace(profile, family="density-shot", baseline=1e-6, scale=1.0)
    return render_compound_poisson(probe, shape).astype(np.float64) - probe.baseline


def _fit_profile(
    reference: np.ndarray,
    naive: CompoundPoissonProfile,
    *,
    pixel_size_factor: int,
    contract: dict[str, Any],
) -> CompoundPoissonProfile:
    shot = _shot_field(naive, reference.shape)
    target_mean = float(np.mean(reference))
    target_variance = float(np.var(reference))
    if naive.family == "density-shot":
        shot_variance = float(np.var(shot))
        if shot_variance <= 0.0:
            raise RuntimeError("development LOD shot field has zero variance")
        scale = math.sqrt(target_variance / shot_variance)
        baseline = target_mean - scale * float(np.mean(shot))
        if baseline < float(contract["density_fit"]["minimum_baseline"]):
            raise RuntimeError("development LOD density fit requires signed baseline")
    else:
        solver = contract["transmittance_fit"]

        def residual(parameters: np.ndarray) -> np.ndarray:
            values = np.exp(-(parameters[0] + parameters[1] * shot))
            return np.asarray(
                [
                    (float(np.mean(values)) - target_mean) / target_mean,
                    (float(np.var(values)) - target_variance) / target_variance,
                ],
                dtype=np.float64,
            )

        result = least_squares(
            residual,
            np.asarray(
                [
                    float(solver["initial_base_scale"][0]),
                    float(solver["initial_base_scale"][1])
                    / (pixel_size_factor * pixel_size_factor),
                ],
                dtype=np.float64,
            ),
            bounds=(
                np.asarray(
                    [
                        float(solver["bounds_base_scale"][0][0]),
                        float(solver["bounds_base_scale"][0][1])
                        / (pixel_size_factor * pixel_size_factor),
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [
                        float(solver["bounds_base_scale"][1][0]),
                        float(solver["bounds_base_scale"][1][1])
                        / (pixel_size_factor * pixel_size_factor),
                    ],
                    dtype=np.float64,
                ),
            ),
            xtol=float(solver["xtol"]),
            ftol=float(solver["ftol"]),
            gtol=float(solver["gtol"]),
            max_nfev=int(solver["max_nfev"]),
        )
        baseline, scale = (float(value) for value in result.x)
    return replace(naive, baseline=baseline, scale=scale)


def _is_domain_valid(
    profile: CompoundPoissonProfile, values: np.ndarray
) -> tuple[bool, int]:
    if profile.family == "density-shot":
        violations = int(np.count_nonzero(values < profile.baseline))
        valid = bool(np.all(np.isfinite(values)) and np.all(values > 0.0))
    else:
        violations = int(
            np.count_nonzero(values > math.exp(-profile.baseline))
        )
        valid = bool(
            np.all(np.isfinite(values))
            and np.all(values > 0.0)
            and np.all(values <= 1.0)
        )
    return valid, violations


def evaluate_moment_corrected_lod(contract: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(int(value) for value in contract["reference_shape"])
    split = contract["split"]
    lags = contract["metrics"]["acf_lags"]
    bands = contract["metrics"]["radial_nps_bands_cycles_per_pixel"]
    compiled: list[dict[str, Any]] = []
    development_rows: list[dict[str, Any]] = []
    confirmatory_rows: list[dict[str, Any]] = []
    repeat_exact: list[bool] = []
    partition_exact: list[bool] = []
    domain_valid: list[bool] = []
    brightening_violations: list[int] = []
    development_eligibility = contract["development_eligibility"]
    for layer, row in enumerate(contract["compiled_base_profiles"]):
        development_base = _profile(
            row, seed=int(split["development_reference_seed"]) + layer
        )
        confirmation_base = replace(
            development_base,
            seed=int(split["confirmatory_reference_seed"]) + layer,
        )
        development_resolved = render_compound_poisson(
            development_base, shape
        )
        confirmation_resolved = render_compound_poisson(
            confirmation_base, shape
        )
        for factor in (int(value) for value in contract["lod_factors"]):
            development_reference = _area_mean(
                development_resolved, factor
            )
            confirmation_reference = _area_mean(
                confirmation_resolved, factor
            )
            naive = rescale_compound_poisson_profile(
                development_base,
                pixel_size_factor=factor,
                seed=(
                    int(split["development_direct_seed"])
                    + layer * 16
                    + factor
                ),
            )
            profile = _fit_profile(
                development_reference,
                naive,
                pixel_size_factor=factor,
                contract=contract,
            )
            try:
                development_candidate = render_compound_poisson(
                    profile, development_reference.shape
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    "moment-corrected LOD left its development domain at "
                    f"layer={layer}, factor={factor}, "
                    f"baseline={profile.baseline}, scale={profile.scale}"
                ) from exc
            development_metrics = _metrics(
                development_reference,
                development_candidate,
                lags=lags,
                bands=bands,
            )
            development_valid, development_brightening = _is_domain_valid(
                profile, development_candidate
            )
            if not (
                development_metrics["mean_abs_error"]
                <= float(development_eligibility["mean_abs_error_max"])
                and development_metrics["variance_ratio"]
                >= float(development_eligibility["variance_ratio_min"])
                and development_metrics["variance_ratio"]
                <= float(development_eligibility["variance_ratio_max"])
                and development_valid
                and development_brightening == 0
            ):
                raise RuntimeError("moment-corrected LOD fails development eligibility")
            confirm_profile = replace(
                profile,
                seed=(
                    int(split["confirmatory_direct_seed"])
                    + layer * 16
                    + factor
                ),
            )
            candidate = render_compound_poisson(
                confirm_profile, confirmation_reference.shape
            )
            split_row = candidate.shape[0] // 2
            pieces = [
                render_compound_poisson_region(
                    confirm_profile,
                    candidate.shape,
                    origin_yx=(0, 0),
                    shape=(split_row, candidate.shape[1]),
                ),
                render_compound_poisson_region(
                    confirm_profile,
                    candidate.shape,
                    origin_yx=(split_row, 0),
                    shape=(candidate.shape[0] - split_row, candidate.shape[1]),
                ),
            ]
            partition_exact.append(
                np.array_equal(np.concatenate(pieces), candidate)
            )
            repeat_exact.append(
                np.array_equal(
                    render_compound_poisson(confirm_profile, candidate.shape),
                    candidate,
                )
            )
            valid, violations = _is_domain_valid(confirm_profile, candidate)
            domain_valid.append(valid)
            brightening_violations.append(violations)
            key = {
                "layer": layer,
                "family": profile.family,
                "lod_factor": factor,
                "output_pitch_um": (
                    float(contract["base_physical_pitch_um"]) * factor
                ),
            }
            compiled.append(
                {
                    **key,
                    "poisson_rate": profile.poisson_rate,
                    "correlation_sigma_pixels": profile.correlation_sigma_pixels,
                    "baseline": profile.baseline,
                    "scale": profile.scale,
                }
            )
            development_rows.append({**key, **development_metrics})
            confirmatory_rows.append(
                {
                    **key,
                    **_metrics(
                        confirmation_reference,
                        candidate,
                        lags=lags,
                        bands=bands,
                    ),
                }
            )
    gates = contract["automatic_gates"]
    decisions = {
        "mean": max(row["mean_abs_error"] for row in confirmatory_rows)
        <= float(gates["mean_abs_error_max"]),
        "variance": min(row["variance_ratio"] for row in confirmatory_rows)
        >= float(gates["variance_ratio_min"])
        and max(row["variance_ratio"] for row in confirmatory_rows)
        <= float(gates["variance_ratio_max"]),
        "acf": max(row["acf_abs_error_max"] for row in confirmatory_rows)
        <= float(gates["acf_abs_error_max"]),
        "nps": max(row["nps_band_relative_error_max"] for row in confirmatory_rows)
        <= float(gates["nps_band_relative_error_max"]),
        "repeat": all(repeat_exact),
        "partition": all(partition_exact),
        "domain": all(domain_valid),
        "brightening": sum(brightening_violations)
        == int(gates["brightening_violation_count"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p4c2_moment_corrected_lod_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "compiled_lod_profiles": compiled,
        "development_metrics": development_rows,
        "confirmatory_metrics": confirmatory_rows,
        "repeat_exact": repeat_exact,
        "partition_exact": partition_exact,
        "domain_valid": domain_valid,
        "brightening_violation_count": sum(brightening_violations),
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
        "negative_control": contract["negative_control"],
    }
    evidence_id = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
    ).hexdigest()
    return {**core, "stable_evidence_id": evidence_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    raw = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()
