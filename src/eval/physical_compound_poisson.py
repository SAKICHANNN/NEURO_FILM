"""Development/confirmatory evaluator for U6.P4B compound-Poisson structure."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.eval.physical_developed_structure import load_contract as load_parent
from src.eval.physical_structure_compiler import _acf, _nps_bands, _reference
from src.film_physics import (
    CompoundPoissonProfile,
    render_compound_poisson,
    render_compound_poisson_region,
)


SCHEMA = "neuro_film.u6_p4b_compound_poisson_compiler_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4B contract")
    return payload


def _fit_profile(
    reference: np.ndarray,
    *,
    family: str,
    rate: float,
    sigma: float,
    seed: int,
    contract: dict[str, Any],
) -> CompoundPoissonProfile | None:
    probe = CompoundPoissonProfile(
        family=family,
        poisson_rate=rate,
        correlation_sigma_pixels=sigma,
        baseline=(
            float(contract["candidate"]["minimum_density_baseline"])
            if family == "density-shot"
            else 0.0
        ),
        scale=1.0,
        seed=seed,
        truncate=float(contract["candidate"]["gaussian_truncate"]),
    )
    raw = render_compound_poisson(probe, reference.shape).astype(np.float64)
    if family == "density-shot":
        shot = (
            raw - probe.baseline
        )
        if float(np.var(shot)) <= 0.0:
            return None
        scale = float(np.sqrt(float(np.var(reference)) / float(np.var(shot))))
        baseline = float(np.mean(reference)) - scale * float(np.mean(shot))
        if baseline < float(contract["candidate"]["minimum_density_baseline"]):
            return None
    else:
        # Reuse the exact renderer to obtain the filtered shot field:
        unit = replace(probe, scale=1.0, baseline=0.0)
        shot = -np.log(
            render_compound_poisson(unit, reference.shape).astype(np.float64)
        )
        solver = contract["candidate"]["bw_solver"]
        target_mean = float(np.mean(reference))
        target_variance = float(np.var(reference))

        def residual(parameters: np.ndarray) -> np.ndarray:
            values = np.exp(-(parameters[0] + parameters[1] * shot))
            return np.asarray(
                [
                    (float(np.mean(values)) - target_mean) / target_mean,
                    (float(np.var(values)) - target_variance) / target_variance,
                ],
                dtype=np.float64,
            )

        fit = least_squares(
            residual,
            np.asarray(solver["initial_base_scale"], dtype=np.float64),
            bounds=tuple(
                np.asarray(value, dtype=np.float64)
                for value in solver["bounds_base_scale"]
            ),
            xtol=float(solver["xtol"]),
            ftol=float(solver["ftol"]),
            gtol=float(solver["gtol"]),
            max_nfev=int(solver["max_nfev"]),
        )
        baseline, scale = (float(value) for value in fit.x)
    return replace(probe, baseline=baseline, scale=scale)


def _profile_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    lags: list[list[int]],
    bands: list[list[float]],
) -> dict[str, Any]:
    reference_acf = _acf(reference, lags)
    candidate_acf = _acf(candidate, lags)
    reference_nps = _nps_bands(reference, bands)
    candidate_nps = _nps_bands(candidate, bands)
    return {
        "mean_abs_error": abs(float(np.mean(candidate)) - float(np.mean(reference))),
        "variance_ratio": float(np.var(candidate)) / float(np.var(reference)),
        "acf_abs_error_max": float(
            np.max(
                np.abs(np.asarray(candidate_acf) - np.asarray(reference_acf))
            )
        ),
        "acf_abs_error_mean": float(
            np.mean(
                np.abs(np.asarray(candidate_acf) - np.asarray(reference_acf))
            )
        ),
        "nps_band_relative_error_max": float(
            np.max(
                np.abs(np.asarray(candidate_nps) - np.asarray(reference_nps))
                / np.maximum(np.asarray(reference_nps), 1e-12)
            )
        ),
        "reference_acf": reference_acf,
        "candidate_acf": candidate_acf,
        "reference_nps_bands": reference_nps,
        "candidate_nps_bands": candidate_nps,
    }


def _compile_layer(
    reference: np.ndarray,
    *,
    family: str,
    seed: int,
    contract: dict[str, Any],
) -> tuple[CompoundPoissonProfile, dict[str, Any]]:
    candidate_contract = contract["candidate"]
    eligibility = candidate_contract["development_candidate_eligibility"]
    lags = contract["metrics"]["acf_lags"]
    bands = contract["metrics"]["radial_nps_bands_cycles_per_pixel"]
    eligible: list[
        tuple[
            tuple[float, float, float, float],
            CompoundPoissonProfile,
            dict[str, Any],
        ]
    ] = []
    for rate in candidate_contract["poisson_rate_grid"]:
        for sigma in candidate_contract["correlation_sigma_grid_pixels"]:
            profile = _fit_profile(
                reference,
                family=family,
                rate=float(rate),
                sigma=float(sigma),
                seed=seed,
                contract=contract,
            )
            if profile is None:
                continue
            try:
                candidate = render_compound_poisson(profile, reference.shape)
            except RuntimeError:
                # A development-grid solution that underflows or otherwise
                # leaves its physical domain is ineligible, never clipped.
                continue
            metrics = _profile_metrics(reference, candidate, lags, bands)
            domain = bool(
                np.all(np.isfinite(candidate))
                and np.all(candidate > 0.0)
                and (family == "density-shot" or np.all(candidate <= 1.0))
            )
            if not (
                metrics["mean_abs_error"]
                <= float(eligibility["mean_abs_error_max"])
                and metrics["variance_ratio"]
                >= float(eligibility["variance_ratio_min"])
                and metrics["variance_ratio"]
                <= float(eligibility["variance_ratio_max"])
                and domain
            ):
                continue
            key = (
                metrics["acf_abs_error_max"],
                metrics["acf_abs_error_mean"],
                profile.poisson_rate,
                profile.correlation_sigma_pixels,
            )
            eligible.append((key, profile, metrics))
    if not eligible:
        raise RuntimeError("no development-eligible compound-Poisson profile")
    _, profile, metrics = min(eligible, key=lambda item: item[0])
    return profile, metrics


def evaluate_compound_poisson(
    parent: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    split = contract["split"]
    development_colour, development_bw = _reference(
        parent, int(split["development_reference_seed"])
    )
    confirmation_colour, confirmation_bw = _reference(
        parent, int(split["confirmatory_reference_seed"])
    )
    margin = int(contract["metrics"]["interior_margin_pixels"])
    development = [
        development_colour[margin:-margin, margin:-margin, layer]
        for layer in range(3)
    ] + [development_bw[margin:-margin, margin:-margin]]
    confirmation = [
        confirmation_colour[margin:-margin, margin:-margin, layer]
        for layer in range(3)
    ] + [confirmation_bw[margin:-margin, margin:-margin]]
    families = ["density-shot", "density-shot", "density-shot", "transmittance-shot"]
    profiles: list[CompoundPoissonProfile] = []
    development_metrics: list[dict[str, Any]] = []
    development_seed = int(split["development_candidate_seed"])
    for index, (reference, family) in enumerate(
        zip(development, families, strict=True)
    ):
        profile, metrics = _compile_layer(
            reference,
            family=family,
            seed=development_seed + index,
            contract=contract,
        )
        profiles.append(profile)
        development_metrics.append(metrics)
    confirmation_seed = int(split["confirmatory_candidate_seed"])
    confirm_profiles = [
        replace(profile, seed=confirmation_seed + index)
        for index, profile in enumerate(profiles)
    ]
    candidates = [
        render_compound_poisson(profile, reference.shape)
        for profile, reference in zip(confirm_profiles, confirmation, strict=True)
    ]
    lags = contract["metrics"]["acf_lags"]
    bands = contract["metrics"]["radial_nps_bands_cycles_per_pixel"]
    metrics = [
        _profile_metrics(reference, candidate, lags, bands)
        for reference, candidate in zip(confirmation, candidates, strict=True)
    ]
    partition = []
    repeat = []
    for profile, candidate in zip(confirm_profiles, candidates, strict=True):
        split_row = candidate.shape[0] // 2
        pieces = [
            render_compound_poisson_region(
                profile,
                candidate.shape,
                origin_yx=(0, 0),
                shape=(split_row, candidate.shape[1]),
            ),
            render_compound_poisson_region(
                profile,
                candidate.shape,
                origin_yx=(split_row, 0),
                shape=(candidate.shape[0] - split_row, candidate.shape[1]),
            ),
        ]
        partition.append(np.array_equal(np.concatenate(pieces), candidate))
        repeat.append(
            np.array_equal(
                render_compound_poisson(profile, candidate.shape), candidate
            )
        )
    gates = contract["automatic_gates"]
    decisions = {
        "mean": max(item["mean_abs_error"] for item in metrics)
        <= float(gates["mean_abs_error_max"]),
        "variance": min(item["variance_ratio"] for item in metrics)
        >= float(gates["variance_ratio_min"])
        and max(item["variance_ratio"] for item in metrics)
        <= float(gates["variance_ratio_max"]),
        "acf": max(item["acf_abs_error_max"] for item in metrics)
        <= float(gates["acf_abs_error_max"]),
        "nps": max(item["nps_band_relative_error_max"] for item in metrics)
        <= float(gates["nps_band_relative_error_max"]),
        "repeat": all(repeat),
        "partition": all(partition),
        "domain": bool(
            all(np.all(np.isfinite(value)) and np.all(value > 0.0) for value in candidates)
            and np.all(candidates[-1] <= 1.0)
        ),
    }
    passed = all(decisions.values())
    profile_rows = [
        {
            "family": profile.family,
            "poisson_rate": profile.poisson_rate,
            "correlation_sigma_pixels": profile.correlation_sigma_pixels,
            "baseline": profile.baseline,
            "scale": profile.scale,
        }
        for profile in profiles
    ]
    core = {
        "schema": "neuro_film.u6_p4b_compound_poisson_compiler_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "compiled_profiles": profile_rows,
        "development_metrics": development_metrics,
        "confirmatory_metrics": metrics,
        "repeat_exact": repeat,
        "partition_exact": partition,
        "decisions": decisions,
        "automatic_pass": passed,
        "branch": contract["branch_rule"]["pass" if passed else "fail"],
        "negative_control": contract["frozen_negative_control"],
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
