"""Frozen physical-scale LOD audit for the U6.P4B structure compiler."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_structure_compiler import _acf, _nps_bands
from src.film_physics import (
    CompoundPoissonProfile,
    render_compound_poisson,
    render_compound_poisson_region,
    rescale_compound_poisson_profile,
)


SCHEMA = "neuro_film.u6_p4c_physical_lod_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P4C contract")
    return payload


def _profile(row: dict[str, Any], *, seed: int) -> CompoundPoissonProfile:
    return CompoundPoissonProfile(
        family=str(row["family"]),
        poisson_rate=float(row["poisson_rate"]),
        correlation_sigma_pixels=float(row["correlation_sigma_pixels"]),
        baseline=float(row["baseline"]),
        scale=float(row["scale"]),
        seed=seed,
    )


def _area_mean(values: np.ndarray, factor: int) -> np.ndarray:
    height, width = values.shape
    if height % factor or width % factor:
        raise ValueError("reference shape is not divisible by the LOD factor")
    output = values.astype(np.float64).reshape(
        height // factor,
        factor,
        width // factor,
        factor,
    ).mean(axis=(1, 3))
    output = np.asarray(output, dtype=np.float32)
    output.setflags(write=False)
    return output


def _metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    lags: list[list[int]],
    bands: list[list[float]],
) -> dict[str, Any]:
    ref_acf = _acf(reference, lags)
    candidate_acf = _acf(candidate, lags)
    ref_nps = _nps_bands(reference, bands)
    candidate_nps = _nps_bands(candidate, bands)
    return {
        "mean_abs_error": abs(float(np.mean(candidate)) - float(np.mean(reference))),
        "variance_ratio": float(np.var(candidate)) / float(np.var(reference)),
        "acf_abs_error_max": float(
            np.max(np.abs(np.asarray(candidate_acf) - np.asarray(ref_acf)))
        ),
        "nps_band_relative_error_max": float(
            np.max(
                np.abs(np.asarray(candidate_nps) - np.asarray(ref_nps))
                / np.maximum(np.asarray(ref_nps), 1e-12)
            )
        ),
        "reference_acf": ref_acf,
        "candidate_acf": candidate_acf,
        "reference_nps_bands": ref_nps,
        "candidate_nps_bands": candidate_nps,
    }


def evaluate_physical_lod(contract: dict[str, Any]) -> dict[str, Any]:
    shape = tuple(int(value) for value in contract["reference_shape"])
    lags = contract["metrics"]["acf_lags"]
    bands = contract["metrics"]["radial_nps_bands_cycles_per_pixel"]
    reference_seed = int(contract["reference_seed"])
    direct_seed = int(contract["direct_lod_seed"])
    rows: list[dict[str, Any]] = []
    repeat_exact: list[bool] = []
    partition_exact: list[bool] = []
    domain_valid: list[bool] = []
    brightening_violations: list[int] = []
    for layer, profile_row in enumerate(contract["compiled_profiles"]):
        base = _profile(profile_row, seed=reference_seed + layer)
        resolved = render_compound_poisson(base, shape)
        for factor in (int(value) for value in contract["lod_factors"]):
            reference = _area_mean(resolved, factor)
            direct_profile = rescale_compound_poisson_profile(
                base,
                pixel_size_factor=factor,
                seed=direct_seed + layer * 16 + factor,
            )
            candidate = render_compound_poisson(
                direct_profile, reference.shape
            )
            split = candidate.shape[0] // 2
            pieces = [
                render_compound_poisson_region(
                    direct_profile,
                    candidate.shape,
                    origin_yx=(0, 0),
                    shape=(split, candidate.shape[1]),
                ),
                render_compound_poisson_region(
                    direct_profile,
                    candidate.shape,
                    origin_yx=(split, 0),
                    shape=(candidate.shape[0] - split, candidate.shape[1]),
                ),
            ]
            partition_exact.append(
                np.array_equal(np.concatenate(pieces), candidate)
            )
            repeat_exact.append(
                np.array_equal(
                    render_compound_poisson(direct_profile, candidate.shape),
                    candidate,
                )
            )
            if direct_profile.family == "density-shot":
                violations = int(np.count_nonzero(candidate < direct_profile.baseline))
                valid = bool(
                    np.all(np.isfinite(candidate)) and np.all(candidate > 0.0)
                )
            else:
                clear_base = math.exp(-direct_profile.baseline)
                violations = int(np.count_nonzero(candidate > clear_base))
                valid = bool(
                    np.all(np.isfinite(candidate))
                    and np.all(candidate > 0.0)
                    and np.all(candidate <= 1.0)
                )
            brightening_violations.append(violations)
            domain_valid.append(valid)
            rows.append(
                {
                    "layer": layer,
                    "family": direct_profile.family,
                    "lod_factor": factor,
                    "output_pitch_um": (
                        float(contract["base_physical_pitch_um"]) * factor
                    ),
                    **_metrics(
                        reference, candidate, lags=lags, bands=bands
                    ),
                }
            )
    gates = contract["automatic_gates"]
    decisions = {
        "mean": max(row["mean_abs_error"] for row in rows)
        <= float(gates["mean_abs_error_max"]),
        "variance": min(row["variance_ratio"] for row in rows)
        >= float(gates["variance_ratio_min"])
        and max(row["variance_ratio"] for row in rows)
        <= float(gates["variance_ratio_max"]),
        "acf": max(row["acf_abs_error_max"] for row in rows)
        <= float(gates["acf_abs_error_max"]),
        "nps": max(row["nps_band_relative_error_max"] for row in rows)
        <= float(gates["nps_band_relative_error_max"]),
        "repeat": all(repeat_exact),
        "partition": all(partition_exact),
        "domain": all(domain_valid),
        "brightening": sum(brightening_violations)
        == int(gates["brightening_violation_count"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": "neuro_film.u6_p4c_physical_lod_report.v1",
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "lod_metrics": rows,
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
