"""U6.P6E group-split RGB approximation audit for a spectral scanner oracle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls
from scipy.special import expit

from src.eval.physical_spectral_scanner_reference import (
    _wavelength_grid,
    load_contract as load_p6d_contract,
)
from src.film_physics.spectral_scanner import (
    apply_spectral_scanner,
    construct_primary_metamer_pair,
    synthetic_profile_from_contract,
)


SCHEMA = "neuro_film.u6_p6e_spectral_scanner_rgb_approximation_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6E contract")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_hashes(values: np.ndarray) -> set[str]:
    contiguous = np.ascontiguousarray(values, dtype=np.float64)
    return {
        hashlib.sha256(contiguous[index].tobytes(order="C")).hexdigest()
        for index in range(contiguous.shape[0])
    }


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _generate_split(
    wavelength: np.ndarray,
    contract: dict[str, Any],
    split_name: str,
) -> tuple[np.ndarray, tuple[int, ...]]:
    data = contract["data_contract"]
    row = data[split_name]
    centers = np.asarray(data["basis_center_nm"], dtype=np.float64)
    jitter = float(data["group_center_jitter_nm"])
    sigma_low, sigma_high = (
        float(value) for value in row["basis_sigma_nm_interval"]
    )
    base_low, base_high = (
        float(value) for value in row["density_base_interval"]
    )
    component_scale = float(row["density_component_scale"])
    samples_per_group = int(row["samples_per_group"])
    spectra: list[np.ndarray] = []
    groups: list[int] = []
    for seed_value in row["group_seeds"]:
        seed = int(seed_value)
        rng = np.random.default_rng(seed)
        group_centers = centers + rng.uniform(-jitter, jitter, centers.shape)
        base = rng.uniform(base_low, base_high, samples_per_group)
        sigmas = rng.uniform(
            sigma_low,
            sigma_high,
            (samples_per_group, centers.size),
        )
        amplitudes = rng.gamma(
            shape=1.5,
            scale=component_scale / 1.5,
            size=(samples_per_group, centers.size),
        )
        distance = wavelength[None, None, :] - group_centers[None, :, None]
        bases = np.exp(-0.5 * (distance / sigmas[:, :, None]) ** 2)
        density = base[:, None] + np.sum(
            amplitudes[:, :, None] * bases, axis=1
        )
        spectra.append(np.power(10.0, -density, dtype=np.float64))
        groups.extend([seed] * samples_per_group)
    return np.concatenate(spectra, axis=0), tuple(groups)


def fit_nonnegative_row_sum_bounded_matrix(
    source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    rows = []
    for channel in range(3):
        weights, _ = nnls(source, target[:, channel])
        total = float(np.sum(weights))
        if total > 1.0:
            weights = weights / total
        rows.append(weights)
    return np.asarray(rows, dtype=np.float64)


def _quadratic_features(source: np.ndarray) -> np.ndarray:
    r, g, b = (source[:, index] for index in range(3))
    return np.column_stack(
        (
            np.ones(source.shape[0], dtype=np.float64),
            r,
            g,
            b,
            r * r,
            g * g,
            b * b,
            r * g,
            r * b,
            g * b,
        )
    )


def _fit_quadratic_logit(
    source: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float,
    ridge: float,
) -> np.ndarray:
    features = _quadratic_features(source)
    bounded_target = np.minimum(np.maximum(target, epsilon), 1.0 - epsilon)
    logits = np.log(bounded_target / (1.0 - bounded_target))
    penalty = np.eye(features.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    return np.linalg.solve(
        features.T @ features + penalty,
        features.T @ logits,
    )


def _predict_quadratic(source: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return expit(_quadratic_features(source) @ coefficients)


def _error_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=1)
    return {
        "mean_l2": float(np.mean(error)),
        "median_l2": float(np.median(error)),
        "p95_l2": float(np.percentile(error, 95.0)),
        "maximum_l2": float(np.max(error)),
    }


def evaluate_spectral_scanner_rgb_approximation(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    parents = contract["parents"]
    p6d_contract_path = root / parents["p6d_contract_path"]
    if _sha256(p6d_contract_path) != parents["p6d_contract_sha256"]:
        raise ValueError("P6D contract hash mismatch")
    p6d_contract = load_p6d_contract(p6d_contract_path)
    wavelength = _wavelength_grid(p6d_contract["wavelength_grid_nm"])
    profiles = {
        name: synthetic_profile_from_contract(wavelength, row)
        for name, row in p6d_contract["synthetic_profiles"].items()
    }
    scanner_a = profiles["scanner_a"]
    scanner_b = profiles["scanner_b"]

    p6d_decision = json.loads(
        (root / parents["p6d_decision_path"]).read_text(encoding="utf-8")
    )
    p6d_report_path = root / p6d_decision["evidence"]["report"]["path"]
    if _sha256(p6d_report_path) != parents["p6d_report_sha256"]:
        raise ValueError("P6D report hash mismatch")
    p6d_report = json.loads(p6d_report_path.read_text(encoding="utf-8"))

    spectra: dict[str, np.ndarray] = {}
    groups: dict[str, tuple[int, ...]] = {}
    scanner_a_rgb: dict[str, np.ndarray] = {}
    scanner_b_rgb: dict[str, np.ndarray] = {}
    for split in ("development", "confirmation", "stress"):
        spectra[split], groups[split] = _generate_split(
            wavelength, contract, split
        )
        scanner_a_rgb[split] = apply_spectral_scanner(
            spectra[split], scanner_a
        )
        scanner_b_rgb[split] = apply_spectral_scanner(
            spectra[split], scanner_b
        )

    candidate = contract["candidate_contract"]
    development_source = scanner_a_rgb["development"]
    development_target = scanner_b_rgb["development"]
    matrix = fit_nonnegative_row_sum_bounded_matrix(
        development_source, development_target
    )
    coefficients = _fit_quadratic_logit(
        development_source,
        development_target,
        epsilon=float(candidate["quadratic_logit_target_epsilon"]),
        ridge=float(candidate["quadratic_ridge"]),
    )

    summaries: dict[str, dict[str, dict[str, float]]] = {}
    output_extrema: dict[str, dict[str, list[float]]] = {}
    for split in ("development", "confirmation", "stress"):
        source = scanner_a_rgb[split]
        target = scanner_b_rgb[split]
        predictions = {
            "identity": source,
            "nonnegative-row-sum-bounded-3x3": source @ matrix.T,
            "fixed-quadratic-logit-ridge": _predict_quadratic(
                source, coefficients
            ),
            "spectral-oracle": target.copy(),
        }
        summaries[split] = {
            name: _error_summary(prediction, target)
            for name, prediction in predictions.items()
        }
        output_extrema[split] = {
            name: [
                float(np.min(prediction)),
                float(np.max(prediction)),
            ]
            for name, prediction in predictions.items()
        }

    group_sets = {name: set(values) for name, values in groups.items()}
    spectrum_hashes = {name: _row_hashes(values) for name, values in spectra.items()}
    rgb_hashes = {
        name: _row_hashes(values) for name, values in scanner_a_rgb.items()
    }
    split_pairs = (
        ("development", "confirmation"),
        ("development", "stress"),
        ("confirmation", "stress"),
    )
    group_overlap = sum(
        len(group_sets[left] & group_sets[right]) for left, right in split_pairs
    )
    spectrum_overlap = sum(
        len(spectrum_hashes[left] & spectrum_hashes[right])
        for left, right in split_pairs
    )
    rgb_overlap = sum(
        len(rgb_hashes[left] & rgb_hashes[right])
        for left, right in split_pairs
    )

    metamer_first, metamer_second = construct_primary_metamer_pair(
        scanner_a,
        scanner_b,
        neutral_transmittance=float(
            p6d_contract["synthetic_witnesses"]["neutral_transmittance"]
        ),
        perturbation_max_abs=float(
            p6d_contract["synthetic_witnesses"][
                "metamer_perturbation_max_abs"
            ]
        ),
    )
    reproduced_lower_bound = float(
        np.linalg.norm(
            apply_spectral_scanner(metamer_first, scanner_b)
            - apply_spectral_scanner(metamer_second, scanner_b)
        )
        * 0.5
    )
    expected_lower_bound = float(
        p6d_report["metrics"]["rgb_only_irreducible_l2_error"]
    )
    confirmation_matrix = summaries["confirmation"][
        "nonnegative-row-sum-bounded-3x3"
    ]["mean_l2"]
    confirmation_quadratic = summaries["confirmation"][
        "fixed-quadratic-logit-ridge"
    ]["mean_l2"]
    relative_improvement = (
        (confirmation_matrix - confirmation_quadratic) / confirmation_matrix
        if confirmation_matrix > 0.0
        else 0.0
    )
    all_candidate_minimum = min(
        interval[0]
        for split in output_extrema.values()
        for name, interval in split.items()
        if name != "spectral-oracle"
    )
    all_candidate_maximum = max(
        interval[1]
        for split in output_extrema.values()
        for name, interval in split.items()
        if name != "spectral-oracle"
    )
    gates = contract["automatic_gates"]
    checks = {
        "development_rows": spectra["development"].shape[0]
        == int(gates["expected_development_rows"]),
        "confirmation_rows": spectra["confirmation"].shape[0]
        == int(gates["expected_confirmation_rows"]),
        "stress_rows": spectra["stress"].shape[0]
        == int(gates["expected_stress_rows"]),
        "group_disjoint": group_overlap
        == int(gates["cross_split_group_overlap"]),
        "spectrum_hash_disjoint": spectrum_overlap
        == int(gates["cross_split_spectrum_hash_overlap"]),
        "scanner_a_rgb_hash_disjoint": rgb_overlap
        == int(gates["cross_split_scanner_a_rgb_hash_overlap"]),
        "spectral_oracle": max(
            summaries[split]["spectral-oracle"]["maximum_l2"]
            for split in summaries
        )
        <= float(gates["spectral_oracle_l2_max"]),
        "candidate_bounded": all_candidate_minimum
        >= float(gates["candidate_output_minimum"])
        and all_candidate_maximum <= float(gates["candidate_output_maximum"]),
        "confirmation_relative": relative_improvement
        >= float(
            gates["confirmation_quadratic_vs_matrix_mean_improvement_min"]
        ),
        "confirmation_absolute": summaries["confirmation"][
            "fixed-quadratic-logit-ridge"
        ]["p95_l2"]
        <= float(gates["confirmation_quadratic_l2_p95_max"]),
        "stress_absolute": summaries["stress"][
            "fixed-quadratic-logit-ridge"
        ]["p95_l2"]
        <= float(gates["stress_quadratic_l2_p95_max"]),
        "p6d_metamer_lower_bound": abs(
            reproduced_lower_bound - expected_lower_bound
        )
        <= float(gates["p6d_metamer_lower_bound_reproduced_abs_error_max"]),
    }
    infrastructure_checks = (
        "development_rows",
        "confirmation_rows",
        "stress_rows",
        "group_disjoint",
        "spectrum_hash_disjoint",
        "scanner_a_rgb_hash_disjoint",
        "spectral_oracle",
        "candidate_bounded",
        "p6d_metamer_lower_bound",
    )
    infrastructure_pass = all(checks[name] for name in infrastructure_checks)
    absolute_pass = checks["confirmation_absolute"] and checks["stress_absolute"]
    automatic_pass = all(checks.values())
    if automatic_pass:
        branch = contract["branch_rule"]["pass"]
    elif infrastructure_pass and absolute_pass and not checks["confirmation_relative"]:
        branch = contract["branch_rule"]["fail_relative"]
    else:
        branch = contract["branch_rule"]["fail_absolute_or_stress"]

    core = {
        "schema": (
            "neuro_film.u6_p6e_spectral_scanner_rgb_approximation_report.v1"
        ),
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "dataset": {
            "row_counts": {
                split: int(values.shape[0]) for split, values in spectra.items()
            },
            "group_counts": {
                split: len(set(values)) for split, values in groups.items()
            },
            "group_overlap": group_overlap,
            "spectrum_hash_overlap": spectrum_overlap,
            "scanner_a_rgb_hash_overlap": rgb_overlap,
            "spectrum_sha256": {
                split: hashlib.sha256(
                    np.ascontiguousarray(values).tobytes(order="C")
                ).hexdigest()
                for split, values in spectra.items()
            },
        },
        "fit": {
            "split": "development",
            "bounded_matrix": matrix.tolist(),
            "bounded_matrix_row_sums": np.sum(matrix, axis=1).tolist(),
            "quadratic_coefficients": coefficients.tolist(),
        },
        "summaries": summaries,
        "output_extrema": output_extrema,
        "confirmation_quadratic_vs_matrix_mean_improvement": float(
            relative_improvement
        ),
        "p6d_metamer_expected_lower_bound": expected_lower_bound,
        "p6d_metamer_reproduced_lower_bound": reproduced_lower_bound,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "branch": branch,
    }
    return {**core, "stable_evidence_id": _stable_id(core)}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "evaluate_spectral_scanner_rgb_approximation",
    "fit_nonnegative_row_sum_bounded_matrix",
    "load_contract",
    "write_report",
]
