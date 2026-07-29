"""U6.P6F measured Velvia 100F target spectra through synthetic scanners."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.physical_spectral_scanner_reference import (
    _wavelength_grid,
    load_contract as load_p6d_contract,
)
from src.eval.physical_spectral_scanner_rgb_approximation import (
    fit_nonnegative_row_sum_bounded_matrix,
)
from src.film_physics.spectral_scanner import (
    apply_spectral_scanner,
    synthetic_profile_from_contract,
)


SCHEMA = "neuro_film.u6_p6f_measured_slide_spectral_scanner_contract.v1"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported U6.P6F contract")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_id(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "ascii"
        )
    ).hexdigest()


def _load_measured_table(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    test_sets: list[int] = []
    slides: list[int] = []
    spectra: list[list[float]] = []
    sample_ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {
            "test_set",
            "slide_index",
            "sample_id",
            "spectral_pct_380_to_780",
        }
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise ValueError("measured pair table fields do not match P6F")
        for row in reader:
            values = [
                float(value)
                for value in row["spectral_pct_380_to_780"].split(";")
            ]
            if len(values) != 41:
                raise ValueError("measured spectrum must contain 41 values")
            test_sets.append(int(row["test_set"]))
            slides.append(int(row["slide_index"]))
            sample_ids.append(str(row["sample_id"]))
            spectra.append(values)
    return (
        np.asarray(test_sets, dtype=np.int64),
        np.asarray(slides, dtype=np.int64),
        np.asarray(spectra, dtype=np.float64),
        tuple(sample_ids),
    )


def _error_summary(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    error = np.linalg.norm(prediction - target, axis=1)
    return {
        "mean_l2": float(np.mean(error)),
        "median_l2": float(np.median(error)),
        "p95_l2": float(np.percentile(error, 95.0)),
        "maximum_l2": float(np.max(error)),
    }


def evaluate_measured_slide_spectral_scanner(
    root: Path,
    contract: dict[str, Any],
) -> dict[str, Any]:
    parents = contract["parents"]
    for key, hash_key in (
        ("p6d_contract_path", "p6d_contract_sha256"),
        ("aq1_report_path", "aq1_report_sha256"),
        ("pair_table_path", "pair_table_sha256"),
    ):
        if _sha256(root / parents[key]) != parents[hash_key]:
            raise ValueError(f"{key} hash mismatch")
    p6e_decision = json.loads(
        (root / parents["p6e_decision_path"]).read_text(encoding="utf-8")
    )
    if (
        p6e_decision.get("decision")
        != "close_nonlinear_rgb_rescue_retain_bounded_matrix_and_spectral_reference"
        or parents["p6e_nonlinear_rgb_route_closed"] is not True
    ):
        raise ValueError("P6E nonlinear closure is not exact")
    aq1_report = json.loads(
        (root / parents["aq1_report_path"]).read_text(encoding="utf-8")
    )
    if not aq1_report.get("automatic_pass") or (
        aq1_report["table"]["sha256"] != parents["pair_table_sha256"]
    ):
        raise ValueError("AQ1 measured table evidence is not eligible")

    test_sets, slides, spectral_pct, sample_ids = _load_measured_table(
        root / parents["pair_table_path"]
    )
    measured = contract["measured_data_contract"]
    if spectral_pct.shape != (int(measured["expected_rows"]), 41):
        raise ValueError("measured table row or wavelength count mismatch")
    if sorted(set(test_sets.tolist())) != measured["expected_test_sets"]:
        raise ValueError("measured test-set inventory mismatch")
    if sorted(set(slides.tolist())) != measured["expected_slides"]:
        raise ValueError("measured slide inventory mismatch")
    unique_cells, cell_counts = np.unique(
        np.column_stack((test_sets, slides)), axis=0, return_counts=True
    )
    if unique_cells.shape[0] != 30 or not np.all(
        cell_counts == int(measured["expected_samples_per_set_slide"])
    ):
        raise ValueError("measured set/slide cell support mismatch")
    if any(
        len(
            {
                sample_ids[index]
                for index in np.flatnonzero(
                    (test_sets == test_set) & (slides == slide)
                )
            }
        )
        != int(measured["expected_samples_per_set_slide"])
        for test_set, slide in unique_cells
    ):
        raise ValueError("measured sample IDs are incomplete within a cell")

    source_grid = measured["source_wavelength_nm"]
    consumed_grid = measured["consumed_wavelength_nm"]
    source_wavelength = np.arange(
        float(source_grid["start"]),
        float(source_grid["stop"]) + float(source_grid["step"]) * 0.5,
        float(source_grid["step"]),
        dtype=np.float64,
    )
    consumed_mask = (
        (source_wavelength >= float(consumed_grid["start"]))
        & (source_wavelength <= float(consumed_grid["stop"]))
    )
    transmittance = spectral_pct[:, consumed_mask] / 100.0

    p6d_contract = load_p6d_contract(root / parents["p6d_contract_path"])
    wavelength = _wavelength_grid(p6d_contract["wavelength_grid_nm"])
    if not np.array_equal(source_wavelength[consumed_mask], wavelength):
        raise ValueError("measured/P6D wavelength grids do not match")
    profiles = {
        name: synthetic_profile_from_contract(wavelength, row)
        for name, row in p6d_contract["synthetic_profiles"].items()
    }
    scanner_a_rgb = apply_spectral_scanner(
        np.asarray(transmittance, dtype=np.float64), profiles["scanner_a"]
    )
    scanner_b_rgb = apply_spectral_scanner(
        np.asarray(transmittance, dtype=np.float64), profiles["scanner_b"]
    )

    split = contract["split_contract"]
    development_sets = np.asarray(split["development_test_sets"], dtype=np.int64)
    held_sets = np.asarray(split["held_test_sets"], dtype=np.int64)
    development_slides = np.asarray(split["development_slides"], dtype=np.int64)
    held_slides = np.asarray(split["held_slides"], dtype=np.int64)
    masks = {
        "development": np.isin(test_sets, development_sets)
        & np.isin(slides, development_slides),
        "held-set": np.isin(test_sets, held_sets)
        & np.isin(slides, development_slides),
        "held-slide": np.isin(test_sets, development_sets)
        & np.isin(slides, held_slides),
        "joint-held": np.isin(test_sets, held_sets)
        & np.isin(slides, held_slides),
    }
    coverage = np.sum(np.column_stack(tuple(masks.values())), axis=1)
    if not np.all(coverage == 1):
        raise ValueError("P6F split cells must cover every row exactly once")

    matrix = fit_nonnegative_row_sum_bounded_matrix(
        scanner_a_rgb[masks["development"]],
        scanner_b_rgb[masks["development"]],
    )
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    extrema: dict[str, dict[str, list[float]]] = {}
    for name, mask in masks.items():
        source = scanner_a_rgb[mask]
        target = scanner_b_rgb[mask]
        predictions = {
            "identity": source,
            "nonnegative-row-sum-bounded-3x3": source @ matrix.T,
            "spectral-oracle": target.copy(),
        }
        summaries[name] = {
            candidate: _error_summary(prediction, target)
            for candidate, prediction in predictions.items()
        }
        extrema[name] = {
            candidate: [
                float(np.min(prediction)),
                float(np.max(prediction)),
            ]
            for candidate, prediction in predictions.items()
        }

    joint_identity = summaries["joint-held"]["identity"]["mean_l2"]
    joint_matrix = summaries["joint-held"][
        "nonnegative-row-sum-bounded-3x3"
    ]["mean_l2"]
    relative_improvement = (
        (joint_identity - joint_matrix) / joint_identity
        if joint_identity > 0.0
        else 0.0
    )
    separation = np.linalg.norm(scanner_a_rgb - scanner_b_rgb, axis=1)
    candidate_minimum = min(
        interval[0]
        for cell in extrema.values()
        for name, interval in cell.items()
        if name != "spectral-oracle"
    )
    candidate_maximum = max(
        interval[1]
        for cell in extrema.values()
        for name, interval in cell.items()
        if name != "spectral-oracle"
    )
    gates = contract["automatic_gates"]
    checks = {
        "development_rows": int(np.sum(masks["development"]))
        == int(gates["expected_development_rows"]),
        "held_set_rows": int(np.sum(masks["held-set"]))
        == int(gates["expected_held_set_rows"]),
        "held_slide_rows": int(np.sum(masks["held-slide"]))
        == int(gates["expected_held_slide_rows"]),
        "joint_held_rows": int(np.sum(masks["joint-held"]))
        == int(gates["expected_joint_held_rows"]),
        "strict_transmittance_domain": bool(
            np.all(transmittance > 0.0) and np.all(transmittance < 1.0)
        )
        is bool(gates["all_transmittance_strictly_inside_zero_one"]),
        "spectral_oracle": max(
            summaries[cell]["spectral-oracle"]["maximum_l2"]
            for cell in summaries
        )
        <= float(gates["spectral_oracle_l2_max"]),
        "candidate_bounded": candidate_minimum
        >= float(gates["candidate_output_minimum"])
        and candidate_maximum <= float(gates["candidate_output_maximum"]),
        "joint_relative": relative_improvement
        >= float(gates["joint_held_matrix_vs_identity_mean_improvement_min"]),
        "joint_mean": joint_matrix
        <= float(gates["joint_held_matrix_mean_l2_max"]),
        "joint_p95": summaries["joint-held"][
            "nonnegative-row-sum-bounded-3x3"
        ]["p95_l2"]
        <= float(gates["joint_held_matrix_p95_l2_max"]),
        "held_set_p95": summaries["held-set"][
            "nonnegative-row-sum-bounded-3x3"
        ]["p95_l2"]
        <= float(gates["held_set_matrix_p95_l2_max"]),
        "held_slide_p95": summaries["held-slide"][
            "nonnegative-row-sum-bounded-3x3"
        ]["p95_l2"]
        <= float(gates["held_slide_matrix_p95_l2_max"]),
        "scanner_profile_separation": float(np.median(separation))
        >= float(gates["scanner_profile_separation_median_l2_min"]),
    }
    absolute_names = (
        "joint_mean",
        "joint_p95",
        "held_set_p95",
        "held_slide_p95",
    )
    automatic_pass = all(checks.values())
    if automatic_pass:
        branch = contract["branch_rule"]["pass"]
    elif all(checks[name] for name in absolute_names) and not checks["joint_relative"]:
        branch = contract["branch_rule"]["fail_relative"]
    else:
        branch = contract["branch_rule"]["fail_absolute"]
    core = {
        "schema": (
            "neuro_film.u6_p6f_measured_slide_spectral_scanner_report.v1"
        ),
        "node": contract["node"],
        "claim_ceiling": contract["claim_ceiling"],
        "measured_data": {
            "rows": int(transmittance.shape[0]),
            "wavelength_count": int(transmittance.shape[1]),
            "test_sets": sorted(set(test_sets.tolist())),
            "slides": sorted(set(slides.tolist())),
            "transmittance_minimum": float(np.min(transmittance)),
            "transmittance_maximum": float(np.max(transmittance)),
            "transmittance_sha256": hashlib.sha256(
                np.ascontiguousarray(transmittance).tobytes(order="C")
            ).hexdigest(),
        },
        "split_rows": {
            name: int(np.sum(mask)) for name, mask in masks.items()
        },
        "fit": {
            "cell": "development",
            "bounded_matrix": matrix.tolist(),
            "bounded_matrix_row_sums": np.sum(matrix, axis=1).tolist(),
        },
        "summaries": summaries,
        "output_extrema": extrema,
        "joint_held_matrix_vs_identity_mean_improvement": float(
            relative_improvement
        ),
        "scanner_profile_separation": {
            "median_l2": float(np.median(separation)),
            "p95_l2": float(np.percentile(separation, 95.0)),
            "maximum_l2": float(np.max(separation)),
        },
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
    "evaluate_measured_slide_spectral_scanner",
    "load_contract",
    "write_report",
]
