"""U5.R2CB4 measured-spectrum conformance for fixed Fujifilm dye bases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.eval.fujifilm_e6_spectral_dye_signature import (
    CHANNELS,
    STOCKS,
    _curve_values,
)
from src.eval.physical_measured_slide_spectral_scanner import _load_measured_table

SCHEMA = "neuro_film.u5_r2cb4_fujifilm_dye_basis_measured_conformance_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb4_fujifilm_dye_basis_measured_conformance_report.v1"
EXPERIMENT_ID = "U5.R2CB4"
CONTRACT_SHA256 = "868e5a40892737768b2a1c054f9e26761eeb44b484515c1147b294e708694c6f"
CORRECT_STOCK = "fujichrome_velvia_100_rvp100_af3_202e"


class FujifilmDyeConformanceError(RuntimeError):
    """Raised when a CB4 input or physical-domain check fails."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmDyeConformanceError("CB4 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmDyeConformanceError("CB4 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmDyeConformanceError("CB4 frozen contract structure drift")
    for key in (
        "cb1_decision_path",
        "trace_path",
        "aq1_report_path",
        "pair_table_path",
    ):
        _relative_path(payload["parents"][key])
    return payload


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _wavelength_grid(config: Mapping[str, Any]) -> np.ndarray:
    row = config["measured_target"]["wavelength_nm"]
    start, stop, step = (float(row[key]) for key in ("start", "stop", "step"))
    count = int(row["count"])
    wavelength = start + np.arange(count, dtype=np.float64) * step
    if step <= 0.0 or not np.isclose(wavelength[-1], stop):
        raise FujifilmDyeConformanceError("CB4 wavelength grid is invalid")
    return wavelength


def _fixed_bases(
    trace: Mapping[str, Any], wavelength: np.ndarray
) -> dict[str, np.ndarray]:
    bases: dict[str, np.ndarray] = {}
    for stock in STOCKS:
        columns = [np.ones(wavelength.size, dtype=np.float64)]
        for channel in CHANNELS:
            source_x, source_y = _curve_values(trace["stocks"][stock], channel)
            columns.append(
                np.interp(
                    wavelength,
                    source_x,
                    source_y,
                    left=float(source_y[0]),
                    right=float(source_y[-1]),
                )
            )
        basis = np.stack(columns, axis=1)
        if not np.all(np.isfinite(basis)) or np.any(basis < 0.0):
            raise FujifilmDyeConformanceError("CB4 fixed dye basis is invalid")
        bases[stock] = basis
    return bases


def _fit_basis_rows(
    density: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(density, dtype=np.float64)
    matrix = np.asarray(basis, dtype=np.float64)
    if (
        values.ndim != 2
        or matrix.ndim != 2
        or values.shape[1] != matrix.shape[0]
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(matrix))
        or np.any(values < 0.0)
        or np.any(matrix < 0.0)
    ):
        raise FujifilmDyeConformanceError("CB4 NNLS input is invalid")
    coefficients = np.empty((values.shape[0], matrix.shape[1]), dtype=np.float64)
    rmse = np.empty(values.shape[0], dtype=np.float64)
    for index, target in enumerate(values):
        coefficient, _ = nnls(matrix, target)
        coefficients[index] = coefficient
        residual = matrix @ coefficient - target
        rmse[index] = np.sqrt(np.mean(residual * residual))
    return coefficients, rmse


def _summary(values: np.ndarray) -> dict[str, float]:
    row = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(row)),
        "p95": float(np.percentile(row, 95.0)),
        "maximum": float(np.max(row)),
    }


def evaluate_conformance(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = config["parents"]
    hash_pairs = (
        ("cb1_decision_path", "cb1_decision_sha256"),
        ("trace_path", "trace_sha256"),
        ("aq1_report_path", "aq1_report_sha256"),
        ("pair_table_path", "pair_table_sha256"),
    )
    for path_key, hash_key in hash_pairs:
        path = root / _relative_path(parents[path_key])
        if not path.is_file() or hash_file(path) != parents[hash_key]:
            raise FujifilmDyeConformanceError(
                f"CB4 input integrity mismatch: {path_key}"
            )
    cb1 = json.loads((root / parents["cb1_decision_path"]).read_text(encoding="utf-8"))
    if (
        cb1.get("decision") != parents["cb1_required_decision"]
        or cb1.get("stable_evidence_id") != parents["cb1_required_stable_evidence_id"]
    ):
        raise FujifilmDyeConformanceError("CB4 CB1 decision mismatch")
    aq1 = json.loads((root / parents["aq1_report_path"]).read_text(encoding="utf-8"))
    if (
        not aq1.get("automatic_pass")
        or aq1["table"]["sha256"] != parents["pair_table_sha256"]
    ):
        raise FujifilmDyeConformanceError("CB4 AQ1 table is not eligible")

    test_sets, slides, spectral_pct, _ = _load_measured_table(
        root / parents["pair_table_path"]
    )
    wavelength = _wavelength_grid(config)
    source_wavelength = np.arange(380.0, 781.0, 10.0, dtype=np.float64)
    consumed = np.isin(source_wavelength, wavelength)
    transmittance = spectral_pct[:, consumed] / 100.0
    if (
        transmittance.shape != (int(config["measured_target"]["rows"]), wavelength.size)
        or np.any(transmittance <= 0.0)
        or not np.all(np.isfinite(transmittance))
    ):
        raise FujifilmDyeConformanceError("CB4 measured transmittance is invalid")
    density = -np.log10(transmittance)

    split = config["split"]
    development_sets = np.asarray(split["development_test_sets"], dtype=np.int64)
    development_slides = np.asarray(split["development_slides"], dtype=np.int64)
    held_sets = np.asarray(split["held_test_sets"], dtype=np.int64)
    held_slides = np.asarray(split["held_slides"], dtype=np.int64)
    masks = {
        "development": np.isin(test_sets, development_sets)
        & np.isin(slides, development_slides),
        "held-set": np.isin(test_sets, held_sets) & np.isin(slides, development_slides),
        "held-slide": np.isin(test_sets, development_sets)
        & np.isin(slides, held_slides),
        "joint-held": np.isin(test_sets, held_sets) & np.isin(slides, held_slides),
    }
    if not np.all(np.sum(np.column_stack(tuple(masks.values())), axis=1) == 1):
        raise FujifilmDyeConformanceError("CB4 split does not partition rows")

    trace = json.loads((root / parents["trace_path"]).read_text(encoding="utf-8"))
    bases = _fixed_bases(trace, wavelength)
    coefficients: dict[str, np.ndarray] = {}
    errors: dict[str, np.ndarray] = {}
    constant = np.ones((wavelength.size, 1), dtype=np.float64)
    coefficients["constant-density-only"], errors["constant-density-only"] = (
        _fit_basis_rows(density, constant)
    )
    for stock, basis in bases.items():
        coefficients[stock], errors[stock] = _fit_basis_rows(density, basis)

    wrong_stocks = tuple(stock for stock in STOCKS if stock != CORRECT_STOCK)
    correct_error = errors[CORRECT_STOCK]
    best_wrong = np.minimum(*(errors[stock] for stock in wrong_stocks))
    constant_error = errors["constant-density-only"]
    epsilon = np.finfo(np.float64).tiny
    cell_records: dict[str, Any] = {}
    for name, mask in masks.items():
        correct = correct_error[mask]
        wrong = best_wrong[mask]
        baseline = constant_error[mask]
        cell_records[name] = {
            "rows": int(np.sum(mask)),
            "candidate_density_rmse": {
                "constant-density-only": _summary(baseline),
                **{stock: _summary(errors[stock][mask]) for stock in STOCKS},
            },
            "correct_basis_win_fraction_vs_best_wrong": float(np.mean(correct < wrong)),
            "correct_basis_relative_improvement_vs_best_wrong": _summary(
                (wrong - correct) / np.maximum(wrong, epsilon)
            ),
            "correct_basis_relative_improvement_vs_constant": _summary(
                (baseline - correct) / np.maximum(baseline, epsilon)
            ),
            "correct_basis_dye_zero_coefficient_fraction": np.mean(
                coefficients[CORRECT_STOCK][mask, 1:] <= 1e-12, axis=0
            ).tolist(),
        }

    gates = config["gates"]
    joint = cell_records["joint-held"]
    correct_joint = joint["candidate_density_rmse"][CORRECT_STOCK]
    checks = {
        "rows": density.shape[0] == int(gates["required_rows"]),
        "split_rows": (
            cell_records["development"]["rows"]
            == int(gates["required_development_rows"])
            and cell_records["held-set"]["rows"] == int(gates["required_held_set_rows"])
            and cell_records["held-slide"]["rows"]
            == int(gates["required_held_slide_rows"])
            and cell_records["joint-held"]["rows"]
            == int(gates["required_joint_held_rows"])
        ),
        "finite": bool(
            np.all(np.isfinite(density))
            and all(np.all(np.isfinite(row)) for row in errors.values())
        )
        is bool(gates["all_values_finite"]),
        "transmittance_domain": bool(np.all(transmittance > 0.0))
        is bool(gates["all_transmittance_strictly_positive"]),
        "joint_correct_basis_wins": joint["correct_basis_win_fraction_vs_best_wrong"]
        >= float(gates["joint_held_correct_basis_win_fraction_minimum"]),
        "joint_wrong_basis_margin": joint[
            "correct_basis_relative_improvement_vs_best_wrong"
        ]["median"]
        >= float(gates["joint_held_median_relative_improvement_vs_best_wrong_minimum"]),
        "joint_absolute_error": correct_joint["median"]
        <= float(gates["joint_held_median_density_rmse_maximum"])
        and correct_joint["p95"] <= float(gates["joint_held_p95_density_rmse_maximum"]),
        "joint_constant_control": joint[
            "correct_basis_relative_improvement_vs_constant"
        ]["median"]
        >= float(gates["joint_held_median_relative_improvement_vs_constant_minimum"]),
        "each_cell_correct_basis_wins": all(
            row["correct_basis_win_fraction_vs_best_wrong"]
            >= float(gates["each_cell_correct_basis_win_fraction_minimum"])
            for row in cell_records.values()
        ),
        "dye_support": all(
            fraction <= float(gates["maximum_each_dye_zero_coefficient_fraction"])
            for row in cell_records.values()
            for fraction in row["correct_basis_dye_zero_coefficient_fraction"]
        ),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent_cb1_stable_evidence_id": parents["cb1_required_stable_evidence_id"],
        "pair_table_sha256": parents["pair_table_sha256"],
        "wavelength_nm": wavelength.tolist(),
        "measured_density_sha256": _array_sha256(density),
        "basis_sha256": {stock: _array_sha256(basis) for stock, basis in bases.items()},
        "coefficient_sha256": {
            name: _array_sha256(value) for name, value in coefficients.items()
        },
        "cells": cell_records,
        "checks": checks,
        "failed_gates": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
        "decision": (
            "retain_measured_velvia100f_dye_basis_compatibility"
            if passed
            else "close_direct_measured_spectrum_dye_basis_conformance"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "FujifilmDyeConformanceError",
    "_fit_basis_rows",
    "evaluate_conformance",
    "hash_file",
    "load_contract",
    "write_report",
]
