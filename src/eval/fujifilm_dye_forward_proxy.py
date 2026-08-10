"""U5.R2CB5 source-only forward proxy for fixed Fujifilm dye coordinates."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from src.eval.fujifilm_dye_basis_measured_conformance import (
    CORRECT_STOCK,
    _array_sha256,
    _fit_basis_rows,
    _fixed_bases,
    canonical_json,
    hash_file,
)
from src.eval.fujifilm_e6_spectral_dye_signature import STOCKS
from src.eval.physical_measured_slide_spectral_scanner import _load_measured_table

SCHEMA = "neuro_film.u5_r2cb5_fujifilm_dye_forward_proxy_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb5_fujifilm_dye_forward_proxy_report.v1"
EXPERIMENT_ID = "U5.R2CB5"
CONTRACT_SHA256 = "614ff361874f015e83ca8a3b95e7a0df5f60eec3a80424b9b1b7d917c33e4014"


class FujifilmDyeForwardError(RuntimeError):
    """Raised when a CB5 contract, input or fit violates its boundary."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmDyeForwardError("CB5 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmDyeForwardError("CB5 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmDyeForwardError("CB5 frozen contract structure drift")
    for key in ("cb4_decision_path", "pair_table_path", "trace_path"):
        _relative_path(payload["parents"][key])
    return payload


def _load_source_rgb(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        expected = {"source_r", "source_g", "source_b"}
        if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
            raise FujifilmDyeForwardError("CB5 source RGB fields are missing")
        for row in reader:
            rows.append(
                [float(row[key]) for key in ("source_r", "source_g", "source_b")]
            )
    values = np.asarray(rows, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise FujifilmDyeForwardError("CB5 source RGB is invalid")
    return values


def _fit_nonnegative_mapping(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = np.asarray(source, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[0] < 12
        or not np.all(np.isfinite(x))
        or not np.all(np.isfinite(y))
        or np.any(x < 0.0)
        or np.any(y < 0.0)
    ):
        raise FujifilmDyeForwardError("CB5 mapping fit input is invalid")
    columns = []
    for channel in range(y.shape[1]):
        result = lsq_linear(
            x,
            y[:, channel],
            bounds=(0.0, np.inf),
            max_iter=1000,
            lsmr_tol="auto",
            verbose=0,
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            raise FujifilmDyeForwardError("CB5 nonnegative mapping did not converge")
        columns.append(result.x)
    return np.stack(columns, axis=1)


def _summary(values: np.ndarray) -> dict[str, float]:
    row = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(row)),
        "median": float(np.median(row)),
        "p95": float(np.percentile(row, 95.0)),
        "maximum": float(np.max(row)),
    }


def evaluate_forward_proxy(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    parents = config["parents"]
    for path_key, hash_key in (
        ("cb4_decision_path", "cb4_decision_sha256"),
        ("pair_table_path", "pair_table_sha256"),
        ("trace_path", "trace_sha256"),
    ):
        path = root / _relative_path(parents[path_key])
        if not path.is_file() or hash_file(path) != parents[hash_key]:
            raise FujifilmDyeForwardError(f"CB5 input integrity mismatch: {path_key}")
    cb4 = json.loads((root / parents["cb4_decision_path"]).read_text(encoding="utf-8"))
    if (
        cb4.get("decision") != parents["cb4_required_decision"]
        or cb4.get("stable_evidence_id") != parents["cb4_required_stable_evidence_id"]
    ):
        raise FujifilmDyeForwardError("CB5 CB4 decision mismatch")

    pair_path = root / parents["pair_table_path"]
    test_sets, slides, spectral_pct, _ = _load_measured_table(pair_path)
    source_rgb = _load_source_rgb(pair_path)
    wavelength_all = np.arange(380.0, 781.0, 10.0, dtype=np.float64)
    wavelength = np.arange(410.0, 691.0, 10.0, dtype=np.float64)
    consumed = np.isin(wavelength_all, wavelength)
    transmittance = spectral_pct[:, consumed] / 100.0
    if (
        source_rgb.shape[0] != int(config["gates"]["required_rows"])
        or transmittance.shape != (source_rgb.shape[0], wavelength.size)
        or np.any(transmittance <= 0.0)
        or not np.all(np.isfinite(transmittance))
    ):
        raise FujifilmDyeForwardError("CB5 measured rows are invalid")
    density = -np.log10(transmittance)
    features = np.column_stack((np.ones(source_rgb.shape[0]), 1.0 - source_rgb))
    trace = json.loads((root / parents["trace_path"]).read_text(encoding="utf-8"))
    bases = _fixed_bases(trace, wavelength)
    oracle_coefficients = {
        stock: _fit_basis_rows(density, basis)[0] for stock, basis in bases.items()
    }

    predictions = {stock: np.empty_like(density) for stock in STOCKS}
    direct_prediction = np.empty_like(density)
    mean_prediction = np.empty_like(density)
    oracle_prediction = oracle_coefficients[CORRECT_STOCK] @ bases[CORRECT_STOCK].T
    coverage = np.zeros(source_rgb.shape[0], dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    for held_set in sorted(set(test_sets.tolist())):
        for held_slide in sorted(set(slides.tolist())):
            development = (test_sets != held_set) & (slides != held_slide)
            confirmation = (test_sets == held_set) & (slides == held_slide)
            if int(np.sum(development)) != int(
                config["evaluation"]["expected_development_rows_per_fold"]
            ) or int(np.sum(confirmation)) != int(
                config["evaluation"]["expected_confirmation_rows_per_fold"]
            ):
                raise FujifilmDyeForwardError("CB5 fold support mismatch")
            coverage[confirmation] += 1
            for stock in STOCKS:
                mapping = _fit_nonnegative_mapping(
                    features[development], oracle_coefficients[stock][development]
                )
                predictions[stock][confirmation] = (
                    features[confirmation] @ mapping @ bases[stock].T
                )
            direct_mapping = _fit_nonnegative_mapping(
                features[development], density[development]
            )
            direct_prediction[confirmation] = features[confirmation] @ direct_mapping
            mean_prediction[confirmation] = np.mean(density[development], axis=0)
            fold_records.append(
                {
                    "held_test_set": held_set,
                    "held_slide": held_slide,
                    "development_rows": int(np.sum(development)),
                    "confirmation_rows": int(np.sum(confirmation)),
                }
            )
    if not np.all(coverage == 1):
        raise FujifilmDyeForwardError("CB5 confirmation folds do not partition rows")

    def rmse(prediction: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((prediction - density) ** 2, axis=1))

    errors = {stock: rmse(prediction) for stock, prediction in predictions.items()}
    correct = errors[CORRECT_STOCK]
    wrong = np.minimum(*(errors[stock] for stock in STOCKS if stock != CORRECT_STOCK))
    direct = rmse(direct_prediction)
    mean_error = rmse(mean_prediction)
    oracle_error = rmse(oracle_prediction)
    epsilon = np.finfo(np.float64).tiny
    comparisons = {
        "correct_basis_win_fraction_vs_best_wrong": float(np.mean(correct < wrong)),
        "correct_basis_relative_improvement_vs_best_wrong": _summary(
            (wrong - correct) / np.maximum(wrong, epsilon)
        ),
        "correct_basis_win_fraction_vs_direct_nonnegative_spectral": float(
            np.mean(correct < direct)
        ),
        "correct_basis_relative_improvement_vs_direct_nonnegative_spectral": _summary(
            (direct - correct) / np.maximum(direct, epsilon)
        ),
        "correct_basis_relative_improvement_vs_development_mean": _summary(
            (mean_error - correct) / np.maximum(mean_error, epsilon)
        ),
    }
    summaries = {
        "development-mean-spectrum": _summary(mean_error),
        "direct-nonnegative-spectral": _summary(direct),
        "target-oracle-correct-basis": _summary(oracle_error),
        **{stock: _summary(errors[stock]) for stock in STOCKS},
    }
    gates = config["gates"]
    checks = {
        "rows": density.shape[0] == int(gates["required_rows"]),
        "folds": len(fold_records) == int(gates["required_fold_count"]),
        "finite": bool(
            all(np.all(np.isfinite(row)) for row in predictions.values())
            and np.all(np.isfinite(direct_prediction))
        )
        is bool(gates["all_values_finite"]),
        "nonnegative": bool(
            all(np.all(row >= 0.0) for row in predictions.values())
            and np.all(direct_prediction >= 0.0)
        )
        is bool(gates["all_predicted_density_nonnegative"]),
        "wrong_basis": comparisons["correct_basis_win_fraction_vs_best_wrong"]
        >= float(gates["correct_basis_win_fraction_vs_best_wrong_minimum"])
        and comparisons["correct_basis_relative_improvement_vs_best_wrong"]["median"]
        >= float(gates["median_relative_improvement_vs_best_wrong_minimum"]),
        "direct_spectral": comparisons[
            "correct_basis_win_fraction_vs_direct_nonnegative_spectral"
        ]
        >= float(gates["win_fraction_vs_direct_nonnegative_spectral_minimum"])
        and comparisons[
            "correct_basis_relative_improvement_vs_direct_nonnegative_spectral"
        ]["median"]
        >= float(
            gates["median_relative_improvement_vs_direct_nonnegative_spectral_minimum"]
        ),
        "absolute_error": summaries[CORRECT_STOCK]["median"]
        <= float(gates["median_density_rmse_maximum"])
        and summaries[CORRECT_STOCK]["p95"] <= float(gates["p95_density_rmse_maximum"])
        and summaries[CORRECT_STOCK]["maximum"] <= float(gates["maximum_density_rmse"]),
        "mean_control": comparisons[
            "correct_basis_relative_improvement_vs_development_mean"
        ]["median"]
        >= float(gates["median_relative_improvement_vs_development_mean_minimum"]),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent_cb4_stable_evidence_id": parents["cb4_required_stable_evidence_id"],
        "source_rgb_sha256": _array_sha256(source_rgb),
        "measured_density_sha256": _array_sha256(density),
        "prediction_sha256": {
            **{stock: _array_sha256(row) for stock, row in predictions.items()},
            "direct-nonnegative-spectral": _array_sha256(direct_prediction),
            "development-mean-spectrum": _array_sha256(mean_prediction),
        },
        "folds": fold_records,
        "summaries": summaries,
        "comparisons": comparisons,
        "checks": checks,
        "failed_gates": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
        "decision": (
            "retain_source_only_fujifilm_dye_forward_proxy"
            if passed
            else "close_low_capacity_recorder_rgb_to_dye_forward_family"
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
    "FujifilmDyeForwardError",
    "_fit_nonnegative_mapping",
    "evaluate_forward_proxy",
    "load_contract",
    "write_report",
]
