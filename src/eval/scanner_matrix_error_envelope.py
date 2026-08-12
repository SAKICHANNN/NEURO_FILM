"""Scanner-matrix calibration-error sensitivity for layer correlation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file

SCHEMA = "neuro_film.u6_p4cs_scanner_matrix_error_envelope_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cs_scanner_matrix_error_envelope_report.v1"


class ScannerMatrixEnvelopeError(RuntimeError):
    """Raised when the frozen P4CS experiment is invalid."""


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise ScannerMatrixEnvelopeError(f"invalid {name}")
    return result


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "contract_frozen_implementation_ready":
        raise ScannerMatrixEnvelopeError("P4CS contract identity drift")
    parent = payload["parents"]["p4cr_evidence"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if sha256_file(parent_path) != parent["sha256"] or parent_payload.get("decision") != parent["required_decision"]:
        raise ScannerMatrixEnvelopeError("P4CS parent drift")
    simulation = payload["simulation"]
    levels = simulation.get("relative_frobenius_error_levels", [])
    directions = simulation.get("zero_row_sum_perturbation_directions", [])
    if (
        levels != sorted(set(levels))
        or not levels
        or levels[0] != 0.0
        or len(directions) != 3
        or set(simulation["development_seeds"]) & set(simulation["confirmation_seeds"])
    ):
        raise ScannerMatrixEnvelopeError("P4CS role or grid drift")
    for direction in directions:
        matrix = _matrix(direction, "perturbation direction")
        if not np.allclose(np.sum(matrix, axis=1), 0.0, atol=1e-15) or np.linalg.norm(matrix) <= 0.0:
            raise ScannerMatrixEnvelopeError("perturbation direction does not preserve row sums")
    return payload


def _correlation(values: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    residual -= np.mean(residual, axis=0, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(residual), axis=0, dtype=np.float64))
    if np.any(rms <= 0.0) or not np.all(np.isfinite(rms)):
        raise ScannerMatrixEnvelopeError("degenerate correlation observation")
    normalized = residual / rms
    result = normalized.T @ normalized / float(len(normalized))
    result = (result + result.T) * 0.5
    np.fill_diagonal(result, 1.0)
    return result


def _rmse(observed: np.ndarray, truth: np.ndarray) -> float:
    indexes = np.triu_indices(3, 1)
    return float(np.sqrt(np.mean(np.square(observed[indexes] - truth[indexes]))))


def _seed_rows(seed: int, contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    simulation = contract["simulation"]
    shape = tuple(int(value) for value in simulation["shape"])
    truth = _matrix(simulation["layer_correlation"], "layer correlation")
    scanner = _matrix(simulation["true_scanner_matrix"], "true scanner matrix")
    rng = np.random.default_rng(seed)
    correlated = rng.standard_normal((*shape, 3)) @ np.linalg.cholesky(truth).T
    density = np.asarray(simulation["mean_density_rgb"], dtype=np.float64) + float(simulation["density_sigma"]) * correlated
    transmittance = np.power(10.0, -density)
    scan = transmittance @ scanner.T
    scanner_norm = float(np.linalg.norm(scanner))
    rows: list[dict[str, Any]] = []
    for level_index, level in enumerate(simulation["relative_frobenius_error_levels"]):
        for direction_index, raw_direction in enumerate(simulation["zero_row_sum_perturbation_directions"]):
            direction = _matrix(raw_direction, "perturbation direction")
            direction /= np.linalg.norm(direction)
            estimate = scanner + float(level) * scanner_norm * direction
            if np.any(estimate <= 0.0) or not np.isfinite(np.linalg.det(estimate)) or abs(np.linalg.det(estimate)) <= 1e-12:
                raise ScannerMatrixEnvelopeError("perturbed scanner left positive invertible domain")
            recovered_t = scan @ np.linalg.inv(estimate).T
            if np.any(recovered_t <= 0.0) or not np.all(np.isfinite(recovered_t)):
                raise ScannerMatrixEnvelopeError("perturbed inverse left transmittance domain")
            recovered_density = -np.log10(recovered_t)
            rows.append(
                {
                    "seed": seed,
                    "level_index": level_index,
                    "relative_frobenius_error": float(level),
                    "direction_index": direction_index,
                    "matrix_condition_number": float(np.linalg.cond(estimate)),
                    "correlation_rmse": _rmse(_correlation(recovered_density), truth),
                }
            )
    return rows


def _aggregate(rows: Sequence[Mapping[str, Any]], levels: Sequence[float]) -> list[dict[str, Any]]:
    result = []
    for index, level in enumerate(levels):
        errors = [float(row["correlation_rmse"]) for row in rows if int(row["level_index"]) == index]
        result.append(
            {
                "level_index": index,
                "relative_frobenius_error": float(level),
                "median_correlation_rmse": float(np.median(errors)),
                "worst_correlation_rmse": max(errors),
            }
        )
    return result


def _largest_passing_index(summary: Sequence[Mapping[str, Any]], maximum: float) -> int:
    passing = [int(row["level_index"]) for row in summary if float(row["worst_correlation_rmse"]) <= maximum]
    if not passing:
        raise ScannerMatrixEnvelopeError("even the exact scanner matrix failed")
    return max(passing)


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    simulation = contract["simulation"]
    levels = simulation["relative_frobenius_error_levels"]
    development_rows = [row for seed in simulation["development_seeds"] for row in _seed_rows(int(seed), contract)]
    development_summary = _aggregate(development_rows, levels)
    metrics = contract["metrics"]
    selected_index = _largest_passing_index(development_summary, metrics["maximum_development_selected_worst_rmse"])
    confirmation_rows = [row for seed in simulation["confirmation_seeds"] for row in _seed_rows(int(seed), contract)]
    confirmation_summary = _aggregate(confirmation_rows, levels)
    confirmation_index = _largest_passing_index(confirmation_summary, metrics["maximum_confirmation_selected_worst_rmse"])
    selected = confirmation_summary[selected_index]
    next_row = confirmation_summary[selected_index + 1] if selected_index + 1 < len(confirmation_summary) else None
    gates = {
        "development_selected_worst": development_summary[selected_index]["worst_correlation_rmse"] <= metrics["maximum_development_selected_worst_rmse"],
        "confirmation_selected_median": selected["median_correlation_rmse"] <= metrics["maximum_confirmation_selected_median_rmse"],
        "confirmation_selected_worst": selected["worst_correlation_rmse"] <= metrics["maximum_confirmation_selected_worst_rmse"],
        "selected_index_stable": abs(selected_index - confirmation_index) <= metrics["maximum_selected_level_index_gap"],
        "next_level_discriminates": next_row is None or next_row["worst_correlation_rmse"] > metrics["maximum_confirmation_selected_worst_rmse"],
        "finite": all(np.isfinite(float(row["correlation_rmse"])) for row in (*development_rows, *confirmation_rows)),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "development_selected_level_index": selected_index,
        "development_selected_relative_frobenius_error": float(levels[selected_index]),
        "confirmation_largest_passing_level_index": confirmation_index,
        "development_summary": development_summary,
        "confirmation_summary": confirmation_summary,
        "gates": gates,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "development_rows": development_rows,
        "confirmation_rows": confirmation_rows,
    }


__all__ = ["ScannerMatrixEnvelopeError", "evaluate", "load_contract"]
