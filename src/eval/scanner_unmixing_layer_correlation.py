"""Controlled layer-correlation identifiability under scanner mixing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "neuro_film.u6_p4cr_scanner_unmixing_layer_correlation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cr_scanner_unmixing_layer_correlation_report.v1"


class ScannerUnmixingCorrelationError(RuntimeError):
    """Raised when the frozen P4CR experiment drifts."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "contract_frozen_implementation_ready":
        raise ScannerUnmixingCorrelationError("P4CR contract identity drift")
    parent = payload["parents"]["p4cq_evidence"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        sha256_file(parent_path) != parent["sha256"]
        or parent_payload.get("decision") != parent["required_decision"]
    ):
        raise ScannerUnmixingCorrelationError("P4CR parent drift")
    simulation = payload["simulation"]
    if (
        simulation.get("shape") != [384, 512]
        or len(simulation.get("development_seeds", [])) != 4
        or len(simulation.get("confirmation_seeds", [])) != 4
        or set(simulation["development_seeds"]) & set(simulation["confirmation_seeds"])
    ):
        raise ScannerUnmixingCorrelationError("P4CR role drift")
    return payload


def _as_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ScannerUnmixingCorrelationError(f"invalid {name}")
    return matrix


def _correlation(values: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    residual -= np.mean(residual, axis=0, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(residual), axis=0, dtype=np.float64))
    if np.any(rms <= 0.0) or not np.all(np.isfinite(rms)):
        raise ScannerUnmixingCorrelationError("degenerate correlation observation")
    normalized = residual / rms
    matrix = normalized.T @ normalized / float(len(normalized))
    matrix = (matrix + matrix.T) * 0.5
    np.fill_diagonal(matrix, 1.0)
    return matrix


def _rmse(observed: np.ndarray, truth: np.ndarray) -> float:
    indexes = np.triu_indices(3, 1)
    return float(np.sqrt(np.mean(np.square(observed[indexes] - truth[indexes]))))


def _simulate(seed: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    simulation = contract["simulation"]
    shape = tuple(int(value) for value in simulation["shape"])
    truth = _as_matrix(simulation["layer_correlation"], "layer correlation")
    if np.min(np.linalg.eigvalsh(truth)) <= 0.0 or not np.allclose(np.diag(truth), 1.0):
        raise ScannerUnmixingCorrelationError("layer correlation is not positive definite")
    scanner = _as_matrix(simulation["scanner_matrix"], "scanner matrix")
    wrong = _as_matrix(simulation["wrong_scanner_matrix"], "wrong scanner matrix")
    condition = float(np.linalg.cond(scanner))
    gates = contract["metrics"]
    if not gates["minimum_scanner_condition_number"] <= condition <= gates["maximum_scanner_condition_number"]:
        raise ScannerUnmixingCorrelationError("scanner condition number drift")
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((*shape, 3))
    correlated = independent @ np.linalg.cholesky(truth).T
    density = np.asarray(simulation["mean_density_rgb"], dtype=np.float64) + (
        float(simulation["density_sigma"]) * correlated
    )
    transmittance = np.power(10.0, -density)
    scan = transmittance @ scanner.T
    if (
        not np.all(np.isfinite(density))
        or not np.all(np.isfinite(transmittance))
        or np.any(transmittance <= 0.0)
        or np.any(transmittance >= 1.0)
        or not np.all(np.isfinite(scan))
        or np.any(scan <= 0.0)
    ):
        raise ScannerUnmixingCorrelationError("simulated physical domain escaped")
    recovered_transmittance = scan @ np.linalg.inv(scanner).T
    wrong_transmittance = scan @ np.linalg.inv(wrong).T
    if np.any(recovered_transmittance <= 0.0) or np.any(wrong_transmittance <= 0.0):
        raise ScannerUnmixingCorrelationError("scanner inverse left positive domain")
    recovered_density = -np.log10(recovered_transmittance)
    wrong_density = -np.log10(wrong_transmittance)
    observations = {
        "layer_density": _correlation(density),
        "scanner_rgb": _correlation(scan),
        "correct_inverse_density": _correlation(recovered_density),
        "wrong_inverse_density": _correlation(wrong_density),
    }
    return {
        "seed": seed,
        "scanner_condition_number": condition,
        "correlations": {name: value.tolist() for name, value in observations.items()},
        "errors": {name: _rmse(value, truth) for name, value in observations.items()},
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = tuple(rows[0]["errors"])
    medians = {
        name: float(np.median([float(row["errors"][name]) for row in rows]))
        for name in names
    }
    return {"rows": list(rows), "median_errors": medians}


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    simulation = contract["simulation"]
    development = _aggregate([_simulate(int(seed), contract) for seed in simulation["development_seeds"]])
    confirmation = _aggregate([_simulate(int(seed), contract) for seed in simulation["confirmation_seeds"]])
    dev = development["median_errors"]
    conf = confirmation["median_errors"]
    metrics = contract["metrics"]
    correct = conf["correct_inverse_density"]
    direct = conf["scanner_rgb"]
    wrong = conf["wrong_inverse_density"]
    gates = {
        "correct_inverse_rmse": correct <= metrics["maximum_correct_inverse_rmse"],
        "direct_scanner_confounded": direct >= metrics["minimum_direct_scanner_rmse"],
        "wrong_inverse_confounded": wrong >= metrics["minimum_wrong_inverse_rmse"],
        "correct_beats_direct": (direct - correct) / direct >= metrics["minimum_correct_improvement_over_direct"],
        "correct_beats_wrong": (wrong - correct) / wrong >= metrics["minimum_correct_improvement_over_wrong"],
        "development_confirmation_stable": max(abs(float(dev[name]) - float(conf[name])) for name in conf) <= metrics["maximum_development_confirmation_rmse_gap"],
        "finite": all(np.isfinite(value) for value in (*dev.values(), *conf.values())),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "development_median_errors": dev,
        "confirmation_median_errors": conf,
        "gates": gates,
        "decision": contract["decision_if_pass"] if passed else contract["decision_if_fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {
        "schema": REPORT_SCHEMA,
        "automatic_pass": passed,
        "stable_evidence_id": hashlib.sha256(canonical_json(stable)).hexdigest(),
        "stable": stable,
        "development": development,
        "confirmation": confirmation,
    }


__all__ = ["ScannerUnmixingCorrelationError", "canonical_json", "evaluate", "load_contract"]
