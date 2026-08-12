"""Signal-dependent broad-PSF scanner-flare correlation control."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file

SCHEMA = "neuro_film.u6_p4cv_signal_dependent_scanner_flare_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cv_signal_dependent_scanner_flare_report.v1"


class SignalDependentScannerFlareError(RuntimeError):
    """Raised when the frozen P4CV experiment drifts."""


def _matrix(value: Any, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3, 3) or not np.all(np.isfinite(result)):
        raise SignalDependentScannerFlareError(f"invalid {name}")
    return result


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("status") != "contract_frozen_implementation_ready":
        raise SignalDependentScannerFlareError("P4CV contract identity drift")
    parent = payload["parents"]["p4cu_evidence"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if sha256_file(parent_path) != parent["sha256"] or parent_payload.get("decision") != parent["required_decision"]:
        raise SignalDependentScannerFlareError("P4CV parent drift")
    simulation = payload["simulation"]
    if (
        simulation.get("shape") != [384, 512]
        or set(simulation["development_seeds"]) & set(simulation["confirmation_seeds"])
        or not 0.0 < float(simulation["wrong_psf_sigma_pixels"]) < float(simulation["gaussian_sigma_pixels"])
    ):
        raise SignalDependentScannerFlareError("P4CV role drift")
    return payload


def _correlation(values: np.ndarray) -> np.ndarray:
    residual = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    residual -= np.mean(residual, axis=0, dtype=np.float64)
    rms = np.sqrt(np.mean(np.square(residual), axis=0, dtype=np.float64))
    if np.any(rms <= 0.0) or not np.all(np.isfinite(rms)):
        raise SignalDependentScannerFlareError("degenerate correlation observation")
    normalized = residual / rms
    result = normalized.T @ normalized / float(len(normalized))
    result = (result + result.T) * 0.5
    np.fill_diagonal(result, 1.0)
    return result


def _rmse(observed: np.ndarray, truth: np.ndarray) -> float:
    indexes = np.triu_indices(3, 1)
    return float(np.sqrt(np.mean(np.square(observed[indexes] - truth[indexes]))))


def _gaussian_blur_circular(image: np.ndarray, sigma: float) -> np.ndarray:
    height, width, _ = image.shape
    fy = np.fft.fftfreq(height)[:, None]
    fx = np.fft.rfftfreq(width)[None, :]
    response = np.exp(-2.0 * np.pi**2 * sigma**2 * (fy * fy + fx * fx))
    transformed = np.fft.rfft2(image, axes=(0, 1))
    return np.fft.irfft2(transformed * response[..., None], s=(height, width), axes=(0, 1))


def _simulate(seed: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    simulation = contract["simulation"]
    truth = _matrix(simulation["layer_correlation"], "layer correlation")
    scanner = _matrix(simulation["scanner_matrix"], "scanner matrix")
    rng = np.random.default_rng(seed)
    independent = rng.standard_normal((*simulation["shape"], 3))
    density = np.asarray(simulation["mean_density_rgb"], dtype=np.float64) + float(simulation["density_sigma"]) * (
        independent @ np.linalg.cholesky(truth).T
    )
    clean_scan = np.power(10.0, -density) @ scanner.T
    fraction = float(simulation["flare_fraction"])
    true_source = _gaussian_blur_circular(clean_scan, float(simulation["gaussian_sigma_pixels"]))
    wrong_source = _gaussian_blur_circular(clean_scan, float(simulation["wrong_psf_sigma_pixels"]))
    flared_scan = clean_scan + fraction * true_source
    inverse = np.linalg.inv(scanner)

    def recover(subtracted: np.ndarray) -> np.ndarray:
        corrected_scan = flared_scan - fraction * subtracted
        recovered_t = corrected_scan @ inverse.T
        if np.any(corrected_scan <= 0.0) or np.any(recovered_t <= 0.0) or not np.all(np.isfinite(recovered_t)):
            raise SignalDependentScannerFlareError("flare correction left physical domain")
        return -np.log10(recovered_t)

    observations = {
        "correct_psf": recover(true_source),
        "uncorrected_psf": recover(np.zeros_like(clean_scan)),
        "wrong_psf": recover(wrong_source),
    }
    return {
        "seed": seed,
        "errors": {name: _rmse(_correlation(value), truth) for name, value in observations.items()},
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = tuple(rows[0]["errors"])
    return {
        "rows": list(rows),
        "median_errors": {name: float(np.median([float(row["errors"][name]) for row in rows])) for name in names},
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    simulation = contract["simulation"]
    development = _aggregate([_simulate(int(seed), contract) for seed in simulation["development_seeds"]])
    confirmation = _aggregate([_simulate(int(seed), contract) for seed in simulation["confirmation_seeds"]])
    dev, conf = development["median_errors"], confirmation["median_errors"]
    metrics = contract["metrics"]
    correct, uncorrected, wrong = conf["correct_psf"], conf["uncorrected_psf"], conf["wrong_psf"]
    gates = {
        "corrected_recovers": correct <= metrics["maximum_corrected_correlation_rmse"],
        "uncorrected_is_confounded": uncorrected >= metrics["minimum_uncorrected_correlation_rmse"],
        "wrong_psf_is_confounded": wrong >= metrics["minimum_wrong_psf_correlation_rmse"],
        "corrected_beats_uncorrected": (uncorrected - correct) / uncorrected >= metrics["minimum_corrected_improvement_over_uncorrected"],
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


__all__ = ["SignalDependentScannerFlareError", "evaluate", "load_contract"]
