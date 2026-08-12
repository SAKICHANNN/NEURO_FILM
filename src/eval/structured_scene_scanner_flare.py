"""Structured-scene scanner-flare test for layer residual correlation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.scanner_unmixing_layer_correlation import canonical_json, sha256_file
from src.eval.signal_dependent_scanner_flare import (
    _correlation,
    _gaussian_blur_circular,
    _matrix,
    _rmse,
)

SCHEMA = "neuro_film.u6_p4cw_structured_scene_scanner_flare_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cw_structured_scene_scanner_flare_report.v1"


class StructuredSceneScannerFlareError(RuntimeError):
    """Raised when the frozen P4CW experiment drifts."""


def load_contract(root: Path, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise StructuredSceneScannerFlareError("P4CW contract identity drift")
    parent = payload["parents"]["p4cv_evidence"]
    parent_path = root / parent["path"]
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        sha256_file(parent_path) != parent["sha256"]
        or parent_payload.get("decision") != parent["required_decision"]
    ):
        raise StructuredSceneScannerFlareError("P4CW parent drift")
    simulation = payload["simulation"]
    if (
        simulation.get("shape") != [384, 512]
        or set(simulation["development_seeds"]) & set(simulation["confirmation_seeds"])
        or len(simulation["scene_frequencies_xy"]) != len(simulation["scene_weights"])
        or not np.isclose(sum(simulation["scene_weights"]), 1.0)
    ):
        raise StructuredSceneScannerFlareError("P4CW role drift")
    return payload


def _scene_density(seed: int, simulation: Mapping[str, Any]) -> np.ndarray:
    height, width = (int(value) for value in simulation["shape"])
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    yy, xx = np.meshgrid(y, x, indexing="ij")
    rng = np.random.default_rng(seed ^ 0x50444357)
    channels = []
    for channel in range(3):
        field = np.zeros((height, width), dtype=np.float64)
        for weight, (fx, fy) in zip(
            simulation["scene_weights"], simulation["scene_frequencies_xy"], strict=True
        ):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            angle = (
                2.0
                * np.pi
                * (
                    (float(fx) + 0.25 * channel) * xx
                    + (float(fy) + 0.125 * channel) * yy
                )
                + phase
            )
            field += float(weight) * np.cos(angle)
        field -= np.mean(field, dtype=np.float64)
        field /= np.max(np.abs(field))
        channels.append(field)
    return np.stack(channels, axis=-1)


def _simulate(seed: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    simulation = contract["simulation"]
    truth = _matrix(simulation["layer_correlation"], "layer correlation")
    scanner = _matrix(simulation["scanner_matrix"], "scanner matrix")
    scene = _scene_density(seed, simulation)
    baseline = np.asarray(
        simulation["mean_density_rgb"], dtype=np.float64
    ) + scene * np.asarray(simulation["scene_density_amplitude_rgb"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    grain = float(simulation["grain_density_sigma"]) * (
        rng.standard_normal((*simulation["shape"], 3)) @ np.linalg.cholesky(truth).T
    )
    density = baseline + grain
    clean_scan = np.power(10.0, -density) @ scanner.T
    fraction = float(simulation["flare_fraction"])
    true_source = _gaussian_blur_circular(
        clean_scan, float(simulation["gaussian_sigma_pixels"])
    )
    wrong_source = _gaussian_blur_circular(
        clean_scan, float(simulation["wrong_psf_sigma_pixels"])
    )
    flared_scan = clean_scan + fraction * true_source
    inverse = np.linalg.inv(scanner)

    def recover(subtracted: np.ndarray) -> np.ndarray:
        corrected_scan = flared_scan - fraction * subtracted
        recovered_t = corrected_scan @ inverse.T
        if (
            np.any(corrected_scan <= 0.0)
            or np.any(recovered_t <= 0.0)
            or not np.all(np.isfinite(recovered_t))
        ):
            raise StructuredSceneScannerFlareError(
                "flare correction left physical domain"
            )
        return -np.log10(recovered_t) - baseline

    observations = {
        "correct_psf": recover(true_source),
        "uncorrected_psf": recover(np.zeros_like(clean_scan)),
        "wrong_psf": recover(wrong_source),
    }
    return {
        "seed": seed,
        "errors": {
            name: _rmse(_correlation(value), truth)
            for name, value in observations.items()
        },
    }


def _aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = tuple(rows[0]["errors"])
    return {
        "rows": list(rows),
        "median_errors": {
            name: float(np.median([float(row["errors"][name]) for row in rows]))
            for name in names
        },
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract = load_contract(root, contract_path)
    simulation = contract["simulation"]
    development = _aggregate(
        [_simulate(int(seed), contract) for seed in simulation["development_seeds"]]
    )
    confirmation = _aggregate(
        [_simulate(int(seed), contract) for seed in simulation["confirmation_seeds"]]
    )
    dev, conf = development["median_errors"], confirmation["median_errors"]
    metrics = contract["metrics"]
    correct, uncorrected, wrong = (
        conf["correct_psf"],
        conf["uncorrected_psf"],
        conf["wrong_psf"],
    )
    gates = {
        "corrected_recovers": correct <= metrics["maximum_corrected_correlation_rmse"],
        "uncorrected_is_confounded": uncorrected
        >= metrics["minimum_uncorrected_correlation_rmse"],
        "wrong_psf_is_confounded": wrong
        >= metrics["minimum_wrong_psf_correlation_rmse"],
        "corrected_beats_uncorrected": (uncorrected - correct) / uncorrected
        >= metrics["minimum_corrected_improvement_over_uncorrected"],
        "development_confirmation_stable": max(
            abs(float(dev[name]) - float(conf[name])) for name in conf
        )
        <= metrics["maximum_development_confirmation_rmse_gap"],
        "finite": all(np.isfinite(value) for value in (*dev.values(), *conf.values())),
    }
    passed = all(gates.values())
    stable = {
        "contract_sha256": sha256_file(contract_path),
        "development_median_errors": dev,
        "confirmation_median_errors": conf,
        "gates": gates,
        "decision": contract["decision_if_pass"]
        if passed
        else contract["decision_if_fail"],
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


__all__ = ["StructuredSceneScannerFlareError", "evaluate", "load_contract"]
