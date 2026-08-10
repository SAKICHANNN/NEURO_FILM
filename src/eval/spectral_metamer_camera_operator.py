"""Physical spectral fixture for white-point descriptor identifiability."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class SpectralMetamerError(ValueError):
    """Raised when the frozen spectral fixture or its bindings drift."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path: Path, root: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u5_r2bx1_spectral_metamer_camera_operator_contract.v1"
        or payload.get("experiment_id") != "U5.R2BX1"
        or payload.get("status") != "contract_frozen_implementation_ready"
    ):
        raise SpectralMetamerError("contract identity drift")
    fixture = payload.get("fixture", {})
    gates = payload.get("gates", {})
    if (
        fixture.get("development_reflectance_count") != 384
        or fixture.get("confirmation_reflectance_count") != 192
        or fixture.get("null_amplitude_fraction_of_positive_limit") != 0.8
        or gates.get("minimum_shared_confirmation_error_penalty_median") != 0.1
    ):
        raise SpectralMetamerError("contract boundary drift")
    parent = payload["parent"]
    parent_path = root / parent["decision"]
    if not parent_path.is_file() or _sha256(parent_path) != parent["sha256"]:
        raise SpectralMetamerError("parent decision drift")
    parent_payload = json.loads(parent_path.read_text(encoding="utf-8"))
    if parent_payload.get("decision") != parent["required_decision"]:
        raise SpectralMetamerError("parent decision status drift")
    return payload


def _gaussian(wavelengths: np.ndarray, center: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * np.square((wavelengths - center) / sigma))


def _observer_curves(wavelengths: np.ndarray) -> np.ndarray:
    x = _gaussian(wavelengths, 600.0, 42.0) + 0.28 * _gaussian(
        wavelengths, 445.0, 24.0
    )
    y = _gaussian(wavelengths, 550.0, 36.0)
    z = 1.18 * _gaussian(wavelengths, 445.0, 28.0)
    curves = np.stack([x, y, z], axis=0)
    return curves / np.sum(curves, axis=1, keepdims=True)


def _camera_curves(wavelengths: np.ndarray) -> np.ndarray:
    red = _gaussian(wavelengths, 612.0, 31.0) + 0.10 * _gaussian(
        wavelengths, 500.0, 35.0
    )
    green = _gaussian(wavelengths, 538.0, 28.0) + 0.08 * _gaussian(
        wavelengths, 625.0, 30.0
    )
    blue = _gaussian(wavelengths, 462.0, 25.0) + 0.12 * _gaussian(
        wavelengths, 550.0, 32.0
    )
    curves = np.stack([red, green, blue], axis=1)
    return curves / np.sum(curves, axis=0, keepdims=True)


def _metameric_illuminants(
    wavelengths: np.ndarray,
    observer: np.ndarray,
    cycles: list[float],
    minimum: float,
    amplitude_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = (wavelengths - wavelengths[0]) / (wavelengths[-1] - wavelengths[0])
    base = 1.0 + 0.18 * np.cos(np.pi * phase) + 0.12 * np.sin(2.0 * np.pi * phase)
    base = base / np.mean(base)
    seed = sum(
        np.sin(2.0 * np.pi * float(cycle) * phase + 0.31 * index)
        for index, cycle in enumerate(cycles)
    )
    projection = observer.T @ np.linalg.solve(observer @ observer.T, observer @ seed)
    null = seed - projection
    null = null / np.max(np.abs(null))
    positive_limit = float(np.min((base - minimum) / np.maximum(np.abs(null), 1e-15)))
    amplitude = amplitude_fraction * positive_limit
    first = base + amplitude * null
    second = base - amplitude * null
    if np.any(first < minimum) or np.any(second < minimum):
        raise SpectralMetamerError("metameric illuminant lost positivity")
    return base, first, second


def _reflectances(
    wavelengths: np.ndarray,
    count: int,
    basis_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.linspace(wavelengths[1], wavelengths[-2], basis_count)
    widths = np.linspace(18.0, 58.0, basis_count)
    basis = np.stack(
        [_gaussian(wavelengths, center, width) for center, width in zip(centers, widths)],
        axis=1,
    )
    coefficients = rng.normal(0.0, 1.15, size=(count, basis_count))
    slopes = rng.normal(0.0, 0.65, size=(count, 1))
    phase = (wavelengths - np.mean(wavelengths)) / np.ptp(wavelengths)
    logits = coefficients @ basis.T / np.sqrt(basis_count) + slopes * phase[None, :]
    return 0.02 + 0.96 / (1.0 + np.exp(-logits))


def _camera_response(
    reflectances: np.ndarray, illuminant: np.ndarray, camera: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    white = illuminant @ camera
    response = (reflectances * illuminant[None, :]) @ camera
    return response / white[None, :], white


def _observer_target(
    reflectances: np.ndarray, illuminant: np.ndarray, observer: np.ndarray
) -> np.ndarray:
    white = observer @ illuminant
    response = (reflectances * illuminant[None, :]) @ observer.T
    return response / white[None, :]


def _fit_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(source, target, rcond=None)[0]


def _rmse(source: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(source - target), dtype=np.float64)))


def evaluate_contract(contract_path: Path, root: Path) -> dict[str, Any]:
    config = load_contract(contract_path, root)
    fixture = config["fixture"]
    gates = config["gates"]
    wavelengths = np.arange(
        fixture["wavelength_start_nm"],
        fixture["wavelength_stop_nm"] + fixture["wavelength_step_nm"],
        fixture["wavelength_step_nm"],
        dtype=np.float64,
    )
    observer = _observer_curves(wavelengths)
    camera = _camera_curves(wavelengths)
    base, first, second = _metameric_illuminants(
        wavelengths,
        observer,
        [float(value) for value in fixture["null_seed_cycles"]],
        float(fixture["illuminant_minimum"]),
        float(fixture["null_amplitude_fraction_of_positive_limit"]),
    )
    illuminants = [first, second]
    white_xyz = [observer @ illuminant for illuminant in illuminants]
    chromaticities = [value / np.sum(value) for value in white_xyz]
    rng = np.random.default_rng(int(fixture["seed"]))
    development = _reflectances(
        wavelengths,
        int(fixture["development_reflectance_count"]),
        int(fixture["reflectance_basis_count"]),
        rng,
    )
    confirmation = _reflectances(
        wavelengths,
        int(fixture["confirmation_reflectance_count"]),
        int(fixture["reflectance_basis_count"]),
        rng,
    )
    development_target = _observer_target(development, base, observer)
    development_sources = []
    camera_whites = []
    oracle_matrices = []
    for illuminant in illuminants:
        source, camera_white = _camera_response(development, illuminant, camera)
        development_sources.append(source)
        camera_whites.append(camera_white)
        oracle_matrices.append(_fit_matrix(source, development_target))
    shared_matrix = _fit_matrix(
        np.concatenate(development_sources, axis=0),
        np.concatenate([development_target, development_target], axis=0),
    )

    confirmation_target_reads_before_operator_freeze = 0
    confirmation_target = _observer_target(confirmation, base, observer)
    rows = []
    for index, illuminant in enumerate(illuminants):
        source, _ = _camera_response(confirmation, illuminant, camera)
        oracle_rmse = _rmse(source @ oracle_matrices[index], confirmation_target)
        shared_rmse = _rmse(source @ shared_matrix, confirmation_target)
        rows.append(
            {
                "illuminant_index": index,
                "oracle_confirmation_rmse": oracle_rmse,
                "shared_confirmation_rmse": shared_rmse,
                "shared_error_penalty": shared_rmse / oracle_rmse - 1.0,
            }
        )
    white_xyz_difference = float(np.max(np.abs(white_xyz[0] - white_xyz[1])))
    chromaticity_difference = float(
        np.max(np.abs(chromaticities[0] - chromaticities[1]))
    )
    illuminant_relative_rms = _rmse(first, second) / float(
        np.sqrt(np.mean(np.square(base)))
    )
    camera_white_relative_difference = _rmse(camera_whites[0], camera_whites[1]) / float(
        np.sqrt(np.mean(np.square(np.mean(camera_whites, axis=0))))
    )
    operator_relative_distance = _rmse(oracle_matrices[0], oracle_matrices[1]) / float(
        np.sqrt(np.mean(np.square(np.mean(oracle_matrices, axis=0))))
    )
    penalties = [row["shared_error_penalty"] for row in rows]
    gate_results = {
        "observer_white_xyz": white_xyz_difference
        <= gates["maximum_observer_white_xyz_difference"],
        "observer_chromaticity": chromaticity_difference
        <= gates["maximum_observer_chromaticity_difference"],
        "illuminant_separation": illuminant_relative_rms
        >= gates["minimum_illuminant_relative_rms_difference"],
        "camera_white_separation": camera_white_relative_difference
        >= gates["minimum_camera_white_relative_difference"],
        "oracle_quality": all(
            row["oracle_confirmation_rmse"]
            <= gates["maximum_oracle_confirmation_rmse_each"]
            for row in rows
        ),
        "operator_separation": operator_relative_distance
        >= gates["minimum_operator_relative_frobenius_distance"],
        "shared_penalty_each": all(
            penalty >= gates["minimum_shared_confirmation_error_penalty_each"]
            for penalty in penalties
        ),
        "shared_penalty_median": float(np.median(penalties))
        >= gates["minimum_shared_confirmation_error_penalty_median"],
        "confirmation_target_isolation": confirmation_target_reads_before_operator_freeze
        <= gates["maximum_target_reads_before_operator_freeze"],
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2bx1_spectral_metamer_camera_operator_result.v1",
        "experiment_id": config["experiment_id"],
        "contract": contract_path.relative_to(root).as_posix(),
        "contract_sha256": _sha256(contract_path),
        "primary_source": config["primary_source"],
        "wavelength_count": len(wavelengths),
        "observer_white_xyz_difference": white_xyz_difference,
        "observer_chromaticity_difference": chromaticity_difference,
        "illuminant_minimum": float(min(np.min(first), np.min(second))),
        "illuminant_relative_rms_difference": illuminant_relative_rms,
        "camera_white_relative_difference": camera_white_relative_difference,
        "operator_relative_frobenius_distance": operator_relative_distance,
        "confirmation_target_reads_before_operator_freeze": confirmation_target_reads_before_operator_freeze,
        "rows": rows,
        "shared_error_penalty_median": float(np.median(penalties)),
        "gate_results": gate_results,
        "passed": passed,
        "decision": (
            "retain_synthetic_spectral_descriptor_insufficiency"
            if passed
            else "close_exact_spectral_metamer_fixture"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(
        json.dumps(report, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return report


def write_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
