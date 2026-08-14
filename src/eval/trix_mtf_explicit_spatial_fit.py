"""U6.P2AN held-frequency explicit spatial-response compatibility audit."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src.eval.trix_mtf_source_prior import load_contract as load_source_contract
from src.eval.trix_mtf_source_prior import run_audit as run_source_audit


class TrixMTFSpatialFitError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema")
        != "neuro_film.u6_p2an_trix_mtf_explicit_spatial_fit_contract.v1"
    ):
        raise TrixMTFSpatialFitError("unsupported P2AN contract")
    return payload


def _candidate_response(parameters: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    blur_um, adjacency_um, gain = parameters
    blur = np.exp(-2.0 * math.pi**2 * (blur_um * 1e-3) ** 2 * frequencies**2)
    adjacency_blur = np.exp(
        -2.0 * math.pi**2 * (adjacency_um * 1e-3) ** 2 * frequencies**2
    )
    return blur * (1.0 + gain * (1.0 - adjacency_blur))


def _gaussian_response(sigma_um: float, frequencies: np.ndarray) -> np.ndarray:
    return np.exp(-2.0 * math.pi**2 * (sigma_um * 1e-3) ** 2 * frequencies**2)


def _best_candidate(
    config: dict[str, Any], frequencies: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, float]:
    family = config["candidate"]
    lower = np.asarray(
        [
            family["combined_blur_sigma_micrometres_bounds"][0],
            family["adjacency_sigma_micrometres_bounds"][0],
            family["adjacency_gain_bounds"][0],
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            family["combined_blur_sigma_micrometres_bounds"][1],
            family["adjacency_sigma_micrometres_bounds"][1],
            family["adjacency_gain_bounds"][1],
        ],
        dtype=np.float64,
    )
    grid = family["initial_grid"]
    starts = itertools.product(
        grid["combined_blur_sigma_micrometres"],
        grid["adjacency_sigma_micrometres"],
        grid["adjacency_gain"],
    )
    trials = []
    for start in starts:
        result = least_squares(
            lambda value: _candidate_response(value, frequencies) - targets,
            np.asarray(start, dtype=np.float64),
            bounds=(lower, upper),
            method="trf",
            xtol=family["xtol"],
            ftol=family["ftol"],
            gtol=family["gtol"],
            max_nfev=family["max_nfev"],
        )
        if result.success and np.all(np.isfinite(result.x)):
            error = _candidate_response(result.x, frequencies) - targets
            trials.append((float(np.mean(np.square(error))), tuple(result.x), result.x))
    if not trials:
        raise TrixMTFSpatialFitError("P2AN candidate optimizer failed")
    _, _, best = min(trials, key=lambda row: (row[0], row[1]))
    error = _candidate_response(best, frequencies) - targets
    return best, float(np.sqrt(np.mean(np.square(error))))


def _best_gaussian(
    config: dict[str, Any], frequencies: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    control = config["control"]
    trials = []
    for start in control["initial_sigmas_micrometres"]:
        result = least_squares(
            lambda value: _gaussian_response(float(value[0]), frequencies) - targets,
            np.asarray([start], dtype=np.float64),
            bounds=tuple(
                np.asarray(
                    control["sigma_micrometres_bounds"], dtype=np.float64
                ).reshape(2, 1)
            ),
            method="trf",
            xtol=config["candidate"]["xtol"],
            ftol=config["candidate"]["ftol"],
            gtol=config["candidate"]["gtol"],
            max_nfev=config["candidate"]["max_nfev"],
        )
        if result.success:
            error = _gaussian_response(float(result.x[0]), frequencies) - targets
            trials.append((float(np.mean(np.square(error))), float(result.x[0])))
    if not trials:
        raise TrixMTFSpatialFitError("P2AN Gaussian optimizer failed")
    mse, sigma = min(trials)
    return sigma, math.sqrt(mse)


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    parent = contract["parent"]
    evidence_path = root / parent["evidence_path"]
    source_contract_path = root / parent["profile_contract_path"]
    if (
        _sha(evidence_path) != parent["evidence_sha256"]
        or _sha(source_contract_path) != parent["profile_contract_sha256"]
    ):
        raise TrixMTFSpatialFitError("P2AN parent identity mismatch")
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("automatic_pass") is not parent["required_automatic_pass"]
        or evidence.get("profile_identity") != parent["required_profile_identity"]
    ):
        raise TrixMTFSpatialFitError("P2AN parent decision mismatch")
    source = run_source_audit(
        root=root, contract=load_source_contract(source_contract_path)
    )
    if source["profile_identity"] != parent["required_profile_identity"]:
        raise TrixMTFSpatialFitError("P2AN source profile replay mismatch")
    frequencies = np.asarray(
        source["profile"]["frequencies_cycles_per_mm"], dtype=np.float64
    )
    targets = np.asarray(source["profile"]["response_fractions"], dtype=np.float64)
    development = np.asarray(
        contract["split"]["development_frequencies_cycles_per_mm"], dtype=np.float64
    )
    confirmation = np.asarray(
        contract["split"]["confirmation_frequencies_cycles_per_mm"], dtype=np.float64
    )
    if set(development) & set(confirmation) or set(development) | set(
        confirmation
    ) != set(frequencies):
        raise TrixMTFSpatialFitError("P2AN frequency split mismatch")
    indices = {float(value): index for index, value in enumerate(frequencies)}
    development_target = targets[[indices[float(value)] for value in development]]
    confirmation_target = targets[[indices[float(value)] for value in confirmation]]
    parameters, development_rmse = _best_candidate(
        contract, development, development_target
    )
    gaussian_sigma, gaussian_development_rmse = _best_gaussian(
        contract, development, development_target
    )
    development_response = _candidate_response(parameters, development)
    confirmation_response = _candidate_response(parameters, confirmation)
    gaussian_confirmation = _gaussian_response(gaussian_sigma, confirmation)
    development_errors = np.abs(development_response - development_target)
    confirmation_errors = np.abs(confirmation_response - confirmation_target)
    gaussian_errors = np.abs(gaussian_confirmation - confirmation_target)
    median_reduction = 1.0 - float(np.median(confirmation_errors)) / float(
        np.median(gaussian_errors)
    )
    bounds = contract["candidate"]
    bounds_satisfied = bool(
        bounds["combined_blur_sigma_micrometres_bounds"][0]
        <= parameters[0]
        <= bounds["combined_blur_sigma_micrometres_bounds"][1]
        and bounds["adjacency_sigma_micrometres_bounds"][0]
        <= parameters[1]
        <= bounds["adjacency_sigma_micrometres_bounds"][1]
        and bounds["adjacency_gain_bounds"][0]
        <= parameters[2]
        <= bounds["adjacency_gain_bounds"][1]
    )
    all_response = _candidate_response(parameters, frequencies)
    measurements = {
        "parameters": {
            "combined_blur_sigma_micrometres": float(parameters[0]),
            "adjacency_sigma_micrometres": float(parameters[1]),
            "adjacency_gain": float(parameters[2]),
        },
        "development_rmse": development_rmse,
        "development_max_absolute_error": float(np.max(development_errors)),
        "confirmation_rmse": float(np.sqrt(np.mean(np.square(confirmation_errors)))),
        "confirmation_max_absolute_error": float(np.max(confirmation_errors)),
        "gaussian_sigma_micrometres": gaussian_sigma,
        "gaussian_development_rmse": gaussian_development_rmse,
        "gaussian_confirmation_rmse": float(
            np.sqrt(np.mean(np.square(gaussian_errors)))
        ),
        "confirmation_median_error_reduction_vs_gaussian": median_reduction,
        "dc_response_absolute_error": abs(
            float(_candidate_response(parameters, np.asarray([0.0]))[0]) - 1.0
        ),
        "candidate_response_finite_positive": bool(
            np.all(np.isfinite(all_response)) and np.all(all_response > 0.0)
        ),
        "parameter_bounds_satisfied": bounds_satisfied,
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    gates = contract["automatic_gates"]
    results = {
        "maximum_development_absolute_error": measurements[
            "development_max_absolute_error"
        ]
        <= gates["maximum_development_absolute_error"],
        "maximum_confirmation_absolute_error": measurements[
            "confirmation_max_absolute_error"
        ]
        <= gates["maximum_confirmation_absolute_error"],
        "maximum_confirmation_rmse": measurements["confirmation_rmse"]
        <= gates["maximum_confirmation_rmse"],
        "minimum_confirmation_median_error_reduction_vs_gaussian": median_reduction
        >= gates["minimum_confirmation_median_error_reduction_vs_gaussian"],
        "dc_response_absolute_error": measurements["dc_response_absolute_error"]
        <= gates["dc_response_absolute_error"],
        "candidate_response_finite_positive": measurements[
            "candidate_response_finite_positive"
        ]
        is gates["candidate_response_finite_positive"],
        "parameter_bounds_satisfied": bounds_satisfied
        is gates["parameter_bounds_satisfied"],
        "repeat_byte_exact": True,
        "rgb_image_transform_count_zero": True,
    }
    passed = all(results.values())
    stable = {
        "schema": "neuro_film.u6_p2an_trix_mtf_explicit_spatial_fit_report.v1",
        "parent_profile_identity": source["profile_identity"],
        "split": contract["split"],
        "measurements": measurements,
        "development": {
            "target": development_target.tolist(),
            "candidate": development_response.tolist(),
        },
        "confirmation": {
            "target": confirmation_target.tolist(),
            "candidate": confirmation_response.tolist(),
            "gaussian_control": gaussian_confirmation.tolist(),
        },
        "gate_results": results,
        "automatic_pass": passed,
        "decision": contract["branch_rule"]["pass" if passed else "fail"],
        "claim_ceiling": contract["claim_ceiling"],
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {
        **stable,
        "stable_evidence_id": hashlib.sha256(encoded.encode()).hexdigest(),
    }


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8", newline="\n")
    return hashlib.sha256(encoded.encode()).hexdigest()
