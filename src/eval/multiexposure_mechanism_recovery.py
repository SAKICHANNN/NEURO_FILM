"""Recover two explicit multi-exposure highlight-spread equation families.

This is a synthetic inverse-problem evaluator.  It reuses the frozen P3S
patterns and response equation, fits only development observations, evaluates
on disjoint confirmation observations, and must abstain on mixed mechanisms.
It does not estimate a real film or lens profile.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from src.eval.promist_halation_identifiability import (
    _blur,
    _response,
    build_rows,
)
from src.eval.promist_halation_identifiability import (
    load_contract as load_p3s_contract,
)

SCHEMA = "neuro_film.u6_p3t_multiexposure_mechanism_recovery_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p3t_multiexposure_mechanism_recovery_report.v1"


class MechanismRecoveryError(ValueError):
    """Raised when the frozen P3T recovery contract or its parents drift."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def _verify_file(root: Path, binding: dict[str, Any], label: str) -> Path:
    path = root / binding["path"]
    if _sha256_bytes(path.read_bytes()) != binding["sha256"]:
        raise MechanismRecoveryError(f"P3T {label} hash drift")
    return path


def load_contract(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != SCHEMA
        or contract.get("node") != "ULT > U6 > U6.P3 > U6.P3T"
    ):
        raise MechanismRecoveryError("unsupported P3T contract")

    parents = contract["parents"]
    parent_contract_path = _verify_file(
        root, parents["p3s_contract"], "parent contract"
    )
    parent_decision_path = _verify_file(
        root, parents["p3s_decision"], "parent decision"
    )
    _verify_file(root, parents["p3s_evaluator"], "parent evaluator")
    parent_contract = load_p3s_contract(parent_contract_path)
    parent_decision = json.loads(parent_decision_path.read_text(encoding="utf-8"))
    if parent_decision.get("decision") != parents["p3s_decision"]["required_decision"]:
        raise MechanismRecoveryError("P3T parent decision drift")

    pure = [row for row in contract["truth_cases"] if row["family"] != "mixed-abstain"]
    mixed = [row for row in contract["truth_cases"] if row["family"] == "mixed-abstain"]
    gates = contract["automatic_gates"]
    if len(pure) != int(gates["pure_case_count"]) or len(mixed) != int(
        gates["mixed_case_count"]
    ):
        raise MechanismRecoveryError("P3T truth-case count drift")
    physical = contract["candidate_families"]["physical-backing-return"]
    lens = contract["candidate_families"]["lens-diffusion"]
    if physical["fraction_bounds"] != [0.0, 0.2] or lens["total_weight_bounds"] != [
        0.0,
        0.3,
    ]:
        raise MechanismRecoveryError("P3T parameter bounds drift")
    shape = np.asarray(lens["normalized_weight_shape"], dtype=np.float64)
    sigmas = np.asarray(lens["sigma_pixels"], dtype=np.float64)
    if (
        shape.shape != (6,)
        or sigmas.shape != (6,)
        or not np.isclose(np.sum(shape), 1.0)
    ):
        raise MechanismRecoveryError("P3T lens basis drift")
    if not np.all(np.isfinite(shape)) or np.any(shape < 0.0):
        raise MechanismRecoveryError("P3T lens weights must be finite and nonnegative")
    return contract, parent_contract


def _seed_for(seed: int, case_id: str, role: str, row: dict[str, Any]) -> int:
    token = (
        f"{seed}|{case_id}|{role}|{row['pattern']}|"
        f"{float(row['exposure_scale']).hex()}|{row['base_sha256']}"
    ).encode("ascii")
    return int.from_bytes(hashlib.sha256(token).digest()[:8], "little")


def _lens_delta(
    source: np.ndarray, sigmas: np.ndarray, shape: np.ndarray
) -> np.ndarray:
    delta = np.zeros_like(source, dtype=np.float64)
    for sigma, weight in zip(sigmas, shape, strict=True):
        delta += float(weight) * (_blur(source, float(sigma)) - source)
    return delta


def _prepare_role(
    rows: list[dict[str, Any]],
    role: str,
    truth_case: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    observation = contract["observation"]
    physical = contract["candidate_families"]["physical-backing-return"]
    lens = contract["candidate_families"]["lens-diffusion"]
    toe = float(observation["response_toe"])
    density_scale = float(observation["response_density_scale"])
    noise_sigma = float(observation["deterministic_gaussian_noise_sigma"])
    seed = int(observation["noise_seed"])
    physical_sigmas = tuple(float(value) for value in physical["sigma_grid_pixels"])
    lens_sigmas = np.asarray(lens["sigma_pixels"], dtype=np.float64)
    lens_shape = np.asarray(lens["normalized_weight_shape"], dtype=np.float64)

    sources: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    lens_deltas: list[np.ndarray] = []
    physical_blurs: dict[float, list[np.ndarray]] = {
        sigma: [] for sigma in physical_sigmas
    }
    metadata: list[dict[str, Any]] = []
    for row in rows:
        source = np.asarray(row["exposure"], dtype=np.float64)
        blurs = {sigma: _blur(source, sigma) for sigma in physical_sigmas}
        lens_delta = _lens_delta(source, lens_sigmas, lens_shape)
        truth_exposure = source.copy()
        if truth_case["family"] in {"physical-backing-return", "mixed-abstain"}:
            truth_sigma = float(truth_case["sigma_pixels"])
            truth_exposure += float(truth_case["fraction"]) * blurs[truth_sigma]
        if truth_case["family"] in {"lens-diffusion", "mixed-abstain"}:
            truth_exposure += float(truth_case["total_weight"]) * lens_delta
        target = _response(truth_exposure, toe, density_scale)
        rng = np.random.default_rng(_seed_for(seed, truth_case["case_id"], role, row))
        target = target + rng.normal(0.0, noise_sigma, size=target.shape)
        if (
            not np.all(np.isfinite(target))
            or float(np.min(target)) < 0.0
            or float(np.max(target)) > 1.0
        ):
            raise RuntimeError("P3T noisy observation escaped the display envelope")
        sources.append(source.reshape(-1))
        targets.append(target.reshape(-1))
        lens_deltas.append(lens_delta.reshape(-1))
        for sigma, image in blurs.items():
            physical_blurs[sigma].append(image.reshape(-1))
        metadata.append(
            {key: row[key] for key in ("pattern", "exposure_scale", "base_sha256")}
        )
    return {
        "source": np.concatenate(sources),
        "target": np.concatenate(targets),
        "lens_delta": np.concatenate(lens_deltas),
        "physical_blurs": {
            sigma: np.concatenate(images) for sigma, images in physical_blurs.items()
        },
        "metadata": metadata,
    }


def _rmse(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right), dtype=np.float64)))


def _fit_scalar(
    source: np.ndarray,
    basis: np.ndarray,
    target: np.ndarray,
    bounds: tuple[float, float],
    toe: float,
    density_scale: float,
) -> tuple[float, float]:
    def objective(value: float) -> float:
        prediction = _response(source + float(value) * basis, toe, density_scale)
        return float(np.mean(np.square(prediction - target), dtype=np.float64))

    result = minimize_scalar(
        objective,
        method="bounded",
        bounds=bounds,
        options={"xatol": 1e-13, "maxiter": 500},
    )
    if not result.success or not math.isfinite(float(result.fun)):
        raise RuntimeError("P3T bounded scalar fit failed")
    return float(result.x), math.sqrt(float(result.fun))


def _predict(
    source: np.ndarray,
    basis: np.ndarray,
    parameter: float,
    toe: float,
    density_scale: float,
) -> np.ndarray:
    exposure = source + parameter * basis
    if not np.all(np.isfinite(exposure)) or float(np.min(exposure)) < 0.0:
        raise RuntimeError("P3T candidate produced invalid exposure")
    return _response(exposure, toe, density_scale)


def _evaluate_case(
    truth_case: dict[str, Any],
    development_rows: list[dict[str, Any]],
    confirmation_rows: list[dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, Any]:
    development = _prepare_role(development_rows, "development", truth_case, contract)
    confirmation = _prepare_role(
        confirmation_rows, "confirmation", truth_case, contract
    )
    observation = contract["observation"]
    toe = float(observation["response_toe"])
    density_scale = float(observation["response_density_scale"])
    physical_contract = contract["candidate_families"]["physical-backing-return"]
    lens_contract = contract["candidate_families"]["lens-diffusion"]

    physical_fits: list[dict[str, float]] = []
    for sigma in (float(value) for value in physical_contract["sigma_grid_pixels"]):
        fraction, development_rmse = _fit_scalar(
            development["source"],
            development["physical_blurs"][sigma],
            development["target"],
            tuple(float(value) for value in physical_contract["fraction_bounds"]),
            toe,
            density_scale,
        )
        physical_fits.append(
            {
                "sigma_pixels": sigma,
                "fraction": fraction,
                "development_rmse": development_rmse,
            }
        )
    physical_fit = min(
        physical_fits, key=lambda row: (row["development_rmse"], row["sigma_pixels"])
    )
    physical_prediction = _predict(
        confirmation["source"],
        confirmation["physical_blurs"][physical_fit["sigma_pixels"]],
        physical_fit["fraction"],
        toe,
        density_scale,
    )
    physical_confirmation_rmse = _rmse(physical_prediction, confirmation["target"])

    lens_weight, lens_development_rmse = _fit_scalar(
        development["source"],
        development["lens_delta"],
        development["target"],
        tuple(float(value) for value in lens_contract["total_weight_bounds"]),
        toe,
        density_scale,
    )
    lens_prediction = _predict(
        confirmation["source"],
        confirmation["lens_delta"],
        lens_weight,
        toe,
        density_scale,
    )
    lens_confirmation_rmse = _rmse(lens_prediction, confirmation["target"])
    identity_prediction = _response(confirmation["source"], toe, density_scale)
    identity_rmse = _rmse(identity_prediction, confirmation["target"])

    family_rmse = {
        "physical-backing-return": physical_confirmation_rmse,
        "lens-diffusion": lens_confirmation_rmse,
    }
    selected_family = min(family_rmse, key=lambda key: (family_rmse[key], key))
    best_rmse = family_rmse[selected_family]
    other_rmse = max(family_rmse.values())
    is_mixed = truth_case["family"] == "mixed-abstain"
    abstention_floor = float(
        contract["automatic_gates"]["mixed_case_minimum_best_confirmation_rmse"]
    )
    decision = (
        "abstain" if is_mixed and best_rmse >= abstention_floor else selected_family
    )
    parameter_error: float | None
    sigma_exact: bool | None
    if truth_case["family"] == "physical-backing-return":
        parameter_error = abs(
            physical_fit["fraction"] - float(truth_case["fraction"])
        ) / float(truth_case["fraction"])
        sigma_exact = physical_fit["sigma_pixels"] == float(truth_case["sigma_pixels"])
    elif truth_case["family"] == "lens-diffusion":
        parameter_error = abs(lens_weight - float(truth_case["total_weight"])) / float(
            truth_case["total_weight"]
        )
        sigma_exact = None
    else:
        parameter_error = None
        sigma_exact = None
    return {
        "case_id": truth_case["case_id"],
        "truth_family": truth_case["family"],
        "physical_fit": {
            "sigma_pixels": physical_fit["sigma_pixels"],
            "fraction": physical_fit["fraction"],
            "development_rmse": physical_fit["development_rmse"],
            "confirmation_rmse": physical_confirmation_rmse,
        },
        "lens_fit": {
            "total_weight": lens_weight,
            "development_rmse": lens_development_rmse,
            "confirmation_rmse": lens_confirmation_rmse,
        },
        "identity_confirmation_rmse": identity_rmse,
        "selected_family": selected_family,
        "decision": decision,
        "best_confirmation_rmse": best_rmse,
        "wrong_to_correct_rmse_ratio": (
            None if is_mixed else other_rmse / max(best_rmse, 1e-30)
        ),
        "relative_parameter_error": parameter_error,
        "physical_sigma_exact": sigma_exact,
    }


def evaluate(root: Path, contract_path: Path) -> dict[str, Any]:
    contract, parent_contract = load_contract(root, contract_path)
    development_rows = build_rows(parent_contract, "development")
    confirmation_rows = build_rows(parent_contract, "confirmation")
    cases = [
        _evaluate_case(row, development_rows, confirmation_rows, contract)
        for row in contract["truth_cases"]
    ]
    pure = [row for row in cases if row["truth_family"] != "mixed-abstain"]
    mixed = [row for row in cases if row["truth_family"] == "mixed-abstain"]
    gates = contract["automatic_gates"]
    correct_family_count = sum(row["decision"] == row["truth_family"] for row in pure)
    physical_sigma_count = sum(
        row["physical_sigma_exact"] is True
        for row in pure
        if row["truth_family"] == "physical-backing-return"
    )
    mixed_abstention_count = sum(row["decision"] == "abstain" for row in mixed)
    pattern_overlap = len(
        {row["pattern"] for row in development_rows}
        & {row["pattern"] for row in confirmation_rows}
    )
    gate_results = {
        "pure_correct_family": correct_family_count
        >= int(gates["pure_case_correct_family_count_min"]),
        "pure_confirmation_error": max(row["best_confirmation_rmse"] for row in pure)
        <= float(gates["pure_case_maximum_confirmation_rmse"]),
        "pure_wrong_family_separation": min(
            float(row["wrong_to_correct_rmse_ratio"]) for row in pure
        )
        >= float(gates["pure_case_minimum_wrong_to_correct_rmse_ratio"]),
        "pure_parameter_recovery": max(
            float(row["relative_parameter_error"]) for row in pure
        )
        <= float(gates["pure_case_maximum_relative_parameter_error"]),
        "physical_sigma_recovery": physical_sigma_count
        >= int(gates["physical_sigma_exact_count_min"]),
        "mixed_abstention": mixed_abstention_count
        >= int(gates["mixed_case_abstention_count_min"]),
        "mixed_residual_floor": min(row["best_confirmation_rmse"] for row in mixed)
        >= float(gates["mixed_case_minimum_best_confirmation_rmse"]),
        "role_separation": pattern_overlap
        <= int(gates["development_confirmation_pattern_overlap_count_max"]),
    }
    core = {
        "schema": REPORT_SCHEMA,
        "node": contract["node"],
        "contract_sha256": _sha256_bytes(contract_path.read_bytes()),
        "parent_bindings": contract["parents"],
        "role_facts": {
            "development_rows": len(development_rows),
            "confirmation_rows": len(confirmation_rows),
            "development_confirmation_pattern_overlap_count": pattern_overlap,
            "fit_reads_confirmation": False,
        },
        "case_results": cases,
        "summary": {
            "pure_correct_family_count": correct_family_count,
            "pure_case_count": len(pure),
            "physical_sigma_exact_count": physical_sigma_count,
            "mixed_abstention_count": mixed_abstention_count,
            "mixed_case_count": len(mixed),
            "maximum_pure_confirmation_rmse": max(
                row["best_confirmation_rmse"] for row in pure
            ),
            "minimum_wrong_to_correct_rmse_ratio": min(
                float(row["wrong_to_correct_rmse_ratio"]) for row in pure
            ),
            "maximum_relative_parameter_error": max(
                float(row["relative_parameter_error"]) for row in pure
            ),
            "minimum_mixed_best_confirmation_rmse": min(
                row["best_confirmation_rmse"] for row in mixed
            ),
        },
        "gate_results": gate_results,
        "passed": all(gate_results.values()),
        "decision": (
            "retain-bounded-synthetic-mechanism-recovery"
            if all(gate_results.values())
            else "close-bounded-synthetic-mechanism-recovery"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    return {**core, "scientific_stable_id": _sha256_bytes(_canonical_bytes(core))}


def write_report(report: dict[str, Any], path: Path) -> str:
    payload = _canonical_bytes(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256_bytes(payload)


__all__ = [
    "MechanismRecoveryError",
    "evaluate",
    "load_contract",
    "write_report",
]
