"""U6.P4ID compact dye-density to scanner compiler mechanism D0."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.kodak_negative_print_nuisance import _interp_curve, _physical
from src.film_physics.spectral_scanner import (
    apply_spectral_scanner,
    synthetic_profile_from_contract,
)

SCHEMA = "neuro-film.u6-p4id-compact-spectral-scanner-compiler-d0-contract.v1"
REPORT_SCHEMA = "neuro-film.u6-p4id-compact-spectral-scanner-compiler-d0-result.v1"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported P4ID contract")
    return payload


def _bound_json(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path) != binding["sha256"]:
        raise ValueError(f"P4ID parent integrity mismatch: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _grid(row: Mapping[str, Any]) -> np.ndarray:
    start, stop, step = (float(row[key]) for key in ("start", "stop", "step"))
    count = round((stop - start) / step) + 1
    result = start + np.arange(count, dtype=np.float64) * step
    if step <= 0.0 or not np.isclose(result[-1], stop):
        raise ValueError("invalid P4ID wavelength grid")
    return result


def _dye_curves(config: Mapping[str, Any], curve_data: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    source = config["source_model"]
    wavelength = _grid(source["wavelength_nm"])
    curves = []
    for name in source["dye_order"]:
        x, y = _physical(
            curve_data,
            source["dye_family"],
            curve_data["curves"][source["dye_family"]][name],
        )
        values = np.clip(_interp_curve(x, y, wavelength, outside="edge"), 0.0, None)
        maximum = float(np.max(values))
        if maximum <= 0.0:
            raise ValueError("P4ID dye curve is empty")
        curves.append(values / maximum)
    return wavelength, np.stack(curves, axis=0)


def _population(levels: Sequence[float]) -> np.ndarray:
    values = np.asarray(levels, dtype=np.float64)
    if values.ndim != 1 or np.any(np.diff(values) <= 0.0) or np.any(values < 0.0):
        raise ValueError("P4ID dye amounts must be ordered and nonnegative")
    return np.asarray(
        [(a, b, c) for a in values for b in values for c in values],
        dtype=np.float64,
    )


def _scan(amounts: np.ndarray, curves: np.ndarray, profile: Any) -> np.ndarray:
    density = np.asarray(amounts, dtype=np.float64) @ curves
    return apply_spectral_scanner(np.power(10.0, -density), profile)


def _fit_apply(
    fit_x: np.ndarray,
    fit_y: np.ndarray,
    confirmation_x: np.ndarray,
    *,
    log_response: bool,
) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((fit_x, np.ones(len(fit_x), dtype=np.float64)))
    target = np.log10(fit_y) if log_response else fit_y
    matrix = np.linalg.lstsq(design, target, rcond=None)[0]
    prediction = np.column_stack(
        (confirmation_x, np.ones(len(confirmation_x), dtype=np.float64))
    ) @ matrix
    if log_response:
        prediction = np.power(10.0, prediction)
    return prediction, matrix


def _error(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    absolute = np.abs(prediction - truth)
    return {
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "maximum_absolute_error": float(np.max(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
    }


def evaluate(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    curve_data = _bound_json(root, config["parents"]["curve_data"])
    aa1 = _bound_json(root, config["parents"]["aa1_decision"])
    if aa1.get("decision_branch") != config["parents"]["aa1_decision"]["required_decision"]:
        raise ValueError("P4ID AA1 decision drift")
    wavelength, curves = _dye_curves(config, curve_data)
    profile = synthetic_profile_from_contract(
        wavelength, config["source_model"]["scanner_profile"]
    )
    fit_x = _population(config["population"]["fit_dye_amounts"])
    confirmation_x = _population(config["population"]["confirmation_dye_amounts"])
    fit_y = _scan(fit_x, curves, profile)
    truth = _scan(confirmation_x, curves, profile)
    models: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for name, log_response in (
        ("linear-density-affine", False),
        ("beer-lambert-log-response-affine", True),
    ):
        prediction, matrix = _fit_apply(
            fit_x, fit_y, confirmation_x, log_response=log_response
        )
        predictions[name] = prediction
        models[name] = {
            **_error(prediction, truth),
            "matrix_sha256": hashlib.sha256(
                np.ascontiguousarray(matrix).tobytes()
            ).hexdigest(),
        }
    gates = config["gates"]
    linear = models["linear-density-affine"]
    log_model = models["beer-lambert-log-response-affine"]
    clear = _scan(np.zeros((1, 3), dtype=np.float64), curves, profile)[0]
    neutral_amounts = np.linspace(0.0, 1.0, 33, dtype=np.float64)
    neutral_transmittance = np.power(
        10.0, -neutral_amounts[:, None]
    ) * np.ones((1, wavelength.size), dtype=np.float64)
    neutral = apply_spectral_scanner(neutral_transmittance, profile)
    improvement = 1.0 - log_model["p95_absolute_error"] / linear["p95_absolute_error"]
    decisions = {
        "clear_response": float(np.max(np.abs(clear - 1.0)))
        <= float(gates["maximum_clear_response_absolute_error"]),
        "neutral_response": float(np.max(np.ptp(neutral, axis=1)))
        <= float(gates["maximum_neutral_channel_spread"]),
        "spectral_truth_monotone": float(np.max(np.diff(neutral, axis=0)))
        <= float(gates["maximum_monotonicity_positive_step"]),
        "log_domain_p95": log_model["p95_absolute_error"]
        <= float(gates["maximum_confirmation_p95_absolute_error"]),
        "log_domain_maximum": log_model["maximum_absolute_error"]
        <= float(gates["maximum_confirmation_maximum_absolute_error"]),
        "log_domain_materially_better": improvement
        >= float(gates["minimum_log_domain_p95_relative_improvement"]),
        "literal_linear_control_fails": (
            linear["p95_absolute_error"]
            > float(gates["maximum_confirmation_p95_absolute_error"])
        )
        is bool(gates["require_literal_linear_density_control_to_fail_p95"]),
    }
    passed = all(decisions.values())
    core = {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(_canonical(config)).hexdigest(),
        "curve_data_sha256": config["parents"]["curve_data"]["sha256"],
        "scanner_profile_sha256": profile.profile_sha256,
        "dye_curve_sha256": hashlib.sha256(
            np.ascontiguousarray(curves).tobytes()
        ).hexdigest(),
        "fit_count": len(fit_x),
        "confirmation_count": len(confirmation_x),
        "models": models,
        "log_domain_p95_relative_improvement": improvement,
        "clear_response_max_abs_error": float(np.max(np.abs(clear - 1.0))),
        "neutral_channel_spread_max": float(np.max(np.ptp(neutral, axis=1))),
        "spectral_truth_maximum_positive_step": float(np.max(np.diff(neutral, axis=0))),
        "decisions": decisions,
        "automatic_pass": passed,
        "decision": config["decision_if_pass"] if passed else config["decision_if_fail"],
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**core, "stable_evidence_id": hashlib.sha256(_canonical(core)).hexdigest()}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = ["evaluate", "load_contract", "write_report"]
