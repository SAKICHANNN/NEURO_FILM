"""Corrected calibrated-positive-wedge scanner OECF evaluation."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from scipy.optimize import least_squares

from src.eval.scanner_step_wedge_oecf import canonical_json, hash_file

SCHEMA = "neuro_film.u6_p6ak_uchicago_step_wedge_oecf_correction_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p6ak_uchicago_step_wedge_oecf_correction_report.v1"


class CorrectedOecfError(RuntimeError):
    """Raised when the corrected wedge contract or source is invalid."""


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA:
        raise CorrectedOecfError("unsupported corrected OECF contract")
    boundaries = payload["positive_wedge_row_boundaries"]
    development = set(payload["development_step_indices_zero_based"])
    confirmation = set(payload["confirmation_step_indices_zero_based"])
    if (
        len(boundaries) != len(payload["density_values"]) + 1
        or boundaries != sorted(set(boundaries))
        or development & confirmation
        or development | confirmation != set(range(len(payload["density_values"])))
        or payload["negative_wedge_role"]
        != "unscored polarity/control image without an independently authorized density table"
    ):
        raise CorrectedOecfError("corrected OECF contract drift")
    return payload


def _plateau_codes(
    path: Path, boundaries: list[int], inner_fraction: float
) -> np.ndarray:
    with tifffile.TiffFile(path) as document:
        if len(document.pages) != 1:
            raise CorrectedOecfError("positive wedge must be single frame")
        image = document.asarray()
    if image.ndim != 2 or image.dtype != np.uint16 or boundaries[-1] != image.shape[0]:
        raise CorrectedOecfError("positive wedge geometry drift")
    margin_fraction = (1.0 - inner_fraction) / 2.0
    codes = []
    for lo, hi in pairwise(boundaries):
        margin = round((hi - lo) * margin_fraction)
        region = image[lo + margin : hi - margin, :]
        if region.size == 0:
            raise CorrectedOecfError("positive wedge plateau collapsed")
        codes.append(float(np.median(region)))
    result = np.asarray(codes, dtype=np.float64)
    if not np.all(np.diff(result) < 0.0):
        raise CorrectedOecfError("positive wedge plateau codes are not decreasing")
    return result


def _density_from_code(
    codes: np.ndarray, black_code: float, gamma: float
) -> np.ndarray:
    shifted = codes - black_code
    if not np.all(shifted > 0.0) or not math.isfinite(gamma) or gamma <= 0.0:
        raise CorrectedOecfError("invalid log OECF parameters")
    return gamma * np.log10(65535.0 / shifted)


def _metrics(
    prediction: np.ndarray, truth: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    error = prediction[indices] - truth[indices]
    return {
        "rmse": float(math.sqrt(float(np.mean(error * error)))),
        "maximum_absolute_error": float(np.max(np.abs(error))),
    }


def evaluate(config: dict[str, Any], root: Path) -> dict[str, Any]:
    source = root / config["source"]["positive_wedge_path"]
    density_table = root / config["source"]["density_table_path"]
    if hash_file(source) != config["source"]["positive_wedge_sha256"]:
        raise CorrectedOecfError("positive wedge hash mismatch")
    if hash_file(density_table) != config["source"]["density_table_sha256"]:
        raise CorrectedOecfError("density table hash mismatch")
    codes = _plateau_codes(
        source,
        [int(value) for value in config["positive_wedge_row_boundaries"]],
        float(config["plateau_inner_fraction"]),
    )
    densities = np.asarray(config["density_values"], dtype=np.float64)
    development = np.asarray(
        config["development_step_indices_zero_based"], dtype=np.int64
    )
    confirmation = np.asarray(
        config["confirmation_step_indices_zero_based"], dtype=np.int64
    )

    published = config["published_equation"]
    published_density = _density_from_code(
        codes, float(published["black_code"]), float(published["gamma"])
    )
    published_metrics = _metrics(published_density, densities, confirmation)

    refit = config["development_refit"]

    def residual(parameters: np.ndarray) -> np.ndarray:
        return (
            _density_from_code(
                codes[development], float(parameters[0]), float(parameters[1])
            )
            - densities[development]
        )

    fit = least_squares(
        residual,
        np.asarray(refit["initial"], dtype=np.float64),
        bounds=(
            np.asarray([refit["black_code_bounds"][0], refit["gamma_bounds"][0]]),
            np.asarray([refit["black_code_bounds"][1], refit["gamma_bounds"][1]]),
        ),
        max_nfev=int(refit["maximum_function_evaluations"]),
        ftol=1e-14,
        xtol=1e-14,
        gtol=1e-14,
    )
    if not fit.success:
        raise CorrectedOecfError("development-only OECF fit failed")
    black_code, gamma = (float(fit.x[0]), float(fit.x[1]))
    fitted_density = _density_from_code(codes, black_code, gamma)
    fitted_metrics = _metrics(fitted_density, densities, confirmation)

    normalized_codes = codes / 65535.0
    monotone_prediction = np.interp(
        densities, densities[development], normalized_codes[development]
    )
    monotone_metrics = _metrics(monotone_prediction, normalized_codes, confirmation)
    improvement = 1.0 - fitted_metrics["rmse"] / published_metrics["rmse"]
    gates = config["gates"]
    refit_pass = (
        fitted_metrics["rmse"]
        <= float(gates["refit_confirmation_density_rmse_maximum"])
        and fitted_metrics["maximum_absolute_error"]
        <= float(gates["refit_confirmation_density_max_abs_error_maximum"])
        and improvement
        >= float(gates["refit_confirmation_improvement_vs_published_minimum"])
    )
    monotone_pass = monotone_metrics["rmse"] <= float(
        gates["monotone_confirmation_code_rmse_maximum"]
    )
    if refit_pass:
        status = "pass-exact-workflow-refit-log-oecf"
        decision = "retain_exact_workflow_bounded_log_oecf_profile"
    elif monotone_pass:
        status = "pass-monotone-only-scanner-oecf"
        decision = "retain_exact_workflow_monotone_oecf_table_only"
    else:
        status = "fail-closed-corrected-scanner-oecf"
        decision = "close_corrected_step_wedge_oecf_route"
    return {
        "schema": REPORT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "status": status,
        "decision": decision,
        "mode": config["mode"],
        "supersedes": config["supersedes"],
        "source": {
            "positive_wedge_sha256": hash_file(source),
            "density_table_sha256": hash_file(density_table),
            "plateau_codes": codes.tolist(),
        },
        "published_equation": {
            "black_code": float(published["black_code"]),
            "gamma": float(published["gamma"]),
            "confirmation_density_metrics": published_metrics,
        },
        "development_refit": {
            "black_code": black_code,
            "gamma": gamma,
            "function_evaluations": int(fit.nfev),
            "confirmation_density_metrics": fitted_metrics,
            "confirmation_rmse_improvement_vs_published": improvement,
        },
        "monotone_control": {
            "confirmation_code_metrics": monotone_metrics,
        },
        "gates": {
            "strictly_decreasing_plateau_codes": True,
            "refit_pass": refit_pass,
            "monotone_control_pass": monotone_pass,
        },
        "negative_wedge_role": config["negative_wedge_role"],
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "CorrectedOecfError",
    "canonical_json",
    "evaluate",
    "load_contract",
]
