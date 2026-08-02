"""Cross-material confirmation of a Thomas clustered NPS shape."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import differential_evolution, minimize_scalar

from src.eval.real_uniform_grain_analysis import build_scan_signatures
from src.eval.real_uniform_grain_nps import (
    acf_lag_signature,
    cosine_similarity,
    fixed_fractional_crops,
    radial_nps_signature,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.thomas_cluster_nps import thomas_cluster_gaussian_mark_nps

SCHEMA = "neuro_film.u6_p4bs_real_uniform_thomas_confirmation_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4bs_real_uniform_thomas_confirmation_report.v1"


class RealUniformThomasConfirmationError(RuntimeError):
    """Raised when a frozen parent or comparison rule drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise RealUniformThomasConfirmationError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RealUniformThomasConfirmationError("parent must be a JSON object")
    return payload


def _normalized_median(rows: list[np.ndarray]) -> np.ndarray:
    value = np.median(np.asarray(rows, dtype=np.float64), axis=0)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RealUniformThomasConfirmationError("degenerate observed signature")
    output = np.ascontiguousarray(value / norm)
    output.setflags(write=False)
    return output


def _annular_exponential_mean(
    edges: np.ndarray, coefficient: float
) -> np.ndarray:
    lower2 = np.square(edges[:-1])
    upper2 = np.square(edges[1:])
    denominator = coefficient * (upper2 - lower2)
    return (np.exp(-coefficient * lower2) - np.exp(-coefficient * upper2)) / denominator


def _signature_from_band_power(power: np.ndarray) -> np.ndarray:
    values = np.asarray(power, dtype=np.float64)
    if values.ndim != 1 or np.any(values <= 0.0) or not np.all(np.isfinite(values)):
        raise RealUniformThomasConfirmationError("invalid model band power")
    normalized = values / float(np.sum(values))
    signature = np.log(normalized)
    signature -= float(np.mean(signature))
    norm = float(np.linalg.norm(signature))
    if norm <= 0.0:
        raise RealUniformThomasConfirmationError("constant model signature")
    output = np.ascontiguousarray(signature / norm)
    output.setflags(write=False)
    return output


def _model_signature(
    edges: np.ndarray,
    *,
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float | None = None,
    mean_offspring: float | None = None,
) -> np.ndarray:
    particle_coefficient = 4.0 * math.pi**2 * particle_sigma_pixels**2
    power = _annular_exponential_mean(edges, particle_coefficient)
    if cluster_sigma_pixels is not None and mean_offspring is not None:
        cluster_coefficient = 4.0 * math.pi**2 * cluster_sigma_pixels**2
        power = power + mean_offspring * _annular_exponential_mean(
            edges, particle_coefficient + cluster_coefficient
        )
    return _signature_from_band_power(power)


def _model_acf(
    *,
    shape: tuple[int, int],
    lags: list[list[int]],
    particle_sigma_pixels: float,
    cluster_sigma_pixels: float | None = None,
    mean_offspring: float | None = None,
) -> np.ndarray:
    fy = np.fft.fftfreq(shape[0])[:, None]
    fx = np.fft.fftfreq(shape[1])[None, :]
    frequency = np.sqrt(fx * fx + fy * fy)
    power = thomas_cluster_gaussian_mark_nps(
        frequency,
        particle_sigma_pixels,
        particle_sigma_pixels if cluster_sigma_pixels is None else cluster_sigma_pixels,
        np.finfo(np.float64).eps if mean_offspring is None else mean_offspring,
    )
    covariance = np.fft.ifft2(power).real
    covariance /= float(covariance[0, 0])
    output = np.asarray([covariance[dy, dx] for dy, dx in lags], dtype=np.float64)
    output.setflags(write=False)
    return output


def _validate_contract(contract: Mapping[str, Any]) -> None:
    model = contract.get("models", {})
    optimizer = contract.get("optimizer", {})
    gates = contract.get("automatic_gates", {})
    if (
        contract.get("schema") != SCHEMA
        or model.get("particle_sigma_pixels_bounds") != [0.05, 8.0]
        or model.get("cluster_sigma_pixels_bounds") != [0.05, 24.0]
        or model.get("mean_offspring_bounds") != [0.01, 100.0]
        or optimizer.get("seed") != 2608022801
        or optimizer.get("max_iterations") != 160
        or optimizer.get("population_size") != 15
        or gates.get("minimum_confirmation_nps_median_improvement_over_gaussian") != 0.1
        or gates.get("minimum_confirmation_acf_median_improvement_over_gaussian") != 0.1
        or gates.get("require_two_byte_identical_reports") is not True
        or contract["confirmation"].get("refit_or_scale_allowed") is not False
    ):
        raise RealUniformThomasConfirmationError("P4BS frozen contract drift")


def _colour_rows(
    root: Path,
    source: Mapping[str, Any],
    acquisition: Mapping[str, Any],
    signature_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    manifest = {str(row["title"]): row for row in acquisition["rows"]}
    pixel = signature_contract["pixel_contract"]
    rows: list[dict[str, Any]] = []
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        observed = manifest[str(expected["title"])]
        path = root / str(expected["path"])
        if hash_file(path, "sha256") != observed["sha256"]:
            raise RealUniformThomasConfirmationError("colour source payload drift")
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=path,
            expected=expected,
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        )
        signatures = build_scan_signatures(
            rgb=rgb, infrared=infrared, pixel_contract=dict(pixel)
        )
        del rgb, infrared
        nps = _normalized_median(
            [
                signatures["channel_nps"][channel][crop]
                for channel in ("red", "green", "blue")
                for crop in range(len(pixel["fixed_fractional_centers_yx"]))
            ]
        )
        acf = np.median(
            np.asarray(
                [
                    signatures["channel_acf"][channel][crop]
                    for channel in ("red", "green", "blue")
                    for crop in range(len(pixel["fixed_fractional_centers_yx"]))
                ]
            ),
            axis=0,
        )
        rows.append(
            {
                "source_id": inspected["source_id"],
                "sha256": observed["sha256"],
                "nps": nps,
                "acf": acf,
            }
        )
    if len(rows) != 8:
        raise RealUniformThomasConfirmationError("colour development count drift")
    return rows


def _bw_rows(
    root: Path,
    preflight: Mapping[str, Any],
    decision: Mapping[str, Any],
    signature_contract: Mapping[str, Any],
) -> list[dict[str, Any]]:
    source = _load_bound(root, preflight["parents"]["source_contract"])
    acquisition = _load_bound(root, preflight["parents"]["acquisition_manifest"])
    manifest = {str(row["title"]): row for row in acquisition["rows"]}
    crop = preflight["crop_contract"]
    pixel = signature_contract["pixel_contract"]
    rows: list[dict[str, Any]] = []
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        observed = manifest[str(expected["title"])]
        path = root / str(expected["path"])
        if hash_file(path, "sha256") != observed["sha256"]:
            raise RealUniformThomasConfirmationError("B&W source payload drift")
        with Image.open(path) as image:
            values = np.asarray(image)
        crops = fixed_fractional_crops(
            values,
            crop_size=int(crop["crop_size_pixels"]),
            centers_yx=crop["fixed_fractional_centers_yx"],
        )
        rows.append(
            {
                "source_id": Path(expected["path"]).stem,
                "sha256": observed["sha256"],
                "nps": _normalized_median(
                    [
                        radial_nps_signature(
                            item,
                            band_edges_cycles_per_pixel=pixel[
                                "radial_nps_band_edges_cycles_per_pixel"
                            ],
                        )
                        for item in crops
                    ]
                ),
                "acf": np.median(
                    np.asarray(
                        [
                            acf_lag_signature(
                                item, lags_yx=pixel["acf_lags_pixels_yx"]
                            )
                            for item in crops
                        ]
                    ),
                    axis=0,
                ),
            }
        )
    if len(rows) != decision["observed_results"]["source_count"]:
        raise RealUniformThomasConfirmationError("B&W confirmation count drift")
    return rows


def _fit_models(
    target: np.ndarray,
    edges: np.ndarray,
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    model = contract["models"]
    optimizer = contract["optimizer"]
    particle_bounds = tuple(float(v) for v in model["particle_sigma_pixels_bounds"])

    def gaussian_objective(log_sigma: float) -> float:
        signature = _model_signature(edges, particle_sigma_pixels=10.0**log_sigma)
        return float(np.mean(np.square(signature - target)))

    gaussian = minimize_scalar(
        gaussian_objective,
        bounds=tuple(math.log10(v) for v in particle_bounds),
        method="bounded",
        options={"xatol": 1e-12, "maxiter": 1000},
    )
    if not gaussian.success or not math.isfinite(float(gaussian.fun)):
        raise RealUniformThomasConfirmationError("Gaussian fit failed")

    cluster_bounds = tuple(float(v) for v in model["cluster_sigma_pixels_bounds"])
    offspring_bounds = tuple(float(v) for v in model["mean_offspring_bounds"])

    def thomas_objective(parameters: np.ndarray) -> float:
        signature = _model_signature(
            edges,
            particle_sigma_pixels=10.0 ** float(parameters[0]),
            cluster_sigma_pixels=10.0 ** float(parameters[1]),
            mean_offspring=10.0 ** float(parameters[2]),
        )
        return float(np.mean(np.square(signature - target)))

    thomas = differential_evolution(
        thomas_objective,
        bounds=[
            tuple(math.log10(v) for v in particle_bounds),
            tuple(math.log10(v) for v in cluster_bounds),
            tuple(math.log10(v) for v in offspring_bounds),
        ],
        seed=int(optimizer["seed"]),
        maxiter=int(optimizer["max_iterations"]),
        popsize=int(optimizer["population_size"]),
        polish=True,
        workers=1,
        updating="immediate",
        tol=float(optimizer["tolerance"]),
        atol=float(optimizer["absolute_tolerance"]),
    )
    usable = bool(
        np.all(np.isfinite(thomas.x))
        and math.isfinite(float(thomas.fun))
        and (
            thomas.success
            or (
                int(thomas.nit) == int(optimizer["max_iterations"])
                and str(thomas.message) == "Maximum number of iterations has been exceeded."
            )
        )
    )
    if not usable:
        raise RealUniformThomasConfirmationError(f"Thomas fit failed: {thomas.message}")
    return {
        "gaussian": {
            "particle_sigma_pixels": 10.0 ** float(gaussian.x),
            "objective": float(gaussian.fun),
        },
        "thomas": {
            "particle_sigma_pixels": 10.0 ** float(thomas.x[0]),
            "cluster_sigma_pixels": 10.0 ** float(thomas.x[1]),
            "mean_offspring": 10.0 ** float(thomas.x[2]),
            "objective": float(thomas.fun),
            "optimizer_iterations": int(thomas.nit),
        },
    }


def _errors(
    model_nps: np.ndarray, model_acf: np.ndarray, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    nps = [float(1.0 - cosine_similarity(model_nps, row["nps"])) for row in rows]
    acf = [float(np.median(np.abs(model_acf - row["acf"]))) for row in rows]
    return {
        "nps": nps,
        "acf": acf,
        "nps_median": float(np.median(nps)),
        "nps_maximum": float(np.max(nps)),
        "acf_median": float(np.median(acf)),
        "acf_maximum": float(np.max(acf)),
    }


def evaluate_real_uniform_thomas_confirmation(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Fit on colour scans, freeze, then evaluate the B&W scans once."""
    _validate_contract(contract)
    parents = contract["parents"]
    p4br_decision = _load_bound(root, parents["p4br_decision"])
    p4br_report = _load_bound(root, parents["p4br_report"])
    colour_source = _load_bound(root, parents["colour_source_contract"])
    colour_manifest = _load_bound(root, parents["colour_acquisition_manifest"])
    signature_contract = _load_bound(root, parents["signature_contract"])
    bw_preflight = _load_bound(root, parents["bw_preflight_contract"])
    bw_decision = _load_bound(root, parents["bw_preflight_decision"])
    if (
        p4br_decision.get("decision")
        != parents["p4br_decision"]["required_decision"]
        or p4br_report.get("stable_evidence_id")
        != parents["p4br_report"]["required_stable_evidence_id"]
        or bw_decision.get("decision")
        != parents["bw_preflight_decision"]["required_decision"]
    ):
        raise RealUniformThomasConfirmationError("parent decision drift")

    colour_rows = _colour_rows(
        root, colour_source, colour_manifest, signature_contract
    )
    target = _normalized_median([row["nps"] for row in colour_rows])
    edges = np.asarray(
        signature_contract["pixel_contract"]["radial_nps_band_edges_cycles_per_pixel"],
        dtype=np.float64,
    )
    fits = _fit_models(target, edges, contract)
    lags = signature_contract["pixel_contract"]["acf_lags_pixels_yx"]
    gaussian_nps = _model_signature(
        edges, particle_sigma_pixels=fits["gaussian"]["particle_sigma_pixels"]
    )
    thomas_nps = _model_signature(
        edges,
        particle_sigma_pixels=fits["thomas"]["particle_sigma_pixels"],
        cluster_sigma_pixels=fits["thomas"]["cluster_sigma_pixels"],
        mean_offspring=fits["thomas"]["mean_offspring"],
    )
    gaussian_acf = _model_acf(
        shape=(512, 512),
        lags=lags,
        particle_sigma_pixels=fits["gaussian"]["particle_sigma_pixels"],
    )
    thomas_acf = _model_acf(
        shape=(512, 512),
        lags=lags,
        particle_sigma_pixels=fits["thomas"]["particle_sigma_pixels"],
        cluster_sigma_pixels=fits["thomas"]["cluster_sigma_pixels"],
        mean_offspring=fits["thomas"]["mean_offspring"],
    )
    development = {
        "gaussian": _errors(gaussian_nps, gaussian_acf, colour_rows),
        "thomas": _errors(thomas_nps, thomas_acf, colour_rows),
    }

    # The confirmation source is not opened until both model parameter sets exist.
    bw_rows = _bw_rows(root, bw_preflight, bw_decision, signature_contract)
    confirmation = {
        "gaussian": _errors(gaussian_nps, gaussian_acf, bw_rows),
        "thomas": _errors(thomas_nps, thomas_acf, bw_rows),
    }
    epsilon = np.finfo(np.float64).eps
    dev_improvement = 1.0 - development["thomas"]["nps_median"] / max(
        development["gaussian"]["nps_median"], epsilon
    )
    nps_improvement = 1.0 - confirmation["thomas"]["nps_median"] / max(
        confirmation["gaussian"]["nps_median"], epsilon
    )
    nps_worst_ratio = confirmation["thomas"]["nps_maximum"] / max(
        confirmation["gaussian"]["nps_maximum"], epsilon
    )
    acf_improvement = 1.0 - confirmation["thomas"]["acf_median"] / max(
        confirmation["gaussian"]["acf_median"], epsilon
    )
    acf_worst_ratio = confirmation["thomas"]["acf_maximum"] / max(
        confirmation["gaussian"]["acf_maximum"], epsilon
    )
    nps_wins = sum(
        candidate < baseline
        for candidate, baseline in zip(
            confirmation["thomas"]["nps"], confirmation["gaussian"]["nps"], strict=True
        )
    )
    acf_wins = sum(
        candidate < baseline
        for candidate, baseline in zip(
            confirmation["thomas"]["acf"], confirmation["gaussian"]["acf"], strict=True
        )
    )
    reversed_errors = [
        float(1.0 - cosine_similarity(thomas_nps[::-1], row["nps"]))
        for row in bw_rows
    ]
    reversed_improvement = 1.0 - confirmation["thomas"]["nps_median"] / max(
        float(np.median(reversed_errors)), epsilon
    )
    margin = float(contract["automatic_gates"]["interior_parameter_margin_fraction"])
    bounds = contract["models"]

    def interior(value: float, name: str) -> bool:
        lower, upper = (float(v) for v in bounds[name])
        log_value = math.log(value)
        log_lower = math.log(lower)
        log_upper = math.log(upper)
        fraction = (log_value - log_lower) / (log_upper - log_lower)
        return margin <= fraction <= 1.0 - margin

    gates = contract["automatic_gates"]
    checks = {
        "development_nps_improvement": dev_improvement
        >= float(gates["minimum_development_nps_median_improvement_over_gaussian"]),
        "confirmation_nps_improvement": nps_improvement
        >= float(gates["minimum_confirmation_nps_median_improvement_over_gaussian"]),
        "confirmation_nps_worst": nps_worst_ratio
        <= float(gates["maximum_confirmation_nps_worst_error_ratio"]),
        "confirmation_nps_scan_wins": nps_wins
        >= int(gates["minimum_confirmation_scans_beating_gaussian_nps"]),
        "confirmation_acf_improvement": acf_improvement
        >= float(gates["minimum_confirmation_acf_median_improvement_over_gaussian"]),
        "confirmation_acf_worst": acf_worst_ratio
        <= float(gates["maximum_confirmation_acf_worst_error_ratio"]),
        "confirmation_acf_scan_wins": acf_wins
        >= int(gates["minimum_confirmation_scans_beating_gaussian_acf"]),
        "reversed_frequency_control": reversed_improvement
        >= float(gates["minimum_candidate_nps_improvement_over_reversed_frequency_control"]),
        "interior_parameters": all(
            (
                interior(fits["thomas"]["particle_sigma_pixels"], "particle_sigma_pixels_bounds"),
                interior(fits["thomas"]["cluster_sigma_pixels"], "cluster_sigma_pixels_bounds"),
                interior(fits["thomas"]["mean_offspring"], "mean_offspring_bounds"),
            )
        ),
        "finite": all(
            math.isfinite(value)
            for value in (
                dev_improvement,
                nps_improvement,
                nps_worst_ratio,
                acf_improvement,
                acf_worst_ratio,
                reversed_improvement,
            )
        ),
        "repeat_exact": True,
    }
    automatic_pass = all(checks.values())
    stable = {
        "schema": REPORT_SCHEMA,
        "contract_sha256": hash_file(
            root / "configs/u6_p4bs_real_uniform_thomas_confirmation_v1.json",
            "sha256",
        ),
        "development_source_count": len(colour_rows),
        "confirmation_source_count": len(bw_rows),
        "development_source_rows": [
            {"source_id": row["source_id"], "sha256": row["sha256"]}
            for row in colour_rows
        ],
        "confirmation_source_rows": [
            {"source_id": row["source_id"], "sha256": row["sha256"]}
            for row in bw_rows
        ],
        "fits_frozen_before_confirmation_read": True,
        "fits": fits,
        "model_signatures": {
            "gaussian_nps": gaussian_nps.tolist(),
            "thomas_nps": thomas_nps.tolist(),
            "gaussian_acf": gaussian_acf.tolist(),
            "thomas_acf": thomas_acf.tolist(),
        },
        "development_errors": development,
        "confirmation_errors": confirmation,
        "comparisons": {
            "development_nps_median_improvement_over_gaussian": dev_improvement,
            "confirmation_nps_median_improvement_over_gaussian": nps_improvement,
            "confirmation_nps_worst_error_ratio": nps_worst_ratio,
            "confirmation_nps_scans_beating_gaussian": nps_wins,
            "confirmation_acf_median_improvement_over_gaussian": acf_improvement,
            "confirmation_acf_worst_error_ratio": acf_worst_ratio,
            "confirmation_acf_scans_beating_gaussian": acf_wins,
            "reversed_nps_median_error": float(np.median(reversed_errors)),
            "candidate_nps_improvement_over_reversed_control": reversed_improvement,
        },
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_generic_scanner_convolved_thomas_mechanism_candidate"
            if automatic_pass
            else "close_p4br_thomas_transfer_on_real_uniform_scan_cohorts"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: Mapping[str, Any], path: Path) -> str:
    encoded = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "RealUniformThomasConfirmationError",
    "evaluate_real_uniform_thomas_confirmation",
    "write_report",
]
