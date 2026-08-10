"""U5.R2CB2 bounded film-inspired dye-operator compiler."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.eval.fujifilm_e6_spectral_dye_signature import (
    CHANNELS,
    STOCKS,
    _curve_values,
)
from src.eval.spectral_film_lut_bank import (
    _RGB_TO_XYZ,
    array_sha256,
    local_jacobian_metrics,
    synthetic_cube,
    xyz_to_lab,
)
from src.film_physics.spectral_scanner import (
    SpectralScannerProfile,
    apply_spectral_scanner,
    synthetic_profile_from_contract,
)

SCHEMA = "neuro_film.u5_r2cb2_fujifilm_e6_bounded_dye_operator_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb2_fujifilm_e6_bounded_dye_operator_report.v1"
EXPERIMENT_ID = "U5.R2CB2"
CONTRACT_SHA256 = "5663d54698b3e6f4efed1bc64cdd526fbac0f46ec36fa05d749e793bf7f12985"


class FujifilmDyeOperatorError(RuntimeError):
    """Raised when a CB2 contract, input or physical domain check fails."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmDyeOperatorError("CB2 paths must be repository-relative")
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmDyeOperatorError("CB2 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema") != SCHEMA
        or payload.get("experiment_id") != EXPERIMENT_ID
        or tuple(payload.get("stocks", ())) != STOCKS
        or len(payload.get("scanner_profiles", ())) != 2
    ):
        raise FujifilmDyeOperatorError("CB2 frozen contract structure drift")
    _relative_path(payload["parent_source_signature"]["path"])
    _relative_path(payload["trace"]["path"])
    return payload


def _wavelength_grid(config: Mapping[str, Any]) -> np.ndarray:
    row = config["compiler"]["wavelength_nm"]
    start, stop, step = (float(row[key]) for key in ("start", "stop", "step"))
    count = round((stop - start) / step) + 1
    grid = start + np.arange(count, dtype=np.float64) * step
    if step <= 0.0 or not np.isclose(grid[-1], stop):
        raise FujifilmDyeOperatorError("CB2 invalid wavelength grid")
    return grid


def _load_source_curves(
    config: Mapping[str, Any], root: Path
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    parent_path = root / _relative_path(config["parent_source_signature"]["path"])
    trace_path = root / _relative_path(config["trace"]["path"])
    for path, expected in (
        (parent_path, config["parent_source_signature"]["sha256"]),
        (trace_path, config["trace"]["sha256"]),
    ):
        if not path.is_file() or hash_file(path) != expected:
            raise FujifilmDyeOperatorError(f"CB2 input integrity mismatch: {path}")
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    if (
        parent.get("decision")
        != config["parent_source_signature"]["required_decision"]
        or parent.get("stable_evidence_id")
        != config["parent_source_signature"]["required_stable_evidence_id"]
    ):
        raise FujifilmDyeOperatorError("CB2 parent decision mismatch")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    if tuple(trace.get("stocks", {})) != STOCKS:
        raise FujifilmDyeOperatorError("CB2 trace stock order mismatch")
    wavelength = _wavelength_grid(config)
    curves: dict[str, np.ndarray] = {}
    for stock in STOCKS:
        channels = []
        for channel in CHANNELS:
            source_x, source_y = _curve_values(trace["stocks"][stock], channel)
            channels.append(
                np.interp(
                    wavelength,
                    source_x,
                    source_y,
                    left=float(source_y[0]),
                    right=float(source_y[-1]),
                )
            )
        values = np.stack(channels, axis=0)
        if not np.all(np.isfinite(values)):
            raise FujifilmDyeOperatorError("CB2 nonfinite source curve")
        curves[stock] = values
    return wavelength, curves


def _balanced_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(curves, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 3:
        raise FujifilmDyeOperatorError("CB2 dye curves must be 3xN")
    scales, _ = nnls(values.T, np.ones(values.shape[1], dtype=np.float64))
    balanced = scales[:, None] * values
    mean_density = float(np.mean(np.sum(balanced, axis=0)))
    if not math.isfinite(mean_density) or mean_density <= 0.0:
        raise FujifilmDyeOperatorError("CB2 dye balance is degenerate")
    scales = scales / mean_density
    balanced = scales[:, None] * values
    rmse = float(np.sqrt(np.mean((np.sum(balanced, axis=0) - 1.0) ** 2)))
    if np.any(balanced < 0.0):
        raise FujifilmDyeOperatorError("CB2 digitized density produced negative dye")
    return balanced, scales, rmse


def _raw_scan(
    linear_rgb: np.ndarray,
    balanced_curves: np.ndarray,
    profile: SpectralScannerProfile,
    density_strength: float,
) -> np.ndarray:
    rgb = np.asarray(linear_rgb, dtype=np.float64)
    complementary = np.stack(
        (1.0 - rgb[..., 2], 1.0 - rgb[..., 1], 1.0 - rgb[..., 0]), axis=-1
    )
    density = density_strength * np.einsum(
        "...c,cw->...w", complementary, balanced_curves
    )
    transmittance = np.power(10.0, -density)
    return apply_spectral_scanner(transmittance, profile)


def _neutral_calibrated_scan(
    linear_rgb: np.ndarray,
    balanced_curves: np.ndarray,
    profile: SpectralScannerProfile,
    *,
    density_strength: float,
    neutral_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    levels = np.linspace(0.0, 1.0, neutral_samples, dtype=np.float64)
    neutral_rgb = np.repeat(levels[:, None], 3, axis=1)
    neutral_raw = _raw_scan(
        neutral_rgb, balanced_curves, profile, density_strength
    )
    if np.any(np.diff(neutral_raw, axis=0) <= 0.0):
        raise FujifilmDyeOperatorError("CB2 neutral scanner ramp is not strict")
    raw = _raw_scan(linear_rgb, balanced_curves, profile, density_strength)
    epsilon = 32.0 * np.finfo(np.float64).eps
    if np.any(raw < neutral_raw[0] - epsilon) or np.any(raw > neutral_raw[-1] + epsilon):
        raise FujifilmDyeOperatorError("CB2 scanner value exceeds neutral endpoints")
    calibrated = np.empty_like(raw)
    for channel in range(3):
        calibrated[..., channel] = np.interp(
            raw[..., channel], neutral_raw[:, channel], levels
        )
    return calibrated, neutral_raw


def compile_operator(
    cube: np.ndarray,
    curves: np.ndarray,
    profile: SpectralScannerProfile,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    compiler = config["compiler"]
    balanced, scales, balance_rmse = _balanced_curves(curves)
    calibrated, neutral_raw = _neutral_calibrated_scan(
        cube,
        balanced,
        profile,
        density_strength=float(compiler["density_strength"]),
        neutral_samples=int(compiler["neutral_calibration_samples"]),
    )
    alpha = float(compiler["residual_strength"])
    output64 = (1.0 - alpha) * np.asarray(cube, dtype=np.float64) + alpha * calibrated
    output = output64.astype(np.float32)
    neutral_levels = np.linspace(
        0.0, 1.0, int(compiler["neutral_calibration_samples"]), dtype=np.float64
    )
    neutral_cube = np.repeat(neutral_levels[:, None], 3, axis=1)
    neutral_calibrated, _ = _neutral_calibrated_scan(
        neutral_cube,
        balanced,
        profile,
        density_strength=float(compiler["density_strength"]),
        neutral_samples=int(compiler["neutral_calibration_samples"]),
    )
    neutral_output = (
        (1.0 - alpha) * neutral_cube + alpha * neutral_calibrated
    ).astype(np.float32)
    return output, {
        "balance_rmse": balance_rmse,
        "balance_scales": scales.tolist(),
        "balanced_curve_sha256": array_sha256(balanced),
        "neutral_raw_sha256": array_sha256(neutral_raw),
        "neutral_output": neutral_output,
    }


def _linear_lab(values: np.ndarray) -> np.ndarray:
    return xyz_to_lab(np.asarray(values, dtype=np.float64) @ _RGB_TO_XYZ.T)


def _median_delta_e76(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.median(np.linalg.norm(_linear_lab(left) - _linear_lab(right), axis=-1)))


def _stable_identity(report: Mapping[str, Any]) -> str:
    stable = dict(report)
    stable.pop("stable_evidence_id", None)
    return hashlib.sha256(canonical_json(stable)).hexdigest()


def evaluate_operator(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    wavelength, source_curves = _load_source_curves(config, root)
    scanners = {
        row["profile_id"]: synthetic_profile_from_contract(wavelength, row)
        for row in config["scanner_profiles"]
    }
    size = int(config["compiler"]["lut_size"])
    cube = synthetic_cube(size)
    pooled = np.mean(np.stack(list(source_curves.values()), axis=0), axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict[str, np.ndarray]] = {}
    records: dict[str, Any] = {}
    pooled_outputs: dict[str, np.ndarray] = {}
    for scanner_id, scanner in scanners.items():
        outputs[scanner_id] = {}
        pooled_output, pooled_info = compile_operator(cube, pooled, scanner, config)
        pooled_outputs[scanner_id] = pooled_output
        pooled_path = output_dir / f"{scanner_id}__pooled.npy"
        np.save(pooled_path, pooled_output, allow_pickle=False)
        for stock in STOCKS:
            output, info = compile_operator(cube, source_curves[stock], scanner, config)
            outputs[scanner_id][stock] = output
            path = output_dir / f"{scanner_id}__{stock}.npy"
            np.save(path, output, allow_pickle=False)
            neutral = info.pop("neutral_output")
            jacobian = local_jacobian_metrics(output)
            interior = np.all((cube > 0.0) & (cube < 1.0), axis=-1)
            interior_values = output[interior]
            records[f"{scanner_id}__{stock}"] = {
                "array_sha256": array_sha256(output),
                "artifact_file_sha256": hash_file(path),
                "balance": info,
                "finite": bool(np.all(np.isfinite(output))),
                "interior_hard_clip_fraction": float(
                    np.mean((interior_values <= 0.0) | (interior_values >= 1.0))
                ),
                "jacobian": jacobian,
                "neutral_array_sha256": array_sha256(neutral),
                "neutral_max_rgb_spread": float(np.max(np.ptp(neutral, axis=1))),
                "output_maximum": float(np.max(output)),
                "output_minimum": float(np.min(output)),
                "stock_style_delta_e76_median": _median_delta_e76(cube, output),
                "stock_vs_pooled_delta_e76_median": _median_delta_e76(
                    output, pooled_output
                ),
            }
        records[f"{scanner_id}__pooled"] = {
            "array_sha256": array_sha256(pooled_output),
            "artifact_file_sha256": hash_file(pooled_path),
            "balance_rmse": pooled_info["balance_rmse"],
        }

    pairs: dict[str, Any] = {}
    scanner_ids = tuple(scanners)
    for first, second in combinations(STOCKS, 2):
        per_scanner = {}
        for scanner_id in scanner_ids:
            left, right = outputs[scanner_id][first], outputs[scanner_id][second]
            per_scanner[scanner_id] = {
                "linear_rgb_rmse": float(np.sqrt(np.mean((left - right) ** 2))),
                "delta_e76_median": _median_delta_e76(left, right),
            }
        primary = per_scanner[scanner_ids[0]]["linear_rgb_rmse"]
        observer = per_scanner[scanner_ids[1]]["linear_rgb_rmse"]
        pairs[f"{first}__{second}"] = {
            "per_scanner": per_scanner,
            "observer_to_primary_separation_ratio": observer / primary,
        }

    gates = config["gates"]
    stock_records = [
        records[f"{scanner_id}__{stock}"]
        for scanner_id in scanner_ids
        for stock in STOCKS
    ]
    pair_rows = [row for pair in pairs.values() for row in pair["per_scanner"].values()]
    gate_results = {
        "array_hashes_exact": True,
        "balance": all(
            row["balance"]["balance_rmse"] <= gates["maximum_dye_balance_rmse"]
            for row in stock_records
        ),
        "finite": all(row["finite"] is gates["all_values_finite"] for row in stock_records),
        "interior_clip": all(
            row["interior_hard_clip_fraction"]
            <= gates["maximum_interior_hard_clip_fraction"]
            for row in stock_records
        ),
        "jacobian": all(
            row["jacobian"]["negative_jacobian_fraction"]
            <= gates["maximum_negative_jacobian_fraction"]
            and row["jacobian"]["minimum_jacobian_determinant"]
            >= gates["minimum_jacobian_determinant"]
            and row["jacobian"]["maximum_jacobian_spectral_norm"]
            <= gates["maximum_jacobian_spectral_norm"]
            for row in stock_records
        ),
        "neutral": all(
            row["neutral_max_rgb_spread"] <= gates["maximum_neutral_rgb_spread"]
            for row in stock_records
        ),
        "pair_separation": all(
            row["linear_rgb_rmse"] >= gates["minimum_each_pair_output_linear_rgb_rmse"]
            and row["delta_e76_median"] >= gates["minimum_each_pair_output_delta_e76_median"]
            for row in pair_rows
        ),
        "range": all(
            row["output_minimum"] >= gates["output_minimum"]
            and row["output_maximum"] <= gates["output_maximum"]
            for row in stock_records
        ),
        "scanner_retention": all(
            row["observer_to_primary_separation_ratio"]
            >= gates["minimum_observer_to_primary_pair_separation_ratio"]
            for row in pairs.values()
        ),
        "stock_style": all(
            row["stock_style_delta_e76_median"]
            >= gates["minimum_stock_style_delta_e76_median"]
            for row in stock_records
        ),
        "stock_vs_pooled": all(
            row["stock_vs_pooled_delta_e76_median"]
            >= gates["minimum_each_stock_vs_pooled_delta_e76_median"]
            for row in stock_records
        ),
    }
    passed = all(gate_results.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "trace_sha256": config["trace"]["sha256"],
        "parent_stable_evidence_id": config["parent_source_signature"][
            "required_stable_evidence_id"
        ],
        "wavelength_nm": wavelength.tolist(),
        "scanner_profile_sha256": {
            key: value.profile_sha256 for key, value in scanners.items()
        },
        "records": records,
        "pairs": pairs,
        "gate_results": gate_results,
        "failed_gates": sorted(key for key, value in gate_results.items() if not value),
        "passed": passed,
        "decision": (
            "retain_bounded_film_inspired_dye_operator_capacity"
            if passed
            else "close_exact_bounded_dye_operator_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _stable_identity(report)
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "FujifilmDyeOperatorError",
    "compile_operator",
    "evaluate_operator",
    "hash_file",
    "load_contract",
    "write_report",
]
