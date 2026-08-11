"""U5.R2CB6 characteristic-constrained Fujifilm dye forward proxy."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

from src.eval.fujifilm_dye_basis_measured_conformance import (
    CORRECT_STOCK,
    _array_sha256,
    _fit_basis_rows,
    _fixed_bases,
    canonical_json,
    hash_file,
)
from src.eval.fujifilm_dye_forward_proxy import (
    _fit_nonnegative_mapping,
    _load_source_rgb,
    _summary,
)
from src.eval.physical_measured_slide_spectral_scanner import _load_measured_table

SCHEMA = "neuro_film.u5_r2cb6_fujifilm_characteristic_forward_proxy_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb6_fujifilm_characteristic_forward_proxy_report.v1"
EXPERIMENT_ID = "U5.R2CB6"
CONTRACT_SHA256 = "e8f754c96b88491f9e6c1f467ddc1e3f9ec4d8b87afde9c40d0bea4e49b78150"


class FujifilmCharacteristicForwardError(RuntimeError):
    """Raised when a CB6 contract, source observation or fit is invalid."""


def _relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise FujifilmCharacteristicForwardError(
            "CB6 paths must be repository-relative"
        )
    return path


def load_contract(path: Path) -> dict[str, Any]:
    if hash_file(path) != CONTRACT_SHA256:
        raise FujifilmCharacteristicForwardError("CB6 contract hash drift")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FujifilmCharacteristicForwardError("CB6 frozen contract structure drift")
    for key in (
        "cb5_decision_path",
        "cb4_decision_path",
        "pair_table_path",
        "dye_trace_path",
        "characteristic_source_path",
        "characteristic_page_render_path",
    ):
        _relative_path(payload["parents"][key])
    return payload


def _characteristic_curves(
    config: Mapping[str, Any],
) -> tuple[np.ndarray, tuple[PchipInterpolator, ...], np.ndarray]:
    source = config["source_observation"]
    exposure = np.asarray(source["log_exposure"], dtype=np.float64)
    names = (
        "blue_sensitive_yellow_dye",
        "green_sensitive_magenta_dye",
        "red_sensitive_cyan_dye",
    )
    density = np.stack(
        [np.asarray(source["density"][name], dtype=np.float64) for name in names],
        axis=0,
    )
    if (
        exposure.ndim != 1
        or exposure.size < 8
        or not np.all(np.diff(exposure) > 0.0)
        or density.shape != (3, exposure.size)
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
        or np.any(np.diff(density, axis=1) > 0.0)
    ):
        raise FujifilmCharacteristicForwardError("CB6 characteristic trace is invalid")
    floors = density[:, -1].copy()
    curves = tuple(
        PchipInterpolator(exposure, row - floor, extrapolate=False)
        for row, floor in zip(density, floors, strict=True)
    )
    return exposure, curves, floors


def _apply_curve(
    source_rgb: np.ndarray,
    parameters: np.ndarray,
    exposure: np.ndarray,
    curve: PchipInterpolator,
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.float64)
    row = np.asarray(parameters, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or row.shape != (4,)
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(row))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(row[1:] < 0.0)
    ):
        raise FujifilmCharacteristicForwardError("CB6 curve application is invalid")
    log_h = row[0] + source @ row[1:]
    clipped = np.clip(log_h, exposure[0], exposure[-1])
    result = np.asarray(curve(clipped), dtype=np.float64)
    if not np.all(np.isfinite(result)) or np.any(result < -1e-12):
        raise FujifilmCharacteristicForwardError("CB6 curve emitted invalid density")
    return np.maximum(result, 0.0)


def _fit_characteristic_layer(
    source_rgb: np.ndarray,
    target: np.ndarray,
    exposure: np.ndarray,
    curve: PchipInterpolator,
    layer_index: int,
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape[1] != 3
        or values.shape != (source.shape[0],)
        or source.shape[0] < 12
        or not np.all(np.isfinite(source))
        or not np.all(np.isfinite(values))
        or np.any(source < 0.0)
        or np.any(source > 1.0)
        or np.any(values < 0.0)
        or layer_index not in (0, 1, 2)
    ):
        raise FujifilmCharacteristicForwardError("CB6 layer fit input is invalid")
    initial = np.zeros(4, dtype=np.float64)
    initial[0] = exposure[0]
    # Yellow, magenta and cyan are driven primarily by recorder B, G and R.
    initial[1 + (2 - layer_index)] = exposure[-1] - exposure[0]
    lower = np.array([exposure[0], 0.0, 0.0, 0.0], dtype=np.float64)
    upper = np.array(
        [
            exposure[-1],
            exposure[-1] - exposure[0],
            exposure[-1] - exposure[0],
            exposure[-1] - exposure[0],
        ],
        dtype=np.float64,
    )

    result = least_squares(
        lambda row: _apply_curve(source, row, exposure, curve) - values,
        initial,
        bounds=(lower, upper),
        method="trf",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=200,
        x_scale="jac",
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise FujifilmCharacteristicForwardError("CB6 characteristic fit failed")
    return np.asarray(result.x, dtype=np.float64)


def _relative_summary(candidate: np.ndarray, control: np.ndarray) -> dict[str, float]:
    epsilon = np.finfo(np.float64).tiny
    return _summary((control - candidate) / np.maximum(control, epsilon))


def evaluate_characteristic_forward_proxy(
    config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    parents = config["parents"]
    for path_key, hash_key in (
        ("cb5_decision_path", "cb5_decision_sha256"),
        ("cb4_decision_path", "cb4_decision_sha256"),
        ("pair_table_path", "pair_table_sha256"),
        ("dye_trace_path", "dye_trace_sha256"),
        ("characteristic_source_path", "characteristic_source_sha256"),
        ("characteristic_page_render_path", "characteristic_page_render_sha256"),
    ):
        path = root / _relative_path(parents[path_key])
        if not path.is_file() or hash_file(path) != parents[hash_key]:
            raise FujifilmCharacteristicForwardError(
                f"CB6 input integrity mismatch: {path_key}"
            )
    cb5 = json.loads((root / parents["cb5_decision_path"]).read_text(encoding="utf-8"))
    cb4 = json.loads((root / parents["cb4_decision_path"]).read_text(encoding="utf-8"))
    if cb5.get("decision") != parents["cb5_required_decision"]:
        raise FujifilmCharacteristicForwardError("CB6 CB5 decision mismatch")
    if cb4.get("stable_evidence_id") != parents["cb4_required_stable_evidence_id"]:
        raise FujifilmCharacteristicForwardError("CB6 CB4 decision mismatch")

    pair_path = root / parents["pair_table_path"]
    test_sets, slides, spectral_pct, _ = _load_measured_table(pair_path)
    source_rgb = _load_source_rgb(pair_path)
    wavelength_all = np.arange(380.0, 781.0, 10.0, dtype=np.float64)
    wavelength = np.arange(410.0, 691.0, 10.0, dtype=np.float64)
    density = -np.log10(spectral_pct[:, np.isin(wavelength_all, wavelength)] / 100.0)
    if (
        density.shape != (int(config["gates"]["required_rows"]), wavelength.size)
        or source_rgb.shape != (density.shape[0], 3)
        or not np.all(np.isfinite(density))
        or np.any(density < 0.0)
    ):
        raise FujifilmCharacteristicForwardError("CB6 measured rows are invalid")

    trace = json.loads((root / parents["dye_trace_path"]).read_text(encoding="utf-8"))
    basis = _fixed_bases(trace, wavelength)[CORRECT_STOCK]
    oracle_coefficients = _fit_basis_rows(density, basis)[0]
    exposure, curves, floors = _characteristic_curves(config)
    characteristic_prediction = np.empty_like(density)
    affine_prediction = np.empty_like(density)
    direct_prediction = np.empty_like(density)
    cyclic_prediction = np.empty_like(density)
    mean_prediction = np.empty_like(density)
    coverage = np.zeros(density.shape[0], dtype=np.int16)
    fold_records: list[dict[str, Any]] = []
    features = np.column_stack((np.ones(source_rgb.shape[0]), 1.0 - source_rgb))

    for held_set in sorted(set(test_sets.tolist())):
        for held_slide in sorted(set(slides.tolist())):
            development = (test_sets != held_set) & (slides != held_slide)
            confirmation = (test_sets == held_set) & (slides == held_slide)
            if int(np.sum(development)) != int(
                config["evaluation"]["expected_development_rows_per_fold"]
            ) or int(np.sum(confirmation)) != int(
                config["evaluation"]["expected_confirmation_rows_per_fold"]
            ):
                raise FujifilmCharacteristicForwardError("CB6 fold support mismatch")
            coverage[confirmation] += 1
            base_mapping = _fit_nonnegative_mapping(
                features[development], oracle_coefficients[development, :1]
            )
            coefficient = np.empty((int(np.sum(confirmation)), 4), dtype=np.float64)
            coefficient[:, 0] = (features[confirmation] @ base_mapping)[:, 0]
            parameters = []
            for layer_index, curve in enumerate(curves):
                row = _fit_characteristic_layer(
                    source_rgb[development],
                    oracle_coefficients[development, layer_index + 1],
                    exposure,
                    curve,
                    layer_index,
                )
                parameters.append(row)
                coefficient[:, layer_index + 1] = _apply_curve(
                    source_rgb[confirmation], row, exposure, curve
                )
            characteristic_prediction[confirmation] = coefficient @ basis.T
            cyclic_coefficient = coefficient.copy()
            cyclic_coefficient[:, 1:] = coefficient[:, [2, 3, 1]]
            cyclic_prediction[confirmation] = cyclic_coefficient @ basis.T

            affine_mapping = _fit_nonnegative_mapping(
                features[development], oracle_coefficients[development]
            )
            affine_prediction[confirmation] = (
                features[confirmation] @ affine_mapping @ basis.T
            )
            direct_mapping = _fit_nonnegative_mapping(
                features[development], density[development]
            )
            direct_prediction[confirmation] = features[confirmation] @ direct_mapping
            mean_prediction[confirmation] = np.mean(density[development], axis=0)
            fold_records.append(
                {
                    "held_test_set": held_set,
                    "held_slide": held_slide,
                    "development_rows": int(np.sum(development)),
                    "confirmation_rows": int(np.sum(confirmation)),
                    "parameters": [row.tolist() for row in parameters],
                }
            )
    if not np.all(coverage == 1):
        raise FujifilmCharacteristicForwardError("CB6 folds do not partition rows")

    def rmse(prediction: np.ndarray) -> np.ndarray:
        return np.sqrt(np.mean((prediction - density) ** 2, axis=1))

    errors = {
        "characteristic": rmse(characteristic_prediction),
        "cb5-affine": rmse(affine_prediction),
        "direct-spectral": rmse(direct_prediction),
        "cyclic-layer": rmse(cyclic_prediction),
        "development-mean": rmse(mean_prediction),
    }
    candidate = errors["characteristic"]
    comparisons = {
        "win_fraction_vs_cb5_affine": float(np.mean(candidate < errors["cb5-affine"])),
        "relative_improvement_vs_cb5_affine": _relative_summary(
            candidate, errors["cb5-affine"]
        ),
        "win_fraction_vs_direct_nonnegative_spectral": float(
            np.mean(candidate < errors["direct-spectral"])
        ),
        "relative_improvement_vs_direct_nonnegative_spectral": _relative_summary(
            candidate, errors["direct-spectral"]
        ),
        "win_fraction_vs_cyclic_layer": float(
            np.mean(candidate < errors["cyclic-layer"])
        ),
        "relative_improvement_vs_cyclic_layer": _relative_summary(
            candidate, errors["cyclic-layer"]
        ),
        "relative_improvement_vs_development_mean": _relative_summary(
            candidate, errors["development-mean"]
        ),
    }
    summaries = {name: _summary(value) for name, value in errors.items()}
    gates = config["gates"]
    checks = {
        "rows": density.shape[0] == int(gates["required_rows"]),
        "folds": len(fold_records) == int(gates["required_fold_count"]),
        "finite": bool(
            all(
                np.all(np.isfinite(row))
                for row in (
                    characteristic_prediction,
                    affine_prediction,
                    direct_prediction,
                    cyclic_prediction,
                )
            )
        )
        is bool(gates["all_values_finite"]),
        "nonnegative": bool(
            np.all(characteristic_prediction >= 0.0)
            and all(
                np.all(np.asarray(record["parameters"])[:, 1:] >= 0.0)
                for record in fold_records
            )
        )
        is bool(gates["all_predicted_coefficients_nonnegative"]),
        "cb5_affine": comparisons["win_fraction_vs_cb5_affine"]
        >= float(gates["win_fraction_vs_cb5_affine_minimum"])
        and comparisons["relative_improvement_vs_cb5_affine"]["median"]
        >= float(gates["median_relative_improvement_vs_cb5_affine_minimum"]),
        "direct_spectral": comparisons["win_fraction_vs_direct_nonnegative_spectral"]
        >= float(gates["win_fraction_vs_direct_nonnegative_spectral_minimum"])
        and comparisons["relative_improvement_vs_direct_nonnegative_spectral"]["median"]
        >= float(
            gates["median_relative_improvement_vs_direct_nonnegative_spectral_minimum"]
        ),
        "cyclic_layer": comparisons["win_fraction_vs_cyclic_layer"]
        >= float(gates["win_fraction_vs_cyclic_layer_minimum"])
        and comparisons["relative_improvement_vs_cyclic_layer"]["median"]
        >= float(gates["median_relative_improvement_vs_cyclic_layer_minimum"]),
        "absolute_error": summaries["characteristic"]["median"]
        <= float(gates["median_density_rmse_maximum"])
        and summaries["characteristic"]["p95"]
        <= float(gates["p95_density_rmse_maximum"])
        and summaries["characteristic"]["maximum"]
        <= float(gates["maximum_density_rmse"]),
        "mean_control": comparisons["relative_improvement_vs_development_mean"][
            "median"
        ]
        >= float(gates["median_relative_improvement_vs_development_mean_minimum"]),
    }
    passed = all(checks.values())
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "contract_sha256": CONTRACT_SHA256,
        "parent_cb4_stable_evidence_id": parents["cb4_required_stable_evidence_id"],
        "source_rgb_sha256": _array_sha256(source_rgb),
        "measured_density_sha256": _array_sha256(density),
        "characteristic_floor_density": floors.tolist(),
        "prediction_sha256": {
            "characteristic": _array_sha256(characteristic_prediction),
            "cb5-affine": _array_sha256(affine_prediction),
            "direct-spectral": _array_sha256(direct_prediction),
            "cyclic-layer": _array_sha256(cyclic_prediction),
            "development-mean": _array_sha256(mean_prediction),
        },
        "folds": fold_records,
        "summaries": summaries,
        "comparisons": comparisons,
        "checks": checks,
        "failed_gates": sorted(name for name, value in checks.items() if not value),
        "passed": passed,
        "decision": (
            "retain_characteristic_constrained_forward_mechanism"
            if passed
            else "close_characteristic_constrained_forward_family"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def write_report(report: Mapping[str, Any], output: Path) -> str:
    payload = canonical_json(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONTRACT_SHA256",
    "FujifilmCharacteristicForwardError",
    "_apply_curve",
    "_characteristic_curves",
    "_fit_characteristic_layer",
    "evaluate_characteristic_forward_proxy",
    "load_contract",
    "write_report",
]
