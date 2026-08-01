"""Frozen U6.P4AP standards-derived autoregressive grain-shape audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.real_uniform_grain_nps import (
    _quadratic_detrend,
    acf_lag_signature,
    cosine_similarity,
    fixed_fractional_crops,
    radial_nps_signature,
)
from src.eval.real_uniform_grain_physical import (
    render_anisotropic_density_region,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file, validate_contract
from src.film_physics.autoregressive_structure import (
    ar_impulse_metrics,
    causal_ar_normal_region,
    causal_ar_positions,
    quantize_ar_coefficients,
)


class StandardArGrainShapeError(RuntimeError):
    """Raised when a parent, source, fit or frozen evaluation drifts."""


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = root / binding["path"]
    if hash_file(path, "sha256") != binding["sha256"]:
        raise StandardArGrainShapeError(f"parent hash mismatch: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StandardArGrainShapeError("bound parent must be an object")
    return payload


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StandardArGrainShapeError("P4AP contract must be an object")
    return payload


def _source_id(expected: dict[str, Any]) -> str:
    title = str(expected["title"])
    if not title.startswith("File:") or not title.endswith(".tif"):
        raise StandardArGrainShapeError("unexpected P4R source title")
    return title[5:-4]


def _normalize(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0 or not np.all(np.isfinite(values)):
        raise StandardArGrainShapeError("degenerate signature")
    output = np.ascontiguousarray(values / norm)
    output.setflags(write=False)
    return output


def _scan_observation(
    *,
    rgb: np.ndarray,
    analysis: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    pixel = analysis["pixel_contract"]
    channels = ("red", "green", "blue")
    nps: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    acf: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    for channel, name in enumerate(channels):
        for crop in fixed_fractional_crops(
            rgb[..., channel],
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        ):
            nps[name].append(
                radial_nps_signature(
                    crop,
                    band_edges_cycles_per_pixel=pixel[
                        "radial_nps_band_edges_cycles_per_pixel"
                    ],
                )
            )
            acf[name].append(
                acf_lag_signature(crop, lags_yx=pixel["acf_lags_pixels_yx"])
            )
    return {
        "nps": {
            name: _normalize(np.median(nps[name], axis=0))
            for name in channels
        },
        "acf": {
            name: np.ascontiguousarray(
                np.median(acf[name], axis=0),
                dtype=np.float64,
            )
            for name in channels
        },
    }


def _standardized_fit_crops(
    *,
    rgb: np.ndarray,
    analysis: dict[str, Any],
) -> list[np.ndarray]:
    pixel = analysis["pixel_contract"]
    rows: list[np.ndarray] = []
    for channel in range(3):
        for crop in fixed_fractional_crops(
            rgb[..., channel],
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        ):
            values = np.asarray(crop, dtype=np.float64)
            mean = float(np.mean(values))
            if mean <= 0.0:
                raise StandardArGrainShapeError("fit crop has nonpositive mean")
            residual = _quadratic_detrend(values / mean)
            deviation = float(np.std(residual))
            if deviation <= 0.0 or not np.all(np.isfinite(residual)):
                raise StandardArGrainShapeError("fit crop is degenerate")
            normalized = np.ascontiguousarray(residual / deviation)
            normalized.setflags(write=False)
            rows.append(normalized)
    return rows


def _design_matrix(
    crop: np.ndarray,
    *,
    maximum_lag: int,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[int, int], ...]]:
    height, width = crop.shape
    positions = causal_ar_positions(maximum_lag)
    target = crop[maximum_lag:, maximum_lag : width - maximum_lag]
    features = []
    for dy, dx in positions:
        features.append(
            crop[
                maximum_lag + dy : height + dy,
                maximum_lag + dx : width - maximum_lag + dx,
            ]
        )
    matrix = np.stack(features, axis=-1).reshape(-1, len(positions))
    return matrix, target.reshape(-1), positions


def _fit_candidates(
    *,
    residuals: list[np.ndarray],
    model: dict[str, Any],
    gates: dict[str, Any],
) -> list[dict[str, Any]]:
    lags = [int(value) for value in model["candidate_lags"]]
    maximum_lag = max(lags)
    maximum_positions = causal_ar_positions(maximum_lag)
    indexes = {
        lag: [maximum_positions.index(position) for position in causal_ar_positions(lag)]
        for lag in lags
    }
    xtx = {
        lag: np.zeros((len(indexes[lag]), len(indexes[lag])), dtype=np.float64)
        for lag in lags
    }
    xty = {
        lag: np.zeros(len(indexes[lag]), dtype=np.float64) for lag in lags
    }
    sample_count = 0
    for crop in residuals:
        matrix, target, observed_positions = _design_matrix(
            crop,
            maximum_lag=maximum_lag,
        )
        if observed_positions != maximum_positions:
            raise StandardArGrainShapeError("AR design position drift")
        count = len(target)
        sample_count += count
        for lag in lags:
            selected = matrix[:, indexes[lag]]
            xtx[lag] += selected.T @ selected / count
            xty[lag] += selected.T @ target / count
    candidates: list[dict[str, Any]] = []
    for lag in lags:
        matrix = xtx[lag] / len(residuals)
        vector = xty[lag] / len(residuals)
        ridge = (
            float(model["ridge_relative_trace"])
            * float(np.trace(matrix))
            / len(vector)
        )
        raw = np.linalg.solve(matrix + ridge * np.eye(len(vector)), vector)
        row: dict[str, Any] = {
            "lag": lag,
            "positions_yx": [list(value) for value in causal_ar_positions(lag)],
            "raw_coefficients": raw.tolist(),
            "ridge": ridge,
            "fit_crop_count": len(residuals),
            "fit_sample_count": sample_count,
            "admitted": False,
        }
        try:
            coefficients = quantize_ar_coefficients(
                raw,
                step=float(model["coefficient_step"]),
                minimum=float(model["minimum_coefficient"]),
                maximum=float(model["maximum_coefficient"]),
            )
            impulse = ar_impulse_metrics(
                coefficients,
                lag=lag,
                shape=tuple(int(value) for value in model["impulse_shape"]),
                tail_width=int(model["impulse_tail_width"]),
            )
            admitted = bool(
                impulse["tail_energy_fraction"]
                <= float(gates["maximum_impulse_tail_energy_fraction"])
                and impulse["peak_absolute"]
                <= float(gates["maximum_impulse_peak_absolute"])
            )
            row.update(
                {
                    "quantized_coefficients": coefficients.tolist(),
                    "impulse": impulse,
                    "admitted": admitted,
                    "rejection": None if admitted else "impulse_stability_gate",
                }
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            row["rejection"] = f"{type(error).__name__}:{error}"
        candidates.append(row)
    return candidates


def _synthetic_signatures(
    *,
    contract: dict[str, Any],
    analysis: dict[str, Any],
    p4t: dict[str, Any],
    seeds: list[int],
    lag: int | None,
    coefficients: np.ndarray | None,
    normalization_energy: float | None,
) -> dict[str, dict[str, np.ndarray]]:
    model = contract["model"]
    field_shape = tuple(int(value) for value in model["field_shape"])
    crop_y, crop_x, crop_height, crop_width = (
        int(value) for value in model["central_analysis_crop"]
    )
    channels = ("red", "green", "blue")
    nps: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    acf: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    pixel = analysis["pixel_contract"]
    candidate = p4t["candidate"]
    for seed in seeds:
        for channel, name in enumerate(channels):
            channel_seed = seed + 104729 * channel
            if lag is None:
                target = np.full(
                    field_shape,
                    float(model["comparison_target_density"]),
                    dtype=np.float64,
                )
                field = render_anisotropic_density_region(
                    target,
                    origin_yx=(0, 0),
                    shape=field_shape,
                    grain_optical_density=float(
                        candidate["grain_optical_density_by_rgb_layer"][channel]
                    ),
                    sigma_yx=tuple(
                        float(value) for value in candidate["sigma_yx_pixels"]
                    ),
                    seed=channel_seed,
                    maximum_target_density=float(candidate["maximum_target_density"]),
                    truncate=float(candidate["truncate"]),
                )
            else:
                if coefficients is None or normalization_energy is None:
                    raise StandardArGrainShapeError("AR signature lacks parameters")
                normal = causal_ar_normal_region(
                    field_shape,
                    origin_yx=(0, 0),
                    shape=field_shape,
                    lag=lag,
                    coefficients=coefficients,
                    seed=channel_seed,
                    normalization_energy=normalization_energy,
                )
                field = 1.0 + float(model["ar_positive_proxy_scale"]) * normal
                if np.min(field) <= 0.0:
                    raise StandardArGrainShapeError("AR positive proxy left domain")
            crop = field[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ]
            nps[name].append(
                radial_nps_signature(
                    crop,
                    band_edges_cycles_per_pixel=pixel[
                        "radial_nps_band_edges_cycles_per_pixel"
                    ],
                )
            )
            acf[name].append(
                acf_lag_signature(crop, lags_yx=pixel["acf_lags_pixels_yx"])
            )
    return {
        "nps": {
            name: _normalize(np.median(rows, axis=0))
            for name, rows in nps.items()
        },
        "acf": {
            name: np.ascontiguousarray(np.median(rows, axis=0), dtype=np.float64)
            for name, rows in acf.items()
        },
    }


def _errors(
    synthetic: dict[str, dict[str, np.ndarray]],
    observed: dict[str, dict[str, dict[str, np.ndarray]]],
) -> dict[str, Any]:
    channels = ("red", "green", "blue")
    synthetic_nps = _normalize(
        np.concatenate([synthetic["nps"][name] for name in channels])
    )
    nps_errors: list[float] = []
    acf_errors: list[float] = []
    for row in observed.values():
        observed_nps = _normalize(
            np.concatenate([row["nps"][name] for name in channels])
        )
        nps_errors.append(1.0 - cosine_similarity(synthetic_nps, observed_nps))
        acf_errors.append(
            float(
                np.median(
                    np.concatenate(
                        [
                            np.abs(synthetic["acf"][name] - row["acf"][name])
                            for name in channels
                        ]
                    )
                )
            )
        )
    return {
        "nps": nps_errors,
        "acf": acf_errors,
        "nps_median": float(np.median(nps_errors)),
        "nps_maximum": float(np.max(nps_errors)),
        "acf_median": float(np.median(acf_errors)),
        "acf_maximum": float(np.max(acf_errors)),
    }


def _reversed_nps(
    signatures: dict[str, dict[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "nps": {
            name: _normalize(value[::-1])
            for name, value in signatures["nps"].items()
        },
        "acf": signatures["acf"],
    }


def _partition_exact(
    *,
    lag: int,
    coefficients: np.ndarray,
    energy: float,
    seed: int,
) -> bool:
    full_shape = (137, 143)
    full = causal_ar_normal_region(
        full_shape,
        origin_yx=(0, 0),
        shape=full_shape,
        lag=lag,
        coefficients=coefficients,
        seed=seed,
        normalization_energy=energy,
    )
    stitched = np.concatenate(
        [
            causal_ar_normal_region(
                full_shape,
                origin_yx=(y0, 0),
                shape=(min(31, full_shape[0] - y0), full_shape[1]),
                lag=lag,
                coefficients=coefficients,
                seed=seed,
                normalization_energy=energy,
            )
            for y0 in range(0, full_shape[0], 31)
        ],
        axis=0,
    )
    return bool(np.array_equal(full, stitched))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Execute development fit, freeze, then sealed confirmation evaluation."""
    if contract.get("schema") != (
        "neuro_film.u6_p4ap_standard_ar_grain_shape_contract.v1"
    ):
        raise StandardArGrainShapeError("unsupported P4AP contract")
    model = contract["model"]
    split = contract["split"]
    if (
        model.get("per_channel_coefficients_allowed") is not False
        or model.get("stock_specific_fit_allowed") is not False
        or model.get("confirmation_refit_allowed") is not False
        or model.get("per_image_normalization_allowed") is not False
        or model.get("display_rgb_noise_allowed") is not False
        or model.get("hard_clipping_allowed") is not False
        or split.get("stock_labels_available_to_fit") is not False
    ):
        raise StandardArGrainShapeError("forbidden AR capacity is enabled")
    parents = contract["parents"]
    source = _load_bound(root, parents["source_contract"])
    analysis = _load_bound(root, parents["analysis_contract"])
    p4r = _load_bound(root, parents["p4r_report"])
    p4t = _load_bound(root, parents["p4t_contract"])
    p4t_decision = _load_bound(root, parents["p4t_decision"])
    _load_bound(root, parents["p4t_report"])
    validate_contract(source)
    if (
        p4r.get("decision") != parents["p4r_report"]["required_decision"]
        or p4t_decision.get("result", {}).get("status")
        != parents["p4t_decision"]["required_status"]
    ):
        raise StandardArGrainShapeError("parent decision does not admit P4AP")
    expected_by_id = {_source_id(row): row for row in source["files"]}
    development_ids = [str(value) for value in split["development_source_ids"]]
    confirmation_ids = [str(value) for value in split["confirmation_source_ids"]]
    if (
        set(expected_by_id) != set(development_ids) | set(confirmation_ids)
        or set(development_ids) & set(confirmation_ids)
    ):
        raise StandardArGrainShapeError("P4AP split does not cover P4R sources")
    expected_sha = {str(row["source_id"]): str(row["sha256"]) for row in p4r["source_rows"]}
    development: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    fit_crops: list[np.ndarray] = []
    for source_id in development_ids:
        expected = expected_by_id[source_id]
        inspected, rgb, _infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(analysis["pixel_contract"]["crop_size_pixels"]),
            centers_yx=analysis["pixel_contract"]["fixed_fractional_centers_yx"],
        )
        if inspected["source_id"] != source_id or inspected["sha256"] != expected_sha[source_id]:
            raise StandardArGrainShapeError("development source identity drift")
        development[source_id] = _scan_observation(rgb=rgb, analysis=analysis)
        fit_crops.extend(_standardized_fit_crops(rgb=rgb, analysis=analysis))
        del rgb, _infrared
    candidates = _fit_candidates(
        residuals=fit_crops,
        model=model,
        gates=contract["automatic_gates"],
    )
    del fit_crops
    admitted = [row for row in candidates if row["admitted"]]
    if not admitted:
        stable = {
            "schema": "neuro_film.u6_p4ap_standard_ar_grain_shape_report.v1",
            "contract_sha256": hash_file(
                root / "configs/u6_p4ap_standard_ar_grain_shape_v1.json", "sha256"
            ),
            "candidate_rows": candidates,
            "confirmation_pixel_reads": 0,
            "automatic_pass": False,
            "decision": "close_no_admitted_ar_candidate_before_confirmation",
            "claim_ceiling": contract["claim_ceiling"],
        }
        stable_id = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {**stable, "stable_evidence_id": stable_id}
    baseline_development = _synthetic_signatures(
        contract=contract,
        analysis=analysis,
        p4t=p4t,
        seeds=[int(value) for value in model["development_seeds"]],
        lag=None,
        coefficients=None,
        normalization_energy=None,
    )
    baseline_dev_errors = _errors(baseline_development, development)
    epsilon = np.finfo(np.float64).eps
    for row in admitted:
        coefficients = np.asarray(row["quantized_coefficients"], dtype=np.float64)
        signatures = _synthetic_signatures(
            contract=contract,
            analysis=analysis,
            p4t=p4t,
            seeds=[int(value) for value in model["development_seeds"]],
            lag=int(row["lag"]),
            coefficients=coefficients,
            normalization_energy=float(row["impulse"]["energy"]),
        )
        errors = _errors(signatures, development)
        nps_ratio = errors["nps_median"] / max(baseline_dev_errors["nps_median"], epsilon)
        acf_ratio = errors["acf_median"] / max(baseline_dev_errors["acf_median"], epsilon)
        row["development_errors"] = errors
        row["selection_key"] = [max(nps_ratio, acf_ratio), nps_ratio + acf_ratio, int(row["lag"])]
    selected = min(admitted, key=lambda row: tuple(row["selection_key"]))
    selected_lag = int(selected["lag"])
    selected_coefficients = np.asarray(selected["quantized_coefficients"], dtype=np.float64)
    selected_energy = float(selected["impulse"]["energy"])
    frozen_candidate = {
        "lag": selected_lag,
        "positions_yx": selected["positions_yx"],
        "quantized_coefficients": selected["quantized_coefficients"],
        "normalization_energy": selected_energy,
        "selection_key": selected["selection_key"],
    }
    confirmation_reads_before_candidate_freeze = 0
    confirmation: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for source_id in confirmation_ids:
        expected = expected_by_id[source_id]
        inspected, rgb, _infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(analysis["pixel_contract"]["crop_size_pixels"]),
            centers_yx=analysis["pixel_contract"]["fixed_fractional_centers_yx"],
        )
        if inspected["source_id"] != source_id or inspected["sha256"] != expected_sha[source_id]:
            raise StandardArGrainShapeError("confirmation source identity drift")
        confirmation[source_id] = _scan_observation(rgb=rgb, analysis=analysis)
        del rgb, _infrared
    confirmation_seeds = [int(value) for value in model["confirmation_seeds"]]
    baseline_confirmation = _synthetic_signatures(
        contract=contract,
        analysis=analysis,
        p4t=p4t,
        seeds=confirmation_seeds,
        lag=None,
        coefficients=None,
        normalization_energy=None,
    )
    candidate_confirmation = _synthetic_signatures(
        contract=contract,
        analysis=analysis,
        p4t=p4t,
        seeds=confirmation_seeds,
        lag=selected_lag,
        coefficients=selected_coefficients,
        normalization_energy=selected_energy,
    )
    repeated_confirmation = _synthetic_signatures(
        contract=contract,
        analysis=analysis,
        p4t=p4t,
        seeds=confirmation_seeds,
        lag=selected_lag,
        coefficients=selected_coefficients,
        normalization_energy=selected_energy,
    )
    baseline_errors = _errors(baseline_confirmation, confirmation)
    candidate_errors = _errors(candidate_confirmation, confirmation)
    reversed_errors = _errors(_reversed_nps(candidate_confirmation), confirmation)
    nps_improvement = 1.0 - candidate_errors["nps_median"] / max(baseline_errors["nps_median"], epsilon)
    nps_worst_ratio = candidate_errors["nps_maximum"] / max(baseline_errors["nps_maximum"], epsilon)
    acf_improvement = 1.0 - candidate_errors["acf_median"] / max(baseline_errors["acf_median"], epsilon)
    acf_worst_ratio = candidate_errors["acf_maximum"] / max(baseline_errors["acf_maximum"], epsilon)
    reversed_improvement = 1.0 - candidate_errors["nps_median"] / max(reversed_errors["nps_median"], epsilon)
    repeat_exact = all(
        np.array_equal(candidate_confirmation[k][name], repeated_confirmation[k][name])
        for k in ("nps", "acf")
        for name in ("red", "green", "blue")
    )
    partition_exact = _partition_exact(
        lag=selected_lag,
        coefficients=selected_coefficients,
        energy=selected_energy,
        seed=confirmation_seeds[0],
    )
    input_hashes_unchanged = all(
        hash_file(root / expected_by_id[source_id]["path"], "sha256") == expected_sha[source_id]
        for source_id in development_ids + confirmation_ids
    )
    gates = contract["automatic_gates"]
    gate_results = {
        "confirmation_nps_median_improvement": nps_improvement >= float(gates["minimum_confirmation_nps_median_improvement_over_p4t"]),
        "confirmation_nps_worst_not_worse": nps_worst_ratio <= float(gates["maximum_confirmation_nps_worst_error_ratio_to_p4t"]),
        "confirmation_acf_median_improvement": acf_improvement >= float(gates["minimum_confirmation_acf_median_improvement_over_p4t"]),
        "confirmation_acf_worst_not_worse": acf_worst_ratio <= float(gates["maximum_confirmation_acf_worst_error_ratio_to_p4t"]),
        "reversed_frequency_control": reversed_improvement >= float(gates["minimum_candidate_vs_reversed_frequency_control_improvement"]),
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
        "input_hashes_unchanged": input_hashes_unchanged,
        "stock_labels_used_for_fit": False,
        "confirmation_reads_before_candidate_freeze": confirmation_reads_before_candidate_freeze == 0,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p4ap_standard_ar_grain_shape_report.v1",
        "contract_sha256": hash_file(root / "configs/u6_p4ap_standard_ar_grain_shape_v1.json", "sha256"),
        "development_source_ids": development_ids,
        "confirmation_source_ids": confirmation_ids,
        "candidate_rows": candidates,
        "selected_candidate": frozen_candidate,
        "development_baseline_errors": baseline_dev_errors,
        "confirmation": {
            "baseline_errors": baseline_errors,
            "candidate_errors": candidate_errors,
            "reversed_frequency_errors": reversed_errors,
            "nps_median_improvement_over_p4t": nps_improvement,
            "nps_worst_error_ratio_to_p4t": nps_worst_ratio,
            "acf_median_improvement_over_p4t": acf_improvement,
            "acf_worst_error_ratio_to_p4t": acf_worst_ratio,
            "candidate_vs_reversed_frequency_improvement": reversed_improvement,
        },
        "confirmation_pixel_reads": len(confirmation),
        "confirmation_reads_before_candidate_freeze": confirmation_reads_before_candidate_freeze,
        "stock_labels_used_for_fit": False,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_generic_standard_ar_source_candidate"
            if automatic_pass
            else "close_standard_ar_source_below_confirmation_gate"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["StandardArGrainShapeError", "load_contract", "run_audit", "write_report"]
