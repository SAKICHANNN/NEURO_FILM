"""Frozen U6.P4AR nonnegative spectral-mixture grain-shape audit."""

from __future__ import annotations

import hashlib
import json
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls

from src.eval.real_uniform_grain_nps import (
    _quadratic_detrend,
    fixed_fractional_crops,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file, validate_contract
from src.eval.standard_ar_grain_shape import (
    _errors,
    _load_bound,
    _reversed_nps,
    _scan_observation,
    _source_id,
    _synthetic_signatures,
)
from src.film_physics.spectral_structure import (
    quantize_simplex_weights,
    spectral_normal_region,
)


class SpectralMixtureGrainShapeError(RuntimeError):
    """Raised when a parent, source, fit or frozen evaluation drifts."""


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SpectralMixtureGrainShapeError("P4AR contract must be an object")
    return payload


def _linear_radial_profile(
    crop: np.ndarray,
    *,
    edges: list[float],
) -> np.ndarray:
    values = np.asarray(crop, dtype=np.float64)
    mean = float(np.mean(values))
    if values.ndim != 2 or mean <= 0.0 or not np.all(np.isfinite(values)):
        raise SpectralMixtureGrainShapeError("invalid spectral fit crop")
    relative = _quadratic_detrend(values / mean)
    window = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    power = np.square(np.abs(np.fft.rfft2(relative * window)))
    fy = np.fft.fftfreq(values.shape[0])[:, None]
    fx = np.fft.rfftfreq(values.shape[1])[None, :]
    radius = np.sqrt(fx * fx + fy * fy)
    profile = np.asarray(
        [
            float(np.mean(power[(radius >= lower) & (radius < upper)]))
            for lower, upper in pairwise(edges)
        ],
        dtype=np.float64,
    )
    total = float(np.sum(profile))
    if total <= 0.0 or not np.all(np.isfinite(profile)):
        raise SpectralMixtureGrainShapeError("degenerate spectral fit profile")
    output = np.ascontiguousarray(profile / total)
    output.setflags(write=False)
    return output


def _fit_profiles(
    *,
    rgb: np.ndarray,
    analysis: dict[str, Any],
) -> list[np.ndarray]:
    pixel = analysis["pixel_contract"]
    profiles: list[np.ndarray] = []
    for channel in range(3):
        for crop in fixed_fractional_crops(
            rgb[..., channel],
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        ):
            profiles.append(
                _linear_radial_profile(
                    crop,
                    edges=pixel["radial_nps_band_edges_cycles_per_pixel"],
                )
            )
    return profiles


def _spectral_field(
    *,
    model: dict[str, Any],
    weights: np.ndarray,
    seed: int,
) -> np.ndarray:
    field_shape = tuple(int(value) for value in model["field_shape"])
    return spectral_normal_region(
        field_shape,
        origin_yx=(0, 0),
        shape=field_shape,
        centers_cycles_per_pixel=np.asarray(
            model["center_frequencies_cycles_per_pixel"], dtype=np.float64
        ),
        bandwidth_octaves=float(model["bandwidth_octaves"]),
        weights=weights,
        white_floor_fraction=float(model["white_floor_fraction"]),
        seed=seed,
    )


def _basis_profiles(
    *,
    model: dict[str, Any],
    analysis: dict[str, Any],
) -> np.ndarray:
    component_count = len(model["center_frequencies_cycles_per_pixel"])
    pixel = analysis["pixel_contract"]
    crop_y, crop_x, crop_height, crop_width = (
        int(value) for value in model["central_analysis_crop"]
    )
    columns = []
    for component in range(component_count):
        weights = np.zeros(component_count, dtype=np.float64)
        weights[component] = 1.0
        rows = []
        for seed in model["development_seeds"]:
            field = _spectral_field(model=model, weights=weights, seed=int(seed))
            positive = 1.0 + float(model["positive_proxy_scale"]) * field
            crop = positive[
                crop_y : crop_y + crop_height,
                crop_x : crop_x + crop_width,
            ]
            rows.append(
                _linear_radial_profile(
                    crop,
                    edges=pixel["radial_nps_band_edges_cycles_per_pixel"],
                )
            )
        column = np.median(rows, axis=0)
        column /= float(np.sum(column))
        columns.append(column)
    return np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)


def _candidate_signatures(
    *,
    contract: dict[str, Any],
    analysis: dict[str, Any],
    weights: np.ndarray,
    seeds: list[int],
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    model = contract["model"]
    crop_y, crop_x, crop_height, crop_width = (
        int(value) for value in model["central_analysis_crop"]
    )
    channels = ("red", "green", "blue")
    nps: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    acf: dict[str, list[np.ndarray]] = {name: [] for name in channels}
    pixel = analysis["pixel_contract"]
    minimum = float("inf")
    from src.eval.real_uniform_grain_nps import acf_lag_signature, radial_nps_signature

    for seed in seeds:
        for channel, name in enumerate(channels):
            field = _spectral_field(
                model=model,
                weights=weights,
                seed=seed + 104729 * channel,
            )
            minimum = min(minimum, float(np.min(field)))
            positive = 1.0 + float(model["positive_proxy_scale"]) * field
            if np.min(positive) <= 0.0:
                raise SpectralMixtureGrainShapeError("positive proxy left domain")
            crop = positive[
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
    signatures = {
        "nps": {
            name: _normalize_signature(np.median(rows, axis=0))
            for name, rows in nps.items()
        },
        "acf": {
            name: np.ascontiguousarray(np.median(rows, axis=0), dtype=np.float64)
            for name, rows in acf.items()
        },
    }
    return signatures, minimum


def _normalize_signature(value: np.ndarray) -> np.ndarray:
    values = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0 or not np.all(np.isfinite(values)):
        raise SpectralMixtureGrainShapeError("degenerate synthetic signature")
    output = np.ascontiguousarray(values / norm)
    output.setflags(write=False)
    return output


def _partition_exact(*, model: dict[str, Any], weights: np.ndarray, seed: int) -> bool:
    full_shape = (137, 143)
    kwargs = {
        "full_shape": full_shape,
        "centers_cycles_per_pixel": np.asarray(
            model["center_frequencies_cycles_per_pixel"], dtype=np.float64
        ),
        "bandwidth_octaves": float(model["bandwidth_octaves"]),
        "weights": weights,
        "white_floor_fraction": float(model["white_floor_fraction"]),
        "seed": seed,
    }
    full = spectral_normal_region(**kwargs, origin_yx=(0, 0), shape=full_shape)
    stitched = np.concatenate(
        [
            spectral_normal_region(
                **kwargs,
                origin_yx=(y0, 0),
                shape=(min(31, full_shape[0] - y0), full_shape[1]),
            )
            for y0 in range(0, full_shape[0], 31)
        ],
        axis=0,
    )
    return bool(np.array_equal(full, stitched))


def run_audit(*, root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    """Fit on development groups, freeze one PSD, then evaluate confirmation."""
    if contract.get("schema") != (
        "neuro_film.u6_p4ar_spectral_mixture_grain_shape_contract.v1"
    ):
        raise SpectralMixtureGrainShapeError("unsupported P4AR contract")
    model = contract["model"]
    split = contract["split"]
    if (
        model.get("per_channel_weights_allowed") is not False
        or model.get("stock_specific_fit_allowed") is not False
        or model.get("confirmation_refit_allowed") is not False
        or model.get("per_image_normalization_allowed") is not False
        or model.get("display_rgb_noise_allowed") is not False
        or model.get("hard_clipping_allowed") is not False
        or split.get("stock_labels_available_to_fit") is not False
    ):
        raise SpectralMixtureGrainShapeError("forbidden spectral capacity is enabled")
    parents = contract["parents"]
    source = _load_bound(root, parents["source_contract"])
    analysis = _load_bound(root, parents["analysis_contract"])
    p4r = _load_bound(root, parents["p4r_report"])
    p4t = _load_bound(root, parents["p4t_contract"])
    p4t_decision = _load_bound(root, parents["p4t_decision"])
    _load_bound(root, parents["p4t_report"])
    _load_bound(root, parents["p4ap_contract"])
    p4ap = _load_bound(root, parents["p4ap_evidence"])
    validate_contract(source)
    if (
        p4r.get("decision") != parents["p4r_report"]["required_decision"]
        or p4t_decision.get("result", {}).get("status")
        != parents["p4t_decision"]["required_status"]
        or p4ap.get("decision") != parents["p4ap_evidence"]["required_decision"]
    ):
        raise SpectralMixtureGrainShapeError("parent decision does not admit P4AR")
    expected_by_id = {_source_id(row): row for row in source["files"]}
    development_ids = [str(value) for value in split["development_source_ids"]]
    confirmation_ids = [str(value) for value in split["confirmation_source_ids"]]
    if (
        set(expected_by_id) != set(development_ids) | set(confirmation_ids)
        or set(development_ids) & set(confirmation_ids)
    ):
        raise SpectralMixtureGrainShapeError("P4AR split does not cover P4R sources")
    expected_sha = {
        str(row["source_id"]): str(row["sha256"]) for row in p4r["source_rows"]
    }
    development: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    fit_profiles: list[np.ndarray] = []
    for source_id in development_ids:
        expected = expected_by_id[source_id]
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(analysis["pixel_contract"]["crop_size_pixels"]),
            centers_yx=analysis["pixel_contract"]["fixed_fractional_centers_yx"],
        )
        if inspected["source_id"] != source_id or inspected["sha256"] != expected_sha[source_id]:
            raise SpectralMixtureGrainShapeError("development source identity drift")
        development[source_id] = _scan_observation(rgb=rgb, analysis=analysis)
        fit_profiles.extend(_fit_profiles(rgb=rgb, analysis=analysis))
        del rgb, infrared
    target = np.median(fit_profiles, axis=0)
    target /= float(np.sum(target))
    basis = _basis_profiles(model=model, analysis=analysis)
    raw_weights, residual_norm = nnls(basis, target)
    weights = quantize_simplex_weights(
        raw_weights,
        denominator=int(model["weight_quantization_denominator"]),
    )
    frozen_bundle = {
        "center_frequencies_cycles_per_pixel": model[
            "center_frequencies_cycles_per_pixel"
        ],
        "bandwidth_octaves": model["bandwidth_octaves"],
        "white_floor_fraction": model["white_floor_fraction"],
        "quantized_weights": weights.tolist(),
        "active_components": int(np.count_nonzero(weights)),
        "maximum_component_weight": float(np.max(weights)),
        "nnls_residual_norm": float(residual_norm),
    }
    baseline_development = _synthetic_signatures(
        contract=contract,
        analysis=analysis,
        p4t=p4t,
        seeds=[int(value) for value in model["development_seeds"]],
        lag=None,
        coefficients=None,
        normalization_energy=None,
    )
    candidate_development, development_minimum = _candidate_signatures(
        contract=contract,
        analysis=analysis,
        weights=weights,
        seeds=[int(value) for value in model["development_seeds"]],
    )
    baseline_dev_errors = _errors(baseline_development, development)
    candidate_dev_errors = _errors(candidate_development, development)
    epsilon = np.finfo(np.float64).eps
    development_nps_improvement = 1.0 - candidate_dev_errors["nps_median"] / max(
        baseline_dev_errors["nps_median"], epsilon
    )
    development_acf_improvement = 1.0 - candidate_dev_errors["acf_median"] / max(
        baseline_dev_errors["acf_median"], epsilon
    )
    development_gates = contract["development_gates_before_confirmation"]
    development_gate_results = {
        "minimum_active_components": frozen_bundle["active_components"]
        >= int(development_gates["minimum_active_components"]),
        "maximum_single_component_weight": frozen_bundle["maximum_component_weight"]
        <= float(development_gates["maximum_single_component_weight"]),
        "nps_median_improvement": development_nps_improvement
        >= float(development_gates["minimum_nps_median_improvement_over_p4t"]),
        "acf_median_improvement": development_acf_improvement
        >= float(development_gates["minimum_acf_median_improvement_over_p4t"]),
    }
    contract_sha = hash_file(
        root / "configs/u6_p4ar_spectral_mixture_grain_shape_v1.json", "sha256"
    )
    if not all(development_gate_results.values()):
        stable = {
            "schema": "neuro_film.u6_p4ar_spectral_mixture_grain_shape_report.v1",
            "contract_sha256": contract_sha,
            "development_source_ids": development_ids,
            "frozen_bundle": frozen_bundle,
            "development": {
                "target_linear_profile": target.tolist(),
                "baseline_errors": baseline_dev_errors,
                "candidate_errors": candidate_dev_errors,
                "nps_median_improvement_over_p4t": development_nps_improvement,
                "acf_median_improvement_over_p4t": development_acf_improvement,
                "minimum_field_value": development_minimum,
            },
            "development_gate_results": development_gate_results,
            "confirmation_pixel_reads": 0,
            "automatic_pass": False,
            "decision": "close_spectral_mixture_before_confirmation",
            "claim_ceiling": contract["claim_ceiling"],
        }
        stable_id = hashlib.sha256(
            json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        return {**stable, "stable_evidence_id": stable_id}
    confirmation: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    confirmation_reads_before_bundle_freeze = 0
    for source_id in confirmation_ids:
        expected = expected_by_id[source_id]
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=root / expected["path"],
            expected=expected,
            crop_size=int(analysis["pixel_contract"]["crop_size_pixels"]),
            centers_yx=analysis["pixel_contract"]["fixed_fractional_centers_yx"],
        )
        if inspected["source_id"] != source_id or inspected["sha256"] != expected_sha[source_id]:
            raise SpectralMixtureGrainShapeError("confirmation source identity drift")
        confirmation[source_id] = _scan_observation(rgb=rgb, analysis=analysis)
        del rgb, infrared
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
    candidate_confirmation, confirmation_minimum = _candidate_signatures(
        contract=contract,
        analysis=analysis,
        weights=weights,
        seeds=confirmation_seeds,
    )
    repeated_confirmation, repeated_minimum = _candidate_signatures(
        contract=contract,
        analysis=analysis,
        weights=weights,
        seeds=confirmation_seeds,
    )
    baseline_errors = _errors(baseline_confirmation, confirmation)
    candidate_errors = _errors(candidate_confirmation, confirmation)
    reversed_errors = _errors(_reversed_nps(candidate_confirmation), confirmation)
    nps_improvement = 1.0 - candidate_errors["nps_median"] / max(
        baseline_errors["nps_median"], epsilon
    )
    nps_worst_ratio = candidate_errors["nps_maximum"] / max(
        baseline_errors["nps_maximum"], epsilon
    )
    acf_improvement = 1.0 - candidate_errors["acf_median"] / max(
        baseline_errors["acf_median"], epsilon
    )
    acf_worst_ratio = candidate_errors["acf_maximum"] / max(
        baseline_errors["acf_maximum"], epsilon
    )
    reversed_improvement = 1.0 - candidate_errors["nps_median"] / max(
        reversed_errors["nps_median"], epsilon
    )
    repeat_exact = bool(
        confirmation_minimum == repeated_minimum
        and all(
            np.array_equal(candidate_confirmation[k][name], repeated_confirmation[k][name])
            for k in ("nps", "acf")
            for name in ("red", "green", "blue")
        )
    )
    partition_exact = _partition_exact(
        model=model,
        weights=weights,
        seed=confirmation_seeds[0],
    )
    input_hashes_unchanged = all(
        hash_file(root / expected_by_id[source_id]["path"], "sha256")
        == expected_sha[source_id]
        for source_id in development_ids + confirmation_ids
    )
    gates = contract["confirmation_gates"]
    gate_results = {
        "nps_median_improvement": nps_improvement
        >= float(gates["minimum_nps_median_improvement_over_p4t"]),
        "nps_worst_not_worse": nps_worst_ratio
        <= float(gates["maximum_nps_worst_error_ratio_to_p4t"]),
        "acf_median_improvement": acf_improvement
        >= float(gates["minimum_acf_median_improvement_over_p4t"]),
        "acf_worst_not_worse": acf_worst_ratio
        <= float(gates["maximum_acf_worst_error_ratio_to_p4t"]),
        "reversed_frequency_control": reversed_improvement
        >= float(gates["minimum_candidate_vs_reversed_frequency_control_improvement"]),
        "minimum_field_value": confirmation_minimum
        >= float(gates["minimum_field_value_before_positive_proxy"]),
        "repeat_exact": repeat_exact,
        "row_partition_exact": partition_exact,
        "input_hashes_unchanged": input_hashes_unchanged,
        "stock_labels_used_for_fit": False,
        "confirmation_reads_before_bundle_freeze": confirmation_reads_before_bundle_freeze == 0,
    }
    automatic_pass = all(gate_results.values())
    stable = {
        "schema": "neuro_film.u6_p4ar_spectral_mixture_grain_shape_report.v1",
        "contract_sha256": contract_sha,
        "development_source_ids": development_ids,
        "confirmation_source_ids": confirmation_ids,
        "frozen_bundle": frozen_bundle,
        "development": {
            "target_linear_profile": target.tolist(),
            "baseline_errors": baseline_dev_errors,
            "candidate_errors": candidate_dev_errors,
            "nps_median_improvement_over_p4t": development_nps_improvement,
            "acf_median_improvement_over_p4t": development_acf_improvement,
            "minimum_field_value": development_minimum,
        },
        "development_gate_results": development_gate_results,
        "confirmation": {
            "baseline_errors": baseline_errors,
            "candidate_errors": candidate_errors,
            "reversed_frequency_errors": reversed_errors,
            "nps_median_improvement_over_p4t": nps_improvement,
            "nps_worst_error_ratio_to_p4t": nps_worst_ratio,
            "acf_median_improvement_over_p4t": acf_improvement,
            "acf_worst_error_ratio_to_p4t": acf_worst_ratio,
            "candidate_vs_reversed_frequency_improvement": reversed_improvement,
            "minimum_field_value": confirmation_minimum,
        },
        "confirmation_pixel_reads": len(confirmation),
        "confirmation_reads_before_bundle_freeze": confirmation_reads_before_bundle_freeze,
        "stock_labels_used_for_fit": False,
        "gate_results": gate_results,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_generic_spectral_mixture_source_candidate"
            if automatic_pass
            else "close_spectral_mixture_below_confirmation_gate"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_id = hashlib.sha256(
        json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    return {**stable, "stable_evidence_id": stable_id}


def write_report(report: dict[str, Any], path: Path) -> str:
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return hashlib.sha256(encoded).hexdigest()


__all__ = ["SpectralMixtureGrainShapeError", "load_contract", "run_audit", "write_report"]
