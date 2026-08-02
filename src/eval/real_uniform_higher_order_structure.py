"""Higher-order comparison of fixed Gaussian and bounded cloud fields."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.ndimage import label

from src.eval.real_uniform_grain_nps import (
    fixed_fractional_crops,
    standardized_quadratic_residual,
)
from src.eval.real_uniform_grain_preflight import inspect_uniform_grain_tiff
from src.eval.real_uniform_grain_source import hash_file
from src.film_physics.bounded_cloud_occupancy import render_bounded_cloud_region
from src.film_physics.finite_support_thomas import (
    render_finite_support_thomas_region,
)

SCHEMA = "neuro_film.u6_p4cf_real_uniform_higher_order_structure_contract.v1"
REPORT_SCHEMA = "neuro_film.u6_p4cf_real_uniform_higher_order_structure_report.v1"


class RealUniformHigherOrderError(RuntimeError):
    """Raised when evidence or the frozen comparison contract drifts."""


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    if not path.is_file() or hash_file(path, "sha256") != binding["sha256"]:
        raise RealUniformHigherOrderError(f"parent hash mismatch: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RealUniformHigherOrderError("bound parent must be a JSON object")
    return value


def validate_contract(contract: Mapping[str, Any]) -> None:
    models = contract.get("fixed_models", {})
    processing = contract.get("preprocessing", {})
    metrics = contract.get("higher_order_metrics", {})
    gates = contract.get("automatic_gates", {})
    if (
        contract.get("schema") != SCHEMA
        or models.get("occupancy_trials") != 16
        or models.get("occupancy_probability") != 0.5
        or len(models.get("model_realization_seeds", [])) != 8
        or processing.get("crop_size_pixels") != 512
        or processing.get("local_energy_block_pixels") != 16
        or metrics.get("excursion_thresholds_sigma") != [2.5, 3.0]
        or gates.get("minimum_development_scans_won") != 6
        or gates.get("minimum_confirmation_scans_won") != 2
        or gates.get("maximum_confirmation_median_metric_ratio") != 0.9
        or gates.get("require_two_byte_identical_reports") is not True
        or contract["observed_roles"].get("refit_rescale_or_feature_selection_allowed")
        is not False
    ):
        raise RealUniformHigherOrderError("P4CF frozen contract drift")


def structure_features(
    values: np.ndarray,
    contract: Mapping[str, Any],
    *,
    relative_to_mean: bool,
) -> dict[str, np.ndarray]:
    """Build the three candidate-independent higher-order feature families."""
    residual = standardized_quadratic_residual(
        values, relative_to_mean=relative_to_mean
    )
    metrics = contract["higher_order_metrics"]
    processing = contract["preprocessing"]
    marginal = np.quantile(
        residual,
        np.asarray(metrics["marginal_quantile_probabilities"], dtype=np.float64),
        method="linear",
    )
    block = int(processing["local_energy_block_pixels"])
    if residual.shape[0] % block or residual.shape[1] % block:
        raise RealUniformHigherOrderError("crop is not divisible by local block")
    local_rms = np.sqrt(
        np.mean(
            np.square(
                residual.reshape(
                    residual.shape[0] // block,
                    block,
                    residual.shape[1] // block,
                    block,
                )
            ),
            axis=(1, 3),
            dtype=np.float64,
        )
    )
    energy = np.quantile(
        local_rms,
        np.asarray(metrics["local_rms_quantile_probabilities"], dtype=np.float64),
        method="linear",
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    per_megapixel = 1_000_000.0 / residual.size
    topology: list[float] = []
    bins = metrics["excursion_component_area_bins_pixels"]
    for threshold in metrics["excursion_thresholds_sigma"]:
        for mask in (residual >= float(threshold), residual <= -float(threshold)):
            components, count = label(mask, structure=structure)
            sizes = np.bincount(components.reshape(-1), minlength=count + 1)[1:]
            for lower, upper in bins:
                selected = int(np.count_nonzero((sizes >= lower) & (sizes <= upper)))
                topology.append(math.log1p(selected * per_megapixel))
    result = {
        "marginal_quantiles": np.ascontiguousarray(marginal, dtype=np.float64),
        "local_rms_quantiles": np.ascontiguousarray(energy, dtype=np.float64),
        "excursion_topology": np.ascontiguousarray(topology, dtype=np.float64),
    }
    if any(not np.all(np.isfinite(item)) for item in result.values()):
        raise RealUniformHigherOrderError("higher-order feature is non-finite")
    return result


def _aggregate_features(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise RealUniformHigherOrderError("feature aggregation is empty")
    return {
        key: np.ascontiguousarray(
            np.median(np.asarray([row[key] for row in rows]), axis=0),
            dtype=np.float64,
        )
        for key in rows[0]
    }


def compare_feature_sets(
    observed: Mapping[str, np.ndarray],
    gaussian: Mapping[str, np.ndarray],
    bounded: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    errors: dict[str, dict[str, float]] = {}
    ratios: list[float] = []
    bounded_wins = 0
    for key in ("marginal_quantiles", "local_rms_quantiles", "excursion_topology"):
        target = np.asarray(observed[key], dtype=np.float64)
        gaussian_error = float(
            np.sqrt(np.mean(np.square(target - np.asarray(gaussian[key]))))
        )
        bounded_error = float(
            np.sqrt(np.mean(np.square(target - np.asarray(bounded[key]))))
        )
        ratio = bounded_error / max(gaussian_error, np.finfo(np.float64).tiny)
        errors[key] = {
            "gaussian_rmse": gaussian_error,
            "bounded_rmse": bounded_error,
            "bounded_to_gaussian_ratio": ratio,
        }
        ratios.append(ratio)
        bounded_wins += int(bounded_error < gaussian_error)
    return {
        "metrics": errors,
        "bounded_metric_win_count": bounded_wins,
        "bounded_scan_win": bounded_wins >= 2,
        "median_metric_ratio": float(np.median(ratios)),
    }


def _model_features(contract: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    model = contract["fixed_models"]
    shape = tuple(int(v) for v in contract["preprocessing"]["model_field_shape"])
    common = {
        "full_shape": shape,
        "origin_yx": (0, 0),
        "shape": shape,
        "particle_sigma_pixels": float(model["particle_sigma_pixels"]),
        "cluster_sigma_pixels": float(model["cluster_sigma_pixels"]),
        "mean_offspring": float(model["mean_offspring"]),
        "component_seeds": tuple(int(v) for v in model["component_seeds"]),
        "truncate": float(model["truncate_sigma"]),
    }
    gaussian_rows: list[dict[str, np.ndarray]] = []
    bounded_rows: list[dict[str, np.ndarray]] = []
    gaussian_hashes: list[str] = []
    bounded_hashes: list[str] = []
    for seed in model["model_realization_seeds"]:
        gaussian = render_finite_support_thomas_region(
            **common, realization_seed=int(seed)
        )
        bounded = render_bounded_cloud_region(
            **common,
            realization_seed=int(seed),
            trials=int(model["occupancy_trials"]),
            probability=float(model["occupancy_probability"]),
        )
        gaussian_hashes.append(hashlib.sha256(gaussian.tobytes()).hexdigest())
        bounded_hashes.append(hashlib.sha256(bounded.tobytes()).hexdigest())
        gaussian_rows.append(
            structure_features(gaussian, contract, relative_to_mean=False)
        )
        bounded_rows.append(
            structure_features(bounded, contract, relative_to_mean=False)
        )
    distinct = bool(
        len(set(gaussian_hashes)) == len(gaussian_hashes)
        and len(set(bounded_hashes)) == len(bounded_hashes)
        and set(gaussian_hashes).isdisjoint(bounded_hashes)
    )
    return (
        _aggregate_features(gaussian_rows),
        _aggregate_features(bounded_rows),
        {
            "gaussian_field_sha256": gaussian_hashes,
            "bounded_field_sha256": bounded_hashes,
            "all_fields_distinct": distinct,
        },
    )


def _colour_observations(
    root: Path, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    parents = contract["parents"]
    source = _load_bound(root, parents["colour_source_contract"])
    acquisition = _load_bound(root, parents["colour_acquisition_manifest"])
    signature = _load_bound(root, parents["colour_signature_contract"])
    manifest = {str(row["title"]): row for row in acquisition["rows"]}
    pixel = signature["pixel_contract"]
    rows: list[dict[str, Any]] = []
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        observed = manifest[str(expected["title"])]
        path = root / str(expected["path"])
        if hash_file(path, "sha256") != observed["sha256"]:
            raise RealUniformHigherOrderError("colour source payload drift")
        inspected, rgb, infrared = inspect_uniform_grain_tiff(
            path=path,
            expected=expected,
            crop_size=int(pixel["crop_size_pixels"]),
            centers_yx=pixel["fixed_fractional_centers_yx"],
        )
        del infrared
        features: list[dict[str, np.ndarray]] = []
        for channel in range(3):
            for crop in fixed_fractional_crops(
                rgb[..., channel],
                crop_size=int(pixel["crop_size_pixels"]),
                centers_yx=pixel["fixed_fractional_centers_yx"],
            ):
                features.append(
                    structure_features(crop, contract, relative_to_mean=True)
                )
        del rgb
        rows.append(
            {
                "source_id": inspected["source_id"],
                "sha256": observed["sha256"],
                "features": _aggregate_features(features),
            }
        )
    if len(rows) != 8:
        raise RealUniformHigherOrderError("colour development count drift")
    return rows


def _bw_observations(
    root: Path, contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    parents = contract["parents"]
    preflight = _load_bound(root, parents["bw_preflight_contract"])
    decision = _load_bound(root, parents["bw_preflight_decision"])
    if (
        decision.get("decision") != "open_fit_forbidden_bw_model_confirmation"
        or decision.get("automatic_pass") is not True
    ):
        raise RealUniformHigherOrderError("B&W source admission drift")
    source = _load_bound(root, preflight["parents"]["source_contract"])
    acquisition = _load_bound(root, preflight["parents"]["acquisition_manifest"])
    manifest = {str(row["title"]): row for row in acquisition["rows"]}
    crop_contract = preflight["crop_contract"]
    rows: list[dict[str, Any]] = []
    for expected in sorted(source["files"], key=lambda row: str(row["title"])):
        observed = manifest[str(expected["title"])]
        path = root / str(expected["path"])
        if hash_file(path, "sha256") != observed["sha256"]:
            raise RealUniformHigherOrderError("B&W source payload drift")
        with Image.open(path) as image:
            values = np.asarray(image)
        features = [
            structure_features(crop, contract, relative_to_mean=True)
            for crop in fixed_fractional_crops(
                values,
                crop_size=int(crop_contract["crop_size_pixels"]),
                centers_yx=crop_contract["fixed_fractional_centers_yx"],
            )
        ]
        del values
        rows.append(
            {
                "source_id": Path(expected["path"]).stem,
                "sha256": observed["sha256"],
                "features": _aggregate_features(features),
            }
        )
    if len(rows) != int(decision["observed_results"]["source_count"]):
        raise RealUniformHigherOrderError("B&W confirmation count drift")
    return rows


def _role_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    ratios = [float(row["comparison"]["median_metric_ratio"]) for row in rows]
    return {
        "source_count": len(rows),
        "bounded_scan_win_count": sum(
            bool(row["comparison"]["bounded_scan_win"]) for row in rows
        ),
        "median_metric_ratio": float(np.median(ratios)),
        "worst_scan_metric_ratio": float(np.max(ratios)),
    }


def evaluate_real_uniform_higher_order_structure(
    contract: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    """Evaluate frozen models on colour development then B&W confirmation."""
    validate_contract(contract)
    p4bs = _load_bound(root, contract["parents"]["p4bs_decision"])
    p4ca = _load_bound(root, contract["parents"]["p4ca_decision"])
    p4bu = _load_bound(root, contract["parents"]["p4bu_decision"])
    if (
        p4bs.get("decision")
        != contract["parents"]["p4bs_decision"]["required_decision"]
        or p4ca.get("decision")
        != contract["parents"]["p4ca_decision"]["required_decision"]
        or p4bu.get("decision")
        != contract["parents"]["p4bu_decision"]["required_decision"]
    ):
        raise RealUniformHigherOrderError("model parent decision drift")
    gaussian, bounded, model_evidence = _model_features(contract)
    development = _colour_observations(root, contract)
    for row in development:
        row["comparison"] = compare_feature_sets(row.pop("features"), gaussian, bounded)
    confirmation = _bw_observations(root, contract)
    for row in confirmation:
        row["comparison"] = compare_feature_sets(row.pop("features"), gaussian, bounded)
    development_summary = _role_summary(development)
    confirmation_summary = _role_summary(confirmation)
    gates = contract["automatic_gates"]
    checks = {
        "second_order_radial_parity": float(
            p4ca["radial_signature_cosine_error_vs_gaussian"]
        )
        <= float(gates["maximum_parent_radial_signature_cosine_error"]),
        "second_order_covariance_parity": float(
            p4ca["maximum_covariance_absolute_error_vs_gaussian"]
        )
        <= float(gates["maximum_parent_covariance_absolute_error"]),
        "development_scan_wins": development_summary["bounded_scan_win_count"]
        >= int(gates["minimum_development_scans_won"]),
        "development_median_gain": development_summary["median_metric_ratio"]
        <= float(gates["maximum_development_median_metric_ratio"]),
        "confirmation_scan_wins": confirmation_summary["bounded_scan_win_count"]
        >= int(gates["minimum_confirmation_scans_won"]),
        "confirmation_median_gain": confirmation_summary["median_metric_ratio"]
        <= float(gates["maximum_confirmation_median_metric_ratio"]),
        "confirmation_worst_tail": confirmation_summary["worst_scan_metric_ratio"]
        <= float(gates["maximum_confirmation_worst_scan_metric_ratio"]),
        "model_seed_distinctness": bool(model_evidence["all_fields_distinct"]),
    }
    automatic_pass = bool(all(checks.values()))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "experiment_id": contract["experiment_id"],
        "contract_sha256": hashlib.sha256(_canonical_json(contract)).hexdigest(),
        "parent_stable_evidence": {
            "p4bs": p4bs["stable_evidence_id"],
            "p4ca": p4ca["stable_evidence_id"],
        },
        "model_evidence": model_evidence,
        "development": development,
        "confirmation": confirmation,
        "development_summary": development_summary,
        "confirmation_summary": confirmation_summary,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_bounded_occupancy_higher_order_mechanism_candidate"
            if automatic_pass
            else "close_bounded_occupancy_higher_order_mechanism_without_rescue"
        ),
        "claim_ceiling": contract["claim_ceiling"],
    }
    stable_payload = dict(report)
    report["stable_evidence_id"] = hashlib.sha256(
        _canonical_json(stable_payload)
    ).hexdigest()
    return report


def write_report(report: Mapping[str, Any], path: Path) -> str:
    payload = _canonical_json(report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "RealUniformHigherOrderError",
    "compare_feature_sets",
    "evaluate_real_uniform_higher_order_structure",
    "structure_features",
    "validate_contract",
    "write_report",
]
