"""Scene-held explicit operator development on the BO2 weak-pair source."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import least_squares

from src.eval.flickr_single_author_pair_acquisition import canonical_bytes, sha256_file
from src.roll2film.monotone_curve_matrix import fit_monotone_curve_positive_matrix
from src.roll2film.triangular_logit_transport import (
    PARAMETER_COUNT,
    TriangularLogitTransport,
    fit_triangular_logit_transport,
    select_safe_transport,
)


SCHEMA = "neuro-film.u5-r2bo3-flickr-weak-pair-operator-development.v1"


class FlickrWeakPairDevelopmentError(ValueError):
    """Raised on contract, split, pixel or fit drift."""


def _srgb_to_linear(value: np.ndarray) -> np.ndarray:
    encoded = np.asarray(value, dtype=np.float64)
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        np.power((encoded + 0.055) / 1.055, 2.4),
    )


def _load_rgb(path: Path, digest: str) -> np.ndarray:
    if sha256_file(path) != digest:
        raise FlickrWeakPairDevelopmentError(f"pixel hash drift: {path.as_posix()}")
    with Image.open(path) as image:
        return np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.uint8)


def validate_inputs(root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if config.get("schema") != SCHEMA or config.get("status") != "contract_frozen_before_formal_execution":
        raise FlickrWeakPairDevelopmentError("invalid BO3 contract")
    loaded = {}
    for key, value in config["parents"].items():
        path = root / str(value["path"])
        if sha256_file(path) != value["sha256"]:
            raise FlickrWeakPairDevelopmentError(f"parent hash drift: {key}")
        loaded[key] = json.loads(path.read_text(encoding="utf-8"))
    report = loaded["registration_report"]
    if (
        not report.get("automatic_pass")
        or report.get("stable_evidence_id")
        != config["parents"]["registration_report"]["required_stable_evidence_id"]
    ):
        raise FlickrWeakPairDevelopmentError("registration parent did not pass exactly")
    return report, loaded["download_manifest"]


def _registered_samples(
    digital_u8: np.ndarray,
    film_u8: np.ndarray,
    homography: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, int | float]]:
    """Build deterministic registered linear-light samples for one scene."""

    digital_linear = _srgb_to_linear(digital_u8.astype(np.float64) / 255.0)
    film_linear = _srgb_to_linear(film_u8.astype(np.float64) / 255.0)
    height, width = film_u8.shape[:2]
    warped = cv2.warpPerspective(
        digital_linear,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
    )
    valid = cv2.warpPerspective(
        np.ones(digital_u8.shape[:2], dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    ).astype(bool)
    erosion = int(contract["mask_erosion_pixels"])
    if erosion:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((erosion, erosion), np.uint8)).astype(bool)
    low = int(contract["encoded_boundary_code_minimum"]) / 255.0
    high = int(contract["encoded_boundary_code_maximum"]) / 255.0
    warped_encoded = np.where(
        warped <= 0.0031308,
        warped * 12.92,
        1.055 * np.power(np.maximum(warped, 0.0), 1.0 / 2.4) - 0.055,
    )
    film_encoded = film_u8.astype(np.float64) / 255.0
    valid &= np.all((warped_encoded > low) & (warped_encoded < high), axis=2)
    valid &= np.all((film_encoded > low) & (film_encoded < high), axis=2)
    source_luma = cv2.cvtColor(np.rint(np.clip(warped_encoded, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    target_luma = cv2.cvtColor(film_u8, cv2.COLOR_RGB2GRAY)
    source_grad = cv2.magnitude(cv2.Sobel(source_luma, cv2.CV_32F, 1, 0), cv2.Sobel(source_luma, cv2.CV_32F, 0, 1))
    target_grad = cv2.magnitude(cv2.Sobel(target_luma, cv2.CV_32F, 1, 0), cv2.Sobel(target_luma, cv2.CV_32F, 0, 1))
    combined = np.maximum(source_grad, target_grad)
    cutoff = float(np.quantile(combined[valid], float(contract["maximum_combined_encoded_luma_gradient_quantile"])))
    valid &= combined <= cutoff
    indices = np.flatnonzero(valid)
    maximum = int(contract["maximum_samples_per_scene"])
    if len(indices) < 256:
        raise FlickrWeakPairDevelopmentError("too few registered low-gradient samples")
    if len(indices) > maximum:
        positions = np.linspace(0, len(indices) - 1, maximum, dtype=np.int64)
        indices = indices[positions]
    source = warped.reshape(-1, 3)[indices]
    target = film_linear.reshape(-1, 3)[indices]
    return source, target, {
        "eligible_pixels": int(np.count_nonzero(valid)),
        "sampled_pixels": len(indices),
        "gradient_cutoff": cutoff,
    }


def _fit_basic(source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]) -> tuple[TriangularLogitTransport, bool]:
    bounds = config["operators"]["family_basic_logit"]
    parameters = np.zeros(PARAMETER_COUNT, dtype=np.float64)
    success: list[bool] = []
    for channel, (scale_index, shift_index) in enumerate(((0, 1), (2, 4), (6, 10))):
        x = source[:, channel]
        y = target[:, channel]
        interior = (x > 0) & (x < 1) & (y > 0) & (y < 1)
        logit_x = np.log(x[interior]) - np.log1p(-x[interior])
        logit_y = np.log(y[interior]) - np.log1p(-y[interior])
        shrinkage = float(bounds["identity_shrinkage"])

        def residual(value: np.ndarray) -> np.ndarray:
            prediction = np.exp(value[0]) * logit_x + value[1]
            return np.concatenate((prediction - logit_y, np.sqrt(shrinkage) * value))

        result = least_squares(
            residual,
            np.zeros(2),
            bounds=(
                [float(bounds["log_scale_bounds"][0]), float(bounds["shift_bounds"][0])],
                [float(bounds["log_scale_bounds"][1]), float(bounds["shift_bounds"][1])],
            ),
            max_nfev=int(bounds["maximum_fit_evaluations"]),
        )
        parameters[scale_index] = result.x[0]
        parameters[shift_index] = result.x[1]
        success.append(bool(result.success))
    return TriangularLogitTransport(parameters), all(success)


def _fit_triangle(source: np.ndarray, target: np.ndarray, config: Mapping[str, Any]) -> tuple[TriangularLogitTransport, bool, dict[str, Any]]:
    values = config["operators"]["family_triangular_logit"]
    parameters, converged = fit_triangular_logit_transport(
        source,
        target,
        lower_bounds=np.asarray(values["lower_bounds"], dtype=np.float64),
        upper_bounds=np.asarray(values["upper_bounds"], dtype=np.float64),
        sample_stride=1,
        identity_shrinkage=float(values["identity_shrinkage"]),
        maximum_evaluations=int(values["maximum_fit_evaluations"]),
    )
    operator, diagnostics = select_safe_transport(
        parameters=parameters,
        grid_size=int(values["safe_dose_grid_size"]),
        finite_difference=float(values["safe_dose_finite_difference"]),
        minimum_jacobian_determinant=float(values["minimum_jacobian_determinant"]),
        maximum_jacobian_condition=float(values["maximum_jacobian_condition"]),
        bisection_iterations=int(values["safe_dose_bisection_iterations"]),
    )
    return operator, converged, diagnostics


def _rmse(output: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(output - target))))


def _summaries(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    errors = np.asarray([row["errors"][method] for row in rows], dtype=np.float64)
    families = sorted({row["family_id"] for row in rows})
    return {
        "mean_rmse": float(np.mean(errors)),
        "median_rmse": float(np.median(errors)),
        "p95_rmse": float(np.quantile(errors, 0.95)),
        "worst_rmse": float(np.max(errors)),
        "per_family_median_rmse": {
            family: float(np.median([row["errors"][method] for row in rows if row["family_id"] == family]))
            for family in families
        },
    }


def evaluate(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    registration, manifest = validate_inputs(root, config)
    manifest_rows = {str(row["local_path"]): row for row in manifest["rows"]}
    accepted = [row for row in registration["pairs"] if row["diagnostics"]["registration_gate_passed"]]
    development = [row for row in accepted if int(row["scene_id"]) % 4 != 0]
    confirmation = [row for row in accepted if int(row["scene_id"]) % 4 == 0]
    samples: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    sample_facts: dict[str, Any] = {}
    data_root = root / str(config["data_root"])
    for row in development:
        digital_meta = manifest_rows[str(row["digital_local_path"])]
        film_meta = manifest_rows[str(row["film_local_path"])]
        digital = _load_rgb(data_root / str(row["digital_local_path"]), str(digital_meta["sha256"]))
        film = _load_rgb(data_root / str(row["film_local_path"]), str(film_meta["sha256"]))
        source, target, facts = _registered_samples(
            digital,
            film,
            np.asarray(row["diagnostics"]["homography_digital_to_film"], dtype=np.float64),
            config["paired_samples"],
        )
        samples[str(row["pair_id"])] = (source, target)
        sample_facts[str(row["pair_id"])] = facts
    results: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    for fold in range(3):
        fold_test = [row for row in development if int(row["scene_id"]) % 3 == fold]
        pooled_train = [row for row in development if int(row["scene_id"]) % 3 != fold]
        pooled_source = np.concatenate([samples[str(row["pair_id"])][0] for row in pooled_train])
        pooled_target = np.concatenate([samples[str(row["pair_id"])][1] for row in pooled_train])
        pooled_triangle, pooled_ok, pooled_diag = _fit_triangle(pooled_source, pooled_target, config)
        fits.append({"fold": fold, "family_id": "pooled", "method": "pooled_triangular_logit", "converged": pooled_ok, "operator": {"parameters": pooled_triangle.parameters.tolist(), "dose": pooled_triangle.dose}, "diagnostics": pooled_diag})
        for family in sorted({row["family_id"] for row in development}):
            train = [row for row in development if row["family_id"] == family and int(row["scene_id"]) % 3 != fold]
            test = [row for row in fold_test if row["family_id"] == family]
            source = np.concatenate([samples[str(row["pair_id"])][0] for row in train])
            target = np.concatenate([samples[str(row["pair_id"])][1] for row in train])
            basic, basic_ok = _fit_basic(source, target, config)
            curve_cfg = config["operators"]["family_curve_positive_matrix"]
            curve_fit = fit_monotone_curve_positive_matrix(
                source,
                target,
                curve_identity_mixture=float(curve_cfg["curve_identity_mixture"]),
                matrix_identity_mixture=float(curve_cfg["matrix_identity_mixture"]),
                free_logit_bounds=tuple(float(v) for v in curve_cfg["free_logit_bounds"]),
                restart_count=int(curve_cfg["restart_count"]),
                maximum_function_evaluations=int(curve_cfg["maximum_fit_evaluations"]),
                loss=str(curve_cfg["loss"]),
                loss_scale=float(curve_cfg["loss_scale"]),
                seed=int(curve_cfg["seed"]) + fold,
            )
            triangle, triangle_ok, triangle_diag = _fit_triangle(source, target, config)
            fits.extend(
                [
                    {"fold": fold, "family_id": family, "method": "family_basic_logit", "converged": basic_ok, "operator": {"parameters": basic.parameters.tolist(), "dose": basic.dose}},
                    {"fold": fold, "family_id": family, "method": "family_curve_positive_matrix", "converged": curve_fit.converged, "operator": curve_fit.operator.to_dict()},
                    {"fold": fold, "family_id": family, "method": "family_triangular_logit", "converged": triangle_ok, "operator": {"parameters": triangle.parameters.tolist(), "dose": triangle.dose}, "diagnostics": triangle_diag},
                ]
            )
            for row in test:
                pair_id = str(row["pair_id"])
                test_source, test_target = samples[pair_id]
                outputs = {
                    "identity": test_source,
                    "family_basic_logit": basic.apply(test_source),
                    "family_curve_positive_matrix": curve_fit.operator.apply(test_source),
                    "family_triangular_logit": triangle.apply(test_source),
                    "pooled_triangular_logit": pooled_triangle.apply(test_source),
                }
                epsilon = float(config["evaluation"]["new_boundary_epsilon"])
                interior = np.all((test_source > epsilon) & (test_source < 1.0 - epsilon), axis=1)
                results.append(
                    {
                        "pair_id": pair_id,
                        "family_id": family,
                        "scene_id": row["scene_id"],
                        "fold": fold,
                        "samples": sample_facts[pair_id],
                        "errors": {method: _rmse(output, test_target) for method, output in outputs.items()},
                        "new_boundary_fraction": {
                            method: float(np.mean(np.any((output[interior] <= epsilon) | (output[interior] >= 1.0 - epsilon), axis=1)))
                            for method, output in outputs.items()
                        },
                    }
                )
    methods = ["identity", "family_basic_logit", "family_curve_positive_matrix", "family_triangular_logit", "pooled_triangular_logit"]
    summaries = {method: _summaries(results, method) for method in methods}
    baseline = np.asarray([row["errors"]["family_basic_logit"] for row in results])
    identity = np.asarray([row["errors"]["identity"] for row in results])
    candidates = {}
    gates = config["evaluation"]
    for method in ("family_curve_positive_matrix", "family_triangular_logit"):
        error = np.asarray([row["errors"][method] for row in results])
        improvements = (baseline - error) / baseline
        family_improvements = {
            family: float(np.median([improvements[index] for index, row in enumerate(results) if row["family_id"] == family]))
            for family in sorted({row["family_id"] for row in results})
        }
        checks = {
            "mean_improvement_over_identity": (float(np.mean(identity)) - float(np.mean(error))) / float(np.mean(identity)) >= float(gates["minimum_candidate_mean_improvement_over_identity"]),
            "median_improvement_over_basic": float(np.median(improvements)) >= float(gates["minimum_candidate_median_improvement_over_family_basic"]),
            "scene_win_fraction_over_basic": float(np.mean(error < baseline)) >= float(gates["minimum_candidate_scene_win_fraction_over_family_basic"]),
            "each_family_median_improvement": all(value >= float(gates["minimum_each_family_median_improvement_over_family_basic"]) for value in family_improvements.values()),
            "p95_tail": float(np.quantile(error, 0.95)) / float(np.quantile(baseline, 0.95)) <= float(gates["maximum_candidate_p95_error_ratio_to_family_basic"]),
            "worst_tail": float(np.max(error)) / float(np.max(baseline)) <= float(gates["maximum_candidate_worst_error_ratio_to_family_basic"]),
            "new_boundary": max(row["new_boundary_fraction"][method] for row in results) <= float(gates["maximum_new_boundary_fraction"]),
            "all_fits_converged": all(fit["converged"] for fit in fits if fit["method"] == method),
        }
        candidates[method] = {
            "checks": checks,
            "eligible": all(checks.values()),
            "median_improvement_over_basic": float(np.median(improvements)),
            "scene_win_fraction_over_basic": float(np.mean(error < baseline)),
            "per_family_median_improvement_over_basic": family_improvements,
            "p95_error_ratio_to_basic": float(np.quantile(error, 0.95)) / float(np.quantile(baseline, 0.95)),
            "worst_error_ratio_to_basic": float(np.max(error)) / float(np.max(baseline)),
            "maximum_new_boundary_fraction": max(row["new_boundary_fraction"][method] for row in results),
        }
    eligible = [method for method, value in candidates.items() if value["eligible"]]
    parameter_counts = {method: int(config["operators"][method]["parameter_count"]) for method in eligible}
    selected = min(eligible, key=lambda method: (parameter_counts[method], summaries[method]["mean_rmse"])) if eligible else None
    stable = {
        "schema": "neuro-film.u5-r2bo3-flickr-weak-pair-operator-development-report.v1",
        "node": config["node"],
        "split": {
            "development_pair_ids": sorted(str(row["pair_id"]) for row in development),
            "confirmation_pair_ids_held_unread": sorted(str(row["pair_id"]) for row in confirmation),
            "development_pairs": len(development),
            "confirmation_pairs": len(confirmation),
        },
        "summaries": summaries,
        "candidate_decisions": candidates,
        "selected_candidate": selected,
        "automatic_pass": selected is not None,
        "branch": config["branches"]["candidate_pass" if selected is not None else "all_candidates_fail"],
        "fold_results": results,
        "fit_records": fits,
        "confirmation_pixels_loaded": False,
        "training_allowed": False,
        "operator_fitting_allowed": "development pixels only",
        "claim_ceiling": config["claim_ceiling"],
    }
    return {**stable, "stable_evidence_id": hashlib.sha256(canonical_bytes(stable)).hexdigest()}


__all__ = ["FlickrWeakPairDevelopmentError", "SCHEMA", "evaluate", "validate_inputs"]
