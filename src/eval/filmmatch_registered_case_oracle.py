"""Registered evaluator Oracle for the FilmMatch explicit case bank."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.eval.filmmatch_condition_oracle import _fit
from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file


SCHEMA = "neuro_film.u5_r2be0_filmmatch_registered_case_oracle.v1"


class FilmMatchRegisteredOracleError(ValueError):
    """Raised when the frozen registered-Oracle contract is violated."""


def validate_contract(root: Path, config: Mapping[str, Any]) -> None:
    if config.get("schema") != SCHEMA:
        raise FilmMatchRegisteredOracleError("unsupported BE0 schema")
    if config.get("status") != "contract_frozen_before_formal_execution":
        raise FilmMatchRegisteredOracleError("BE0 contract is not frozen")
    for parent in config["parents"].values():
        path = root / str(parent["path"])
        if sha256_file(path) != str(parent["sha256"]):
            raise FilmMatchRegisteredOracleError(
                f"parent identity drift: {path.as_posix()}"
            )
        required = parent.get("required_decision")
        if required is not None:
            decision = json.loads(path.read_text(encoding="utf-8"))
            observed = decision.get("decision", {}).get(
                "status"
            ) if isinstance(decision.get("decision"), dict) else decision.get(
                "decision"
            )
            if observed != required:
                raise FilmMatchRegisteredOracleError(
                    f"parent decision drift: {path.as_posix()}"
                )


def _registration_gray(image: np.ndarray) -> np.ndarray:
    encoded = np.asarray(image)
    if encoded.dtype != np.uint16 or encoded.ndim != 3 or encoded.shape[2] != 3:
        raise FilmMatchRegisteredOracleError(
            "registration expects uint16 RGB"
        )
    return cv2.cvtColor(
        np.rint(encoded.astype(np.float64) / 257.0).astype(np.uint8),
        cv2.COLOR_RGB2GRAY,
    )


def register_validation_pair(
    source_u16: np.ndarray,
    target_u16: np.ndarray,
    contract: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Estimate the frozen source-to-target homography and evaluation mask."""

    if source_u16.shape != target_u16.shape:
        raise FilmMatchRegisteredOracleError("validation shape mismatch")
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    source_gray = _registration_gray(source_u16)
    target_gray = _registration_gray(target_u16)
    sift = cv2.SIFT_create(
        nfeatures=int(contract["maximum_features"])
    )
    source_points, source_desc = sift.detectAndCompute(source_gray, None)
    target_points, target_desc = sift.detectAndCompute(target_gray, None)
    if source_desc is None or target_desc is None:
        raise FilmMatchRegisteredOracleError("SIFT produced no descriptors")
    matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(
        source_desc, target_desc, k=2
    )
    ratio = float(contract["ratio_threshold"])
    good = [left for left, right in matches if left.distance < ratio * right.distance]
    if len(good) < int(contract["minimum_good_matches"]):
        raise FilmMatchRegisteredOracleError(
            "insufficient registered matches"
        )
    source_xy = np.float32(
        [source_points[match.queryIdx].pt for match in good]
    )
    target_xy = np.float32(
        [target_points[match.trainIdx].pt for match in good]
    )
    homography, inlier_mask = cv2.findHomography(
        source_xy,
        target_xy,
        cv2.RANSAC,
        float(contract["ransac_reprojection_threshold_pixels"]),
    )
    if homography is None or inlier_mask is None:
        raise FilmMatchRegisteredOracleError("homography fit failed")
    inliers = inlier_mask.ravel().astype(bool)
    projected = cv2.perspectiveTransform(
        source_xy[:, None, :], homography
    )[:, 0, :]
    reprojection = np.linalg.norm(projected - target_xy, axis=1)[inliers]
    inlier_fraction = float(np.mean(inliers))
    median_error = float(np.median(reprojection))
    p95_error = float(np.quantile(reprojection, 0.95))

    height, width = source_gray.shape
    valid = cv2.warpPerspective(
        np.ones((height, width), dtype=np.uint8),
        homography,
        (width, height),
        flags=cv2.INTER_NEAREST,
    )
    target = target_u16.astype(np.float64) / 65535.0
    valid = valid.astype(bool)
    valid &= np.all(
        (target > float(contract["target_code_minimum"]))
        & (target < float(contract["target_code_maximum"])),
        axis=2,
    )
    erosion = int(contract["mask_erosion_pixels"])
    if erosion > 0:
        valid = cv2.erode(
            valid.astype(np.uint8),
            np.ones((erosion, erosion), dtype=np.uint8),
        ).astype(bool)
    luma = (
        target[..., 0] * 0.2126
        + target[..., 1] * 0.7152
        + target[..., 2] * 0.0722
    )
    gradient = np.zeros_like(luma)
    gradient[:, 1:] += np.abs(np.diff(luma, axis=1))
    gradient[1:, :] += np.abs(np.diff(luma, axis=0))
    cutoff = float(
        np.quantile(
            gradient[valid],
            float(contract["maximum_target_luma_gradient_quantile"]),
        )
    )
    valid &= gradient <= cutoff
    evaluation_fraction = float(np.mean(valid))
    diagnostics = {
        "source_keypoints": len(source_points),
        "target_keypoints": len(target_points),
        "good_matches": len(good),
        "inliers": int(np.count_nonzero(inliers)),
        "inlier_fraction": inlier_fraction,
        "median_inlier_reprojection_error_pixels": median_error,
        "p95_inlier_reprojection_error_pixels": p95_error,
        "evaluation_pixels": int(np.count_nonzero(valid)),
        "evaluation_pixel_fraction": evaluation_fraction,
        "target_luma_gradient_cutoff": cutoff,
        "homography_source_to_target": homography.tolist(),
    }
    passed = bool(
        inlier_fraction >= float(contract["minimum_inlier_fraction"])
        and median_error
        <= float(
            contract["maximum_median_inlier_reprojection_error_pixels"]
        )
        and p95_error
        <= float(contract["maximum_p95_inlier_reprojection_error_pixels"])
        and evaluation_fraction
        >= float(contract["minimum_evaluation_pixel_fraction"])
    )
    diagnostics["registration_gate_passed"] = passed
    if not passed:
        raise FilmMatchRegisteredOracleError("registration gate failed")
    return homography.astype(np.float64), valid, diagnostics


def _operator_groups(
    records: list[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    exposure = np.asarray([int(row["exposure_ev"]) for row in records])
    groups: dict[str, np.ndarray] = {
        "global": np.ones(exposure.shape, dtype=bool),
        "regime_negative": exposure < 0,
        "regime_zero": exposure == 0,
        "regime_positive": exposure > 0,
    }
    for value in range(-5, 6):
        groups[f"ev_{value:+d}"] = exposure == value
    if any(np.count_nonzero(mask) < 12 for mask in groups.values()):
        raise FilmMatchRegisteredOracleError(
            "one operator group lacks fit support"
        )
    return groups


def _apply_rows(operator: Any, image: np.ndarray, rows: int) -> np.ndarray:
    output = np.empty_like(image, dtype=np.float64)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        output[start:stop] = operator.apply(image[start:stop])
    return output


def _aligned_metrics(
    output: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    homography: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    height, width = target.shape[:2]
    warped = cv2.warpPerspective(
        output,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    difference = warped[mask] - target[mask]
    norm = np.linalg.norm(difference, axis=1)
    source_boundary = np.any((source <= 0.0) | (source >= 1.0), axis=2)
    output_boundary = np.any((output <= 0.0) | (output >= 1.0), axis=2)
    metrics = {
        "rgb_rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "median_rgb_euclidean_error": float(np.median(norm)),
        "p95_rgb_euclidean_error": float(np.quantile(norm, 0.95)),
        "minimum_output": float(np.min(output)),
        "maximum_output": float(np.max(output)),
        "new_source_relative_boundary_fraction": float(
            np.mean(output_boundary & ~source_boundary)
        ),
    }
    return metrics, warped[mask].astype(np.float32)


def evaluate_registered_case_oracle(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    software_commit: str,
    config_sha256: str,
) -> dict[str, Any]:
    validate_contract(root, config)
    data = config["data"]
    data_root = root / str(data["root"])
    source_path = data_root / str(data["source_relative_path"])
    target_path = data_root / str(data["target_relative_path"])
    if (
        sha256_file(source_path) != data["source_sha256"]
        or sha256_file(target_path) != data["target_sha256"]
    ):
        raise FilmMatchRegisteredOracleError("validation file identity drift")
    source_u16 = np.asarray(tifffile.imread(source_path))
    target_u16 = np.asarray(tifffile.imread(target_path))
    homography, evaluation_mask, registration = register_validation_pair(
        source_u16, target_u16, config["registration"]
    )
    source_image = source_u16.astype(np.float64) / 65535.0
    target_image = target_u16.astype(np.float64) / 65535.0
    fit_source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    fit_target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = list(datasets["reflective_records"])
    groups = _operator_groups(records)
    fit_config = {"candidate": {"fit": config["operator_bank"]["fit"]}}
    rows = int(config["evaluation"]["row_chunk"])

    results: dict[str, Any] = {}
    aligned: dict[str, np.ndarray] = {}
    identity_metrics, identity_aligned = _aligned_metrics(
        source_image,
        source_image,
        target_image,
        homography,
        evaluation_mask,
    )
    results["identity"] = {"metrics": identity_metrics}
    aligned["identity"] = identity_aligned
    for name, group in groups.items():
        operator = _fit(
            fit_source[group], fit_target[group], fit_config
        )
        output = _apply_rows(operator, source_image, rows)
        metrics, aligned_values = _aligned_metrics(
            output,
            source_image,
            target_image,
            homography,
            evaluation_mask,
        )
        results[name] = {
            "fit_samples": int(np.count_nonzero(group)),
            "operator": operator.to_dict(),
            "metrics": metrics,
        }
        aligned[name] = aligned_values

    bank_names = [name for name in results if name != "identity"]
    pairwise = []
    for index, left in enumerate(bank_names):
        for right in bank_names[index + 1 :]:
            pairwise.append(
                {
                    "left": left,
                    "right": right,
                    "rgb_rmse": float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    aligned[left].astype(np.float64)
                                    - aligned[right].astype(np.float64)
                                )
                            )
                        )
                    ),
                }
            )
    global_error = results["global"]["metrics"]["rgb_rmse"]
    identity_error = results["identity"]["metrics"]["rgb_rmse"]
    regime_names = [
        "regime_negative",
        "regime_zero",
        "regime_positive",
    ]
    exact_names = [f"ev_{value:+d}" for value in range(-5, 6)]
    ranked_regimes = sorted(
        regime_names, key=lambda name: results[name]["metrics"]["rgb_rmse"]
    )
    ranked_exact = sorted(
        exact_names, key=lambda name: results[name]["metrics"]["rgb_rmse"]
    )
    best_cases = sorted(
        regime_names + exact_names,
        key=lambda name: results[name]["metrics"]["rgb_rmse"],
    )
    aggregates = {
        "identity_rgb_rmse": identity_error,
        "global_rgb_rmse": global_error,
        "global_improvement_over_identity": float(
            1.0 - global_error / identity_error
        ),
        "best_regime": ranked_regimes[0],
        "best_regime_rgb_rmse": results[ranked_regimes[0]]["metrics"][
            "rgb_rmse"
        ],
        "regime_oracle_improvement_over_global": float(
            1.0
            - results[ranked_regimes[0]]["metrics"]["rgb_rmse"]
            / global_error
        ),
        "best_exact_ev": ranked_exact[0],
        "best_exact_ev_rgb_rmse": results[ranked_exact[0]]["metrics"][
            "rgb_rmse"
        ],
        "exact_ev_oracle_improvement_over_global": float(
            1.0
            - results[ranked_exact[0]]["metrics"]["rgb_rmse"]
            / global_error
        ),
        "best_case": best_cases[0],
        "second_best_case": best_cases[1],
        "best_case_improvement_over_second_best_case": float(
            1.0
            - results[best_cases[0]]["metrics"]["rgb_rmse"]
            / results[best_cases[1]]["metrics"]["rgb_rmse"]
        ),
        "minimum_operator_bank_output_pairwise_rmse": float(
            min(row["rgb_rmse"] for row in pairwise)
        ),
        "maximum_new_source_relative_boundary_fraction": float(
            max(
                results[name]["metrics"][
                    "new_source_relative_boundary_fraction"
                ]
                for name in bank_names
            )
        ),
    }
    gates = config["automatic_gates"]
    gate_results = {
        "global_beats_identity": aggregates[
            "global_improvement_over_identity"
        ]
        >= float(gates["minimum_global_improvement_over_identity"]),
        "regime_oracle_value": aggregates[
            "regime_oracle_improvement_over_global"
        ]
        >= float(gates["minimum_regime_oracle_improvement_over_global"]),
        "exact_ev_oracle_value": aggregates[
            "exact_ev_oracle_improvement_over_global"
        ]
        >= float(gates["minimum_exact_ev_oracle_improvement_over_global"]),
        "best_case_separation": aggregates[
            "best_case_improvement_over_second_best_case"
        ]
        >= float(
            gates["minimum_best_case_improvement_over_second_best_case"]
        ),
        "operator_bank_diversity": aggregates[
            "minimum_operator_bank_output_pairwise_rmse"
        ]
        >= float(gates["minimum_operator_bank_output_pairwise_rmse"]),
        "source_relative_boundary": aggregates[
            "maximum_new_source_relative_boundary_fraction"
        ]
        <= float(gates["maximum_new_source_relative_boundary_fraction"]),
    }
    report = {
        "schema": SCHEMA,
        "experiment_id": config["experiment_id"],
        "software_commit": software_commit,
        "config_sha256": config_sha256,
        "registration": registration,
        "operators": results,
        "operator_pairwise_output_rmse": pairwise,
        "aggregates": aggregates,
        "automatic_gates": gate_results,
        "automatic_passed": all(gate_results.values()),
        "retrieval_contract_opened": all(gate_results.values()),
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


def report_file_sha256(report: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "FilmMatchRegisteredOracleError",
    "SCHEMA",
    "_operator_groups",
    "evaluate_registered_case_oracle",
    "register_validation_pair",
    "report_file_sha256",
    "validate_contract",
]
