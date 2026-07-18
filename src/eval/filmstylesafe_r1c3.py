"""A0-only conditioned style-control research for FilmStyleSafe R1C3.

Fits an explicit low-capacity colour relation independently for each source /
candidate pair. This is evaluation-time canonicalization, not corpus training or
a promoted safety gate.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.color import rgb2lab
from skimage.measure import label

from .filmstylesafe_r1c import FilmStyleSafeR1CError, proxy_severe_label

CONTRACT_ID = "kmcfm.filmstylesafe-r1c3.v1"


def load_r1c3_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilmStyleSafeR1CError(f"cannot read R1C3 contract {path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("contract_id") != CONTRACT_ID:
        raise FilmStyleSafeR1CError("unexpected FilmStyleSafe R1C3 contract id")
    authorization = contract.get("authorization", {})
    for forbidden in (
        "model_training",
        "hidden_split_population",
        "external_human_recruitment",
        "paid_resource",
        "production_gate",
    ):
        if authorization.get(forbidden) is not False:
            raise FilmStyleSafeR1CError(f"R1C3 must keep {forbidden} false")
    if contract.get("decision", {}).get("style_controls_may_be_excluded") is not False:
        raise FilmStyleSafeR1CError("R1C3 must retain all style controls")
    return contract


def _basis(sample: np.ndarray, degree: int) -> np.ndarray:
    if sample.ndim != 2 or sample.shape[1] != 3:
        raise FilmStyleSafeR1CError("conditioned basis requires Nx3 scaled Lab")
    if degree not in {1, 2}:
        raise FilmStyleSafeR1CError("conditioned basis degree must be 1 or 2")
    luma, a_axis, b_axis = sample.T
    columns = [np.ones_like(luma), luma, a_axis, b_axis]
    if degree == 2:
        columns.extend(
            [
                luma * luma,
                a_axis * a_axis,
                b_axis * b_axis,
                luma * a_axis,
                luma * b_axis,
                a_axis * b_axis,
            ]
        )
    return np.stack(columns, axis=1)


def _sample_indices(height: int, width: int, maximum: int) -> np.ndarray:
    if maximum <= 0:
        raise FilmStyleSafeR1CError("maximum_sample_pixels must be positive")
    count = height * width
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    stride = int(np.ceil(np.sqrt(count / float(maximum))))
    ys = np.arange(0, height, stride, dtype=np.int64)
    xs = np.arange(0, width, stride, dtype=np.int64)
    grid = (ys[:, None] * width + xs[None, :]).reshape(-1)
    if grid.size > maximum:
        positions = np.linspace(0, grid.size - 1, maximum, dtype=np.int64)
        grid = grid[positions]
    return grid


def _robust_coefficients(
    source_scaled: np.ndarray,
    candidate_ab_scaled: np.ndarray,
    *,
    degree: int,
    maximum_sample_pixels: int,
    ridge: float,
    huber_delta: float,
    irls_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width, channels = source_scaled.shape
    if channels != 3 or candidate_ab_scaled.shape != (height, width, 2):
        raise FilmStyleSafeR1CError("conditioned fit shape mismatch")
    if ridge < 0 or huber_delta <= 0 or irls_iterations <= 0:
        raise FilmStyleSafeR1CError("invalid conditioned fit parameters")
    indices = _sample_indices(height, width, int(maximum_sample_pixels))
    return _robust_coefficients_from_indices(
        source_scaled,
        candidate_ab_scaled,
        indices=indices,
        degree=degree,
        ridge=ridge,
        huber_delta=huber_delta,
        irls_iterations=irls_iterations,
    )


def _robust_coefficients_from_indices(
    source_scaled: np.ndarray,
    candidate_ab_scaled: np.ndarray,
    *,
    indices: np.ndarray,
    degree: int,
    ridge: float,
    huber_delta: float,
    irls_iterations: int,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width, channels = source_scaled.shape
    if channels != 3 or candidate_ab_scaled.shape != (height, width, 2):
        raise FilmStyleSafeR1CError("conditioned fit shape mismatch")
    if indices.ndim != 1 or indices.size == 0:
        raise FilmStyleSafeR1CError("conditioned fit requires sampled pixels")
    source_sample = source_scaled.reshape(-1, 3)[indices].astype(np.float64)
    target_sample = candidate_ab_scaled.reshape(-1, 2)[indices].astype(np.float64)
    design = _basis(source_sample, degree)
    weights = np.ones(design.shape[0], dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.zeros((design.shape[1], 2), dtype=np.float64)
    for _ in range(int(irls_iterations)):
        weighted = design * weights[:, None]
        lhs = design.T @ weighted + penalty
        rhs = design.T @ (target_sample * weights[:, None])
        try:
            coefficients = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError as exc:
            raise FilmStyleSafeR1CError("conditioned fit is singular") from exc
        residual = np.linalg.norm(target_sample - design @ coefficients, axis=1)
        weights = np.minimum(1.0, float(huber_delta) / np.maximum(residual, 1e-12))
    if not np.isfinite(coefficients).all():
        raise FilmStyleSafeR1CError("conditioned fit produced non-finite coefficients")
    final_residual = np.linalg.norm(target_sample - design @ coefficients, axis=1)
    diagnostics = {
        "fit_sample_pixels": float(indices.size),
        "fit_basis_columns": float(design.shape[1]),
        "fit_residual_median_scaled": float(np.median(final_residual)),
        "fit_residual_p95_scaled": float(np.quantile(final_residual, 0.95)),
        "fit_downweighted_fraction": float(np.mean(weights < 1.0)),
    }
    return coefficients, diagnostics


def _predict_ab(source_scaled: np.ndarray, coefficients: np.ndarray, degree: int) -> np.ndarray:
    luma = source_scaled[..., 0]
    a_axis = source_scaled[..., 1]
    b_axis = source_scaled[..., 2]
    prediction = np.broadcast_to(coefficients[0], (*source_scaled.shape[:2], 2)).copy()
    values = [luma, a_axis, b_axis]
    if degree == 2:
        values.extend(
            [
                luma * luma,
                a_axis * a_axis,
                b_axis * b_axis,
                luma * a_axis,
                luma * b_axis,
                a_axis * b_axis,
            ]
        )
    for index, value in enumerate(values, start=1):
        prediction += value[..., None] * coefficients[index]
    return prediction


def conditioned_scis(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    degree: int,
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, float]:
    """Return an explicit SCIS score after robust global colour canonicalization."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise FilmStyleSafeR1CError("reference/candidate shape mismatch")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise FilmStyleSafeR1CError("conditioned SCIS requires finite RGB")
    lab_ref = rgb2lab(np.clip(reference, 0.0, 1.0)).astype(np.float64)
    lab_candidate = rgb2lab(np.clip(candidate, 0.0, 1.0)).astype(np.float64)
    source_scaled = np.stack(
        [lab_ref[..., 0] / 100.0, lab_ref[..., 1] / 128.0, lab_ref[..., 2] / 128.0],
        axis=2,
    )
    candidate_ab_scaled = lab_candidate[..., 1:3] / 128.0
    coefficients, diagnostics = _robust_coefficients(
        source_scaled,
        candidate_ab_scaled,
        degree=degree,
        maximum_sample_pixels=int(fit["maximum_sample_pixels"]),
        ridge=float(fit["ridge"]),
        huber_delta=float(fit["huber_delta"]),
        irls_iterations=int(fit["irls_iterations"]),
    )
    predicted_ab = _predict_ab(source_scaled, coefficients, degree) * 128.0
    residual = np.linalg.norm(lab_candidate[..., 1:3] - predicted_ab, axis=2)
    luma = lab_ref[..., 0] / 100.0
    gy, gx = np.gradient(luma)
    smooth = np.hypot(gx, gy) <= float(score["smooth_grad_threshold"])
    masked = np.where(smooth, residual, 0.0)
    smooth_values = masked[smooth]
    quantile = float(np.quantile(smooth_values, float(score["quantile"]))) if smooth_values.size else 0.0
    total = float(reference.shape[0] * reference.shape[1])
    best_max_fraction = 0.0
    best_component_count = 0.0
    for threshold in score["residual_thresholds"]:
        labeled = label((masked >= float(threshold)) & smooth, connectivity=2)
        counts = np.bincount(labeled.reshape(-1))[1:]
        areas = [
            int(area) for area in counts if area >= int(score["min_component_pixels"])
        ]
        maximum = (float(max(areas)) / total) if areas else 0.0
        if maximum > best_max_fraction:
            best_max_fraction = maximum
            best_component_count = float(len(areas))
    sparse_density = float(np.mean((masked >= float(score["sparse_threshold"])) & smooth))
    value = float(
        best_max_fraction * float(score["island_weight"])
        + quantile
        + sparse_density * float(score["sparse_weight"])
    )
    prefix = "conditioned_affine_v1" if degree == 1 else "conditioned_quadratic_v1"
    return {
        f"{prefix}_score": value,
        f"{prefix}_max_component_area_frac": best_max_fraction,
        f"{prefix}_q99_residual": quantile,
        f"{prefix}_sparse_density": sparse_density,
        f"{prefix}_component_count": best_component_count,
        **{f"{prefix}_{key}": val for key, val in diagnostics.items()},
    }


def conditioned_scis_spatial_crossfit(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> dict[str, float]:
    """Score affine residuals predicted from non-neighbouring spatial tiles."""
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[2] != 3:
        raise FilmStyleSafeR1CError("reference/candidate shape mismatch")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise FilmStyleSafeR1CError("conditioned SCIS requires finite RGB")
    rows = int(grid["rows"])
    columns = int(grid["columns"])
    radius = int(grid["chebyshev_exclusion_radius_tiles"])
    minimum_fit = int(grid["minimum_fit_pixels"])
    if rows != 4 or columns != 4 or radius != 1 or minimum_fit != 4096:
        raise FilmStyleSafeR1CError("unexpected frozen R1C3D grid")
    lab_ref = rgb2lab(np.clip(reference, 0.0, 1.0)).astype(np.float64)
    lab_candidate = rgb2lab(np.clip(candidate, 0.0, 1.0)).astype(np.float64)
    source_scaled = np.stack(
        [lab_ref[..., 0] / 100.0, lab_ref[..., 1] / 128.0, lab_ref[..., 2] / 128.0],
        axis=2,
    )
    candidate_ab_scaled = lab_candidate[..., 1:3] / 128.0
    height, width = reference.shape[:2]
    y_bounds = np.linspace(0, height, rows + 1, dtype=np.int64)
    x_bounds = np.linspace(0, width, columns + 1, dtype=np.int64)
    prediction_scaled = np.empty((height, width, 2), dtype=np.float64)
    sample_counts: list[int] = []
    for row in range(rows):
        for column in range(columns):
            excluded_row_start = max(0, row - radius)
            excluded_row_stop = min(rows, row + radius + 1)
            excluded_column_start = max(0, column - radius)
            excluded_column_stop = min(columns, column + radius + 1)
            fit_mask = np.ones((height, width), dtype=bool)
            fit_mask[
                y_bounds[excluded_row_start] : y_bounds[excluded_row_stop],
                x_bounds[excluded_column_start] : x_bounds[excluded_column_stop],
            ] = False
            indices = np.flatnonzero(fit_mask.reshape(-1))
            if indices.size < minimum_fit:
                raise FilmStyleSafeR1CError("R1C3D tile has insufficient fit pixels")
            maximum = int(fit["maximum_sample_pixels"])
            if indices.size > maximum:
                positions = np.linspace(0, indices.size - 1, maximum, dtype=np.int64)
                indices = indices[positions]
            coefficients, _ = _robust_coefficients_from_indices(
                source_scaled,
                candidate_ab_scaled,
                indices=indices,
                degree=1,
                ridge=float(fit["ridge"]),
                huber_delta=float(fit["huber_delta"]),
                irls_iterations=int(fit["irls_iterations"]),
            )
            y0, y1 = int(y_bounds[row]), int(y_bounds[row + 1])
            x0, x1 = int(x_bounds[column]), int(x_bounds[column + 1])
            prediction_scaled[y0:y1, x0:x1] = _predict_ab(
                source_scaled[y0:y1, x0:x1], coefficients, degree=1
            )
            sample_counts.append(int(indices.size))
    residual = np.linalg.norm(
        lab_candidate[..., 1:3] - prediction_scaled * 128.0, axis=2
    )
    luma = lab_ref[..., 0] / 100.0
    gy, gx = np.gradient(luma)
    smooth = np.hypot(gx, gy) <= float(score["smooth_grad_threshold"])
    masked = np.where(smooth, residual, 0.0)
    smooth_values = masked[smooth]
    quantile = float(np.quantile(smooth_values, float(score["quantile"]))) if smooth_values.size else 0.0
    total = float(height * width)
    best_max_fraction = 0.0
    best_component_count = 0.0
    for threshold in score["residual_thresholds"]:
        labeled = label((masked >= float(threshold)) & smooth, connectivity=2)
        counts = np.bincount(labeled.reshape(-1))[1:]
        areas = [
            int(area) for area in counts if area >= int(score["min_component_pixels"])
        ]
        maximum = (float(max(areas)) / total) if areas else 0.0
        if maximum > best_max_fraction:
            best_max_fraction = maximum
            best_component_count = float(len(areas))
    sparse_density = float(np.mean((masked >= float(score["sparse_threshold"])) & smooth))
    value = float(
        best_max_fraction * float(score["island_weight"])
        + quantile
        + sparse_density * float(score["sparse_weight"])
    )
    prefix = "conditioned_affine_spatial_crossfit_v1"
    return {
        f"{prefix}_score": value,
        f"{prefix}_max_component_area_frac": best_max_fraction,
        f"{prefix}_q99_residual": quantile,
        f"{prefix}_sparse_density": sparse_density,
        f"{prefix}_component_count": best_component_count,
        f"{prefix}_tiles": float(rows * columns),
        f"{prefix}_minimum_fit_sample_pixels": float(min(sample_counts)),
        f"{prefix}_maximum_fit_sample_pixels": float(max(sample_counts)),
    }


def evaluate_conditioned_pair(
    reference_path: Path,
    candidate_path: Path,
    *,
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, float]:
    with Image.open(reference_path) as ref_image, Image.open(candidate_path) as candidate_image:
        ref_image.load()
        candidate_image.load()
        reference = np.asarray(ref_image.convert("RGB"), dtype=np.float32) / 255.0
        candidate_rgb = candidate_image.convert("RGB")
        if candidate_rgb.size != ref_image.size:
            candidate_rgb = candidate_rgb.resize(ref_image.size, Image.Resampling.BICUBIC)
        candidate = np.asarray(candidate_rgb, dtype=np.float32) / 255.0
    result: dict[str, float] = {}
    for degree in (1, 2):
        result.update(conditioned_scis(reference, candidate, degree=degree, fit=fit, score=score))
    return result


def analyze_conditioned_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row.get("proxy_severe") is True]
    negatives = [row for row in rows if row.get("proxy_severe") is False]
    if len(positives) != 3 or len(negatives) != 5:
        raise FilmStyleSafeR1CError("R1C3 requires exactly three positives and five negatives")
    candidates = []
    for candidate_id, degree in (("conditioned_affine_v1", 1), ("conditioned_quadratic_v1", 2)):
        key = f"{candidate_id}_score"
        threshold = max(float(row[key]) for row in negatives)
        detected = [float(row[key]) > threshold for row in positives]
        hardneg_external = [
            row
            for row in negatives
            if row.get("hard_negative") or row.get("role") == "external_style_control"
        ]
        candidates.append(
            {
                "candidate_id": candidate_id,
                "degree": degree,
                "zero_fpr_threshold": threshold,
                "positives_detected": int(sum(detected)),
                "positives_total": 3,
                "negatives_total": 5,
                "sensitivity_at_zero_fpr": float(sum(detected) / 3.0),
                "hardneg_external_below_all_positives": all(
                    float(negative[key]) < float(positive[key])
                    for negative in hardneg_external
                    for positive in positives
                ),
            }
        )
    full_passes = [row for row in candidates if row["positives_detected"] == 3]
    if full_passes:
        selected = min(full_passes, key=lambda row: int(row["degree"]))
        decision = "pass"
    elif max(int(row["positives_detected"]) for row in candidates) == 2:
        selected = None
        decision = "weak_pass"
    else:
        selected = None
        decision = "fail"
    return {
        "rows": list(rows),
        "candidates": candidates,
        "decision": decision,
        "selected_candidate": selected,
        "claim_ceiling": (
            "A0 proxy-label conditioned-style-control research only; not human "
            "confirmation, population risk evidence, a validated detector or product gate"
        ),
    }


def run_conditioned_pilot(
    inventory: Mapping[str, Any],
    *,
    root: Path,
    parent_sources: Mapping[str, str],
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for member in inventory.get("members", []):
        if member.get("binding_status") != "bound" or member.get("role") == "source_scene":
            continue
        parent_id = str(member["parent_scene_id"])
        reference = root / parent_sources[parent_id]
        candidate = root / str(member["artifact_path"])
        if not reference.is_file() or not candidate.is_file():
            raise FilmStyleSafeR1CError(f"missing artifact for {member['member_id']}")
        metrics = evaluate_conditioned_pair(reference, candidate, fit=fit, score=score)
        role = str(member["role"])
        rows.append(
            {
                "member_id": member["member_id"],
                "role": role,
                "parent_scene_id": parent_id,
                "proxy_severe": proxy_severe_label(role),
                "hard_negative": role == "legitimate_local_hard_negative",
                **metrics,
            }
        )
    return analyze_conditioned_rows(rows)


def run_spatial_crossfit_pilot(
    inventory: Mapping[str, Any],
    *,
    root: Path,
    parent_sources: Mapping[str, str],
    fit: Mapping[str, Any],
    score: Mapping[str, Any],
    grid: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for member in inventory.get("members", []):
        if member.get("binding_status") != "bound" or member.get("role") == "source_scene":
            continue
        parent_id = str(member["parent_scene_id"])
        reference_path = root / parent_sources[parent_id]
        candidate_path = root / str(member["artifact_path"])
        with Image.open(reference_path) as ref_image, Image.open(candidate_path) as cand_image:
            reference = np.asarray(ref_image.convert("RGB"), dtype=np.float32) / 255.0
            candidate_image = cand_image.convert("RGB")
            if candidate_image.size != ref_image.size:
                candidate_image = candidate_image.resize(ref_image.size, Image.Resampling.BICUBIC)
            candidate = np.asarray(candidate_image, dtype=np.float32) / 255.0
        metrics = conditioned_scis_spatial_crossfit(
            reference, candidate, fit=fit, score=score, grid=grid
        )
        role = str(member["role"])
        rows.append(
            {
                "member_id": member["member_id"],
                "role": role,
                "proxy_severe": proxy_severe_label(role),
                "hard_negative": role == "legitimate_local_hard_negative",
                **metrics,
            }
        )
    key = "conditioned_affine_spatial_crossfit_v1_score"
    positives = [row for row in rows if row["proxy_severe"] is True]
    negatives = [row for row in rows if row["proxy_severe"] is False]
    if len(positives) != 3 or len(negatives) != 5:
        raise FilmStyleSafeR1CError("R1C3D requires exactly three positives and five negatives")
    threshold = max(float(row[key]) for row in negatives)
    detected = [float(row[key]) > threshold for row in positives]
    hardneg_external = [
        row
        for row in negatives
        if row["hard_negative"] or row["role"] == "external_style_control"
    ]
    hardneg_control = all(
        float(negative[key]) < float(positive[key])
        for negative in hardneg_external
        for positive in positives
    )
    passed = sum(detected) == 3 and hardneg_control
    return {
        "rows": rows,
        "zero_fpr_threshold": threshold,
        "positives_detected": int(sum(detected)),
        "positives_total": 3,
        "negatives_total": 5,
        "hardneg_external_below_all_positives": hardneg_control,
        "decision": "pass" if passed else "fail_close_conditioned_scis",
        "claim_ceiling": (
            "A0 spatial self-absorption mechanism diagnostic only; not a validated "
            "severe detector or product gate"
        ),
    }
