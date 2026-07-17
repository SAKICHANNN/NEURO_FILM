"""FilmStyleSafe R1C conventional-metric failure pilot and SCIS v0.

A0-only development tooling. Does not train models, populate hidden splits,
recruit humans, or claim population risk bounds.
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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

CONTRACT_ID = "kmcfm.filmstylesafe-r1c.v1"


class FilmStyleSafeR1CError(ValueError):
    """Raised when an R1C contract or pilot fails closed."""


def load_r1c_contract(path: Path) -> dict[str, Any]:
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FilmStyleSafeR1CError(f"cannot read R1C contract {path}: {exc}") from exc
    if not isinstance(contract, dict) or contract.get("contract_id") != CONTRACT_ID:
        raise FilmStyleSafeR1CError("unexpected FilmStyleSafe R1C contract id")
    if contract.get("authorization", {}).get("hidden_split_population") is not False:
        raise FilmStyleSafeR1CError("R1C must keep hidden_split_population false")
    if contract.get("authorization", {}).get("model_training") is not False:
        raise FilmStyleSafeR1CError("R1C must keep model_training false")
    if contract.get("authorization", {}).get("external_human_recruitment") is not False:
        raise FilmStyleSafeR1CError("R1C must keep external_human_recruitment false")
    return contract


def _as_rgb_float(image: Image.Image) -> np.ndarray:
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def conventional_pair_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Global fidelity metrics that ignore intentional local film edits poorly."""
    if reference.shape != candidate.shape:
        raise FilmStyleSafeR1CError("reference/candidate shape mismatch")
    ref_u8 = np.clip(np.round(reference * 255.0), 0, 255).astype(np.uint8)
    cand_u8 = np.clip(np.round(candidate * 255.0), 0, 255).astype(np.uint8)
    psnr = float(peak_signal_noise_ratio(ref_u8, cand_u8, data_range=255))
    ssim = float(
        structural_similarity(ref_u8, cand_u8, channel_axis=2, data_range=255)
    )
    mean_abs = float(np.mean(np.abs(candidate - reference)))
    lab_ref = rgb2lab(reference)
    lab_cand = rgb2lab(candidate)
    delta_e76 = float(np.mean(np.linalg.norm(lab_cand - lab_ref, axis=2)))
    return {
        "psnr": psnr,
        "ssim": ssim,
        "mean_abs_rgb": mean_abs,
        "mean_delta_e76": delta_e76,
    }


def scis_v0(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    smooth_grad_threshold: float = 0.035,
    residual_threshold: float = 12.0,
    min_component_pixels: int = 16,
) -> dict[str, float]:
    """Semantic Chroma-Island Score v0 (explicit, non-learned).

    1. Convert to Lab.
    2. Remove a global ab mean shift (smooth global colour relation).
    3. Residual chroma magnitude restricted to input-smooth regions.
    4. Threshold and connected-component area statistics.
    """
    if reference.shape != candidate.shape:
        raise FilmStyleSafeR1CError("reference/candidate shape mismatch")
    lab_ref = rgb2lab(np.clip(reference, 0.0, 1.0))
    lab_cand = rgb2lab(np.clip(candidate, 0.0, 1.0))
    global_shift = lab_cand[..., 1:3].reshape(-1, 2).mean(axis=0) - lab_ref[..., 1:3].reshape(
        -1, 2
    ).mean(axis=0)
    residual_ab = lab_cand[..., 1:3] - (lab_ref[..., 1:3] + global_shift.reshape(1, 1, 2))
    residual = np.linalg.norm(residual_ab, axis=2)
    luma = lab_ref[..., 0] / 100.0
    gy, gx = np.gradient(luma)
    grad = np.hypot(gx, gy)
    smooth = grad <= float(smooth_grad_threshold)
    masked = residual.copy()
    masked[~smooth] = 0.0
    q95 = float(np.quantile(masked[smooth], 0.95)) if np.any(smooth) else 0.0
    q99 = float(np.quantile(masked[smooth], 0.99)) if np.any(smooth) else 0.0
    binary = (masked >= float(residual_threshold)) & smooth
    labeled = label(binary, connectivity=2)
    areas = []
    for idx in range(1, int(labeled.max()) + 1):
        area = int(np.sum(labeled == idx))
        if area >= int(min_component_pixels):
            areas.append(area)
    total = float(reference.shape[0] * reference.shape[1])
    max_area = float(max(areas)) if areas else 0.0
    topk = float(sum(sorted(areas, reverse=True)[:3]))
    max_frac = max_area / total
    topk_frac = topk / total
    # Scalar score: emphasize large smooth-region chroma islands.
    score = float(max_frac * 1000.0 + q99)
    return {
        "scis_v0_score": score,
        "scis_max_component_area_frac": max_frac,
        "scis_topk3_area_frac": topk_frac,
        "scis_q95_residual": q95,
        "scis_q99_residual": q99,
        "scis_smooth_pixel_fraction": float(np.mean(smooth)),
        "scis_component_count": float(len(areas)),
    }


def scis_v0_1(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    smooth_grad_threshold: float = 0.035,
    residual_thresholds: Sequence[float] | None = None,
    min_component_pixels: int = 4,
    low_frequency_sigma: float = 8.0,
    sparse_threshold: float = 6.0,
) -> dict[str, float]:
    """SCIS v0.1: style-robust high-frequency residual + multi-threshold islands.

    Removes a blurred low-frequency ab residual so strong global looks (53/55/56)
    contribute less, then scores both large islands and sparse speckles.
    """
    from src.filmfx.fast_blur import gaussian_filter_safe

    if reference.shape != candidate.shape:
        raise FilmStyleSafeR1CError("reference/candidate shape mismatch")
    thresholds = list(residual_thresholds or (4.0, 8.0, 12.0))
    lab_ref = rgb2lab(np.clip(reference, 0.0, 1.0))
    lab_cand = rgb2lab(np.clip(candidate, 0.0, 1.0))
    global_shift = lab_cand[..., 1:3].reshape(-1, 2).mean(axis=0) - lab_ref[..., 1:3].reshape(
        -1, 2
    ).mean(axis=0)
    residual_ab = lab_cand[..., 1:3] - (lab_ref[..., 1:3] + global_shift.reshape(1, 1, 2))
    low = np.stack(
        [
            gaussian_filter_safe(residual_ab[..., 0], sigma=float(low_frequency_sigma)),
            gaussian_filter_safe(residual_ab[..., 1], sigma=float(low_frequency_sigma)),
        ],
        axis=2,
    )
    high = residual_ab - low
    residual = np.linalg.norm(high, axis=2)
    luma = lab_ref[..., 0] / 100.0
    gy, gx = np.gradient(luma)
    smooth = np.hypot(gx, gy) <= float(smooth_grad_threshold)
    masked = residual.copy()
    masked[~smooth] = 0.0
    q99 = float(np.quantile(masked[smooth], 0.99)) if np.any(smooth) else 0.0
    total = float(reference.shape[0] * reference.shape[1])
    best_max_frac = 0.0
    best_components = 0.0
    for thr in thresholds:
        binary = (masked >= float(thr)) & smooth
        labeled = label(binary, connectivity=2)
        areas = []
        for idx in range(1, int(labeled.max()) + 1):
            area = int(np.sum(labeled == idx))
            if area >= int(min_component_pixels):
                areas.append(area)
        max_frac = (float(max(areas)) / total) if areas else 0.0
        if max_frac > best_max_frac:
            best_max_frac = max_frac
            best_components = float(len(areas))
    sparse = float(np.mean((masked >= float(sparse_threshold)) & smooth))
    score = float(best_max_frac * 1000.0 + q99 + sparse * 200.0)
    return {
        "scis_v0_1_score": score,
        "scis_v0_1_max_component_area_frac": best_max_frac,
        "scis_v0_1_q99_hf_residual": q99,
        "scis_v0_1_sparse_density": sparse,
        "scis_v0_1_component_count": best_components,
    }


def proxy_severe_label(role: str) -> bool | None:
    """A0 development proxy labels from suite roles (not human population labels)."""
    if role in {"synthetic_failure", "regression_case"}:
        return True
    if role in {
        "source_scene",
        "strength_control",
        "external_style_control",
        "legitimate_local_hard_negative",
    }:
        return False
    return None


def evaluate_pair_metrics(
    reference_path: Path,
    candidate_path: Path,
    *,
    scis_params: Mapping[str, Any] | None = None,
    max_side: int | None = None,
) -> dict[str, float]:
    with Image.open(reference_path) as ref_im, Image.open(candidate_path) as cand_im:
        ref_im.load()
        cand_im.load()
        ref_im = ref_im.convert("RGB")
        cand_im = cand_im.convert("RGB")
        if max_side is not None and max_side > 0:
            ref_im.thumbnail((int(max_side), int(max_side)), Image.Resampling.BICUBIC)
        if ref_im.size != cand_im.size:
            cand_im = cand_im.resize(ref_im.size, Image.Resampling.BICUBIC)
        reference = _as_rgb_float(ref_im)
        candidate = _as_rgb_float(cand_im)
    metrics = conventional_pair_metrics(reference, candidate)
    params = {
        key: value
        for key, value in dict(scis_params or {}).items()
        if key
        in {
            "smooth_grad_threshold",
            "residual_threshold",
            "min_component_pixels",
        }
    }
    metrics.update(scis_v0(reference, candidate, **params))
    v01_params = {
        key: value
        for key, value in dict(scis_params or {}).items()
        if key
        in {
            "smooth_grad_threshold",
            "min_component_pixels",
            "low_frequency_sigma",
            "sparse_threshold",
            "residual_thresholds",
        }
    }
    # v0.1 uses a lower default min component size than v0.
    if "min_component_pixels" not in v01_params:
        v01_params["min_component_pixels"] = 4
    metrics.update(scis_v0_1(reference, candidate, **v01_params))
    return metrics


def run_a0_metric_pilot(
    inventory: Mapping[str, Any],
    *,
    root: Path,
    parent_sources: Mapping[str, str],
    scis_params: Mapping[str, Any] | None = None,
    max_side: int | None = None,
) -> dict[str, Any]:
    """Score bound A0 members against parent sources using frozen metrics."""
    rows: list[dict[str, Any]] = []
    for member in inventory.get("members", []):
        if member.get("binding_status") != "bound":
            continue
        role = str(member["role"])
        if role == "source_scene":
            continue
        parent = str(member["parent_scene_id"])
        if parent not in parent_sources:
            raise FilmStyleSafeR1CError(f"missing parent source for {parent}")
        ref = root / parent_sources[parent]
        cand = root / str(member["artifact_path"])
        if not ref.is_file() or not cand.is_file():
            raise FilmStyleSafeR1CError(f"missing artifact for {member['member_id']}")
        metrics = evaluate_pair_metrics(
            ref, cand, scis_params=scis_params, max_side=max_side
        )
        label = proxy_severe_label(role)
        rows.append(
            {
                "member_id": member["member_id"],
                "role": role,
                "parent_scene_id": parent,
                "proxy_severe": label,
                "hard_negative": role == "legitimate_local_hard_negative",
                **metrics,
            }
        )
    return analyze_metric_separation(rows)


def analyze_metric_separation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare conventional vs SCIS separation at zero false-positive on non-severe."""
    positives = [row for row in rows if row.get("proxy_severe") is True]
    negatives = [row for row in rows if row.get("proxy_severe") is False]
    if not positives or not negatives:
        raise FilmStyleSafeR1CError("pilot requires both proxy-severe and non-severe members")
    hardneg_like = [
        row
        for row in negatives
        if row.get("hard_negative") or row.get("role") == "external_style_control"
    ]

    def _roc_at_zero_fpr(
        score_key: str,
        neg_rows: Sequence[Mapping[str, Any]],
        *,
        higher_is_more_severe: bool,
        cohort: str,
    ) -> dict[str, Any]:
        neg_scores = [float(row[score_key]) for row in neg_rows]
        pos_scores = [float(row[score_key]) for row in positives]
        if not neg_scores:
            raise FilmStyleSafeR1CError(f"empty negative cohort for {cohort}")
        if higher_is_more_severe:
            threshold = max(neg_scores)
            detected = [score > threshold for score in pos_scores]
            direction = "higher"
        else:
            threshold = min(neg_scores)
            detected = [score < threshold for score in pos_scores]
            direction = "lower"
        sensitivity = float(sum(detected) / len(detected))
        return {
            "metric": score_key,
            "cohort": cohort,
            "direction": direction,
            "zero_fpr_threshold": threshold,
            "sensitivity_at_zero_fpr": sensitivity,
            "positives_detected": int(sum(detected)),
            "positives_total": len(detected),
            "negatives_total": len(neg_rows),
        }

    conventional_all = [
        _roc_at_zero_fpr("mean_delta_e76", negatives, higher_is_more_severe=True, cohort="all_nonsevere"),
        _roc_at_zero_fpr("mean_abs_rgb", negatives, higher_is_more_severe=True, cohort="all_nonsevere"),
        _roc_at_zero_fpr("psnr", negatives, higher_is_more_severe=False, cohort="all_nonsevere"),
        _roc_at_zero_fpr("ssim", negatives, higher_is_more_severe=False, cohort="all_nonsevere"),
    ]
    conventional_hardneg = [
        _roc_at_zero_fpr(
            "mean_delta_e76", hardneg_like, higher_is_more_severe=True, cohort="hardneg_external"
        ),
        _roc_at_zero_fpr(
            "mean_abs_rgb", hardneg_like, higher_is_more_severe=True, cohort="hardneg_external"
        ),
        _roc_at_zero_fpr("psnr", hardneg_like, higher_is_more_severe=False, cohort="hardneg_external"),
        _roc_at_zero_fpr("ssim", hardneg_like, higher_is_more_severe=False, cohort="hardneg_external"),
    ]
    scis_all = _roc_at_zero_fpr(
        "scis_v0_score", negatives, higher_is_more_severe=True, cohort="all_nonsevere"
    )
    scis_hardneg = _roc_at_zero_fpr(
        "scis_v0_score", hardneg_like, higher_is_more_severe=True, cohort="hardneg_external"
    )
    scis01_all = _roc_at_zero_fpr(
        "scis_v0_1_score", negatives, higher_is_more_severe=True, cohort="all_nonsevere"
    )
    scis01_hardneg = _roc_at_zero_fpr(
        "scis_v0_1_score", hardneg_like, higher_is_more_severe=True, cohort="hardneg_external"
    )
    best_conventional_all = max(conventional_all, key=lambda row: row["sensitivity_at_zero_fpr"])
    best_conventional_hardneg = max(
        conventional_hardneg, key=lambda row: row["sensitivity_at_zero_fpr"]
    )
    hardneg = next((row for row in rows if row.get("hard_negative")), None)
    return {
        "rows": list(rows),
        "n_positives": len(positives),
        "n_negatives": len(negatives),
        "n_hardneg_external": len(hardneg_like),
        "conventional_zero_fpr_all_nonsevere": conventional_all,
        "conventional_zero_fpr_hardneg_external": conventional_hardneg,
        "scis_v0_zero_fpr_all_nonsevere": scis_all,
        "scis_v0_zero_fpr_hardneg_external": scis_hardneg,
        "scis_v0_1_zero_fpr_all_nonsevere": scis01_all,
        "scis_v0_1_zero_fpr_hardneg_external": scis01_hardneg,
        "best_conventional_zero_fpr_all_nonsevere": best_conventional_all,
        "best_conventional_zero_fpr_hardneg_external": best_conventional_hardneg,
        "conventional_metric_gap_at_zero_fpr": best_conventional_all["sensitivity_at_zero_fpr"] < 1.0,
        "conventional_metric_gap_vs_hardneg_external": best_conventional_hardneg[
            "sensitivity_at_zero_fpr"
        ]
        < 1.0,
        "scis_v0_perfect_sensitivity_at_zero_fpr": scis_all["sensitivity_at_zero_fpr"] == 1.0,
        "scis_v0_perfect_vs_hardneg_external": scis_hardneg["sensitivity_at_zero_fpr"] == 1.0,
        "scis_v0_1_perfect_sensitivity_at_zero_fpr": scis01_all["sensitivity_at_zero_fpr"] == 1.0,
        "scis_v0_1_perfect_vs_hardneg_external": scis01_hardneg["sensitivity_at_zero_fpr"] == 1.0,
        "hardneg_scis_below_all_positives": (
            hardneg is not None
            and all(float(hardneg["scis_v0_score"]) < float(row["scis_v0_score"]) for row in positives)
        ),
        "hardneg_scis_v0_1_below_all_positives": (
            hardneg is not None
            and "scis_v0_1_score" in hardneg
            and all(
                float(hardneg["scis_v0_1_score"]) < float(row["scis_v0_1_score"]) for row in positives
            )
        ),
        "claim_ceiling": (
            "A0 development proxy-label pilot only; not human population evidence, "
            "not a frozen safety gate, not A1 confirmation"
        ),
    }
