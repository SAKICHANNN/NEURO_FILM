"""Fail-closed RF2.S0 archive-display matrix transplant evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps
from skimage.color import rgb2lab

from src.roll2film.baselines import fit_joint_basic_adjustment

from .gold_transform_consistency import (
    BoundedAffineOperator,
    GoldTransformConsistencyError,
    PairedFrameSamples,
    fit_full_affine,
)


class GoldMatrixTransplantError(ValueError):
    """Raised when the frozen RF2.S0 contract or evidence is invalid."""


@dataclass(frozen=True)
class DigitalSample:
    sample_id: str
    split: str
    pixels: np.ndarray
    source_path: str

    def __post_init__(self) -> None:
        pixels = np.asarray(self.pixels, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 3 or len(pixels) < 64:
            raise GoldMatrixTransplantError("digital samples require at least 64 RGB pixels")
        if not np.all(np.isfinite(pixels)) or self.split not in {"gold", "stress"}:
            raise GoldMatrixTransplantError("digital sample values or split are invalid")
        object.__setattr__(self, "pixels", pixels)


def sample_rgb_image(path: Path, maximum_pixels: int) -> np.ndarray:
    """Read unprofiled sRGB8-compatible pixels on the frozen uniform grid."""
    if maximum_pixels < 64:
        raise GoldMatrixTransplantError("pixel budget must be at least 64")
    with Image.open(path) as image:
        values = np.asarray(ImageOps.exif_transpose(image).convert("RGB"), dtype=np.float64) / 255.0
    height, width = values.shape[:2]
    rows = min(height, max(1, int(np.floor(np.sqrt(maximum_pixels * height / width)))))
    columns = min(width, max(1, maximum_pixels // rows))
    yy, xx = np.meshgrid(
        np.linspace(0, height - 1, rows, dtype=np.int64),
        np.linspace(0, width - 1, columns, dtype=np.int64),
        indexing="ij",
    )
    return values[yy, xx].reshape(-1, 3)


def colour_distribution_descriptor(pixels: np.ndarray) -> np.ndarray:
    values = np.asarray(pixels, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 64:
        raise GoldMatrixTransplantError("descriptor pixels have invalid shape")
    luma = values @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
    return np.concatenate((values.mean(axis=0), values.std(axis=0), np.quantile(luma, [0.05, 0.5, 0.95])))


def build_archive_ood_model(
    frames: Sequence[PairedFrameSamples], *, threshold_quantile: float, mad_floor: float = 1e-4
) -> dict[str, Any]:
    """Build an archive-only threshold from cross-roll nearest neighbours."""
    if not 0.5 <= threshold_quantile < 1.0:
        raise GoldMatrixTransplantError("OOD threshold quantile is invalid")
    descriptors = np.stack([colour_distribution_descriptor(frame.source) for frame in frames])
    centre = np.median(descriptors, axis=0)
    scale = np.maximum(np.median(np.abs(descriptors - centre), axis=0) * 1.4826, mad_floor)
    standardized = (descriptors - centre) / scale
    distances: list[float] = []
    for index, frame in enumerate(frames):
        allowed = np.asarray([other.roll_id != frame.roll_id for other in frames])
        if not np.any(allowed):
            raise GoldMatrixTransplantError("OOD model needs multiple physical rolls")
        distances.append(float(np.min(np.linalg.norm(standardized[allowed] - standardized[index], axis=1))))
    threshold = float(np.quantile(distances, threshold_quantile))
    return {
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "archive_descriptors": descriptors.tolist(),
        "leave_one_roll_nearest_distances": distances,
        "threshold_quantile": threshold_quantile,
        "threshold": threshold,
    }


def ood_distance(pixels: np.ndarray, model: Mapping[str, Any]) -> float:
    descriptor = colour_distribution_descriptor(pixels)
    centre = np.asarray(model["centre"], dtype=np.float64)
    scale = np.asarray(model["scale"], dtype=np.float64)
    archive = np.asarray(model["archive_descriptors"], dtype=np.float64)
    standardized = (descriptor - centre) / scale
    archive_standardized = (archive - centre) / scale
    return float(np.min(np.linalg.norm(archive_standardized - standardized, axis=1)))


def composite_affine(
    operator: BoundedAffineOperator, strength: float
) -> BoundedAffineOperator | None:
    if not 0.0 < strength <= 1.0:
        raise GoldMatrixTransplantError("transplant strength must be in (0, 1]")
    matrix = np.eye(3) + strength * (operator.matrix - np.eye(3))
    bias = strength * operator.bias
    if float(np.linalg.det(matrix)) <= 1e-10:
        return None
    return BoundedAffineOperator(matrix, bias, operator.working_space)


def style_and_basic_residual(source: np.ndarray, output: np.ndarray) -> tuple[float, float]:
    source = np.asarray(source, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    basic = fit_joint_basic_adjustment(source, output).apply(source)
    source_lab = _lab(source)
    output_lab = _lab(output)
    basic_lab = _lab(np.clip(basic, 0.0, 1.0))
    return (
        float(np.median(np.linalg.norm(output_lab - source_lab, axis=1))),
        float(np.median(np.linalg.norm(output_lab - basic_lab, axis=1))),
    )


def evaluate_transplant(
    *,
    archive_frames: Sequence[PairedFrameSamples],
    digital_samples: Sequence[DigitalSample],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit archive-only recipe, evaluate fixed strengths, and branch fail-closed."""
    fit_config = config["all_roll_fit"]
    bounds = {
        "matrix_coefficient": fit_config["matrix_coefficient"],
        "channel_bias": fit_config["channel_bias"],
        "ridge": fit_config["ridge"],
    }
    operator = fit_full_affine(archive_frames, bounds, str(config["working_space"]))
    ood_config = config["ood"]
    ood_model = build_archive_ood_model(
        archive_frames, threshold_quantile=float(ood_config["threshold_quantile"])
    )
    digital_ood: list[dict[str, Any]] = []
    for sample in digital_samples:
        distance = ood_distance(sample.pixels, ood_model)
        digital_ood.append({
            "sample_id": sample.sample_id,
            "split": sample.split,
            "distance": distance,
            "eligible": distance <= float(ood_model["threshold"]),
        })
    gold_eligible = sum(row["eligible"] for row in digital_ood if row["split"] == "gold")
    stress_rows = [row for row in digital_ood if row["split"] == "stress"]
    stress_fraction = float(np.mean([row["eligible"] for row in stress_rows]))

    floors = config["anchor_derived_floors_frozen_before_candidate_render"]
    gates = config["gates"]
    strength_results: list[dict[str, Any]] = []
    selected_strength: float | None = None
    selected_checks: dict[str, bool] | None = None
    for strength_value in config["transplant_strengths_strongest_first"]:
        strength = float(strength_value)
        composite = composite_affine(operator, strength)
        per_image: list[dict[str, Any]] = []
        if composite is not None:
            for sample in digital_samples:
                raw = composite.apply(sample.pixels)
                finite = bool(np.all(np.isfinite(raw)))
                rendered = np.clip(raw, 0.0, 1.0)
                style, residual = style_and_basic_residual(sample.pixels, rendered)
                per_image.append({
                    "sample_id": sample.sample_id,
                    "split": sample.split,
                    "finite": finite,
                    "raw_clip_fraction": float(np.mean((raw < 0.0) | (raw > 1.0))),
                    "median_style_delta_e76": style,
                    "median_residual_delta_e76_after_matched_basic": residual,
                })
        gold = [row for row in per_image if row["split"] == "gold"]
        style_median = float(np.median([row["median_style_delta_e76"] for row in gold])) if gold else 0.0
        residual_median = float(np.median([row["median_residual_delta_e76_after_matched_basic"] for row in gold])) if gold else 0.0
        maximum_clip = max((row["raw_clip_fraction"] for row in gold), default=1.0)
        checks = {
            "gold_ood_coverage": gold_eligible >= int(gates["minimum_gold_ood_eligible_images"]),
            "stress_ood_coverage": stress_fraction >= float(gates["minimum_stress_ood_eligible_fraction"]),
            "style_floor": style_median >= float(floors["minimum_candidate_gold_median_style_delta_e76"]),
            "matched_basic_residual_floor": residual_median >= float(floors["minimum_candidate_gold_median_residual_delta_e76_after_matched_basic"]),
            "gold_raw_clip": maximum_clip <= float(gates["maximum_gold_per_image_raw_clip_fraction"]),
            "finite_outputs": bool(per_image) and all(row["finite"] for row in per_image),
            "positive_composite_determinant": composite is not None,
        }
        strength_results.append({
            "strength": strength,
            "operator": composite.to_dict() if composite is not None else None,
            "gold_median_style_delta_e76": style_median,
            "gold_median_residual_delta_e76_after_matched_basic": residual_median,
            "gold_maximum_per_image_raw_clip_fraction": maximum_clip,
            "checks": checks,
            "automatic_pass": all(checks.values()),
            "per_image": per_image,
        })
        if selected_strength is None and all(checks.values()):
            selected_strength = strength
            selected_checks = checks

    coverage_pass = (
        gold_eligible >= int(gates["minimum_gold_ood_eligible_images"])
        and stress_fraction >= float(gates["minimum_stress_ood_eligible_fraction"])
    )
    if not coverage_pass:
        decision = "ood_coverage_fail"
    elif selected_strength is None:
        failures = {key for row in strength_results for key, passed in row["checks"].items() if not passed}
        if failures & {"style_floor", "matched_basic_residual_floor"}:
            decision = "style_or_residual_floor_fail"
        else:
            decision = "clip_or_operator_safety_fail"
    else:
        decision = "automatic_pass_visual_adjudication_required"
    return {
        "archive_all_roll_operator": operator.to_dict(),
        "archive_ood_model": ood_model,
        "digital_ood": digital_ood,
        "gold_ood_eligible_images": gold_eligible,
        "stress_ood_eligible_fraction": stress_fraction,
        "strength_results": strength_results,
        "selected_strength": selected_strength,
        "selected_checks": selected_checks,
        "automatic_passed": selected_strength is not None,
        "decision": decision,
    }


def _lab(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return rgb2lab(clipped.reshape(-1, 1, 3)).reshape(-1, 3)
