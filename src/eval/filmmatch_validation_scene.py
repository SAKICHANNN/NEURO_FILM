"""Fit-forbidden FilmMatch validation-scene renderer and diagnostics."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from src.eval.filmmatch_condition_oracle import _fit
from src.eval.filmmatch_luma_proxy_router import fit_conservative_threshold
from src.eval.filmmatch_paired_source import canonical_sha256, sha256_file


def _apply_rows(operator: Any, image: np.ndarray, rows: int) -> np.ndarray:
    if rows <= 0:
        raise ValueError("row chunk must be positive")
    output = np.empty_like(image, dtype=np.float64)
    for start in range(0, image.shape[0], rows):
        stop = min(start + rows, image.shape[0])
        output[start:stop] = operator.apply(image[start:stop])
    return output


def _save_rgb16_png(path: Path, image: np.ndarray) -> str:
    encoded = np.rint(np.clip(image, 0.0, 1.0) * 65535.0).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.png")
    if not cv2.imwrite(
        str(temporary), cv2.cvtColor(encoded, cv2.COLOR_RGB2BGR)
    ):
        raise RuntimeError("validation PNG encode failed")
    temporary.replace(path)
    return sha256_file(path)


def evaluate_validation_scene(
    datasets: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = np.asarray(datasets["reflective_source"], dtype=np.float64)
    target = np.asarray(datasets["reflective_target"], dtype=np.float64)
    records = datasets["reflective_records"]
    exposure = np.asarray([int(row["exposure_ev"]) for row in records])
    group_ids = np.asarray(
        [
            f"{row['illuminant']}|ev={int(row['exposure_ev']):+d}"
            for row in records
        ]
    )
    groups = sorted(set(group_ids.tolist()))
    features = np.asarray(
        [float(np.median(source[group_ids == group])) for group in groups]
    )
    labels = np.asarray(
        [
            int(group.rsplit("=", 1)[1]) >= int(
                config["router"]["minimum_exposure_ev"]
            )
            for group in groups
        ]
    )
    threshold, threshold_fit = fit_conservative_threshold(features, labels)
    global_operator = _fit(source, target, config)
    positive = exposure > 0
    positive_operator = _fit(source[positive], target[positive], config)

    validation = config["validation"]
    source_path = root / validation["source_relative_path"]
    target_path = root / validation["target_relative_path"]
    if (
        sha256_file(source_path) != validation["source_sha256"]
        or sha256_file(target_path) != validation["target_sha256"]
    ):
        raise ValueError("validation image identity drift")
    source_image = (
        np.asarray(tifffile.memmap(source_path), dtype=np.float64) / 65535.0
    )
    target_image = (
        np.asarray(tifffile.memmap(target_path), dtype=np.float64) / 65535.0
    )
    if source_image.shape != target_image.shape:
        raise ValueError("validation image shape drift")
    validation_feature = float(np.median(source_image))
    route = "positive_exposure_expert" if validation_feature >= threshold else "global"
    global_output = _apply_rows(
        global_operator,
        source_image,
        int(config["execution"]["row_chunk"]),
    )
    positive_output = _apply_rows(
        positive_operator,
        source_image,
        int(config["execution"]["row_chunk"]),
    )
    routed_output = (
        positive_output if route == "positive_exposure_expert" else global_output
    )
    output_sha = {
        "source": _save_rgb16_png(output_dir / "source.png", source_image),
        "global": _save_rgb16_png(output_dir / "global.png", global_output),
        "positive_exposure_expert": _save_rgb16_png(
            output_dir / "positive_exposure_expert.png", positive_output
        ),
        "routed": _save_rgb16_png(output_dir / "routed.png", routed_output),
        "target_reference": _save_rgb16_png(
            output_dir / "target_reference.png", target_image
        ),
    }
    automatic = {
        "finite": bool(np.all(np.isfinite(routed_output))),
        "minimum_output": float(np.min(routed_output)),
        "maximum_output": float(np.max(routed_output)),
        "new_boundary_fraction": float(
            np.mean(
                np.any(
                    ((routed_output <= 0.0) & (source_image > 0.0))
                    | ((routed_output >= 1.0) & (source_image < 1.0)),
                    axis=2,
                )
            )
        ),
        "style_rgb_rmse_from_source": float(
            np.sqrt(np.mean(np.square(routed_output - source_image)))
        ),
    }
    gates = config["automatic_gate"]
    automatic_pass = bool(
        automatic["finite"]
        and automatic["minimum_output"] >= 0.0
        and automatic["maximum_output"] <= 1.0
        and automatic["new_boundary_fraction"]
        <= float(gates["maximum_new_boundary_fraction"])
        and automatic["style_rgb_rmse_from_source"]
        >= float(gates["minimum_style_rgb_rmse_from_source"])
    )
    report = {
        "schema": "neuro_film.u5_r2aw9_filmmatch_validation_scene.v1",
        "experiment_id": config["experiment_id"],
        "router": {
            "threshold": threshold,
            "threshold_fit": threshold_fit,
            "validation_feature": validation_feature,
            "route": route,
        },
        "operators": {
            "global": global_operator.to_dict(),
            "positive_exposure_expert": positive_operator.to_dict(),
        },
        "outputs": output_sha,
        "automatic_diagnostics": automatic,
        "automatic_gate_passed": automatic_pass,
        "visual_review_required": automatic_pass,
        "visual_review_opened": automatic_pass,
        "promotion_opened": False,
        "target_pixel_metric_forbidden_reason": (
            "validation captures have materially different framing, crop, "
            "film border and scene geometry"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_validation_scene"]
