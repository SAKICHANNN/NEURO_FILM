"""Dense numerical and visual stress for a fixed FilmMatch operator."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.eval.filmmatch_paired_source import canonical_sha256
from src.eval.filmmatch_validation_scene import _save_rgb16_png
from src.roll2film.factorized_monotone_bernstein import (
    FactorizedMonotoneBernsteinOperator,
)


def _chunked_apply(
    operator: FactorizedMonotoneBernsteinOperator,
    values: np.ndarray,
    chunk: int,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    output = np.empty_like(flat)
    for start in range(0, len(flat), chunk):
        stop = min(start + chunk, len(flat))
        output[start:stop] = operator.apply(flat[start:stop])
    return output.reshape(np.asarray(values).shape)


def _chunked_jacobian(
    operator: FactorizedMonotoneBernsteinOperator,
    values: np.ndarray,
    chunk: int,
) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float64).reshape(-1, 3)
    output = np.empty(len(flat), dtype=np.float64)
    for start in range(0, len(flat), chunk):
        stop = min(start + chunk, len(flat))
        output[start:stop] = operator.jacobian_determinants(flat[start:stop])
    return output


def _hue_luma_stress(size: int) -> np.ndarray:
    hue = np.linspace(0.0, 360.0, size, endpoint=False, dtype=np.float32)
    value = np.linspace(0.01, 0.99, size, dtype=np.float32)
    hsv = np.empty((size, size, 3), dtype=np.float32)
    hsv[:, :, 0] = hue[None, :]
    hsv[:, :, 1] = 0.9
    hsv[:, :, 2] = value[:, None]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB).astype(np.float64)


def evaluate_dense_safety_stress(
    parent_report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    operator = FactorizedMonotoneBernsteinOperator.from_dict(
        parent_report["final_fit"]["operator"]
    )
    execution = config["execution"]
    chunk = int(execution["sample_chunk"])
    axis = np.linspace(0.0, 1.0, int(execution["cube_grid_size"]))
    cube = np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    cube_output = _chunked_apply(operator, cube, chunk)
    determinants = _chunked_jacobian(operator, cube, chunk)

    t = np.linspace(0.0, 1.0, 65536)
    anchor_values = tuple(float(value) for value in execution["axis_anchors"])
    ramp_rows = []
    maximum_code_jump = 0
    maximum_euclidean_jump = 0.0
    for channel in range(3):
        other = [index for index in range(3) if index != channel]
        for first in anchor_values:
            for second in anchor_values:
                ramp = np.empty((len(t), 3), dtype=np.float64)
                ramp[:, channel] = t
                ramp[:, other[0]] = first
                ramp[:, other[1]] = second
                rendered = _chunked_apply(operator, ramp, chunk)
                differences = np.diff(rendered, axis=0)
                code = np.rint(rendered * 65535.0).astype(np.int64)
                code_jump = int(np.max(np.abs(np.diff(code, axis=0))))
                euclidean_jump = float(
                    np.max(np.linalg.norm(differences, axis=1))
                )
                maximum_code_jump = max(maximum_code_jump, code_jump)
                maximum_euclidean_jump = max(
                    maximum_euclidean_jump, euclidean_jump
                )
                ramp_rows.append(
                    {
                        "varied_channel": channel,
                        "fixed_values": [first, second],
                        "maximum_code_jump": code_jump,
                        "maximum_euclidean_jump": euclidean_jump,
                    }
                )
    neutral = np.repeat(t[:, None], 3, axis=1)
    neutral_output = _chunked_apply(operator, neutral, chunk)
    neutral_luma = neutral_output @ np.asarray(
        config["metrics"]["luma_coefficients"], dtype=np.float64
    )
    minimum_neutral_luma_step = float(np.min(np.diff(neutral_luma)))

    gradient = _hue_luma_stress(int(execution["gradient_size"]))
    gradient_output = _chunked_apply(operator, gradient, chunk)
    outputs = {
        "hue_luma_source": _save_rgb16_png(
            output_dir / "hue_luma_source.png", gradient
        ),
        "hue_luma_candidate": _save_rgb16_png(
            output_dir / "hue_luma_candidate.png", gradient_output
        ),
    }
    diagnostics = {
        "cube_sample_count": int(len(cube)),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "p001_jacobian_determinant": float(
            np.quantile(determinants, 0.001)
        ),
        "nonpositive_jacobian_fraction": float(np.mean(determinants <= 0.0)),
        "minimum_output": float(np.min(cube_output)),
        "maximum_output": float(np.max(cube_output)),
        "new_interior_boundary_fraction": float(
            np.mean(
                np.any(
                    ((cube_output <= 0.0) & (cube > 0.0))
                    | ((cube_output >= 1.0) & (cube < 1.0)),
                    axis=1,
                )
            )
        ),
        "ramp_count": len(ramp_rows),
        "maximum_ramp_code_jump": maximum_code_jump,
        "maximum_ramp_euclidean_jump": maximum_euclidean_jump,
        "minimum_neutral_luma_step": minimum_neutral_luma_step,
        "gradient_minimum": float(np.min(gradient_output)),
        "gradient_maximum": float(np.max(gradient_output)),
    }
    gates = config["automatic_gate"]
    passed = bool(
        diagnostics["minimum_jacobian_determinant"]
        >= float(gates["minimum_jacobian_determinant"])
        and diagnostics["nonpositive_jacobian_fraction"] == 0.0
        and diagnostics["minimum_output"] >= 0.0
        and diagnostics["maximum_output"] <= 1.0
        and diagnostics["new_interior_boundary_fraction"]
        <= float(gates["maximum_new_interior_boundary_fraction"])
        and diagnostics["maximum_ramp_code_jump"]
        <= int(gates["maximum_ramp_code_jump"])
        and diagnostics["minimum_neutral_luma_step"]
        >= float(gates["minimum_neutral_luma_step"])
    )
    report = {
        "schema": "neuro_film.u5_r2ax7_filmmatch_dense_safety_stress.v1",
        "experiment_id": config["experiment_id"],
        "operator": operator.to_dict(),
        "automatic_diagnostics": diagnostics,
        "ramp_diagnostics": ramp_rows,
        "outputs": outputs,
        "automatic_gate_passed": passed,
        "visual_review_required": passed,
        "visual_review_opened": passed,
        "promotion_opened": False,
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = canonical_sha256(report)
    return report


__all__ = ["evaluate_dense_safety_stress"]
