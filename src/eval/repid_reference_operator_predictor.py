"""Reference-only prediction of one bounded explicit REPID colour operator."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.eval.filmmatch_source_conditioned_operator import (
    fit_ridge,
    predict_ridge,
    source_descriptor,
)
from src.eval.repid_shared_operator import load_jpeg_as_srgb
from src.eval.spcp_global_logit_affine import (
    LogitAffineOperator,
    apply_operator,
    fit_operator_rows,
    gradient_p999_ratio,
    matrix_diagnostics,
    mean_oklab_error,
    new_exact_boundary_fraction,
    sample_indexes,
)


def load_original_srgb(path: Path, *, expected_sha256: str) -> np.ndarray:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError(f"original hash drift: {path}")
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        if (
            image.format != "JPEG"
            or image.mode != "RGB"
            or image.info.get("icc_profile")
        ):
            raise ValueError(f"original ingress drift: {path}")
        return np.asarray(image, dtype=np.float32) / np.float32(255.0)


def resize_srgb(source: np.ndarray, width: int, height: int) -> np.ndarray:
    values = np.rint(np.clip(source, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(values, mode="RGB").resize(
        (int(width), int(height)), Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.float32) / np.float32(255.0)


def reference_descriptor(
    reference: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    maximum_side = int(config["reference_descriptor_maximum_side"])
    height, width = reference.shape[:2]
    if max(height, width) > maximum_side:
        scale = maximum_side / max(height, width)
        reference = resize_srgb(
            reference, max(1, round(width * scale)), max(1, round(height * scale))
        )
    return source_descriptor(reference.reshape(-1, 3), config)


def encode_effect(operator: LogitAffineOperator) -> np.ndarray:
    return np.concatenate(((operator.matrix - np.eye(3)).reshape(-1), operator.bias))


def decode_safe_effect(
    parameters: np.ndarray, config: Mapping[str, Any]
) -> tuple[LogitAffineOperator, int]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (12,) or not np.all(np.isfinite(values)):
        raise ValueError("effect parameters must be finite 12D")
    lower = np.asarray(config["parameter_lower_bounds"], dtype=np.float64)
    upper = np.asarray(config["parameter_upper_bounds"], dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    clip_count = int(np.count_nonzero(clipped != values))
    delta = clipped[:9].reshape(3, 3)
    bias = clipped[9:]
    gates = config["matrix_gates"]
    for dose in map(float, config["safe_dose_grid"]):
        operator = LogitAffineOperator(np.eye(3) + dose * delta, dose * bias, dose)
        diagnostics = matrix_diagnostics(operator)
        if (
            diagnostics["determinant"] >= float(gates["determinant_min"])
            and diagnostics["condition_number"] <= float(gates["condition_number_max"])
            and diagnostics["minimum_singular_value"]
            >= float(gates["minimum_singular_value"])
        ):
            return operator, clip_count
    raise ValueError("safe dose grid must contain an admissible identity dose")


def fit_scene_effect(
    original: np.ndarray,
    winner: np.ndarray,
    scene_id: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    target = resize_srgb(winner, original.shape[1], original.shape[0])
    indexes = sample_indexes(
        scene_id, original.shape[0] * original.shape[1], int(config["pixels_per_scene"])
    )
    matrix, bias = fit_operator_rows(
        original.reshape(-1, 3)[indexes],
        target.reshape(-1, 3)[indexes],
        ridge_alpha=float(config["operator_ridge_alpha"]),
    )
    raw = np.concatenate(((matrix - np.eye(3)).reshape(-1), bias))
    operator, clip_count = decode_safe_effect(raw, config)
    return encode_effect(operator), {
        **matrix_diagnostics(operator),
        "dose": float(operator.dose),
        "clip_count": float(clip_count),
    }


def evaluate_effects(
    original: np.ndarray,
    winner: np.ndarray,
    operators: Mapping[str, LogitAffineOperator],
) -> dict[str, Any]:
    target = resize_srgb(winner, original.shape[1], original.shape[0])
    identity_error = mean_oklab_error(original, target)
    errors: dict[str, float] = {}
    candidate_output = None
    for name, operator in operators.items():
        output = apply_operator(original, operator)
        errors[name] = mean_oklab_error(output, target)
        if name == "candidate":
            candidate_output = output
    if candidate_output is None:
        raise ValueError("candidate operator is required")
    return {
        "identity_error": identity_error,
        "errors": errors,
        "candidate_output_delta_e_oklab": mean_oklab_error(candidate_output, original),
        "candidate_new_exact_boundary_fraction": new_exact_boundary_fraction(
            original, candidate_output
        ),
        "candidate_p999_gradient_ratio": gradient_p999_ratio(
            original, candidate_output
        ),
    }


def relative_gain(baseline: float, candidate: float) -> float:
    return float((baseline - candidate) / max(baseline, 1.0e-12))


def model_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: value.tolist() if isinstance(value, np.ndarray) else value
        for name, value in sorted(model.items())
    }


__all__ = [
    "decode_safe_effect",
    "encode_effect",
    "evaluate_effects",
    "fit_ridge",
    "fit_scene_effect",
    "load_jpeg_as_srgb",
    "load_original_srgb",
    "matrix_diagnostics",
    "model_payload",
    "predict_ridge",
    "reference_descriptor",
    "relative_gain",
]
