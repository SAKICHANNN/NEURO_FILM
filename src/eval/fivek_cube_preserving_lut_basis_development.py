"""Intrinsically cube-preserving adaptive explicit LUT-basis experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_adaptive_lut_basis_development import (
    FiveKAdaptiveLUTError,
    _load_hashed_json,
    _smoothness_matrix,
    _trilinear_features,
)


class FiveKCubeLUTError(ValueError):
    """Raised when the frozen cube-preserving LUT contract drifts."""


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKCubeLUTError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    if (
        operator.get("grid_size") != 4
        or operator.get("channel_envelope") != "4*x_c*(1-x_c)"
        or operator.get("node_coefficient_minimum") != -0.25
        or operator.get("node_coefficient_maximum") != 0.25
        or operator.get("hard_output_clipping_allowed")
        or operator.get("post_operator_gamut_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or prediction.get("basis_rank") != 8
        or prediction.get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise FiveKCubeLUTError("cube-preserving boundary drift")
    try:
        parent = _load_hashed_json(root, config["parent"], "decision")
    except FiveKAdaptiveLUTError as exc:
        raise FiveKCubeLUTError(str(exc)) from exc
    if parent.get("status") != config["parent"]["required_status"]:
        raise FiveKCubeLUTError("parent decision drift")
    populations = []
    for item in config["development_populations"]:
        try:
            manifest = _load_hashed_json(root, item, "manifest")
        except FiveKAdaptiveLUTError as exc:
            raise FiveKCubeLUTError(str(exc)) from exc
        if len(manifest.get("rows", [])) != int(item["rows"]):
            raise FiveKCubeLUTError(f"row drift: {item['name']}")
        populations.append({**item, "manifest_payload": manifest})
    return {"populations": populations}


def apply_cube_preserving_lut(
    source: np.ndarray, coefficient_lut: np.ndarray
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    lut = np.asarray(coefficient_lut, dtype=np.float64)
    if (
        source_array.ndim != 3
        or source_array.shape[2] != 3
        or not np.all(np.isfinite(source_array))
        or np.any(source_array < 0.0)
        or np.any(source_array > 1.0)
        or lut.shape != (4, 4, 4, 3)
        or not np.all(np.isfinite(lut))
        or np.any(lut < -0.25)
        or np.any(lut > 0.25)
    ):
        raise FiveKCubeLUTError("invalid cube-preserving LUT peers")
    coefficients = _trilinear_features(source_array, 4) @ lut.reshape(-1, 3)
    coefficients = coefficients.reshape(source_array.shape)
    output = source_array + (
        4.0 * source_array * (1.0 - source_array) * coefficients
    )
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise FiveKCubeLUTError("cube-preserving proof failed")
    return output


def fit_cube_preserving_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    grid_size: int,
    sample_stride: int,
    identity_shrinkage: float,
    smoothness: float,
    coefficient_minimum: float,
    coefficient_maximum: float,
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 3
        or source_array.shape[2] != 3
        or grid_size != 4
        or sample_stride < 1
        or coefficient_minimum != -0.25
        or coefficient_maximum != 0.25
    ):
        raise FiveKCubeLUTError("invalid cube-preserving fit peers")
    sampled_source = source_array[::sample_stride, ::sample_stride]
    sampled_target = target_array[::sample_stride, ::sample_stride]
    features = _trilinear_features(sampled_source, grid_size)
    residual = (sampled_target - sampled_source).reshape(-1, 3)
    envelope = (
        4.0 * sampled_source * (1.0 - sampled_source)
    ).reshape(-1, 3)
    differences = _smoothness_matrix(grid_size)
    regularizer = (
        identity_shrinkage * np.eye(grid_size**3)
        + smoothness * (differences.T @ differences)
    )
    nodes = np.empty((grid_size**3, 3), dtype=np.float64)
    for channel in range(3):
        design = features * envelope[:, channel, None]
        nodes[:, channel] = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ residual[:, channel],
        )
    return np.clip(
        nodes.reshape(grid_size, grid_size, grid_size, 3),
        coefficient_minimum,
        coefficient_maximum,
    )


__all__ = [
    "FiveKCubeLUTError",
    "apply_cube_preserving_lut",
    "fit_cube_preserving_lut",
    "validate_contract",
]
