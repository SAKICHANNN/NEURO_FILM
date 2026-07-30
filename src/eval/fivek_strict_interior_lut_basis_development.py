"""Strict-epsilon-interior adaptive explicit LUT-basis experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.eval.fivek_adaptive_lut_basis_development import (
    FiveKAdaptiveLUTError,
    _load_hashed_json,
    _sha256,
    _smoothness_matrix,
    _trilinear_features,
    validate_contract as validate_bj0_contract,
)
from src.eval.fivek_cube_preserving_lut_basis_development import (
    run_validated_development,
)


EPSILON = 1.0 / 510.0
HEADROOM_SAFETY_FACTOR = 0.9999999999


class FiveKStrictInteriorLUTError(ValueError):
    """Raised when strict-interior LUT evidence or its proof drifts."""


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKStrictInteriorLUTError("contract is not frozen")
    operator = config["operator"]
    if (
        operator.get("grid_size") != 4
        or operator.get("boundary_epsilon") != EPSILON
        or operator.get("headroom_safety_factor") != HEADROOM_SAFETY_FACTOR
        or operator.get("node_coefficient_minimum") != -1.0
        or operator.get("node_coefficient_maximum") != 1.0
        or operator.get("hard_output_clipping_allowed")
        or operator.get("post_operator_gamut_scaling_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or config["basis_prediction"].get("basis_rank") != 8
        or config["basis_prediction"].get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
    ):
        raise FiveKStrictInteriorLUTError("strict-interior boundary drift")
    try:
        parent = _load_hashed_json(root, config["parent"], "decision")
    except FiveKAdaptiveLUTError as exc:
        raise FiveKStrictInteriorLUTError(str(exc)) from exc
    if parent.get("status") != config["parent"]["required_status"]:
        raise FiveKStrictInteriorLUTError("parent decision drift")
    populations = []
    for item in config["development_populations"]:
        try:
            manifest = _load_hashed_json(root, item, "manifest")
        except FiveKAdaptiveLUTError as exc:
            raise FiveKStrictInteriorLUTError(str(exc)) from exc
        if len(manifest.get("rows", [])) != int(item["rows"]):
            raise FiveKStrictInteriorLUTError(f"row drift: {item['name']}")
        populations.append({**item, "manifest_payload": manifest})
    return {"populations": populations}


def _headroom(source: np.ndarray) -> np.ndarray:
    return HEADROOM_SAFETY_FACTOR * np.maximum(
        0.0, np.minimum(source - EPSILON, 1.0 - EPSILON - source)
    )


def apply_strict_interior_lut(
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
        or np.any(lut < -1.0)
        or np.any(lut > 1.0)
    ):
        raise FiveKStrictInteriorLUTError("invalid strict-interior peers")
    coefficient = (
        _trilinear_features(source_array, 4) @ lut.reshape(-1, 3)
    ).reshape(source_array.shape)
    output = source_array + _headroom(source_array) * coefficient
    if np.any(output < 0.0) or np.any(output > 1.0):
        raise FiveKStrictInteriorLUTError("strict-interior proof failed")
    return output


def fit_strict_interior_lut(
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
        or coefficient_minimum != -1.0
        or coefficient_maximum != 1.0
    ):
        raise FiveKStrictInteriorLUTError("invalid strict-interior fit")
    sampled_source = source_array[::sample_stride, ::sample_stride]
    sampled_target = target_array[::sample_stride, ::sample_stride]
    features = _trilinear_features(sampled_source, grid_size)
    residual = (sampled_target - sampled_source).reshape(-1, 3)
    headroom = _headroom(sampled_source).reshape(-1, 3)
    differences = _smoothness_matrix(grid_size)
    regularizer = (
        identity_shrinkage * np.eye(grid_size**3)
        + smoothness * (differences.T @ differences)
    )
    nodes = np.empty((grid_size**3, 3), dtype=np.float64)
    for channel in range(3):
        design = features * headroom[:, channel, None]
        nodes[:, channel] = np.linalg.solve(
            design.T @ design + regularizer,
            design.T @ residual[:, channel],
        )
    return np.clip(
        nodes.reshape(grid_size, grid_size, grid_size, 3),
        coefficient_minimum,
        coefficient_maximum,
    )


def run_development(
    *,
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    output_dir: Path,
    software_commit: str,
) -> dict[str, Any]:
    validated = validate_contract(root, config)
    parent = _load_hashed_json(root, config["parent"], "decision")
    bj1_config_path = root / parent["config"]
    bj1_report = json.loads(
        (root / parent["report"]).read_text(encoding="utf-8")
    )
    if _sha256(bj1_config_path) != bj1_report["config_sha256"]:
        raise FiveKStrictInteriorLUTError("BJ1 config drift")
    bj1_config = json.loads(bj1_config_path.read_text(encoding="utf-8"))
    bj0_decision = _load_hashed_json(
        root, bj1_config["parent"], "decision"
    )
    bj0_config_path = root / bj0_decision["config"]
    bj0_report = json.loads(
        (root / bj0_decision["report"]).read_text(encoding="utf-8")
    )
    if _sha256(bj0_config_path) != bj0_report["config_sha256"]:
        raise FiveKStrictInteriorLUTError("BJ0 config drift")
    bj0 = validate_bj0_contract(
        root, json.loads(bj0_config_path.read_text(encoding="utf-8"))
    )
    return run_validated_development(
        root=root,
        config=config,
        config_path=config_path,
        output_dir=output_dir,
        software_commit=software_commit,
        validated=validated,
        bj0=bj0,
        fit_function=fit_strict_interior_lut,
        apply_function=apply_strict_interior_lut,
    )


__all__ = [
    "FiveKStrictInteriorLUTError",
    "HEADROOM_SAFETY_FACTOR",
    "apply_strict_interior_lut",
    "fit_strict_interior_lut",
    "run_development",
    "validate_contract",
]
