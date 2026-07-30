"""Bounded image-adaptive explicit LUT-basis development experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


class FiveKAdaptiveLUTError(ValueError):
    """Raised when the frozen adaptive-LUT contract or evidence drifts."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_hashed_json(
    root: Path, item: Mapping[str, Any], key: str
) -> dict[str, Any]:
    path = root / str(item[key])
    if not path.is_file() or _sha256(path) != str(
        item[f"{key}_sha256"]
    ).lower():
        raise FiveKAdaptiveLUTError(f"evidence drift: {item[key]}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_contract(
    root: Path, config: Mapping[str, Any]
) -> dict[str, Any]:
    if config.get("status") != "contract_frozen_implementation_ready":
        raise FiveKAdaptiveLUTError("contract is not frozen")
    operator = config["operator"]
    prediction = config["basis_prediction"]
    if (
        operator.get("grid_size") != 4
        or operator.get("hard_output_clipping_allowed")
        or operator.get("spatial_or_semantic_features_allowed")
        or prediction.get("basis_rank") != 8
        or prediction.get("learned_final_rgb_allowed")
        or config.get("new_data_download_allowed")
        or config.get("production_integration_allowed")
        or config.get("film_or_stock_claim_allowed")
        or set(config["methods"])
        != {
            "identity",
            "global_lut",
            "adaptive_basis_lut",
            "oracle_fitted_lut",
        }
    ):
        raise FiveKAdaptiveLUTError("adaptive-LUT boundary drift")
    curve = _load_hashed_json(
        root, config["parents"]["confirmed_curve"], "decision"
    )
    product = _load_hashed_json(
        root, config["parents"]["closed_curve_product_value"], "decision"
    )
    if (
        curve.get("status")
        != config["parents"]["confirmed_curve"]["required_status"]
        or product.get("status")
        != config["parents"]["closed_curve_product_value"]["required_status"]
    ):
        raise FiveKAdaptiveLUTError("parent decision drift")
    populations = []
    for item in config["development_populations"]:
        manifest = _load_hashed_json(root, item, "manifest")
        rows = manifest.get("rows", [])
        if len(rows) != int(item["rows"]):
            raise FiveKAdaptiveLUTError(
                f"population row drift: {item['name']}"
            )
        populations.append({**item, "manifest_payload": manifest})
    return {"populations": populations}


def _trilinear_features(rgb: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64).reshape(-1, 3)
    if (
        grid_size < 2
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise FiveKAdaptiveLUTError("invalid LUT input")
    scaled = values * (grid_size - 1)
    lower = np.minimum(
        np.floor(scaled).astype(np.int64), grid_size - 2
    )
    fraction = scaled - lower
    features = np.zeros(
        (len(values), grid_size**3), dtype=np.float64
    )
    rows = np.arange(len(values))
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                weight = (
                    (fraction[:, 0] if dr else 1.0 - fraction[:, 0])
                    * (fraction[:, 1] if dg else 1.0 - fraction[:, 1])
                    * (fraction[:, 2] if db else 1.0 - fraction[:, 2])
                )
                node = (
                    (lower[:, 0] + dr) * grid_size * grid_size
                    + (lower[:, 1] + dg) * grid_size
                    + lower[:, 2]
                    + db
                )
                features[rows, node] += weight
    return features


def _smoothness_matrix(grid_size: int) -> np.ndarray:
    rows = []
    for r in range(grid_size):
        for g in range(grid_size):
            for b in range(grid_size):
                current = (r * grid_size + g) * grid_size + b
                for axis in range(3):
                    peer = [r, g, b]
                    if peer[axis] + 1 >= grid_size:
                        continue
                    peer[axis] += 1
                    adjacent = (
                        (peer[0] * grid_size + peer[1]) * grid_size
                        + peer[2]
                    )
                    row = np.zeros(grid_size**3, dtype=np.float64)
                    row[current] = 1.0
                    row[adjacent] = -1.0
                    rows.append(row)
    return np.stack(rows)


def fit_residual_lut(
    source: np.ndarray,
    target: np.ndarray,
    *,
    grid_size: int,
    sample_stride: int,
    identity_shrinkage: float,
    smoothness: float,
    maximum_absolute_residual: float,
) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if (
        source_array.shape != target_array.shape
        or source_array.ndim != 3
        or source_array.shape[2] != 3
        or sample_stride < 1
        or identity_shrinkage <= 0.0
        or smoothness < 0.0
        or maximum_absolute_residual <= 0.0
    ):
        raise FiveKAdaptiveLUTError("invalid LUT fit peers")
    sampled_source = source_array[::sample_stride, ::sample_stride]
    sampled_target = target_array[::sample_stride, ::sample_stride]
    features = _trilinear_features(sampled_source, grid_size)
    residual = (sampled_target - sampled_source).reshape(-1, 3)
    differences = _smoothness_matrix(grid_size)
    system = (
        features.T @ features
        + identity_shrinkage * np.eye(grid_size**3)
        + smoothness * (differences.T @ differences)
    )
    nodes = np.linalg.solve(system, features.T @ residual)
    return np.clip(
        nodes.reshape(grid_size, grid_size, grid_size, 3),
        -maximum_absolute_residual,
        maximum_absolute_residual,
    )


def apply_residual_lut(source: np.ndarray, residual_lut: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source, dtype=np.float64)
    lut = np.asarray(residual_lut, dtype=np.float64)
    if (
        source_array.ndim != 3
        or source_array.shape[2] != 3
        or lut.ndim != 4
        or lut.shape[:3] != (lut.shape[0],) * 3
        or lut.shape[3] != 3
    ):
        raise FiveKAdaptiveLUTError("invalid residual LUT")
    residual = _trilinear_features(source_array, lut.shape[0]) @ lut.reshape(
        -1, 3
    )
    return source_array + residual.reshape(source_array.shape)


__all__ = [
    "FiveKAdaptiveLUTError",
    "apply_residual_lut",
    "fit_residual_lut",
    "validate_contract",
]
