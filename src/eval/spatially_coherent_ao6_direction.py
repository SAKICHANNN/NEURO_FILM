"""CB19 fixed finite-support spatially coherent AO6 direction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import convolve1d

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb19_spatially_coherent_ao6_direction_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb19_spatially_coherent_ao6_direction_report.v1"
EXPERIMENT_ID = "U5.R2CB19"
KERNEL = np.asarray([0.25, 0.5, 0.25], dtype=np.float64)


class SpatiallyCoherentAo6DirectionError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise SpatiallyCoherentAo6DirectionError("CB19 contract structure drift")
    return payload


def spatially_coherent_direction_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
) -> np.ndarray:
    del boundary_epsilon
    base = np.asarray(safe_base_linear)
    ao6 = np.asarray(ao6_linear)
    w = np.asarray(weights, dtype=np.float64)
    if (
        base.dtype != np.float32
        or ao6.dtype != np.float32
        or base.shape != ao6.shape
        or base.ndim != 3
        or base.shape[-1] != 3
    ):
        raise SpatiallyCoherentAo6DirectionError("CB19 direction input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    ao6_luma = np.sum(ao664 * w, axis=-1)
    base_chroma = base64 - base_luma[..., None]
    ao6_chroma = ao664 - ao6_luma[..., None]
    filtered = convolve1d(ao6_chroma, KERNEL, axis=0, mode="reflect")
    filtered = convolve1d(filtered, KERNEL, axis=1, mode="reflect")
    base_norm = np.linalg.norm(base_chroma, axis=-1)
    filtered_norm = np.linalg.norm(filtered, axis=-1)
    target_chroma = base_chroma.copy()
    valid = filtered_norm > 1e-12
    target_chroma[valid] = (
        base_norm[valid, None] * filtered[valid] / filtered_norm[valid, None]
    )
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(target).all():
        raise SpatiallyCoherentAo6DirectionError("CB19 target is nonfinite")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb18_decision_path"],
        config["parents"]["cb18_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb18_required_status"]:
        raise SpatiallyCoherentAo6DirectionError("CB18 decision drift")
    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=spatially_coherent_direction_target,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb19_spatially_coherent_ao6_direction_v1.json",
    )


__all__ = [
    "KERNEL",
    "SpatiallyCoherentAo6DirectionError",
    "evaluate",
    "load_contract",
    "spatially_coherent_direction_target",
]
