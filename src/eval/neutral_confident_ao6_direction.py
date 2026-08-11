"""CB18 neutral-confident AO6 direction with CB11 magnitude."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb18_neutral_confident_ao6_direction_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb18_neutral_confident_ao6_direction_report.v1"
EXPERIMENT_ID = "U5.R2CB18"


class NeutralConfidentAo6DirectionError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise NeutralConfidentAo6DirectionError("CB18 contract structure drift")
    return payload


def neutral_confident_direction_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
) -> np.ndarray:
    base = np.asarray(safe_base_linear)
    ao6 = np.asarray(ao6_linear)
    w = np.asarray(weights, dtype=np.float64)
    if base.dtype != np.float32 or ao6.dtype != np.float32 or base.shape != ao6.shape:
        raise NeutralConfidentAo6DirectionError("CB18 direction input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    ao6_luma = np.sum(ao664 * w, axis=-1)
    base_chroma = base64 - base_luma[..., None]
    ao6_chroma = ao664 - ao6_luma[..., None]
    base_norm = np.linalg.norm(base_chroma, axis=-1)
    ao6_norm = np.linalg.norm(ao6_chroma, axis=-1)
    base_unit = np.zeros_like(base_chroma)
    ao6_unit = np.zeros_like(ao6_chroma)
    base_valid = base_norm > 1e-12
    ao6_valid = ao6_norm > 1e-12
    base_unit[base_valid] = base_chroma[base_valid] / base_norm[base_valid, None]
    ao6_unit[ao6_valid] = ao6_chroma[ao6_valid] / ao6_norm[ao6_valid, None]
    base_relative = np.clip(
        base_norm / (np.abs(base_luma) + boundary_epsilon), 0.0, 1.0
    )
    ao6_relative = np.clip(ao6_norm / (np.abs(ao6_luma) + boundary_epsilon), 0.0, 1.0)
    confidence = np.minimum(base_relative, ao6_relative)
    confidence[~(base_valid & ao6_valid)] = 0.0
    mixed = (1.0 - confidence[..., None]) * base_unit + confidence[..., None] * ao6_unit
    mixed_norm = np.linalg.norm(mixed, axis=-1)
    mixed_valid = mixed_norm > 1e-12
    target_unit = base_unit.copy()
    target_unit[mixed_valid] = mixed[mixed_valid] / mixed_norm[mixed_valid, None]
    target_chroma = base_norm[..., None] * target_unit
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(target).all():
        raise NeutralConfidentAo6DirectionError("CB18 target is nonfinite")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb17_decision_path"],
        config["parents"]["cb17_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb17_required_status"]:
        raise NeutralConfidentAo6DirectionError("CB17 decision drift")
    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=neutral_confident_direction_target,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb18_neutral_confident_ao6_direction_v1.json",
    )


__all__ = [
    "NeutralConfidentAo6DirectionError",
    "evaluate",
    "load_contract",
    "neutral_confident_direction_target",
]
