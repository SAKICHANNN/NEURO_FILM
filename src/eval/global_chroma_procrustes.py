"""CB20 image-global zero-luminance chroma Procrustes rotation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb20_global_chroma_procrustes_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb20_global_chroma_procrustes_report.v1"
EXPERIMENT_ID = "U5.R2CB20"


class GlobalChromaProcrustesError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GlobalChromaProcrustesError("CB20 contract structure drift")
    return payload


def _plane_basis(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = w / np.linalg.norm(w)
    first = np.asarray([w[1], -w[0], 0.0], dtype=np.float64)
    first /= np.linalg.norm(first)
    second = np.cross(w, first)
    second /= np.linalg.norm(second)
    return np.stack([first, second], axis=1)


def global_chroma_procrustes_target(
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
        raise GlobalChromaProcrustesError("CB20 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    ao6_luma = np.sum(ao664 * w, axis=-1)
    basis = _plane_basis(w)
    base_xy = (base64 - base_luma[..., None]) @ basis
    ao6_xy = (ao664 - ao6_luma[..., None]) @ basis
    dot = float(
        np.sum(base_xy[..., 0] * ao6_xy[..., 0] + base_xy[..., 1] * ao6_xy[..., 1])
    )
    cross = float(
        np.sum(base_xy[..., 0] * ao6_xy[..., 1] - base_xy[..., 1] * ao6_xy[..., 0])
    )
    angle = np.arctan2(cross, dot)
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated_xy = np.empty_like(base_xy)
    rotated_xy[..., 0] = cosine * base_xy[..., 0] - sine * base_xy[..., 1]
    rotated_xy[..., 1] = sine * base_xy[..., 0] + cosine * base_xy[..., 1]
    target_chroma = rotated_xy @ basis.T
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(target).all():
        raise GlobalChromaProcrustesError("CB20 target is nonfinite")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb19_decision_path"],
        config["parents"]["cb19_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb19_required_status"]:
        raise GlobalChromaProcrustesError("CB19 decision drift")
    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=global_chroma_procrustes_target,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb20_global_chroma_procrustes_v1.json",
    )


__all__ = [
    "GlobalChromaProcrustesError",
    "evaluate",
    "global_chroma_procrustes_target",
    "load_contract",
]
