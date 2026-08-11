"""CB22 image-global Gaussian transport in the zero-luminance chroma plane."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb22_global_chroma_gaussian_transport_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb22_global_chroma_gaussian_transport_report.v1"
EXPERIMENT_ID = "U5.R2CB22"


class GlobalChromaGaussianTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise GlobalChromaGaussianTransportError("CB22 contract structure drift")
    return payload


def _symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if not np.isfinite(eigenvalues).all() or np.min(eigenvalues) <= 0.0:
        raise GlobalChromaGaussianTransportError(
            "CB22 covariance is not positive definite"
        )
    return (eigenvectors * np.power(eigenvalues, power)) @ eigenvectors.T


def global_chroma_gaussian_transport_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_raw_covariance_eigenvalue: float = 1e-8,
    minimum_transport_eigenvalue: float = 0.25,
    maximum_transport_eigenvalue: float = 4.0,
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
        or w.shape != (3,)
        or not np.isfinite(base).all()
        or not np.isfinite(ao6).all()
    ):
        raise GlobalChromaGaussianTransportError("CB22 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    base_luma = np.sum(base64 * w, axis=-1)
    ao6_luma = np.sum(ao664 * w, axis=-1)
    basis = _plane_basis(w)
    base_xy = ((base64 - base_luma[..., None]) @ basis).reshape(-1, 2)
    ao6_xy = ((ao664 - ao6_luma[..., None]) @ basis).reshape(-1, 2)
    base_mean = np.mean(base_xy, axis=0)
    ao6_mean = np.mean(ao6_xy, axis=0)
    base_centered = base_xy - base_mean
    ao6_centered = ao6_xy - ao6_mean
    base_covariance = base_centered.T @ base_centered / base_centered.shape[0]
    ao6_covariance = ao6_centered.T @ ao6_centered / ao6_centered.shape[0]
    base_eigenvalues = np.linalg.eigvalsh(base_covariance)
    ao6_eigenvalues = np.linalg.eigvalsh(ao6_covariance)
    if (
        not np.isfinite(base_eigenvalues).all()
        or not np.isfinite(ao6_eigenvalues).all()
        or np.min(base_eigenvalues) < minimum_raw_covariance_eigenvalue
        or np.min(ao6_eigenvalues) < minimum_raw_covariance_eigenvalue
    ):
        raise GlobalChromaGaussianTransportError("CB22 raw covariance envelope failed")
    base_sqrt = _symmetric_matrix_power(base_covariance, 0.5)
    base_inverse_sqrt = _symmetric_matrix_power(base_covariance, -0.5)
    middle_sqrt = _symmetric_matrix_power(base_sqrt @ ao6_covariance @ base_sqrt, 0.5)
    transport = base_inverse_sqrt @ middle_sqrt @ base_inverse_sqrt
    transport = 0.5 * (transport + transport.T)
    transport_eigenvalues = np.linalg.eigvalsh(transport)
    if (
        not np.isfinite(transport_eigenvalues).all()
        or np.min(transport_eigenvalues) < minimum_transport_eigenvalue
        or np.max(transport_eigenvalues) > maximum_transport_eigenvalue
    ):
        raise GlobalChromaGaussianTransportError("CB22 transport envelope failed")
    transported_xy = (base_xy - base_mean) @ transport.T + ao6_mean
    target_chroma = transported_xy.reshape(base.shape[:-1] + (2,)) @ basis.T
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(target).all():
        raise GlobalChromaGaussianTransportError("CB22 target is nonfinite")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb21_decision_path"],
        config["parents"]["cb21_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb21_required_status"]:
        raise GlobalChromaGaussianTransportError("CB21 decision drift")
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return global_chroma_gaussian_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_raw_covariance_eigenvalue=float(
                operator["minimum_raw_covariance_eigenvalue"]
            ),
            minimum_transport_eigenvalue=float(
                operator["minimum_transport_eigenvalue"]
            ),
            maximum_transport_eigenvalue=float(
                operator["maximum_transport_eigenvalue"]
            ),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb22_global_chroma_gaussian_transport_v1.json",
        blind_seed=20260811 + 2200,
    )


__all__ = [
    "GlobalChromaGaussianTransportError",
    "evaluate",
    "global_chroma_gaussian_transport_target",
    "load_contract",
]
