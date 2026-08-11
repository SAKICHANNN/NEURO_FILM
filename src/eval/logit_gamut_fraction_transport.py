"""CB25 logit-RGB Gaussian transport with gamut-fraction reconstruction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.global_chroma_gaussian_transport import _symmetric_matrix_power
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb25_logit_gamut_fraction_transport_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb25_logit_gamut_fraction_transport_report.v1"
EXPERIMENT_ID = "U5.R2CB25"


class LogitGamutFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise LogitGamutFractionTransportError("CB25 contract structure drift")
    return payload


def _maximum_chroma_magnitude(luma: np.ndarray, unit: np.ndarray) -> np.ndarray:
    maximum = np.full(luma.shape, np.inf, dtype=np.float64)
    for channel in range(3):
        direction = unit[..., channel]
        positive = direction > 0.0
        negative = direction < 0.0
        maximum = np.minimum(
            maximum,
            np.divide(
                1.0 - luma,
                direction,
                out=np.full_like(luma, np.inf),
                where=positive,
            ),
        )
        maximum = np.minimum(
            maximum,
            np.divide(
                luma,
                -direction,
                out=np.full_like(luma, np.inf),
                where=negative,
            ),
        )
    return maximum


def logit_gamut_fraction_transport_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    logit_epsilon: float = 1.0 / 65535.0,
    minimum_raw_covariance_eigenvalue: float = 1e-6,
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
        or abs(float(np.sum(w)) - 1.0) > 1e-12
    ):
        raise LogitGamutFractionTransportError("CB25 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    clipped_base = np.clip(base64, logit_epsilon, 1.0 - logit_epsilon)
    clipped_ao6 = np.clip(ao664, logit_epsilon, 1.0 - logit_epsilon)
    base_logit = np.log(clipped_base / (1.0 - clipped_base)).reshape(-1, 3)
    ao6_logit = np.log(clipped_ao6 / (1.0 - clipped_ao6)).reshape(-1, 3)
    base_mean = np.mean(base_logit, axis=0)
    ao6_mean = np.mean(ao6_logit, axis=0)
    base_centered = base_logit - base_mean
    ao6_centered = ao6_logit - ao6_mean
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
        raise LogitGamutFractionTransportError("CB25 raw covariance envelope failed")
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
        raise LogitGamutFractionTransportError("CB25 transport envelope failed")
    mapped_logit = (base_logit - base_mean) @ transport.T + ao6_mean
    mapped = (1.0 / (1.0 + np.exp(-mapped_logit))).reshape(base.shape)
    mapped_luma = np.sum(mapped * w, axis=-1)
    mapped_chroma = mapped - mapped_luma[..., None]
    mapped_magnitude = np.linalg.norm(mapped_chroma, axis=-1)
    unit = np.zeros_like(mapped_chroma)
    nonneutral = mapped_magnitude > 1e-12
    unit[nonneutral] = mapped_chroma[nonneutral] / mapped_magnitude[nonneutral, None]
    mapped_maximum = _maximum_chroma_magnitude(mapped_luma, unit)
    gamut_fraction = np.zeros_like(mapped_luma)
    gamut_fraction[nonneutral] = (
        mapped_magnitude[nonneutral] / mapped_maximum[nonneutral]
    )
    base_luma = np.sum(base64 * w, axis=-1)
    base_maximum = _maximum_chroma_magnitude(base_luma, unit)
    target = np.asarray(
        base_luma[..., None]
        + gamut_fraction[..., None] * base_maximum[..., None] * unit,
        dtype=np.float32,
    )
    if (
        not np.isfinite(target).all()
        or np.min(target) < -1e-7
        or np.max(target) > 1.0 + 1e-7
    ):
        raise LogitGamutFractionTransportError("CB25 target invariant failed")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb24_decision_path"],
        config["parents"]["cb24_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb24_required_status"]:
        raise LogitGamutFractionTransportError("CB24 decision drift")
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return logit_gamut_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            logit_epsilon=float(operator["logit_epsilon"]),
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
        contract_filename="u5_r2cb25_logit_gamut_fraction_transport_v1.json",
        blind_seed=20260811 + 2500,
    )


__all__ = [
    "LogitGamutFractionTransportError",
    "evaluate",
    "load_contract",
    "logit_gamut_fraction_transport_target",
]
