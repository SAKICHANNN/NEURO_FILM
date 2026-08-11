"""CB23 globally dose CB22 transport from an analytical feasibility quantile."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.global_chroma_gaussian_transport import (
    global_chroma_gaussian_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb23_feasibility_bounded_gaussian_transport_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb23_feasibility_bounded_gaussian_transport_report.v1"
EXPERIMENT_ID = "U5.R2CB23"


class FeasibilityBoundedGaussianTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise FeasibilityBoundedGaussianTransportError("CB23 contract structure drift")
    return payload


def _maximum_feasible_step(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    target_linear: np.ndarray,
    *,
    boundary_epsilon: float,
) -> np.ndarray:
    source = np.asarray(source_linear)
    base = np.asarray(safe_base_linear)
    target = np.asarray(target_linear)
    if (
        source.dtype != np.float32
        or base.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != base.shape
        or source.shape != target.shape
        or source.ndim != 3
        or source.shape[-1] != 3
        or not np.isfinite(source).all()
        or not np.isfinite(base).all()
        or not np.isfinite(target).all()
    ):
        raise FeasibilityBoundedGaussianTransportError("CB23 feasibility input drift")
    base64 = base.astype(np.float64)
    residual = target.astype(np.float64) - base64
    feasible = np.ones(source.shape[:-1], dtype=np.float64)
    lower_target = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper_target = float(upper_edge - np.float32(4.0) * np.spacing(upper_edge))
    for channel in range(3):
        delta = residual[..., channel]
        lower = np.where(source[..., channel] > boundary_epsilon, lower_target, 0.0)
        upper = np.where(
            source[..., channel] < 1.0 - boundary_epsilon, upper_target, 1.0
        )
        positive = delta > 0.0
        negative = delta < 0.0
        feasible = np.minimum(
            feasible,
            np.where(
                positive,
                np.divide(
                    upper - base64[..., channel],
                    delta,
                    out=np.full_like(delta, np.inf),
                    where=positive,
                ),
                np.inf,
            ),
        )
        feasible = np.minimum(
            feasible,
            np.where(
                negative,
                np.divide(
                    base64[..., channel] - lower,
                    -delta,
                    out=np.full_like(delta, np.inf),
                    where=negative,
                ),
                np.inf,
            ),
        )
    return np.clip(feasible, 0.0, 1.0)


def feasibility_bounded_gaussian_transport_target(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
    minimum_raw_covariance_eigenvalue: float = 1e-8,
    minimum_transport_eigenvalue: float = 0.25,
    maximum_transport_eigenvalue: float = 4.0,
    feasibility_quantile: float = 0.1,
    minimum_retained_step_at_quantile: float = 0.5,
) -> np.ndarray:
    gaussian_target = global_chroma_gaussian_transport_target(
        safe_base_linear,
        ao6_linear,
        weights=weights,
        boundary_epsilon=boundary_epsilon,
        minimum_raw_covariance_eigenvalue=minimum_raw_covariance_eigenvalue,
        minimum_transport_eigenvalue=minimum_transport_eigenvalue,
        maximum_transport_eigenvalue=maximum_transport_eigenvalue,
    )
    feasible = _maximum_feasible_step(
        source_linear,
        safe_base_linear,
        gaussian_target,
        boundary_epsilon=boundary_epsilon,
    )
    quantile_step = float(np.quantile(feasible, feasibility_quantile, method="lower"))
    global_dose = min(1.0, quantile_step / minimum_retained_step_at_quantile)
    if not np.isfinite(global_dose) or global_dose <= 0.0:
        raise FeasibilityBoundedGaussianTransportError("CB23 global dose failed")
    base64 = np.asarray(safe_base_linear, dtype=np.float64)
    target = np.asarray(
        base64 + global_dose * (gaussian_target.astype(np.float64) - base64),
        dtype=np.float32,
    )
    if not np.isfinite(target).all():
        raise FeasibilityBoundedGaussianTransportError("CB23 target is nonfinite")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb22_decision_path"],
        config["parents"]["cb22_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb22_required_status"]:
        raise FeasibilityBoundedGaussianTransportError("CB22 decision drift")
    operator = config["operator"]

    def target_builder(
        source_linear: np.ndarray,
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return feasibility_bounded_gaussian_transport_target(
            source_linear,
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
            feasibility_quantile=float(operator["feasibility_quantile"]),
            minimum_retained_step_at_quantile=float(
                operator["minimum_retained_step_at_quantile"]
            ),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb23_feasibility_bounded_gaussian_transport_v1.json",
        blind_seed=20260811 + 2300,
        source_aware_target_builder=True,
    )


__all__ = [
    "FeasibilityBoundedGaussianTransportError",
    "_maximum_feasible_step",
    "evaluate",
    "feasibility_bounded_gaussian_transport_target",
    "load_contract",
]
