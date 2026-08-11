"""CB27 circular hue plus empirical monotone gamut-fraction transport."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.circular_hue_fraction_transport import _hue_fraction
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb27_monotone_fraction_quantile_transport_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb27_monotone_fraction_quantile_transport_report.v1"
EXPERIMENT_ID = "U5.R2CB27"


class MonotoneFractionQuantileTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise MonotoneFractionQuantileTransportError("CB27 contract structure drift")
    return payload


def _circular_rotation_angle(
    base_unit: np.ndarray,
    ao6_unit: np.ndarray,
    base_fraction: np.ndarray,
    ao6_fraction: np.ndarray,
    base_valid: np.ndarray,
    ao6_valid: np.ndarray,
    *,
    minimum_valid_fraction: float,
) -> float:
    fit_valid = (
        base_valid
        & ao6_valid
        & (base_fraction >= minimum_valid_fraction)
        & (ao6_fraction >= minimum_valid_fraction)
    )
    if np.count_nonzero(fit_valid) < 2:
        raise MonotoneFractionQuantileTransportError("CB27 hue population failed")
    base_fit = base_unit[fit_valid]
    ao6_fit = ao6_unit[fit_valid]
    dot = float(np.sum(base_fit[:, 0] * ao6_fit[:, 0] + base_fit[:, 1] * ao6_fit[:, 1]))
    cross = float(
        np.sum(base_fit[:, 0] * ao6_fit[:, 1] - base_fit[:, 1] * ao6_fit[:, 0])
    )
    return float(np.arctan2(cross, dot))


def _empirical_quantile_transport(
    base_fraction: np.ndarray,
    ao6_fraction: np.ndarray,
    base_valid: np.ndarray,
    ao6_valid: np.ndarray,
    *,
    minimum_valid_fraction: float,
) -> np.ndarray:
    base_mask = base_valid & (base_fraction >= minimum_valid_fraction)
    ao6_mask = ao6_valid & (ao6_fraction >= minimum_valid_fraction)
    base_values = base_fraction[base_mask]
    ao6_values = np.sort(ao6_fraction[ao6_mask], kind="stable")
    if base_values.size < 2 or ao6_values.size < 2:
        raise MonotoneFractionQuantileTransportError("CB27 fraction population failed")
    order = np.argsort(base_values, kind="stable")
    quantiles = (np.arange(base_values.size, dtype=np.float64) + 0.5) / float(
        base_values.size
    )
    position = quantiles * float(ao6_values.size - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, ao6_values.size - 1)
    alpha = position - lower
    sorted_mapped = (1.0 - alpha) * ao6_values[lower] + alpha * ao6_values[upper]
    mapped_values = np.empty_like(base_values)
    mapped_values[order] = sorted_mapped
    mapped = np.zeros_like(base_fraction)
    mapped[base_mask] = mapped_values
    if (
        not np.isfinite(mapped).all()
        or np.min(mapped) < 0.0
        or np.max(mapped) > 1.0 + 1e-12
    ):
        raise MonotoneFractionQuantileTransportError("CB27 fraction invariant failed")
    return mapped


def monotone_fraction_quantile_transport_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_valid_fraction: float = 1e-4,
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
        raise MonotoneFractionQuantileTransportError("CB27 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    basis = _plane_basis(w)
    base_luma, base_unit, base_fraction, base_valid = _hue_fraction(base64, w, basis)
    _, ao6_unit, ao6_fraction, ao6_valid = _hue_fraction(ao664, w, basis)
    angle = _circular_rotation_angle(
        base_unit,
        ao6_unit,
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    mapped_fraction = _empirical_quantile_transport(
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated_unit = np.empty_like(base_unit)
    rotated_unit[..., 0] = cosine * base_unit[..., 0] - sine * base_unit[..., 1]
    rotated_unit[..., 1] = sine * base_unit[..., 0] + cosine * base_unit[..., 1]
    rotated_rgb = rotated_unit @ basis.T
    maximum = _maximum_chroma_magnitude(base_luma, rotated_rgb)
    target_chroma = np.zeros_like(rotated_rgb)
    mapped_valid = mapped_fraction > 0.0
    target_chroma[mapped_valid] = (
        mapped_fraction[mapped_valid, None]
        * maximum[mapped_valid, None]
        * rotated_rgb[mapped_valid]
    )
    target = np.asarray(base_luma[..., None] + target_chroma, dtype=np.float32)
    if (
        not np.isfinite(target).all()
        or np.min(target) < -1e-7
        or np.max(target) > 1.0 + 1e-7
    ):
        raise MonotoneFractionQuantileTransportError("CB27 target invariant failed")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb26_decision_path"],
        config["parents"]["cb26_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb26_required_status"]:
        raise MonotoneFractionQuantileTransportError("CB26 decision drift")
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return monotone_fraction_quantile_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=float(operator["minimum_valid_fraction"]),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb27_monotone_fraction_quantile_transport_v1.json",
        blind_seed=20260811 + 2700,
    )


__all__ = [
    "MonotoneFractionQuantileTransportError",
    "evaluate",
    "load_contract",
    "monotone_fraction_quantile_transport_target",
]
