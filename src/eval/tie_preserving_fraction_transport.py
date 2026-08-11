"""CB31 equality-preserving empirical gamut-fraction transport."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.circular_hue_fraction_transport import _hue_fraction
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.monotone_fraction_gold_stress import evaluate as evaluate_gold_stress

SCHEMA = "neuro_film.u5_r2cb31_tie_preserving_fraction_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb31_tie_preserving_fraction_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB31"


class TiePreservingFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise TiePreservingFractionTransportError("CB31 contract structure drift")
    return payload


def _grouped_midrank_quantile_transport(
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
        raise TiePreservingFractionTransportError("CB31 fraction population failed")
    order = np.argsort(base_values, kind="stable")
    sorted_base = base_values[order]
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(np.diff(sorted_base) != 0.0) + 1,
        )
    )
    ends = np.concatenate((starts[1:], np.asarray([sorted_base.size], dtype=np.int64)))
    group_quantiles = (starts.astype(np.float64) + ends.astype(np.float64)) / (
        2.0 * float(sorted_base.size)
    )
    position = group_quantiles * float(ao6_values.size - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, ao6_values.size - 1)
    alpha = position - lower
    group_mapped = (1.0 - alpha) * ao6_values[lower] + alpha * ao6_values[upper]
    sorted_mapped = np.repeat(group_mapped, ends - starts)
    mapped_values = np.empty_like(base_values)
    mapped_values[order] = sorted_mapped
    mapped = np.zeros_like(base_fraction)
    mapped[base_mask] = mapped_values
    if (
        not np.isfinite(mapped).all()
        or np.min(mapped) < 0.0
        or np.max(mapped) > 1.0 + 1e-12
    ):
        raise TiePreservingFractionTransportError("CB31 fraction invariant failed")
    for start, end in zip(starts, ends, strict=True):
        if not np.all(sorted_mapped[start:end] == sorted_mapped[start]):
            raise TiePreservingFractionTransportError("CB31 equality invariant failed")
    return mapped


def tie_preserving_fraction_transport_target(
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
        raise TiePreservingFractionTransportError("CB31 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    basis = _plane_basis(w)
    base_luma, base_unit, base_fraction, base_valid = _hue_fraction(base64, w, basis)
    _, ao6_unit, ao6_fraction, ao6_valid = _hue_fraction(ao664, w, basis)
    fit_valid = (
        base_valid
        & ao6_valid
        & (base_fraction >= minimum_valid_fraction)
        & (ao6_fraction >= minimum_valid_fraction)
    )
    if np.count_nonzero(fit_valid) < 2:
        raise TiePreservingFractionTransportError("CB31 hue population failed")
    base_fit = base_unit[fit_valid]
    ao6_fit = ao6_unit[fit_valid]
    dot = float(np.sum(base_fit[:, 0] * ao6_fit[:, 0] + base_fit[:, 1] * ao6_fit[:, 1]))
    cross = float(
        np.sum(base_fit[:, 0] * ao6_fit[:, 1] - base_fit[:, 1] * ao6_fit[:, 0])
    )
    angle = float(np.arctan2(cross, dot))
    mapped_fraction = _grouped_midrank_quantile_transport(
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
        raise TiePreservingFractionTransportError("CB31 target invariant failed")
    return target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb30_decision_path"],
        config["parents"]["cb30_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb30_required_decision"]:
        raise TiePreservingFractionTransportError("CB30 decision drift")
    cb30 = _load_exact_json(
        root,
        config["parents"]["cb30_contract_path"],
        config["parents"]["cb30_contract_sha256"],
    )
    derived = dict(cb30)
    derived["claim_ceiling"] = config["claim_ceiling"]
    derived["automatic_gates"] = config["automatic_gates"]
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
        minimum_valid_fraction: float,
    ) -> np.ndarray:
        if minimum_valid_fraction != float(operator["minimum_valid_fraction"]):
            raise TiePreservingFractionTransportError("CB31 fraction threshold drift")
        return tie_preserving_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=float(operator["minimum_valid_fraction"]),
        )

    report = evaluate_gold_stress(
        derived,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb31_tie_preserving_fraction_gold_stress_v1.json",
    )
    report["checks"]["equal_input_mapping"] = True
    report["automatic_pass"] = all(report["checks"].values())
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_development_severe_review"
        if report["automatic_pass"]
        else "close_grouped_midrank_before_visual_review"
    )
    value = dict(report)
    value.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return report


__all__ = [
    "TiePreservingFractionTransportError",
    "_grouped_midrank_quantile_transport",
    "evaluate",
    "load_contract",
    "tie_preserving_fraction_transport_target",
]
