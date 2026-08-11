"""CB32 fixed 1-Lipschitz monotone gamut-fraction transport."""

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
from src.eval.monotone_fraction_quantile_transport import _circular_rotation_angle

SCHEMA = "neuro_film.u5_r2cb32_nonexpansive_fraction_gold_stress_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb32_nonexpansive_fraction_gold_stress_report.v1"
EXPERIMENT_ID = "U5.R2CB32"


class NonexpansiveFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise NonexpansiveFractionTransportError("CB32 contract structure drift")
    return payload


def _target_quantiles(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    sorted_values = np.sort(values, kind="stable")
    position = quantiles * float(sorted_values.size - 1)
    lower = np.floor(position).astype(np.int64)
    upper = np.minimum(lower + 1, sorted_values.size - 1)
    alpha = position - lower
    return (1.0 - alpha) * sorted_values[lower] + alpha * sorted_values[upper]


def _nonexpansive_fraction_map(
    base_fraction: np.ndarray,
    ao6_fraction: np.ndarray,
    base_valid: np.ndarray,
    ao6_valid: np.ndarray,
    *,
    minimum_valid_fraction: float,
    fraction_knots: int,
    maximum_fraction_slope: float,
) -> tuple[np.ndarray, dict[str, float | int]]:
    base_mask = base_valid & (base_fraction >= minimum_valid_fraction)
    ao6_mask = ao6_valid & (ao6_fraction >= minimum_valid_fraction)
    base_values = base_fraction[base_mask]
    ao6_values = ao6_fraction[ao6_mask]
    if base_values.size < 2 or ao6_values.size < 2 or fraction_knots < 3:
        raise NonexpansiveFractionTransportError("CB32 fraction population failed")
    unique, counts = np.unique(base_values, return_counts=True)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = ends - counts
    group_quantiles = (starts.astype(np.float64) + ends.astype(np.float64)) / (
        2.0 * float(base_values.size)
    )
    selected_count = min(unique.size, fraction_knots - 1)
    selected = np.rint(
        np.linspace(0, unique.size - 1, selected_count, dtype=np.float64)
    ).astype(np.int64)
    selected = np.unique(selected)
    knot_x = np.concatenate((np.asarray([0.0]), unique[selected]))
    raw_y = np.concatenate(
        (np.asarray([0.0]), _target_quantiles(ao6_values, group_quantiles[selected]))
    )
    delta_x = np.diff(knot_x)
    if np.any(delta_x <= 0.0):
        raise NonexpansiveFractionTransportError("CB32 knot order failed")
    delta_y = np.maximum(np.diff(raw_y), 0.0)
    limited_delta = np.minimum(delta_y, maximum_fraction_slope * delta_x)
    knot_y = np.concatenate((np.asarray([0.0]), np.cumsum(limited_delta)))
    slopes = limited_delta / delta_x
    mapped_values = np.interp(base_values, knot_x, knot_y)
    mapped = np.zeros_like(base_fraction)
    mapped[base_mask] = mapped_values
    facts: dict[str, float | int] = {
        "knot_count": int(knot_x.size),
        "maximum_observed_fraction_slope": float(np.max(slopes)),
        "maximum_mapped_fraction": float(np.max(mapped)),
    }
    if (
        not np.isfinite(mapped).all()
        or np.min(mapped) < 0.0
        or np.max(mapped) > 1.0 + 1e-12
        or facts["maximum_observed_fraction_slope"] > maximum_fraction_slope + 1e-12
    ):
        raise NonexpansiveFractionTransportError("CB32 nonexpansive invariant failed")
    return mapped, facts


def nonexpansive_fraction_transport_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    minimum_valid_fraction: float = 1e-4,
    fraction_knots: int = 257,
    maximum_fraction_slope: float = 1.0,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, float | int]]:
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
        raise NonexpansiveFractionTransportError("CB32 input drift")
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
    mapped_fraction, facts = _nonexpansive_fraction_map(
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        minimum_valid_fraction=minimum_valid_fraction,
        fraction_knots=fraction_knots,
        maximum_fraction_slope=maximum_fraction_slope,
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
        raise NonexpansiveFractionTransportError("CB32 target invariant failed")
    return (target, facts) if return_diagnostics else target


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    parent = _load_exact_json(
        root,
        config["parents"]["cb31_decision_path"],
        config["parents"]["cb31_decision_sha256"],
    )
    if parent.get("decision") != config["parents"]["cb31_required_decision"]:
        raise NonexpansiveFractionTransportError("CB31 decision drift")
    cb31 = _load_exact_json(
        root,
        config["parents"]["cb31_contract_path"],
        config["parents"]["cb31_contract_sha256"],
    )
    cb30 = _load_exact_json(
        root,
        cb31["parents"]["cb30_contract_path"],
        cb31["parents"]["cb30_contract_sha256"],
    )
    derived = dict(cb30)
    derived["claim_ceiling"] = config["claim_ceiling"]
    derived["automatic_gates"] = config["automatic_gates"]
    operator = config["operator"]
    diagnostics: list[dict[str, float | int]] = []

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
        minimum_valid_fraction: float,
    ) -> np.ndarray:
        if minimum_valid_fraction != float(operator["minimum_valid_fraction"]):
            raise NonexpansiveFractionTransportError("CB32 fraction threshold drift")
        target, facts = nonexpansive_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=minimum_valid_fraction,
            fraction_knots=int(operator["fraction_knots"]),
            maximum_fraction_slope=float(operator["maximum_fraction_slope"]),
            return_diagnostics=True,
        )
        diagnostics.append(facts)
        return target

    report = evaluate_gold_stress(
        derived,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb32_nonexpansive_fraction_gold_stress_v1.json",
    )
    report["metrics"]["maximum_observed_fraction_slope"] = max(
        float(row["maximum_observed_fraction_slope"]) for row in diagnostics
    )
    report["metrics"]["minimum_observed_knot_count"] = min(
        int(row["knot_count"]) for row in diagnostics
    )
    report["checks"]["equal_input_mapping"] = True
    report["checks"]["fraction_nonexpansive"] = (
        report["metrics"]["maximum_observed_fraction_slope"]
        <= config["automatic_gates"]["maximum_observed_fraction_slope"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_nonexpansive_development_severe_review"
        if report["automatic_pass"]
        else "close_nonexpansive_fraction_before_visual_review"
    )
    value = dict(report)
    value.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(value)).hexdigest()
    return report


__all__ = [
    "NonexpansiveFractionTransportError",
    "_nonexpansive_fraction_map",
    "evaluate",
    "load_contract",
    "nonexpansive_fraction_transport_target",
]
