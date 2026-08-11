"""CB26 circular hue rotation plus bounded gamut-fraction transport."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from src.eval.characteristic_ao6_factorized import _inputs
from src.eval.fixed_global_policy_confirmation import render_fixed_pair
from src.eval.fujifilm_characteristic_luma_chroma import (
    apply_characteristic_luma_chroma,
)
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.global_chroma_procrustes import _plane_basis
from src.eval.kci_velvia_tone_photographic_stress import _load_rgb
from src.eval.logit_gamut_fraction_transport import _maximum_chroma_magnitude
from src.eval.safe_base_ao6_chroma_direction import evaluate_direction_candidate

SCHEMA = "neuro_film.u5_r2cb26_circular_hue_fraction_transport_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb26_circular_hue_fraction_transport_report.v1"
EXPERIMENT_ID = "U5.R2CB26"


class CircularHueFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CircularHueFractionTransportError("CB26 contract structure drift")
    return payload


def _hue_fraction(
    image: np.ndarray, weights: np.ndarray, basis: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    luma = np.sum(image * weights, axis=-1)
    chroma = image - luma[..., None]
    xy = chroma @ basis
    magnitude = np.linalg.norm(xy, axis=-1)
    valid = magnitude > 1e-12
    unit_xy = np.zeros_like(xy)
    unit_xy[valid] = xy[valid] / magnitude[valid, None]
    unit_rgb = unit_xy @ basis.T
    maximum = _maximum_chroma_magnitude(luma, unit_rgb)
    fraction = np.zeros_like(luma)
    fraction[valid] = magnitude[valid] / maximum[valid]
    return luma, unit_xy, fraction, valid


def _fit_parameters(
    base_unit: np.ndarray,
    ao6_unit: np.ndarray,
    base_fraction: np.ndarray,
    ao6_fraction: np.ndarray,
    base_valid: np.ndarray,
    ao6_valid: np.ndarray,
    *,
    fraction_logit_epsilon: float,
    minimum_valid_fraction: float,
) -> tuple[float, float, float, int]:
    fit_valid = (
        base_valid
        & ao6_valid
        & (base_fraction >= minimum_valid_fraction)
        & (ao6_fraction >= minimum_valid_fraction)
    )
    valid_count = int(np.count_nonzero(fit_valid))
    if valid_count < 2:
        raise CircularHueFractionTransportError("CB26 valid population failed")
    base_fit = base_unit[fit_valid]
    ao6_fit = ao6_unit[fit_valid]
    dot = float(np.sum(base_fit[:, 0] * ao6_fit[:, 0] + base_fit[:, 1] * ao6_fit[:, 1]))
    cross = float(
        np.sum(base_fit[:, 0] * ao6_fit[:, 1] - base_fit[:, 1] * ao6_fit[:, 0])
    )
    angle = float(np.arctan2(cross, dot))
    clipped_base = np.clip(
        base_fraction[fit_valid],
        fraction_logit_epsilon,
        1.0 - fraction_logit_epsilon,
    )
    clipped_ao6 = np.clip(
        ao6_fraction[fit_valid],
        fraction_logit_epsilon,
        1.0 - fraction_logit_epsilon,
    )
    base_logit = np.log(clipped_base / (1.0 - clipped_base))
    ao6_logit = np.log(clipped_ao6 / (1.0 - clipped_ao6))
    centered = base_logit - float(np.mean(base_logit))
    denominator = float(np.sum(centered * centered))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise CircularHueFractionTransportError("CB26 fraction variance failed")
    slope = float(
        np.sum(centered * (ao6_logit - float(np.mean(ao6_logit)))) / denominator
    )
    intercept = float(np.mean(ao6_logit) - slope * np.mean(base_logit))
    return angle, slope, intercept, valid_count


def circular_hue_fraction_transport_target(
    safe_base_linear: np.ndarray,
    ao6_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float | None = None,
    fraction_logit_epsilon: float = 1.0 / 65535.0,
    minimum_valid_fraction: float = 1e-4,
    minimum_fraction_slope: float = 0.25,
    maximum_fraction_slope: float = 4.0,
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
        raise CircularHueFractionTransportError("CB26 input drift")
    base64 = base.astype(np.float64)
    ao664 = ao6.astype(np.float64)
    basis = _plane_basis(w)
    base_luma, base_unit, base_fraction, base_valid = _hue_fraction(base64, w, basis)
    _, ao6_unit, ao6_fraction, ao6_valid = _hue_fraction(ao664, w, basis)
    angle, slope, intercept, _ = _fit_parameters(
        base_unit,
        ao6_unit,
        base_fraction,
        ao6_fraction,
        base_valid,
        ao6_valid,
        fraction_logit_epsilon=fraction_logit_epsilon,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    cosine = float(np.cos(angle))
    sine = float(np.sin(angle))
    rotated_unit = np.empty_like(base_unit)
    rotated_unit[..., 0] = cosine * base_unit[..., 0] - sine * base_unit[..., 1]
    rotated_unit[..., 1] = sine * base_unit[..., 0] + cosine * base_unit[..., 1]

    if (
        not np.isfinite(slope)
        or slope < minimum_fraction_slope
        or slope > maximum_fraction_slope
    ):
        raise CircularHueFractionTransportError("CB26 fraction slope envelope failed")
    all_base_fraction = np.clip(
        base_fraction, fraction_logit_epsilon, 1.0 - fraction_logit_epsilon
    )
    mapped_logit = intercept + slope * np.log(
        all_base_fraction / (1.0 - all_base_fraction)
    )
    mapped_fraction = 1.0 / (1.0 + np.exp(-mapped_logit))
    rotated_rgb = rotated_unit @ basis.T
    maximum = _maximum_chroma_magnitude(base_luma, rotated_rgb)
    target_chroma = np.zeros_like(rotated_rgb)
    target_chroma[base_valid] = (
        mapped_fraction[base_valid, None]
        * maximum[base_valid, None]
        * rotated_rgb[base_valid]
    )
    target = base_luma[..., None] + target_chroma
    target = np.asarray(target, dtype=np.float32)
    if (
        not np.isfinite(target).all()
        or np.min(target) < -1e-7
        or np.max(target) > 1.0 + 1e-7
    ):
        raise CircularHueFractionTransportError("CB26 target invariant failed")
    return target


def evaluate_preflight(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb25_decision_path"],
        config["parents"]["cb25_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb25_required_status"]:
        raise CircularHueFractionTransportError("CB25 decision drift")
    cb11, ao6_config, artifact, source_rows, curve = _inputs(config, root)
    cb11_operator = cb11["operator"]
    operator = config["operator"]
    weights = np.asarray(cb11_operator["luminance_weights"], dtype=np.float64)
    epsilon = float(cb11_operator["boundary_epsilon"])
    rows: list[dict[str, Any]] = []
    for source_row in source_rows:
        source = _load_rgb(
            root / source_row["decoded_path"],
            maximum_long_edge=int(config["population"]["maximum_long_edge"]),
        )
        ao6 = render_fixed_pair(source, artifact, ao6_config["component"])[
            ao6_config["arm_id"]
        ]
        safe_base, _, _ = apply_characteristic_luma_chroma(
            source,
            curve,
            weights=weights,
            strength=float(cb11_operator["nominal_strength"]),
            boundary_epsilon=epsilon,
        )
        basis = _plane_basis(weights)
        _, base_unit, base_fraction, base_valid = _hue_fraction(
            safe_base.astype(np.float64), weights, basis
        )
        _, ao6_unit, ao6_fraction, ao6_valid = _hue_fraction(
            ao6.astype(np.float64), weights, basis
        )
        angle, slope, intercept, valid_count = _fit_parameters(
            base_unit,
            ao6_unit,
            base_fraction,
            ao6_fraction,
            base_valid,
            ao6_valid,
            fraction_logit_epsilon=float(operator["fraction_logit_epsilon"]),
            minimum_valid_fraction=float(operator["minimum_valid_fraction"]),
        )
        passed = bool(
            np.isfinite(slope)
            and float(operator["minimum_fraction_slope"])
            <= slope
            <= float(operator["maximum_fraction_slope"])
        )
        rows.append(
            {
                "id": source_row["id"],
                "make": source_row["make"],
                "valid_pixel_count": valid_count,
                "hue_rotation_radians": angle,
                "fraction_slope": slope,
                "fraction_intercept": intercept,
                "passed_fraction_slope_envelope": passed,
            }
        )
        if not passed:
            break
    failed = bool(rows and not rows[-1]["passed_fraction_slope_envelope"])
    report: dict[str, Any] = {
        "schema": "neuro_film.u5_r2cb26_circular_hue_fraction_transport_preflight_report.v1",
        "experiment_id": EXPERIMENT_ID,
        "rows": rows,
        "source_count_read": len(rows),
        "source_count_unread": int(config["population"]["source_count_exact"])
        - len(rows),
        "failed_source_id": rows[-1]["id"] if failed else None,
        "passed": not failed
        and len(rows) == int(config["population"]["source_count_exact"]),
        "decision": (
            "close_circular_hue_fraction_transport_on_slope_envelope"
            if failed
            else "preflight_fraction_slope_envelope_pass"
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb25_decision_path"],
        config["parents"]["cb25_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb25_required_status"]:
        raise CircularHueFractionTransportError("CB25 decision drift")
    operator = config["operator"]

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return circular_hue_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            fraction_logit_epsilon=float(operator["fraction_logit_epsilon"]),
            minimum_valid_fraction=float(operator["minimum_valid_fraction"]),
            minimum_fraction_slope=float(operator["minimum_fraction_slope"]),
            maximum_fraction_slope=float(operator["maximum_fraction_slope"]),
        )

    return evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb26_circular_hue_fraction_transport_v1.json",
        blind_seed=20260811 + 2600,
    )


__all__ = [
    "CircularHueFractionTransportError",
    "circular_hue_fraction_transport_target",
    "evaluate",
    "evaluate_preflight",
    "load_contract",
]
