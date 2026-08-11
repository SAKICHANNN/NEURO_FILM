"""CB36 source-luminance-rebased execution of the fixed CB32 chroma target."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import (
    SafeBaseAo6DirectionError,
    apply_safe_base_direction_target,
    evaluate_direction_candidate,
)

SCHEMA = "neuro_film.u5_r2cb36_source_luminance_rebased_fraction_development_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb36_source_luminance_rebased_fraction_development_report.v1"
)
EXPERIMENT_ID = "U5.R2CB36"


class SourceLuminanceRebasedFractionTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 contract structure drift"
        )
    return payload


def source_luminance_rebased_target(
    source_linear: np.ndarray,
    chroma_target_linear: np.ndarray,
    *,
    weights: np.ndarray,
) -> np.ndarray:
    """Place the fixed target chroma plane on each source pixel's luminance."""
    source = np.asarray(source_linear)
    target = np.asarray(chroma_target_linear)
    w = np.asarray(weights, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or source.ndim != 3
        or source.shape[-1] != 3
        or w.shape != (3,)
        or abs(float(np.sum(w)) - 1.0) > 1e-12
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 rebase input drift"
        )
    source64 = source.astype(np.float64)
    target64 = target.astype(np.float64)
    source_luma = np.sum(source64 * w, axis=-1)
    target_luma = np.sum(target64 * w, axis=-1)
    target_chroma = target64 - target_luma[..., None]
    rebased = np.asarray(source_luma[..., None] + target_chroma, dtype=np.float32)
    if not np.isfinite(rebased).all():
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 rebased target is nonfinite"
        )
    return rebased


def select_source_luminance_rebased_candidate(
    source_linear: np.ndarray,
    chroma_target_linear: np.ndarray,
    *,
    weights: np.ndarray,
    boundary_epsilon: float,
    dose_grid: Sequence[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 selector input drift"
        )
    target = source_luminance_rebased_target(
        source, chroma_target_linear, weights=weights
    )
    try:
        full_candidate, full_scale, _ = apply_safe_base_direction_target(
            source,
            source,
            target,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
        )
    except SafeBaseAo6DirectionError as exc:
        raise SourceLuminanceRebasedFractionTransportError(str(exc)) from exc
    source64 = source.astype(np.float64)
    residual = full_candidate.astype(np.float64) - source64
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    for dose in doses:
        candidate = np.asarray(source64 + float(dose) * residual, dtype=np.float32)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(
            candidate, working_space="linear_srgb"
        )[..., 0]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        if (
            gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
        ):
            selected = candidate
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            break
    if selected is None:
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 dose grid has no gradient- and order-safe source-luminance rebase"
        )
    w = np.asarray(weights, dtype=np.float64)
    source_luma = np.sum(source64 * w, axis=-1)
    luma_error = np.sum(selected.astype(np.float64) * w, axis=-1) - source_luma
    effective_scale = np.asarray(
        full_scale.astype(np.float64) * selected_dose, dtype=np.float32
    )
    return selected, effective_scale, luma_error, {
        "global_dose": selected_dose,
        "selected_gradient_ratio": selected_gradient,
        "selected_lstar_inversion_fraction": selected_inversion,
    }


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb35_decision_path"],
        config["parents"]["cb35_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb35_required_decision"]:
        raise SourceLuminanceRebasedFractionTransportError("CB35 decision drift")
    cb33 = _load_exact_json(
        root,
        config["parents"]["cb33_contract_path"],
        config["parents"]["cb33_contract_sha256"],
    )
    _load_exact_json(
        root,
        cb33["parents"]["cb32_contract_path"],
        cb33["parents"]["cb32_contract_sha256"],
    )
    op = config["operator"]
    facts: list[dict[str, float]] = []

    def target_builder(
        safe_base_linear: np.ndarray,
        ao6_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> np.ndarray:
        return nonexpansive_fraction_transport_target(
            safe_base_linear,
            ao6_linear,
            weights=weights,
            boundary_epsilon=boundary_epsilon,
            minimum_valid_fraction=float(op["minimum_valid_fraction"]),
            fraction_knots=int(op["fraction_knots"]),
            maximum_fraction_slope=float(op["maximum_fraction_slope"]),
        )

    def candidate_builder(
        source_linear: np.ndarray,
        safe_base_linear: np.ndarray,
        full_target_linear: np.ndarray,
        *,
        weights: np.ndarray,
        boundary_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del safe_base_linear
        candidate, scale, luma_error, row_facts = (
            select_source_luminance_rebased_candidate(
                source_linear,
                full_target_linear,
                weights=weights,
                boundary_epsilon=boundary_epsilon,
                dose_grid=op["dose_grid"],
                maximum_gradient_ratio=float(
                    config["automatic_gates"][
                        "maximum_p999_gradient_ratio_vs_source"
                    ]
                ),
                maximum_lstar_inversion_fraction=float(
                    config["automatic_gates"][
                        "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
                    ]
                ),
                lstar_order_epsilon=float(op["lstar_order_epsilon"]),
            )
        )
        facts.append(row_facts)
        return candidate, scale, luma_error

    report = evaluate_direction_candidate(
        config,
        root,
        output_dir,
        target_builder=target_builder,
        candidate_builder=candidate_builder,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename=(
            "u5_r2cb36_source_luminance_rebased_fraction_development_v1.json"
        ),
        blind_seed=int(config["blind_protocol"]["seed"]),
    )
    if len(facts) != len(report["rows"]):
        raise SourceLuminanceRebasedFractionTransportError(
            "CB36 diagnostic count drift"
        )
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([fact["global_dose"] for fact in facts], dtype=np.float64)
    report["metrics"]["population_median_global_dose"] = float(np.median(doses))
    report["metrics"]["fraction_global_dose_below_0p25"] = float(
        np.mean(doses < 0.25)
    )
    gates = config["automatic_gates"]
    report["checks"]["global_dose"] = (
        report["metrics"]["population_median_global_dose"]
        >= gates["minimum_population_median_global_dose"]
    )
    report["checks"]["global_dose_tail"] = (
        report["metrics"]["fraction_global_dose_below_0p25"]
        <= gates["maximum_fraction_global_dose_below_0p25"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    if not report["automatic_pass"]:
        report["blind_sheets"] = []
        report["sealed_mappings"] = {}
    report["decision"] = (
        "open_source_luminance_rebased_severe_review_then_blind_development"
        if report["automatic_pass"]
        else "close_source_luminance_rebased_transport_without_rescue"
    )
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "SourceLuminanceRebasedFractionTransportError",
    "evaluate",
    "load_contract",
    "select_source_luminance_rebased_candidate",
    "source_luminance_rebased_target",
]
