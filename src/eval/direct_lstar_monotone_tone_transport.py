"""CB39 direct frozen-CIELAB-L-star monotone tone plus fixed CB32 chroma."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_photographic import _load_exact_json
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.monotone_source_tone_fraction_transport import _sample_sorted_quantiles
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import (
    apply_safe_base_direction_target,
    evaluate_direction_candidate,
)

SCHEMA = "neuro_film.u5_r2cb39_direct_lstar_monotone_tone_development_contract.v1"
REPORT_SCHEMA = (
    "neuro_film.u5_r2cb39_direct_lstar_monotone_tone_development_report.v1"
)
EXPERIMENT_ID = "U5.R2CB39"


class DirectLstarMonotoneToneTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise DirectLstarMonotoneToneTransportError("CB39 contract structure drift")
    return payload


def monotone_lstar_quantile_map(
    source_lstar: np.ndarray,
    target_lstar: np.ndarray,
    *,
    knot_count: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    source = np.asarray(source_lstar, dtype=np.float64)
    target = np.asarray(target_lstar, dtype=np.float64)
    if (
        source.shape != target.shape
        or source.ndim != 2
        or knot_count < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or np.min(source) < 0.0
        or np.max(source) > 100.0
        or np.min(target) < 0.0
        or np.max(target) > 100.0
    ):
        raise DirectLstarMonotoneToneTransportError("CB39 L-star input drift")
    quantiles = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    source_knots = _sample_sorted_quantiles(
        np.sort(source.reshape(-1), kind="stable"), quantiles
    )
    target_knots = _sample_sorted_quantiles(
        np.sort(target.reshape(-1), kind="stable"), quantiles
    )
    unique_x, first, counts = np.unique(
        source_knots, return_index=True, return_counts=True
    )
    if unique_x.size < 2:
        raise DirectLstarMonotoneToneTransportError("CB39 L-star knots degenerate")
    unique_y = np.maximum.accumulate(target_knots[first + counts - 1])
    mapped = np.interp(source, unique_x, unique_y)
    return mapped, {
        "tone_knot_count": int(unique_x.size),
        "tone_lstar_minimum": float(np.min(mapped)),
        "tone_lstar_maximum": float(np.max(mapped)),
    }


def _lstar_to_neutral_linear(lstar: np.ndarray) -> np.ndarray:
    value = np.asarray(lstar, dtype=np.float64)
    fy = (value + 16.0) / 116.0
    return np.where(
        fy > 0.2068966,
        fy**3,
        (fy - 16.0 / 116.0) / 7.787,
    )


def select_direct_lstar_candidate(
    source_linear: np.ndarray,
    safe_base_linear: np.ndarray,
    chroma_target_linear: np.ndarray,
    *,
    boundary_epsilon: float,
    tone_knot_count: int,
    dose_grid: Sequence[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    source = np.asarray(source_linear)
    base = np.asarray(safe_base_linear)
    target = np.asarray(chroma_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or base.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != base.shape
        or source.shape != target.shape
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise DirectLstarMonotoneToneTransportError("CB39 selector input drift")
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    base_lstar = linear_rgb_to_lab(base, working_space="linear_srgb")[..., 0]
    mapped_lstar, tone_facts = monotone_lstar_quantile_map(
        source_lstar, base_lstar, knot_count=tone_knot_count
    )
    tone_y = _lstar_to_neutral_linear(mapped_lstar)
    lower_target = float(
        np.float32(boundary_epsilon)
        + np.float32(4.0) * np.spacing(np.float32(boundary_epsilon))
    )
    upper_edge = np.float32(1.0 - boundary_epsilon)
    upper_target = float(upper_edge - np.float32(4.0) * np.spacing(upper_edge))
    lower = np.where(np.any(source > boundary_epsilon, axis=-1), lower_target, 0.0)
    upper = np.where(
        np.any(source < 1.0 - boundary_epsilon, axis=-1), upper_target, 1.0
    )
    tone_y = np.clip(tone_y, lower, upper)
    tone_base = np.repeat(tone_y[..., None], 3, axis=-1).astype(np.float32)
    target64 = target.astype(np.float64)
    target_y = np.sum(target64 * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    target_chroma = target64 - target_y[..., None]
    desired = np.asarray(tone_y[..., None] + target_chroma, dtype=np.float32)
    full_candidate, full_scale, _ = apply_safe_base_direction_target(
        source,
        tone_base,
        desired,
        weights=LEGACY_LAB_Y_WEIGHTS,
        boundary_epsilon=boundary_epsilon,
    )
    tone64 = tone_base.astype(np.float64)
    residual = full_candidate.astype(np.float64) - tone64
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    diagnostics: list[dict[str, float]] = []
    for dose in doses:
        candidate = np.asarray(tone64 + float(dose) * residual, dtype=np.float32)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(
            candidate, working_space="linear_srgb"
        )[..., 0]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        diagnostics.append(
            {
                "dose": float(dose),
                "gradient_ratio": float(gradient),
                "lstar_inversion_fraction": float(inversion),
            }
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
        best_gradient = min(diagnostics, key=lambda row: row["gradient_ratio"])
        best_order = min(diagnostics, key=lambda row: row["lstar_inversion_fraction"])
        raise DirectLstarMonotoneToneTransportError(
            "CB39 dose grid has no safe direct-L-star candidate; "
            f"minimum_gradient={best_gradient['gradient_ratio']:.17g}; "
            f"minimum_inversion={best_order['lstar_inversion_fraction']:.17g}"
        )
    selected_y = np.sum(
        selected.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1
    )
    effective_scale = np.asarray(
        full_scale.astype(np.float64) * selected_dose, dtype=np.float32
    )
    facts: dict[str, float | int] = dict(tone_facts)
    facts.update(
        {
            "global_dose": selected_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
        }
    )
    return selected, effective_scale, selected_y - tone_y, facts


def evaluate(config: Mapping[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    decision = _load_exact_json(
        root,
        config["parents"]["cb38_decision_path"],
        config["parents"]["cb38_decision_sha256"],
    )
    if decision.get("decision") != config["parents"]["cb38_required_decision"]:
        raise DirectLstarMonotoneToneTransportError("CB38 decision drift")
    source_decision = _load_exact_json(
        root,
        config["population"]["decision_path"],
        config["population"]["decision_sha256"],
    )
    parent_source = _load_exact_json(
        root,
        source_decision["parent_decision_path"],
        source_decision["parent_decision_sha256"],
    )
    if parent_source.get("decision") != source_decision["parent_required_decision"]:
        raise DirectLstarMonotoneToneTransportError("CB39 source parent drift")
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
    facts: list[dict[str, float | int]] = []
    failure: dict[str, Any] | None = None

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
        nonlocal failure
        del weights
        try:
            candidate, scale, luma_error, row_facts = select_direct_lstar_candidate(
                source_linear,
                safe_base_linear,
                full_target_linear,
                boundary_epsilon=boundary_epsilon,
                tone_knot_count=int(op["tone_knot_count"]),
                dose_grid=op["dose_grid"],
                maximum_gradient_ratio=float(
                    config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
                ),
                maximum_lstar_inversion_fraction=float(
                    config["automatic_gates"][
                        "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
                    ]
                ),
                lstar_order_epsilon=float(op["lstar_order_epsilon"]),
            )
        except DirectLstarMonotoneToneTransportError as exc:
            failure = {"completed_source_count": len(facts), "reason": str(exc)}
            raise
        facts.append(row_facts)
        return candidate, scale, luma_error

    try:
        report = evaluate_direction_candidate(
            config,
            root,
            output_dir,
            target_builder=target_builder,
            candidate_builder=candidate_builder,
            report_schema=REPORT_SCHEMA,
            experiment_id=EXPERIMENT_ID,
            contract_filename="u5_r2cb39_direct_lstar_monotone_tone_development_v1.json",
            blind_seed=int(config["blind_protocol"]["seed"]),
        )
    except DirectLstarMonotoneToneTransportError:
        if failure is None:
            raise
        shutil.rmtree(output_dir)
        diagnostic: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "experiment_id": EXPERIMENT_ID,
            "contract_sha256": hashlib.sha256(
                (
                    root
                    / "configs/u5_r2cb39_direct_lstar_monotone_tone_development_v1.json"
                ).read_bytes()
            ).hexdigest(),
            "automatic_pass": False,
            "checks": {"gradient_and_order_safe_candidate": False},
            "failure": failure,
            "partial_artifacts_removed": True,
            "visual_review_status": "forbidden",
            "decision": "close_direct_lstar_tone_before_complete_render",
            "claim_ceiling": config["claim_ceiling"],
        }
        diagnostic["stable_evidence_id"] = hashlib.sha256(
            canonical_json(diagnostic)
        ).hexdigest()
        return diagnostic
    if len(facts) != len(report["rows"]):
        raise DirectLstarMonotoneToneTransportError("CB39 diagnostic count drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([float(fact["global_dose"]) for fact in facts])
    report["metrics"]["population_median_global_dose"] = float(np.median(doses))
    report["metrics"]["fraction_global_dose_below_0p25"] = float(
        np.mean(doses < 0.25)
    )
    report["metrics"]["minimum_tone_knot_count"] = min(
        int(fact["tone_knot_count"]) for fact in facts
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
    report["checks"]["tone_support"] = (
        report["metrics"]["minimum_tone_knot_count"]
        >= gates["minimum_tone_knot_count"]
    )
    report["automatic_pass"] = all(report["checks"].values())
    if not report["automatic_pass"]:
        report["blind_sheets"] = []
        report["sealed_mappings"] = {}
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = (
        "open_direct_lstar_tone_severe_review_then_blind_development"
        if report["automatic_pass"]
        else "close_direct_lstar_tone_without_rescue"
    )
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "DirectLstarMonotoneToneTransportError",
    "evaluate",
    "load_contract",
    "monotone_lstar_quantile_map",
    "select_direct_lstar_candidate",
]
