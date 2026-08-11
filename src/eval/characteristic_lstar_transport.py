"""CB41 frozen characteristic curve applied directly in actual L-star."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.direct_lstar_monotone_tone_transport import _lstar_to_neutral_linear
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_forward_proxy import load_contract as load_cb6
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_photographic import (
    _compiled_curve,
    _load_exact_json,
)
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import _gradient_p999_ratio
from src.eval.nonexpansive_fraction_transport import (
    nonexpansive_fraction_transport_target,
)
from src.eval.safe_base_ao6_chroma_direction import (
    apply_safe_base_direction_target,
    evaluate_direction_candidate,
)

SCHEMA = "neuro_film.u5_r2cb41_characteristic_lstar_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb41_characteristic_lstar_development_report.v1"
EXPERIMENT_ID = "U5.R2CB41"


class CharacteristicLstarTransportError(RuntimeError):
    pass


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB41 contract structure drift")
    return payload


def select_characteristic_lstar_candidate(
    source_linear: np.ndarray,
    chroma_target_linear: np.ndarray,
    *,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose_grid: Sequence[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_linear)
    target = np.asarray(chroma_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or strength <= 0.0
        or strength > 1.0
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
    ):
        raise CharacteristicLstarTransportError("CB41 selector input drift")

    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    mapped_lstar = 100.0 * _anchored_curve(
        source_lstar.astype(np.float64) / 100.0,
        curve,
        strength=strength,
        epsilon=boundary_epsilon,
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
    diagnostics: list[dict[str, float]] = []
    selected: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    for dose in doses:
        candidate = np.asarray(tone64 + float(dose) * residual, dtype=np.float32)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(candidate, working_space="linear_srgb")[
            ..., 0
        ]
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
        raise CharacteristicLstarTransportError(
            "CB41 dose grid has no safe characteristic-L-star candidate; "
            f"minimum_gradient={best_gradient['gradient_ratio']:.17g}; "
            f"minimum_inversion={best_order['lstar_inversion_fraction']:.17g}"
        )
    selected_y = np.sum(selected.astype(np.float64) * LEGACY_LAB_Y_WEIGHTS, axis=-1)
    effective_scale = np.asarray(
        full_scale.astype(np.float64) * selected_dose, dtype=np.float32
    )
    facts = {
        "characteristic_strength": float(strength),
        "global_dose": selected_dose,
        "selected_gradient_ratio": selected_gradient,
        "selected_lstar_inversion_fraction": selected_inversion,
    }
    return selected, effective_scale, selected_y - tone_y, facts


def evaluate(
    config: Mapping[str, Any],
    root: Path,
    output_dir: Path,
    *,
    selector: Any = select_characteristic_lstar_candidate,
    report_schema: str = REPORT_SCHEMA,
    experiment_id: str = EXPERIMENT_ID,
    contract_filename: str = "u5_r2cb41_characteristic_lstar_development_v1.json",
    prerequisite_path_key: str = "cb40_decision_path",
    prerequisite_sha_key: str = "cb40_decision_sha256",
    prerequisite_required_key: str = "cb40_required_decision",
    diagnostic_decision: str = "close_characteristic_lstar_before_complete_render",
    pass_decision: str = "open_characteristic_lstar_severe_review_then_blind_development",
    close_decision: str = "close_characteristic_lstar_without_rescue",
    selector_receives_safe_base: bool = False,
) -> dict[str, Any]:
    cb40 = _load_exact_json(
        root,
        config["parents"][prerequisite_path_key],
        config["parents"][prerequisite_sha_key],
    )
    if cb40.get("decision") != config["parents"][prerequisite_required_key]:
        raise CharacteristicLstarTransportError("CB41 prerequisite decision drift")
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
    parent_source_decision = parent_source.get("decision")
    if parent_source_decision is None and isinstance(parent_source.get("result"), dict):
        parent_source_decision = parent_source["result"].get("decision")
    if parent_source_decision != source_decision["parent_required_decision"]:
        raise CharacteristicLstarTransportError("CB41 source parent drift")
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
    cb11 = _load_exact_json(
        root,
        config["parents"]["cb11_contract_path"],
        config["parents"]["cb11_contract_sha256"],
    )
    cb6 = load_cb6(root / cb11["parents"]["cb6_contract_path"])
    curve = _compiled_curve(cb6)
    strength = float(cb11["operator"]["nominal_strength"])
    op = config["operator"]
    facts: list[dict[str, float]] = []
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
            selector_kwargs = {
                "curve": curve,
                "strength": strength,
                "boundary_epsilon": boundary_epsilon,
                "dose_grid": op["dose_grid"],
                "maximum_gradient_ratio": float(
                    config["automatic_gates"]["maximum_p999_gradient_ratio_vs_source"]
                ),
                "maximum_lstar_inversion_fraction": float(
                    config["automatic_gates"][
                        "maximum_adjacent_lstar_gradient_sign_inversion_fraction"
                    ]
                ),
                "lstar_order_epsilon": float(op["lstar_order_epsilon"]),
            }
            if selector_receives_safe_base:
                selector_kwargs["safe_base_linear"] = safe_base_linear
            candidate, scale, luma_error, row_facts = selector(
                source_linear,
                full_target_linear,
                **selector_kwargs,
            )
        except CharacteristicLstarTransportError as exc:
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
            report_schema=report_schema,
            experiment_id=experiment_id,
            contract_filename=contract_filename,
            blind_seed=int(config["blind_protocol"]["seed"]),
        )
    except CharacteristicLstarTransportError:
        if failure is None:
            raise
        shutil.rmtree(output_dir)
        diagnostic: dict[str, Any] = {
            "schema": report_schema,
            "experiment_id": experiment_id,
            "contract_sha256": hashlib.sha256(
                (root / "configs" / contract_filename).read_bytes()
            ).hexdigest(),
            "automatic_pass": False,
            "checks": {"gradient_and_order_safe_candidate": False},
            "failure": failure,
            "partial_artifacts_removed": True,
            "visual_review_status": "forbidden",
            "decision": diagnostic_decision,
            "claim_ceiling": config["claim_ceiling"],
        }
        diagnostic["stable_evidence_id"] = hashlib.sha256(
            canonical_json(diagnostic)
        ).hexdigest()
        return diagnostic

    if len(facts) != len(report["rows"]):
        raise CharacteristicLstarTransportError("CB41 diagnostic count drift")
    for row, row_facts in zip(report["rows"], facts, strict=True):
        row.update(row_facts)
    doses = np.asarray([fact["global_dose"] for fact in facts], dtype=np.float64)
    report["metrics"]["population_median_global_dose"] = float(np.median(doses))
    report["metrics"]["fraction_global_dose_below_0p25"] = float(np.mean(doses < 0.25))
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
    report["visual_review_status"] = (
        "pending" if report["automatic_pass"] else "forbidden"
    )
    report["decision"] = pass_decision if report["automatic_pass"] else close_decision
    report.pop("stable_evidence_id", None)
    report["stable_evidence_id"] = hashlib.sha256(canonical_json(report)).hexdigest()
    return report


__all__ = [
    "CharacteristicLstarTransportError",
    "evaluate",
    "load_contract",
    "select_characteristic_lstar_candidate",
]
