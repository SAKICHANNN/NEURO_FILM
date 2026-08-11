"""CB46 direct CIELAB lightness and hue-preserving chroma construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator

from src.color_engine.gamut import compress_chroma_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.eval.characteristic_lstar_transport import (
    CharacteristicLstarTransportError,
)
from src.eval.characteristic_lstar_transport import evaluate as evaluate_characteristic
from src.eval.fujifilm_characteristic_minmax import _anchored_curve
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _new_boundary_fraction,
)

SCHEMA = "neuro_film.u5_r2cb46_direct_lab_lch_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb46_direct_lab_lch_development_report.v1"
EXPERIMENT_ID = "U5.R2CB46"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB46 contract structure drift")
    return payload


def select_direct_lab_lch_candidate(
    source_linear: np.ndarray,
    full_target_linear: np.ndarray,
    *,
    curve: PchipInterpolator,
    strength: float,
    boundary_epsilon: float,
    dose_grid: list[float],
    maximum_gradient_ratio: float,
    maximum_lstar_inversion_fraction: float,
    lstar_order_epsilon: float,
    gamut_iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    source = np.asarray(source_linear)
    target = np.asarray(full_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
        or doses.ndim != 1
        or doses.size < 2
        or doses[0] != 1.0
        or doses[-1] != 0.0
        or np.any(np.diff(doses) >= 0.0)
        or gamut_iterations <= 0
    ):
        raise CharacteristicLstarTransportError("CB46 selector input drift")
    source_lab = linear_rgb_to_lab(source, working_space="linear_srgb")
    target_lab = linear_rgb_to_lab(target, working_space="linear_srgb")
    mapped_lstar = np.asarray(
        100.0
        * _anchored_curve(
            source_lab[..., 0].astype(np.float64) / 100.0,
            curve,
            strength=strength,
            epsilon=boundary_epsilon,
        ),
        dtype=np.float32,
    )
    mapped_lstar = np.clip(mapped_lstar, 0.1, 99.9)
    chroma_delta = target_lab[..., 1:].astype(np.float64) - source_lab[
        ..., 1:
    ].astype(np.float64)
    selected: np.ndarray | None = None
    selected_scale: np.ndarray | None = None
    selected_luma_error: np.ndarray | None = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    selected_boundary = float("inf")
    selected_median_gamut_retention = 0.0
    for dose in doses:
        desired_lab = np.empty_like(source_lab)
        desired_lab[..., 0] = mapped_lstar
        desired_lab[..., 1:] = np.asarray(
            source_lab[..., 1:].astype(np.float64) + float(dose) * chroma_delta,
            dtype=np.float32,
        )
        compressed_lab = compress_chroma_to_working_gamut(
            desired_lab,
            working_space="linear_srgb",
            iterations=gamut_iterations,
            tolerance=2e-6,
        )
        compressed_lab = compressed_lab.copy()
        compressed_lab[..., 1:] *= np.float32(1.0 - boundary_epsilon)
        candidate = lab_to_linear_rgb(
            compressed_lab, working_space="linear_srgb"
        )
        if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
            raise CharacteristicLstarTransportError(
                "CB46 fixed interior chroma guard escaped linear-sRGB"
            )
        actual_lab = linear_rgb_to_lab(candidate, working_space="linear_srgb")
        gradient = _gradient_p999_ratio(source, candidate)
        inversion = _gradient_inversion_fraction(
            source_lab[..., 0], actual_lab[..., 0], epsilon=lstar_order_epsilon
        )
        boundary = _new_boundary_fraction(source, candidate, boundary_epsilon)
        if (
            gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
            and boundary == 0.0
        ):
            desired_chroma = np.linalg.norm(
                desired_lab[..., 1:].astype(np.float64), axis=-1
            )
            actual_chroma = np.linalg.norm(
                compressed_lab[..., 1:].astype(np.float64), axis=-1
            )
            gamut_retention = np.divide(
                actual_chroma,
                desired_chroma,
                out=np.ones_like(actual_chroma),
                where=desired_chroma > 1e-12,
            )
            selected = candidate
            selected_scale = np.asarray(
                float(dose) * gamut_retention, dtype=np.float32
            )
            selected_luma_error = (
                actual_lab[..., 0].astype(np.float64)
                - mapped_lstar.astype(np.float64)
            )
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            selected_boundary = float(boundary)
            selected_median_gamut_retention = float(np.median(gamut_retention))
            break
    if selected is None or selected_scale is None or selected_luma_error is None:
        raise CharacteristicLstarTransportError(
            "CB46 dose grid has no safe direct-Lab candidate"
        )
    return (
        selected,
        selected_scale,
        selected_luma_error,
        {
            "characteristic_strength": float(strength),
            "global_dose": selected_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
            "selected_new_boundary_fraction": selected_boundary,
            "median_gamut_chroma_retention": selected_median_gamut_retention,
        },
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_direct_lab_lch_candidate(
            *args,
            **kwargs,
            gamut_iterations=int(op["gamut_iterations"]),
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb46_direct_lab_lch_development_v1.json",
        prerequisite_path_key="cb45_decision_path",
        prerequisite_sha_key="cb45_decision_sha256",
        prerequisite_required_key="cb45_required_decision",
        diagnostic_decision="close_direct_lab_lch_before_complete_render",
        pass_decision="open_direct_lab_lch_severe_review_then_blind_development",
        close_decision="close_direct_lab_lch_without_rescue",
    )
    if report.get("rows"):
        doses = np.asarray(
            [row["global_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_global_dose"] = float(np.median(doses))
        report["metrics"]["fraction_global_dose_below_0p25"] = float(
            np.mean(doses < 0.25)
        )
        report["metrics"]["population_median_gamut_chroma_retention"] = float(
            np.median(
                [row["median_gamut_chroma_retention"] for row in report["rows"]]
            )
        )
        gates = config["automatic_gates"]
        report["checks"]["global_dose"] = (
            report["metrics"]["population_median_global_dose"]
            >= gates["minimum_population_median_global_dose"]
            and report["metrics"]["fraction_global_dose_below_0p25"]
            <= gates["maximum_fraction_global_dose_below_0p25"]
        )
        report["checks"]["gamut_chroma_retention"] = (
            report["metrics"]["population_median_gamut_chroma_retention"]
            >= gates["minimum_population_median_gamut_chroma_retention"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_direct_lab_lch_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract", "select_direct_lab_lch_candidate"]
