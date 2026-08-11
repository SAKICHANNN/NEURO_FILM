"""CB49 nonnegative, cube-safe global matrix with an exact luminance eigen-row."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize

from src.color_engine.lab import linear_rgb_to_lab
from src.eval.characteristic_lstar_transport import CharacteristicLstarTransportError
from src.eval.characteristic_lstar_transport import evaluate as evaluate_characteristic
from src.eval.exact_lab_y_monotone_tone_transport import LEGACY_LAB_Y_WEIGHTS
from src.eval.fujifilm_characteristic_rgb import _gradient_inversion_fraction
from src.eval.fujifilm_dye_basis_measured_conformance import canonical_json
from src.eval.fujifilm_e6_dye_operator_photographic import (
    _gradient_p999_ratio,
    _new_boundary_fraction,
)

SCHEMA = "neuro_film.u5_r2cb49_nonnegative_y_eigen_matrix_development_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2cb49_nonnegative_y_eigen_matrix_development_report.v1"
EXPERIMENT_ID = "U5.R2CB49"


def load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != SCHEMA or payload.get("experiment_id") != EXPERIMENT_ID:
        raise CharacteristicLstarTransportError("CB49 contract structure drift")
    return payload


def _fit_matrix(
    source: np.ndarray,
    target: np.ndarray,
    *,
    minimum_luminance_slope: float,
    maximum_luminance_slope: float,
) -> tuple[np.ndarray, float, int]:
    x = source.reshape(-1, 3).astype(np.float64)
    y = target.reshape(-1, 3).astype(np.float64)
    source_y = x @ LEGACY_LAB_Y_WEIGHTS
    target_y = y @ LEGACY_LAB_Y_WEIGHTS
    denominator = float(np.dot(source_y, source_y))
    if denominator <= 0.0:
        raise CharacteristicLstarTransportError("CB49 source luminance is degenerate")
    slope = float(
        np.clip(
            np.dot(source_y, target_y) / denominator,
            minimum_luminance_slope,
            maximum_luminance_slope,
        )
    )
    xtx = x.T @ x
    ytx = y.T @ x
    count = float(x.shape[0])

    def objective(flat: np.ndarray) -> float:
        matrix = flat.reshape(3, 3)
        return float(
            (np.trace(matrix @ xtx @ matrix.T) - 2.0 * np.sum(matrix * ytx)) / count
        )

    def gradient(flat: np.ndarray) -> np.ndarray:
        matrix = flat.reshape(3, 3)
        return np.asarray(2.0 * (matrix @ xtx - ytx) / count).reshape(-1)

    equality_jacobian = np.zeros((3, 9), dtype=np.float64)
    for column in range(3):
        for row in range(3):
            equality_jacobian[column, 3 * row + column] = LEGACY_LAB_Y_WEIGHTS[row]
    equality_target = slope * LEGACY_LAB_Y_WEIGHTS

    def equality(flat: np.ndarray) -> np.ndarray:
        return equality_jacobian @ flat - equality_target

    row_sum_jacobian = np.zeros((3, 9), dtype=np.float64)
    for row in range(3):
        row_sum_jacobian[row, 3 * row : 3 * row + 3] = -1.0

    result = minimize(
        objective,
        (slope * np.eye(3, dtype=np.float64)).reshape(-1),
        jac=gradient,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 9,
        constraints=[
            {"type": "eq", "fun": equality, "jac": lambda _: equality_jacobian},
            {
                "type": "ineq",
                "fun": lambda flat: 1.0 - flat.reshape(3, 3).sum(axis=1),
                "jac": lambda _: row_sum_jacobian,
            },
        ],
        options={"ftol": 1e-12, "maxiter": 500, "disp": False},
    )
    matrix = np.asarray(result.x, dtype=np.float64).reshape(3, 3)
    if (
        not result.success
        or np.min(matrix) < -1e-10
        or np.max(matrix.sum(axis=1)) > 1.0 + 1e-10
        or np.max(np.abs(LEGACY_LAB_Y_WEIGHTS @ matrix - equality_target)) > 1e-9
    ):
        raise CharacteristicLstarTransportError(
            f"CB49 constrained matrix fit failed: {result.message}"
        )
    return matrix, slope, int(result.nit)


def select_nonnegative_y_eigen_candidate(
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
    minimum_luminance_slope: float,
    maximum_luminance_slope: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    del curve, strength
    source = np.asarray(source_linear)
    target = np.asarray(full_target_linear)
    doses = np.asarray(dose_grid, dtype=np.float64)
    if (
        source.dtype != np.float32
        or target.dtype != np.float32
        or source.shape != target.shape
    ):
        raise CharacteristicLstarTransportError("CB49 selector input drift")
    matrix, slope, iterations = _fit_matrix(
        source,
        target,
        minimum_luminance_slope=minimum_luminance_slope,
        maximum_luminance_slope=maximum_luminance_slope,
    )
    source64 = source.astype(np.float64)
    fitted = source64 @ matrix.T
    source_lstar = linear_rgb_to_lab(source, working_space="linear_srgb")[..., 0]
    selected = None
    selected_luma_error = None
    selected_dose = -1.0
    selected_gradient = float("inf")
    selected_inversion = float("inf")
    selected_boundary = float("inf")
    for dose in doses:
        candidate = np.asarray(
            source64 + float(dose) * (fitted - source64), dtype=np.float32
        )
        if np.min(candidate) < 0.0 or np.max(candidate) > 1.0:
            continue
        boundary = _new_boundary_fraction(source, candidate, boundary_epsilon)
        gradient = _gradient_p999_ratio(source, candidate)
        candidate_lstar = linear_rgb_to_lab(candidate, working_space="linear_srgb")[
            ..., 0
        ]
        inversion = _gradient_inversion_fraction(
            source_lstar, candidate_lstar, epsilon=lstar_order_epsilon
        )
        if (
            boundary == 0.0
            and gradient <= maximum_gradient_ratio
            and inversion <= maximum_lstar_inversion_fraction
        ):
            expected_y = ((1.0 - float(dose)) + float(dose) * slope) * (
                source64 @ LEGACY_LAB_Y_WEIGHTS
            )
            selected = candidate
            selected_luma_error = (
                candidate.astype(np.float64) @ LEGACY_LAB_Y_WEIGHTS - expected_y
            )
            selected_dose = float(dose)
            selected_gradient = float(gradient)
            selected_inversion = float(inversion)
            selected_boundary = float(boundary)
            break
    if selected is None or selected_luma_error is None:
        raise CharacteristicLstarTransportError(
            "CB49 dose grid has no safe constrained matrix"
        )
    return (
        selected,
        np.full(source.shape[:-1], selected_dose, dtype=np.float32),
        selected_luma_error,
        {
            "fitted_luminance_slope": slope,
            "global_dose": selected_dose,
            "selected_gradient_ratio": selected_gradient,
            "selected_lstar_inversion_fraction": selected_inversion,
            "selected_new_boundary_fraction": selected_boundary,
            "minimum_matrix_coefficient": float(np.min(matrix)),
            "maximum_matrix_row_sum": float(np.max(matrix.sum(axis=1))),
            "maximum_luminance_eigen_error": float(
                np.max(
                    np.abs(LEGACY_LAB_Y_WEIGHTS @ matrix - slope * LEGACY_LAB_Y_WEIGHTS)
                )
            ),
            "optimizer_iterations": float(iterations),
            "matrix_fit_rmse": float(
                np.sqrt(np.mean((fitted - target.astype(np.float64)) ** 2))
            ),
        },
    )


def evaluate(config: dict[str, Any], root: Path, output_dir: Path) -> dict[str, Any]:
    op = config["operator"]

    def selector(*args: Any, **kwargs: Any) -> Any:
        return select_nonnegative_y_eigen_candidate(
            *args,
            **kwargs,
            minimum_luminance_slope=float(op["minimum_luminance_slope"]),
            maximum_luminance_slope=float(op["maximum_luminance_slope"]),
        )

    report = evaluate_characteristic(
        config,
        root,
        output_dir,
        selector=selector,
        report_schema=REPORT_SCHEMA,
        experiment_id=EXPERIMENT_ID,
        contract_filename="u5_r2cb49_nonnegative_y_eigen_matrix_development_v1.json",
        prerequisite_path_key="cb48_decision_path",
        prerequisite_sha_key="cb48_decision_sha256",
        prerequisite_required_key="cb48_required_decision",
        diagnostic_decision="close_nonnegative_y_eigen_before_complete_render",
        pass_decision="open_nonnegative_y_eigen_severe_review_then_blind_development",
        close_decision="close_nonnegative_y_eigen_without_rescue",
    )
    if report.get("rows"):
        doses = np.asarray(
            [row["global_dose"] for row in report["rows"]], dtype=np.float64
        )
        report["metrics"]["population_median_global_dose"] = float(np.median(doses))
        report["metrics"]["fraction_global_dose_below_0p25"] = float(
            np.mean(doses < 0.25)
        )
        report["metrics"]["minimum_matrix_coefficient"] = float(
            min(row["minimum_matrix_coefficient"] for row in report["rows"])
        )
        report["metrics"]["maximum_matrix_row_sum"] = float(
            max(row["maximum_matrix_row_sum"] for row in report["rows"])
        )
        gates = config["automatic_gates"]
        report["checks"]["matrix_constraints"] = (
            report["metrics"]["minimum_matrix_coefficient"] >= -1e-10
            and report["metrics"]["maximum_matrix_row_sum"] <= 1.0 + 1e-10
        )
        report["checks"]["global_dose"] = (
            report["metrics"]["population_median_global_dose"]
            >= gates["minimum_population_median_global_dose"]
            and report["metrics"]["fraction_global_dose_below_0p25"]
            <= gates["maximum_fraction_global_dose_below_0p25"]
        )
        report["automatic_pass"] = all(report["checks"].values())
        if not report["automatic_pass"]:
            report["blind_sheets"] = []
            report["sealed_mappings"] = {}
            report["visual_review_status"] = "forbidden"
            report["decision"] = "close_nonnegative_y_eigen_without_rescue"
        report.pop("stable_evidence_id", None)
        report["stable_evidence_id"] = hashlib.sha256(
            canonical_json(report)
        ).hexdigest()
    return report


__all__ = ["evaluate", "load_contract", "select_nonnegative_y_eigen_candidate"]
