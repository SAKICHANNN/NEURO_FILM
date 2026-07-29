"""U5.R2AR0 bounded cube-flow challenge on two Velvia display proxies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.real_film.combined_velvia_operator import load_combined_velvia_pairs
from src.real_film.velvia_chart_explainability import (
    _fit_affine,
    _fit_record,
    _metrics,
)
from src.roll2film.cube_diffeomorphic_flow import finite_difference_jacobians
from src.roll2film.hierarchical_colour_coupling import (
    fit_paired_cube_diffeomorphic_flow,
)
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


SCHEMA = "neuro_film.u5_r2ar0_velvia_proxy_cube_flow_contract.v1"
REPORT_SCHEMA = "neuro_film.u5_r2ar0_velvia_proxy_cube_flow_report.v1"


class VelviaProxyCubeFlowError(ValueError):
    """Raised when frozen inputs or AR0 semantics drift."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_sha256(payload: Any) -> str:
    return _sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def _read_exact_json(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = root / str(binding["path"])
    raw = path.read_bytes()
    if _sha256(raw) != binding["sha256"]:
        raise VelviaProxyCubeFlowError(f"input hash drift: {binding['path']}")
    return json.loads(raw)


def validate_contract(config: dict[str, Any], root: Path) -> None:
    """Validate frozen lineage and the one fixed post-exploratory candidate."""

    if config.get("schema") != SCHEMA:
        raise VelviaProxyCubeFlowError("unsupported AR0 contract")
    parent = _read_exact_json(root, config["parents"]["cross_domain_decision"])
    if parent.get("decision") != config["parents"]["cross_domain_decision"][
        "required_decision"
    ]:
        raise VelviaProxyCubeFlowError("AO4C decision drift")
    operator = _read_exact_json(root, config["parents"]["combined_operator"])
    required = config["parents"]["combined_operator"]["required_operator"]
    if required not in operator.get("witnesses", {}):
        raise VelviaProxyCubeFlowError("AO5 operator drift")
    for name in ("chart_pairs", "palette_pairs"):
        _read_exact_json(root, config["inputs"][name])
    candidate = config["candidate"]
    expected = {
        "family": "stationary_boundary_vanishing_cube_diffeomorphic_flow",
        "axis_size": 3,
        "integration_steps": 8,
        "coefficient_vector_norm_cap": 2.0,
        "optimization_steps": 150,
        "learning_rate": 0.05,
        "coefficient_l2": 0.001,
        "velocity_smoothness_l2": 0.01,
        "gradient_clip_norm": 2.0,
        "seed": 27101,
        "device": "cpu",
        "optimization_dtype": "float64",
        "deterministic_algorithms": True,
    }
    if candidate != expected:
        raise VelviaProxyCubeFlowError("AR0 candidate drift")
    directions = config["evaluation"]["directions"]
    if directions != [
        {"fit": "velvia_chart", "score": "velvia_palette"},
        {"fit": "velvia_palette", "score": "velvia_chart"},
        {"fit": "combined", "score": "combined"},
    ]:
        raise VelviaProxyCubeFlowError("AR0 direction drift")
    disclosure = config["development_disclosure"]
    if not all(bool(value) for value in disclosure.values()):
        raise VelviaProxyCubeFlowError("AR0 exploration disclosure drift")


def _load_pairs(
    config: dict[str, Any], root: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    inputs = config["inputs"]
    loader_config = {
        "inputs": {
            "chart_paired_u8_sha256": inputs["chart_pairs"][
                "paired_u8_sha256"
            ],
            "palette_asset_sha256": inputs["palette_pairs"]["asset_sha256"],
            "velvia_palette_paired_u8_sha256": inputs["palette_pairs"][
                "paired_u8_sha256"
            ],
            "chart_rows": inputs["chart_pairs"]["rows"],
            "palette_rows": inputs["palette_pairs"]["rows"],
        }
    }
    return load_combined_velvia_pairs(
        root / inputs["chart_pairs"]["path"],
        root / inputs["palette_pairs"]["path"],
        loader_config,
    )


def _fit_one_matrix(
    source: np.ndarray, target: np.ndarray, fit: dict[str, Any]
):
    return fit_positive_film_response_operator(
        source,
        target,
        model="one_matrix",
        identity_mixture=float(fit["identity_mixture"]),
        restart_count=int(fit["restart_count"]),
        maximum_function_evaluations=int(
            fit["maximum_function_evaluations"]
        ),
        function_tolerance=float(fit["function_tolerance"]),
        parameter_tolerance=float(fit["parameter_tolerance"]),
        gradient_tolerance=float(fit["gradient_tolerance"]),
        loss=str(fit["loss"]),
        loss_scale=float(fit["loss_scale"]),
        seed=int(fit["seed"]),
    )


def _structural_metrics(operator, config: dict[str, Any]) -> dict[str, Any]:
    evaluation = config["evaluation"]
    jacobian_axis = np.linspace(
        0.05, 0.95, int(evaluation["jacobian_axis_count"])
    )
    jacobian_points = np.stack(
        np.meshgrid(jacobian_axis, jacobian_axis, jacobian_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    jacobians = finite_difference_jacobians(
        operator,
        jacobian_points,
        step=float(evaluation["jacobian_step"]),
    )
    determinants = np.linalg.det(jacobians)
    norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    inverse_axis = np.linspace(
        0.0, 1.0, int(evaluation["inverse_axis_count"])
    )
    inverse_points = np.stack(
        np.meshgrid(inverse_axis, inverse_axis, inverse_axis, indexing="ij"),
        axis=-1,
    ).reshape(-1, 3)
    mapped = operator.apply(inverse_points)
    restored = operator.inverse(mapped)
    return {
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "maximum_jacobian_spectral_norm": float(np.max(norms)),
        "maximum_inverse_roundtrip_error": float(
            np.max(np.abs(restored - inverse_points))
        ),
        "raw_out_of_cube_fraction": float(
            np.mean(np.any((mapped < 0.0) | (mapped > 1.0), axis=1))
        ),
    }


def evaluate_velvia_proxy_cube_flow(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    """Fit and score one fixed flow/control set in both proxy directions."""

    validate_contract(config, root)
    datasets = _load_pairs(config, root)
    candidate = config["candidate"]
    one_matrix_fit = config["controls"]["one_matrix_fit"]
    rows: list[dict[str, Any]] = []
    for direction in config["evaluation"]["directions"]:
        fit_name = direction["fit"]
        score_name = direction["score"]
        fit_source, fit_target = datasets[fit_name]
        score_source, score_target = datasets[score_name]
        flow, trace = fit_paired_cube_diffeomorphic_flow(
            fit_source,
            fit_target,
            axis_size=int(candidate["axis_size"]),
            integration_steps=int(candidate["integration_steps"]),
            coefficient_vector_norm_cap=float(
                candidate["coefficient_vector_norm_cap"]
            ),
            steps=int(candidate["optimization_steps"]),
            learning_rate=float(candidate["learning_rate"]),
            coefficient_l2=float(candidate["coefficient_l2"]),
            velocity_smoothness_l2=float(
                candidate["velocity_smoothness_l2"]
            ),
            gradient_clip_norm=float(candidate["gradient_clip_norm"]),
            seed=int(candidate["seed"]),
            device=str(candidate["device"]),
            deterministic_algorithms=bool(
                candidate["deterministic_algorithms"]
            ),
            optimization_dtype=str(candidate["optimization_dtype"]),
        )
        one_matrix = _fit_one_matrix(
            fit_source, fit_target, one_matrix_fit
        )
        _, affine = _fit_affine(fit_source, fit_target, per_channel=False)
        predictions = {
            "identity": score_source,
            "full_affine": score_source @ np.asarray(affine["matrix"]).T
            + np.asarray(affine["bias"]),
            "one_matrix": one_matrix.operator.apply(score_source),
            "cube_flow": flow.apply(score_source),
        }
        metrics = {
            name: _metrics(prediction, score_target)
            for name, prediction in predictions.items()
        }
        flow_rmse = metrics["cube_flow"]["rgb_rmse"]
        row = {
            "fit_domain": fit_name,
            "score_domain": score_name,
            "fit_rows": int(len(fit_source)),
            "score_rows": int(len(score_source)),
            "metrics": metrics,
            "flow_gain_over_one_matrix": float(
                1.0 - flow_rmse / metrics["one_matrix"]["rgb_rmse"]
            ),
            "flow_gain_over_full_affine": float(
                1.0 - flow_rmse / metrics["full_affine"]["rgb_rmse"]
            ),
            "flow_fit_trace": trace,
            "flow_operator": flow.to_dict(),
            "flow_operator_sha256": _canonical_sha256(flow.to_dict()),
            "one_matrix_fit": _fit_record(one_matrix),
            "structural": _structural_metrics(flow, config),
        }
        rows.append(row)

    gates = config["automatic_gates"]
    cross = rows[:2]
    combined = rows[2]
    structural_rows = [row["structural"] for row in rows]
    checks = {
        "cross_direction_gain_over_one_matrix": all(
            row["flow_gain_over_one_matrix"]
            >= float(
                gates[
                    "minimum_cross_direction_flow_gain_over_one_matrix_each"
                ]
            )
            for row in cross
        ),
        "combined_gain_over_one_matrix": combined[
            "flow_gain_over_one_matrix"
        ]
        >= float(gates["minimum_combined_flow_gain_over_one_matrix"]),
        "cross_direction_gain_over_full_affine": all(
            row["flow_gain_over_full_affine"]
            >= float(
                gates[
                    "minimum_cross_direction_flow_gain_over_full_affine_each"
                ]
            )
            for row in cross
        ),
        "positive_jacobian": all(
            row["minimum_jacobian_determinant"]
            >= float(gates["minimum_jacobian_determinant"])
            for row in structural_rows
        ),
        "bounded_jacobian_norm": all(
            row["maximum_jacobian_spectral_norm"]
            <= float(gates["maximum_jacobian_spectral_norm"])
            for row in structural_rows
        ),
        "inverse_roundtrip": all(
            row["maximum_inverse_roundtrip_error"]
            <= float(gates["maximum_inverse_roundtrip_error"])
            for row in structural_rows
        ),
        "in_cube": all(
            row["raw_out_of_cube_fraction"]
            <= float(gates["maximum_raw_out_of_cube_fraction"])
            for row in structural_rows
        ),
    }
    automatic_pass = all(checks.values())
    report = {
        "schema": REPORT_SCHEMA,
        "node": config["node"],
        "development_disclosure": config["development_disclosure"],
        "directions": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
        "decision": (
            "retain_development_flow_challenger"
            if automatic_pass
            else "close_flow_on_display_proxy_family"
        ),
        "photographic_render_allowed": bool(automatic_pass),
        "claim_ceiling": config["claim_ceiling"],
    }
    report["stable_evidence_id"] = _canonical_sha256(report)
    return report


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "VelviaProxyCubeFlowError",
    "evaluate_velvia_proxy_cube_flow",
    "validate_contract",
]
