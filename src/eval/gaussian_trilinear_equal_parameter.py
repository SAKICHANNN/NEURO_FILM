"""Equal-parameter diagnostic for Gaussian versus trilinear local bases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from src.eval.fixed_gaussian_log_odds_capacity import (
    _jacobians,
    _grid,
    validate_contract as validate_as0_contract,
)
from src.roll2film.fixed_gaussian_residual import (
    fit_fixed_neutral_gaussian_log_odds,
    fit_fixed_neutral_trilinear_log_odds,
    fixed_cube_centers,
)
from src.roll2film.gamut_polar_palette import operator_from_config
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


SCHEMA = "neuro_film.u5_r2as1_gaussian_trilinear_equal_parameter.v1"
REPORT_SCHEMA = "neuro_film.u5_r2as1_gaussian_trilinear_report.v1"


class _Operator(Protocol):
    def apply(self, rgb: np.ndarray) -> np.ndarray: ...


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_contract(
    config: dict[str, Any], root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        config.get("schema") != SCHEMA
        or config.get("status") != "post_as0_result_baseline_diagnostic"
        or not config["disclosure"]["as0_confirmation_rows_reused"]
        or not config["disclosure"]["exploratory_trilinear_results_seen_before_freeze"]
        or config["disclosure"]["confirmation_claim_allowed"]
        or config["models"]["shared_fit"]["fitted_scalar_count"] != 24
        or config["training_allowed"]
        or config["image_rendering_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AS1 disclosure or boundary drift")
    parent = config["parent"]
    decision_raw = (root / parent["decision_path"]).read_bytes()
    as0_raw = (root / parent["contract_path"]).read_bytes()
    if (
        _sha256(decision_raw) != parent["decision_sha256"]
        or json.loads(decision_raw)["decision"] != parent["required_decision"]
        or _sha256(as0_raw) != parent["contract_sha256"]
    ):
        raise ValueError("AS1 parent identity drift")
    as0 = json.loads(as0_raw)
    witness = validate_as0_contract(as0, root)
    return as0, witness


def _audit_operator(
    operator: _Operator,
    *,
    cube: np.ndarray,
    jacobian_points: np.ndarray,
    jacobian_step: float,
    neutral: np.ndarray,
    base: _Operator,
) -> dict[str, float]:
    mapped = operator.apply(cube)
    jacobians = _jacobians(operator, jacobian_points, jacobian_step)
    determinants = np.linalg.det(jacobians)
    norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
    return {
        "output_minimum": float(np.min(mapped)),
        "output_maximum": float(np.max(mapped)),
        "minimum_jacobian_determinant": float(np.min(determinants)),
        "nonpositive_jacobian_fraction": float(
            np.mean(determinants <= 0.0)
        ),
        "maximum_jacobian_spectral_norm": float(np.max(norms)),
        "maximum_neutral_axis_residual": float(
            np.max(np.abs(operator.apply(neutral) - base.apply(neutral)))
        ),
    }


def evaluate_gaussian_trilinear_equal_parameter(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    as0, witness_config = validate_contract(config, root)
    fit = as0["fit"]
    audit = as0["audit"]
    shared = config["models"]["shared_fit"]
    gaussian_spec = config["models"]["candidate"]
    development = np.random.default_rng(
        int(fit["development_seed"])
    ).random((int(fit["development_rows"]), 3))
    confirmation = np.random.default_rng(
        int(fit["confirmation_seed"])
    ).random((int(fit["confirmation_rows"]), 3))
    cube = _grid(
        np.linspace(0.0, 1.0, int(audit["cube_axis_size"]))
    )
    jacobian_points = _grid(
        np.linspace(
            float(audit["jacobian_margin"]),
            1.0 - float(audit["jacobian_margin"]),
            int(audit["jacobian_axis_size"]),
        )
    )
    neutral = np.repeat(
        np.linspace(0.0, 1.0, int(audit["neutral_axis_rows"]))[:, None],
        3,
        axis=1,
    )
    centers = fixed_cube_centers(tuple(gaussian_spec["center_levels"]))

    rows: list[dict[str, Any]] = []
    gaussian_wins = 0
    improvements: list[float] = []
    for witness_index, witness_name in enumerate(
        as0["parent_witness"]["witnesses"]
    ):
        truth = operator_from_config(
            witness_config["candidate"],
            witness_config["witnesses"][witness_name],
        )
        development_target = truth.apply(development)
        confirmation_target = truth.apply(confirmation)
        base = fit_positive_film_response_operator(
            development,
            development_target,
            model="one_matrix",
            restart_count=int(fit["base_restart_count"]),
            maximum_function_evaluations=int(
                fit["base_maximum_function_evaluations"]
            ),
            seed=int(fit["base_seed"]) + witness_index,
        ).operator
        gaussian = fit_fixed_neutral_gaussian_log_odds(
            base,
            development,
            development_target,
            centers=centers,
            sigma=float(gaussian_spec["sigma"]),
            ridge=float(shared["ridge"]),
            fit_epsilon=float(shared["fit_epsilon"]),
        )
        trilinear = fit_fixed_neutral_trilinear_log_odds(
            base,
            development,
            development_target,
            ridge=float(shared["ridge"]),
            fit_epsilon=float(shared["fit_epsilon"]),
        )
        predictions = {
            "gaussian": gaussian.apply(confirmation),
            "trilinear": trilinear.apply(confirmation),
        }
        rmse = {
            name: float(
                np.sqrt(np.mean(np.square(value - confirmation_target)))
            )
            for name, value in predictions.items()
        }
        improvement = float(1.0 - rmse["gaussian"] / rmse["trilinear"])
        gaussian_win = rmse["gaussian"] < rmse["trilinear"]
        gaussian_wins += int(gaussian_win)
        improvements.append(improvement)
        rows.append(
            {
                "witness": witness_name,
                "confirmation_rgb_rmse": rmse,
                "gaussian_improvement_over_trilinear": improvement,
                "gaussian_wins": gaussian_win,
                "gaussian": _audit_operator(
                    gaussian,
                    cube=cube,
                    jacobian_points=jacobian_points,
                    jacobian_step=float(audit["jacobian_step"]),
                    neutral=neutral,
                    base=base,
                ),
                "trilinear": _audit_operator(
                    trilinear,
                    cube=cube,
                    jacobian_points=jacobian_points,
                    jacobian_step=float(audit["jacobian_step"]),
                    neutral=neutral,
                    base=base,
                ),
            }
        )

    gates = config["gates"]
    structures = [
        row[name] for row in rows for name in ("gaussian", "trilinear")
    ]
    checks = [
        {
            "name": "gaussian_witness_wins",
            "passed": gaussian_wins
            >= int(gates["minimum_gaussian_witness_wins"]),
        },
        {
            "name": "gaussian_median_improvement",
            "passed": float(np.median(improvements))
            >= float(
                gates[
                    "minimum_gaussian_median_rmse_improvement_over_trilinear"
                ]
            ),
        },
        {
            "name": "both_cube_safe",
            "passed": min(row["output_minimum"] for row in structures)
            >= float(gates["output_minimum_both"])
            and max(row["output_maximum"] for row in structures)
            <= float(gates["output_maximum_both"]),
        },
        {
            "name": "both_jacobian_safe",
            "passed": min(
                row["minimum_jacobian_determinant"] for row in structures
            )
            >= float(gates["minimum_jacobian_determinant_both"])
            and max(
                row["maximum_jacobian_spectral_norm"] for row in structures
            )
            <= float(gates["maximum_jacobian_spectral_norm_both"]),
        },
        {
            "name": "both_neutral_safe",
            "passed": max(
                row["maximum_neutral_axis_residual"] for row in structures
            )
            <= float(gates["maximum_neutral_axis_residual_both"]),
        },
    ]
    automatic_pass = all(bool(row["passed"]) for row in checks)
    stable = {
        "experiment_id": config["experiment_id"],
        "rows": rows,
        "gaussian_witness_wins": gaussian_wins,
        "median_gaussian_improvement_over_trilinear": float(
            np.median(improvements)
        ),
        "checks": checks,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable,
        "stable_evidence_id": _sha256(_canonical_json(stable)),
        "decision": (
            config["decision_if_gaussian_passes"]
            if automatic_pass
            else config["decision_if_gaussian_fails"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_gaussian_trilinear_equal_parameter",
    "validate_contract",
]
