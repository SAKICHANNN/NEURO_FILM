"""Synthetic held-sample capacity audit for a compact Gaussian colour operator."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.roll2film.fixed_gaussian_residual import (
    FixedNeutralGaussianLogOddsOperator,
    fit_fixed_neutral_gaussian_log_odds,
    fixed_cube_centers,
)
from src.roll2film.gamut_polar_palette import operator_from_config
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
)


SCHEMA = "neuro_film.u5_r2as0_fixed_gaussian_log_odds_capacity.v1"
REPORT_SCHEMA = "neuro_film.u5_r2as0_fixed_gaussian_log_odds_report.v1"


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _grid(axis: np.ndarray) -> np.ndarray:
    return np.stack(
        np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1
    ).reshape(-1, 3)


def _jacobians(
    operator: FixedNeutralGaussianLogOddsOperator,
    points: np.ndarray,
    step: float,
) -> np.ndarray:
    columns = []
    for channel in range(3):
        offset = np.zeros(3, dtype=np.float64)
        offset[channel] = step
        columns.append(
            (operator.apply(points + offset) - operator.apply(points - offset))
            / (2.0 * step)
        )
    return np.stack(columns, axis=-1)


def validate_contract(config: dict[str, Any], root: Path) -> dict[str, Any]:
    if config.get("schema") != SCHEMA:
        raise ValueError("unsupported AS0 contract")
    disclosure = config["development_disclosure"]
    candidate = config["candidate"]
    boundary = config["frozen_boundary"]
    if (
        disclosure["status"] != "post_exploratory_representation_contract"
        or candidate["primitive_count"] != 8
        or candidate["center_levels"] != [0.25, 0.75]
        or candidate["hard_output_clipping"]
        or not boundary["ao9_same_71_pair_capacity_rescue_forbidden"]
        or boundary["real_film_or_display_proxy_rows_accessed"]
        or boundary["image_rendering_allowed"]
        or config["training_allowed"]
        or config["production_integration_allowed"]
    ):
        raise ValueError("AS0 frozen boundary drift")
    parent = config["parent_witness"]
    raw = (root / parent["path"]).read_bytes()
    if _sha256(raw) != parent["sha256"]:
        raise ValueError("AS0 analytic witness hash drift")
    payload = json.loads(raw)
    if any(name not in payload["witnesses"] for name in parent["witnesses"]):
        raise ValueError("AS0 analytic witness inventory drift")
    return payload


def evaluate_fixed_gaussian_log_odds_capacity(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    witness_config = validate_contract(config, root)
    fit = config["fit"]
    candidate = config["candidate"]
    audit = config["audit"]
    centers = fixed_cube_centers(tuple(candidate["center_levels"]))
    development = np.random.default_rng(
        int(fit["development_seed"])
    ).random((int(fit["development_rows"]), 3))
    confirmation = np.random.default_rng(
        int(fit["confirmation_seed"])
    ).random((int(fit["confirmation_rows"]), 3))
    originals = (development.copy(), confirmation.copy())

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

    rows: list[dict[str, Any]] = []
    for witness_index, witness_name in enumerate(
        config["parent_witness"]["witnesses"]
    ):
        truth = operator_from_config(
            witness_config["candidate"],
            witness_config["witnesses"][witness_name],
        )
        development_target = truth.apply(development)
        confirmation_target = truth.apply(confirmation)
        base_fit = fit_positive_film_response_operator(
            development,
            development_target,
            model="one_matrix",
            restart_count=int(fit["base_restart_count"]),
            maximum_function_evaluations=int(
                fit["base_maximum_function_evaluations"]
            ),
            seed=int(fit["base_seed"]) + witness_index,
        )
        operator = fit_fixed_neutral_gaussian_log_odds(
            base_fit.operator,
            development,
            development_target,
            centers=centers,
            sigma=float(candidate["sigma"]),
            ridge=float(candidate["ridge"]),
            fit_epsilon=float(candidate["fit_epsilon"]),
        )
        base_prediction = base_fit.operator.apply(confirmation)
        prediction = operator.apply(confirmation)
        base_rmse = float(
            np.sqrt(np.mean(np.square(base_prediction - confirmation_target)))
        )
        candidate_rmse = float(
            np.sqrt(np.mean(np.square(prediction - confirmation_target)))
        )
        mapped_cube = operator.apply(cube)
        jacobians = _jacobians(
            operator, jacobian_points, float(audit["jacobian_step"])
        )
        determinants = np.linalg.det(jacobians)
        norms = np.linalg.svd(jacobians, compute_uv=False)[:, 0]
        replay = FixedNeutralGaussianLogOddsOperator.from_dict(
            json.loads(json.dumps(operator.to_dict(), sort_keys=True))
        )
        replay_error = float(
            np.max(np.abs(replay.apply(confirmation) - prediction))
        )
        neutral_error = float(
            np.max(
                np.abs(
                    operator.apply(neutral)
                    - base_fit.operator.apply(neutral)
                )
            )
        )
        rows.append(
            {
                "witness": witness_name,
                "base_development_rgb_rmse": base_fit.development_rgb_rmse,
                "base_confirmation_rgb_rmse": base_rmse,
                "candidate_confirmation_rgb_rmse": candidate_rmse,
                "candidate_confirmation_rmse_gain": float(
                    1.0 - candidate_rmse / base_rmse
                ),
                "output_minimum": float(np.min(mapped_cube)),
                "output_maximum": float(np.max(mapped_cube)),
                "minimum_jacobian_determinant": float(
                    np.min(determinants)
                ),
                "nonpositive_jacobian_fraction": float(
                    np.mean(determinants <= 0.0)
                ),
                "maximum_jacobian_spectral_norm": float(np.max(norms)),
                "maximum_neutral_axis_residual": neutral_error,
                "serialization_replay_maximum_absolute_error": replay_error,
                "maximum_coefficient_row_norm": float(
                    np.max(np.linalg.norm(operator.coefficients, axis=1))
                ),
                "operator_sha256": _sha256(
                    _canonical_json(operator.to_dict())
                ),
            }
        )

    gains = [row["candidate_confirmation_rmse_gain"] for row in rows]
    gates = config["gates"]
    source_nonmutation = np.array_equal(
        development, originals[0]
    ) and np.array_equal(confirmation, originals[1])
    checks = [
        {
            "name": "each_witness_confirmation_gain",
            "passed": min(gains)
            >= float(gates["minimum_each_witness_confirmation_rmse_gain"]),
        },
        {
            "name": "median_confirmation_gain",
            "passed": float(np.median(gains))
            >= float(gates["minimum_median_confirmation_rmse_gain"]),
        },
        {
            "name": "candidate_confirmation_rmse",
            "passed": max(
                row["candidate_confirmation_rgb_rmse"] for row in rows
            )
            <= float(gates["maximum_candidate_confirmation_rgb_rmse"]),
        },
        {
            "name": "cube_range",
            "passed": min(row["output_minimum"] for row in rows)
            >= float(gates["output_minimum"])
            and max(row["output_maximum"] for row in rows)
            <= float(gates["output_maximum"]),
        },
        {
            "name": "positive_jacobian",
            "passed": min(
                row["minimum_jacobian_determinant"] for row in rows
            )
            >= float(gates["minimum_jacobian_determinant"]),
        },
        {
            "name": "jacobian_norm",
            "passed": max(
                row["maximum_jacobian_spectral_norm"] for row in rows
            )
            <= float(gates["maximum_jacobian_spectral_norm"]),
        },
        {
            "name": "neutral_axis",
            "passed": max(
                row["maximum_neutral_axis_residual"] for row in rows
            )
            <= float(gates["maximum_neutral_axis_residual"]),
        },
        {
            "name": "serialization_replay",
            "passed": max(
                row["serialization_replay_maximum_absolute_error"]
                for row in rows
            )
            <= float(
                gates["serialization_replay_maximum_absolute_error"]
            ),
        },
        {
            "name": "source_nonmutation",
            "passed": source_nonmutation == bool(
                gates["source_nonmutation"]
            ),
        },
    ]
    automatic_pass = all(bool(row["passed"]) for row in checks)
    stable_payload = {
        "experiment_id": config["experiment_id"],
        "rows": rows,
        "checks": checks,
        "automatic_pass": automatic_pass,
    }
    return {
        "schema": REPORT_SCHEMA,
        **stable_payload,
        "stable_evidence_id": _sha256(_canonical_json(stable_payload)),
        "decision": (
            config["frozen_boundary"]["next_if_pass"]
            if automatic_pass
            else config["frozen_boundary"]["next_if_fail"]
        ),
        "claim_ceiling": config["claim_ceiling"],
    }


__all__ = [
    "REPORT_SCHEMA",
    "SCHEMA",
    "evaluate_fixed_gaussian_log_odds_capacity",
    "validate_contract",
]
