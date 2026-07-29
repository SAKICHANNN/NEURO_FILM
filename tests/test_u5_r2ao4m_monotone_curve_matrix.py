from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao4m_film2paint_curve_matrix_capacity import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.film2paint_curve_matrix_capacity import (
    evaluate_curve_matrix_capacity,
)
from src.roll2film.monotone_curve_matrix import (
    MonotoneCurvePositiveMatrixOperator,
    fit_monotone_curve_positive_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao4m_film2paint_curve_matrix_capacity_v1.json"


def _identity() -> MonotoneCurvePositiveMatrixOperator:
    return MonotoneCurvePositiveMatrixOperator(
        curve_segment_weights=np.full((3, 3), 1.0 / 3.0),
        matrix=np.eye(3),
    )


def test_identity_endpoints_jacobian_and_serialization() -> None:
    operator = _identity()
    rng = np.random.default_rng(20250729)
    rgb = np.concatenate(
        (
            np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            rng.random((64, 3)),
        )
    )
    output = operator.apply(rgb)
    assert np.max(np.abs(output - rgb)) < 2e-16
    assert np.max(np.abs(operator.inverse(output) - rgb)) < 2e-16
    assert np.array_equal(operator.jacobian_determinants(rgb), np.ones(len(rgb)))
    replay = MonotoneCurvePositiveMatrixOperator.from_dict(operator.to_dict())
    assert np.array_equal(replay.apply(rgb), output)


def test_nontrivial_operator_is_bounded_invertible_and_positive() -> None:
    operator = MonotoneCurvePositiveMatrixOperator(
        curve_segment_weights=np.asarray(
            [
                [0.20, 0.35, 0.45],
                [0.45, 0.30, 0.25],
                [0.30, 0.45, 0.25],
            ]
        ),
        matrix=np.asarray(
            [
                [0.90, 0.06, 0.04],
                [0.04, 0.91, 0.05],
                [0.03, 0.07, 0.90],
            ]
        ),
    )
    axis = np.linspace(0.0, 1.0, 17)
    rgb = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    output = operator.apply(rgb)
    assert float(np.min(output)) >= 0.0
    assert float(np.max(output)) <= 1.0
    assert float(np.min(operator.jacobian_determinants(rgb))) > 0.0
    assert np.max(np.abs(operator.inverse(output) - rgb)) < 2e-15


def test_fit_recovers_a_synthetic_curve_matrix_mapping() -> None:
    truth = MonotoneCurvePositiveMatrixOperator(
        curve_segment_weights=np.asarray(
            [
                [0.27, 0.31, 0.42],
                [0.39, 0.34, 0.27],
                [0.31, 0.41, 0.28],
            ]
        ),
        matrix=np.asarray(
            [
                [0.91, 0.05, 0.04],
                [0.03, 0.92, 0.05],
                [0.04, 0.04, 0.92],
            ]
        ),
    )
    rng = np.random.default_rng(91)
    source = rng.random((96, 3))
    target = truth.apply(source)
    fit = fit_monotone_curve_positive_matrix(
        source,
        target,
        restart_count=3,
        maximum_function_evaluations=2000,
        seed=17,
    )
    assert fit.converged
    assert fit.development_rgb_rmse < 2e-7
    assert np.max(np.abs(fit.operator.apply(source) - target)) < 2e-6
    held = rng.random((64, 3))
    assert np.sqrt(
        np.mean(np.square(fit.operator.apply(held) - truth.apply(held)))
    ) < 2e-6


def test_contracts_fail_closed() -> None:
    with pytest.raises(ValueError):
        MonotoneCurvePositiveMatrixOperator(
            curve_segment_weights=np.full((3, 3), 1.0 / 3.0),
            matrix=np.asarray(
                [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
            ),
        )
    with pytest.raises(ValueError):
        _identity().apply(np.asarray([[1.1, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        _identity().inverse(np.asarray([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        fit_monotone_curve_positive_matrix(
            np.zeros((11, 3)), np.zeros((11, 3))
        )


def test_frozen_config_is_exact_equal_parameter_and_claim_limited() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["models"]["candidate_parameter_count"] == 12
    assert config["models"]["control_parameter_count"] == 12
    assert config["models"]["candidate_parameterization"][
        "hard_output_clipping"
    ] is False
    assert config["external_method"]["source_code_obtained"] is False
    assert config["external_method"]["dataset_obtained"] is False
    assert config["training_allowed"] is False
    assert config["image_rendering_allowed"] is False


def test_config_hash_mutation_and_incomplete_datasets_fail_closed(
    tmp_path: Path,
) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["minimum_candidate_rgb_rmse_gain_over_bounded_one_matrix"] = -1
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(mutated, expected_sha256=CONFIG_SHA256)
    with pytest.raises(ValueError, match="exact chart and palette"):
        evaluate_curve_matrix_capacity(
            {"velvia_chart": (np.zeros((24, 3)), np.zeros((24, 3)))},
            payload,
        )
