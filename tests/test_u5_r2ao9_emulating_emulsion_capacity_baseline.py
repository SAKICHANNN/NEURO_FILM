from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2ao9_emulating_emulsion_capacity_baseline import (
    CONFIG_SHA256,
    load_config,
)
from src.eval.emulating_emulsion_capacity_baseline import (
    evaluate_capacity_baseline,
)
from src.roll2film.emulating_emulsion_baseline import (
    EMULATING_EMULSION_BASELINE_SCHEMA,
    EmulatingEmulsionEquationOperator,
    fit_emulating_emulsion_equation,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2ao9_emulating_emulsion_capacity_baseline_v1.json"


def _operator() -> EmulatingEmulsionEquationOperator:
    return EmulatingEmulsionEquationOperator(
        capture_matrix=np.asarray(
            [
                [1.05, -0.04, 0.01],
                [0.02, 0.96, 0.02],
                [-0.01, 0.05, 0.96],
            ]
        ),
        scan_matrix=np.asarray(
            [
                [0.95, 0.04, 0.01],
                [0.02, 1.00, -0.02],
                [0.01, 0.03, 0.96],
            ]
        ),
        response_amplitudes=np.asarray([1.10, 1.00, 0.95]),
        response_slopes=np.asarray([2.0, 1.7, 2.2]),
        response_midpoints=np.asarray([0.45, 0.50, 0.55]),
        response_offsets=np.asarray([-0.04, 0.01, 0.03]),
    )


def test_config_is_exact_and_claim_limited() -> None:
    config = load_config(CONFIG, expected_sha256=CONFIG_SHA256)
    assert config["external_method"]["source_code_obtained"] is False
    assert config["external_method"]["dataset_obtained"] is False
    assert config["models"]["candidate_parameterization"][
        "hard_output_clipping"
    ] is False
    assert config["training_allowed"] is False
    assert config["image_rendering_allowed"] is False
    assert "not real digital-film pair evidence" in config["claim_ceiling"]


def test_config_hash_mutation_fails(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"]["maximum_mean_raw_out_of_cube_fraction"] = 1.0
    path = tmp_path / "mutated.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_config(path, expected_sha256=CONFIG_SHA256)


def test_operator_matches_literal_equation_and_does_not_clip() -> None:
    operator = _operator()
    source = np.asarray([[0.0, 0.5, 1.0], [1.2, -0.1, 0.3]])
    exposure = source @ operator.capture_matrix.T
    sigmoid = 1.0 / (
        1.0
        + np.exp(
            -operator.response_slopes[None, :]
            * (exposure - operator.response_midpoints[None, :])
        )
    )
    expected = (
        operator.response_amplitudes[None, :] * sigmoid
        + operator.response_offsets[None, :]
    ) @ operator.scan_matrix.T
    actual = operator.apply(source)
    assert np.array_equal(actual, expected)
    assert operator.to_dict()["schema"] == EMULATING_EMULSION_BASELINE_SCHEMA
    assert operator.to_dict()["hard_output_clipping"] is False


def test_operator_jacobian_matches_finite_difference() -> None:
    operator = _operator()
    point = np.asarray([[0.25, 0.50, 0.75]])
    step = 1e-6
    columns = []
    for channel in range(3):
        delta = np.zeros_like(point)
        delta[0, channel] = step
        columns.append(
            ((operator.apply(point + delta) - operator.apply(point - delta)) / (2 * step))[0]
        )
    finite_difference = np.stack(columns, axis=1)
    assert operator.jacobian_determinants(point)[0] == pytest.approx(
        np.linalg.det(finite_difference), abs=1e-9
    )


def test_fitter_recovers_synthetic_observable_mapping_deterministically() -> None:
    rng = np.random.default_rng(42)
    source = rng.uniform(0.0, 1.0, size=(160, 3))
    target = _operator().apply(source)
    kwargs = {
        "restart_count": 2,
        "maximum_function_evaluations": 3000,
        "seed": 17,
    }
    first = fit_emulating_emulsion_equation(source, target, **kwargs)
    second = fit_emulating_emulsion_equation(source, target, **kwargs)
    assert first.converged
    assert first.development_rgb_rmse < 1e-6
    assert np.array_equal(first.operator.apply(source), second.operator.apply(source))


def test_invalid_operator_and_dataset_fail_closed() -> None:
    payload = _operator().to_dict()
    with pytest.raises(ValueError, match="positive"):
        EmulatingEmulsionEquationOperator(
            capture_matrix=payload["capture_matrix"],
            scan_matrix=payload["scan_matrix"],
            response_amplitudes=[1.0, 0.0, 1.0],
            response_slopes=payload["response_slopes"],
            response_midpoints=payload["response_midpoints"],
            response_offsets=payload["response_offsets"],
        )
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="exact chart and palette"):
        evaluate_capacity_baseline(
            {"velvia_chart": (np.zeros((24, 3)), np.zeros((24, 3)))},
            config,
        )
