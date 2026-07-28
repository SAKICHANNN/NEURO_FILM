from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.roll2film.positive_film import positive_film_operator_from_config
from src.roll2film.positive_film_fitting import (
    fit_positive_film_response_operator,
    row_stochastic_identity_mixture,
)
from scripts.run_u5_r2an0_paired_positive_film_recovery import (
    CONFIG_SHA256,
    generate_paired_design,
    load_config,
    split_masks,
)


ROOT = Path(__file__).resolve().parents[1]
J0_CONFIG = ROOT / "configs/u5_r2j0_positive_film_response_v1.json"
AN0_CONFIG = ROOT / "configs/u5_r2an0_paired_positive_film_recovery_v1.json"


def _truth(name: str = "cross_bias_like"):
    config = json.loads(J0_CONFIG.read_text(encoding="utf-8"))
    return positive_film_operator_from_config(config["witnesses"][name])


def test_identity_mixture_is_row_stochastic_and_structurally_safe() -> None:
    rng = np.random.default_rng(181)
    for _ in range(100):
        matrix = row_stochastic_identity_mixture(
            rng.uniform(-8.0, 4.0, 6), identity_mixture=0.25
        )
        assert np.min(matrix) >= 0.0
        assert np.max(np.abs(matrix.sum(axis=1) - 1.0)) <= 1e-15
        assert np.linalg.det(matrix) >= 0.25 - 1e-12


def test_two_matrix_fit_recovers_observable_mapping_with_nonunit_truth_amplitude() -> None:
    rng = np.random.default_rng(182)
    source = np.concatenate(
        (
            rng.random((160, 3)),
            np.broadcast_to(np.geomspace(2.0**-12, 1.0, 48)[:, None], (48, 3)),
        )
    )
    truth = _truth()
    target = truth.apply(source)
    fit = fit_positive_film_response_operator(
        source,
        target,
        model="two_matrix",
        restart_count=2,
        maximum_function_evaluations=1200,
        seed=183,
    )
    assert fit.converged
    assert fit.development_rgb_rmse < 5e-4
    assert np.max(np.abs(fit.operator.apply(source) - target)) < 0.005
    assert np.array_equal(fit.operator.maximum_responses, np.ones(3))


def test_invalid_pairs_and_controls_fail_closed() -> None:
    source = np.zeros((12, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="paired"):
        fit_positive_film_response_operator(
            source, source[:11], model="two_matrix", restart_count=1
        )
    with pytest.raises(ValueError, match="paired"):
        fit_positive_film_response_operator(
            source - 0.1, source, model="two_matrix", restart_count=1
        )
    with pytest.raises(ValueError, match="controls"):
        fit_positive_film_response_operator(
            source, source, model="two_matrix", restart_count=0
        )
    with pytest.raises(ValueError, match="controls"):
        fit_positive_film_response_operator(
            source, source, model="two_matrix", loss="invalid"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="identity_mixture"):
        row_stochastic_identity_mixture(np.zeros(6), identity_mixture=0.4)


def test_frozen_design_has_exact_paper_inspired_shape_and_complement_split() -> None:
    config = load_config(AN0_CONFIG, expected_sha256=CONFIG_SHA256)
    source, patch, illuminant, exposure = generate_paired_design(config)
    masks = split_masks(config, patch, illuminant, exposure)
    assert source.shape == (3168, 3)
    assert np.unique(patch).size == 96
    assert np.unique(illuminant).size == 3
    assert np.unique(exposure).size == 11
    assert np.count_nonzero(masks["development"]) == 912
    assert np.count_nonzero(masks["confirmation"]) == 2256
    assert np.count_nonzero(masks["held_patch"]) == 660
    assert np.count_nonzero(masks["held_illuminant"]) == 836
    assert np.count_nonzero(masks["held_exposure"]) == 760
    assert not np.any(masks["development"] & masks["confirmation"])
    assert np.all(masks["development"] | masks["confirmation"])


def test_soft_l1_is_robust_to_sparse_correspondence_outliers() -> None:
    rng = np.random.default_rng(184)
    source = rng.random((360, 3))
    truth = _truth("cyan_shadow_warm_highlight_like")
    clean_target = truth.apply(source)
    contaminated = clean_target + rng.normal(0.0, 0.001, clean_target.shape)
    outlier_rows = rng.choice(source.shape[0], size=24, replace=False)
    contaminated[outlier_rows] += rng.uniform(-0.2, 0.2, (outlier_rows.size, 3))
    contaminated = np.clip(contaminated, 0.0, 1.0)
    linear = fit_positive_film_response_operator(
        source,
        contaminated,
        model="two_matrix",
        restart_count=1,
        maximum_function_evaluations=1200,
        loss="linear",
        seed=185,
    )
    robust = fit_positive_film_response_operator(
        source,
        contaminated,
        model="two_matrix",
        restart_count=1,
        maximum_function_evaluations=1200,
        loss="soft_l1",
        loss_scale=0.005,
        seed=185,
    )
    evaluation = np.random.default_rng(186).random((240, 3))
    clean = truth.apply(evaluation)
    linear_rmse = np.sqrt(np.mean(np.square(linear.operator.apply(evaluation) - clean)))
    robust_rmse = np.sqrt(np.mean(np.square(robust.operator.apply(evaluation) - clean)))
    assert robust.loss == "soft_l1"
    assert robust_rmse < 0.5 * linear_rmse
