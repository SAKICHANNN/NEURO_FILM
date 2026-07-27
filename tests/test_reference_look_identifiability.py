from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2w1_reference_look_identifiability_development import (
    _smooth_random_velocity_grids,
    _validate_activation,
    _validate_experiment_partition,
    run_development,
)
from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.reference_look_identifiability import (
    aggregate_reference_descriptors,
    build_direction_strength_bank,
    build_reference_look_bank,
    factorized_reference_descriptor,
    fit_reference_descriptor_ridge,
    raw_rgb_histogram_descriptor,
)


def test_factorized_descriptor_is_finite_and_row_permutation_invariant() -> None:
    rng = np.random.default_rng(28510)
    rgb = rng.uniform(0.0, 1.0, size=(257, 3))
    first = factorized_reference_descriptor(rgb)
    second = factorized_reference_descriptor(rgb[rng.permutation(len(rgb))])
    assert first.shape == (82,)
    assert np.all(np.isfinite(first))
    assert np.allclose(first, second, atol=1e-14, rtol=0.0)


def test_descriptor_distinguishes_channel_relations_at_fixed_luma() -> None:
    rng = np.random.default_rng(28511)
    neutral = rng.uniform(0.1, 0.8, size=(512, 3))
    warm = neutral.copy()
    warm[:, 0] = np.minimum(1.0, warm[:, 0] + 0.08)
    warm[:, 2] = np.maximum(0.0, warm[:, 2] - 0.05)
    distance = np.linalg.norm(
        factorized_reference_descriptor(warm)
        - factorized_reference_descriptor(neutral)
    )
    assert distance > 0.05


def test_multi_reference_aggregation_is_symmetric() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    first = aggregate_reference_descriptors(values)
    second = aggregate_reference_descriptors(values[[2, 0, 3, 1]])
    assert np.array_equal(first, second)
    with pytest.raises(ValueError):
        aggregate_reference_descriptors(values[:1])


def test_raw_histogram_is_normalized_per_channel_and_permutation_exact() -> None:
    rng = np.random.default_rng(28512)
    rgb = rng.uniform(0.0, 1.0, size=(400, 3))
    first = raw_rgb_histogram_descriptor(rgb, bins_per_channel=12)
    second = raw_rgb_histogram_descriptor(
        rgb[rng.permutation(len(rgb))], bins_per_channel=12
    )
    assert np.array_equal(first, second)
    assert np.allclose(first.reshape(3, 12).sum(axis=1), 1.0)


def test_reference_bank_hard_retrieval_is_bounded_and_repeatable() -> None:
    features = np.array(
        [[-2.0, -1.0], [0.0, 0.5], [2.0, 1.5]], dtype=np.float64
    )
    grids = np.zeros((3, 2, 2, 2, 3), dtype=np.float64)
    grids[0, ..., 0] = -0.2
    grids[1, ..., 1] = 0.3
    grids[2, ..., 2] = 0.4
    bank = build_reference_look_bank(
        features, grids, look_ids=("cool", "neutral", "warm")
    )
    query = np.array([[1.9, 1.45], [-1.8, -0.9]], dtype=np.float64)
    first = bank.hard_retrieve(query)
    second = bank.hard_retrieve(query)
    assert first[1] == ("warm", "cool")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[2], second[2])
    assert np.max(np.linalg.norm(first[0], axis=-1)) <= 0.4 + 1e-12


def test_direction_strength_bank_interpolates_without_splitting_direction() -> None:
    strengths = np.array([0.0, 0.5, 1.0])
    prototypes = np.array(
        [
            [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.5], [0.0, 1.0]],
        ]
    )
    grids = np.zeros((2, 2, 2, 2, 3))
    grids[0, ..., 0] = 0.8
    grids[1, ..., 1] = 0.7
    bank = build_direction_strength_bank(
        prototypes, strengths, grids, direction_ids=("red", "green")
    )
    predicted, direction_ids, values, distances = bank.retrieve(
        np.array([[0.58, 0.0], [0.0, 0.72]])
    )
    assert direction_ids == ("red", "green")
    assert np.allclose(values, [0.58, 0.72], atol=1e-12, rtol=0.0)
    assert np.allclose(distances, 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(predicted[0], 0.58 * grids[0])
    assert np.allclose(predicted[1], 0.72 * grids[1])


def test_reference_ridge_predicts_only_norm_bounded_explicit_grids() -> None:
    rng = np.random.default_rng(28513)
    features = rng.normal(size=(30, 8))
    mapping = rng.normal(scale=0.1, size=(8, 24))
    grids = (features @ mapping).reshape(30, 2, 2, 2, 3)
    model = fit_reference_descriptor_ridge(
        features,
        grids,
        alpha=0.25,
        maximum_vector_norm=0.6,
    )
    prediction = model.predict(features[:5])
    assert prediction.shape == (5, 2, 2, 2, 3)
    assert np.max(np.linalg.norm(prediction, axis=-1)) <= 0.6 + 1e-12
    assert np.all(np.isfinite(prediction))


def test_reference_descriptor_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        factorized_reference_descriptor(np.zeros((15, 3)))
    with pytest.raises(ValueError):
        factorized_reference_descriptor(np.full((16, 3), np.nan))
    with pytest.raises(ValueError):
        raw_rgb_histogram_descriptor(np.zeros((16, 3)), bins_per_channel=1)


def _tiny_runner_config() -> dict:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (
            root
            / "configs"
            / "u5_r2w1_reference_look_identifiability_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    config = copy.deepcopy(config)
    look = config["look_generator"]
    look["development_base_direction_count"] = 8
    look["development_fit_direction_ids"] = [0, 1, 2, 3]
    look["development_unseen_direction_ids"] = [4, 5, 6, 7]
    look["strengths"] = [0.5]
    look["development_operator_instance_count"] = 8
    look["base_direction_generators"][0]["direction_count"] = 4
    look["base_direction_generators"][0]["seed"] = 91000
    look["base_direction_generators"][1]["direction_count"] = 4
    look["base_direction_generators"][1]["seed"] = 91012
    look["integration_steps"] = 2
    content = config["content_generator"]
    content["scenes_per_group"] = 1
    content["samples_per_scene"] = 16
    content["fit_content_seed"] = 91001
    content["heldout_content_seed"] = 91002
    content["identity_reference_fit_content_seed"] = 91003
    content["identity_reference_heldout_content_seed"] = 91004
    content["owner_strength_fixture_content_seed"] = 91005
    nuisance = config["reference_nuisance"]
    nuisance["fit_nuisance_seed"] = 91006
    nuisance["heldout_nuisance_seed"] = 91007
    nuisance["identity_fit_nuisance_seed"] = 91008
    nuisance["identity_heldout_nuisance_seed"] = 91009
    nuisance["owner_strength_fixture_nuisance_seed"] = 91010
    config["descriptor"]["ridge_alpha_candidates"] = [1.0]
    config["evaluation"]["uniform_grid_axis_size"] = 3
    paired = config["paired_upper_bound_optimization"]
    paired["seed"] = 91011
    paired["steps"] = 1
    paired["device"] = "cpu"
    paired["dtype"] = "float64"
    return config


def test_w1_activation_requires_repeat_and_conditionally_requires_u1() -> None:
    config = _tiny_runner_config()
    s4 = {
        "decision_branch": "primary_fails_or_does_not_beat_pooled",
        "repeat_report_sha256_equal": True,
    }
    u1 = {
        "decision_branch": "primary_does_not_beat_controls",
        "repeat_report_sha256_equal": True,
    }
    _validate_activation(config, s4, u1)
    with pytest.raises(RuntimeError):
        _validate_activation(config, s4, None)
    _validate_activation(
        config,
        {
            "decision_branch": "shuffled_negative_passes",
            "repeat_report_sha256_equal": True,
        },
        None,
    )


def test_w1_partition_rejects_direction_or_seed_leakage() -> None:
    config = _tiny_runner_config()
    _validate_experiment_partition(config)
    invalid = copy.deepcopy(config)
    invalid["look_generator"]["development_unseen_direction_ids"][0] = 0
    with pytest.raises(ValueError):
        _validate_experiment_partition(invalid)
    invalid = copy.deepcopy(config)
    invalid["reference_nuisance"]["reserved_confirmation_seed"] = 91001
    with pytest.raises(ValueError):
        _validate_experiment_partition(invalid)


def test_smooth_random_operator_family_is_exact_bounded_and_cube_safe() -> None:
    first = _smooth_random_velocity_grids(
        count=3,
        seed=91020,
        axis_size=4,
        smoothing_passes=3,
        minimum_norm=0.7,
        maximum_norm=1.1,
    )
    second = _smooth_random_velocity_grids(
        count=3,
        seed=91020,
        axis_size=4,
        smoothing_passes=3,
        minimum_norm=0.7,
        maximum_norm=1.1,
    )
    assert np.array_equal(first, second)
    maximum_norms = np.max(np.linalg.norm(first, axis=-1), axis=(1, 2, 3))
    assert np.all(maximum_norms >= 0.7)
    assert np.all(maximum_norms <= 1.1)
    points = np.array([[0.0, 0.0, 0.0], [0.2, 0.5, 0.8], [1.0, 1.0, 1.0]])
    for grid in first:
        output = CubeDiffeomorphicColourFlow(
            grid, integration_steps=4
        ).apply(points)
        assert np.all(output >= 0.0)
        assert np.all(output <= 1.0)


def test_w1_tiny_runner_covers_all_information_regimes() -> None:
    config = _tiny_runner_config()
    s4 = {
        "decision_branch": "primary_fails_or_does_not_beat_pooled",
        "repeat_report_sha256_equal": True,
    }
    u1 = {
        "decision_branch": "primary_does_not_beat_controls",
        "repeat_report_sha256_equal": True,
    }
    report = run_development(
        config,
        s4,
        u1,
        config_sha256="test-config",
        software_commit="test-commit",
    )
    repeated = run_development(
        config,
        s4,
        u1,
        config_sha256="test-config",
        software_commit="test-commit",
    )
    assert set(report["regimes"]) == {
        "seen_single_output_only",
        "seen_four_output_only",
        "unseen_single_output_only",
        "unseen_four_output_only",
    }
    assert report["selected_ridge_alpha"] == {
        "single_reference": 1.0,
        "four_reference": 1.0,
    }
    assert report["paired_reference_upper_bound"]["sample_count"] == 4
    assert set(report["owner_53_55_56_strength_path"]) >= {
        "factorized_descriptor_hard_retrieval",
        "factorized_descriptor_ridge_operator",
    }
    assert report["reserved_confirmation_seeds_accessed"] is False
    assert report["decision_branch_before_repeat"]
    assert json.dumps(report, sort_keys=True) == json.dumps(
        repeated, sort_keys=True
    )
