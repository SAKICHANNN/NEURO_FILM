from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2u1_hierarchical_colour_coupling_development import (
    _couple_method,
    _decision_branch,
    _validate_activation,
    run_development,
)
from src.roll2film.hierarchical_colour_coupling import (
    fit_paired_cube_diffeomorphic_flow,
    hierarchical_colour_coupling,
    random_colour_coupling,
)

ROOT = Path(__file__).resolve().parents[1]


def test_hierarchical_coupling_is_repeatable_and_non_reusing() -> None:
    rng = np.random.default_rng(19)
    source = rng.uniform(0.0, 1.0, size=(97, 3))
    target = rng.uniform(0.0, 1.0, size=(83, 3))
    first = hierarchical_colour_coupling(
        source, target, maximum_depth=3, seed=71
    )
    second = hierarchical_colour_coupling(
        source, target, maximum_depth=3, seed=71
    )
    assert np.array_equal(first.source, second.source)
    assert np.array_equal(first.target, second.target)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert np.array_equal(first.target_indices, second.target_indices)
    assert len(np.unique(first.source_indices)) == len(first.source_indices)
    assert len(np.unique(first.target_indices)) == len(first.target_indices)
    assert np.array_equal(first.source, source[first.source_indices])
    assert np.array_equal(first.target, target[first.target_indices])


def test_hierarchical_coupling_improves_monotone_shift_pairing() -> None:
    rng = np.random.default_rng(23)
    source = np.clip(rng.normal(0.5, 0.18, size=(512, 3)), 0.0, 1.0)
    target_base = np.clip(
        rng.normal(0.5, 0.18, size=(512, 3)), 0.0, 1.0
    )
    target = np.clip(
        target_base * np.array([0.78, 1.05, 0.92])
        + np.array([0.12, -0.02, 0.04]),
        0.0,
        1.0,
    )
    random_pairs = random_colour_coupling(source, target, seed=29)
    hcc_pairs = hierarchical_colour_coupling(
        source, target, maximum_depth=3, seed=29
    )
    random_error = np.mean((random_pairs.source - random_pairs.target) ** 2)
    hcc_error = np.mean((hcc_pairs.source - hcc_pairs.target) ** 2)
    assert hcc_error < 0.45 * random_error


def test_hierarchical_coupling_falls_back_inside_empty_subtree() -> None:
    source = np.array(
        [
            [0.2, 0.2, 0.2],
            [0.3, 0.3, 0.3],
            [0.7, 0.7, 0.7],
            [0.8, 0.8, 0.8],
        ]
    )
    target = np.array(
        [
            [0.2, 0.8, 0.2],
            [0.3, 0.7, 0.3],
            [0.7, 0.3, 0.7],
            [0.8, 0.2, 0.8],
        ]
    )
    pairs = hierarchical_colour_coupling(
        source, target, maximum_depth=1, seed=27
    )
    assert len(pairs.source) == len(source)
    assert sorted(pairs.source_indices.tolist()) == list(range(len(source)))
    assert sorted(pairs.target_indices.tolist()) == list(range(len(target)))


def test_paired_flow_fit_is_repeatable_bounded_and_improves() -> None:
    rng = np.random.default_rng(31)
    source = rng.uniform(0.08, 0.92, size=(96, 3))
    target = np.clip(
        source * np.array([0.88, 1.04, 0.96])
        + np.array([0.06, -0.01, 0.02]),
        0.0,
        1.0,
    )
    kwargs = dict(
        axis_size=3,
        integration_steps=4,
        coefficient_vector_norm_cap=1.0,
        steps=20,
        learning_rate=0.04,
        coefficient_l2=1e-5,
        velocity_smoothness_l2=1e-4,
        gradient_clip_norm=5.0,
        seed=37,
        device="cpu",
        deterministic_algorithms=True,
        optimization_dtype="float32",
    )
    first, first_trace = fit_paired_cube_diffeomorphic_flow(
        source, target, **kwargs
    )
    second, second_trace = fit_paired_cube_diffeomorphic_flow(
        source, target, **kwargs
    )
    assert np.array_equal(first.velocity_grid, second.velocity_grid)
    assert first_trace == second_trace
    assert first_trace["final_pair_mse"] < first_trace["initial_pair_mse"]
    assert np.max(np.linalg.norm(first.velocity_grid, axis=-1)) <= 1.0
    assert np.all(first.apply(source) >= 0.0)
    assert np.all(first.apply(source) <= 1.0)


def test_coupling_and_fit_inputs_fail_closed() -> None:
    valid = np.full((4, 3), 0.5)
    with pytest.raises(ValueError):
        hierarchical_colour_coupling(
            valid, valid, maximum_depth=-1, seed=1
        )
    with pytest.raises(ValueError):
        random_colour_coupling(valid, np.full((4, 2), 0.5), seed=1)
    with pytest.raises(ValueError):
        fit_paired_cube_diffeomorphic_flow(
            valid,
            valid[:3],
            axis_size=3,
            integration_steps=4,
            coefficient_vector_norm_cap=1.0,
            steps=2,
            learning_rate=0.1,
            coefficient_l2=0.0,
            velocity_smoothness_l2=0.0,
            gradient_clip_norm=1.0,
            seed=1,
            device="cpu",
            deterministic_algorithms=True,
        )


def test_runner_activation_requires_repeated_eligible_s4_branch() -> None:
    config = {
        "status": "pending_u5_r2s4_repeated_adjudication",
        "activation_gate": {
            "allowed_parent_branches": [
                "primary_all_gates_pass",
                "primary_matches_distribution_but_operator_fails",
                "primary_fails_or_does_not_beat_pooled",
            ]
        },
    }
    _validate_activation(
        config,
        {
            "decision_branch": "primary_fails_or_does_not_beat_pooled",
            "repeat_report_sha256_equal": True,
        },
    )
    for decision in (
        {
            "decision_branch": "shuffled_negative_passes",
            "repeat_report_sha256_equal": True,
        },
        {
            "decision_branch": "primary_all_gates_pass",
            "repeat_report_sha256_equal": False,
        },
        {},
    ):
        with pytest.raises(RuntimeError):
            _validate_activation(config, decision)


def test_runner_coupling_methods_keep_control_semantics() -> None:
    rng = np.random.default_rng(41)
    sources = [rng.uniform(0.0, 0.3, size=(24, 3)) for _ in range(4)]
    targets = [rng.uniform(0.7, 1.0, size=(24, 3)) for _ in range(4)]
    coupling = {
        "hierarchical_maximum_depth": 2,
        "shuffled_target_condition_permutation": [1, 3, 0, 2],
    }
    for method_id in (
        "random_correct_condition",
        "pooled_hcc",
        "correct_condition_hcc",
        "shuffled_condition_hcc",
    ):
        left, right, counts = _couple_method(
            {"id": method_id},
            sources,
            targets,
            coupling_config=coupling,
            seed=43,
        )
        assert left.shape == right.shape
        assert left.shape[1] == 3
        assert sum(counts) == len(left)
    _, _, pooled_counts = _couple_method(
        {"id": "pooled_hcc"},
        sources,
        targets,
        coupling_config=coupling,
        seed=43,
    )
    assert len(pooled_counts) == 1


def test_runner_decision_separates_pair_fit_nonidentification() -> None:
    passing = {
        "all_except_repeat": True,
        "pair_fit": True,
        "oracle_median": True,
        "oracle_p90": True,
    }
    assert (
        _decision_branch(passing, shuffled_fails=True)
        == "pending_repeat_primary_all_gates_pass"
    )
    base = dict(passing, all_except_repeat=False)
    assert (
        _decision_branch(base, shuffled_fails=False)
        == "shuffled_negative_passes"
    )
    nonidentified = dict(base, pair_fit=True, oracle_median=True, oracle_p90=False)
    assert (
        _decision_branch(nonidentified, shuffled_fails=True)
        == "operator_fails_despite_pair_fit"
    )
    other = dict(base, pair_fit=False)
    assert (
        _decision_branch(other, shuffled_fails=True)
        == "primary_does_not_beat_controls"
    )


def test_tiny_runner_smoke_keeps_formal_seeds_unaccessed() -> None:
    config = json.loads(
        (
            ROOT
            / "configs"
            / "u5_r2u1_hierarchical_colour_coupling_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    config["style_generator"].update(
        {
            "development_style_count": 2,
            "development_style_seed": 501,
            "reserved_confirmation_style_seed": 599,
        }
    )
    content = config["content_condition_generator"]
    content.update(
        {
            "condition_count": 2,
            "condition_palette_centres": [
                [0.25, 0.25, 0.25],
                [0.70, 0.25, 0.20],
            ],
            "scenes_per_condition_per_domain": 1,
            "samples_per_scene": 12,
            "development_observation_seed_a": 502,
            "development_observation_seed_b": 503,
        }
    )
    config["coupling"].update(
        {
            "development_pair_seed_a": 504,
            "development_pair_seed_b": 505,
            "shuffled_target_condition_permutation": [1, 0],
        }
    )
    config["operator"].update(
        {
            "velocity_grid_axis_size": 3,
            "integration_steps": 2,
            "evaluation_grid_axis_size": 3,
        }
    )
    config["optimization"].update({"steps": 2, "seed": 506})
    report = run_development(
        config,
        {
            "decision_branch": "primary_fails_or_does_not_beat_pooled",
            "repeat_report_sha256_equal": True,
        },
        config_sha256="test-config",
        software_commit="test-commit",
    )
    assert set(report["methods"]) == {
        "random_correct_condition",
        "pooled_hcc",
        "correct_condition_hcc",
        "shuffled_condition_hcc",
    }
    assert report["reserved_confirmation_seed_accessed"] is False
    assert report["config_sha256"] == "test-config"
    assert report["software_commit"] == "test-commit"
