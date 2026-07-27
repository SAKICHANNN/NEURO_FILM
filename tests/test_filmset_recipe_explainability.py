from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_u5_r2w2f0_filmset_recipe_global_explainability import (
    _validate_activation,
    _validate_preflight,
)
from src.roll2film.cube_diffeomorphic_flow import (
    CubeDiffeomorphicColourFlow,
)
from src.roll2film.filmset_recipe_explainability import (
    apply_spatial_indices,
    decide_recipe_branch,
    evaluate_flow_structure,
    evaluate_recipe_global_explainability,
    sample_aligned_spatial_pixels,
    spatial_residual_between_cell_fraction,
)
from src.roll2film.filmset_reference_preflight import (
    partition_filmset_content_ids,
)
from src.roll2film.manifests import FilmSetManifestError


def _config() -> dict:
    root = Path(__file__).resolve().parents[1]
    return json.loads(
        (
            root
            / "configs"
            / "u5_r2w2f0_filmset_recipe_global_explainability_v1.json"
        ).read_text(encoding="utf-8")
    )


def test_activation_requires_exact_repeated_paired_only_w1() -> None:
    config = _config()
    decision = {
        "decision_branch": "paired_upper_bound_only_passes",
        "repeat_report_sha256_equal": True,
        "report_a_sha256": "same",
        "report_b_sha256": "same",
    }
    _validate_activation(config, decision)
    for key, value in (
        ("decision_branch", "practical_reference_method_passes"),
        ("repeat_report_sha256_equal", False),
        ("report_b_sha256", "different"),
    ):
        invalid = copy.deepcopy(decision)
        invalid[key] = value
        with pytest.raises(RuntimeError):
            _validate_activation(config, invalid)


def test_frozen_partition_wrapper_is_exact_and_rejects_other_pools() -> None:
    identifiers = ["c", "a", "b", "d"]
    first = partition_filmset_content_ids(
        identifiers, seed=2026072701, pool="internal"
    )
    second = partition_filmset_content_ids(
        reversed(identifiers), seed=2026072701, pool="internal"
    )
    assert first == second
    assert sorted(first) == sorted(identifiers)
    with pytest.raises(FilmSetManifestError):
        partition_filmset_content_ids(identifiers, seed=1, pool="final")


def test_preflight_is_exact_and_fails_on_wrong_expected_hash() -> None:
    config = _config()
    assert _validate_preflight(config) == config["activation_gate"][
        "required_w2f_preflight_report_sha256"
    ]
    invalid = copy.deepcopy(config)
    invalid["activation_gate"]["required_w2f_preflight_report_sha256"] = "0" * 64
    with pytest.raises(RuntimeError):
        _validate_preflight(invalid)


def test_spatial_sampling_is_exact_disjoint_and_aligned() -> None:
    image = np.arange(96 * 112 * 3, dtype=np.float64).reshape(96, 112, 3)
    image /= image.max()
    first = sample_aligned_spatial_pixels(
        image,
        identity="fixture",
        grid_rows=4,
        grid_columns=4,
        pixels_per_cell_fit=12,
        pixels_per_cell_evaluation=9,
        fit_seed=100,
        evaluation_seed=101,
    )
    second = sample_aligned_spatial_pixels(
        image,
        identity="fixture",
        grid_rows=4,
        grid_columns=4,
        pixels_per_cell_fit=12,
        pixels_per_cell_evaluation=9,
        fit_seed=100,
        evaluation_seed=101,
    )
    assert np.array_equal(first.fit_flat_indices, second.fit_flat_indices)
    assert np.array_equal(
        first.evaluation_flat_indices, second.evaluation_flat_indices
    )
    changed_evaluation_seed = sample_aligned_spatial_pixels(
        image,
        identity="fixture",
        grid_rows=4,
        grid_columns=4,
        pixels_per_cell_fit=12,
        pixels_per_cell_evaluation=9,
        fit_seed=100,
        evaluation_seed=102,
    )
    assert np.array_equal(
        first.fit_flat_indices, changed_evaluation_seed.fit_flat_indices
    )
    assert not np.array_equal(
        first.evaluation_flat_indices,
        changed_evaluation_seed.evaluation_flat_indices,
    )
    assert not (
        set(first.fit_flat_indices) & set(first.evaluation_flat_indices)
    )
    assert len(first.fit_pixels) == 16 * 12
    assert len(first.evaluation_pixels) == 16 * 9
    transformed = np.clip(image * 0.8 + 0.03, 0.0, 1.0)
    assert np.array_equal(
        apply_spatial_indices(transformed, first.fit_flat_indices),
        transformed.reshape(-1, 3)[first.fit_flat_indices],
    )


def test_spatial_residual_detects_cell_specific_bias() -> None:
    rng = np.random.default_rng(29100)
    cells = np.repeat(np.arange(16), 16)
    unstructured = rng.normal(scale=0.02, size=(4, len(cells), 3))
    structured = unstructured.copy()
    structured[:, cells == 0, 0] += 0.2
    assert np.median(
        spatial_residual_between_cell_fraction(structured, cells)
    ) > np.median(spatial_residual_between_cell_fraction(unstructured, cells))


def test_flow_structure_is_cube_safe_and_replay_exact() -> None:
    operator = CubeDiffeomorphicColourFlow.identity(
        axis_size=4, integration_steps=4
    )
    structure = evaluate_flow_structure([operator], coefficient_cap=2.0)
    assert structure["minimum_output"] >= 0.0
    assert structure["maximum_output"] <= 1.0
    assert structure["minimum_jacobian_determinant"] > 0.99
    assert structure["maximum_inverse_error"] == 0.0
    assert structure["maximum_replay_error"] == 0.0


def test_branch_ordering_keeps_structure_and_k1_controls_hard() -> None:
    base = {
        "shared_beats_identity": True,
        "shared_beats_basic": True,
        "oracle_regret": True,
        "grid_dispersion": True,
        "spatial_residual": True,
        "basic_beats_identity": True,
        "shared_incremental_is_basic_only": False,
        "per_pair_beats_identity": True,
        "structure_all": True,
    }
    assert decide_recipe_branch(base) == "global_operator_coherent"
    invalid = dict(base, structure_all=False)
    assert decide_recipe_branch(invalid) == "structure_or_repeat_fails"
    basic = dict(
        base,
        shared_beats_basic=False,
        shared_incremental_is_basic_only=True,
    )
    assert decide_recipe_branch(basic) == "basic_only"
    adaptive = dict(base, oracle_regret=False, basic_beats_identity=False)
    assert decide_recipe_branch(adaptive) == "adaptive_or_spatial_recipe"
    unidentified = dict(
        adaptive,
        shared_beats_identity=False,
        per_pair_beats_identity=False,
        oracle_regret=True,
    )
    assert decide_recipe_branch(unidentified) == "operator_unidentified"


def test_tiny_paired_global_fit_returns_only_explicit_operators() -> None:
    rng = np.random.default_rng(29101)
    dev = rng.uniform(0.05, 0.95, size=(3, 64, 3))
    confirm_fit = rng.uniform(0.05, 0.95, size=(2, 64, 3))
    confirm_eval = rng.uniform(0.05, 0.95, size=(2, 64, 3))
    oracle = CubeDiffeomorphicColourFlow(
        np.full((2, 2, 2, 3), [0.08, -0.04, 0.03]),
        integration_steps=2,
    )
    dev_target = oracle.apply(dev.reshape(-1, 3)).reshape(dev.shape)
    confirm_target_fit = oracle.apply(
        confirm_fit.reshape(-1, 3)
    ).reshape(confirm_fit.shape)
    confirm_target_eval = oracle.apply(
        confirm_eval.reshape(-1, 3)
    ).reshape(confirm_eval.shape)
    settings = {
        "axis_size": 2,
        "integration_steps": 2,
        "coefficient_vector_norm_cap": 2.0,
        "steps": 2,
        "learning_rate": 0.03,
        "coefficient_l2": 0.0001,
        "velocity_smoothness_l2": 0.001,
        "gradient_clip_norm": 5.0,
        "deterministic_algorithms": True,
        "optimization_dtype": "float64",
    }
    report = evaluate_recipe_global_explainability(
        dev,
        dev_target,
        confirm_fit,
        confirm_target_fit,
        confirm_eval,
        confirm_target_eval,
        np.repeat(np.arange(4), 16),
        operator_settings=settings,
        gates=_config()["gates"],
        shared_seed=29102,
        per_pair_seed_base=29103,
        device="cpu",
    )
    assert report["development_images"] == 3
    assert report["confirmatory_images"] == 2
    assert report["metrics"]["shared_operator"]["schema"].endswith(
        "colour_flow.v1"
    )
    assert "decision_branch_before_repeat" in report
    assert np.isfinite(
        report["metrics"]["linear_rgb_rmse"]["shared_o0"]["median"]
    )
