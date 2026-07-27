from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.roll2film.cube_diffeomorphic_flow import CubeDiffeomorphicColourFlow
from src.roll2film.group_invariant_reference_operator import (
    HierarchicalReferenceOperatorPredictor,
    apply_velocity_grids_torch,
    canonicalize_reference_groups,
    generate_episode_population,
    radial_tanh_bound,
    vicreg_terms,
)


ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    config = json.loads(
        (
            ROOT
            / "configs/u5_r2ah1_group_invariant_reference_operator_development_v1.json"
        ).read_text(encoding="utf-8")
    )
    operator = config["operator_population"]
    operator["training_palette_score_direction_count"] = 1
    operator["training_smooth_random_direction_count"] = 1
    operator["development_palette_score_direction_count"] = 1
    operator["development_smooth_random_direction_count"] = 1
    operator["strengths"] = [0.5]
    content = config["content_population"]
    content["samples_per_reference"] = 16
    content["training_groups_per_operator_instance"] = 2
    content["development_groups_per_operator_instance"] = 2
    return config


def test_canonicalization_is_exactly_permutation_invariant() -> None:
    rng = np.random.default_rng(30131)
    groups = rng.uniform(size=(3, 4, 19, 3)).astype(np.float32)
    canonical = canonicalize_reference_groups(groups)
    permuted = groups[:, [2, 0, 3, 1]][:, :, ::-1]
    assert np.array_equal(canonicalize_reference_groups(permuted), canonical)


def test_radial_head_is_bounded_and_model_has_expected_shapes() -> None:
    raw = torch.randn(7, 4, 4, 4, 3)
    bounded = radial_tanh_bound(raw, maximum_vector_norm=1.7)
    assert torch.max(torch.linalg.vector_norm(bounded, dim=-1)) < 1.7
    model = HierarchicalReferenceOperatorPredictor()
    references = torch.rand(2, 4, 32, 3)
    result = model(references)
    assert result["group_grid"].shape == (2, 4, 4, 4, 3)
    assert result["reference_grids"].shape == (2, 4, 4, 4, 4, 3)
    assert sum(parameter.numel() for parameter in model.parameters()) < 150000


def test_batched_torch_renderer_matches_numpy_o0() -> None:
    rng = np.random.default_rng(30132)
    points = rng.uniform(size=(2, 23, 3)).astype(np.float32)
    grids = rng.normal(scale=0.2, size=(2, 4, 4, 4, 3)).astype(np.float32)
    actual = apply_velocity_grids_torch(
        torch.from_numpy(points),
        torch.from_numpy(grids),
        integration_steps=8,
    ).detach().numpy()
    expected = np.stack(
        [
            CubeDiffeomorphicColourFlow(
                grids[index].astype(np.float64), integration_steps=8
            )
            .apply(points[index].astype(np.float64))
            .astype(np.float32)
            for index in range(2)
        ]
    )
    assert np.max(np.abs(actual - expected)) < 2e-6


def test_vicreg_terms_are_finite_and_reject_invalid_batch() -> None:
    variance, covariance = vicreg_terms(torch.randn(8, 16))
    assert torch.isfinite(variance)
    assert torch.isfinite(covariance)
    assert variance >= 0.0
    assert covariance >= 0.0


def test_episode_population_is_repeat_exact_and_crossed() -> None:
    config = _small_config()
    first = generate_episode_population(config, split="training")
    second = generate_episode_population(config, split="training")
    assert np.array_equal(first.references, second.references)
    assert np.array_equal(first.target_grids, second.target_grids)
    assert first.references.shape == (3, 2, 4, 16, 3)
    assert first.identity_index == 2
    assert set(first.content_labels.reshape(-1)) == set(range(8))
    assert set(first.nuisance_labels.reshape(-1)) == set(range(4))
