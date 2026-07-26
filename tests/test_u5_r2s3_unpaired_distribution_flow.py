from __future__ import annotations

import numpy as np
import torch

from src.roll2film.unpaired_distribution_flow import (
    distribution_loss,
    fit_grouped_unpaired_distribution_flow,
    fit_unpaired_distribution_flow,
    grouped_distribution_loss,
    make_distribution_loss_assets,
)


def test_distribution_loss_assets_replay_exactly() -> None:
    spec = {
        "kind": "fixed_projection_sorted_quantile",
        "projection_count": 8,
        "projection_seed": 4,
    }
    left = make_distribution_loss_assets(spec)
    right = make_distribution_loss_assets(spec)
    assert np.array_equal(left.tensors["projections"], right.tensors["projections"])


def test_distribution_losses_are_zero_for_row_permutations() -> None:
    values = np.random.default_rng(5).uniform(size=(32, 3))
    shuffled = values[np.random.default_rng(6).permutation(len(values))]
    for spec in (
        {
            "kind": "fixed_projection_sorted_quantile",
            "projection_count": 8,
            "projection_seed": 7,
        },
        {
            "kind": "fixed_random_fourier_mmd",
            "frequencies_per_bandwidth": 4,
            "bandwidths": [0.1, 0.2],
            "feature_seed": 8,
        },
    ):
        assets = make_distribution_loss_assets(spec)
        loss = distribution_loss(
            torch.as_tensor(values),
            torch.as_tensor(shuffled),
            assets,
        )
        assert float(loss) < 1e-28


def test_small_cpu_fit_returns_bounded_explicit_operator() -> None:
    source = np.random.default_rng(9).uniform(0.1, 0.7, size=(32, 3))
    target = np.clip(source + np.array([0.08, -0.03, 0.04]), 0.0, 1.0)
    assets = make_distribution_loss_assets(
        {
            "kind": "fixed_projection_sorted_quantile",
            "projection_count": 8,
            "projection_seed": 10,
        }
    )
    operator, metrics = fit_unpaired_distribution_flow(
        source,
        target,
        loss_assets=assets,
        axis_size=3,
        integration_steps=4,
        coefficient_vector_norm_cap=1.0,
        steps=8,
        learning_rate=0.03,
        coefficient_l2=1e-4,
        velocity_smoothness_l2=1e-3,
        gradient_clip_norm=5.0,
        seed=11,
        device="cpu",
        deterministic_algorithms=True,
    )
    assert metrics["final_distribution_loss"] < metrics["initial_distribution_loss"]
    assert np.max(np.linalg.norm(operator.velocity_grid, axis=-1)) <= 1.0
    output = operator.apply(source)
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0


def test_grouped_loss_preserves_groups_and_singleton_arithmetic() -> None:
    assets = make_distribution_loss_assets(
        {
            "kind": "fixed_random_fourier_mmd",
            "frequencies_per_bandwidth": 8,
            "bandwidths": [0.1, 0.2],
            "feature_seed": 12,
        }
    )
    first = torch.full((16, 3), 0.2, dtype=torch.float64)
    second = torch.full((16, 3), 0.8, dtype=torch.float64)
    singleton = distribution_loss(first, second, assets)
    assert torch.equal(
        grouped_distribution_loss([first], [second], assets), singleton
    )
    correct = grouped_distribution_loss(
        [first, second], [first.clone(), second.clone()], assets
    )
    shuffled = grouped_distribution_loss(
        [first, second], [second.clone(), first.clone()], assets
    )
    assert float(correct) == 0.0
    assert float(shuffled) > 0.0


def test_grouped_float32_fit_repeats_and_remains_bounded() -> None:
    rng = np.random.default_rng(13)
    sources = [
        rng.uniform(0.1, 0.5, size=(24, 3)),
        rng.uniform(0.5, 0.9, size=(24, 3)),
    ]
    targets = [
        np.clip(source + np.array([0.04, -0.02, 0.03]), 0.0, 1.0)
        for source in sources
    ]
    assets = make_distribution_loss_assets(
        {
            "kind": "fixed_random_fourier_mmd",
            "frequencies_per_bandwidth": 8,
            "bandwidths": [0.1, 0.2],
            "feature_seed": 14,
        }
    )
    kwargs = {
        "loss_assets": assets,
        "axis_size": 3,
        "integration_steps": 4,
        "coefficient_vector_norm_cap": 1.0,
        "steps": 8,
        "learning_rate": 0.03,
        "coefficient_l2": 1e-4,
        "velocity_smoothness_l2": 1e-3,
        "gradient_clip_norm": 5.0,
        "seed": 15,
        "device": "cpu",
        "deterministic_algorithms": True,
        "optimization_dtype": "float32",
    }
    first, first_metrics = fit_grouped_unpaired_distribution_flow(
        sources, targets, **kwargs
    )
    second, second_metrics = fit_grouped_unpaired_distribution_flow(
        sources, targets, **kwargs
    )
    assert np.array_equal(first.velocity_grid, second.velocity_grid)
    assert first_metrics == second_metrics
    assert (
        first_metrics["final_distribution_loss"]
        < first_metrics["initial_distribution_loss"]
    )
    assert np.max(np.linalg.norm(first.velocity_grid, axis=-1)) <= 1.0
