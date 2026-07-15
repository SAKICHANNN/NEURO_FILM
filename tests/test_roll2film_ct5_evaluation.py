from __future__ import annotations

import numpy as np

from src.roll2film.ct5_evaluation import (
    cluster_bootstrap_improvement,
    evaluate_sampled_candidate,
    paired_per_image_affine_oracle,
)


def test_sampled_metrics_separate_identity_strength_from_fidelity() -> None:
    rng = np.random.default_rng(601)
    source = rng.uniform(0.05, 0.85, size=(3, 64, 3))
    target = source * np.array([1.08, 0.96, 1.02]) + 0.01
    identity, _ = evaluate_sampled_candidate(source, target, source)
    styled, _ = evaluate_sampled_candidate(source, target, target)

    assert identity["median_delta_e00_from_input"] == 0.0
    assert styled["mean_delta_e00_to_target"] == 0.0
    assert float(styled["median_delta_e00_from_input"]) > 0.0
    assert styled["sampled_ssim"] is None


def test_paired_affine_oracle_recovers_affine_targets() -> None:
    rng = np.random.default_rng(602)
    source = rng.uniform(0.1, 0.8, size=(4, 128, 3))
    target = source @ np.array([[1.05, 0.02, -0.01], [0.01, 0.96, 0.03], [0.0, 0.02, 1.04]]).T
    target += np.array([0.01, -0.005, 0.008])
    rendered = paired_per_image_affine_oracle(source, target)

    assert np.sqrt(np.mean((rendered - target) ** 2)) < 1e-7


def test_cluster_bootstrap_uses_cluster_not_pixel_units() -> None:
    summary = cluster_bootstrap_improvement(
        [5.0, 5.2, 4.8, 6.0],
        [3.0, 3.2, 2.8, 5.0],
        ["a", "a", "b", "c"],
        seed=8,
        resamples=500,
    )

    assert summary["clusters"] == 3
    assert summary["ci95_low"] > 0.0
