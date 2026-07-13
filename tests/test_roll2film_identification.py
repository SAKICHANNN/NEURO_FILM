from __future__ import annotations

import numpy as np

from src.roll2film.evaluation import classify_e0, recovery_metrics
from src.roll2film.identification import estimate_gaussian_transport_operator
from src.roll2film.simulator import PseudoRollConfig, sample_neutral_prior, simulate_pseudo_roll


def test_gaussian_transport_recovers_known_spd_operator_without_pairs() -> None:
    roll = simulate_pseudo_roll(PseudoRollConfig(frames=32, pixels_per_frame=512, seed=11))
    neutral = sample_neutral_prior(32768, seed=12)
    holdout = sample_neutral_prior(4096, seed=13)

    estimate = estimate_gaussian_transport_operator(neutral, roll.target_frames)
    metrics = recovery_metrics(estimate, roll.operator, holdout)

    assert metrics.holdout_rgb_rmse < 0.01
    assert 0.75 < metrics.style_ratio < 1.25


def test_more_group_frames_reduce_mean_recovery_error() -> None:
    neutral = sample_neutral_prior(16384, seed=20)
    holdout = sample_neutral_prior(4096, seed=21)
    errors: dict[int, list[float]] = {1: [], 16: []}
    for replicate in range(12):
        for size in errors:
            roll = simulate_pseudo_roll(
                PseudoRollConfig(frames=size, pixels_per_frame=96, seed=1000 + replicate * 20 + size)
            )
            estimate = estimate_gaussian_transport_operator(neutral, roll.target_frames)
            errors[size].append(recovery_metrics(estimate, roll.operator, holdout).holdout_rgb_rmse)

    assert np.mean(errors[16]) < np.mean(errors[1]) * 0.65


def test_e0_gate_requires_scaling_and_shuffled_gap() -> None:
    result = classify_e0(
        {1: 0.10, 2: 0.08, 4: 0.06, 8: 0.04},
        {1: 0.11, 2: 0.11, 4: 0.10, 8: 0.10},
    )
    assert result["decision"] == "pass"

    failed = classify_e0(
        {1: 0.10, 2: 0.095, 4: 0.09, 8: 0.085},
        {1: 0.10, 2: 0.10, 4: 0.10, 8: 0.10},
    )
    assert failed["decision"] == "fail"
