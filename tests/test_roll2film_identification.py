from __future__ import annotations

import numpy as np

from src.roll2film.evaluation import (
    classify_e0,
    classify_fixed_budget_e0,
    paired_improvement_summary,
    recovery_metrics,
)
from src.roll2film.identification import (
    estimate_affine_spline_transport_operator,
    estimate_gaussian_transport_operator,
)
from src.roll2film.simulator import (
    PseudoRollConfig,
    default_l2_truth_operator,
    sample_neutral_prior,
    simulate_pseudo_roll,
)


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


def test_fixed_budget_gate_separates_method_controls_from_roll_claim() -> None:
    summary = paired_improvement_summary(
        [0.10, 0.12, 0.11, 0.09],
        [0.05, 0.06, 0.055, 0.045],
        seed=7,
        bootstrap_resamples=500,
    )
    result = classify_fixed_budget_e0(
        partition_parameter_max_abs=0.0,
        nuisance_boundary=summary,
        independent_support=summary,
        mixed_operator=summary,
    )

    assert result["method_control_decision"] == "pass"
    assert result["roll_information_decision"] == "not_established"


def test_paired_improvement_rejects_mismatched_inputs() -> None:
    import pytest

    with pytest.raises(ValueError, match="equally sized"):
        paired_improvement_summary([0.1, 0.2], [0.1], seed=1)


def test_affine_spline_transport_recovers_l2_truth_without_pairs() -> None:
    truth = default_l2_truth_operator()
    roll = simulate_pseudo_roll(
        PseudoRollConfig(frames=32, pixels_per_frame=512, seed=301),
        truth,
    )
    neutral = sample_neutral_prior(32768, seed=302)
    holdout = sample_neutral_prior(8192, seed=303)

    estimate = estimate_affine_spline_transport_operator(neutral, roll.target_frames)
    error = float(np.sqrt(np.mean((estimate.apply(holdout) - truth.apply(holdout)) ** 2)))

    assert error < 0.02
    assert float(estimate.jacobian_determinant(holdout).min()) > 0.0
