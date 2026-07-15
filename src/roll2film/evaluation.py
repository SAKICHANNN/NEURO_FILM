"""Metrics and preregistered decision logic for the no-data E0 gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .operators import AffineColorOperator


@dataclass(frozen=True)
class RecoveryMetrics:
    matrix_rmse: float
    bias_rmse: float
    holdout_rgb_rmse: float
    style_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "matrix_rmse": self.matrix_rmse,
            "bias_rmse": self.bias_rmse,
            "holdout_rgb_rmse": self.holdout_rgb_rmse,
            "style_ratio": self.style_ratio,
        }


def recovery_metrics(
    estimate: AffineColorOperator,
    truth: AffineColorOperator,
    holdout: np.ndarray,
) -> RecoveryMetrics:
    matrix_rmse = float(np.sqrt(np.mean((estimate.matrix - truth.matrix) ** 2)))
    bias_rmse = float(np.sqrt(np.mean((estimate.bias - truth.bias) ** 2)))
    holdout_rmse = float(np.sqrt(np.mean((estimate.apply(holdout) - truth.apply(holdout)) ** 2)))
    truth_style = float(np.linalg.norm(truth.matrix - np.eye(3)) + np.linalg.norm(truth.bias))
    estimate_style = float(np.linalg.norm(estimate.matrix - np.eye(3)) + np.linalg.norm(estimate.bias))
    style_ratio = estimate_style / max(truth_style, 1e-12)
    return RecoveryMetrics(matrix_rmse, bias_rmse, holdout_rmse, style_ratio)


def paired_improvement_summary(
    baseline_errors: list[float] | np.ndarray,
    candidate_errors: list[float] | np.ndarray,
    *,
    seed: int,
    bootstrap_resamples: int = 5000,
) -> dict[str, float]:
    """Summarize paired error reduction with a deterministic bootstrap CI.

    Positive values mean the candidate improves on the baseline. Replicates,
    rather than correlated pixels, are the resampling unit.
    """
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    candidate = np.asarray(candidate_errors, dtype=np.float64)
    if baseline.ndim != 1 or candidate.ndim != 1 or len(baseline) != len(candidate):
        raise ValueError("paired errors must be one-dimensional and equally sized")
    if len(baseline) < 2 or bootstrap_resamples < 100:
        raise ValueError("at least two pairs and 100 bootstrap resamples are required")
    if not np.all(np.isfinite(baseline)) or not np.all(np.isfinite(candidate)):
        raise ValueError("paired errors must be finite")
    delta = baseline - candidate
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(bootstrap_resamples, len(delta)))
    bootstrap_means = delta[indices].mean(axis=1)
    baseline_mean = float(baseline.mean())
    return {
        "baseline_mean": baseline_mean,
        "candidate_mean": float(candidate.mean()),
        "absolute_improvement_mean": float(delta.mean()),
        "relative_improvement_mean": float(delta.mean() / max(baseline_mean, 1e-12)),
        "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
        "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
    }


def classify_fixed_budget_e0(
    *,
    partition_parameter_max_abs: float,
    nuisance_boundary: dict[str, float],
    independent_support: dict[str, float],
    mixed_operator: dict[str, float],
    partition_tolerance: float = 1e-12,
    relative_improvement_min: float = 0.10,
) -> dict[str, object]:
    """Classify implementation controls without claiming real-roll evidence."""

    def passes(summary: dict[str, float]) -> bool:
        return (
            summary["relative_improvement_mean"] >= relative_improvement_min
            and summary["ci95_low"] > 0.0
        )

    checks = {
        "partition_equivalence": partition_parameter_max_abs <= partition_tolerance,
        "nuisance_boundary_information": passes(nuisance_boundary),
        "independent_support_information": passes(independent_support),
        "mixed_operator_rejection": passes(mixed_operator),
    }
    return {
        "method_control_decision": "pass" if all(checks.values()) else "fail",
        "roll_information_decision": "not_established",
        "checks": checks,
        "thresholds": {
            "partition_parameter_max_abs": partition_tolerance,
            "paired_relative_improvement_min": relative_improvement_min,
            "paired_bootstrap_ci95_low_must_exceed_zero": True,
        },
        "claim_boundary": (
            "Passing validates fixed-budget affine controls and frame-boundary/support diagnostics only; "
            "it does not establish physical-roll information or real-film identifiability."
        ),
    }


def classify_fixed_budget_l2(
    *,
    partition_operator_grid_max_abs: float,
    nuisance_boundary: dict[str, float],
    independent_support: dict[str, float],
    mixed_operator: dict[str, float],
    partition_tolerance: float = 1e-12,
    relative_improvement_min: float = 0.10,
) -> dict[str, object]:
    """Classify L2 simulator controls without promoting real-roll evidence."""

    affine_result = classify_fixed_budget_e0(
        partition_parameter_max_abs=partition_operator_grid_max_abs,
        nuisance_boundary=nuisance_boundary,
        independent_support=independent_support,
        mixed_operator=mixed_operator,
        partition_tolerance=partition_tolerance,
        relative_improvement_min=relative_improvement_min,
    )
    affine_result["claim_boundary"] = (
        "Passing validates fixed-budget affine-plus-monotone-spline method controls only; "
        "it does not establish physical-roll information, film identity, stock calibration, "
        "or product safety."
    )
    return affine_result


def classify_e0(
    correct_by_size: dict[int, float],
    shuffled_by_size: dict[int, float],
    *,
    endpoint_relative_gain_min: float = 0.20,
    correct_vs_shuffled_final_gap_min: float = 0.20,
) -> dict[str, object]:
    sizes = sorted(correct_by_size)
    if sizes != sorted(shuffled_by_size) or len(sizes) < 3:
        raise ValueError("correct and shuffled results must share at least three group sizes")
    x = np.log2(np.asarray(sizes, dtype=np.float64))
    correct = np.asarray([correct_by_size[size] for size in sizes])
    shuffled = np.asarray([shuffled_by_size[size] for size in sizes])
    slope = float(np.polyfit(x, correct, 1)[0])
    endpoint_gain = float(1.0 - correct[-1] / correct[0])
    shuffled_gap = float(1.0 - correct[-1] / shuffled[-1])
    passed = (
        slope < 0
        and endpoint_gain >= endpoint_relative_gain_min
        and shuffled_gap >= correct_vs_shuffled_final_gap_min
    )
    return {
        "decision": "pass" if passed else "fail",
        "negative_log2_size_slope": slope,
        "endpoint_relative_gain": endpoint_gain,
        "correct_vs_shuffled_final_gap": shuffled_gap,
        "thresholds": {
            "slope_must_be_negative": True,
            "endpoint_relative_gain_min": endpoint_relative_gain_min,
            "correct_vs_shuffled_final_gap_min": correct_vs_shuffled_final_gap_min,
        },
    }
