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
