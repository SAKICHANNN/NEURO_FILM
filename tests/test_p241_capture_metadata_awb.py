from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_p241_capture_metadata_awb_development import _relative_reduction
from src.eval.capture_metadata_awb import (
    EXPOSURE_FEATURE_INDICES,
    FULL_FEATURE_NAMES,
    TIME_FEATURE_INDICES,
    CaptureMetadataAwbError,
    angular_errors_degrees,
    apply_ridge,
    capture_features,
    fit_ridge,
    log_target_to_neutral,
    neutral_log_target,
)


def _facts(index: int) -> dict[str, object]:
    return {
        "date_time_original": f"2026-08-{index + 1:02d}T{index + 1:02d}:00:00",
        "exposure_time_seconds": 1.0 / (30.0 + index * 10.0),
        "iso_speed": 100 + index * 40,
        "f_number": 2.0 + index * 0.25,
        "as_shot_neutral_green_normalized": [0.5 + 0.01 * index, 1.0, 0.75],
    }


def test_capture_features_and_target_are_finite_and_exact_shape() -> None:
    facts = _facts(0)
    features = capture_features(facts)
    target = neutral_log_target(facts)
    assert features.shape == (len(FULL_FEATURE_NAMES),)
    assert target.tolist() == pytest.approx(np.log([0.5, 0.75]).tolist())
    assert np.all(np.isfinite(features))


@pytest.mark.parametrize("indices", [tuple(range(7)), TIME_FEATURE_INDICES, EXPOSURE_FEATURE_INDICES])
def test_ridge_freeze_and_apply_are_repeat_exact(indices: tuple[int, ...]) -> None:
    features = np.stack([capture_features(_facts(index)) for index in range(8)])
    targets = np.stack([neutral_log_target(_facts(index)) for index in range(8)])
    first = fit_ridge(features, targets, feature_indices=indices, alpha=10.0)
    second = fit_ridge(features, targets, feature_indices=indices, alpha=10.0)
    assert first == second
    assert np.array_equal(apply_ridge(first, features), apply_ridge(second, features))


def test_angular_error_is_zero_for_identical_neutrals() -> None:
    log_values = np.asarray([[-0.5, -0.25], [-0.75, 0.1]], dtype=np.float64)
    neutral = log_target_to_neutral(log_values)
    assert angular_errors_degrees(neutral, neutral).tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_invalid_metadata_and_degenerate_fit_fail_closed() -> None:
    with pytest.raises(CaptureMetadataAwbError, match="incomplete"):
        capture_features({})
    features = np.ones((4, len(FULL_FEATURE_NAMES)), dtype=np.float64)
    targets = np.zeros((4, 2), dtype=np.float64)
    with pytest.raises(CaptureMetadataAwbError, match="degenerate"):
        fit_ridge(features, targets, feature_indices=(0, 1), alpha=10.0)


def test_relative_reduction_uses_a_strict_positive_denominator() -> None:
    candidate = np.asarray([0.5, 0.0], dtype=np.float64)
    control = np.asarray([1.0, 0.0], dtype=np.float64)
    assert _relative_reduction(candidate, control).tolist() == pytest.approx([0.5, 0.0])
