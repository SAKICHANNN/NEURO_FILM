from __future__ import annotations

import numpy as np

from scripts.audit_u1_6g4f_explicit_new_operator import (
    _deterministic_payload_sha,
    _effect_pass,
    _seam_max,
    effect_metrics,
)


def _gates() -> dict:
    return {
        "effect_changed_uint8_channel_fraction_min": 0.04,
        "effect_composite_abs_p99_min": 0.0035,
        "effect_uint8_max_code_delta_min": 3,
        "effect_alpha_active_fraction_min": 0.05,
        "effect_highlight_to_other_alpha_ratio_min": 3.0,
        "effect_new_high_clip_fraction_max": 0.0001,
        "effect_new_low_clip_fraction_max": 0.0001,
    }


def test_effect_metrics_detect_selective_nonempty_effect() -> None:
    base = np.full((20, 20, 3), 0.2, np.float32)
    base[:2] = 0.8
    alpha = np.full((20, 20, 1), 0.0002, np.float32)
    alpha[:2] = 0.04
    output = base + alpha * np.asarray([0.8, 0.7, 0.5], np.float32)
    metrics = effect_metrics(base, output, alpha)
    assert metrics["changed_uint8_channel_fraction"] > 0.04
    assert metrics["composite_abs_p99"] > 0.0035
    assert metrics["uint8_max_code_delta"] >= 3
    assert metrics["highlight_to_other_alpha_ratio"] > 3.0
    assert metrics["new_high_clip_fraction"] == 0.0
    assert metrics["new_low_clip_fraction"] == 0.0
    assert _effect_pass(metrics, _gates())


def test_effect_gate_rejects_empty_and_unselective() -> None:
    base = np.full((20, 20, 3), 0.2, np.float32)
    metrics = effect_metrics(base, base.copy(), np.zeros((20, 20, 1), np.float32))
    assert not _effect_pass(metrics, _gates())


def test_seam_metric_reads_frozen_boundaries() -> None:
    difference = np.zeros((12, 13, 3), np.float32)
    difference[3, 7, 1] = 0.25
    assert _seam_max(difference, 4) == 0.25
    assert _seam_max(difference, 5) == 0.0


def test_deterministic_payload_ignores_runtime_only() -> None:
    first = [{"case_id": "x", "metric": 1.0, "runtime": {"seconds": [1.0]}, "automatic_pass": True}]
    second = [{"case_id": "x", "metric": 1.0, "runtime": {"seconds": [9.0]}, "automatic_pass": False}]
    assert _deterministic_payload_sha(first) == _deterministic_payload_sha(second)
    second[0]["metric"] = 2.0
    assert _deterministic_payload_sha(first) != _deterministic_payload_sha(second)
