from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.color_engine.srgb_transfer import linear_srgb_to_encoded
from src.eval.fivek_bilateral_gain_capacity import (
    _apply_safe_log_gain,
    _gain_features,
    _held_mask,
    _stable_midrank,
    evaluate_gain_capacity,
)
from src.eval.fivek_bilateral_gain_capacity_run import validate_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2bt0_fivek_bilateral_gain_capacity_v1.json"


def test_bt0_contract_is_development_only_and_explicit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["confirmation_pixels_allowed"] is False
    assert config["learned_final_rgb_allowed"] is False
    assert config["candidate"]["parameter_count"] == 384
    assert (
        config["controls"]["parameter_matched_global_rgb_lut"]["parameter_count"] == 375
    )


def test_midrank_and_checkerboard_are_exact() -> None:
    values = np.asarray([[2.0, 1.0], [1.0, 4.0]])
    assert np.array_equal(
        _stable_midrank(values),
        np.asarray([[2.0 / 3.0, 1.0 / 6.0], [1.0 / 6.0, 1.0]]),
    )
    spec = {"macro_rows": 2, "macro_columns": 2}
    first = _held_mask((4, 4), spec, 0)
    second = _held_mask((4, 4), spec, 1)
    assert np.all(first ^ second)
    assert not np.any(first & second)


def test_gain_features_and_guard_are_convex_and_bounded() -> None:
    rank = np.linspace(0.0, 1.0, 35).reshape(5, 7)
    features = _gain_features((5, 7), rank, (4, 4, 8))
    assert features.shape == (35, 128)
    assert np.allclose(features.sum(axis=1), 1.0)
    source = np.full((5, 7, 3), 0.5)
    coefficients = np.full((128, 3), 1.5)
    spec = {"epsilon": 1.0 / 4096.0, "log_gain_bound": 1.5}
    output, dose = _apply_safe_log_gain(source, features, coefficients, spec)
    assert np.all(np.isfinite(output))
    assert np.min(output) >= 0.0
    assert np.max(output) <= 1.0
    assert np.min(dose) < 1.0


def test_spatial_luma_gain_beats_global_on_synthetic_local_effect() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["population"]["source_count_exact"] = 2
    config["evaluation"]["automatic_gates"]["source_count_exact"] = 2
    rows = []
    for index in range(2):
        y, x = np.mgrid[0:32, 0:32]
        linear = np.stack(
            [
                0.1 + 0.7 * x / 31.0,
                0.1 + 0.7 * y / 31.0,
                0.2 + 0.4 * (x + y) / 62.0,
            ],
            axis=-1,
        )
        local = 0.28 * np.sin(2.0 * np.pi * x / 31.0) * (0.4 + 0.6 * y / 31.0)
        gains = np.stack([local, -0.7 * local, 0.4 * local], axis=-1)
        target_linear = np.clip(linear * np.exp(gains), 0.0, 1.0)
        rows.append(
            {
                "pair_id": f"synthetic-{index}",
                "group": f"group-{index}",
                "target_variant": "aligned_expert",
                "source": linear_srgb_to_encoded(linear),
                "target": linear_srgb_to_encoded(target_linear),
            }
        )
    result = evaluate_gain_capacity(rows, config)
    assert result["metrics"]["mean_improvement_over_parameter_matched_global_lut"] > 0.0
    assert result["metrics"]["maximum_out_of_cube_fraction"] == 0.0
