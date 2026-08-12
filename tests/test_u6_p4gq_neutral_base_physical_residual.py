import json
from pathlib import Path

import numpy as np

from src.film_physics.bounded_linear_residual import apply_bounded_linear_residual

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gq_neutral_base_physical_residual_v1.json"


def test_p4gq_freezes_neutral_base_physical_residual():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert candidate["neutral_base"] == "exact-input-scene-linear"
    assert "cloud-free" in candidate["physical_residual"]
    assert candidate["empirical_scan_inverse_used"] is False
    assert candidate["hard_clipping_allowed"] is False
    assert contract["gates"]["minimum_physical_residual_rms"] == 0.0001
    assert contract["gates"]["maximum_limited_fraction"] == 0.05
    assert contract["execution"]["post_result_retuning_allowed"] is False


def test_p4gq_scales_physical_residual_without_clipping():
    base = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    reference = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)
    physical = np.array([[[0.1, 0.7, 0.9]]], dtype=np.float32)
    output, diagnostics = apply_bounded_linear_residual(base, physical, reference)
    assert np.all(output >= 0.0) and np.all(output <= 1.0)
    assert output[0, 0, 0] == 0.0
    assert output[0, 0, 1] > 0.5
    assert output[0, 0, 2] == 1.0
    assert diagnostics["limited_fraction"] == 2.0 / 3.0
    assert diagnostics["hard_clipping_used"] == 0.0
