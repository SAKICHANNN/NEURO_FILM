from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.nps_preserving_cloud_residual_lod_v5 import (
    _apply_residual_gain,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4dv_nps_preserving_cloud_residual_lod_v5.json"


def test_p4dv_contract_is_frozen_and_parent_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["fixture"]["development_seeds"] == [57181, 57187, 57193]
    assert contract["fixture"]["confirmation_seeds"] == [54187, 54193, 54203]
    assert contract["compiler"]["hard_clip_forbidden"] is True


def test_p4dv_residual_gain_is_exact_and_unclipped() -> None:
    expected = np.full((2, 3, 3), 0.5, dtype=np.float64)
    base = expected + np.array([0.1, -0.1, 0.05], dtype=np.float64)
    gain = np.array([2.0, 1.5, 1.25], dtype=np.float64)
    output = _apply_residual_gain(base, expected, gain)
    np.testing.assert_array_equal(output, expected + (base - expected) * gain)


def test_p4dv_evaluation_is_repeat_identical() -> None:
    first = evaluate(ROOT, CONTRACT)
    assert first["stable"]["gates"]["repeat"] is True
    assert len(first["stable"]["compiled_channel_gain"]) == 3
