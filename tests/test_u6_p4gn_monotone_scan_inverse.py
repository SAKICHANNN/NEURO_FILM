import json
from pathlib import Path

import numpy as np
import pytest

from src.film_physics.monotone_scan_inverse import MonotoneScanInverseV1

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gn_monotone_scan_inverse_v1.json"


def test_p4gn_freezes_independent_monotone_inverse_before_chart():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    compiler = contract["compiler"]
    assert compiler["build_levels"] == 33
    assert compiler["confirmation_midpoints"] == 32
    assert compiler["chart_pixels_used_for_fit"] is False
    assert compiler["extrapolation_allowed"] is False
    assert compiler["hard_clipping_allowed"] is False
    assert contract["execution"]["stop_before_chart_on_calibration_failure"] is True
    assert contract["execution"]["post_result_retuning_allowed"] is False


def test_p4gn_monotone_inverse_interpolates_without_clipping():
    profile = MonotoneScanInverseV1(
        ((0.1, 0.4, 0.9), (0.2, 0.5, 0.8), (0.3, 0.6, 0.7)),
        (0.0, 0.5, 1.0),
    )
    scan = np.array([[[0.25, 0.35, 0.65]]], dtype=np.float32)
    result = profile.apply(scan)
    assert result[0, 0] == pytest.approx([0.25, 0.25, 0.75])
    assert profile.identity() == profile.identity()


def test_p4gn_monotone_inverse_rejects_extrapolation():
    profile = MonotoneScanInverseV1(
        ((0.1, 0.9), (0.1, 0.9), (0.1, 0.9)), (0.0, 1.0)
    )
    with pytest.raises(ValueError, match="forbidden extrapolation"):
        profile.apply(np.array([[[0.09, 0.5, 0.5]]], dtype=np.float32))
