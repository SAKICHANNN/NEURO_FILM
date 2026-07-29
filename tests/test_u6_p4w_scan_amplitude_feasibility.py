from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.eval.scan_amplitude_feasibility import (
    robust_relative_amplitude,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4w_scan_amplitude_feasibility_v1.json"


def test_p4w_contract_binds_all_eight_scans_without_labels() -> None:
    config = json.loads(CONTRACT.read_text(encoding="utf-8"))
    source, analysis = validate_contract(ROOT, config)
    assert len(source["files"]) == 8
    assert analysis["pixel_contract"]["crop_size_pixels"] == 512
    assert config["split"]["stock_labels_available_to_analysis"] is False


def test_robust_amplitude_is_scale_invariant_and_detrends_plane() -> None:
    y, x = np.mgrid[0:128, 0:128]
    rng = np.random.default_rng(2026072905)
    values = 1.0 + 0.001 * x + 0.002 * y
    values += rng.normal(0.0, 0.02, values.shape)
    first = robust_relative_amplitude(values)
    second = robust_relative_amplitude(values * 37.0)
    assert 0.015 < first < 0.025
    assert abs(first - second) <= 1e-12
