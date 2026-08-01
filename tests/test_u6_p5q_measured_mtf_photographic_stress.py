from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.physical_measured_mtf_photographic_stress import (
    MeasuredMtfPhotographicError,
    _strong_edge_ratio,
    _texture_energy_ratio,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p5q_measured_mtf_photographic_stress_v1.json"


def test_contract_is_frozen_and_rejects_gate_drift(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    assert contract["pipeline"]["row_partitions"] == [257, 509]
    contract["automatic_gates"]["minimum_strong_edge_gradient_ratio"] = 0.4
    path = tmp_path / "drift.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(MeasuredMtfPhotographicError, match="contract drift"):
        load_contract(path)


def test_edge_and_texture_metrics_order_blur_below_identity() -> None:
    image = np.zeros((64, 96, 3), dtype=np.float64)
    image[:, 48:] = 1.0
    blurred = image.copy()
    blurred[:, 47:50] = np.asarray([0.25, 0.5, 0.75])[None, :, None]
    edge = _strong_edge_ratio(image, blurred)
    texture = _texture_energy_ratio(image, blurred)
    assert 0.0 < edge < 1.0
    assert 0.0 < texture < 1.0


def test_edge_metric_rejects_empty_support() -> None:
    constant = np.ones((16, 16, 3), dtype=np.float64)
    with pytest.raises(MeasuredMtfPhotographicError, match="support is empty"):
        _strong_edge_ratio(constant, constant)
