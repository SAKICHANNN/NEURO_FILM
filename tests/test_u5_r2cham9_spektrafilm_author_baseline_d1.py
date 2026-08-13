from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.portra400_spektrafilm_author_baseline_d1 import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham9_spektrafilm_author_baseline_d1_v1.json"


def test_cham9_contract_is_external_unfitted_colour_baseline() -> None:
    contract = load_contract(CONFIG)
    runtime = contract["external_runtime"]
    assert runtime["chart_fitting_allowed"] is False
    assert runtime["spatial_effects"] is False
    assert runtime["stochastic_effects"] is False
    assert "not Portra 400 calibration" in contract["claim_ceiling"]


def test_cham9_rejects_spatial_effect_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["external_runtime"]["spatial_effects"] = True
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported CHAM9 contract"):
        load_contract(path)

