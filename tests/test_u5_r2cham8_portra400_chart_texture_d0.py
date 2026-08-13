from __future__ import annotations

from pathlib import Path

from src.eval.portra400_chart_texture_d0 import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cham8_contract_preserves_nuisance_ceiling() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cham8_portra400_chart_texture_d0_v1.json"
    )
    assert contract["analysis"]["minimum_patches_per_scene"] >= 24
    assert contract["analysis"]["shift_control_pixels"] == 17
    assert "scanner MTF/noise/sharpening" in contract["claim_ceiling"]
    assert "not stock calibration" in contract["claim_ceiling"]
