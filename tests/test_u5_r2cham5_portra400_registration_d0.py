from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.flickr_single_author_pair_registration import register_pair
from src.eval.portra400_same_scene_registration import load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_cham5_requires_only_chart_registration() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cham5_portra400_registration_d0_v1.json"
    )
    assert [row["registration_required"] for row in contract["pairs"]] == [
        True,
        False,
        False,
    ]
    assert all(row["operator_fit_allowed"] is False for row in contract["pairs"])


def test_registration_rejects_structureless_pair() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cham5_portra400_registration_d0_v1.json"
    )
    source = np.zeros((64, 64, 3), dtype=np.uint8)
    homography, diagnostics = register_pair(source, source, contract["registration"])
    assert homography is None
    assert diagnostics["registration_gate_passed"] is False
