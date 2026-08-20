from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.three_stock_spectral_shape import (
    ThreeStockSpectralShapeError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/rf3_d5_three_stock_spectral_shape_v1.json"


def test_contract_loads() -> None:
    contract = load_contract(CONFIG)
    assert contract["wavelengths_nm"] == list(range(400, 681, 10))
    assert len(contract["sources"]) == 3


def test_contract_rejects_gate_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["gates"][
        "minimum_pairwise_active_union_rmse_log_shape_difference_lower_bound"
    ] = 0.07
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ThreeStockSpectralShapeError):
        load_contract(path)
