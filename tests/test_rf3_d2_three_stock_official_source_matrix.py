from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.real_film.official_three_stock_prior_source import (
    OfficialStockSourceError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/rf3_d2_three_stock_official_source_matrix_v1.json"


def test_contract_freezes_three_named_stocks_and_nonrenderable_role() -> None:
    contract = load_contract(CONTRACT)
    assert [row["stock_id"] for row in contract["stocks"]] == [
        "fujifilm_velvia_50",
        "kodak_portra_400",
        "kodak_ektar_100",
    ]
    assert all(row["renderable_rgb_target"] is False for row in contract["stocks"])


def test_contract_rejects_host_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["stocks"][0]["allowed_host"] = "example.com"
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfficialStockSourceError, match="URL/host drift"):
        load_contract(path)


def test_contract_rejects_renderable_target_inflation(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["stocks"][1]["renderable_rgb_target"] = True
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OfficialStockSourceError, match="invalid stock source row"):
        load_contract(path)


def test_missing_source_fails_without_acquisition(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    with pytest.raises(OfficialStockSourceError, match="source missing"):
        evaluate(contract, tmp_path, acquire_missing=False)


@pytest.mark.skipif(
    not all((ROOT / row["path"]).is_file() for row in load_contract(CONTRACT)["stocks"]),
    reason="RF3.D2 exact official sources are unavailable",
)
def test_official_source_audit_is_exact_and_nonrenderable() -> None:
    contract = load_contract(CONTRACT)
    first = evaluate(contract, ROOT, acquire_missing=False)
    second = evaluate(contract, ROOT, acquire_missing=False)
    assert first == second
    assert first["automatic_pass"]
    assert first["common_domain_count"] >= 4
    assert not first["incomparable_granularity_semantics"][
        "direct_numeric_cross_manufacturer_comparison_allowed"
    ]
