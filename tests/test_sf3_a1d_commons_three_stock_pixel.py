from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from src.real_film.commons_three_stock_pixel import (
    CommonsThreeStockPixelError,
    build_selection,
    load_contract,
    load_metadata_report,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/sf3_a1d_commons_three_stock_pixel_integrity_v1.json"
CORRECTED_CONTRACT = ROOT / "configs/sf3_a1d2_commons_three_stock_pixel_integrity_v1.json"


def test_frozen_metadata_builds_balanced_three_stock_selection() -> None:
    contract = load_contract(CONTRACT)
    report = load_metadata_report(ROOT / contract["metadata_report"], contract)
    first = build_selection(report, contract)
    second = build_selection(deepcopy(report), contract)
    assert first == second
    assert first["selected_files"] == 72
    assert first["selected_by_stock"] == {
        "fujifilm_velvia_50": 24,
        "kodak_ektar_100": 24,
        "kodak_portra_400": 24,
    }
    for stock in contract["allowed_stock_ids"]:
        rows = [row for row in first["rows"] if row["film_stock_id"] == stock]
        assert len(rows) == len({row["normalized_author_group"] for row in rows}) == 24


def test_metadata_hash_drift_fails_closed(tmp_path: Path) -> None:
    contract = load_contract(CONTRACT)
    path = tmp_path / "report.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CommonsThreeStockPixelError, match="hash drifted"):
        load_metadata_report(path, contract)


def test_contract_and_parent_hashes_are_current() -> None:
    contract = load_contract(CONTRACT)
    assert sha256_file(ROOT / contract["metadata_report"]) == contract["metadata_report_sha256"]
    assert contract["operator_fitting_allowed"] is False
    json.dumps(contract, sort_keys=True)


def test_corrected_contract_vetoes_simulations_and_product_photos() -> None:
    contract = load_contract(CORRECTED_CONTRACT)
    report = load_metadata_report(ROOT / contract["metadata_report"], contract)
    selection = build_selection(report, contract)
    excluded = {int(value) for value in contract["selection"]["excluded_page_ids"]}
    selected = {int(row["page_id"]) for row in selection["rows"]}
    assert not selected.intersection(excluded)
    assert selection["selected_files"] == 72
