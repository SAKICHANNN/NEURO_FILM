from __future__ import annotations

from pathlib import Path

import pytest

from src.eval.noever_vision3_resolution_source import (
    NoeverVision3SourceError,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu11_noever_vision3_resolution_source_v1.json"


def test_contract_freezes_complete_balanced_resolution_chart_inventory() -> None:
    contract = load_contract(CONTRACT)
    rows = contract["source"]["selected_members"]
    assert len(rows) == 18
    assert sum(row["compressed_size"] for row in rows) == 319810605
    names = [row["name"] for row in rows]
    assert all("_RC_" in name and "_T4_" in name for name in names)
    assert {name.split("/")[2] for name in names} == {"50D", "200T", "500T"}


def test_contract_rejects_member_inventory_drift(tmp_path: Path) -> None:
    payload = CONTRACT.read_text(encoding="utf-8").replace(
        '"selected_member_count": 18', '"selected_member_count": 17'
    )
    path = tmp_path / "contract.json"
    path.write_text(payload, encoding="utf-8")
    # The declared count is part of the source identity even if rows remain present.
    with pytest.raises(NoeverVision3SourceError):
        load_contract(path)
