from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.run_u5_r2cham4_portra400_same_scene_source import (
    SourceError,
    enumerate_remote,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2cham4_portra400_same_scene_source_v1.json"


def test_contract_binds_six_unique_roles_under_budget() -> None:
    contract = load_contract(CONFIG)
    rows = contract["acquisition"]["expected_files"]
    assert len(rows) == 6
    assert len({row["role"] for row in rows}) == 6
    assert sum(row["bytes"] for row in rows) == 183875041
    assert contract["source"]["explicit_reuse_licence_observed"] is False
    assert contract["gates"]["operator_fit_allowed"] is False


def test_remote_inventory_drift_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = load_contract(CONFIG)
    rows = deepcopy(contract["acquisition"]["expected_files"])
    first = rows[0]
    fake = type("Row", (), {"id": first["file_id"], "path": "wrong.json"})()
    monkeypatch.setattr(
        "scripts.run_u5_r2cham4_portra400_same_scene_source.gdown.download_folder",
        lambda **_kwargs: [fake],
    )
    with pytest.raises(SourceError, match="inventory drift"):
        enumerate_remote(contract)
