from __future__ import annotations

import json
from pathlib import Path

from scripts.run_sf2_9b_colorreference_batch_family_source import (
    _assets,
    _year_from_name,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_source_inventory_is_balanced_and_unique() -> None:
    config = json.loads(
        (ROOT / "configs/sf2_9b_colorreference_batch_family_v1.json").read_text()
    )
    assets = _assets(config)
    assert len(assets) == 100
    assert len({asset["path"] for asset in assets}) == 100
    assert sum(asset["family_code"] == "E" for asset in assets) == 51
    assert sum(asset["family_code"] == "V" for asset in assets) == 49
    assert all(asset["path"].endswith(".zip") for asset in assets)


def test_charge_year_parsing_is_explicitly_2000s_only() -> None:
    assert _year_from_name("E040227.zip") == 2004
    assert _year_from_name("V240219.zip") == 2024
