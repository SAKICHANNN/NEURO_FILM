from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_pair_preflight import (
    select_pair_names,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/u5_r2ay6s0_fivek_confirmation_preflight_v1.json"
)


def test_contract_binds_third_disjoint_population() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validate_contract(ROOT, config)
    assert config["exclusion"]["expected_total_unique_names"] == 192
    assert config["image_download_allowed"] is False


def test_selection_excludes_union_not_only_original_rows() -> None:
    selected = select_pair_names(
        licensed_names=["a", "b", "c", "d"],
        retained_names={"a", "c"},
        seed=7,
        count=2,
    )
    assert selected == ["b", "d"]
