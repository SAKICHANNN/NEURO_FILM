from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_monotone_curve_blind_review import (
    select_rows,
    validate_contract,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay7_monotone_curve_blind_product_value_v1.json"
)


def test_contract_binds_confirmed_parent_and_hidden_labels() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert validated["report"]["automatic_pass"] is True
    assert config["comparison"]["candidate_labels_hidden"] is True
    assert config["production_integration_allowed"] is False


def test_selection_is_fixed_across_four_benefit_strata() -> None:
    rows = [
        {
            "pair_id": f"pair-{index:02d}",
            "ridge": {"neutral_rmse": float(index)},
            "global": {"neutral_rmse": 0.0},
        }
        for index in range(64)
    ]
    selected = select_rows(rows)
    assert len(selected) == 16
    assert [row["pair_id"] for row in selected] == [
        f"pair-{index:02d}"
        for index in (1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49, 53, 57, 61)
    ]
