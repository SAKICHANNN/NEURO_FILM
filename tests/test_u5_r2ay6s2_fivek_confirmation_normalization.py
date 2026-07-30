from __future__ import annotations

import json
from pathlib import Path

from src.eval.fivek_fresh_normalization_support import validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/u5_r2ay6s2_fivek_confirmation_normalization_v1.json"
)


def test_contract_binds_all_prior_normalized_rows_without_candidate_use() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    validated = validate_contract(ROOT, config)
    assert len(validated["acquisition"]["rows"]) == 64
    assert config["support_and_leakage"]["existing_total_rows"] == 191
    assert config["development_candidate"]["use_during_this_leaf"] is False
    assert config["candidate_rendering_allowed"] is False
