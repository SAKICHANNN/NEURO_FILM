from __future__ import annotations

import json
from pathlib import Path

from src.eval.sensitometry_print_frontier import candidate_bank, validate_contract
from src.roll2film.sensitometry_gauge import NeutralAxisGaugeOperator


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2i1b_neutral_gauge_frontier_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_i1b_contract_builds_frozen_gauged_operator_and_bank() -> None:
    config = _config()
    validated = validate_contract(ROOT, config)
    assert isinstance(validated["operator"], NeutralAxisGaugeOperator)
    assert len(validated["samples"]) == 41
    assert [row["candidate_id"] for row in candidate_bank(config)] == [
        "neutral_gauged__s035",
        "neutral_gauged__s050",
        "neutral_gauged__s065",
        "neutral_gauged__s080",
        "neutral_gauged__s100",
    ]


def test_i1b_script_uses_dedicated_default_without_duplicate_evaluator() -> None:
    source = (ROOT / "scripts/run_u5_r2i1b_neutral_gauge_frontier.py").read_text(encoding="utf-8")
    assert "run_u5_r2i0_sensitometry_print_frontier import main" in source
    assert "u5_r2i1b_neutral_gauge_frontier_v1.json" in source
