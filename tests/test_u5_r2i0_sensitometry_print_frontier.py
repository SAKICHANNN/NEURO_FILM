from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.sensitometry_print_frontier import SensitometryFrontierError, candidate_bank, shortlist_candidates, validate_contract


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u5_r2i0_sensitometry_print_frontier_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_candidate_bank_is_one_strength_path() -> None:
    rows = candidate_bank(_config())
    assert [row["candidate_id"] for row in rows] == ["sensitometry_print__s035", "sensitometry_print__s050", "sensitometry_print__s065", "sensitometry_print__s080", "sensitometry_print__s100"]
    assert [row["strength"] for row in rows] == [0.35, 0.5, 0.65, 0.8, 1.0]


def test_contract_hashes_and_frozen_set_validate() -> None:
    validated = validate_contract(ROOT, _config())
    assert len(validated["samples"]) == 41
    assert len(validated["candidates"]) == 5


def test_shortlist_orders_residual_then_style_then_lower_strength() -> None:
    summaries = {
        "a": {"automatic_survivor": True, "gold_median_non_basic_residual_delta_e76": 8.0, "gold_median_style_delta_e76": 9.0, "strength": 0.8},
        "b": {"automatic_survivor": True, "gold_median_non_basic_residual_delta_e76": 8.0, "gold_median_style_delta_e76": 9.0, "strength": 0.5},
        "c": {"automatic_survivor": True, "gold_median_non_basic_residual_delta_e76": 10.0, "gold_median_style_delta_e76": 7.0, "strength": 1.0},
        "d": {"automatic_survivor": False, "gold_median_non_basic_residual_delta_e76": 100.0, "gold_median_style_delta_e76": 100.0, "strength": 0.35},
    }
    assert shortlist_candidates(summaries, {"shortlist": {"maximum_strengths": 2}}) == ["c", "b"]


def test_tampered_parent_hash_fails_closed() -> None:
    config = _config()
    config["composition_config_sha256"] = "0" * 64
    with pytest.raises(SensitometryFrontierError, match="hash mismatch"):
        validate_contract(ROOT, config)
