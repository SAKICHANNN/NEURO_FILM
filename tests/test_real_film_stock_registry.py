from __future__ import annotations

import pytest

from src.real_film.stock_registry import StockRegistryError, crosscheck_blueneg, validate_registry


def _stock(label: str, stock_id: str, selected: bool = False) -> dict:
    return {
        "film_stock_id": stock_id, "display_name": label, "manufacturer": "x",
        "product_line": "y", "nominal_iso": 100, "film_type": "color_negative",
        "emulsion_generation": "unknown", "catalog_code": "unknown",
        "market_status": "unknown", "source_dataset": "blueneg", "source_label": label,
        "label_status": "research_eligible_dataset_declared", "data_evidence_grade": "S1-candidate",
        "expert_evidence_grade": "none", "roll_count": 1, "frame_count": 1,
        "aligned_public_frame_count": 0, "selected_first_pilot": selected,
        "claim_ceiling": "test",
    }


def _registry(stocks: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "source_datasets": {"blueneg": {}},
        "coverage": {
            "named_stock_data_s1_or_higher": 0,
            "named_stock_experts_s2_or_higher": 0,
            "calibrated_stock_experts_s3": 0,
            "coverage_accounts_must_not_be_merged": True,
        },
        "stocks": stocks,
    }


def test_registry_and_source_crosscheck_pass() -> None:
    registry = _registry([_stock("a", "a", True), _stock("b", "b", True)])
    validation = validate_registry(registry)
    assert validation["selected_pilots"] == ["a", "b"]
    frames = [{"film_type": "a"}, {"film_type": "b"}]
    rolls = [
        {"film_type": "a", "roll_id": "ra", "public_non_test_pseudogt_frames": 0},
        {"film_type": "b", "roll_id": "rb", "public_non_test_pseudogt_frames": 0},
    ]
    assert crosscheck_blueneg(registry, frames, rolls)["passed"] is True


def test_claimed_hint_cannot_be_promoted_above_s0() -> None:
    left, right = _stock("a", "a", True), _stock("b", "b", True)
    left["label_status"] = "claimed_stock_hint"
    with pytest.raises(StockRegistryError):
        validate_registry(_registry([left, right]))


def test_selected_pilot_cannot_be_s0() -> None:
    left, right = _stock("a", "a", True), _stock("b", "b", True)
    left["label_status"] = "claimed_stock_hint"
    left["data_evidence_grade"] = "S0"
    with pytest.raises(StockRegistryError, match="cannot be selected"):
        validate_registry(_registry([left, right]))


def test_coverage_counts_are_computed_not_claimed() -> None:
    left, right = _stock("a", "a", True), _stock("b", "b", True)
    left["data_evidence_grade"] = "S1"
    with pytest.raises(StockRegistryError, match="coverage mismatch"):
        validate_registry(_registry([left, right]))
