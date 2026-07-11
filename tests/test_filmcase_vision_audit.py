from __future__ import annotations

import pytest

from src.filmcase.vision_audit import VisionAuditError, aggregate_reviews, build_blind_audit


def _all_reviews(plan, *, severe_for: tuple[str, str] | None = None):
    rows = []
    for mapping in plan.mapping:
        for label, candidate in mapping["label_to_candidate"].items():
            severe = "yes" if severe_for == (candidate, mapping["sample_id"]) else "no"
            rows.append({"round": mapping["round"], "sample_id": mapping["sample_id"], "label": label, "severe": severe, "style_strength": 4, "appeal": 3})
    return rows


def test_blind_plan_is_deterministic_and_hides_candidate_ids_from_sheet() -> None:
    first = build_blind_audit(["01", "02"], ["anchor09", "anchor56", "bland"], seed=7)
    second = build_blind_audit(["01", "02"], ["anchor09", "anchor56", "bland"], seed=7)
    assert first == second
    assert {label for row in first.sheet for label in row["labels"]} == {"A", "B", "C"}
    assert "anchor09" not in str(first.sheet)


def test_two_of_three_severe_votes_veto_candidate_sample() -> None:
    plan = build_blind_audit(["01"], ["anchor09", "anchor56"], seed=3)
    reviews = _all_reviews(plan, severe_for=("anchor56", "01"))
    result = aggregate_reviews(plan, reviews)
    target = next(row for row in result["cases"] if row["candidate_id"] == "anchor56")
    assert target["decision"] == "veto"
    assert result["candidates"]["anchor56"]["blind_pass_complete"] is False


def test_one_severe_vote_requires_original_resolution_adjudication() -> None:
    plan = build_blind_audit(["01"], ["anchor09", "anchor56"], seed=3)
    reviews = _all_reviews(plan)
    target_mapping = next(row for row in plan.mapping if row["round"] == 1)
    label = next(label for label, candidate in target_mapping["label_to_candidate"].items() if candidate == "anchor09")
    next(row for row in reviews if row["round"] == 1 and row["label"] == label)["severe"] = "yes"
    result = aggregate_reviews(plan, reviews)
    target = next(row for row in result["cases"] if row["candidate_id"] == "anchor09")
    assert target["decision"] == "needs_original_resolution_adjudication"


def test_duplicate_review_is_rejected() -> None:
    plan = build_blind_audit(["01"], ["anchor09", "anchor56"], seed=3)
    reviews = _all_reviews(plan)
    with pytest.raises(VisionAuditError, match="duplicate"):
        aggregate_reviews(plan, [*reviews, reviews[0]])
