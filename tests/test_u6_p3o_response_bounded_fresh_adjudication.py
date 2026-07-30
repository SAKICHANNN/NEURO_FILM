from __future__ import annotations

from src.eval.response_bounded_fresh_adjudication import score_blind_round


def test_score_blind_round_resolves_candidate_parent_and_tie() -> None:
    scored = score_blind_round(
        judgments={"one": "A", "two": "B", "three": "tie"},
        mapping={
            "one": {"A": "bounded", "B": "no_spatial"},
            "two": {"A": "bounded", "B": "no_spatial"},
            "three": {"A": "no_spatial", "B": "bounded"},
        },
    )
    assert scored["candidate_votes"] == 1
    assert scored["parent_votes"] == 1
    assert scored["ties"] == 1
    assert scored["resolved"] == {
        "one": "bounded",
        "two": "no_spatial",
        "three": "tie",
    }
