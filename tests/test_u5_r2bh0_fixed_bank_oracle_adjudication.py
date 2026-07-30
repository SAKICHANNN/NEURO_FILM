from __future__ import annotations

import pytest

from src.eval.fixed_bank_oracle_adjudication import (
    FixedBankAdjudicationError,
    adjudicate_rank_tables,
)


ARMS = ("global", "case_a", "case_b")
THRESHOLDS = {
    "minimum_improved_leave_one_round_out_folds": 2,
    "minimum_aggregate_heldout_rank_gain": 2,
    "minimum_aggregate_heldout_relative_rank_gain": 0.1,
    "minimum_sources_with_stable_non_global_choice": 1,
    "maximum_confirmed_severe_artifact_count": 0,
}


def _table(
    source_a: tuple[int, int, int],
    source_b: tuple[int, int, int],
) -> dict[str, dict[str, int]]:
    return {
        "a": dict(zip(ARMS, source_a, strict=True)),
        "b": dict(zip(ARMS, source_b, strict=True)),
    }


def test_cross_round_gate_passes_stable_hard_choice() -> None:
    result = adjudicate_rank_tables(
        rank_tables=[
            _table((2, 1, 3), (1, 3, 2)),
            _table((2, 1, 3), (1, 3, 2)),
            _table((2, 1, 3), (1, 3, 2)),
        ],
        arms=ARMS,
        tie_order=ARMS,
        thresholds=THRESHOLDS,
        confirmed_severe_count=0,
    )
    assert result["pass"]
    assert result["leave_one_round_out"]["improved_folds"] == 3
    assert result["leave_one_round_out"]["aggregate_rank_gain"] == 3
    assert result["leave_one_round_out"]["stable_non_global_choices"] == {
        "a": {"arm": "case_a", "fold_count": 3}
    }


def test_development_tie_falls_back_to_fold_global() -> None:
    result = adjudicate_rank_tables(
        rank_tables=[
            _table((1, 2, 3), (1, 2, 3)),
            _table((2, 1, 3), (1, 2, 3)),
            _table((1, 2, 3), (1, 2, 3)),
        ],
        arms=ARMS,
        tie_order=ARMS,
        thresholds=THRESHOLDS,
        confirmed_severe_count=0,
    )
    fold_two = result["leave_one_round_out"]["folds"][1]
    assert fold_two["selected_global_arm"] == "global"
    assert fold_two["selected_source_arms"]["a"] == "global"


def test_severe_veto_blocks_otherwise_passing_oracle() -> None:
    result = adjudicate_rank_tables(
        rank_tables=[
            _table((2, 1, 3), (1, 3, 2)),
            _table((2, 1, 3), (1, 3, 2)),
            _table((2, 1, 3), (1, 3, 2)),
        ],
        arms=ARMS,
        tie_order=ARMS,
        thresholds=THRESHOLDS,
        confirmed_severe_count=1,
    )
    assert not result["pass"]
    assert not result["gates"][
        "maximum_confirmed_severe_artifact_count"
    ]["pass"]


def test_rejects_incomplete_ranking() -> None:
    bad = _table((1, 1, 3), (1, 2, 3))
    with pytest.raises(FixedBankAdjudicationError):
        adjudicate_rank_tables(
            rank_tables=[bad, bad, bad],
            arms=ARMS,
            tie_order=ARMS,
            thresholds=THRESHOLDS,
            confirmed_severe_count=0,
        )
