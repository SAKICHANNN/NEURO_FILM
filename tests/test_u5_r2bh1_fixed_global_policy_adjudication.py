from __future__ import annotations

import pytest

from src.eval.fixed_global_policy_adjudication import (
    ARMS,
    FixedGlobalPolicyAdjudicationError,
    adjudicate_choices,
)


THRESHOLDS = {
    "minimum_b0_round_wins": 2,
    "minimum_b0_aggregate_choices": 5,
    "maximum_confirmed_severe_artifact_count": 0,
}


def _mapping(flip: bool = False) -> dict[str, dict[str, str]]:
    arms = ARMS[::-1] if flip else ARMS
    return {
        source: {"A": arms[0], "B": arms[1]}
        for source in ("a", "b", "c")
    }


def test_passes_round_and_aggregate_gates() -> None:
    result = adjudicate_choices(
        choices_by_round=[
            {"a": "A", "b": "A", "c": "B"},
            {"a": "B", "b": "B", "c": "A"},
            {"a": "A", "b": "A", "c": "B"},
        ],
        mappings_by_round=[_mapping(), _mapping(True), _mapping()],
        thresholds=THRESHOLDS,
        confirmed_severe_count=0,
        automatic_gate_pass=True,
        repeat_exact=True,
    )
    assert result["pass"]
    assert result["aggregate_counts"][ARMS[0]] == 6
    assert result["b0_round_wins"] == 3


def test_aggregate_gate_can_fail_after_round_gate_passes() -> None:
    thresholds = dict(THRESHOLDS, minimum_b0_aggregate_choices=7)
    result = adjudicate_choices(
        choices_by_round=[
            {"a": "A", "b": "A", "c": "B"},
            {"a": "B", "b": "B", "c": "A"},
            {"a": "A", "b": "A", "c": "B"},
        ],
        mappings_by_round=[_mapping(), _mapping(True), _mapping()],
        thresholds=thresholds,
        confirmed_severe_count=0,
        automatic_gate_pass=True,
        repeat_exact=True,
    )
    assert not result["pass"]
    assert result["gates"]["minimum_b0_round_wins"]["pass"]
    assert not result["gates"]["minimum_b0_aggregate_choices"]["pass"]


def test_severe_veto_blocks_preference_pass() -> None:
    result = adjudicate_choices(
        choices_by_round=[{"a": "A", "b": "A", "c": "A"}] * 3,
        mappings_by_round=[_mapping()] * 3,
        thresholds=THRESHOLDS,
        confirmed_severe_count=1,
        automatic_gate_pass=True,
        repeat_exact=True,
    )
    assert not result["pass"]
    assert not result["gates"][
        "maximum_confirmed_severe_artifact_count"
    ]["pass"]


def test_rejects_invalid_mapping() -> None:
    bad = _mapping()
    bad["a"] = {"A": ARMS[0], "B": ARMS[0]}
    with pytest.raises(FixedGlobalPolicyAdjudicationError):
        adjudicate_choices(
            choices_by_round=[{"a": "A", "b": "A", "c": "A"}] * 3,
            mappings_by_round=[bad, _mapping(), _mapping()],
            thresholds=THRESHOLDS,
            confirmed_severe_count=0,
            automatic_gate_pass=True,
            repeat_exact=True,
        )
