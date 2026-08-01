from __future__ import annotations

import random

import pytest

from src.eval.fivek_group_split import (
    FiveKGroupSplitError,
    assign_group_split,
)


def _rows() -> list[dict[str, str]]:
    sizes = {"camera-a": 2, "camera-b": 3, "camera-c": 2, "camera-d": 4}
    return [
        {"pair_id": f"{group}-{index}", "camera_model": group}
        for group, size in sizes.items()
        for index in range(size)
    ]


def test_group_split_is_target_blind_input_order_independent_and_exact() -> None:
    rows = _rows()
    shuffled = rows.copy()
    random.Random(19).shuffle(shuffled)
    kwargs = {
        "group_field": "camera_model",
        "id_field": "pair_id",
        "seed": 2026080117,
        "target_confirmation_rows": 4,
        "minimum_confirmation_rows": 3,
        "maximum_confirmation_rows": 5,
        "minimum_development_groups": 2,
    }
    first = assign_group_split(rows, **kwargs)
    second = assign_group_split(shuffled, **kwargs)
    assert first == second
    assert first["confirmation_rows"] == 4
    assert first["development_rows"] == 7
    assert first["selection_used_target_or_pixels"] is False
    assert not set(first["development_groups"]) & set(
        first["confirmation_groups"]
    )


def test_group_split_rejects_duplicate_identity() -> None:
    rows = _rows()
    rows.append(dict(rows[0]))
    with pytest.raises(FiveKGroupSplitError, match="duplicate row identity"):
        assign_group_split(
            rows,
            group_field="camera_model",
            id_field="pair_id",
            seed=1,
            target_confirmation_rows=4,
            minimum_confirmation_rows=3,
            maximum_confirmation_rows=5,
            minimum_development_groups=2,
        )


def test_group_split_fails_when_whole_groups_cannot_meet_support() -> None:
    rows = [
        {"pair_id": f"a-{index}", "camera_model": "camera-a"}
        for index in range(9)
    ] + [{"pair_id": "b-0", "camera_model": "camera-b"}]
    with pytest.raises(FiveKGroupSplitError):
        assign_group_split(
            rows,
            group_field="camera_model",
            id_field="pair_id",
            seed=1,
            target_confirmation_rows=5,
            minimum_confirmation_rows=4,
            maximum_confirmation_rows=6,
            minimum_development_groups=1,
        )
