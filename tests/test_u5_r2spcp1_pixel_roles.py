from __future__ import annotations

from scripts.freeze_u5_r2spcp1_pixel_roles import ROLE_COUNTS, selection_key


def test_role_counts_consume_exact_eligible_pool() -> None:
    assert sum(count for _, count in ROLE_COUNTS) == 457


def test_role_names_are_unique_and_ordered() -> None:
    names = [name for name, _ in ROLE_COUNTS]
    assert names == [
        "development",
        "confirmation",
        "operator_fit",
        "operator_calibration",
        "operator_sealed",
        "reserve",
    ]
    assert len(names) == len(set(names))


def test_selection_key_is_stable_and_scene_sensitive() -> None:
    assert selection_key("I0001") == selection_key("I0001")
    assert selection_key("I0001") != selection_key("I0002")
