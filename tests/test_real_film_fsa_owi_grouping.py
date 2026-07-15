from __future__ import annotations

from src.real_film.fsa_owi_grouping import (
    build_location_guard_groups,
    build_sequence_guard_groups,
    evaluate_group_gate,
    evaluate_location_recovery_gate,
    extract_assignment_location,
)


def _rows() -> list[dict]:
    return [
        {"loc_fsac_id": "fsac.1a00001", "creator_group": "a"},
        {"loc_fsac_id": "fsac.1a00003", "creator_group": "a"},
        {"loc_fsac_id": "fsac.1a00010", "creator_group": "a"},
        {"loc_fsac_id": "fsac.1a00011", "creator_group": None},
    ]


def test_sequence_groups_merge_only_same_creator_and_bounded_gap() -> None:
    rows = build_sequence_guard_groups(_rows(), maximum_adjacent_gap=5)
    groups = {row["loc_fsac_id"]: row["sequence_guard_group"] for row in rows}
    assert groups["fsac.1a00001"] == groups["fsac.1a00003"]
    assert groups["fsac.1a00003"] != groups["fsac.1a00010"]
    unknown = next(row for row in rows if row["loc_fsac_id"] == "fsac.1a00011")
    assert unknown["learning_eligible"] is False


def test_group_gate_reports_pass_on_sufficient_groups() -> None:
    raw = [
        {"loc_fsac_id": f"fsac.1a{i:05d}", "creator_group": creator}
        for creator, start in (("a", 0), ("b", 100))
        for i in range(start, start + 4)
    ]
    rows = build_sequence_guard_groups(raw, maximum_adjacent_gap=1)
    config = {
        "gates": {
            "minimum_known_creator_records": 8,
            "minimum_creators_with_eight_records": 0,
            "minimum_sequence_groups_with_three_records": 2,
            "maximum_largest_sequence_group_fraction": 0.6,
            "minimum_known_creator_coverage": 1.0,
        }
    }
    assert evaluate_group_gate(rows, config)["decision"] == "grouped_phase_c_allowed"


def test_location_recovery_uses_caption_and_sequence_fallback() -> None:
    raw = _rows()
    raw[0]["description"] = "Farm in Virginia"
    raw[1]["description"] = "Skyline Drive"
    raw[2]["description"] = "unknown place"
    raw[3]["description"] = "Washington, D.C."
    sequence = build_sequence_guard_groups(raw, maximum_adjacent_gap=5)
    rows = build_location_guard_groups(sequence)
    assert extract_assignment_location("Farm, vicinity of Atlanta, Ga.") == "GA"
    first = next(row for row in rows if row["loc_fsac_id"] == "fsac.1a00001")
    second = next(row for row in rows if row["loc_fsac_id"] == "fsac.1a00003")
    fallback = next(row for row in rows if row["loc_fsac_id"] == "fsac.1a00010")
    assert first["location_guard_group"] == second["location_guard_group"]
    assert "fallback:" in fallback["location_guard_group"]
    config = {"location_recovery": {
        "minimum_location_coverage": 0.5,
        "minimum_location_guard_groups_with_three_records": 0,
        "maximum_largest_location_guard_fraction": 1.0,
    }}
    assert evaluate_location_recovery_gate(rows, config)["passed"] is True
