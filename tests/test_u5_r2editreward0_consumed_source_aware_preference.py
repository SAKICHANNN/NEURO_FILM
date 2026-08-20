from __future__ import annotations

from pathlib import Path

from src.eval.editreward_consumed_source_aware_preference import (
    ScoreRow,
    _presentation_index,
    wrong_source_controls,
)


def _row(dataset: str, index: int, label: str) -> ScoreRow:
    source_prefix = "p399" if dataset == "p401" else "p402"
    path = Path(f"{dataset}-{index}-{label}.png")
    return ScoreRow(
        dataset=dataset,
        source_id=f"{source_prefix}_{index:02d}",
        presentation_id=f"R1-{index + 1:02d}",
        role=("landscape_nature", "architecture_interior", "night_artificial_light")[
            index // 4
        ],
        label=label,
        source_path=path,
        candidate_path=path,
        source_sha256=f"source-{dataset}-{index}",
        candidate_sha256=f"candidate-{dataset}-{index}-{label}",
        source_geometry=(8, 8),
        candidate_geometry=(8, 8),
    )


def test_presentation_index_accepts_only_frozen_round_one() -> None:
    assert _presentation_index("R1-01") == 0
    assert _presentation_index("R1-12") == 11
    for invalid in ("R2-01", "R1-00", "R1-13"):
        try:
            _presentation_index(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{invalid} must fail closed")


def test_wrong_source_control_is_deterministic_cyclic_and_candidate_blind() -> None:
    rows = []
    for dataset, labels in (("p401", "AB"), ("p402", "ABCD")):
        for index in range(12):
            rows.extend(_row(dataset, index, label) for label in labels)
    controls = wrong_source_controls(rows)
    assert len(controls) == 24
    assert len({candidate.row_id for candidate, _ in controls}) == 24
    for candidate, wrong_source in controls:
        source_index = int(candidate.source_id[-2:])
        expected_index = (source_index + 1) % 12
        assert wrong_source.source_id.endswith(f"{expected_index:02d}")
        assert candidate.source_id != wrong_source.source_id


def test_row_id_contains_no_private_candidate_identity() -> None:
    row = _row("p402", 3, "C")
    assert row.row_id == "p402:R1-04:C"
    assert "identity" not in row.row_id
    assert "complete" not in row.row_id
