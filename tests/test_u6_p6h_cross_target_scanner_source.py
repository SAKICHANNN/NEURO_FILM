from __future__ import annotations

from src.eval.physical_cross_target_scanner_source import (
    _connectivity,
    _safe_member_name,
)


def test_zip_member_safety_rejects_traversal_and_absolute_paths() -> None:
    assert _safe_member_name("set/image1.tif")
    assert not _safe_member_name("../image1.tif")
    assert not _safe_member_name("set/../../image1.tif")
    assert not _safe_member_name("/absolute/image1.tif")
    assert not _safe_member_name("C:/absolute/image1.tif")


def test_connectivity_distinguishes_same_target_from_cross_target_edges() -> None:
    rows = [
        {"target_set": 1, "scanner": "a", "role": "a-one"},
        {"target_set": 1, "scanner": "b", "role": "b-one"},
        {"target_set": 2, "scanner": "c", "role": "c-two"},
        {"target_set": 2, "scanner": "d", "role": "d-two"},
    ]
    result = _connectivity(rows)
    assert result["target_set_unique_scanner_counts"] == {"1": 2, "2": 2}
    assert result["scanners_repeated_across_target_sets"] == []
    assert result["scanner_target_graph_connected_across_sets"] is False


def test_connectivity_detects_repeated_scanner_edge() -> None:
    rows = [
        {"target_set": 1, "scanner": "a", "role": "a-one"},
        {"target_set": 2, "scanner": "a", "role": "a-two"},
    ]
    result = _connectivity(rows)
    assert result["scanners_repeated_across_target_sets"] == ["a"]
    assert result["scanner_target_graph_connected_across_sets"] is True
