from __future__ import annotations

import numpy as np
import pytest

from src.real_film.filmr_signal_audit import (
    FilmRSignalAuditError,
    build_support_matrix,
    evaluate_support_gate,
)


def test_support_matrix_requires_exact_label_coverage() -> None:
    rows = [{"pair_id": "a", "filename_family_claim": "x"}]
    with pytest.raises(FilmRSignalAuditError):
        build_support_matrix(rows, {"other": "scene"})


def test_structural_gate_rejects_single_content_families() -> None:
    families = ("a", "b", "c")
    contents = ("objects", "scenes")
    matrix = np.array([[4, 0], [3, 0], [0, 5]])
    result = evaluate_support_gate(
        families,
        contents,
        matrix,
        minimum_family_samples=3,
        minimum_cell_samples=2,
        minimum_supported_cells=2,
        minimum_shared_cells=1,
        minimum_multiclass_families=3,
    )
    assert result["decision"] == "structurally_unidentified"
    assert result["cross_content_families"] == []


def test_structural_gate_accepts_balanced_overlap() -> None:
    families = ("a", "b", "c")
    contents = ("objects", "scenes")
    matrix = np.array([[3, 2], [2, 3], [4, 2]])
    result = evaluate_support_gate(
        families,
        contents,
        matrix,
        minimum_family_samples=3,
        minimum_cell_samples=2,
        minimum_supported_cells=2,
        minimum_shared_cells=1,
        minimum_multiclass_families=3,
    )
    assert result["decision"] == "feature_audit_allowed"
    assert result["largest_pairwise_comparable_clique"] == ["a", "b", "c"]
