"""Metadata-only identifiability gate for the FILM-R RF1.1 audit."""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.stats import chi2_contingency


class FilmRSignalAuditError(ValueError):
    """Raised when the frozen FILM-R signal-audit contract is violated."""


def build_support_matrix(
    rows: Sequence[Mapping[str, Any]],
    content_labels: Mapping[str, str],
) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    """Build exact filename-family by manual-content counts."""
    pair_ids = [str(row["pair_id"]) for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise FilmRSignalAuditError("FILM-R pair ids must be unique")
    if set(pair_ids) != set(content_labels):
        raise FilmRSignalAuditError("manual content labels do not exactly cover pairs")
    families = tuple(sorted({str(row["filename_family_claim"]) for row in rows}))
    contents = tuple(sorted({str(content_labels[pair_id]) for pair_id in pair_ids}))
    family_index = {name: index for index, name in enumerate(families)}
    content_index = {name: index for index, name in enumerate(contents)}
    matrix = np.zeros((len(families), len(contents)), dtype=np.int64)
    for row in rows:
        pair_id = str(row["pair_id"])
        matrix[
            family_index[str(row["filename_family_claim"])],
            content_index[str(content_labels[pair_id])],
        ] += 1
    if int(matrix.sum()) != len(rows):
        raise FilmRSignalAuditError("support matrix lost FILM-R rows")
    return families, contents, matrix


def evaluate_support_gate(
    families: Sequence[str],
    contents: Sequence[str],
    matrix: np.ndarray,
    *,
    minimum_family_samples: int,
    minimum_cell_samples: int,
    minimum_supported_cells: int,
    minimum_shared_cells: int,
    minimum_multiclass_families: int,
) -> dict[str, Any]:
    """Apply the frozen structural-overlap gate without image features."""
    counts = np.asarray(matrix, dtype=np.int64)
    if counts.shape != (len(families), len(contents)) or np.any(counts < 0):
        raise FilmRSignalAuditError("invalid support matrix shape or counts")
    totals = counts.sum(axis=1)
    supported = counts >= minimum_cell_samples
    eligible = [
        index for index, total in enumerate(totals) if int(total) >= minimum_family_samples
    ]
    cross_content = [
        index
        for index in eligible
        if int(np.sum(supported[index])) >= minimum_supported_cells
    ]
    adjacency: dict[int, set[int]] = {index: set() for index in eligible}
    for left, right in itertools.combinations(eligible, 2):
        shared = int(np.sum(supported[left] & supported[right]))
        if shared >= minimum_shared_cells:
            adjacency[left].add(right)
            adjacency[right].add(left)
    largest_clique: tuple[int, ...] = ()
    for size in range(1, len(eligible) + 1):
        for candidate in itertools.combinations(eligible, size):
            if all(right in adjacency[left] for left, right in itertools.combinations(candidate, 2)):
                largest_clique = candidate
    comparable_cross_content = [index for index in cross_content if index in largest_clique]
    passed = (
        len(cross_content) >= minimum_multiclass_families
        and len(comparable_cross_content) >= minimum_multiclass_families
    )
    chi2, p_value, _, _ = chi2_contingency(counts, correction=False)
    total = int(counts.sum())
    denominator = total * max(min(counts.shape) - 1, 1)
    cramers_v = float(np.sqrt(float(chi2) / denominator)) if total else 0.0
    return {
        "passed": passed,
        "decision": "feature_audit_allowed" if passed else "structurally_unidentified",
        "eligible_families": [str(families[index]) for index in eligible],
        "cross_content_families": [str(families[index]) for index in cross_content],
        "largest_pairwise_comparable_clique": [
            str(families[index]) for index in largest_clique
        ],
        "comparable_cross_content_families": [
            str(families[index]) for index in comparable_cross_content
        ],
        "supported_content_cells": {
            str(families[index]): [
                str(contents[column])
                for column in range(len(contents))
                if supported[index, column]
            ]
            for index in range(len(families))
        },
        "family_totals": {
            str(families[index]): int(totals[index]) for index in range(len(families))
        },
        "family_content_cramers_v": cramers_v,
        "chi_square_p_value_descriptive_only": float(p_value),
        "requirements": {
            "minimum_family_samples": minimum_family_samples,
            "minimum_cell_samples": minimum_cell_samples,
            "minimum_supported_cells_per_family": minimum_supported_cells,
            "minimum_shared_cells": minimum_shared_cells,
            "minimum_multiclass_families": minimum_multiclass_families,
        },
    }


def matrix_records(
    families: Sequence[str], contents: Sequence[str], matrix: np.ndarray
) -> list[dict[str, Any]]:
    """Return stable JSON records for the support matrix."""
    counts = np.asarray(matrix, dtype=np.int64)
    return [
        {
            "family": str(family),
            "total": int(counts[index].sum()),
            "content_counts": {
                str(content): int(counts[index, column])
                for column, content in enumerate(contents)
            },
        }
        for index, family in enumerate(families)
    ]


def content_totals(content_labels: Mapping[str, str]) -> dict[str, int]:
    """Count frozen manual content labels."""
    return dict(sorted(Counter(str(value) for value in content_labels.values()).items()))
