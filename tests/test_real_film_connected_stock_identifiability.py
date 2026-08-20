from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.real_film.connected_stock_identifiability import (
    ConnectedStockIdentifiabilityError,
    group_label_permutation_test,
    group_loo_centroid,
    load_connected_rows,
    paired_group_bootstrap_delta,
)


def _separable() -> tuple[np.ndarray, list[str], list[str]]:
    groups = [f"a{i}" for i in range(5) for _ in range(2)] + [f"b{i}" for i in range(5) for _ in range(2)]
    labels = ["a"] * 10 + ["b"] * 10
    values = [[i * 0.02] for i in range(5) for _ in range(2)] + [[10 + i * 0.02] for i in range(5) for _ in range(2)]
    return np.asarray(values), groups, labels


def test_group_loo_centroid_uses_equal_author_group_votes() -> None:
    features, groups, labels = _separable()
    result = group_loo_centroid(features, groups, labels)
    assert result["held_out_author_groups"] == 10
    assert result["author_label_units"] == 10
    assert result["balanced_accuracy"] == 1.0
    assert result["per_class_group_recall"] == {"a": 1.0, "b": 1.0}


def test_group_permutation_is_deterministic() -> None:
    features, groups, labels = _separable()
    first = group_label_permutation_test(features, groups, labels, permutations=31, seed=7)
    second = group_label_permutation_test(features, groups, labels, permutations=31, seed=7)
    assert first == second
    assert first["observed_balanced_accuracy"] == 1.0


def test_paired_bootstrap_reports_positive_primary_delta() -> None:
    features, groups, labels = _separable()
    primary = group_loo_centroid(features, groups, labels)
    control = group_loo_centroid(np.zeros_like(features), groups, labels)
    first = paired_group_bootstrap_delta(primary, control, iterations=200, seed=9)
    second = paired_group_bootstrap_delta(primary, control, iterations=200, seed=9)
    assert first == second
    assert first["observed_delta"] > 0
    assert first["ci95_low"] > 0


def test_multilabel_author_is_held_out_as_one_split_group() -> None:
    features, groups, labels = _separable()
    features = np.concatenate([features, np.asarray([[0.05], [10.05]])])
    groups = [*groups, "paired", "paired"]
    labels = [*labels, "a", "b"]
    result = group_loo_centroid(features, groups, labels)
    paired = [row for row in result["predictions"] if row["held_out_author_group"] == "paired"]
    assert result["held_out_author_groups"] == 11
    assert result["author_label_units"] == 12
    assert {row["true_label"] for row in paired} == {"a", "b"}
    assert all(row["correct"] for row in paired)


def test_load_connected_rows_applies_exact_page_allowlist(tmp_path: Path) -> None:
    pixel_root = tmp_path / "pixels"
    pixel_root.mkdir()
    manifest_rows = []
    for page_id in (11, 12, 13):
        path = pixel_root / f"{page_id}.png"
        Image.new("RGB", (2, 2), (page_id, 0, 0)).save(path)
        manifest_rows.append({
            "page_id": page_id,
            "film_stock_id": "stock-a",
            "normalized_author_group": f"author-{page_id}",
            "local_path": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 2,
            "height": 2,
        })
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"rows": manifest_rows}), encoding="utf-8")
    cell = {
        "cell_id": "clean-stock-a",
        "source_id": "source",
        "film_stock_id": "stock-a",
        "manifest": manifest_path.name,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "pixel_root": pixel_root.name,
        "expected_files": 2,
        "included_page_ids": [13, 11],
    }
    rows = load_connected_rows(tmp_path, {"cells": [cell]})
    assert [row["source_record"]["page_id"] for row in rows] == [13, 11]

    cell["included_page_ids"] = [13, 99]
    with pytest.raises(ConnectedStockIdentifiabilityError, match="included page id drift"):
        load_connected_rows(tmp_path, {"cells": [cell]})
