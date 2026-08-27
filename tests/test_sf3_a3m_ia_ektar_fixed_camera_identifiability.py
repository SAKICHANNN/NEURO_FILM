from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.real_film.internet_archive_fixed_camera_stock import (
    _exact_binary_permutation,
    _rank,
    canonical_json,
)


def test_rank_and_canonical_json_are_deterministic() -> None:
    assert _rank("experiment", "item", "a.jpg", "abcd") == _rank("experiment", "item", "a.jpg", "abcd")
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}\n'


def test_exact_binary_permutation_enumerates_all_fifteen_assignments() -> None:
    groups = [f"g{index}" for index in range(6)]
    labels = ["ektar", "ektar", "control", "control", "control", "control"]
    features = np.asarray([[0.0], [0.1], [1.0], [1.1], [1.2], [1.3]], dtype=np.float64)
    from src.real_film.connected_stock_identifiability import group_loo_centroid

    observed = group_loo_centroid(features, groups, labels)["balanced_accuracy"]
    result = _exact_binary_permutation(features, groups, labels, observed)
    assert result["assignments"] == 15
    assert 0.0 < result["p_value_greater_equal"] <= 1.0


def test_project_contract_is_frozen_before_pixels() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs/sf3_a3m_ia_ektar_fixed_camera_identifiability_v1.json").read_text(encoding="utf-8"))
    assert len(config["items"]) == 6
    assert {row["film_stock_id"] for row in config["items"]} == {
        "kodak_ektar_100",
        "kodak_pro_image_100",
        "kodak_ultramax_400",
    }
    assert config["selection"]["frames_per_item"] == 12
    assert config["gates"]["minimum_ektar_recall"] == 1.0
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
