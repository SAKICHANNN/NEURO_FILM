from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.real_film.portra_three_source_heldout import (
    fit_centroid_predict,
    primary_selection,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/sf3_a3u_portra_three_source_heldout_v1.json"


def test_primary_selection_excludes_old_portra_generation_and_is_balanced() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    rows = primary_selection(config)
    assert len(rows) == 24
    assert [row["index"] for row in rows if row["label"] == "portra"] == list(
        range(39, 51)
    )
    assert sorted(row["index"] for row in rows if row["label"] == "wrong_stock") == [
        12,
        14,
        19,
        21,
        22,
        24,
        26,
        28,
        29,
        30,
        32,
        33,
    ]
    assert not set(range(1, 11)) & {row["index"] for row in rows}


def test_centroid_prediction_uses_development_only_geometry() -> None:
    development = np.asarray(
        [[-2.0, -1.0], [-1.0, -2.0], [2.0, 1.0], [1.0, 2.0]],
        dtype=np.float64,
    )
    development_labels = ["portra", "portra", "wrong_stock", "wrong_stock"]
    confirmation = np.asarray([[-3.0, -2.0], [3.0, 2.0]], dtype=np.float64)
    result = fit_centroid_predict(
        development,
        development_labels,
        confirmation,
        ["portra", "wrong_stock"],
    )
    assert result["balanced_accuracy"] == 1.0
    assert [row["predicted_label"] for row in result["predictions"]] == [
        "portra",
        "wrong_stock",
    ]


def test_contract_keeps_luminant_conditional_and_operator_closed() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert (
        config["conditional_positive_confirmation"]["body_reads_before_primary_pass"]
        == 0
    )
    assert config["operation_limits"]["excluded_nicknick_image_body_requests"] == 0
    assert config["operator_fitting_allowed"] is False
    assert config["training_allowed"] is False
