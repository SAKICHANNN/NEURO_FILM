from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.eval.fgaesq_consumed_preference_concordance import (
    _decode_png,
    _resolve_within,
    _sign,
    _truth_winner,
    audit_score_lock_mechanics,
    image_statistic_controls,
    load_public_round_one,
    official_confidence_adjustment,
)


def test_confidence_adjustment_matches_official_large_series_weights() -> None:
    adjusted = official_confidence_adjustment(
        {
            "A": [1.0],
            "B": [2.0, 4.0],
            "C": [3.0, 6.0, 9.0],
            "D": [5.0, 7.0, 8.0],
        },
        item_count=4,
    )
    series_mean = np.mean([1.0, 2.0, 4.0, 3.0, 6.0, 9.0, 5.0, 7.0, 8.0])
    assert adjusted["A"] == 0.3 * 1.0 + 0.7 * series_mean
    assert adjusted["B"] == 0.7 * 3.0 + 0.3 * series_mean
    assert adjusted["C"] == 0.9 * 6.0 + 0.1 * series_mean


def test_small_series_uses_unadjusted_pair_mean() -> None:
    assert official_confidence_adjustment({"A": [2.0], "B": [8.0]}, item_count=2) == {
        "A": 2.0,
        "B": 8.0,
    }


def test_statistic_controls_are_finite_and_chroma_sensitive() -> None:
    gray = Image.fromarray(np.full((8, 8, 3), 128, dtype=np.uint8), "RGB")
    red = Image.fromarray(
        np.tile(np.array([255, 0, 0], dtype=np.uint8), (8, 8, 1)), "RGB"
    )
    gray_stats = image_statistic_controls(gray)
    red_stats = image_statistic_controls(red)
    assert gray_stats["mean_oklab_chroma"] < red_stats["mean_oklab_chroma"]
    assert gray_stats["linear_srgb_rms_luminance_contrast"] < 1e-15
    assert all(np.isfinite(list(red_stats.values())))


def test_decode_rejects_extension_format_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    Image.new("RGB", (4, 4), "red").save(path, format="JPEG")
    try:
        _decode_png(path)
    except ValueError as error:
        assert "extension/format mismatch" in str(error)
    else:
        raise AssertionError("format mismatch must fail closed")


def test_resolve_within_rejects_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    try:
        _resolve_within(root, "../escape.png")
    except ValueError as error:
        assert "escapes root" in str(error)
    else:
        raise AssertionError("root escape must fail closed")


def test_public_inventory_reads_only_round_one_png(tmp_path: Path) -> None:
    for label, colour in (("A", "red"), ("B", "blue")):
        asset = tmp_path / "assets" / "round_1" / "R1-01" / f"{label}.png"
        asset.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8), colour).save(asset)
    manifest = {
        "rows": [
            {
                "round": 1,
                "presentation_id": "R1-01",
                "role": "test",
                "labels": ["A", "B"],
                "assets": {
                    "A": "assets/round_1/R1-01/A.png",
                    "B": "assets/round_1/R1-01/B.png",
                },
            },
            {
                "round": 2,
                "presentation_id": "R2-01",
                "role": "forbidden",
                "labels": ["A", "B"],
                "assets": {},
            },
        ]
    }
    (tmp_path / "public_manifest.json").write_text(json.dumps(manifest), "utf-8")
    rows = load_public_round_one(
        dataset_id="test",
        root=tmp_path,
        expected_labels=("A", "B"),
        expected_sources=1,
    )
    assert len(rows) == 1
    assert rows[0]["presentation_id"] == "R1-01"
    assert sorted(rows[0]["hashes"]) == ["A", "B"]


def test_truth_winner_requires_one_directed_edge() -> None:
    row = {
        "pairwise_winners": {
            "candidate": ["identity"],
            "identity": [],
        }
    }
    assert _truth_winner(row, "candidate", "identity") == "candidate"
    row["pairwise_winners"]["identity"] = ["candidate"]
    try:
        _truth_winner(row, "candidate", "identity")
    except ValueError as error:
        assert "non-binary" in str(error)
    else:
        raise AssertionError("contradictory truth must fail closed")


def test_sign_has_exact_frozen_tie_band() -> None:
    assert _sign(1.0, 0.0) == 1
    assert _sign(0.0, 1.0) == -1
    assert _sign(1.0 + 1e-6, 1.0) == 0


def test_mechanics_audit_closes_order_dependent_scores(tmp_path: Path) -> None:
    config = {
        "gates": {"fresh_process_score_max_abs_error_max": 1e-6},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, sort_keys=True), "utf-8")
    import hashlib

    config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
    rows = []
    for dataset, labels in (("p401", ["A", "B"]), ("p402", ["A", "B", "C", "D"])):
        for index in range(12):
            canonical = {
                label: float(position) for position, label in enumerate(labels)
            }
            reversed_scores = dict(canonical)
            if dataset == "p401" and index == 0:
                reversed_scores = {"A": 2.0, "B": 0.0}
            rows.append(
                {
                    "dataset_id": dataset,
                    "presentation_id": f"R1-{index + 1:02d}",
                    "series_scores": canonical,
                    "reversed_series_scores": reversed_scores,
                    "single_scores": canonical,
                }
            )
    lock = {
        "config_sha256": config_sha,
        "private_mapping_reads": 0,
        "direct_result_reads": 0,
        "rows": rows,
    }
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for path in paths:
        path.write_text(json.dumps(lock, sort_keys=True), "utf-8")
    result = audit_score_lock_mechanics(
        config_path=config_path,
        score_lock_paths=paths,
    )
    assert result["status"] == "INVALID_MECHANICS_ORDER_DEPENDENT_FGAESQ_SERIES"
    assert result["metrics"]["pair_sign_total"] == 84
    assert result["metrics"]["pair_sign_match_count"] == 83
    assert result["scientific_preference_metrics_computed"] is False
