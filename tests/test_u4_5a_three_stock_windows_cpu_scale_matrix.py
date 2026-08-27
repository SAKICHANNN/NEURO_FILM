from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.audit_u4_5a_three_stock_windows_cpu_scale_matrix import (
    U45AError,
    summarize_tier,
)

ROOT = Path(__file__).resolve().parents[1]


def test_u4_5a_contract_freezes_three_product_scales() -> None:
    config = json.loads(
        (ROOT / "configs/u4_5a_three_stock_windows_cpu_scale_matrix_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert [row["maximum_pixels"] for row in config["tiers"]] == [
        1_000_000,
        12_000_000,
        24_000_000,
    ]
    assert config["render"]["runs_per_tier"] == 2
    assert config["render"]["look_amount"] == 1.0


def test_u4_5a_summary_separates_mechanics_from_product_targets() -> None:
    tier = {
        "tier_id": "example",
        "maximum_pixels": 100,
        "provisional_maximum_wall_seconds": 1.0,
        "provisional_maximum_process_tree_rss_bytes": 1000,
    }
    hashes = {"velvia_50": "a", "portra_400": "b", "ektar_100": "c"}
    runs = [
        {
            "wall_seconds": 2.0,
            "peak_process_tree_rss_bytes": 900,
            "width": 10,
            "height": 10,
            "pixels": 100,
            "output_hashes": hashes,
            "three_outputs_distinct": True,
        },
        {
            "wall_seconds": 2.1,
            "peak_process_tree_rss_bytes": 950,
            "width": 10,
            "height": 10,
            "pixels": 100,
            "output_hashes": hashes,
            "three_outputs_distinct": True,
        },
    ]
    summary = summarize_tier(tier, runs, 1.15)
    assert summary["mechanical_pass"] is True
    assert summary["provisional_product_targets_pass"] is False


def test_u4_5a_summary_requires_two_runs() -> None:
    with pytest.raises(U45AError, match="exactly two"):
        summarize_tier({}, [], 1.15)
