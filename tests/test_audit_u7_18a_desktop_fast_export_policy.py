from __future__ import annotations

import json
from pathlib import Path

from scripts import audit_u7_18a_desktop_fast_export_policy as audit

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_u7_18a_policy_matches_launcher_and_parent_evidence() -> None:
    config = json.loads(
        (ROOT / "configs/u7_18a_desktop_fast_export_policy_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["node_id"] == "U7.18A"
    assert config["policies"] == {
        "preview": {"tile_size": 256, "tile_workers": 1},
        "baseline_export": {"tile_size": 256, "tile_workers": 1},
        "candidate_export": {"tile_size": 512, "tile_workers": 8},
        "png_compression": 6,
    }
    assert audit.DESKTOP_EXPORT_TILE_SIZE == 512
    assert audit.DESKTOP_EXPORT_TILE_WORKERS == 8
    assert config["gates"]["maximum_candidate_to_baseline_median_wall_ratio"] == 0.85
    assert config["primary"]["effects"] == {
        "grain": 0.05,
        "halation": 0.15,
        "dust": 0.02,
        "seed": 7,
    }
    assert set(config["parent_bindings"]) == {
        "product_desktop",
        "desktop_launcher",
        "u7_2b_evidence",
        "u7_2d_evidence",
        "u7_17a_evidence",
    }


def test_recipe_normalization_changes_only_output_identity() -> None:
    recipe = {
        "render": {"style": "ektar_100", "look_amount": 0.65},
        "output": {"path": "one.png", "sha256": "1" * 64, "bit_depth": 16},
    }
    normalized = audit._normalized_recipe(recipe)
    assert normalized == {
        "render": {"style": "ektar_100", "look_amount": 0.65},
        "output": {
            "path": "<OUTPUT>",
            "sha256": "<OUTPUT_SHA256>",
            "bit_depth": 16,
        },
    }
    assert recipe["output"]["path"] == "one.png"
