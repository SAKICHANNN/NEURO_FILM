from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_19c_product_bounded_memory_policy import (
    _canonical_bytes,
    _scientific_view,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_19c_product_bounded_memory_policy_v1.json"


def test_frozen_contract_has_one_non_rescuable_policy() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["status"] == "FROZEN_BEFORE_U7_19C_X2D_RENDER"
    assert config["policy"] == {
        "profile_id": "safe-rich-product-v1",
        "default_tile_size": 256,
        "default_tile_workers": 1,
        "explicit_policy_remains_authoritative": True,
        "recipe_semantics_unchanged": True,
    }
    assert [
        (row["extension"], row["height"], row["width"]) for row in config["sources"]
    ] == [
        (".3fr", 8842, 11904),
        (".fff", 8842, 11904),
    ]
    assert config["limits"]["peak_process_tree_rss_bytes"] == 16 * 1024**3
    assert "only candidate" in config["stop_rule"]


def test_canonical_report_and_scientific_view_ignore_only_execution_noise() -> None:
    report = {
        "execution_commit": "abc",
        "order": "forward",
        "records": [
            {
                "source_id": "x",
                "resource": {"wall_seconds": 1.0, "peak_process_tree_rss_bytes": 2},
                "value": 3,
            }
        ],
    }
    assert _scientific_view(report) == {"records": [{"source_id": "x", "value": 3}]}
    assert _canonical_bytes({"z": 2, "a": 1}) == b'{\n  "a": 1,\n  "z": 2\n}\n'
