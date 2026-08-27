from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/U4_5A_THREE_STOCK_WINDOWS_CPU_SCALE_MATRIX_RESULT.json"


def test_u4_5a_evidence_preserves_mechanical_pass_and_product_gap() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_MECHANICS_PRODUCT_TARGETS_OPEN"
    assert evidence["gates"] == {
        "all_mechanical": True,
        "all_provisional_product_targets": False,
        "owned_residue_empty": True,
    }
    assert [row["tier_id"] for row in evidence["tiers"]] == [
        "preview_1mp",
        "standard_12mp",
        "desktop_24mp",
    ]
    assert all(row["mechanical_pass"] for row in evidence["tiers"])
    assert not any(row["provisional_product_targets_pass"] for row in evidence["tiers"])
    for tier in evidence["tiers"]:
        assert tier["repeat_wall_ratio"] <= 1.15
        assert tier["runs"][0]["output_hashes"] == tier["runs"][1]["output_hashes"]
        assert len(set(tier["runs"][0]["output_hashes"].values())) == 3
