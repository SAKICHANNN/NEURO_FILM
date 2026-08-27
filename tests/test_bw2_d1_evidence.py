from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.inference import list_generic_bw_looks

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/BW2_D1_GENERIC_BW_LOOK_RESULT.json"


def test_bw2_d1_evidence_is_bound_to_current_product_correction() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "PASS_GENERIC_BW_LOOK_PRODUCT_TRUTH_CORRECTION"
    for binding in evidence["bindings"].values():
        path = ROOT / binding["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["sha256"]
    assert evidence["formal_probe"]["rows"][0]["identity_exact"] is True
    assert evidence["formal_probe"]["rows"][-1][
        "legacy_hp5_execution_parity_exact"
    ] is True
    assert evidence["compatibility"] == {
        "legacy_hp5_profile_preserved": True,
        "legacy_tri_x_400_profile_preserved": True,
        "legacy_recipe_replay_changed": False,
        "frozen_profile_or_asset_hash_changed": False,
    }
    assert list_generic_bw_looks()[0]["film_stock_id"] == "generic_black_and_white"
    assert evidence["multi_stock_completion"] is False
