from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8cg_24mp_native_thomas_png_v1.json"


def test_p8cg_freezes_real_24mp_complete_png_target() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["candidate"]["shape_chw"] == [3, 4000, 6000]
    assert contract["candidate"]["bit_depth"] == 16
    assert contract["candidate"]["full_output_allowed"] is False


def test_p8cg_uses_provisional_ordinary_cpu_limits_without_claim_expansion() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["gates"]["maximum_wall_seconds"] == 15.0
    assert contract["gates"]["maximum_process_tree_rss_bytes"] == 1_000_000_000
    assert "One measured Windows" in contract["claim_ceiling"]
