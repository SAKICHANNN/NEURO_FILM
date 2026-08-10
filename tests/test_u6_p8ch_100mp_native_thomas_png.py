from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8ch_100mp_native_thomas_png_v1.json"


def test_p8ch_freezes_complete_100mp_row_streamed_png_target() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["status"] == "contract_frozen_implementation_ready"
    assert contract["candidate"]["shape_chw"] == [3, 10_000, 10_000]
    assert contract["candidate"]["row_partition"] == 128
    assert contract["candidate"]["bit_depth"] == 16
    assert contract["candidate"]["full_output_allowed"] is False


def test_p8ch_freezes_provisional_desktop_limits_without_claim_expansion() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["gates"]["maximum_wall_seconds"] == 75.0
    assert contract["gates"]["maximum_process_tree_rss_bytes"] == 4_000_000_000
    assert "100MP" in contract["claim_ceiling"]
    assert "not mobile" in contract["claim_ceiling"]
