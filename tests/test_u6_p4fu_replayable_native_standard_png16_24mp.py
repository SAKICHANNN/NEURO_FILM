from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_p4fu_contract_freezes_complete_24mp_transaction() -> None:
    contract = json.loads(
        (
            ROOT
            / "configs/u6_p4fu_replayable_native_standard_png16_24mp_v1.json"
        ).read_text()
    )
    assert contract["scenario"]["height"] * contract["scenario"]["width"] == 24_000_000
    assert contract["gates"]["require_exact_png_repeat"]
    assert contract["gates"]["require_restart_verification"]
    assert contract["gates"]["require_cleanup"]
