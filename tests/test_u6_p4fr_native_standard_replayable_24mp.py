from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p4fr_native_standard_replayable_24mp import _input_sha

ROOT = Path(__file__).resolve().parents[1]


def test_p4fr_contract_and_input_are_deterministic() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4fr_native_standard_replayable_24mp_v1.json").read_text()
    )
    scenario = contract["scenario"]
    assert scenario["height"] * scenario["width"] == 24_000_000
    assert _input_sha(16, 24, 8) == _input_sha(16, 24, 8)
    assert contract["claim_ceiling"].endswith("product promotion.")
