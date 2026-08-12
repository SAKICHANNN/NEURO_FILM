from __future__ import annotations

import json
from pathlib import Path

from scripts.benchmark_u6_p4fr_native_standard_replayable_24mp import _input_sha

ROOT = Path(__file__).resolve().parents[1]


def test_p4fz_contract_and_input_are_deterministic() -> None:
    contract = json.loads(
        (ROOT / "configs/u6_p4fz_cloud_standard_display_24mp_v1.json").read_text()
    )
    scenario = contract["scenario"]
    assert scenario["height"] * scenario["width"] == 24_000_000
    assert _input_sha(16, 24, 8) == _input_sha(16, 24, 8)
    assert contract["gates"]["require_source_passes"] == 4
    assert contract["claim_ceiling"].endswith("product promotion.")


def test_p4fz_frozen_result_passes_all_resource_gates() -> None:
    result = json.loads(
        (ROOT / "docs/evidence/U6_P4FZ_CLOUD_STANDARD_DISPLAY_24MP_RESULT.json").read_text()
    )
    assert result["status"] == "PASS"
    assert result["gates"]["all_pass"] is True
    assert result["metrics"]["source_passes"] == 4
    assert len(set(result["metrics"]["peak_process_tree_rss_bytes"])) == 2
