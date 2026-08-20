from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_u1_4c29_streaming_staged_prophoto_v3_24mp import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c32_spilled_context_staged_prophoto_24mp_v1.json"


def test_c32_contract_binds_spilled_context_execution() -> None:
    payload, contract_sha256 = load_contract(CONTRACT, root=ROOT)
    assert payload["experiment_id"] == "U1.4C32"
    assert payload["candidate"]["spill_mapped_for_context"] is True
    assert payload["candidate"]["preprocess_workers"] == 2
    assert payload["gates"]["maximum_peak_process_tree_rss_bytes"] == 1_073_741_824
    assert payload["gates"]["maximum_worker_wall_seconds"] == 15.0
    assert len(contract_sha256) == 64


def test_c32_contract_rejects_missing_spilled_context(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["spill_mapped_for_context"] = False
    mutated = tmp_path / "mutated.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen execution drift"):
        load_contract(mutated, root=ROOT)
