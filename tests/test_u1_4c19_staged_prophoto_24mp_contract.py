from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_u1_4c19_staged_prophoto_24mp import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c19_staged_prophoto_24mp_v1.json"


def test_c19_contract_preserves_c18_output_and_resource_gates() -> None:
    payload, digest = load_contract(CONTRACT)
    assert digest
    assert payload["candidate"]["row_chunk"] == 128
    assert payload["gates"]["required_output_sha256"] == (
        "ff30c351261c0c9004c05f21e5a2929e448716153756407cde7e36225509bd7a"
    )
    assert payload["gates"]["maximum_peak_process_tree_rss_bytes"] == 2**30
    assert payload["gates"]["maximum_worker_wall_seconds"] == 15.0


def test_c19_contract_rejects_row_chunk_rescue(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["row_chunk"] = 256
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="execution drift"):
        load_contract(changed)
