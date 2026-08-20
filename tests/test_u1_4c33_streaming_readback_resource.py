from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_u1_4c29_streaming_staged_prophoto_v3_24mp import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c33_streaming_readback_staged_prophoto_24mp_v1.json"


def test_c33_contract_binds_streaming_postpublication_readback() -> None:
    payload, contract_sha256 = load_contract(CONTRACT, root=ROOT)
    assert payload["experiment_id"] == "U1.4C33"
    assert (
        payload["candidate"]["postpublication_sample_readback"]
        == "strict-bounded-streaming-rgb16-png"
    )
    assert len(contract_sha256) == 64


def test_c33_contract_rejects_readback_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["postpublication_sample_readback"] = "full-frame"
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen execution drift"):
        load_contract(changed, root=ROOT)
