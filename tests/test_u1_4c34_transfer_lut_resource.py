from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.benchmark_u1_4c29_streaming_staged_prophoto_v3_24mp import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c34_transfer_lut_staged_prophoto_24mp_v1.json"


def test_c34_contract_binds_prepared_rgb16_transfer_lut() -> None:
    payload, contract_sha256 = load_contract(CONTRACT, root=ROOT)
    assert payload["experiment_id"] == "U1.4C34"
    assert payload["candidate"]["prepared_rgb16_transfer_lut"] is True
    assert len(contract_sha256) == 64


def test_c34_contract_rejects_transfer_lut_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["candidate"]["prepared_rgb16_transfer_lut"] = False
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen execution drift"):
        load_contract(changed, root=ROOT)
