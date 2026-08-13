from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.vision3_mtf_resolution_transfer import (
    ResolutionTransferError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u5_r2bu13_vision3_mtf_resolution_transfer_v1.json"


def test_bu13_formal_result_is_fail_closed_and_replay_exact() -> None:
    config = load_contract(CONTRACT)
    first = evaluate(config, ROOT)
    second = evaluate(config, ROOT)
    assert first == second
    assert first["automatic_pass"] is False
    assert first["decision"] == (
        "retain_source_and_processed_resolution_evidence_separately_and_forbid_stock_psf_compilation"
    )
    assert first["gate_results"]["fast_noninferiority_rate"] is False
    assert first["gate_results"]["no_pixel_fit_or_render"] is True


def test_bu13_rejects_protocol_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["comparison"]["frequencies_cycles_per_mm"] = [42.0, 53.0, 67.0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ResolutionTransferError, match="contract drift"):
        load_contract(path)
