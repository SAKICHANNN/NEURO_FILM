from __future__ import annotations

import json
from pathlib import Path

from src.eval.scan_signal_normalization import SCHEMA, evaluate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p2i_scan_signal_normalization_v1.json"


def test_frozen_scan_normalization_evaluator_is_exact() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["schema"] == SCHEMA
    first = evaluate(contract, root=ROOT)
    second = evaluate(contract, root=ROOT)
    assert first == second
    assert first["automatic_pass"] is True
    assert all(first["checks"].values())
    assert first["normalization"]["calibrated"] is False
    assert first["normalization"]["production_eligible"] is False
