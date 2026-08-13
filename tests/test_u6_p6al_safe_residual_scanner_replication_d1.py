from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.safe_residual_scanner_replication_d1 import load_contract

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u6_p6al_safe_residual_scanner_replication_d1_v1.json"


def test_contract_freezes_p6g_fit_and_p6ak_execution() -> None:
    contract = load_contract(CONFIG)
    assert contract["execution"]["fit"].startswith("exact_p6g")
    assert contract["execution"]["operator"].startswith("p6ak-")
    assert contract["execution"]["hard_clipping_allowed"] is False


def test_contract_rejects_fit_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG.read_text(encoding="utf-8"))
    payload["execution"]["fit"] = "refit"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        load_contract(path)
