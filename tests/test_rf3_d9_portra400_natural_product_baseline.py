from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.portra400_natural_product_baseline import (
    Portra400NaturalProductBaselineError,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/rf3_d9_portra400_natural_product_baseline_v1.json"


def test_rf3_d9_contract_rejects_drift(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["sources"]["cham10_evidence"]["sha256"] = "0" * 64
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Portra400NaturalProductBaselineError, match="source identity drift"):
        evaluate(root=ROOT, contract_path=path)


def test_rf3_d9_natural_product_baseline_is_stable() -> None:
    forward = evaluate(root=ROOT, contract_path=CONTRACT)
    reverse = evaluate(root=ROOT, contract_path=CONTRACT, reverse=True)
    assert forward == reverse
    assert len(forward["rows"]) == 2
    assert all(row["eligible_correspondences"] >= 32 for row in forward["rows"])
    assert forward["decision"] in {
        "retain_current_portra_400_as_same_session_natural_baseline_pending_independent_roll_scanner_confirmation",
        "record_current_portra_400_natural_transfer_gap_and_require_new_controlled_observations",
    }
