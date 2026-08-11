from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.eval.monotone_fraction_transport_wide_ood import (
    MonotoneFractionTransportWideOodError,
    evaluate,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cb29_contract_uses_exact_screened_source_subset() -> None:
    contract = load_contract(
        ROOT / "configs/u5_r2cb29_monotone_fraction_transport_wide_ood_v1.json"
    )
    review = json.loads(
        (ROOT / contract["population"]["visual_review_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert contract["population"]["included_source_ids"] == review["eligible_ids"]
    assert contract["population"]["decision_field"] == "result.decision"
    assert len(contract["population"]["included_source_ids"]) == 10
    assert "canon_powershot_v1" not in contract["population"]["included_source_ids"]


def test_cb29_rejects_source_review_drift(tmp_path: Path) -> None:
    contract_path = (
        ROOT / "configs/u5_r2cb29_monotone_fraction_transport_wide_ood_v1.json"
    )
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    payload["population"]["included_source_ids"] = list(
        reversed(payload["population"]["included_source_ids"])
    )
    with pytest.raises(
        MonotoneFractionTransportWideOodError, match="source review drift"
    ):
        evaluate(payload, ROOT, tmp_path / "out")
