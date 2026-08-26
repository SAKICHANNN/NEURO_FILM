from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_6i_three_stock_native_pointwise import (
    CONTRACT_SCHEMA,
    REPORT_SCHEMA,
    _load_contract,
    evaluate,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u7_6i_three_stock_native_pointwise_v1.json"


def test_u7_6i_contract_is_bound_and_pointwise_only() -> None:
    payload, digest = _load_contract(CONTRACT)
    assert payload["schema"] == CONTRACT_SCHEMA
    assert len(digest) == 64
    assert payload["styles"] == ["velvia_50", "portra_400", "ektar_100"]
    assert payload["execution"]["full_renderer_integration_allowed"] is False
    assert "not a complete renderer" in payload["claim_ceiling"]


def test_u7_6i_formal_native_pointwise_gate(tmp_path: Path) -> None:
    report = evaluate(CONTRACT, tmp_path / "report.json")
    assert report["schema"] == REPORT_SCHEMA
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["observations"]["maximum_lab_absolute_error"] == 0.0
    assert len({row["output_sha256"] for row in report["workers"][0]["rows"]}) == 3
    assert json.loads((tmp_path / "report.json").read_text("utf-8")) == report
