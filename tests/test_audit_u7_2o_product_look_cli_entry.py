from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_2o_product_look_cli_entry import build_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2o_product_look_cli_entry_v1.json"


def test_u7_2o_report_passes_all_frozen_gates() -> None:
    report = build_report(CONFIG, "reverse")
    assert report["status"] == "PASS"
    assert len(report["comparisons"]) == 9
    assert all(report["gates"].values())
    assert json.loads(CONFIG.read_text(encoding="utf-8"))["decision"] == "prospective"
