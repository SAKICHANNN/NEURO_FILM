from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_u7_2n_product_auxiliary_output_transaction import build_report

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/u7_2n_product_auxiliary_output_transaction_v1.json"


def test_formal_report_passes_all_frozen_gates() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    report = build_report(
        config_path=CONFIG,
        order=tuple(config["available_product_looks"]),
    )
    assert report["status"] == "PASS"
    assert all(report["gates"].values())
    assert report["owned_runtime_residue_zero"] is True
