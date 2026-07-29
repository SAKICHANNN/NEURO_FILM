from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u6_p8am_native_end_to_end_decision_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8am_decision_binds_formal_evidence() -> None:
    decision = json.loads(DECISION.read_text())
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    assert _sha256(ROOT / decision["formal_report"]) == decision[
        "formal_report_sha256"
    ]
    result = decision["result"]
    assert result["status"] == "pass"
    assert result["all_stage_partition_rows_byte_exact"]
    assert result["no_double_count_receipts_pass"]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AN")
