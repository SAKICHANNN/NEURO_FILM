from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u6_p8av_native_ao6_display_v4_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8av_decision_binds_exact_pass_report() -> None:
    decision = json.loads(DECISION.read_text())
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    assert _sha256(ROOT / decision["formal_report"]) == decision[
        "formal_report_sha256"
    ]
    report = json.loads((ROOT / decision["formal_report"]).read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    assert decision["result"]["status"] == "pass"
    assert decision["result"]["display_v3_v4_byte_exact"] is True
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AW")
