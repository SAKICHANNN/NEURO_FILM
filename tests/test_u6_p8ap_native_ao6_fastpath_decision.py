from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u6_p8ap_native_ao6_fastpath_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ap_decision_binds_exact_pass_report() -> None:
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
    for key in (
        "context_v1_v2_byte_exact",
        "base_v1_v2_byte_exact",
        "display_v1_v2_byte_exact",
    ):
        assert decision["result"][key] is True
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AQ")
