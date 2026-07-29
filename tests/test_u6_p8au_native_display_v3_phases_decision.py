from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u6_p8au_native_display_v3_phases_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8au_decision_selects_repeat_stable_display_phase() -> None:
    decision = json.loads(DECISION.read_text())
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    assert _sha256(ROOT / decision["formal_report"]) == decision[
        "formal_report_sha256"
    ]
    report = json.loads((ROOT / decision["formal_report"]).read_text())
    assert report["stable_evidence_id"] == decision["stable_evidence_id"]
    result = decision["result"]
    assert result["status"] == "pass-attribution"
    assert result["output_repeat_and_v1_exact"] is True
    assert result["dominant_phase"] == "ao6_native_display_v3_seconds"
    assert result["minimum_dominant_phase_share"] >= result[
        "dominant_phase_share_gate"
    ]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AV")
