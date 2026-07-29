from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = (
    ROOT / "configs/u6_p8ao_native_end_to_end_phases_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8ao_decision_binds_repeat_stable_dominant_phase() -> None:
    decision = json.loads(DECISION.read_text())
    assert _sha256(ROOT / decision["contract"]) == decision[
        "contract_sha256"
    ]
    assert _sha256(ROOT / decision["formal_report"]) == decision[
        "formal_report_sha256"
    ]
    result = decision["result"]
    assert result["status"] == "pass-attribution"
    assert result["output_repeat_exact"]
    assert result["dominant_phase_repeat_exact"]
    assert result["dominant_phase"] == "ao6_native_display_seconds"
    assert result["minimum_dominant_phase_share"] >= result[
        "dominant_phase_share_gate"
    ]
    assert decision["production_default_changed"] is False
    assert decision["next_leaf"].startswith("U6.P8AP")
