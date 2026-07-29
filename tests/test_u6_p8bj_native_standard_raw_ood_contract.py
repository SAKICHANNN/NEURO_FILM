from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p8bj_native_standard_raw_ood_v1.json"
DECISION = (
    ROOT / "configs/u6_p8bj_native_standard_raw_ood_decision_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_p8bj_decision_binds_exact_contract_and_visual_evidence() -> None:
    decision = json.loads(DECISION.read_text())
    assert decision["contract_sha256"] == _sha256(CONTRACT)
    assert decision["contact_sheet_sha256"] == (
        "b870bcde0255256729bb789b411ca9e8eaa41bf9c18d494fbb6593cd82f95b94"
    )
    result = decision["result"]
    assert result["status"] == "pass_limited"
    assert result["automatic_gate_pass"]
    assert result["transactions_verified"] == 4
    assert result["output_code_boundary_fraction"] == [0.0] * 4
    visual = result["autonomous_visual_review"]
    assert visual["confirmed_severe_artifact_count"] == 0
    assert visual["severe_artifact_gate_pass"]


def test_p8bj_does_not_promote_or_overclaim_raw_ood_result() -> None:
    decision = json.loads(DECISION.read_text())
    assert not decision["production_default_changed"]
    assert "not population preference" in decision["claim_ceiling"]
    assert "stock response" in decision["claim_ceiling"]
    assert "face-skin-text-highlight" in decision["next_leaf"]
