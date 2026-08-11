import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb44_constrained_affine_tone_source_decision_v1.json"


def test_cb44_source_role_is_explicitly_consumed_development_only() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["status"] == "source_eligible_fixed_comparison_open"
    assert len(decision["included_source_ids"]) == 12
    assert len(set(decision["included_source_ids"])) == 12
    assert decision["camera_make_count_exact"] == 12
    assert decision["previously_consumed_mechanism_development_role"] is True
    assert decision["operator_outputs_inspected_for_this_role"] is False
