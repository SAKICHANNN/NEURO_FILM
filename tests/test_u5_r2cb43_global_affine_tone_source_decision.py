import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb43_global_affine_tone_source_decision_v1.json"


def test_cb43_source_role_is_exact_and_disjoint() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["status"] == "source_eligible_fixed_comparison_open"
    assert len(decision["included_source_ids"]) == 7
    assert len(set(decision["included_source_ids"])) == 7
    assert decision["camera_make_count_exact"] == 7
    assert decision["exact_decoded_sha_overlap_with_cb27_through_cb42"] == 0
    assert decision["operator_outputs_inspected_for_this_role"] is False
