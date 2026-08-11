import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "configs/u5_r2cb41_characteristic_lstar_source_decision_v1.json"


def test_cb41_source_role_is_exact_and_disjoint() -> None:
    decision = json.loads(DECISION.read_text(encoding="utf-8"))
    assert decision["status"] == "source_eligible_fixed_comparison_open"
    assert len(decision["included_source_ids"]) == 17
    assert len(set(decision["included_source_ids"])) == 17
    assert decision["camera_make_count_exact"] == 9
    assert decision["exact_decoded_sha_overlap_with_cb27_through_cb40"] == 0
    assert decision["operator_outputs_inspected_for_this_role"] is False
    assert decision["excluded_parent_rows"] == {
        "fujifilm_s2pro": (
            "visible control target excluded by the frozen AI1S selection rule"
        )
    }
