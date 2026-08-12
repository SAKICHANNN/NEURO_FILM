import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4go_scan_domain_anchored_inverse_v1.json"


def test_p4go_changes_only_physical_scan_domain_endpoints():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert candidate["lower_anchor"] == [0.0, 0.0]
    assert candidate["upper_anchor"] == [1.0, 1.0]
    assert candidate["interior_knots_changed"] is False
    assert candidate["chart_pixels_used_for_fit"] is False
    assert candidate["extrapolation_allowed"] is False
    assert candidate["hard_clipping_allowed"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
