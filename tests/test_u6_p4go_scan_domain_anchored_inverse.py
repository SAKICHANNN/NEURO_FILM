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


def test_p4go_formal_result_stops_before_chart():
    evidence = json.loads(
        (
            ROOT
            / "docs/evidence/U6_P4GO_SCAN_DOMAIN_ANCHORED_INVERSE_RESULT.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["maximum_error_gate_pass"] is True
    assert evidence["p95_error_gate_pass"] is False
    assert evidence["confirmation_p95_absolute_error"] > 0.005
    assert evidence["chart_pixels_read"] is False
    assert evidence["automatic_pass"] is False
    assert evidence["decision"] == "close_scan_domain_anchored_inverse_v1"
