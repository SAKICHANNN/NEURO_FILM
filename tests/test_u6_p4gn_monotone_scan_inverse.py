import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gn_monotone_scan_inverse_v1.json"


def test_p4gn_freezes_independent_monotone_inverse_before_chart():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    compiler = contract["compiler"]
    assert compiler["build_levels"] == 33
    assert compiler["confirmation_midpoints"] == 32
    assert compiler["chart_pixels_used_for_fit"] is False
    assert compiler["extrapolation_allowed"] is False
    assert compiler["hard_clipping_allowed"] is False
    assert contract["execution"]["stop_before_chart_on_calibration_failure"] is True
    assert contract["execution"]["post_result_retuning_allowed"] is False
