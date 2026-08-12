import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gi_cloud_profile_bound_neutral_gauge_v1.json"


def test_p4gi_contract_keeps_chart_out_of_fit_and_freezes_stop_rules():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["execution"]["chart_pixels_used_for_fit"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["calibration"]["layout"] == "vertical-neutral-levels"
    assert contract["confirmation"]["layout"] == "horizontal-neutral-midpoints"
    assert contract["gates"]["maximum_confirmation_neutral_chroma_p99"] == 0.09
    assert contract["decision_if_fail"] == "close_cloud_profile_bound_neutral_gauge_v1"
