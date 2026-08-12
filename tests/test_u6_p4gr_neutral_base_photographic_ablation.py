import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gr_neutral_base_photographic_ablation_v1.json"


def test_p4gr_freezes_fresh_photographic_ablation_before_execution():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["source"]["expected_rows"] == 9
    assert contract["source"]["expected_camera_makes"] == 9
    assert contract["source"]["required_rights_scope"].startswith("CC0_")
    assert contract["candidate"]["arms"] == [
        "source",
        "ao6-only",
        "physical-residual-only",
        "physical-residual-plus-ao6",
    ]
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["execution"]["formal_processes"] == 2
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert "not scene exposure" in contract["claim_ceiling"]
