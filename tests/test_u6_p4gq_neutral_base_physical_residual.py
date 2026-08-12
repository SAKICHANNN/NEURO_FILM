import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gq_neutral_base_physical_residual_v1.json"


def test_p4gq_freezes_neutral_base_physical_residual():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    candidate = contract["candidate"]
    assert candidate["neutral_base"] == "exact-input-scene-linear"
    assert "cloud-free" in candidate["physical_residual"]
    assert candidate["empirical_scan_inverse_used"] is False
    assert candidate["hard_clipping_allowed"] is False
    assert contract["gates"]["minimum_physical_residual_rms"] == 0.0001
    assert contract["gates"]["maximum_limited_fraction"] == 0.05
    assert contract["execution"]["post_result_retuning_allowed"] is False
