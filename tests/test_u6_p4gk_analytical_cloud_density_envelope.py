import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gk_analytical_cloud_density_envelope_v1.json"


def test_p4gk_contract_requires_analytical_bounds_before_chart():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["candidate"]["hard_clipping_allowed"] is False
    assert contract["candidate"]["bounds"] == "exact-profile-black-and-white-reference-density"
    assert contract["execution"]["p4gh_chart_pixels_read"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["gates"]["minimum_channel_scan_response_span"] == 0.5
    assert contract["gates"]["minimum_interior_cloud_density_residual_rms"] == 0.0001
