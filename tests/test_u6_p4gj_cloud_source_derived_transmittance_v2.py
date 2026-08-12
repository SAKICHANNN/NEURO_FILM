import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u6_p4gj_cloud_source_derived_transmittance_v2.json"


def test_p4gj_contract_replaces_only_fixture_transmittance():
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    mechanism = contract["mechanism"]
    assert mechanism["historical_expected_transmittance"] == "constant-0.6-test-fixture"
    assert mechanism["candidate_expected_transmittance"] == "pow10-negative-developed-density-core"
    assert mechanism["post_spatial_chain"] == "unchanged-p4fb"
    assert contract["execution"]["historical_v1_modified"] is False
    assert contract["execution"]["post_result_retuning_allowed"] is False
    assert contract["gates"]["minimum_channel_scan_response_span"] == 0.5
