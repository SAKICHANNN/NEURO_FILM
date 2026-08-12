import json
from pathlib import Path

import numpy as np
import pytest

from src.eval.native_cloud_spatial_partition import (
    _build,
    _configure_source_derived,
    render_physical_partition,
)

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


def test_p4gj_source_derived_bridge_exposes_density_envelope_gap(tmp_path: Path):
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    fixture = contract["fixture"]
    physical = json.loads(
        (ROOT / "configs/u6_p4fb_native_cloud_spatial_partition_v1.json").read_text(
            encoding="utf-8"
        )
    )
    physical["fixture"]["full_height"] = fixture["height"]
    physical["fixture"]["width"] = fixture["width"]
    dll = _build(
        ROOT,
        tmp_path / "build",
        None,
        bridge_source="nf_sensitometry_cloud_bridge_f32_v2.c",
    )
    library = _configure_source_derived(dll)
    assert library.nf_sensitometry_cloud_bridge_f32_abi_version_v2() == 2
    source = np.zeros(
        (fixture["height"], fixture["width"], 3), dtype=np.float32
    )
    with pytest.raises(RuntimeError, match="post failed: 23"):
        render_physical_partition(
            library,
            physical,
            source,
            0,
            fixture["height"],
            source_derived_expected=True,
        )


def test_p4gj_formal_result_closes_on_black_density_envelope():
    result = json.loads(
        (ROOT / "docs/evidence/U6_P4GJ_CLOUD_SOURCE_DERIVED_TRANSMITTANCE_RESULT.json").read_text(
            encoding="utf-8"
        )
    )
    assert result["automatic_pass"] is False
    assert result["stable"]["rows"] == [
        {"error": "P4FB physical provider post failed: 23", "source_level": 0.0},
        {"error": None, "source_level": 1.0},
    ]
    assert result["stable"]["decision"] == "close_source_derived_cloud_transmittance_v2"
    assert result["stable_evidence_id"] == "fb8e389fcb46f9840e43138f1249a17157cb39c1592dfd68bd63f889ebcb453d"
