from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p295_openexr_carrots_aces2065_interoperability_v1.json"


def test_p295_source_is_exact_official_carrots_object() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = config["source"]
    assert source["bytes"] == 914_825
    assert source["git_blob"] == "b4697022c7120065b1d6131b20da363a8ff88069"
    assert (
        source["sha256"]
        == "892d9eb1d2b22a3c7c57d22fa60c33aaa4a49ba3ead5304ec594b274aa39e4e1"
    )
    assert source["repository_path"] == "ScanLines/Carrots.exr"


def test_p295_requires_unchanged_strict_p251_identity() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    p251 = config["p251"]
    assert p251["required_channel_type"] == "FLOAT"
    assert p251["required_aces_image_container_flag"] == 1
    assert p251["required_color_interop_id"] == "lin_ap0_scene"
    assert (
        p251["module_sha256"]
        == "9ec0e671144624176d122cb72a38b61de40eba55f8c52772234c438dd33d2390"
    )


def test_p295_observed_header_is_ap0_but_not_strict_p251_container() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    header = config["expected_header"]
    assert header["chromaticities"] == config["p251"]["required_chromaticities"]
    assert header["adopted_neutral"] == config["p251"]["required_adopted_neutral"]
    assert sorted(header["channels"]) == ["A", "B", "G", "R"]
    assert all(row["type"] == "HALF" for row in header["channels"].values())
    assert header["aces_image_container_flag"] is None
    assert header["color_interop_id"] is None


def test_p295_stop_rule_forbids_same_family_rescue() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stop = config["stop_rule"]
    assert "another sample" in stop
    assert "metadata rewrite" in stop
    assert "storage conversion" in stop
    assert config["boundaries"]["pixel_channel_reads_on_header_failure"] == 0
