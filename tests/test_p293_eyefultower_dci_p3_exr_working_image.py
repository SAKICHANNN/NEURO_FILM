from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p293_eyefultower_dci_p3_exr_working_image_v1.json"


def test_p293_source_is_exact_p292_selected_object() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = config["source"]
    assert source["bytes"] == 9_399_313
    assert source["md5"] == "29c574af8fc3a543cd773eb812d56a49"
    assert source["sha256"] == (
        "c7b65fe69568b4d04bbf162381fd6d74c9b9682057a0f9dd8b931ec6b038f174"
    )
    assert source["path"].endswith("40_DSC0001.exr")
    assert config["parent"]["report_sha256"].startswith("aba2796f")


def test_p293_header_gate_forbids_readme_metadata_substitution() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["gates"]["require_container_dci_p3_chromaticities"]
    assert "Do not add or infer missing chromaticities" in config["stop_rule"]
    assert "same-source JPEG" in config["stop_rule"]
    assert config["boundaries"]["jpeg_reads"] == 0


def test_p293_observed_header_is_rgb_float_no_compression_without_color_id() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = config["expected_header"]
    assert expected["compression"] == "NO_COMPRESSION"
    assert sorted(expected["channels"]) == ["B", "G", "R"]
    assert all(channel["type"] == "FLOAT" for channel in expected["channels"].values())
    assert expected["chromaticities_present"] is False
    assert expected["data_window"] == {"min": [0, 0], "max": [1081, 722]}
