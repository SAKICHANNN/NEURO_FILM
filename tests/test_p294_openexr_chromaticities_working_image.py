from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p294_openexr_chromaticities_working_image_v1.json"


def test_p294_binds_two_exact_official_rgb_sources() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["official_source"]["revision"] == (
        "e38ffb0790f62f05a6f083a6fa4cac150b3b7452"
    )
    assert [(row["id"], row["bytes"], row["git_blob"]) for row in config["sources"]] == [
        ("rec709", 908_168, "13b3fca94da1426e4793bd66a20299c955b73d00"),
        (
            "xyz_equal_energy",
            930_048,
            "cd4423ab12669b4caaaeb157ddd08fe430f5a775",
        ),
    ]


def test_p294_requires_explicit_file_identity_without_adjacent_rescue() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "sidecar" in config["stop_rule"]
    assert "clipping" in config["stop_rule"]
    assert "P293 EyefulTower EXR" in config["forbidden_payloads"]
    assert all("_YC" not in row["path"] for row in config["sources"])


def test_p294_header_lock_records_the_missing_rec709_attribute() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = {row["id"]: row for row in config["expected_headers"]}
    assert expected["rec709"]["chromaticities"] is None
    assert expected["xyz_equal_energy"]["chromaticities"] == [
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.3333333432674408,
        0.3333333432674408,
    ]
    assert all(row["channels"]["R"]["type"] == "HALF" for row in expected.values())
