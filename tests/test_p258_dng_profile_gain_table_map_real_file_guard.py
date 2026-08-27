from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.preprocess.dng_forward_raster as raster

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/p258_dng_profile_gain_table_map_real_file_guard_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_p258_frozen_source_matches_existing_p7h_row_without_metadata_read() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = config["source"]
    p7h_path = ROOT / config["bindings"]["p7h_config_path"]
    p7h = json.loads(p7h_path.read_text(encoding="utf-8"))
    rows = [row for row in p7h["candidates"] if row["repository_id"] == 6728]
    assert len(rows) == 1
    row = rows[0]
    assert row["id"] == source["source_id"]
    assert row["path"] == source["logical_path"]
    assert row["sha256"] == source["sha256"]
    assert row["url"] == source["url"]
    assert "CC0" in p7h["source"]["declared_license"]
    path = ROOT / source["logical_path"]
    assert path.stat().st_size == source["bytes"]
    assert _sha256(path) == source["sha256"]


def test_p258_expected_real_tag_is_guarded_by_unchanged_p257_implementation() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    expected = config["source"]["expected_present_guarded_tags"]
    assert expected == [52525]
    assert config["source"]["expected_absent_guarded_tags"] == [52544]
    assert raster._PROFILE_GAIN_TABLE_MAP_TAGS == {
        item["code"]: item["name"] for item in config["guarded_tags"]
    }


def test_p258_guard_diagnostic_carries_name_code_and_ifd() -> None:
    page = SimpleNamespace(tags={52525: object()}, pages=None)
    with pytest.raises(raster.DngForwardRasterError) as caught:
        raster._guard_unsupported_profile_gain_table_map([("0/2", page)])
    assert "ProfileGainTableMap(52525)@0/2" in str(caught.value)
