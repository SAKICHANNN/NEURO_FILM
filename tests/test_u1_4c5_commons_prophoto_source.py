from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from src.eval.commons_prophoto_source_preflight import (
    CommonsProPhotoSourceError,
    _icc_xyz_tag,
    load_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/u1_4c5_commons_prophoto_source_v1.json"


def test_contract_freezes_rights_bytes_and_stops() -> None:
    contract = load_contract(CONTRACT)
    assert len(contract["rows"]) == 12
    assert len({row["artist"] for row in contract["rows"]}) == 4
    assert sum(row["bytes"] for row in contract["rows"]) == 198661368
    assert contract["eligibility"]["minimum_native_16bit_sources"] == 2
    assert (
        contract["eligibility"]["visual_review_required_before_algorithm_role"] is True
    )
    assert all(
        row["url"].startswith("https://upload.wikimedia.org/")
        for row in contract["rows"]
    )


def test_contract_hash_or_source_inventory_drift_rejects(tmp_path: Path) -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    payload["rows"][0]["bytes"] += 1
    mutated = tmp_path / "contract.json"
    mutated.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CommonsProPhotoSourceError, match="contract hash drift"):
        load_contract(mutated)


def test_existing_fivek_prophoto_icc_has_expected_xyz_tags() -> None:
    manifest = json.loads(
        (ROOT / "configs/u1_4c4_native_prophoto_source_manifest_v1.json").read_text(
            encoding="utf-8"
        )
    )
    path = ROOT / manifest["rows"][0]["path"]
    if not path.is_file():
        pytest.skip("P-backed FiveK profile fixture unavailable")
    with Image.open(path) as image:
        profile = image.info.get("icc_profile")
    assert isinstance(profile, bytes)
    assert _icc_xyz_tag(profile, b"rXYZ").shape == (3,)
    damaged = bytearray(profile)
    damaged[16:20] = b"CMYK"
    with pytest.raises(CommonsProPhotoSourceError, match="not an RGB profile"):
        _icc_xyz_tag(bytes(damaged), b"rXYZ")
