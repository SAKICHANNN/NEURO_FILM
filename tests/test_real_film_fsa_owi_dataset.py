from __future__ import annotations

import hashlib
from pathlib import Path

from src.real_film.fsa_owi_dataset import (
    merge_pixel_evidence,
    prepare_download_records,
    seed_verified_pilot,
)


def _grouped() -> list[dict]:
    return [
        {"loc_fsac_id": "fsac.1a00002", "evaluation_creator_group": "b"},
        {"loc_fsac_id": "fsac.1a00001", "evaluation_creator_group": "a"},
    ]


def test_prepare_and_merge_pixel_evidence_are_complete() -> None:
    prepared = prepare_download_records(_grouped())
    assert [row["loc_fsac_id"] for row in prepared] == ["fsac.1a00001", "fsac.1a00002"]
    pixels = []
    for row in prepared:
        pixels.append({
            "loc_fsac_id": row["loc_fsac_id"], "local_path": "x", "download_url": "u",
            "bytes": 1, "sha256": "a", "dhash64": "0" * 16, "width": 1,
            "height": 1, "mode": "RGB", "format": "JPEG", "content_type": "image/jpeg",
            "icc_profile_bytes": 0, "icc_profile_sha256": None, "exif_tags": 0,
        })
    merged = merge_pixel_evidence(_grouped(), pixels)
    assert len(merged) == 2
    assert merged[0]["pixel_evidence"]["format"] == "JPEG"


def test_seed_verified_pilot_reuses_exact_payload(tmp_path: Path) -> None:
    prepared = prepare_download_records(_grouped())
    source = tmp_path / "pilot.jpg"
    source.write_bytes(b"jpeg-payload")
    pilot = [{
        "loc_fsac_id": "fsac.1a00001",
        "local_path": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }]
    evidence = seed_verified_pilot(prepared, pilot, tmp_path / "phase-c")
    assert evidence == {"seeded_records": 1, "seeded_bytes": len(b"jpeg-payload")}
    assert len(list((tmp_path / "phase-c").glob("*.jpg"))) == 1
