from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

from src.real_film.ppisp_capture_pair_source_lock import sha256_bytes
from src.real_film.rgb2raw_capture_pair_source_lock import (
    analyze_members,
    run_source_lock,
)


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for camera in ("iphone-x", "samsung-s9"):
            archive.writestr(f"train/{camera}/001.pkl", b"metadata")
            for patch in range(2):
                archive.writestr(f"train/{camera}/001_{patch}.png", b"png")
                archive.writestr(f"train/{camera}/001_{patch}.npy", b"npy")
    return stream.getvalue()


def _central(archive: bytes) -> tuple[int, bytes]:
    eocd = archive.rfind(b"PK\x05\x06")
    values = struct.unpack_from("<4s4H2IH", archive, eocd)
    size, offset = values[5], values[6]
    return offset, archive[offset : offset + size]


def test_member_analysis_pairs_and_groups() -> None:
    from src.real_film.ppisp_capture_pair_source_lock import parse_central_directory

    archive = _archive()
    _, central = _central(archive)
    facts = analyze_members(parse_central_directory(central))
    assert facts["primary_pair_count"] == 4
    assert facts["primary_scene_group_count"] == 2
    assert facts["metadata_covers_primary_scene_groups"] is True
    assert facts["unsafe_member_count"] == 0


def test_source_lock_reads_only_central_directory(tmp_path: Path) -> None:
    archive = _archive()
    offset, central = _central(archive)
    readme = b"---\nlicense: mit\n---\n"
    config = {
        "schema": "neuro-film.sf3-a0r-rgb2raw-capture-pair-source-lock-contract.v1",
        "experiment_id": "test",
        "source": {
            "dataset_id": "test/data",
            "revision": "rev",
            "readme_url": "https://invalid/readme",
            "readme_size": len(readme),
            "readme_sha256": sha256_bytes(readme),
            "license_declaration": "mit",
            "license_text_present": False,
            "archive": {
                "path": "data.zip",
                "size": len(archive),
                "lfs_sha256": "0" * 64,
                "xet_hash": "1" * 64,
                "central_offset": offset,
                "central_size": len(central),
                "central_sha256": sha256_bytes(central),
            },
        },
        "gates": {
            "required_primary_camera_groups": ["iphone-x", "samsung-s9"],
            "required_primary_pair_count": 4,
            "required_primary_scene_group_count": 2,
            "required_metadata_member_count": 2,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    ranges: list[tuple[int, int]] = []

    def range_reader(_url: str, start: int, end: int, size: int) -> bytes:
        assert size == len(archive)
        ranges.append((start, end))
        return archive[start : end + 1]

    report = run_source_lock(
        path, range_reader=range_reader, url_reader=lambda _url: readme
    )
    assert report["automatic_pass"] is True
    assert report["member_payload_reads"] == report["pixel_decodes"] == 0
    assert ranges == [(offset, offset + len(central) - 1)]
