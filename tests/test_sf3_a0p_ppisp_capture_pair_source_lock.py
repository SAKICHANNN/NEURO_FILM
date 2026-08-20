from __future__ import annotations

import io
import json
import struct
import zipfile
from pathlib import Path

from src.real_film.ppisp_capture_pair_source_lock import (
    analyze_members,
    locate_central_directory,
    parse_central_directory,
    run_source_lock,
    sha256_bytes,
)


def _archive(*, auto_name: str = "auto") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for variant in ("scene", f"scene_{auto_name}"):
            for index in range(4):
                archive.writestr(f"{variant}/images/cam0/{index:03d}.jpg", b"jpeg")
            archive.writestr(f"{variant}/sparse/0/cameras.bin", b"camera")
            archive.writestr(f"{variant}/sparse/0/images.bin", b"images")
    return stream.getvalue()


def _central(archive: bytes) -> bytes:
    offset = archive.rfind(b"PK\x05\x06")
    values = struct.unpack_from("<4s4H2IH", archive, offset)
    size, start = values[5], values[6]
    return archive[start : start + size]


def test_central_directory_and_pair_analysis() -> None:
    archive = _archive()
    members = parse_central_directory(_central(archive))
    facts = analyze_members(members)
    assert facts["exact_pair_count"] == 4
    assert facts["standard_jpeg_count"] == 4
    assert facts["auto_jpeg_count"] == 4
    assert facts["camera_metadata_member_count"] == 2
    assert facts["image_metadata_member_count"] == 2
    assert facts["image_member_reads"] == 0


def test_eocd_locator_returns_exact_central_range() -> None:
    archive = _archive()
    tail = archive[-512:]
    start, size, count = locate_central_directory(tail, archive_size=len(archive))
    assert archive[start : start + size] == _central(archive)
    assert count == len(parse_central_directory(_central(archive)))


def test_full_source_lock_uses_only_tail_and_central_ranges(tmp_path: Path) -> None:
    archive = _archive()
    readme = (
        b"CC BY 4.0 commercial and non-commercial Nikon Z7 OM-1 Mark II "
        b"iPhone 13 Pro auto exposure and auto white balance"
    )
    config = {
        "schema": "neuro-film.sf3-a0p-ppisp-capture-pair-source-lock-contract.v1",
        "experiment_id": "test",
        "source": {
            "dataset_id": "test/dataset",
            "revision": "revision",
            "readme_url": "https://invalid/readme",
            "readme_size": len(readme),
            "readme_sha256": sha256_bytes(readme),
            "required_readme_phrases": [
                "CC BY 4.0",
                "commercial and non-commercial",
                "Nikon Z7",
                "OM-1 Mark II",
                "iPhone 13 Pro",
                "auto exposure and auto white balance",
            ],
        },
        "archives": [
            {
                "scene_id": f"scene{index}",
                "path": f"colmap/scene{index}.zip",
                "size": len(archive),
                "sha256": "0" * 64,
            }
            for index in range(4)
        ],
        "network_limits": {
            "tail_probe_bytes_per_archive": 512,
            "maximum_total_archive_bytes": 8192,
            "member_payload_reads": 0,
        },
        "gates": {
            "required_scene_count": 4,
            "minimum_paired_jpegs_per_scene": 4,
            "minimum_distinct_pair_parent_groups_per_scene": 1,
            "require_standard_and_auto_namespaces": True,
            "require_colmap_camera_and_image_metadata_names": True,
            "require_safe_unencrypted_members": True,
            "require_explicit_cc_by_4_commercial_scope": True,
        },
        "decision_if_pass": "PASS",
        "decision_if_fail": "FAIL",
        "claim_ceiling": "test",
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    ranges: list[tuple[int, int]] = []

    def read_range(_url: str, start: int, end: int, size: int) -> bytes:
        assert size == len(archive)
        ranges.append((start, end))
        return archive[start : end + 1]

    report = run_source_lock(
        config_path, range_reader=read_range, url_reader=lambda _url: readme
    )
    assert report["automatic_pass"] is True
    assert report["decision"] == "PASS"
    assert report["image_member_reads"] == report["pixel_decodes"] == 0
    assert all(end >= start for start, end in ranges)
    assert 4 <= len(ranges) <= 8


def test_auto_detection_does_not_match_automatic_word() -> None:
    archive = _archive(auto_name="automatic")
    members = parse_central_directory(_central(archive))
    facts = analyze_members(members)
    assert facts["auto_namespace_present"] is False
