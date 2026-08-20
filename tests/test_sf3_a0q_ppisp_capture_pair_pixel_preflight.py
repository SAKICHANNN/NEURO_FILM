from __future__ import annotations

import io
import zipfile

import numpy as np

from src.real_film.ppisp_capture_pair_pixel_preflight import (
    census_agreement,
    extract_member,
    select_pairs,
)
from src.real_film.ppisp_capture_pair_source_lock import parse_central_directory


def _archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for scene in ("scene", "scene_auto"):
            for camera in ("nikon", "oms"):
                for index in range(4):
                    archive.writestr(
                        f"{scene}/images/{camera}/{index:03d}.jpg",
                        f"{scene}-{camera}-{index}".encode(),
                    )
    return stream.getvalue()


def _central(archive: bytes) -> bytes:
    eocd = archive.rfind(b"PK\x05\x06")
    size = int.from_bytes(archive[eocd + 12 : eocd + 16], "little")
    start = int.from_bytes(archive[eocd + 16 : eocd + 20], "little")
    return archive[start : start + size]


def test_hash_selection_is_order_independent_and_balanced() -> None:
    members = parse_central_directory(_central(_archive()))
    forward = select_pairs(
        members,
        scene_id="scene",
        camera_groups=["nikon", "oms"],
        count_per_camera=2,
        seed="fixed",
    )
    reverse = select_pairs(
        list(reversed(members)),
        scene_id="scene",
        camera_groups=["nikon", "oms"],
        count_per_camera=2,
        seed="fixed",
    )
    assert [row["pair_key"] for row in forward] == [
        row["pair_key"] for row in reverse
    ]
    assert [row["camera_group"] for row in forward].count("nikon") == 2
    assert [row["camera_group"] for row in forward].count("oms") == 2


def test_member_extraction_checks_local_and_central_facts(monkeypatch) -> None:
    archive = _archive()
    member = parse_central_directory(_central(archive))[0]

    def read(_url: str, start: int, end: int, size: int) -> bytes:
        assert size == len(archive)
        return archive[start : end + 1]

    monkeypatch.setattr(
        "src.real_film.ppisp_capture_pair_pixel_preflight._range_get", read
    )
    raw, network_bytes = extract_member("unused", member, archive_size=len(archive))
    assert raw == b"scene-nikon-0"
    assert network_bytes > len(raw)


def test_census_is_monotone_robust_and_detects_geometry_change() -> None:
    base = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert census_agreement(base, base * 2.0 + 7.0) == 1.0
    changed = base.copy()
    changed[2:6, 2:6] = np.flip(changed[2:6, 2:6], axis=0)
    assert census_agreement(base, changed) < 1.0
