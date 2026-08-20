from __future__ import annotations

from src.real_film.ntire_night_capture_pair_source_lock import analyze_archives
from src.real_film.ppisp_capture_pair_source_lock import ZipMember


def _member(name: str) -> ZipMember:
    return ZipMember(
        name=name,
        flags=0,
        method=8,
        crc32=1,
        compressed_size=10,
        uncompressed_size=20,
        local_offset=30,
        external_attributes=0,
        version_made_by=20,
    )


def test_analyze_archives_requires_exact_numeric_pair_graph() -> None:
    raw = [_member("raw/"), _member("raw/1.png"), _member("raw/1.json")]
    target = [_member("sony/"), _member("sony/1.JPG")]
    facts = analyze_archives(raw, target)
    assert facts["complete_pair_count"] == 1
    assert facts["incomplete_id_count"] == 0
    assert facts["unsafe_member_count"] == 0


def test_analyze_archives_reports_incomplete_and_unexpected_members() -> None:
    raw = [
        _member("raw/"),
        _member("raw/1.png"),
        _member("raw/1.json"),
        _member("raw/2.png"),
        _member("raw/readme.txt"),
    ]
    target = [_member("sony/"), _member("sony/1.JPG"), _member("sony/3.JPG")]
    facts = analyze_archives(raw, target)
    assert facts["complete_pair_count"] == 1
    assert facts["incomplete_id_count"] == 2
    assert facts["unexpected_raw_member_count"] == 1


def test_analyze_archives_rejects_unsafe_member() -> None:
    raw = [_member("raw/"), _member("raw/1.png"), _member("raw/1.json")]
    target = [_member("sony/"), _member("../sony/1.JPG")]
    facts = analyze_archives(raw, target)
    assert facts["complete_pair_count"] == 0
    assert facts["unsafe_member_count"] == 1
