from __future__ import annotations

import io
import json
import zipfile

from src.real_film.ntire_night_metadata_preflight import (
    analyze_metadata,
    derive_selection,
    extract_member,
)
from src.real_film.ppisp_capture_pair_source_lock import (
    locate_central_directory,
    parse_central_directory,
)


def _config() -> dict[str, object]:
    return {
        "field_families": {
            "black_level": ["blacklevel"],
            "white_level": ["whitelevel"],
            "noise_profile": ["noiseprofile"],
            "cfa_pattern": ["rawpattern"],
            "orientation": ["orientation"],
            "white_balance": ["wb"],
            "crop_bounds": ["crop"],
        },
        "crop_numeric_contract": {
            "minimum_numeric_values_per_row": 4,
            "maximum_absolute_value": 1_000_000_000,
        },
    }


def test_selection_is_stable_and_counted() -> None:
    first = derive_selection("seed", 0, 99, 12)
    assert first == derive_selection("seed", 0, 99, 12)
    assert len(first) == len(set(first)) == 12


def test_metadata_analysis_requires_all_frozen_families() -> None:
    payload = json.dumps(
        {
            "black_level": 64,
            "white_level": 4095,
            "noise_profile": [0.1, 0.01],
            "raw_pattern": [[0, 1], [1, 2]],
            "orientation": 1,
            "wb": [2.0, 1.0, 1.5],
            "target_crop": {"left": 1, "top": 2, "right": 100, "bottom": 80},
        }
    ).encode()
    facts = analyze_metadata(payload, _config())
    assert facts["all_field_families_present"]
    assert facts["finite_numeric_values"]
    assert facts["crop_values_bounded"]
    assert facts["crop_numeric_value_count"] == 4


def test_bounded_member_extract_checks_crc_and_payload() -> None:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("raw/7.json", b'{"black_level": 64}')
    value = stream.getvalue()
    offset, size, count = locate_central_directory(value, archive_size=len(value))
    members = parse_central_directory(value[offset : offset + size])
    assert count == 1

    def reader(_url: str, start: int, end: int, archive_size: int) -> bytes:
        assert archive_size == len(value)
        return value[start : end + 1]

    payload, bytes_read = extract_member("memory", members[0], len(value), reader)
    assert payload == b'{"black_level": 64}'
    assert bytes_read == 30 + members[0].compressed_size
