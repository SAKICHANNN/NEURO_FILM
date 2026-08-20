from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.preprocess.dng_metadata import (
    DngMetadataError,
    build_dng_capture_metadata_receipt,
    canonical_json_bytes,
)


def _required_extratags() -> list[tuple[int, str, int, object, bool]]:
    return [
        (50706, "B", 4, b"\x01\x04\x00\x00", False),
        (50707, "B", 4, b"\x01\x03\x00\x00", False),
        (50708, "s", 0, "Synthetic Camera", False),
        (274, "H", 1, 1, False),
        (33421, "H", 2, (2, 2), False),
        (33422, "B", 4, b"\x00\x01\x01\x02", False),
        (50710, "B", 3, b"\x00\x01\x02", False),
        (50711, "H", 1, 1, False),
        (50713, "H", 2, (1, 1), False),
        (50714, "2I", 1, (64, 1), False),
        (50717, "I", 1, 1023, False),
        (50728, "2I", 3, (1, 2, 1, 1, 3, 4), False),
        (51041, "d", 2, (0.001, 0.00001), False),
    ]


def _write_synthetic_dng(path: Path) -> None:
    tifffile.imwrite(
        path,
        np.zeros((4, 6), dtype=np.uint16),
        photometric=32803,
        extratags=_required_extratags(),
    )


def test_build_receipt_preserves_exact_types_without_raster_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "fixture.dng"
    _write_synthetic_dng(source)

    def forbidden_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raster samples must not be decoded")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden_decode)
    receipt = build_dng_capture_metadata_receipt(
        source,
        source_id="fixture",
        logical_path="data/fixtures/fixture.dng",
    )

    assert receipt["raw_kind"] == "cfa"
    assert receipt["raw_ifd_path"] == "0"
    assert receipt["raster_decode_calls"] == 0
    assert receipt["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["facts"]["dng_version"]["value"] == [1, 4, 0, 0]
    assert receipt["facts"]["black_level"]["value"] == [
        {"decimal": 64.0, "denominator": 1, "numerator": 64}
    ]
    assert receipt["facts"]["as_shot_neutral"]["value"] == [
        {"decimal": 0.5, "denominator": 2, "numerator": 1},
        {"decimal": 1.0, "denominator": 1, "numerator": 1},
        {"decimal": 0.75, "denominator": 4, "numerator": 3},
    ]
    body = {key: value for key, value in receipt.items() if key != "receipt_body_sha256"}
    assert receipt["receipt_body_sha256"] == hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def test_receipt_is_byte_exact_across_replay(tmp_path: Path) -> None:
    source = tmp_path / "fixture.dng"
    _write_synthetic_dng(source)
    kwargs = {
        "source_id": "fixture",
        "logical_path": r"data\fixtures\fixture.dng",
    }
    first = build_dng_capture_metadata_receipt(source, **kwargs)
    second = build_dng_capture_metadata_receipt(source, **kwargs)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["logical_path"] == "data/fixtures/fixture.dng"


def test_non_dng_and_missing_required_tags_fail_closed(tmp_path: Path) -> None:
    non_dng = tmp_path / "fixture.tif"
    tifffile.imwrite(non_dng, np.zeros((2, 2), dtype=np.uint16))
    with pytest.raises(DngMetadataError, match="only .dng"):
        build_dng_capture_metadata_receipt(
            non_dng, source_id="bad", logical_path="data/bad.tif"
        )

    incomplete = tmp_path / "incomplete.dng"
    tifffile.imwrite(incomplete, np.zeros((2, 2), dtype=np.uint16), photometric=32803)
    with pytest.raises(DngMetadataError, match="missing required DNG facts"):
        build_dng_capture_metadata_receipt(
            incomplete, source_id="bad", logical_path="data/incomplete.dng"
        )


def test_multiple_raw_ifds_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "ambiguous.dng"
    with tifffile.TiffWriter(source) as writer:
        for _ in range(2):
            writer.write(
                np.zeros((2, 2), dtype=np.uint16),
                photometric=32803,
                extratags=_required_extratags(),
            )
    with pytest.raises(DngMetadataError, match="exactly one raw IFD, found 2"):
        build_dng_capture_metadata_receipt(
            source, source_id="bad", logical_path="data/ambiguous.dng"
        )
