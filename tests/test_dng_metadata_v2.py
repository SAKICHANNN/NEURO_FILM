from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.preprocess.dng_metadata import (
    DngMetadataError,
    build_dng_capture_metadata_receipt,
    canonical_json_bytes,
)
from src.preprocess.dng_metadata_v2 import build_dng_capture_metadata_receipt_v2


def _extratags(
    *,
    cfa_pattern: bytes = b"\x00\x01\x01\x02",
    include_plane_color: bool,
    include_layout: bool,
) -> list[tuple[int, str, int, object, bool]]:
    tags: list[tuple[int, str, int, object, bool]] = [
        (50706, "B", 4, b"\x01\x04\x00\x00", False),
        (50707, "B", 4, b"\x01\x03\x00\x00", False),
        (50708, "s", 0, "Synthetic Camera", False),
        (274, "H", 1, 1, False),
        (33421, "H", 2, (2, 2), False),
        (33422, "B", 4, cfa_pattern, False),
        (50713, "H", 2, (1, 1), False),
        (50714, "2I", 1, (64, 1), False),
        (50717, "I", 1, 1023, False),
        (50728, "2I", 3, (1, 2, 1, 1, 3, 4), False),
    ]
    if include_plane_color:
        tags.append((50710, "B", 3, b"\x00\x01\x02", False))
    if include_layout:
        tags.append((50711, "H", 1, 1, False))
    return tags


def _write(
    path: Path,
    *,
    cfa_pattern: bytes = b"\x00\x01\x01\x02",
    include_plane_color: bool = False,
    include_layout: bool = False,
) -> None:
    tifffile.imwrite(
        path,
        np.zeros((4, 6), dtype=np.uint16),
        photometric=32803,
        extratags=_extratags(
            cfa_pattern=cfa_pattern,
            include_plane_color=include_plane_color,
            include_layout=include_layout,
        ),
    )


def test_v2_applies_only_standard_rgb_cfa_defaults(tmp_path: Path) -> None:
    source = tmp_path / "defaulted.dng"
    _write(source)

    with pytest.raises(DngMetadataError, match="missing required CFA facts"):
        build_dng_capture_metadata_receipt(
            source, source_id="fixture", logical_path="data/fixture.dng"
        )

    receipt = build_dng_capture_metadata_receipt_v2(
        source, source_id="fixture", logical_path="data/fixture.dng"
    )
    assert receipt["schema"] == "neuro_film.dng_capture_metadata_receipt.v2"
    assert receipt["standard_defaults_used"] == ["cfa_plane_color", "cfa_layout"]
    assert receipt["facts"]["cfa_plane_color"] == {
        "count": 3,
        "ifd_paths": [],
        "tiff_type": "BYTE",
        "value": [0, 1, 2],
        "value_origin": "dng_standard_default_1_7_1_0",
    }
    assert receipt["facts"]["cfa_layout"] == {
        "count": 1,
        "ifd_paths": [],
        "tiff_type": "SHORT",
        "value": [1],
        "value_origin": "dng_standard_default_1_7_1_0",
    }
    assert receipt["facts"]["black_level"]["value_origin"] == "explicit_ifd_tag"


def test_v2_never_overrides_explicit_cfa_tags(tmp_path: Path) -> None:
    source = tmp_path / "explicit.dng"
    _write(source, include_plane_color=True, include_layout=True)
    receipt = build_dng_capture_metadata_receipt_v2(
        source, source_id="fixture", logical_path="data/fixture.dng"
    )
    assert receipt["standard_defaults_used"] == []
    assert receipt["facts"]["cfa_plane_color"]["ifd_paths"] == ["0"]
    assert receipt["facts"]["cfa_plane_color"]["value_origin"] == "explicit_ifd_tag"
    assert receipt["facts"]["cfa_layout"]["value_origin"] == "explicit_ifd_tag"


def test_v2_rejects_omitted_plane_mapping_for_non_rgb_cfa(tmp_path: Path) -> None:
    source = tmp_path / "non_rgb.dng"
    _write(source, cfa_pattern=b"\x00\x01\x02\x03", include_layout=True)
    with pytest.raises(DngMetadataError, match="valid only for an RGB CFA pattern"):
        build_dng_capture_metadata_receipt_v2(
            source, source_id="fixture", logical_path="data/fixture.dng"
        )


def test_v2_is_byte_exact_and_never_decodes_raster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "defaulted.dng"
    _write(source)

    def forbidden_decode(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raster samples must not be decoded")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden_decode)
    kwargs = {"source_id": "fixture", "logical_path": r"data\fixture.dng"}
    first = build_dng_capture_metadata_receipt_v2(source, **kwargs)
    second = build_dng_capture_metadata_receipt_v2(source, **kwargs)
    assert canonical_json_bytes(first) == canonical_json_bytes(second)
    assert first["raster_decode_calls"] == 0
    assert first["logical_path"] == "data/fixture.dng"
