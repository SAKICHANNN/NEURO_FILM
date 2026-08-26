from __future__ import annotations

import pytest

from src.preprocess.dng_profile_huesatmap_audit import (
    DngProfileHueSatMapAuditError,
    parse_profile_huesatmap_exif,
    table_summary,
)

VALID = """\
Exif.Image.ProfileHueSatMapDims               2 2 1
Exif.Image.ProfileHueSatMapData1              0 1 1 2 0.9 1.1 0 1 1 -2 1.1 0.8
Exif.Image.ProfileHueSatMapData2              0 1 1 1 1 1 0 1 1 -1 1 1
"""


def test_parse_valid_2_5d_map_and_default_semantics() -> None:
    result = parse_profile_huesatmap_exif(VALID)
    assert result.dimensions == (2, 2, 1)
    assert result.data1.shape == (1, 2, 2, 3)
    assert result.encoding == 0
    assert result.dynamic_range == 0
    assert table_summary(result.data1)["entry_count"] == 4


def test_parse_explicit_3d_map() -> None:
    data = " ".join("0 1 1" for _ in range(8))
    text = (
        "Exif.Image.ProfileHueSatMapDims               2 2 2\n"
        f"Exif.Image.ProfileHueSatMapData1              {data}\n"
        f"Exif.Image.ProfileHueSatMapData2              {data}\n"
        "Exif.Image.ProfileHueSatMapEncoding           1\n"
        "Exif.Image.ProfileDynamicRange                1\n"
    )
    result = parse_profile_huesatmap_exif(text)
    assert result.data1.shape == (2, 2, 2, 3)
    assert result.encoding == 1
    assert result.dynamic_range == 1


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (VALID.replace("2 2 1", "0 2 1", 1), "out of range"),
        (VALID.replace("2 0.9 1.1 ", "", 1), "count mismatch"),
        (VALID.replace("0 1 1 2", "0 1 0.5 2", 1), "must equal one"),
        (VALID.replace("2 0.9 1.1", "nan 0.9 1.1", 1), "nonfinite"),
    ],
)
def test_invalid_map_fails_closed(text: str, message: str) -> None:
    with pytest.raises(DngProfileHueSatMapAuditError, match=message):
        parse_profile_huesatmap_exif(text)
