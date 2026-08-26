from __future__ import annotations

import pytest

from src.preprocess.rawpixls_capture_metadata_audit import (
    RawPixlsCaptureMetadataError,
    parse_rawpixls_exif_text,
)

GOOD = """\
Exif.Image.Make                               Example
Exif.Photo.ExposureTime                       1/125 s
Exif.Photo.FNumber                            F2.8
Exif.Photo.ISOSpeedRatings                    100
Exif.Photo.DateTimeOriginal                   2026:8:6 12:34:56
Exif.Image.UniqueCameraModel                  Camera One
Exif.Image.AsShotNeutral                      64/128 128/128 96/128
"""


def test_parse_complete_capture_metadata() -> None:
    result = parse_rawpixls_exif_text(GOOD)
    assert result["complete"] is True
    assert result["missing"] == []
    assert result["facts"]["date_time_original"] == "2026-08-06T12:34:56"
    assert result["facts"]["exposure_time_seconds"] == pytest.approx(0.008)
    assert result["facts"]["as_shot_neutral_green_normalized"] == [0.5, 1.0, 0.75]


def test_missing_fact_is_retained_without_guessing() -> None:
    result = parse_rawpixls_exif_text(
        GOOD.replace("Exif.Image.UniqueCameraModel                  Camera One\n", "")
    )
    assert result["complete"] is False
    assert result["missing"] == ["unique_camera_model"]


def test_equal_image_and_photo_aliases_are_allowed() -> None:
    result = parse_rawpixls_exif_text(
        GOOD + "Exif.Image.ExposureTime                       1/125 s\n"
    )
    assert result["complete"] is True


def test_conflicting_aliases_fail_closed() -> None:
    with pytest.raises(RawPixlsCaptureMetadataError, match="conflicting exposure_time"):
        parse_rawpixls_exif_text(
            GOOD + "Exif.Image.ExposureTime                       1/60 s\n"
        )


@pytest.mark.parametrize(
    "replacement, message",
    [
        ("0/1 1/1 1/1", "positive"),
        ("1/1 1/1", "three values"),
        ("1/100 1/1 1/1", "implausibly"),
    ],
)
def test_invalid_as_shot_neutral_fails_closed(replacement: str, message: str) -> None:
    text = GOOD.replace("64/128 128/128 96/128", replacement)
    with pytest.raises(RawPixlsCaptureMetadataError, match=message):
        parse_rawpixls_exif_text(text)
