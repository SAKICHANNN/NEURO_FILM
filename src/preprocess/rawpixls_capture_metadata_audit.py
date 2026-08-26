"""Strict parsing for the P240 raw.pixls.us capture-metadata source audit."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any


class RawPixlsCaptureMetadataError(ValueError):
    """Raised when an EXIF text payload is structurally ambiguous."""


_LINE = re.compile(r"^(Exif\.(?:Image|Photo)\.([A-Za-z0-9]+))\s+(.+?)\s*$")
_DATE = re.compile(
    r"^(\d{4}):(\d{1,2}):(\d{1,2}) (\d{2}):(\d{2}):(\d{2})$"
)
_FIELDS = {
    "Make": "make",
    "UniqueCameraModel": "unique_camera_model",
    "DateTimeOriginal": "date_time_original",
    "ExposureTime": "exposure_time",
    "FNumber": "f_number",
    "ISOSpeedRatings": "iso_speed",
    "AsShotNeutral": "as_shot_neutral",
}
REQUIRED_FACTS = tuple(_FIELDS.values())


def _one(values: list[str], label: str) -> str | None:
    if not values:
        return None
    unique = sorted(set(values))
    if len(unique) != 1:
        raise RawPixlsCaptureMetadataError(f"conflicting {label} values")
    return unique[0]


def _positive_number(token: str, label: str) -> float:
    value = float(Fraction(token))
    if not math.isfinite(value) or value <= 0.0:
        raise RawPixlsCaptureMetadataError(f"{label} must be positive and finite")
    return value


def _exposure(value: str) -> float:
    token = value.removesuffix(" s").strip()
    return _positive_number(token, "exposure_time")


def _f_number(value: str) -> float:
    return _positive_number(value.removeprefix("F").strip(), "f_number")


def _iso(value: str) -> int:
    if not re.fullmatch(r"\d+", value):
        raise RawPixlsCaptureMetadataError("iso_speed must be an integer")
    result = int(value)
    if result <= 0:
        raise RawPixlsCaptureMetadataError("iso_speed must be positive")
    return result


def _date(value: str) -> str:
    match = _DATE.fullmatch(value)
    if match is None:
        raise RawPixlsCaptureMetadataError("date_time_original is invalid")
    year, month, day, hour, minute, second = (int(item) for item in match.groups())
    if not (
        1 <= month <= 12
        and 1 <= day <= 31
        and 0 <= hour <= 23
        and 0 <= minute <= 59
        and 0 <= second <= 59
    ):
        raise RawPixlsCaptureMetadataError("date_time_original is out of range")
    return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}:{second:02d}"


def _neutral(value: str) -> list[float]:
    tokens = value.split()
    if len(tokens) != 3:
        raise RawPixlsCaptureMetadataError("as_shot_neutral must contain three values")
    raw = [_positive_number(token, "as_shot_neutral") for token in tokens]
    green = raw[1]
    normalized = [item / green for item in raw]
    if any(item < 0.125 or item > 8.0 for item in normalized):
        raise RawPixlsCaptureMetadataError("as_shot_neutral is implausibly scaled")
    return normalized


def parse_rawpixls_exif_text(text: str) -> dict[str, Any]:
    """Parse only the exact fields required by the frozen P240 contract."""

    if not isinstance(text, str) or not text.strip():
        raise RawPixlsCaptureMetadataError("EXIF text must be non-empty")
    collected: dict[str, list[str]] = {name: [] for name in _FIELDS.values()}
    for line in text.splitlines():
        match = _LINE.fullmatch(line)
        if match is None:
            continue
        tag_name = match.group(2)
        if tag_name in _FIELDS:
            collected[_FIELDS[tag_name]].append(match.group(3))

    raw = {name: _one(values, name) for name, values in collected.items()}
    missing = sorted(name for name, value in raw.items() if value is None)
    facts: dict[str, Any] = {}
    if raw["make"] is not None:
        facts["make"] = raw["make"]
    if raw["unique_camera_model"] is not None:
        facts["unique_camera_model"] = raw["unique_camera_model"]
    if raw["date_time_original"] is not None:
        facts["date_time_original"] = _date(raw["date_time_original"])
    if raw["exposure_time"] is not None:
        facts["exposure_time_seconds"] = _exposure(raw["exposure_time"])
    if raw["f_number"] is not None:
        facts["f_number"] = _f_number(raw["f_number"])
    if raw["iso_speed"] is not None:
        facts["iso_speed"] = _iso(raw["iso_speed"])
    if raw["as_shot_neutral"] is not None:
        facts["as_shot_neutral_green_normalized"] = _neutral(raw["as_shot_neutral"])
    return {"complete": not missing, "facts": facts, "missing": missing}


__all__ = [
    "REQUIRED_FACTS",
    "RawPixlsCaptureMetadataError",
    "parse_rawpixls_exif_text",
]
