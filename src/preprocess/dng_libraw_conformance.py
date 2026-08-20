"""Compare exact DNG receipt facts with metadata exposed by LibRaw."""

from __future__ import annotations

import math
from typing import Any

from .types import InputInspection


class DngLibRawConformanceError(ValueError):
    """Raised when a comparison would require a camera-specific guess."""


def _values(receipt: dict[str, Any], name: str) -> list[float]:
    try:
        value = receipt["facts"][name]["value"]
    except (KeyError, TypeError) as exc:
        raise DngLibRawConformanceError(f"receipt is missing {name}") from exc
    if not isinstance(value, list):
        raise DngLibRawConformanceError(f"receipt fact {name} is not a value list")
    result: list[float] = []
    for item in value:
        if isinstance(item, dict):
            numeric = float(item["decimal"])
        elif isinstance(item, int | float) and not isinstance(item, bool):
            numeric = float(item)
        else:
            raise DngLibRawConformanceError(f"receipt fact {name} is not numeric")
        if not math.isfinite(numeric):
            raise DngLibRawConformanceError(f"receipt fact {name} is not finite")
        result.append(numeric)
    return result


def _uniform(values: list[float], name: str) -> float:
    if not values:
        raise DngLibRawConformanceError(f"{name} is empty")
    if any(value != values[0] for value in values[1:]):
        raise DngLibRawConformanceError(f"{name} is not uniform")
    return values[0]


def _black_reference(values: list[float], channel_count: int) -> list[int]:
    if len(values) == channel_count:
        expanded = values
    elif len(values) == 1 or all(value == values[0] for value in values[1:]):
        expanded = [values[0]] * channel_count
    else:
        raise DngLibRawConformanceError("non-uniform BlackLevel cannot be mapped to LibRaw channels")
    return [round(value) for value in expanded]


def compare_dng_receipt_to_libraw(
    receipt: dict[str, Any],
    inspection: InputInspection,
) -> dict[str, Any]:
    """Return deterministic comparison facts without decoding image samples."""

    metadata = inspection.raw_metadata
    required = {
        "raw_width",
        "raw_height",
        "visible_width",
        "visible_height",
        "black_level_per_channel",
        "white_level",
        "camera_whitebalance",
    }
    missing = sorted(required - metadata.keys())
    if missing:
        raise DngLibRawConformanceError(f"LibRaw metadata is missing: {', '.join(missing)}")

    raw_width = int(_values(receipt, "image_width")[0])
    raw_height = int(_values(receipt, "image_length")[0])
    libraw_raw = [int(metadata["raw_width"]), int(metadata["raw_height"])]
    raw_geometry_match = libraw_raw == [raw_width, raw_height]

    dng_white = _uniform(_values(receipt, "white_level"), "WhiteLevel")
    libraw_white = float(metadata["white_level"])
    white_error = abs(libraw_white - dng_white)

    libraw_black = [int(value) for value in metadata["black_level_per_channel"]]
    if not libraw_black:
        raise DngLibRawConformanceError("LibRaw BlackLevel is empty")
    dng_black_projected = _black_reference(_values(receipt, "black_level"), len(libraw_black))
    black_errors = [abs(actual - expected) for actual, expected in zip(libraw_black, dng_black_projected)]

    neutral = _values(receipt, "as_shot_neutral")
    if len(neutral) != 3 or any(value <= 0.0 for value in neutral):
        raise DngLibRawConformanceError("AsShotNeutral must contain three positive values")
    expected_wb = [neutral[1] / neutral[0], 1.0, neutral[1] / neutral[2]]
    raw_wb = [float(value) for value in metadata["camera_whitebalance"][:3]]
    if len(raw_wb) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in raw_wb):
        raise DngLibRawConformanceError("LibRaw camera white balance is not three positive values")
    actual_wb = [value / raw_wb[1] for value in raw_wb]
    wb_relative_errors = [
        abs(actual - expected) / max(abs(expected), 1e-12)
        for actual, expected in zip(actual_wb, expected_wb)
    ]

    visible = [int(metadata["visible_width"]), int(metadata["visible_height"])]
    crop_values = _values(receipt, "default_crop_size") if "default_crop_size" in receipt["facts"] else []
    crop = [round(crop_values[0]), round(crop_values[1])] if len(crop_values) == 2 else None
    matches_raw = visible == [raw_width, raw_height]
    matches_crop = crop is not None and visible == crop
    if matches_raw and matches_crop:
        visible_classification = "raw_and_default_crop"
    elif matches_raw:
        visible_classification = "raw_ifd"
    elif matches_crop:
        visible_classification = "default_crop"
    else:
        visible_classification = "unexplained"

    numeric_values = [
        dng_white,
        libraw_white,
        *libraw_black,
        *dng_black_projected,
        *expected_wb,
        *actual_wb,
        *wb_relative_errors,
    ]
    if not all(math.isfinite(float(value)) for value in numeric_values):
        raise DngLibRawConformanceError("comparison produced non-finite values")

    return {
        "black_level": {
            "dng_nearest_integer": dng_black_projected,
            "libraw": libraw_black,
            "maximum_error_codes": max(black_errors),
        },
        "camera_white_balance": {
            "dng_reciprocal_as_shot_neutral": expected_wb,
            "libraw_green_normalized": actual_wb,
            "maximum_relative_error": max(wb_relative_errors),
        },
        "raw_geometry": {
            "dng": [raw_width, raw_height],
            "libraw": libraw_raw,
            "match": raw_geometry_match,
        },
        "visible_geometry": {
            "classification": visible_classification,
            "dng_default_crop": crop,
            "libraw": visible,
        },
        "warning_count": len(inspection.warnings),
        "white_level": {
            "dng_uniform": dng_white,
            "error_codes": white_error,
            "libraw": libraw_white,
        },
    }

