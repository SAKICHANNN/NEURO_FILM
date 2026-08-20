from __future__ import annotations

from pathlib import Path

import pytest

from src.preprocess.dng_libraw_conformance import (
    DngLibRawConformanceError,
    compare_dng_receipt_to_libraw,
)
from src.preprocess.types import InputInspection


def _tag(values: list[float]) -> dict[str, object]:
    return {"value": values}


def _receipt() -> dict[str, object]:
    return {
        "facts": {
            "image_width": _tag([4000]),
            "image_length": _tag([3000]),
            "white_level": _tag([1023]),
            "black_level": _tag(
                [
                    {"decimal": 64.2},
                    {"decimal": 64.4},
                    {"decimal": 64.1},
                    {"decimal": 64.3},
                ]
            ),
            "as_shot_neutral": _tag(
                [{"decimal": 0.5}, {"decimal": 1.0}, {"decimal": 0.25}]
            ),
            "default_crop_size": _tag([3968, 2976]),
        }
    }


def _inspection() -> InputInspection:
    return InputInspection(
        path=Path("fixture.dng"),
        exists=True,
        source_kind="raw",
        raw_metadata={
            "raw_width": 4000,
            "raw_height": 3000,
            "visible_width": 3968,
            "visible_height": 2976,
            "black_level_per_channel": [64, 64, 64, 64],
            "white_level": 1023,
            "camera_whitebalance": [2.0, 1.0, 4.0, 0.0],
        }
    )


def test_compare_exact_capture_facts() -> None:
    result = compare_dng_receipt_to_libraw(_receipt(), _inspection())
    assert result["raw_geometry"]["match"] is True
    assert result["visible_geometry"]["classification"] == "default_crop"
    assert result["black_level"]["maximum_error_codes"] == 0
    assert result["white_level"]["error_codes"] == 0.0
    assert result["camera_white_balance"]["maximum_relative_error"] == 0.0


def test_nonuniform_unmappable_black_level_fails_closed() -> None:
    receipt = _receipt()
    receipt["facts"]["black_level"] = _tag([1, 2, 3])
    with pytest.raises(DngLibRawConformanceError, match="cannot be mapped"):
        compare_dng_receipt_to_libraw(receipt, _inspection())
