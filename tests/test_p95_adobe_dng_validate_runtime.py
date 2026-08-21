from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.audit_p95_adobe_dng_validate_runtime import (
    P95Error,
    _canonical_bytes,
    _inspect_tiff,
    _output_path,
)


def test_canonical_report_is_order_independent() -> None:
    assert _canonical_bytes({"z": 1, "a": 2}) == b'{"a":2,"z":1}\n'


def test_output_path_accepts_official_appended_suffix(tmp_path: Path) -> None:
    base = tmp_path / "stage2"
    appended = tmp_path / "stage2.tif"
    appended.write_bytes(b"bound")
    assert _output_path(base) == appended


def test_output_path_fails_closed_when_absent(tmp_path: Path) -> None:
    with pytest.raises(P95Error, match="output is absent"):
        _output_path(tmp_path / "missing")


def test_tiff_inspection_binds_uint16_rgb(tmp_path: Path) -> None:
    path = tmp_path / "rgb.tif"
    image = np.arange(8 * 9 * 3, dtype=np.uint16).reshape(8, 9, 3)
    tifffile.imwrite(path, image, photometric="rgb")
    facts = _inspect_tiff(path, expected_components=3)
    assert facts["dtype"] == "uint16"
    assert facts["height"] == 8
    assert facts["width"] == 9
    assert facts["samples_per_pixel"] == 3
    assert len(facts["sha256"]) == 64


def test_config_is_frozen_to_five_distinct_makes() -> None:
    config = json.loads(
        Path("configs/p95_adobe_dng_validate_runtime_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(config["rows"]) == 5
    assert len({row["camera_make"] for row in config["rows"]}) == 5
    assert [row["source_id"] for row in config["rows"]] == sorted(
        row["source_id"] for row in config["rows"]
    )
