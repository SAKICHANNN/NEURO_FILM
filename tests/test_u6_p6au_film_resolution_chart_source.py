from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.film_resolution_chart_source import _validate_tiff, load_contract

ROOT = Path(__file__).resolve().parents[1]


def test_p6au_contract_freezes_six_complete_resolution_groups() -> None:
    contract = load_contract(
        ROOT / "configs/u6_p6au_film_resolution_chart_source_v1.json"
    )
    members = contract["selection"]["members"]
    assert len(members) == 18
    for format_name in contract["selection"]["formats"]:
        for stock in contract["selection"]["stocks"]:
            names = [
                row["name"]
                for row in members
                if f"TIFF/{format_name}/{stock}/" in row["name"]
            ]
            assert sorted(name.split("Scans ")[1].split("/")[0] for name in names) == [
                "2K",
                "4K",
                "6K",
            ]


def test_p6au_rgb16_tiff_validation(tmp_path: Path) -> None:
    path = tmp_path / "rgb16.tif"
    values = np.arange(6 * 8 * 3, dtype=np.uint16).reshape(6, 8, 3)
    tifffile.imwrite(path, values, photometric="rgb")
    facts = _validate_tiff(path)
    assert facts["shape"] == [6, 8, 3]
    assert facts["dtype"] == "uint16"


def test_p6au_rejects_rgb8(tmp_path: Path) -> None:
    path = tmp_path / "rgb8.tif"
    tifffile.imwrite(path, np.zeros((4, 5, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError, match="not RGB16"):
        _validate_tiff(path)
