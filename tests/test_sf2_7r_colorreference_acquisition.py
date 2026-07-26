from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from scripts.run_sf2_7r_colorreference_acquisition import (
    _slide_id,
    audit_asset,
)


def _tiff_bytes(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (7, 5), (value, 2 * value, 3 * value)).save(
        output, format="TIFF"
    )
    return output.getvalue()


def test_slide_id_parses_frozen_archive_conventions() -> None:
    assert _slide_id("slide1.tif") == "1"
    assert _slide_id("Scan_5.TIFF") == "5"
    assert _slide_id("readme.txt") is None


def test_zip_crc_decode_and_slide_inventory(tmp_path: Path) -> None:
    path = tmp_path / "scan.zip"
    with ZipFile(path, "w") as archive:
        for slide_id in range(1, 6):
            archive.writestr(f"slide{slide_id}.tif", _tiff_bytes(slide_id * 10))
        archive.writestr("readme.txt", "fixture")
    record = audit_asset(path, "nikon_fixture")
    assert record["archive_crc_clean"] is True
    images = [member for member in record["members"] if "image" in member]
    assert [member["slide_id"] for member in images] == ["1", "2", "3", "4", "5"]
    assert all(member["image"]["format"] == "TIFF" for member in images)


def test_standalone_tiff_decodes(tmp_path: Path) -> None:
    path = tmp_path / "slide.tif"
    path.write_bytes(_tiff_bytes(20))
    record = audit_asset(path, "recorder_space_source")
    assert record["image"]["width"] == 7
    assert record["image"]["height"] == 5
