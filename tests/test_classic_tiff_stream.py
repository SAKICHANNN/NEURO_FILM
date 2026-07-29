from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from src.eval.classic_tiff_stream import (
    ClassicTiffStreamError,
    read_classic_tiff_zip_roi,
    summarize_classic_tiff_zip_member,
)


def _archive(tmp_path: Path, pixels: np.ndarray, *, name: str = "scan.tif") -> Path:
    tiff = tmp_path / name
    tifffile.imwrite(tiff, pixels, compression=None, rowsperstrip=1)
    archive = tmp_path / "scan.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        output.write(tiff, arcname=name)
    return archive


@pytest.mark.parametrize("channels", [1, 3])
def test_streamed_summary_matches_exact_uint16_pixels(
    tmp_path: Path, channels: int
) -> None:
    shape = (17, 23) if channels == 1 else (17, 23, 3)
    source = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    archive = _archive(tmp_path, source)
    summary = summarize_classic_tiff_zip_member(
        archive,
        "scan.tif",
        maximum_sample_rows=9,
        maximum_sample_columns=11,
    )
    expected = source[..., None] if channels == 1 else source
    assert summary.layout.width == 23
    assert summary.layout.height == 17
    assert summary.layout.samples_per_pixel == channels
    np.testing.assert_array_equal(
        summary.sampled_u16,
        expected[
            summary.sampled_row_indices[:, None],
            summary.sampled_column_indices[None, :],
        ],
    )
    np.testing.assert_allclose(summary.row_means, np.mean(expected, axis=1))
    np.testing.assert_allclose(summary.column_means, np.mean(expected, axis=0))


def test_wrong_member_and_compressed_tiff_fail_closed(tmp_path: Path) -> None:
    source = np.arange(17 * 23 * 3, dtype=np.uint16).reshape(17, 23, 3)
    archive = _archive(tmp_path, source)
    with pytest.raises(ClassicTiffStreamError, match="identity"):
        summarize_classic_tiff_zip_member(archive, "other.tif")

    tiff = tmp_path / "compressed.tif"
    tifffile.imwrite(tiff, source, compression="deflate", rowsperstrip=1)
    compressed = tmp_path / "compressed.zip"
    with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as output:
        output.write(tiff, arcname="compressed.tif")
    with pytest.raises(ClassicTiffStreamError, match="unsupported"):
        summarize_classic_tiff_zip_member(compressed, "compressed.tif")


def test_bounded_roi_matches_exact_pixels(tmp_path: Path) -> None:
    source = np.arange(29 * 31 * 3, dtype=np.uint16).reshape(29, 31, 3)
    archive = _archive(tmp_path, source)
    layout, roi = read_classic_tiff_zip_roi(
        archive,
        "scan.tif",
        row_start=5,
        row_stop=23,
        column_start=7,
        column_stop=19,
    )
    assert layout.width == 31
    np.testing.assert_array_equal(roi, source[5:23, 7:19])
    with pytest.raises(ClassicTiffStreamError, match="outside"):
        read_classic_tiff_zip_roi(
            archive,
            "scan.tif",
            row_start=-1,
            row_stop=3,
            column_start=0,
            column_stop=2,
        )
    with pytest.raises(ClassicTiffStreamError, match="exceeds"):
        read_classic_tiff_zip_roi(
            archive,
            "scan.tif",
            row_start=0,
            row_stop=29,
            column_start=0,
            column_stop=31,
            maximum_output_bytes=8,
        )


def test_unbounded_sample_request_and_truncation_fail_closed(
    tmp_path: Path,
) -> None:
    source = np.arange(9 * 11, dtype=np.uint16).reshape(9, 11)
    archive = _archive(tmp_path, source)
    with pytest.raises(ClassicTiffStreamError, match="positive"):
        summarize_classic_tiff_zip_member(
            archive, "scan.tif", maximum_sample_rows=0
        )
    with pytest.raises(ClassicTiffStreamError, match="outside"):
        summarize_classic_tiff_zip_member(
            archive, "scan.tif", maximum_uncompressed_bytes=8
        )
