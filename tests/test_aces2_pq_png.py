from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from src.preprocess.aces2_pq_png import publish_working_image_aces2_hdr_pq_png_v1
from src.preprocess.png_stream import sha256_rec2100_pq_rgb16_png_samples
from src.preprocess.types import SourceProfile, WorkingImage


def _working() -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray(
            [[[0.01, 0.18, 1.0], [4.0, 0.1, 0.02]]],
            dtype=np.float32,
        ),
        working_space="linear_srgb",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "p90-test"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=Path("fixture.dng"),
    )


def test_aces2_pq_png_preserves_quantized_official_output(tmp_path: Path) -> None:
    output = tmp_path / "aces.png"
    file_sha, samples, encoded = publish_working_image_aces2_hdr_pq_png_v1(
        _working(),
        output,
        row_count=1,
    )
    expected = np.floor(encoded.astype(np.float64) * 65535.0 + 0.5).astype(np.uint16)
    np.testing.assert_array_equal(samples, expected)
    assert file_sha == hashlib.sha256(output.read_bytes()).hexdigest()
    assert sha256_rec2100_pq_rgb16_png_samples(
        output,
        width=2,
        height=1,
    ) == hashlib.sha256(samples.tobytes()).hexdigest()


def test_aces2_pq_png_is_create_only_and_validates_rows(tmp_path: Path) -> None:
    output = tmp_path / "existing.png"
    output.write_bytes(b"retained")
    with pytest.raises(FileExistsError, match="create-only"):
        publish_working_image_aces2_hdr_pq_png_v1(_working(), output)
    assert output.read_bytes() == b"retained"
    with pytest.raises(ValueError, match="row_count"):
        publish_working_image_aces2_hdr_pq_png_v1(
            _working(),
            tmp_path / "invalid.png",
            row_count=0,
        )
    assert not (tmp_path / "invalid.png").exists()
