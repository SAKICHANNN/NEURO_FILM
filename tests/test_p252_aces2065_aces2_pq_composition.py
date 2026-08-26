from __future__ import annotations

from pathlib import Path

import numpy as np

from src.preprocess import (
    publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1,
)
from src.preprocess.ocio_aces2_output import convert_working_image_to_acescg
from src.preprocess.types import SourceProfile, WorkingImage


def _working(pixels: np.ndarray, working_space: str = "acescg_ap1_d60") -> WorkingImage:
    return WorkingImage(
        pixels=pixels,
        working_space=working_space,
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("icc", "test ACES2065-1"),
        hdr_metadata={},
        orientation_applied=False,
        alpha_policy="absent",
        bit_depth_in=32,
        source_path=Path("test.exr"),
    )


def test_exact_acescg_working_space_is_owned_identity() -> None:
    pixels = np.asarray(
        [[[-0.25, 0.0, 0.18], [1.0, 4.0, 16.0]]], dtype=np.float32
    )
    converted = convert_working_image_to_acescg(_working(pixels))
    assert np.array_equal(converted, pixels)
    assert converted.flags.c_contiguous
    assert converted.flags.owndata
    assert not np.shares_memory(converted, pixels)
    converted[...] = 0.0
    assert np.any(pixels != 0.0)


def test_composition_delegates_strict_loader_and_canonical_publisher(
    monkeypatch, tmp_path: Path
) -> None:
    import src.preprocess.aces2065_aces2_pq as module

    source = tmp_path / "master.exr"
    output = tmp_path / "display.png"
    working = _working(np.full((2, 3, 3), 0.18, dtype=np.float32))
    expected_samples = np.zeros((2, 3, 3), dtype=np.uint16)
    expected_encoded = np.zeros((2, 3, 3), dtype=np.float32)
    calls: list[tuple[object, ...]] = []

    def load(path: Path | str) -> WorkingImage:
        calls.append(("load", Path(path)))
        return working

    def publish(
        value: WorkingImage,
        path: Path,
        *,
        row_count: int,
        reverse_partition: bool,
    ) -> tuple[str, np.ndarray, np.ndarray]:
        calls.append(
            ("publish", value, path, row_count, reverse_partition)
        )
        return "digest", expected_samples, expected_encoded

    monkeypatch.setattr(module, "load_aces2065_openexr_working_image", load)
    monkeypatch.setattr(
        module, "publish_working_image_aces2_canonical_hdr_pq_png_v1", publish
    )
    result = publish_aces2065_openexr_aces2_canonical_hdr_pq_png_v1(
        source, output, row_count=7, reverse_partition=True
    )
    assert result[0] == "digest"
    assert result[1] is expected_samples
    assert result[2] is expected_encoded
    assert calls == [
        ("load", source),
        ("publish", working, output, 7, True),
    ]
