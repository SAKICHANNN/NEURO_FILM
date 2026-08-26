from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import src.preprocess.dng_baseline_exposure as module
from src.preprocess.dng_baseline_exposure import (
    DngBaselineExposureError,
    DngBaselineExposureFacts,
    _apply_baseline_exposure,
    _single_rational,
    load_dng_baseline_exposed_working_image,
)
from src.preprocess.types import SourceProfile, WorkingImage


def _working(path: Path) -> WorkingImage:
    return WorkingImage(
        pixels=np.asarray([[[0.25, -0.5, 2.0]]], dtype=np.float32),
        working_space="linear_rec2020",
        transfer_state="scene_linear",
        source_transfer_state="scene_linear",
        source_profile=SourceProfile("raw_metadata", "parent"),
        hdr_metadata={},
        orientation_applied=True,
        alpha_policy="absent",
        bit_depth_in=16,
        source_path=path,
        warnings=[],
    )


def test_exposure_math_is_float32_and_does_not_mutate() -> None:
    pixels = np.asarray([[[0.25, -0.5, 2.0]]], dtype=np.float32)
    frozen = pixels.copy()
    output = _apply_baseline_exposure(pixels, 4.0, row_block=1)
    assert output.dtype == np.float32
    assert output.tolist() == [[[1.0, -2.0, 8.0]]]
    assert np.array_equal(pixels, frozen)


@pytest.mark.parametrize(
    ("value", "match"),
    [((1, 0), "zero denominator"), ((1, 2, 3, 4), "one rational")],
)
def test_rational_rejects_invalid_values(value: tuple[int, ...], match: str) -> None:
    with pytest.raises(DngBaselineExposureError, match=match):
        _single_rational(value, name="test")


@pytest.mark.parametrize(
    ("pixels", "scale", "row_block", "match"),
    [
        (np.zeros((1, 1, 3), np.float64), 1.0, 1, "float32"),
        (np.zeros((1, 1, 3), np.float32), 257.0, 1, "frozen exposure"),
        (np.zeros((1, 1, 3), np.float32), 1.0, 0, "positive integer"),
    ],
)
def test_exposure_rejects_invalid_inputs(
    pixels: np.ndarray, scale: float, row_block: int, match: str
) -> None:
    with pytest.raises(DngBaselineExposureError, match=match):
        _apply_baseline_exposure(pixels, scale, row_block=row_block)


def test_loader_applies_facts_and_preserves_parent(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "input.dng"
    source.write_bytes(b"source")
    parent = _working(source)
    frozen = parent.pixels.copy()
    facts = DngBaselineExposureFacts(
        baseline_exposure_ev=2.0,
        baseline_exposure_provenance="explicit_ifd_tag",
        baseline_exposure_offset_ev=0.0,
        baseline_exposure_offset_provenance="dng_standard_default",
        total_ev=2.0,
        scale=4.0,
    )
    monkeypatch.setattr(module, "_read_baseline_exposure", lambda path: facts)
    monkeypatch.setattr(
        module, "load_dng_forward_working_image", lambda *args, **kwargs: parent
    )
    output = load_dng_baseline_exposed_working_image(source)
    assert output.pixels.tolist() == [[[1.0, -2.0, 8.0]]]
    assert np.array_equal(parent.pixels, frozen)
    assert output.hdr_metadata["dng_baseline_exposure"] == facts.to_dict()
    assert output.warnings[-1].code == "private_dng_baseline_exposure"


def test_loader_rejects_source_hash_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "input.dng"
    source.write_bytes(b"source")
    with pytest.raises(DngBaselineExposureError, match="SHA-256"):
        load_dng_baseline_exposed_working_image(source, expected_source_sha256="0" * 64)
