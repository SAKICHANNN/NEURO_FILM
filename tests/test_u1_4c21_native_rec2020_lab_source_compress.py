from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.color_engine.gamut import compress_source_to_working_gamut
from src.color_engine.lab import lab_to_linear_rgb, linear_rgb_to_lab
from src.eval.native_rec2020_lab_source_compress import (
    NativeRec2020LabCompressError,
    apply_native_rec2020_lab_compress,
    build_native_rec2020_lab_compress,
    load_native_rec2020_lab_compress,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory):
    build = build_native_rec2020_lab_compress(
        root=ROOT, output_dir=tmp_path_factory.mktemp("u1_4c21_native")
    )
    return load_native_rec2020_lab_compress(Path(build["dll_path"]))


def _fixture() -> tuple[np.ndarray, np.ndarray]:
    source_rgb = np.asarray(
        [[[0.10, 0.20, 0.30], [0.85, 0.10, 0.05]], [[0.02, 0.90, 0.12], [0.95, 0.95, 0.95]]],
        dtype=np.float32,
    )
    source_lab = linear_rgb_to_lab(source_rgb, working_space="linear_rec2020")
    target_lab = source_lab.copy()
    target_lab[..., 0] += np.asarray([[4.0, 18.0], [-8.0, 3.0]], dtype=np.float32)
    target_lab[..., 1] += np.asarray([[2.0, 42.0], [-35.0, 1.0]], dtype=np.float32)
    target_lab[..., 2] += np.asarray([[-3.0, 31.0], [28.0, -2.0]], dtype=np.float32)
    return np.ascontiguousarray(source_lab), np.ascontiguousarray(target_lab)


def test_native_matches_python_source_compression(native_library) -> None:
    source_lab, target_lab = _fixture()
    expected_lab = compress_source_to_working_gamut(
        source_lab, target_lab, working_space="linear_rec2020"
    )
    expected_rgb = lab_to_linear_rgb(expected_lab, working_space="linear_rec2020")
    actual_lab, actual_rgb, _ = apply_native_rec2020_lab_compress(
        native_library, source_lab, target_lab, thread_count=2
    )
    assert np.max(np.abs(actual_lab - expected_lab)) <= 2.5e-5
    assert np.max(np.abs(actual_rgb - expected_rgb)) <= 2.5e-6
    assert np.isfinite(actual_rgb).all()
    assert np.all(actual_rgb >= -2e-6)
    assert np.all(actual_rgb <= 1.0 + 2e-6)


def test_thread_partition_is_exact(native_library) -> None:
    source_lab, target_lab = _fixture()
    serial = apply_native_rec2020_lab_compress(
        native_library, source_lab, target_lab, thread_count=1
    )
    parallel = apply_native_rec2020_lab_compress(
        native_library, source_lab, target_lab, thread_count=4
    )
    for left, right in zip(serial, parallel, strict=True):
        assert np.array_equal(left, right)


def test_nonfinite_failure_is_atomic(native_library) -> None:
    source_lab, target_lab = _fixture()
    target_lab[0, 0, 0] = np.nan
    output_lab = np.full_like(source_lab, 17.0)
    output_rgb = np.full_like(source_lab, 19.0)
    scale = np.full(source_lab.shape[:2], 23.0, dtype=np.float32)
    with pytest.raises(NativeRec2020LabCompressError):
        apply_native_rec2020_lab_compress(
            native_library,
            source_lab,
            target_lab,
            output_lab=output_lab,
            output_rgb=output_rgb,
            scale=scale,
        )
    assert np.all(output_lab == 17.0)
    assert np.all(output_rgb == 19.0)
    assert np.all(scale == 23.0)
