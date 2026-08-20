from pathlib import Path

import numpy as np
import pytest

from src.color_engine.safe_lab import apply_tone_rolloff, preserve_luma_detail
from src.eval.native_safe_lab_spatial_tone import (
    NativeSafeLabSpatialToneError,
    apply_native_safe_lab_spatial_tone,
    build_native_safe_lab_spatial_tone,
    load_native_safe_lab_spatial_tone,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory):
    build = build_native_safe_lab_spatial_tone(
        root=ROOT, output_dir=tmp_path_factory.mktemp("u1_4c25")
    )
    return load_native_safe_lab_spatial_tone(Path(build["dll_path"]))


def test_native_spatial_tone_matches_python_oracle(library) -> None:
    rng = np.random.default_rng(20260820)
    source = np.ascontiguousarray(rng.normal((52.0, 3.0, -2.0), (18.0, 15.0, 16.0), (31, 47, 3)), dtype=np.float32)
    pointwise = np.ascontiguousarray(source + rng.normal(0.0, (2.0, 4.0, 4.0), source.shape), dtype=np.float32)
    oracle = apply_tone_rolloff(
        preserve_luma_detail(source, pointwise, 0.9), 0.04, 1.0, 99.0
    )
    output = apply_native_safe_lab_spatial_tone(
        library,
        source,
        pointwise,
        detail_strength=0.9,
        tone_strength=0.04,
        shadow_floor_l=1.0,
        highlight_ceiling_l=99.0,
        thread_count=4,
    )
    assert np.max(np.abs(output - oracle)) <= 2.5e-5
    assert np.array_equal(output[..., 1:], pointwise[..., 1:])


def test_native_spatial_tone_replays_and_rejects_nonfinite_atomically(library) -> None:
    source = np.ascontiguousarray(np.full((9, 11, 3), (50.0, 2.0, -3.0), np.float32))
    pointwise = np.ascontiguousarray(source + np.float32(1.0))
    first = apply_native_safe_lab_spatial_tone(
        library,
        source,
        pointwise,
        detail_strength=0.9,
        tone_strength=0.04,
        shadow_floor_l=1.0,
        highlight_ceiling_l=99.0,
        thread_count=2,
    )
    second = apply_native_safe_lab_spatial_tone(
        library,
        source,
        pointwise,
        detail_strength=0.9,
        tone_strength=0.04,
        shadow_floor_l=1.0,
        highlight_ceiling_l=99.0,
        thread_count=2,
    )
    assert np.array_equal(first, second)
    invalid = source.copy()
    invalid[0, 0, 0] = np.nan
    output = np.full_like(source, -77.0)
    with pytest.raises(NativeSafeLabSpatialToneError):
        apply_native_safe_lab_spatial_tone(
            library,
            invalid,
            pointwise,
            detail_strength=0.9,
            tone_strength=0.04,
            shadow_floor_l=1.0,
            highlight_ceiling_l=99.0,
            output=output,
        )
    assert np.all(output == -77.0)
