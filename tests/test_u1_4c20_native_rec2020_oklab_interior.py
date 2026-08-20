from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
import pytest

from src.color_engine.oklab_analytical_interior import (
    analytical_oklab_interior_rec2020,
)
from src.eval.native_rec2020_oklab_interior import (
    NativeRec2020InteriorError,
    apply_native_rec2020_interior,
    build_native_rec2020_interior,
    load_native_rec2020_interior,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> ctypes.CDLL:
    build = build_native_rec2020_interior(
        root=ROOT, output_dir=tmp_path_factory.mktemp("u1_4c20_native")
    )
    return load_native_rec2020_interior(Path(build["dll_path"]))


def test_native_mapper_matches_python_and_preserves_in_gamut(
    native_library: ctypes.CDLL,
) -> None:
    rng = np.random.default_rng(20260820)
    pixels = rng.uniform(-0.2, 1.2, size=(37, 53, 3)).astype(np.float32)
    pixels[::2, ::2] = rng.uniform(0.0, 1.0, size=pixels[::2, ::2].shape)
    expected, expected_scale = analytical_oklab_interior_rec2020(pixels)
    actual, actual_scale = apply_native_rec2020_interior(
        native_library, np.ascontiguousarray(pixels), thread_count=7
    )
    in_gamut = np.all((pixels >= 0.0) & (pixels <= 1.0), axis=2)
    assert np.array_equal(actual[in_gamut], pixels[in_gamut])
    assert np.max(np.abs(actual - expected)) <= 2.5e-7
    assert np.max(np.abs(actual_scale - expected_scale)) <= 2.5e-7
    assert np.isfinite(actual).all()


def test_native_mapper_replays_across_thread_partitions(
    native_library: ctypes.CDLL,
) -> None:
    values = np.linspace(-0.1, 1.1, 91 * 47 * 3, dtype=np.float32).reshape(91, 47, 3)
    serial = apply_native_rec2020_interior(native_library, values, thread_count=1)
    parallel = apply_native_rec2020_interior(native_library, values, thread_count=11)
    assert np.array_equal(serial[0], parallel[0])
    assert np.array_equal(serial[1], parallel[1])


def test_native_mapper_rejects_nonfinite_without_writing(
    native_library: ctypes.CDLL,
) -> None:
    pixels = np.zeros((3, 5, 3), dtype=np.float32)
    pixels[1, 2, 0] = np.nan
    output = np.full_like(pixels, 0.25)
    scale = np.full(pixels.shape[:2], 0.75, dtype=np.float32)
    with pytest.raises(NativeRec2020InteriorError, match="status 2"):
        apply_native_rec2020_interior(
            native_library, pixels, output=output, chroma_scale=scale
        )
    assert np.array_equal(output, np.full_like(pixels, 0.25))
    assert np.array_equal(scale, np.full(pixels.shape[:2], 0.75, dtype=np.float32))
