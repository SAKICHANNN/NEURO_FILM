from __future__ import annotations

from pathlib import Path

import numpy as np

from src.eval.native_rec2020_postcolor_rgb16_v1 import (
    NativeRec2020PostcolorError,
    apply_native_rec2020_postcolor_v1,
    build_exact_rec2020_rgb16_thresholds,
    build_native_rec2020_postcolor_v1,
    load_native_rec2020_postcolor_v1,
)
from src.inference.romm_rec2020_velvia import _source_anchored_interior_residual
from src.preprocess import linear_rec2020_to_rec2020

ROOT = Path(__file__).resolve().parents[1]


def _reference(
    source: np.ndarray, candidate: np.ndarray, margin: float
) -> tuple[np.ndarray, np.ndarray]:
    output, scale = _source_anchored_interior_residual(
        source, candidate, margin=margin
    )
    encoded = linear_rec2020_to_rec2020(output)
    return np.rint(encoded * 65535.0).astype(np.uint16), scale


def test_native_postcolor_is_exact_on_random_and_boundary_values(tmp_path: Path) -> None:
    library = load_native_rec2020_postcolor_v1(
        Path(build_native_rec2020_postcolor_v1(root=ROOT, output_dir=tmp_path)["dll_path"])
    )
    rng = np.random.default_rng(14029)
    source = rng.random((129, 193, 3), dtype=np.float32)
    candidate = rng.random(source.shape, dtype=np.float32)
    source[0, :8] = np.array(
        [[0.0, 1.0, 0.5], [1.0, 0.0, 0.018053969]], dtype=np.float32
    ).repeat(4, axis=0)
    candidate[0, :8] = source[0, :8, ::-1]
    expected, expected_scale = _reference(source, candidate, 2.0 / 65535.0)
    actual, actual_scale = apply_native_rec2020_postcolor_v1(
        library,
        np.ascontiguousarray(source),
        np.ascontiguousarray(candidate),
        2.0 / 65535.0,
        thread_count=4,
    )
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_scale, expected_scale)


def test_native_postcolor_rejects_nonfinite_before_returning_output(tmp_path: Path) -> None:
    library = load_native_rec2020_postcolor_v1(
        Path(build_native_rec2020_postcolor_v1(root=ROOT, output_dir=tmp_path)["dll_path"])
    )
    source = np.zeros((2, 3, 3), dtype=np.float32)
    candidate = source.copy()
    candidate[1, 1, 1] = np.nan
    try:
        apply_native_rec2020_postcolor_v1(library, source, candidate, 2.0 / 65535.0)
    except NativeRec2020PostcolorError as error:
        assert "status 2" in str(error)
    else:
        raise AssertionError("nonfinite input must fail closed")


def test_native_postcolor_is_exact_at_every_quantization_threshold(tmp_path: Path) -> None:
    library = load_native_rec2020_postcolor_v1(
        Path(build_native_rec2020_postcolor_v1(root=ROOT, output_dir=tmp_path)["dll_path"])
    )
    thresholds = build_exact_rec2020_rgb16_thresholds()
    adjacent = np.nextafter(thresholds, np.float32(0.0))
    values = np.concatenate((adjacent, thresholds)).reshape(-1, 1, 1)
    source = np.ascontiguousarray(np.repeat(values, 3, axis=2))
    expected, expected_scale = _reference(source, source.copy(), 2.0 / 65535.0)
    actual, actual_scale = apply_native_rec2020_postcolor_v1(
        library,
        source,
        source.copy(),
        2.0 / 65535.0,
        thread_count=8,
        thresholds=thresholds,
    )
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual_scale, expected_scale)
