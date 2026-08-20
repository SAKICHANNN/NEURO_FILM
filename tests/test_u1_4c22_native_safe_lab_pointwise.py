from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.color_engine.safe_lab import (
    apply_chroma_curve,
    apply_color_guardrails,
    safe_lab_context_from_lab,
)
from src.eval.native_safe_lab_pointwise import (
    NativeSafeLabPointwiseError,
    apply_native_safe_lab_pointwise,
    build_native_safe_lab_pointwise,
    load_native_safe_lab_pointwise,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory):
    build = build_native_safe_lab_pointwise(
        root=ROOT, output_dir=tmp_path_factory.mktemp("u1_4c22_native")
    )
    return load_native_safe_lab_pointwise(Path(build["dll_path"]))


def _fixture() -> np.ndarray:
    return np.asarray(
        [
            [[15.0, 1.0, -2.0], [55.0, 16.0, 22.0], [82.0, 35.0, -45.0]],
            [[32.0, -18.0, 12.0], [66.0, 5.0, 8.0], [98.0, 0.5, 0.25]],
        ],
        dtype=np.float32,
    )


def _python_pointwise(source: np.ndarray) -> np.ndarray:
    context = safe_lab_context_from_lab(source)
    destination_mean = np.asarray([41.78747, 2.28839, -3.78775], dtype=np.float32)
    destination_std = np.asarray([26.84055, 12.73181, 23.40809], dtype=np.float32)
    source_mean = np.asarray(context.lab_mean, dtype=np.float32)
    source_std = np.asarray(context.lab_std, dtype=np.float32)
    transferred = (source - source_mean) / source_std * destination_std + destination_mean
    output = source.copy()
    output[..., 0] = source[..., 0] + 0.02 * 0.35 * (transferred[..., 0] - source[..., 0])
    output[..., 1:] = source[..., 1:] + 0.35 * (transferred[..., 1:] - source[..., 1:])
    output = apply_chroma_curve(source, output, 0.45)
    return apply_color_guardrails(
        source,
        output,
        neutral_protect=0.55,
        skin_protect=0.35,
        max_chroma_gain=2.1,
        max_chroma_boost=16.0,
        max_chroma_absolute=None,
    )


def test_native_matches_python_pointwise_chain(native_library) -> None:
    source = _fixture()
    context = safe_lab_context_from_lab(source)
    actual = apply_native_safe_lab_pointwise(
        native_library,
        source,
        source_context=context,
        destination_mean=np.asarray([41.78747, 2.28839, -3.78775], dtype=np.float32),
        destination_std=np.asarray([26.84055, 12.73181, 23.40809], dtype=np.float32),
        strength=0.35,
        luma_strength=0.02,
        chroma_curve_strength=0.45,
        neutral_protect=0.55,
        skin_protect=0.35,
        max_chroma_gain=2.1,
        max_chroma_boost=16.0,
        max_chroma_absolute=None,
        thread_count=2,
    )
    assert np.max(np.abs(actual - _python_pointwise(source))) <= 2.5e-5


def test_thread_partition_is_exact(native_library) -> None:
    source = _fixture()
    context = safe_lab_context_from_lab(source)
    kwargs = {
        "source_context": context,
        "destination_mean": np.asarray(
            [41.78747, 2.28839, -3.78775], dtype=np.float32
        ),
        "destination_std": np.asarray(
            [26.84055, 12.73181, 23.40809], dtype=np.float32
        ),
        "strength": 0.35,
        "luma_strength": 0.02,
        "chroma_curve_strength": 0.45,
        "neutral_protect": 0.55,
        "skin_protect": 0.35,
        "max_chroma_gain": 2.1,
        "max_chroma_boost": 16.0,
        "max_chroma_absolute": None,
    }
    serial = apply_native_safe_lab_pointwise(native_library, source, thread_count=1, **kwargs)
    parallel = apply_native_safe_lab_pointwise(native_library, source, thread_count=4, **kwargs)
    assert np.array_equal(serial, parallel)


def test_nonfinite_failure_is_atomic(native_library) -> None:
    source = _fixture()
    context = safe_lab_context_from_lab(source)
    source[0, 0, 0] = np.nan
    output = np.full_like(source, 17.0)
    with pytest.raises(NativeSafeLabPointwiseError):
        apply_native_safe_lab_pointwise(
            native_library,
            source,
            source_context=context,
            destination_mean=np.ones(3, dtype=np.float32),
            destination_std=np.ones(3, dtype=np.float32),
            strength=0.35,
            luma_strength=0.02,
            chroma_curve_strength=0.45,
            neutral_protect=0.55,
            skin_protect=0.35,
            max_chroma_gain=2.1,
            max_chroma_boost=16.0,
            max_chroma_absolute=None,
            output=output,
        )
    assert np.all(output == 17.0)
