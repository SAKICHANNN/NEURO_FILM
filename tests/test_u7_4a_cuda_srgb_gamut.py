from __future__ import annotations

import numpy as np
import pytest

from scripts.pipeline_color_baseline import compress_to_srgb_gamut
from src.eval.cuda_srgb_gamut import compress_to_srgb_gamut_cuda

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def _pair() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260826)
    source = np.empty((127, 191, 3), dtype=np.float32)
    source[..., 0] = rng.uniform(3.0, 97.0, source.shape[:2])
    source[..., 1:] = rng.uniform(-45.0, 45.0, source.shape[:2] + (2,))
    target = source + rng.normal(0.0, (8.0, 28.0, 28.0), source.shape).astype(np.float32)
    return source, target


def test_cuda_candidate_is_close_and_repeat_deterministic() -> None:
    source, target = _pair()
    expected = compress_to_srgb_gamut(source, target, iterations=14)
    first = compress_to_srgb_gamut_cuda(source, target, iterations=14)
    second = compress_to_srgb_gamut_cuda(source, target, iterations=14)
    error = np.abs(first.output_lab - expected)
    assert float(error.max()) <= 0.01
    assert float(np.quantile(error, 0.99)) <= 2e-5
    assert np.array_equal(first.output_lab, second.output_lab)
    assert first.peak_allocated_bytes > 0


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_cuda_rejects_nonfinite_input(bad: float) -> None:
    source, target = _pair()
    target[0, 0, 0] = bad
    with pytest.raises(ValueError, match="finite"):
        compress_to_srgb_gamut_cuda(source, target)
