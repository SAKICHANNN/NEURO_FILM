from __future__ import annotations

import numpy as np
import pytest

from scripts.run_u5_r2spcp1_pixel_alignment_preflight import (
    alignment_metrics,
    normalized_correlation,
    tie_aware_midrank,
)


def test_midrank_is_invariant_to_strict_monotone_code_change() -> None:
    source = np.array([[0, 0, 4], [9, 9, 255]], dtype=np.uint8)
    mapped = np.array([[2, 2, 17], [50, 50, 251]], dtype=np.uint8)
    np.testing.assert_array_equal(tie_aware_midrank(source), tie_aware_midrank(mapped))


def test_alignment_metrics_are_exact_for_identical_input() -> None:
    y, x = np.mgrid[:64, :64]
    rgb = np.stack(((x * 3) % 256, (y * 5) % 256, ((x + y) * 7) % 256), axis=-1).astype(
        np.uint8
    )
    metrics = alignment_metrics(rgb, rgb.copy())
    assert metrics["gradient_ncc"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["rank_gradient_ncc"] == pytest.approx(1.0, abs=1e-12)
    assert metrics["phase_translation_pixels"] == pytest.approx(0.0, abs=1e-8)


def test_ncc_rejects_constant_degenerate_arrays() -> None:
    constant = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="denominator"):
        normalized_correlation(constant, constant)
