from __future__ import annotations

import numpy as np
import pytest

from src.eval.filmmatch_fresh_nonbasic_audit import FreshNonBasicAuditError, _sample_aligned


def test_bl9_aligned_sampling_is_deterministic_and_keeps_endpoints() -> None:
    source = np.arange(16 * 16 * 3, dtype=np.float32).reshape(16, 16, 3) / float(16 * 16 * 3)
    output = source * np.float32(0.75)
    first = _sample_aligned(source, output, 256)
    second = _sample_aligned(source, output, 256)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[0][[0, -1]], source.reshape(-1, 3)[[0, -1]])


def test_bl9_sampling_rejects_shape_mismatch() -> None:
    with pytest.raises(FreshNonBasicAuditError, match="shape mismatch"):
        _sample_aligned(np.zeros((16, 16, 3)), np.zeros((8, 8, 3)), 256)
