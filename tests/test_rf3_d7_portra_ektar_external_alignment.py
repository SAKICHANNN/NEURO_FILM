from __future__ import annotations

import numpy as np
import pytest

from src.eval.portra_ektar_external_alignment import (
    PortraEktarExternalAlignmentError,
    _median_delta_e76,
)


def test_delta_e_is_zero_only_for_identical_input() -> None:
    left = np.full((3, 4, 3), 0.25, dtype=np.float32)
    right = left.copy()
    assert _median_delta_e76(left, right) == 0.0
    right[:, :, 1] = 0.5
    assert _median_delta_e76(left, right) > 0.0


def test_delta_e_rejects_geometry_drift() -> None:
    with pytest.raises(PortraEktarExternalAlignmentError, match="geometry"):
        _median_delta_e76(
            np.zeros((2, 2, 3), dtype=np.float32),
            np.zeros((2, 3, 3), dtype=np.float32),
        )
