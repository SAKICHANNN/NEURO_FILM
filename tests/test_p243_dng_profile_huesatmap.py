from __future__ import annotations

import numpy as np
import pytest

from src.preprocess.dng_profile_huesatmap import apply_profile_huesatmap


def test_identity_map_is_identity() -> None:
    table = np.ones((1, 6, 6, 3), dtype=np.float32)
    table[..., 0] = 0.0
    rgb = np.array([[0.0, 0.0, 0.0], [0.2, 0.5, 0.7], [1.0, 0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(apply_profile_huesatmap(rgb, table), rgb, atol=2e-7, rtol=0)


def test_inputs_are_immutable() -> None:
    table = np.ones((2, 2, 2, 3), dtype=np.float32)
    table[..., 0] = 0.0
    rgb = np.array([[0.1, 0.3, 0.9]], dtype=np.float32)
    rgb_before, table_before = rgb.copy(), table.copy()
    apply_profile_huesatmap(rgb, table)
    np.testing.assert_array_equal(rgb, rgb_before)
    np.testing.assert_array_equal(table, table_before)


@pytest.mark.parametrize("shape", [(3,), (2, 2), (2, 2, 2, 2)])
def test_invalid_shape_rejects(shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError):
        apply_profile_huesatmap(np.zeros((1, 3), np.float32), np.ones(shape, np.float32))
