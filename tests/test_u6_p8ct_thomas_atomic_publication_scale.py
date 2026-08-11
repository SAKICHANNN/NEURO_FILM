from __future__ import annotations

import hashlib

import numpy as np

from scripts.evaluate_u6_p8ct_thomas_atomic_publication_scale import (
    _exposure_fixture,
)


def test_exposure_fixture_matches_frozen_scalar_formula() -> None:
    height, width = 7, 11
    actual = _exposure_fixture(height, width)
    expected = np.empty((3, height, width), dtype=np.float32)
    for channel in range(3):
        for index in range(height * width):
            expected[channel].flat[index] = np.float32(
                ((index * 17 + channel * 13) % 193 - 96) / 64.0
            )
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert actual.tobytes() == expected.tobytes()


def test_exposure_fixture_replays_exactly() -> None:
    first = _exposure_fixture(65, 67)
    second = _exposure_fixture(65, 67)
    assert hashlib.sha256(first.tobytes()).digest() == hashlib.sha256(
        second.tobytes()
    ).digest()
