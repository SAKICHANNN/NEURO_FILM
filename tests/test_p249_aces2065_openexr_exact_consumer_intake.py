from __future__ import annotations

import hashlib

import numpy as np

from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
    _synthetic_lattice,
)
from scripts.audit_p249_aces2065_openexr_exact_consumer_intake import (
    _expected_ap0,
    _normalize_text,
)

MATRIX = [
    [0.6954522414, 0.1406786965, 0.1638690622],
    [0.0447945634, 0.8596711185, 0.0955343182],
    [-0.0055258826, 0.0040252103, 1.0015006723],
]


def test_independent_ap0_conversion_matches_frozen_pixel_identity() -> None:
    value = _expected_ap0(_synthetic_lattice(), MATRIX)
    assert value.shape == (7, 9, 3)
    assert value.dtype == np.float32
    assert (
        hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
        == "2b3417e344aa7c26962a109f55d14237a0926d0e03eb46767477b274c4179cb4"
    )
    assert value.min() < 0.0
    assert value.max() > 1.0


def test_header_text_normalization_is_narrow() -> None:
    assert _normalize_text(b"lin_ap0_scene") == "lin_ap0_scene"
    assert _normalize_text("lin_ap0_scene") == "lin_ap0_scene"
