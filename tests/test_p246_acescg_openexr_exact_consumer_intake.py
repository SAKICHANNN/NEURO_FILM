from __future__ import annotations

import hashlib
import json

from scripts.audit_p246_acescg_openexr_exact_consumer_intake import (
    _canonical_bytes,
    _synthetic_lattice,
)


def test_synthetic_lattice_matches_frozen_producer_pixel_identity() -> None:
    value = _synthetic_lattice()
    assert value.shape == (7, 9, 3)
    assert str(value.dtype) == "float32"
    assert (
        hashlib.sha256(value.astype("<f4", copy=False).tobytes()).hexdigest()
        == "3b14e75aa87a09794e0c4cb6339a4fc09862657d9aea190a9e3311c155bdb727"
    )
    assert value.min() < 0.0
    assert value.max() > 1.0


def test_canonical_report_is_order_independent_and_strict_json() -> None:
    first = _canonical_bytes({"z": [2, 1], "a": True})
    second = _canonical_bytes({"a": True, "z": [2, 1]})
    assert first == second
    assert first.endswith(b"\n")
    assert json.loads(first) == {"a": True, "z": [2, 1]}
