from __future__ import annotations

import math

import pytest

from src.color_match import (
    CanonicalEncodingError,
    canonical_bytes,
    canonical_sha256,
)


def test_canonical_v1_frozen_multitype_vector() -> None:
    payload = {
        "a": [None, True, -2, 1.5, "色"],
        "z": -0.0,
    }
    assert canonical_bytes(payload) == (
        b"d2:s1:al5:n;b1;i-2;f3ff8000000000000;"
        b"s3:\xe8\x89\xb2s1:zf8000000000000000;"
    )
    assert canonical_sha256(payload) == (
        "283b5aeb8ea476e896c26c821b83812976c7afa8ac30f1d87a70fcb6687188a5"
    )


def test_canonical_objects_ignore_insertion_order_but_preserve_types() -> None:
    first = {"b": 1.0, "a": 1}
    second = {"a": 1, "b": 1.0}
    assert canonical_bytes(first) == canonical_bytes(second)
    assert canonical_bytes(1) != canonical_bytes(1.0)
    assert canonical_bytes(0.0) != canonical_bytes(-0.0)


@pytest.mark.parametrize(
    "value",
    [
        float("nan"),
        float("inf"),
        b"bytes",
        {1: "non-string-key"},
    ],
)
def test_canonical_encoding_rejects_values_outside_wire_contract(
    value: object,
) -> None:
    with pytest.raises(CanonicalEncodingError):
        canonical_bytes(value)


def test_finite_float_bit_encoding_is_big_endian_binary64() -> None:
    encoded = canonical_bytes(math.pi)
    assert encoded == b"f400921fb54442d18;"
