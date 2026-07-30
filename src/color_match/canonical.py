"""Language-neutral canonical bytes for reference-match identity hashes."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any, Mapping, Sequence


class CanonicalEncodingError(ValueError):
    """Raised when a value cannot enter the v1 canonical identity stream."""


def _length(value: int) -> bytes:
    return str(value).encode("ascii") + b":"


def canonical_bytes(value: Any) -> bytes:
    """Encode the supported JSON value subset without text-float ambiguity."""

    if value is None:
        return b"n;"
    if isinstance(value, bool):
        return b"b1;" if value else b"b0;"
    if isinstance(value, int):
        return b"i" + str(value).encode("ascii") + b";"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalEncodingError(
                "canonical floats must be finite"
            )
        return b"f" + struct.pack(">d", value).hex().encode("ascii") + b";"
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return b"s" + _length(len(encoded)) + encoded
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise CanonicalEncodingError(
                "canonical object keys must be strings"
            )
        parts = [b"d", _length(len(value))]
        for key in sorted(value):
            parts.append(canonical_bytes(key))
            parts.append(canonical_bytes(value[key]))
        return b"".join(parts)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        parts = [b"l", _length(len(value))]
        parts.extend(canonical_bytes(item) for item in value)
        return b"".join(parts)
    raise CanonicalEncodingError(
        f"unsupported canonical value type: {type(value).__name__}"
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


__all__ = [
    "CanonicalEncodingError",
    "canonical_bytes",
    "canonical_sha256",
]
