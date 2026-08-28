"""Strict private Radiance RGBE scanline decoder.

The returned values are unlabelled linear RGB radiance.  This module does not
assign primaries, a white point, transfer characteristics, or WorkingImage
state.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


class RadianceRgbeError(ValueError):
    """Raised when a Radiance RGBE payload violates the strict contract."""


def _line(payload: bytes, offset: int) -> tuple[bytes, int]:
    end = payload.find(b"\n", offset)
    if end < 0:
        raise RadianceRgbeError("truncated Radiance header")
    line = payload[offset:end]
    if line.endswith(b"\r"):
        line = line[:-1]
    return line, end + 1


def _parse_header(payload: bytes) -> tuple[int, int, int]:
    magic, offset = _line(payload, 0)
    if magic not in {b"#?RADIANCE", b"#?RGBE"}:
        raise RadianceRgbeError("unsupported Radiance magic")
    formats: list[bytes] = []
    while True:
        line, offset = _line(payload, offset)
        if not line:
            break
        if line.startswith(b"FORMAT="):
            formats.append(line.removeprefix(b"FORMAT="))
    if formats != [b"32-bit_rle_rgbe"]:
        raise RadianceRgbeError("unsupported or ambiguous Radiance format")
    resolution, offset = _line(payload, offset)
    match = re.fullmatch(rb"-Y ([1-9][0-9]*) \+X ([1-9][0-9]*)", resolution)
    if match is None:
        raise RadianceRgbeError("unsupported Radiance orientation")
    height, width = (int(value) for value in match.groups())
    if width < 8 or width > 32767:
        raise RadianceRgbeError("unsupported Radiance scanline width")
    return height, width, offset


def _decode_radiance_rgbe_codes_bytes(payload: bytes) -> np.ndarray:
    """Decode one strict payload to owned RGBE code bytes for audit use."""

    height, width, offset = _parse_header(payload)
    rgbe = np.empty((height, width, 4), dtype=np.uint8)
    view = memoryview(payload)
    for y_index in range(height):
        if offset + 4 > len(payload):
            raise RadianceRgbeError("truncated Radiance scanline marker")
        marker = payload[offset : offset + 4]
        offset += 4
        if marker[:2] != b"\x02\x02" or marker[2] & 0x80:
            raise RadianceRgbeError("old-style Radiance scanline is unsupported")
        encoded_width = (marker[2] << 8) | marker[3]
        if encoded_width != width:
            raise RadianceRgbeError("Radiance scanline width mismatch")
        for channel in range(4):
            x_index = 0
            while x_index < width:
                if offset >= len(payload):
                    raise RadianceRgbeError("truncated Radiance RLE packet")
                count = payload[offset]
                offset += 1
                if count == 0:
                    raise RadianceRgbeError("zero-length Radiance RLE packet")
                if count > 128:
                    run = count - 128
                    if x_index + run > width or offset >= len(payload):
                        raise RadianceRgbeError("invalid Radiance RLE run")
                    rgbe[y_index, x_index : x_index + run, channel] = payload[offset]
                    offset += 1
                    x_index += run
                else:
                    if x_index + count > width or offset + count > len(payload):
                        raise RadianceRgbeError("invalid Radiance RLE literal")
                    rgbe[y_index, x_index : x_index + count, channel] = view[
                        offset : offset + count
                    ]
                    offset += count
                    x_index += count
    if offset != len(payload):
        raise RadianceRgbeError("trailing bytes after Radiance image")
    return np.require(rgbe, dtype=np.uint8, requirements=["C", "W", "O"])


def decode_radiance_rgbe_bytes(payload: bytes) -> np.ndarray:
    """Decode one strict new-style RLE Radiance RGBE payload."""

    rgbe = _decode_radiance_rgbe_codes_bytes(payload)
    height, width = rgbe.shape[:2]
    exponent = rgbe[..., 3]
    output = np.zeros((height, width, 3), dtype=np.float32)
    nonzero = exponent != 0
    if np.any(nonzero):
        scale = np.ldexp(
            np.ones(np.count_nonzero(nonzero), dtype=np.float32),
            exponent[nonzero].astype(np.int32) - 136,
        )
        output[nonzero] = (rgbe[nonzero, :3].astype(np.float32) + 0.5) * scale[:, None]
    if not np.all(np.isfinite(output)) or np.any(output < 0):
        raise RadianceRgbeError("decoded Radiance values are invalid")
    return np.require(output, dtype=np.float32, requirements=["C", "W", "O"])


def read_radiance_rgbe(path: str | Path) -> np.ndarray:
    """Read and decode one strict Radiance RGBE file."""

    return decode_radiance_rgbe_bytes(Path(path).read_bytes())


__all__ = [
    "RadianceRgbeError",
    "decode_radiance_rgbe_bytes",
    "read_radiance_rgbe",
]
