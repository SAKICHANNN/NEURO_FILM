"""Strict private Radiance RGBE scanline decoder.

The returned values are unlabelled linear RGB radiance.  This module does not
assign primaries, a white point, transfer characteristics, or WorkingImage
state.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

import numpy as np

from src.film_physics.create_only_file import publish_create_only


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


def _quantize_radiance_rgbe(values: np.ndarray) -> np.ndarray:
    """Apply the official Radiance ``setcolr`` quantizer."""

    source = np.asarray(values)
    if (
        source.dtype == np.bool_
        or source.ndim != 3
        or source.shape[2] != 3
        or source.shape[0] <= 0
        or source.shape[1] < 8
        or source.shape[1] > 32767
        or not np.all(np.isfinite(source))
        or np.any(source < 0)
    ):
        raise RadianceRgbeError("invalid Radiance RGB input")
    working = np.ascontiguousarray(source, dtype=np.float64)
    maximum = np.max(working, axis=2)
    mantissa, exponent = np.frexp(maximum)
    nonzero = maximum > 1e-32
    biased = exponent + 128
    if np.any(nonzero & ((biased <= 0) | (biased > 255))):
        raise RadianceRgbeError("Radiance RGB exponent is out of range")
    scale = np.zeros_like(maximum)
    scale[nonzero] = mantissa[nonzero] * 255.9999 / maximum[nonzero]
    primaries = np.where(working > 0, working * scale[..., None], 0.0)
    if np.any(primaries >= 256.0):
        raise RadianceRgbeError("Radiance RGB quantization overflow")
    rgbe = np.empty((*source.shape[:2], 4), dtype=np.uint8)
    rgbe[..., :3] = primaries.astype(np.uint8)
    rgbe[..., 3] = np.where(nonzero, biased, 0).astype(np.uint8)
    return np.require(rgbe, dtype=np.uint8, requirements=["C", "W", "O"])


def _encode_rle_channel(channel: np.ndarray) -> bytes:
    """Encode one scanline channel using Radiance ``fwritecolrs`` rules."""

    result = bytearray()
    length = int(channel.size)
    index = 0
    count = 1
    while index < length:
        begin = index
        while begin < length:
            count = 1
            while (
                count < 127
                and begin + count < length
                and channel[begin + count] == channel[begin]
            ):
                count += 1
            if count >= 4:
                break
            begin += count
        if begin - index > 1 and begin - index < 4:
            second = index + 1
            while second < begin and channel[second] == channel[index]:
                second += 1
            if second == begin:
                result.extend((128 + begin - index, int(channel[index])))
                index = begin
        while index < begin:
            literal = min(begin - index, 128)
            result.append(literal)
            result.extend(channel[index : index + literal].tobytes())
            index += literal
        if count >= 4:
            result.extend((128 + count, int(channel[begin])))
        else:
            count = 0
        index += count
    return bytes(result)


def encode_radiance_rgbe_bytes(values: np.ndarray) -> bytes:
    """Encode unlabelled linear RGB radiance as canonical new-style RGBE."""

    rgbe = _quantize_radiance_rgbe(values)
    height, width = rgbe.shape[:2]
    output = bytearray(
        b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"
        + f"-Y {height} +X {width}\n".encode("ascii")
    )
    marker = bytes((2, 2, width >> 8, width & 255))
    for row in rgbe:
        output.extend(marker)
        for channel in range(4):
            output.extend(_encode_rle_channel(row[:, channel]))
    return bytes(output)


def write_radiance_rgbe_create_only(path: str | Path, values: np.ndarray) -> None:
    """Atomically publish one canonical RGBE file without overwriting."""

    destination = Path(path)
    if not destination.is_absolute() or not destination.parent.is_dir():
        raise RadianceRgbeError("Radiance destination preflight failed")
    payload = encode_radiance_rgbe_bytes(values)
    stage = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.stage"
    try:
        stage.write_bytes(payload)
        publish_create_only(stage, destination)
    except (OSError, ValueError) as exc:
        raise RadianceRgbeError("Radiance publication failed") from exc
    finally:
        stage.unlink(missing_ok=True)


def read_radiance_rgbe(path: str | Path) -> np.ndarray:
    """Read and decode one strict Radiance RGBE file."""

    return decode_radiance_rgbe_bytes(Path(path).read_bytes())


__all__ = [
    "RadianceRgbeError",
    "decode_radiance_rgbe_bytes",
    "encode_radiance_rgbe_bytes",
    "read_radiance_rgbe",
    "write_radiance_rgbe_create_only",
]
