"""Deterministic bounded-row RGB PNG publication."""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Self

import numpy as np

from .color_management import REC2020_SDR_CICP
from .output_encode import srgb_icc_profile

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDAT_PAYLOAD_BYTES = 64 * 1024


def _chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("PNG chunk type must contain four bytes")
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


class _StreamingRgbPngWriter:
    """Consume complete RGB rows in order and atomically publish one PNG."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        bit_depth: int,
        compression_level: int = 6,
        cicp: bytes | None = None,
    ) -> None:
        if path.suffix.casefold() != ".png":
            raise ValueError("streaming RGB output requires a .png extension")
        if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
            raise ValueError("width must be a positive integer")
        if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
            raise ValueError("height must be a positive integer")
        if isinstance(bit_depth, bool) or bit_depth not in (8, 16):
            raise ValueError("bit_depth must be 8 or 16")
        if not 0 <= compression_level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        self.path = path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.dtype = np.dtype(np.uint8 if bit_depth == 8 else np.uint16)
        self._row = 0
        self._closed = False
        self._pending = bytearray()
        self._digest = hashlib.sha256()
        self._compressor = zlib.compressobj(compression_level)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = path.with_suffix(path.suffix + ".stream.tmp")
        self._handle = self.temporary.open("xb")
        try:
            self._emit(_PNG_SIGNATURE)
            self._emit(
                _chunk(
                    b"IHDR",
                    struct.pack(">IIBBBBB", width, height, bit_depth, 2, 0, 0, 0),
                )
            )
            if cicp is None:
                profile = srgb_icc_profile()
                self._emit(
                    _chunk(
                        b"iCCP",
                        b"K-MCFM sRGB\x00\x00" + zlib.compress(profile, level=9),
                    )
                )
            else:
                if len(cicp) != 4:
                    raise ValueError("cICP payload must contain four bytes")
                self._emit(_chunk(b"cICP", cicp))
        except BaseException:
            self.abort()
            raise

    @property
    def rows_written(self) -> int:
        return self._row

    def _emit(self, payload: bytes) -> None:
        self._handle.write(payload)
        self._digest.update(payload)

    def _drain_idat(self, *, final: bool) -> None:
        while len(self._pending) >= _IDAT_PAYLOAD_BYTES or (final and self._pending):
            count = min(len(self._pending), _IDAT_PAYLOAD_BYTES)
            payload = bytes(self._pending[:count])
            del self._pending[:count]
            self._emit(_chunk(b"IDAT", payload))

    def write_rows(self, row_start: int, samples: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("PNG writer is closed")
        values = np.asarray(samples)
        if (
            isinstance(row_start, bool)
            or not isinstance(row_start, int)
            or row_start != self._row
        ):
            raise ValueError("PNG rows must be complete and strictly ordered")
        if (
            values.dtype != self.dtype
            or values.ndim != 3
            or values.shape[1:] != (self.width, 3)
            or values.shape[0] <= 0
            or self._row + values.shape[0] > self.height
            or not values.flags.c_contiguous
        ):
            raise ValueError("PNG tile must be contiguous RGB rows with exact sample type")
        rows = values.shape[0]
        row_bytes = self.width * 3 * self.dtype.itemsize
        filtered = np.empty((rows, row_bytes + 1), dtype=np.uint8)
        filtered[:, 0] = 0
        if self.bit_depth == 8:
            encoded = values.reshape(rows, row_bytes)
        else:
            encoded = values.byteswap().view(np.uint8).reshape(rows, row_bytes)
        filtered[:, 1:] = encoded
        self._pending.extend(self._compressor.compress(filtered.tobytes()))
        self._drain_idat(final=False)
        self._row += rows

    def finish(self) -> str:
        if self._closed:
            raise RuntimeError("PNG writer is closed")
        if self._row != self.height:
            self.abort()
            raise ValueError("PNG stream ended before all rows were written")
        try:
            self._pending.extend(self._compressor.flush())
            self._drain_idat(final=True)
            self._emit(_chunk(b"IEND", b""))
            self._handle.flush()
            os.fsync(self._handle.fileno())
            self._handle.close()
            os.replace(self.temporary, self.path)
            self._closed = True
            return self._digest.hexdigest()
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            self.temporary.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        if not self._closed:
            self.abort()


class StreamingSrgbPngWriter(_StreamingRgbPngWriter):
    """Consume sRGB rows and publish a PNG with the exact embedded sRGB ICC."""


class StreamingRec2020PngWriter(_StreamingRgbPngWriter):
    """Consume relative BT.2020 SDR rows and publish a CICP-tagged PNG."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        bit_depth: int,
        compression_level: int = 0,
    ) -> None:
        super().__init__(
            path,
            width=width,
            height=height,
            bit_depth=bit_depth,
            compression_level=compression_level,
            cicp=REC2020_SDR_CICP,
        )


__all__ = ["StreamingRec2020PngWriter", "StreamingSrgbPngWriter"]
