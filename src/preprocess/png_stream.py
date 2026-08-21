"""Deterministic bounded-row RGB PNG publication."""

from __future__ import annotations

import hashlib
import os
import struct
import zlib
from pathlib import Path
from typing import Self

import numpy as np

from .color_management import REC2020_SDR_CICP, REC2100_PQ_CICP
from .output_encode import srgb_icc_profile

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDAT_PAYLOAD_BYTES = 64 * 1024
_MAX_VERIFIER_CHUNK_BYTES = 64 * 1024 * 1024


def _chunk(kind: bytes, payload: bytes) -> bytes:
    if len(kind) != 4:
        raise ValueError("PNG chunk type must contain four bytes")
    crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)


def sha256_rec2020_rgb16_png_samples(
    path: Path,
    *,
    width: int,
    height: int,
) -> str:
    """Strictly stream and hash native-order RGB16 samples from our PNG rail."""

    return _sha256_cicp_rgb16_png_samples(
        path, width=width, height=height, expected_cicp=REC2020_SDR_CICP
    )


def sha256_rec2100_pq_rgb16_png_samples(
    path: Path,
    *,
    width: int,
    height: int,
) -> str:
    """Strictly hash native RGB16 samples from the Rec.2100 PQ PNG rail."""

    return _sha256_cicp_rgb16_png_samples(
        path, width=width, height=height, expected_cicp=REC2100_PQ_CICP
    )


def _sha256_cicp_rgb16_png_samples(
    path: Path,
    *,
    width: int,
    height: int,
    expected_cicp: bytes,
) -> str:

    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("width must be a positive integer")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("height must be a positive integer")
    row_bytes = width * 3 * 2
    scanline_bytes = row_bytes + 1
    pending = bytearray()
    digest = hashlib.sha256()
    decoder = zlib.decompressobj()
    rows = 0
    seen_ihdr = False
    seen_cicp = False
    seen_idat = False
    idat_ended = False
    seen_iend = False

    def consume(decoded: bytes) -> None:
        nonlocal rows
        pending.extend(decoded)
        complete = len(pending) // scanline_bytes
        if complete == 0:
            return
        if rows + complete > height:
            raise ValueError("PNG contains more rows than declared")
        count = complete * scanline_bytes
        block = bytes(pending[:count])
        del pending[:count]
        matrix = np.frombuffer(block, dtype=np.uint8).reshape(complete, scanline_bytes)
        if np.any(matrix[:, 0] != 0):
            raise ValueError("streaming RGB16 PNG must use filter type zero")
        encoded = np.ascontiguousarray(matrix[:, 1:])
        samples = np.frombuffer(encoded, dtype=">u2").astype(np.uint16)
        digest.update(samples.tobytes())
        rows += complete

    with Path(path).open("rb") as handle:
        if handle.read(len(_PNG_SIGNATURE)) != _PNG_SIGNATURE:
            raise ValueError("invalid PNG signature")
        while not seen_iend:
            header = handle.read(8)
            if len(header) != 8:
                raise ValueError("truncated PNG chunk header")
            length, kind = struct.unpack(">I4s", header)
            if length > _MAX_VERIFIER_CHUNK_BYTES:
                raise ValueError("PNG chunk exceeds verifier bound")
            payload = handle.read(length)
            stored_crc = handle.read(4)
            if len(payload) != length or len(stored_crc) != 4:
                raise ValueError("truncated PNG chunk")
            actual_crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
            if struct.unpack(">I", stored_crc)[0] != actual_crc:
                raise ValueError("PNG chunk CRC mismatch")

            if kind == b"IHDR":
                if seen_ihdr or seen_cicp or seen_idat or length != 13:
                    raise ValueError("invalid IHDR placement")
                expected = struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0)
                if payload != expected:
                    raise ValueError("unexpected RGB16 PNG layout")
                seen_ihdr = True
            elif kind == b"cICP":
                if not seen_ihdr or seen_cicp or seen_idat or payload != expected_cicp:
                    raise ValueError("unexpected RGB16 cICP metadata")
                seen_cicp = True
            elif kind == b"IDAT":
                if not seen_ihdr or not seen_cicp or idat_ended:
                    raise ValueError("invalid IDAT placement")
                seen_idat = True
                compressed = payload
                output_bound = max(scanline_bytes, _IDAT_PAYLOAD_BYTES)
                while compressed:
                    consume(decoder.decompress(compressed, output_bound))
                    if decoder.unused_data:
                        raise ValueError("invalid compressed PNG sample stream")
                    tail = decoder.unconsumed_tail
                    if tail and len(tail) >= len(compressed):
                        raise ValueError("PNG decompressor made no progress")
                    compressed = tail
            elif kind == b"IEND":
                if not seen_idat or length != 0:
                    raise ValueError("invalid IEND chunk")
                idat_ended = True
                consume(decoder.flush())
                if (
                    not decoder.eof
                    or decoder.unused_data
                    or decoder.unconsumed_tail
                    or pending
                    or rows != height
                ):
                    raise ValueError("incomplete PNG sample stream")
                seen_iend = True
            else:
                raise ValueError("unexpected chunk in deterministic Rec.2020 PNG")

            if seen_idat and kind not in {b"IDAT", b"IEND"}:
                idat_ended = True

        if handle.read(1):
            raise ValueError("trailing bytes after IEND")
    return digest.hexdigest()


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
        canonical_feed_bytes: int | None = None,
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
        if canonical_feed_bytes is not None and (
            isinstance(canonical_feed_bytes, bool)
            or not isinstance(canonical_feed_bytes, int)
            or canonical_feed_bytes <= 0
        ):
            raise ValueError("canonical_feed_bytes must be a positive integer")
        self.path = path
        self.width = width
        self.height = height
        self.bit_depth = bit_depth
        self.dtype = np.dtype(np.uint8 if bit_depth == 8 else np.uint16)
        self._row = 0
        self._closed = False
        self._pending = bytearray()
        self._canonical_pending = bytearray()
        self._canonical_feed_bytes = canonical_feed_bytes
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

    def _compress_filtered(self, payload: bytes) -> None:
        if self._canonical_feed_bytes is None:
            self._pending.extend(self._compressor.compress(payload))
            return
        self._canonical_pending.extend(payload)
        block_size = self._canonical_feed_bytes
        while len(self._canonical_pending) >= block_size:
            block = bytes(self._canonical_pending[:block_size])
            del self._canonical_pending[:block_size]
            self._pending.extend(self._compressor.compress(block))

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
        self._compress_filtered(filtered.tobytes())
        self._drain_idat(final=False)
        self._row += rows

    def finish(self) -> str:
        if self._closed:
            raise RuntimeError("PNG writer is closed")
        if self._row != self.height:
            self.abort()
            raise ValueError("PNG stream ended before all rows were written")
        try:
            if self._canonical_pending:
                self._pending.extend(
                    self._compressor.compress(bytes(self._canonical_pending))
                )
                self._canonical_pending.clear()
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


class StreamingRec2100PqPngWriter(_StreamingRgbPngWriter):
    """Consume full-range BT.2100 PQ RGB rows and publish a CICP PNG."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        bit_depth: int = 16,
        compression_level: int = 0,
    ) -> None:
        if bit_depth != 16:
            raise ValueError("Rec.2100 PQ rail requires 16-bit RGB samples")
        super().__init__(
            path,
            width=width,
            height=height,
            bit_depth=bit_depth,
            compression_level=compression_level,
            cicp=REC2100_PQ_CICP,
        )


class CanonicalStreamingRec2100PqPngWriter(_StreamingRgbPngWriter):
    """PQ writer with zlib input boundaries independent of caller partitions."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        bit_depth: int = 16,
        compression_level: int = 0,
        canonical_feed_bytes: int = 64 * 1024,
    ) -> None:
        if bit_depth != 16:
            raise ValueError("Rec.2100 PQ rail requires 16-bit RGB samples")
        super().__init__(
            path,
            width=width,
            height=height,
            bit_depth=bit_depth,
            compression_level=compression_level,
            cicp=REC2100_PQ_CICP,
            canonical_feed_bytes=canonical_feed_bytes,
        )


__all__ = [
    "CanonicalStreamingRec2100PqPngWriter",
    "StreamingRec2020PngWriter",
    "StreamingRec2100PqPngWriter",
    "StreamingSrgbPngWriter",
    "sha256_rec2020_rgb16_png_samples",
    "sha256_rec2100_pq_rgb16_png_samples",
]
