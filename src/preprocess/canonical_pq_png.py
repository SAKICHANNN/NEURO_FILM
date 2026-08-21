"""Versioned caller-partition-invariant Rec.2100 PQ PNG writer."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .color_management import REC2100_PQ_CICP
from .png_stream import _chunk, _StreamingRgbPngWriter


class CanonicalStreamingRec2100PqPngWriter(_StreamingRgbPngWriter):
    """Feed zlib fixed uncompressed blocks independent of caller writes."""

    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        canonical_feed_bytes: int = 64 * 1024,
    ) -> None:
        if canonical_feed_bytes != 64 * 1024:
            raise ValueError("canonical_feed_bytes must equal the frozen 65536")
        super().__init__(
            path,
            width=width,
            height=height,
            bit_depth=16,
            compression_level=0,
            cicp=REC2100_PQ_CICP,
        )
        self._canonical_pending = bytearray()
        self._canonical_feed_bytes = canonical_feed_bytes

    def _compress_canonical(self, payload: bytes) -> None:
        self._canonical_pending.extend(payload)
        while len(self._canonical_pending) >= self._canonical_feed_bytes:
            block = bytes(self._canonical_pending[: self._canonical_feed_bytes])
            del self._canonical_pending[: self._canonical_feed_bytes]
            self._pending.extend(self._compressor.compress(block))

    def write_rows(self, row_start: int, samples: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("PNG writer is closed")
        values = np.asarray(samples)
        if isinstance(row_start, bool) or not isinstance(row_start, int) or row_start != self._row:
            raise ValueError("PNG rows must be complete and strictly ordered")
        if (
            values.dtype != np.uint16
            or values.ndim != 3
            or values.shape[1:] != (self.width, 3)
            or values.shape[0] <= 0
            or self._row + values.shape[0] > self.height
            or not values.flags.c_contiguous
        ):
            raise ValueError("PNG tile must be contiguous RGB rows with exact sample type")
        rows = values.shape[0]
        row_bytes = self.width * 3 * 2
        filtered = np.empty((rows, row_bytes + 1), dtype=np.uint8)
        filtered[:, 0] = 0
        filtered[:, 1:] = values.byteswap().view(np.uint8).reshape(rows, row_bytes)
        self._compress_canonical(filtered.tobytes())
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
                self._pending.extend(self._compressor.compress(bytes(self._canonical_pending)))
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


__all__ = ["CanonicalStreamingRec2100PqPngWriter"]
