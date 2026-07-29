"""Bounded row-streaming summaries for classic uint16 strip TIFF members."""

from __future__ import annotations

import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np


class ClassicTiffStreamError(ValueError):
    """Raised when an input escapes the intentionally narrow TIFF contract."""


_TYPE_SIZE = {1: 1, 2: 1, 3: 2, 4: 4, 5: 8}
_MAX_IFD_OFFSET = 16 * 1024 * 1024
_MAX_IFD_ENTRIES = 256
_MAX_PREFIX_BYTES = 32 * 1024 * 1024


@dataclass(frozen=True)
class ClassicTiffLayout:
    endian: str
    width: int
    height: int
    samples_per_pixel: int
    bits_per_sample: tuple[int, ...]
    photometric: int
    rows_per_strip: int
    strip_offsets: tuple[int, ...]
    strip_byte_counts: tuple[int, ...]
    sample_format: tuple[int, ...]
    first_strip_offset: int
    row_bytes: int


@dataclass(frozen=True)
class ClassicTiffSummary:
    layout: ClassicTiffLayout
    sampled_row_indices: np.ndarray
    sampled_column_indices: np.ndarray
    sampled_u16: np.ndarray
    row_means: np.ndarray
    column_means: np.ndarray


def _read_exact(handle: BinaryIO, count: int) -> bytes:
    if count < 0:
        raise ClassicTiffStreamError("negative read length")
    output = bytearray()
    while len(output) < count:
        block = handle.read(count - len(output))
        if not block:
            raise ClassicTiffStreamError("truncated TIFF member")
        output.extend(block)
    return bytes(output)


def _discard_exact(handle: BinaryIO, count: int) -> None:
    remaining = count
    while remaining:
        block = handle.read(min(remaining, 1024 * 1024))
        if not block:
            raise ClassicTiffStreamError("truncated TIFF padding")
        remaining -= len(block)


def _scalar_or_values(
    prefix: bytes,
    *,
    endian: str,
    type_code: int,
    count: int,
    value_field: bytes,
    value_offset: int,
) -> tuple[int, ...]:
    if type_code not in (3, 4):
        raise ClassicTiffStreamError("unsupported numeric TIFF tag type")
    size = _TYPE_SIZE[type_code] * count
    raw = (
        value_field[:size]
        if size <= 4
        else prefix[value_offset : value_offset + size]
    )
    if len(raw) != size:
        raise ClassicTiffStreamError("TIFF tag payload escaped prefix")
    code = "H" if type_code == 3 else "I"
    return tuple(int(value) for value in struct.unpack(endian + code * count, raw))


def read_classic_tiff_layout(handle: BinaryIO) -> tuple[ClassicTiffLayout, int]:
    """Read a narrow classic TIFF header and leave the stream at prefix end."""

    header = _read_exact(handle, 8)
    if header[:2] == b"II":
        endian = "<"
    elif header[:2] == b"MM":
        endian = ">"
    else:
        raise ClassicTiffStreamError("unsupported TIFF byte order")
    magic, ifd_offset = struct.unpack(endian + "HI", header[2:8])
    if magic != 42:
        raise ClassicTiffStreamError("only classic TIFF is supported")
    if ifd_offset < 8 or ifd_offset > _MAX_IFD_OFFSET:
        raise ClassicTiffStreamError("IFD offset is outside the bound")
    prefix = bytearray(header)
    prefix.extend(_read_exact(handle, ifd_offset + 2 - len(prefix)))
    entry_count = struct.unpack(
        endian + "H", prefix[ifd_offset : ifd_offset + 2]
    )[0]
    if entry_count <= 0 or entry_count > _MAX_IFD_ENTRIES:
        raise ClassicTiffStreamError("IFD entry count is outside the bound")
    ifd_end = ifd_offset + 2 + 12 * entry_count + 4
    prefix.extend(_read_exact(handle, ifd_end - len(prefix)))

    tags: dict[int, tuple[int, int, bytes, int]] = {}
    maximum_payload_end = ifd_end
    for index in range(entry_count):
        offset = ifd_offset + 2 + index * 12
        type_code, count = struct.unpack(
            endian + "HI", prefix[offset + 2 : offset + 8]
        )
        tag = struct.unpack(endian + "H", prefix[offset : offset + 2])[0]
        value_field = bytes(prefix[offset + 8 : offset + 12])
        value_offset = struct.unpack(endian + "I", value_field)[0]
        if type_code in _TYPE_SIZE and _TYPE_SIZE[type_code] * count > 4:
            maximum_payload_end = max(
                maximum_payload_end,
                value_offset + _TYPE_SIZE[type_code] * count,
            )
        tags[tag] = (type_code, count, value_field, value_offset)
    if maximum_payload_end > _MAX_PREFIX_BYTES:
        raise ClassicTiffStreamError("TIFF metadata prefix is outside the bound")
    prefix.extend(_read_exact(handle, maximum_payload_end - len(prefix)))

    def values(tag: int, default: tuple[int, ...] | None = None) -> tuple[int, ...]:
        if tag not in tags:
            if default is None:
                raise ClassicTiffStreamError(f"required TIFF tag {tag} is missing")
            return default
        type_code, count, value_field, value_offset = tags[tag]
        return _scalar_or_values(
            bytes(prefix),
            endian=endian,
            type_code=type_code,
            count=count,
            value_field=value_field,
            value_offset=value_offset,
        )

    width = values(256)[0]
    height = values(257)[0]
    compression = values(259)[0]
    photometric = values(262)[0]
    samples = values(277, (1,))[0]
    rows_per_strip = values(278, (height,))[0]
    bits = values(258, (1,))
    offsets = values(273)
    byte_counts = values(279)
    planar = values(284, (1,))[0]
    sample_format = values(339, tuple(1 for _ in range(samples)))
    orientation = values(274, (1,))[0]
    if (
        width <= 0
        or height <= 0
        or samples not in (1, 3)
        or bits != tuple(16 for _ in range(samples))
        or compression != 1
        or planar != 1
        or orientation != 1
        or sample_format != tuple(1 for _ in range(samples))
        or photometric not in ((1, 0) if samples == 1 else (2,))
        or rows_per_strip <= 0
    ):
        raise ClassicTiffStreamError("TIFF pixel layout is unsupported")
    expected_strips = (height + rows_per_strip - 1) // rows_per_strip
    if len(offsets) != expected_strips or len(byte_counts) != expected_strips:
        raise ClassicTiffStreamError("TIFF strip inventory is inconsistent")
    row_bytes = width * samples * 2
    for strip_index, byte_count in enumerate(byte_counts):
        rows = min(rows_per_strip, height - strip_index * rows_per_strip)
        if byte_count != rows * row_bytes:
            raise ClassicTiffStreamError("TIFF strip byte count is inconsistent")
    if any(right < left for left, right in zip(offsets, offsets[1:])):
        raise ClassicTiffStreamError("TIFF strip offsets are not ordered")
    first_strip = offsets[0]
    if first_strip < len(prefix):
        raise ClassicTiffStreamError("TIFF strip overlaps metadata")
    layout = ClassicTiffLayout(
        endian=endian,
        width=width,
        height=height,
        samples_per_pixel=samples,
        bits_per_sample=bits,
        photometric=photometric,
        rows_per_strip=rows_per_strip,
        strip_offsets=offsets,
        strip_byte_counts=byte_counts,
        sample_format=sample_format,
        first_strip_offset=first_strip,
        row_bytes=row_bytes,
    )
    return layout, len(prefix)


def summarize_classic_tiff_zip_member(
    archive_path: Path,
    member_name: str,
    *,
    maximum_sample_rows: int = 1024,
    maximum_sample_columns: int = 1024,
    maximum_uncompressed_bytes: int = 6_000_000_000,
) -> ClassicTiffSummary:
    """Stream one exact member and retain bounded geometry/statistics only."""

    if maximum_sample_rows <= 0 or maximum_sample_columns <= 0:
        raise ClassicTiffStreamError("sample bounds must be positive")
    with zipfile.ZipFile(archive_path, "r") as archive:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if len(infos) != 1 or infos[0].filename != member_name:
            raise ClassicTiffStreamError("ZIP member identity is not exact")
        info = infos[0]
        if info.file_size <= 0 or info.file_size > maximum_uncompressed_bytes:
            raise ClassicTiffStreamError("TIFF member size is outside the bound")
        with archive.open(info, "r") as handle:
            layout, position = read_classic_tiff_layout(handle)
            row_indices = np.unique(
                np.linspace(
                    0,
                    layout.height - 1,
                    min(layout.height, maximum_sample_rows),
                    dtype=np.int64,
                )
            )
            column_indices = np.unique(
                np.linspace(
                    0,
                    layout.width - 1,
                    min(layout.width, maximum_sample_columns),
                    dtype=np.int64,
                )
            )
            sampled = np.empty(
                (
                    len(row_indices),
                    len(column_indices),
                    layout.samples_per_pixel,
                ),
                dtype=np.uint16,
            )
            row_means = np.empty(
                (layout.height, layout.samples_per_pixel), dtype=np.float64
            )
            column_sums = np.zeros(
                (layout.width, layout.samples_per_pixel), dtype=np.float64
            )
            row_lookup = {
                int(source_index): output_index
                for output_index, source_index in enumerate(row_indices)
            }
            dtype = np.dtype(layout.endian + "u2")
            current_row = 0
            for strip_offset, byte_count in zip(
                layout.strip_offsets,
                layout.strip_byte_counts,
                strict=True,
            ):
                if strip_offset < position:
                    raise ClassicTiffStreamError("TIFF strips overlap or regress")
                _discard_exact(handle, strip_offset - position)
                position = strip_offset
                raw = _read_exact(handle, byte_count)
                position += byte_count
                rows = byte_count // layout.row_bytes
                pixels = np.frombuffer(raw, dtype=dtype).reshape(
                    rows, layout.width, layout.samples_per_pixel
                )
                native = pixels.astype(np.uint16, copy=False)
                row_means[current_row : current_row + rows] = np.mean(
                    native, axis=1, dtype=np.float64
                )
                column_sums += np.sum(native, axis=0, dtype=np.float64)
                for local_row in range(rows):
                    output_row = row_lookup.get(current_row + local_row)
                    if output_row is not None:
                        sampled[output_row] = native[
                            local_row, column_indices
                        ]
                current_row += rows
            if current_row != layout.height:
                raise ClassicTiffStreamError("TIFF row coverage is incomplete")
            while handle.read(1024 * 1024):
                pass
    return ClassicTiffSummary(
        layout=layout,
        sampled_row_indices=row_indices,
        sampled_column_indices=column_indices,
        sampled_u16=sampled,
        row_means=row_means,
        column_means=column_sums / float(layout.height),
    )


__all__ = [
    "ClassicTiffLayout",
    "ClassicTiffStreamError",
    "ClassicTiffSummary",
    "read_classic_tiff_layout",
    "summarize_classic_tiff_zip_member",
]
