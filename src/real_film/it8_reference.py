"""Deterministic parsing helpers for IT8/CGATS transmission references."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable
from zipfile import ZipFile


_HEADER_VALUE = re.compile(r'^([A-Z][A-Z0-9_]*)\s+(?:"([^"]*)"|([^#\s]+))')


def find_unique_suffix_member(archive: ZipFile, suffix: str) -> str:
    names = sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir() and info.filename.lower().endswith(suffix)
    )
    if len(names) != 1:
        raise ValueError(f"expected one {suffix} member, found {names}")
    return names[0]


def find_charge_table_member(archive: ZipFile, archive_stem: str) -> str:
    """Find one primary charge table across historical .txt/.it8 naming."""

    names = sorted(
        info.filename
        for info in archive.infolist()
        if not info.is_dir()
        and Path(info.filename).stem.casefold() == archive_stem.casefold()
        and Path(info.filename).suffix.casefold() in {".it8", ".txt"}
        and "extras"
        not in {part.casefold() for part in Path(info.filename).parts[:-1]}
    )
    if len(names) != 1:
        raise ValueError(
            f"expected one exact charge IT8/.txt member, found {names}"
        )
    return names[0]


@dataclass(frozen=True)
class IT8Reference:
    """The fields needed by the bounded SF2.9R measurement audit."""

    header: dict[str, str]
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]

    @property
    def sample_ids(self) -> tuple[str, ...]:
        return tuple(row["SAMPLE_ID"] for row in self.rows)


@dataclass(frozen=True)
class SpectralReference:
    """Parsed CGATS spectral rows with explicit wavelength/value pairs."""

    header: dict[str, str]
    sample_ids: tuple[str, ...]
    lab: tuple[tuple[float, float, float], ...]
    wavelengths_nm: tuple[int, ...]
    spectra_pct: tuple[tuple[float, ...], ...]


def _normalise_lines(payload: bytes) -> list[str]:
    text = payload.decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    return text.split("\n")


def _parse_header(lines: Iterable[str]) -> dict[str, str]:
    header: dict[str, str] = {}
    for line in lines:
        match = _HEADER_VALUE.match(line.strip())
        if match:
            header[match.group(1)] = match.group(2) or match.group(3)
    return header


def _section(lines: list[str], begin: str, end: str) -> list[str]:
    try:
        start = lines.index(begin) + 1
        stop = lines.index(end, start)
    except ValueError as error:
        raise ValueError(f"missing IT8 section {begin}/{end}") from error
    return lines[start:stop]


def _table_header(lines: list[str]) -> dict[str, str]:
    format_start = lines.index("BEGIN_DATA_FORMAT")
    format_end = lines.index("END_DATA_FORMAT", format_start)
    data_start = lines.index("BEGIN_DATA", format_end)
    return _parse_header(lines[:format_start] + lines[format_end + 1 : data_start])


def parse_it8(payload: bytes) -> IT8Reference:
    """Parse the ordinary one-row-per-sample IT8 reference table."""

    lines = _normalise_lines(payload)
    header = _table_header(lines)
    fields = tuple(
        " ".join(_section(lines, "BEGIN_DATA_FORMAT", "END_DATA_FORMAT")).split()
    )
    if not fields or fields[0] != "SAMPLE_ID":
        raise ValueError("IT8 data format has no SAMPLE_ID")
    tokens = " ".join(_section(lines, "BEGIN_DATA", "END_DATA")).split()
    field_count = int(header["NUMBER_OF_FIELDS"])
    set_count = int(header["NUMBER_OF_SETS"])
    if field_count != len(fields):
        raise ValueError(f"field count mismatch: {field_count} != {len(fields)}")
    if len(tokens) != field_count * set_count:
        raise ValueError(
            f"data token count mismatch: {len(tokens)} != {field_count * set_count}"
        )
    rows = tuple(
        dict(zip(fields, tokens[offset : offset + field_count], strict=True))
        for offset in range(0, len(tokens), field_count)
    )
    sample_ids = [row["SAMPLE_ID"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate SAMPLE_ID")
    return IT8Reference(header, fields, rows)


def parse_cgats_spectral(payload: bytes) -> SpectralReference:
    """Parse the wrapped CGATS spectral table used by ColorReference."""

    lines = _normalise_lines(payload)
    header = _table_header(lines)
    fields = tuple(
        " ".join(_section(lines, "BEGIN_DATA_FORMAT", "END_DATA_FORMAT")).split()
    )
    tokens = " ".join(_section(lines, "BEGIN_DATA", "END_DATA")).split()
    field_count = int(header["NUMBER_OF_FIELDS"])
    set_count = int(header["NUMBER_OF_SETS"])
    if field_count != len(fields):
        raise ValueError(f"field count mismatch: {field_count} != {len(fields)}")
    expected_prefix = (
        "SAMPLE_ID",
        "XYZ_X",
        "XYZ_Y",
        "XYZ_Z",
        "LAB_L",
        "LAB_A",
        "LAB_B",
        "LAB_C",
        "LAB_H",
    )
    if fields[:9] != expected_prefix or any(
        pair != ("SPECTRAL_NM", "SPECTRAL_PCT")
        for pair in zip(fields[9::2], fields[10::2], strict=True)
    ):
        raise ValueError("unsupported spectral data format")
    if len(tokens) != field_count * set_count:
        raise ValueError(
            f"spectral token count mismatch: {len(tokens)} "
            f"!= {field_count * set_count}"
        )
    if field_count < 11 or (field_count - 9) % 2:
        raise ValueError(f"unsupported spectral field count: {field_count}")
    sample_ids: list[str] = []
    labs: list[tuple[float, float, float]] = []
    spectra: list[tuple[float, ...]] = []
    wavelengths: tuple[int, ...] | None = None
    for offset in range(0, len(tokens), field_count):
        row = tokens[offset : offset + field_count]
        sample_ids.append(row[0])
        labs.append((float(row[4]), float(row[5]), float(row[6])))
        row_wavelengths = tuple(int(float(value)) for value in row[9::2])
        row_spectrum = tuple(float(value) for value in row[10::2])
        if wavelengths is None:
            wavelengths = row_wavelengths
        elif wavelengths != row_wavelengths:
            raise ValueError("spectral wavelength grid changes between samples")
        spectra.append(row_spectrum)
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("duplicate spectral SAMPLE_ID")
    assert wavelengths is not None
    return SpectralReference(
        header,
        tuple(sample_ids),
        tuple(labs),
        wavelengths,
        tuple(spectra),
    )
