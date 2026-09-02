"""Bounded metadata-first admission for a strict ordinary SDR AVIF subset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

_MAX_TOP_LEVEL_BOXES = 64
_MAX_CHILD_BOXES = 4096
_MAX_META_BYTES = 8 * 1024 * 1024
_MAX_ITEMS = 128
_MAX_PROPERTIES = 256
_REJECTED_BRANDS = {b"avis", b"msf1", b"tmap"}
_IMAGE_ITEM_TYPES = {b"av01", b"grid", b"iden", b"iovl", b"tmap"}
_REJECTED_REFERENCE_TYPES = {b"dimg", b"auxl", b"prem"}
_ALLOWED_METADATA_ITEM_TYPES = {b"Exif", b"mime", b"uri "}
_REQUIRED_PRIMARY_PROPERTIES = {b"ispe", b"pixi", b"av1C", b"colr"}


class StrictSdrAvifError(ValueError):
    """Raised when AVIF cannot prove the frozen ordinary-SDR subset."""


@dataclass(frozen=True)
class StrictSdrAvifInfo:
    """Metadata proven before any AV1 sample is decoded."""

    width: int
    height: int
    bits_per_channel: tuple[int, ...]
    nclx: tuple[int, int, int, bool]
    primary_item_id: int
    item_count: int
    compatible_brands: tuple[str, ...]


@dataclass(frozen=True)
class _Box:
    kind: bytes
    payload_start: int
    end: int


def _u16(data: bytes, offset: int, label: str) -> int:
    if offset + 2 > len(data):
        raise StrictSdrAvifError(f"truncated {label}")
    return int.from_bytes(data[offset : offset + 2], "big")


def _u32(data: bytes, offset: int, label: str) -> int:
    if offset + 4 > len(data):
        raise StrictSdrAvifError(f"truncated {label}")
    return int.from_bytes(data[offset : offset + 4], "big")


def _boxes(data: bytes, start: int, end: int, *, label: str) -> list[_Box]:
    boxes: list[_Box] = []
    cursor = start
    while cursor < end:
        if len(boxes) >= _MAX_CHILD_BOXES:
            raise StrictSdrAvifError(f"{label} exceeds box-count budget")
        if cursor + 8 > end:
            raise StrictSdrAvifError(f"truncated {label} box header")
        size = _u32(data, cursor, f"{label} box size")
        kind = data[cursor + 4 : cursor + 8]
        header = 8
        if size == 1:
            if cursor + 16 > end:
                raise StrictSdrAvifError(f"truncated {label} large box header")
            size = int.from_bytes(data[cursor + 8 : cursor + 16], "big")
            header = 16
        if size == 0:
            size = end - cursor
        if size < header or cursor + size > end:
            raise StrictSdrAvifError(f"invalid {label} box size")
        boxes.append(_Box(kind, cursor + header, cursor + size))
        cursor += size
    if cursor != end:
        raise StrictSdrAvifError(f"invalid {label} box layout")
    return boxes


def _read_top_level(handle: BinaryIO, file_size: int) -> dict[bytes, list[bytes]]:
    retained: dict[bytes, list[bytes]] = {}
    cursor = 0
    count = 0
    while cursor < file_size:
        count += 1
        if count > _MAX_TOP_LEVEL_BOXES:
            raise StrictSdrAvifError("AVIF exceeds top-level box-count budget")
        handle.seek(cursor)
        header = handle.read(16)
        if len(header) < 8:
            raise StrictSdrAvifError("truncated AVIF top-level box")
        size = int.from_bytes(header[:4], "big")
        kind = header[4:8]
        header_size = 8
        if size == 1:
            if len(header) < 16:
                raise StrictSdrAvifError("truncated AVIF large box")
            size = int.from_bytes(header[8:16], "big")
            header_size = 16
        elif size == 0:
            size = file_size - cursor
        if size < header_size or cursor + size > file_size:
            raise StrictSdrAvifError("invalid AVIF top-level box size")
        if kind in {b"ftyp", b"meta"}:
            payload_size = size - header_size
            if kind == b"meta" and payload_size > _MAX_META_BYTES:
                raise StrictSdrAvifError("AVIF meta payload exceeds inspection budget")
            if payload_size > _MAX_META_BYTES:
                raise StrictSdrAvifError("AVIF header payload exceeds inspection budget")
            handle.seek(cursor + header_size)
            payload = handle.read(payload_size)
            if len(payload) != payload_size:
                raise StrictSdrAvifError("truncated AVIF header payload")
            retained.setdefault(kind, []).append(payload)
        elif kind not in {b"mdat", b"free", b"skip"}:
            qualifier = "sequence " if kind == b"moov" else ""
            raise StrictSdrAvifError(
                f"unsupported AVIF {qualifier}top-level box {kind.decode('latin-1')}"
            )
        else:
            retained.setdefault(kind, []).append(b"")
        cursor += size
    return retained


def looks_like_avif(path: Path) -> bool:
    """Return whether the file begins with an AVIF-family file-type box."""

    try:
        with Path(path).open("rb") as handle:
            header = handle.read(16)
    except OSError:
        return False
    if len(header) < 16 or header[4:8] != b"ftyp":
        return False
    size = int.from_bytes(header[:4], "big")
    if size < 16:
        return False
    return header[8:12] in {b"avif", b"avis"}


def _parse_ftyp(payload: bytes) -> tuple[str, ...]:
    if len(payload) < 8 or (len(payload) - 8) % 4:
        raise StrictSdrAvifError("invalid AVIF file-type box")
    major = payload[:4]
    brands = [payload[index : index + 4] for index in range(8, len(payload), 4)]
    if major != b"avif" or b"avif" not in brands:
        raise StrictSdrAvifError("AVIF still-image brand is required")
    rejected = ({major, *brands} & _REJECTED_BRANDS)
    if rejected:
        names = ",".join(sorted(value.decode("latin-1") for value in rejected))
        raise StrictSdrAvifError(f"AVIF sequence/tone-map brand is forbidden: {names}")
    return tuple(value.decode("latin-1") for value in brands)


def _parse_pitm(payload: bytes) -> int:
    if len(payload) < 6:
        raise StrictSdrAvifError("truncated AVIF primary-item box")
    version = payload[0]
    if version == 0:
        return _u16(payload, 4, "AVIF primary item")
    if version == 1:
        return _u32(payload, 4, "AVIF primary item")
    raise StrictSdrAvifError("unsupported AVIF primary-item version")


def _parse_iinf(payload: bytes) -> dict[int, bytes]:
    if len(payload) < 6:
        raise StrictSdrAvifError("truncated AVIF item-information box")
    version = payload[0]
    if version == 0:
        expected = _u16(payload, 4, "AVIF item count")
        start = 6
    else:
        expected = _u32(payload, 4, "AVIF item count")
        start = 8
    if expected > _MAX_ITEMS:
        raise StrictSdrAvifError("AVIF item count exceeds inspection budget")
    entries = _boxes(payload, start, len(payload), label="AVIF iinf")
    if len(entries) != expected or any(entry.kind != b"infe" for entry in entries):
        raise StrictSdrAvifError("AVIF item-information count mismatch")
    items: dict[int, bytes] = {}
    for entry in entries:
        data = payload[entry.payload_start : entry.end]
        if len(data) < 12:
            raise StrictSdrAvifError("truncated AVIF item entry")
        infe_version = data[0]
        if infe_version == 2:
            item_id = _u16(data, 4, "AVIF item id")
            item_type = data[8:12]
        elif infe_version == 3:
            item_id = _u32(data, 4, "AVIF item id")
            item_type = data[10:14]
        else:
            raise StrictSdrAvifError("unsupported AVIF item-entry version")
        if item_id in items:
            raise StrictSdrAvifError("duplicate AVIF item id")
        items[item_id] = item_type
    return items


def _parse_ipma(payload: bytes) -> dict[int, list[tuple[int, bool]]]:
    if len(payload) < 8:
        raise StrictSdrAvifError("truncated AVIF property-association box")
    version = payload[0]
    flags = int.from_bytes(payload[1:4], "big")
    entry_count = _u32(payload, 4, "AVIF property association count")
    if entry_count > _MAX_ITEMS:
        raise StrictSdrAvifError("AVIF property associations exceed budget")
    cursor = 8
    associations: dict[int, list[tuple[int, bool]]] = {}
    for _ in range(entry_count):
        if version < 1:
            item_id = _u16(payload, cursor, "AVIF property item id")
            cursor += 2
        else:
            item_id = _u32(payload, cursor, "AVIF property item id")
            cursor += 4
        if cursor >= len(payload):
            raise StrictSdrAvifError("truncated AVIF property association")
        count = payload[cursor]
        cursor += 1
        values: list[tuple[int, bool]] = []
        for _ in range(count):
            if flags & 1:
                raw = _u16(payload, cursor, "AVIF property index")
                cursor += 2
                values.append((raw & 0x7FFF, bool(raw & 0x8000)))
            else:
                if cursor >= len(payload):
                    raise StrictSdrAvifError("truncated AVIF property index")
                raw = payload[cursor]
                cursor += 1
                values.append((raw & 0x7F, bool(raw & 0x80)))
        if item_id in associations:
            raise StrictSdrAvifError("duplicate AVIF property association entry")
        associations[item_id] = values
    if cursor != len(payload):
        raise StrictSdrAvifError("trailing AVIF property association data")
    return associations


def _parse_iprp(payload: bytes) -> tuple[list[tuple[bytes, bytes]], dict[int, list[tuple[int, bool]]]]:
    children = _boxes(payload, 0, len(payload), label="AVIF iprp")
    ipco = [child for child in children if child.kind == b"ipco"]
    ipma = [child for child in children if child.kind == b"ipma"]
    if len(ipco) != 1 or len(ipma) != 1 or len(children) != 2:
        raise StrictSdrAvifError("AVIF requires one ipco and one ipma box")
    property_boxes = _boxes(
        payload, ipco[0].payload_start, ipco[0].end, label="AVIF ipco"
    )
    if len(property_boxes) > _MAX_PROPERTIES:
        raise StrictSdrAvifError("AVIF property count exceeds inspection budget")
    properties = [
        (box.kind, payload[box.payload_start : box.end]) for box in property_boxes
    ]
    associations = _parse_ipma(payload[ipma[0].payload_start : ipma[0].end])
    return properties, associations


def _parse_iref(payload: bytes) -> list[tuple[bytes, int, tuple[int, ...]]]:
    if len(payload) < 4:
        raise StrictSdrAvifError("truncated AVIF item-reference box")
    version = payload[0]
    if version not in {0, 1}:
        raise StrictSdrAvifError("unsupported AVIF item-reference version")
    references: list[tuple[bytes, int, tuple[int, ...]]] = []
    for box in _boxes(payload, 4, len(payload), label="AVIF iref"):
        data = payload[box.payload_start : box.end]
        cursor = 0
        if version == 0:
            source = _u16(data, cursor, "AVIF reference source")
            cursor += 2
        else:
            source = _u32(data, cursor, "AVIF reference source")
            cursor += 4
        count = _u16(data, cursor, "AVIF reference count")
        cursor += 2
        targets: list[int] = []
        for _ in range(count):
            if version == 0:
                targets.append(_u16(data, cursor, "AVIF reference target"))
                cursor += 2
            else:
                targets.append(_u32(data, cursor, "AVIF reference target"))
                cursor += 4
        if cursor != len(data):
            raise StrictSdrAvifError("trailing AVIF item-reference data")
        references.append((box.kind, source, tuple(targets)))
    return references


def _primary_properties(
    properties: list[tuple[bytes, bytes]],
    associations: dict[int, list[tuple[int, bool]]],
    primary: int,
) -> dict[bytes, bytes]:
    selected: dict[bytes, bytes] = {}
    for index, essential in associations.get(primary, []):
        if index == 0:
            continue
        if index > len(properties):
            raise StrictSdrAvifError("AVIF property index is out of range")
        kind, payload = properties[index - 1]
        if kind not in _REQUIRED_PRIMARY_PROPERTIES:
            qualifier = "essential " if essential else ""
            raise StrictSdrAvifError(
                f"unsupported {qualifier}primary AVIF property {kind.decode('latin-1')}"
            )
        if kind in selected:
            raise StrictSdrAvifError("duplicate primary AVIF property")
        selected[kind] = payload
    if set(selected) != _REQUIRED_PRIMARY_PROPERTIES:
        missing = _REQUIRED_PRIMARY_PROPERTIES - set(selected)
        names = ",".join(sorted(value.decode("latin-1") for value in missing))
        raise StrictSdrAvifError(f"missing primary AVIF properties: {names}")
    return selected


def _parse_primary_values(selected: dict[bytes, bytes]) -> tuple[int, int, tuple[int, ...], tuple[int, int, int, bool]]:
    ispe = selected[b"ispe"]
    if len(ispe) != 12 or ispe[:4] != b"\x00\x00\x00\x00":
        raise StrictSdrAvifError("invalid AVIF spatial-extents property")
    width = _u32(ispe, 4, "AVIF width")
    height = _u32(ispe, 8, "AVIF height")
    if width < 1 or height < 1:
        raise StrictSdrAvifError("invalid AVIF dimensions")

    pixi = selected[b"pixi"]
    if len(pixi) < 5 or pixi[:4] != b"\x00\x00\x00\x00":
        raise StrictSdrAvifError("invalid AVIF pixel-information property")
    count = pixi[4]
    if len(pixi) != 5 + count:
        raise StrictSdrAvifError("invalid AVIF pixel-information length")
    bits = tuple(pixi[5:])
    if bits != (8, 8, 8):
        raise StrictSdrAvifError("strict SDR AVIF requires three 8-bit channels")

    colr = selected[b"colr"]
    if len(colr) != 11 or colr[:4] != b"nclx":
        raise StrictSdrAvifError("strict SDR AVIF requires one NCLX property")
    nclx = (
        _u16(colr, 4, "AVIF colour primaries"),
        _u16(colr, 6, "AVIF transfer characteristics"),
        _u16(colr, 8, "AVIF matrix coefficients"),
        bool(colr[10] & 0x80),
    )
    if nclx != (1, 13, 6, True):
        raise StrictSdrAvifError(
            "strict SDR AVIF requires NCLX primaries=1 transfer=13 matrix=6 full-range"
        )
    if not selected[b"av1C"]:
        raise StrictSdrAvifError("empty AVIF codec configuration")
    return width, height, bits, nclx


def inspect_strict_sdr_avif(path: Path) -> StrictSdrAvifInfo:
    """Inspect a narrow SDR AVIF subset without decoding image samples."""

    path = Path(path)
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            top = _read_top_level(handle, file_size)
    except OSError as exc:
        raise StrictSdrAvifError("unable to read AVIF container") from exc
    if len(top.get(b"ftyp", [])) != 1 or len(top.get(b"meta", [])) != 1:
        raise StrictSdrAvifError("AVIF requires exactly one ftyp and one meta box")
    if len(top.get(b"mdat", [])) != 1:
        raise StrictSdrAvifError("AVIF requires exactly one media-data box")
    brands = _parse_ftyp(top[b"ftyp"][0])

    meta = top[b"meta"][0]
    if len(meta) < 4 or meta[:4] != b"\x00\x00\x00\x00":
        raise StrictSdrAvifError("unsupported AVIF meta version")
    children = _boxes(meta, 4, len(meta), label="AVIF meta")
    grouped: dict[bytes, list[_Box]] = {}
    for child in children:
        grouped.setdefault(child.kind, []).append(child)
    required = {b"hdlr", b"pitm", b"iloc", b"iinf", b"iprp"}
    if any(len(grouped.get(kind, [])) != 1 for kind in required):
        raise StrictSdrAvifError("AVIF meta required-box cardinality mismatch")
    allowed = required | {b"iref", b"idat"}
    unknown = set(grouped) - allowed
    if unknown or b"grpl" in grouped:
        names = ",".join(sorted(value.decode("latin-1") for value in unknown))
        raise StrictSdrAvifError(f"unsupported AVIF meta box: {names or 'grpl'}")
    if len(grouped.get(b"iref", [])) > 1 or len(grouped.get(b"idat", [])) > 1:
        raise StrictSdrAvifError("duplicate optional AVIF meta box")

    def payload(kind: bytes) -> bytes:
        box = grouped[kind][0]
        return meta[box.payload_start : box.end]

    primary = _parse_pitm(payload(b"pitm"))
    items = _parse_iinf(payload(b"iinf"))
    if items.get(primary) != b"av01":
        raise StrictSdrAvifError("AVIF primary item must be av01")
    image_items = {item_id: kind for item_id, kind in items.items() if kind in _IMAGE_ITEM_TYPES}
    if image_items != {primary: b"av01"}:
        raise StrictSdrAvifError(
            "AVIF alpha/auxiliary or derived image items are forbidden"
        )
    for item_id, kind in items.items():
        if item_id != primary and kind not in _ALLOWED_METADATA_ITEM_TYPES:
            raise StrictSdrAvifError(
                f"unsupported AVIF non-image item {kind.decode('latin-1')}"
            )

    properties, associations = _parse_iprp(payload(b"iprp"))
    selected = _primary_properties(properties, associations, primary)
    width, height, bits, nclx = _parse_primary_values(selected)

    if b"iref" in grouped:
        for kind, source, targets in _parse_iref(payload(b"iref")):
            if kind in _REJECTED_REFERENCE_TYPES or kind != b"cdsc":
                raise StrictSdrAvifError(
                    f"unsupported AVIF item reference {kind.decode('latin-1')}"
                )
            if source not in items or items[source] not in _ALLOWED_METADATA_ITEM_TYPES:
                raise StrictSdrAvifError("AVIF descriptive reference source is not metadata")
            if targets != (primary,):
                raise StrictSdrAvifError("AVIF metadata must describe only the primary item")

    return StrictSdrAvifInfo(
        width=width,
        height=height,
        bits_per_channel=bits,
        nclx=nclx,
        primary_item_id=primary,
        item_count=len(items),
        compatible_brands=brands,
    )
