"""Deterministic, metadata-only DNG capture receipts.

This module intentionally never decodes TIFF image samples.  It records only
standard TIFF/DNG IFD facts and binds them to the exact source bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path
from typing import Any

import tifffile

SCHEMA = "neuro_film.dng_capture_metadata_receipt.v1"
_CFA = 32803
_LINEAR_RAW = 34892
_DISALLOWED_SUBFILE_BITS = 1 | 4  # reduced-resolution image or transparency mask


class DngMetadataError(ValueError):
    """Raised when a DNG receipt cannot be built without guessing."""


_TAG_CODES = {
    "orientation": 274,
    "bits_per_sample": 258,
    "photometric_interpretation": 262,
    "image_width": 256,
    "image_length": 257,
    "dng_version": 50706,
    "dng_backward_version": 50707,
    "unique_camera_model": 50708,
    "cfa_repeat_pattern_dim": 33421,
    "cfa_pattern": 33422,
    "cfa_plane_color": 50710,
    "cfa_layout": 50711,
    "black_level_repeat_dim": 50713,
    "black_level": 50714,
    "white_level": 50717,
    "default_scale": 50718,
    "default_crop_origin": 50719,
    "default_crop_size": 50720,
    "color_matrix_1": 50721,
    "color_matrix_2": 50722,
    "as_shot_neutral": 50728,
    "active_area": 50829,
    "noise_profile": 51041,
}

_REQUIRED = {
    "orientation",
    "bits_per_sample",
    "photometric_interpretation",
    "image_width",
    "image_length",
    "dng_version",
    "unique_camera_model",
    "black_level",
    "white_level",
    "as_shot_neutral",
}

_RATIONAL_TYPES = {5, 10}
_FLOAT_TYPES = {11, 12}


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a receipt using the project's stable compact JSON convention."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _as_sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _integer(value: Any) -> int:
    if isinstance(value, IntEnum):
        return int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DngMetadataError(f"expected TIFF integer, got {type(value).__name__}")
    return int(value)


def _canonical_tag(tag: tifffile.TiffTag) -> dict[str, Any]:
    dtype = int(tag.dtype)
    value = tag.value
    if dtype in _RATIONAL_TYPES:
        flat = _as_sequence(value)
        if len(flat) != int(tag.count) * 2:
            raise DngMetadataError(f"invalid rational count for tag {tag.code}")
        pairs = []
        for offset in range(0, len(flat), 2):
            numerator = _integer(flat[offset])
            denominator = _integer(flat[offset + 1])
            if denominator == 0:
                raise DngMetadataError(f"zero denominator in tag {tag.code}")
            decimal = numerator / denominator
            if not math.isfinite(decimal):
                raise DngMetadataError(f"non-finite rational in tag {tag.code}")
            pairs.append(
                {
                    "decimal": decimal,
                    "denominator": denominator,
                    "numerator": numerator,
                }
            )
        normalized: Any = pairs
    elif dtype in _FLOAT_TYPES:
        values = [float(item) for item in _as_sequence(value)]
        if not all(math.isfinite(item) for item in values):
            raise DngMetadataError(f"non-finite floating value in tag {tag.code}")
        normalized = values
    elif isinstance(value, bytes):
        normalized = list(value)
    elif isinstance(value, str):
        normalized = value
    else:
        normalized = [_integer(item) for item in _as_sequence(value)]
    return {
        "count": int(tag.count),
        "tiff_type": tag.dtype.name,
        "value": normalized,
    }


def _walk_pages(
    pages: Iterable[tifffile.TiffPage], prefix: str = ""
) -> list[tuple[str, tifffile.TiffPage]]:
    result: list[tuple[str, tifffile.TiffPage]] = []
    for index, page in enumerate(pages):
        path = f"{prefix}{index}"
        result.append((path, page))
        if page.pages is not None:
            result.extend(_walk_pages(page.pages, f"{path}."))
    return result


def _page_integer(page: tifffile.TiffPage, code: int) -> int:
    tag = page.tags.get(code)
    if tag is None:
        raise DngMetadataError(f"raw IFD is missing TIFF tag {code}")
    record = _canonical_tag(tag)
    values = record["value"]
    if not isinstance(values, list) or len(values) != 1 or not isinstance(values[0], int):
        raise DngMetadataError(f"TIFF tag {code} is not a scalar integer")
    return values[0]


def _select_raw_page(
    pages: list[tuple[str, tifffile.TiffPage]],
) -> tuple[str, tifffile.TiffPage]:
    eligible: list[tuple[str, tifffile.TiffPage]] = []
    for path, page in pages:
        photometric = int(page.photometric)
        subfile_type = int(page.subfiletype or 0)
        if (
            photometric in {_CFA, _LINEAR_RAW}
            and not subfile_type & _DISALLOWED_SUBFILE_BITS
            and int(page.imagewidth) > 0
            and int(page.imagelength) > 0
        ):
            eligible.append((path, page))
    if len(eligible) != 1:
        raise DngMetadataError(f"expected exactly one raw IFD, found {len(eligible)}")
    return eligible[0]


def _resolved_tag(
    pages: list[tuple[str, tifffile.TiffPage]],
    raw_path: str,
    raw_page: tifffile.TiffPage,
    code: int,
) -> dict[str, Any] | None:
    raw_tag = raw_page.tags.get(code)
    if raw_tag is not None:
        record = _canonical_tag(raw_tag)
        record["ifd_paths"] = [raw_path]
        return record

    matches: list[tuple[str, dict[str, Any]]] = []
    for path, page in pages:
        tag = page.tags.get(code)
        if tag is not None:
            matches.append((path, _canonical_tag(tag)))
    if not matches:
        return None
    first = matches[0][1]
    if any(record != first for _, record in matches[1:]):
        raise DngMetadataError(f"ambiguous values for TIFF/DNG tag {code}")
    result = dict(first)
    result["ifd_paths"] = [path for path, _ in matches]
    return result


def _page_summary(path: str, page: tifffile.TiffPage) -> dict[str, Any]:
    return {
        "height": int(page.imagelength),
        "ifd_path": path,
        "photometric_interpretation": int(page.photometric),
        "subfile_type": int(page.subfiletype or 0),
        "width": int(page.imagewidth),
    }


def build_dng_capture_metadata_receipt(
    path: str | Path,
    *,
    source_id: str,
    logical_path: str,
) -> dict[str, Any]:
    """Build a source-bound DNG receipt without decoding raster samples."""

    source = Path(path)
    if source.suffix.lower() != ".dng":
        raise DngMetadataError("capture-metadata receipts accept only .dng files")
    if not source.is_file():
        raise DngMetadataError("DNG source does not exist or is not a file")
    if not source_id or not logical_path:
        raise DngMetadataError("source_id and logical_path must be non-empty")

    try:
        with tifffile.TiffFile(source) as document:
            pages = _walk_pages(document.pages)
            raw_path, raw_page = _select_raw_page(pages)
            facts: dict[str, Any] = {}
            for name, code in _TAG_CODES.items():
                record = _resolved_tag(pages, raw_path, raw_page, code)
                if record is not None:
                    facts[name] = record
            missing = sorted(_REQUIRED - facts.keys())
            if missing:
                raise DngMetadataError(f"missing required DNG facts: {', '.join(missing)}")

            # Width/height/photometric must describe the selected raw IFD, not a preview.
            if _page_integer(raw_page, 256) != int(raw_page.imagewidth):
                raise DngMetadataError("raw width tag disagrees with raw IFD geometry")
            if _page_integer(raw_page, 257) != int(raw_page.imagelength):
                raise DngMetadataError("raw height tag disagrees with raw IFD geometry")
            photometric = int(raw_page.photometric)
            if photometric == _CFA:
                cfa_required = {
                    "cfa_repeat_pattern_dim",
                    "cfa_pattern",
                    "cfa_plane_color",
                    "cfa_layout",
                }
                cfa_missing = sorted(cfa_required - facts.keys())
                if cfa_missing:
                    raise DngMetadataError(f"missing required CFA facts: {', '.join(cfa_missing)}")
                raw_kind = "cfa"
            elif photometric == _LINEAR_RAW:
                if any(name.startswith("cfa_") for name in facts):
                    raise DngMetadataError("LinearRaw IFD unexpectedly exposes CFA facts")
                raw_kind = "linear_raw"
            else:  # guarded by selection, retained as an explicit invariant
                raise DngMetadataError("selected IFD is not CFA or LinearRaw")

            body = {
                "facts": facts,
                "ifds": [_page_summary(ifd_path, page) for ifd_path, page in pages],
                "logical_path": logical_path.replace("\\", "/"),
                "noise_profile_present": "noise_profile" in facts,
                "raster_decode_calls": 0,
                "raw_ifd_path": raw_path,
                "raw_kind": raw_kind,
                "schema": SCHEMA,
                "source_bytes": source.stat().st_size,
                "source_id": source_id,
                "source_sha256": _sha256_file(source),
            }
    except DngMetadataError:
        raise
    except (OSError, TypeError, ValueError, tifffile.TiffFileError) as exc:
        raise DngMetadataError(f"invalid or unsupported DNG metadata: {exc}") from exc

    receipt = dict(body)
    receipt["receipt_body_sha256"] = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    return receipt
