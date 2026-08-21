"""Versioned DNG receipts with explicit standard-default provenance.

The historical v1 receipt intentionally requires explicit CFA tags.  This
module leaves that implementation untouched and applies only the two defaults
defined by DNG 1.7.1.0.  Raster samples are never decoded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import tifffile

from src.preprocess.dng_metadata import (
    _CFA,
    _LINEAR_RAW,
    _REQUIRED,
    _TAG_CODES,
    DngMetadataError,
    _page_integer,
    _page_summary,
    _resolved_tag,
    _select_raw_page,
    _sha256_file,
    _walk_pages,
    canonical_json_bytes,
)

SCHEMA = "neuro_film.dng_capture_metadata_receipt.v2"
STANDARD_NAME = "Adobe Digital Negative (DNG) Specification 1.7.1.0"
STANDARD_SHA256 = "abdecfd8e104e8b86cc054d3a40b677bb2ebe87131bfde080a1df3ee16eb2f8d"
STANDARD_LOGICAL_PATH = "data/external/adobe_dng_spec_1_7_1_0/DNG_Spec_1_7_1_0.pdf"
_EXPLICIT = "explicit_ifd_tag"
_DEFAULT = "dng_standard_default_1_7_1_0"
_STANDARD_DEFAULTS: dict[str, dict[str, Any]] = {
    "cfa_plane_color": {
        "count": 3,
        "ifd_paths": [],
        "tiff_type": "BYTE",
        "value": [0, 1, 2],
        "value_origin": _DEFAULT,
    },
    "cfa_layout": {
        "count": 1,
        "ifd_paths": [],
        "tiff_type": "SHORT",
        "value": [1],
        "value_origin": _DEFAULT,
    },
}


def _with_explicit_origin(record: dict[str, Any]) -> dict[str, Any]:
    result = dict(record)
    result["value_origin"] = _EXPLICIT
    return result


def _standard_default(name: str) -> dict[str, Any]:
    default = _STANDARD_DEFAULTS[name]
    return {
        **default,
        "ifd_paths": list(default["ifd_paths"]),
        "value": list(default["value"]),
    }


def _require_rgb_default_cfa(facts: dict[str, Any]) -> None:
    pattern = facts["cfa_pattern"]["value"]
    if (
        not isinstance(pattern, list)
        or not pattern
        or any(
            not isinstance(value, int) or value not in {0, 1, 2} for value in pattern
        )
    ):
        raise DngMetadataError(
            "omitted CFAPlaneColor default is valid only for an RGB CFA pattern"
        )


def build_dng_capture_metadata_receipt_v2(
    path: str | Path,
    *,
    source_id: str,
    logical_path: str,
) -> dict[str, Any]:
    """Build a v2 receipt using only the two DNG 1.7.1.0 CFA defaults."""

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
                    facts[name] = _with_explicit_origin(record)

            missing = sorted(_REQUIRED - facts.keys())
            if missing:
                raise DngMetadataError(
                    f"missing required DNG facts: {', '.join(missing)}"
                )

            if _page_integer(raw_page, 256) != int(raw_page.imagewidth):
                raise DngMetadataError("raw width tag disagrees with raw IFD geometry")
            if _page_integer(raw_page, 257) != int(raw_page.imagelength):
                raise DngMetadataError("raw height tag disagrees with raw IFD geometry")

            photometric = int(raw_page.photometric)
            standard_defaults_used: list[str] = []
            if photometric == _CFA:
                base_cfa_required = {"cfa_repeat_pattern_dim", "cfa_pattern"}
                cfa_missing = sorted(base_cfa_required - facts.keys())
                if cfa_missing:
                    raise DngMetadataError(
                        f"missing required CFA facts: {', '.join(cfa_missing)}"
                    )
                if "cfa_plane_color" not in facts:
                    _require_rgb_default_cfa(facts)
                    facts["cfa_plane_color"] = _standard_default("cfa_plane_color")
                    standard_defaults_used.append("cfa_plane_color")
                if "cfa_layout" not in facts:
                    facts["cfa_layout"] = _standard_default("cfa_layout")
                    standard_defaults_used.append("cfa_layout")
                raw_kind = "cfa"
            elif photometric == _LINEAR_RAW:
                if any(name.startswith("cfa_") for name in facts):
                    raise DngMetadataError(
                        "LinearRaw IFD unexpectedly exposes CFA facts"
                    )
                raw_kind = "linear_raw"
            else:
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
                "standard_defaults_used": standard_defaults_used,
                "standard_specification": {
                    "logical_path": STANDARD_LOGICAL_PATH,
                    "name": STANDARD_NAME,
                    "sha256": STANDARD_SHA256,
                },
            }
    except DngMetadataError:
        raise
    except (OSError, TypeError, ValueError, tifffile.TiffFileError) as exc:
        raise DngMetadataError(f"invalid or unsupported DNG metadata: {exc}") from exc

    receipt = dict(body)
    receipt["receipt_body_sha256"] = hashlib.sha256(
        canonical_json_bytes(body)
    ).hexdigest()
    return receipt


__all__ = [
    "SCHEMA",
    "STANDARD_LOGICAL_PATH",
    "STANDARD_NAME",
    "STANDARD_SHA256",
    "build_dng_capture_metadata_receipt_v2",
]
