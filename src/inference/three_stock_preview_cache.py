"""Hash-validated warm lookup for already-rendered three-stock previews."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .render_contract import atomic_write_json, sha256_file

CACHE_INDEX_NAME = "preview-cache.json"
_PREVIEW_SCHEMA = "neuro-film.three-stock-direct-preview.v1"
_CACHE_SCHEMA = "neuro-film.three-stock-preview-cache.v1"
_STYLES = ("velvia_50", "portra_400", "ektar_100")


class ThreeStockPreviewCacheError(ValueError):
    """Raised when a preview cache is missing, drifted or malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ThreeStockPreviewCacheError(f"invalid cache JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise ThreeStockPreviewCacheError(f"cache JSON must be an object: {path.name}")
    return value


def _preview_rows(preview: dict[str, Any]) -> list[dict[str, str]]:
    if preview.get("schema_version") != _PREVIEW_SCHEMA:
        raise ThreeStockPreviewCacheError("preview manifest schema drift")
    rows = preview.get("rows")
    if not isinstance(rows, list) or len(rows) != len(_STYLES):
        raise ThreeStockPreviewCacheError("preview row inventory drift")
    normalized: list[dict[str, str]] = []
    for expected_style, row in zip(_STYLES, rows, strict=True):
        if not isinstance(row, dict) or row.get("style_id") != expected_style:
            raise ThreeStockPreviewCacheError("preview style order drift")
        filename = f"{expected_style}.preview.png"
        output_sha256 = row.get("output_sha256")
        if not isinstance(output_sha256, str) or len(output_sha256) != 64:
            raise ThreeStockPreviewCacheError("preview output identity drift")
        normalized.append(
            {
                "style_id": expected_style,
                "filename": filename,
                "output_sha256": output_sha256,
            }
        )
    return normalized


def publish_three_stock_preview_cache_index(
    preview_directory: Path,
    *,
    input_path: Path,
    profile_path: Path,
    parent_contract_sha256: str,
) -> dict[str, Any]:
    """Publish one portable cache index after complete-file validation."""

    preview_directory = Path(preview_directory)
    input_path = Path(input_path)
    profile_path = Path(profile_path)
    index_path = preview_directory / CACHE_INDEX_NAME
    if index_path.exists():
        raise ThreeStockPreviewCacheError("preview cache index already exists")
    preview = _load_json(preview_directory / "preview.json")
    rows = _preview_rows(preview)
    input_sha256 = sha256_file(input_path)
    profile_sha256 = sha256_file(profile_path)
    if preview.get("input_sha256") != input_sha256:
        raise ThreeStockPreviewCacheError("preview input identity drift")
    for row in rows:
        if sha256_file(preview_directory / row["filename"]) != row["output_sha256"]:
            raise ThreeStockPreviewCacheError("preview output identity drift")
    index = {
        "schema_version": _CACHE_SCHEMA,
        "parent_contract_sha256": parent_contract_sha256,
        "input_sha256": input_sha256,
        "profile_sha256": profile_sha256,
        "preview_width": preview["preview_width"],
        "preview_height": preview["preview_height"],
        "preview_pixels": preview["preview_pixels"],
        "max_preview_pixels": preview["max_preview_pixels"],
        "look_amount": preview["look_amount"],
        "rows": rows,
        "claim_ceiling": (
            "Warm lookup of existing non-calibrated Look Approximation previews; "
            "not render authorization or final export."
        ),
    }
    atomic_write_json(index_path, index)
    return index


def inspect_three_stock_preview_cache(
    preview_directory: Path, *, input_path: Path, profile_path: Path
) -> dict[str, Any]:
    """Return a cache index only after hashing its input, profile and outputs."""

    preview_directory = Path(preview_directory)
    index = _load_json(preview_directory / CACHE_INDEX_NAME)
    expected_keys = {
        "schema_version",
        "parent_contract_sha256",
        "input_sha256",
        "profile_sha256",
        "preview_width",
        "preview_height",
        "preview_pixels",
        "max_preview_pixels",
        "look_amount",
        "rows",
        "claim_ceiling",
    }
    if set(index) != expected_keys or index.get("schema_version") != _CACHE_SCHEMA:
        raise ThreeStockPreviewCacheError("preview cache index field drift")
    if sha256_file(Path(input_path)) != index["input_sha256"]:
        raise ThreeStockPreviewCacheError("preview cache input drift")
    if sha256_file(Path(profile_path)) != index["profile_sha256"]:
        raise ThreeStockPreviewCacheError("preview cache profile drift")
    rows = index.get("rows")
    if not isinstance(rows, list) or len(rows) != len(_STYLES):
        raise ThreeStockPreviewCacheError("preview cache row inventory drift")
    for expected_style, row in zip(_STYLES, rows, strict=True):
        expected_filename = f"{expected_style}.preview.png"
        if (
            not isinstance(row, dict)
            or set(row) != {"style_id", "filename", "output_sha256"}
            or row.get("style_id") != expected_style
            or row.get("filename") != expected_filename
            or not isinstance(row.get("output_sha256"), str)
            or len(row["output_sha256"]) != 64
        ):
            raise ThreeStockPreviewCacheError("preview cache row drift")
        output = preview_directory / expected_filename
        if sha256_file(output) != row["output_sha256"]:
            raise ThreeStockPreviewCacheError("preview cache output drift")
    return index


__all__ = [
    "CACHE_INDEX_NAME",
    "ThreeStockPreviewCacheError",
    "inspect_three_stock_preview_cache",
    "publish_three_stock_preview_cache_index",
]
