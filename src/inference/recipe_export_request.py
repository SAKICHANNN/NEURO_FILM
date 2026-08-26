"""Portable, hash-bound export requests for validated recipe history."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from html import escape
from pathlib import Path, PurePosixPath
from typing import Any

from .recipe_history import (
    DEFAULT_MAXIMUM_RECIPE_BYTES,
    DEFAULT_MAXIMUM_RECIPE_FILES,
    build_render_recipe_history,
)
from .recipe_history_export import export_recipe_history_entry

RECIPE_EXPORT_REQUEST_SCHEMA_ID = "kmcfm.recipe-export-request.v1"
RECIPE_EXPORT_REQUEST_SET_SCHEMA_ID = "kmcfm.recipe-export-request-set.v1"
DEFAULT_MAXIMUM_REQUEST_BYTES = 64 * 1024

_REQUEST_KEYS = {
    "schema",
    "recipe_path",
    "recipe_sha256",
    "style",
    "output_format",
    "output_bit_depth",
    "output_label",
    "evidence_grade",
}


class RecipeExportRequestError(ValueError):
    """Raised when an offline export request is invalid or cannot be used."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RecipeExportRequestError("export request contains a duplicate key")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise RecipeExportRequestError(f"export request contains non-finite JSON: {value}")


def _validated_relative_recipe_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise RecipeExportRequestError("recipe_path must be a non-empty string")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RecipeExportRequestError("recipe_path must be normalized and relative")
    if path.as_posix() != value or not value.endswith(".recipe.json"):
        raise RecipeExportRequestError("recipe_path must name a normalized recipe file")
    return value


def validate_recipe_export_request(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one portable request without reading any image."""

    if set(value) != _REQUEST_KEYS:
        raise RecipeExportRequestError("export request fields drift")
    if value.get("schema") != RECIPE_EXPORT_REQUEST_SCHEMA_ID:
        raise RecipeExportRequestError("export request schema drift")
    recipe_path = _validated_relative_recipe_path(value.get("recipe_path"))
    recipe_sha256 = value.get("recipe_sha256")
    if (
        not isinstance(recipe_sha256, str)
        or len(recipe_sha256) != 64
        or any(char not in "0123456789abcdef" for char in recipe_sha256)
    ):
        raise RecipeExportRequestError("recipe_sha256 must be lowercase SHA-256")
    for key in ("style", "output_format", "output_label", "evidence_grade"):
        if not isinstance(value.get(key), str) or not value[key]:
            raise RecipeExportRequestError(f"{key} must be a non-empty string")
    if type(value.get("output_bit_depth")) is not int or value["output_bit_depth"] < 1:
        raise RecipeExportRequestError("output_bit_depth must be a positive integer")
    return {
        "schema": RECIPE_EXPORT_REQUEST_SCHEMA_ID,
        "recipe_path": recipe_path,
        "recipe_sha256": recipe_sha256,
        "style": value["style"],
        "output_format": value["output_format"],
        "output_bit_depth": value["output_bit_depth"],
        "output_label": value["output_label"],
        "evidence_grade": value["evidence_grade"],
    }


def _read_recipe_export_request(
    path: Path,
    *,
    maximum_request_bytes: int = DEFAULT_MAXIMUM_REQUEST_BYTES,
) -> tuple[bytes, dict[str, Any]]:

    if type(maximum_request_bytes) is not int or maximum_request_bytes < 1:
        raise RecipeExportRequestError("maximum_request_bytes must be positive")
    try:
        with path.open("rb") as handle:
            payload = handle.read(maximum_request_bytes + 1)
    except OSError as exc:
        raise RecipeExportRequestError("export request is unavailable") from exc
    if len(payload) > maximum_request_bytes:
        raise RecipeExportRequestError("export request exceeds the byte limit")
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RecipeExportRequestError("export request is not strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise RecipeExportRequestError("export request must be an object")
    return payload, validate_recipe_export_request(value)


def load_recipe_export_request(
    path: Path,
    *,
    maximum_request_bytes: int = DEFAULT_MAXIMUM_REQUEST_BYTES,
) -> dict[str, Any]:
    """Load one bounded strict-JSON request."""

    return _read_recipe_export_request(
        path, maximum_request_bytes=maximum_request_bytes
    )[1]


def _request_for_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return validate_recipe_export_request(
        {
            "schema": RECIPE_EXPORT_REQUEST_SCHEMA_ID,
            "recipe_path": row["recipe_path"],
            "recipe_sha256": row["recipe_sha256"],
            "style": row["style"],
            "output_format": row["output_format"],
            "output_bit_depth": row["output_bit_depth"],
            "output_label": row["output_label"],
            "evidence_grade": row["evidence_grade"],
        }
    )


def build_recipe_export_request_set(
    history_root: Path,
    *,
    maximum_recipe_files: int = DEFAULT_MAXIMUM_RECIPE_FILES,
    maximum_recipe_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES,
) -> dict[str, Any]:
    """Build request files and an offline selection page without image reads."""

    catalog = build_render_recipe_history(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    rows: list[dict[str, Any]] = []
    files: dict[str, bytes] = {}
    for row in catalog["entries"]:
        if row["status"] != "valid":
            continue
        request = _request_for_row(row)
        payload = _canonical_json(request)
        name = f"request-{row['recipe_sha256']}.json"
        files[name] = payload
        rows.append(
            {
                "request_file": name,
                "request_sha256": _sha256(payload),
                **request,
            }
        )

    cards = "".join(
        "<article><p class=\"eyebrow\">{style}</p><h2>{format} · {depth}-bit</h2>"
        "<p>{label} · {grade}</p><a download=\"{filename}\" href=\"{filename}\">"
        "Save export request</a></article>".format(
            style=escape(str(row["style"])),
            format=escape(str(row["output_format"])),
            depth=row["output_bit_depth"],
            label=escape(str(row["output_label"])),
            grade=escape(str(row["evidence_grade"])),
            filename=escape(str(row["request_file"]), quote=True),
        )
        for row in rows
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; navigate-to 'self'; base-uri 'none'; form-action 'none'">
<title>Export requests · K-MCFM</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#101210;color:#f2f4ef;font:16px/1.5 system-ui,"Segoe UI",sans-serif}}main{{width:min(900px,calc(100vw - 24px));margin:0 auto;padding:42px 0 64px}}.eyebrow{{color:#d7ff74;font-size:.75rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}article{{padding:22px;border:1px solid #3a403a;border-radius:16px;background:#191c19}}h2{{margin:.3rem 0}}p{{color:#aab2aa}}a{{display:inline-block;margin-top:8px;color:#101210;background:#d7ff74;padding:10px 14px;border-radius:10px;font-weight:800;text-decoration:none}}a:focus-visible{{outline:3px solid #fff;outline-offset:3px}}code{{overflow-wrap:anywhere}}
</style></head><body><main><p class="eyebrow">K-MCFM local workspace</p><h1>Choose a recipe to export</h1><p>Save one portable request, then pass it to <code>scripts/export_recipe_request.py</code> with a new output path. Requests contain no machine-local input or output paths.</p><section class="grid" aria-label="Validated export requests">{cards}</section><p>All entries remain film-inspired Look Approximations. This page does not render, upload, or access image pixels.</p></main></body></html>""".encode()
    files["index.html"] = page
    receipt = {
        "schema": RECIPE_EXPORT_REQUEST_SET_SCHEMA_ID,
        "catalog_status": catalog["status"],
        "request_count": len(rows),
        "requests": rows,
        "index": {"bytes": len(page), "sha256": _sha256(page)},
        "claim_ceiling": "Private offline selection for existing film-inspired look-approximation recipes only.",
    }
    files["request_set_receipt.json"] = _canonical_json(receipt)
    return {"files": files, "receipt": receipt}


def materialize_recipe_export_request_set(
    history_root: Path,
    destination: Path,
    *,
    maximum_recipe_files: int = DEFAULT_MAXIMUM_RECIPE_FILES,
    maximum_recipe_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES,
) -> dict[str, Any]:
    """Publish one create-only request set."""

    if destination.exists() or destination.is_symlink():
        raise RecipeExportRequestError("request-set destination already exists")
    result = build_recipe_export_request_set(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for name, payload in result["files"].items():
            (destination / name).write_bytes(payload)
    except BaseException:
        if destination.exists() and destination.is_dir():
            shutil.rmtree(destination)
        raise
    return result["receipt"]


def export_recipe_request(
    history_root: Path,
    request_path: Path,
    *,
    profile_path: Path,
    output_path: Path,
    root: Path,
    maximum_recipe_files: int = DEFAULT_MAXIMUM_RECIPE_FILES,
    maximum_recipe_bytes: int = DEFAULT_MAXIMUM_RECIPE_BYTES,
    maximum_request_bytes: int = DEFAULT_MAXIMUM_REQUEST_BYTES,
    tile_size: int | None = None,
) -> dict[str, Any]:
    """Validate a request against current history, then invoke exact replay."""

    request_payload, request = _read_recipe_export_request(
        request_path, maximum_request_bytes=maximum_request_bytes
    )
    catalog = build_render_recipe_history(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    selected = [
        row
        for row in catalog["entries"]
        if row.get("recipe_path") == request["recipe_path"]
        and row.get("status") == "valid"
    ]
    if len(selected) != 1:
        raise RecipeExportRequestError("requested recipe is missing or invalid")
    row = selected[0]
    for key in (
        "recipe_sha256",
        "style",
        "output_format",
        "output_bit_depth",
        "output_label",
        "evidence_grade",
    ):
        if row[key] != request[key]:
            raise RecipeExportRequestError(f"requested recipe {key} drift")
    receipt = export_recipe_history_entry(
        history_root,
        recipe_path=request["recipe_path"],
        profile_path=profile_path,
        output_path=output_path,
        root=root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
        tile_size=tile_size,
    )
    return {
        "schema": "kmcfm.recipe-export-request-receipt.v1",
        "request_sha256": _sha256(request_payload),
        **receipt,
    }


__all__ = [
    "DEFAULT_MAXIMUM_REQUEST_BYTES",
    "RECIPE_EXPORT_REQUEST_SCHEMA_ID",
    "RECIPE_EXPORT_REQUEST_SET_SCHEMA_ID",
    "RecipeExportRequestError",
    "build_recipe_export_request_set",
    "export_recipe_request",
    "load_recipe_export_request",
    "materialize_recipe_export_request_set",
    "validate_recipe_export_request",
]
