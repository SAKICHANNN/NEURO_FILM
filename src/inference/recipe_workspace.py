"""Offline desktop workspace over validated recipe history and previews."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from html import escape
from pathlib import Path
from typing import Any

from .recipe_history import build_render_recipe_history
from .recipe_history_html import render_recipe_history_html
from .recipe_preview import build_recipe_output_previews, render_recipe_preview_html

RECIPE_WORKSPACE_SCHEMA_ID = "kmcfm.offline-recipe-workspace.v1"
_WORKSPACE_NAMES = ("index.html", "history.html", "previews.html")


class RecipeWorkspaceError(ValueError):
    """Raised when an offline recipe workspace cannot be built or published."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _placeholder_preview_html(status: str) -> bytes:
    rendered_status = escape(status)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>Render previews · K-MCFM</title><style>
:root{{color-scheme:dark}}*{{box-sizing:border-box}}body{{margin:0;background:#101210;color:#f2f4ef;font:16px/1.5 system-ui,"Segoe UI",sans-serif}}main{{width:min(760px,calc(100vw - 24px));margin:0 auto;padding:48px 0}}a{{color:#d7ff74}}.panel{{padding:28px;border:1px solid #3a403a;border-radius:18px;background:#191c19}}code{{overflow-wrap:anywhere}}@media(max-width:520px){{main{{padding-top:24px}}}}
</style></head><body><main><a href="index.html">Back to workspace</a><section class="panel"><p>Catalog status: <code>{rendered_status}</code></p><h1>No verified previews available</h1><p>Valid, hash-bound RGB16 output recipes are required before this page embeds any pixels.</p></section></main></body></html>""".encode()


def _index_html(catalog: Mapping[str, Any]) -> bytes:
    counts = catalog["counts"]
    status = escape(str(catalog["status"]), quote=True)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; navigate-to 'self'; base-uri 'none'; form-action 'none'">
<title>Film workspace · K-MCFM</title><style>
:root{{color-scheme:dark;--bg:#101210;--panel:#191c19;--line:#3a403a;--text:#f2f4ef;--muted:#aab2aa;--accent:#d7ff74}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top left,#22291f,#101210 38rem);color:var(--text);font:16px/1.5 Inter,system-ui,"Segoe UI",sans-serif}}main{{width:min(1040px,calc(100vw - 32px));margin:0 auto;padding:56px 0 72px}}.eyebrow{{color:var(--accent);font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}}h1{{margin:.2rem 0 .6rem;font-size:clamp(2.4rem,7vw,5.4rem);letter-spacing:-.06em;line-height:.94}}.lede{{max-width:62ch;color:var(--muted)}}.summary{{display:flex;gap:10px;flex-wrap:wrap;margin:28px 0}}.metric{{padding:10px 13px;border:1px solid var(--line);border-radius:12px;background:#191c19}}.metric strong{{font-size:1.35rem}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:34px}}.card{{display:block;min-width:0;padding:28px;border:1px solid var(--line);border-radius:20px;background:linear-gradient(145deg,#191c19,#222622);color:var(--text);text-decoration:none;box-shadow:0 18px 42px #0004}}.card:hover{{border-color:#7d9050}}.card:focus-visible{{outline:3px solid var(--accent);outline-offset:4px}}.card span{{color:var(--accent);font-weight:800}}.card h2{{margin:.45rem 0 .35rem;font-size:1.45rem}}.card p,footer{{color:var(--muted)}}footer{{margin-top:34px;padding-top:18px;border-top:1px solid var(--line);font-size:.82rem}}@media(max-width:650px){{main{{width:calc(100vw - 20px);padding-top:30px}}.grid{{grid-template-columns:1fr}}}}@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style></head><body><main><header><p class="eyebrow">K-MCFM local workspace</p><h1>Film workspace</h1><p class="lede">Review deterministic film-inspired recipes and existing hash-verified previews. All looks remain Look Approximation evidence.</p><div class="summary" aria-label="Catalog summary"><div class="metric"><strong>{counts['valid']}</strong> valid</div><div class="metric"><strong>{counts['invalid']}</strong> invalid</div><div class="metric">status <strong>{status}</strong></div></div></header><nav class="grid" aria-label="Workspace destinations"><a class="card" href="history.html"><span>Recipe records</span><h2>Open render history</h2><p>Search immutable recipe, input, output, format and claim metadata.</p></a><a class="card" href="previews.html"><span>Existing outputs</span><h2>Open render previews</h2><p>Inspect bounded thumbnails decoded only after full output hash verification.</p></a></nav><footer>Private offline workspace · no telemetry, remote network, rerender or calibrated-stock claim</footer></main></body></html>""".encode()


def build_offline_recipe_workspace(
    history_root: Path,
    *,
    maximum_recipe_files: int = 10_000,
    maximum_recipe_bytes: int = 2 * 1024 * 1024,
) -> dict[str, bytes]:
    """Build deterministic workspace files without writing them to disk."""

    catalog = build_render_recipe_history(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    history_html = render_recipe_history_html(catalog)
    if catalog["counts"]["valid"]:
        preview_html = render_recipe_preview_html(build_recipe_output_previews(catalog))
    else:
        preview_html = _placeholder_preview_html(str(catalog["status"]))
    pages = {
        "index.html": _index_html(catalog),
        "history.html": history_html,
        "previews.html": preview_html,
    }
    receipt: dict[str, Any] = {
        "schema": RECIPE_WORKSPACE_SCHEMA_ID,
        "catalog_status": catalog["status"],
        "catalog_counts": catalog["counts"],
        "source_recipe_sha256": {
            row["recipe_path"]: row["recipe_sha256"]
            for row in catalog["entries"]
            if row["status"] == "valid"
        },
        "pages": {
            name: {"bytes": len(payload), "sha256": _sha256(payload)}
            for name, payload in pages.items()
        },
        "claim_ceiling": "Private offline workspace over existing look-approximation recipes and outputs only.",
    }
    return {**pages, "workspace_receipt.json": _canonical_json(receipt)}


def materialize_offline_recipe_workspace(
    history_root: Path,
    destination: Path,
    *,
    maximum_recipe_files: int = 10_000,
    maximum_recipe_bytes: int = 2 * 1024 * 1024,
) -> dict[str, Any]:
    """Create one new exact workspace directory and return its receipt."""

    if destination.exists() or destination.is_symlink():
        raise RecipeWorkspaceError("workspace destination already exists")
    files = build_offline_recipe_workspace(
        history_root,
        maximum_recipe_files=maximum_recipe_files,
        maximum_recipe_bytes=maximum_recipe_bytes,
    )
    try:
        destination.mkdir(parents=True, exist_ok=False)
        for name in (*_WORKSPACE_NAMES, "workspace_receipt.json"):
            (destination / name).write_bytes(files[name])
    except BaseException:
        if destination.exists() and destination.is_dir():
            shutil.rmtree(destination)
        raise
    return json.loads(files["workspace_receipt.json"])


def open_offline_recipe_workspace(destination: Path) -> None:
    """Open an existing workspace entry point in the Windows desktop shell."""

    index = destination / "index.html"
    if not index.is_file():
        raise RecipeWorkspaceError("workspace index is unavailable")
    if os.name != "nt":
        raise RecipeWorkspaceError("desktop workspace launch requires Windows")
    os.startfile(index)  # type: ignore[attr-defined]


__all__ = [
    "RECIPE_WORKSPACE_SCHEMA_ID",
    "RecipeWorkspaceError",
    "build_offline_recipe_workspace",
    "materialize_offline_recipe_workspace",
    "open_offline_recipe_workspace",
]
