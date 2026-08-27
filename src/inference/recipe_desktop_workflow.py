"""Integrated local browser workflow over verified recipes and previews."""

from __future__ import annotations

import base64
import html
from pathlib import Path
from typing import Any

from .recipe_browser_export import RecipeBrowserExportSession
from .recipe_history import build_render_recipe_history
from .recipe_preview import RecipePreview, build_recipe_output_previews

_STYLE_LABELS = {
    "velvia_50": ("Fujifilm Velvia 50", "E-6 slide"),
    "portra_400": ("Kodak Portra 400", "C-41 colour negative"),
    "ektar_100": ("Kodak Ektar 100", "C-41 colour negative"),
}


class RecipeDesktopWorkflowError(ValueError):
    """Raised when the integrated three-look workflow cannot be constructed."""


def _integrated_page(
    rows: list[dict[str, Any]],
    token: str,
    previews: dict[str, RecipePreview],
) -> bytes:
    cards: list[str] = []
    for row in rows:
        style = str(row["style"])
        if style not in previews or style not in _STYLE_LABELS:
            raise RecipeDesktopWorkflowError("recipe and preview styles drift")
        preview = previews[style]
        display_name, process = _STYLE_LABELS[style]
        encoded = base64.b64encode(preview.png_bytes).decode("ascii")
        cards.append(
            '<article class="look-card">'
            f'<img src="data:image/png;base64,{encoded}" '
            f'width="{preview.preview_width}" height="{preview.preview_height}" '
            f'alt="{html.escape(display_name)} Look Approximation preview">'
            '<div class="copy"><div class="badges"><span>film-inspired</span>'
            '<span>Look Approximation</span></div>'
            f'<p class="process">{html.escape(process)}</p>'
            f'<h2>{html.escape(display_name)}</h2>'
            f'<p>{html.escape(str(row["output_format"]))} · '
            f'{int(row["output_bit_depth"])}-bit · existing strict recipe</p>'
            '<form method="post" action="/export">'
            f'<input type="hidden" name="token" value="{html.escape(token, quote=True)}">'
            f'<input type="hidden" name="request_file" value="{html.escape(str(row["request_file"]), quote=True)}">'
            f'<button type="submit" aria-label="Export {html.escape(display_name, quote=True)}">'
            'Export this look</button></form></div></article>'
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; form-action 'self'; base-uri 'none'">
<title>Choose a film look · K-MCFM</title><style>
:root{{color-scheme:dark;--bg:#0e110e;--panel:#191d19;--line:#3a4238;--text:#f2f5ef;--muted:#abb4a9;--accent:#d7ff74}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0,#263020,var(--bg) 42rem);color:var(--text);font:16px/1.5 Inter,system-ui,"Segoe UI",sans-serif}}main{{width:min(1200px,calc(100vw - 28px));margin:auto;padding:48px 0 70px}}header{{max-width:760px;margin-bottom:30px}}.eyebrow,.process{{color:var(--accent);font-size:.76rem;font-weight:800;letter-spacing:.12em;text-transform:uppercase}}h1{{margin:.15rem 0 .5rem;font-size:clamp(2.5rem,7vw,5.6rem);line-height:.92;letter-spacing:-.06em}}header p,.copy>p,footer{{color:var(--muted)}}.grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}}.look-card{{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:20px;background:linear-gradient(150deg,var(--panel),#202620);box-shadow:0 20px 48px #0005}}img{{display:block;width:100%;height:auto;aspect-ratio:3/2;object-fit:cover;background:#090b09}}.copy{{padding:20px}}.badges{{display:flex;flex-wrap:wrap;gap:7px}}.badges span{{padding:4px 9px;border-radius:999px;background:#343b33;font-size:.68rem;font-weight:800}}.badges span:first-child{{background:var(--accent);color:#152005}}.process{{margin:18px 0 0}}h2{{margin:.25rem 0;font-size:1.55rem}}button{{width:100%;margin-top:10px;border:0;border-radius:11px;padding:12px 15px;background:var(--accent);color:#11180a;font:inherit;font-weight:850;cursor:pointer}}button:hover{{background:#e4ff9d}}button:focus-visible{{outline:3px solid #fff;outline-offset:3px}}footer{{margin-top:28px;padding-top:18px;border-top:1px solid var(--line);font-size:.82rem}}.status{{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0)}}@media(max-width:780px){{main{{width:calc(100vw - 20px);padding-top:28px}}.grid{{grid-template-columns:1fr}}h1{{font-size:clamp(2.35rem,15vw,4.5rem)}}}}@media(prefers-reduced-motion:reduce){{*{{scroll-behavior:auto!important}}}}
</style></head><body><main><header><p class="eyebrow">K-MCFM local desktop</p><h1>Choose a film look</h1><p>Review three hash-verified previews, then export one existing strict recipe. These are deterministic film-inspired Look Approximations, not calibrated stock responses.</p></header><section class="grid" aria-label="Available film looks">{''.join(cards)}</section><p class="status" role="status" aria-live="polite">Ready to export one verified look.</p><footer>Private IPv4-loopback session · one create-only export · no telemetry or remote service</footer></main></body></html>""".encode()


class IntegratedRecipeDesktopSession(RecipeBrowserExportSession):
    """One-shot browser session with hash-bound previews for all three looks."""

    def __init__(
        self,
        history_root: Path,
        output_root: Path,
        *,
        profile_path: Path,
        root: Path,
        tile_size: int | None = None,
        maximum_recipe_files: int = 10_000,
        maximum_recipe_bytes: int = 2 * 1024 * 1024,
        maximum_request_bytes: int = 64 * 1024,
        maximum_form_bytes: int = 4096,
    ) -> None:
        catalog = build_render_recipe_history(
            history_root,
            maximum_recipe_files=maximum_recipe_files,
            maximum_recipe_bytes=maximum_recipe_bytes,
        )
        previews = build_recipe_output_previews(catalog)
        preview_by_style = {preview.style: preview for preview in previews}
        if len(previews) != 3 or set(preview_by_style) != set(_STYLE_LABELS):
            raise RecipeDesktopWorkflowError(
                "integrated workflow requires exact Velvia, Portra and Ektar previews"
            )

        def render_page(rows: list[dict[str, Any]], token: str) -> bytes:
            row_styles = [str(row["style"]) for row in rows]
            if len(rows) != 3 or set(row_styles) != set(_STYLE_LABELS):
                raise RecipeDesktopWorkflowError(
                    "integrated workflow requires exact Velvia, Portra and Ektar recipes"
                )
            return _integrated_page(rows, token, preview_by_style)

        super().__init__(
            history_root,
            output_root,
            profile_path=profile_path,
            root=root,
            tile_size=tile_size,
            maximum_recipe_files=maximum_recipe_files,
            maximum_recipe_bytes=maximum_recipe_bytes,
            maximum_request_bytes=maximum_request_bytes,
            maximum_form_bytes=maximum_form_bytes,
            page_renderer=render_page,
        )


__all__ = [
    "IntegratedRecipeDesktopSession",
    "RecipeDesktopWorkflowError",
]
