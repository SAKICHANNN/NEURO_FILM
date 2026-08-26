"""Hash-bound, read-only offline previews for existing render recipes."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.preprocess.output_encode import srgb_icc_profile

MAX_OUTPUT_BYTES = 192 * 1024 * 1024
MAX_DECODED_PIXELS = 30_000_000
MAX_PREVIEW_ROWS = 8
MAX_PREVIEW_WIDTH = 960
MAX_PREVIEW_HEIGHT = 640
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class RecipePreviewError(ValueError):
    """Raised when a recipe-bound output cannot be previewed safely."""


@dataclass(frozen=True)
class RecipePreview:
    """One deterministic in-memory preview and its bound source facts."""

    recipe_path: str
    style: str
    output_sha256: str
    source_width: int
    source_height: int
    preview_width: int
    preview_height: int
    png_sha256: str
    png_bytes: bytes


def _required_text(entry: Mapping[str, Any], key: str) -> str:
    value = entry.get(key)
    if not isinstance(value, str) or not value:
        raise RecipePreviewError(f"catalog entry {key} must be non-empty text")
    return value


def _validate_entry(entry: Mapping[str, Any]) -> tuple[Path, str, str, str]:
    if not isinstance(entry, Mapping):
        raise TypeError("catalog entry must be a mapping")
    if entry.get("status") != "valid":
        raise RecipePreviewError("preview requires a valid catalog row")
    if entry.get("output_format") != "PNG" or entry.get("output_bit_depth") != 16:
        raise RecipePreviewError("preview requires a recipe-bound RGB16 PNG")
    if entry.get("output_label") != "film-inspired":
        raise RecipePreviewError("preview output label must remain film-inspired")
    if entry.get("evidence_grade") != "look-approximation":
        raise RecipePreviewError("preview evidence grade must remain look-approximation")
    output_path = Path(_required_text(entry, "output_path"))
    recipe_path = _required_text(entry, "recipe_path")
    style = _required_text(entry, "style")
    expected_sha256 = _required_text(entry, "output_sha256")
    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise RecipePreviewError("output SHA-256 must be lowercase hexadecimal")
    if output_path.suffix.casefold() != ".png":
        raise RecipePreviewError("preview output path must use .png")
    return output_path, recipe_path, style, expected_sha256


def _bounded_png16_bytes(path: Path, expected_sha256: str) -> bytes:
    if not path.exists() or not path.is_file():
        raise RecipePreviewError("recipe-bound output file is missing or not regular")
    size = path.stat().st_size
    if size <= 0 or size > MAX_OUTPUT_BYTES:
        raise RecipePreviewError("recipe-bound output file exceeds the preview byte budget")
    payload = path.read_bytes()
    if len(payload) != size or len(payload) > MAX_OUTPUT_BYTES:
        raise RecipePreviewError("recipe-bound output changed or exceeded the byte budget")
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RecipePreviewError("recipe-bound output SHA-256 mismatch")
    if (
        len(payload) < 26
        or payload[:8] != _PNG_SIGNATURE
        or payload[12:16] != b"IHDR"
        or payload[24] != 16
        or payload[25] != 2
    ):
        raise RecipePreviewError("recipe-bound output is not a true RGB16 PNG")
    return payload


def _preview_dimensions(width: int, height: int) -> tuple[int, int]:
    if width <= MAX_PREVIEW_WIDTH and height <= MAX_PREVIEW_HEIGHT:
        return width, height
    if MAX_PREVIEW_WIDTH * height <= MAX_PREVIEW_HEIGHT * width:
        return MAX_PREVIEW_WIDTH, max(
            1, (height * MAX_PREVIEW_WIDTH + width // 2) // width
        )
    return max(1, (width * MAX_PREVIEW_HEIGHT + height // 2) // height), MAX_PREVIEW_HEIGHT


def _encode_preview_png(rgb16: np.ndarray) -> bytes:
    rgb8 = ((rgb16.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
    image = Image.fromarray(rgb8, mode="RGB")
    buffer = BytesIO()
    image.save(
        buffer,
        format="PNG",
        compress_level=6,
        optimize=False,
        icc_profile=srgb_icc_profile(),
    )
    return buffer.getvalue()


def build_recipe_output_preview(entry: Mapping[str, Any]) -> RecipePreview:
    """Verify then decode one existing RGB16 output into a bounded PNG8 preview."""

    output_path, recipe_path, style, expected_sha256 = _validate_entry(entry)
    payload = _bounded_png16_bytes(output_path, expected_sha256)
    decoded_bgr = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if (
        decoded_bgr is None
        or decoded_bgr.dtype != np.uint16
        or decoded_bgr.ndim != 3
        or decoded_bgr.shape[2] != 3
    ):
        raise RecipePreviewError("recipe-bound output did not decode as HxWx3 uint16")
    height, width = (int(decoded_bgr.shape[0]), int(decoded_bgr.shape[1]))
    if width <= 0 or height <= 0 or width * height > MAX_DECODED_PIXELS:
        raise RecipePreviewError("recipe-bound output exceeds the decoded-pixel budget")
    preview_width, preview_height = _preview_dimensions(width, height)
    rgb16 = decoded_bgr[..., ::-1]
    if (preview_width, preview_height) != (width, height):
        rgb16 = cv2.resize(
            rgb16,
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
    png_bytes = _encode_preview_png(np.ascontiguousarray(rgb16))
    return RecipePreview(
        recipe_path=recipe_path,
        style=style,
        output_sha256=expected_sha256,
        source_width=width,
        source_height=height,
        preview_width=preview_width,
        preview_height=preview_height,
        png_sha256=hashlib.sha256(png_bytes).hexdigest(),
        png_bytes=png_bytes,
    )


def build_recipe_output_previews(catalog: Mapping[str, Any]) -> tuple[RecipePreview, ...]:
    """Build bounded previews for every valid row in one ready catalog."""

    if not isinstance(catalog, Mapping):
        raise TypeError("catalog must be a mapping")
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise RecipePreviewError("catalog entries must be a list")
    valid = [
        entry
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("status") == "valid"
    ]
    if not valid or len(valid) > MAX_PREVIEW_ROWS:
        raise RecipePreviewError("catalog valid-row count is outside the preview budget")
    return tuple(build_recipe_output_preview(entry) for entry in valid)


def render_recipe_preview_html(previews: Sequence[RecipePreview]) -> bytes:
    """Render previews as one deterministic, self-contained offline document."""

    if not isinstance(previews, Sequence) or isinstance(previews, (str, bytes)):
        raise TypeError("previews must be a sequence")
    if not previews or len(previews) > MAX_PREVIEW_ROWS:
        raise RecipePreviewError("preview count is outside the document budget")
    cards: list[str] = []
    for preview in previews:
        if not isinstance(preview, RecipePreview):
            raise TypeError("preview sequence contains an invalid item")
        data = base64.b64encode(preview.png_bytes).decode("ascii")
        style = escape(preview.style)
        cards.append(
            '<article class="preview-card">'
            f'<img src="data:image/png;base64,{data}" width="{preview.preview_width}" '
            f'height="{preview.preview_height}" alt="{style} look-approximation preview">'
            '<div class="preview-copy">'
            '<div class="badges"><span>film-inspired</span><span>look-approximation</span></div>'
            f"<h2>{style}</h2>"
            f'<p>Existing render · {preview.source_width}×{preview.source_height} RGB16</p>'
            f'<code>output {escape(preview.output_sha256)}</code>'
            f'<code>preview {escape(preview.png_sha256)}</code>'
            "</div></article>"
        )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:; base-uri 'none'; form-action 'none'">
<title>Render previews · K-MCFM</title><style>
:root{{color-scheme:dark;--panel:#1a1e1a;--line:#3b423b;--text:#f3f5f0;--muted:#aeb6ad;--accent:#d7ff74}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top left,#22291f,#0f110f 38rem);color:var(--text);font-family:Inter,system-ui,"Segoe UI",sans-serif;line-height:1.5}}
main{{width:min(1180px,calc(100vw - 32px));margin:0 auto;padding:48px 0 72px}} header{{max-width:760px;margin-bottom:30px}} .eyebrow{{color:var(--accent);font-size:.75rem;font-weight:800;letter-spacing:.14em;text-transform:uppercase}} h1{{margin:.15rem 0 .4rem;font-size:clamp(2.2rem,6vw,4.8rem);line-height:.95;letter-spacing:-.055em}} header p,.preview-copy p{{color:var(--muted)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,330px),1fr));gap:18px}} .preview-card{{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:18px;background:var(--panel);box-shadow:0 18px 45px #0005}} img{{display:block;width:100%;height:auto;aspect-ratio:3/2;object-fit:cover;background:#080a08}} .preview-copy{{padding:18px}} h2{{margin:.6rem 0 .25rem;font-size:1.35rem}} .badges{{display:flex;flex-wrap:wrap;gap:7px}} .badges span{{padding:4px 9px;border-radius:999px;background:#343a33;font-size:.72rem;font-weight:800}} .badges span:first-child{{background:var(--accent);color:#172006}} code{{display:block;margin-top:7px;color:#dce5d8;font:clamp(.66rem,1.5vw,.76rem)/1.45 "Cascadia Mono",Consolas,monospace;overflow-wrap:anywhere;word-break:break-all}} footer{{margin-top:28px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:.82rem}}
@media(max-width:620px){{main{{width:calc(100vw - 20px);padding-top:28px}}}}
</style></head><body><main><header><p class="eyebrow">K-MCFM local workspace</p><h1>Render previews</h1><p>Read-only thumbnails of existing, hash-verified look-approximation outputs. No image is re-rendered or exported.</p></header><section class="grid" aria-label="Recipe previews">{''.join(cards)}</section><footer>Private offline preview · existing outputs only · no calibrated stock claim</footer></main></body></html>"""
    return html.encode("utf-8")


__all__ = [
    "MAX_DECODED_PIXELS",
    "MAX_OUTPUT_BYTES",
    "MAX_PREVIEW_HEIGHT",
    "MAX_PREVIEW_ROWS",
    "MAX_PREVIEW_WIDTH",
    "RecipePreview",
    "RecipePreviewError",
    "build_recipe_output_preview",
    "build_recipe_output_previews",
    "render_recipe_preview_html",
]
