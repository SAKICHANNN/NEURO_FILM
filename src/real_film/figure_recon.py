"""Bounded official-figure acquisition for real-film source reconnaissance."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw
import requests


OFFICIAL_CDN_PREFIX = "https://ars.els-cdn.com/content/image/"
MAXIMUM_DECODED_PIXELS = 50_000_000


def acquire_official_figure(
    url: str,
    destination: Path,
    *,
    maximum_bytes: int,
    allowed_media_types: tuple[str, ...],
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Download one official figure with bounded bytes and strict media type."""

    if not url.startswith(OFFICIAL_CDN_PREFIX):
        raise ValueError("figure URL is outside the frozen official CDN")
    if maximum_bytes < 1 or timeout_seconds <= 0:
        raise ValueError("invalid acquisition bounds")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    total = 0
    digest = hashlib.sha256()
    media_type = ""
    try:
        with requests.get(
            url,
            stream=True,
            timeout=timeout_seconds,
            headers={"User-Agent": "neuro-film-research/1.0"},
        ) as response:
            response.raise_for_status()
            media_type = response.headers.get("Content-Type", "").split(";", 1)[0]
            if media_type not in allowed_media_types:
                raise ValueError("unexpected figure media type")
            content_length = response.headers.get("Content-Length")
            if content_length is not None and int(content_length) > maximum_bytes:
                raise ValueError("figure Content-Length exceeds frozen maximum")
            with temporary.open("wb") as handle:
                for block in response.iter_content(chunk_size=64 * 1024):
                    if not block:
                        continue
                    total += len(block)
                    if total > maximum_bytes:
                        raise ValueError("figure body exceeds frozen maximum")
                    handle.write(block)
                    digest.update(block)
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise

    try:
        with Image.open(destination) as image:
            image.verify()
        with Image.open(destination) as image:
            width, height = image.size
            if image.format != "JPEG" or image.mode != "RGB":
                raise ValueError("figure is not an RGB JPEG")
            if width * height > MAXIMUM_DECODED_PIXELS:
                raise ValueError("figure exceeds decoded-pixel limit")
    except Exception:
        destination.unlink(missing_ok=True)
        raise

    return {
        "bytes": total,
        "sha256": digest.hexdigest(),
        "media_type": media_type,
        "format": "JPEG",
        "mode": "RGB",
        "width": width,
        "height": height,
    }


def build_vertical_contact_sheet(
    figures: list[tuple[int, Path]],
    destination: Path,
    *,
    maximum_width: int = 1200,
) -> dict[str, Any]:
    """Build a deterministic labelled reconnaissance sheet."""

    if not figures or maximum_width < 64:
        raise ValueError("invalid contact-sheet request")
    prepared: list[tuple[int, Image.Image]] = []
    for figure_number, path in figures:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if rgb.width > maximum_width:
                height = max(1, round(rgb.height * maximum_width / rgb.width))
                rgb = rgb.resize((maximum_width, height), Image.Resampling.LANCZOS)
            prepared.append((figure_number, rgb.copy()))
    label_height = 32
    width = max(image.width for _, image in prepared)
    height = sum(label_height + image.height for _, image in prepared)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    y = 0
    for figure_number, image in prepared:
        draw.text((8, y + 8), f"Figure {figure_number}", fill="black")
        y += label_height
        sheet.paste(image, (0, y))
        y += image.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    sheet.save(temporary, format="PNG", compress_level=9)
    temporary.replace(destination)
    raw = destination.read_bytes()
    return {
        "path": destination.as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "width": width,
        "height": height,
    }
