#!/usr/bin/env python3
"""Build horizontal Velvia 50 comparison sheets across existing schemes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
STYLE_DIR = "velvia_50"


@dataclass
class Render:
    source_key: str
    row_id: str
    original: Path
    after: Path


@dataclass
class Scheme:
    label: str
    manifest: Path
    renders: dict[str, Render]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create original + cross-scheme Velvia 50 comparison contact sheets."
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "contact_sheets" / "velvia50_scheme_comparison_20260612")
    parser.add_argument("--tile-width", type=int, default=176)
    parser.add_argument("--tile-height", type=int, default=132)
    parser.add_argument("--max-schemes-per-sheet", type=int, default=6)
    parser.add_argument("--min-overlap", type=int, default=3)
    parser.add_argument(
        "--include-smoke",
        action="store_true",
        help="Include smoke/debug manifests. By default they are indexed but not rendered.",
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=0,
        help="Limit source rows per collection for quick checks. Zero means all rows.",
    )
    return parser.parse_args()


def repo_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def resolve_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    value = raw.strip().strip('"')
    if not value or value.startswith("local:"):
        return None
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def read_json_rows(path: Path) -> list[dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def manifest_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".json":
        return read_json_rows(path)
    return read_csv(path)


def source_key(path: Path) -> str:
    stem = path.stem.lower()
    replacements = [
        "_raw-neutral",
        "_velvia_50_before",
        "_velvia50_before",
        "_before",
        "_digital_input",
    ]
    for suffix in replacements:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def source_sort_key(key: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", key)
    if match:
        return int(match.group(1)), key
    return 9999, key


def scheme_label(manifest: Path) -> str:
    rel = manifest.resolve().relative_to(ROOT)
    parts = list(rel.parts)
    if "eval" in parts and STYLE_DIR in parts:
        start = parts.index("eval") + 1
        end = parts.index(STYLE_DIR)
        return "/".join(parts[start:end])
    if "color_baseline" in parts:
        return "/".join(parts[parts.index("color_baseline") + 1 : -1])
    return "/".join(parts[:-1])


def is_smoke_or_debug(label: str) -> bool:
    lowered = label.lower()
    return any(token in lowered for token in ("smoke", "debug"))


def discover_manifests() -> list[Path]:
    candidates: list[Path] = []
    candidates.extend((ROOT / "outputs" / "eval").glob(f"**/{STYLE_DIR}/manifest.csv"))
    color_root = ROOT / "outputs" / "color_baseline"
    if color_root.exists():
        candidates.extend(color_root.glob("velvia50*/manifest.csv"))
        candidates.extend(color_root.glob("velvia50*/manifest.json"))
    return sorted(
        {candidate.resolve() for candidate in candidates},
        key=lambda path: (str(path.parent).lower(), 0 if path.suffix.lower() == ".csv" else 1, path.name.lower()),
    )


def load_scheme(manifest: Path) -> Scheme | None:
    renders: dict[str, Render] = {}
    for row in manifest_rows(manifest):
        original = (
            resolve_path(str(row.get("source_original") or ""))
            or resolve_path(str(row.get("before") or ""))
            or resolve_path(str(row.get("input") or ""))
        )
        after = (
            resolve_path(str(row.get("after") or ""))
            or resolve_path(str(row.get("output") or ""))
            or resolve_path(str(row.get("safe") or ""))
        )
        if original is None or after is None:
            continue
        if not original.exists() or not after.exists():
            continue
        key = source_key(original)
        row_id = str(row.get("id") or row.get("index") or key[:18])
        renders[key] = Render(source_key=key, row_id=row_id, original=original, after=after)
    if not renders:
        return None
    return Scheme(label=scheme_label(manifest), manifest=manifest, renders=renders)


def assign_collections(schemes: list[Scheme], min_overlap: int) -> list[tuple[str, list[str], list[Scheme]]]:
    key_counts: dict[str, int] = defaultdict(int)
    for scheme in schemes:
        for key in scheme.renders:
            key_counts[key] += 1
    canonical = [key for key, _ in sorted(key_counts.items(), key=lambda item: (-item[1], source_sort_key(item[0])))]
    canonical = [key for key in canonical if key_counts[key] >= min_overlap]
    if not canonical:
        canonical = sorted({key for scheme in schemes for key in scheme.renders}, key=source_sort_key)

    primary_keys = sorted(set(canonical), key=source_sort_key)
    primary: list[Scheme] = []
    remaining: list[Scheme] = []
    for scheme in schemes:
        overlap = len(set(scheme.renders) & set(primary_keys))
        if overlap >= min_overlap:
            primary.append(scheme)
        else:
            remaining.append(scheme)

    collections: list[tuple[str, list[str], list[Scheme]]] = []
    if primary:
        collections.append(("rawpixls20_main", primary_keys, primary))

    for index, scheme in enumerate(remaining, start=1):
        keys = sorted(scheme.renders, key=source_sort_key)
        collections.append((f"legacy_or_misc_{index:02d}", keys, [scheme]))
    return collections


def text_lines(text: str, max_chars: int = 22, max_lines: int = 3) -> list[str]:
    text = text.replace("_", " ")
    chunks: list[str] = []
    current = ""
    for part in re.split(r"([/ -])", text):
        if len(current) + len(part) <= max_chars:
            current += part
            continue
        if current:
            chunks.append(current.strip())
        current = part.strip()
    if current:
        chunks.append(current.strip())
    if len(chunks) > max_lines:
        chunks = chunks[:max_lines]
        chunks[-1] = chunks[-1][: max_chars - 1] + "."
    return chunks or [text[:max_chars]]


def make_thumb(path: Path, width: int, height: int, cache: dict[tuple[Path, int, int], Image.Image]) -> Image.Image:
    key = (path, width, height)
    if key in cache:
        return cache[key].copy()
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    thumb = Image.new("RGB", (width, height), (246, 246, 246))
    thumb.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    cache[key] = thumb
    return thumb.copy()


def missing_tile(width: int, height: int, label: str) -> Image.Image:
    image = Image.new("RGB", (width, height), (232, 232, 232))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width - 1, height - 1), outline=(190, 190, 190))
    draw.text((8, height // 2 - 5), label, fill=(80, 80, 80), font=font)
    return image


def draw_header(draw: ImageDraw.ImageDraw, x: int, y: int, width: int, label: str, font: ImageFont.ImageFont) -> None:
    for offset, line in enumerate(text_lines(label)):
        draw.text((x + 6, y + 5 + offset * 11), line, fill="black", font=font)
    draw.line((x, y + 48, x + width, y + 48), fill=(205, 205, 205))


def write_sheet(
    output: Path,
    title: str,
    source_keys: list[str],
    schemes: list[Scheme],
    tile_width: int,
    tile_height: int,
    cache: dict[tuple[Path, int, int], Image.Image],
) -> None:
    font = ImageFont.load_default()
    columns = 1 + len(schemes)
    gutter = 8
    header_h = 70
    row_label_h = 18
    row_h = tile_height + row_label_h + gutter
    width = columns * tile_width + (columns + 1) * gutter
    height = header_h + len(source_keys) * row_h + gutter
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((gutter, 8), title, fill="black", font=font)

    x = gutter
    draw_header(draw, x, 20, tile_width, "original", font)
    for scheme in schemes:
        x += tile_width + gutter
        draw_header(draw, x, 20, tile_width, scheme.label, font)

    originals: dict[str, Path] = {}
    for scheme in schemes:
        for key, render in scheme.renders.items():
            originals.setdefault(key, render.original)

    for row_index, key in enumerate(source_keys):
        y = header_h + row_index * row_h
        x = gutter
        original = originals.get(key)
        if original and original.exists():
            tile = make_thumb(original, tile_width, tile_height, cache)
        else:
            tile = missing_tile(tile_width, tile_height, "missing original")
        sheet.paste(tile, (x, y))
        draw.text((x + 5, y + tile_height + 2), key[:28], fill=(30, 30, 30), font=font)

        for scheme in schemes:
            x += tile_width + gutter
            render = scheme.renders.get(key)
            if render is None:
                tile = missing_tile(tile_width, tile_height, "not rendered")
            else:
                tile = make_thumb(render.after, tile_width, tile_height, cache)
            sheet.paste(tile, (x, y))
            if render is not None:
                draw.text((x + 5, y + tile_height + 2), render.row_id[:28], fill=(30, 30, 30), font=font)

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, "PNG")


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    manifests = discover_manifests()
    indexed: list[dict[str, object]] = []
    schemes: list[Scheme] = []
    seen_labels: set[str] = set()
    for manifest in manifests:
        scheme = load_scheme(manifest)
        if scheme is None:
            continue
        if scheme.label in seen_labels:
            continue
        seen_labels.add(scheme.label)
        indexed.append(
            {
                "label": scheme.label,
                "manifest": repo_path(manifest),
                "render_count": len(scheme.renders),
                "is_smoke_or_debug": is_smoke_or_debug(scheme.label),
            }
        )
        if not args.include_smoke and is_smoke_or_debug(scheme.label):
            continue
        schemes.append(scheme)

    collections = assign_collections(schemes, args.min_overlap)
    cache: dict[tuple[Path, int, int], Image.Image] = {}
    sheets: list[dict[str, object]] = []

    for collection_name, source_keys, collection_schemes in collections:
        if args.limit_rows > 0:
            source_keys = source_keys[: args.limit_rows]
        collection_schemes = sorted(collection_schemes, key=lambda scheme: scheme.label.lower())
        for page_index, start in enumerate(range(0, len(collection_schemes), args.max_schemes_per_sheet), start=1):
            page_schemes = collection_schemes[start : start + args.max_schemes_per_sheet]
            page_total = (len(collection_schemes) + args.max_schemes_per_sheet - 1) // args.max_schemes_per_sheet
            output = output_dir / f"{collection_name}_page_{page_index:02d}_of_{page_total:02d}.png"
            title = f"Velvia 50 scheme comparison - {collection_name} - page {page_index}/{page_total}"
            write_sheet(output, title, source_keys, page_schemes, args.tile_width, args.tile_height, cache)
            sheets.append(
                {
                    "collection": collection_name,
                    "page": page_index,
                    "page_count": page_total,
                    "file": repo_path(output),
                    "source_count": len(source_keys),
                    "schemes": [scheme.label for scheme in page_schemes],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "style": STYLE_DIR,
        "include_smoke": args.include_smoke,
        "indexed_manifest_count": len(indexed),
        "rendered_scheme_count": len(schemes),
        "sheet_count": len(sheets),
        "tile_width": args.tile_width,
        "tile_height": args.tile_height,
        "max_schemes_per_sheet": args.max_schemes_per_sheet,
        "indexed_manifests": indexed,
        "sheets": sheets,
    }
    (output_dir / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    with (output_dir / "SCHEMES.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "manifest", "render_count", "is_smoke_or_debug"])
        writer.writeheader()
        writer.writerows(indexed)

    print(f"indexed_manifests={len(indexed)}")
    print(f"rendered_schemes={len(schemes)}")
    print(f"sheets={len(sheets)}")
    print(repo_path(output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
