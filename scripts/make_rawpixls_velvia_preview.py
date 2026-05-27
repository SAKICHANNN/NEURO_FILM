#!/usr/bin/env python3
"""Build a Velvia 50 before/after preview batch from raw.pixls.us RAW files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import time
import urllib.parse
import urllib.request
from html import unescape
from pathlib import Path

import numpy as np
import rawpy
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipeline_color_baseline import load_image, style_transfer  # noqa: E402


API_URL = "https://raw.pixls.us/json/getrepository.php"
BASE_URL = "https://raw.pixls.us/"
ALLOWED_EXTS = {".3FR", ".ARW", ".CR2", ".CR3", ".DNG", ".NEF", ".NRW", ".ORF", ".PEF", ".RAF", ".RW2", ".SRW"}
CAMERA_MAKES = {
    "Canon",
    "Fujifilm",
    "Leica",
    "Nikon",
    "Olympus",
    "Panasonic",
    "Pentax",
    "Ricoh",
    "Samsung",
    "Sigma",
    "Sony",
}
BAD_TERMS = (
    "monochrom",
    "monochrome",
    "black and white",
    "infrared",
    "ir converted",
)


def slug_float(value: float) -> str:
    return f"{value:.3g}".replace(".", "p")


def log(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    safe = str(message).encode("ascii", "backslashreplace").decode("ascii")
    print(safe, file=stream, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a raw.pixls.us Velvia 50 preview batch.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--strength", type=float, default=0.58)
    parser.add_argument("--luma-strength", type=float, default=0.38)
    parser.add_argument("--max-mb", type=float, default=26.0)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--output-margin", type=int, default=4, help="Compress generated RGB into [margin, 255-margin].")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing generated preview files.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "color_baseline" / "velvia50_rawpixls20_s0p58_gamutsafe")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "rawpixls_cache")
    parser.add_argument("--stats", type=Path, default=ROOT / "configs" / "film_color_stats.json")
    return parser.parse_args()


def safe_url(url: str) -> str:
    absolute = urllib.parse.urljoin(BASE_URL, unescape(url))
    parts = urllib.parse.urlsplit(absolute)
    return urllib.parse.urlunsplit(
        (parts.scheme, parts.netloc, urllib.parse.quote(parts.path), parts.query, parts.fragment)
    )


def fetch_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "neuro-film-preview/1.0"})
    with urllib.request.urlopen(request, timeout=60) as handle:
        return json.load(handle)


def parse_repository_row(row: list) -> dict | None:
    make, model, mode, megapixels, note, license_html, date, file_html, exif_html = row
    if make not in CAMERA_MAKES:
        return None
    text = " ".join(str(part) for part in (make, model, mode, note, file_html)).lower()
    if any(term in text for term in BAD_TERMS):
        return None
    if "zero" not in license_html.lower() and "public domain" not in license_html.lower():
        return None

    link_match = re.search(r"href=['\"]([^'\"]+)['\"]", file_html)
    name_match = re.search(r">([^<>]+)</a>", file_html)
    hash_match = re.search(r"Checksum['\"]>([0-9a-fA-F]{64})</span>", file_html)
    size_match = re.search(r"\(([0-9.]+)MB\)", file_html)
    exif_match = re.search(r"href=['\"]([^'\"]+)['\"]", exif_html)
    if not link_match or not name_match or not size_match:
        return None

    filename = unescape(name_match.group(1))
    ext = Path(filename).suffix.upper()
    if ext not in ALLOWED_EXTS:
        return None

    return {
        "make": make,
        "model": model,
        "mode": mode,
        "megapixels": megapixels,
        "note": note,
        "date": date,
        "license": "CC0/Public Domain",
        "filename": filename,
        "extension": ext,
        "size_mb": float(size_match.group(1)),
        "sha256": hash_match.group(1).lower() if hash_match else "",
        "download_url": safe_url(link_match.group(1)),
        "exif_url": safe_url(exif_match.group(1)) if exif_match else "",
    }


def build_candidates(max_mb: float) -> list[dict]:
    repo = fetch_json(API_URL)["data"]
    candidates = [parsed for row in repo if (parsed := parse_repository_row(row))]
    candidates = [item for item in candidates if item["size_mb"] <= max_mb]

    def score(item: dict) -> tuple:
        raw_mode = "raw" not in item["mode"].lower()
        small = abs(item["size_mb"] - 14.0)
        return (raw_mode, small, item["make"], item["model"], item["filename"])

    candidates = sorted(candidates, key=score)
    groups: dict[str, list[dict]] = {}
    for item in candidates:
        groups.setdefault(item["make"], []).append(item)

    make_order = ["Canon", "Nikon", "Sony", "Fujifilm", "Olympus", "Panasonic", "Pentax", "Ricoh", "Leica", "Samsung", "Sigma"]
    diversified: list[dict] = []
    while any(groups.values()):
        for make in make_order:
            if groups.get(make):
                diversified.append(groups[make].pop(0))
    return diversified


def download_file(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1024:
        return
    tmp = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "neuro-film-preview/1.0"})
    with urllib.request.urlopen(request, timeout=180) as response, tmp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    tmp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_raw(raw_path: Path, out_path: Path, max_side: int) -> Image.Image:
    with rawpy.imread(str(raw_path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            no_auto_bright=False,
            output_bps=8,
            gamma=(2.222, 4.5),
            user_flip=0,
        )
    image = Image.fromarray(rgb, mode="RGB")
    image = ImageOps.exif_transpose(image)
    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, "JPEG", quality=95)
    return image


def soften_output_bounds(image: Image.Image, margin: int) -> Image.Image:
    if margin <= 0:
        return image
    margin = max(0, min(32, int(margin)))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr = margin + (arr / 255.0) * (255 - margin * 2)
    return Image.fromarray(np.rint(arr).astype(np.uint8), mode="RGB")


def is_color_photo(image: Image.Image) -> bool:
    small = image.resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    if arr.mean() < 0.055 or arr.mean() > 0.96:
        return False
    channel_delta = np.mean(np.abs(arr[..., 0] - arr[..., 1]) + np.abs(arr[..., 1] - arr[..., 2]))
    saturation_proxy = np.std(arr[..., 0] - arr[..., 1]) + np.std(arr[..., 1] - arr[..., 2])
    return bool(channel_delta > 0.012 and saturation_proxy > 0.018)


def make_pair(before: Path, after: Path, out_path: Path, title: str, strength: float) -> None:
    font = ImageFont.load_default()
    left = load_image(before)
    right = load_image(after)
    target_h = min(900, max(left.height, right.height))
    for image in (left, right):
        image.thumbnail((target_h * 2, target_h), Image.Resampling.LANCZOS)
    tile_w = max(left.width, right.width)
    tile_h = max(left.height, right.height)
    label_h = 42
    canvas = Image.new("RGB", (tile_w * 2, tile_h + label_h), "white")
    canvas.paste(left, ((tile_w - left.width) // 2, label_h))
    canvas.paste(right, (tile_w + (tile_w - right.width) // 2, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), f"Before RAW neutral render | {title}", fill="black", font=font)
    draw.text((tile_w + 10, 8), f"After Velvia 50 s={strength:.2f} gamut-safe", fill="black", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, "PNG")


def make_contact_sheet(pair_paths: list[Path], out_path: Path, strength: float) -> None:
    font = ImageFont.load_default()
    thumbs = []
    for idx, path in enumerate(pair_paths, start=1):
        image = Image.open(path).convert("RGB")
        image.thumbnail((520, 260), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (520, 284), "white")
        canvas.paste(image, ((520 - image.width) // 2, 0))
        ImageDraw.Draw(canvas).text((8, 264), f"{idx:02d} {path.stem}", fill="black", font=font)
        thumbs.append(canvas)
    cols = 2
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 520, rows * 284 + 34), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (8, 10),
        f"Velvia 50 preview from raw.pixls.us RAW files, strength={strength:.2f}, no grain, gamut-safe",
        fill="black",
        font=font,
    )
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, ((idx % cols) * 520, 34 + (idx // cols) * 284))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "PNG")


def clean_generated_dirs(output_dir: Path, strength: float) -> None:
    for directory_name in ("inputs", "after", "pairs"):
        directory = output_dir / directory_name
        if directory.exists():
            shutil.rmtree(directory)
    for filename in (
        "manifest.csv",
        "manifest.json",
        "skipped.json",
        f"velvia50_rawpixls20_s{slug_float(strength)}_before_after_contact_sheet.jpg",
        f"velvia50_rawpixls20_s{slug_float(strength)}_before_after_contact_sheet.png",
    ):
        (output_dir / filename).unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if not args.no_clean:
        clean_generated_dirs(output_dir, args.strength)
    input_dir = output_dir / "inputs"
    after_dir = output_dir / "after"
    pair_dir = output_dir / "pairs"
    raw_dir = args.cache_dir.resolve()
    for directory in (input_dir, after_dir, pair_dir, raw_dir):
        directory.mkdir(parents=True, exist_ok=True)

    stats_doc = json.loads(args.stats.read_text(encoding="utf-8"))
    style_stats = stats_doc["styles"]["velvia_50"]
    candidates = build_candidates(args.max_mb)
    rows: list[dict] = []
    pair_paths: list[Path] = []
    skipped: list[dict] = []
    used_models: set[str] = set()

    log(f"candidate_count={len(candidates)} max_mb={args.max_mb}")
    for candidate in candidates:
        if len(rows) >= args.count:
            break
        model_key = f"{candidate['make']} {candidate['model']}"
        if model_key in used_models and len(rows) < min(args.count, 12):
            continue

        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{candidate['make']}_{candidate['model']}_{Path(candidate['filename']).stem}")[:120]
        raw_path = raw_dir / f"{slug}{candidate['extension'].lower()}"
        before_path = input_dir / f"{len(rows)+1:02d}_{slug}_raw-neutral.jpg"
        strength_slug = slug_float(args.strength)
        after_path = after_dir / f"{len(rows)+1:02d}_{slug}_velvia50_s{strength_slug}.png"
        pair_path = pair_dir / f"{len(rows)+1:02d}_{slug}_before_after.png"

        try:
            log(f"download {len(rows)+1:02d}: {candidate['make']} {candidate['model']} {candidate['size_mb']:.2f}MB")
            download_file(candidate["download_url"], raw_path)
            actual_hash = sha256(raw_path)
            if candidate["sha256"] and actual_hash != candidate["sha256"]:
                raise ValueError(f"sha256 mismatch: expected {candidate['sha256']} got {actual_hash}")
            before_image = render_raw(raw_path, before_path, args.max_side)
            if not is_color_photo(before_image):
                skipped.append({**candidate, "reason": "rendered image looked monochrome or near-empty"})
                before_path.unlink(missing_ok=True)
                continue

            after_image = style_transfer(
                before_image,
                style_stats,
                "velvia_50",
                strength=args.strength,
                luma_strength=args.luma_strength,
                grain=0.0,
                seed=42 + len(rows),
                gamut_safe=True,
            )
            after_image = soften_output_bounds(after_image, args.output_margin)
            after_image.save(after_path, "PNG")
            make_pair(before_path, after_path, pair_path, f"{candidate['make']} {candidate['model']}", args.strength)

            row = {
                **candidate,
                "raw_path": str(raw_path),
                "before": str(before_path),
                "after": str(after_path),
                "pair": str(pair_path),
                "style": "velvia_50",
                "strength": args.strength,
                "luma_strength": args.luma_strength,
                "grain": 0.0,
                "gamut_safe": True,
                "output_margin": args.output_margin,
                "source": "raw.pixls.us repository API",
            }
            rows.append(row)
            pair_paths.append(pair_path)
            used_models.add(model_key)
            log(f"accepted {len(rows):02d}: {candidate['make']} {candidate['model']}")
        except Exception as exc:  # noqa: BLE001 - preview builder should keep collecting.
            skipped.append({**candidate, "reason": repr(exc)})
            log(f"skip: {candidate['make']} {candidate['model']} {exc}")
            time.sleep(0.5)

    if len(rows) < args.count:
        log(f"warning: requested {args.count}, generated {len(rows)}", error=True)

    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        make_contact_sheet(
            pair_paths,
            output_dir / f"velvia50_rawpixls20_s{slug_float(args.strength)}_before_after_contact_sheet.png",
            args.strength,
        )
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    log(f"output_dir={output_dir}")
    log(f"generated={len(rows)} skipped={len(skipped)}")
    return 0 if len(rows) >= args.count else 2


if __name__ == "__main__":
    raise SystemExit(main())
