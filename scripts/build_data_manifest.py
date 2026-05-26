#!/usr/bin/env python3
"""Build a local JSONL manifest for film-domain training images."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FILM_DOMAIN = ROOT / "data" / "film_domain"
DEFAULT_OUTPUT = ROOT / "data" / "processed" / "manifest.jsonl"
DEFAULT_SUMMARY = ROOT / "data" / "processed" / "manifest_summary.json"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

EXPECTED_COUNTS = {
    "portra_400": 500,
    "portra_800": 500,
    "vision3_500t": 500,
    "vision3_250d": 403,
    "ektar_100": 499,
    "tri_x_400": 500,
    "velvia_50": 384,
    "hp5": 500,
}


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    style: str
    width: int
    height: int
    bytes: int
    sha256: str
    split: str


def iter_images(root: Path) -> Iterable[tuple[str, Path]]:
    for style_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for image_path in sorted(style_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield style_dir.name, image_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_split(digest: str, val_percent: int) -> str:
    bucket = int(digest[:8], 16) % 100
    return "val" if bucket < val_percent else "train"


def inspect_image(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return image.size


def build_manifest(
    film_domain: Path,
    output: Path,
    summary_path: Path,
    min_bytes: int,
    min_side: int,
    val_percent: int,
    dry_run: bool,
) -> int:
    if not film_domain.exists():
        raise SystemExit(f"Missing film domain directory: {film_domain}")

    records: list[ImageRecord] = []
    rejected: list[dict[str, object]] = []
    per_style: dict[str, dict[str, int]] = {}

    for style, path in iter_images(film_domain):
        stat = path.stat()
        rel = path.relative_to(ROOT)
        if stat.st_size < min_bytes:
            rejected.append({"path": str(rel), "reason": f"too_small<{min_bytes}"})
            continue
        try:
            width, height = inspect_image(path)
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            rejected.append({"path": str(rel), "reason": type(exc).__name__})
            continue
        if min(width, height) < min_side:
            rejected.append({"path": str(rel), "reason": f"short_side<{min_side}"})
            continue
        digest = sha256_file(path)
        split = choose_split(digest, val_percent)
        records.append(
            ImageRecord(
                path=path,
                style=style,
                width=width,
                height=height,
                bytes=stat.st_size,
                sha256=digest,
                split=split,
            )
        )
        style_summary = per_style.setdefault(
            style,
            {"accepted": 0, "train": 0, "val": 0, "expected": EXPECTED_COUNTS.get(style, 0)},
        )
        style_summary["accepted"] += 1
        style_summary[split] += 1

    rows = []
    for record in sorted(records, key=lambda item: (item.style, str(item.path))):
        rows.append(
            {
                "path": str(record.path.relative_to(ROOT)).replace("\\", "/"),
                "source": "Flickr API",
                "license": "unknown-flickr-user-content",
                "split": record.split,
                "task": "film_lora_training",
                "style": record.style,
                "redistributable": False,
                "width": record.width,
                "height": record.height,
                "bytes": record.bytes,
                "sha256": record.sha256,
                "caption": f"{record.style.replace('_', ' ')} photograph, film grain, analog photography",
            }
        )

    total_expected = sum(EXPECTED_COUNTS.values())
    summary = {
        "schema_version": 1,
        "film_domain": str(film_domain.relative_to(ROOT)).replace("\\", "/"),
        "output": str(output.relative_to(ROOT)).replace("\\", "/"),
        "accepted": len(rows),
        "expected_total": total_expected,
        "missing_total": max(0, total_expected - len(rows)),
        "rejected": len(rejected),
        "per_style": dict(sorted(per_style.items())),
        "missing_styles": {
            style: expected - per_style.get(style, {}).get("accepted", 0)
            for style, expected in sorted(EXPECTED_COUNTS.items())
            if per_style.get(style, {}).get("accepted", 0) < expected
        },
        "rejected_examples": rejected[:100],
    }

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not summary["missing_styles"] and not rejected else 2


def main() -> None:
    parser = argparse.ArgumentParser(description="Build data/processed/manifest.jsonl from film scans.")
    parser.add_argument("--film-domain", type=Path, default=DEFAULT_FILM_DOMAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--min-bytes", type=int, default=30 * 1024)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--val-percent", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.val_percent < 100:
        raise SystemExit("--val-percent must be in [0, 99]")

    raise_code = build_manifest(
        film_domain=args.film_domain.resolve(),
        output=args.output.resolve(),
        summary_path=args.summary.resolve(),
        min_bytes=args.min_bytes,
        min_side=args.min_side,
        val_percent=args.val_percent,
        dry_run=args.dry_run,
    )
    raise SystemExit(raise_code)


if __name__ == "__main__":
    main()
